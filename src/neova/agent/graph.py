"""The conversation graph, four LLM-driven steps:

    precheck ──match──▶ handoff ──▶ END
       │
       ▼
     agent ⇄ tools        the LLM picks its tools (ReAct loop); the tools node runs them with the
       │                  state and owns every write
       ▼
   post_review ──human──▶ handoff ──▶ END
       │
       ▼
      END

One run per customer message; the checkpointer keeps the state between messages. Every text the
customer reads is either the agent's reply after the checks, or a fixed sentence below."""
import json
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from pydantic import BaseModel

from neova import llm
from neova.agent import prompts
from neova.agent.prompts import data
from neova.agent.tools import TOOLS, APIError, APIUnavailable, NeovaAPI
from neova.config import LOGS_DIR, get_settings
from neova.rag.index import load_index, retrieve

HTTP = None  # httpx client for the API; None means API_BASE_URL (tests plug the in-process app)
TICKETS_DIR = LOGS_DIR / "tickets"   # one file per ticket, for reviewing transfers by hand
MAX_TOOL_CALLS = 8
HISTORY_MESSAGES = 12
CUSTOMER_FIELDS = ("customer_id", "plan", "monthly_price", "contract_start_date", "engagement_months",
                   "seniority_months", "balance_due", "last_invoices", "equipment", "open_incident_id")

BOOKED = "C'est confirmé : un technicien passera le {when}. {cost}"
UNCERTAIN_BOOKING = ("Je n'ai pas pu confirmer la réservation à cause d'un incident technique. Je vérifie à votre "
                     "prochain message ; rien n'est confirmé tant que ce n'est pas certain.")
UNCERTAIN_TICKET = ("Je n'ai pas pu confirmer la transmission de votre demande à un conseiller à cause d'un "
                    "incident technique. Je vérifie à votre prochain message.")
REDIRECTS = {"payment_plan": "un échéancier de paiement", "incident_follow_up": "le suivi de l'incident",
             "technician_appointment": "un rendez-vous avec un technicien"}
WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]


class Confirmation(BaseModel):
    decision: Literal["yes", "no", "other_slot", "question", "new_request"]
    preference: str


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # the conversation, as LangChain messages
    message: str                              # the customer message of this turn
    session: str | None
    customer_id: str | None
    identity_failures: int
    facts: dict                               # customer record and incidents read this conversation
    passages: list[str]                       # chunk ids retrieved this turn
    proposal: dict | None                     # {"proposal": ..., "key": ...} waiting for a yes
    proposed: list[str]                       # slot ids already proposed in this conversation
    consulted: bool                           # documents or the gesture policy were used this turn
    uncertain: dict | None                    # a write whose result is unknown: {"kind", ...}
    gesture: dict | None                      # last assess_gesture outcome
    ticket: dict | None                       # the last ticket created in this conversation
    actions: list[str]                        # what the agent did, for the ticket
    final: dict | None                        # the reply to check: {"text", "sources" = passages of this turn}
    reason: str                               # why the turn goes to a human
    review: list[str]                         # the after-review verdict, shown in the chat logs
    calls: int                                # tool calls this turn
    next: str
    reply: str
    trace: list[str]


@lru_cache
def index():
    return load_index()


def api(state: State) -> NeovaAPI:
    return NeovaAPI(HTTP, session=state.get("session"), customer_id=state.get("customer_id"))


def agent_llm(messages):
    """The one place the agent's LLM is called; tests replace it. A provider failure can come back as
    a 200 with an empty reply and finish_reason "error", which OpenRouter's model routing does not
    fall back on, so the fallback model is asked explicitly."""
    settings = get_settings()
    for model in (settings.chat_model, settings.chat_model_fallback):
        chat = llm.get_chat(model, callbacks=[llm.UsageLogger()]).bind_tools(TOOLS)
        reply = llm.invoke_with_retry(chat, messages)
        if reply.response_metadata.get("finish_reason") != "error":
            return reply
    raise RuntimeError("le modèle et son modèle de secours ont renvoyé une erreur")


def when(slot: dict) -> str:
    start, end = datetime.fromisoformat(slot["start"]), datetime.fromisoformat(slot["end"])
    return f"{WEEKDAYS[start.weekday()]} {start.day} {MONTHS[start.month - 1]} entre {start:%Hh%M} et {end:%Hh%M}"


def transcript(state: State) -> str:
    lines = []
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            lines.append(f"client : {m.content}")
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            lines.append(f"agent : {m.content}")
    return "\n".join(lines[-HISTORY_MESSAGES:])


def visit(state: State, name: str, **update) -> dict:
    return {"trace": state.get("trace", []) + [name], **update}


def to_human(state: State, name: str, reason: str) -> dict:
    return visit(state, name, reason=reason, next="handoff")


# --------------------------------------------------------------------------- nodes

def precheck(state: State) -> dict:
    message = state["message"]
    update = {"trace": ["precheck"], "reply": "", "reason": "", "final": None, "calls": 0, "consulted": False, "passages": [],
              "messages": [HumanMessage(message)]}
    verdict = prompts.precheck(message, transcript(state))
    if verdict is None:
        return {**update, "reason": "contrôle d'entrée indisponible", "next": "handoff"}
    if verdict.matched_items:
        return {**update, "reason": f"transfert immédiat (situations {verdict.matched_items}) : « {verdict.evidence} »",
                "next": "handoff"}
    uncertain = state.get("uncertain")
    if uncertain and uncertain["kind"] == "ticket":
        return {**update, "next": "handoff"}
    if uncertain and uncertain["kind"] == "booking":
        outcome = book(state, uncertain["proposal"], uncertain["key"], uncertain.get("attempts", 0))
        if "booking_error" in outcome:
            text = f"Le créneau n'a finalement pas pu être réservé ({outcome.pop('booking_error')})."
            outcome.update(reply=text, messages=[AIMessage(text)], next="finish")
        return {**update, **outcome}
    return {**update, "next": "agent"}


def agent(state: State) -> dict:
    if state.get("calls", 0) >= MAX_TOOL_CALLS:
        return to_human(state, "agent", "trop d'étapes sans conclusion")
    messages = [SystemMessage(prompts.template("agent"))] + state["messages"][-HISTORY_MESSAGES * 3:]
    try:
        reply = agent_llm(messages)
    except Exception as exc:
        return to_human(state, "agent", f"assistant indisponible : {exc}")
    if reply.tool_calls:
        return visit(state, "agent", messages=[reply], next="tools")
    text = (reply.content if isinstance(reply.content, str) else " ".join(map(str, reply.content))).strip()
    if not text:
        return visit(state, "agent", calls=state.get("calls", 0) + 1, next="agent")
    return visit(state, "agent", final={"text": text, "sources": list(state.get("passages", []))}, next="post_review")


def tools(state: State) -> dict:
    """Runs the tool calls of the last AI message. Reads are free; writes are gated here."""
    update: dict = {"messages": [], "calls": state.get("calls", 0)}
    view = dict(state)
    next_node = "agent"
    for call in state["messages"][-1].tool_calls:
        update["calls"] += 1
        result, changes, route = TOOL_RUNNERS[call["name"]](view, **call["args"])
        update["messages"][:0] = changes.pop("messages", [])
        view.update(changes)
        update.update(changes)
        update["messages"].append(ToolMessage(result, tool_call_id=call["id"]))
        if route:
            next_node = route
            break
    return visit(state, "tools", **update, next=next_node)


def post_review(state: State) -> dict:
    """The after-review check reads the list of situations and says what it still needs to judge
    the candidates; the code fetches it (documents by search, facts from the API) and asks once
    more. No loop: one round of needs, then a verdict."""
    facts = dict(state.get("facts", {}))
    if state.get("gesture"):
        facts["geste_commercial"] = {k: state["gesture"][k] for k in ("outcome", "redirects")}
    documents = "\n\n".join(index().by_id[i].text for i in state.get("passages", []) if i in index().by_id)
    if not documents:   # the agent acted without reading: the check reads for it, on the request and the draft reply
        contract_start = (facts.get("client") or {}).get("contract_start_date")
        found = retrieve(index(), [state["message"], state["final"]["text"]], contract_start=contract_start)
        documents = "\n\n".join(f"[{r.chunk.chunk_id}]\n{r.chunk.text}" for r in found)
    context, draft = transcript(state), state["final"]["text"]   # context explains a bare "oui" or identifiers
    verdict = prompts.post_review(state["message"], context, draft, facts, documents)
    review = []
    if verdict is not None and verdict.needs:
        review.append(f"demande : {[f'{n.kind} : {n.query}' for n in verdict.needs]}")
        documents, facts = resolve_needs(state, verdict.needs, documents, facts)
        verdict = prompts.post_review(state["message"], context, draft, facts, documents, final=True)
    if verdict is None:
        return visit(state, "post_review", review=review, reason="contrôle après examen indisponible", next="handoff")
    review += [f"situations : {verdict.matched_items} · documents suffisants : {verdict.documents_cover}",
               f"conditions conseiller : {[(c.condition, c.met) for c in verdict.advisor_conditions]}",
               f"raison : {verdict.reason}"]
    if not verdict.can_conclude:
        return visit(state, "post_review", review=review, next="handoff",
                     reason=f"transfert après examen (situations {verdict.matched_items}) : {verdict.reason}")
    return visit(state, "post_review", review=review, reply=draft, messages=[AIMessage(draft)], next="finish")


def resolve_needs(state: State, needs, documents: str, facts: dict):
    """Fetches what the check asked for: a search per document need, the customer record and
    incidents for a fact need (only if the customer is identified; otherwise the fact stays
    unavailable and the check cannot count it as established)."""
    contract_start = (facts.get("client") or {}).get("contract_start_date")
    for need in needs:
        if need.kind == "document":
            found = retrieve(index(), [need.query], contract_start=contract_start)
            documents += "\n\n" + "\n\n".join(f"[{r.chunk.chunk_id}]\n{r.chunk.text}" for r in found)
        elif state.get("session"):
            try:
                customer = api(state).customer()
                facts["client"] = {k: customer.get(k) for k in CUSTOMER_FIELDS}
                facts["incidents"] = api(state).incidents(customer["postal_code"])
            except (APIError, APIUnavailable):
                facts.setdefault("indisponible", []).append(need.query)
        else:
            facts.setdefault("non verifiable, client non identifie", []).append(need.query)
    return documents, facts


def handoff(state: State) -> dict:
    """Ticket first, then the announcement of the real result. One ticket per transfer; an uncertain
    creation keeps its content and key so replaying it never creates a second one."""
    uncertain = state.get("uncertain") or {}
    if uncertain.get("kind") == "ticket":
        ticket, key, reason = uncertain["ticket"], uncertain["key"], uncertain["reason"]
    else:
        reason = state.get("reason") or "transfert demandé"
        ticket, key = prompts.draft_ticket(transcript(state), state.get("actions", []), reason), str(uuid.uuid4())
    try:
        created = api(state).create_ticket(**ticket, idempotency_key=key)
    except APIUnavailable:
        return visit(state, "handoff", reply=UNCERTAIN_TICKET, messages=[AIMessage(UNCERTAIN_TICKET)], next="finish",
                     uncertain={"kind": "ticket", "ticket": ticket, "key": key, "reason": reason})
    except APIError:
        return visit(state, "handoff", reply=prompts.NO_TICKET_MESSAGE, messages=[AIMessage(prompts.NO_TICKET_MESSAGE)],
                     uncertain=None, next="finish")
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    (TICKETS_DIR / f"{created['ticket_id']}.json").write_text(json.dumps(
        {**created, **ticket, "reason": reason, "trace": state.get("trace", []), "conversation": transcript(state)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    text = prompts.handoff_message(state["message"], created["callback_eta"])
    return visit(state, "handoff", reply=text, messages=[AIMessage(text)], ticket=created, uncertain=None, proposal=None,
                 actions=state.get("actions", []) + [f"ticket créé ({created['ticket_id']})"], next="finish")


def finish(state: State) -> dict:
    return {}


# --------------------------------------------------------------------------- tool runners
# Each returns (text for the LLM, state changes, next node or None to keep looping).

SESSION_REQUIRED = "Session requise : demandez au client son numéro client (NEO-XXXXX) et le téléphone du contrat."


def run_search(state, question: str, historical: bool = False):
    contract_start = (state.get("facts", {}).get("client") or {}).get("contract_start_date")
    results = retrieve(index(), [question], historical=historical, contract_start=contract_start)
    text = "\n\n".join(f"[{r.chunk.chunk_id}]\n{r.chunk.text}" for r in results) or "Aucun passage trouvé."
    passages = (state.get("passages", []) + [r.chunk.chunk_id for r in results])[-12:]
    return text, {"passages": passages, "consulted": True}, None


def run_verify(state, customer_id: str, phone: str):
    client = api(state)
    try:
        customer = client.verify(customer_id.strip().upper(), "".join(ch for ch in phone if ch.isdigit()))
    except APIError:
        failures = state.get("identity_failures", 0) + 1
        if failures >= 2:
            return ("Vérification refusée une deuxième fois : n'insistez pas, transférez à un conseiller "
                    "(request_handoff).", {"identity_failures": failures}, None)
        return "Vérification refusée : numéro client ou téléphone incorrect. Demandez-les à nouveau.", \
            {"identity_failures": failures}, None
    except APIUnavailable:
        return "Service indisponible : l'identité ne peut pas être vérifiée pour le moment.", {}, None
    changes = {"session": client.session, "customer_id": customer["customer_id"], "identity_failures": 0,
               "actions": state.get("actions", []) + ["identité vérifiée"]}
    if state.get("customer_id") not in (None, customer["customer_id"]):
        changes.update(facts={}, actions=["identité vérifiée"], gesture=None, proposal=None, proposed=[],
                       ticket=None, uncertain=None, passages=[],
                       messages=[RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(state["message"]), state["messages"][-1]])
    if customer["plan"].startswith("Néova Pro"):
        return (f"Identité vérifiée : le contrat de ce client est professionnel ({customer['plan']}).",
                changes, None)
    return "Identité vérifiée.", changes, None


def run_get_customer(state):
    if not state.get("session"):
        return SESSION_REQUIRED, {}, None
    try:
        customer = api(state).customer()
    except APIUnavailable:
        return "Service indisponible : le dossier ne peut pas être lu pour le moment.", {}, None
    view = {k: customer.get(k) for k in CUSTOMER_FIELDS}
    facts = {**state.get("facts", {}), "client": view}
    return json.dumps(view, ensure_ascii=False), {"facts": facts}, None


def run_get_incidents(state):
    if not state.get("session"):
        return SESSION_REQUIRED, {}, None
    try:
        postal_code = api(state).customer()["postal_code"]
        incidents = api(state).incidents(postal_code)
    except APIUnavailable:
        return "Service indisponible : les incidents ne peuvent pas être lus pour le moment.", {}, None
    facts = {**state.get("facts", {}), "incidents": incidents}
    return json.dumps(incidents, ensure_ascii=False) or "Aucun incident.", {"facts": facts}, None


def run_propose(state, reason: str, override_reason: str | None = None, another_slot: bool = False):
    """Picks the first slot the API accepts (skipping the ones already proposed in this
    conversation) and prepares it. The LLM never sees a slot it has not proposed."""
    if not state.get("session"):
        return SESSION_REQUIRED, {}, None
    skipped = list(state.get("proposed", []))
    if another_slot and state.get("proposal"):
        skipped.append(state["proposal"]["proposal"]["slot"]["slot_id"])
    try:
        slots = [s for s in api(state).slots() if s["slot_id"] not in skipped]
        for slot in slots:
            try:
                proposal = api(state).propose_appointment(slot["slot_id"], reason, override_reason or None)
            except APIError as error:
                if error.body.get("error") == "slot_taken":
                    continue
                if error.body.get("error") == "already_has_appointment":
                    return (f"Proposition refusée : le client a déjà un rendez-vous confirmé le "
                            f"{when(error.body['appointment'])}. Un client ne peut avoir qu'un rendez-vous : "
                            "aucune autre réservation n'est possible."), {"proposal": None}, None
                detail = error.body.get("detail") or error.body.get("error")
                return f"Proposition refusée : {detail}", {"proposal": None}, None
            text = (f"Proposition prête : {when(proposal['slot'])}. Frais : {proposal['cost_notice']} "
                    "Transmettez ces informations au client et demandez-lui de confirmer (oui / non).")
            return text, {"proposal": {"proposal": proposal, "key": str(uuid.uuid4())},
                          "proposed": skipped + [slot["slot_id"]], "consulted": True}, None
    except APIUnavailable:
        return "Service indisponible : aucun créneau ne peut être proposé pour le moment.", {"proposal": None}, None
    return "Aucun autre créneau disponible dans la zone du client.", {"proposal": None}, None


def customer_said_yes(state) -> bool:
    proposal = state["proposal"]["proposal"]
    offer = f"{when(proposal['slot'])}. {proposal['cost_notice']}"
    verdict = prompts.ask("confirmation", Confirmation, data("proposition", offer), data("message_client", state["message"]))
    return verdict is not None and verdict.decision == "yes"


def book(state, proposal: dict, key: str, attempts: int = 0) -> dict:
    """The only place a booking is written, always with the stored proposal and its key."""
    try:
        booked = api(state).book_appointment(proposal["proposal_id"], key)
    except APIUnavailable:
        if attempts + 1 >= 3:
            return {"booking_error": "réservation non confirmée après trois essais", "proposal": None, "uncertain": None}
        return {"reply": UNCERTAIN_BOOKING, "messages": [AIMessage(UNCERTAIN_BOOKING)], "next": "finish",
                "uncertain": {"kind": "booking", "proposal": proposal, "key": key, "attempts": attempts + 1}}
    except APIError as error:
        return {"booking_error": error.body.get("detail") or error.body.get("error"), "proposal": None, "uncertain": None}
    text = BOOKED.format(when=when(proposal["slot"]), cost=proposal["cost_notice"])
    return {"reply": text, "messages": [AIMessage(text)], "proposal": None, "uncertain": None, "next": "finish",
            "actions": state.get("actions", []) + [f"rendez-vous technicien réservé ({booked['appointment_id']})"]}


def run_book(state):
    if not state.get("proposal"):
        return "Aucune proposition en attente : appelez d'abord propose_appointment.", {}, None
    if not customer_said_yes(state):
        return "Le client n'a pas clairement accepté la proposition : ne réservez pas.", {}, None
    outcome = book(state, state["proposal"]["proposal"], state["proposal"]["key"])
    route = outcome.pop("next", None)
    if "booking_error" in outcome:
        return f"Réservation refusée : {outcome.pop('booking_error')}. Proposez un autre créneau.", outcome, None
    return "Réservation effectuée.", outcome, route


def run_gesture(state, request: str):
    if not state.get("session"):
        return SESSION_REQUIRED, {}, None
    facts = state.get("facts", {})
    if "client" not in facts:
        _, changes, _ = run_get_customer(state)
        facts = changes.get("facts", facts)
    verdict = prompts.gesture(request, facts)
    if verdict is None:
        return "Décision indisponible pour le moment : transférez à un conseiller (request_handoff).", {}, None
    if not verdict.is_gesture_request and not verdict.out_of_scope:
        return ("Ce n'est pas une demande de geste commercial : traitez-la avec search_documents et le dossier du client.",
                {"facts": facts}, None)
    redirects = [REDIRECTS[r] for r in verdict.redirect_to]
    outcome = {"outcome": verdict.customer_outcome, "redirects": redirects,
               "handoff_required": verdict.decision == "needs_review", "note": verdict.internal_note}
    if outcome["handoff_required"]:
        return ("Issue : ce geste doit être examiné par un conseiller. Transférez la demande (request_handoff) "
                "sans annoncer de geste au client.", {"gesture": outcome, "facts": facts, "consulted": True}, None)
    text = "Issue : geste non accordé." + (f" Orientations à proposer : {', '.join(redirects)}." if redirects else "")
    return text, {"gesture": outcome, "facts": facts, "consulted": True}, None


def run_handoff(state, reason: str):
    return "Transfert en cours.", {"reason": reason}, "handoff"


TOOL_RUNNERS = {
    "search_documents": run_search, "verify_customer": run_verify, "get_customer": run_get_customer,
    "get_incidents": run_get_incidents, "propose_appointment": run_propose,
    "book_appointment": run_book, "assess_gesture": run_gesture, "request_handoff": run_handoff,
}

ROUTES = {
    "precheck": ["handoff", "agent", "finish"],
    "agent": ["tools", "agent", "post_review", "handoff"],
    "tools": ["agent", "handoff", "finish"],
    "post_review": ["finish", "handoff"],
    "handoff": ["finish"],
}


def build(checkpointer=None):
    graph = StateGraph(State)
    for node in (precheck, agent, tools, post_review, handoff, finish):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "precheck")
    for name, targets in ROUTES.items():
        graph.add_conditional_edges(name, lambda state: state["next"], {t: t for t in targets})
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_turn(graph, thread_id: str, message: str) -> dict:
    return graph.invoke({"message": message, "next": ""}, {"configurable": {"thread_id": thread_id}})
