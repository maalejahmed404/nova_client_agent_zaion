"""The conversation graph: one run per customer message, the checkpointer keeps the state between
messages. Every text sent to the customer is either written by a checked LLM step or one of the
fixed sentences below.

precheck → handoff | resume a waiting step | planner
planner → clarification | identity | collect → gesture (gesture route) → post_review
post_review → handoff | respond (info, gesture) | technical_check → propose (action)"""
import re
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from neova.agent import prompts
from neova.agent.prompts import data
from neova.agent.tools import APIError, APIUnavailable, NeovaAPI
from neova.rag.answer import answer
from neova.rag.index import Retrieved, load_index, retrieve
from neova.rag.ingest import fold

HTTP = None  # the httpx client used to reach the API; None means API_BASE_URL (tests plug the in-process app)

ASK_IDENTITY = ("Pour accéder à votre dossier, j'ai besoin de votre numéro client (NEO-XXXXX) et du numéro "
                "de téléphone associé à votre contrat.")
IDENTIFIED = "Merci, votre identité est vérifiée. Que puis-je faire pour vous ?"
IDENTITY_RETRY = ("Je n'ai pas pu vérifier ces informations. Pouvez-vous me redonner votre numéro client "
                  "(NEO-XXXXX) et votre numéro de téléphone ?")
ASK_CONFIRM = "Je peux vous proposer un technicien le {when}. {cost} Confirmez-vous ce rendez-vous (oui / non) ?"
ASK_AGAIN = "Confirmez-vous le rendez-vous du {when} (oui / non) ?"
BOOKED = "C'est confirmé : un technicien passera le {when}. {cost}"
NOT_BOOKED = "D'accord, aucun rendez-vous n'a été réservé."
BOOKING_LOST = "Ce créneau n'a finalement pas pu être réservé."
NO_SLOT = "Je n'ai trouvé aucun créneau disponible dans votre zone pour le moment."
ALREADY_BOOKED = ("Vous avez déjà un rendez-vous avec un technicien le {when}. Un seul rendez-vous peut être "
                  "programmé à la fois.")
OUTAGE = ("Une panne réseau est en cours dans votre zone : un technicien ne pourrait pas la résoudre. "
          "Vous pouvez suivre l'incident depuis votre espace client.")
UNKNOWN = "Je n'ai pas cette information."
WELCOME = ("Je suis l'assistant du service client Néova. Je peux vous aider pour votre box ou votre ligne, une "
           "facture, un déménagement, une résiliation ou un rendez-vous avec un technicien. Que puis-je faire pour vous ?")
ALREADY_TRANSFERRED = ("Votre demande est déjà transmise à un conseiller, qui vous recontactera {eta}. "
                       "Il aura l'ensemble de notre échange.")
UNCERTAIN_BOOKING = ("Je n'ai pas pu confirmer la réservation à cause d'un incident technique. Je vérifie à votre "
                     "prochain message ; rien n'est confirmé tant que ce n'est pas certain.")
UNCERTAIN_TICKET = ("Je n'ai pas pu confirmer la transmission de votre demande à un conseiller à cause d'un "
                    "incident technique. Je vérifie à votre prochain message.")
REDIRECTS = {"payment_plan": "un échéancier de paiement", "incident_follow_up": "le suivi de l'incident",
             "technician_appointment": "un rendez-vous avec un technicien"}
WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]


class Plan(BaseModel):
    query: str
    route: Literal["info", "gesture", "action", "conversation"]
    historical: bool
    needs_customer: bool
    clarification: str


class TechnicalCheck(BaseModel):
    justified: Literal["yes", "no", "missing_info"]
    reason: Literal["no_internet", "slow_internet", "installation", "equipment_swap"]
    override: Literal["none", "pto_damaged", "equipment_damaged"]
    message: str


class Confirmation(BaseModel):
    decision: Literal["yes", "no", "other_slot", "question", "new_request"]
    preference: str


class State(TypedDict, total=False):
    message: str
    history: list[str]
    session: str | None
    customer_id: str | None
    identity_failures: int
    pending: dict | None     # the waiting step: {"step": ..., "request": ..., "plan": ..., ...}
    request: str
    plan: dict
    facts: dict
    passages: list[str]      # chunk ids of the retrieved context
    gesture: dict | None
    booking: dict | None     # {"reason", "override", "preference", "exclude"} while choosing a slot
    actions: list[str]       # what the agent did, for the ticket
    reason: str              # set when the conversation must go to a human
    next: str                # routing decision of the node that just ran
    reply: str
    ticket: dict | None      # the ticket created in this conversation: one per conversation
    trace: list[str]         # nodes visited during this message, for tests and the evaluation


@lru_cache
def index():
    return load_index()


def api(state: State) -> NeovaAPI:
    return NeovaAPI(HTTP, session=state.get("session"), customer_id=state.get("customer_id"))


def conversation(state: State) -> str:
    return "\n".join(state.get("history", [])[-8:])


def when(slot: dict) -> str:
    start, end = datetime.fromisoformat(slot["start"]), datetime.fromisoformat(slot["end"])
    return f"{WEEKDAYS[start.weekday()]} {start.day} {MONTHS[start.month - 1]} entre {start:%Hh%M} et {end:%Hh%M}"


def visit(state: State, name: str, **update) -> dict:
    return {"trace": state.get("trace", []) + [name], **update}


def to_human(state: State, name: str, reason: str) -> dict:
    return visit(state, name, reason=reason, next="handoff")


def search(plan: dict, facts: dict) -> list[Retrieved]:
    contract_start = (facts.get("client") or {}).get("contract_start_date")
    return retrieve(index(), [plan["query"]], historical=plan["historical"], contract_start=contract_start)


def passages(state: State) -> list[Retrieved]:
    return [Retrieved(index().by_id[i], 0.0, "search") for i in state.get("passages", []) if i in index().by_id]


# --------------------------------------------------------------------------- nodes

def precheck(state: State) -> dict:
    message = state["message"]
    history = state.get("history", [])
    update = {"trace": ["precheck"], "reply": "", "reason": "", "history": history + [f"client : {message}"]}
    verdict = prompts.precheck(message, "\n".join(history[-8:]))
    if verdict is None:
        return {**update, "reason": "contrôle d'entrée indisponible", "next": "handoff"}
    if verdict.matched_items:
        return {**update, "reason": f"transfert immédiat (situations {verdict.matched_items}) : « {verdict.evidence} »",
                "next": "handoff"}
    step = (state.get("pending") or {}).get("step")
    resume = {"awaiting_identity": "identify", "awaiting_clarification": "planner",
              "awaiting_technical_info": "technical_check", "awaiting_booking_confirmation": "confirm",
              "uncertain_operation": "replay"}
    if step not in ("awaiting_booking_confirmation", "uncertain_operation") and all(identifiers(message)):
        return {**update, "next": "identify"}
    return {**update, "next": resume.get(step, "planner")}


def planner(state: State) -> dict:
    pending = state.get("pending") or {}
    request = state["message"]
    if pending.get("step") == "awaiting_clarification":
        request = f"{pending['request']}\n{state['message']}"
    plan = prompts.ask("plan", Plan, data("contexte", conversation(state)), data("message_client", request))
    if plan is None:
        return to_human(state, "planner", "analyse de la demande indisponible")
    update = {"request": request, "plan": plan.model_dump(), "pending": None, "gesture": None, "booking": None}
    if plan.clarification:
        if prompts.leaked(plan.clarification):
            return to_human(state, "planner", "texte non sûr")
        return visit(state, "planner", **{**update, "reply": plan.clarification, "next": "finish",
                                          "pending": {"step": "awaiting_clarification", "request": request}})
    if plan.route == "conversation":
        return visit(state, "planner", **update, reply=WELCOME, next="finish")
    if (plan.route != "info" or plan.needs_customer) and not state.get("session"):
        return visit(state, "planner", **{**update, "reply": ASK_IDENTITY, "next": "finish",
                                          "pending": {"step": "awaiting_identity", "request": request,
                                                      "plan": plan.model_dump()}})
    return visit(state, "planner", **update, next="collect")


def identifiers(message: str):
    return (re.search(r"NEO[-\s]?(\d{5})", message, re.I),
            re.search(r"(?<!\d)0\d(?:[\s.]?\d{2}){4}(?!\d)", message))


def identify(state: State) -> dict:
    """Verifies the identifiers, then resumes the request that was waiting for them. Identifiers
    given without being asked are accepted too; the rest of the message then goes to the planner."""
    pending, message = state.get("pending") or {}, state["message"]
    number, phone = identifiers(message)
    customer = None
    if number and phone:
        client = api(state)
        try:
            customer = client.verify(f"NEO-{number.group(1)}", re.sub(r"\D", "", phone.group(0)))
        except APIError:
            customer = None
        except APIUnavailable:
            return to_human(state, "identify", "API indisponible pendant l'identification")
    if customer is None:
        failures = state.get("identity_failures", 0) + 1
        if failures >= 2:
            return to_human(state, "identify", "identification impossible après deux essais")
        return visit(state, "identify", identity_failures=failures, reply=IDENTITY_RETRY, next="finish")
    update = {"session": client.session, "customer_id": customer["customer_id"], "identity_failures": 0,
              "pending": None, "actions": state.get("actions", []) + ["identité vérifiée"]}
    if customer["plan"].startswith("Néova Pro"):
        return visit(state, "identify", **update, reason="contrat professionnel", next="handoff")
    if pending.get("step") == "awaiting_identity":
        return visit(state, "identify", **update, request=pending["request"], plan=pending["plan"], next="collect")
    rest = message.replace(number.group(0), "").replace(phone.group(0), "")
    if len(re.findall(r"\w+", rest)) < 3:
        return visit(state, "identify", **update, reply=IDENTIFIED, next="finish")
    return visit(state, "identify", **update, next="planner")


def collect(state: State) -> dict:
    facts = {}
    if state.get("session"):
        client = api(state)
        try:
            customer = client.customer()
            facts = {"client": customer, "incidents": client.incidents(customer["postal_code"])}
        except (APIError, APIUnavailable):
            return to_human(state, "collect", "données client indisponibles")
    results = search(state["plan"], facts)
    route = state["plan"]["route"]
    return visit(state, "collect", facts=facts, passages=[r.chunk.chunk_id for r in results],
                 next="gesture" if route == "gesture" else "post_review")


def gesture(state: State) -> dict:
    verdict = prompts.gesture(state["request"], state["facts"])
    if verdict is None:
        return to_human(state, "gesture", "décision de geste commercial indisponible")
    if verdict.decision == "needs_review":
        return visit(state, "gesture", gesture=verdict.model_dump(), next="handoff",
                     reason=f"geste commercial à examiner par un conseiller : {verdict.internal_note}")
    return visit(state, "gesture", gesture=verdict.model_dump(), next="post_review")


def post_review(state: State) -> dict:
    documents = "\n\n".join(r.chunk.text for r in passages(state))
    verdict = prompts.post_review(state["plan"]["query"], state["facts"], documents)
    route = state["plan"]["route"]
    if verdict is None:
        return to_human(state, "post_review", "contrôle après examen indisponible")
    # "À traiter en premier lieu, puis transférer si le cadre ne permet pas de conclure": a situation of the
    # list alone is not a transfer; it becomes one only when the documents and facts cannot settle the case.
    if not verdict.can_conclude and (verdict.matched_items or route == "info"):
        return to_human(state, "post_review",
                        f"transfert après examen (situations {verdict.matched_items}) : {verdict.reason}")
    return visit(state, "post_review", next="technical_check" if route == "action" else "respond")


def respond(state: State) -> dict:
    decision = None
    if state.get("gesture"):
        redirects = [REDIRECTS[r] for r in state["gesture"]["redirect_to"]]
        decision = "Geste commercial non accordé." + (f" Orientations possibles : {', '.join(redirects)}." if redirects else "")
    reply = answer(state["plan"]["query"], passages(state), state.get("facts"), decision)
    if not reply.in_corpus:
        return to_human(state, "respond", "réponse non établie par la documentation")
    if prompts.leaked(reply.answer):
        return to_human(state, "respond", "texte non sûr")
    return visit(state, "respond", reply=reply.answer, next="finish")


def technical_check(state: State) -> dict:
    pending = state.get("pending") or {}
    request, plan = pending.get("request", state.get("request")), pending.get("plan", state.get("plan"))
    faq = "\n\n".join(r.chunk.text for r in passages(state))
    verdict = prompts.ask("technician_check", TechnicalCheck, data("faq", faq),
                          data("incidents", (state.get("facts") or {}).get("incidents", [])),
                          data("conversation", conversation(state)))
    if verdict is None:
        return to_human(state, "technical_check", "vérification technique indisponible")
    if verdict.justified != "yes" and prompts.leaked(verdict.message):
        return to_human(state, "technical_check", "texte non sûr")
    if verdict.justified == "missing_info":
        return visit(state, "technical_check", reply=verdict.message, next="finish",
                     pending={"step": "awaiting_technical_info", "request": request, "plan": plan})
    if verdict.justified == "no":
        return visit(state, "technical_check", reply=verdict.message, pending=None, next="finish")
    booking = {"reason": verdict.reason, "override": None if verdict.override == "none" else verdict.override,
               "preference": "", "exclude": []}
    return visit(state, "technical_check", request=request, plan=plan, pending=None, booking=booking, next="propose")


def propose(state: State) -> dict:
    booking, client = state["booking"], api(state)
    try:
        slots = [s for s in client.slots() if s["slot_id"] not in booking["exclude"]]
        preferred = fold(booking["preference"])
        slots.sort(key=lambda s: (WEEKDAYS[datetime.fromisoformat(s["start"]).weekday()] not in preferred, s["start"]))
        for slot in slots:
            try:
                proposal = client.propose_appointment(slot["slot_id"], booking["reason"], booking["override"])
            except APIError as error:
                if error.body.get("error") == "slot_taken":
                    continue
                if error.body.get("error") == "active_outage":
                    return visit(state, "propose", reply=OUTAGE, pending=None, next="finish")
                if error.body.get("error") == "already_has_appointment":
                    return visit(state, "propose", pending=None, next="finish",
                                 reply=ALREADY_BOOKED.format(when=when(error.body["appointment"])))
                return to_human(state, "propose", f"réservation refusée par l'API : {error.body.get('error')}")
            pending = {"step": "awaiting_booking_confirmation", "proposal": proposal, "key": str(uuid.uuid4()),
                       "request": state.get("request"), "plan": state.get("plan")}
            text = ASK_CONFIRM.format(when=when(proposal["slot"]), cost=proposal["cost_notice"])
            return visit(state, "propose", pending=pending, reply=text, next="finish")
    except APIUnavailable:
        return to_human(state, "propose", "API indisponible pendant la recherche de créneau")
    return visit(state, "propose", reply=NO_SLOT, pending=None, next="finish")


def confirm(state: State) -> dict:
    pending = state["pending"]
    proposal = pending["proposal"]
    offer = ASK_CONFIRM.format(when=when(proposal["slot"]), cost=proposal["cost_notice"])
    verdict = prompts.ask("confirmation", Confirmation, data("proposition", offer), data("message_client", state["message"]))
    if verdict is None:
        return visit(state, "confirm", reply=ASK_AGAIN.format(when=when(proposal["slot"])), next="finish")
    if verdict.decision == "yes":
        return book(state, "confirm", pending)
    if verdict.decision == "no":
        return visit(state, "confirm", reply=NOT_BOOKED, pending=None, next="finish")
    if verdict.decision == "other_slot":
        booking = {"reason": proposal["reason"], "override": proposal["override_reason"],
                   "preference": verdict.preference, "exclude": [proposal["slot"]["slot_id"]]}
        return visit(state, "confirm", booking=booking, pending=None, next="propose")
    if verdict.decision == "question":
        return visit(state, "confirm", next="side_question")
    return visit(state, "confirm", pending=None, next="planner")


def book(state: State, name: str, pending: dict) -> dict:
    """The only place a booking is written, always with the proposal and the key kept in pending."""
    proposal = pending["proposal"]
    try:
        booked = api(state).book_appointment(proposal["proposal_id"], pending["key"])
    except APIUnavailable:
        attempts = pending.get("attempts", 0) + 1
        if attempts >= 3:
            return to_human(state, name, "réservation non confirmée après trois essais")
        return visit(state, name, reply=UNCERTAIN_BOOKING, next="finish",
                     pending={**pending, "step": "uncertain_operation", "kind": "booking", "attempts": attempts})
    except APIError:
        return visit(state, name, reply=BOOKING_LOST, pending=None, next="finish")
    text = BOOKED.format(when=when(proposal["slot"]), cost=proposal["cost_notice"])
    return visit(state, name, reply=text, pending=None, next="finish",
                 actions=state.get("actions", []) + [f"rendez-vous technicien réservé ({booked['appointment_id']})"])


def side_question(state: State) -> dict:
    """A question asked while a booking waits for confirmation: answer it, keep the proposal."""
    results = retrieve(index(), [state["message"]])
    reply = answer(state["message"], results, state.get("facts"))
    text = reply.answer if reply.in_corpus and not prompts.leaked(reply.answer) else UNKNOWN
    proposal = state["pending"]["proposal"]
    return visit(state, "side_question", reply=f"{text}\n\n{ASK_AGAIN.format(when=when(proposal['slot']))}", next="finish")


def replay(state: State) -> dict:
    """An operation whose result is unknown is replayed with the same key before anything else."""
    pending = state["pending"]
    if pending["kind"] == "ticket":
        return visit(state, "replay", next="handoff", reason=pending["reason"])
    return book(state, "replay", pending)


def handoff(state: State) -> dict:
    """Ticket first, then the announcement of the real result. A ticket whose creation is uncertain
    keeps its content and key, so replaying it never creates a second one."""
    pending = state.get("pending") or {}
    if state.get("ticket") and pending.get("kind") != "ticket":
        return visit(state, "handoff", reply=ALREADY_TRANSFERRED.format(eta=state["ticket"]["callback_eta"]),
                     pending=None, next="finish")
    if pending.get("kind") == "ticket":
        ticket, key, reason = pending["ticket"], pending["key"], pending["reason"]
    else:
        reason = state.get("reason") or "transfert demandé"
        ticket, key = prompts.draft_ticket(conversation(state), state.get("actions", []) + [reason]), str(uuid.uuid4())
    try:
        created = api(state).create_ticket(**ticket, idempotency_key=key)
    except APIUnavailable:
        return visit(state, "handoff", reply=UNCERTAIN_TICKET, next="finish", pending={
            "step": "uncertain_operation", "kind": "ticket", "ticket": ticket, "key": key, "reason": reason})
    except APIError:
        return visit(state, "handoff", reply=prompts.NO_TICKET_MESSAGE, pending=None, next="finish")
    text = prompts.handoff_message(state["message"], created["callback_eta"])
    return visit(state, "handoff", reply=text, pending=None, next="finish", ticket=created,
                 actions=state.get("actions", []) + [f"ticket créé ({created['ticket_id']})"])


def finish(state: State) -> dict:
    return {"history": state.get("history", []) + [f"agent : {state.get('reply', '')}"]}


NODES = [precheck, planner, identify, collect, gesture, post_review, respond, technical_check, propose, confirm,
         side_question, replay, handoff, finish]


ROUTES = {  # every node's possible next nodes; any other transition is rejected by LangGraph
    "precheck": ["handoff", "identify", "planner", "technical_check", "confirm", "replay"],
    "planner": ["handoff", "finish", "collect"],
    "identify": ["handoff", "finish", "collect", "planner"],
    "collect": ["handoff", "gesture", "post_review"],
    "gesture": ["handoff", "post_review"],
    "post_review": ["handoff", "technical_check", "respond"],
    "respond": ["handoff", "finish"],
    "technical_check": ["handoff", "finish", "propose"],
    "propose": ["handoff", "finish"],
    "confirm": ["handoff", "finish", "propose", "side_question", "planner"],
    "side_question": ["finish"],
    "replay": ["handoff", "finish"],
    "handoff": ["finish"],
}


def build(checkpointer=None):
    graph = StateGraph(State)
    for node in NODES:
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "precheck")
    for name, targets in ROUTES.items():
        graph.add_conditional_edges(name, lambda state: state["next"], {t: t for t in targets})
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_turn(graph, thread_id: str, message: str) -> dict:
    return graph.invoke({"message": message, "next": ""}, {"configurable": {"thread_id": thread_id}})
