import json
import logging
import uuid
from typing import Annotated, Any, TypedDict

import httpx
import openai
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from neova import llm
from neova.agent import prompts
from neova.agent.client import APIClient, APIError, APIUnavailable
from neova.rag.index import format_results, load_index, retrieve

logger = logging.getLogger(__name__)

_api_client = APIClient()
_index: tuple[list, Any] | None = None


def _get_index() -> tuple[list, Any]:
    global _index
    if _index is None:
        _index = load_index()
    return _index


MAX_TOOL_CALLS = 8


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session: str | None
    customer_id: str | None
    facts: dict[str, Any]
    passages: list[str]
    proposal: dict[str, Any] | None
    actions: list[str]
    tool_calls: int
    identity_failures: int
    needs_done: bool
    needs: dict[str, Any] | None
    escalated: bool
    next: str | None
    after_tools: str | None


@tool
def chercher_documentation(question: str, historical: bool = False) -> None:
    """Recherche dans la documentation publique de Néova : offres, tarifs, FAQ, conditions, procédures.
    historical est vrai uniquement si le client demande des informations sur une ancienne offre ou un ancien tarif."""


@tool
def verifier_identite(customer_id: str, phone: str) -> None:
    """Vérifie le client avec son numéro de client et le numéro de téléphone du contrat,
    tels qu'il les a fournis. Ouvre l'accès à son dossier."""


@tool
def dossier_client() -> None:
    """Le dossier du client identifié. Nécessite une identité vérifiée."""


@tool
def incidents_zone() -> None:
    """Les incidents réseau dans la zone du client identifié."""


@tool
def proposer_rendez_vous(
    reason: str, override_reason: str | None = None, another_slot: bool = False
) -> None:
    """Prépare un rendez-vous avec un technicien sans le réserver. Ne l'appeler qu'une fois que
    la documentation et le client ont établi qu'une visite est justifiée.
    another_slot est vrai lorsque le client refuse le créneau proposé."""


@tool
def confirmer_rendez_vous() -> None:
    """Réserve le rendez-vous qui a été proposé. Uniquement après que le client a clairement
    accepté. Rien n'est réservé si sa réponse n'est pas une acceptation claire."""


@tool
def verifier_geste_commercial() -> None:
    """Décide si un geste commercial peut être accordé au client identifié,
    et retourne le résultat pour le lui communiquer."""


class PrecheckResult(BaseModel):
    matched_items: list[int]
    evidence: str


def precheck(state: State) -> dict[str, Any]:
    if not state.get("messages"):
        return {"next": "agent"}
    last_message = state["messages"][-1]
    if last_message.type != "human":
        return {"next": "agent"}

    blocks = [
        ("message_client", last_message.content),
        ("contexte", [m.content for m in state["messages"][:-1] if m.type == "human"]),
    ]
    sys_msg, human_msg = prompts.build_messages("precheck", blocks)

    try:
        result = llm.structured_answer(
            messages=[SystemMessage(content=sys_msg), HumanMessage(content=human_msg)],
            output_model=PrecheckResult,
            temperature=0.0,
        )
    except (openai.APIError, httpx.HTTPError) as e:
        logger.warning(f"precheck failed: {e}")
        return {"next": "agent"}

    valid = False
    if result.matched_items and result.evidence:
        evidence = result.evidence.lower()
        if evidence in last_message.content.lower():
            valid = True
        else:
            for m in state["messages"][:-1]:
                if m.type == "human" and evidence in m.content.lower():
                    valid = True
                    break

    if valid:
        return {"next": "escalade"}
    else:
        return {"next": "agent"}


def agent(state: State) -> dict[str, Any]:
    blocks = [
        ("date", prompts.get_current_date_string()),
        ("faits", state.get("facts", {})),
        ("actions", state.get("actions", [])),
    ]
    sys_msg, _ = prompts.build_messages("agent", blocks)

    tools = [
        chercher_documentation,
        verifier_identite,
        dossier_client,
        incidents_zone,
        proposer_rendez_vous,
        confirmer_rendez_vous,
        verifier_geste_commercial,
    ]

    client = llm.create_chat_client(temperature=0.0).bind_tools(tools)

    messages = [SystemMessage(content=sys_msg)] + state.get("messages", [])

    try:
        response = llm.invoke_with_retry(client, messages)
    except (openai.APIError, httpx.HTTPError):
        return {"next": "postreview"}

    if response.tool_calls:
        if state.get("tool_calls", 0) < MAX_TOOL_CALLS:
            return {
                "messages": [response],
                "next": "outils",
                "after_tools": "agent",
                "tool_calls": state.get("tool_calls", 0) + len(response.tool_calls),
            }
        else:
            response.tool_calls = []
            if not response.content:
                response.content = "Je ne peux pas vous répondre pour le moment."
            return {"messages": [response], "next": "postreview"}
    else:
        if not response.content:
            response.content = "Je ne peux pas vous répondre pour le moment."
        return {"messages": [response], "next": "postreview"}


class ConfirmationResult(BaseModel):
    confirmed: bool
    evidence: str


class GesteCommercialResult(BaseModel):
    is_gesture_request: bool
    condition_status: list[str]
    decision: str
    cap_row: int | None
    escalate_n2: bool
    out_of_scope: bool
    customer_outcome: str
    redirect_to: str
    internal_note: str


def outils(state: State) -> dict[str, Any]:
    after_tools = state.get("after_tools")
    if after_tools not in ("agent", "postreview"):
        raise ValueError(f"outils reached without a destination: after_tools={after_tools!r}")

    tool_messages = []
    actions = list(state.get("actions", []))
    facts = dict(state.get("facts", {}))
    proposal = dict(state.get("proposal", {})) if state.get("proposal") else None
    passages = list(state.get("passages", []))
    customer_id = state.get("customer_id")
    identity_failures = state.get("identity_failures", 0)
    next_node = after_tools
    new_after_tools = after_tools

    needs = state.get("needs") or {}
    if needs:
        if state.get("needs_done"):
            return {"next": after_tools, "needs": None}

        if "document" in needs:
            args = {"question": needs["document"]}
            call_id = str(uuid.uuid4())
            call = {
                "id": call_id,
                "name": "chercher_documentation",
                "args": args,
                "type": "tool_call",
            }
            ai_msg = AIMessage(content="", tool_calls=[call])

            chunks, vectors = _get_index()
            historical = args.get("historical", False)
            contract_start = None
            if "dossier" in facts and "contract_start_date" in facts["dossier"]:
                contract_start = facts["dossier"]["contract_start_date"]

            results = retrieve(
                chunks,
                vectors,
                args["question"],
                historical=historical,
                contract_start=contract_start,
            )
            result_content = format_results(results)

            for r in results:
                passages.append(r["chunk"].chunk_id)
            passages = passages[-12:]

            tool_msg = ToolMessage(content=result_content, tool_call_id=call_id)
            return {
                "messages": [ai_msg, tool_msg],
                "passages": passages,
                "needs": None,
                "needs_done": True,
                "next": after_tools,
            }
        elif "fact" in needs and needs["fact"] == "dossier":
            call_id = str(uuid.uuid4())
            call = {"id": call_id, "name": "dossier_client", "args": {}, "type": "tool_call"}
            ai_msg = AIMessage(content="", tool_calls=[call])

            try:
                data = _api_client.get_customer()
                facts["dossier"] = data
                result_content = json.dumps(data, ensure_ascii=False)
            except (APIError, APIUnavailable) as e:
                if isinstance(e, APIError):
                    result_content = e.detail
                else:
                    result_content = "Le service est temporairement indisponible."

            tool_msg = ToolMessage(content=result_content, tool_call_id=call_id)
            return {
                "messages": [ai_msg, tool_msg],
                "facts": facts,
                "needs": None,
                "needs_done": True,
                "next": after_tools,
            }
        return {"next": after_tools, "needs": None}

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_calls_to_execute = last_message.tool_calls
    else:
        return {"next": after_tools, "after_tools": new_after_tools}

    for call in tool_calls_to_execute:
        call_id = call["id"]
        name = call["name"]
        args = call["args"]

        result_content = ""

        try:
            if name == "chercher_documentation":
                chunks, vectors = _get_index()
                historical = args.get("historical", False)
                contract_start = None
                if "dossier" in facts and "contract_start_date" in facts["dossier"]:
                    contract_start = facts["dossier"]["contract_start_date"]

                results = retrieve(
                    chunks,
                    vectors,
                    args.get("question", ""),
                    historical=historical,
                    contract_start=contract_start,
                )
                result_content = format_results(results)

                for r in results:
                    passages.append(r["chunk"].chunk_id)
                passages = passages[-12:]

            elif name == "verifier_identite":
                try:
                    customer = _api_client.verify(args["customer_id"], args["phone"])

                    if customer_id and customer_id != customer["customer_id"]:
                        facts = {}
                        proposal = None
                        passages = []

                    customer_id = customer["customer_id"]
                    result_content = "Identité vérifiée avec succès."

                except APIError as e:
                    identity_failures += 1
                    if identity_failures >= 2:
                        result_content = "L'identité ne peut pas être vérifiée."
                        next_node = "escalade"
                        new_after_tools = None
                    else:
                        result_content = f"Erreur de vérification: {e.detail}"

            elif name == "dossier_client":
                data = _api_client.get_customer()
                facts["dossier"] = data
                result_content = json.dumps(data, ensure_ascii=False)

            elif name == "incidents_zone":
                data = _api_client.get_incidents()
                facts["incidents"] = data
                result_content = json.dumps(data, ensure_ascii=False)

            elif name == "proposer_rendez_vous":
                data = _api_client.propose(
                    reason=args.get("reason", ""),
                    override_reason=args.get("override_reason"),
                    another_slot=args.get("another_slot", False),
                )
                key = str(uuid.uuid4())
                proposal = {
                    "offer": data,
                    "idempotency_key": key,
                    "proposal_id": data.get("proposal_id"),
                }
                result_content = json.dumps(data, ensure_ascii=False)

            elif name == "confirmer_rendez_vous":
                if not proposal:
                    result_content = (
                        "Il faut d'abord proposer un créneau avant de pouvoir le confirmer."
                    )
                else:
                    human_msg = None
                    for m in reversed(state["messages"]):
                        if m.type == "human":
                            human_msg = m.content
                            break

                    if not human_msg:
                        human_msg = ""

                    blocks = [
                        ("message_client", human_msg),
                        ("proposition", proposal.get("offer", {})),
                        ("contexte", [m.content for m in state["messages"] if m.type == "human"]),
                    ]

                    sys_msg, text = prompts.build_messages("confirmation", blocks)

                    decision = llm.structured_answer(
                        messages=[SystemMessage(content=sys_msg), HumanMessage(content=text)],
                        output_model=ConfirmationResult,
                        temperature=0.0,
                    )

                    if not decision.confirmed:
                        result_content = "Le client n'a pas confirmé la proposition de rendez-vous."
                    else:
                        try:
                            res = _api_client.book(
                                proposal["proposal_id"], proposal["idempotency_key"]
                            )
                            actions.append(f"Rendez-vous confirmé: {res.get('appointment_id')}")
                            proposal = None
                            result_content = "Le rendez-vous a été réservé avec succès."
                        except APIUnavailable:
                            result_content = "Le service de réservation est temporairement indisponible. La confirmation n'a pas pu aboutir."
                        except APIError as e:
                            result_content = f"Erreur lors de la réservation: {e.detail}"

            elif name == "verifier_geste_commercial":
                blocks = [("faits", facts)]
                sys_msg, text = prompts.build_messages("geste_commercial", blocks)

                decision = llm.structured_answer(
                    messages=[SystemMessage(content=sys_msg), HumanMessage(content=text)],
                    output_model=GesteCommercialResult,
                    temperature=0.0,
                )

                facts["geste_commercial"] = decision.model_dump()
                result_content = (
                    f"Issue: {decision.customer_outcome}\nOrientation: {decision.redirect_to}"
                )

            else:
                result_content = f"Outil inconnu: {name}"

        except (APIError, APIUnavailable) as e:
            if isinstance(e, APIError):
                result_content = e.detail
            else:
                result_content = "Le service est temporairement indisponible."
            logger.error(f"API Error in tool {name}: {e}")

        tool_messages.append(ToolMessage(content=result_content, tool_call_id=call_id))

    return {
        "messages": tool_messages,
        "actions": actions,
        "facts": facts,
        "proposal": proposal,
        "passages": passages,
        "customer_id": customer_id,
        "identity_failures": identity_failures,
        "next": next_node,
        "after_tools": new_after_tools,
    }


def postreview(state: State) -> None:
    pass


def escalade(state: State) -> None:
    pass


def fin(state: State) -> None:
    pass


builder = StateGraph(State)
builder.add_node("precheck", precheck)
builder.add_node("agent", agent)
builder.add_node("outils", outils)
builder.add_node("postreview", postreview)
builder.add_node("escalade", escalade)
builder.add_node("fin", fin)

builder.set_entry_point("precheck")
builder.add_conditional_edges("precheck", lambda state: state["next"], ["agent", "escalade"])
builder.add_conditional_edges("agent", lambda state: state["next"], ["outils", "postreview"])
builder.add_conditional_edges("outils", lambda state: state["next"], ["agent", "postreview"])
builder.add_conditional_edges(
    "postreview", lambda state: state["next"], ["outils", "agent", "escalade", "fin"]
)
builder.add_edge("escalade", "fin")
builder.add_edge("fin", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
