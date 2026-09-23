from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

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


def precheck(state: State) -> None:
    pass


def agent(state: State) -> None:
    pass


def outils(state: State) -> None:
    pass


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
