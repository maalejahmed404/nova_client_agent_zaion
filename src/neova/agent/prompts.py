"""The five nodes that use the internal documents: static prompt, imposed output, checks. Any
failure returns None or a safe fallback, and the caller then hands off to a human."""
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from neova import llm
from neova.rag.ingest import fold

log = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).with_name("templates")
NO_TICKET_MESSAGE = (
    "Votre demande doit être traitée par un conseiller, mais je n'ai pas pu la transmettre pour le moment. "
    "Merci de renouveler votre demande dans quelques instants."
)
FALLBACK_MESSAGE = "Votre demande est transmise à un conseiller, qui vous recontactera {eta}."


class PrecheckVerdict(BaseModel):
    matched_items: list[int]
    evidence: str


class Element(BaseModel):
    element: str
    established: bool


class Candidate(BaseModel):
    item: int
    elements: list[Element]


class Condition(BaseModel):
    condition: str
    met: Literal["yes", "no", "unknown"]


class Need(BaseModel):
    kind: Literal["document", "fact"]
    query: str


class PostReviewVerdict(BaseModel):
    candidates: list[Candidate]
    needs: list[Need]             # what is missing to judge a candidate: a document search or a customer fact
    documents_cover: bool
    advisor_conditions: list[Condition]
    unsupported_claims: list[str]  # statements of the draft reply that the documents and facts do not establish
    reason: str

    @property
    def matched_items(self) -> list[int]:
        """A situation counts only if every one of its elements is established."""
        return sorted(c.item for c in self.candidates if c.elements and all(e.established for e in c.elements))

    @property
    def can_conclude(self) -> bool:
        """Transfer only for an established situation the documents cannot settle, or when the documents
        send the request to an advisor and no documented condition for it is known to fail."""
        advisor_needed = bool(self.advisor_conditions) and all(c.met != "no" for c in self.advisor_conditions)
        return (not self.matched_items or self.documents_cover) and not advisor_needed


class GestureVerdict(BaseModel):
    is_gesture_request: bool      # false: the customer asks something else (payment plan, dispute, information)
    decision: Literal["eligible", "not_eligible", "needs_review"]
    condition_status: list[Literal["met", "not_met", "unknown"]]
    cap_row: int | None
    escalate_n2: bool
    out_of_scope: bool

    @field_validator("cap_row", mode="before")
    @classmethod
    def _row_or_none(cls, value):
        return value if isinstance(value, int) else None   # models write "aucune" or "" when no cap applies
    internal_note: str
    customer_outcome: Literal["credit_next_invoice", "refused", "under_review"]
    redirect_to: list[Literal["payment_plan", "incident_follow_up", "technician_appointment"]]


class HandoffTicket(BaseModel):
    category: Literal["billing_dispute", "technical", "termination", "commercial_gesture", "other"]
    motif: str
    summary: str
    actions_taken: list[str]
    urgency: Literal["low", "normal", "high"]


@lru_cache
def template(name: str) -> str:
    return (TEMPLATES_DIR / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")


def internal_rules(name: str) -> str:
    return template(name).split("<regles_internes>")[1].split("</regles_internes>")[0]


def item_count(name: str) -> int:
    return len(re.findall(r"^\d+\. ", internal_rules(name), re.M))


def data(tag: str, value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)
    return f"<{tag}>\n{text.replace('<', '‹').replace('>', '›')}\n</{tag}>"


def ask(name: str, schema, *blocks: str):
    try:
        return llm.structured_call([("system", template(name)), ("human", "\n\n".join(blocks))], schema)
    except Exception as exc:
        log.warning("%s call failed, handing off: %s", name, exc)
        return None


def precheck(message: str, context: str = "") -> PrecheckVerdict | None:
    v = ask("precheck", PrecheckVerdict, data("contexte", context or "(aucun)"), data("message_client", message))
    if v is None or any(not 1 <= i <= item_count("precheck") for i in v.matched_items):
        return None
    if v.evidence and fold(v.evidence) not in fold(f"{context} {message}"):
        return None
    return v


def post_review(message: str, context: str, draft: str, facts: dict, documents: str,
                final: bool = False) -> PostReviewVerdict | None:
    """final=True is the second pass, after the missing pieces were fetched: the verdict must be given."""
    blocks = [data("contexte", context or "(aucun)"), data("message_client", message),
              data("reponse_proposee", draft), data("faits", facts), data("documents", documents or "(aucun)")]
    if final:
        blocks.append("Toutes les pièces disponibles ont été réunies : rends le verdict, needs doit rester vide.")
    v = ask("post_review", PostReviewVerdict, *blocks)
    if v is None or any(not 1 <= c.item <= item_count("post_review") for c in v.candidates):
        return None
    return v


def gesture(message: str, facts: dict) -> GestureVerdict | None:
    """The LLM applies the policy to the facts; the code then overrides it wherever the data
    decides. needs_review always means a handoff. No credit is ever announced as applied: no tool
    applies one, so a gesture the customer may get goes to an advisor."""
    v = ask("gesture", GestureVerdict, data("faits", facts), data("message_client", message))
    if v is None:
        return None
    balance = (facts.get("client") or {}).get("balance_due")
    if v.out_of_scope or v.escalate_n2 or balance is None:
        decision = "needs_review"
    elif balance > 0 or "not_met" in v.condition_status:
        decision = "not_eligible"
    else:
        decision = "needs_review"  # the gesture history is unavailable, so eligibility can never be confirmed
    return v.model_copy(update={"decision": decision,
                                "customer_outcome": "refused" if decision == "not_eligible" else "under_review"})


def draft_ticket(conversation: str, actions: list[str], reason: str = "") -> dict:
    """actions = what the agent really did; reason = why the graph decided to transfer (context for the
    motif, never listed as an action)."""
    v = ask("ticket", HandoffTicket, data("contexte", conversation), data("actions", actions), data("motif_interne", reason))
    if v is None:
        return {"category": "other", "summary": f"{reason} — {conversation[-1000:]}".strip(" —"),
                "actions_taken": actions, "urgency": "normal"}
    return {"category": v.category, "summary": f"{v.motif} — {v.summary}", "actions_taken": v.actions_taken,
            "urgency": v.urgency}


def leaked(text: str) -> list[str]:
    """Sentences of the internal rules (six words or more) found in a customer-facing text."""
    folded, found = fold(text), []
    for name in ("precheck", "post_review", "gesture", "ticket"):
        for sentence in re.split(r"[.;:!?\n]", fold(internal_rules(name).replace("**", ""))):
            sentence = re.sub(r"^(-|\d+)\s*", "", sentence.strip())
            if len(sentence.split()) >= 6 and sentence in folded:
                found.append(sentence)
    return found


def handoff_message(message: str, callback_eta: str | None) -> str:
    """callback_eta is the value returned by the ticket API; None means no ticket was created."""
    if callback_eta is None:
        return NO_TICKET_MESSAGE
    try:
        reply = llm.invoke_with_retry(llm.get_chat(), [
            ("system", template("handoff_message")),
            ("human", data("delai", callback_eta) + "\n\n" + data("message_client", message)),
        ])
        text = reply.content.strip()
    except Exception as exc:
        log.warning("handoff_message call failed, using the fallback: %s", exc)
        text = ""
    if callback_eta not in text or leaked(text):
        return FALLBACK_MESSAGE.format(eta=callback_eta)
    return text
