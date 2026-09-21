"""Graph routes and tool-call order, offline: a scripted agent LLM that emits tool calls, scripted
verdicts for the checks, and the real API in process."""
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from neova.agent import graph, prompts
from neova.agent.tools import APIUnavailable, NeovaAPI
from neova.api.app import app
from neova.api.store import store

AHMED = ("NEO-10467", "0778115402")      # fibre, 69007, degraded incident in his zone, 61,98 € due
CAMILLE = ("NEO-88213", "0612840193")    # fibre, 75019, active outage, 0 € due
PATRICK = ("NEO-40318", "0698441207")    # fibre, 59000, no incident
PRO = ("NEO-71925", "0142886310")


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c-{uuid.uuid4().hex[:6]}"}])


def answer(text, sources=()):
    return AIMessage(content=text)


class Script:
    """The agent's tool calls, in order, and the verdicts of the check nodes."""

    def __init__(self):
        self.steps, self.calls, self.precheck = [], [], []
        self.confirmation = graph.Confirmation(decision="yes", preference="")
        self.gesture = None
        self.ticket = SimpleNamespace(category="technical", motif="motif", summary="résumé", actions_taken=[], urgency="normal")
        self.review = prompts.PostReviewVerdict(candidates=[], documents_cover=True, advisor_conditions=[], reason="")

    def agent(self, messages):
        assert self.steps, "the scripted agent has no step left"
        step = self.steps.pop(0)
        self.calls.extend(c["name"] for c in step.tool_calls)
        return step

    def ask(self, name, schema, *blocks):
        self.calls.append(name)
        return {"precheck": lambda: prompts.PrecheckVerdict(matched_items=self.precheck, evidence=""),
                "post_review": lambda: self.review, "confirmation": lambda: self.confirmation,
                "gesture": lambda: self.gesture, "ticket": lambda: self.ticket}[name]()


@pytest.fixture
def script(monkeypatch):
    s = Script()
    monkeypatch.setattr(graph, "agent_llm", s.agent)
    monkeypatch.setattr(prompts, "ask", s.ask)
    monkeypatch.setattr(prompts, "handoff_message", lambda message, eta: s.calls.append("handoff_message") or f"transmis, rappel {eta}")
    monkeypatch.setattr(graph, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(graph, "index", lambda: SimpleNamespace(by_id={}))
    for method in ("verify", "customer", "incidents", "slots", "propose_appointment", "book_appointment", "create_ticket"):
        original = getattr(NeovaAPI, method)
        def recorded(self, *args, _original=original, _name=method, **kwargs):
            s.calls.append(f"api:{_name}")
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(NeovaAPI, method, recorded)
    monkeypatch.setattr(graph, "HTTP", TestClient(app))
    return s


@pytest.fixture
def talk():
    agent, thread = graph.build(), str(uuid.uuid4())
    return lambda message: graph.run_turn(agent, thread, message)


def api_calls(script):
    return [c[4:] for c in script.calls if c.startswith("api:")]


# --- handoff -------------------------------------------------------------------------------

def test_mandatory_handoff_before_any_tool_and_one_ticket_per_conversation(script, talk):
    script.precheck = [3]
    state = talk("Mon père est décédé")
    assert state["trace"] == ["precheck", "handoff"] and state["reply"].startswith("transmis, rappel")
    assert script.calls[:4] == ["precheck", "ticket", "api:create_ticket", "handoff_message"]
    assert len(store.data["tickets"]) == 1
    script.precheck = []
    assert talk("allô ?")["reply"].startswith("Votre demande est déjà transmise") and len(store.data["tickets"]) == 1


def test_agent_can_request_a_handoff_and_the_reason_reaches_the_ticket(script, talk, monkeypatch):
    seen = {}
    original = prompts.draft_ticket
    monkeypatch.setattr(prompts, "draft_ticket", lambda t, a, r="": seen.update(reason=r) or original(t, a, r))
    script.steps = [call("search_documents", question="Netflix inclus ?"), call("request_handoff", reason="réponse non établie")]
    state = talk("Netflix est inclus ?")
    assert state["reply"].startswith("transmis") and seen["reason"] == "réponse non établie"
    assert store.data["tickets"][0]["actions_taken"] == []


def test_failed_ticket_never_claims_a_transfer(script, talk):
    script.precheck = [1]
    script.ticket = SimpleNamespace(category="not-a-category", motif="m", summary="s", actions_taken=[], urgency="normal")
    assert talk("Je veux mes données RGPD")["reply"] == prompts.NO_TICKET_MESSAGE and store.data["tickets"] == []


def test_uncertain_ticket_is_replayed_with_the_same_key(script, talk, monkeypatch):
    script.precheck = [3]
    original, lost = NeovaAPI.create_ticket, iter([True])
    def created_then_lost(self, *a, **k):
        result = original(self, *a, **k)
        if next(lost, False):
            raise APIUnavailable("response lost")
        return result
    monkeypatch.setattr(NeovaAPI, "create_ticket", created_then_lost)
    assert talk("Mon père est décédé")["reply"] == graph.UNCERTAIN_TICKET
    script.precheck = []
    assert talk("allô ?")["reply"].startswith("transmis") and len(store.data["tickets"]) == 1


# --- answers, sources, guard ---------------------------------------------------------------

def test_answer_with_sources_from_this_turn_passes_the_checks(script, talk, monkeypatch):
    chunk = SimpleNamespace(chunk_id="grille#fibre", text="Fibre 1 Gb/s : 39,99 €")
    monkeypatch.setattr(graph, "retrieve", lambda *a, **k: [SimpleNamespace(chunk=chunk)])
    monkeypatch.setattr(graph, "index", lambda: SimpleNamespace(by_id={"grille#fibre": chunk}))
    script.steps = [call("search_documents", question="prix fibre 1 Gb/s"), answer("39,99 € par mois.", ["grille#fibre"])]
    state = talk("Combien coûte la fibre 1 Gb/s ?")
    assert state["reply"] == "39,99 € par mois." and state["trace"][-2:] == ["post_review", "guard"]
    assert "post_review" in script.calls and store.data["tickets"] == []


def test_a_number_without_any_source_or_fact_is_a_handoff(script, talk):
    script.steps = [answer("La fibre coûte 39,99 €.")]
    assert talk("prix ?")["reply"].startswith("transmis") and len(store.data["tickets"]) == 1


def test_empty_replies_are_capped_then_handed_off(script, talk):
    script.steps = [AIMessage(content="")] * graph.MAX_TOOL_CALLS
    state = talk("bonjour")
    assert state["reply"].startswith("transmis") and "trop d'étapes" in state["reason"]


def test_greeting_without_tools_and_without_ticket(script, talk):
    script.steps = [answer("Bonjour, je peux vous aider pour votre box, une facture ou un rendez-vous.")]
    assert talk("bonjour")["reply"].startswith("Bonjour") and store.data["tickets"] == [] and api_calls(script) == []


def test_post_review_can_send_an_answer_to_a_human(script, talk, monkeypatch):
    chunk = SimpleNamespace(chunk_id="faq#x", text="…")
    monkeypatch.setattr(graph, "retrieve", lambda *a, **k: [SimpleNamespace(chunk=chunk)])
    monkeypatch.setattr(graph, "index", lambda: SimpleNamespace(by_id={"faq#x": chunk}))
    script.review = prompts.PostReviewVerdict(candidates=[], documents_cover=False, advisor_conditions=[], reason="pas couvert")
    script.steps = [call("search_documents", question="q"), answer("réponse", [])]
    assert talk("question")["reply"].startswith("transmis")


# --- identification ------------------------------------------------------------------------

def test_reads_need_a_verified_session_then_work(script, talk):
    script.steps = [call("get_customer"), answer("Pour accéder à votre dossier, donnez-moi votre numéro client et téléphone.")]
    talk("Quels sont mes frais de résiliation ?")
    assert api_calls(script) == []            # nothing read without a session
    script.steps = [call("verify_customer", customer_id=CAMILLE[0], phone=CAMILLE[1]), call("get_customer"),
                    answer("Votre engagement est terminé : aucun frais.")]
    state = talk("NEO-88213 0612840193")
    assert api_calls(script) == ["verify", "customer"] and state["session"]
    assert "full_name" not in str(state["facts"]) and state["facts"]["client"]["balance_due"] == 0


def test_two_refused_verifications_hand_off(script, talk):
    script.steps = [call("verify_customer", customer_id="NEO-00000", phone="0600000000"),
                    answer("Je n'ai pas pu vérifier, pouvez-vous redonner vos identifiants ?"),
                    call("verify_customer", customer_id="NEO-00000", phone="0600000000")]
    talk("NEO-00000 0600000000")
    assert talk("NEO-00000 0600000000")["reply"].startswith("transmis")


def test_pro_account_hands_off(script, talk):
    script.steps = [call("verify_customer", customer_id=PRO[0], phone=PRO[1])]
    assert talk("NEO-71925 0142886310")["reply"].startswith("transmis") and store.data["tickets"][0]["customer_id"] == PRO[0]


# --- booking -------------------------------------------------------------------------------

def identify_and_propose(script, talk, who=PATRICK):
    script.steps = [call("verify_customer", customer_id=who[0], phone=who[1]), call("get_incidents"),
                    call("propose_appointment", reason="no_internet"),
                    answer("Je peux vous proposer le vendredi 28 août entre 9h et 11h. Confirmez-vous (oui / non) ?")]
    return talk(f"{who[0]} {who[1]} plus d'internet, voyant rouge fixe après deux redémarrages, je veux un technicien")


def test_booking_order_and_nothing_booked_before_yes(script, talk):
    state = identify_and_propose(script, talk)
    assert api_calls(script) == ["verify", "customer", "incidents", "slots", "propose_appointment"]
    assert state["proposal"] and store.data["appointments"] == []
    key = state["proposal"]["key"]
    script.steps = [call("book_appointment")]
    booked = talk("oui")
    assert booked["reply"].startswith("C'est confirmé") and booked["proposal"] is None
    (appointment,) = store.data["appointments"]
    assert appointment["slot_id"] == "SLOT-5D22" and store.idempotency_keys.get(f"appointments:{PATRICK[0]}:{key}")


def test_book_without_an_explicit_yes_is_refused_by_code(script, talk):
    identify_and_propose(script, talk)
    script.confirmation = graph.Confirmation(decision="question", preference="")
    script.steps = [call("book_appointment"), answer("Les 69 € ne s'appliquent que si… Confirmez-vous ?")]
    state = talk("c'est quoi les 69 € ?")
    assert store.data["appointments"] == [] and state["proposal"] is not None
    assert "book_appointment" in script.calls and "book_appointment" not in api_calls(script)


def test_book_without_a_proposal_is_refused_by_code(script, talk):
    script.steps = [call("verify_customer", customer_id=PATRICK[0], phone=PATRICK[1]), call("book_appointment"),
                    answer("Je dois d'abord vous proposer un créneau.")]
    talk("NEO-40318 0698441207 réservez-moi un technicien")
    assert store.data["appointments"] == [] and "book_appointment" not in api_calls(script)


def test_uncertain_booking_is_replayed_with_the_same_key_and_books_once(script, talk, monkeypatch):
    identify_and_propose(script, talk)
    original, lost = NeovaAPI.book_appointment, iter([True])
    def booked_then_lost(self, *a, **k):
        result = original(self, *a, **k)
        if next(lost, False):
            raise APIUnavailable("response lost")
        return result
    monkeypatch.setattr(NeovaAPI, "book_appointment", booked_then_lost)
    script.steps = [call("book_appointment")]
    assert talk("oui")["reply"] == graph.UNCERTAIN_BOOKING
    assert talk("alors ?")["reply"].startswith("C'est confirmé") and len(store.data["appointments"]) == 1


def test_proposal_refused_by_the_api_is_explained_to_the_agent(script, talk):
    script.steps = [call("verify_customer", customer_id=CAMILLE[0], phone=CAMILLE[1]),
                    call("propose_appointment", reason="no_internet"),
                    answer("Une panne réseau est en cours dans votre zone, un technicien ne pourrait rien faire.")]
    state = talk("NEO-88213 0612840193 je veux un technicien")
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    assert any("panne active" in m.content for m in tool_messages) and not state.get("proposal")


# --- gesture -------------------------------------------------------------------------------

def gesture_verdict(**update):
    base = dict(decision="eligible", condition_status=["met", "met", "met"], cap_row=1, escalate_n2=False,
                out_of_scope=False, internal_note="note", customer_outcome="credit_next_invoice", redirect_to=[])
    return prompts.GestureVerdict(**{**base, **update})


def test_gesture_under_review_goes_to_a_human_without_criteria(script, talk):
    script.gesture = gesture_verdict()
    script.steps = [call("verify_customer", customer_id=CAMILLE[0], phone=CAMILLE[1]), call("assess_gesture", request="remise panne")]
    state = talk("NEO-88213 0612840193 ma fibre est coupée, je veux un geste")
    assert state["reply"].startswith("transmis") and state["gesture"]["outcome"] == "under_review"
    assert len(store.data["tickets"]) == 1


def test_refused_gesture_is_announced_without_criteria(script, talk):
    script.gesture = gesture_verdict(decision="not_eligible", condition_status=["not_met", "met", "met"],
                                     redirect_to=["payment_plan"])
    script.steps = [call("verify_customer", customer_id=AHMED[0], phone=AHMED[1]), call("assess_gesture", request="remise"),
                    answer("Aucun geste ne peut être accordé. Je peux vous proposer un échéancier de paiement.")]
    state = talk("NEO-10467 0778115402 je veux une remise")
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    assert any("échéancier" in m.content and "impayé" not in m.content for m in tool_messages)
    assert state["reply"].startswith("Aucun geste") and store.data["tickets"] == []


def test_another_slot_is_a_new_proposal_with_a_new_key(script, talk):
    first = identify_and_propose(script, talk)["proposal"]
    script.steps = [call("propose_appointment", reason="no_internet", another_slot=True),
                    answer("Je peux vous proposer le lundi 31 août. Confirmez-vous ?")]
    second = talk("un autre jour svp")["proposal"]
    assert second["proposal"]["slot"]["slot_id"] != first["proposal"]["slot"]["slot_id"] and second["key"] != first["key"]
    assert store.data["appointments"] == []


def test_identification_only_turn_skips_the_after_review_check(script, talk):
    script.steps = [call("verify_customer", customer_id=CAMILLE[0], phone=CAMILLE[1]), answer("Merci, que puis-je faire pour vous ?")]
    state = talk("NEO-88213 0612840193")
    assert "post_review" not in script.calls and state["reply"].startswith("Merci") and store.data["tickets"] == []
