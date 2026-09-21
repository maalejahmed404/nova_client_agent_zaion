"""Graph routes and tool-call order, offline: scripted LLM answers, the real API in process."""
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from neova.agent import graph, prompts
from neova.agent.graph import Confirmation, Plan, TechnicalCheck
from neova.agent.tools import APIUnavailable, NeovaAPI
from neova.api.app import app
from neova.api.store import store
from neova.rag.answer import Answer

AHMED = "NEO-10467 0778115402"      # fibre, 69007, no outage
CAMILLE = "NEO-88213 0612840193"    # fibre, 75019, balance 0


class Script:
    """What each LLM step answers; every call and every tool call is recorded in order."""

    def __init__(self):
        self.calls = []
        self.plan = lambda text: Plan(query=text, route="info", historical=False,
                                      needs_customer=False, clarification="")
        self.precheck = []
        self.technical = TechnicalCheck(justified="yes", reason="no_internet", override="none", message="")
        self.confirmation = Confirmation(decision="yes", preference="")
        self.gesture = None
        self.ticket = SimpleNamespace(category="technical", motif="motif", summary="résumé",
                                      actions_taken=[], urgency="normal")

    def ask(self, name, schema, *blocks):
        self.calls.append(name)
        text = "\n".join(blocks)
        if "<message_client>" in text:
            text = text.split("<message_client>\n")[1].split("\n</message_client>")[0]
        return {
            "precheck": lambda: prompts.PrecheckVerdict(matched_items=self.precheck, evidence=""),
            "plan": lambda: self.plan(text),
            "post_review": lambda: prompts.PostReviewVerdict(candidates=[], documents_cover=True, advisor_conditions=[], reason=""),
            "technician_check": lambda: self.technical,
            "confirmation": lambda: self.confirmation,
            "gesture": lambda: self.gesture,
            "ticket": lambda: self.ticket,
        }[name]()


@pytest.fixture
def script(monkeypatch):
    s = Script()
    monkeypatch.setattr(prompts, "ask", s.ask)
    monkeypatch.setattr(prompts, "handoff_message",
                        lambda message, eta: s.calls.append("handoff_message") or f"transmis, rappel {eta}")
    monkeypatch.setattr(graph, "answer", lambda question, results, facts=None, decision=None:
                        s.calls.append(f"answer:{question}") or Answer(in_corpus=True, answer=f"réponse ({decision})", sources=[]))
    monkeypatch.setattr(graph, "search", lambda plan, facts: [])
    monkeypatch.setattr(graph, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(graph, "index", lambda: SimpleNamespace(by_id={}))
    for method in ("verify", "customer", "incidents", "slots", "propose_appointment", "book_appointment", "create_ticket"):
        original = getattr(NeovaAPI, method)
        def recorded(self, *args, _original=original, _name=method, **kwargs):
            s.calls.append(_name)
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(NeovaAPI, method, recorded)
    monkeypatch.setattr(graph, "HTTP", TestClient(app))
    return s


@pytest.fixture
def talk():
    agent, thread = graph.build(), str(uuid.uuid4())
    return lambda message: graph.run_turn(agent, thread, message)


def tools(script):
    return [c for c in script.calls if c in ("verify", "customer", "incidents", "slots", "propose_appointment",
                                             "book_appointment", "create_ticket", "ticket", "handoff_message")]


def action_plan(text):
    return Plan(query=text, route="action", historical=False, needs_customer=True, clarification="")


# --- handoff -------------------------------------------------------------------------------

def test_mandatory_handoff_drafts_then_creates_then_announces_the_real_delay(script, talk):
    script.precheck = [3]
    state = talk("Mon père est décédé")
    assert tools(script) == ["ticket", "create_ticket", "handoff_message"]
    assert state["reply"].startswith("transmis, rappel ") and len(store.data["tickets"]) == 1
    assert "plan" not in script.calls and "verify" not in script.calls


def test_failed_ticket_never_claims_a_transfer(script, talk):
    script.precheck = [1]
    script.ticket = SimpleNamespace(category="not-a-category", motif="m", summary="s", actions_taken=[], urgency="normal")
    assert talk("Je veux mes données RGPD")["reply"] == prompts.NO_TICKET_MESSAGE
    assert store.data["tickets"] == []


def test_uncertain_ticket_is_replayed_with_the_same_key(script, talk, monkeypatch):
    script.precheck = [3]
    original, failures = NeovaAPI.create_ticket, iter([True])
    def created_then_lost(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if next(failures, False):
            raise APIUnavailable("response lost")
        return result
    monkeypatch.setattr(NeovaAPI, "create_ticket", created_then_lost)
    assert talk("Mon père est décédé")["reply"] == graph.UNCERTAIN_TICKET
    script.precheck = []
    assert talk("allô ?")["reply"].startswith("transmis")
    assert len(store.data["tickets"]) == 1


# --- identification and clarification ------------------------------------------------------

def test_identification_then_the_original_request_resumes(script, talk):
    script.plan = lambda text: Plan(query=text, route="info", historical=False,
                                    needs_customer=True, clarification="")
    assert talk("Combien pour résilier ?")["reply"] == graph.ASK_IDENTITY
    state = talk(CAMILLE)
    assert tools(script) == ["verify", "customer", "incidents"]
    assert "answer:Combien pour résilier ?" in script.calls and state["reply"].startswith("réponse")


def test_two_identity_failures_hand_off(script, talk):
    script.plan = lambda text: action_plan(text)
    talk("Je veux un technicien")
    assert talk("NEO-00000 0600000000")["reply"] == graph.IDENTITY_RETRY
    assert talk("NEO-00000 0600000000")["reply"].startswith("transmis")


def test_clarification_then_the_answer_uses_both_messages(script, talk):
    script.plan = lambda text: Plan(query=text, route="info", historical=False, needs_customer=False,
                                    clarification="" if "prix mensuel" in text else "Le prix en option ou l'indemnité ?")
    assert talk("Combien le décodeur ?")["reply"] == "Le prix en option ou l'indemnité ?"
    talk("le prix mensuel")
    assert "answer:Combien le décodeur ?\nle prix mensuel" in script.calls


# --- booking -------------------------------------------------------------------------------

def start_booking(script, talk):
    script.plan = lambda text: action_plan(text)
    talk("Je n'ai plus internet, je veux un technicien")
    return talk(AHMED)


def test_booking_order_and_nothing_booked_before_yes(script, talk):
    state = start_booking(script, talk)
    assert tools(script) == ["verify", "customer", "incidents", "slots", "propose_appointment"]
    assert state["pending"]["step"] == "awaiting_booking_confirmation" and "Confirmez-vous" in state["reply"]
    assert store.data["appointments"] == []
    proposal, key = state["pending"]["proposal"], state["pending"]["key"]
    booked = talk("oui")
    assert booked["reply"].startswith("C'est confirmé") and booked["pending"] is None
    (appointment,) = store.data["appointments"]
    assert appointment["slot_id"] == proposal["slot"]["slot_id"]
    assert store.idempotency_keys.get(f"appointments:NEO-10467:{key}") is not None


def test_no_booking_without_a_technical_reason(script, talk):
    script.technical = TechnicalCheck(justified="missing_info", reason="no_internet", override="none",
                                      message="Avez-vous redémarré la box ?")
    state = start_booking(script, talk)
    assert state["reply"] == "Avez-vous redémarré la box ?"
    assert "slots" not in script.calls and "propose_appointment" not in script.calls


def test_refusal_books_nothing(script, talk):
    start_booking(script, talk)
    script.confirmation = Confirmation(decision="no", preference="")
    assert talk("non merci")["reply"] == graph.NOT_BOOKED
    assert store.data["appointments"] == []


def test_other_slot_is_a_new_proposal_with_a_new_key(script, talk):
    first = start_booking(script, talk)["pending"]
    script.confirmation = Confirmation(decision="other_slot", preference="vendredi")
    second = talk("je préfère vendredi")["pending"]
    assert second["proposal"]["slot"]["slot_id"] != first["proposal"]["slot"]["slot_id"]
    assert second["key"] != first["key"] and store.data["appointments"] == []


def test_question_during_confirmation_keeps_the_same_proposal(script, talk):
    first = start_booking(script, talk)["pending"]
    script.confirmation = Confirmation(decision="question", preference="")
    state = talk("c'est quoi les 69 € ?")
    assert state["pending"] == first and "Confirmez-vous" in state["reply"]


def test_new_request_during_confirmation_drops_the_proposal(script, talk):
    start_booking(script, talk)
    script.confirmation = Confirmation(decision="new_request", preference="")
    script.plan = lambda text: Plan(query=text, route="info", historical=False,
                                    needs_customer=False, clarification="")
    state = talk("Finalement, combien coûte la fibre ?")
    assert state["pending"] is None and state["reply"].startswith("réponse")
    assert store.data["appointments"] == []


def test_uncertain_booking_is_replayed_with_the_same_key_and_books_once(script, talk, monkeypatch):
    start_booking(script, talk)
    original, failures = NeovaAPI.book_appointment, iter([True])
    def booked_then_lost(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if next(failures, False):
            raise APIUnavailable("response lost")
        return result
    monkeypatch.setattr(NeovaAPI, "book_appointment", booked_then_lost)
    state = talk("oui")
    assert state["reply"] == graph.UNCERTAIN_BOOKING and state["pending"]["step"] == "uncertain_operation"
    assert talk("alors ?")["reply"].startswith("C'est confirmé")
    assert len(store.data["appointments"]) == 1


# --- gesture -------------------------------------------------------------------------------

def gesture_verdict(**update):
    base = dict(decision="eligible", condition_status=["met", "met", "met"], cap_row=1, escalate_n2=False,
                out_of_scope=False, internal_note="note", customer_outcome="credit_next_invoice", redirect_to=[])
    return prompts.GestureVerdict(**{**base, **update})


def start_gesture(script, talk, verdict):
    script.gesture = verdict
    script.plan = lambda text: Plan(query=text, route="gesture", historical=False,
                                    needs_customer=True, clarification="")
    talk("Je veux un geste commercial pour la panne")
    return talk(CAMILLE)


def test_eligible_from_the_model_is_blocked_by_the_code(script, talk):
    state = start_gesture(script, talk, gesture_verdict())
    assert state["reply"].startswith("transmis") and state["gesture"]["decision"] == "needs_review"
    assert len(store.data["tickets"]) == 1 and store.data["appointments"] == []


def test_out_of_scope_goes_to_a_human_even_with_a_condition_not_met(script, talk):
    state = start_gesture(script, talk, gesture_verdict(decision="not_eligible", out_of_scope=True,
                                                        condition_status=["met", "not_met", "met"]))
    assert state["reply"].startswith("transmis") and len(store.data["tickets"]) == 1


def test_refused_gesture_is_announced_without_criteria(script, talk):
    state = start_gesture(script, talk, gesture_verdict(decision="not_eligible", condition_status=["met", "not_met", "met"],
                                                        redirect_to=["incident_follow_up"]))
    assert state["reply"] == "réponse (Geste commercial non accordé. Orientations possibles : le suivi de l'incident.)"


def test_identifiers_given_without_being_asked_are_accepted(script, talk):
    state = talk(CAMILLE)
    assert state["reply"] == graph.IDENTIFIED and state["session"]
    assert tools(script) == ["verify"] and "plan" not in script.calls


def test_one_ticket_per_conversation(script, talk):
    script.precheck = [1]
    talk("Je veux supprimer mes données personnelles")
    state = talk("allô ? NEO-40318")
    assert state["reply"].startswith("Votre demande est déjà transmise") and len(store.data["tickets"]) == 1
    assert tools(script).count("create_ticket") == 1


def test_new_identifiers_are_verified_even_with_an_open_session(script, talk):
    talk(AHMED)
    state = talk(CAMILLE)
    assert state["customer_id"] == "NEO-88213" and tools(script).count("verify") == 2


def test_a_listed_situation_is_handled_first_and_transferred_only_if_it_cannot_be_concluded(script, talk, monkeypatch):
    echeancier = [{"item": 3, "elements": [{"element": "demande d'échéancier de paiement", "established": True}]}]
    verdicts = iter([prompts.PostReviewVerdict(candidates=echeancier, documents_cover=True, reason="aucun montant dû",
                                               advisor_conditions=[{"condition": "dette supérieure au seuil", "met": "no"}]),
                     prompts.PostReviewVerdict(candidates=echeancier, documents_cover=True, reason="validation conseiller",
                                               advisor_conditions=[{"condition": "dette supérieure au seuil", "met": "yes"}])])
    original = script.ask
    monkeypatch.setattr(prompts, "ask", lambda name, schema, *b: next(verdicts) if name == "post_review" else original(name, schema, *b))
    assert talk("Je peux payer en plusieurs fois ?")["reply"].startswith("réponse")
    assert talk("Et en trois fois ?")["reply"].startswith("transmis") and len(store.data["tickets"]) == 1


def test_a_message_that_is_not_a_request_gets_a_welcome_and_no_ticket(script, talk):
    script.plan = lambda text: Plan(query=text, route="conversation", historical=False,
                                    needs_customer=False, clarification="")
    assert talk("comment ça va ?")["reply"] == graph.WELCOME
    assert store.data["tickets"] == [] and "post_review" not in script.calls


def test_a_second_appointment_is_not_proposed(script, talk):
    start_booking(script, talk)
    talk("oui")
    state = talk("Je voudrais un autre technicien")
    assert state["reply"].startswith("Vous avez déjà un rendez-vous") and len(store.data["appointments"]) == 1
    assert store.data["tickets"] == []


def test_identifiers_are_verified_even_while_a_technical_question_waits(script, talk):
    script.technical = TechnicalCheck(justified="missing_info", reason="no_internet", override="none",
                                      message="Avez-vous redémarré la box ?")
    start_booking(script, talk)
    assert talk(CAMILLE)["reply"] == graph.IDENTIFIED
