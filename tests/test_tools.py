import pytest
import tenacity
from fastapi.testclient import TestClient

import neova.api.app as app_module
from neova.agent.tools import APIError, APIUnavailable, NeovaAPI

CAMILLE = ("NEO-88213", "0612840193")   # 75019, linked outage INC-4471 started 06:12


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(NeovaAPI._send.retry, "wait", tenacity.wait_none())
    return NeovaAPI(TestClient(app_module.app))


def test_customer_facts_come_from_the_api(api):
    api.verify(*CAMILLE)
    customer = api.customer()
    assert customer["seniority_months"] == 29 and customer["balance_due"] == 0
    (incident,) = api.incidents(customer["postal_code"])
    assert incident["incident_id"] == customer["open_incident_id"] == "INC-4471"
    assert incident["phase"] == "active" and incident["observed_duration_hours"] == 3.8   # 06:12 -> 10:00 = 3 h 48


def test_planned_incident_has_no_duration(api):
    (incident,) = api.incidents("33000")
    assert incident["phase"] == "planned" and incident["observed_duration_hours"] is None


def test_personal_data_requires_verification(api):
    with pytest.raises(APIError) as refused:
        api.customer()
    assert refused.value.status == 401


def test_booking_goes_through_a_proposal(api):
    api.verify(*CAMILLE)
    proposal = api.propose_appointment("SLOT-7A31", "no_internet", "pto_damaged")
    assert api.book_appointment(proposal["proposal_id"])["slot_id"] == "SLOT-7A31"


def test_transient_500_is_retried_and_creates_one_ticket(api, monkeypatch):
    draws = iter([0.0, 0.9])                       # first request hit by chaos, second goes through
    monkeypatch.setattr(app_module, "chaos_rate", 0.5)
    monkeypatch.setattr(app_module.random, "random", lambda: next(draws))
    assert api.create_ticket("technical", "box en panne", [], "normal")["ticket_id"]
    assert len(TestClient(app_module.app).get("/admin/tickets").json()) == 1


def test_persistent_500_raises_unavailable(api, monkeypatch):
    monkeypatch.setattr(app_module, "chaos_rate", 1.0)
    with pytest.raises(APIUnavailable):
        api.incidents("75019")
