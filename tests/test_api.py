from fastapi.testclient import TestClient
import pytest
from neova.api.app import app

client = TestClient(app)

CAMILLE = ("NEO-88213", "0612840193")   # 75019, Fibre, active outage INC-4471
AHMED = ("NEO-10467", "0778115402")     # 69007, Fibre
SYLVIE = ("NEO-53190", "0640027781")    # 44000, Mobile
PRO = ("NEO-71925", "0142886310")       # 75116, Néova Pro


def session(customer=CAMILLE) -> dict:
    resp = client.post("/customers/verify", json={"customer_id": customer[0], "phone": customer[1]})
    assert resp.status_code == 200
    return {"X-Session": resp.json()["session"]}


def propose(headers, slot_id="SLOT-7A31", reason="no_internet", override="pto_damaged"):
    return client.post(
        "/appointments/proposals",
        json={"slot_id": slot_id, "reason": reason, "override_reason": override},
        headers=headers,
    )


def book(headers, proposal_id, key):
    return client.post(
        "/appointments",
        json={"proposal_id": proposal_id},
        headers={**headers, "Idempotency-Key": key},
    )


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["clock"] == "simulated"

def test_incidents_active():
    data = client.get("/incidents?postal_code=69007").json()
    assert any(inc["incident_id"] == "INC-4502" and inc["phase"] == "active" for inc in data)

def test_incidents_planned():
    data = client.get("/incidents?postal_code=33000").json()
    assert any(inc["incident_id"] == "INC-4519" and inc["phase"] == "planned" for inc in data)

def test_slots_from_session_zone():
    data = client.get("/slots", headers=session(("NEO-40318", "0698441207"))).json()  # Lille 59000
    assert any(slot["slot_id"] == "SLOT-5D23" for slot in data)

def test_slots_zone_incident():
    data = client.get("/slots", headers=session()).json()
    assert len(data) > 0
    for item in data:
        assert item["zone_incident"]["incident_id"] == "INC-4471"

def test_verify_wrong_phone():
    response = client.post("/customers/verify", json={"customer_id": "NEO-88213", "phone": "wrong"})
    assert response.status_code == 404
    assert response.json() == {"error": "verification_failed"}

def test_verify_returns_session_and_no_phone():
    response = client.post("/customers/verify", json={"customer_id": CAMILLE[0], "phone": CAMILLE[1]})
    assert response.status_code == 200
    body = response.json()
    assert body["session"]
    assert "phone" not in body["customer"]


# --- verified session protects personal routes ---

def test_personal_routes_require_session():
    assert client.get("/customers/NEO-88213").status_code == 401
    assert client.get("/customers/NEO-88213/invoices").status_code == 401
    assert client.get("/slots").status_code == 401
    assert propose({}).status_code == 401
    assert book({}, "x", "k").status_code == 401

def test_unknown_session_is_rejected():
    assert client.get("/customers/NEO-88213", headers={"X-Session": "forged"}).status_code == 401

def test_session_cannot_read_another_customer():
    headers = session(CAMILLE)
    assert client.get("/customers/NEO-40318", headers=headers).status_code == 403
    assert client.get("/customers/NEO-40318/invoices", headers=headers).status_code == 403
    assert client.get("/customers/NEO-88213", headers=headers).status_code == 200


# --- proposals ---

def test_proposal_active_outage_without_override():
    response = propose(session(), override=None)
    assert response.status_code == 409
    assert response.json()["error"] == "active_outage"

def test_proposal_returns_notice_and_books_nothing():
    response = propose(session())
    assert response.status_code == 201
    body = response.json()
    assert body["proposal_id"]
    assert "69 €" in body["cost_notice"]
    assert body["slot"]["slot_id"] == "SLOT-7A31"
    slot = next(s for s in client.get("/slots", headers=session()).json() if s["slot_id"] == "SLOT-7A31")
    assert slot["available"] is True

def test_proposal_id_depends_on_override():
    headers = session()
    with_override = propose(headers, override="pto_damaged").json()["proposal_id"]
    without = propose(session(AHMED), slot_id="SLOT-6B10", override=None).json()["proposal_id"]
    assert with_override != without

def test_proposal_pro_contract_rejected():
    response = propose(session(PRO), slot_id="SLOT-7A31", override=None)
    assert response.status_code == 422
    assert response.json()["error"] == "pro_contract"

def test_proposal_non_fibre_plan():
    response = propose(session(SYLVIE), slot_id="SLOT-4N01", override=None)
    assert response.status_code == 422
    assert response.json()["error"] == "non_fibre_plan"

def test_proposal_zone_mismatch():
    response = propose(session(AHMED), slot_id="SLOT-7A31", override=None)
    assert response.status_code == 422
    assert response.json()["error"] == "zone_mismatch"

def test_proposal_slot_not_found():
    assert propose(session(), slot_id="SLOT-XXXX").json()["error"] == "slot_not_found"

def test_proposal_invalid_reason():
    assert propose(session(), reason="invalid_reason").json()["error"] == "invalid_reason"

def test_proposal_slot_in_past():
    from neova.api.store import store
    with store.lock:
        store.data["technician_slots"].append({
            "slot_id": "SLOT-PAST1", "postal_codes": ["75019"],
            "start": "2026-08-24T09:00:00+02:00", "end": "2026-08-24T11:00:00+02:00", "available": True,
        })
    response = propose(session(), slot_id="SLOT-PAST1")
    assert response.status_code == 422
    assert response.json()["error"] == "slot_in_past"


# --- booking bound to the exact proposal ---

def test_book_happy_path_consumes_proposal():
    headers = session()
    pid = propose(headers).json()["proposal_id"]
    response = book(headers, pid, "key-1")
    assert response.status_code == 201
    body = response.json()
    assert body["appointment_id"].startswith("APT-")
    assert body["override_reason"] == "pto_damaged"
    assert "69 €" in body["cost_notice"]
    # a second "oui" on the same proposal cannot book again
    again = book(headers, pid, "key-2")
    assert again.status_code == 404
    assert again.json()["error"] == "proposal_unknown"

def test_book_missing_idempotency_header():
    headers = session()
    pid = propose(headers).json()["proposal_id"]
    response = client.post("/appointments", json={"proposal_id": pid}, headers=headers)
    assert response.status_code == 400

def test_book_unknown_proposal():
    response = book(session(), "0" * 64, "key-3")
    assert response.status_code == 404
    assert response.json()["error"] == "proposal_unknown"

def test_book_another_customers_proposal():
    pid = propose(session(CAMILLE)).json()["proposal_id"]
    response = book(session(AHMED), pid, "key-4")
    assert response.status_code == 403
    assert response.json()["error"] == "proposal_mismatch"

def test_book_slot_taken_between_proposal_and_confirmation():
    first = session(CAMILLE)
    pid_first = propose(first, slot_id="SLOT-7A33").json()["proposal_id"]
    second = session(PRO)  # same zone 75116 shares SLOT-7A33, but Pro is rejected at proposal
    assert propose(second, slot_id="SLOT-7A33").status_code == 422
    # take the slot with another Fibre customer in the same zone is impossible in the dataset,
    # so mark it unavailable directly to simulate the race
    from neova.api.store import store
    with store.lock:
        next(s for s in store.data["technician_slots"] if s["slot_id"] == "SLOT-7A33")["available"] = False
    response = book(first, pid_first, "key-5")
    assert response.status_code == 409
    assert response.json()["error"] == "slot_taken"

def test_book_already_has_appointment():
    headers = session()
    pid1 = propose(headers, slot_id="SLOT-7A33").json()["proposal_id"]
    assert book(headers, pid1, "key-6").status_code == 201
    pid2 = propose(headers, slot_id="SLOT-7A34").json()["proposal_id"]
    response = book(headers, pid2, "key-7")
    assert response.status_code == 409
    assert response.json()["error"] == "already_has_appointment"


# --- idempotency ---

def test_book_retry_after_consumption_replays_same_result():
    headers = session()
    before = sum(1 for s in client.get("/slots", headers=headers).json() if s["available"])
    pid = propose(headers, slot_id="SLOT-7A33").json()["proposal_id"]
    first = book(headers, pid, "retry-key")
    assert first.status_code == 201
    retry = book(headers, pid, "retry-key")
    assert retry.status_code == 201
    assert retry.json()["appointment_id"] == first.json()["appointment_id"]
    after = sum(1 for s in client.get("/slots", headers=headers).json() if s["available"])
    assert after == before - 1

def test_book_same_key_different_body_conflicts():
    headers = session()
    pid1 = propose(headers, slot_id="SLOT-7A33").json()["proposal_id"]
    assert book(headers, pid1, "shared-key").status_code == 201
    pid2 = propose(headers, slot_id="SLOT-7A34").json()["proposal_id"]
    response = book(headers, pid2, "shared-key")
    assert response.status_code == 409
    assert response.json()["error"] == "idempotency_conflict"


# --- delete ---

def test_delete_appointment_frees_slot():
    headers = session()
    pid = propose(headers, slot_id="SLOT-7A34").json()["proposal_id"]
    appt_id = book(headers, pid, "delete-key").json()["appointment_id"]
    assert not any(s["slot_id"] == "SLOT-7A34" for s in client.get("/slots", headers=headers).json())
    assert client.delete(f"/appointments/{appt_id}").json()["ok"] is True
    assert any(s["slot_id"] == "SLOT-7A34" for s in client.get("/slots", headers=headers).json())
    assert client.delete("/appointments/UNKNOWN").status_code == 404


# --- tickets ---

TICKET = {"category": "technical", "summary": "test", "actions_taken": ["rebooted"], "urgency": "normal"}

def test_ticket_bad_category():
    response = client.post("/tickets", json={**TICKET, "category": "invalid"})
    assert response.status_code == 422

def test_ticket_without_session_for_mandatory_handoff():
    response = client.post("/tickets", json={**TICKET, "customer_id": "NEO-88213"})
    assert response.status_code == 201
    body = response.json()
    assert body["customer_id"] is None          # body id is ignored without a session
    assert body["callback_eta"] == "sous 45 minutes"

def test_ticket_with_session_uses_session_identity():
    response = client.post("/tickets", json={**TICKET, "customer_id": "NEO-40318"}, headers=session(CAMILLE))
    assert response.status_code == 201
    assert response.json()["customer_id"] == "NEO-88213"

def test_ticket_idempotent_replay():
    first = client.post("/tickets", json=TICKET, headers={"Idempotency-Key": "t-key"})
    second = client.post("/tickets", json=TICKET, headers={"Idempotency-Key": "t-key"})
    assert first.json()["ticket_id"] == second.json()["ticket_id"]
    assert len(client.get("/admin/tickets").json()) == 1
    conflict = client.post("/tickets", json={**TICKET, "summary": "other"}, headers={"Idempotency-Key": "t-key"})
    assert conflict.status_code == 409


# --- regressions: idempotency bound to identity, ticket creation atomic ---

def test_replay_is_bound_to_the_customer_who_used_the_key():
    camille = session(CAMILLE)
    pid = propose(camille).json()["proposal_id"]
    stored = book(camille, pid, "shared-key")
    assert stored.status_code == 201
    # Ahmed reuses Camille's key and proposal id: must not receive her booking
    response = book(session(AHMED), pid, "shared-key")
    assert response.status_code in (403, 404)
    assert "appointment_id" not in response.json()
    # ticket: same key, same body, another customer -> a different ticket, not a replay
    first = client.post("/tickets", json=TICKET, headers={**camille, "Idempotency-Key": "t-shared"})
    second = client.post("/tickets", json=TICKET, headers={**session(AHMED), "Idempotency-Key": "t-shared"})
    assert first.json()["customer_id"] == "NEO-88213"
    assert second.json()["customer_id"] == "NEO-10467"
    assert first.json()["ticket_id"] != second.json()["ticket_id"]

def test_concurrent_ticket_retries_create_one_ticket():
    from concurrent.futures import ThreadPoolExecutor
    from neova.api import app as app_module
    from neova.api.store import store

    # Widen the race: hold the store lock briefly on every write so both threads are in flight.
    original_write = store.write
    def slow_write():
        import time
        time.sleep(0.05)
        original_write()
    store.write = slow_write
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda _: client.post("/tickets", json=TICKET, headers={"Idempotency-Key": "race-key"}),
                range(2),
            ))
    finally:
        store.write = original_write
    assert {r.status_code for r in results} == {201}
    assert len({r.json()["ticket_id"] for r in results}) == 1
    assert len(client.get("/admin/tickets").json()) == 1


def test_chaos_middleware():
    client.post("/admin/chaos", json={"rate": 1.0})
    assert client.get("/health").status_code == 200
    resp = client.get("/incidents?postal_code=75019")
    assert resp.status_code == 500 and resp.json()["error"] == "chaos"
    client.post("/admin/chaos", json={"rate": 0.0})
    assert client.get("/incidents?postal_code=75019").status_code == 200
