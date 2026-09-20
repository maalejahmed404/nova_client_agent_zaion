from fastapi.testclient import TestClient
import pytest
from neova.api.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_store():
    client.post("/admin/reset")

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["clock"] == "simulated"

def test_incidents_active():
    response = client.get("/incidents?postal_code=69007")
    assert response.status_code == 200
    data = response.json()
    assert any(inc["incident_id"] == "INC-4502" and inc["phase"] == "active" for inc in data)

def test_incidents_planned():
    response = client.get("/incidents?postal_code=33000")
    assert response.status_code == 200
    data = response.json()
    assert any(inc["incident_id"] == "INC-4519" and inc["phase"] == "planned" for inc in data)

def test_slots_contains():
    response = client.get("/slots?postal_code=59000")
    assert response.status_code == 200
    data = response.json()
    assert any(slot["slot_id"] == "SLOT-5D23" for slot in data)

def test_slots_zone_incident():
    response = client.get("/slots?postal_code=75019")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for item in data:
        assert item["zone_incident"] is not None
        assert item["zone_incident"]["incident_id"] == "INC-4471"

def test_verify_wrong_phone():
    response = client.post("/customers/verify", json={"customer_id": "NEO-88213", "phone": "wrong"})
    assert response.status_code == 404
    assert response.json() == {"error": "verification_failed"}

def test_customer_no_phone():
    # Via get customer
    response = client.get("/customers/NEO-88213")
    assert response.status_code == 200
    assert "phone" not in response.json()
    
    # Via verify customer with correct phone
    response = client.post("/customers/verify", json={"customer_id": "NEO-88213", "phone": "0612840193"})
    assert response.status_code == 200
    assert "phone" not in response.json()

# New tests for step 2.2
def test_appointment_no_override():
    # Camille NEO-88213 has active outage INC-4471 in 75019
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A31",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": None,
        },
        headers={"Idempotency-Key": "test-key-1"},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "active_outage"

def test_appointment_with_override():
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A31",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-2"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "appointment_id" in data
    assert data["override_reason"] == "pto_damaged"
    assert "cost_notice" in data

def test_appointment_non_fibre_plan():
    # Sylvie NEO-53190 has "Mobile Néova 80 Go"
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-53190",
            "slot_id": "SLOT-4N01",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": None,
        },
        headers={"Idempotency-Key": "test-key-3"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "non_fibre_plan"

def test_appointment_zone_mismatch():
    # Ahmed NEO-10467 is in 69007, try to book a 75019 slot
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-10467",
            "slot_id": "SLOT-7A31",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": None,
        },
        headers={"Idempotency-Key": "test-key-4"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "zone_mismatch"

def test_appointment_missing_header():
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A31",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
    )
    assert response.status_code == 400
    assert response.json() == {"error": "idempotency_key_required"}

def test_appointment_idempotency():
    key = "idemp-key-123"
    # Get initial available slots count
    before_slots = client.get("/slots?postal_code=75019").json()
    available_before = sum(1 for s in before_slots if s["available"])
    
    # First call
    resp1 = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A33",  # different from above
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": key},
    )
    assert resp1.status_code == 201
    appt_id1 = resp1.json()["appointment_id"]
    
    # Check slots count after first booking
    mid_slots = client.get("/slots?postal_code=75019").json()
    available_mid = sum(1 for s in mid_slots if s["available"])
    assert available_mid == available_before - 1  # Should decrease by exactly one
    
    # Second call with same key
    resp2 = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A33",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": key},
    )
    assert resp2.status_code == 201
    appt_id2 = resp2.json()["appointment_id"]
    assert appt_id1 == appt_id2  # Same appointment ID
    
    # Verify slot count unchanged after second call (idempotency)
    after_slots = client.get("/slots?postal_code=75019").json()
    available_after = sum(1 for s in after_slots if s["available"])
    assert available_after == available_mid  # No additional decrease

def test_delete_appointment():
    # First create an appointment
    create_resp = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A34",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "delete-test-key"},
    )
    assert create_resp.status_code == 201
    appt_id = create_resp.json()["appointment_id"]
    
    # Get all slots (including unavailable) to check status
    # First, let's check if the slot appears in the slots endpoint
    slots_before_delete = client.get("/slots?postal_code=75019").json()
    
    # Find the specific slot
    slot_found = False
    for slot in slots_before_delete:
        if slot["slot_id"] == "SLOT-7A34":
            slot_found = True
            assert slot["available"] == False  # Should be unavailable after booking
            break
    
    # If not found in available slots (because it's now unavailable), 
    # check through a different endpoint or verify by trying to book it again
    if not slot_found:
        # Try to book the same slot again - should fail with slot_taken
        response = client.post(
            "/appointments",
            json={
                "customer_id": "NEO-71925",  # Different customer
                "slot_id": "SLOT-7A34",
                "reason": "no_internet",
                "cost_notice_acknowledged": True,
                "override_reason": "pto_damaged",
            },
            headers={"Idempotency-Key": "delete-test-key-2"},
        )
        assert response.status_code == 409
        assert response.json()["error"] == "slot_taken"
    
    # Delete the appointment
    delete_resp = client.delete(f"/appointments/{appt_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["ok"] is True
    
    # Verify slot is available again by checking it appears in available slots
    slots_after_delete = client.get("/slots?postal_code=75019").json()
    slot_7a34_after = next((s for s in slots_after_delete if s["slot_id"] == "SLOT-7A34"), None)
    # SLOT-7A34 should now be available and appear in the list
    assert slot_7a34_after is not None
    assert slot_7a34_after["available"] == True
    
    # Ensure 404 on unknown
    resp = client.delete("/appointments/UNKNOWN")
    assert resp.status_code == 404

def test_ticket_bad_category():
    response = client.post(
        "/tickets",
        json={
            "customer_id": "NEO-88213",
            "category": "invalid",
            "summary": "test",
            "actions_taken": [],
            "urgency": "normal",
        },
    )
    assert response.status_code == 422

def test_ticket_callback_eta():
    response = client.post(
        "/tickets",
        json={
            "customer_id": "NEO-88213",
            "category": "technical",
            "summary": "test",
            "actions_taken": ["rebooted"],
            "urgency": "normal",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "callback_eta" in data
    # "sous 45 minutes" because now is simulated 2026-08-25 10:00 weekday
    assert data["callback_eta"] == "sous 45 minutes"

def test_chaos_middleware():
    # Set chaos rate to 1.0
    client.post("/admin/chaos", json={"rate": 1.0})
    
    # /health should still work (admin route)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    
    # /slots should always 500
    slots_resp = client.get("/slots?postal_code=75019")
    assert slots_resp.status_code == 500
    assert slots_resp.json()["error"] == "chaos"
    
    # Set back to 0.0
    client.post("/admin/chaos", json={"rate": 0.0})
    slots_resp2 = client.get("/slots?postal_code=75019")
    assert slots_resp2.status_code == 200

# Additional error code tests
def test_appointment_customer_not_found():
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-99999",
            "slot_id": "SLOT-7A31",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-5"},
    )
    assert response.status_code == 404
    assert response.json()["error"] == "customer_not_found"

def test_appointment_slot_not_found():
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-XXXX",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-6"},
    )
    assert response.status_code == 404
    assert response.json()["error"] == "slot_not_found"

def test_appointment_slot_in_past():
    # Inject a synthetic past technician slot into the store
    from neova.api.store import store
    from neova.config import now
    current = now()
    past_start = current.isoformat()
    # Create a slot that started in the past but is still available
    with store.lock:
        store.data["technician_slots"].append({
            "slot_id": "SLOT-PAST1",
            "postal_codes": ["75019"],
            "start": "2026-08-24T09:00:00+02:00",  # Past relative to simulated now (2026-08-25)
            "end": "2026-08-24T11:00:00+02:00",
            "available": True
        })
    
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-PAST1",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-7"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "slot_in_past"

def test_appointment_invalid_reason():
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A31",
            "reason": "invalid_reason",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-8"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_reason"

def test_appointment_cost_notice_required():
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A31",
            "reason": "no_internet",
            "cost_notice_acknowledged": False,  # False should fail
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-9"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "cost_notice_required"

def test_appointment_slot_taken():
    # First book a slot
    client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A33",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-10"},
    )
    
    # Try to book the same slot with different customer
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-71925",  # Different customer
            "slot_id": "SLOT-7A33",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-11"},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "slot_taken"

def test_appointment_already_has_appointment():
    # First book an appointment for Camille
    client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A33",
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-12"},
    )
    
    # Try to book another appointment for same customer
    response = client.post(
        "/appointments",
        json={
            "customer_id": "NEO-88213",
            "slot_id": "SLOT-7A34",  # Different slot
            "reason": "no_internet",
            "cost_notice_acknowledged": True,
            "override_reason": "pto_damaged",
        },
        headers={"Idempotency-Key": "test-key-13"},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "already_has_appointment"