from fastapi.testclient import TestClient
from neova.api.app import app

client = TestClient(app)

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
    
    # Via verify customer
    response = client.post("/customers/verify", json={"customer_id": "NEO-88213", "phone": "0612840193"})
    if response.status_code == 200:
        assert "phone" not in response.json()
