import json
from datetime import datetime
from zoneinfo import ZoneInfo
from neova.policies import (
    contract_facts, gesture_eligibility, callback_eta, 
    in_deprecated_window, mandatory_escalation_reason, leak_check
)

# Load test data
with open("data/neova_data.json", "r", encoding="utf-8") as f:
    neova_data = json.load(f)

def test_contract_facts():
    now = datetime(2026, 8, 25, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))
    
    # Camille NEO-88213
    camille = next(c for c in neova_data["customers"] if c["customer_id"] == "NEO-88213")
    facts = contract_facts(camille, now)
    assert facts["seniority_months"] == 29
    assert facts["engagement_active"] == False
    assert facts["early_termination_fee"] == 0.0
    assert facts["equipment_penalty_total"] == 138
    assert facts["fee_rule"] == "none"
    
    # Ahmed NEO-10467
    ahmed = next(c for c in neova_data["customers"] if c["customer_id"] == "NEO-10467")
    facts = contract_facts(ahmed, now)
    assert facts["seniority_months"] == 3
    assert facts["remaining_months"] == 21
    assert facts["early_termination_fee"] == 629.79
    assert facts["fee_rule"] == "full_remaining"
    
    # Patrick NEO-40318
    patrick = next(c for c in neova_data["customers"] if c["customer_id"] == "NEO-40318")
    facts = contract_facts(patrick, now)
    assert facts["seniority_months"] == 0
    assert facts["remaining_months"] == 24
    assert facts["early_termination_fee"] == 959.76
    assert facts["fee_rule"] == "full_remaining"
    
    # Léa NEO-27604
    lea = next(c for c in neova_data["customers"] if c["customer_id"] == "NEO-27604")
    facts = contract_facts(lea, now)
    assert facts["engagement_active"] == False
    assert facts["early_termination_fee"] == 0.0
    assert facts["fee_rule"] == "none"
    
    # Sylvie NEO-53190
    sylvie = next(c for c in neova_data["customers"] if c["customer_id"] == "NEO-53190")
    facts = contract_facts(sylvie, now)
    assert facts["engagement_months"] == 0
    assert facts["early_termination_fee"] == 0.0
    assert facts["fee_rule"] == "none"

def test_gesture_eligibility():
    now = datetime(2026, 8, 25, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))
    
    # Camille NEO-88213
    camille = next(c for c in neova_data["customers"] if c["customer_id"] == "NEO-88213")
    facts = contract_facts(camille, now)
    assert facts["seniority_months"] == 29
    
    # Pass all network incidents - gesture_eligibility() will filter by postal code internally
    eligibility = gesture_eligibility(camille, neova_data["network_incidents"], now)
    assert eligibility["eligible"] == True
    assert eligibility["cap"] == "50%"
    
    # Ahmed NEO-10467
    ahmed = next(c for c in neova_data["customers"] if c["customer_id"] == "NEO-10467")
    eligibility = gesture_eligibility(ahmed, neova_data["network_incidents"], now)
    assert eligibility["eligible"] == False
    assert eligibility["reason_code"] == "unpaid_balance"
    
    # Léa NEO-27604
    lea = next(c for c in neova_data["customers"] if c["customer_id"] == "NEO-27604")
    eligibility = gesture_eligibility(lea, neova_data["network_incidents"], now)
    assert eligibility["eligible"] == False
    assert eligibility["reason_code"] == "no_incident"

def test_callback_eta():
    now = datetime(2026, 8, 25, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))
    assert callback_eta(now) == "sous 45 minutes"
    
    # Saturday
    sat = datetime(2026, 8, 29, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))
    assert callback_eta(sat) == "lundi matin"
    
    # Tuesday evening
    tue_evening = datetime(2026, 8, 25, 20, 0, tzinfo=ZoneInfo("Europe/Paris"))
    assert callback_eta(tue_evening) == "demain matin"
    
    # Friday evening
    fri_evening = datetime(2026, 8, 28, 19, 0, tzinfo=ZoneInfo("Europe/Paris"))
    assert callback_eta(fri_evening) == "lundi matin"
    
    # Wednesday evening
    wed = datetime(2026, 8, 26, 19, 0, tzinfo=ZoneInfo("Europe/Paris"))
    assert callback_eta(wed) == "demain matin"

def test_mandatory_escalation_reason():
    assert mandatory_escalation_reason("Mon père est décédé") == "bereavement"
    assert mandatory_escalation_reason("je veux exercer mes droits RGPD") == "rgpd"
    assert mandatory_escalation_reason("ma box est morte") is None
    assert mandatory_escalation_reason("ma box clignote") is None
    assert mandatory_escalation_reason("je vais porter plainte") == "legal_threat"

def test_leak_check():
    result = leak_check("Vous devez être client depuis au moins 6 mois")
    assert len(result) > 0
    assert "au moins 6 mois" in result
    
    result = leak_check("L'espace client reste accessible 6 mois après la résiliation")
    assert len(result) == 0

if __name__ == "__main__":
    test_contract_facts()
    test_gesture_eligibility()
    test_callback_eta()
    test_mandatory_escalation_reason()
    test_leak_check()
    print("All tests passed!")