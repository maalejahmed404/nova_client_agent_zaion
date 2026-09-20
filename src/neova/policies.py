from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from zoneinfo import ZoneInfo
import re
import calendar

EQUIPMENT_PENALTIES = {"Box Néova 6": 89, "Box Néova 5": 69, "Décodeur TV": 49,
                       "Routeur secours 4G": 79, "Routeur de secours 4G": 79}


def contract_facts(customer: dict, now: datetime) -> dict:
    # Calculate seniority in whole months
    contract_start_date = datetime.fromisoformat(customer["contract_start_date"]).date()
    seniority_months = (now.year - contract_start_date.year) * 12 + (now.month - contract_start_date.month)
    
    # Adjust for day-of-month awareness
    if now.day < contract_start_date.day:
        seniority_months -= 1
    
    # Engagement calculations
    engagement_months = customer.get("engagement_months", 0)
    if engagement_months == 0:
        engagement_end = None
    else:
        # Calculate engagement end date
        year = contract_start_date.year
        month = contract_start_date.month + engagement_months
        while month > 12:
            year += 1
            month -= 12
        try:
            engagement_end = datetime(year, month, contract_start_date.day, tzinfo=ZoneInfo("Europe/Paris"))
        except ValueError:  # Handle cases like Feb 31
            # Set to last day of the month
            if month in [1, 3, 5, 7, 8, 10, 12]:
                day = 31
            elif month in [4, 6, 9, 11]:
                day = 30
            else:  # February
                day = 28
            engagement_end = datetime(year, month, day, tzinfo=ZoneInfo("Europe/Paris"))
    
    engagement_active = engagement_months > 0 and (engagement_end is None or engagement_end > now)
    
    # Calculate remaining months
    if not engagement_active:
        remaining_months = 0
    else:
        # Calculate remaining months from now to engagement_end
        remaining_months = (engagement_end.year - now.year) * 12 + (engagement_end.month - now.month)
        # Round up
        if engagement_end.day > now.day:
            remaining_months += 1
    
    # Calculate early termination fee
    if not engagement_active:
        early_termination_fee = 0.0
    else:
        monthly_price = customer.get("monthly_price", 0)
        if seniority_months < 12:
            fee = remaining_months * monthly_price
        else:
            fee = remaining_months * monthly_price * 0.25
        early_termination_fee = round(fee, 2)
    
    # Determine fee rule
    if not engagement_active:
        fee_rule = "none"
    elif seniority_months < 12:
        fee_rule = "full_remaining"
    else:
        fee_rule = "quarter_remaining"
    
    # Equipment penalties
    equipment_penalties = {}
    for item in customer.get("equipment", []):
        if item in EQUIPMENT_PENALTIES:
            equipment_penalties[item] = EQUIPMENT_PENALTIES[item]
    
    equipment_penalty_total = sum(equipment_penalties.values())
    
    return {
        "seniority_months": seniority_months,
        "engagement_months": engagement_months,
        "engagement_end": engagement_end,
        "engagement_active": engagement_active,
        "remaining_months": remaining_months,
        "early_termination_fee": early_termination_fee,
        "fee_rule": fee_rule,
        "equipment_penalties": equipment_penalties,
        "equipment_penalty_total": equipment_penalty_total,
        "notice_days": 10
    }


def gesture_eligibility(customer: dict, incidents: List[dict], now: datetime) -> dict:
    # Check unpaid balance
    if customer.get("balance_due", 0) != 0:
        return {"eligible": False, "cap": None, "reason_code": "unpaid_balance"}
    
    # Check seniority
    contract_start_date = datetime.fromisoformat(customer["contract_start_date"]).date()
    seniority_months = (now.year - contract_start_date.year) * 12 + (now.month - contract_start_date.month)
    if now.day < contract_start_date.day:
        seniority_months -= 1
    
    if seniority_months < 6:
        return {"eligible": False, "cap": None, "reason_code": "insufficient_seniority"}
    
    # Check for qualifying incidents
    qualifying_incident = None
    for incident in incidents:
        started_at = datetime.fromisoformat(incident['started_at'])

        estimated_resolution = (
            datetime.fromisoformat(incident['estimated_resolution'])
            if incident['estimated_resolution']
            else None
        )
        if started_at > now :
            continue
        is_ongoing  = (
            estimated_resolution is None
            or estimated_resolution > now
        )
        is_recently_ended = (
            estimated_resolution is not None and estimated_resolution <= now and (now - estimated_resolution).days <= 30
        )

        if ( (is_ongoing or is_recently_ended) and customer['postal_code'] in incident['postal_codes']) :
            qualifying_incident = incident
            break
        
    
    if not qualifying_incident:
        return {"eligible": False, "cap": None, "reason_code": "no_incident"}
    
    # Calculate duration and determine cap
    start_time = datetime.fromisoformat(qualifying_incident["started_at"])
    end_time = (
        datetime.fromisoformat(qualifying_incident["estimated_resolution"])
        if qualifying_incident["estimated_resolution"]
        else now
    )

    duration = end_time - start_time
    
    hours = duration.total_seconds() / 3600
    if hours < 48:
        cap = "50%"
    elif hours <= 168:  # 7 days
        cap = "1 month"
    else:
        cap = "escalate_n2"
    
    return {"eligible": True, "cap": cap, "reason_code": None}


def callback_eta(now: datetime) -> str:
    # Check if it's a weekday (Monday-Friday) and within business hours
    if now.weekday() < 5 and 9 <= now.hour < 18:
        return "sous 45 minutes"
    
    # If Friday after 18:00, Saturday, or Sunday
    if (now.weekday() == 4 and now.hour >= 18) or now.weekday() >= 5:
        return "lundi matin"
    
    # Otherwise, it's after business hours on a weekday
    return "demain matin"


def in_deprecated_window(customer: dict, windows: List[Tuple[str, str]]) -> bool:
    contract_start_date = datetime.fromisoformat(customer["contract_start_date"]).date()
    
    for valid_from, valid_to in windows:
        from_date = datetime.fromisoformat(valid_from).date()
        to_date = datetime.fromisoformat(valid_to).date()
        
        if from_date <= contract_start_date <= to_date:
            return True
    
    return False


def mandatory_escalation_reason(text: str) -> Optional[str]:
    if not text:
        return None
    
    # Normalize text: lowercase and remove accents
    normalized = text.lower()
    # Remove accents
    normalized = re.sub(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', 
                        lambda m: {'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae', 'ç': 'c',
                                  'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
                                  'ð': 'd', 'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
                                  'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y', 'þ': 'th', 'ÿ': 'y'}[m.group(0)],
                        normalized)
    
    # Test regex patterns
    patterns = [
        ("rgpd", r"\b(rgpd|donnees personnelles|droit d'acces|droit a l'effacement|portabilite)\b"),
        ("legal_threat", r"\b(avocat|mediateur|tribunal|mise en demeure|association de consommateurs|porter plainte|huissier)\b"),
        ("bereavement", r"\b(deces|decede|(pere|mere|mari|femme|epoux|epouse|conjoint|fils|fille|frere|soeur|titulaire) est (mort|morte))\b"),
        ("fraud", r"\b(fraude|usurpation|usurpe|pas reconnu|reconnais pas|jamais souscrit|prelevement inconnu)\b"),
        ("minor_or_protected", r"\b(je suis mineur|mineure|tutelle|curatelle)\b"),
        ("distress", r"\b(je n'en peux plus|suicid|me faire du mal|detresse|plus rien a manger)\b")
    ]
    
    # Special handling for bereavement to exclude "est mort" alone
    for code, pattern in patterns:
        if re.search(pattern, normalized):
            return code

    return None

def leak_check(text: str) -> List[str]:
    if not text:
        return []
    
    # Normalize text: lowercase and remove accents
    normalized = text.lower()
    # Remove accents
    normalized = re.sub(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]',
                        lambda m: {'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae', 'ç': 'c',
                                  'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
                                  'ð': 'd', 'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
                                  'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y', 'þ': 'th', 'ÿ': 'y'}[m.group(0)],
                        normalized)
    
    sentinels = [
        "au moins 6 mois",
        "depuis 6 mois", 
        "30 derniers jours",
        "12 mois glissants",
        "strictement egal",
        "trois conditions",
        "conditions cumulatives",
        "niveau 2",
        "transmettre au niveau",
        "plafond absolu"
    ]
    
    matched = []
    for sentinel in sentinels:
        if sentinel in normalized:
            matched.append(sentinel)
    
    return matched