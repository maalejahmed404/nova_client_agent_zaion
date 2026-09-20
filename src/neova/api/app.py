from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from neova.api.models import (
    Customer, Incident, SlotWithZone, Ticket, Invoice, VerifyRequest, VerifyResponse,
    ProposalRequest, ProposalResponse, Appointment,
)
from neova.api.store import store, proposal_id, body_hash
from neova.config import DATA_DIR, now, clock_mode, get_settings
import random
import string
import json

app = FastAPI(title="Néova API")

# Global chaos rate that can be modified
chaos_rate: float = get_settings().api_chaos_rate

class AppointmentRequest(BaseModel):
    proposal_id: str

class TicketRequest(BaseModel):
    customer_id: Optional[str] = None
    category: str
    summary: str
    actions_taken: List[str]
    urgency: Literal["low", "normal", "high"]
    
    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v):
        if v not in ("low", "normal", "high"):
            raise ValueError("urgency must be 'low', 'normal', or 'high'")
        return v

class ChaosRequest(BaseModel):
    rate: float

cost_notice_text = "L'intervention est gratuite si la cause se situe en amont de la prise optique ou si l'équipement fourni par Néova est défectueux ; elle est facturée 69 € si le technicien constate une dégradation imputable au client."

def random_id(prefix: str) -> str:
    return prefix + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

@app.middleware("http")
async def chaos_middleware(request: Request, call_next):
    global chaos_rate
    if request.url.path.startswith("/health") or request.url.path.startswith("/admin/"):
        return await call_next(request)
    if random.random() < chaos_rate:
        return JSONResponse(status_code=500, content={"error": "chaos"})
    return await call_next(request)

@app.get("/health")
def health():
    return {"now": now().isoformat(), "clock": clock_mode()}

def session_error(customer, id: Optional[str] = None):
    """401 when the session is missing/unknown, 403 when it belongs to another customer."""
    if customer is None:
        return JSONResponse(status_code=401, content={"error": "session_required"})
    if id is not None and id != customer.customer_id:
        return JSONResponse(status_code=403, content={"error": "session_mismatch"})
    return None

@app.get("/customers/{id}", response_model=Customer, responses={401: {"model": dict}, 403: {"model": dict}})
def get_customer(id: str, x_session: Optional[str] = Header(None, alias="X-Session")):
    customer = store.customer_for(x_session)
    return session_error(customer, id) or customer

@app.post("/customers/verify", response_model=VerifyResponse, responses={404: {"model": dict}})
def verify_customer(req: VerifyRequest):
    customer = store.verify(req.customer_id, req.phone)
    if not customer:
        return JSONResponse(status_code=404, content={"error": "verification_failed"})
    return VerifyResponse(customer=customer, session=store.create_session(customer.customer_id))

@app.get("/customers/{id}/invoices", response_model=List[Invoice])
def get_invoices(id: str, x_session: Optional[str] = Header(None, alias="X-Session")):
    customer = store.customer_for(x_session)
    return session_error(customer, id) or store.invoices(id)

@app.get("/incidents", response_model=List[Incident])
def get_incidents(postal_code: str):
    return store.incidents(postal_code)

@app.get("/slots", response_model=List[SlotWithZone])
def get_slots(x_session: Optional[str] = Header(None, alias="X-Session")):
    customer = store.customer_for(x_session)
    error = session_error(customer)
    if error:
        return error
    postal_code = customer.postal_code
    slots = store.available_slots(postal_code)
    incidents = store.incidents(postal_code)
    zone_incident = None
    for inc in incidents:
        if inc.phase == "active" and inc.status == "outage":
            zone_incident = inc
            break
            
    results = []
    for s in slots:
        results.append(SlotWithZone(**s.model_dump(), zone_incident=zone_incident))
    return results

def find_slot(slot_id: str):
    for s in store.data.get("technician_slots", []):
        if s["slot_id"] == slot_id:
            return SlotWithZone(**s, zone_incident=None)
    return None

def has_active_outage(postal_code: str) -> bool:
    for inc_data in store.data.get("network_incidents", []):
        inc = Incident(**inc_data)
        if postal_code in inc.postal_codes and inc.phase == "active" and inc.status == "outage":
            return True
    return False

def idempotent_replay(scope: str, key: str, body: dict):
    """Return the stored response for a repeated key, a 409 for a reused key with another body, else None."""
    entry = store.idempotency_keys.get(f"{scope}:{key}")
    if entry is None:
        return None
    if entry["hash"] != body_hash(body):
        return JSONResponse(status_code=409, content={"error": "idempotency_conflict"})
    return JSONResponse(status_code=201, content=entry["response"])

def remember_response(scope: str, key: str, body: dict, response: dict):
    store.idempotency_keys[f"{scope}:{key}"] = {"hash": body_hash(body), "response": response}

@app.post("/appointments/proposals", status_code=201, response_model=ProposalResponse)
def create_proposal(req: ProposalRequest, x_session: Optional[str] = Header(None, alias="X-Session")):
    customer = store.customer_for(x_session)
    error = session_error(customer)
    if error:
        return error

    with store.lock:
        slot = find_slot(req.slot_id)
        if not slot:
            return JSONResponse(status_code=404, content={"error": "slot_not_found", "detail": "Créneau introuvable."})
        if slot.start <= now():
            return JSONResponse(status_code=422, content={"error": "slot_in_past", "detail": "Le créneau est déjà passé."})
        if req.reason not in store.data.get("appointment_reasons", []):
            return JSONResponse(status_code=422, content={"error": "invalid_reason", "detail": "Motif de rendez-vous invalide."})
        if customer.postal_code not in slot.postal_codes:
            return JSONResponse(status_code=422, content={"error": "zone_mismatch", "detail": "Le client ne réside pas dans la zone du créneau."})
        if customer.plan.startswith("Néova Pro"):
            return JSONResponse(status_code=422, content={"error": "pro_contract", "detail": "Les contrats professionnels relèvent d'un service dédié."})
        if not customer.plan.startswith("Fibre"):
            return JSONResponse(status_code=422, content={"error": "non_fibre_plan", "detail": "Le plan du client n'est pas une offre Fibre."})
        if has_active_outage(customer.postal_code) and not req.override_reason:
            return JSONResponse(status_code=409, content={"error": "active_outage", "detail": "Une panne active est en cours dans votre zone."})
        if not slot.available:
            return JSONResponse(status_code=409, content={"error": "slot_taken", "detail": "Le créneau n'est plus disponible."})

        pid = proposal_id(customer.customer_id, slot.slot_id, req.reason, req.override_reason, cost_notice_text)
        store.proposals[pid] = {
            "customer_id": customer.customer_id,
            "slot_id": slot.slot_id,
            "reason": req.reason,
            "override_reason": req.override_reason,
            "cost_notice": cost_notice_text,
            "created_at": now().isoformat(),
        }
        store.write()

    return ProposalResponse(proposal_id=pid, slot=slot, reason=req.reason,
                            override_reason=req.override_reason, cost_notice=cost_notice_text)

@app.post("/appointments", status_code=201)
def create_appointment(
    req: AppointmentRequest,
    x_session: Optional[str] = Header(None, alias="X-Session"),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    if not idempotency_key:
        return JSONResponse(status_code=400, content={"error": "idempotency_key_required"})
    customer = store.customer_for(x_session)
    error = session_error(customer)
    if error:
        return error
    body = req.model_dump()
    scope = f"appointments:{customer.customer_id}"  # a key is only replayable by the customer who used it

    with store.lock:
        # Replay before anything else: a retry after the proposal was consumed must still succeed.
        replay = idempotent_replay(scope, idempotency_key, body)
        if replay is not None:
            return replay

        proposal = store.proposals.get(req.proposal_id)
        if proposal is None:
            return JSONResponse(status_code=404, content={"error": "proposal_unknown"})
        if proposal["customer_id"] != customer.customer_id:
            return JSONResponse(status_code=403, content={"error": "proposal_mismatch"})
        expected = proposal_id(proposal["customer_id"], proposal["slot_id"], proposal["reason"],
                               proposal["override_reason"], proposal["cost_notice"])
        if expected != req.proposal_id:
            return JSONResponse(status_code=409, content={"error": "proposal_tampered"})

        slot = find_slot(proposal["slot_id"])
        if not slot or slot.start <= now():
            return JSONResponse(status_code=422, content={"error": "slot_in_past", "detail": "Le créneau est déjà passé."})
        if has_active_outage(customer.postal_code) and not proposal["override_reason"]:
            return JSONResponse(status_code=409, content={"error": "active_outage", "detail": "Une panne active est en cours dans votre zone."})
        if not slot.available:
            return JSONResponse(status_code=409, content={"error": "slot_taken", "detail": "Le créneau n'est plus disponible."})
        for appt_data in store.data.get("appointments", []):
            if appt_data["customer_id"] == customer.customer_id:
                return JSONResponse(status_code=409, content={"error": "already_has_appointment", "detail": "Le client a déjà un rendez-vous."})

        appointment_id = random_id("APT-")
        for s in store.data.get("technician_slots", []):
            if s["slot_id"] == slot.slot_id:
                s["available"] = False
                break
        response_body = {
            "appointment_id": appointment_id,
            "customer_id": customer.customer_id,
            "slot_id": slot.slot_id,
            "start": slot.start.isoformat(),
            "end": slot.end.isoformat(),
            "reason": proposal["reason"],
            "override_reason": proposal["override_reason"],
            "created_at": now().isoformat(),
            "cost_notice": proposal["cost_notice"],
        }
        store.data["appointments"].append({k: v for k, v in response_body.items() if k != "cost_notice"})
        del store.proposals[req.proposal_id]
        remember_response(scope, idempotency_key, body, response_body)
        store.write()

    return JSONResponse(status_code=201, content=response_body)

@app.delete("/appointments/{appointment_id}")
def delete_appointment(appointment_id: str):
    with store.lock:
        # Find appointment
        appt_to_remove = None
        for i, appt_data in enumerate(store.data.get("appointments", [])):
            if appt_data["appointment_id"] == appointment_id:
                appt_to_remove = appt_data
                break
        
        if not appt_to_remove:
            return JSONResponse(status_code=404, content={"error": "Appointment not found"})
        
        # Remove appointment
        store.data["appointments"].pop(i)
        
        # Mark slot available again
        slot_id = appt_to_remove["slot_id"]
        for s in store.data.get("technician_slots", []):
            if s["slot_id"] == slot_id:
                s["available"] = True
                break
        
        store.write()

    return {"ok": True}

@app.post("/tickets", status_code=201)
def create_ticket(
    req: TicketRequest,
    x_session: Optional[str] = Header(None, alias="X-Session"),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    # No session required: mandatory hand-offs happen before any verification.
    # The customer id comes only from a valid session, never from the body.
    customer = store.customer_for(x_session)
    customer_id = customer.customer_id if customer else None
    body = req.model_dump()
    scope = f"tickets:{customer_id or 'anon'}"
    from neova.policies import callback_eta

    # One critical section: replay check, creation and persistence together, so two concurrent
    # retries cannot both miss the replay and create two tickets.
    with store.lock:
        if idempotency_key:
            replay = idempotent_replay(scope, idempotency_key, body)
            if replay is not None:
                return replay
        if req.category not in store.data.get("escalation_categories", []):
            return JSONResponse(
                status_code=422,
                content={"error": "Invalid category", "detail": "Catégorie invalide."}
            )
        ticket_id = random_id("TKT-")
        eta = callback_eta(now())
        store.data["tickets"].append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "category": req.category,
            "summary": req.summary,
            "actions_taken": req.actions_taken,
            "urgency": req.urgency,
            "created_at": now().isoformat(),
            "callback_eta": eta,
        })
        response_body = {"ticket_id": ticket_id, "callback_eta": eta, "customer_id": customer_id}
        if idempotency_key:
            remember_response(scope, idempotency_key, body, response_body)
        store.write()

    return JSONResponse(status_code=201, content=response_body)

@app.post("/admin/chaos")
def set_chaos(req: ChaosRequest):
    global chaos_rate
    chaos_rate = req.rate
    return {"rate": req.rate}

@app.post("/admin/reset")
def reset():
    store.reset()
    return {"ok": True}

@app.get("/admin/tickets", response_model=List[Ticket])
def get_tickets():
    with store.lock:
        return [Ticket(**t) for t in store.data.get("tickets", [])]
