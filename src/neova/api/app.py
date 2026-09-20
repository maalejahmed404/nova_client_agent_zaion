from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from neova.api.models import Customer, Incident, SlotWithZone, Ticket, Invoice, VerifyRequest, Appointment
from neova.api.store import store
from neova.config import DATA_DIR, now, clock_mode, get_settings
import random
import string
import json

app = FastAPI(title="Néova API")

# Global chaos rate that can be modified
chaos_rate: float = get_settings().api_chaos_rate

class AppointmentRequest(BaseModel):
    customer_id: str
    slot_id: str
    reason: str
    cost_notice_acknowledged: bool
    override_reason: Optional[Literal["pto_damaged", "equipment_damaged"]] = None
    
    @field_validator("override_reason")
    @classmethod
    def validate_override_reason(cls, v):
        if v not in ("pto_damaged", "equipment_damaged", None):
            raise ValueError("override_reason must be 'pto_damaged', 'equipment_damaged', or null")
        return v

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

@app.get("/customers/{id}", response_model=Customer, responses={404: {"model": dict}})
def get_customer(id: str):
    customer = store.get_customer(id)
    if not customer:
        return JSONResponse(status_code=404, content={"error": "customer_not_found"})
    return customer

@app.post("/customers/verify", response_model=Customer, responses={404: {"model": dict}})
def verify_customer(req: VerifyRequest):
    customer = store.verify(req.customer_id, req.phone)
    if not customer:
        return JSONResponse(status_code=404, content={"error": "verification_failed"})
    return customer

@app.get("/customers/{id}/invoices", response_model=List[Invoice])
def get_invoices(id: str):
    invs = store.invoices(id)
    if invs is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return invs

@app.get("/incidents", response_model=List[Incident])
def get_incidents(postal_code: str):
    return store.incidents(postal_code)

@app.get("/slots", response_model=List[SlotWithZone])
def get_slots(postal_code: str):
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

@app.post("/appointments", status_code=201)
def create_appointment(
    req: AppointmentRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
):
    if not idempotency_key:
        return JSONResponse(
            status_code=400,
            content={"error": "idempotency_key_required"}
        )
    
    with store.lock:
        # Check idempotency first
        if idempotency_key in store.idempotency_keys:
            return JSONResponse(
                status_code=201,
                content=store.idempotency_keys[idempotency_key]
            )
        
        # 1. customer exists
        customer = None
        for c in store.data.get("customers", []):
            if c["customer_id"] == req.customer_id:
                customer = Customer(**c)
                break
        
        if not customer:
            return JSONResponse(
                status_code=404,
                content={"error": "customer_not_found", "detail": "Client introuvable."}
            )
        
        # 2. slot exists
        slot = None
        for s in store.data.get("technician_slots", []):
            if s["slot_id"] == req.slot_id:
                slot = SlotWithZone(**s, zone_incident=None)
                break
        
        if not slot:
            return JSONResponse(
                status_code=404,
                content={"error": "slot_not_found", "detail": "Créneau introuvable."}
            )
        
        # 3. slot.start > now()
        if slot.start <= now():
            return JSONResponse(
                status_code=422,
                content={"error": "slot_in_past", "detail": "Le créneau est déjà passé."}
            )
        
        # 4. reason in data["appointment_reasons"]
        reasons = store.data.get("appointment_reasons", [])
        if req.reason not in reasons:
            return JSONResponse(
                status_code=422,
                content={"error": "invalid_reason", "detail": "Motif de rendez-vous invalide."}
            )
        
        # 5. customer.postal_code in slot.postal_codes
        if customer.postal_code not in slot.postal_codes:
            return JSONResponse(
                status_code=422,
                content={"error": "zone_mismatch", "detail": "Le client ne réside pas dans la zone du créneau."}
            )
        
        # 6. customer.plan starts with "Fibre" or "Néova Pro"
        if not (customer.plan.startswith("Fibre") or customer.plan.startswith("Néova Pro")):
            return JSONResponse(
                status_code=422,
                content={"error": "non_fibre_plan", "detail": "Le plan du client n'est pas une offre Fibre."}
            )
        
        # 7. cost_notice_acknowledged is true
        if not req.cost_notice_acknowledged:
            return JSONResponse(
                status_code=422,
                content={"error": "cost_notice_required", "detail": "L'accusé réception de l'information tarifaire est requis."}
            )
        
        # 8. no active incident with status "outage" on the customer's postal code
        active_outage = False
        for inc_data in store.data.get("network_incidents", []):
            inc = Incident(**inc_data)
            if (customer.postal_code in inc.postal_codes and 
                inc.phase == "active" and 
                inc.status == "outage"):
                active_outage = True
                break
        
        if active_outage and not req.override_reason:
            return JSONResponse(
                status_code=409,
                content={"error": "active_outage", "detail": "Une panne active est en cours dans votre zone."}
            )
        
        # 9. slot.available (re-check inside lock)
        if not slot.available:
            return JSONResponse(
                status_code=409,
                content={"error": "slot_taken", "detail": "Le créneau n'est plus disponible."}
            )
        
        # 10. customer has no existing appointment
        for appt_data in store.data.get("appointments", []):
            if appt_data["customer_id"] == customer.customer_id:
                return JSONResponse(
                    status_code=409,
                    content={"error": "already_has_appointment", "detail": "Le client a déjà un rendez-vous."}
                )
        
        # Success - atomic booking
        appointment_id = random_id("APT-")
        
        # Mark slot unavailable
        for s in store.data.get("technician_slots", []):
            if s["slot_id"] == slot.slot_id:
                s["available"] = False
                break
        
        # Create appointment
        new_appt = {
            "appointment_id": appointment_id,
            "customer_id": customer.customer_id,
            "slot_id": slot.slot_id,
            "start": slot.start.isoformat(),
            "end": slot.end.isoformat(),
            "reason": req.reason,
            "override_reason": req.override_reason,
            "created_at": now().isoformat(),
        }
        store.data["appointments"].append(new_appt)
        
        # Store idempotency response
        response_body = {
            "appointment_id": appointment_id,
            "customer_id": customer.customer_id,
            "slot_id": slot.slot_id,
            "start": slot.start.isoformat(),
            "end": slot.end.isoformat(),
            "reason": req.reason,
            "override_reason": req.override_reason,
            "created_at": now().isoformat(),
            "cost_notice": cost_notice_text,
        }
        store.idempotency_keys[idempotency_key] = response_body
        
        # Save state
        with open(DATA_DIR / "runtime_state.json", "w", encoding="utf-8") as f:
            json.dump(store.data, f, indent=2, ensure_ascii=False)
    
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
        
        # Save state
        with open(DATA_DIR / "runtime_state.json", "w", encoding="utf-8") as f:
            json.dump(store.data, f, indent=2, ensure_ascii=False)
    
    return {"ok": True}

@app.post("/tickets", status_code=201)
def create_ticket(req: TicketRequest):
    # Validate category
    with store.lock:
        categories = store.data.get("escalation_categories", [])
    if req.category not in categories:
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid category", "detail": "Catégorie invalide."}
        )
    
    # Generate ticket id
    ticket_id = random_id("TKT-")
    
    # Get callback_eta from policies
    from neova.policies import callback_eta
    eta = callback_eta(now())
    
    store.create_ticket(
        ticket_id=ticket_id,
        customer_id=req.customer_id,
        category=req.category,
        summary=req.summary,
        actions_taken=req.actions_taken,
        urgency=req.urgency,
        callback_eta=eta
    )
    store.save()
    
    return JSONResponse(
        status_code=201,
        content={
            "ticket_id": ticket_id,
            "callback_eta": eta,
        }
    )

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
