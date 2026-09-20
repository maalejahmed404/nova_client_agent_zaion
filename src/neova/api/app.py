from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from neova.api.models import Customer, Incident, SlotWithZone, Ticket, Invoice, VerifyRequest
from neova.api.store import store
from neova.config import now, clock_mode

app = FastAPI(title="Néova API")

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

@app.post("/admin/reset")
def reset():
    store.reset()
    return {"ok": True}

@app.get("/admin/tickets", response_model=List[Ticket])
def get_tickets():
    with store.lock:
        return [Ticket(**t) for t in store.data.get("tickets", [])]
