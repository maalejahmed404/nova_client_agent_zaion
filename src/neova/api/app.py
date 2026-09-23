import random
import secrets
from datetime import timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from neova.api.models import (
    Appointment,
    BookingRequest,
    Customer,
    Incident,
    ProposalRequest,
    ProposalResponse,
    Ticket,
    TicketRequest,
    VerifyRequest,
    VerifyResponse,
)
from neova.api.store import Store, body_hash, proposal_id
from neova.config import BUSINESS_DAYS, BUSINESS_HOURS, get_settings, now

store = Store()
failure_rate = get_settings().api_chaos_rate

app = FastAPI()


@app.middleware("http")
async def chaos_middleware(request: Request, call_next):
    if request.url.path.startswith("/admin") or request.url.path == "/health":
        return await call_next(request)
    if random.random() < failure_rate:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Le service est temporairement indisponible."},
        )
    return await call_next(request)


def session_customer(x_session: Annotated[str | None, Header()] = None) -> str:
    if x_session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identité manquante.",
        )
    customer_id = store.session_customer(x_session)
    if customer_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session inconnue.",
        )
    return customer_id


def optional_session_customer(x_session: Annotated[str | None, Header()] = None) -> str | None:
    if x_session is None:
        return None
    return store.session_customer(x_session)


@app.post("/customers/verify", response_model=VerifyResponse)
def verify_customer(request: VerifyRequest):
    customer = store.get_customer(request.customer_id)
    if customer is None or not store.phone_matches(request.customer_id, request.phone):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="L'identité n'a pas pu être vérifiée.",
        )
    session = store.open_session(request.customer_id)
    return VerifyResponse(customer=customer, session=session)


@app.get("/customers/me", response_model=Customer)
def get_customer(customer_id: Annotated[str, Depends(session_customer)]):
    customer = store.get_customer(customer_id)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client introuvable.",
        )
    return customer


@app.get("/incidents", response_model=list[Incident])
def get_incidents(customer_id: Annotated[str, Depends(session_customer)]):
    customer = store.get_customer(customer_id)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client introuvable.",
        )
    return store.incidents_for(customer.postal_code)


@app.post("/admin/chaos")
def set_chaos_rate(rate: float):
    global failure_rate
    if rate < 0 or rate > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le taux doit être compris entre 0 et 1.",
        )
    failure_rate = rate
    return {"chaos_rate": failure_rate}


@app.post("/admin/reset")
def reset_store():
    store.reset()
    return {"status": "reset"}


@app.post("/appointments/proposals", response_model=ProposalResponse)
def create_proposal(
    request: ProposalRequest,
    customer_id: Annotated[str, Depends(session_customer)],
):
    customer = store.get_customer(customer_id)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client introuvable.",
        )
    if store.appointment_for(customer_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Le client a déjà un rendez-vous.",
        )
    slots = store.slots_for(customer.postal_code)
    slots = [slot for slot in slots if not slot.is_past]
    if not slots:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucun créneau disponible dans ce code postal.",
        )
    if request.another_slot:
        if len(slots) < 2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pas d'autre créneau disponible.",
            )
        slot = slots[1]
    else:
        slot = slots[0]
    prop_id = proposal_id(customer_id, slot.slot_id, request.reason)
    store.save_proposal(
        prop_id,
        {
            "customer_id": customer_id,
            "slot_id": slot.slot_id,
            "start": slot.start,
            "end": slot.end,
            "reason": request.reason,
        },
    )
    return ProposalResponse(
        proposal_id=prop_id,
        slot_id=slot.slot_id,
        start=slot.start,
        end=slot.end,
    )


@app.post("/appointments")
def book_appointment(
    request: BookingRequest,
    customer_id: Annotated[str, Depends(session_customer)],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
):
    if idempotency_key is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="L'en-tête Idempotency-Key est obligatoire.",
        )
    body = request.model_dump()
    hash = body_hash(body)
    remembered = store.recall_key(idempotency_key)
    if remembered is not None:
        stored_hash, stored_answer = remembered
        if stored_hash == hash:
            return stored_answer
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="La clé d'idempotence a déjà été utilisée avec un autre corps.",
            )
    proposal = store.peek_proposal(request.proposal_id)
    if proposal is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="L'offre proposée a été modifiée.",
        )
    slot = store.get_slot(proposal["slot_id"])
    if slot is None or not slot.available or slot.is_past:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ce créneau n'est plus disponible.",
        )
    store.take_proposal(request.proposal_id)
    appointment_id = secrets.token_urlsafe(16)
    appointment = Appointment(
        appointment_id=appointment_id,
        customer_id=customer_id,
        slot_id=proposal["slot_id"],
        reason=proposal["reason"],
        start=proposal["start"],
        end=proposal["end"],
        created_at=now(),
    )
    store.add_appointment(appointment)
    answer = appointment.model_dump(mode="json")
    store.remember_key(idempotency_key, hash, answer)
    return answer


@app.delete("/appointments/{appointment_id}")
def delete_appointment(
    appointment_id: str,
    customer_id: Annotated[str, Depends(session_customer)],
):
    appointment = store.get_appointment(appointment_id)
    if appointment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rendez-vous introuvable.",
        )
    if appointment.customer_id != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ce rendez-vous ne vous appartient pas.",
        )
    store.remove_appointment(appointment_id)
    return {"deleted": appointment_id}


@app.post("/tickets", response_model=Ticket)
def create_ticket(
    request: TicketRequest,
    customer_id: Annotated[str | None, Depends(optional_session_customer)] = None,
):
    current = now()
    if current.weekday() in BUSINESS_DAYS and BUSINESS_HOURS[0] <= current.hour < BUSINESS_HOURS[1]:
        callback = current + timedelta(minutes=45)
    else:
        tomorrow = current + timedelta(days=1)
        while tomorrow.weekday() not in BUSINESS_DAYS:
            tomorrow += timedelta(days=1)
        callback = tomorrow.replace(hour=BUSINESS_HOURS[0], minute=0, second=0, microsecond=0)
    ticket_id = secrets.token_urlsafe(16)
    ticket = Ticket(
        ticket_id=ticket_id,
        category=request.category,
        summary=request.summary,
        customer_id=customer_id,
        created_at=current,
        callback_eta=callback.isoformat(),
    )
    store.add_ticket(ticket)
    return ticket


@app.get("/health")
def health():
    return {"status": "up"}
