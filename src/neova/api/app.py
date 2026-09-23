import random
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from neova.api.models import Customer, Incident, VerifyRequest, VerifyResponse
from neova.api.store import Store
from neova.config import get_settings

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


@app.get("/health")
def health():
    return {"status": "up"}