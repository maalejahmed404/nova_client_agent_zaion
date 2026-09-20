from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, computed_field, field_validator
from neova.config import now

class Invoice(BaseModel):
    invoice_id: str
    date: str
    amount: float
    status: str

class Customer(BaseModel):
    customer_id: str
    full_name: str
    plan: str
    monthly_price: float
    contract_start_date: str
    engagement_months: int
    address: str
    postal_code: str
    balance_due: float
    open_incident_id: Optional[str] = None
    equipment: List[str]
    last_invoices: List[Invoice]

class Incident(BaseModel):
    incident_id: str
    postal_codes: List[str]
    status: str
    cause: str
    affected_customers: int
    started_at: datetime
    estimated_resolution: Optional[datetime] = None

    @computed_field
    @property
    def phase(self) -> str:
        current_time = now()
        if self.started_at > current_time:
            return "planned"
        if self.started_at <= current_time and (self.estimated_resolution is None or self.estimated_resolution > current_time):
            return "active"
        return "past_eta"

class Slot(BaseModel):
    slot_id: str
    postal_codes: List[str]
    start: datetime
    end: datetime
    available: bool

class SlotWithZone(Slot):
    zone_incident: Optional[Incident] = None

class Appointment(BaseModel):
    appointment_id: str
    customer_id: str
    slot_id: str
    start: datetime
    end: datetime
    reason: str
    override_reason: Optional[Literal["pto_damaged", "equipment_damaged"]] = None
    created_at: datetime

class Ticket(BaseModel):
    ticket_id: str
    customer_id: Optional[str] = None
    category: str
    summary: str
    actions_taken: List[str]
    urgency: Literal["low", "normal", "high"]
    created_at: datetime
    callback_eta: str

class VerifyRequest(BaseModel):
    customer_id: str
    phone: str

class VerifyResponse(BaseModel):
    customer: Customer
    session: str

class ProposalRequest(BaseModel):
    slot_id: str
    reason: str
    override_reason: Optional[Literal["pto_damaged", "equipment_damaged"]] = None

class ProposalResponse(BaseModel):
    proposal_id: str
    slot: Slot
    reason: str
    override_reason: Optional[str] = None
    cost_notice: str
