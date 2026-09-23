from datetime import date, datetime

from pydantic import BaseModel, computed_field, field_validator

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
    open_incident_id: str | None = None
    equipment: list[str]
    last_invoices: list[Invoice]

    @computed_field
    @property
    def seniority_months(self) -> int:
        start, current = date.fromisoformat(self.contract_start_date), now()
        months = (current.year - start.year) * 12 + (current.month - start.month)
        return months - 1 if current.day < start.day else months

    @computed_field
    @property
    def months_left(self) -> int:
        if self.engagement_months == 0:
            return 0
        months_passed = self.seniority_months
        months_remaining = self.engagement_months - months_passed
        return max(0, months_remaining)

    @computed_field
    @property
    def engagement_over(self) -> bool:
        return self.months_left == 0

    @property
    def is_pro(self) -> bool:
        return "Pro" in self.plan

    @property
    def is_fibre(self) -> bool:
        return "Fibre" in self.plan


class Incident(BaseModel):
    incident_id: str
    postal_codes: list[str]
    status: str
    cause: str
    affected_customers: int
    started_at: datetime
    estimated_resolution: datetime | None = None

    @computed_field
    @property
    def phase(self) -> str:
        current_time = now()
        if self.started_at > current_time:
            return "planned"
        if self.started_at <= current_time and (
            self.estimated_resolution is None or self.estimated_resolution > current_time
        ):
            return "active"
        return "past_eta"

    @computed_field
    @property
    def observed_duration_hours(self) -> float | None:
        current_time = now()
        if self.started_at > current_time:
            return None
        return round((current_time - self.started_at).total_seconds() / 3600, 2)


class Slot(BaseModel):
    slot_id: str
    postal_codes: list[str]
    start: datetime
    end: datetime
    available: bool

    @property
    def is_past(self) -> bool:
        return self.start < now()


class Appointment(BaseModel):
    appointment_id: str
    customer_id: str
    slot_id: str
    reason: str
    start: datetime
    end: datetime
    created_at: datetime

    @field_validator("reason")
    def validate_reason(cls, v):
        allowed = ["no_internet", "slow_internet", "installation", "equipment_swap"]
        if v not in allowed:
            raise ValueError(f"Reason must be one of {allowed}")
        return v


class Ticket(BaseModel):
    ticket_id: str
    category: str
    summary: str
    customer_id: str | None = None
    created_at: datetime
    callback_eta: str

    @field_validator("category")
    def validate_category(cls, v):
        allowed = ["billing_dispute", "technical", "termination", "commercial_gesture", "other"]
        if v not in allowed:
            raise ValueError(f"Category must be one of {allowed}")
        return v


class VerifyRequest(BaseModel):
    customer_id: str
    phone: str


class VerifyResponse(BaseModel):
    customer: Customer
    session: str


class ProposalRequest(BaseModel):
    reason: str
    another_slot: bool = False

    @field_validator("reason")
    def validate_reason(cls, v):
        allowed = ["no_internet", "slow_internet", "installation", "equipment_swap"]
        if v not in allowed:
            raise ValueError(f"Reason must be one of {allowed}")
        return v


class ProposalResponse(BaseModel):
    proposal_id: str
    slot_id: str
    start: datetime
    end: datetime


class BookingRequest(BaseModel):
    proposal_id: str


class TicketRequest(BaseModel):
    category: str
    summary: str

    @field_validator("category")
    def validate_category(cls, v):
        allowed = ["billing_dispute", "technical", "termination", "commercial_gesture", "other"]
        if v not in allowed:
            raise ValueError(f"Category must be one of {allowed}")
        return v
