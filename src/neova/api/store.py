import hashlib
import json
import secrets
import threading

from neova.api.models import Appointment, Customer, Incident, Slot, Ticket
from neova.config import DATA_DIR


def proposal_id(
    customer_id: str, slot_id: str, reason: str, override_reason: str | None
) -> str:
    raw = "|".join([customer_id, slot_id, reason, override_reason or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def body_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data_path = DATA_DIR / "neova_data.json"
        self.runtime_path = DATA_DIR / "runtime_state.json"
        self.customers: dict[str, Customer] = {}
        self.customer_phones: dict[str, str] = {}
        self.incidents: dict[str, Incident] = {}
        self.slots: dict[str, Slot] = {}
        self.sessions: dict[str, str] = {}
        self.appointments: dict[str, Appointment] = {}
        self.tickets: dict[str, Ticket] = {}
        self.proposals: dict[str, dict] = {}
        self.idempotency_keys: dict[str, tuple[str, dict]] = {}
        with self.lock:
            self._load()

    def _load(self) -> None:
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for customer_data in data["customers"]:
            customer = Customer(**customer_data)
            self.customers[customer.customer_id] = customer
            self.customer_phones[customer.customer_id] = customer_data["phone"]
        for incident_data in data["network_incidents"]:
            incident = Incident(**incident_data)
            self.incidents[incident.incident_id] = incident
        for slot_data in data["technician_slots"]:
            slot = Slot(**slot_data)
            self.slots[slot.slot_id] = slot
        if self.runtime_path.exists():
            with open(self.runtime_path, "r", encoding="utf-8") as f:
                runtime_data = json.load(f)
            for appointment_data in runtime_data.get("appointments", []):
                appointment = Appointment(**appointment_data)
                self.appointments[appointment.appointment_id] = appointment
            for ticket_data in runtime_data.get("tickets", []):
                ticket = Ticket(**ticket_data)
                self.tickets[ticket.ticket_id] = ticket

    def _save_runtime(self) -> None:
        appointments = [
            appointment.model_dump(mode="json") for appointment in self.appointments.values()
        ]
        tickets = [ticket.model_dump(mode="json") for ticket in self.tickets.values()]
        runtime_data = {"appointments": appointments, "tickets": tickets}
        with open(self.runtime_path, "w", encoding="utf-8") as f:
            json.dump(runtime_data, f, indent=2, ensure_ascii=False)

    def open_session(self, customer_id: str) -> str:
        token = secrets.token_urlsafe(16)
        with self.lock:
            self.sessions[token] = customer_id
        return token

    def session_customer(self, token: str) -> str | None:
        with self.lock:
            return self.sessions.get(token)

    def get_customer(self, customer_id: str) -> Customer | None:
        with self.lock:
            return self.customers.get(customer_id)

    def phone_matches(self, customer_id: str, phone: str) -> bool:
        with self.lock:
            return self.customer_phones.get(customer_id) == phone

    def incidents_for(self, postal_code: str) -> list[Incident]:
        with self.lock:
            result = []
            for incident in self.incidents.values():
                if postal_code in incident.postal_codes:
                    result.append(incident)
            return result

    def slots_for(self, postal_code: str) -> list[Slot]:
        with self.lock:
            result = []
            for slot in self.slots.values():
                if postal_code in slot.postal_codes and slot.available:
                    result.append(slot)
            return result

    def save_proposal(self, proposal_id: str, offer: dict) -> None:
        with self.lock:
            self.proposals[proposal_id] = offer

    def take_proposal(self, proposal_id: str) -> dict | None:
        with self.lock:
            return self.proposals.pop(proposal_id, None)

    def add_appointment(self, appointment: Appointment) -> None:
        with self.lock:
            self.appointments[appointment.appointment_id] = appointment
            self._save_runtime()

    def appointment_for(self, customer_id: str) -> Appointment | None:
        with self.lock:
            for appointment in self.appointments.values():
                if appointment.customer_id == customer_id:
                    return appointment
            return None

    def remove_appointment(self, appointment_id: str) -> None:
        with self.lock:
            if appointment_id in self.appointments:
                del self.appointments[appointment_id]
                self._save_runtime()

    def add_ticket(self, ticket: Ticket) -> None:
        with self.lock:
            self.tickets[ticket.ticket_id] = ticket
            self._save_runtime()

    def remember_key(self, key: str, body_hash: str, answer: dict) -> None:
        with self.lock:
            self.idempotency_keys[key] = (body_hash, answer)

    def recall_key(self, key: str) -> tuple[str, dict] | None:
        with self.lock:
            return self.idempotency_keys.get(key)

    def reset(self) -> None:
        with self.lock:
            self.sessions.clear()
            self.proposals.clear()
            self.idempotency_keys.clear()
            self.customers.clear()
            self.customer_phones.clear()
            self.incidents.clear()
            self.slots.clear()
            self.appointments.clear()
            self.tickets.clear()
            if self.runtime_path.exists():
                self.runtime_path.unlink()
            self._load()
