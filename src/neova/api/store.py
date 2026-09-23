import hashlib
import json
import secrets
import threading

from neova.api.models import Appointment, Customer, Incident, Slot, Ticket
from neova.config import DATA_DIR


def proposal_id(customer_id: str, slot_id: str, reason: str) -> str:
    raw = f"{customer_id}|{slot_id}|{reason}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def body_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seed_path = DATA_DIR / "neova_data.json"
        self.runtime_path = DATA_DIR / "runtime_state.json"
        self.data: dict = {}
        self.sessions: dict[str, str] = {}
        self.proposals: dict[str, dict] = {}
        self.idempotency_keys: dict[str, tuple[str, dict]] = {}
        with self.lock:
            self._load()

    def _load(self) -> None:
        source = self.runtime_path if self.runtime_path.exists() else self.seed_path
        self.data = json.loads(source.read_text(encoding="utf-8"))
        self.data.setdefault("appointments", [])
        self.data.setdefault("tickets", [])
        if source is self.seed_path:
            self._save()

    def _save(self) -> None:
        self.runtime_path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _customer_row(self, customer_id: str) -> dict | None:
        for row in self.data["customers"]:
            if row["customer_id"] == customer_id:
                return row
        return None

    def _slot_row(self, slot_id: str) -> dict | None:
        for row in self.data["technician_slots"]:
            if row["slot_id"] == slot_id:
                return row
        return None

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
            row = self._customer_row(customer_id)
            return Customer(**row) if row else None

    def phone_matches(self, customer_id: str, phone: str) -> bool:
        with self.lock:
            row = self._customer_row(customer_id)
            return bool(row) and row["phone"] == phone

    def incidents_for(self, postal_code: str) -> list[Incident]:
        with self.lock:
            return [
                Incident(**row)
                for row in self.data["network_incidents"]
                if postal_code in row["postal_codes"]
            ]

    def slots_for(self, postal_code: str) -> list[Slot]:
        with self.lock:
            return [
                Slot(**row)
                for row in self.data["technician_slots"]
                if postal_code in row["postal_codes"] and row["available"]
            ]

    def get_slot(self, slot_id: str) -> Slot | None:
        with self.lock:
            row = self._slot_row(slot_id)
            return Slot(**row) if row else None

    def save_proposal(self, proposal_id: str, offer: dict) -> None:
        with self.lock:
            self.proposals[proposal_id] = offer

    def peek_proposal(self, proposal_id: str) -> dict | None:
        with self.lock:
            return self.proposals.get(proposal_id)

    def take_proposal(self, proposal_id: str) -> dict | None:
        with self.lock:
            return self.proposals.pop(proposal_id, None)

    def add_appointment(self, appointment: Appointment) -> None:
        with self.lock:
            self.data["appointments"].append(appointment.model_dump(mode="json"))
            slot = self._slot_row(appointment.slot_id)
            if slot:
                slot["available"] = False
            self._save()

    def appointment_for(self, customer_id: str) -> Appointment | None:
        with self.lock:
            for row in self.data["appointments"]:
                if row["customer_id"] == customer_id:
                    return Appointment(**row)
            return None

    def get_appointment(self, appointment_id: str) -> Appointment | None:
        with self.lock:
            for row in self.data["appointments"]:
                if row["appointment_id"] == appointment_id:
                    return Appointment(**row)
            return None

    def remove_appointment(self, appointment_id: str) -> None:
        with self.lock:
            kept = [r for r in self.data["appointments"] if r["appointment_id"] != appointment_id]
            removed = [
                r for r in self.data["appointments"] if r["appointment_id"] == appointment_id
            ]
            if not removed:
                return
            self.data["appointments"] = kept
            slot = self._slot_row(removed[0]["slot_id"])
            if slot:
                slot["available"] = True
            self._save()

    def add_ticket(self, ticket: Ticket) -> None:
        with self.lock:
            self.data["tickets"].append(ticket.model_dump(mode="json"))
            self._save()

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
            self.runtime_path.unlink(missing_ok=True)
            self._load()
