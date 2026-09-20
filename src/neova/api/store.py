import hashlib
import json
import secrets
import threading
import random
import string
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from neova.config import DATA_DIR, now, get_settings
from neova.api.models import Customer, Incident, Slot, SlotWithZone, Invoice, Appointment, Ticket


def proposal_id(customer_id: str, slot_id: str, reason: str, override_reason: Optional[str], cost_notice: str) -> str:
    raw = "|".join([customer_id, slot_id, reason, override_reason or "", cost_notice])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def body_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


class Store:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.lock = threading.Lock()
        self.seed_path = data_dir / "neova_data.json"
        self.runtime_path = data_dir / "runtime_state.json"
        self.data = {}
        self.idempotency_keys: Dict[str, dict] = {}
        self.sessions: Dict[str, str] = {}
        self.proposals: Dict[str, dict] = {}
        self.load()

    def load(self):
        with self.lock:
            if self.runtime_path.exists():
                with open(self.runtime_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if "data" not in state:  # file written before idempotency/proposals were persisted
                    state = {"data": state}
                self.data = state["data"]
                self.idempotency_keys = state.get("idempotency", {})
                self.proposals = state.get("proposals", {})
            else:
                with open(self.seed_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)

    def reset(self):
        with self.lock:
            with open(self.seed_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.idempotency_keys.clear()
            self.sessions.clear()
            self.proposals.clear()

    def write(self):
        """Persist state. Caller must hold self.lock."""
        state = {"data": self.data, "idempotency": self.idempotency_keys, "proposals": self.proposals}
        with open(self.runtime_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def save(self):
        with self.lock:
            self.write()

    def create_session(self, customer_id: str) -> str:
        token = secrets.token_urlsafe(16)
        with self.lock:
            self.sessions[token] = customer_id
        return token

    def customer_for(self, token: Optional[str]) -> Optional[Customer]:
        if not token:
            return None
        with self.lock:
            customer_id = self.sessions.get(token)
        if not customer_id:
            return None
        return self.get_customer(customer_id)

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        with self.lock:
            for c in self.data.get("customers", []):
                if c["customer_id"] == customer_id:
                    return Customer(**c)
            return None

    def verify(self, customer_id: str, phone: str) -> Optional[Customer]:
        with self.lock:
            for c in self.data.get("customers", []):
                if c["customer_id"] == customer_id and c["phone"] == phone:
                    return Customer(**c)
            return None

    def invoices(self, customer_id: str) -> Optional[List[Invoice]]:
        with self.lock:
            for c in self.data.get("customers", []):
                if c["customer_id"] == customer_id:
                    return [Invoice(**inv) for inv in c.get("last_invoices", [])]
            return None

    def incidents(self, postal_code: str) -> List[Incident]:
        with self.lock:
            results = []
            for inc in self.data.get("network_incidents", []):
                if postal_code in inc.get("postal_codes", []):
                    results.append(Incident(**inc))
            return results

    def available_slots(self, postal_code: str) -> List[Slot]:
        with self.lock:
            current_time = now()
            results = []
            for s in self.data.get("technician_slots", []):
                if not s.get("available", False):
                    continue
                if postal_code not in s.get("postal_codes", []):
                    continue
                slot = Slot(**s)
                if slot.start > current_time:
                    results.append(slot)
            return results

    def get_slot(self, slot_id: str) -> Optional[Slot]:
        with self.lock:
            for s in self.data.get("technician_slots", []):
                if s["slot_id"] == slot_id:
                    return Slot(**s)
            return None

    def get_appointment_by_customer(self, customer_id: str) -> Optional[Appointment]:
        with self.lock:
            for appt_data in self.data.get("appointments", []):
                if appt_data["customer_id"] == customer_id:
                    return Appointment(**appt_data)
            return None

    def get_appointment_by_id(self, appointment_id: str) -> Optional[Appointment]:
        with self.lock:
            for appt_data in self.data.get("appointments", []):
                if appt_data["appointment_id"] == appointment_id:
                    return Appointment(**appt_data)
            return None

    def create_ticket(
        self,
        ticket_id: str,
        customer_id: Optional[str],
        category: str,
        summary: str,
        actions_taken: List[str],
        urgency: str,
        callback_eta: str
    ):
        with self.lock:
            new_ticket = {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "category": category,
                "summary": summary,
                "actions_taken": actions_taken,
                "urgency": urgency,
                "created_at": now().isoformat(),
                "callback_eta": callback_eta,
            }
            self.data["tickets"].append(new_ticket)

store = Store()