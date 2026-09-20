import json
import threading
import random
import string
from typing import List, Optional, Dict
from datetime import datetime
from neova.config import DATA_DIR, now, get_settings
from neova.api.models import Customer, Incident, Slot, SlotWithZone, Invoice, Appointment, Ticket

class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}
        self.idempotency_keys: Dict[str, dict] = {}
        self.reset()

    def reset(self):
        with self.lock:
            with open(DATA_DIR / "neova_data.json", "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.idempotency_keys.clear()

    def save(self):
        with self.lock:
            with open(DATA_DIR / "runtime_state.json", "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

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