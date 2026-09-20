import json
import threading
from typing import List, Optional
from neova.config import DATA_DIR, now
from neova.api.models import Customer, Incident, Slot, SlotWithZone, Invoice

class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}
        self.reset()

    def reset(self):
        with self.lock:
            with open(DATA_DIR / "neova_data.json", "r", encoding="utf-8") as f:
                self.data = json.load(f)

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
        customer = self.get_customer(customer_id)
        if customer:
            return customer.last_invoices
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

store = Store()
