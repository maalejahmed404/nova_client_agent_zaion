"""The agent's only access to customer data: the Néova API, one method per endpoint. Transient
failures (5xx, network) are retried; every write carries one idempotency key for all its retries."""
import uuid

import httpx
import tenacity

from neova.config import get_settings


class APIUnavailable(Exception):
    """The API kept failing after the retries: the caller hands off to a human."""


class APIError(Exception):
    """The API refused the request (4xx); `body["error"]` says why."""

    def __init__(self, status: int, body: dict):
        super().__init__(f"{status}: {body}")
        self.status = status
        self.body = body


class NeovaAPI:
    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(base_url=get_settings().api_base_url, timeout=10)
        self.session = None
        self.customer_id = None

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.TransportError, APIUnavailable)),
        wait=tenacity.wait_exponential(min=0.2, max=2),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def _send(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = self.http.request(method, path, **kwargs)
        if response.status_code >= 500:
            raise APIUnavailable(f"{method} {path} returned {response.status_code}")
        return response

    def _call(self, method: str, path: str, json: dict | None = None, params: dict | None = None,
              write: bool = False):
        headers = {"X-Session": self.session} if self.session else {}
        if write:
            headers["Idempotency-Key"] = str(uuid.uuid4())
        response = self._send(method, path, json=json, params=params, headers=headers)
        body = response.json()
        if response.status_code >= 400:
            raise APIError(response.status_code, body)
        return body

    def verify(self, customer_id: str, phone: str) -> dict:
        body = self._call("POST", "/customers/verify", json={"customer_id": customer_id, "phone": phone})
        self.session, self.customer_id = body["session"], body["customer"]["customer_id"]
        return body["customer"]

    def customer(self) -> dict:
        return self._call("GET", f"/customers/{self.customer_id}")

    def invoices(self) -> list:
        return self._call("GET", f"/customers/{self.customer_id}/invoices")

    def incidents(self, postal_code: str) -> list:
        return self._call("GET", "/incidents", params={"postal_code": postal_code})

    def slots(self) -> list:
        return self._call("GET", "/slots")

    def propose_appointment(self, slot_id: str, reason: str, override_reason: str | None = None) -> dict:
        return self._call("POST", "/appointments/proposals",
                          json={"slot_id": slot_id, "reason": reason, "override_reason": override_reason})

    def book_appointment(self, proposal_id: str) -> dict:
        return self._call("POST", "/appointments", json={"proposal_id": proposal_id}, write=True)

    def cancel_appointment(self, appointment_id: str) -> dict:
        return self._call("DELETE", f"/appointments/{appointment_id}")

    def create_ticket(self, category: str, summary: str, actions_taken: list[str], urgency: str) -> dict:
        return self._call("POST", "/tickets", write=True, json={
            "category": category, "summary": summary, "actions_taken": actions_taken, "urgency": urgency})
