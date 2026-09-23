import time
from typing import Any

import httpx

from neova.config import get_settings


class APIError(Exception):
    """Raised when the API answers 4xx."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"API error {status}: {detail}")


class APIUnavailable(Exception):
    """Raised when the service does not answer or answers 5xx after retries."""


class APIClient:
    """Plain httpx client for the Néova API."""

    def __init__(self) -> None:
        self._base_url = get_settings().api_base_url
        self._session_token: str | None = None
        self._client = httpx.Client(timeout=10.0)

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
    ) -> Any:
        """Send a request with retries on connection errors and 5xx."""
        request_headers: dict[str, str] = {}
        if self._session_token:
            request_headers["X-Session"] = self._session_token
        if headers:
            request_headers.update(headers)

        for attempt in range(retries):
            try:
                response = self._client.request(
                    method, f"{self._base_url}{path}", json=json, headers=request_headers
                )
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt == retries - 1:
                    raise APIUnavailable("Service unreachable after retries.")
                time.sleep(0.5 * (attempt + 1))
                continue

            if response.status_code >= 500:
                if attempt == retries - 1:
                    raise APIUnavailable(f"Service answered {response.status_code}")
                time.sleep(0.5 * (attempt + 1))
                continue

            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = ""
                raise APIError(response.status_code, detail)

            return response.json()

        raise APIUnavailable("Maximum retries exceeded.")

    def verify(self, customer_id: str, phone: str) -> dict[str, Any]:
        """POST /customers/verify, store the token, return the customer."""
        data = self._request(
            "POST", "/customers/verify", {"customer_id": customer_id, "phone": phone}
        )
        self._session_token = data["session"]
        return data["customer"]

    def get_customer(self) -> dict[str, Any]:
        """GET /customers/me."""
        return self._request("GET", "/customers/me")

    def get_incidents(self) -> list[dict[str, Any]]:
        """GET /incidents."""
        return self._request("GET", "/incidents")

    def propose(
        self, reason: str, override_reason: str | None = None, another_slot: bool = False
    ) -> dict[str, Any]:
        """POST /appointments/proposals."""
        json: dict[str, Any] = {"reason": reason, "another_slot": another_slot}
        if override_reason is not None:
            json["override_reason"] = override_reason
        return self._request("POST", "/appointments/proposals", json)

    def book(self, proposal_id: str, idempotency_key: str) -> dict[str, Any]:
        """POST /appointments with the Idempotency-Key header."""
        headers = {"Idempotency-Key": idempotency_key}
        return self._request("POST", "/appointments", {"proposal_id": proposal_id}, headers)

    def cancel(self, appointment_id: str) -> dict[str, Any]:
        """DELETE /appointments/{id}."""
        return self._request("DELETE", f"/appointments/{appointment_id}")

    def create_ticket(self, category: str, summary: str) -> dict[str, Any]:
        """POST /tickets."""
        return self._request("POST", "/tickets", {"category": category, "summary": summary})

    def health(self) -> bool:
        """GET /health, true when the service answers."""
        try:
            response = self._client.get(f"{self._base_url}/health", timeout=2.0)
            return response.status_code < 400
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
