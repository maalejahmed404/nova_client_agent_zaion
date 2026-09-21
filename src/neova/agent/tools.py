"""The agent's tools. `NeovaAPI` is the HTTP client of our FastAPI service (retries on 5xx, one
idempotency key per write). The `@tool` functions below are what the LLM sees: their docstrings are
its only documentation. They are executed by the graph's `tools` node, which owns the state."""
import uuid

import httpx
import tenacity
from langchain_core.tools import tool

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
    def __init__(self, http: httpx.Client | None = None, session: str | None = None, customer_id: str | None = None):
        self.http = http or httpx.Client(base_url=get_settings().api_base_url, timeout=10)
        self.session = session
        self.customer_id = customer_id

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
              idempotency_key: str | None = None):
        headers = {"X-Session": self.session} if self.session else {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
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

    def book_appointment(self, proposal_id: str, idempotency_key: str | None = None) -> dict:
        """Pass the same key again to replay an operation whose result is unknown."""
        return self._call("POST", "/appointments", json={"proposal_id": proposal_id},
                          idempotency_key=idempotency_key or str(uuid.uuid4()))

    def cancel_appointment(self, appointment_id: str) -> dict:
        return self._call("DELETE", f"/appointments/{appointment_id}")

    def create_ticket(self, category: str, summary: str, actions_taken: list[str], urgency: str,
                      idempotency_key: str | None = None) -> dict:
        return self._call("POST", "/tickets", idempotency_key=idempotency_key or str(uuid.uuid4()), json={
            "category": category, "summary": summary, "actions_taken": actions_taken, "urgency": urgency})


# --------------------------------------------------------------------------- what the LLM sees
# Bodies are placeholders: the graph's `tools` node executes each call with access to the state.

@tool
def search_documents(question: str, historical: bool = False) -> str:
    """Cherche dans la documentation publique de Néova (offres, tarifs, FAQ, CGV, procédures).
    question : une question complète et autonome en français. historical : true seulement si le
    client parle d'une offre ou d'un tarif passé. Renvoie des passages avec leur identifiant."""


@tool
def verify_customer(customer_id: str, phone: str) -> str:
    """Vérifie l'identité du client avec son numéro client (format NEO-XXXXX) et le numéro de
    téléphone de son contrat, tels qu'il les a donnés. Ouvre l'accès à son dossier."""


@tool
def get_customer() -> str:
    """Le dossier du client identifié : offre, prix mensuel, dates, ancienneté, engagement, solde
    dû, factures, équipements, incident en cours sur sa ligne. Nécessite une identité vérifiée."""


@tool
def get_incidents() -> str:
    """Les incidents réseau dans la zone du client identifié, avec leur état et leur durée."""


@tool
def propose_appointment(reason: str, override_reason: str | None = None, another_slot: bool = False) -> str:
    """Prépare un rendez-vous technicien sur le prochain créneau disponible dans la zone du client
    identifié, sans le réserver. reason parmi no_internet, slow_internet, installation,
    equipment_swap. override_reason : pto_damaged ou equipment_damaged si le client décrit une prise
    ou un équipement endommagé, sinon rien. another_slot : true si le client refuse le créneau
    proposé et en veut un autre. Renvoie la date, l'heure et l'information sur les frais à
    transmettre au client."""


@tool
def book_appointment() -> str:
    """Réserve le rendez-vous proposé. À appeler seulement après que le client a clairement
    accepté la proposition (oui). La réservation n'est faite que si sa réponse est bien positive."""


@tool
def assess_gesture(request: str) -> str:
    """Décide d'une demande de geste commercial (remise, dédommagement, avoir) pour le client
    identifié. request : la demande du client en une phrase. Renvoie l'issue à annoncer et les
    orientations possibles ; ne renvoie aucun critère."""


@tool
def request_handoff(reason: str) -> str:
    """Transfère la conversation à un conseiller humain. reason : le motif, en une phrase, pour
    le conseiller. À utiliser quand la documentation ne permet pas de répondre, quand un outil le
    demande, ou quand la situation dépasse ce que tu peux traiter."""


TOOLS = [search_documents, verify_customer, get_customer, get_incidents, propose_appointment,
         book_appointment, assess_gesture, request_handoff]
