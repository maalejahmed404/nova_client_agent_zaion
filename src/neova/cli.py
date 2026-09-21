"""uv run neova chat   – talk to the agent (starts the API in the background if it is not running)
uv run neova api    – run the API alone"""
import difflib
import json
import threading
import time
import uuid
from urllib.parse import urlparse

import httpx
import typer
import uvicorn

from neova.agent.graph import build, run_turn
from neova.config import DATA_DIR, get_settings
from neova.rag.ingest import fold

app = typer.Typer(add_completion=False)


def demo_logins() -> dict:
    """Demo shortcut for the terminal only: typing a customer's full name sends their customer number
    and phone, so the agent verifies them through the API as usual. Not an identity check."""
    customers = json.loads((DATA_DIR / "neova_data.json").read_text(encoding="utf-8"))["customers"]
    return {fold(c["full_name"]): f"{c['customer_id']} {c['phone']}" for c in customers}


def api_is_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


@app.command()
def api(port: int = 8000):
    uvicorn.run("neova.api.app:app", port=port)


@app.command()
def chat():
    url = get_settings().api_base_url.rstrip("/")
    if not api_is_up(url):
        server = uvicorn.Server(uvicorn.Config("neova.api.app:app", port=urlparse(url).port or 8000, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(50):
            if api_is_up(url):
                break
            time.sleep(0.2)
    graph, thread, logins = build(), str(uuid.uuid4()), demo_logins()
    typer.echo("Néova — assistant client. Ligne vide pour quitter.")
    typer.echo("Démo : tapez le nom complet d'un client des données (ex. Ahmed Belkacem) pour vous identifier.\n")
    while message := input("vous > ").strip():
        name = difflib.get_close_matches(fold(message), logins, n=1, cutoff=0.8)
        if name:
            message = logins[name[0]]
            typer.echo(f"(démo : identifiants envoyés {message})")
        state = run_turn(graph, thread, message)
        typer.echo(f"néova > {state['reply']}\n")


if __name__ == "__main__":
    app()
