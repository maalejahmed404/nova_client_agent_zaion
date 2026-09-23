import difflib
import json
import threading
import time
import unicodedata
import uuid
from urllib.parse import urlparse

import httpx
import typer
import uvicorn
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from neova.agent.graph import graph
from neova.config import DATA_DIR, get_settings

app = typer.Typer(add_completion=False)

EMPTY_STATE = {
    "session": None,
    "customer_id": None,
    "facts": {},
    "passages": [],
    "proposal": None,
    "actions": [],
    "tool_calls": 0,
    "needs_done": False,
    "needs": None,
    "escalated": False,
    "next": None,
    "after_tools": None,
}


def fold(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in stripped if not unicodedata.combining(c))


def demo_logins() -> dict[str, str]:
    customers = json.loads((DATA_DIR / "neova_data.json").read_text(encoding="utf-8"))["customers"]
    return {fold(c["full_name"]): f"{c['customer_id']} {c['phone']}" for c in customers}


def api_is_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


def show(node: str, update: dict | None) -> None:
    update = update or {}
    line = f"  [{node}]" + (f" → {update['next']}" if update.get("next") else "")
    if update.get("matched_items"):
        line += f" (situations : {update['matched_items']})"
    if update.get("reason"):
        line += f" (motif : {update['reason']})"
    typer.secho(line, fg="green")
    for message in update.get("messages", []):
        if isinstance(message, AIMessage):
            if message.content:
                typer.secho(f"      brouillon : {message.content[:200]}", fg="magenta")
            for call in message.tool_calls:
                args = json.dumps(call["args"], ensure_ascii=False)
                typer.secho(f"      outil {call['name']}({args})", fg="blue")
        elif isinstance(message, ToolMessage):
            typer.secho(f"      résultat : {message.content[:200]}", fg="blue")


def last_reply(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content
    return "(pas de réponse)"


@app.command()
def api(port: int = 8000):
    uvicorn.run("neova.api.app:app", port=port)


@app.command()
def chat(logs: bool = typer.Option(True, help="Affiche les nœuds et les outils exécutés.")):
    url = get_settings().api_base_url.rstrip("/")
    own_api = not api_is_up(url)
    if own_api:
        config = uvicorn.Config(
            "neova.api.app:app", port=urlparse(url).port or 8000, log_level="warning"
        )
        threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()
        for _ in range(50):
            if api_is_up(url):
                break
            time.sleep(0.2)

    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    logins = demo_logins()
    first = True
    typer.echo("Néova — assistant client. Ligne vide pour quitter.")
    typer.echo(
        "Démo : tapez le nom complet d'un client (ex. Ahmed Belkacem) pour vous identifier.\n"
    )
    try:
        while message := input("vous > ").strip():
            match = difflib.get_close_matches(fold(message), logins, n=1, cutoff=0.8)
            if match:
                customer_id, phone = logins[match[0]].split()
                message = (
                    f"Bonjour, je suis {message}. Mon numéro client est {customer_id} "
                    f"et le téléphone du contrat est {phone}."
                )
                typer.secho(f"(démo : identifiants envoyés — {customer_id} {phone})", fg="yellow")
            turn = {"messages": [HumanMessage(content=message)], "next": None}
            if first:
                turn = {**EMPTY_STATE, **turn}
                first = False
            for step in graph.stream(turn, thread, stream_mode="updates"):
                for node, update in step.items():
                    if logs:
                        show(node, update)
            typer.echo(f"néova > {last_reply(graph.get_state(thread).values['messages'])}\n")
    finally:
        if own_api:
            (DATA_DIR / "runtime_state.json").unlink(missing_ok=True)


if __name__ == "__main__":
    app()
