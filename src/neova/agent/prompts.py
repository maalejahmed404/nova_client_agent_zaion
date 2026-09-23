import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from neova.config import PARIS, now


def _escape_angle_brackets(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _wrap_tag(tag: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return f"<{tag}>{_escape_angle_brackets(serialized)}</{tag}>"
    return f"<{tag}>{_escape_angle_brackets(str(value))}</{tag}>"


def load_template(name: str) -> str:
    template_path = Path(__file__).parent / "templates" / f"{name}.txt"
    return template_path.read_text(encoding="utf-8")


def build_messages(template_name: str, blocks: list[tuple[str, Any]]) -> tuple[str, str]:
    template = load_template(template_name)
    human_parts = []

    for tag, value in blocks:
        wrapped = _wrap_tag(tag, value)
        human_parts.append(wrapped)

    human_text = "\n".join(human_parts)
    return template, human_text


def get_current_date_string() -> str:
    current = now()
    days_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    months_fr = [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ]
    day_name = days_fr[current.weekday()]
    month_name = months_fr[current.month - 1]
    return (
        f"{day_name} {current.day} {month_name} {current.year}, {current.hour}h{current.minute:02d}"
    )


def format_handoff_delay(ticket_timestamp: str) -> str:
    ticket_time = datetime.fromisoformat(ticket_timestamp)
    if ticket_time.tzinfo is None:
        ticket_time = ticket_time.replace(tzinfo=PARIS)
    else:
        ticket_time = ticket_time.astimezone(PARIS)

    if ticket_time - now() <= timedelta(minutes=45):
        return "Nous vous rappelons sous 45 minutes."
    return "Nous vous rappelons demain matin."
