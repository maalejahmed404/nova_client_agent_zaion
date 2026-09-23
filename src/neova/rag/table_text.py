"""Stage B, data preparation only: a sentence version of each table, written once by an LLM and
appended to the chunk's search text. It is never shown to the answering LLM. A table whose
generated text fails a check keeps its verbatim records only."""

import hashlib
import json
import logging
import re
import unicodedata
from decimal import Decimal

from pydantic import BaseModel

from neova import llm
from neova.config import CACHE_DIR, get_settings
from neova.rag.models import Chunk, Table

log = logging.getLogger(__name__)
PROMPT_VERSION = "1"


def fold(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in stripped if not unicodedata.combining(c))


SYSTEM = (
    "Tu prépares un texte de recherche pour un tableau d'un document du service client de Néova Télécom.\n"
    "- topic : une ligne qui dit de quoi parle le tableau, suivie entre parenthèses des mots qu'un client "
    "emploierait pour le chercher. Si le document est une archive, dis-le.\n"
    "- sentences : exactement une phrase par ligne du tableau, dans l'ordre. Chaque phrase reprend le premier "
    "élément de sa ligne tel qu'il est écrit et toutes les valeurs de cette ligne.\n"
    "N'ajoute aucune valeur, date, produit ou condition absent du tableau ou de la note."
)


class TableText(BaseModel):
    topic: str
    sentences: list[str]


def numbers(text: str) -> set[str]:
    return {
        str(Decimal(n.replace(",", ".")).normalize()) for n in re.findall(r"\d+(?:[.,]\d+)?", text)
    }


def problems(table: Table, notice: str, generated: TableText) -> list[str]:
    rows, found = table.rows, []
    if len(generated.sentences) != len(rows):
        return [f"{len(generated.sentences)} sentences for {len(rows)} rows"]
    allowed = numbers(notice + " " + " ".join(table.headers))
    all_cells = allowed | numbers(" ".join(x for r in rows for x in r))
    for row, sentence in zip(rows, generated.sentences):
        own, said = numbers(" ".join(row)), numbers(sentence)
        if fold(row[0]) not in fold(sentence):
            found.append(f"row {row[0]!r} not named")
        if not own <= said:
            found.append(f"row {row[0]!r} misses {sorted(own - said)}")
        if said - own - allowed:
            found.append(f"row {row[0]!r} carries foreign numbers {sorted(said - own - allowed)}")
    if numbers(generated.topic) - all_cells:
        found.append(f"topic carries numbers {sorted(numbers(generated.topic) - all_cells)}")
    return found


def _input(chunk: Chunk, table: Table) -> tuple[str, str]:
    description = chunk.text.split("\n\n", 1)[0]
    notice = "\n".join(line for line in description.splitlines() if line != "# description")
    records = "\n".join(
        " · ".join(f"{h} : {c}" for h, c in zip(table.headers, r, strict=False)) for r in table.rows
    )
    return notice, f"{notice}\n\nTableau :\n{records}"


def generate(chunk: Chunk, table: Table, offline: bool = False) -> str | None:
    notice, prompt = _input(chunk, table)
    key = hashlib.sha256(
        f"{get_settings().chat_model}|{PROMPT_VERSION}|{prompt}".encode()
    ).hexdigest()
    path = CACHE_DIR / "table_text" / f"{key}.json"
    if path.exists():
        generated = TableText(**json.loads(path.read_text(encoding="utf-8")))
    elif offline:
        raise RuntimeError(f"no cached table text for {chunk.chunk_id} and offline mode is on")
    else:
        chat = llm.create_chat_client(callbacks=[llm.CostLoggerCallback("table_text")])
        generated = llm.invoke_with_retry(
            chat.with_structured_output(TableText, method="function_calling"),
            [("system", SYSTEM), ("human", prompt)],
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated.model_dump_json(), encoding="utf-8")
    issues = problems(table, notice, generated)
    if issues:
        log.warning("table text rejected for %s: %s", chunk.chunk_id, issues)
        return None
    return "\n".join([generated.topic, *generated.sentences])


def enriched_search_text(chunk: Chunk, offline: bool = False) -> str:
    extra = [t for t in (generate(chunk, table, offline) for table in chunk.tables) if t]
    return "\n".join([chunk.search_text, *extra])
