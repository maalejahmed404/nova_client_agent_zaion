"""Answer step: answers only from the retrieved context (plus the customer's API facts and, on the
gesture route, the decision already taken), or says the corpus does not contain the answer."""
import logging
from pathlib import Path

from pydantic import BaseModel

from neova import llm
from neova.agent.prompts import data
from neova.rag.index import Retrieved

log = logging.getLogger(__name__)
TEMPLATE_PATH = Path(__file__).with_name("templates") / "answer.txt"


class Answer(BaseModel):
    in_corpus: bool
    answer: str
    sources: list[str]


def _escape(text: str) -> str:
    return text.replace("<", "‹").replace(">", "›")


def context_block(results: list[Retrieved]) -> str:
    extracts = "\n\n".join(f"<extrait id={r.chunk.chunk_id}>\n{_escape(r.chunk.text)}\n</extrait>" for r in results)
    return f"<contexte>\n{extracts}\n</contexte>"


def answer(question: str, results: list[Retrieved], facts: dict | None = None, decision: str | None = None) -> Answer:
    blocks = [context_block(results)]
    if facts:
        blocks.append(data("faits_client", facts))
    if decision:
        blocks.append(data("decision", decision))
    blocks.append(f"<question>\n{_escape(question)}\n</question>")
    messages = [("system", TEMPLATE_PATH.read_text(encoding="utf-8")), ("human", "\n\n".join(blocks))]
    try:
        reply = llm.structured_call(messages, Answer)
    except Exception as exc:
        log.warning("answer call failed, treated as not in corpus: %s", exc)
        return Answer(in_corpus=False, answer="", sources=[])
    known = {r.chunk.chunk_id for r in results}
    sources = [s for s in reply.sources if s in known]
    grounded = bool(sources) or decision is not None
    return reply.model_copy(update={"in_corpus": reply.in_corpus and grounded, "sources": sources})
