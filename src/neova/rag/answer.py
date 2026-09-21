"""Minimal answer step: answers only from the retrieved context, or says the corpus does not
contain the answer. Used by the retrieval evaluation now, by the graph later."""
import logging
from pathlib import Path

from pydantic import BaseModel

from neova import llm
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


def answer(question: str, results: list[Retrieved]) -> Answer:
    messages = [("system", TEMPLATE_PATH.read_text(encoding="utf-8")),
                ("human", f"{context_block(results)}\n\n<question>\n{_escape(question)}\n</question>")]
    try:
        reply = llm.structured_call(messages, Answer)
    except Exception as exc:
        log.warning("answer call failed, treated as not in corpus: %s", exc)
        return Answer(in_corpus=False, answer="", sources=[])
    known = {r.chunk.chunk_id for r in results}
    sources = [s for s in reply.sources if s in known]
    return reply.model_copy(update={"in_corpus": reply.in_corpus and bool(sources), "sources": sources})
