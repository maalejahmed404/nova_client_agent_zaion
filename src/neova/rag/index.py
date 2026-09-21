"""Hybrid search over the public chunks: BM25 (exact names, numbers) + embeddings (meaning),
fused by reciprocal rank. Metadata decides what may be searched (archives) and what is added
after the top-k (references).

Two variants are built so their benefit can be measured: A indexes the verbatim sections, B (the
default) also indexes an LLM-written sentence version of each table (see table_text.py).

Build:  uv run python -m neova.rag.index --build --variant A|B [--offline]"""
import argparse
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from neova import llm
from neova.config import CACHE_DIR, CORPUS_DIR, get_settings
from neova.rag.ingest import build_corpus, fold
from neova.rag import table_text
from neova.rag.models import Chunk, Corpus

QUERY_INSTRUCTION = "Given a customer question in French, retrieve the support document section that answers it"
RRF_K = 60
_stemmer = snowballstemmer.stemmer("french")


class StaleIndex(Exception):
    """The index was built from other source files, another embedding model or other search texts."""


def tokens(text: str) -> list[str]:
    return _stemmer.stemWords(re.findall(r"[a-z0-9]+", fold(text)))


def index_dir(variant: str = "B") -> Path:
    return CACHE_DIR / "index" / get_settings().embedding_model.replace("/", "_") / variant


def fingerprint(variant: str = "B", corpus_dir: Path = CORPUS_DIR) -> str:
    text_version = f"B{table_text.PROMPT_VERSION}|{get_settings().chat_model}" if variant == "B" else "A"
    h = hashlib.sha256(f"{get_settings().embedding_model}|{text_version}".encode())
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() in (".pdf", ".png"):
            h.update(path.name.encode() + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray  # one L2-normalised row per chunk

    def __post_init__(self):
        self.bm25 = BM25Okapi([tokens(c.search_text) for c in self.chunks])
        self.by_id = {c.chunk_id: c for c in self.chunks}


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    via: str  # search | reference


def _normalise(vectors) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def build_index(corpus: Corpus, directory: Path | None = None, variant: str = "B", offline: bool = False) -> Index:
    directory = directory or index_dir(variant)
    chunks = corpus.chunks
    if variant == "B":
        chunks = [replace(c, search_text=table_text.enriched_search_text(c, offline)) if c.tables else c for c in chunks]
    vectors = _normalise(llm.embed_texts([c.search_text for c in chunks]))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False), encoding="utf-8")
    np.save(directory / "vectors.npy", vectors)
    (directory / "fingerprint.txt").write_text(fingerprint(variant), encoding="utf-8")
    return Index(chunks, vectors)


def load_index(directory: Path | None = None, variant: str = "B") -> Index:
    directory = directory or index_dir(variant)
    if not (directory / "fingerprint.txt").exists():
        raise StaleIndex(f"no index in {directory}: run `uv run python -m neova.rag.index --build`")
    if (directory / "fingerprint.txt").read_text(encoding="utf-8") != fingerprint(variant):
        raise StaleIndex("corpus files, embedding model or search texts changed: rebuild the index")
    chunks = [Chunk.from_dict(d) for d in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
    if any(c.audience != "public" for c in chunks):
        raise StaleIndex("a non-public chunk is in the index: rebuild it")
    return Index(chunks, np.load(directory / "vectors.npy"))


def searchable(chunk: Chunk, historical: bool, contract_start: str | None) -> bool:
    """Archives are searched only for an explicit historical question, or when the customer's
    contract started inside the archived offer's window (relevant to consult, not proof of the offer)."""
    if chunk.audience != "public":
        return False
    if chunk.statut != "deprecated":
        return True
    window = chunk.offer_window
    in_window = bool(contract_start and window and window["from"] <= contract_start[:10] <= window["to"])
    return historical or in_window


def _recency(chunk: Chunk) -> tuple:
    return (chunk.statut == "current", chunk.effective_from or chunk.updated or "")


def retrieve(index: Index, queries: list[str], *, historical: bool = False, contract_start: str | None = None,
             k: int = 5) -> list[Retrieved]:
    allowed = [i for i, c in enumerate(index.chunks) if searchable(c, historical, contract_start)]
    fused: dict[int, float] = {}
    query_vectors = _normalise(llm.embed_texts(queries, instruction=QUERY_INSTRUCTION))
    for query, qv in zip(queries, query_vectors):
        lexical = index.bm25.get_scores(tokens(query))
        dense = index.vectors @ qv
        for scores in (lexical, dense):
            ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    ranked = sorted(fused, key=lambda i: (fused[i], _recency(index.chunks[i])), reverse=True)[:k]

    results = [Retrieved(index.chunks[i], fused[i], "search") for i in ranked]
    seen = {r.chunk.chunk_id for r in results}

    def add(chunk_id: str, via: str):
        chunk = index.by_id.get(chunk_id)
        if chunk and chunk_id not in seen and searchable(chunk, historical, contract_start):
            seen.add(chunk_id)
            results.append(Retrieved(chunk, 0.0, via))

    for r in list(results):
        for chunk_id in r.chunk.references + r.chunk.referenced_by:
            add(chunk_id, "reference")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--variant", choices=["A", "B"], default="B")
    ap.add_argument("--offline", action="store_true", help="never call the vision or table-text model")
    args = ap.parse_args()
    if args.build:
        index = build_index(build_corpus(offline=args.offline), variant=args.variant, offline=args.offline)
        print(f"indexed {len(index.chunks)} chunks in {index_dir(args.variant)}")
