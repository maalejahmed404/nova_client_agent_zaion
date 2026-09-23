import argparse
import hashlib
import json
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi
from snowballstemmer import FrenchStemmer

from neova.config import CACHE_DIR, CORPUS_DIR, get_settings
from neova.llm import embed_texts
from neova.rag.ingestion import build_corpus
from neova.rag.models import Chunk
from neova.rag.references import build_references
from neova.rag.table_text import PROMPT_VERSION, enriched_search_text

_STEMMER = FrenchStemmer()


class StaleIndex(Exception):
    pass


def tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    words = []
    current = []
    for c in normalized:
        if c.isalpha() or c.isdigit():
            current.append(c)
        else:
            if current:
                words.append("".join(current))
                current.clear()
    if current:
        words.append("".join(current))
    stems = []
    for word in words:
        stem = _STEMMER.stemWord(word)
        stems.append(stem)
    return stems


def corpus_fingerprint() -> str:
    settings = get_settings()
    files = []
    if CORPUS_DIR.exists():
        for file_path in sorted(CORPUS_DIR.glob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                files.append((file_path.name, file_hash))
    model_info = f"{settings.embedding_model}|{settings.chat_model}|{PROMPT_VERSION}"
    combined = "\n".join(f"{name}:{hash}" for name, hash in files) + "\n" + model_info
    return hashlib.sha256(combined.encode()).hexdigest()


def build_index(variant: str = "B", offline: bool = False) -> tuple[list[Chunk], np.ndarray]:
    if variant not in ("A", "B"):
        raise ValueError("variant must be 'A' or 'B'")
    chunks = build_corpus(offline=offline)
    build_references(chunks)
    if variant == "B":
        for chunk in chunks:
            chunk.search_text = enriched_search_text(chunk, offline=offline)
    texts = [chunk.search_text for chunk in chunks]
    vectors = embed_texts(texts)
    vectors_np = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    vectors_np = vectors_np / np.where(norms == 0, 1, norms)
    index_dir = CACHE_DIR / "index" / variant
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = index_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([chunk.to_dict() for chunk in chunks], f, ensure_ascii=False, indent=2)
    vectors_path = index_dir / "vectors.npy"
    np.save(vectors_path, vectors_np)
    fingerprint_path = index_dir / "fingerprint.txt"
    fingerprint_path.write_text(corpus_fingerprint(), encoding="utf-8")
    return chunks, vectors_np


def load_index(variant: str = "B") -> tuple[list[Chunk], np.ndarray]:
    index_dir = CACHE_DIR / "index" / variant
    fingerprint_path = index_dir / "fingerprint.txt"
    if not fingerprint_path.exists():
        raise StaleIndex(f"No fingerprint found. Run build_index(variant='{variant}') first.")
    stored_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip()
    current_fingerprint = corpus_fingerprint()
    if stored_fingerprint != current_fingerprint:
        raise StaleIndex("Corpus or models changed. Rebuild the index with build_index().")
    chunks_path = index_dir / "chunks.json"
    with open(chunks_path, encoding="utf-8") as f:
        chunk_dicts = json.load(f)
    chunks = [Chunk.from_dict(d) for d in chunk_dicts]
    for chunk in chunks:
        if chunk.audience != "public":
            raise StaleIndex(f"Chunk {chunk.chunk_id} has audience {chunk.audience}, not public.")
    vectors_path = index_dir / "vectors.npy"
    vectors = np.load(vectors_path)
    return chunks, vectors


def searchable(chunk: Chunk, historical: bool, contract_start: str | None) -> bool:
    if chunk.audience != "public":
        return False
    if chunk.statut != "deprecated":
        return True
    if historical:
        return True
    if contract_start and chunk.offer_window:
        window_from = chunk.offer_window.get("from")
        window_to = chunk.offer_window.get("to")
        if window_from and window_to:
            return window_from <= contract_start[:10] <= window_to
    return False


def retrieve(
    chunks: list[Chunk],
    vectors: np.ndarray,
    question: str,
    historical: bool = False,
    contract_start: str | None = None,
    k: int = 5,
) -> list[dict]:
    searchable_indices = [
        i for i, chunk in enumerate(chunks) if searchable(chunk, historical, contract_start)
    ]
    if not searchable_indices:
        return []
    searchable_chunks = [chunks[i] for i in searchable_indices]
    searchable_vectors = vectors[searchable_indices]
    question_tokens = tokens(question)
    corpus_tokens = [tokens(chunk.search_text) for chunk in searchable_chunks]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(question_tokens)
    bm25_ranking = np.argsort(bm25_scores)[::-1]
    bm25_position = {idx: pos for pos, idx in enumerate(bm25_ranking)}
    question_vector = embed_texts([question])[0]
    question_vector_np = np.array(question_vector, dtype=np.float32)
    question_vector_norm = np.linalg.norm(question_vector_np)
    question_vector_np = question_vector_np / np.where(
        question_vector_norm == 0, 1, question_vector_norm
    )
    cosine_scores = np.dot(searchable_vectors, question_vector_np)
    cosine_ranking = np.argsort(cosine_scores)[::-1]
    cosine_position = {idx: pos for pos, idx in enumerate(cosine_ranking)}
    scores = {}
    for i in range(len(searchable_chunks)):
        score = 0.0
        if i in bm25_position:
            score += 1.0 / (60.0 + bm25_position[i])
        if i in cosine_position:
            score += 1.0 / (60.0 + cosine_position[i])
        scores[i] = score
    sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    def tie_breaker(i: int) -> tuple:
        chunk = searchable_chunks[i]
        is_current = chunk.statut == "current"
        date = chunk.updated or chunk.effective_from or ""
        return (is_current, date)

    sorted_indices = sorted(sorted_indices, key=lambda i: (scores[i], tie_breaker(i)), reverse=True)
    kept_indices = sorted_indices[:k]
    kept_chunks = [searchable_chunks[i] for i in kept_indices]
    ref_map_path = CACHE_DIR / "corpus" / "ref_map.json"
    if ref_map_path.exists():
        with open(ref_map_path, encoding="utf-8") as f:
            ref_map = json.load(f)
    else:
        ref_map = {}
    searched_results = []
    seen_ids = set()
    for i, idx in enumerate(kept_indices):
        chunk = searchable_chunks[idx]
        if chunk.chunk_id in seen_ids:
            continue
        seen_ids.add(chunk.chunk_id)
        searched_results.append(
            {
                "chunk": chunk,
                "score": scores[idx],
                "origin": "search",
                "cited_by": None,
                "rank": i,
            }
        )
    all_results = searched_results.copy()
    for i, result in enumerate(searched_results):
        chunk = result["chunk"]
        if chunk.chunk_id in ref_map:
            for ref in ref_map[chunk.chunk_id]:
                target_id = ref["target"]
                if target_id in seen_ids:
                    continue
                target_idx = next(
                    (j for j, c in enumerate(searchable_chunks) if c.chunk_id == target_id),
                    None,
                )
                if target_idx is not None:
                    target_chunk = searchable_chunks[target_idx]
                    seen_ids.add(target_id)
                    all_results.append(
                        {
                            "chunk": target_chunk,
                            "score": scores[target_idx] if target_idx in scores else 0.0,
                            "origin": "reference",
                            "cited_by": chunk.chunk_id,
                            "rank": i,
                        }
                    )
    return all_results


def format_results(results: list[dict]) -> str:
    if not results:
        return "Aucun passage trouvé."
    lines = []
    for i, result in enumerate(results):
        chunk = result["chunk"]
        if result["origin"] == "search":
            rank = result.get("rank", i)
            header = f"# passage {rank + 1} · {chunk.chunk_id}"
        else:
            header = f"# passage cité par {result['rank'] + 1} · {chunk.chunk_id}"
        lines.append(header)
        lines.append(chunk.text)
        if i < len(results) - 1:
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the index")
    parser.add_argument("--variant", choices=["A", "B"], default="B", help="Index variant")
    parser.add_argument("--offline", action="store_true", help="Run offline")
    parser.add_argument("--question", type=str, help="Question to retrieve passages for")
    parser.add_argument(
        "--historical", action="store_true", help="Include historical/deprecated chunks"
    )
    parser.add_argument(
        "--contract-start", type=str, help="Customer contract start date (YYYY-MM-DD)"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of top passages to retrieve")
    args = parser.parse_args()
    if args.build:
        chunks, vectors = build_index(variant=args.variant, offline=args.offline)
        index_dir = CACHE_DIR / "index" / args.variant
        print(f"Indexed {len(chunks)} chunks.")
        print(f"Files written to {index_dir}")
    elif args.question:
        chunks, vectors = load_index(variant=args.variant)
        results = retrieve(
            chunks,
            vectors,
            args.question,
            historical=args.historical,
            contract_start=args.contract_start,
            k=args.k,
        )
        print(format_results(results))
        for i, result in enumerate(results):
            print(
                f"{i + 1}. {result['chunk'].chunk_id} - score: {result['score']:.4f} - origin: {result['origin']}"
            )
    else:
        parser.print_help()
