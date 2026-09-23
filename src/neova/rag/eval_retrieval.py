import argparse
import json
import statistics
import numpy as np

from neova.config import PROJECT_ROOT, get_settings
from neova.llm import embed_texts
from neova.rag.index import load_index, retrieve
from neova.rag.models import Chunk

GOLD_PATH = PROJECT_ROOT / "eval" / "retrieval_gold.jsonl"


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_cosine_similarity(chunks: list[Chunk], vectors: np.ndarray, question: str) -> float:
    question_embedding = embed_texts([question])[0]
    question_vector = np.array(question_embedding, dtype=np.float32)
    norm = np.linalg.norm(question_vector)
    question_vector = question_vector / np.where(norm == 0, 1, norm)
    cosine_scores = np.dot(vectors, question_vector)
    return float(np.max(cosine_scores))


def evaluate(chunks: list[Chunk], vectors: np.ndarray, gold: list[dict]) -> list[dict]:
    rows = []
    for g in gold:
        results = retrieve(
            chunks, vectors, g["q"], 
            historical=g.get("historical", False), 
            contract_start=g.get("contract_start")
        )
        returned_ids = [result["chunk"].chunk_id for result in results]
        expected = g["expected"]
        in_corpus = g.get("in_corpus", True)
        
        archive_leak = []
        if not g.get("historical", False):
            for result in results:
                chunk = result["chunk"]
                if chunk.statut == "deprecated":
                    archive_leak.append(chunk.chunk_id)
        
        searched_results = [r for r in results if r["origin"] == "search"]
        searched_ids = [r["chunk"].chunk_id for r in searched_results]
        
        full_match = True
        for exp in expected:
            if exp not in returned_ids:
                full_match = False
                break
        
        first_expected_rank = None
        if expected:
            for i, chunk_id in enumerate(searched_ids):
                if chunk_id == expected[0]:
                    first_expected_rank = i + 1
                    break
        
        similarity = best_cosine_similarity(chunks, vectors, g["q"])
        
        rows.append({
            "id": g["id"],
            "question": g["q"],
            "expected": expected,
            "returned": returned_ids,
            "in_corpus": in_corpus,
            "similarity": similarity,
            "full_match": full_match,
            "any_match": any(exp in returned_ids for exp in expected) if expected else False,
            "first_expected_rank": first_expected_rank,
            "archive_leak": archive_leak,
            "searched_top3": searched_ids[:3],
        })
    return rows


def report(rows: list[dict]) -> None:
    with_answer = [r for r in rows if r["in_corpus"]]
    without_answer = [r for r in rows if not r["in_corpus"]]
    
    print(f"\n== retrieval ({len(with_answer)} questions with an answer)")
    if with_answer:
        full_recall = sum(r["full_match"] for r in with_answer) / len(with_answer)
        print(f"full-evidence recall    {full_recall:.2f}")
        
        any_recall = sum(r["any_match"] for r in with_answer) / len(with_answer)
        print(f"any-evidence recall    {any_recall:.2f}")
        
        mrr_numerator = 0
        for r in with_answer:
            if r["first_expected_rank"] is not None:
                mrr_numerator += 1 / r["first_expected_rank"]
        mrr = mrr_numerator / len(with_answer) if with_answer else 0
        print(f"MRR (first expected)    {mrr:.2f}")
    
    archive_leak_count = sum(len(r["archive_leak"]) > 0 for r in rows if not r.get("historical", False))
    print(f"archive leaks           {archive_leak_count}")
    
    for r in with_answer:
        if not r["full_match"] or r["archive_leak"]:
            print(f"  FAIL {r['id']}: {r['question']}")
            print(f"    expected: {r['expected']}")
            print(f"    returned: {r['returned']}")
            if r["archive_leak"]:
                print(f"    archive leak: {r['archive_leak']}")
    
    print(f"\n== best cosine similarity ({len(with_answer)} with an answer, {len(without_answer)} without)")
    if with_answer:
        with_similarities = sorted(r["similarity"] for r in with_answer)
        print(f"answer in corpus         min {with_similarities[0]:.3f}  median {statistics.median(with_similarities):.3f}  max {with_similarities[-1]:.3f}")
    
    if without_answer:
        without_similarities = sorted(r["similarity"] for r in without_answer)
        print(f"answer not in corpus     min {without_similarities[0]:.3f}  median {statistics.median(without_similarities):.3f}  max {without_similarities[-1]:.3f}")
    
    if with_answer and without_answer:
        max_negative = max(r["similarity"] for r in without_answer)
        wrongly_cut = [r["id"] for r in with_answer if r["similarity"] <= max_negative]
        print(f"a threshold above every negative ({max_negative:.3f}) would also cut {len(wrongly_cut)} answerable questions: {wrongly_cut}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "B"], default="B")
    args = parser.parse_args()
    
    chunks, vectors = load_index(variant=args.variant)
    gold = load_gold()
    rows = evaluate(chunks, vectors, gold)
    report(rows)