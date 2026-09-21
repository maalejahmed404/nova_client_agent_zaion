"""Evaluation on the hand-written gold in eval/retrieval_gold.jsonl:
- retrieval: do the chunks an answer needs reach the context (questions with an answer);
- similarity: best cosine for questions with and without an answer in the corpus;
- LLM behaviour: does the answer step say "not in corpus" exactly when it should.

uv run python -m neova.rag.eval_retrieval [--no-llm]"""
import argparse
import json
import statistics

from neova import llm
from neova.config import PROJECT_ROOT
from neova.rag.answer import answer
from neova.rag.index import QUERY_INSTRUCTION, _normalise, load_index, retrieve, searchable

GOLD_PATH = PROJECT_ROOT / "eval" / "retrieval_gold.jsonl"


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_similarity(index, g: dict) -> float:
    query = _normalise(llm.embed_texts([g["q"]], instruction=QUERY_INSTRUCTION))[0]
    allowed = [i for i, c in enumerate(index.chunks) if searchable(c, g.get("historical", False), g.get("contract_start"))]
    return float(max(index.vectors[i] @ query for i in allowed))


def evaluate(index, gold: list[dict], use_llm: bool) -> list[dict]:
    rows = []
    for g in gold:
        results = retrieve(index, [g["q"]], historical=g.get("historical", False), contract_start=g.get("contract_start"))
        context = [r.chunk.chunk_id for r in results]
        searched = [r.chunk.chunk_id for r in results if r.via == "search"]
        expected, in_corpus = g["expected"], g.get("in_corpus", True)
        archive_expected = any(index.by_id[e].statut == "deprecated" for e in expected)
        row = {
            "id": g["id"], "in_corpus": in_corpus, "similarity": best_similarity(index, g),
            "full": set(expected) <= set(context),
            "rank": searched.index(expected[0]) + 1 if expected and expected[0] in searched else None,
            "missing": sorted(set(expected) - set(context)),
            "archive_leak": [] if archive_expected else [c for c in context if index.by_id[c].statut == "deprecated"],
            "top3": searched[:3],
        }
        if use_llm:
            reply = answer(g["q"], results)
            row["said_in_corpus"], row["answer"] = reply.in_corpus, reply.answer
        rows.append(row)
    return rows


def report(rows: list[dict], use_llm: bool) -> None:
    pos = [r for r in rows if r["in_corpus"]]
    neg = [r for r in rows if not r["in_corpus"]]
    print(f"\n== retrieval ({len(pos)} questions with an answer)")
    print(f"full-evidence recall    {sum(r['full'] for r in pos) / len(pos):.2f}")
    print(f"MRR (first expected)    {sum(1 / r['rank'] for r in pos if r['rank']) / len(pos):.2f}")
    print(f"archive leaks           {sum(bool(r['archive_leak']) for r in rows)}")
    for r in pos:
        if not r["full"] or r["archive_leak"]:
            print(f"  MISS {r['id']}: missing {r['missing']} leak {r['archive_leak']} top3 {r['top3']}")

    print(f"\n== best cosine similarity ({len(pos)} with an answer, {len(neg)} without)")
    for name, group in (("answer in corpus", pos), ("answer not in corpus", neg)):
        s = sorted(r["similarity"] for r in group)
        print(f"{name:22}  min {s[0]:.3f}  median {statistics.median(s):.3f}  max {s[-1]:.3f}")
    threshold = max(r["similarity"] for r in neg)
    wrongly_cut = [r["id"] for r in pos if r["similarity"] <= threshold]
    print(f"a threshold above every negative ({threshold:.3f}) would also cut {len(wrongly_cut)} answerable questions: {wrongly_cut}")

    if use_llm:
        print("\n== LLM behaviour")
        refused_neg = [r for r in neg if not r["said_in_corpus"]]
        refused_pos = [r for r in pos if not r["said_in_corpus"]]
        print(f"negatives answered 'not in corpus'   {len(refused_neg)}/{len(neg)}")
        print(f"answerable questions refused         {len(refused_pos)}/{len(pos)}")
        for r in neg:
            if r["said_in_corpus"]:
                print(f"  ANSWERED A NEGATIVE {r['id']}: {r['answer']}")
        for r in refused_pos:
            print(f"  REFUSED {r['id']}: {r['answer']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B"], default="B")
    ap.add_argument("--no-llm", action="store_true", help="skip the answer step (no chat model call)")
    args = ap.parse_args()
    report(evaluate(load_index(variant=args.variant), load_gold(), not args.no_llm), not args.no_llm)
