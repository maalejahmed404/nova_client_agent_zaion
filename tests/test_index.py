import tempfile
import zlib
from pathlib import Path

import pytest

from neova.agent import workflow
from neova.rag import index as rag_index
from neova.rag.index import StaleIndex, build_index, load_index, retrieve, searchable, tokens
from neova.rag.ingest import build_corpus

ART12 = "cgv-resiliation#article-12-frais-de-resiliation-anticipee"
ART14 = "cgv-resiliation#article-14-restitution-des-equipements"
GRID_FIBRE = "grille-tarifaire-2026#offres-fibre-residentielles"
PROMO = "promo-rentree-2024#tarifs-promotionnels"


def fake_embed(texts, instruction=None):
    """Deterministic bag-of-stems vectors: enough to test the plumbing without any API call."""
    vectors = []
    for text in texts:
        v = [0.0] * 256
        for t in tokens(text):
            v[zlib.crc32(t.encode()) % 256] += 1.0
        vectors.append([x + 1e-6 for x in v])
    return vectors


@pytest.fixture(scope="module")
def corpus():
    try:
        return build_corpus(offline=True)
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def index(corpus, monkeypatch):
    monkeypatch.setattr(rag_index.llm, "embed_texts", fake_embed)
    with tempfile.TemporaryDirectory() as root:
        yield build_index(corpus, Path(root), variant="A")


def ids(results, via=None):
    return [r.chunk.chunk_id for r in results if via is None or r.via == via]


def test_tokens_keep_numbers_and_stem_french():
    assert tokens("80 Go") == ["80", "go"]
    assert tokens("résiliation")[0][:5] == tokens("résilier")[0][:5] == "resil"


def test_current_question_never_sees_the_archive(index):
    results = retrieve(index, ["combien coûte la fibre 500 Mb/s par mois ?"])
    assert GRID_FIBRE in ids(results, "search")
    assert not any(r.chunk.statut == "deprecated" for r in results)


def test_explicit_historical_question_reaches_the_archive_without_a_customer(index):
    results = retrieve(index, ["prix de l'offre Rentrée 2024 fibre 500 première année"], historical=True)
    promo = next(r for r in results if r.chunk.chunk_id == PROMO)
    assert "ARCHIVE" in promo.chunk.text


def test_contract_date_in_the_window_makes_the_archive_searchable(index):
    question = ["prix fibre 500 première année prix ensuite"]
    assert PROMO in ids(retrieve(index, question, contract_start="2024-10-15"))
    assert PROMO not in ids(retrieve(index, question, contract_start="2025-03-01"))


def test_article_12_first_still_brings_article_14(index):
    results = retrieve(index, ["frais de résiliation anticipée mensualités restant dues"], k=1,
                       required=workflow.required_chunk_ids("termination_cost"))
    assert ids(results)[0] == ART12
    assert ART14 in ids(results)


def test_required_evidence_survives_a_full_top_k(index):
    required = workflow.required_chunk_ids("box_diagnosis")
    results = retrieve(index, ["appel hors Union européenne tarif minute"], k=5, required=required)
    assert len(ids(results, "search")) == 5
    assert set(required) <= set(ids(results))


def test_every_workflow_target_exists(corpus):
    known = {c.chunk_id for c in corpus.chunks}
    for intent in workflow.intents():
        assert set(workflow.required_chunk_ids(intent)) <= known, intent


def test_internal_chunks_are_never_searchable(corpus):
    internal = corpus.chunks[0].__class__.from_dict({**corpus.chunks[0].to_dict(), "audience": "internal"})
    assert not searchable(internal, historical=True, contract_start=None)
    assert all(c.audience == "public" for c in corpus.chunks)


def test_stale_index_is_refused(corpus, monkeypatch):
    monkeypatch.setattr(rag_index.llm, "embed_texts", fake_embed)
    with tempfile.TemporaryDirectory() as root:
        build_index(corpus, Path(root), variant="A")
        load_index(Path(root), variant="A")
        (Path(root) / "fingerprint.txt").write_text("built from other files", encoding="utf-8")
        with pytest.raises(StaleIndex):
            load_index(Path(root), variant="A")
