import tempfile
from pathlib import Path

import pytest

from neova.rag import table_text
from neova.rag.models import Chunk
from neova.rag.table_text import TableText, problems

TABLE = {"headers": ["Offre", "Prix mensuel", "Engagement"],
         "rows": [["Fibre Néova 500 Mb/s", "29,99 €", "12 ou 24 mois"], ["Fibre Néova 1 Gb/s", "39,99 €", "12 ou 24 mois"]]}
NOTICE = "Les tarifs sont exprimés en euros TTC par mois."
GOOD = TableText(topic="Tarifs des abonnements fibre Néova (prix, combien coûte, box internet)",
                 sentences=["L'offre Fibre Néova 500 Mb/s coûte 29.99 € par mois, engagement 12 ou 24 mois.",
                            "L'offre Fibre Néova 1 Gb/s coûte 39,99 € par mois, engagement 12 ou 24 mois."])


def chunk() -> Chunk:
    text = ("Grille > Offres fibre\n[Source : grille · EN VIGUEUR]\nNote : " + NOTICE +
            "\n\n- Offre : Fibre Néova 500 Mb/s · Prix mensuel : 29,99 €")
    return Chunk("grille#fibre", "grille", "Grille", ["Grille", "Offres fibre"], [1], "public", "current",
                 None, None, None, None, "h", "pdf", True, [TABLE], text, "Grille > Offres fibre\n...")


def test_faithful_text_passes():
    assert problems(TABLE, NOTICE, GOOD) == []


def test_swapped_prices_are_rejected():
    swapped = GOOD.model_copy(update={"sentences": [
        "L'offre Fibre Néova 500 Mb/s coûte 39,99 € par mois, engagement 12 ou 24 mois.",
        "L'offre Fibre Néova 1 Gb/s coûte 29,99 € par mois, engagement 12 ou 24 mois."]})
    assert problems(TABLE, NOTICE, swapped)


def test_invented_number_and_unnamed_row_are_rejected():
    invented = GOOD.model_copy(update={"topic": "Tarifs fibre, frais d'activation 12,50 €"})
    assert problems(TABLE, NOTICE, invented)
    unnamed = GOOD.model_copy(update={"sentences": ["La fibre coûte 29,99 € par mois, engagement 12 ou 24 mois.",
                                                    GOOD.sentences[1]]})
    assert problems(TABLE, NOTICE, unnamed)
    assert problems(TABLE, NOTICE, GOOD.model_copy(update={"sentences": GOOD.sentences[:1]}))


def test_cache_avoids_the_second_call_and_offline_needs_the_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(table_text.llm, "get_chat", lambda **kw: type("Chat", (), {"with_structured_output": lambda self, *a, **k: None})())
    monkeypatch.setattr(table_text.llm, "invoke_with_retry", lambda runnable, messages: calls.append(1) or GOOD)
    with tempfile.TemporaryDirectory() as root:
        monkeypatch.setattr(table_text, "CACHE_DIR", Path(root))
        with pytest.raises(RuntimeError):
            table_text.generate(chunk(), TABLE, offline=True)
        first = table_text.generate(chunk(), TABLE)
        second = table_text.generate(chunk(), TABLE, offline=True)
    assert first == second and len(calls) == 1


def test_generated_text_is_appended_never_substituted(monkeypatch):
    monkeypatch.setattr(table_text, "generate", lambda c, t, offline=False: "Tarifs fibre (prix)")
    text = table_text.enriched_search_text(chunk())
    assert text.startswith(chunk().search_text) and text.endswith("Tarifs fibre (prix)")
