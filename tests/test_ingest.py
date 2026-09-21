"""Expectations about this corpus, written from the documents. The pipeline never reads them."""
import tempfile
from pathlib import Path

import pymupdf
import pytest

from neova import llm
from neova.config import CORPUS_DIR
from neova.rag import ingest
from neova.rag.ingest import build_corpus, fold, parse_markdown
from neova.rag.models import Chunk, Corpus

ART = "cgv-resiliation#article-{}"
ART11, ART12 = ART.format("11-demande-de-resiliation"), ART.format("12-frais-de-resiliation-anticipee")
ART13, ART14 = ART.format("13-cas-d-exoneration"), ART.format("14-restitution-des-equipements")

TABLES = {
    ART14: (["Équipement", "Indemnité"], [
        ["Box Néova 6", "89 €"], ["Box Néova 5", "69 €"], ["Décodeur TV", "49 €"],
        ["Routeur de secours 4G", "79 €"], ["Bloc d’alimentation ou télécommande seuls", "12 €"]]),
    "faq-box-internet#signification-des-voyants": (["Voyant", "État", "Signification"], [
        ["Blanc fixe", "Normal", "La connexion est établie, aucune action nécessaire."],
        ["Blanc clignotant", "Démarrage", "La box s’initialise. Compter 3 à 5 minutes."],
        ["Rouge fixe", "Panne", "Aucun signal reçu sur la ligne fibre."],
        ["Rouge clignotant", "Incident réseau", "Signal interrompu en amont du logement."],
        ["Orange", "Dégradé", "Connexion active mais débit réduit ou instable."],
        ["Éteint", "Sans alimentation", "Vérifier le bloc secteur et la prise murale."]]),
    "faq-espace-client#ce-que-le-client-peut-faire-seul": (["Action", "Disponible"], [
        ["Télécharger ses factures (24 derniers mois)", "oui"],
        ["Modifier son IBAN", "oui, effet à la facture suivante"],
        ["Modifier la date de prélèvement", "oui, une fois par an, vers le 20"],
        ["Activer / désactiver le blocage des données hors UE", "oui, immédiat"],
        ["Éditer une étiquette de retour d’équipement", "oui"],
        ["Suivre un incident réseau sur sa commune", "oui"],
        ["Modifier l’adresse de facturation", "oui"],
        ["Modifier le nom du titulaire", "non — vérification d’identité requise"],
        ["Résilier", "oui, avec préavis de 10 jours"]]),
    "fic-roam-2026-02#union-europeenne-dom-inclus": (["Offre", "Données utilisables depuis l'UE"], [
        ["Mobile Néova 5 Go", "5 Go"], ["Mobile Néova 80 Go", "25 Go"], ["Mobile Néova 200 Go", "35 Go"]]),
    "fic-roam-2026-02#hors-union-europeenne": (["Consommation", "Tarif"], [
        ["Appel émis", "0,50 € / minute"], ["Appel reçu", "0,25 € / minute"], ["SMS émis", "0,20 € par SMS"],
        ["Données", "5,00 € par tranche de 100 Mo entamée"]]),
    "grille-tarifaire-2026#offres-fibre-residentielles": (
        ["Offre", "Débit descendant / montant", "Prix mensuel", "Engagement"], [
            ["Fibre Néova 500 Mb/s", "500 Mb/s / 500 Mb/s", "29,99 €", "12 ou 24 mois"],
            ["Fibre Néova 1 Gb/s", "1 Gb/s / 700 Mb/s", "39,99 €", "12 ou 24 mois"]]),
    "grille-tarifaire-2026#offres-mobiles": (["Offre", "Données en France", "Prix mensuel", "Engagement"], [
        ["Mobile Néova 5 Go", "5 Go", "4,99 €", "sans engagement"],
        ["Mobile Néova 80 Go", "80 Go", "14,99 €", "sans engagement"],
        ["Mobile Néova 200 Go", "200 Go", "19,99 €", "sans engagement"]]),
    "grille-tarifaire-2026#options": (["Option", "Prix mensuel"], [
        ["Décodeur TV", "5,00 €"], ["Routeur de secours 4G", "9,00 €"], ["Extension de garantie équipement", "3,00 €"],
        ["Ligne mobile supplémentaire (client fibre)", "−20 % sur l’offre mobile choisie"]]),
    "grille-tarifaire-2026#frais-ponctuels": (["Frais", "Montant"], [
        ["Activation de la ligne", "12,50 €"], ["Rejet de prélèvement", "2,00 €"],
        ["Rétablissement après suspension", "15,00 €"], ["Intervention technicien imputable au client", "69,00 €"],
        ["Remplacement de carte SIM", "10,00 €"]]),
    "procedure-demenagement#frais": (["Situation", "Frais"], [
        ["Transfert vers un logement déjà raccordé", "Gratuit"], ["Transfert nécessitant un raccordement", "12,50 €"],
        ["Conservation de la box existante", "Gratuit"], ["Déménagement en zone non couverte", "Voir ci-dessous"]]),
    "promo-rentree-2024#tarifs-promotionnels": (["Offre", "Prix la première année", "Prix ensuite"], [
        ["Fibre Néova 500 Mb/s", "19,99 €/mois", "24,99 €/mois"],
        ["Fibre Néova 1 Gb/s", "24,99 €/mois", "29,99 €/mois"],
        ["Mobile Néova 80 Go", "9,99 €/mois", "12,99 €/mois"]]),
}

METADATA = {  # statut, updated, effective_from, offer_window, supersedes, source_format
    "cgv-resiliation": ("current", "2026-04-02", None, None, None, "pdf"),
    "faq-box-internet": ("current", "2026-05-18", None, None, None, "pdf"),
    "faq-espace-client": ("current", "2026-04-29", None, None, None, "pdf"),
    "faq-facturation": ("current", "2026-06-20", None, None, None, "pdf"),
    "faq-retour-equipement": ("current", "2026-01-15", None, None, None, "pdf"),
    "fic-roam-2026-02": ("current", "2026-02-27", None, None, "FIC-ROAM-2025-04", "image"),
    "grille-tarifaire-2026": ("current", "2026-06-01", "2026-06-01", None, None, "pdf"),
    "procedure-demenagement": ("current", "2026-03-11", None, None, None, "pdf"),
    "promo-rentree-2024": ("deprecated", "2024-09-01", None, {"from": "2024-09-01", "to": "2024-10-31"}, None, "pdf"),
}

REFERENCES = {  # read in the source text: "article N" / "voir CGV, article N"
    ART11: [ART12],
    ART14: [ART12],
    "grille-tarifaire-2026#offres-fibre-residentielles": [ART12],
    "faq-retour-equipement#delai": [ART14],
    "procedure-demenagement#zone-non-couverte": [ART13],
}


@pytest.fixture(scope="module")
def corpus():
    try:
        return build_corpus(offline=True)
    except RuntimeError as exc:  # no cached transcription of the scanned sheet
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def chunks(corpus):
    return {c.chunk_id: c for c in corpus.chunks}


def source_text(doc_id: str) -> str:
    """Raw text of the source file, read without the table extractor."""
    if doc_id.startswith("fic-roam"):
        return fold(llm.transcribe_image(CORPUS_DIR / "fiche-roaming-international-scan.png", offline=True))
    with pymupdf.open(CORPUS_DIR / f"{doc_id}.pdf") as pdf:
        return fold(" ".join(page.get_text() for page in pdf))


def test_inventory(corpus):
    assert {c.doc_id for c in corpus.chunks} == set(METADATA)
    assert corpus.excluded == {"politique-geste-commercial": "internal", "procedure-escalade-n2": "internal"}
    assert len(corpus.chunks) == 49


def test_internal_documents_never_enter_the_public_corpus(corpus):
    assert all(c.audience == "public" for c in corpus.chunks)
    assert not {c.doc_id for c in corpus.chunks} & {"procedure-escalade-n2", "politique-geste-commercial"}


def test_every_table_cell(chunks):
    found = {cid: c.tables for cid, c in chunks.items() if c.tables}
    assert set(found) == set(TABLES)
    for cid, (headers, rows) in TABLES.items():
        (table,) = found[cid]
        assert [fold(h) for h in table["headers"]] == [fold(h) for h in headers], cid
        assert [[fold(x) for x in r] for r in table["rows"]] == [[fold(x) for x in r] for r in rows], cid


def test_table_gold_is_in_the_source_files():
    for cid, (headers, rows) in TABLES.items():
        text = source_text(cid.split("#")[0])
        for cell in headers + [x for r in rows for x in r]:
            assert fold(cell) in text, f"{cid}: {cell!r} not in the source"


def test_tables_keep_column_value_pairs_in_the_text(chunks):
    fibre = chunks["grille-tarifaire-2026#offres-fibre-residentielles"].text
    assert "Offre : Fibre Néova 500 Mb/s · Débit descendant / montant : 500 Mb/s / 500 Mb/s · Prix mensuel : 29,99 €" in fibre
    assert "Équipement : Box Néova 5 · Indemnité : 69 €" in chunks[ART14].text


def test_metadata_per_document(corpus):
    for c in corpus.chunks:
        got = (c.statut, c.updated, c.effective_from, c.offer_window, c.supersedes, c.source_format)
        assert got == METADATA[c.doc_id], c.chunk_id


def test_banners(chunks):
    assert "[Source : grille-tarifaire-2026 · EN VIGUEUR depuis le 2026-06-01 · mise à jour 2026-06-01]" in \
        chunks["grille-tarifaire-2026#offres-fibre-residentielles"].text
    assert "ARCHIVE · offre valable pour les souscriptions du 2024-09-01 au 2024-10-31" in \
        chunks["promo-rentree-2024#tarifs-promotionnels"].text


def test_references_both_ways(corpus, chunks):
    assert {c.chunk_id: c.references for c in corpus.chunks if c.references} == REFERENCES
    assert sorted(chunks[ART12].referenced_by) == sorted([ART11, ART14, "grille-tarifaire-2026#offres-fibre-residentielles"])
    assert all(not c.unresolved_references for c in corpus.chunks)


def test_page_continuation_and_no_furniture(chunks):
    assert chunks[ART13].pages == [1, 2]
    for c in chunks.values():
        for furniture in ("corpus/", "cookies", "CO R P U S"):
            assert furniture not in c.text, (c.chunk_id, furniture)


def test_headerless_document_has_unknown_status_and_audience():
    with tempfile.TemporaryDirectory() as root:
        image = Path(root) / "x.png"
        image.write_bytes(b"scan")
        doc = parse_markdown("# Titre\n\nUn texte sans en-tête.\n\n## Section\n\nContenu.", image)
    assert (doc.meta["statut"], doc.audience, doc.meta["maj"]) == ("unknown", "unknown", None)


def test_validate_rejects_a_dangling_reference(corpus):
    broken = Chunk.from_dict({**corpus.chunks[0].to_dict(), "references": ["nowhere#x"]})
    with pytest.raises(ValueError, match="dangling"):
        ingest.validate_corpus(Corpus([broken], {}))


def test_validate_has_no_corpus_specific_expectation(corpus):
    unlinked = [c for c in corpus.chunks if not c.references and not c.referenced_by][:2]
    ingest.validate_corpus(Corpus(unlinked, {}))
