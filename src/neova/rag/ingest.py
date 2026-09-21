"""Corpus ingestion: every public PDF (and the scanned sheet) becomes one chunk per section, with
metadata read from the document itself. Internal documents are never turned into chunks."""

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from neova import llm
from neova.config import CORPUS_DIR, LOGS_DIR
from neova.rag.models import Chunk, Corpus

HEADER_FILL = (0.06, 0.46, 0.43)  # teal background of every table header row in the PDFs

FRENCH_MONTHS = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
}
_MONTHS = "|".join(FRENCH_MONTHS)
_DATE_RE = rf"(\d{{1,2}})(?:er)?\s+({_MONTHS})\s+(\d{{4}})"
VALIDITY_RE = re.compile(rf"valable\s+du\s+(\d{{1,2}})(?:er)?\s+({_MONTHS})(?:\s+(\d{{4}}))?\s+au\s+{_DATE_RE}", re.I)
EFFECTIVE_RE = re.compile(rf"en\s+vigueur\s+depuis\s+le\s+{_DATE_RE}", re.I)
SUPERSEDES_RE = re.compile(r"remplace\s+et\s+annule\s+(?:la\s+fiche\s+)?([A-Z][A-Z0-9-]+)", re.I)
SCAN_REF_RE = re.compile(r"Réf\.?\s*([A-Za-z0-9-]+)")
SCAN_DATE_RE = re.compile(rf"Mise\s+à\s+jour\s*:?\s*{_DATE_RE}", re.I)
MARKER_RE = re.compile(r"^(•|-|\d{1,2}\.|[a-z]\))$")
INLINE_MARKER_RE = re.compile(r"^(•|-|\d{1,2}\.)\s+(.*)$")
ARTICLE_HEADING_RE = re.compile(r"^Article\s+(\d+)")
REFERENCE_RE = re.compile(r"\barticles?\s+(\d+)(?:\s*[àa]\s*(\d+))?", re.I)
# The HTML export of the customer-area help page carries site boilerplate at body size.
CONTENT_DROP = [re.compile(p) for p in (
    r"^Accueil > Aide.*$", r".*cookies.*\[Accepter\].*", r"^(Accueil|Aide|Espace client)$", r"^© Néova Télécom.*$")]


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.replace("’", "'")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", fold(text))).strip("-")[:60]


def iso_date(day, month, year) -> str:
    return f"{int(year):04d}-{FRENCH_MONTHS[month.lower()]:02d}-{int(day):02d}"


# --------------------------------------------------------------------------- layout

@dataclass
class Line:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    spans: list  # [(text, bold)]

    @property
    def text(self) -> str:
        return "".join(t for t, _ in self.spans).strip()

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class Table:
    page: int
    headers: list
    rows: list  # list[list[str]]
    x0: float
    y0: float
    y1: float
    line_ids: set = field(default_factory=set)


def _layout(path: Path):
    lines, drawings, heights = [], {}, {}
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc):
            heights[pno] = page.rect.height
            drawings[pno] = [(d["rect"], d.get("fill")) for d in page.get_drawings()]
            for block in page.get_text("dict")["blocks"]:
                for raw in block.get("lines", []):
                    spans = [(s["text"], bool(s["flags"] & 16) or "bold" in s["font"].lower())
                             for s in raw["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    x0, y0, x1, y1 = raw["bbox"]
                    size = max(s["size"] for s in raw["spans"] if s["text"].strip())
                    lines.append(Line(pno, x0, y0, x1, y1, round(size, 1), spans))
    lines.sort(key=lambda l: (l.page, l.y0, l.x0))
    return lines, drawings, heights


def _is_header_fill(fill) -> bool:
    return fill is not None and all(abs(fill[i] - HEADER_FILL[i]) < 0.05 for i in range(3))


def _find_tables(lines: list[Line], drawings: dict) -> list[Table]:
    """A table starts at a teal header rectangle; rows come from the zebra/separator drawings
    below it; columns from the header cells' x intervals."""
    tables = []
    by_page = {}
    for i, line in enumerate(lines):
        by_page.setdefault(line.page, []).append(i)
    for pno, items in drawings.items():
        # header rows are ~23pt tall; the teal rule under the title is only a few points
        headers = [r for r, fill in items if _is_header_fill(fill) and 12 < r.height < 40]
        for hr in sorted(headers, key=lambda r: r.y0):
            page_lines = [i for i in by_page.get(pno, []) if hr.x0 - 2 <= lines[i].x0 <= hr.x1]
            header_ids = [i for i in page_lines if hr.y0 <= (lines[i].y0 + lines[i].y1) / 2 <= hr.y1]
            if not header_ids:
                continue
            header_cells = sorted(header_ids, key=lambda i: lines[i].x0)
            col_starts = [lines[i].x0 for i in header_cells]
            # Row boundaries: header bottom + every drawing edge between this header and the next one.
            next_header = min([r.y0 for r in headers if r.y0 > hr.y1] + [10**6])
            edges = {round(hr.y1, 1)}
            for r, fill in items:
                if hr.y1 - 1 <= r.y0 and r.y1 <= next_header and r.x0 <= hr.x0 + 5 and r.x1 >= hr.x0 + 100:
                    edges.update((round(r.y0, 1), round(r.y1, 1)))
            edges = sorted(edges)
            bounds = [edges[0]]
            for e in edges[1:]:
                if e - bounds[-1] > 1.5:
                    bounds.append(e)
            gaps = [b - a for a, b in zip(bounds, bounds[1:])]
            row_h = sorted(gaps)[len(gaps) // 2] if gaps else lines[header_cells[0]].height * 2
            # a gap much larger than a row means the drawings belong to something else below
            for k, g in enumerate(gaps):
                if g > row_h * 2.5:
                    bounds = bounds[: k + 1]
                    break
            # A last zebra-white row has no bottom edge: accept lines one row further down only
            # when they sit on a column start at cell font size (prose and headings do not).
            cell_size = max(lines[i].size for i in header_cells) + 1.0
            def in_table(i):
                mid = (lines[i].y0 + lines[i].y1) / 2
                if hr.y1 <= mid <= bounds[-1]:
                    return True
                return (bounds[-1] < mid <= bounds[-1] + row_h * 1.1 and lines[i].size <= cell_size
                        and any(abs(lines[i].x0 - s) <= 2 for s in col_starts))
            body_ids = [i for i in page_lines if i not in header_ids and in_table(i)]
            rows_by_bound: dict[int, list[int]] = {}
            for i in body_ids:
                mid = (lines[i].y0 + lines[i].y1) / 2
                k = sum(1 for b in bounds if b <= mid)
                rows_by_bound.setdefault(k, []).append(i)
            rows = []
            for k in sorted(rows_by_bound):
                cells = [""] * len(col_starts)
                for i in sorted(rows_by_bound[k], key=lambda i: (lines[i].y0, lines[i].x0)):
                    col = max(c for c, start in enumerate(col_starts) if lines[i].x0 >= start - 2) \
                        if lines[i].x0 >= col_starts[0] - 2 else 0
                    cells[col] = (cells[col] + " " + lines[i].text).strip()
                if any(cells):
                    rows.append(cells)
            tables.append(Table(pno, [lines[i].text for i in header_cells], rows, hr.x0, hr.y0,
                                max([lines[i].y1 for i in body_ids] + [hr.y1]), set(header_ids) | set(body_ids)))
    return tables


def _render_spans(spans) -> str:
    out, bold_run = [], []
    def flush():
        if bold_run:
            text = "".join(bold_run)
            lead, core, trail = text[: len(text) - len(text.lstrip())], text.strip(), text[len(text.rstrip()):]
            out.append(f"{lead}**{core}**{trail}" if core else text)
            bold_run.clear()
    for text, bold in spans:
        if bold:
            bold_run.append(text)
        else:
            flush()
            out.append(text)
    flush()
    return "".join(out).strip()


# --------------------------------------------------------------------------- documents

@dataclass
class Element:
    kind: str  # heading | paragraph | item | table
    text: str = ""
    level: int = 0
    marker: str = ""
    table: Table | None = None
    pages: set = field(default_factory=set)


@dataclass
class Document:
    doc_id: str
    source_format: str
    audience: str
    title: str
    meta: dict
    preamble: str
    elements: list
    source_hash: str


def _parse_meta(header: str, body: str, path: Path, title: str = "") -> dict:
    def grab(pattern, text, default=None):
        m = re.search(pattern, text)
        return m.group(1) if m else default
    everything = header + "\n" + title + "\n" + body  # applicability dates live in any of the three
    doc_id = grab(r"\bid\s+([a-z0-9-]+)", header)
    if doc_id:
        meta = {"id": doc_id, "statut": grab(r"\bstatut\s+(current|deprecated)\b", header),
                "audience": grab(r"\bpublic\s+(public|internal)\b", header),
                "maj": grab(r"\bmaj\s+(\d{4}-\d{2}-\d{2})", header)}
    else:
        ref, date = SCAN_REF_RE.search(header), SCAN_DATE_RE.search(header)
        meta = {"id": ref.group(1).lower() if ref else path.stem, "statut": None,
                "audience": "public" if re.search(r"Diffusion\s*:\s*clients", everything, re.I) else None,
                "maj": iso_date(date.group(1), date.group(2), date.group(3)) if date else None}
    if meta["statut"] is None:
        meta["statut"] = "current" if re.search(r"version en vigueur|en vigueur depuis", everything, re.I) else "unknown"
    meta["audience"] = meta["audience"] or "unknown"
    m = VALIDITY_RE.search(everything)
    meta["subscription_window"] = {"from": iso_date(m.group(1), m.group(2), m.group(3) or m.group(6)),
                                   "to": iso_date(m.group(4), m.group(5), m.group(6))} if m else None
    m = EFFECTIVE_RE.search(everything)
    meta["effective_from"] = iso_date(m.group(1), m.group(2), m.group(3)) if m else None
    m = SUPERSEDES_RE.search(everything)
    meta["supersedes"] = m.group(1).upper() if m else None
    return meta


def parse_pdf(path: Path) -> Document:
    lines, drawings, heights = _layout(path)
    tables = _find_tables(lines, drawings)
    in_table = {i: t for t in tables for i in t.line_ids}
    free = [i for i in range(len(lines)) if i not in in_table]

    chars: dict[float, int] = {}
    for i in free:
        chars[lines[i].size] = chars.get(lines[i].size, 0) + len(lines[i].text)
    body = max(chars, key=chars.get)
    heading_sizes = sorted({lines[i].size for i in free if lines[i].size > body + 0.3}, reverse=True)
    level_of = {s: k + 1 for k, s in enumerate(heading_sizes)}
    title_size = heading_sizes[0] if heading_sizes else None

    title_ids = []
    for i in free:
        if lines[i].page == 0 and lines[i].size == title_size:
            if not title_ids or i == title_ids[-1] + 1:
                title_ids.append(i)
        elif title_ids:
            break
    title = " ".join(lines[i].text for i in title_ids)
    title_top = lines[title_ids[0]].y0 if title_ids else 0
    title_bottom = lines[title_ids[-1]].y1 if title_ids else 0
    first_heading_y = next((lines[i].y0 for i in free if lines[i].page == 0 and i not in title_ids
                            and lines[i].size in level_of), heights[0])

    header_block, elements, order = [], [], []
    for i in free:
        line = lines[i]
        h = heights[line.page]
        if i in title_ids:
            continue
        if line.page == 0 and line.y0 < title_top:                        # R1 banner
            continue
        if line.y1 > 0.94 * h or (line.page > 0 and line.y0 < 0.07 * h):  # R2 margins
            continue
        if line.page == 0 and title_bottom <= line.y0 < first_heading_y and line.size < body:  # R3
            header_block.append(line.text)
            continue
        if line.size <= body and any(p.match(line.text) for p in CONTENT_DROP):
            continue
        order.append(i)

    # walk in reading order, emitting tables where their first line appears
    emitted, bands, prev = set(), [], None
    for i in sorted(set(order) | set(in_table), key=lambda i: (lines[i].page, lines[i].y0, lines[i].x0)):
        line = lines[i]
        if i in in_table:
            t = in_table[i]
            if id(t) not in emitted:
                emitted.add(id(t))
                elements.append(Element("table", table=t, pages={t.page}))
            prev = None
            continue
        if line.size in level_of:
            elements.append(Element("heading", line.text, level_of[line.size], pages={line.page}))
            prev = None
            continue
        # group cells of one visual band (marker + text)
        if prev is not None and line.page == prev.page and line.y0 < prev.y1 - 1:
            bands[-1].append(line)
        else:
            bands.append([line])
            elements.append(Element("band", str(len(bands) - 1), pages={line.page}))  # resolved below
        prev = line

    resolved, last_item, last_para = [], None, None
    for e in elements:
        if e.kind != "band":
            resolved.append(e)
            last_item = last_para = None
            continue
        band = sorted(bands[int(e.text)], key=lambda l: l.x0)
        first = band[0]
        marker_match = MARKER_RE.match(first.text)
        inline = INLINE_MARKER_RE.match(_render_spans(first.spans)) if len(band) == 1 else None
        if marker_match and len(band) == 1:
            continue  # orphan bullet glyph (its text was dropped or never existed)
        if marker_match and len(band) > 1:
            item = Element("item", " ".join(_render_spans(l.spans) for l in band[1:]), marker=first.text, pages=e.pages)
            item.level = int(band[1].x0)
            resolved.append(item)
            last_item, last_para = item, None
        elif inline:
            item = Element("item", inline.group(2), marker=inline.group(1), pages=e.pages)
            item.level = int(first.x0) + 12
            resolved.append(item)
            last_item, last_para = item, None
        elif last_item is not None and abs(first.x0 - last_item.level) <= 3 and len(band) == 1:
            last_item.text += " " + _render_spans(first.spans)
            last_item.pages |= e.pages
        else:
            text = " ".join(_render_spans(l.spans) for l in band)
            if last_para is not None and first.page in last_para.pages and first.y0 - last_para.level < first.height * 1.6:
                last_para.text += " " + text
            else:
                last_para = Element("paragraph", text, pages=set(e.pages))
                resolved.append(last_para)
            last_para.level = int(first.y1)
            last_item = None
    for e in resolved:
        e.level = e.level if e.kind == "heading" else 0

    preamble = " ".join(e.text for e in _take_until_heading(resolved) if e.kind == "paragraph")
    meta = _parse_meta("\n".join(header_block), " ".join(e.text for e in resolved if e.kind == "paragraph"), path, title)
    return Document(meta["id"], "pdf", meta["audience"], title, meta, preamble,
                    resolved, hashlib.sha256(path.read_bytes()).hexdigest())


def _take_until_heading(elements):
    for e in elements:
        if e.kind == "heading":
            return
        yield e


def parse_markdown(text: str, path: Path) -> Document:
    """The scanned sheet arrives as markdown from the vision model."""
    elements, para, table_lines = [], [], []
    levels = sorted({len(m.group(1)) for m in re.finditer(r"^(#{1,6})\s", text, re.M)})
    def flush_para():
        if para:
            elements.append(Element("paragraph", " ".join(para), pages={0}))
            para.clear()
    def flush_table():
        if table_lines:
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in table_lines
                    if not re.match(r"^\|?\s*:?-{2,}", l.strip())]
            elements.append(Element("table", table=Table(0, rows[0], rows[1:], 0, 0, 0), pages={0}))
            table_lines.clear()
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            flush_para(); table_lines.append(line); continue
        flush_table()
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para()
            elements.append(Element("heading", m.group(2).strip(), levels.index(len(m.group(1))) + 1, pages={0}))
        elif INLINE_MARKER_RE.match(line):
            flush_para(); im = INLINE_MARKER_RE.match(line)
            elements.append(Element("item", im.group(2), marker=im.group(1), pages={0}))
        elif line:
            para.append(line)
        else:
            flush_para()
    flush_para(); flush_table()
    title_el = next((e for e in elements if e.kind == "heading" and e.level == 1), None)
    title = title_el.text if title_el else path.stem
    elements = [e for e in elements if e is not title_el]
    preamble = " ".join(e.text for e in _take_until_heading(elements) if e.kind == "paragraph")
    meta = _parse_meta(text, text, path)
    return Document(meta["id"], "image", meta["audience"], title, meta, preamble,
                    elements, hashlib.sha256(path.read_bytes()).hexdigest())


# --------------------------------------------------------------------------- chunks

def _record(headers: list, cells: list) -> str:
    return " · ".join(f"{h} : {c}" for h, c in zip(headers, cells))


def _render_table(table: Table) -> str:
    return "\n".join(f"- {_record(table.headers, r)}" for r in table.rows)


def banner(doc_id: str, meta: dict) -> str:
    if meta["statut"] == "deprecated":
        w = meta.get("subscription_window")
        status = f"ARCHIVE · offre valable pour les souscriptions du {w['from']} au {w['to']}" if w else "ARCHIVE"
    elif meta["statut"] == "current":
        status = "EN VIGUEUR" + (f" depuis le {meta['effective_from']}" if meta.get("effective_from") else "")
    else:
        status = "STATUT NON INDIQUÉ"
    parts = [f"Source : {doc_id}", status]
    if meta.get("supersedes"):
        parts.append(f"remplace {meta['supersedes']}")
    if meta.get("maj"):
        parts.append(f"mise à jour {meta['maj']}")
    return "[" + " · ".join(parts) + "]"


def _chunk(doc: Document, path: list[str], elements: list) -> Chunk | None:
    if not elements:
        return None
    body_parts, tables, pages = [], [], set()
    for e in elements:
        pages |= e.pages
        if e.kind == "paragraph":
            body_parts.append(e.text)
        elif e.kind == "item":
            body_parts.append(f"{e.marker if e.marker.endswith('.') else '-'} {e.text}")
        elif e.kind == "table":
            body_parts.append(_render_table(e.table))
            tables.append({"headers": list(e.table.headers), "rows": [list(r) for r in e.table.rows]})
    body = "\n\n".join(body_parts)
    heading = " > ".join(path)
    text = f"{heading}\n{banner(doc.doc_id, doc.meta)}"
    if doc.preamble and path[-1] != "Introduction":
        text += f"\nNote : {doc.preamble}"
    text += "\n\n" + body
    article = ARTICLE_HEADING_RE.match(path[-1])
    cited = set()
    for m in REFERENCE_RE.finditer(body):
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        cited.update(range(a, b + 1))
    if article:
        cited.discard(int(article.group(1)))
    return Chunk(
        chunk_id=f"{doc.doc_id}#{slug(path[-1])}", doc_id=doc.doc_id, title=doc.title, section_path=path,
        pages=sorted(p + 1 for p in pages), audience=doc.audience, statut=doc.meta["statut"],
        updated=doc.meta["maj"], effective_from=doc.meta["effective_from"],
        offer_window=doc.meta["subscription_window"], supersedes=doc.meta["supersedes"],
        source_hash=doc.source_hash, source_format=doc.source_format, has_table=bool(tables), tables=tables,
        text=text, search_text=f"{heading}\n{body.replace('**', '')}",
        article_no=int(article.group(1)) if article else None,
        unresolved_references=[f"article-{n}" for n in sorted(cited)],
    )


def chunks_from_document(doc: Document) -> list[Chunk]:
    chunks, stack, elements = [], [], []

    def close():
        path = [doc.title] + ([h.text for h in stack] if stack else ["Introduction"])
        chunk = _chunk(doc, path, elements)
        if chunk:
            chunks.append(chunk)
        elements.clear()

    for e in doc.elements:
        if e.kind == "heading":
            close()
            while stack and stack[-1].level >= e.level:
                stack.pop()
            stack.append(e)
        else:
            elements.append(e)
    close()
    return chunks


def link_references(chunks: list[Chunk]) -> None:
    """Resolve "article N" to the chunk of that article when exactly one exists; keep the rest as
    unresolved. referenced_by is the reverse link."""
    by_article: dict[int, list[str]] = {}
    for c in chunks:
        if c.article_no is not None:
            by_article.setdefault(c.article_no, []).append(c.chunk_id)
    by_id = {c.chunk_id: c for c in chunks}
    for c in chunks:
        unresolved = []
        for ref in c.unresolved_references:
            targets = by_article.get(int(ref.split("-")[1]), [])
            if len(targets) == 1:
                c.references.append(targets[0])
                by_id[targets[0]].referenced_by.append(c.chunk_id)
            else:
                unresolved.append(ref)
        c.unresolved_references = unresolved


# --------------------------------------------------------------------------- build and validate

def parse_document(path: Path, offline: bool = False) -> Document:
    if path.suffix.lower() == ".png":
        return parse_markdown(llm.transcribe_image(path, offline=offline), path)
    return parse_pdf(path)


def build_corpus(corpus_dir: Path = CORPUS_DIR, offline: bool = False) -> Corpus:
    chunks, excluded = [], {}
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in (".pdf", ".png"):
            continue
        doc = parse_document(path, offline)
        if doc.audience != "public":
            excluded[doc.doc_id] = doc.audience
            continue
        chunks.extend(chunks_from_document(doc))
    link_references(chunks)
    corpus = Corpus(chunks, excluded)
    validate_corpus(corpus)
    return corpus


def validate_corpus(corpus: Corpus) -> None:
    """Invariants that hold for any corpus. Expectations about this corpus live in tests/."""
    problems, ids = [], [c.chunk_id for c in corpus.chunks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate chunk ids")
    known = set(ids)
    for c in corpus.chunks:
        if c.audience != "public":
            problems.append(f"{c.chunk_id}: audience {c.audience} in the public corpus")
        if not c.text.split("\n\n", 1)[-1].strip():
            problems.append(f"{c.chunk_id}: empty body")
        for t in c.tables:
            if not t["headers"] or not t["rows"] or any(len(r) != len(t["headers"]) for r in t["rows"]):
                problems.append(f"{c.chunk_id}: malformed table {t['headers']}")
        for ref in c.references + c.referenced_by:
            if ref not in known:
                problems.append(f"{c.chunk_id}: dangling reference {ref}")
    if problems:
        raise ValueError("corpus validation failed:\n- " + "\n- ".join(problems))


# --------------------------------------------------------------------------- inspection

def write_artefacts(corpus: Corpus) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "ingest_chunks.json").write_text(json.dumps(
        {"chunks": [c.to_dict() for c in corpus.chunks], "excluded": corpus.excluded},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="print every chunk in full")
    ap.add_argument("--offline", action="store_true", help="never call the vision model")
    args = ap.parse_args()
    corpus = build_corpus(offline=args.offline)
    write_artefacts(corpus)
    current = None
    for c in corpus.chunks:
        if c.doc_id != current:
            current = c.doc_id
            print(f"\n=== {c.doc_id}  {c.statut}  maj {c.updated}  ===")
        print(f"  [{c.chunk_id}] p{c.pages}{'  table' if c.has_table else ''}"
              f"{'  refs ' + ','.join(c.references) if c.references else ''}")
        if args.inspect:
            print("    " + c.text.replace("\n", "\n    ") + "\n")
    print(f"\n{len({c.doc_id for c in corpus.chunks})} public documents, {len(corpus.chunks)} chunks; "
          f"excluded: {corpus.excluded}")
    print(f"artefact: {LOGS_DIR / 'ingest_chunks.json'}")
