import hashlib
import re
import sys
from pathlib import Path

import pymupdf

from neova.config import CORPUS_DIR

from .models import Document, Element, Line, Table, Chunk

HEADER_FILL = (0.06, 0.46, 0.43)


def layout(path: Path):
    with pymupdf.open(path) as doc:
        lines: list[Line] = []
        drawings_per_page: dict[
            int, list[tuple[tuple[float, float, float, float], tuple[float, float, float] | None]]
        ] = {}
        page_heights: dict[int, float] = {}

        for page_num, page in enumerate(doc, start=1):
            page_heights[page_num] = page.rect.height

            drawings: list[
                tuple[tuple[float, float, float, float], tuple[float, float, float] | None]
            ] = []
            for d in page.get_drawings():
                rect = d["rect"]
                fill_color = d.get("fill")
                if fill_color is not None and isinstance(fill_color, float):
                    fill_color = (fill_color, fill_color, fill_color)
                drawings.append(((rect.x0, rect.y0, rect.x1, rect.y1), fill_color))
            drawings_per_page[page_num] = drawings

            blocks = page.get_text("dict")
            page_lines: list[Line] = []

            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue
                for line_info in block["lines"]:
                    spans = line_info.get("spans", [])
                    if not spans:
                        continue

                    visible_spans = []
                    max_size = 0.0
                    line_x0, line_y0, line_x1, line_y1 = line_info["bbox"]

                    for span in spans:
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        font_flags = span.get("flags", 0)
                        font_name = span.get("font", "").lower()
                        is_bold = bool(font_flags & 16) or "bold" in font_name
                        visible_spans.append((span["text"], is_bold))

                        size = span.get("size", 0.0)
                        max_size = max(max_size, size)

                    if not visible_spans:
                        continue

                    page_lines.append(
                        Line(
                            page=page_num,
                            x0=line_x0,
                            y0=line_y0,
                            x1=line_x1,
                            y1=line_y1,
                            size=round(max_size, 1),
                            spans=visible_spans,
                        )
                    )

            page_lines.sort(key=lambda l: (l.y0, l.x0))
            lines.extend(page_lines)

    return lines, drawings_per_page, page_heights


def find_tables(
    lines: list[Line],
    drawings: dict[
        int, list[tuple[tuple[float, float, float, float], tuple[float, float, float] | None]]
    ],
) -> list[Table]:
    tables: list[Table] = []

    line_by_page: dict[int, list[tuple[int, Line]]] = {}
    for idx, line in enumerate(lines):
        line_by_page.setdefault(line.page, []).append((idx, line))

    for page, page_drawings in drawings.items():
        page_lines = line_by_page.get(page, [])
        if not page_lines:
            continue

        headers: list[tuple[tuple[float, float, float, float], float]] = []
        for rect, fill in page_drawings:
            if fill is None:
                continue
            if all(abs(fill[i] - HEADER_FILL[i]) <= 0.05 for i in range(3)):
                height = rect[3] - rect[1]
                if 12 <= height <= 40:
                    headers.append((rect, height))

        headers.sort(key=lambda h: h[0][1])
        for i, (header_rect, header_height) in enumerate(headers):
            hx0, hy0, hx1, hy1 = header_rect
            header_line_indices = []
            for line_idx, line in page_lines:
                line_center_y = (line.y0 + line.y1) / 2
                if hy0 <= line_center_y <= hy1 and hx0 <= line.x0 <= hx1:
                    header_line_indices.append((line_idx, line))

            header_line_indices.sort(key=lambda x: x[1].x0)
            column_starts = [line.x0 for _, line in header_line_indices]
            column_labels = [line.text for _, line in header_line_indices]

            header_indices = {line_idx for line_idx, _ in header_line_indices}

            row_boundaries = [hy1]
            next_header_bottom = headers[i + 1][0][1] if i + 1 < len(headers) else float("inf")
            for rect, _ in page_drawings:
                rx0, ry0, rx1, ry1 = rect
                if hy1 <= ry0 < next_header_bottom and abs(rx0 - hx0) <= 5 and rx1 - rx0 >= 100:
                    row_boundaries.append(ry0)
                    row_boundaries.append(ry1)

            row_boundaries.sort()
            merged_boundaries = []
            for b in row_boundaries:
                if not merged_boundaries or b - merged_boundaries[-1] >= 1.5:
                    merged_boundaries.append(b)
            row_boundaries = merged_boundaries

            if len(row_boundaries) < 2:
                continue

            gaps = [
                row_boundaries[j] - row_boundaries[j - 1] for j in range(1, len(row_boundaries))
            ]
            if not gaps:
                continue
            median_gap = sorted(gaps)[len(gaps) // 2]
            row_height = median_gap

            last_boundary_idx = len(row_boundaries) - 1
            for j in range(1, len(row_boundaries)):
                if row_boundaries[j] - row_boundaries[j - 1] > 2.5 * row_height:
                    last_boundary_idx = j - 1
                    break
            last_boundary = row_boundaries[last_boundary_idx]

            header_line_size = header_line_indices[0][1].size if header_line_indices else 0

            body_line_indices: list[tuple[int, Line, float]] = []
            for line_idx, line in page_lines:
                if line_idx in header_indices:
                    continue
                line_center_y = (line.y0 + line.y1) / 2
                if hy1 <= line_center_y <= last_boundary:
                    body_line_indices.append((line_idx, line, line_center_y))
                elif (
                    line_center_y - last_boundary <= row_height
                    and header_line_indices
                    and line.size <= header_line_size + 1
                ):
                    col_match = False
                    for col_start in column_starts:
                        if abs(line.x0 - col_start) <= 2:
                            col_match = True
                            break
                    if col_match:
                        body_line_indices.append((line_idx, line, line_center_y))

            rows: list[list[str]] = []
            row_indices: list[set[int]] = []
            for j in range(1, len(row_boundaries[: last_boundary_idx + 1])):
                row_start = row_boundaries[j - 1]
                row_end = row_boundaries[j]
                row_lines = []
                indices = set()
                for line_idx, line, center_y in body_line_indices:
                    if row_start <= center_y < row_end:
                        row_lines.append((line_idx, line))
                        indices.add(line_idx)
                if not row_lines:
                    continue

                cells: dict[int, list[str]] = {}
                for line_idx, line in row_lines:
                    col_idx = 0
                    for k, col_start in enumerate(column_starts):
                        if line.x0 >= col_start - 2:
                            col_idx = k + 1
                    if line.x0 < column_starts[0] - 2:
                        col_idx = 0
                    cells.setdefault(col_idx, []).append(line.text)

                row_cells = []
                for col in range(len(column_labels)):
                    cell_text = " ".join(cells.get(col + 1, []))
                    row_cells.append(cell_text)
                rows.append(row_cells)
                row_indices.append(indices)

            if not rows:
                continue

            all_indices = header_indices.union(*row_indices)
            tables.append(Table(page=page, headers=column_labels, rows=rows, line_ids=all_indices))

    return tables


def heading_levels(lines: list[Line], tables_line_ids: set[int]) -> tuple[float, dict[float, int]]:
    char_count: dict[float, int] = {}
    for idx, line in enumerate(lines):
        if idx in tables_line_ids:
            continue
        char_count[line.size] = char_count.get(line.size, 0) + len(line.text)

    if not char_count:
        return 0.0, {}

    body_size = max(char_count.items(), key=lambda x: x[1])[0]

    heading_sizes = sorted([size for size in char_count if size > body_size + 0.3], reverse=True)
    heading_levels_dict = {size: level for level, size in enumerate(heading_sizes, 1)}

    return body_size, heading_levels_dict


def split_furniture(
    lines: list[Line], table_line_idx: set[int], body_size: float, page_height: dict[int, float]
) -> tuple[str, list[int]]:
    header_bands: dict[tuple[int, float], list[str]] = {}
    content_indices: list[int] = []
    running_headers_seen: set[int] = set()

    page1_body_line_found = False
    for idx, line in enumerate(lines):
        if idx in table_line_idx:
            continue

        page_h = page_height.get(line.page, 0)
        y_center = (line.y0 + line.y1) / 2

        if y_center > page_h - 60 and line.size < body_size:
            continue

        if line.page == 1:
            if line.size < body_size and not page1_body_line_found:
                band_key = (line.page, round(y_center / 2) * 2)
                header_bands.setdefault(band_key, []).append(line.text)
            else:
                if line.size == body_size and not page1_body_line_found:
                    page1_body_line_found = True
                content_indices.append(idx)
        else:
            if line.size < body_size:
                if y_center < 40:
                    running_headers_seen.add(idx)
                else:
                    content_indices.append(idx)
            else:
                content_indices.append(idx)

    boilerplate_texts = ["Accueil", "cookies", "©"]
    for idx, line in enumerate(lines):
        if idx in running_headers_seen or idx in table_line_idx:
            continue
        text = line.text
        if any(text.startswith(b) or b in text for b in boilerplate_texts):
            running_headers_seen.add(idx)

    header_lines = []
    for band_key in sorted(header_bands.keys()):
        band_texts = header_bands[band_key]
        header_lines.append(" · ".join(band_texts))

    content_indices = [idx for idx in content_indices if idx not in running_headers_seen]

    return "\n".join(header_lines), content_indices


FRENCH_MONTHS = {
    "janvier": 1,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12,
}


def french_date_to_iso(day_str: str, month_str: str, year_str: str) -> str:
    day = int(day_str.rstrip("er"))
    month = FRENCH_MONTHS[month_str.lower()]
    year = int(year_str)
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_meta(header: str, title: str, body: str, path: Path) -> dict:
    full_text = header + "\n" + title + "\n" + body
    result = {
        "id": None,
        "statut": None,
        "audience": None,
        "updated": None,
        "effective_from": None,
        "offer_window": None,
        "supersedes": None,
    }

    id_match = re.search(r"id\s+(\S+)", header, re.IGNORECASE)
    if id_match:
        result["id"] = id_match.group(1)

    statut_match = re.search(r"statut\s+(\S+)", header, re.IGNORECASE)
    if statut_match:
        result["statut"] = statut_match.group(1).lower()

    audience_match = re.search(r"public\s+(\S+)", header, re.IGNORECASE)
    if audience_match:
        result["audience"] = audience_match.group(1).lower()

    updated_match = re.search(r"maj\s+(\d{4}-\d{2}-\d{2})", header, re.IGNORECASE)
    if updated_match:
        result["updated"] = updated_match.group(1)

    ref_match = re.search(r"réf\.\s+([a-zA-Z0-9-]+)", full_text, re.IGNORECASE)
    if ref_match and not result["id"]:
        result["id"] = ref_match.group(1).lower()

    if not result["id"]:
        result["id"] = path.stem

    mise_match = re.search(
        r"mise à jour\s*:\s*(\d+)\s+([a-zéû]+)\s+(\d{4})", full_text, re.IGNORECASE
    )
    if mise_match and not result["updated"]:
        day, month, year = mise_match.groups()
        result["updated"] = french_date_to_iso(day, month, year)

    if re.search(r"diffusion\s*:\s*clients", full_text, re.IGNORECASE):
        result["audience"] = "public"
    elif not result["audience"]:
        result["audience"] = "unknown"

    if re.search(r"version en vigueur|en vigueur depuis", full_text, re.IGNORECASE):
        result["statut"] = "current"
    elif not result["statut"]:
        result["statut"] = "unknown"

    offer_match = re.search(
        r"valable du (\d+er|\d+)\s+([a-zéû]+)(?:\s+(\d{4}))?\s+au\s+(\d+er|\d+)\s+([a-zéû]+)\s+(\d{4})",
        full_text,
        re.IGNORECASE,
    )
    if offer_match:
        day1, month1, year1, day2, month2, year2 = offer_match.groups()
        if not year1:
            year1 = year2
        from_date = french_date_to_iso(day1, month1, year1)
        to_date = french_date_to_iso(day2, month2, year2)
        result["offer_window"] = {"from": from_date, "to": to_date}

    effective_match = re.search(
        r"en vigueur depuis le (\d+er|\d+)\s+([a-zéû]+)\s+(\d{4})", full_text, re.IGNORECASE
    )
    if effective_match:
        day, month, year = effective_match.groups()
        result["effective_from"] = french_date_to_iso(day, month, year)

    supersedes_match = re.search(
        r"remplace et annule (?:la fiche )?([A-Z0-9-]+)", full_text, re.IGNORECASE
    )
    if supersedes_match:
        result["supersedes"] = supersedes_match.group(1).upper()

    return result


def build_elements(
    lines: list[Line], content_ids: list[int], tables: list[Table], levels: dict[float, int]
) -> list[Element]:
    table_line_ids = set()
    for table in tables:
        table_line_ids.update(table.line_ids)

    table_by_first_line: dict[int, Table] = {}
    for table in tables:
        first_line = min(table.line_ids)
        table_by_first_line[first_line] = table

    all_ids = sorted(set(content_ids) | table_line_ids)
    elements: list[Element] = []
    current_para_lines: list[int] = []
    current_para_pages: set[int] = set()
    current_para_x0: float | None = None
    prev_line_bottom: float | None = None
    item_text_lines: list[int] = []
    item_x0: float | None = None
    item_bottom: float | None = None
    item_marker: str | None = None

    def format_line_text(line_idx: int) -> str:
        line = lines[line_idx]
        parts = []
        current_run = []
        current_bold = None
        for text, bold in line.spans:
            if current_bold is None:
                current_bold = bold
                current_run.append(text)
            elif bold == current_bold:
                current_run.append(text)
            else:
                joined = "".join(current_run)
                if current_bold:
                    parts.append(f"**{joined}**")
                else:
                    parts.append(joined)
                current_run = [text]
                current_bold = bold
        if current_run:
            joined = "".join(current_run)
            if current_bold:
                parts.append(f"**{joined}**")
            else:
                parts.append(joined)
        return "".join(parts)

    def flush_item():
        nonlocal item_text_lines, item_x0, item_bottom, item_marker
        if not item_text_lines:
            return
        text_parts = [format_line_text(idx) for idx in item_text_lines]
        text = " ".join(text_parts)
        pages = {lines[idx].page for idx in item_text_lines}
        elements.append(Element("item", text, marker=item_marker, pages=pages))
        item_text_lines.clear()
        item_x0 = None
        item_bottom = None
        item_marker = None

    def flush_para():
        nonlocal current_para_lines, current_para_x0, prev_line_bottom
        if not current_para_lines:
            return
        text_parts = [format_line_text(idx) for idx in current_para_lines]
        text = " ".join(text_parts)
        pages = {lines[idx].page for idx in current_para_lines}
        elements.append(Element("paragraph", text, pages=pages))
        current_para_lines.clear()
        current_para_pages.clear()
        current_para_x0 = None
        prev_line_bottom = None

    i = 0
    while i < len(all_ids):
        idx = all_ids[i]

        if idx in table_line_ids and idx in table_by_first_line:
            flush_item()
            flush_para()
            table = table_by_first_line[idx]
            while i < len(all_ids) and all_ids[i] in table_line_ids:
                i += 1
            i -= 1
            elements.append(Element("table", "", table=table, pages={table.page}))
            prev_line_bottom = None
            i += 1
            continue

        if idx in table_line_ids:
            i += 1
            continue

        line = lines[idx]

        if idx not in content_ids:
            i += 1
            continue

        if line.size in levels:
            flush_item()
            flush_para()
            text = line.text
            elements.append(Element("heading", text, level=levels[line.size], pages={line.page}))
            prev_line_bottom = line.y1
            i += 1
            continue

        marker_text = line.text.strip()
        if 66 <= line.x0 <= 72 and re.fullmatch(r"•|-|\d{1,2}\.|[a-z]\)", marker_text):
            flush_item()
            flush_para()
            item_marker = marker_text
            i += 1
            if i < len(all_ids):
                next_idx = all_ids[i]
                if next_idx not in table_line_ids and next_idx in content_ids:
                    next_line = lines[next_idx]
                    if abs(next_line.y0 - line.y0) < 2:
                        item_text_lines.append(next_idx)
                        item_x0 = next_line.x0
                        item_bottom = next_line.y1
                        i += 1
            continue

        if item_text_lines:
            current_line = lines[idx]
            if (
                abs(current_line.x0 - item_x0) <= 2
                and current_line.page == lines[item_text_lines[-1]].page
                and current_line.y0 - item_bottom < 7
            ):
                item_text_lines.append(idx)
                item_bottom = current_line.y1
                i += 1
                continue

        if current_para_x0 is not None and current_para_pages:
            same_page = line.page == max(current_para_pages)
            gap = line.y0 - prev_line_bottom if prev_line_bottom else 999
            if same_page and abs(line.x0 - current_para_x0) <= 2 and gap < 7:
                current_para_lines.append(idx)
                current_para_pages.add(line.page)
                prev_line_bottom = line.y1
                i += 1
                continue

        flush_item()
        flush_para()
        current_para_lines.append(idx)
        current_para_pages.add(line.page)
        current_para_x0 = line.x0
        prev_line_bottom = line.y1
        i += 1

    flush_item()
    flush_para()
    return elements


def parse_pdf(path: Path) -> Document:
    lines, drawings, page_heights = layout(path)
    tables = find_tables(lines, drawings)
    tables_line_ids = set()
    for table in tables:
        tables_line_ids.update(table.line_ids)

    body_size, levels = heading_levels(lines, tables_line_ids)
    header, content_ids = split_furniture(lines, tables_line_ids, body_size, page_heights)

    elements = build_elements(lines, content_ids, tables, levels)

    title_parts = []
    while elements and elements[0].kind == "heading" and elements[0].level == 1:
        title_parts.append(elements[0].text)
        elements.pop(0)
    title = " ".join(title_parts) if title_parts else ""

    preamble_parts = []
    while elements and elements[0].kind in {"paragraph", "item"}:
        preamble_parts.append(elements[0].text)
        elements.pop(0)
    preamble = " ".join(preamble_parts)

    body_text = " ".join(elem.text for elem in elements if elem.kind in {"paragraph", "item"})

    meta = parse_meta(header, title, preamble + "\n" + body_text, path)

    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    return Document(
        doc_id=meta["id"],
        source_format="pdf",
        title=title,
        audience=meta["audience"],
        statut=meta["statut"],
        updated=meta["updated"],
        effective_from=meta["effective_from"],
        offer_window=meta["offer_window"],
        supersedes=meta["supersedes"],
        preamble=preamble,
        elements=elements,
        source_hash=file_hash,
    )


def parse_markdown(path: Path, offline: bool = False) -> Document:
    from neova.llm import image_to_markdown

    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    markdown = image_to_markdown(path, offline=offline)
    lines = markdown.splitlines()
    if lines and lines[-1].startswith("Néova Télécom") and "document non contractuel" in lines[-1]:
        lines = lines[:-1]
    i = 0
    header_lines = []
    while i < len(lines) and not lines[i].startswith("#"):
        header_lines.append(lines[i])
        i += 1

    header = "\n".join(header_lines)

    heading_levels_count = {}
    j = len(header_lines)
    while j < len(lines):
        line = lines[j]
        if line.startswith("#"):
            level_str = line.split()[0]
            level = len(level_str)
            heading_levels_count[level] = heading_levels_count.get(level, 0) + 1
        j += 1

    shallowest = min(heading_levels_count.keys()) if heading_levels_count else 0
    i = len(header_lines)
    title_parts = []
    elements: list[Element] = []
    current_para: list[str] = []
    current_list_items: list[str] = []
    current_list_marker: str | None = None
    in_table = False
    table_lines: list[str] = []
    table_page = 1

    def flush_para():
        nonlocal current_para
        if current_para:
            text = " ".join(current_para)
            elements.append(Element("paragraph", text, pages={table_page}))
            current_para.clear()

    def flush_list():
        nonlocal current_list_items, current_list_marker
        if current_list_items:
            for item in current_list_items:
                elements.append(Element("item", item, marker=current_list_marker, pages={table_page}))
            current_list_items.clear()
            current_list_marker = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_para()
            flush_list()
            i += 1
            continue

        if stripped.startswith("|"):
            flush_para()
            flush_list()
            if not in_table:
                in_table = True
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            flush_para()
            flush_list()
            if table_lines:
                headers = []
                rows = []
                for j, tbl_line in enumerate(table_lines):
                    if j == 0:
                        cells = [cell.strip() for cell in tbl_line.strip("|").split("|")]
                        headers = cells
                    elif j == 1 and all(c in "-:" for c in tbl_line.replace("|", "").replace(" ", "")):
                        continue
                    else:
                        cells = [cell.strip() for cell in tbl_line.strip("|").split("|")]
                        rows.append(cells)
                if headers:
                    table = Table(page=table_page, headers=headers, rows=rows)
                    elements.append(Element("table", "", table=table, pages={table_page}))
            table_lines.clear()
            in_table = False

        if stripped.startswith(("#", "##", "###", "####", "#####", "######")):
            flush_para()
            flush_list()
            level_str, rest = stripped.split(maxsplit=1)
            heading_level = len(level_str)
            if shallowest > 0 and heading_level == shallowest:
                title_parts.append(rest)
            else:
                adjusted_level = heading_level - shallowest if shallowest > 0 else heading_level
                elements.append(Element("heading", rest, level=adjusted_level, pages={table_page}))
            i += 1
            continue

        list_prefixes = ["- ", "* ", "• ", "1. "]
        is_list_item = any(stripped.startswith(prefix) for prefix in list_prefixes)
        if is_list_item:
            flush_para()
            for prefix in list_prefixes:
                if stripped.startswith(prefix):
                    marker = prefix.rstrip()
                    if marker == "1.":
                        marker = "1"
                    item_text = stripped[len(prefix):].strip()
                    if current_list_marker != marker:
                        flush_list()
                        current_list_marker = marker
                    current_list_items.append(item_text)
                    break
            i += 1
            continue

        current_para.append(stripped)
        i += 1

    flush_para()
    flush_list()
    if in_table and table_lines:
        headers = []
        rows = []
        for j, tbl_line in enumerate(table_lines):
            if j == 0:
                cells = [cell.strip() for cell in tbl_line.strip("|").split("|")]
                headers = cells
            elif j == 1 and all(c in "-:" for c in tbl_line.replace("|", "").replace(" ", "")):
                continue
            else:
                cells = [cell.strip() for cell in tbl_line.strip("|").split("|")]
                rows.append(cells)
        if headers:
            table = Table(page=table_page, headers=headers, rows=rows)
            elements.append(Element("table", "", table=table, pages={table_page}))

    title = " ".join(title_parts) if title_parts else ""

    preamble_parts = []
    while elements and elements[0].kind in {"paragraph", "item"}:
        preamble_parts.append(elements[0].text)
        elements.pop(0)
    preamble = " ".join(preamble_parts)

    body_text = " ".join(elem.text for elem in elements if elem.kind in {"paragraph", "item"})

    meta = parse_meta(header, title, preamble + "\n" + body_text, path)

    return Document(
        doc_id=meta["id"],
        source_format="image",
        title=title,
        audience=meta["audience"],
        statut=meta["statut"],
        updated=meta["updated"],
        effective_from=meta["effective_from"],
        offer_window=meta["offer_window"],
        supersedes=meta["supersedes"],
        preamble=preamble,
        elements=elements,
        source_hash=file_hash,
    )


def banner(doc: Document) -> str:
    if doc.statut == "deprecated":
        if doc.offer_window:
            from_date = doc.offer_window.get("from")
            to_date = doc.offer_window.get("to")
            if from_date and to_date:
                return f"ARCHIVE — offre valable du {from_date} au {to_date}, remplacée depuis."
        return "ARCHIVE — offre passée, remplacée depuis."
    elif doc.statut == "current" and doc.effective_from:
        return f"EN VIGUEUR depuis le {doc.effective_from}."
    return ""

def _slug(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text

def chunk_document(doc: Document) -> list[Chunk]:
    chunks = []
    shallowest = min((elem.level for elem in doc.elements if elem.kind == "heading"), default=0)
    current_section_title_parts = []
    section_elements: list[Element] = []
    section_tables: list[Table] = []
    section_pages: set[int] = set()
    
    def flush_section():
        nonlocal current_section_title_parts, section_elements, section_tables, section_pages
        if not section_elements:
            return
        section_title = doc.title if not current_section_title_parts else " > ".join(current_section_title_parts)
        text_parts = []
        text_parts.append("# description")
        text_parts.append(doc.title)
        banner_text = banner(doc)
        if banner_text:
            text_parts.append(banner_text)
        if doc.preamble.strip():
            text_parts.append(doc.preamble)
        text_parts.append("")
        
        if doc.doc_id.startswith("faq-"):
            text_parts.append("# demande de")
        else:
            text_parts.append("# section")
        text_parts.append(section_title)
        text_parts.append("")
        
        text_parts.append("# texte")
        for elem in section_elements:
            if elem.kind == "paragraph":
                text_parts.append(elem.text)
            elif elem.kind == "item":
                text_parts.append(f"{elem.marker} {elem.text}" if elem.marker else elem.text)
            elif elem.kind == "table" and elem.table:
                text_parts.append(render_table(elem.table))
            text_parts.append("")
        
        text = "\n".join(text_parts).strip()
        slug = "preambule" if not current_section_title_parts else _slug(section_title)
        chunk_id = f"{doc.doc_id}#{slug}"
        
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            title=doc.title,
            heading=section_title,
            text=text,
            search_text=text,
            tables=section_tables.copy(),
            pages=sorted(section_pages),
            audience=doc.audience,
            statut=doc.statut,
            updated=doc.updated,
            effective_from=doc.effective_from,
            offer_window=doc.offer_window,
            supersedes=doc.supersedes
        ))
        section_elements.clear()
        section_tables.clear()
        section_pages.clear()
    
    current_section_title_parts = []
    for elem in doc.elements:
        if elem.kind == "heading":
            flush_section()
            if elem.level == shallowest:
                current_section_title_parts = [elem.text]
            else:
                current_section_title_parts = current_section_title_parts[:elem.level - shallowest] + [elem.text]
        else:
            section_elements.append(elem)
            if elem.pages:
                section_pages.update(elem.pages)
            if elem.kind == "table" and elem.table:
                section_tables.append(elem.table)
    
    flush_section()
    return chunks

def render_table(table: Table) -> str:
    if not table.headers:
        return ""
    
    header_row = "| " + " | ".join(table.headers) + " |"
    separator = "|-" + "-|-".join("-" * len(h) for h in table.headers) + "-|"
    
    rows = []
    for row in table.rows:
        row_cells = [cell if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(row_cells) + " |")
    
    return "\n".join([header_row, separator] + rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python -m neova.rag.ingestion <pdf_file>")
        sys.exit(1)

    pdf_path = CORPUS_DIR / sys.argv[1]
    if not pdf_path.exists():
        print(f"Fichier introuvable dans corpus/ : {sys.argv[1]}")
        sys.exit(1)

    lines, drawings, _ = layout(pdf_path)
    tables = find_tables(lines, drawings)

    if not tables:
        print("No tables found.")
    else:
        print(f"Found {len(tables)} table(s):")
        for i, table in enumerate(tables, 1):
            print(f"\nTable {i} (page {table.page}):")
            print("Headers:", table.headers)
            print("Rows:")
            for row in table.rows:
                print("  ", row)

    tables_line_ids = set()
    for table in tables:
        tables_line_ids.update(table.line_ids)

    body_size, heading_levels = heading_levels(lines, tables_line_ids)
    print(f"\nBody size: {body_size}")
    print("Heading levels:")
    for size, level in sorted(heading_levels.items(), key=lambda x: x[0], reverse=True):
        print(f"  {size} -> H{level}")

    print("\nHeading lines:")
    for idx, line in enumerate(lines):
        if idx in tables_line_ids:
            continue
        if line.size in heading_levels:
            level = heading_levels[line.size]
            print(f"  H{level}: {line.text}")

    print("\n=== parse_pdf ===")
    doc = parse_pdf(pdf_path)
    print(f"Title: {doc.title}")
    print(f"Audience: {doc.audience}")
    print(f"Statut: {doc.statut}")
    print(f"Preamble: {doc.preamble[:200]}" if doc.preamble else "Preamble: (empty)")
    print("\nElements:")
    for elem in doc.elements:
        if elem.kind == "table":
            preview = f"<table> {elem.table.headers if elem.table else []}"
        elif elem.kind == "heading":
            preview = f"<heading> L{elem.level} {elem.text[:80]}"
        else:
            preview = f"<{elem.kind}> {elem.text[:80]}"
        print(f"  {preview}")
