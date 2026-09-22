from pathlib import Path

import pymupdf

from .models import Line, Table

HEADER_FILL = (0.06, 0.46, 0.43)


def layout(path: Path):
    with pymupdf.open(path) as doc:
        lines: list[Line] = []
        drawings_per_page: dict[int, list[tuple[tuple[float, float, float, float], tuple[float, float, float] | None]]] = {}
        page_heights: dict[int, float] = {}
        
        for page_num, page in enumerate(doc, start=1):
            page_heights[page_num] = page.rect.height
            
            drawings: list[tuple[tuple[float, float, float, float], tuple[float, float, float] | None]] = []
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
                        if size > max_size:
                            max_size = size
                    
                    if not visible_spans:
                        continue
                    
                    page_lines.append(Line(
                        page=page_num,
                        x0=line_x0,
                        y0=line_y0,
                        x1=line_x1,
                        y1=line_y1,
                        size=round(max_size, 1),
                        spans=visible_spans
                    ))
            
            page_lines.sort(key=lambda l: (l.y0, l.x0))
            lines.extend(page_lines)
    
    return lines, drawings_per_page, page_heights


def find_tables(lines: list[Line], drawings: dict[int, list[tuple[tuple[float, float, float, float], tuple[float, float, float] | None]]]) -> list[Table]:
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
            next_header_bottom = headers[i + 1][0][1] if i + 1 < len(headers) else float('inf')
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
            
            gaps = [row_boundaries[j] - row_boundaries[j - 1] for j in range(1, len(row_boundaries))]
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
                elif line_center_y - last_boundary <= row_height and header_line_indices and line.size <= header_line_size + 1:
                    col_match = False
                    for col_start in column_starts:
                        if abs(line.x0 - col_start) <= 2:
                            col_match = True
                            break
                    if col_match:
                        body_line_indices.append((line_idx, line, line_center_y))
            
            rows: list[list[str]] = []
            row_indices: list[set[int]] = []
            for j in range(1, len(row_boundaries[:last_boundary_idx + 1])):
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
                    cell_text = ' '.join(cells.get(col + 1, []))
                    row_cells.append(cell_text)
                rows.append(row_cells)
                row_indices.append(indices)
            
            if not rows:
                continue
            
            all_indices = header_indices.union(*row_indices)
            tables.append(Table(
                page=page,
                headers=column_labels,
                rows=rows,
                line_ids=all_indices
            ))
    
    return tables


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) != 2:
        print("Usage: uv run python -m neova.rag.ingestion <pdf_file>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
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