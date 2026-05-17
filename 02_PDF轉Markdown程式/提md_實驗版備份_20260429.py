from pathlib import Path

import pdfplumber


def normalize_cell(cell):
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").strip()


def pad_rows(rows):
    max_cols = max((len(row) for row in rows), default=0)
    padded = []
    for row in rows:
        normalized = [normalize_cell(cell) for cell in row]
        if len(normalized) < max_cols:
            normalized.extend([""] * (max_cols - len(normalized)))
        padded.append(normalized)
    return padded


def cluster_coords(coords, tolerance=1.0):
    if not coords:
        return []

    sorted_coords = sorted(float(coord) for coord in coords)
    clusters = [[sorted_coords[0]]]
    for coord in sorted_coords[1:]:
        if abs(coord - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(coord)
        else:
            clusters.append([coord])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def overlapping_interval_indices(start, end, boundaries, tolerance=1.0):
    indices = []
    for idx in range(len(boundaries) - 1):
        left = boundaries[idx]
        right = boundaries[idx + 1]
        if start < right - tolerance and end > left + tolerance:
            indices.append(idx)
    return indices


def build_merge_marker(value, row_span, col_span, merge_id):
    if row_span <= 1 and col_span <= 1:
        return value
    return f"[@{merge_id}] {value}"


def is_red_color(color, threshold=0.85):
    if color is None:
        return False
    if isinstance(color, (int, float)):
        return False
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        r, g, b = color[:3]
        return r >= threshold and g <= 0.2 and b <= 0.2
    return False


def is_red_underline_object(obj):
    object_type = obj.get("object_type")

    if object_type == "line":
        color = obj.get("stroking_color")
        x0 = float(obj.get("x0", 0) or 0)
        x1 = float(obj.get("x1", 0) or 0)
        top = float(obj.get("top", 0) or 0)
        bottom = float(obj.get("bottom", 0) or 0)
        is_horizontal = abs(top - bottom) <= 1.0
        return is_red_color(color) and is_horizontal and (x1 - x0) >= 20

    if object_type == "rect":
        fill_color = obj.get("non_stroking_color")
        stroke_color = obj.get("stroking_color")
        x0 = float(obj.get("x0", 0) or 0)
        x1 = float(obj.get("x1", 0) or 0)
        top = float(obj.get("top", 0) or 0)
        bottom = float(obj.get("bottom", 0) or 0)
        height = bottom - top
        width = x1 - x0
        return (is_red_color(fill_color) or is_red_color(stroke_color)) and height <= 2.0 and width >= 20

    return False


def remove_red_underlines(page):
    return page.filter(lambda obj: not is_red_underline_object(obj))


def unique_sorted_coords(coords, tolerance=0.1):
    if not coords:
        return []

    sorted_coords = sorted(float(coord) for coord in coords)
    unique_coords = [sorted_coords[0]]
    for coord in sorted_coords[1:]:
        if abs(coord - unique_coords[-1]) > tolerance:
            unique_coords.append(coord)
    return unique_coords


def bbox_gap(a_bbox, b_bbox):
    ax0, atop, ax1, abottom = a_bbox
    bx0, btop, bx1, bbottom = b_bbox
    x_gap = max(0.0, max(bx0 - ax1, ax0 - bx1))
    y_gap = max(0.0, max(btop - abottom, atop - bbottom))
    return x_gap, y_gap


def bboxes_connected(a_bbox, b_bbox, tolerance=6.0):
    x_gap, y_gap = bbox_gap(a_bbox, b_bbox)
    return x_gap <= tolerance and y_gap <= tolerance


def extract_cell_entries(table_obj, table_data):
    raw_rows = pad_rows(table_data)
    table_rows = getattr(table_obj, "rows", None)
    if not raw_rows or not table_rows:
        return []

    entries = []
    for row_idx, table_row in enumerate(table_rows):
        row_cells = getattr(table_row, "cells", None)
        if not row_cells:
            continue

        values = raw_rows[row_idx] if row_idx < len(raw_rows) else []
        if len(values) < len(row_cells):
            values = values + [""] * (len(row_cells) - len(values))

        for cell_idx, bbox in enumerate(row_cells):
            if not bbox:
                continue
            value = values[cell_idx] if cell_idx < len(values) else ""
            entries.append(
                {
                    "bbox": bbox,
                    "value": value,
                }
            )
    return entries


def build_boundary_block(entries):
    x_boundaries = unique_sorted_coords(
        [coord for entry in entries for coord in (entry["bbox"][0], entry["bbox"][2])]
    )
    y_boundaries = unique_sorted_coords(
        [coord for entry in entries for coord in (entry["bbox"][1], entry["bbox"][3])]
    )
    return {
        "entries": entries,
        "top": min(entry["bbox"][1] for entry in entries),
        "x_boundaries": x_boundaries,
        "y_boundaries": y_boundaries,
    }


def extract_table_boundary_blocks(table_obj, table_data):
    entries = extract_cell_entries(table_obj, table_data)
    if not entries:
        return []

    blocks = []
    for block_entries in split_entries_into_blocks(entries):
        blocks.append(build_boundary_block(block_entries))
    return blocks


def split_entries_into_blocks(entries, tolerance=6.0):
    if not entries:
        return []

    remaining = list(range(len(entries)))
    blocks = []

    while remaining:
        seed = remaining.pop(0)
        component = [seed]
        queue = [seed]

        while queue:
            current = queue.pop(0)
            current_bbox = entries[current]["bbox"]
            next_remaining = []
            for idx in remaining:
                if bboxes_connected(current_bbox, entries[idx]["bbox"], tolerance=tolerance):
                    component.append(idx)
                    queue.append(idx)
                else:
                    next_remaining.append(idx)
            remaining = next_remaining

        block_entries = [entries[idx] for idx in component]
        block_entries.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        blocks.append(block_entries)

    blocks.sort(key=lambda block: min(entry["bbox"][1] for entry in block))
    return blocks


def restore_boundary_block(block):
    entries = block["entries"]
    x_boundaries = block["x_boundaries"]
    y_boundaries = block["y_boundaries"]

    if not entries or len(x_boundaries) < 2 or len(y_boundaries) < 2:
        return None

    expanded = [[""] * (len(x_boundaries) - 1) for _ in range(len(y_boundaries) - 1)]
    merge_id = 1

    for entry in entries:
        x0, top, x1, bottom = entry["bbox"]
        value = entry["value"]

        if not value:
            continue

        col_indices = overlapping_interval_indices(x0, x1, x_boundaries)
        row_indices = overlapping_interval_indices(top, bottom, y_boundaries)

        if not col_indices or not row_indices:
            continue

        row_span = len(row_indices)
        col_span = len(col_indices)
        marker_value = build_merge_marker(value, row_span, col_span, merge_id)

        for r in row_indices:
            for c in col_indices:
                expanded[r][c] = marker_value

        if row_span > 1 or col_span > 1:
            merge_id += 1

    return expanded


def restore_simple_merged_cells(table_obj, table_data):
    boundary_blocks = extract_table_boundary_blocks(table_obj, table_data)
    if not boundary_blocks:
        return [{"rows": pad_rows(table_data), "top": table_obj.bbox[1]}]

    restored_blocks = []
    for block in boundary_blocks:
        expanded = restore_boundary_block(block)
        rows = expanded if expanded else [[entry["value"]] for entry in block["entries"]]
        restored_blocks.append(
            {
                "rows": rows,
                "top": block["top"],
            }
        )
    return restored_blocks


def render_markdown_table(rows):
    if not rows:
        return ""

    md_lines = []
    for row_idx, row in enumerate(rows):
        md_lines.append("| " + " | ".join(row) + " |")
        if row_idx == 0:
            md_lines.append("|" + "|".join(["---"] * len(row)) + "|")
    return "\n".join(md_lines)


def is_single_cell_table(rows):
    if not rows:
        return False
    return len(rows) == 1 and len(rows[0]) == 1


def strip_merge_tags(text):
    return (text or "").replace("\n", " ").strip().removeprefix("")


def normalize_compare_text(text):
    text = text or ""
    text = text.replace("\n", " ").strip()
    text = text.replace("\u3000", " ")
    text = text.strip()
    text = __import__("re").sub(r"^\[@\d+\]\s*", "", text)
    text = __import__("re").sub(r"\s+", " ", text)
    return text.strip()


def is_duplicated_two_col_artifact(rows):
    if not rows:
        return False

    width = max((len(row) for row in rows), default=0)
    if width != 2:
        return False

    duplicated_row_count = 0
    for row in rows:
        padded = row + [""] * (2 - len(row))
        left = normalize_compare_text(padded[0])
        right = normalize_compare_text(padded[1])
        non_empty_values = [value for value in (left, right) if value]

        if not non_empty_values:
            continue

        if len(set(non_empty_values)) > 1:
            return False

        if left and right and left == right:
            duplicated_row_count += 1

    return duplicated_row_count >= 2


def word_within_any_table(word, table_bboxes):
    x0, top, x1, bottom = word["x0"], word["top"], word["x1"], word["bottom"]
    for bbox in table_bboxes:
        if (
            x0 >= bbox[0]
            and x1 <= bbox[2]
            and top >= bbox[1]
            and bottom <= bbox[3]
        ):
            return True
    return False


def extract_text_lines_with_positions(page, table_bboxes, line_tolerance=3.0):
    filtered_page = page.filter(
        lambda obj: obj.get("object_type") != "char"
        or not any(
            obj["x0"] >= bbox[0]
            and obj["x1"] <= bbox[2]
            and obj["top"] >= bbox[1]
            and obj["bottom"] <= bbox[3]
            for bbox in table_bboxes
        )
    )

    clean_text = filtered_page.extract_text()
    if not clean_text:
        return []

    raw_lines = [line for line in clean_text.splitlines() if line.strip()]
    if not raw_lines:
        return []

    words = page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
        x_tolerance=1,
        y_tolerance=3,
    )
    words = [word for word in words if not word_within_any_table(word, table_bboxes)]
    if not words:
        return [{"kind": "text", "top": 0.0, "content": line} for line in raw_lines]

    words.sort(key=lambda word: (word["top"], word["x0"]))

    word_lines = []
    current_line = []
    current_top = None
    for word in words:
        if current_top is None or abs(word["top"] - current_top) <= line_tolerance:
            current_line.append(word)
            if current_top is None:
                current_top = word["top"]
            else:
                current_top = (current_top + word["top"]) / 2
        else:
            word_lines.append(current_line)
            current_line = [word]
            current_top = word["top"]
    if current_line:
        word_lines.append(current_line)

    line_tops = [
        min(word["top"] for word in sorted(line, key=lambda word: word["x0"]))
        for line in word_lines
    ]

    count = min(len(raw_lines), len(line_tops))
    entries = [
        {
            "kind": "text",
            "top": line_tops[idx],
            "content": raw_lines[idx],
        }
        for idx in range(count)
    ]

    if len(raw_lines) > count:
        fallback_top = line_tops[-1] if line_tops else 0.0
        for idx in range(count, len(raw_lines)):
            entries.append(
                {
                    "kind": "text",
                    "top": fallback_top + (idx - count + 1) * 0.01,
                    "content": raw_lines[idx],
                }
            )

    return entries


def merge_text_lines_and_tables(text_lines, table_blocks):
    merged = []
    table_index = 0

    for line in text_lines:
        while table_index < len(table_blocks) and table_blocks[table_index]["top"] <= line["top"]:
            if merged and merged[-1] != "":
                merged.append("")
            merged.append(table_blocks[table_index]["content"])
            merged.append("")
            table_index += 1
        merged.append(line["content"])

    while table_index < len(table_blocks):
        if merged and merged[-1] != "":
            merged.append("")
        merged.append(table_blocks[table_index]["content"])
        merged.append("")
        table_index += 1

    return merged


def extract_table_blocks(page):
    blocks = []
    filtered_page = remove_red_underlines(page)
    for table_obj in filtered_page.find_tables():
        table_data = table_obj.extract()
        if not table_data:
            continue

        restored_blocks = restore_simple_merged_cells(table_obj, table_data)
        for restored_block in restored_blocks:
            if is_single_cell_table(restored_block["rows"]):
                continue
            if is_duplicated_two_col_artifact(restored_block["rows"]):
                continue

            md_table = render_markdown_table(restored_block["rows"])
            if not md_table:
                continue

            blocks.append(
                {
                    "kind": "table",
                    "top": restored_block["top"],
                    "content": md_table,
                    "bbox": table_obj.bbox,
                }
            )
    return blocks


def unique_pdf_files(input_dir):
    seen = set()
    unique_files = []
    for pdf_path in sorted(input_dir.glob("*.pdf")) + sorted(input_dir.glob("*.PDF")):
        key = pdf_path.resolve().as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(pdf_path)
    return unique_files


def pdf_to_markdown(target_pdf_name=None, max_pages=None, page_number=None):
    base_dir = Path(__file__).resolve().parent.parent
    input_dir = base_dir / "法規資料_copy"
    md_output_dir = base_dir /"03_Markdown生成區"/"法規資料_md"

    md_output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = unique_pdf_files(input_dir)
    if target_pdf_name:
        pdf_files = [pdf_path for pdf_path in pdf_files if pdf_path.name == target_pdf_name]
    if not pdf_files:
        print(f"找不到 PDF 檔案，請檢查路徑：{input_dir}")
        return

    for pdf_path in pdf_files:
        print(f"正在處理: {pdf_path.name}")
        page_chunks = [f"# {pdf_path.stem}"]

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number is not None:
                    if 1 <= page_number <= len(pdf.pages):
                        pages = [pdf.pages[page_number - 1]]
                    else:
                        print(f"略過 {pdf_path.name}：找不到第 {page_number} 頁")
                        continue
                else:
                    pages = pdf.pages[:max_pages] if max_pages else pdf.pages

                for page in pages:
                    page_chunks.append(f"## Page {page.page_number}")

                    table_blocks = extract_table_blocks(page)
                    table_blocks.sort(key=lambda block: block["top"])
                    table_bboxes = [block["bbox"] for block in table_blocks]
                    text_lines = extract_text_lines_with_positions(page, table_bboxes)
                    page_chunks.extend(merge_text_lines_and_tables(text_lines, table_blocks))

            output_path = md_output_dir / f"{pdf_path.stem}.md"
            output_path.write_text("\n".join(page_chunks) + "\n", encoding="utf-8")

        except Exception as exc:
            print(f"處理 {pdf_path.name} 時出錯: {exc}")

    print("\n全部轉換完成！")


if __name__ == "__main__":
    pdf_to_markdown()
