from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
JSON_ROOT = DATA_ROOT / "json"
TREE_INPUT_DIR = JSON_ROOT / "single_tree_per_file"
OUTPUT_DIR = DATA_ROOT / "embedding_inputs"

TABLE_START_RE = re.compile(r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$")
MERGE_TAG_RE = re.compile(r"^\[@(?P<tag>\d+)\]\s*")


def text_value(node: dict[str, Any]) -> str:
    return str(node.get("normalized_content") or "").strip()


def original_value(node: dict[str, Any]) -> str:
    return str(node.get("original_content") or "")


def strip_merge_tag(value: str) -> str:
    return MERGE_TAG_RE.sub("", value or "", count=1).strip()


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def line_metadata(node: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(node.get("metadata") or {})
    result = {
        "page_range": metadata.get("page_range") or {"start": None, "end": None},
        "line_range": metadata.get("line_range") or {"start": None, "end": None},
        "line_char_range": metadata.get("line_char_range") or {"min": None, "max": None},
    }
    for key in ("table_sequence", "cell_row", "cell_col"):
        if key in node:
            result[key] = node.get(key)
    return result


def position_hash(position: str) -> str:
    # Keep the full coordinate for humans; use this stable short key for machine lookup.
    return hashlib.blake2b(position.encode("utf-8"), digest_size=8).hexdigest()


def position_parts(position: str | None, file_stem: str | None = None) -> list[str]:
    if not position:
        return []
    position_text = str(position)
    if file_stem:
        file_stem_text = str(file_stem)
        if position_text == file_stem_text:
            return [file_stem_text]
        prefix = f"{file_stem_text}."
        if position_text.startswith(prefix):
            suffix = position_text[len(prefix) :]
            return [file_stem_text] + [part for part in suffix.split(".") if part]
    return [part for part in position_text.split(".") if part]


def table_position_from_position(position: str | None, file_stem: str | None = None) -> str | None:
    parts = position_parts(position, file_stem)
    for index, part in enumerate(parts):
        if part.startswith("表格_"):
            return ".".join(parts[: index + 1])
    return None


def machine_position_fields(
    *,
    file_stem: str | None,
    position: str,
    covered_positions: list[str],
    ancestor_positions: list[str],
    table_position: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hash_algo": "blake2b-64",
        "file_hash": position_hash(file_stem or ""),
        "position_parts": position_parts(position, file_stem),
        "position_hash": position_hash(position),
        "covered_position_parts": [position_parts(covered_position, file_stem) for covered_position in covered_positions],
        "covered_position_hashes": [position_hash(covered_position) for covered_position in covered_positions],
    }
    if ancestor_positions:
        result["ancestor_position_parts"] = [position_parts(ancestor_position, file_stem) for ancestor_position in ancestor_positions]
        result["ancestor_position_hashes"] = [position_hash(ancestor_position) for ancestor_position in ancestor_positions]
    if table_position:
        result["table_position_parts"] = position_parts(table_position, file_stem)
        result["table_position_hash"] = position_hash(table_position)
    return result


def make_record(
    record_id: str,
    source: dict[str, Any],
    record_kind: str,
    position: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    covered_positions: list[str] | None = None,
    ancestor_positions: list[str] | None = None,
    table_position: str | None = None,
) -> dict[str, Any]:
    final_covered_positions = covered_positions if covered_positions is not None else [position]
    final_ancestor_positions = ancestor_positions or []
    file_stem = source.get("file_stem")
    final_table_position = table_position if table_position is not None else table_position_from_position(position, file_stem)
    record = {
        "id": record_id,
        "source_id": source.get("id"),
        "file_name": source.get("file_name"),
        "file_stem": source.get("file_stem"),
        "record_kind": record_kind,
        "position": position,
        "covered_positions": final_covered_positions,
        "ancestor_positions": final_ancestor_positions,
        "table_position": final_table_position,
        "text": f"位置：{position}\n內容：{content}",
        "metadata": metadata if metadata is not None else line_metadata(source),
    }
    record.update(
        machine_position_fields(
            file_stem=source.get("file_stem"),
            position=position,
            covered_positions=final_covered_positions,
            ancestor_positions=final_ancestor_positions,
            table_position=final_table_position,
        )
    )
    return record


def validate_position_hashes(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    hash_to_position: dict[str, str] = {}
    coordinate_count = 0
    for file_name, records in outputs.items():
        for record_index, record in enumerate(records, start=1):
            position_pairs: list[tuple[Any, Any]] = [(record.get("position"), record.get("position_hash"))]
            position_pairs.extend(zip(record.get("covered_positions") or [], record.get("covered_position_hashes") or []))
            position_pairs.extend(zip(record.get("ancestor_positions") or [], record.get("ancestor_position_hashes") or []))
            if record.get("table_position"):
                position_pairs.append((record.get("table_position"), record.get("table_position_hash")))
            expected_count = 1 + len(record.get("covered_positions") or []) + len(record.get("ancestor_positions") or [])
            expected_count += 1 if record.get("table_position") else 0
            if len(position_pairs) != expected_count:
                raise ValueError(f"{file_name}:{record_index} position/hash count mismatch")
            for coordinate, coordinate_hash in position_pairs:
                if not coordinate:
                    continue
                coordinate_count += 1
                coordinate = str(coordinate)
                coordinate_hash = str(coordinate_hash)
                existing = hash_to_position.get(coordinate_hash)
                if existing is not None and existing != coordinate:
                    raise ValueError(
                        "position hash collision: "
                        f"{coordinate_hash} maps to both {existing!r} and {coordinate!r}"
                    )
                hash_to_position[coordinate_hash] = coordinate
    return {
        "coordinate_refs": coordinate_count,
        "unique_coordinates": len(set(hash_to_position.values())),
        "unique_hashes": len(hash_to_position),
        "hash_algo": "blake2b-64",
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def clean_output_dir(path: Path, active_file_names: set[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old_path in path.glob("*.jsonl"):
        if old_path.name not in active_file_names:
            old_path.unlink()


def load_trees(input_dir: Path, file_names: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*.json"), key=lambda path: path.name)
    if file_names:
        wanted = {Path(name).with_suffix(".json").name for name in file_names}
        paths = [path for path in paths if path.name in wanted]
    if limit is not None:
        paths = paths[:limit]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def iter_nodes(root: dict[str, Any]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    pairs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def visit(node: dict[str, Any], ancestors: list[dict[str, Any]]) -> None:
        pairs.append((node, ancestors))
        for child in node.get("children") or []:
            visit(child, ancestors + [node])

    visit(root, [])
    return pairs


def ancestor_content_chain(ancestors: list[dict[str, Any]], include_root: bool = True) -> list[str]:
    parts: list[str] = []
    for ancestor in ancestors:
        if not include_root and ancestor.get("node_kind") == "root":
            continue
        if ancestor.get("node_kind") in {"root", "table", "table_cell"}:
            parts.append(str(ancestor.get("id") or ""))
            continue
        content = compact_text(text_value(ancestor))
        parts.append(content if content else str(ancestor.get("id") or ""))
    return [part for part in parts if part]


def ancestor_covered_positions(ancestors: list[dict[str, Any]], include_root: bool = True) -> list[str]:
    positions: list[str] = []
    for ancestor in ancestors:
        if not include_root and ancestor.get("node_kind") == "root":
            continue
        node_id = ancestor.get("id")
        if node_id:
            positions.append(str(node_id))
    return dedupe_keep_order(positions)


def non_table_ancestor_chain(ancestors: list[dict[str, Any]]) -> list[str]:
    parts: list[str] = []
    for ancestor in ancestors:
        if str(ancestor.get("node_kind") or "").startswith("table"):
            continue
        if ancestor.get("node_kind") == "root":
            parts.append(str(ancestor.get("id") or ""))
            continue
        content = compact_text(text_value(ancestor))
        parts.append(content if content else str(ancestor.get("id") or ""))
    return [part for part in parts if part]


def parse_table_cr(table_node: dict[str, Any]) -> int:
    header_row_count = table_node.get("header_row_count")
    if isinstance(header_row_count, int) and header_row_count > 0:
        return header_row_count
    first_line = original_value(table_node).splitlines()[0].strip() if original_value(table_node).splitlines() else ""
    match = TABLE_START_RE.match(first_line)
    if match and match.group("cr"):
        return max(1, int(match.group("cr")))
    return 1


def table_cells(table_node: dict[str, Any]) -> list[dict[str, Any]]:
    return [child for child in table_node.get("children") or [] if child.get("node_kind") == "table_cell"]


def table_cell_grid(table_node: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    grid: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in table_cells(table_node):
        row = cell.get("cell_row")
        col = cell.get("cell_col")
        if isinstance(row, int) and isinstance(col, int):
            grid[(row, col)] = cell
    return grid


def is_effective_data_cell(cell: dict[str, Any]) -> bool:
    if cell.get("is_table_header") is True:
        return False
    if cell.get("merge_tag") is not None:
        return False
    return bool(text_value(cell))


def dedupe_keep_order(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        normalized = compact_text(part)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def join_table_header(parts: list[str]) -> str:
    return "／".join(dedupe_keep_order(parts))


def header_chain_for_col(grid: dict[tuple[int, int], dict[str, Any]], col: int, cr: int) -> list[str]:
    candidates: list[str] = []
    for row in range(1, cr + 1):
        cell = grid.get((row, col))
        if cell is None:
            continue
        raw = strip_merge_tag(original_value(cell))
        normalized = text_value(cell)
        candidates.append(raw or normalized)
    return dedupe_keep_order(candidates)


def repeated_merge_contexts_by_row(cells: list[dict[str, Any]]) -> list[tuple[int, str, str, int]]:
    groups: dict[str, dict[str, Any]] = {}
    for cell in cells:
        tag = cell.get("merge_tag")
        if tag is None:
            continue
        text = str(cell.get("merge_text") or "").strip()
        if not text:
            continue
        group = groups.setdefault(tag, {"text": text, "cols": []})
        group["cols"].append(cell.get("cell_col"))

    contexts: list[tuple[int, str, str, int]] = []
    for tag, group in groups.items():
        cols = [col for col in group["cols"] if isinstance(col, int)]
        if len(cols) < 2:
            continue
        # Span length is the merge width. When multiple previous context rows
        # have the same width, the nearest row wins later.
        span_length = max(cols) - min(cols) + 1
        for col in cols:
            contexts.append((col, tag, str(group["text"]), span_length))
    return contexts


def row_context_chain(rows: dict[int, list[dict[str, Any]]], current_row: int, cr: int) -> list[str]:
    nearest_by_span: dict[int, tuple[int, list[str]]] = {}
    for row_index in sorted(row for row in rows if cr < row < current_row):
        by_span: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for col, _tag, text, span_length in repeated_merge_contexts_by_row(rows[row_index]):
            by_span[span_length].append((col, text))
        for span_length, texts in by_span.items():
            nearest_by_span[span_length] = (row_index, [text for _col, text in sorted(texts)])
    ordered: list[str] = []
    for _span, (_row_index, texts) in sorted(nearest_by_span.items(), key=lambda item: item[1][0]):
        ordered.extend(texts)
    return [part for part in ordered if compact_text(part)]


def cell_context_chain(rows: dict[int, list[dict[str, Any]]], current_row: int, col: int, cr: int) -> list[str]:
    contexts: list[str] = []
    seen_vertical: set[str] = set()
    for row_index in sorted((row for row in rows if cr < row < current_row), reverse=True):
        cell = next((candidate for candidate in rows[row_index] if candidate.get("cell_col") == col), None)
        if cell is None:
            continue
        raw = original_value(cell)
        if cell.get("merge_tag") is None:
            continue
        text = compact_text(str(cell.get("merge_text") or "") or strip_merge_tag(raw) or text_value(cell))
        if not text or text in seen_vertical:
            continue
        seen_vertical.add(text)
        contexts.append(text)
    contexts.reverse()
    return contexts


def paired_cell_prefix(grid: dict[tuple[int, int], dict[str, Any]], rows: dict[int, list[dict[str, Any]]], row: int, col: int, cr: int) -> str:
    parts = header_chain_for_col(grid, col, cr) + cell_context_chain(rows, row, col, cr)
    return "／".join(dedupe_keep_order(parts))


def paired_cell_text(
    grid: dict[tuple[int, int], dict[str, Any]],
    rows: dict[int, list[dict[str, Any]]],
    cell: dict[str, Any],
    content: str,
    cr: int,
) -> str:
    row = cell.get("cell_row")
    col = cell.get("cell_col")
    if not isinstance(row, int) or not isinstance(col, int):
        return content
    prefix = paired_cell_prefix(grid, rows, row, col, cr)
    return f"{prefix}：{content}" if prefix else content


def table_metadata(table_node: dict[str, Any], cell: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = line_metadata(cell or table_node)
    metadata["table_sequence"] = table_node.get("table_sequence")
    metadata["header_row_count"] = table_node.get("header_row_count")
    if cell is not None:
        metadata["cell_row"] = cell.get("cell_row")
        metadata["cell_col"] = cell.get("cell_col")
        metadata["is_table_header"] = cell.get("is_table_header")
        metadata["merge_tag"] = cell.get("merge_tag")
        metadata["merge_text"] = cell.get("merge_text")
        metadata["merge_col_start"] = cell.get("merge_col_start")
        metadata["merge_col_end"] = cell.get("merge_col_end")
        metadata["merge_col_span"] = cell.get("merge_col_span")
        metadata["merge_row_start"] = cell.get("merge_row_start")
        metadata["merge_row_end"] = cell.get("merge_row_end")
        metadata["merge_row_span"] = cell.get("merge_row_span")
    return metadata


def column_headers_for_cell(
    cell: dict[str, Any],
    grid: dict[tuple[int, int], dict[str, Any]],
    rows: dict[int, list[dict[str, Any]]],
    cr: int,
) -> list[str]:
    row = cell.get("cell_row")
    col = cell.get("cell_col")
    if not isinstance(row, int) or not isinstance(col, int):
        return []
    return dedupe_keep_order(header_chain_for_col(grid, col, cr) + cell_context_chain(rows, row, col, cr))


def table_metadata_with_header(
    table_node: dict[str, Any],
    cell: dict[str, Any],
    grid: dict[tuple[int, int], dict[str, Any]],
    rows: dict[int, list[dict[str, Any]]],
    cr: int,
) -> dict[str, Any]:
    metadata = table_metadata(table_node, cell)
    headers = column_headers_for_cell(cell, grid, rows, cr)
    metadata["column_headers"] = headers
    metadata["column_header"] = "／".join(headers)
    return metadata


def range_values(cells: list[dict[str, Any]], range_name: str, key: str) -> list[int]:
    values: list[int] = []
    for cell in cells:
        value = ((cell.get("metadata") or {}).get(range_name) or {}).get(key)
        if isinstance(value, int):
            values.append(value)
    return values


def row_metadata(
    table_node: dict[str, Any],
    row_index: int,
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    page_starts = range_values(cells, "page_range", "start")
    page_ends = range_values(cells, "page_range", "end")
    line_starts = range_values(cells, "line_range", "start")
    line_ends = range_values(cells, "line_range", "end")
    char_mins = range_values(cells, "line_char_range", "min")
    char_maxes = range_values(cells, "line_char_range", "max")
    metadata = {
        "page_range": {
            "start": min(page_starts) if page_starts else None,
            "end": max(page_ends) if page_ends else None,
        },
        "line_range": {
            "start": min(line_starts) if line_starts else None,
            "end": max(line_ends) if line_ends else None,
        },
        "line_char_range": {
            "min": min(char_mins) if char_mins else None,
            "max": max(char_maxes) if char_maxes else None,
        },
        "table_sequence": table_node.get("table_sequence"),
        "header_row_count": table_node.get("header_row_count"),
        "cell_row": row_index,
        "cell_col": None,
    }
    return metadata


def is_table_node(node: dict[str, Any]) -> bool:
    return str(node.get("node_kind") or "").startswith("table")


def build_all_raw_data_records(trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tree in trees:
        for node, ancestors in iter_nodes(tree):
            content = text_value(node)
            if not content:
                continue
            position = str(node.get("id") or "")
            records.append(
                make_record(
                    f"all_raw_data::{position}",
                    node,
                    "all_raw_data",
                    position,
                    content,
                    ancestor_positions=ancestor_covered_positions(ancestors),
                )
            )
    return records


def build_all_node_records(trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tree in trees:
        for node, ancestors in iter_nodes(tree):
            if is_table_node(node):
                continue
            content = text_value(node)
            if not content:
                continue
            position = str(node.get("id") or "")
            records.append(
                make_record(
                    f"all_node::{position}",
                    node,
                    "all_node",
                    position,
                    content,
                    ancestor_positions=ancestor_covered_positions(ancestors),
                    table_position=None,
                )
            )
    return records


def build_leaf_with_ancestor_records(trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tree in trees:
        for node, ancestors in iter_nodes(tree):
            if node.get("children"):
                continue
            if is_table_node(node):
                continue
            own_content = compact_text(text_value(node))
            if not own_content:
                continue
            chain = ancestor_content_chain(ancestors) + [own_content]
            position = str(node.get("id") or "")
            ancestor_positions = ancestor_covered_positions(ancestors)
            records.append(
                make_record(
                    f"leaf_with_ancestors::{position}",
                    node,
                    "leaf_with_ancestors",
                    position,
                    " > ".join(chain),
                    covered_positions=ancestor_positions + [position],
                    ancestor_positions=ancestor_positions,
                )
            )
    return records


def row_cells_by_row(table_node: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in table_cells(table_node):
        row = cell.get("cell_row")
        if isinstance(row, int):
            grouped[row].append(cell)
    for cells in grouped.values():
        cells.sort(key=lambda cell: cell.get("cell_col") or 0)
    return dict(grouped)


def row_covered_positions(cells: list[dict[str, Any]]) -> list[str]:
    return [str(cell.get("id")) for cell in cells if is_effective_data_cell(cell) and cell.get("id")]


def build_table_row_records(trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tree in trees:
        for table_node, ancestors in iter_nodes(tree):
            if table_node.get("node_kind") != "table":
                continue
            cr = parse_table_cr(table_node)
            grid = table_cell_grid(table_node)
            parent_chain = non_table_ancestor_chain(ancestors)
            rows = row_cells_by_row(table_node)
            for row_index in sorted(row for row in rows if row > cr):
                row_parts: list[str] = []
                for cell in rows[row_index]:
                    cell_content = compact_text(text_value(cell))
                    if not cell_content:
                        continue
                    row_parts.append(paired_cell_text(grid, rows, cell, cell_content, cr))
                row_parts = dedupe_keep_order(row_parts)
                if not row_parts:
                    continue
                position = f"{table_node.get('id')}.r{row_index}"
                table_parts = [f"{table_node.get('id')} {'；'.join(row_parts)}".strip()]
                content_parts = parent_chain + table_parts
                records.append(
                    make_record(
                        f"table_row::{position}",
                        table_node,
                        "table_row",
                        position,
                        " > ".join(content_parts),
                        row_metadata(table_node, row_index, rows[row_index]),
                        row_covered_positions(rows[row_index]),
                    )
                )
    return records


def table_cell_item_leaves(cell: dict[str, Any]) -> list[dict[str, Any]]:
    item_children = [child for child in cell.get("children") or [] if child.get("node_kind") == "table_cell_item"]
    if not item_children:
        return [cell] if text_value(cell) else []

    leaves: list[dict[str, Any]] = []

    def visit(item: dict[str, Any]) -> None:
        child_items = [child for child in item.get("children") or [] if child.get("node_kind") == "table_cell_item"]
        if not child_items:
            if text_value(item):
                leaves.append(item)
            return
        for child in child_items:
            visit(child)

    for child in item_children:
        visit(child)
    return leaves


def has_table_cell_item_children(cell: dict[str, Any]) -> bool:
    return any(child.get("node_kind") == "table_cell_item" for child in cell.get("children") or [])


def path_between(root: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    if root is target:
        return [root]
    for child in root.get("children") or []:
        child_path = path_between(child, target)
        if child_path:
            return [root] + child_path
    return []


def build_table_hierarchy_leaf_records(trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tree in trees:
        for table_node, _ancestors in iter_nodes(tree):
            if table_node.get("node_kind") != "table":
                continue
            cr = parse_table_cr(table_node)
            grid = table_cell_grid(table_node)
            rows = row_cells_by_row(table_node)
            for cell in table_cells(table_node):
                row = cell.get("cell_row")
                col = cell.get("cell_col")
                if not isinstance(row, int) or not isinstance(col, int) or row <= cr or not is_effective_data_cell(cell):
                    continue
                if not has_table_cell_item_children(cell):
                    continue
                for leaf in table_cell_item_leaves(cell):
                    if leaf is cell:
                        continue
                    else:
                        path = path_between(cell, leaf)
                        item_path = path[1:]
                        item_chain = [compact_text(text_value(item)) for item in item_path if compact_text(text_value(item))]
                        covered_positions = [str(item.get("id")) for item in path if item.get("id")]
                    if not item_chain:
                        continue
                    position = str(leaf.get("id") or cell.get("id") or "")
                    content = paired_cell_text(grid, rows, cell, "／".join(item_chain), cr)
                    records.append(
                        make_record(
                            f"table_hierarchy_leaf::{position}",
                            leaf,
                            "table_hierarchy_leaf",
                            position,
                            content,
                            table_metadata_with_header(table_node, cell, grid, rows, cr),
                            covered_positions,
                        )
                    )
    return records


def build_table_inner_records(trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tree in trees:
        for table_node, _ancestors in iter_nodes(tree):
            if table_node.get("node_kind") != "table":
                continue
            cr = parse_table_cr(table_node)
            grid = table_cell_grid(table_node)
            rows = row_cells_by_row(table_node)
            for cell in table_cells(table_node):
                row = cell.get("cell_row")
                col = cell.get("cell_col")
                if not isinstance(row, int) or not isinstance(col, int) or row <= cr or not is_effective_data_cell(cell):
                    continue
                for leaf in table_cell_item_leaves(cell):
                    if leaf is cell:
                        item_chain = [compact_text(text_value(cell))]
                        covered_positions = [str(cell.get("id"))]
                    else:
                        path = path_between(cell, leaf)
                        item_path = path[1:]
                        item_chain = [compact_text(text_value(item)) for item in item_path if compact_text(text_value(item))]
                        covered_positions = [str(item.get("id")) for item in path if item.get("id")]
                    if not item_chain:
                        continue
                    position = str(leaf.get("id") or cell.get("id") or "")
                    content = paired_cell_text(grid, rows, cell, "／".join(item_chain), cr)
                    records.append(
                        make_record(
                            f"table_inner::{position}",
                            leaf,
                            "table_inner",
                            position,
                            content,
                            table_metadata_with_header(table_node, cell, grid, rows, cr),
                            covered_positions,
                        )
                    )
    return records


def build_table_inner_row_records(trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tree in trees:
        for table_node, _ancestors in iter_nodes(tree):
            if table_node.get("node_kind") != "table":
                continue
            cr = parse_table_cr(table_node)
            grid = table_cell_grid(table_node)
            rows = row_cells_by_row(table_node)
            for row_index in sorted(row for row in rows if row > cr):
                row_parts: list[str] = []
                for cell in rows[row_index]:
                    if not is_effective_data_cell(cell):
                        continue
                    cell_content = compact_text(text_value(cell))
                    if not cell_content:
                        continue
                    row_parts.append(paired_cell_text(grid, rows, cell, cell_content, cr))
                row_parts = dedupe_keep_order(row_parts)
                if not row_parts:
                    continue
                position = f"{table_node.get('id')}.r{row_index}"
                records.append(
                    make_record(
                        f"table_inner_row::{position}",
                        table_node,
                        "table_inner_row",
                        position,
                        "；".join(row_parts),
                        row_metadata(table_node, row_index, rows[row_index]),
                        row_covered_positions(rows[row_index]),
                    )
                )
    return records


def build_embedding_inputs(input_dir: Path = TREE_INPUT_DIR, output_dir: Path = OUTPUT_DIR, file_names: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    trees = load_trees(input_dir, file_names=file_names, limit=limit)
    outputs = {
        "all_raw_data.jsonl": build_all_raw_data_records(trees),
        "all_nodes.jsonl": build_all_node_records(trees),
        "leaf_with_ancestors.jsonl": build_leaf_with_ancestor_records(trees),
        "table_hierarchy_leaves.jsonl": build_table_hierarchy_leaf_records(trees),
        "table_inner.jsonl": build_table_inner_records(trees),
        "table_inner_rows.jsonl": build_table_inner_row_records(trees),
    }
    hash_summary = validate_position_hashes(outputs)
    clean_output_dir(output_dir, set(outputs))
    for file_name, records in outputs.items():
        write_jsonl(output_dir / file_name, records)
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "file_count": len(trees),
        "outputs": {file_name: len(records) for file_name, records in outputs.items()},
        "position_hashes": hash_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL embedding inputs from single-tree structured JSON.")
    parser.add_argument("--input-dir", type=Path, default=TREE_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--file-name", action="append", help="Only process the given source file name. Can be passed multiple times.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = build_embedding_inputs(args.input_dir, args.output_dir, args.file_name, args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
