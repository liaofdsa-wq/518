from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
JSON_ROOT = DATA_ROOT / "json"
SUMMARY_DIR = DATA_ROOT / "summary"

TABLE_SPLIT_ROOT = PROJECT_ROOT.parents[1]
TREE_DIR = TABLE_SPLIT_ROOT / "06_CleanTree生成區" / "法規資料_md_clean" / "tree"
MD_DIR = TABLE_SPLIT_ROOT / "07.6連續表格合併"


PAGE_RE = re.compile(r"^\[page\s+(\d+)\]$", re.IGNORECASE)
OPEN_TAG_RE = re.compile(r"^\[(.+?)\]\s*$")
CLOSE_TAG_RE = re.compile(r"^\[/([^\]]+)\]\s*$")
TABLE_START_RE = re.compile(
    r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$"
)
TABLE_END_RE = re.compile(r"^\[/TABLE id=([^\]]+)\]\s*$")
MERGE_TAG_RE = re.compile(r"^\[@(?P<tag>\d+)\]\s*")
SPECIAL_NODE_RE = re.compile(
    r"^SPECIAL\s+r=(?P<rule>[^\s\]]+)\s+s=(?P<serial>[^\s\]]+)\s+t=(?P<type>\d+)$"
)


def position_hash(position: str | None) -> str | None:
    if position is None:
        return None
    return hashlib.blake2b(str(position).encode("utf-8"), digest_size=8).hexdigest()


def table_position_from_position(position: str | None, file_stem: str | None) -> str | None:
    if not position or not file_stem:
        return None
    position_text = str(position)
    file_stem_text = str(file_stem)
    if position_text == file_stem_text:
        return None
    prefix = f"{file_stem_text}."
    if not position_text.startswith(prefix):
        return None
    parts = [file_stem_text] + [part for part in position_text[len(prefix) :].split(".") if part]
    for index, part in enumerate(parts):
        if part.startswith("表格_"):
            return ".".join(parts[: index + 1])
    return None


@dataclass
class TreeNode:
    name: str
    level: int
    sibling_order: int
    path_orders: list[int]
    line_number: int
    parent: "TreeNode | None" = None
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def path_key(self) -> str:
        return ".".join(str(order) for order in self.path_orders)


@dataclass
class MdNode:
    kind: str
    name: str
    line_start: int
    parent: "MdNode | None" = None
    children: list["MdNode"] = field(default_factory=list)
    content_lines: list[tuple[int, str]] = field(default_factory=list)
    pages: set[int] = field(default_factory=set)
    line_end: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def add_page(self, page: int | None) -> None:
        if page is not None:
            self.pages.add(page)


def build_structured_json_from_tree_special(
    tree_dir: Path = TREE_DIR,
    md_dir: Path = MD_DIR,
    file_names: list[str] | None = None,
    limit: int | None = None,
) -> None:
    def ensure_base_dirs() -> None:
        for path in (DATA_ROOT, JSON_ROOT, SUMMARY_DIR):
            path.mkdir(parents=True, exist_ok=True)

    def write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_sqlite(path: Path, trees: list[dict[str, Any]]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        target_path = path
        if target_path.exists():
            try:
                target_path.unlink()
            except PermissionError:
                target_path = target_path.with_name(f"{target_path.stem}_locked.db")
                if target_path.exists():
                    target_path.unlink()

        connection = sqlite3.connect(target_path)
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_stem TEXT NOT NULL,
                    file_hash TEXT,
                    position_hash TEXT,
                    parent_position_hash TEXT,
                    table_position TEXT,
                    table_position_hash TEXT,
                    parent_id TEXT,
                    original_content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    node_kind TEXT,
                    table_sequence TEXT,
                    cell_row INTEGER,
                    cell_col INTEGER,
                    is_table_header INTEGER,
                    header_row_count INTEGER,
                    merge_tag TEXT,
                    merge_text TEXT,
                    merge_col_start INTEGER,
                    merge_col_end INTEGER,
                    merge_col_span INTEGER,
                    merge_row_start INTEGER,
                    merge_row_end INTEGER,
                    merge_row_span INTEGER,
                    page_start INTEGER,
                    page_end INTEGER,
                    line_start INTEGER,
                    line_end INTEGER,
                    line_count INTEGER,
                    line_char_min INTEGER,
                    line_char_max INTEGER,
                    original_char_count INTEGER,
                    normalized_char_count INTEGER
                )
                """
            )
            cursor.execute("CREATE INDEX idx_nodes_parent_id ON nodes(parent_id)")
            cursor.execute("CREATE INDEX idx_nodes_file_stem ON nodes(file_stem)")
            cursor.execute("CREATE INDEX idx_nodes_file_hash ON nodes(file_hash)")
            cursor.execute("CREATE INDEX idx_nodes_position_hash ON nodes(position_hash)")
            cursor.execute("CREATE INDEX idx_nodes_table_position_hash ON nodes(table_position_hash)")

            def insert_node(node: dict[str, Any], parent_id: str | None) -> None:
                metadata = node.get("metadata", {})
                page_range = metadata.get("page_range", {})
                line_range = metadata.get("line_range", {})
                line_char_range = metadata.get("line_char_range", {})
                node_id = node.get("id")
                file_stem = node.get("file_stem")
                table_position = table_position_from_position(node_id, file_stem)
                cursor.execute(
                    """
                    INSERT INTO nodes (
                        id,
                        file_name,
                        file_stem,
                        file_hash,
                        position_hash,
                        parent_position_hash,
                        table_position,
                        table_position_hash,
                        parent_id,
                        original_content,
                        normalized_content,
                        node_kind,
                        table_sequence,
                        cell_row,
                        cell_col,
                        is_table_header,
                        header_row_count,
                        merge_tag,
                        merge_text,
                        merge_col_start,
                        merge_col_end,
                        merge_col_span,
                        merge_row_start,
                        merge_row_end,
                        merge_row_span,
                        page_start,
                        page_end,
                        line_start,
                        line_end,
                        line_count,
                        line_char_min,
                        line_char_max,
                        original_char_count,
                        normalized_char_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        node.get("file_name"),
                        file_stem,
                        position_hash(file_stem),
                        position_hash(node_id),
                        position_hash(parent_id),
                        table_position,
                        position_hash(table_position),
                        parent_id,
                        node.get("original_content", ""),
                        node.get("normalized_content", ""),
                        node.get("node_kind"),
                        node.get("table_sequence"),
                        node.get("cell_row"),
                        node.get("cell_col"),
                        (
                            1
                            if node.get("is_table_header") is True
                            else 0
                            if node.get("is_table_header") is False
                            else None
                        ),
                        node.get("header_row_count"),
                        node.get("merge_tag"),
                        node.get("merge_text"),
                        node.get("merge_col_start"),
                        node.get("merge_col_end"),
                        node.get("merge_col_span"),
                        node.get("merge_row_start"),
                        node.get("merge_row_end"),
                        node.get("merge_row_span"),
                        page_range.get("start"),
                        page_range.get("end"),
                        line_range.get("start"),
                        line_range.get("end"),
                        (
                            line_range.get("end") - line_range.get("start") + 1
                            if line_range.get("start") is not None and line_range.get("end") is not None
                            else None
                        ),
                        line_char_range.get("min"),
                        line_char_range.get("max"),
                        len(node.get("original_content", "")),
                        len(node.get("normalized_content", "")),
                    ),
                )
                for child in node.get("children", []):
                    insert_node(child, str(node.get("id")))

            for tree in trees:
                insert_node(tree, None)
            connection.commit()
        finally:
            connection.close()
        return target_path

    def collect_md_files() -> list[Path]:
        selected = set(file_names) if file_names else None
        md_files = sorted(path for path in md_dir.glob("*.md") if path.is_file())
        if selected is not None:
            md_files = [path for path in md_files if path.name in selected]
        if limit is not None:
            md_files = md_files[:limit]
        return md_files

    def parse_tree_file(path: Path) -> TreeNode:
        root = TreeNode(name="__root__", level=-1, sibling_order=0, path_orders=[], line_number=0)
        stack: list[TreeNode] = [root]
        sibling_counters: dict[tuple[int, ...], int] = {}
        base_indent: int | None = None

        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = re.match(r"^(?P<indent>\s*)\[(?P<name>.+?)\]\s*$", raw_line)
            if not match:
                continue

            indent = len(match.group("indent"))
            if base_indent is None:
                base_indent = indent
            relative_indent = max(indent - base_indent, 0)
            level = relative_indent // 2

            while len(stack) - 1 > level:
                stack.pop()

            parent = stack[-1]
            parent_key = tuple(parent.path_orders)
            sibling_counters[parent_key] = sibling_counters.get(parent_key, 0) + 1
            sibling_order = sibling_counters[parent_key]

            node = TreeNode(
                name=match.group("name"),
                level=level,
                sibling_order=sibling_order,
                path_orders=parent.path_orders + [sibling_order],
                line_number=line_number,
                parent=parent,
            )
            parent.children.append(node)
            stack.append(node)

        return root

    def parse_special_parts(node_name: str) -> dict[str, str] | None:
        match = SPECIAL_NODE_RE.match(node_name)
        if not match:
            return None
        return {
            "special_rule": match.group("rule"),
            "special_serial": match.group("serial"),
            "special_type": match.group("type"),
        }

    def parse_markdown_row(line: str) -> list[str] | None:
        stripped = line.strip()
        if "|" not in stripped:
            return None
        return [cell.strip() for cell in stripped.strip("|").split("|")]

    def is_separator_row(cells: list[str]) -> bool:
        if not cells:
            return False
        return all(re.fullmatch(r"[-: ]+", cell or "") for cell in cells)

    def normalize_cell_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def merge_tag(cell_text: str) -> str | None:
        match = MERGE_TAG_RE.match(cell_text or "")
        return match.group("tag") if match else None

    def strip_merge_tag(cell_text: str) -> str:
        return MERGE_TAG_RE.sub("", cell_text or "", count=1).strip()

    def build_merge_infos(raw_rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        merge_infos: list[list[dict[str, Any]]] = []
        tag_positions: dict[str, list[tuple[int, int]]] = {}

        for row_index, raw_row_data in enumerate(raw_rows, start=1):
            row_infos: list[dict[str, Any]] = []
            for col_index, raw_cell in enumerate(list(raw_row_data.get("cells", [])), start=1):
                tag = merge_tag(str(raw_cell))
                text = strip_merge_tag(str(raw_cell)) if tag is not None else None
                info = {
                    "merge_tag": tag,
                    "merge_text": text,
                    "merge_col_start": None,
                    "merge_col_end": None,
                    "merge_col_span": None,
                    "merge_row_start": None,
                    "merge_row_end": None,
                    "merge_row_span": None,
                }
                row_infos.append(info)
                if tag is not None:
                    tag_positions.setdefault(tag, []).append((row_index, col_index))
            merge_infos.append(row_infos)

        for positions in tag_positions.values():
            rows = [row for row, _col in positions]
            cols = [col for _row, col in positions]
            row_start = min(rows)
            row_end = max(rows)
            col_start = min(cols)
            col_end = max(cols)
            for row_index, col_index in positions:
                info = merge_infos[row_index - 1][col_index - 1]
                info["merge_col_start"] = col_start
                info["merge_col_end"] = col_end
                info["merge_col_span"] = col_end - col_start + 1
                info["merge_row_start"] = row_start
                info["merge_row_end"] = row_end
                info["merge_row_span"] = row_end - row_start + 1

        return merge_infos

    def resolve_table_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        resolved_rows: list[dict[str, Any]] = []
        merge_infos = build_merge_infos(raw_rows)
        for row_index, raw_row_data in enumerate(raw_rows, start=1):
            raw_row = list(raw_row_data.get("cells", []))
            width = len(raw_row)
            resolved_cells: list[str] = []

            for col_index, raw_cell in enumerate(raw_row):
                tag = merge_tag(raw_cell)
                current_text = strip_merge_tag(raw_cell)
                if tag is None:
                    resolved_cells.append(current_text)
                    continue

                source_text = ""
                for previous_row in reversed(resolved_rows):
                    previous_cells = previous_row.get("resolved_cells", [])
                    if len(previous_cells) != width or col_index >= len(previous_cells):
                        continue
                    candidate = previous_cells[col_index]
                    if normalize_cell_text(candidate):
                        source_text = candidate
                        break

                if source_text and current_text and current_text not in source_text and source_text not in current_text:
                    resolved_cells.append(f"{source_text} {current_text}".strip())
                else:
                    resolved_cells.append(current_text or source_text)

            resolved_rows.append(
                {
                    "row_index": row_index,
                    "line_number": raw_row_data.get("line_number"),
                    "page": raw_row_data.get("page"),
                    "raw_cells": list(raw_row),
                    "resolved_cells": resolved_cells,
                    "merge_infos": merge_infos[row_index - 1] if row_index <= len(merge_infos) else [],
                }
            )
        return resolved_rows

    def parse_table_block(lines: list[str], start_index: int, current_page: int | None) -> tuple[MdNode, int]:
        start_line = lines[start_index]
        start_match = TABLE_START_RE.match(start_line.strip())
        if start_match is None:
            raise ValueError(f"Expected table at line {start_index + 1}")

        table_node = MdNode(
            kind="table",
            name=start_match.group("table_id"),
            line_start=start_index + 1,
            attrs={
                "table_id": start_match.group("table_id"),
                "header_row_count": max(1, int(start_match.group("cr") or 1)),
            },
        )
        table_node.add_page(current_page)
        table_node.content_lines.append((start_index + 1, start_line))

        raw_rows: list[dict[str, Any]] = []
        cursor = start_index + 1
        local_page = current_page

        while cursor < len(lines):
            raw_line = lines[cursor]
            line_number = cursor + 1
            page_match = PAGE_RE.match(raw_line)
            if page_match:
                local_page = int(page_match.group(1))
                table_node.add_page(local_page)
                table_node.content_lines.append((line_number, raw_line))
                cursor += 1
                continue

            table_node.content_lines.append((line_number, raw_line))
            if local_page is not None:
                table_node.pages.add(local_page)

            if TABLE_END_RE.match(raw_line.strip()):
                table_node.line_end = line_number
                break

            cells = parse_markdown_row(raw_line)
            if cells and not is_separator_row(cells):
                raw_rows.append({"line_number": line_number, "page": local_page, "cells": cells})
            cursor += 1

        if table_node.line_end is None:
            raise ValueError(f"Unclosed table block starting at line {start_index + 1}")

        resolved_rows = resolve_table_rows(raw_rows)
        table_node.attrs["column_count"] = max((len(row.get("raw_cells", [])) for row in resolved_rows), default=0)
        table_node.attrs["row_count"] = len(raw_rows)
        table_node.attrs["resolved_rows"] = resolved_rows
        return table_node, cursor + 1

    def parse_md_file(path: Path) -> tuple[MdNode, int | None, dict[int, int | None]]:
        root = MdNode(kind="root", name="__root__", line_start=0)
        stack: list[MdNode] = [root]
        lines = path.read_text(encoding="utf-8").splitlines()
        current_page: int | None = None
        last_seen_page: int | None = None
        last_touched_node: MdNode | None = None
        line_to_page: dict[int, int | None] = {}
        cursor = 0

        while cursor < len(lines):
            raw_line = lines[cursor]
            line_number = cursor + 1

            page_match = PAGE_RE.match(raw_line)
            if page_match:
                current_page = int(page_match.group(1))
                last_seen_page = current_page
                line_to_page[line_number] = current_page
                for node in stack[1:]:
                    node.add_page(current_page)
                if last_touched_node is not None:
                    last_touched_node.add_page(current_page)
                cursor += 1
                continue

            table_match = TABLE_START_RE.match(raw_line.strip())
            if table_match:
                table_node, cursor = parse_table_block(lines, cursor, current_page)
                for table_line_number, table_line in table_node.content_lines:
                    if PAGE_RE.match(table_line):
                        page_match_in_table = PAGE_RE.match(table_line)
                        line_to_page[table_line_number] = int(page_match_in_table.group(1)) if page_match_in_table else current_page
                    else:
                        line_to_page[table_line_number] = current_page
                table_node.parent = stack[-1]
                stack[-1].children.append(table_node)
                last_touched_node = table_node
                continue

            close_match = CLOSE_TAG_RE.match(raw_line)
            if close_match:
                closing_name = close_match.group(1)
                for idx in range(len(stack) - 1, 0, -1):
                    if stack[idx].name == closing_name or (closing_name == "SPECIAL" and stack[idx].kind == "special"):
                        stack[idx].line_end = line_number
                        last_touched_node = stack[idx]
                        stack = stack[:idx]
                        break
                cursor += 1
                continue

            open_match = OPEN_TAG_RE.match(raw_line)
            if open_match:
                node_name = open_match.group(1)
                node_kind = "special" if parse_special_parts(node_name) else "article"
                md_node = MdNode(kind=node_kind, name=node_name, line_start=line_number, parent=stack[-1])
                md_node.add_page(current_page)
                if node_kind == "special":
                    md_node.attrs.update(parse_special_parts(node_name) or {})
                stack[-1].children.append(md_node)
                stack.append(md_node)
                last_touched_node = md_node
                cursor += 1
                continue

            stack[-1].content_lines.append((line_number, raw_line))
            stack[-1].add_page(current_page)
            line_to_page[line_number] = current_page
            last_touched_node = stack[-1]

            cursor += 1

        for node in stack[1:]:
            if node.line_end is None:
                node.line_end = len(lines)

        return root, last_seen_page, line_to_page

    def md_name_matches_tree(md_node: MdNode, tree_node: TreeNode) -> bool:
        if md_node.kind not in {"article", "special"}:
            return False
        tree_special = parse_special_parts(tree_node.name)
        md_special = parse_special_parts(md_node.name)
        if tree_special is not None or md_special is not None:
            return tree_special == md_special
        tree_name = tree_node.name.strip()
        md_name = md_node.name.strip()
        return md_name == tree_name or md_name.startswith(tree_name)

    def make_metadata(
        node: MdNode,
        line_to_page: dict[int, int | None],
        page_end_override: int | None = None,
    ) -> dict[str, Any]:
        line_numbers = [line_number for line_number, _ in node.content_lines]
        pages_from_lines = [line_to_page.get(line_number) for line_number in line_numbers if line_to_page.get(line_number) is not None]
        pages = sorted(set(pages_from_lines) | set(node.pages))
        page_start = pages[0] if pages else None
        page_end = page_end_override if page_end_override is not None else (pages[-1] if pages else None)
        line_lengths = [len(text) for _, text in node.content_lines]
        return {
            "page_range": {"start": page_start, "end": page_end},
            "line_range": {
                "start": line_numbers[0] if line_numbers else None,
                "end": line_numbers[-1] if line_numbers else None,
            },
            "line_char_range": {
                "min": min(line_lengths) if line_lengths else None,
                "max": max(line_lengths) if line_lengths else None,
            },
        }

    def ensure_unique_child_id(parent_state: dict[str, int], child_id: str) -> str:
        parent_state[child_id] = parent_state.get(child_id, 0) + 1
        count = parent_state[child_id]
        if count == 1:
            return child_id
        return f"{child_id}({count})"

    def flatten_md_subtree_lines(node: MdNode) -> list[tuple[int, str]]:
        parts: list[tuple[int, str]] = []
        parts.extend(node.content_lines)
        for child in node.children:
            if child.kind == "table":
                parts.extend(child.content_lines)
            else:
                parts.extend(flatten_md_subtree_lines(child))
        parts.sort(key=lambda item: item[0])
        return parts

    def flatten_all_md_output_lines(node: MdNode) -> list[tuple[int, str]]:
        parts: list[tuple[int, str]] = []
        parts.extend(node.content_lines)
        for child in node.children:
            if child.kind == "table":
                parts.extend(child.content_lines)
            else:
                parts.extend(flatten_all_md_output_lines(child))
        parts.sort(key=lambda item: item[0])
        return parts

    def iter_tree_nodes(node: TreeNode) -> list[TreeNode]:
        rows: list[TreeNode] = []
        for child in node.children:
            rows.append(child)
            rows.extend(iter_tree_nodes(child))
        return rows

    def iter_md_nodes(node: MdNode) -> list[MdNode]:
        rows: list[MdNode] = []
        for child in node.children:
            if child.kind in {"article", "special"}:
                rows.append(child)
                if child.kind != "special":
                    rows.extend(iter_md_nodes(child))
        return rows

    def find_matching_md_node(
        tree_node: TreeNode,
        md_nodes: list[MdNode],
        start_index: int,
    ) -> tuple[MdNode | None, int]:
        for idx in range(start_index, len(md_nodes)):
            if md_name_matches_tree(md_nodes[idx], tree_node):
                return md_nodes[idx], idx + 1
        return None, start_index

    def align_tree_and_md(tree_root: TreeNode, md_root: MdNode) -> dict[int, MdNode]:
        # Same strategy as 01_build_structured_json.py: md tags are an ordered
        # stream, while the tree file owns the final hierarchy.
        md_nodes = iter_md_nodes(md_root)
        md_index = 0
        aligned: dict[int, MdNode] = {}

        for tree_node in iter_tree_nodes(tree_root):
            matched, md_index = find_matching_md_node(tree_node, md_nodes, md_index)
            if matched is not None:
                aligned[id(tree_node)] = matched

        return aligned

    def collect_inline_table_nodes(md_node: MdNode, tree_node: TreeNode) -> list[MdNode]:
        tables: list[MdNode] = []
        for child in md_node.children:
            if child.kind == "table":
                tables.append(child)
            elif child.kind == "article" and md_name_matches_tree(child, tree_node):
                # Continuous tables may reopen the same tag across pages and
                # become nested by syntax. Treat only same-tag descendants as
                # the same logical section so real child sections stay separate.
                tables.extend(collect_inline_table_nodes(child, tree_node))
        tables.sort(key=lambda item: item.line_start)
        return tables

    def has_same_article_ancestor(md_node: MdNode) -> bool:
        parent = md_node.parent
        md_name = md_node.name.strip()
        while parent is not None and parent.kind != "root":
            if parent.kind == "article" and parent.name.strip() == md_name:
                return True
            parent = parent.parent
        return False

    def build_table_alignments(tree_root: TreeNode, md_root: MdNode) -> dict[int, list[MdNode]]:
        tree_nodes = [node for node in iter_tree_nodes(tree_root) if parse_special_parts(node.name) is None]
        table_alignments: dict[int, list[MdNode]] = {}
        tree_index = 0

        for md_node in iter_md_nodes(md_root):
            if md_node.kind != "article" or has_same_article_ancestor(md_node):
                continue
            if not any(child.kind == "table" for child in md_node.children):
                continue

            for idx in range(tree_index, len(tree_nodes)):
                tree_node = tree_nodes[idx]
                if md_name_matches_tree(md_node, tree_node):
                    table_alignments.setdefault(id(tree_node), []).append(md_node)
                    tree_index = idx + 1
                    break

        return table_alignments

    def collect_table_nodes_outside_special(node: MdNode, inside_special: bool = False) -> list[MdNode]:
        tables: list[MdNode] = []
        current_inside_special = inside_special or node.kind == "special"
        for child in node.children:
            if child.kind == "table" and not current_inside_special:
                tables.append(child)
            elif child.kind != "table":
                tables.extend(collect_table_nodes_outside_special(child, current_inside_special))
        return tables

    def collect_assigned_table_line_starts(
        tree_root: TreeNode,
        table_alignments: dict[int, list[MdNode]],
        aligned: dict[int, MdNode],
    ) -> set[int]:
        assigned: set[int] = set()
        for tree_node in iter_tree_nodes(tree_root):
            source_nodes = list(table_alignments.get(id(tree_node), []))
            aligned_node = aligned.get(id(tree_node))
            if aligned_node is not None and all(source.line_start != aligned_node.line_start for source in source_nodes):
                source_nodes.append(aligned_node)
            for source_node in source_nodes:
                for table_node in collect_inline_table_nodes(source_node, tree_node):
                    assigned.add(table_node.line_start)
        return assigned

    def build_orphan_table_alignments(
        tree_root: TreeNode,
        md_root: MdNode,
        aligned: dict[int, MdNode],
        table_alignments: dict[int, list[MdNode]],
    ) -> dict[int, list[MdNode]]:
        assigned_line_starts = collect_assigned_table_line_starts(tree_root, table_alignments, aligned)
        anchors = sorted(
            (
                (md_node.line_start, tree_node)
                for tree_node in iter_tree_nodes(tree_root)
                for md_node in [aligned.get(id(tree_node))]
                if md_node is not None
            ),
            key=lambda item: item[0],
        )
        if not anchors:
            return {}

        orphan_alignments: dict[int, list[MdNode]] = {}
        anchor_index = 0
        current_tree_node = anchors[0][1]

        for table_node in sorted(collect_table_nodes_outside_special(md_root), key=lambda item: item.line_start):
            if table_node.line_start in assigned_line_starts:
                continue
            while anchor_index + 1 < len(anchors) and anchors[anchor_index + 1][0] <= table_node.line_start:
                anchor_index += 1
                current_tree_node = anchors[anchor_index][1]
            orphan_alignments.setdefault(id(current_tree_node), []).append(table_node)

        return orphan_alignments

    def build_table_line_owners(
        tree_root: TreeNode,
        table_alignments: dict[int, list[MdNode]],
        orphan_alignments: dict[int, list[MdNode]],
    ) -> dict[int, int]:
        owners: dict[int, int] = {}
        tree_nodes_by_id = {id(tree_node): tree_node for tree_node in iter_tree_nodes(tree_root)}
        for tree_node_id, source_nodes in table_alignments.items():
            tree_node = tree_nodes_by_id.get(tree_node_id)
            if tree_node is None:
                continue
            for source_node in source_nodes:
                for table_node in collect_inline_table_nodes(source_node, tree_node):
                    owners.setdefault(table_node.line_start, tree_node_id)
        for tree_node_id, table_nodes in orphan_alignments.items():
            for table_node in table_nodes:
                owners.setdefault(table_node.line_start, tree_node_id)
        return owners

    def collect_covered_line_numbers(tree_root: TreeNode, md_root: MdNode) -> set[int]:
        covered = {line_number for line_number, _ in md_root.content_lines}
        tree_nodes_by_id = {id(tree_node): tree_node for tree_node in iter_tree_nodes(tree_root)}
        for tree_node in iter_tree_nodes(tree_root):
            md_node = aligned_nodes.get(id(tree_node))
            if md_node is not None:
                if parse_special_parts(tree_node.name) is not None:
                    covered.update(line_number for line_number, _ in flatten_md_subtree_lines(md_node))
                else:
                    covered.update(line_number for line_number, _ in md_node.content_lines)

        for tree_node_id, source_nodes in table_aligned_nodes.items():
            tree_node = tree_nodes_by_id.get(tree_node_id)
            if tree_node is None:
                continue
            for source_node in source_nodes:
                for table_node in collect_inline_table_nodes(source_node, tree_node):
                    covered.update(line_number for line_number, _ in table_node.content_lines)

        for table_nodes in orphan_aligned_tables.values():
            for table_node in table_nodes:
                covered.update(line_number for line_number, _ in table_node.content_lines)

        return covered

    def build_root_content_lines(tree_root: TreeNode, md_root: MdNode) -> list[tuple[int, str]]:
        covered = collect_covered_line_numbers(tree_root, md_root)
        root_lines = list(md_root.content_lines)
        root_line_numbers = {line_number for line_number, _ in root_lines}
        for line_number, text in flatten_all_md_output_lines(md_root):
            if line_number not in covered and line_number not in root_line_numbers:
                root_lines.append((line_number, text))
                root_line_numbers.add(line_number)
        root_lines.sort(key=lambda item: item[0])
        return root_lines

    def make_cell_metadata(line_number: int | None, page: int | None, raw_content: str) -> dict[str, Any]:
        cell_node = MdNode(kind="table_cell", name="", line_start=line_number or 0)
        if line_number is not None:
            cell_node.content_lines = [(line_number, raw_content)]
        cell_node.add_page(page)
        return make_metadata(cell_node, line_to_page)

    def split_cell_segments(cell_text: str) -> list[str]:
        normalized = cell_text.replace("<br>", "\n")
        return [segment.strip() for segment in normalized.splitlines() if segment.strip()]

    def parse_cell_item_segment(segment: str) -> tuple[int, str] | None:
        match = re.match(r"^(?P<marks>-{1,8})\s*(?P<text>.+?)\s*$", segment)
        if not match:
            return None
        return len(match.group("marks")), match.group("text")

    def build_table_cell_item_nodes(
        cell_id: str,
        normalized_content: str,
        line_number: int | None,
        page: int | None,
    ) -> list[dict[str, Any]]:
        root_children: list[dict[str, Any]] = []
        root_state: dict[str, int] = {}
        stack: list[tuple[int, dict[str, Any], dict[str, int]]] = []

        for segment in split_cell_segments(normalized_content):
            parsed = parse_cell_item_segment(segment)
            if parsed is None:
                if stack:
                    current = stack[-1][1]
                    current["original_content"] = f"{current['original_content']}\n{segment}"
                    current["normalized_content"] = f"{current['normalized_content']}\n{segment}"
                    current["metadata"] = make_cell_metadata(line_number, page, current["original_content"])
                continue

            level, text = parsed
            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                parent_node = stack[-1][1]
                parent_state = stack[-1][2]
                parent_id = str(parent_node.get("id"))
                siblings = parent_node.setdefault("children", [])
            else:
                parent_state = root_state
                parent_id = cell_id
                siblings = root_children

            base_child_id = f"{parent_id}.{len(parent_state) + 1}"
            child_id = ensure_unique_child_id(parent_state, base_child_id)
            child_node = {
                "id": child_id,
                "node_kind": "table_cell_item",
                "table_sequence": table_sequence_from_parent_id(cell_id),
                "cell_row": None,
                "cell_col": None,
                "original_content": text,
                "normalized_content": text,
                "children": [],
                "metadata": make_cell_metadata(line_number, page, text),
            }
            siblings.append(child_node)
            stack.append((level, child_node, {}))

        return root_children

    def build_table_cell_node(
        row: dict[str, Any],
        col_index: int,
        parent_id: str,
        parent_state: dict[str, int],
        header_row_count: int,
    ) -> dict[str, Any]:
        raw_cells = list(row.get("raw_cells", []))
        resolved_cells = list(row.get("resolved_cells", []))
        merge_infos = list(row.get("merge_infos", []))
        raw_content = str(raw_cells[col_index - 1]) if col_index <= len(raw_cells) else ""
        normalized_content = str(resolved_cells[col_index - 1]) if col_index <= len(resolved_cells) else strip_merge_tag(raw_content)
        merge_info = dict(merge_infos[col_index - 1]) if col_index <= len(merge_infos) else {}
        base_child_id = f"{parent_id}.r{row.get('row_index', 0)}.c{col_index}"
        child_id = ensure_unique_child_id(parent_state, base_child_id)
        line_number = row.get("line_number")
        page = row.get("page")
        return {
            "id": child_id,
            "node_kind": "table_cell",
            "table_sequence": table_sequence_from_parent_id(parent_id),
            "cell_row": row.get("row_index"),
            "cell_col": col_index,
            "is_table_header": isinstance(row.get("row_index"), int) and row.get("row_index") <= header_row_count,
            "header_row_count": header_row_count,
            "merge_tag": merge_info.get("merge_tag"),
            "merge_text": merge_info.get("merge_text"),
            "merge_col_start": merge_info.get("merge_col_start"),
            "merge_col_end": merge_info.get("merge_col_end"),
            "merge_col_span": merge_info.get("merge_col_span"),
            "merge_row_start": merge_info.get("merge_row_start"),
            "merge_row_end": merge_info.get("merge_row_end"),
            "merge_row_span": merge_info.get("merge_row_span"),
            "original_content": raw_content,
            "normalized_content": normalized_content,
            "children": build_table_cell_item_nodes(child_id, normalized_content, line_number, page),
            "metadata": make_cell_metadata(line_number, page, raw_content),
        }

    def build_table_cell_nodes(md_node: MdNode, parent_id: str, parent_state: dict[str, int]) -> list[dict[str, Any]]:
        cell_nodes: list[dict[str, Any]] = []
        header_row_count = int(md_node.attrs.get("header_row_count", 1) or 1)
        for row in md_node.attrs.get("resolved_rows", []):
            raw_cells = list(row.get("raw_cells", []))
            for col_index in range(1, len(raw_cells) + 1):
                cell_nodes.append(build_table_cell_node(row, col_index, parent_id, parent_state, header_row_count))
        return cell_nodes

    def table_sequence(md_node: MdNode) -> str:
        table_id = str(md_node.attrs.get("table_id", md_node.name))
        match = re.search(r"_table_(.+)$", table_id)
        return match.group(1) if match else table_id

    def table_sequence_from_parent_id(parent_id: str) -> str | None:
        match = re.search(r"(?:^|\.)\u8868\u683c_([^.()]+)", parent_id)
        return match.group(1) if match else None

    def build_table_node(md_node: MdNode, parent_id: str, parent_state: dict[str, int]) -> dict[str, Any]:
        sequence = table_sequence(md_node)
        base_child_id = f"{parent_id}.表格_{sequence}"
        child_id = ensure_unique_child_id(parent_state, base_child_id)
        cell_state: dict[str, int] = {}
        header_row_count = int(md_node.attrs.get("header_row_count", 1) or 1)
        return {
            "id": child_id,
            "node_kind": "table",
            "table_sequence": sequence,
            "cell_row": None,
            "cell_col": None,
            "is_table_header": None,
            "header_row_count": header_row_count,
            "original_content": "\n".join(text for _, text in md_node.content_lines),
            "normalized_content": "\n".join(text for _, text in md_node.content_lines).strip(),
            "children": build_table_cell_nodes(md_node, child_id, cell_state),
            "metadata": make_metadata(md_node, line_to_page),
        }

    aligned_nodes: dict[int, MdNode] = {}
    table_aligned_nodes: dict[int, list[MdNode]] = {}
    orphan_aligned_tables: dict[int, list[MdNode]] = {}
    table_line_owners: dict[int, int] = {}

    def build_tree_node(
        tree_node: TreeNode,
        parent_id: str,
        actual_last_page: int | None,
        is_last_child: bool,
        parent_state: dict[str, int] | None = None,
        md_node: MdNode | None = None,
    ) -> dict[str, Any]:
        if parent_state is None:
            parent_state = {}
        special_parts = parse_special_parts(tree_node.name)
        if special_parts is not None:
            base_id = f"{parent_id}.特殊_{special_parts['special_type']}_{special_parts['special_serial']}" if parent_id else f"特殊_{special_parts['special_type']}_{special_parts['special_serial']}"
            node_id = ensure_unique_child_id(parent_state, base_id)
        else:
            node_id = f"{parent_id}.{tree_node.sibling_order}" if parent_id else str(tree_node.sibling_order)

        if md_node is not None and special_parts is not None:
            original_content = "\n".join(text for _, text in flatten_md_subtree_lines(md_node))
            normalized_content = original_content.strip()
            flat_node = MdNode(kind="special", name=md_node.name, line_start=md_node.line_start)
            flat_node.content_lines = flatten_md_subtree_lines(md_node)
            flat_node.pages = set(md_node.pages)
            for child in md_node.children:
                flat_node.pages.update(child.pages)
            metadata = make_metadata(
                flat_node,
                line_to_page,
                page_end_override=actual_last_page if is_last_child and actual_last_page is not None else None,
            )
        else:
            original_content = "\n".join(text for _, text in (md_node.content_lines if md_node else []))
            normalized_content = original_content.strip()
            metadata = make_metadata(
                md_node if md_node is not None else MdNode(kind="article", name=tree_node.name, line_start=0),
                line_to_page,
                page_end_override=actual_last_page if is_last_child and actual_last_page is not None and md_node is not None else None,
            )

        child_state: dict[str, int] = {}
        children: list[dict[str, Any]] = []
        table_source_nodes: list[MdNode] = []
        if special_parts is None:
            table_source_nodes = list(table_aligned_nodes.get(id(tree_node), []))
            if md_node is not None and all(source.line_start != md_node.line_start for source in table_source_nodes):
                table_source_nodes.insert(0, md_node)
        if table_source_nodes or orphan_aligned_tables.get(id(tree_node)):
            table_children_by_line: dict[int, MdNode] = {}
            for table_source_node in table_source_nodes:
                for table_node in collect_inline_table_nodes(table_source_node, tree_node):
                    owner = table_line_owners.get(table_node.line_start)
                    if owner is not None and owner != id(tree_node):
                        continue
                    table_children_by_line.setdefault(table_node.line_start, table_node)
            for table_node in orphan_aligned_tables.get(id(tree_node), []):
                table_children_by_line.setdefault(table_node.line_start, table_node)
            table_children = [table_children_by_line[line_start] for line_start in sorted(table_children_by_line)]
            for table_child in table_children:
                children.append(build_table_node(table_child, node_id, child_state))

        for index, child_tree_node in enumerate(tree_node.children):
            children.append(
                build_tree_node(
                    child_tree_node,
                    node_id,
                    actual_last_page,
                    is_last_child and index == len(tree_node.children) - 1,
                    child_state,
                    aligned_nodes.get(id(child_tree_node)),
                )
            )

        return {
            "id": node_id,
            "node_kind": "special" if special_parts is not None else "normal",
            "table_sequence": None,
            "cell_row": None,
            "cell_col": None,
            "original_content": original_content,
            "normalized_content": normalized_content,
            "children": children,
            "metadata": metadata,
        }

    def build_document_tree(file_stem: str, tree_root: TreeNode, md_root: MdNode, actual_last_page: int | None) -> dict[str, Any]:
        nonlocal aligned_nodes, table_aligned_nodes, orphan_aligned_tables, table_line_owners
        aligned_nodes = align_tree_and_md(tree_root, md_root)
        table_aligned_nodes = build_table_alignments(tree_root, md_root)
        orphan_aligned_tables = build_orphan_table_alignments(tree_root, md_root, aligned_nodes, table_aligned_nodes)
        table_line_owners = build_table_line_owners(tree_root, table_aligned_nodes, orphan_aligned_tables)
        root_state: dict[str, int] = {}
        root_children: list[dict[str, Any]] = []
        for index, child_tree_node in enumerate(tree_root.children):
            root_children.append(
                build_tree_node(
                    child_tree_node,
                    "",
                    actual_last_page,
                    index == len(tree_root.children) - 1,
                    root_state,
                    aligned_nodes.get(id(child_tree_node)),
                )
            )
        root_content_lines = build_root_content_lines(tree_root, md_root)
        root_node = MdNode(kind="root", name="__root__", line_start=0)
        root_node.content_lines = root_content_lines
        root_node.pages = set(md_root.pages)

        return {
            "id": "",
            "node_kind": "root",
            "table_sequence": None,
            "cell_row": None,
            "cell_col": None,
            "original_content": "\n".join(text for _, text in root_content_lines),
            "normalized_content": "\n".join(text for _, text in root_content_lines).strip(),
            "children": root_children,
            "metadata": make_metadata(root_node, line_to_page, page_end_override=actual_last_page),
        }

    def add_file_identity(node: dict[str, Any], file_name: str, file_stem: str) -> None:
        local_id = str(node.get("id", ""))
        node["file_name"] = file_name
        node["file_stem"] = file_stem
        node["id"] = file_stem if not local_id else f"{file_stem}.{local_id}"
        for child in node.get("children", []):
            add_file_identity(child, file_name, file_stem)

    ensure_base_dirs()

    output_dir = JSON_ROOT / "single_tree_per_file"
    sqlite_output_path = JSON_ROOT / "all_single_tree.db"
    all_trees: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []

    for md_path in collect_md_files():
        file_name = md_path.name
        file_stem = md_path.stem
        tree_path = tree_dir / file_name

        if not tree_path.exists():
            file_summaries.append(
                {
                    "file_name": file_name,
                    "status": "missing_tree",
                    "output_path": None,
                }
            )
            continue

        tree_root = parse_tree_file(tree_path)
        md_root, actual_last_page, line_to_page = parse_md_file(md_path)
        document_tree = build_document_tree(file_stem, tree_root, md_root, actual_last_page)
        add_file_identity(document_tree, file_name, file_stem)

        output_path = output_dir / f"{file_stem}.json"
        write_json(output_path, document_tree)
        all_trees.append(document_tree)
        file_summaries.append(
            {
                "file_name": file_name,
                "status": "ok",
                "output_path": str(output_path),
            }
        )

    write_json(SUMMARY_DIR / "all_single_tree.json", all_trees)
    write_json(SUMMARY_DIR / "file_summary.json", file_summaries)
    actual_sqlite_output_path = write_sqlite(sqlite_output_path, all_trees)

    print(
        json.dumps(
            {
                "tree_dir": str(tree_dir),
                "md_dir": str(md_dir),
                "file_count": len([row for row in file_summaries if row.get("status") == "ok"]),
                "output_dir": str(output_dir),
                "sqlite_output_path": str(actual_sqlite_output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-tree structured JSON from existing tree files and merged markdown.")
    parser.add_argument("--file-name", action="append", help="Only process the given file name. Can be passed multiple times.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files after sorting.")
    args = parser.parse_args()

    build_structured_json_from_tree_special(
        tree_dir=TREE_DIR,
        md_dir=MD_DIR,
        file_names=args.file_name,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
