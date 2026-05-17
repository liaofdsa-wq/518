from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
JSON_ROOT = DATA_ROOT / "json"
ROOT = PROJECT_ROOT


def resolve_source_root() -> Path:
    return ROOT.parents[1] / "07_CleanTree精修區" / "法規資料_md_clean"


SOURCE_ROOT = resolve_source_root()
TREE_DIR = SOURCE_ROOT / "tree"
MD_DIR = SOURCE_ROOT / "法規資料_md"
TABLES_DIR = SOURCE_ROOT / "tables"

ALL_NODE_JSON_DIR = JSON_ROOT / "all_nodes_per_file"
LEAF_JSON_DIR = JSON_ROOT / "leaf_nodes_per_file"
TABLE_CELL_JSON_DIR = JSON_ROOT / "table_cells_per_file"
TABLE_CHUNK_JSON_DIR = JSON_ROOT / "table_chunks_per_file"
SUMMARY_DIR = DATA_ROOT / "summary"

PAGE_RE = re.compile(r"^\[page\s+(\d+)\]$")
OPEN_TAG_RE = re.compile(r"^\[(.+?)\]$")
CLOSE_TAG_RE = re.compile(r"^\[/([^\]]+)\]$")
TABLE_RE = re.compile(r"\[TABLE_REMOVED id=([^\]]+)\]")


@dataclass
class TreeNode:
    name: str
    level: int
    sibling_order: int
    path_names: list[str]
    path_orders: list[int]
    line_number: int
    parent: "TreeNode | None" = None
    children: list["TreeNode"] = field(default_factory=list)
    content: str = ""
    pages: list[int] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    table_anchors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def path_text(self) -> str:
        return " > ".join(self.path_names)

    @property
    def path_key(self) -> str:
        return ".".join(str(order) for order in self.path_orders)


@dataclass
class MdNode:
    name: str
    parent: "MdNode | None" = None
    children: list["MdNode"] = field(default_factory=list)
    content_lines: list[str] = field(default_factory=list)
    pages: set[int] = field(default_factory=set)
    table_anchors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n".join(line for line in self.content_lines).strip()


def ensure_dirs() -> None:
    for path in (
        DATA_ROOT,
        JSON_ROOT,
        ALL_NODE_JSON_DIR,
        LEAF_JSON_DIR,
        TABLE_CELL_JSON_DIR,
        TABLE_CHUNK_JSON_DIR,
        SUMMARY_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def parse_tree_file(path: Path) -> TreeNode:
    root = TreeNode(name="__root__", level=-1, sibling_order=0, path_names=[], path_orders=[], line_number=0)
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
            path_names=parent.path_names + [match.group("name")],
            path_orders=parent.path_orders + [sibling_order],
            line_number=line_number,
            parent=parent,
        )
        parent.children.append(node)
        stack.append(node)

    return root


def parse_md_file(path: Path) -> MdNode:
    root = MdNode(name="__root__")
    stack: list[MdNode] = [root]
    current_page: int | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        page_match = PAGE_RE.match(raw_line)
        if page_match:
            current_page = int(page_match.group(1))
            for node in stack[1:]:
                node.pages.add(current_page)
            continue

        close_match = CLOSE_TAG_RE.match(raw_line)
        if close_match:
            closing_name = close_match.group(1)
            if len(stack) > 1 and stack[-1].name == closing_name:
                stack.pop()
            else:
                for idx in range(len(stack) - 1, 0, -1):
                    if stack[idx].name == closing_name:
                        stack = stack[:idx]
                        break
            continue

        open_match = OPEN_TAG_RE.match(raw_line)
        if open_match and not TABLE_RE.search(raw_line):
            node = MdNode(name=open_match.group(1), parent=stack[-1])
            if current_page is not None:
                node.pages.add(current_page)
            stack[-1].children.append(node)
            stack.append(node)
            continue

        table_ids = TABLE_RE.findall(raw_line)
        if table_ids and len(stack) > 1:
            current = stack[-1]
            for table_id in table_ids:
                current.table_anchors.append(
                    {
                        "table_id": table_id,
                        "page": current_page,
                    }
                )
            cleaned = TABLE_RE.sub("", raw_line).strip()
            if cleaned:
                current.content_lines.append(cleaned)
            continue

        if len(stack) > 1:
            current = stack[-1]
            if current_page is not None:
                current.pages.add(current_page)
            current.content_lines.append(raw_line)

    return root


def align_tree_and_md(tree_root: TreeNode, md_root: MdNode) -> None:
    # The markdown tags are often flat even when the legal hierarchy is nested.
    # Treat md as an ordered stream of tagged content blocks, and let tree define
    # the actual parent/child structure.
    md_nodes = iter_md_nodes(md_root)
    md_index = 0

    for tree_node in iter_tree_nodes(tree_root):
        matched, md_index = find_matching_md_node(tree_node, md_nodes, md_index)
        if matched is None:
            continue

        tree_node.content = matched.content
        tree_node.pages = sorted(matched.pages)
        tree_node.page_start = tree_node.pages[0] if tree_node.pages else None
        tree_node.page_end = tree_node.pages[-1] if tree_node.pages else None
        tree_node.table_anchors = matched.table_anchors


def iter_md_nodes(node: MdNode) -> list[MdNode]:
    rows: list[MdNode] = []
    for child in node.children:
        rows.append(child)
        rows.extend(iter_md_nodes(child))
    return rows


def find_matching_md_node(tree_node: TreeNode, md_nodes: list[MdNode], start_index: int) -> tuple[MdNode | None, int]:
    for idx in range(start_index, len(md_nodes)):
        candidate = md_nodes[idx]
        if candidate.name == tree_node.name:
            return candidate, idx + 1
    return None, start_index


def iter_tree_nodes(node: TreeNode) -> list[TreeNode]:
    rows: list[TreeNode] = []
    for child in node.children:
        rows.append(child)
        rows.extend(iter_tree_nodes(child))
    return rows


def find_node_for_table_page(tree_nodes: list[TreeNode], page: int | None) -> TreeNode | None:
    if page is None:
        return None

    page_matches = [
        node
        for node in tree_nodes
        if node.pages and min(node.pages) <= page <= max(node.pages)
    ]
    if page_matches:
        page_matches.sort(key=lambda node: (node.level, len(node.path_orders), node.line_number))
        return page_matches[-1]

    previous_nodes = [
        node
        for node in tree_nodes
        if node.pages and max(node.pages) <= page
    ]
    if previous_nodes:
        previous_nodes.sort(key=lambda node: (max(node.pages), node.level, len(node.path_orders), node.line_number))
        return previous_nodes[-1]

    return None


def node_to_row(file_name: str, node: TreeNode) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "node_name": node.name,
        "sibling_order": node.sibling_order,
        "path_key": node.path_key,
        "path_names": node.path_names,
        "pages": node.pages,
        "content": node.content,
    }


def parse_table_markdown(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[list[str]] = []
    for line in lines:
        if "|" not in line:
            continue
        trimmed = line.strip()
        if re.match(r"^\|?[-: ]+(\|[-: ]+)+\|?$", trimmed):
            continue
        cells = [cell.strip() for cell in trimmed.strip("|").split("|")]
        rows.append(cells)

    headers = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    return {
        "headers": headers,
        "rows": data_rows,
        "raw_markdown": "\n".join(lines),
    }


def split_long_text(text: str, max_chars: int = 120) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    parts = re.split(r"(?<=[。；：!?])\s*", cleaned)
    chunks: list[str] = []
    buffer = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        subparts = re.split(r"(?<=[，、])\s*", part) if len(part) > max_chars else [part]
        for subpart in subparts:
            subpart = subpart.strip()
            if not subpart:
                continue
            candidate = subpart if not buffer else f"{buffer} {subpart}"
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = subpart

    if buffer:
        chunks.append(buffer)

    return chunks or [cleaned]


def build_table_cell_rows(file_name: str, file_stem: str, tree_root: TreeNode, md_root: MdNode) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    table_folder = TABLES_DIR / file_stem
    if not table_folder.exists():
        return rows

    tree_nodes = iter_tree_nodes(tree_root)
    anchor_index: dict[str, TreeNode] = {}
    for node in tree_nodes:
        for anchor in node.table_anchors:
            table_id = anchor.get("table_id")
            if table_id and table_id not in anchor_index:
                anchor_index[table_id] = node

    anchor_page_index: dict[str, int | None] = {}
    for md_node in iter_md_nodes(md_root):
        for anchor in md_node.table_anchors:
            table_id = anchor.get("table_id")
            if table_id and table_id not in anchor_page_index:
                anchor_page_index[table_id] = anchor.get("page")

    for table_path in sorted(table_folder.glob("*.md")):
        table_id = table_path.stem
        anchor_node = anchor_index.get(table_id)
        if anchor_node is None:
            anchor_node = find_node_for_table_page(tree_nodes, anchor_page_index.get(table_id))
        table_data = parse_table_markdown(table_path)
        data_rows = table_data["rows"]

        for row_idx, row in enumerate(data_rows, start=1):
            for col_idx, cell_text in enumerate(row, start=1):
                rows.append(
                    {
                        "file_name": file_name,
                        "table_id": table_id,
                        "under_path_key": anchor_node.path_key if anchor_node else "",
                        "under_path_names": anchor_node.path_names if anchor_node else [],
                        "pages": anchor_node.pages if anchor_node else [],
                        "row_index": row_idx,
                        "col_index": col_idx,
                        "cell_text": cell_text,
                    }
                )
    return rows


def build_table_chunk_rows(table_cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in table_cell_rows:
        original_cell_text = str(cell.get("cell_text", "")).strip()
        if not original_cell_text:
            continue

        for chunk_index, chunk_text in enumerate(split_long_text(original_cell_text), start=1):
            rows.append(
                {
                    "file_name": cell.get("file_name", ""),
                    "table_id": cell.get("table_id", ""),
                    "under_path_key": cell.get("under_path_key", ""),
                    "pages": cell.get("pages", []),
                    "row_index": cell.get("row_index", 0),
                    "col_index": cell.get("col_index", 0),
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "original_cell_text": original_cell_text,
                }
            )
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    all_node_rows: list[dict[str, Any]] = []
    all_leaf_rows: list[dict[str, Any]] = []
    all_table_cell_rows: list[dict[str, Any]] = []
    all_table_chunk_rows: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []

    tree_files = sorted(TREE_DIR.glob("*.md"))
    for tree_path in tree_files:
        file_name = tree_path.name
        file_stem = tree_path.stem
        md_path = MD_DIR / file_name
        if not md_path.exists():
            continue

        tree_root = parse_tree_file(tree_path)
        md_root = parse_md_file(md_path)
        align_tree_and_md(tree_root, md_root)

        tree_nodes = iter_tree_nodes(tree_root)
        node_rows = [node_to_row(file_name, node) for node in tree_nodes]
        leaf_rows = [row for row, node in zip(node_rows, tree_nodes) if node.is_leaf]
        table_cell_rows = build_table_cell_rows(file_name, file_stem, tree_root, md_root)
        table_chunk_rows = build_table_chunk_rows(table_cell_rows)

        write_json(ALL_NODE_JSON_DIR / f"{file_stem}.json", node_rows)
        write_json(LEAF_JSON_DIR / f"{file_stem}.json", leaf_rows)
        write_json(TABLE_CELL_JSON_DIR / f"{file_stem}.json", table_cell_rows)
        write_json(TABLE_CHUNK_JSON_DIR / f"{file_stem}.json", table_chunk_rows)

        all_node_rows.extend(node_rows)
        all_leaf_rows.extend(leaf_rows)
        all_table_cell_rows.extend(table_cell_rows)
        all_table_chunk_rows.extend(table_chunk_rows)
        file_summaries.append(
            {
                "file_name": file_name,
                "node_count": len(node_rows),
                "leaf_count": len(leaf_rows),
                "table_cell_count": len(table_cell_rows),
                "table_chunk_count": len(table_chunk_rows),
                "all_node_json": str(ALL_NODE_JSON_DIR / f"{file_stem}.json"),
                "leaf_json": str(LEAF_JSON_DIR / f"{file_stem}.json"),
                "table_cell_json": str(TABLE_CELL_JSON_DIR / f"{file_stem}.json"),
                "table_chunk_json": str(TABLE_CHUNK_JSON_DIR / f"{file_stem}.json"),
            }
        )

    write_json(SUMMARY_DIR / "all_nodes.json", all_node_rows)
    write_json(SUMMARY_DIR / "all_leaf_nodes.json", all_leaf_rows)
    write_json(SUMMARY_DIR / "all_table_cells.json", all_table_cell_rows)
    write_json(SUMMARY_DIR / "all_table_chunks.json", all_table_chunk_rows)
    write_json(SUMMARY_DIR / "file_summary.json", file_summaries)


    print(
        json.dumps(
            {
                "output_root": str(DATA_ROOT),
                "file_count": len(file_summaries),
                "node_count": len(all_node_rows),
                "leaf_count": len(all_leaf_rows),
                "table_cell_count": len(all_table_cell_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
