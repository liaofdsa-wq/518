from __future__ import annotations

import argparse
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
EXPERIMENT_ROOT = DATA_ROOT / "experiments" / "06_table_inner_hierarchy"

TABLE_SPLIT_ROOT = PROJECT_ROOT.parents[1]
MD_DIR = TABLE_SPLIT_ROOT / "07.6連續表格合併"

DEFAULT_FILE_NAME = "2.FCM內控CA---壹、業務及收入循環 ---114.04.md"

TABLE_START_RE = re.compile(
    r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$"
)
TABLE_END_RE = re.compile(r"^\[/TABLE id=([^\]]+)\]\s*$")


def pattern_defs() -> list[tuple[str, str]]:
    return [
        ("CHAPTER", r"第\s*([一二三四五六七八九十百0-9]+)\s*章"),
        ("SECTION", r"第\s*([一二三四五六七八九十百0-9]+)\s*節"),
        ("ARTICLE", r"第\s*([一二三四五六七八九十百0-9]+)\s*條"),
        ("POINT_ZH_MAIN", r"第\s*([一二三四五六七八九十百0-9]+)\s*點"),
        ("PARAGRAPH", r"第\s*([一二三四五六七八九十百0-9]+)\s*項"),
        ("SUBITEM", r"第\s*([一二三四五六七八九十百0-9]+)\s*款"),
        ("ITEM_ZH_BIG", r"([壹貳參肆伍陸柒捌玖拾]+)[、．.]"),
        ("PAREN_ZH_BIG", r"[（(]([壹貳參肆伍陸柒捌玖拾]+)[）)]"),
        ("ITEM_ZH", r"([一二三四五六七八九十]+)[、．.]"),
        ("PAREN_ZH", r"[（(]([一二三四五六七八九十]+)[）)]"),
        ("POINT_NUM", r"(\d+)[\.．、]"),
        ("PAREN_NUM", r"[（(](\d+)[）)]"),
        ("ITEM_TIAN_GAN", r"([甲乙丙丁戊己庚辛壬癸]+)[、．.]"),
        ("ROMAN_SMALL", r"([ivx]+)[\.．、]"),
        ("ALPHA_LOWER", r"([a-z])[.．]"),
        ("PAREN_ALPHA", r"[（(]([a-z])[）)]"),
    ]


def inline_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns = []
    for pattern_name, pattern_body in pattern_defs():
        patterns.append(
            (
                pattern_name,
                re.compile(
                    rf"(?:(?<=^)|(?<=[\s「『（(]))(?P<full>{pattern_body})",
                    re.IGNORECASE,
                ),
            )
        )
    return patterns


def pattern_priority() -> dict[str, int]:
    return {name: index for index, (name, _) in enumerate(pattern_defs())}


def zh_num_map() -> dict[str, int]:
    return {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "百": 100,
    }


def zh_big_map() -> dict[str, int]:
    return {
        "壹": 1,
        "貳": 2,
        "參": 3,
        "肆": 4,
        "伍": 5,
        "陸": 6,
        "柒": 7,
        "捌": 8,
        "玖": 9,
        "拾": 10,
    }


def tian_gan_order() -> dict[str, int]:
    return {value: index for index, value in enumerate("甲乙丙丁戊己庚辛壬癸", start=1)}


def roman_order() -> dict[str, int]:
    return {"i": 1, "v": 5, "x": 10}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_markdown_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r"[-: ]+", cell or "") for cell in cells)


def zh_to_int(token: str, mapping: dict[str, int]) -> int | None:
    token = token.strip()
    if not token:
        return None
    if token in mapping and mapping[token] < 10:
        return mapping[token]
    if token.isdigit():
        return int(token)
    total = 0
    current = 0
    for char in token:
        value = mapping.get(char)
        if value is None:
            return None
        if value >= 10:
            current = max(current, 1) * value
            total += current
            current = 0
        else:
            current += value
    return total + current


def roman_to_int(token: str) -> int | None:
    token = token.lower().strip()
    if not token:
        return None
    total = 0
    previous = 0
    for char in reversed(token):
        value = roman_order().get(char)
        if value is None:
            return None
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total


def token_to_order(pattern_name: str, token: str) -> int | None:
    if pattern_name in {"CHAPTER", "SECTION", "ARTICLE", "POINT_ZH_MAIN", "PARAGRAPH", "SUBITEM"}:
        return zh_to_int(token, zh_num_map())
    if pattern_name in {"ITEM_ZH_BIG", "PAREN_ZH_BIG"}:
        return zh_to_int(token, zh_big_map())
    if pattern_name in {"ITEM_ZH", "PAREN_ZH"}:
        return zh_to_int(token, zh_num_map())
    if pattern_name in {"POINT_NUM", "PAREN_NUM"}:
        return int(token)
    if pattern_name == "ITEM_TIAN_GAN":
        return tian_gan_order().get(token)
    if pattern_name == "ROMAN_SMALL":
        return roman_to_int(token)
    if pattern_name in {"ALPHA_LOWER", "PAREN_ALPHA"}:
        return ord(token.lower()) - ord("a") + 1
    return None


def find_pattern_matches(text: str) -> dict[str, list[dict]]:
    matches_by_type = {pattern_name: [] for pattern_name, _ in inline_patterns()}
    for pattern_name, pattern in inline_patterns():
        for match in pattern.finditer(text):
            token = next((group for group in match.groups()[1:] if group), "")
            order = token_to_order(pattern_name, token)
            if order is None:
                continue
            matches_by_type[pattern_name].append(
                {
                    "pattern_name": pattern_name,
                    "token": token,
                    "order": order,
                    "start": match.start("full"),
                    "end": match.end("full"),
                    "full": match.group("full"),
                }
            )
    return matches_by_type


def longest_consecutive_run(matches: list[dict]) -> list[dict]:
    if len(matches) < 2:
        return []
    ordered = sorted(matches, key=lambda item: item["start"])
    best_run: list[dict] = []
    current_run = [ordered[0]]
    for match in ordered[1:]:
        previous = current_run[-1]
        if match["order"] == previous["order"] + 1:
            current_run.append(match)
        elif match["order"] != previous["order"]:
            if len(current_run) > len(best_run):
                best_run = list(current_run)
            current_run = [match]
    if len(current_run) > len(best_run):
        best_run = current_run
    return best_run if len(best_run) >= 2 else []


def choose_split_run(text: str):
    matches_by_type = find_pattern_matches(text)
    candidates = []
    for pattern_name, matches in matches_by_type.items():
        run = longest_consecutive_run(matches)
        if len(run) < 2:
            continue
        candidates.append((run[0]["start"], pattern_priority()[pattern_name], pattern_name, run))
    if not candidates:
        return None
    _, _, pattern_name, run = min(candidates, key=lambda item: (item[0], item[1]))
    return pattern_name, run


def split_by_run(text: str, run: list[dict]) -> tuple[str, list[dict]]:
    lead_text = normalize_text(text[: run[0]["start"]])
    parts = []
    for index, marker in enumerate(run):
        next_start = run[index + 1]["start"] if index + 1 < len(run) else len(text)
        segment_text = normalize_text(text[marker["start"] : next_start])
        body_text = normalize_text(text[marker["end"] : next_start])
        parts.append(
            {
                "pattern_name": marker["pattern_name"],
                "marker_text": normalize_text(marker["full"]),
                "marker_order": marker["order"],
                "body_text": body_text,
                "segment_text": segment_text,
            }
        )
    return lead_text, parts


def build_hierarchy_tree(text: str, depth: int = 0, max_depth: int = 8) -> dict:
    normalized = normalize_text(text)
    if not normalized:
        return {"text": "", "children": []}
    if depth >= max_depth:
        return {"text": normalized, "children": []}
    split_result = choose_split_run(normalized)
    if split_result is None:
        return {"text": normalized, "children": []}
    _, run = split_result
    lead_text, parts = split_by_run(normalized, run)
    children = []
    for part in parts:
        child_tree = build_hierarchy_tree(part["body_text"], depth + 1, max_depth)
        children.append(
            {
                "pattern_name": part["pattern_name"],
                "marker_text": part["marker_text"],
                "marker_order": part["marker_order"],
                "text": child_tree["text"],
                "children": child_tree["children"],
            }
        )
    return {"text": lead_text, "children": children}


def render_hierarchy_tree(tree: dict, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    lines: list[str] = []
    if tree.get("text"):
        lines.append(f"{prefix}- text: {tree['text']}")
    for child in tree.get("children", []):
        lines.append(f"{prefix}- {child['pattern_name']} {child['marker_text']}: {child.get('text', '')}".rstrip())
        lines.extend(render_hierarchy_tree({"text": "", "children": child.get("children", [])}, indent + 1))
    return lines


def load_tables(md_path: Path) -> list[dict]:
    lines = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tables: list[dict] = []
    current_table: dict | None = None

    for line in lines:
        start_match = TABLE_START_RE.match(line.strip())
        if start_match:
            current_table = {
                "table_id": start_match.group("table_id"),
                "rows": [],
            }
            continue

        if current_table is not None and TABLE_END_RE.match(line.strip()):
            tables.append(current_table)
            current_table = None
            continue

        if current_table is None:
            continue

        cells = parse_markdown_row(line)
        if cells and not is_separator_row(cells):
            current_table["rows"].append(cells)

    return tables


def build_preview_lines(file_name: str, tables: list[dict]) -> list[str]:
    preview_lines = [f"# {file_name}", ""]
    for table in tables:
        rows = table["rows"]
        if not rows:
            continue
        headers = rows[0]
        data_rows = rows[1:]
        preview_lines.append(f"## {table['table_id']}")
        preview_lines.append("")
        preview_lines.append(f"欄位: {' | '.join(headers)}")
        preview_lines.append("")
        for row_index, row in enumerate(data_rows, start=1):
            preview_lines.append(f"### row {row_index}")
            preview_lines.append("")
            for col_index, cell in enumerate(row, start=1):
                cell_text = normalize_text(cell)
                if not cell_text:
                    continue
                hierarchy = build_hierarchy_tree(cell_text)
                if not hierarchy.get("children"):
                    continue
                header = headers[col_index - 1] if col_index - 1 < len(headers) else f"col_{col_index}"
                preview_lines.append(f"- col {col_index} / {header}")
                preview_lines.append(f"  原文: {cell_text}")
                preview_lines.append("  tree:")
                preview_lines.extend(render_hierarchy_tree(hierarchy, indent=2))
                preview_lines.append("")
    return preview_lines


def build_preview(file_name: str, output_dir: Path) -> Path:
    md_path = MD_DIR / file_name
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    tables = load_tables(md_path)
    preview_lines = build_preview_lines(file_name, tables)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    output_path.write_text("\n".join(preview_lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Detect inner hierarchy inside merged markdown table cells.")
    parser.add_argument("--file-name", default=DEFAULT_FILE_NAME)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()

    output_path = build_preview(args.file_name, args.output_dir)
    print(f"[OK] preview written to {output_path}")


if __name__ == "__main__":
    main()
