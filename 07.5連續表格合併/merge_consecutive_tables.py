import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "06_CleanTree生成區" / "法規資料_md_clean" / "法規資料_md"
OUTPUT_DIR = BASE_DIR / "07.6連續表格合併"
LOG_FILENAME = "merge_consecutive_tables_log.csv"


@dataclass
class TableProfile:
    table_id: str
    cr: int
    ml: int | None
    start_idx: int
    end_idx: int
    start_line: str
    end_line: str
    raw_row_lines: list[str]
    parsed_rows: list[list[str]]


@dataclass
class MergeLogEntry:
    file_name: str
    base_table_id: str
    merged_table_id: str
    ml: int
    cr_used: int
    action: str
    status: str


def format_merged_tables_inner_hierarchy(output_dir: Path) -> None:
    target_file_names = {
        "1.FCM內控-總則---114.04.md",
        "2.FCM內控CA---壹、業務及收入循環 ---114.04.md",
        "3.FCM內控CP ---貳、採購及付款循環 ---114.04.md",
        "4.FCM內控CW---參 、薪工循環---114.04.md",
        "5.FCM內控CR---肆、融資循環---114.04.md",
        "6.FCM內控CF---伍、不動產及設備循環---114.04.md",
        "7.FCM內控CI---陸、投資循環---114.04.md",
        "8.FCM內控CC---柒、電腦作業與資訊提供---114.04.md",
        "9.FCM內控CM---捌、管理控制制度---114.04.md",
    }
    target_columns = {3}  # 欄位用 1-based set；例如 {3} 只處理第3欄，{2, 3, 5} 代表多欄，None 代表全部欄位
    pattern_defs = [
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
    inline_patterns = [
        (
            pattern_name,
            re.compile(
                rf"(?:(?<=^)|(?<=[\s「『（(]))(?P<full>{pattern_body})",
                re.IGNORECASE,
            ),
        )
        for pattern_name, pattern_body in pattern_defs
    ]
    pattern_priority = {name: index for index, (name, _) in enumerate(pattern_defs)}
    zh_num_map = {
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
    zh_big_map = {
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
    tian_gan_order = {value: index for index, value in enumerate("甲乙丙丁戊己庚辛壬癸", start=1)}
    roman_order = {"i": 1, "v": 5, "x": 10}

    def format_table_inner_hierarchy_text(
        cell_text: str,
        file_name: str | None = None,
        col_idx: int | None = None,
    ) -> str:
        if target_file_names and file_name is not None and file_name not in target_file_names:
            return cell_text
        if target_columns is not None and col_idx is not None and col_idx not in target_columns:
            return cell_text

        def normalize_text(text: str) -> str:
            return re.sub(r"\s+", " ", text or "").strip()

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
                value = roman_order.get(char)
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
                return zh_to_int(token, zh_num_map)
            if pattern_name in {"ITEM_ZH_BIG", "PAREN_ZH_BIG"}:
                return zh_to_int(token, zh_big_map)
            if pattern_name in {"ITEM_ZH", "PAREN_ZH"}:
                return zh_to_int(token, zh_num_map)
            if pattern_name in {"POINT_NUM", "PAREN_NUM"}:
                return int(token)
            if pattern_name == "ITEM_TIAN_GAN":
                return tian_gan_order.get(token)
            if pattern_name == "ROMAN_SMALL":
                return roman_to_int(token)
            if pattern_name in {"ALPHA_LOWER", "PAREN_ALPHA"}:
                return ord(token.lower()) - ord("a") + 1
            return None

        def find_pattern_matches(text: str) -> dict[str, list[dict]]:
            matches_by_type = {pattern_name: [] for pattern_name, _ in inline_patterns}
            for pattern_name, pattern in inline_patterns:
                for match in pattern.finditer(text):
                    token = next((group for group in match.groups()[1:] if group), "")
                    order = token_to_order(pattern_name, token)
                    if order is None:
                        continue
                    matches_by_type[pattern_name].append(
                        {
                            "pattern_name": pattern_name,
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
            best_run = []
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
                candidates.append((run[0]["start"], pattern_priority[pattern_name], run))
            if not candidates:
                return None
            _, _, run = min(candidates, key=lambda item: (item[0], item[1]))
            return run

        def build_hierarchy_tree(text: str, depth: int = 0, max_depth: int = 8) -> dict:
            normalized = normalize_text(text)
            if not normalized:
                return {"text": "", "children": []}
            if depth >= max_depth:
                return {"text": normalized, "children": []}
            run = choose_split_run(normalized)
            if run is None:
                return {"text": normalized, "children": []}
            lead_text = normalize_text(normalized[: run[0]["start"]])
            children = []
            for index, marker in enumerate(run):
                next_start = run[index + 1]["start"] if index + 1 < len(run) else len(normalized)
                body_text = normalize_text(normalized[marker["end"] : next_start])
                child_tree = build_hierarchy_tree(body_text, depth + 1, max_depth)
                children.append(
                    {
                        "label": normalize_text(marker["full"]),
                        "text": child_tree["text"],
                        "children": child_tree["children"],
                    }
                )
            return {"text": lead_text, "children": children}

        def render_tree(tree: dict, indent: int = 0) -> list[str]:
            prefix = f"{'-' * indent} " if indent > 0 else ""
            lines = []
            if tree.get("text"):
                lines.append(f"{prefix}{tree['text']}")
            for child in tree.get("children", []):
                child_prefix = f"{'-' * (indent + 1)} "
                first_line = f"{child_prefix}{child['label']}"
                if child.get("text"):
                    first_line += f" {child['text']}"
                lines.append(first_line.rstrip())
                lines.extend(render_tree({"text": "", "children": child.get("children", [])}, indent + 1))
            return lines

        normalized_text = normalize_text(cell_text)
        if not normalized_text:
            return ""
        tree = build_hierarchy_tree(normalized_text)
        if not tree.get("children"):
            return normalized_text
        return "<br>".join(render_tree(tree))

    def apply_table_inner_hierarchy(table: TableProfile, file_name: str | None = None) -> TableProfile:
        if not table.parsed_rows:
            return table

        transformed_rows = [list(row) for row in table.parsed_rows]
        for row_idx, row in enumerate(transformed_rows):
            if row_idx < table.cr:
                continue
            for col_idx, cell_text in enumerate(row):
                flattened_text = re.sub(r"\s+", " ", cell_text or "").strip()
                transformed_rows[row_idx][col_idx] = format_table_inner_hierarchy_text(
                    flattened_text,
                    file_name=file_name,
                    col_idx=col_idx + 1,
                )
        table.parsed_rows = transformed_rows
        return table

    def is_table_start_line(line: str):
        return re.match(r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$", line.strip())

    def is_table_end_line(line: str):
        return re.match(r"^\[/TABLE id=([^\]]+)\]\s*$", line.strip())

    def parse_markdown_row_line(line: str):
        stripped = line.strip()
        if "|" not in stripped:
            return None
        return [cell.strip() for cell in stripped.strip("|").split("|")]

    def is_separator_row_line(cells):
        if not cells:
            return False
        return all(re.fullmatch(r"[-: ]+", cell or "") for cell in cells)

    def row_to_markdown_line(cells: list[str], width: int):
        padded = list(cells) + [""] * max(0, width - len(cells))
        return "| " + " | ".join(padded[:width]) + " |"

    def build_separator_row_line(width: int):
        return "| " + " | ".join(["---"] * max(1, width)) + " |"

    def format_merged_file_inner_hierarchy(md_file: Path) -> None:
        lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        output_lines: list[str] = []
        idx = 0

        while idx < len(lines):
            start_match = is_table_start_line(lines[idx])
            if not start_match:
                output_lines.append(lines[idx])
                idx += 1
                continue

            table_id = start_match.group("table_id")
            cr = int(start_match.group("cr") or "1")
            start_line = lines[idx]
            idx += 1
            parsed_rows: list[list[str]] = []

            while idx < len(lines) and not is_table_end_line(lines[idx]):
                cells = parse_markdown_row_line(lines[idx])
                if cells and not is_separator_row_line(cells):
                    parsed_rows.append(cells)
                idx += 1

            end_line = lines[idx] if idx < len(lines) else f"[/TABLE id={table_id}]"
            table = TableProfile(
                table_id=table_id,
                cr=cr,
                ml=int(start_match.group("ml")) if start_match.group("ml") else None,
                start_idx=0,
                end_idx=0,
                start_line=start_line,
                end_line=end_line,
                raw_row_lines=[],
                parsed_rows=parsed_rows,
            )
            table = apply_table_inner_hierarchy(table, md_file.name)

            width = max((len(row) for row in table.parsed_rows), default=1)
            output_lines.append(table.start_line)
            if table.parsed_rows:
                output_lines.append(row_to_markdown_line(table.parsed_rows[0], width))
                output_lines.append(build_separator_row_line(width))
                for row in table.parsed_rows[1:]:
                    output_lines.append(row_to_markdown_line(row, width))
            output_lines.append(table.end_line)
            idx += 1

        md_file.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    for md_file in sorted(output_dir.rglob("*.md")):
        format_merged_file_inner_hierarchy(md_file)


def merge_consecutive_tables(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    # region 小工具
    def is_table_start(line: str):
        return re.match(
            r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$",
            line.strip(),
        )

    def is_table_end(line: str):
        return re.match(r"^\[/TABLE id=([^\]]+)\]\s*$", line.strip())

    def parse_markdown_row(line: str):
        stripped = line.strip()
        if "|" not in stripped:
            return None
        return [cell.strip() for cell in stripped.strip("|").split("|")]

    def is_separator_row(cells):
        if not cells:
            return False
        return all(re.fullmatch(r"[-: ]+", cell or "") for cell in cells)

    def get_merge_tag(cell_text: str):
        match = re.match(r"^\[@(\d+)\]", (cell_text or "").strip())
        if not match:
            return None
        return match.group(1)

    def strip_merge_tag(cell_text: str):
        return re.sub(r"^\[@\d+\]\s*", "", (cell_text or "").strip())

    def attach_merge_tag(cell_text: str, tag: str):
        stripped = (cell_text or "").strip()
        if not stripped:
            return f"[@{tag}]"
        if get_merge_tag(stripped) == tag:
            return stripped
        return f"[@{tag}] {stripped}"

    def normalize_cell(cell: str):
        return re.sub(r"\s+", " ", cell or "").strip()

    def join_cells(base: str, extra: str):
        base_text = (base or "").strip()
        extra_text = (extra or "").strip()
        if not extra_text:
            return base_text
        if not base_text:
            return extra_text
        if normalize_cell(base_text) == normalize_cell(extra_text):
            return base_text
        return f"{base_text} {extra_text}"

    def row_to_markdown(cells: list[str], width: int):
        padded = list(cells) + [""] * max(0, width - len(cells))
        return "| " + " | ".join(padded[:width]) + " |"

    def build_separator_row(width: int):
        return "| " + " | ".join(["---"] * max(1, width)) + " |"

    def ensure_width(row: list[str], width: int):
        if len(row) < width:
            row.extend([""] * (width - len(row)))
        return row

    def is_page_marker_row(row: list[str]):
        if not row:
            return False
        first_cell = (row[0] or "").strip()
        if not re.fullmatch(r"\[page\s+\d+\]", first_cell, re.IGNORECASE):
            return False
        return all((cell or "").strip() == "" for cell in row[1:])

    def find_last_content_row_index(rows: list[list[str]]):
        for idx in range(len(rows) - 1, -1, -1):
            if not is_page_marker_row(rows[idx]):
                return idx
        return None

    def cut_header_rows(rows: list[list[str]], cr: int):
        return rows[cr:]

    def parse_table_block(lines, start_idx):
        start_match = is_table_start(lines[start_idx])
        if not start_match:
            return None

        table_id = start_match.group("table_id")
        cursor = start_idx + 1
        raw_row_lines = []
        parsed_rows = []

        while cursor < len(lines):
            line = lines[cursor]
            if is_table_end(line):
                return TableProfile(
                    table_id=table_id,
                    cr=int(start_match.group("cr") or "1"),
                    ml=int(start_match.group("ml")) if start_match.group("ml") else None,
                    start_idx=start_idx,
                    end_idx=cursor,
                    start_line=lines[start_idx].rstrip("\n"),
                    end_line=line.rstrip("\n"),
                    raw_row_lines=raw_row_lines,
                    parsed_rows=parsed_rows,
                )

            raw_row_lines.append(line.rstrip("\n"))
            cells = parse_markdown_row(line)
            if cells and not is_separator_row(cells):
                parsed_rows.append(cells)
            cursor += 1

        return None

    def parse_blocks(lines):
        blocks = []
        idx = 0
        while idx < len(lines):
            table = parse_table_block(lines, idx)
            if table:
                blocks.append({"type": "table", "table": table})
                idx = table.end_idx + 1
                continue

            text_lines = []
            start_idx = idx
            while idx < len(lines) and not is_table_start(lines[idx]):
                text_lines.append(lines[idx].rstrip("\n"))
                idx += 1
            blocks.append(
                {
                    "type": "text",
                    "start_idx": start_idx,
                    "end_idx": idx - 1,
                    "lines": text_lines,
                }
            )
        return blocks

    def merge_row_into_last(base_rows: list[list[str]], incoming_row: list[str]):
        if not base_rows:
            base_rows.append(list(incoming_row))
            return

        target_idx = find_last_content_row_index(base_rows)
        if target_idx is None:
            base_rows.append(list(incoming_row))
            return

        target = base_rows[target_idx]
        width = max(len(target), len(incoming_row))
        ensure_width(target, width)
        incoming = list(incoming_row) + [""] * max(0, width - len(incoming_row))
        for col_idx, value in enumerate(incoming):
            target[col_idx] = join_cells(target[col_idx], value)

    def merge_first_row_then_append_rest(base_rows: list[list[str]], rows_to_use: list[list[str]]):
        if not rows_to_use:
            return "no_rows"
        merge_row_into_last(base_rows, rows_to_use[0])
        if len(rows_to_use) > 1:
            base_rows.extend([list(row) for row in rows_to_use[1:]])
            return "merge_first_row_then_append_rest"
        return "merge_first_row_only"

    def build_next_merge_tag(base_rows: list[list[str]], rows_to_use: list[list[str]]):
        next_tag = 1
        for row in base_rows + rows_to_use:
            for cell in row:
                tag = get_merge_tag(cell)
                if tag is not None:
                    next_tag = max(next_tag, int(tag) + 1)
        return next_tag

    def ensure_tagged_cell(row: list[str], col_idx: int, next_tag_ref: list[int]):
        current_value = row[col_idx]
        tag = get_merge_tag(current_value)
        if tag is None:
            tag = str(next_tag_ref[0])
            next_tag_ref[0] += 1
            row[col_idx] = attach_merge_tag(current_value, tag)
        return row[col_idx]

    def build_ml3_carry_sources(base_rows: list[list[str]], carry_cols: set[int]):
        carry_sources: dict[int, tuple[list[str], int]] = {}
        if not base_rows:
            return carry_sources

        target_idx = find_last_content_row_index(base_rows)
        if target_idx is None:
            return carry_sources
        target = base_rows[target_idx]
        for col_idx in carry_cols:
            if col_idx >= len(target):
                continue
            if normalize_cell(strip_merge_tag(target[col_idx])) == "":
                continue
            carry_sources[col_idx] = (target, col_idx)
        return carry_sources

    def render_table(table: TableProfile):
        parsed_rows = [list(row) for row in table.parsed_rows]
        width = max((len(row) for row in parsed_rows), default=1)
        lines = [table.start_line]
        if parsed_rows:
            lines.append(row_to_markdown(parsed_rows[0], width))
            lines.append(build_separator_row(width))
            for row in parsed_rows[1:]:
                lines.append(row_to_markdown(row, width))
        lines.append(table.end_line)
        return [line + "\n" for line in lines]

    # endregion

    # region 規則
    def handle_ml1(base_rows, lower_table):
        rows_to_use = cut_header_rows(lower_table.parsed_rows, lower_table.cr)
        return {
            "action": merge_first_row_then_append_rest(base_rows, rows_to_use),
            "cr_used": lower_table.cr,
        }

    def handle_ml2(base_rows, lower_table):
        return {
            "action": merge_first_row_then_append_rest(base_rows, lower_table.parsed_rows),
            "cr_used": 0,
        }

    def handle_ml3(base_rows, lower_table):
        rows_to_use = cut_header_rows(lower_table.parsed_rows, lower_table.cr)
        if not rows_to_use:
            return {"action": "no_rows", "cr_used": lower_table.cr}

        normalized_rows = [list(row) for row in rows_to_use]
        if not base_rows:
            base_rows.extend(normalized_rows)
            action = "append_all_rows_no_base"
            if len(normalized_rows) == 1:
                action = "append_first_row_no_base"
            return {"action": action, "cr_used": lower_table.cr}

        next_tag_ref = [build_next_merge_tag(base_rows, normalized_rows)]
        carry_cols = {0, 1, 2}
        required_carry_cols = {1, 2}
        carry_sources = build_ml3_carry_sources(base_rows, carry_cols)

        appended_any_row = False
        carried_any_cell = False
        extended_any_row = False

        for row_idx, source_row in enumerate(normalized_rows):
            working_row = list(source_row)
            width = len(working_row)
            last_content_idx = find_last_content_row_index(base_rows)
            if last_content_idx is not None:
                width = max(width, len(base_rows[last_content_idx]))
            if width:
                ensure_width(working_row, width)

            for col_idx in carry_cols:
                if col_idx >= len(working_row):
                    continue
                current_value = working_row[col_idx]
                if normalize_cell(strip_merge_tag(current_value)) != "":
                    carry_sources[col_idx] = (working_row, col_idx)
                    continue

                source_ref = carry_sources.get(col_idx)
                if source_ref is None:
                    continue

                source_row_ref, source_col_idx = source_ref
                if normalize_cell(strip_merge_tag(source_row_ref[source_col_idx])) == "":
                    continue

                tagged_value = ensure_tagged_cell(source_row_ref, source_col_idx, next_tag_ref)
                working_row[col_idx] = tagged_value
                carry_sources[col_idx] = (working_row, col_idx)
                carried_any_cell = True

            for col_idx in required_carry_cols:
                if col_idx >= len(working_row):
                    continue
                if normalize_cell(strip_merge_tag(working_row[col_idx])) != "":
                    continue
                source_ref = carry_sources.get(col_idx)
                if source_ref is None:
                    continue
                source_row_ref, source_col_idx = source_ref
                if normalize_cell(strip_merge_tag(source_row_ref[source_col_idx])) == "":
                    continue
                tagged_value = ensure_tagged_cell(source_row_ref, source_col_idx, next_tag_ref)
                working_row[col_idx] = tagged_value
                carry_sources[col_idx] = (working_row, col_idx)
                carried_any_cell = True

            base_rows.append(working_row)
            appended_any_row = True

            if row_idx > 0:
                extended_any_row = True

            for col_idx in carry_cols:
                if col_idx >= len(working_row):
                    continue
                if normalize_cell(strip_merge_tag(working_row[col_idx])) == "":
                    continue
                carry_sources[col_idx] = (working_row, col_idx)

        action = "append_rows"
        if carried_any_cell and extended_any_row:
            action = "carry_forward_then_append_rows"
        elif carried_any_cell:
            action = "carry_forward_rows"
        elif extended_any_row:
            action = "append_rows_after_first"

        if not appended_any_row:
            action = "no_rows"

        return {"action": action, "cr_used": lower_table.cr}

    def handle_ml4(base_rows, lower_table):
        rows_to_use = cut_header_rows(lower_table.parsed_rows, lower_table.cr)
        return {
            "action": merge_first_row_then_append_rest(base_rows, rows_to_use),
            "cr_used": lower_table.cr,
        }

    def handle_ml5(base_rows, lower_table):
        return {
            "action": merge_first_row_then_append_rest(base_rows, lower_table.parsed_rows),
            "cr_used": 0,
        }

    def build_ml_decision(base_rows, lower_table):
        if lower_table.ml == 1:
            return handle_ml1(base_rows, lower_table)
        if lower_table.ml == 2:
            return handle_ml2(base_rows, lower_table)
        if lower_table.ml == 3:
            return handle_ml3(base_rows, lower_table)
        if lower_table.ml == 4:
            return handle_ml4(base_rows, lower_table)
        if lower_table.ml == 5:
            return handle_ml5(base_rows, lower_table)
        return None
    # endregion

    # region 主流程
    def merge_file(md_file: Path, input_root: Path, output_root: Path):
        def extract_trailing_page_lines(text_lines: list[str]) -> list[str]:
            page_lines: list[str] = []
            for line in text_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if not re.fullmatch(r"\[page\s+\d+\]", stripped, re.IGNORECASE):
                    return []
                page_lines.append(stripped)
            return page_lines

        lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        blocks = parse_blocks(lines)

        output_blocks = []
        pending_text_lines: list[str] = []
        logs: list[MergeLogEntry] = []
        trailing_page_lines_after_merge: list[str] = []

        for block_idx, block in enumerate(blocks):
            if block["type"] == "text":
                pending_text_lines.extend(block["lines"])
                continue

            table: TableProfile = block["table"]
            can_merge = (
                table.ml is not None
                and output_blocks
                and output_blocks[-1]["type"] == "table"
            )

            if can_merge:
                base_table: TableProfile = output_blocks[-1]["table"]
                decision = build_ml_decision(base_table.parsed_rows, table)
                if decision is not None:
                    action = decision["action"]
                    if pending_text_lines:
                        action += "_drop_intervening_text"
                        if block_idx == len(blocks) - 1:
                            trailing_page_lines_after_merge = extract_trailing_page_lines(pending_text_lines)
                    logs.append(
                        MergeLogEntry(
                            file_name=md_file.name,
                            base_table_id=base_table.table_id,
                            merged_table_id=table.table_id,
                            ml=table.ml,
                            cr_used=decision["cr_used"],
                            action=action,
                            status="merged",
                        )
                    )
                    pending_text_lines = []
                    continue

            if pending_text_lines:
                output_blocks.append({"type": "text", "lines": list(pending_text_lines)})
                pending_text_lines = []
            output_blocks.append({"type": "table", "table": table})

        if pending_text_lines:
            output_blocks.append({"type": "text", "lines": list(pending_text_lines)})
        elif trailing_page_lines_after_merge:
            output_blocks.append({"type": "text", "lines": list(trailing_page_lines_after_merge)})

        output_lines: list[str] = []
        for block in output_blocks:
            if block["type"] == "text":
                output_lines.extend([line + "\n" for line in block["lines"]])
            else:
                output_lines.extend(render_table(block["table"]))

        relative_path = md_file.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(output_lines), encoding="utf-8")
        return logs

    def write_merge_log(log_entries: list[MergeLogEntry], output_root: Path):
        log_path = output_root / LOG_FILENAME
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "file_name",
                    "base_table_id",
                    "merged_table_id",
                    "ml",
                    "cr_used",
                    "action",
                    "status",
                ]
            )
            for entry in log_entries:
                writer.writerow(
                    [
                        entry.file_name,
                        entry.base_table_id,
                        entry.merged_table_id,
                        entry.ml,
                        entry.cr_used,
                        entry.action,
                        entry.status,
                    ]
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    md_files = sorted(input_dir.rglob("*.md"))
    all_logs: list[MergeLogEntry] = []

    for md_file in md_files:
        all_logs.extend(merge_file(md_file, input_dir, output_dir))

    write_merge_log(all_logs, output_dir)
    print(f"[OK] merged {len(md_files)} files into {output_dir}")
    print(f"[OK] log written to {output_dir / LOG_FILENAME}")
    # endregion


def main():
    parser = argparse.ArgumentParser(description="Merge consecutive markdown tables based on ml tags.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    # 兩段式流程：先產出純合併結果，再對 07.6 merged markdown 另外重跑表格內層級。
    merge_consecutive_tables(args.input_dir, args.output_dir)
    format_merged_tables_inner_hierarchy(args.output_dir)


if __name__ == "__main__":
    main()
