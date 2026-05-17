import csv
import re
import shutil
from pathlib import Path
from collections import Counter
from collections import defaultdict
# Codex 副本：
# 1. 不覆蓋原始「結構化法規程式/分類前處理.py」
# 2. 目前此檔只負責前處理；建樹已拆到「建樹_codex.py」

# 1. 路徑定義 (保持你提供的結構)
BASE_DIR = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PACKAGE_ROOT / "03_Markdown生成區" / "法規資料_md"
RESULT_DIR = PACKAGE_ROOT / "06_CleanTree生成區" / "法規資料_md_clean"
LOG_DIR = RESULT_DIR / "log"
working_dir = RESULT_DIR / SOURCE_DIR.name


ENABLED_SPECIAL_RULES = ["r1_three_tables"]


def build_special_block_rules():
    # region 小工具
    def is_table_start(line: str):
        return re.match(
            r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$",
            line.strip(),
        )

    def is_table_end(line: str):
        return re.match(r"^\[/TABLE id=([^\]]+)\]\s*$", line.strip())

    def parse_table_rows(lines, start_idx):
        start_match = is_table_start(lines[start_idx])
        if not start_match:
            return None

        table_id = start_match.group(1)
        end_idx = start_idx + 1
        table_lines = []

        while end_idx < len(lines):
            stripped = lines[end_idx].strip()
            end_match = is_table_end(stripped)
            if end_match:
                return {
                    "table_id": table_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "rows": table_lines,
                }
            table_lines.append(lines[end_idx].rstrip("\n"))
            end_idx += 1

        return None

    def parse_markdown_row(line: str):
        stripped = line.strip()
        if "|" not in stripped:
            return None
        return [cell.strip() for cell in stripped.strip("|").split("|")]

    def is_separator_row(cells):
        if not cells:
            return False
        return all(re.fullmatch(r"[-: ]+", cell or "") for cell in cells)

    def get_data_rows(table_info):
        rows = []
        for line in table_info["rows"]:
            cells = parse_markdown_row(line)
            if not cells or is_separator_row(cells):
                continue
            rows.append(cells)
        return rows

    def chinese_numeral_to_int(text):
        stripped = (text or "").strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return int(stripped)

        normalized = (
            stripped
            .replace("ㄧ", "一")
            .replace("０", "0")
            .replace("１", "1")
            .replace("２", "2")
            .replace("３", "3")
            .replace("４", "4")
            .replace("５", "5")
            .replace("６", "6")
            .replace("７", "7")
            .replace("８", "8")
            .replace("９", "9")
        )
        if normalized.isdigit():
            return int(normalized)

        digit_map = {
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
        }
        total = 0
        current = 0
        for ch in normalized:
            if ch in digit_map:
                current = digit_map[ch]
                continue
            if ch == "十":
                total += (current or 1) * 10
                current = 0
                continue
            if ch == "百":
                total += (current or 1) * 100
                current = 0
                continue
            return None
        return total + current if total or current else None

    def extract_leading_article_no(text):
        match = re.match(r"^第\s*([一二三四五六七八九十百ㄧ0-9０-９]+)\s*條", (text or "").strip())
        if not match:
            return None
        return chinese_numeral_to_int(match.group(1))

    def extract_leading_item_no(text):
        match = re.match(r"^([一二三四五六七八九十百ㄧ0-9０-９]+)[、．\.]", (text or "").strip())
        if not match:
            return None
        return chinese_numeral_to_int(match.group(1))

    def find_current_page(lines, idx):
        cursor = idx
        while cursor >= 0:
            match = re.match(r"^\[page\s+(\d+)\]$", lines[cursor].strip(), re.IGNORECASE)
            if match:
                return int(match.group(1))
            cursor -= 1
        return None

    def collect_three_consecutive_tables(lines, idx):
        tables = []
        cursor = idx

        for _ in range(3):
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1

            table_info = parse_table_rows(lines, cursor)
            if not table_info:
                return None
            tables.append(table_info)
            cursor = table_info["end_idx"] + 1

            if len(tables) < 3:
                while cursor < len(lines) and not lines[cursor].strip():
                    cursor += 1

                if cursor >= len(lines) or not is_table_start(lines[cursor]):
                    return None

        return tables

    def find_previous_table(lines, idx):
        cursor = idx - 1
        while cursor >= 0 and not lines[cursor].strip():
            cursor -= 1

        if cursor < 0 or not is_table_end(lines[cursor]):
            return None

        while cursor >= 0 and not is_table_start(lines[cursor]):
            cursor -= 1

        if cursor < 0:
            return None

        return parse_table_rows(lines, cursor)

    def find_previous_nonempty_idx(lines, idx):
        cursor = idx - 1
        while cursor >= 0 and not lines[cursor].strip():
            cursor -= 1
        return cursor

    def normalize_cell_text(text):
        return re.sub(r"\s+", " ", (text or "").strip())

    def strip_merge_tags(text):
        return re.sub(r"\[@\d+\]\s*", "", text or "")

    def compact_relaxed_item_rows(rows):
        compacted_rows = []
        seen_rows = set()

        for row in rows:
            compacted = []
            last_value = None
            for cell in row:
                normalized = normalize_cell_text(cell)
                if not normalized:
                    continue
                if normalized == last_value:
                    continue
                compacted.append(normalized)
                last_value = normalized

            if not compacted:
                continue

            row_key = tuple(compacted)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            compacted_rows.append(compacted)

        return compacted_rows

    # endregion

    # region 分項規則
    def build_marker_result_r1_three_tables(lines, idx, md_file):
        is_relaxed_security_baseline = (
            md_file.name == "金融機構資訊系統安全基準.md"
            and (find_current_page(lines, idx) or 0) >= 126
        )

        if is_relaxed_security_baseline:
            def is_relaxed_category_line(line):
                stripped = line.strip()
                if not stripped:
                    return False
                if stripped.startswith("[") or stripped.startswith("|"):
                    return False
                return True

            def find_relaxed_applicability_context():
                cursor = find_previous_nonempty_idx(lines, idx)
                category_line_idx = None

                if cursor >= 0 and is_relaxed_category_line(lines[cursor]):
                    category_line_idx = cursor
                    cursor = find_previous_nonempty_idx(lines, category_line_idx)

                if cursor < 0 or not is_table_end(lines[cursor]):
                    return None

                while cursor >= 0 and not is_table_start(lines[cursor]):
                    cursor -= 1

                if cursor < 0:
                    return None

                return parse_table_rows(lines, cursor)

            item_table = parse_table_rows(lines, idx)
            if not item_table:
                return None

            item_rows = compact_relaxed_item_rows(get_data_rows(item_table))
            if len(item_rows) != 1 or len(item_rows[0]) != 2:
                return None

            first_cell = normalize_cell_text(strip_merge_tags(item_rows[0][0]))
            if re.match(r"^(設|運|技)\s*[0-9０-９]+(?:-[0-9０-９]+)?$", first_cell) is None:
                return None

            applicability_table = find_relaxed_applicability_context()
            if not applicability_table:
                return None

            applicability_rows = get_data_rows(applicability_table)
            if len(applicability_rows) != 3:
                return None
            if any(len(cells) not in {4, 5} for cells in applicability_rows):
                return None
            if len({len(cells) for cells in applicability_rows}) != 1:
                return None

            category_table = find_previous_table(lines, applicability_table["start_idx"])
            start_table = applicability_table
            if category_table:
                category_rows = get_data_rows(category_table)
                if len(category_rows) == 2 and all(len(cells) == 1 for cells in category_rows):
                    start_table = category_table

            serial_match = re.search(r"([0-9０-９]+(?:-[0-9０-９]+)?)$", first_cell)
            if not serial_match:
                return None

            return {
                "rule_name": "r1",
                "serial": serial_match.group(1),
                "type": 1,
                "start_idx": start_table["start_idx"],
                "end_idx": item_table["end_idx"],
            }

        tables = collect_three_consecutive_tables(lines, idx)
        if not tables:
            return None

        first_table_rows = get_data_rows(tables[0])
        second_table_rows = get_data_rows(tables[1])
        third_table_rows = get_data_rows(tables[2])

        if len(first_table_rows) != 2:
            return None
        if any(len(cells) != 1 for cells in first_table_rows):
            return None

        if len(second_table_rows) != 3:
            return None
        if any(len(cells) not in {4, 5} for cells in second_table_rows):
            return None
        if len({len(cells) for cells in second_table_rows}) != 1:
            return None

        if len(third_table_rows) != 1 or len(third_table_rows[0]) != 2:
            return None

        first_cell = re.sub(r"\s+", " ", third_table_rows[0][0]).strip()
        serial_match = re.search(r"([0-9０-９]+)$", first_cell)
        if serial_match:
            return {
                "rule_name": "r1",
                "serial": serial_match.group(1),
                "type": 1,
                "start_idx": tables[0]["start_idx"],
                "end_idx": tables[2]["end_idx"],
            }
        return None

    # endregion

    # region 主流程
    return {
        "r1_three_tables": {
            "name": "r1_three_tables",
            "match": lambda lines, idx, md_file: (
                build_marker_result_r1_three_tables(lines, idx, md_file) is not None
            ),
            "build_marker_result": build_marker_result_r1_three_tables,
        },
    }
    # endregion


def initialize_working_dir():
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)

    shutil.copytree(SOURCE_DIR, working_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return working_dir

# 用 working_dir 當主資料夾

def rewrite_page_markers(working_dir):
    import re
    from pathlib import Path

    page_pattern = re.compile(r"^\s*##\s*Page\s*(\d+)", re.IGNORECASE)

    for md_file in Path(working_dir).glob("*.md"):

        lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines = []

        for line in lines:
            stripped = line.strip()

            m = page_pattern.match(stripped)
            if m:
                page_num = m.group(1)
                new_lines.append(f"[page {page_num}]")  # 👉 替換
            else:
                new_lines.append(line)

        # 👉 覆寫原檔
        md_file.write_text("\n".join(new_lines), encoding="utf-8")

    print("[OK] 頁碼已轉換為 [page X]")
def extract_tables():
    """
    母函式：
    - 從 md 抓出表格區塊
    - 原文移除並加標記
    - 表格另存
    - 紀錄 log
    """
    # ========= 1. 判斷是否為表格行 =========
    def is_table_line(line: str):
        line = line.strip()

        # Markdown 表格 or 分隔線
        if "|" in line:
            return True
        if re.match(r"^\s*\|?[-: ]+\|[-|: ]+\|?\s*$", line):
            return True

        return False

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

    def row_has_vertical_merge(upper_row, lower_row):
        width = min(len(upper_row), len(lower_row))
        for col_idx in range(width):
            upper_tag = get_merge_tag(upper_row[col_idx])
            lower_tag = get_merge_tag(lower_row[col_idx])
            if upper_tag and lower_tag and upper_tag == lower_tag:
                return True
        return False

    def detect_column_row_count(block_lines):
        matrix_2d = []
        for line in block_lines:
            cells = parse_markdown_row(line)
            if not cells or is_separator_row(cells):
                continue
            matrix_2d.append(cells)

        if not matrix_2d:
            return 0
        if len(matrix_2d) == 1:
            return 1
        if not row_has_vertical_merge(matrix_2d[0], matrix_2d[1]):
            return 1

        row_count = 2
        for row_idx in range(2, len(matrix_2d)):
            if not row_has_vertical_merge(matrix_2d[row_idx - 1], matrix_2d[row_idx]):
                break
            row_count = row_idx + 1
        return row_count

    # ========= 2. 抓表格區塊 =========
    def extract_table_blocks(lines):
        blocks = []
        current_block = []
        in_table = False

        for line in lines:
            if is_table_line(line):
                current_block.append(line)
                in_table = True
            else:
                if in_table:
                    blocks.append(current_block)
                    current_block = []
                    in_table = False

        # 收尾
        if current_block:
            blocks.append(current_block)

        return blocks

    # ========= 3. 單檔處理 =========
    def process_file(md_path: Path, writer):
        with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        table_blocks = extract_table_blocks(lines)

        new_lines = []
        table_id = 0
        i = 0

        while i < len(lines):
            if is_table_line(lines[i]):
                # 收集整塊表格
                block = []
                while i < len(lines) and is_table_line(lines[i]):
                    block.append(lines[i])
                    i += 1

                table_id += 1
                table_tag = f"{md_path.stem}_table_{table_id}"
                table_name = f"{table_tag}.md"
                column_row_count = detect_column_row_count(block)
                table_path = RESULT_DIR / "tables"
                table_path.mkdir(exist_ok=True)

                # 假設 table_path 是總資料夾
                table_root = Path(table_path)

                # 👉 用 md 檔名當子資料夾名稱（你可以自己改規則）
                subfolder = table_root / md_file.stem

                # 👉 確保資料夾存在
                subfolder.mkdir(parents=True, exist_ok=True)

                # 👉 寫入 table
                with open(subfolder / table_name, "w", encoding="utf-8") as tf:
                    tf.writelines(block)

                # log
                writer.writerow([
                    md_path.name,
                    table_name,
                    f"{len(block)} 行"
                ])

                # 在原文保留表格本體，但以前後標記包起來
                if column_row_count > 1:
                    new_lines.append(f"[TABLE id={table_tag} cr={column_row_count}]\n")
                else:
                    new_lines.append(f"[TABLE id={table_tag}]\n")
                new_lines.extend(block)
                if block and not block[-1].endswith("\n"):
                    new_lines.append("\n")
                new_lines.append(f"[/TABLE id={table_tag}]\n")

            else:
                new_lines.append(lines[i])
                i += 1

        return new_lines

    # ========= 4. 主流程 =========
    LOG_PATH = LOG_DIR / "table_log.csv"

    with open(LOG_PATH, "w", encoding="utf-8-sig", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["來源檔案", "表格檔名", "行數"])

        files = list(working_dir.glob("*.md"))
        print(f"抓到 {len(files)} 個 md 檔")

        for md_file in files:
            new_lines = process_file(md_file, writer)

            output_path = working_dir / md_file.name
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

    print("[OK] 表格抽取完成")

def mark_consecutive_table_merges(input_dir=working_dir):
    # region 小工具
    def is_table_start(line: str):
        return re.match(
            r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$",
            line.strip(),
        )

    def is_table_end(line: str):
        return re.match(r"^\[/TABLE id=([^\]]+)\]\s*$", line.strip())

    def build_plain_table_start_line(table_id: str, cr: int | None = None):
        cr_part = f" cr={cr}" if cr is not None and cr > 1 else ""
        return f"[TABLE id={table_id}{cr_part}]"

    def build_merge_table_start_line(table_id: str, decision, cr: int | None = None):
        logic_id = decision.get("logic_id", 1)
        cr_part = f" cr={cr}" if cr is not None and cr > 1 else ""
        return f"[TABLE id={table_id}{cr_part} ml={logic_id}]"

    def strip_merge_tags(text: str):
        return re.sub(r"\[@\d+\]\s*", "", text or "")

    def normalize_compare_text(text: str):
        stripped = strip_merge_tags(text)
        stripped = re.sub(r"\s+", " ", stripped)
        return stripped.strip()

    def chinese_numeral_to_int(text):
        stripped = (text or "").strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return int(stripped)

        normalized = (
            stripped
            .replace("ㄧ", "一")
            .replace("０", "0")
            .replace("１", "1")
            .replace("２", "2")
            .replace("３", "3")
            .replace("４", "4")
            .replace("５", "5")
            .replace("６", "6")
            .replace("７", "7")
            .replace("８", "8")
            .replace("９", "9")
        )
        if normalized.isdigit():
            return int(normalized)

        digit_map = {
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
        }
        total = 0
        current = 0
        for ch in normalized:
            if ch in digit_map:
                current = digit_map[ch]
                continue
            if ch == "十":
                total += (current or 1) * 10
                current = 0
                continue
            if ch == "百":
                total += (current or 1) * 100
                current = 0
                continue
            return None
        return total + current if total or current else None

    def extract_leading_article_no(text):
        match = re.match(r"^第\s*([一二三四五六七八九十百ㄧ0-9０-９]+)\s*條", (text or "").strip())
        if not match:
            return None
        return chinese_numeral_to_int(match.group(1))

    def extract_leading_item_no(text):
        match = re.match(r"^([一二三四五六七八九十百ㄧ0-9０-９]+)[、．\.]", (text or "").strip())
        if not match:
            return None
        return chinese_numeral_to_int(match.group(1))

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

    def build_matrix_2d(row_lines):
        matrix = []
        clean_lines = []
        for line in row_lines:
            cells = parse_markdown_row(line)
            if not cells or is_separator_row(cells):
                continue
            matrix.append(cells)
            clean_lines.append(line.rstrip("\n"))
        return matrix, clean_lines

    def row_has_vertical_merge(upper_row, lower_row):
        width = min(len(upper_row), len(lower_row))
        for col_idx in range(width):
            upper_tag = get_merge_tag(upper_row[col_idx])
            lower_tag = get_merge_tag(lower_row[col_idx])
            if upper_tag and lower_tag and upper_tag == lower_tag:
                return True
        return False

    def get_column_block_row_count(matrix_2d):
        if not matrix_2d:
            return 0
        if len(matrix_2d) == 1:
            return 1
        if not row_has_vertical_merge(matrix_2d[0], matrix_2d[1]):
            return 1

        row_count = 2
        for row_idx in range(2, len(matrix_2d)):
            if not row_has_vertical_merge(matrix_2d[row_idx - 1], matrix_2d[row_idx]):
                break
            row_count = row_idx + 1
        return row_count

    def build_column_blob(matrix_2d, clean_lines, column_block_row_count):
        block_rows = clean_lines[:column_block_row_count]
        normalized_rows = [normalize_compare_text(row) for row in block_rows]
        return {
            "column_block_rows": block_rows,
            "column_blob_raw": "\n".join(block_rows),
            "column_blob_normalized_for_compare": "\n".join(normalized_rows),
        }

    def find_current_page(lines, idx):
        cursor = idx
        while cursor >= 0:
            match = re.match(r"^\[page\s+(\d+)\]$", lines[cursor].strip(), re.IGNORECASE)
            if match:
                return int(match.group(1))
            cursor -= 1
        return None

    def parse_table_block(lines, start_idx):
        start_match = is_table_start(lines[start_idx])
        if not start_match:
            return None

        table_id = start_match.group("table_id")
        cursor = start_idx + 1
        row_lines = []

        while cursor < len(lines):
            if is_table_end(lines[cursor]):
                return {
                    "table_id": table_id,
                    "cr": start_match.group("cr"),
                    "start_idx": start_idx,
                    "end_idx": cursor,
                    "row_lines": row_lines,
                }
            row_lines.append(lines[cursor])
            cursor += 1

        return None

    def build_table_profiles(lines):
        profiles = []
        idx = 0
        while idx < len(lines):
            table_info = parse_table_block(lines, idx)
            if not table_info:
                idx += 1
                continue

            matrix_2d, clean_lines = build_matrix_2d(table_info["row_lines"])
            col_count = max((len(row) for row in matrix_2d), default=0)
            if table_info["cr"] is not None:
                column_block_row_count = min(int(table_info["cr"]), len(clean_lines))
            else:
                column_block_row_count = 1 if clean_lines else 0
            column_blob_info = build_column_blob(matrix_2d, clean_lines, column_block_row_count)
            first_outer_row_idx = column_block_row_count
            first_data_row_first_col = ""
            first_outer_row_first_col = ""
            first_outer_row_second_col = ""
            last_outer_row_first_col = ""
            last_outer_row_second_col = ""
            if matrix_2d and matrix_2d[0]:
                first_data_row_first_col = normalize_compare_text(matrix_2d[0][0])
            if first_outer_row_idx < len(matrix_2d) and matrix_2d[first_outer_row_idx]:
                first_outer_row_first_col = normalize_compare_text(matrix_2d[first_outer_row_idx][0])
                if len(matrix_2d[first_outer_row_idx]) > 1:
                    first_outer_row_second_col = normalize_compare_text(
                        matrix_2d[first_outer_row_idx][1]
                    )
            for row in reversed(matrix_2d[first_outer_row_idx:]):
                if row:
                    last_outer_row_first_col = normalize_compare_text(row[0])
                    if len(row) > 1:
                        last_outer_row_second_col = normalize_compare_text(row[1])
                    break

            profiles.append({
                "table_id": table_info["table_id"],
                "start_idx": table_info["start_idx"],
                "end_idx": table_info["end_idx"],
                "page_num": find_current_page(lines, table_info["start_idx"]),
                "matrix_2d": matrix_2d,
                "col_count": col_count,
                "column_block_row_count": column_block_row_count,
                "column_block_rows": column_blob_info["column_block_rows"],
                "column_blob_raw": column_blob_info["column_blob_raw"],
                "column_blob_normalized_for_compare": column_blob_info["column_blob_normalized_for_compare"],
                "first_data_row_first_col": first_data_row_first_col,
                "first_outer_row_first_col": first_outer_row_first_col,
                "first_outer_row_second_col": first_outer_row_second_col,
                "last_outer_row_first_col": last_outer_row_first_col,
                "last_outer_row_second_col": last_outer_row_second_col,
            })
            idx = table_info["end_idx"] + 1
        return profiles

    def get_range_tables_2d(profiles, page_start, page_end):
        return [
            profile["matrix_2d"]
            for profile in profiles
            if (
                profile["page_num"] is not None
                and (page_start is None or page_start <= profile["page_num"])
                and (page_end is None or profile["page_num"] <= page_end)
            )
        ]

    # endregion

    # region 特殊規則
    def handle_logic_1_fcm(range_tables_2d, file_name, page_range, upper_table, lower_table):
        _ = range_tables_2d, page_range
        logic_1_fcm_ranges = [
            {"file_name": "1.FCM內控-總則---114.04.md"},
            {"file_name": "2.FCM內控CA---壹、業務及收入循環 ---114.04.md"},
            {"file_name": "3.FCM內控CP ---貳、採購及付款循環 ---114.04.md"},
            {"file_name": "4.FCM內控CW---參 、薪工循環---114.04.md"},
            {"file_name": "5.FCM內控CR---肆、融資循環---114.04.md"},
            {"file_name": "6.FCM內控CF---伍、不動產及設備循環---114.04.md"},
            {"file_name": "7.FCM內控CI---陸、投資循環---114.04.md"},
            {"file_name": "8.FCM內控CC---柒、電腦作業與資訊提供---114.04.md"},
            {"file_name": "9.FCM內控CM---捌、管理控制制度---114.04.md"},
            {"file_name": "~內部控制制度總則(114.04.11修)--FOREWORD.md"},
            {
                "file_name": "金融機構辦理電子銀行業務安全控管作業基準含附錄1150107.md",
                "page_start": 116,
                "page_end": 137,
            },
        ]
        matched_range = None
        for range_rule in logic_1_fcm_ranges:
            if range_rule.get("file_name") != file_name:
                continue
            page_start = range_rule.get("page_start")
            page_end = range_rule.get("page_end")
            if upper_table["page_num"] is None or lower_table["page_num"] is None:
                continue
            if page_start is not None and (
                upper_table["page_num"] < page_start or lower_table["page_num"] < page_start
            ):
                continue
            if page_end is not None and (
                upper_table["page_num"] > page_end or lower_table["page_num"] > page_end
            ):
                continue
            matched_range = range_rule
            break
        if matched_range is None:
            return None
        if upper_table["col_count"] != lower_table["col_count"]:
            return {
                "is_merge_candidate": False,
                "reason": "different_col_count",
            }
        if (
            upper_table["column_blob_normalized_for_compare"]
            != lower_table["column_blob_normalized_for_compare"]
        ):
            return {
                "is_merge_candidate": False,
                "reason": "different_column_blob",
            }
        lower_first_value = lower_table["first_outer_row_first_col"]
        upper_last_value = upper_table["last_outer_row_first_col"]
        if lower_first_value and lower_first_value != upper_last_value:
            return {
                "is_merge_candidate": False,
                "reason": "lower_first_col_not_blank_or_equal_to_upper_last",
            }
        decision = {
            "is_merge_candidate": True,
            "reason": "logic_1_fcm_basic_match",
            "merge_strategy": "append_table_rows",
            "cut_column_block_rows": lower_table["column_block_row_count"],
            "logic_id": 1,
        }
        return decision

    def handle_logic_2_ebanking_blank_first_cell(
        range_tables_2d,
        file_name,
        page_range,
        upper_table,
        lower_table,
    ):
        _ = range_tables_2d, page_range, upper_table
        logic_2_ebanking_ranges = [
            {
                "file_name": "金融機構辦理電子銀行業務安全控管作業基準含附錄1150107.md",
                "page_start": 33,
                "page_end": 115,
            },
        ]
        matched_range = None
        for range_rule in logic_2_ebanking_ranges:
            if range_rule.get("file_name") != file_name:
                continue
            page_start = range_rule.get("page_start")
            page_end = range_rule.get("page_end")
            if upper_table["page_num"] is None or lower_table["page_num"] is None:
                continue
            if page_start is not None and (
                upper_table["page_num"] < page_start or lower_table["page_num"] < page_start
            ):
                continue
            if page_end is not None and (
                upper_table["page_num"] > page_end or lower_table["page_num"] > page_end
            ):
                continue
            matched_range = range_rule
            break
        if matched_range is None:
            return None
        if lower_table["first_data_row_first_col"] != "":
            return {
                "is_merge_candidate": False,
                "reason": "lower_first_data_cell_not_blank",
            }
        return {
            "is_merge_candidate": True,
            "reason": "logic_2_ebanking_blank_first_cell",
            "merge_strategy": "append_table_rows",
            "cut_column_block_rows": lower_table["column_block_row_count"],
            "logic_id": 2,
        }

    def handle_logic_3_ebanking_outer_12(
        range_tables_2d,
        file_name,
        page_range,
        upper_table,
        lower_table,
    ):
        _ = range_tables_2d, page_range
        logic_3_ebanking_ranges = [
            {
                "file_name": "金融機構辦理電子銀行業務安全控管作業基準含附錄1150107.md",
                "page_start": 138,
            },
        ]
        matched_range = None
        for range_rule in logic_3_ebanking_ranges:
            if range_rule.get("file_name") != file_name:
                continue
            page_start = range_rule.get("page_start")
            page_end = range_rule.get("page_end")
            if upper_table["page_num"] is None or lower_table["page_num"] is None:
                continue
            if page_start is not None and (
                upper_table["page_num"] < page_start or lower_table["page_num"] < page_start
            ):
                continue
            if page_end is not None and (
                upper_table["page_num"] > page_end or lower_table["page_num"] > page_end
            ):
                continue
            matched_range = range_rule
            break
        if matched_range is None:
            return None
        if upper_table["col_count"] != lower_table["col_count"]:
            return {
                "is_merge_candidate": False,
                "reason": "different_col_count",
            }
        if (
            upper_table["column_blob_normalized_for_compare"]
            != lower_table["column_blob_normalized_for_compare"]
        ):
            return {
                "is_merge_candidate": False,
                "reason": "different_column_blob",
            }
        lower_first_value = lower_table["first_outer_row_second_col"]
        upper_last_value = upper_table["last_outer_row_second_col"]
        if lower_first_value and lower_first_value != upper_last_value:
            return {
                "is_merge_candidate": False,
                "reason": "lower_outer_12_not_blank_or_equal_to_upper_last_outer_12",
            }
        return {
            "is_merge_candidate": True,
            "reason": "logic_3_ebanking_outer_12",
            "merge_strategy": "append_table_rows",
            "cut_column_block_rows": lower_table["column_block_row_count"],
            "logic_id": 3,
        }

    def handle_logic_4_sto_article_gap(
        range_tables_2d,
        file_name,
        page_range,
        upper_table,
        lower_table,
    ):
        _ = range_tables_2d, page_range, upper_table
        logic_4_sto_ranges = [
            {
                "file_name": "財團法人中華民國證券櫃檯買賣中心證券商經營自行買賣具證券性質之虛擬通貨業務管理辦法1090120.md",
            },
        ]
        matched_range = None
        for range_rule in logic_4_sto_ranges:
            if range_rule.get("file_name") != file_name:
                continue
            page_start = range_rule.get("page_start")
            page_end = range_rule.get("page_end")
            if upper_table["page_num"] is None or lower_table["page_num"] is None:
                continue
            if page_start is not None and (
                upper_table["page_num"] < page_start or lower_table["page_num"] < page_start
            ):
                continue
            if page_end is not None and (
                upper_table["page_num"] > page_end or lower_table["page_num"] > page_end
            ):
                continue
            matched_range = range_rule
            break
        if matched_range is None:
            return None

        lower_body_rows = lower_table["matrix_2d"][lower_table["column_block_row_count"]:]
        if len(lower_body_rows) < 2:
            return {
                "is_merge_candidate": True,
                "reason": "logic_4_sto_single_tail_row",
                "merge_strategy": "append_table_rows",
                "cut_column_block_rows": lower_table["column_block_row_count"],
                "logic_id": 4,
            }

        first_body_article_no = extract_leading_article_no(lower_body_rows[0][0] if lower_body_rows[0] else "")
        next_article_no = None
        for row in lower_body_rows[1:]:
            if not row:
                continue
            next_article_no = extract_leading_article_no(row[0])
            if next_article_no is not None:
                break

        if next_article_no is None:
            return {
                "is_merge_candidate": True,
                "reason": "logic_4_sto_no_next_article_row",
                "merge_strategy": "append_table_rows",
                "cut_column_block_rows": lower_table["column_block_row_count"],
                "logic_id": 4,
            }

        if first_body_article_no == next_article_no - 1:
            return {
                "is_merge_candidate": False,
                "reason": "lower_first_body_article_is_expected_previous_article",
            }

        return {
            "is_merge_candidate": True,
            "reason": "logic_4_sto_article_gap",
            "merge_strategy": "append_table_rows",
            "cut_column_block_rows": lower_table["column_block_row_count"],
            "logic_id": 4,
        }

    def handle_logic_5_futures_item_gap(
        range_tables_2d,
        file_name,
        page_range,
        upper_table,
        lower_table,
    ):
        _ = range_tables_2d, page_range, upper_table
        logic_5_futures_ranges = [
            {
                "file_name": "期貨商作業委託他人處理應注意事項逐點說明.md",
            },
        ]
        matched_range = None
        for range_rule in logic_5_futures_ranges:
            if range_rule.get("file_name") != file_name:
                continue
            page_start = range_rule.get("page_start")
            page_end = range_rule.get("page_end")
            if upper_table["page_num"] is None or lower_table["page_num"] is None:
                continue
            if page_start is not None and (
                upper_table["page_num"] < page_start or lower_table["page_num"] < page_start
            ):
                continue
            if page_end is not None and (
                upper_table["page_num"] > page_end or lower_table["page_num"] > page_end
            ):
                continue
            matched_range = range_rule
            break
        if matched_range is None:
            return None

        lower_body_rows = lower_table["matrix_2d"][lower_table["column_block_row_count"]:]
        if len(lower_body_rows) < 2:
            return {
                "is_merge_candidate": False,
                "reason": "lower_body_rows_less_than_2",
            }

        first_body_item_no = extract_leading_item_no(lower_body_rows[0][0] if lower_body_rows[0] else "")
        next_item_no = None
        for row in lower_body_rows[1:]:
            if not row:
                continue
            next_item_no = extract_leading_item_no(row[0])
            if next_item_no is not None:
                break

        if next_item_no is None:
            return {
                "is_merge_candidate": False,
                "reason": "no_next_item_row_in_lower_first_col",
            }

        if first_body_item_no == next_item_no - 1:
            return {
                "is_merge_candidate": False,
                "reason": "lower_first_body_item_is_expected_previous_item",
            }

        return {
            "is_merge_candidate": True,
            "reason": "logic_5_futures_item_gap",
            "merge_strategy": "append_table_rows",
            "cut_column_block_rows": lower_table["column_block_row_count"],
            "logic_id": 5,
        }

    # endregion

    # region 主流程
    def build_special_decision(profiles, upper_table, lower_table, md_file):
        rule_configs = [
            {
                "handler": handle_logic_1_fcm,
                "ranges": [{}],
            },
            {
                "handler": handle_logic_2_ebanking_blank_first_cell,
                "ranges": [{}],
            },
            {
                "handler": handle_logic_3_ebanking_outer_12,
                "ranges": [{}],
            },
            {
                "handler": handle_logic_4_sto_article_gap,
                "ranges": [{}],
            },
            {
                "handler": handle_logic_5_futures_item_gap,
                "ranges": [{}],
            },
        ]

        for rule_config in rule_configs:
            handler = rule_config["handler"]
            for range_rule in rule_config["ranges"]:
                target_file_name = range_rule.get("file_name")
                if target_file_name is not None and target_file_name != md_file.name:
                    continue

                page_start = range_rule.get("page_start")
                page_end = range_rule.get("page_end")
                if upper_table["page_num"] is None or lower_table["page_num"] is None:
                    continue
                if page_start is not None and upper_table["page_num"] < page_start:
                    continue
                if page_end is not None and upper_table["page_num"] > page_end:
                    continue
                if page_start is not None and lower_table["page_num"] < page_start:
                    continue
                if page_end is not None and lower_table["page_num"] > page_end:
                    continue

                range_tables_2d = get_range_tables_2d(profiles, page_start, page_end)
                decision = handler(
                    range_tables_2d,
                    md_file.name,
                    (page_start, page_end),
                    upper_table,
                    lower_table,
                )
                if decision:
                    return decision
        return None

    def analyze_file(md_file: Path):
        original_lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        lines = []
        for line in original_lines:
            table_match = is_table_start(line)
            if table_match:
                cr = table_match.group("cr")
                lines.append(
                    f"{build_plain_table_start_line(table_match.group('table_id'), int(cr) if cr is not None and int(cr) > 1 else None)}\n"
                )
            else:
                lines.append(line)
        profiles = build_table_profiles(lines)
        if len(profiles) < 2:
            return lines, []

        marker_map = {}
        log_rows = []

        for upper_table, lower_table in zip(profiles, profiles[1:]):
            decision = build_special_decision(profiles, upper_table, lower_table, md_file)
            if decision and decision.get("is_merge_candidate"):
                marker_map[lower_table["start_idx"]] = build_merge_table_start_line(
                    lower_table["table_id"],
                    decision,
                    lower_table["column_block_row_count"],
                )
                log_rows.append([
                    md_file.name,
                    upper_table["table_id"],
                    lower_table["table_id"],
                    upper_table["page_num"],
                    lower_table["page_num"],
                    decision["merge_strategy"],
                    decision["reason"],
                    decision.get("cut_column_block_rows", ""),
                ])
                continue

            continue

        if not marker_map:
            return lines, []

        rebuilt_lines = []
        idx = 0
        while idx < len(lines):
            if idx in marker_map:
                rebuilt_lines.append(f"{marker_map[idx]}\n")
            else:
                rebuilt_lines.append(lines[idx])
            idx += 1

        return rebuilt_lines, log_rows

    input_path = Path(input_dir)
    log_path = LOG_DIR / "consecutive_table_merge_log.csv"
    all_log_rows = []

    for md_file in sorted(input_path.glob("*.md")):
        new_lines, log_rows = analyze_file(md_file)
        md_file.write_text("".join(new_lines), encoding="utf-8")
        all_log_rows.extend(log_rows)
        if log_rows:
            print(f"[MERGE] {md_file.name}: {len(log_rows)}")

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name",
            "upper_table_id",
            "lower_table_id",
            "upper_page",
            "lower_page",
            "merge_strategy",
            "reason",
            "cut_column_block_rows",
        ])
        writer.writerows(all_log_rows)

    print("[OK] 連續表格判定完成")
    # endregion


def mark_special_blocks():
    special_rules = build_special_block_rules()
    enabled_rules = [
        special_rules[rule_name]
        for rule_name in ENABLED_SPECIAL_RULES
        if rule_name in special_rules
    ]

    def build_marker_lines(rule_name, serial, special_type):
        return (
            [f"[SPECIAL r={rule_name} s={serial} t={special_type}]"],
            ["[/SPECIAL]"],
        )

    def apply_marker_result(lines, marker_result):
        def ensure_trailing_blank_line(block_lines):
            if not block_lines:
                return block_lines
            if block_lines[-1].strip():
                return block_lines + ["\n"]
            return block_lines

        def ensure_leading_blank_line(block_lines):
            if not block_lines:
                return block_lines
            if block_lines[0].strip():
                return ["\n"] + block_lines
            return block_lines

        start_idx = marker_result["start_idx"]
        end_idx = marker_result["end_idx"]
        open_marker_lines = marker_result["open_marker_lines"]
        close_marker_lines = marker_result["close_marker_lines"]
        before_lines = ensure_trailing_blank_line(lines[:start_idx])
        original_lines = lines[start_idx:end_idx + 1]
        after_lines = ensure_leading_blank_line(lines[end_idx + 1:])

        return (
            before_lines
            + [f"{line}\n" for line in open_marker_lines]
            + original_lines
            + [f"{line}\n" for line in close_marker_lines]
            + after_lines
        )

    def scan_file(md_file: Path):
        lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        if not enabled_rules:
            return lines

        matched_count = 0
        idx = 0
        while idx < len(lines):
            for rule in enabled_rules:
                matched = rule["match"](lines, idx, md_file)
                if not matched:
                    continue

                marker_result = rule["build_marker_result"](lines, idx, md_file)
                if not marker_result:
                    continue

                if "open_marker_lines" not in marker_result or "close_marker_lines" not in marker_result:
                    open_marker_lines, close_marker_lines = build_marker_lines(
                        marker_result["rule_name"],
                        marker_result["serial"],
                        marker_result.get("type", 1),
                    )
                    marker_result.setdefault("open_marker_lines", open_marker_lines)
                    marker_result.setdefault("close_marker_lines", close_marker_lines)
                lines = apply_marker_result(lines, marker_result)
                idx = (
                    marker_result["start_idx"]
                    + len(marker_result["open_marker_lines"])
                    + (marker_result["end_idx"] - marker_result["start_idx"] + 1)
                    + len(marker_result["close_marker_lines"])
                )
                matched_count += 1
                break
            else:
                idx += 1

        if matched_count:
            print(f"[SPECIAL] {md_file.name}: {matched_count}")
        return lines

    for md_file in working_dir.glob("*.md"):
        new_lines = scan_file(md_file)
        md_file.write_text("".join(new_lines), encoding="utf-8")

    print("[OK] special 標記完成")
def extract_heading_lines():
    import re
    from pathlib import Path

    OUT_DIR = RESULT_DIR / "headings"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def is_table_start(line: str):
        return re.match(
            r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$",
            line.strip(),
        )

    def is_table_end(line: str):
        return re.match(r"^\[/TABLE id=([^\]]+)\]\s*$", line.strip())

    # =====================================
    # 🔧 1. pattern registry（辨識）
    # =====================================
    def build_patterns():
        return [
            ("CHAPTER",  r"^第\s*([一二三四五六七八九十百0-9]+)\s*章"),
            ("SECTION",  r"^第\s*([一二三四五六七八九十百0-9]+)\s*節"),
            ("ARTICLE",  r"^第\s*([一二三四五六七八九十百0-9]+)\s*條(?:之\d+)?"),
            ("PARAGRAPH",r"^第\s*([一二三四五六七八九十百0-9]+)\s*項"),
            ("SUBITEM",  r"^第\s*([一二三四五六七八九十百0-9]+)\s*款"),
            ("SPECIAL_BLOCK", r"^\[SPECIAL\s+r=([^\s\]]+)\s+s=([^\s\]]+)\s+t=(\d+)\]$"),

            ("ITEM_ZH", r"^([一二三四五六七八九十]+)[、．.]"),
            ("ITEM_ZH_BIG", r"^([壹貳參肆伍陸柒捌玖])[、．.]"),
            ("PAREN_ZH_BIG", r"^[（(]([壹貳參肆伍陸柒捌玖拾]+)[）)]"),

            ("PAREN_ZH", r"^[（(]([一二三四五六七八九十]+)[）)]"),
            ("PAREN_NUM", r"^[（(](\d+)[）)]"),

            ("POINT_NUM", r"^(\d+)[\.．、]"),
            ("ALPHA", r"^([a-zA-Z])[\.．、]"),
            ("ROMAN", r"^([ivxlcdmIVXLCDM]+)[\.．、]"),
        ]

    # =====================================
    # 🔧 2. level system（階層）
    # =====================================
    def build_level_maps():
        fixed_level = {
            "CHAPTER": 0,
            "SECTION": 1,
            "ARTICLE": 2,
            "PARAGRAPH": 3,
            "SUBITEM": 4
        }

        dynamic_level_map = {
            "SPECIAL_BLOCK": 5,
            "ITEM_ZH": 5,
            "ITEM_ZH_BIG": 5,
            "PAREN_ZH": 6,
            "PAREN_ZH_BIG": 6,
            "PAREN_NUM": 7,
            "POINT_NUM": 7,
            "ALPHA": 8,
            "ROMAN": 9
        }

        return fixed_level, dynamic_level_map

    def get_level(tag_type, fixed, dynamic):
        if tag_type in fixed:
            return fixed[tag_type]
        if tag_type in dynamic:
            return dynamic[tag_type]
        return None

    # =====================================
    # 🔧 3. heading 判斷（🔥核心）
    # =====================================
    def classify_line(line, patterns, prev_line=None):
        

        # 👉 判斷是否為「區塊開頭」
        is_block_start = (
            prev_line is None or
            prev_line.strip() == "" or
            prev_line.startswith("[page") or
            prev_line.startswith("#")
        )

        if not is_block_start:
            return None

        # 👉 原本的匹配
        for tag_type, pattern in patterns:
            m = re.match(pattern, line)
            if m:
                token = m.group(0)
                number = m.group(1) if m.groups() else None

                return {
                    "type": tag_type,
                    "token": token,
                    "number": number
                }

        return None

    # =====================================
    # 🔧 4. block extraction
    # =====================================
    def extract_blocks(lines, patterns, fixed, dynamic):

        blocks = []
        new_lines = []          # 👉 主md（前後包）
        structure_lines = []    # 👉 純tag
        marked_lines = []       # 👉 如果你還要語意版（可選）

        current_block = []
        current_tag = None

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            stripped = line.strip()
            table_start_match = is_table_start(stripped)
            if table_start_match:
                table_block_lines = [line]
                idx += 1
                while idx < len(lines):
                    table_line = lines[idx]
                    table_block_lines.append(table_line)
                    if is_table_end(table_line.strip()):
                        idx += 1
                        break
                    idx += 1

                if current_block:
                    current_block.extend(table_block_lines)
                new_lines.extend(table_block_lines)
                continue

            if stripped == "[/SPECIAL]":
                if current_block and current_tag == "SPECIAL_BLOCK":
                    current_block.append(line)
                    blocks.append(current_block)
                    new_lines.append(line)
                    current_block = []
                    current_tag = None
                else:
                    new_lines.append(line)
                idx += 1
                continue

            if current_block and current_tag == "SPECIAL_BLOCK":
                current_block.append(line)
                new_lines.append(line)
                idx += 1
                continue

            info = classify_line(stripped, patterns, prev_line=None)
            

            if info:
                level = get_level(info["type"], fixed, dynamic)
                tag = info["token"]

                if current_block:
                    blocks.append(current_block)
                    new_lines.append(f"[/{current_tag}]")

                if info["type"] == "SPECIAL_BLOCK":
                    current_block = [line]
                    current_tag = "SPECIAL_BLOCK"
                    new_lines.append(line)
                    structure_lines.append(line)
                    idx += 1
                    continue

                current_block = [line]
                current_tag = tag

                # 👉 開 tag
                new_lines.append(f"[{tag}]")

                # 🔥 關鍵：保留原始那一行
                new_lines.append(line)

                # 👉 structure
                structure_lines.append(f"[{tag}]")

            else:
                if current_block:
                    current_block.append(line)
                    new_lines.append(line)
                else:
                    new_lines.append(line)
            idx += 1

        if current_block:
            blocks.append(current_block)
            if current_tag == "SPECIAL_BLOCK":
                if not current_block[-1].strip() == "[/SPECIAL]":
                    new_lines.append("[/SPECIAL]")
            else:
                new_lines.append(f"[/{current_tag}]")

        return blocks, new_lines, structure_lines

    # =====================================
    # 🔧 5. IO
    # =====================================
    def save_blocks(blocks, path):
        out = []
        for block in blocks:
            out.extend(block)
            out.append("")
        path.write_text("\n".join(out), encoding="utf-8")

    def overwrite(md_file, new_lines, structure_lines):

        # 👉 主md（可讀版）
        md_file.write_text("\n".join(new_lines), encoding="utf-8")

        # 👉 結構檔（另存）
        structure_dir = md_file.parent.parent / "structure"
        structure_dir.mkdir(exist_ok=True)

        structure_file = structure_dir / md_file.name
        structure_file.write_text("\n".join(structure_lines), encoding="utf-8")

    # =====================================
    # 🚀 主流程
    # =====================================
    patterns = build_patterns()
    fixed, dynamic = build_level_maps()

    for md_file in working_dir.glob("*.md"):

        lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines()


        blocks, new_lines, structure_lines = extract_blocks(lines, patterns, fixed, dynamic)

        save_blocks(blocks, OUT_DIR / md_file.name)
        overwrite(md_file, new_lines, structure_lines)

    print("[OK] 完整 heading parser 重寫完成")
def clean_md_garbage_folder(input_dir, output_dir, header_threshold=3):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = RESULT_DIR/"log.csv"
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== pattern =====
    page_patterns = [
        r"^\s*第\s*\d+\s*頁\s*$",
        r"^\s*第\s*\d+\s*頁\s*共\s*\d+\s*頁\s*$",
        r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$"
    ]

    noise_patterns = [
        r"^\s*\d+\s*$",
        r"\d{2,3}年\d{1,2}月.*臺灣期貨交易所"
    ]

    page_regex = re.compile("|".join(page_patterns), re.IGNORECASE)
    noise_regex = re.compile("|".join(noise_patterns), re.IGNORECASE)

    # ===== CSV log =====
    log_rows = []

    for file in input_dir.glob("*.md"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        file_log = defaultdict(list)

        # ---- 1️⃣ 清頁碼 + OCR ----
        temp_lines = []
        in_table_block = False
        in_special_block = False
        for idx, line in enumerate(lines):
            s = line.strip()
            prev_stripped = lines[idx - 1].strip() if idx > 0 else ""
            next_stripped = lines[idx + 1].strip() if idx + 1 < len(lines) else ""

            if s.startswith("[SPECIAL "):
                in_special_block = True
                temp_lines.append(line)
                continue

            if s == "[/SPECIAL]":
                in_special_block = False
                temp_lines.append(line)
                continue

            if in_special_block:
                temp_lines.append(line)
                continue

            if s.startswith("[TABLE "):
                in_table_block = True
                temp_lines.append(line)
                continue

            if s.startswith("[/TABLE id="):
                in_table_block = False
                temp_lines.append(line)
                continue

            if in_table_block:
                temp_lines.append(line)
                continue

            if not s:
                if (
                    prev_stripped == "[/SPECIAL]"
                    or next_stripped.startswith("[SPECIAL ")
                ):
                    temp_lines.append(line)
                continue

            if page_regex.match(s):
                if len(file_log["page"]) < 3:
                    file_log["page"].append(s)
                continue

            if noise_regex.match(s):
                if len(file_log["noise"]) < 3:
                    file_log["noise"].append(s)
                continue

            temp_lines.append(line)

        # ---- 2️⃣ header 偵測（排除 [ 開頭）----
        stripped = []
        in_table_block = False
        in_special_block = False
        for line in temp_lines:
            s = line.strip()
            if s.startswith("[SPECIAL "):
                in_special_block = True
                continue
            if s == "[/SPECIAL]":
                in_special_block = False
                continue
            if in_special_block:
                continue
            if s.startswith("[TABLE "):
                in_table_block = True
                continue
            if s.startswith("[/TABLE id="):
                in_table_block = False
                continue
            if in_table_block:
                continue
            if s and not s.startswith("["):
                stripped.append(s)

        counter = Counter(stripped)

        header_candidates = {
            line for line, count in counter.items()
            if count >= header_threshold and len(line) <= 30
        }

        # ---- 3️⃣ 刪 header（保護 [）----
        final_lines = []
        in_table_block = False
        in_special_block = False
        for line in temp_lines:
            s = line.strip()

            if s.startswith("[SPECIAL "):
                in_special_block = True
                final_lines.append(line)
                continue

            if s == "[/SPECIAL]":
                in_special_block = False
                final_lines.append(line)
                continue

            if in_special_block:
                final_lines.append(line)
                continue

            if s.startswith("[TABLE "):
                in_table_block = True
                final_lines.append(line)
                continue

            if s.startswith("[/TABLE id="):
                in_table_block = False
                final_lines.append(line)
                continue

            if in_table_block:
                final_lines.append(line)
                continue

            if s.startswith("["):
                final_lines.append(line)
                continue

            if s in header_candidates:
                if len(file_log["header"]) < 3:
                    file_log["header"].append(s)
                continue

            final_lines.append(line)

        # ---- 4️⃣ SPECIAL 前後空白標準化 ----
        normalized_lines = []
        for line in final_lines:
            s = line.strip()

            if s.startswith("[SPECIAL "):
                if normalized_lines and normalized_lines[-1].strip():
                    normalized_lines.append("")
                normalized_lines.append(line)
                continue

            if s == "[/SPECIAL]":
                normalized_lines.append(line)
                normalized_lines.append("")
                continue

            normalized_lines.append(line)

        while normalized_lines and not normalized_lines[-1].strip():
            normalized_lines.pop()

        # ---- 5️⃣ 寫 md ----
        out_path = output_dir / file.name
        out_path.write_text("\n".join(normalized_lines), encoding="utf-8")

        # ---- 5️⃣ 寫 log rows ----
        for log_type, items in file_log.items():
            for item in items:
                log_rows.append([file.name, log_type, item])

        print(f"[OK] cleaned: {file.name}")

    # ---- 6️⃣ 輸出 CSV ----
    with open(log_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "type", "content"])
        writer.writerows(log_rows)

    print("[OK] 完成（CSV log 已輸出）")
def run_preprocess_pipeline():
    initialize_working_dir()
    rewrite_page_markers(working_dir)
    extract_tables()
    mark_special_blocks()
    extract_heading_lines()
    clean_md_garbage_folder(working_dir, working_dir, header_threshold=3)
    mark_consecutive_table_merges(working_dir)
    print(f"[OK] 前處理完成：{working_dir}")


# ========= 執行 =========
if __name__ == "__main__":
    run_preprocess_pipeline()
