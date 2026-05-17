import csv
import re
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_DIR = BASE_DIR / "07.6連續表格合併"
LOG_PATH = MERGED_DIR / "merge_consecutive_tables_log.csv"
OUTPUT_PATH = MERGED_DIR / "merged_tables_review.md"


TABLE_START_RE = re.compile(
    r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$"
)
TABLE_END_RE = re.compile(r"^\[/TABLE id=(?P<table_id>[^\]]+)\]\s*$")


def parse_table_blocks(md_path: Path):
    lines = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tables = {}
    idx = 0

    while idx < len(lines):
        start_match = TABLE_START_RE.match(lines[idx].strip())
        if not start_match:
            idx += 1
            continue

        table_id = start_match.group("table_id")
        block_lines = []
        idx += 1
        while idx < len(lines):
            end_match = TABLE_END_RE.match(lines[idx].strip())
            if not end_match:
                block_lines.append(lines[idx])
            idx += 1
            if end_match:
                break

        tables[table_id] = "\n".join(block_lines)

    return tables


def load_merge_groups():
    groups = defaultdict(list)
    with open(LOG_PATH, "r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("status") != "merged":
                continue
            key = (row["file_name"], row["base_table_id"])
            groups[key].append(row)
    return groups


def build_review_markdown():
    groups = load_merge_groups()
    if not groups:
        OUTPUT_PATH.write_text("# Merged Tables Review\n\nNo merged tables found.\n", encoding="utf-8")
        return

    by_file = defaultdict(list)
    for (file_name, base_table_id), rows in groups.items():
        by_file[file_name].append((base_table_id, rows))

    md_parts = []
    md_parts.append("# Merged Tables Review")
    md_parts.append("")
    md_parts.append(f"- Source log: `{LOG_PATH}`")
    md_parts.append(f"- Merged markdown dir: `{MERGED_DIR}`")
    md_parts.append(f"- Files with merged tables: `{len(by_file)}`")
    md_parts.append(f"- Merged base tables: `{len(groups)}`")
    md_parts.append("")
    md_parts.append("## Index")
    md_parts.append("")

    section_no = 1
    ordered_files = sorted(by_file.items(), key=lambda item: item[0])
    for file_name, entries in ordered_files:
        md_parts.append(f"- {section_no}. `{file_name}` ({len(entries)} tables)")
        section_no += 1

    md_parts.append("")

    section_no = 1
    for file_name, entries in ordered_files:
        md_path = MERGED_DIR / file_name
        table_blocks = parse_table_blocks(md_path)
        md_parts.append(f"## {section_no}. {file_name}")
        md_parts.append("")

        for table_idx, (base_table_id, rows) in enumerate(sorted(entries, key=lambda item: item[0]), start=1):
            merged_table_ids = [row["merged_table_id"] for row in rows]
            ml_values = [row["ml"] for row in rows]
            actions = [row["action"] for row in rows]

            md_parts.append(f"### {section_no}.{table_idx} `{base_table_id}`")
            md_parts.append("")
            md_parts.append(f"- Merged count: `{len(rows)}`")
            md_parts.append(f"- `ml`: `{', '.join(ml_values)}`")
            md_parts.append(f"- Merged table ids: `{', '.join(merged_table_ids)}`")
            md_parts.append(f"- Actions: `{', '.join(actions)}`")
            md_parts.append("")
            md_parts.append(table_blocks.get(base_table_id, "[TABLE BLOCK NOT FOUND]"))
            md_parts.append("")

        section_no += 1

    OUTPUT_PATH.write_text("\n".join(md_parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_review_markdown()
