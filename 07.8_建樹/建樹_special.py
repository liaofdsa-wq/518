import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

# special 版：
# 1. 保留一般建樹邏輯
# 2. 額外開放「特殊節點」與「子樹重建行為」可調

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PACKAGE_ROOT / "06_CleanTree生成區" / "法規資料_md_clean"
INPUT_DIR = RESULT_DIR / "structure"
OUTPUT_DIR = RESULT_DIR / "tree"

PATTERNS = [
    r"^第\s*[一二三四五六七八九十百0-9]+\s*章",
    r"^第\s*[一二三四五六七八九十百0-9]+\s*節",
    r"^第\s*[一二三四五六七八九十百0-9]+\s*條(?:之[一二三四五六七八九十0-9]+)?",
    r"^第\s*[一二三四五六七八九十百0-9]+\s*項",
    r"^第\s*[一二三四五六七八九十百0-9]+\s*款",
    r"^[一二三四五六七八九十]+[、．.]",
    r"^[壹貳參肆伍陸柒捌玖拾]+[、．.]",
    r"^[（(][一二三四五六七八九十]+[）)]",
    r"^[（(][壹貳參肆伍陸柒捌玖拾]+[）)]",
    r"^[（(]\d+[）)]",
    r"^[０-９\d]+[\.．、]",
    r"^[a-zA-Z][\.．、]",
    r"^[ivxlcdmIVXLCDM]+[\.．、]",
]

FIXED_LEVEL = {
    "CHAPTER": 0,
    "SECTION": 1,
    "ARTICLE": 2,
    "PARA": 3,
    "SUBITEM": 4,
}


@dataclass
class Node:
    text: str
    level: int
    children: List["Node"] = field(default_factory=list)


def is_table_start(line: str):
    return re.match(
        r"^\[TABLE\s+id=(?P<table_id>.+?)(?:\s+cr=(?P<cr>\d+))?(?:\s+ml=(?P<ml>\d+))?\]\s*$",
        line.strip(),
    )


def is_table_end(line: str):
    return re.match(r"^\[/TABLE id=([^\]]+)\]\s*$", line.strip())


def build_special_rules():
    # region 小工具
    def parse_special_marker(inner: str) -> Dict[str, str] | None:
        match = re.match(
            r"^SPECIAL\s+r=(?P<rule>[^\s\]]+)\s+s=(?P<serial>[^\s\]]+)\s+t=(?P<type>\d+)$",
            inner,
        )
        if not match:
            return None
        return match.groupdict()

    # endregion

    # region 分項規則
    def resolve_action_special_marker(
        inner: str,
        lines: List[str],
        idx: int,
        stack: List["Node"],
        type_last_level: Dict[str, int],
    ) -> Dict[str, Any] | None:
        _ = lines, idx, type_last_level
        marker_info = parse_special_marker(inner)
        if not marker_info:
            return None

        target_parent = None
        for node in reversed(stack):
            node_text = node.text.strip()
            if not (node_text.startswith("[") and node_text.endswith("]")):
                continue
            node_inner = node_text[1:-1].strip()
            if classify_general_type(node_inner)[0] == "ZH_PAREN":
                target_parent = node
                break

        if target_parent is None:
            return None

        return {
            "rule_name": marker_info["rule"],
            "numbering_type": f"SPECIAL_{marker_info['rule'].upper()}",
            "level": target_parent.level + 1,
            "scope_parent_level": target_parent.level,
            "action": "rebuild_subtree",
            "text": f"[{inner}]",
            "serial": marker_info["serial"],
            "special_type": marker_info["type"],
        }

    # endregion

    # region 主流程
    return [
        {
            "name": "special_marker",
            "resolve_action": resolve_action_special_marker,
        },
    ]
    # endregion


SPECIAL_RULES = build_special_rules()


def classify(inner: str) -> Tuple[str, bool]:
    if "章" in inner:
        return "CHAPTER", True
    if "節" in inner:
        return "SECTION", True
    if "條" in inner:
        return "ARTICLE", True
    if "項" in inner:
        return "PARA", True
    if "款" in inner:
        return "SUBITEM", True

    if re.match(r"^[一二三四五六七八九十]+[、．.]$", inner):
        return "ZH_DOT", False
    if re.match(r"^[壹貳參肆伍陸柒捌玖拾]+[、．.]$", inner):
        return "ZH_BIG_DOT", False
    if re.match(r"^[（(][一二三四五六七八九十]+[）)]$", inner):
        return "ZH_PAREN", False
    if re.match(r"^[（(][壹貳參肆伍陸柒捌玖拾]+[）)]$", inner):
        return "ZH_BIG_PAREN", False
    if re.match(r"^[０-９\d]+[\.．、]$", inner):
        return "NUM_DOT", False
    if re.match(r"^[（(]\d+[）)]$", inner):
        return "NUM_PAREN", False
    if re.match(r"^[a-zA-Z][\.．、]$", inner):
        return "ALPHA", False
    if re.match(r"^[ivxlcdmIVXLCDM]+[\.．、]$", inner):
        return "ROMAN", False

    return "OTHER", False


def normalize_bracket_lines(raw_lines: List[str]) -> List[str]:
    compiled = [re.compile(pattern) for pattern in PATTERNS]
    bracket_lines: List[str] = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue

        if is_table_start(stripped) or is_table_end(stripped):
            bracket_lines.append(stripped)
            continue

        if stripped.startswith("[SPECIAL ") or stripped == "[/SPECIAL]":
            bracket_lines.append(stripped)
            continue

        matched = False
        for pattern in compiled:
            match = pattern.match(stripped)
            if match:
                bracket_lines.append(f"[{match.group(0)}]")
                matched = True
                break

        if not matched:
            bracket_lines.append(stripped)

    return bracket_lines


def chinese_numeral_to_int(text: str) -> int | None:
    mapping = {
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
    if not text:
        return None
    if text == "十":
        return 10

    total = 0
    if "百" in text:
        parts = text.split("百", 1)
        head = mapping.get(parts[0], 1 if parts[0] == "" else None)
        if head is None:
            return None
        total += head * 100
        text = parts[1]

    if "十" in text:
        parts = text.split("十", 1)
        head = mapping.get(parts[0], 1 if parts[0] == "" else None)
        if head is None:
            return None
        total += head * 10
        text = parts[1]

    if text:
        tail = mapping.get(text)
        if tail is None:
            return None
        total += tail

    return total if total > 0 else None


def extract_heading_order(inner: str) -> Tuple[str, int] | None:
    patterns = [
        (r"^第\s*([一二三四五六七八九十百0-9]+)\s*章", "CHAPTER"),
        (r"^第\s*([一二三四五六七八九十百0-9]+)\s*節", "SECTION"),
        (r"^第\s*([一二三四五六七八九十百0-9]+)\s*條", "ARTICLE"),
        (r"^第\s*([一二三四五六七八九十百0-9]+)\s*項", "PARA"),
        (r"^第\s*([一二三四五六七八九十百0-9]+)\s*款", "SUBITEM"),
        (r"^([一二三四五六七八九十百]+)[、．.]", "ZH_DOT"),
        (r"^([壹貳參肆伍陸柒捌玖拾]+)[、．.]", "ZH_BIG_DOT"),
        (r"^[（(]([一二三四五六七八九十百]+)[）)]", "ZH_PAREN"),
        (r"^[（(]([壹貳參肆伍陸柒捌玖拾]+)[）)]", "ZH_BIG_PAREN"),
        (r"^[（(]([0-9０-９]+)[）)]", "NUM_PAREN"),
        (r"^([0-9０-９]+)[\.．、]", "NUM_DOT"),
    ]

    for pattern, tag_type in patterns:
        match = re.match(pattern, inner)
        if not match:
            continue
        raw_order = match.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        if raw_order.isdigit():
            return tag_type, int(raw_order)
        order = chinese_numeral_to_int(raw_order)
        if order is not None:
            return tag_type, order

    return None


def classify_general_type(inner: str) -> Tuple[str, bool]:
    return classify(inner)


def get_general_level_from_type(
    numbering_type: str,
    stack: List["Node"],
    type_last_level: Dict[str, int],
) -> int:
    if numbering_type in FIXED_LEVEL:
        return FIXED_LEVEL[numbering_type]

    if numbering_type in type_last_level:
        return type_last_level[numbering_type]

    level = stack[-1].level + 1
    type_last_level[numbering_type] = level
    return level


def resolve_special_action(
    inner: str,
    lines: List[str],
    idx: int,
    stack: List["Node"],
    type_last_level: Dict[str, int],
) -> Dict[str, Any] | None:
    for rule in SPECIAL_RULES:
        action = rule["resolve_action"](inner, lines, idx, stack, type_last_level)
        if action is not None:
            return action
    return None


def resolve_general_node_context(
    inner: str,
    stack: List["Node"],
    type_last_level: Dict[str, int],
) -> Dict[str, Any]:
    numbering_type, _ = classify_general_type(inner)
    level = get_general_level_from_type(numbering_type, stack, type_last_level)
    return {
        "is_special": False,
        "rule_name": None,
        "numbering_type": numbering_type,
        "level": level,
        "action": "normal",
    }


def resolve_general_level_for_simulation(
    inner: str,
    stack_levels: List[int],
    type_last_level: Dict[str, int],
) -> Dict[str, Any]:
    numbering_type, _ = classify_general_type(inner)
    if numbering_type in FIXED_LEVEL:
        level = FIXED_LEVEL[numbering_type]
    elif numbering_type in type_last_level:
        level = type_last_level[numbering_type]
    else:
        level = stack_levels[-1] + 1
        type_last_level[numbering_type] = level

    while stack_levels and stack_levels[-1] >= level:
        stack_levels.pop()
    stack_levels.append(level)

    return {
        "numbering_type": numbering_type,
        "level": level,
    }


def dump_tree(node: Node, indent: int = 0, out: List[str] | None = None) -> List[str]:
    if out is None:
        out = []

    if node.text != "ROOT":
        out.append("  " * indent + node.text)

    for child in node.children:
        dump_tree(child, indent + 1, out)

    return out


def build_tree_for_file(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    bracket_lines = normalize_bracket_lines(raw_lines)

    root = Node("ROOT", -1)
    stack: List[Node] = [root]
    type_last_level: Dict[str, int] = {}
    stats: Dict[str, Any] = {
        "emitted_nodes": 0,
        "special_hits": 0,
        "special_details": [],
    }

    def attach_heading_node(node_text: str, level: int, active_stack: List[Node]) -> Node:
        node = Node(node_text, level)
        while active_stack and active_stack[-1].level >= level:
            active_stack.pop()
        active_stack[-1].children.append(node)
        active_stack.append(node)
        return node

    def find_table_block_end(start_idx: int) -> int:
        cursor = start_idx + 1
        while cursor < len(bracket_lines):
            if is_table_end(bracket_lines[cursor]):
                return cursor + 1
            cursor += 1
        return cursor

    def rebuild_special_subtree(
        start_idx: int,
        special_node: Node,
        special_action: Dict[str, Any],
        outer_stack_snapshot: List[Node],
        outer_type_last_level_snapshot: Dict[str, int],
    ) -> int:
        local_stack: List[Node] = [special_node]
        local_type_last_level = dict(type_last_level)
        outer_stack_level_sim = [node.level for node in outer_stack_snapshot]
        outer_type_last_level_sim = dict(outer_type_last_level_snapshot)
        cursor = start_idx
        parent_level = special_action.get(
            "scope_parent_level",
            outer_stack_snapshot[-1].level if outer_stack_snapshot else -1,
        )
        level_offset: int | None = None

        while cursor < len(bracket_lines):
            stripped = bracket_lines[cursor].strip()
            if not stripped:
                cursor += 1
                continue

            if is_table_start(stripped):
                next_cursor = find_table_block_end(cursor)
                local_stack[-1].text += "\n" + "\n".join(bracket_lines[cursor:next_cursor])
                cursor = next_cursor
                continue

            if stripped == "[/SPECIAL]":
                cursor += 1
                break

            if stripped.startswith("[") and stripped.endswith("]"):
                inner = stripped[1:-1].strip()
                if resolve_special_action(inner, bracket_lines, cursor, stack, type_last_level) is not None:
                    break

                if (
                    special_action["rule_name"] == "r1"
                    and classify_general_type(inner)[0] == "ZH_PAREN"
                ):
                    break

                simulated_context = resolve_general_level_for_simulation(
                    inner,
                    outer_stack_level_sim,
                    outer_type_last_level_sim,
                )
                if simulated_context["level"] <= parent_level:
                    break

                if special_action["rule_name"] == "r1":
                    if level_offset is None:
                        level_offset = max(
                            1,
                            (special_node.level + 1) - simulated_context["level"],
                        )
                    actual_level = simulated_context["level"] + level_offset
                else:
                    actual_level = simulated_context["level"] + 1
                local_type_last_level[simulated_context["numbering_type"]] = actual_level
                attach_heading_node(stripped, actual_level, local_stack)
                stats["emitted_nodes"] += 1
            else:
                local_stack[-1].text += "\n" + stripped

            cursor += 1

        return cursor

    idx = 0
    while idx < len(bracket_lines):
        stripped = bracket_lines[idx].strip()
        if not stripped:
            idx += 1
            continue

        if is_table_start(stripped):
            next_idx = find_table_block_end(idx)
            stack[-1].text += "\n" + "\n".join(bracket_lines[idx:next_idx])
            idx = next_idx
            continue

        if stripped == "[/SPECIAL]":
            idx += 1
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            special_action = resolve_special_action(inner, bracket_lines, idx, stack, type_last_level)

            if special_action is not None:
                outer_stack_snapshot = list(stack)
                outer_type_last_level_snapshot = dict(type_last_level)
                special_node = attach_heading_node(
                    special_action["text"],
                    special_action["level"],
                    stack,
                )
                stats["emitted_nodes"] += 1
                stats["special_hits"] += 1
                stats["special_details"].append(
                    {
                        "rule_name": special_action["rule_name"],
                        "numbering_type": special_action["numbering_type"],
                        "level": special_action["level"],
                        "action": special_action["action"],
                        "text": special_action["text"],
                    }
                )

                if special_action["action"] == "rebuild_subtree":
                    idx = rebuild_special_subtree(
                        idx + 1,
                        special_node,
                        special_action,
                        outer_stack_snapshot,
                        outer_type_last_level_snapshot,
                    )
                    if stack and stack[-1] is special_node:
                        stack.pop()
                    continue

                idx += 1
                continue

            general_context = resolve_general_node_context(inner, stack, type_last_level)
            attach_heading_node(stripped, general_context["level"], stack)
            stats["emitted_nodes"] += 1
        else:
            stack[-1].text += "\n" + stripped

        idx += 1

    return dump_tree(root), stats


def build_tree_pipeline(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "# tree build summary",
        "",
        f"input_dir={input_dir}",
        f"output_dir={output_dir}",
        "",
    ]

    for file_path in sorted(input_dir.glob("*.md")):
        tree_lines, stats = build_tree_for_file(file_path)
        out_path = output_dir / file_path.name
        out_path.write_text("\n".join(tree_lines), encoding="utf-8")

        summary_lines.append(f"## {file_path.name}")
        summary_lines.append(f"- emitted_nodes={stats['emitted_nodes']}")
        summary_lines.append(f"- special_hits={stats['special_hits']}")
        if stats["special_details"]:
            for detail in stats["special_details"]:
                summary_lines.append(
                    "- special_rule={rule_name}, numbering_type={numbering_type}, "
                    "level={level}, action={action}, text={text}".format(**detail)
                )
        summary_lines.append("")

    (output_dir / "_build_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[OK] special tree 建置完成：{output_dir}")


if __name__ == "__main__":
    build_tree_pipeline()
