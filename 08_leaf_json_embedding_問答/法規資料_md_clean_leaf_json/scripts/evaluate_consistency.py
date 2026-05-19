from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_engineering import call_ollama  # noqa: E402
from rag_runtime import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_TIMEOUT  # noqa: E402


DEFAULT_INPUT_PATH = ROOT / "data" / "evaluation_outputs" / "eval_outputs.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "evaluation_results"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def matches_question_type_filter(row: dict[str, Any], question_type_filter: str | None) -> bool:
    if not question_type_filter:
        return True
    normalized_filter = question_type_filter.strip().lower()
    normalized_question_type = str(row.get("question_type", "") or "").strip().lower()
    normalized_dataset_stem = Path(str(row.get("dataset_file", "") or "")).stem.lower()
    return (
        normalized_question_type == normalized_filter
        or normalized_dataset_stem == normalized_filter
        or normalized_dataset_stem.startswith(normalized_filter)
    )


def filter_rows_by_question_type(rows: list[dict[str, Any]], question_type_filter: str | None) -> list[dict[str, Any]]:
    if not question_type_filter:
        return rows
    return [row for row in rows if matches_question_type_filter(row, question_type_filter)]


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in response: {text}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"Evaluator response is not a JSON object: {parsed}")
    return parsed


def normalize_score(value: Any) -> float:
    score = float(value)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def build_qa_block(items: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(items, start=1):
        block = "\n".join(
            [
                f"[Item {idx}]",
                f"question_id: {str(item.get('question_id', '') or '').strip()}",
                f"question: {str(item.get('question', '') or '').strip()}",
                "answer:",
                str(item.get("generated_answer", "") or "").strip(),
            ]
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_consistency_prompt(qa_block: str) -> str:
    return f"""你是一個 RAG consistency evaluator。

以下是同一組同義問題的多個回答。
請判斷這些回答的核心結論是否一致。

請只回傳 JSON：
{{
  "score": 0.0~1.0,
  "reason": "..."
}}

評分標準：
1.0
所有回答核心結論一致，沒有矛盾。

0.7~0.9
大致一致，但有輕微細節差異或部分回答較不完整。

0.4~0.6
部分一致，但有明顯缺漏或回答重點不同。

0.1~0.3
大多不一致。

0.0
回答彼此矛盾或完全不同。

Questions and Answers:
{qa_block}
"""


def group_rows(rows: list[dict[str, Any]]) -> tuple[list[tuple[str, list[dict[str, Any]]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("group_id", "") or "").strip()
        if group_id:
            grouped[group_id].append(row)
    eligible_groups: list[tuple[str, list[dict[str, Any]]]] = []
    skipped_groups = 0
    for group_id, items in grouped.items():
        if len(items) < 2:
            skipped_groups += 1
            continue
        eligible_groups.append((group_id, items))
    return eligible_groups, skipped_groups


def evaluate_single_group(group_id: str, items: list[dict[str, Any]], ollama_model: str, ollama_timeout: int) -> dict[str, Any]:
    prompt = build_consistency_prompt(build_qa_block(items))
    raw_response = call_ollama(prompt, model_name=ollama_model, timeout=ollama_timeout)
    parsed = extract_json_object(raw_response)
    return {
        "group_id": group_id,
        "question_count": len(items),
        "items": [
            {
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "generated_answer": item.get("generated_answer"),
            }
            for item in items
        ],
        "consistency_score": normalize_score(parsed.get("score")),
        "consistency_reason": str(parsed.get("reason", "") or "").strip(),
        "error": None,
    }


def evaluate_consistency(
    input_path: Path,
    output_dir: Path,
    ollama_model: str,
    limit_groups: int | None,
    question_type_filter: str | None,
    ollama_timeout: int = DEFAULT_OLLAMA_TIMEOUT,
) -> None:
    filtered_rows = filter_rows_by_question_type(load_jsonl(input_path), question_type_filter)
    grouped_rows, skipped_groups = group_rows(filtered_rows)
    if limit_groups is not None:
        grouped_rows = grouped_rows[:limit_groups]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "consistency_results.jsonl"
    summary_path = output_dir / "consistency_summary.json"

    total_groups = len(grouped_rows)
    success_count = 0
    failed_count = 0
    score_sum = 0.0

    with results_path.open("w", encoding="utf-8-sig") as results_file:
        for index, (group_id, items) in enumerate(grouped_rows, start=1):
            print(f"[{index}/{total_groups}] evaluating consistency group_id={group_id}")
            try:
                result = evaluate_single_group(
                    group_id=group_id,
                    items=items,
                    ollama_model=ollama_model,
                    ollama_timeout=ollama_timeout,
                )
                success_count += 1
                score_sum += float(result["consistency_score"])
            except Exception as exc:
                failed_count += 1
                result = {
                    "group_id": group_id,
                    "question_count": len(items),
                    "items": [
                        {
                            "question_id": item.get("question_id"),
                            "question": item.get("question"),
                            "generated_answer": item.get("generated_answer"),
                        }
                        for item in items
                    ],
                    "consistency_score": None,
                    "consistency_reason": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    average_score = (score_sum / success_count) if success_count else None
    summary = {
        "metric": "consistency",
        "average_score": average_score,
        "total_groups": total_groups,
        "successful_groups": success_count,
        "failed_groups": failed_count,
        "skipped_groups": skipped_groups,
        "question_type_filter": question_type_filter,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(f"Completed. Results written to: {results_path}")
    print(f"Summary written to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate consistency for grouped batch RAG outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--question-type", default=None)
    args = parser.parse_args()

    evaluate_consistency(
        input_path=args.input,
        output_dir=args.output_dir,
        ollama_model=args.ollama_model,
        limit_groups=args.limit_groups,
        question_type_filter=args.question_type,
    )


if __name__ == "__main__":
    main()
