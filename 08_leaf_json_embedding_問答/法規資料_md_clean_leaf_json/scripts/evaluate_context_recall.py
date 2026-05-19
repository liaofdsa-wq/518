from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_engineering import call_ollama  # noqa: E402
from rag_runtime import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_TIMEOUT  # noqa: E402


DEFAULT_INPUT_PATH = ROOT / "data" / "evaluation_outputs" / "eval_outputs.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "evaluation_results"
DEFAULT_MAX_CONTEXT_CHARS = 6000
RESULTS_FILE_NAME = "context_recall_results.jsonl"
SUMMARY_FILE_NAME = "context_recall_summary.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        question_id = str(row.get("question_id", "") or "").strip()
        if question_id:
            results[question_id] = row
    return results


def write_all_results(path: Path, results_by_question_id: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as file:
        for row in results_by_question_id.values():
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def build_context_text(retrieved_contexts: list[dict[str, Any]], max_context_chars: int) -> str:
    blocks: list[str] = []
    total_chars = 0
    for idx, item in enumerate(retrieved_contexts, start=1):
        file_name = str(item.get("file_name", "") or "").strip()
        path_text = str(item.get("path_text", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        lines = [f"[Context {idx}]"]
        if file_name:
            lines.append(f"file_name: {file_name}")
        if path_text:
            lines.append(f"path_text: {path_text}")
        lines.append("text:")
        lines.append(text)
        block = "\n".join(lines)
        remaining = max_context_chars - total_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        if not block:
            break
        blocks.append(block)
        total_chars += len(block) + 2
        if total_chars >= max_context_chars:
            break
    return "\n\n".join(blocks)


def build_context_recall_prompt(question: str, reference_answer: str, contexts: str) -> str:
    return f"""你是一個 RAG retrieval evaluator。

請判斷：

retrieved contexts
是否已包含足夠資訊，
能支持 reference answer。

請只回傳 JSON：
{{
  "score": 0.0~1.0,
  "reason": "..."
}}

評分標準：
1.0
contexts 已完整包含 reference answer 所需資訊。

0.7~0.9
大部分重要資訊存在，但略有缺漏。

0.4~0.6
只有部分重要資訊存在。

0.1~0.3
大部分重要資訊缺失。

0.0
contexts 幾乎無法支持 reference answer。

重要規則：
1. 只能根據 retrieved contexts 判斷。
2. 不要使用外部知識。
3. 不要評估 generated_answer。
4. 重點是 retrieval 有沒有抓到足夠資訊。

Question:
{question}

Reference Answer:
{reference_answer}

Retrieved Contexts:
{contexts}
"""


def evaluate_single_record(row: dict[str, Any], ollama_model: str, ollama_timeout: int, max_context_chars: int) -> dict[str, Any]:
    question_id = row.get("question_id")
    question = str(row.get("question", "") or "").strip()
    reference_answer = str(row.get("reference_answer", "") or "").strip()
    retrieved_contexts = row.get("retrieved_contexts") or []
    if not question:
        raise ValueError("Missing question")
    if not reference_answer:
        raise ValueError("Missing reference_answer")
    if not isinstance(retrieved_contexts, list):
        raise ValueError("retrieved_contexts is not a list")

    context_text = build_context_text(retrieved_contexts, max_context_chars=max_context_chars)
    prompt = build_context_recall_prompt(question, reference_answer, context_text)
    raw_response = call_ollama(prompt, model_name=ollama_model, timeout=ollama_timeout)
    parsed = extract_json_object(raw_response)
    return {
        "question_id": question_id,
        "question": question,
        "reference_answer": reference_answer,
        "context_recall_score": normalize_score(parsed.get("score")),
        "context_recall_reason": str(parsed.get("reason", "") or "").strip(),
        "error": None,
        "updated_at": now_iso(),
    }


def build_error_result(row: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "reference_answer": row.get("reference_answer"),
        "context_recall_score": None,
        "context_recall_reason": None,
        "error": error,
        "updated_at": now_iso(),
    }


def build_summary(results_by_question_id: dict[str, dict[str, Any]], question_type_filter: str | None) -> dict[str, Any]:
    rows = list(results_by_question_id.values())
    total = len(rows)
    success_rows = [row for row in rows if row.get("error") is None]
    failed_rows = [row for row in rows if row.get("error") is not None]
    score_sum = sum(float(row["context_recall_score"]) for row in success_rows if row.get("context_recall_score") is not None)
    average_score = (score_sum / len(success_rows)) if success_rows else None
    return {
        "metric": "context_recall",
        "average_score": average_score,
        "total_questions": total,
        "successful_questions": len(success_rows),
        "failed_questions": len(failed_rows),
        "question_type_filter": question_type_filter,
    }


def evaluate_context_recall(
    input_path: Path,
    output_dir: Path,
    ollama_model: str,
    limit: int | None,
    max_context_chars: int,
    question_type_filter: str | None,
    force_rerun: bool,
    ollama_timeout: int = DEFAULT_OLLAMA_TIMEOUT,
) -> None:
    rows = filter_rows_by_question_type(load_jsonl(input_path), question_type_filter)
    if limit is not None:
        rows = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILE_NAME
    summary_path = output_dir / SUMMARY_FILE_NAME
    existing_results = load_existing_results(results_path)

    for index, row in enumerate(rows, start=1):
        question_id = str(row.get("question_id", "") or f"row_{index}").strip()
        existing = existing_results.get(question_id)
        if not force_rerun and existing and existing.get("error") is None:
            print(f"[SKIP_METRIC] question_id={question_id}")
            continue
        if not force_rerun and existing and existing.get("error") is not None:
            print(f"[RERUN_METRIC_ERROR] question_id={question_id}")
        else:
            print(f"[EVAL_METRIC] question_id={question_id}")

        try:
            existing_results[question_id] = evaluate_single_record(
                row=row,
                ollama_model=ollama_model,
                ollama_timeout=ollama_timeout,
                max_context_chars=max_context_chars,
            )
        except Exception as exc:
            existing_results[question_id] = build_error_result(row, f"{type(exc).__name__}: {exc}")

    write_all_results(results_path, existing_results)
    summary = build_summary(existing_results, question_type_filter)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(f"Completed. Results written to: {results_path}")
    print(f"Summary written to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate context recall for batch RAG outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    evaluate_context_recall(
        input_path=args.input,
        output_dir=args.output_dir,
        ollama_model=args.ollama_model,
        limit=args.limit,
        max_context_chars=args.max_context_chars,
        question_type_filter=args.question_type,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
