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
DEFAULT_MAX_CONTEXT_CHARS = 3000
RESULTS_FILE_NAME = "context_precision_results.jsonl"
SUMMARY_FILE_NAME = "context_precision_summary.json"


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


def truncate_context_text(context: dict[str, Any], max_context_chars: int) -> str:
    file_name = str(context.get("file_name", "") or "").strip()
    path_text = str(context.get("path_text", "") or "").strip()
    text = str(context.get("text", "") or "").strip()
    lines = []
    if file_name:
        lines.append(f"file_name: {file_name}")
    if path_text:
        lines.append(f"path_text: {path_text}")
    lines.append("text:")
    lines.append(text)
    return "\n".join(lines)[:max_context_chars].rstrip()


def build_context_precision_prompt(question: str, reference_answer: str, context_text: str) -> str:
    return f"""你是一個 RAG retrieval precision evaluator。

請判斷以下 retrieved context 是否對回答 question 有幫助，或是否能支持 reference answer。

請只回傳 JSON：
{{
  "relevant": true/false,
  "reason": "..."
}}

判斷標準：
- 若 context 包含回答問題所需的資訊，relevant=true。
- 若 context 只是在同一主題但不能回答問題，relevant=false。
- 若 context 來自錯誤法規、錯誤業別、錯誤條文，relevant=false。
- 若 context 只是標題、片段過短且無法提供答案，relevant=false。
- 不要使用外部知識，只能根據 question、reference_answer 和 context 判斷。

Question:
{question}

Reference Answer:
{reference_answer}

Retrieved Context:
{context_text}
"""


def parse_relevant(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"Invalid relevant value: {value}")


def evaluate_single_context(
    question: str,
    reference_answer: str,
    context: dict[str, Any],
    ollama_model: str,
    ollama_timeout: int,
    max_context_chars: int,
) -> dict[str, Any]:
    prompt = build_context_precision_prompt(
        question,
        reference_answer,
        truncate_context_text(context, max_context_chars=max_context_chars),
    )
    raw_response = call_ollama(prompt, model_name=ollama_model, timeout=ollama_timeout)
    parsed = extract_json_object(raw_response)
    return {
        "rank": context.get("rank"),
        "file_name": context.get("file_name"),
        "relevant": parse_relevant(parsed.get("relevant")),
        "reason": str(parsed.get("reason", "") or "").strip(),
    }


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

    context_results: list[dict[str, Any]] = []
    relevant_count = 0
    for context in retrieved_contexts:
        try:
            result = evaluate_single_context(
                question=question,
                reference_answer=reference_answer,
                context=context,
                ollama_model=ollama_model,
                ollama_timeout=ollama_timeout,
                max_context_chars=max_context_chars,
            )
            if result["relevant"] is True:
                relevant_count += 1
        except Exception as exc:
            result = {
                "rank": context.get("rank"),
                "file_name": context.get("file_name"),
                "relevant": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        context_results.append(result)
    total_contexts = len(retrieved_contexts)
    score = (relevant_count / total_contexts) if total_contexts > 0 else None
    return {
        "question_id": question_id,
        "question": question,
        "reference_answer": reference_answer,
        "context_results": context_results,
        "context_precision_score": score,
        "error": None,
        "updated_at": now_iso(),
    }


def build_error_result(row: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "reference_answer": row.get("reference_answer"),
        "context_results": [],
        "context_precision_score": None,
        "error": error,
        "updated_at": now_iso(),
    }


def build_summary(results_by_question_id: dict[str, dict[str, Any]], question_type_filter: str | None) -> dict[str, Any]:
    rows = list(results_by_question_id.values())
    total = len(rows)
    success_rows = [row for row in rows if row.get("error") is None]
    failed_rows = [row for row in rows if row.get("error") is not None]
    score_sum = sum(float(row["context_precision_score"]) for row in success_rows if row.get("context_precision_score") is not None)
    average_score = (score_sum / len(success_rows)) if success_rows else None
    return {
        "metric": "context_precision",
        "average_score": average_score,
        "total_questions": total,
        "successful_questions": len(success_rows),
        "failed_questions": len(failed_rows),
        "question_type_filter": question_type_filter,
    }


def evaluate_context_precision(
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
    parser = argparse.ArgumentParser(description="Evaluate context precision for batch RAG outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    evaluate_context_precision(
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
