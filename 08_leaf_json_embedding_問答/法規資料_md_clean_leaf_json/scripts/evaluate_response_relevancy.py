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
DEFAULT_METRIC_NAME = "response_relevancy"
RESULTS_FILE_NAME = "relevancy_results.jsonl"
SUMMARY_FILE_NAME = "relevancy_summary.json"


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


def build_relevancy_prompt(question: str, generated_answer: str) -> str:
    return f"""你是一個 RAG QA evaluator。

請判斷以下 answer 是否真正回答 question。

請只回傳 JSON：
{{
  "score": 0.0~1.0,
  "reason": "..."
}}

注意：
- Response Relevancy 只評估 answer 是否有針對 question 回答。
- 不要把 relevancy 當成 factual correctness。
- 不要因為答案不完整就直接給極低分。

評分標準：
1.0：
完整且直接回答問題。

0.8~0.9：
有直接回答問題核心，但略有冗長或小缺漏。

0.6~0.7：
有回答到主要方向，但答案不完整或部分偏離。

0.3~0.5：
只回答到很小一部分，或以資訊不足迴避但仍與問題相關。

0.0~0.2：
完全答非所問或沒有回答。

補充規則：
- 若答案有回答到問題核心，但不完整，應給 0.6~0.8。
- 若答案有回答到問題，但有部分多餘內容，仍可給 0.7~0.9。
- 若答案是「資訊不足」但題目需要具體答案，通常給 0.2~0.4。
- 若完全答非所問，才給 0~0.2。

Question:
{question}

Answer:
{generated_answer}
"""


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


def evaluate_single_record(row: dict[str, Any], ollama_model: str, ollama_timeout: int) -> dict[str, Any]:
    question_id = row.get("question_id")
    question = str(row.get("question", "") or "").strip()
    generated_answer = str(row.get("generated_answer", "") or "").strip()

    if not question:
        raise ValueError("Missing question")
    if not generated_answer:
        raise ValueError("Missing generated_answer")

    prompt = build_relevancy_prompt(question, generated_answer)
    raw_response = call_ollama(prompt, model_name=ollama_model, timeout=ollama_timeout)
    parsed = extract_json_object(raw_response)

    return {
        "question_id": question_id,
        "question": question,
        "generated_answer": generated_answer,
        "relevancy_score": normalize_score(parsed.get("score")),
        "relevancy_reason": str(parsed.get("reason", "") or "").strip(),
        "error": None,
        "updated_at": now_iso(),
    }


def build_error_result(row: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "generated_answer": row.get("generated_answer"),
        "relevancy_score": None,
        "relevancy_reason": None,
        "error": error,
        "updated_at": now_iso(),
    }


def build_summary(results_by_question_id: dict[str, dict[str, Any]], question_type_filter: str | None) -> dict[str, Any]:
    rows = list(results_by_question_id.values())
    total = len(rows)
    success_rows = [row for row in rows if row.get("error") is None]
    failed_rows = [row for row in rows if row.get("error") is not None]
    score_sum = sum(float(row["relevancy_score"]) for row in success_rows if row.get("relevancy_score") is not None)
    average_score = (score_sum / len(success_rows)) if success_rows else None
    return {
        "metric": DEFAULT_METRIC_NAME,
        "average_score": average_score,
        "total_questions": total,
        "successful_questions": len(success_rows),
        "failed_questions": len(failed_rows),
        "question_type_filter": question_type_filter,
    }


def evaluate_relevancy(
    input_path: Path,
    output_dir: Path,
    ollama_model: str,
    limit: int | None,
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
            )
        except Exception as exc:
            existing_results[question_id] = build_error_result(row, f"{type(exc).__name__}: {exc}")

    write_all_results(results_path, existing_results)
    summary = build_summary(existing_results, question_type_filter)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(f"Completed. Results written to: {results_path}")
    print(f"Summary written to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate response relevancy for batch RAG outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    evaluate_relevancy(
        input_path=args.input,
        output_dir=args.output_dir,
        ollama_model=args.ollama_model,
        limit=args.limit,
        question_type_filter=args.question_type,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
