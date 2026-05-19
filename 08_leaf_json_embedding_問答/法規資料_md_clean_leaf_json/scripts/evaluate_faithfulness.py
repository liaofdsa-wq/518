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
RESULTS_FILE_NAME = "faithfulness_results.jsonl"
SUMMARY_FILE_NAME = "faithfulness_summary.json"


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


def build_claim_extraction_prompt(generated_answer: str) -> str:
    return f"""你是一個 RAG faithfulness evaluator。

請將以下 answer 拆解成：

「可被 retrieved contexts 驗證的 factual claims」

請只回傳 JSON：
{{
  "claims": ["claim 1", "claim 2"]
}}

規則：
- claims 必須是完整 factual statement。
- 請保留完整句意，不要輸出殘缺短語。
- 不要輸出名詞片段。
- 不要過度切分。
- 若一句話本身就是單一完整規範，請直接保留整句作為 claim。
- claim 應保留必要上下文，不能只留下局部片段。
- 請盡量抽取：
  - 明確事實
  - 規定內容
  - 條件
  - 定義
  - 要求
  - 描述性規範
- 以下都應視為完整 claims：
  - 「應於各頁加蓋騎縫章，以防抽換」
  - 「人員離調職時應儘速移除權限」
  - 「高風險交易包括非約定轉帳交易」
- 以下都不是好的 claims，因為太碎：
  - 「防抽換」
  - 「資訊安全人員」
  - 「核心資通系統」
- 不要抽取客套話、格式標題、引用來源標題。
- 只有以下情況才回傳 {{"claims": []}}：
  - 完全拒答
  - 無內容
  - 單純說不知道
  - 單純說資料不足

Answer:
{generated_answer}
"""


def build_claim_verification_prompt(question: str, contexts: str, claim: str) -> str:
    return f"""你是一個 RAG faithfulness evaluator。
請判斷 claim 是否能被 retrieved contexts 支持。

請只回傳 JSON：
{{
  "score": 0.0~1.0,
  "reason": "..."
}}

評分標準：
- 1.0：claim 可被 retrieved contexts 明確完整支持
- 0.7~0.9：claim 大致可被支持，但有輕微延伸或細節不完整
- 0.4~0.6：claim 部分可被支持，但有明顯缺漏
- 0.1~0.3：claim 只有很弱的支持或高度推測
- 0.0：claim 無法被 contexts 支持、與 contexts 矛盾，或超出 contexts
- 不要使用外部知識，只能根據 contexts 判斷。

Question:
{question}

Retrieved Contexts:
{contexts}

Claim:
{claim}
"""


def normalize_claims(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    claims: list[str] = []
    predicate_markers = ("應", "須", "需", "不得", "包括", "係指", "是指", "屬於", "可", "得", "為")
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        if len(text) <= 6 and not any(marker in text for marker in predicate_markers):
            continue
        if text in {"防抽換", "資訊安全人員", "核心資通系統"}:
            continue
        claims.append(text)
    return claims


def fallback_claims_from_answer(generated_answer: str) -> list[str]:
    text = generated_answer.strip()
    if len(text) <= 20:
        return []
    lowered = text.lower()
    refusal_markers = (
        "不知道",
        "不清楚",
        "無法回答",
        "無法判斷",
        "資料不足",
        "資訊不足",
        "cannot answer",
        "insufficient information",
        "i don't know",
    )
    if any(marker in lowered for marker in refusal_markers):
        return []
    if len(text) < 150:
        return [text]
    return [text]


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


def extract_claims(generated_answer: str, ollama_model: str, ollama_timeout: int) -> list[str]:
    prompt = build_claim_extraction_prompt(generated_answer)
    try:
        raw_response = call_ollama(prompt, model_name=ollama_model, timeout=ollama_timeout)
        parsed = extract_json_object(raw_response)
        claims = normalize_claims(parsed.get("claims"))
        if claims:
            return claims
    except Exception:
        pass
    return fallback_claims_from_answer(generated_answer)


def verify_claim(question: str, contexts: str, claim: str, ollama_model: str, ollama_timeout: int) -> dict[str, Any]:
    prompt = build_claim_verification_prompt(question, contexts, claim)
    raw_response = call_ollama(prompt, model_name=ollama_model, timeout=ollama_timeout)
    parsed = extract_json_object(raw_response)
    return {
        "claim": claim,
        "score": normalize_score(parsed.get("score")),
        "reason": str(parsed.get("reason", "") or "").strip(),
    }


def evaluate_single_record(row: dict[str, Any], ollama_model: str, ollama_timeout: int, max_context_chars: int) -> dict[str, Any]:
    question_id = row.get("question_id")
    question = str(row.get("question", "") or "").strip()
    generated_answer = str(row.get("generated_answer", "") or "").strip()
    retrieved_contexts = row.get("retrieved_contexts") or []
    if not question:
        raise ValueError("Missing question")
    if not isinstance(retrieved_contexts, list):
        raise ValueError("retrieved_contexts is not a list")

    claims = extract_claims(generated_answer=generated_answer, ollama_model=ollama_model, ollama_timeout=ollama_timeout)
    if not claims:
        return {
            "question_id": question_id,
            "question": question,
            "generated_answer": generated_answer,
            "claims": [],
            "claim_results": [],
            "faithfulness_score": None,
            "error": "no_claims",
            "updated_at": now_iso(),
        }

    context_text = build_context_text(retrieved_contexts, max_context_chars=max_context_chars)
    claim_results: list[dict[str, Any]] = []
    score_sum = 0.0
    for claim in claims:
        claim_result = verify_claim(
            question=question,
            contexts=context_text,
            claim=claim,
            ollama_model=ollama_model,
            ollama_timeout=ollama_timeout,
        )
        claim_results.append(claim_result)
        score_sum += float(claim_result["score"])
    score = score_sum / len(claims)
    return {
        "question_id": question_id,
        "question": question,
        "generated_answer": generated_answer,
        "claims": claims,
        "claim_results": claim_results,
        "faithfulness_score": score,
        "error": None,
        "updated_at": now_iso(),
    }


def build_error_result(row: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "generated_answer": row.get("generated_answer"),
        "claims": [],
        "claim_results": [],
        "faithfulness_score": None,
        "error": error,
        "updated_at": now_iso(),
    }


def build_summary(results_by_question_id: dict[str, dict[str, Any]], question_type_filter: str | None) -> dict[str, Any]:
    rows = list(results_by_question_id.values())
    total = len(rows)
    success_rows = [row for row in rows if row.get("error") is None or row.get("error") == "no_claims"]
    failed_rows = [row for row in rows if row.get("error") not in {None, "no_claims"}]
    no_claim_rows = [row for row in rows if row.get("error") == "no_claims"]
    scored_rows = [row for row in rows if row.get("error") is None and row.get("faithfulness_score") is not None]
    score_sum = sum(float(row["faithfulness_score"]) for row in scored_rows)
    average_score = (score_sum / len(scored_rows)) if scored_rows else None
    return {
        "metric": "faithfulness",
        "average_score": average_score,
        "total_questions": total,
        "successful_questions": len(success_rows),
        "failed_questions": len(failed_rows),
        "no_claim_questions": len(no_claim_rows),
        "question_type_filter": question_type_filter,
    }


def evaluate_faithfulness(
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
    parser = argparse.ArgumentParser(description="Evaluate faithfulness for batch RAG outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    evaluate_faithfulness(
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
