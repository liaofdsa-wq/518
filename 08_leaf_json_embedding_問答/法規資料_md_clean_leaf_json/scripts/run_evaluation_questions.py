from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prompt_engineering  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from prompt_engineering import call_ollama  # noqa: E402
from rag_runtime import (  # noqa: E402
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_PROMPT_TOP_N,
    DEFAULT_TOP_K,
    build_eval_prompt,
    load_bm25_index,
    load_embedding_data,
    load_model,
    prepare_prompt_contexts,
    run_search,
)


EVALUATION_DATASET_DIR = ROOT / "evaluation_dataset"
OUTPUT_PATH = ROOT / "data" / "evaluation_outputs" / "eval_outputs.jsonl"
DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_HYBRID_TEXT_MODE = "leaf"
DEFAULT_ALPHA = 0.5
DATASET_FILES = (
    "general_qa_15.xlsx",
    "large_doc_qa.xlsx",
    "table_qa.xlsx",
    "consistency_qa.xlsx",
)


def detect_error_message(answer_text: str | None) -> str | None:
    if not answer_text:
        return None

    normalized = answer_text.strip()
    error_markers = (
        "[錯誤]",
        "[error]",
        "404 Client Error",
        "ConnectionError",
        "Not Found for url:",
        "http://localhost:11434/api/generate",
        "Ollama",
    )
    if normalized.startswith("[錯誤]"):
        return normalized
    if any(marker.lower() in normalized.lower() for marker in error_markers):
        return normalized
    return None


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def as_text(value: Any) -> str | None:
    value = normalize_value(value)
    if value is None:
        return None
    return str(value)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def matches_question_type_filter(
    question_type_value: str | None,
    question_type_filter: str | None,
    dataset_file: str | None = None,
) -> bool:
    if not question_type_filter:
        return True
    normalized_filter = question_type_filter.strip().lower()
    normalized_question_type = str(question_type_value or "").strip().lower()
    normalized_dataset_stem = Path(dataset_file).stem.lower() if dataset_file else ""
    return (
        normalized_question_type == normalized_filter
        or normalized_dataset_stem == normalized_filter
        or normalized_dataset_stem.startswith(normalized_filter)
    )


def resolve_input_files(dataset_dir: Path, input_files: list[str] | None) -> list[Path]:
    file_names = input_files or list(DATASET_FILES)
    resolved_files: list[Path] = []
    for file_name in file_names:
        file_path = dataset_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing evaluation dataset: {file_path}")
        resolved_files.append(file_path)
    return resolved_files


def load_jsonl_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question_id = as_text(row.get("question_id"))
            if not question_id:
                continue
            records[question_id] = row
    return records


def load_question_rows(
    dataset_dir: Path,
    input_files: list[str] | None,
    question_type_filter: str | None,
) -> list[dict[str, Any]]:
    rows_by_question_id: dict[str, dict[str, Any]] = {}

    for file_path in resolve_input_files(dataset_dir, input_files):
        df = pd.read_excel(file_path)
        for excel_row_index, row in df.iterrows():
            raw = row.to_dict()
            question = as_text(raw.get("question"))
            if not question:
                continue

            question_type = as_text(raw.get("question_type"))
            if not matches_question_type_filter(question_type, question_type_filter, file_path.name):
                continue

            question_id = as_text(raw.get("question_id"))
            if question_id and question_id in rows_by_question_id:
                print(f"[WARNING] duplicate question_id={question_id}")

            record = {
                "dataset_file": file_path.name,
                "excel_row_index": int(excel_row_index) + 2,
                "question_id": question_id,
                "group_id": as_text(raw.get("group_id")),
                "question_type": question_type,
                "question": question,
                "reference_answer": as_text(raw.get("reference_answer")),
                "source_document": as_text(raw.get("source_document"))
                or as_text(raw.get("reference_source_regulation")),
                "source_article": as_text(raw.get("source_article"))
                or as_text(raw.get("reference_article")),
                "evidence_text": as_text(raw.get("evidence_text"))
                or as_text(raw.get("reference_evidence")),
                "notes": as_text(raw.get("notes")),
            }
            key = question_id or f"{file_path.name}::row_{excel_row_index + 2}"
            rows_by_question_id[key] = record

    return list(rows_by_question_id.values())


def serialize_context(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": item.get("rank"),
        "text": as_text(item.get("text")),
        "file_name": as_text(item.get("file_name")),
        "doc_type": as_text(item.get("doc_type")),
        "path_text": as_text(item.get("path_text")),
        "page_start": as_text(item.get("page_start")),
        "page_end": as_text(item.get("page_end")),
        "score": float(item.get("score", 0.0) or 0.0),
        "vector_score": float(item.get("vector_score", 0.0) or 0.0),
        "keyword_score": float(item.get("keyword_score", 0.0) or 0.0),
    }


def build_output_record(
    question_row: dict[str, Any],
    retrieved_contexts: list[dict[str, Any]],
    generated_answer_text: str | None,
    retrieval_mode: str,
    hybrid_text_mode: str,
    embedding_model: str,
    ollama_model: str,
    top_k: int,
    prompt_top_n: int,
    max_context_chars: int,
    ollama_timeout: int,
    error: str | None,
) -> dict[str, Any]:
    return {
        "question_id": question_row.get("question_id"),
        "dataset_file": question_row.get("dataset_file"),
        "group_id": question_row.get("group_id"),
        "question_type": question_row.get("question_type"),
        "question": question_row.get("question"),
        "reference_answer": question_row.get("reference_answer"),
        "source_document": question_row.get("source_document"),
        "source_article": question_row.get("source_article"),
        "evidence_text": question_row.get("evidence_text"),
        "generated_answer": generated_answer_text,
        "retrieved_contexts": [serialize_context(item) for item in retrieved_contexts],
        "retrieval_mode": retrieval_mode,
        "hybrid_text_mode": hybrid_text_mode,
        "embedding_model": embedding_model,
        "ollama_model": ollama_model,
        "top_k": top_k,
        "prompt_top_n": prompt_top_n,
        "max_context_chars": max_context_chars,
        "ollama_timeout": ollama_timeout,
        "notes": question_row.get("notes"),
        "error": error,
        "updated_at": now_iso(),
    }


def write_all_records(output_path: Path, records_by_question_id: dict[str, dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig") as output_file:
        for record in records_by_question_id.values():
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_batch(
    dataset_dir: Path,
    output_path: Path,
    embedding_model: str,
    retrieval_mode: str,
    hybrid_text_mode: str,
    top_k: int,
    alpha: float,
    ollama_model: str,
    prompt_top_n: int,
    max_context_chars: int,
    ollama_timeout: int,
    limit: int | None,
    force_rerun: bool,
    question_type_filter: str | None,
    input_files: list[str] | None,
) -> None:
    question_rows = load_question_rows(dataset_dir, input_files, question_type_filter)
    if limit is not None:
        question_rows = question_rows[:limit]

    existing_records = load_jsonl_records(output_path)
    prompt_engineering.OLLAMA_TIMEOUT = ollama_timeout

    print(f"Loading embeddings: mode={retrieval_mode}, hybrid_text_mode={hybrid_text_mode}")
    doc_embeddings, metadata, _summary = load_embedding_data(retrieval_mode, hybrid_text_mode)

    print(f"Loading embedding model: {embedding_model}")
    model = load_model(embedding_model)

    print("Loading BM25 index")
    bm25 = load_bm25_index(
        retrieval_mode,
        hybrid_text_mode,
        tuple(str(item.get("text", "")) for item in metadata),
    )

    skipped_existing = 0
    newly_run = 0
    rerun_due_to_error = 0
    failed = 0

    for index, question_row in enumerate(question_rows, start=1):
        question_id = question_row.get("question_id") or f"row_{index}"
        existing_record = existing_records.get(question_id)

        if not force_rerun and existing_record and existing_record.get("error") is None:
            skipped_existing += 1
            print(f"[SKIP] question_id={question_id}")
            continue

        if not force_rerun and existing_record and existing_record.get("error") is not None:
            rerun_due_to_error += 1
            print(f"[RERUN_ERROR] question_id={question_id}")
        else:
            newly_run += 1
            print(f"[RUN] question_id={question_id}")

        retrieved_contexts: list[dict[str, Any]] = []
        generated_answer_text: str | None = None
        error: str | None = None
        context_chars = 0
        contexts_used = 0

        try:
            question_text = str(question_row["question"])
            question_a = {"raw_text": question_text}
            preprocess(question_a)

            retrieved_contexts = run_search(
                question_text,
                model,
                doc_embeddings,
                metadata,
                bm25,
                top_k,
                alpha,
            )
            prompt_contexts, context_chars = prepare_prompt_contexts(
                retrieved_contexts,
                prompt_top_n=prompt_top_n,
                max_context_chars=max_context_chars,
            )
            contexts_used = len(prompt_contexts)
            print(f"contexts_used={contexts_used}")
            print(f"context_chars={context_chars}")

            prompt = build_eval_prompt(question_text, prompt_contexts)
            generated_answer_text = as_text(
                call_ollama(prompt, model_name=ollama_model, timeout=ollama_timeout)
            )
            answer_error = detect_error_message(generated_answer_text)
            if answer_error:
                error = answer_error
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if generated_answer_text is None:
                generated_answer_text = error
            if contexts_used == 0 and retrieved_contexts:
                print(f"contexts_used={len(retrieved_contexts[:prompt_top_n])}")
            print(f"context_chars={context_chars}")
            print(f"[ERROR] question_id={question_id} -> {error}")
        if error:
            failed += 1

        existing_records[question_id] = build_output_record(
            question_row=question_row,
            retrieved_contexts=retrieved_contexts,
            generated_answer_text=generated_answer_text,
            retrieval_mode=retrieval_mode,
            hybrid_text_mode=hybrid_text_mode,
            embedding_model=embedding_model,
            ollama_model=ollama_model,
            top_k=top_k,
            prompt_top_n=prompt_top_n,
            max_context_chars=max_context_chars,
            ollama_timeout=ollama_timeout,
            error=error,
        )

    write_all_records(output_path, existing_records)

    print(f"total questions loaded: {len(question_rows)}")
    print(f"skipped existing: {skipped_existing}")
    print(f"newly run: {newly_run}")
    print(f"rerun due to previous error: {rerun_due_to_error}")
    print(f"failed: {failed}")
    print(f"output path: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation datasets through the existing RAG pipeline.")
    parser.add_argument("--dataset-dir", type=Path, default=EVALUATION_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", default=DEFAULT_RETRIEVAL_MODE)
    parser.add_argument("--hybrid-text-mode", default=DEFAULT_HYBRID_TEXT_MODE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--prompt-top-n", type=int, default=DEFAULT_PROMPT_TOP_N)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-timeout", type=int, default=DEFAULT_OLLAMA_TIMEOUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--input-files", nargs="+", default=None)
    args = parser.parse_args()

    run_batch(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        embedding_model=args.embedding_model,
        retrieval_mode=args.mode,
        hybrid_text_mode=args.hybrid_text_mode,
        top_k=args.top_k,
        alpha=args.alpha,
        ollama_model=args.ollama_model,
        prompt_top_n=args.prompt_top_n,
        max_context_chars=args.max_context_chars,
        ollama_timeout=args.ollama_timeout,
        limit=args.limit,
        force_rerun=args.force_rerun,
        question_type_filter=args.question_type,
        input_files=args.input_files,
    )


if __name__ == "__main__":
    main()
