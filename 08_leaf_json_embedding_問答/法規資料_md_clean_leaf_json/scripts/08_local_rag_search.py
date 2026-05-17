from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "BAAI/bge-m3"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
EMBEDDING_INPUT_DIR = DATA_ROOT / "embedding_inputs"
RAG_INDEX_ROOT = DATA_ROOT / "rag_index" / "bge_m3_current"

INPUT_FILES = (
    "all_raw_data.jsonl",
    "all_nodes.jsonl",
    "leaf_with_ancestors.jsonl",
    "table_hierarchy_leaves.jsonl",
    "table_inner.jsonl",
    "table_inner_rows.jsonl",
)

DEFAULT_SEARCH_EXCLUDED_KINDS = {"all_raw_data"}

REQUIRED_FIELDS = (
    "id",
    "source_id",
    "file_name",
    "file_stem",
    "record_kind",
    "position",
    "covered_positions",
    "text",
    "metadata",
    "hash_algo",
    "file_hash",
    "position_parts",
    "position_hash",
    "covered_position_parts",
    "covered_position_hashes",
)


def stable_hash(value: str | None) -> str:
    return hashlib.blake2b(str(value or "").encode("utf-8"), digest_size=8).hexdigest()


def dedupe_keep_order(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def parse_include_kinds(value: str | None) -> set[str] | None:
    if not value:
        return None
    kinds = {part.strip() for part in value.split(",") if part.strip()}
    return kinds or None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            rows.append(row)
    return rows


def iter_input_records(input_dir: Path = EMBEDDING_INPUT_DIR) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_name in INPUT_FILES:
        path = input_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding input: {path}")
        for row in load_jsonl(path):
            row["_input_file"] = file_name
            records.append(row)
    return records


def position_hash_pairs(record: dict[str, Any]) -> list[tuple[str | None, str | None]]:
    pairs: list[tuple[str | None, str | None]] = [
        (record.get("position"), record.get("position_hash")),
    ]
    pairs.extend(zip(record.get("covered_positions") or [], record.get("covered_position_hashes") or []))
    pairs.extend(zip(record.get("ancestor_positions") or [], record.get("ancestor_position_hashes") or []))
    if record.get("table_position"):
        pairs.append((record.get("table_position"), record.get("table_position_hash")))
    return pairs


def check_inputs(input_dir: Path = EMBEDDING_INPUT_DIR) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "files": {},
        "record_count": 0,
        "record_kind_counts": {},
        "unique_position_hashes": 0,
        "duplicate_record_id_count": 0,
        "error_count": 0,
        "errors": [],
    }
    errors: list[dict[str, Any]] = []
    id_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    hash_to_position: dict[str, str] = {}

    for file_name in INPUT_FILES:
        path = input_dir / file_name
        file_count = 0
        if not path.exists():
            errors.append({"file": file_name, "line": None, "error": "missing_file", "detail": str(path)})
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                file_count += 1
                summary["record_count"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append({"file": file_name, "line": line_number, "error": "json", "detail": str(exc)})
                    continue
                if not isinstance(record, dict):
                    errors.append({"file": file_name, "line": line_number, "error": "not_object", "detail": ""})
                    continue

                record_id = str(record.get("id") or "")
                id_counts[record_id] += 1
                kind_counts[str(record.get("record_kind") or "")] += 1

                for field in REQUIRED_FIELDS:
                    if field not in record:
                        errors.append({"file": file_name, "line": line_number, "error": "missing_field", "detail": field})

                text = str(record.get("text") or "")
                if not text or not text.startswith("位置：") or "\n內容：" not in text:
                    errors.append({"file": file_name, "line": line_number, "error": "bad_text_format", "detail": record_id})

                covered_positions = record.get("covered_positions") or []
                covered_position_hashes = record.get("covered_position_hashes") or []
                if not covered_positions:
                    errors.append({"file": file_name, "line": line_number, "error": "empty_covered_positions", "detail": record_id})
                if len(covered_positions) != len(covered_position_hashes):
                    errors.append({"file": file_name, "line": line_number, "error": "covered_hash_count", "detail": record_id})

                for position, position_hash in position_hash_pairs(record):
                    if not position:
                        continue
                    expected_hash = stable_hash(str(position))
                    if position_hash != expected_hash:
                        errors.append(
                            {
                                "file": file_name,
                                "line": line_number,
                                "error": "hash_mismatch",
                                "detail": {"position": position, "actual": position_hash, "expected": expected_hash},
                            }
                        )
                    existing = hash_to_position.get(expected_hash)
                    if existing is not None and existing != position:
                        errors.append(
                            {
                                "file": file_name,
                                "line": line_number,
                                "error": "hash_collision",
                                "detail": {"hash": expected_hash, "left": existing, "right": position},
                            }
                        )
                    hash_to_position[expected_hash] = str(position)

        summary["files"][file_name] = file_count

    duplicate_ids = [record_id for record_id, count in id_counts.items() if record_id and count > 1]
    for record_id in duplicate_ids[:20]:
        errors.append({"file": None, "line": None, "error": "duplicate_record_id", "detail": record_id})

    summary["record_kind_counts"] = dict(kind_counts)
    summary["unique_position_hashes"] = len(hash_to_position)
    summary["duplicate_record_id_count"] = len(duplicate_ids)
    summary["error_count"] = len(errors)
    summary["errors"] = errors[:20]
    return summary, errors


def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def source_file_summary(input_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for file_name in INPUT_FILES:
        path = input_dir / file_name
        if not path.exists():
            continue
        stat = path.stat()
        result[file_name] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }
    return result


def build_index(
    input_dir: Path = EMBEDDING_INPUT_DIR,
    index_dir: Path = RAG_INDEX_ROOT,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
) -> dict[str, Any]:
    check_summary, errors = check_inputs(input_dir)
    if errors:
        raise RuntimeError(f"Embedding inputs failed validation: {len(errors)} errors")

    records = iter_input_records(input_dir)
    texts = [str(record.get("text") or "") for record in records]
    model = load_sentence_transformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    metadata_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        metadata_rows.append({"index": index, **record})

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    write_jsonl(index_dir / "metadata.jsonl", metadata_rows)

    summary = {
        "model_name": model_name,
        "record_count": len(metadata_rows),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "input_check": check_summary,
        "sources": source_file_summary(input_dir),
        "files": {
            "embeddings": str(index_dir / "embeddings.npy"),
            "metadata": str(index_dir / "metadata.jsonl"),
            "summary": str(index_dir / "index_summary.json"),
        },
    }
    (index_dir / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def load_index(index_dir: Path = RAG_INDEX_ROOT) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    embedding_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "metadata.jsonl"
    summary_path = index_dir / "index_summary.json"
    if not embedding_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"RAG index not found under {index_dir}; run --rebuild-index first.")

    embeddings = np.load(embedding_path)
    metadata = load_jsonl(metadata_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if len(embeddings) != len(metadata):
        raise RuntimeError(f"Embedding count {len(embeddings)} != metadata count {len(metadata)}")
    return embeddings, metadata, summary


def cosine_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int,
    include_kinds: set[str] | None,
) -> list[tuple[int, float]]:
    if include_kinds:
        candidate_indices = np.asarray(
            [index for index, row in enumerate(metadata) if row.get("record_kind") in include_kinds],
            dtype=np.int64,
        )
    else:
        candidate_indices = np.asarray(
            [
                index
                for index, row in enumerate(metadata)
                if row.get("record_kind") not in DEFAULT_SEARCH_EXCLUDED_KINDS
            ],
            dtype=np.int64,
        )

    if candidate_indices.size == 0:
        return []

    candidate_embeddings = doc_embeddings[candidate_indices]
    scores = candidate_embeddings @ query_embedding
    top_k = min(top_k, len(scores))
    local_indices = np.argsort(-scores)[:top_k]
    return [(int(candidate_indices[local_index]), float(scores[local_index])) for local_index in local_indices]


def support_positions_for_record(record: dict[str, Any]) -> list[str]:
    values: list[str | None] = []
    values.extend(record.get("covered_positions") or [])
    values.extend(record.get("ancestor_positions") or [])
    values.append(record.get("table_position"))
    return dedupe_keep_order(values)


def search_records(
    query: str,
    top_k: int,
    include_kinds: set[str] | None = None,
    index_dir: Path = RAG_INDEX_ROOT,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    embeddings, metadata, index_summary = load_index(index_dir)
    model = load_sentence_transformer(model_name)
    query_embedding = model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    matches = cosine_search(query_embedding, embeddings, metadata, top_k, include_kinds)
    results: list[dict[str, Any]] = []
    for rank, (metadata_index, score) in enumerate(matches, start=1):
        record = metadata[metadata_index]
        evidence_positions = list(record.get("covered_positions") or [])
        support_positions = support_positions_for_record(record)
        result = {
            "rank": rank,
            "score": score,
            "record_kind": record.get("record_kind"),
            "id": record.get("id"),
            "source_id": record.get("source_id"),
            "file_name": record.get("file_name"),
            "file_stem": record.get("file_stem"),
            "position": record.get("position"),
            "position_hash": record.get("position_hash"),
            "covered_positions": evidence_positions,
            "covered_position_hashes": record.get("covered_position_hashes") or [],
            "ancestor_positions": record.get("ancestor_positions") or [],
            "table_position": record.get("table_position"),
            "metadata": record.get("metadata") or {},
            "retrieval_text": record.get("text"),
            "evidence_positions": evidence_positions,
            "evidence_position_hashes": record.get("covered_position_hashes") or [],
            "support_positions": support_positions,
            "support_position_hashes": [stable_hash(position) for position in support_positions],
        }
        result["context_text"] = build_context_block(result)
        results.append(result)

    return {
        "query": query,
        "top_k": top_k,
        "include_kinds": sorted(include_kinds) if include_kinds else None,
        "index": {
            "record_count": index_summary.get("record_count"),
            "embedding_dim": index_summary.get("embedding_dim"),
            "model_name": index_summary.get("model_name", model_name),
        },
        "results": results,
        "combined_context_text": "\n\n---\n\n".join(result["context_text"] for result in results),
    }


def build_context_block(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    page_range = metadata.get("page_range") or {}
    line_range = metadata.get("line_range") or {}
    return "\n".join(
        [
            f"[RAG命中 {result.get('rank')}] score={result.get('score'):.6f}",
            f"record_kind: {result.get('record_kind')}",
            f"file: {result.get('file_name')}",
            f"position: {result.get('position')}",
            f"position_hash: {result.get('position_hash')}",
            f"page_range: {page_range.get('start')} - {page_range.get('end')}",
            f"line_range: {line_range.get('start')} - {line_range.get('end')}",
            "語意命中文字:",
            str(result.get("retrieval_text") or "").strip(),
            "evidence_positions:",
            json.dumps(result.get("evidence_positions") or [], ensure_ascii=False),
            "evidence_position_hashes:",
            json.dumps(result.get("evidence_position_hashes") or [], ensure_ascii=False),
            "support_positions:",
            json.dumps(result.get("support_positions") or [], ensure_ascii=False),
            "support_position_hashes:",
            json.dumps(result.get("support_position_hashes") or [], ensure_ascii=False),
        ]
    )


def print_search_text(payload: dict[str, Any]) -> None:
    print(f"query: {payload['query']}")
    print(f"top_k: {payload['top_k']}")
    if payload.get("include_kinds"):
        print(f"include_kinds: {', '.join(payload['include_kinds'])}")
    print(json.dumps(payload.get("index") or {}, ensure_ascii=False))
    print()
    print(payload["combined_context_text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG search over current embedding_inputs JSONL.")
    parser.add_argument("--query", help="Query text to search.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--include-kinds", help="Comma-separated record kinds to search.")
    parser.add_argument("--output-json", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=EMBEDDING_INPUT_DIR)
    parser.add_argument("--index-dir", type=Path, default=RAG_INDEX_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.check_inputs:
        summary, errors = check_inputs(args.input_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if errors:
            sys.exit(1)
        if not args.rebuild_index and not args.query:
            return

    if args.rebuild_index:
        summary = build_index(args.input_dir, args.index_dir, args.model, args.batch_size)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not args.query:
            return

    if not args.query:
        parser.error("Provide --query, --check-inputs, or --rebuild-index.")

    payload = search_records(
        query=args.query,
        top_k=args.top_k,
        include_kinds=parse_include_kinds(args.include_kinds),
        index_dir=args.index_dir,
        model_name=args.model,
    )
    if args.output_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_search_text(payload)


if __name__ == "__main__":
    main()
