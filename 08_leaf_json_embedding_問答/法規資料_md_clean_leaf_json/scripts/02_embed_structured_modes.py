from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-m3"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"
SUMMARY_DIR = DATA_ROOT / "summary"
DEFAULT_MODE = "all_nodes"  # "all_nodes" / "leaf" / "table" / "hybrid"
DEFAULT_BATCH_SIZE = 8
DEFAULT_NORMALIZE = True
DEFAULT_SAMPLE_SIZE: int | None = None
DEFAULT_OUTPUT_SUFFIX = ""


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return []


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " / ".join(part for item in value if (part := clean_text(item)))
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def parent_path_keys(path_key: str) -> list[str]:
    parts = [part for part in clean_text(path_key).split(".") if part]
    return [".".join(parts[:idx]) for idx in range(1, len(parts))]


def node_payload(file_name: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "node_name": row.get("node_name"),
        "path_key": row.get("path_key"),
        "path_names": row.get("path_names", []),
        "pages": row.get("pages", []),
        "content": row.get("content", ""),
        "sibling_order": row.get("sibling_order"),
    }


def compose_leaf_context(
    file_name: str,
    key_map: dict[str, dict[str, Any]],
    leaf_row: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    leaf_key = clean_text(leaf_row.get("path_key"))
    ancestor_rows = [key_map[key] for key in parent_path_keys(leaf_key) if key in key_map]
    context_rows = ancestor_rows + [leaf_row]

    context_lines: list[str] = []
    payload_chain: list[dict[str, Any]] = []

    for row in context_rows:
        text = clean_text(row.get("content"))
        if not text:
            continue
        context_lines.append(text)
        payload_chain.append(node_payload(file_name, row))

    return "\n\n".join(context_lines), payload_chain


def compose_table_text(row: dict[str, Any]) -> str:
    return clean_text(row.get("chunk_text"))


def build_all_node_records(sample_size: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    all_node_rows = load_json(SUMMARY_DIR / "all_nodes.json")

    for row in all_node_rows:
        file_name = clean_text(row.get("file_name"))
        if not file_name:
            continue

        records.append(
            {
                "doc_type": "all_node",
                "source_id": f"{Path(file_name).stem}::{row.get('path_key', '')}",
                "file_name": file_name,
                "path_text": " > ".join(row.get("path_names", [])),
                "page_start": min(row.get("pages", []) or [0]) or None,
                "page_end": max(row.get("pages", []) or [0]) or None,
                "text": clean_text(row.get("content")),
                "payload": node_payload(file_name, row),
            }
        )

        if sample_size is not None and len(records) >= sample_size:
            return records

    return records


def build_leaf_records(sample_size: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    all_node_rows = load_json(SUMMARY_DIR / "all_nodes.json")
    leaf_rows = load_json(SUMMARY_DIR / "all_leaf_nodes.json")
    all_nodes_by_file: dict[str, list[dict[str, Any]]] = {}
    for row in all_node_rows:
        file_name = clean_text(row.get("file_name"))
        if not file_name:
            continue
        all_nodes_by_file.setdefault(file_name, []).append(row)

    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for row in leaf_rows:
        file_name = clean_text(row.get("file_name"))
        if not file_name:
            continue
        rows_by_file.setdefault(file_name, []).append(row)

    for file_name, file_rows in rows_by_file.items():
        source_rows = all_nodes_by_file.get(file_name, file_rows)
        key_map = {
            clean_text(row.get("path_key")): row
            for row in source_rows
            if clean_text(row.get("path_key"))
        }

        for row in file_rows:
            text, context_chain = compose_leaf_context(file_name, key_map, row)
            records.append(
                {
                    "doc_type": "leaf",
                    "source_id": f"{Path(file_name).stem}::{row.get('path_key', '')}",
                    "file_name": file_name,
                    "path_text": " > ".join(row.get("path_names", [])),
                    "page_start": min(row.get("pages", []) or [0]) or None,
                    "page_end": max(row.get("pages", []) or [0]) or None,
                    "text": text,
                    "payload": {
                        **node_payload(file_name, row),
                        "context_chain": context_chain,
                    },
                }
            )

            if sample_size is not None and len(records) >= sample_size:
                return records

    return records


def build_records(mode: str, sample_size: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if mode == "all_nodes":
        records.extend(build_all_node_records(sample_size))

    if mode == "leaf":
        records.extend(build_leaf_records(sample_size))

    if mode == "hybrid":
        records.extend(build_all_node_records(None))
        records.extend(build_leaf_records(None))

    if mode in {"table", "hybrid"}:
        table_rows = load_json(SUMMARY_DIR / "all_table_chunks.json")
        for row in table_rows:
            under_path_key = clean_text(row.get("under_path_key"))
            table_id = clean_text(row.get("table_id"))
            row_index = row.get("row_index", "")
            col_index = row.get("col_index", "")
            chunk_index = row.get("chunk_index", "")
            source_id = f"{under_path_key}.{table_id}.r{row_index}.c{col_index}.k{chunk_index}"

            records.append(
                {
                    "doc_type": "table_chunk",
                    "source_id": source_id,
                    "file_name": row.get("file_name"),
                    "path_text": under_path_key,
                    "page_start": min(row.get("pages", []) or [0]) or None,
                    "page_end": max(row.get("pages", []) or [0]) or None,
                    "text": compose_table_text(row),
                    "payload": dict(row),
                }
            )
            if sample_size is not None and len(records) >= sample_size:
                return records

    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview_csv(path: Path, rows: list[dict[str, Any]], embeddings: np.ndarray, limit: int = 20) -> None:
    fieldnames = [
        "doc_type",
        "source_id",
        "file_name",
        "path_text",
        "page_start",
        "page_end",
        "text_preview",
        "embedding_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, emb in zip(rows[:limit], embeddings[:limit]):
            writer.writerow(
                {
                    "doc_type": row["doc_type"],
                    "source_id": row["source_id"],
                    "file_name": row["file_name"],
                    "path_text": row["path_text"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "text_preview": row["text"][:180].replace("\n", " "),
                    "embedding_preview": emb[:8].tolist(),
                }
            )


def resolve_settings() -> dict[str, Any]:
    return {
        "model": DEFAULT_MODEL,
        "batch_size": DEFAULT_BATCH_SIZE,
        "normalize": DEFAULT_NORMALIZE,
        "mode": DEFAULT_MODE,
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "output_suffix": DEFAULT_OUTPUT_SUFFIX,
    }


def main() -> None:
    settings = resolve_settings()
    suffix = f"_{settings['output_suffix']}" if clean_text(settings["output_suffix"]) else ""
    output_dir = EMBEDDINGS_ROOT / f"embedding_bge_m3_{settings['mode']}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(settings["mode"], settings["sample_size"])
    if not records:
        raise RuntimeError("No records found for embedding.")

    texts = [record["text"] for record in records]

    print(f"使用模型: {settings['model']}")
    print(f"目前模式: {settings['mode']}")
    model = SentenceTransformer(settings["model"])

    print(f"開始產生 embedding，共 {len(texts)} 筆，mode={settings['mode']}")
    embeddings = model.encode(
        texts,
        batch_size=settings["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=settings["normalize"],
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    metadata_rows = []
    for idx, record in enumerate(records):
        metadata_rows.append(
            {
                "index": idx,
                "doc_type": record["doc_type"],
                "source_id": record["source_id"],
                "file_name": record["file_name"],
                "path_text": record["path_text"],
                "page_start": record["page_start"],
                "page_end": record["page_end"],
                "text": record["text"],
                "payload": record["payload"],
            }
        )

    np.save(output_dir / "embeddings.npy", embeddings)
    write_jsonl(output_dir / "metadata.jsonl", metadata_rows)
    write_preview_csv(output_dir / "embedding_preview.csv", records, embeddings)

    summary = {
        "model_name": settings["model"],
        "mode": settings["mode"],
        "record_count": len(records),
        "sample_size": settings["sample_size"],
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "doc_type_counts": {
            "all_node": sum(1 for r in records if r["doc_type"] == "all_node"),
            "leaf": sum(1 for r in records if r["doc_type"] == "leaf"),
            "table_chunk": sum(1 for r in records if r["doc_type"] == "table_chunk"),
        },
        "sources": {
            "summary_dir": str(SUMMARY_DIR),
        },
        "files": {
            "embeddings": str(output_dir / "embeddings.npy"),
            "metadata": str(output_dir / "metadata.jsonl"),
            "preview_csv": str(output_dir / "embedding_preview.csv"),
        },
    }
    (output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("已完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
