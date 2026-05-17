from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "BAAI/bge-m3"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"
OUTPUT_MODE_NAME = "800200"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 8
DEFAULT_NORMALIZE = True
DEFAULT_SAMPLE_SIZE: int | None = None
DEFAULT_OUTPUT_SUFFIX = ""

PAGE_PATTERN = re.compile(r"\[page\s+(\d+)\]", re.IGNORECASE)


def resolve_source_root() -> Path:
    return PROJECT_ROOT.parents[1] / "04_Markdown精修區" / "法規資料_md"


SOURCE_ROOT = resolve_source_root()


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_page_markers(text: str) -> tuple[str, list[tuple[int, int]]]:
    cleaned_parts: list[str] = []
    page_markers: list[tuple[int, int]] = []
    cleaned_length = 0
    current_page: int | None = None
    last_end = 0

    for match in PAGE_PATTERN.finditer(text):
        segment = text[last_end : match.start()]
        if segment:
            cleaned_parts.append(segment)
            cleaned_length += len(segment)

        current_page = int(match.group(1))
        page_markers.append((cleaned_length, current_page))
        last_end = match.end()

    tail = text[last_end:]
    if tail:
        cleaned_parts.append(tail)
        cleaned_length += len(tail)

    cleaned_text = "".join(cleaned_parts)
    if current_page is not None and (not page_markers or page_markers[-1][0] != len(cleaned_text)):
        page_markers.append((len(cleaned_text), current_page))

    return cleaned_text, page_markers


def page_for_offset(page_markers: list[tuple[int, int]], offset: int) -> int | None:
    if not page_markers:
        return None

    current_page = page_markers[0][1]
    for marker_offset, page in page_markers:
        if marker_offset > offset:
            break
        current_page = page
    return current_page


def split_fixed_chunks(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    chunks: list[tuple[int, int, str]] = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((start, end, chunk_text))
        if end >= len(text):
            break
        start += step

    return chunks


def build_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_files = sorted(SOURCE_ROOT.glob("*.md"))

    for file_path in source_files:
        raw_text = normalize_text(read_text(file_path))
        if not raw_text:
            continue

        cleaned_text, page_markers = strip_page_markers(raw_text)
        cleaned_text = normalize_text(cleaned_text)
        if not cleaned_text:
            continue

        chunks = split_fixed_chunks(cleaned_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        for chunk_idx, (char_start, char_end, chunk_text) in enumerate(chunks, start=1):
            page_start = page_for_offset(page_markers, char_start)
            page_end = page_for_offset(page_markers, max(char_end - 1, char_start))
            records.append(
                {
                    "doc_type": "fixed_chunk_800_200",
                    "source_id": f"{file_path.stem}::chunk_{chunk_idx:04d}",
                    "file_name": file_path.name,
                    "path_text": "",
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": chunk_text,
                    "payload": {
                        "file_name": file_path.name,
                        "chunk_index": chunk_idx,
                        "char_start": char_start,
                        "char_end": char_end,
                        "chunk_size": DEFAULT_CHUNK_SIZE,
                        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                    },
                }
            )
            if DEFAULT_SAMPLE_SIZE is not None and len(records) >= DEFAULT_SAMPLE_SIZE:
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
        "page_start",
        "page_end",
        "chunk_index",
        "char_start",
        "char_end",
        "text_preview",
        "embedding_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, emb in zip(rows[:limit], embeddings[:limit]):
            payload = row.get("payload", {})
            writer.writerow(
                {
                    "doc_type": row["doc_type"],
                    "source_id": row["source_id"],
                    "file_name": row["file_name"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "chunk_index": payload.get("chunk_index"),
                    "char_start": payload.get("char_start"),
                    "char_end": payload.get("char_end"),
                    "text_preview": row["text"][:180].replace("\n", " "),
                    "embedding_preview": emb[:8].tolist(),
                }
            )


def main() -> None:
    suffix = f"_{DEFAULT_OUTPUT_SUFFIX}" if DEFAULT_OUTPUT_SUFFIX else ""
    output_dir = EMBEDDINGS_ROOT / f"embedding_bge_m3_{OUTPUT_MODE_NAME}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records()
    if not records:
        raise RuntimeError(f"No source records found under {SOURCE_ROOT}")

    model = SentenceTransformer(DEFAULT_MODEL)
    embeddings = model.encode(
        [row["text"] for row in records],
        batch_size=DEFAULT_BATCH_SIZE,
        normalize_embeddings=DEFAULT_NORMALIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    metadata_path = output_dir / "metadata.jsonl"
    embeddings_path = output_dir / "embeddings.npy"
    summary_path = output_dir / "embedding_summary.json"
    preview_path = output_dir / "embedding_preview.csv"

    np.save(embeddings_path, embeddings)
    write_jsonl(metadata_path, records)
    write_preview_csv(preview_path, records, embeddings)

    summary = {
        "mode": OUTPUT_MODE_NAME,
        "model": DEFAULT_MODEL,
        "source_root": str(SOURCE_ROOT),
        "record_count": len(records),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "normalize_embeddings": DEFAULT_NORMALIZE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "doc_type_counts": {
            "fixed_chunk_800_200": len(records),
        },
        "files": {
            "metadata": str(metadata_path),
            "embeddings": str(embeddings_path),
            "preview": str(preview_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built {len(records):,} fixed 800/200 chunks")
    print(f"Saved embeddings to {embeddings_path}")
    print("已完成")


if __name__ == "__main__":
    main()
