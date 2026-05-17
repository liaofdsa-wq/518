from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"
AVAILABLE_MODES = ("hybrid", "leaf", "table", "all_nodes", "800200")
HYBRID_TEXT_OPTIONS = ("leaf", "all_nodes")


def load_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_single_embedding_data(mode: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if mode not in AVAILABLE_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    embedding_dir = EMBEDDINGS_ROOT / f"embedding_bge_m3_{mode}"
    embedding_path = embedding_dir / "embeddings.npy"
    metadata_path = embedding_dir / "metadata.jsonl"
    summary_path = embedding_dir / "embedding_summary.json"

    embeddings = np.load(embedding_path)
    metadata = load_metadata(metadata_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))

    if len(embeddings) != len(metadata):
        raise RuntimeError(
            f"Embedding count {len(embeddings)} does not match metadata count {len(metadata)}"
        )

    return embeddings, metadata, summary


def load_embedding_data(mode: str, hybrid_text_mode: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if mode != "hybrid":
        return load_single_embedding_data(mode)

    if hybrid_text_mode not in HYBRID_TEXT_OPTIONS:
        raise ValueError(f"Unsupported hybrid text mode: {hybrid_text_mode}")

    text_embeddings, text_metadata, text_summary = load_single_embedding_data(hybrid_text_mode)
    table_embeddings, table_metadata, table_summary = load_single_embedding_data("table")

    merged_embeddings = np.concatenate([text_embeddings, table_embeddings], axis=0)
    merged_metadata = [*text_metadata, *table_metadata]
    merged_summary = {
        "mode": "hybrid",
        "hybrid_text_mode": hybrid_text_mode,
        "record_count": len(merged_metadata),
        "embedding_dim": int(merged_embeddings.shape[1]) if merged_embeddings.ndim == 2 else None,
        "doc_type_counts": {
            "all_node": sum(1 for row in merged_metadata if row.get("doc_type") == "all_node"),
            "leaf": sum(1 for row in merged_metadata if row.get("doc_type") == "leaf"),
            "table_chunk": sum(1 for row in merged_metadata if row.get("doc_type") == "table_chunk"),
        },
        "sources": {
            "text_mode": hybrid_text_mode,
            "text_metadata": text_summary.get("files", {}).get("metadata"),
            "table_metadata": table_summary.get("files", {}).get("metadata"),
        },
    }
    return merged_embeddings, merged_metadata, merged_summary


def cosine_search(query_embedding: np.ndarray, doc_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = doc_embeddings @ query_embedding
    top_k = min(top_k, len(scores))
    indices = np.argsort(-scores)[:top_k]
    return indices, scores[indices]


def run_search(
    question: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    query_embedding = model.encode(
        [question.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    indices, scores = cosine_search(query_embedding, doc_embeddings, top_k)

    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        item = metadata[int(idx)]
        results.append({"rank": rank, "score": float(score), **item})
    return results


def build_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[參考資料{idx}]",
                    f"文件類型: {item.get('doc_type', '')}",
                    f"檔名: {item.get('file_name', '')}",
                    f"路徑: {item.get('path_text', '')}",
                    f"頁碼: {item.get('page_start', '')} - {item.get('page_end', '')}",
                    "內容:",
                    str(item.get("text", "")),
                ]
            )
        )
    return f"""你是法規問答助手。請根據提供的參考資料回答問題。

回答規則:
1. 只根據參考資料作答，不要自行補充無根據的資訊。
2. 若資料不足以完整回答，請明確說明資料不足。
3. 盡量用精簡、完整的繁體中文回答。

問題:
{question}

參考資料:
{chr(10).join(blocks)}
"""


def generate_with_ollama(prompt: str, model_name: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["response"]).strip()


def format_path(item: dict[str, Any]) -> str:
    payload = item.get("payload", {})
    page_start = item.get("page_start", "")
    page_end = item.get("page_end", "")
    path_text = item.get("path_text", "")
    file_name = item.get("file_name", "")
    doc_type = item.get("doc_type", "")
    extra = ""

    if doc_type == "table_chunk":
        extra = (
            f"table_id={payload.get('table_id', '')},"
            f"row={payload.get('row_index', '')},"
            f"col={payload.get('col_index', '')},"
            f"chunk={payload.get('chunk_index', '')}"
        )
    elif doc_type == "all_node":
        extra = f"node={payload.get('node_name', '')}"
    elif doc_type == "leaf":
        chain = payload.get("context_chain", [])
        node_names = [str(row.get("node_name", "")).strip() for row in chain if isinstance(row, dict)]
        node_names = [name for name in node_names if name]
        if node_names:
            extra = f"context={' > '.join(node_names)}"

    base = f"{file_name} | {path_text} | p.{page_start}-{page_end} | {doc_type}"
    return f"{base} | {extra}" if extra else base


def format_rag_text(items: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"[RAG選取資料{item.get('rank', '')}]",
                    f"score={item.get('score', '')}",
                    f"來源={format_path(item)}",
                    "文字:",
                    str(item.get("text", "")).strip(),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def read_questions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported questions file type: {path.suffix}")


def answer_questions(
    questions_path: Path,
    output_path: Path,
    embedding_model: str,
    mode: str,
    hybrid_text_mode: str,
    top_k: int,
    ollama_model: str,
) -> None:
    df = read_questions(questions_path)
    required_columns = {"問題", "標準答案"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    model = SentenceTransformer(embedding_model)
    doc_embeddings, metadata, _ = load_embedding_data(mode, hybrid_text_mode)

    existing_questions: set[str] = set()
    existing_rows_by_question: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        output_rows = existing_df.to_dict(orient="records")
        existing_questions = {
            str(row.get("問題", "")).strip()
            for row in output_rows
            if str(row.get("問題", "")).strip()
        }
        existing_rows_by_question = {
            str(row.get("問題", "")).strip(): row
            for row in output_rows
            if str(row.get("問題", "")).strip()
        }

    for idx, row in df.iterrows():
        question = str(row["問題"]).strip()
        gold = str(row["標準答案"]).strip()
        if question in existing_questions:
            existing_row = existing_rows_by_question.get(question, {})
            if not str(existing_row.get("RAG選取文字", "")).strip() or str(existing_row.get("RAG選取文字", "")).lower() == "nan":
                results = run_search(question, model, doc_embeddings, metadata, top_k)
                selected_results = results[:3]
                existing_row["RAG選取文字"] = format_rag_text(selected_results)
                if not str(existing_row.get("LLM答案路徑", "")).strip() or str(existing_row.get("LLM答案路徑", "")).lower() == "nan":
                    existing_row["LLM答案路徑"] = " || ".join(format_path(item) for item in selected_results)
                pd.DataFrame(output_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
                print(f"[{idx + 1}/{len(df)}] fill rag text")
                continue
            print(f"[{idx + 1}/{len(df)}] skip")
            continue

        results = run_search(question, model, doc_embeddings, metadata, top_k)
        selected_results = results[:3]
        prompt = build_prompt(question, selected_results)
        answer = generate_with_ollama(prompt, ollama_model)
        path_csv = " || ".join(format_path(item) for item in selected_results)
        rag_text = format_rag_text(selected_results)

        output_rows.append(
            {
                "問題": question,
                "標準答案": gold,
                "LLM答案": answer,
                "LLM答案路徑": path_csv,
                "RAG選取文字": rag_text,
            }
        )
        pd.DataFrame(output_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[{idx + 1}/{len(df)}] done")
        existing_questions.add(question)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--hybrid-text-mode", default="leaf")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ollama-model", default="qwen2.5:3b")
    args = parser.parse_args()

    answer_questions(
        questions_path=args.questions,
        output_path=args.output,
        embedding_model=args.embedding_model,
        mode=args.mode,
        hybrid_text_mode=args.hybrid_text_mode,
        top_k=args.top_k,
        ollama_model=args.ollama_model,
    )


if __name__ == "__main__":
    main()
