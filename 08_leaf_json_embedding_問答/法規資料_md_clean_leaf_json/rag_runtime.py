from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from preprocessing import preprocess


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_TOP_K = 3
DEFAULT_PROMPT_TOP_N = 1
DEFAULT_MAX_CONTEXT_CHARS = 3000
DEFAULT_OLLAMA_TIMEOUT = 300
#========
TABLE_CHUNK_MIN_CHARS = 30
TABLE_ADJACENT_WINDOW = 1
TABLE_EXPANDED_MAX_CHARS = 2500
#========
#========
AVAILABLE_MODES = (
    "hybrid",
    "leaf",
    "table",
    "all_nodes",
    "800200",
    "all_node",
    "leaf_with_ancestors",
    "table_hierarchy_leaf",
    "table_inner",
    "table_inner_row",
)
#========
HYBRID_TEXT_OPTIONS = ("leaf", "all_nodes")
VECTOR_THRESHOLD = 0.2


def load_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def load_single_embedding_data(mode: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if mode not in AVAILABLE_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    embedding_dir = EMBEDDINGS_ROOT / f"embedding_bge_m3_{mode}"
    embedding_path = embedding_dir / "embeddings.npy"
    metadata_path = embedding_dir / "metadata.jsonl"
    summary_path = embedding_dir / "embedding_summary.json"

    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embedding_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

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


def _tokenize_2gram(text: str) -> list[str]:
    return [text[i:i + 2] for i in range(len(text) - 1)]

#==========
def _bm25_cache_path(mode: str, hybrid_text_mode: str) -> Path:
    if mode == "hybrid":
        return DATA_ROOT / f"bm25_index_{mode}_{hybrid_text_mode}.pkl"
    return DATA_ROOT / f"bm25_index_{mode}.pkl"
#==========

def load_bm25_index(mode: str, hybrid_text_mode: str, metadata: tuple[str, ...]) -> BM25Okapi:
    cache_path = _bm25_cache_path(mode, hybrid_text_mode)

    if cache_path.exists():
        with cache_path.open("rb") as file:
            return pickle.load(file)

    corpus = [_tokenize_2gram(text) for text in metadata]
    bm25 = BM25Okapi(corpus)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(bm25, file)

    return bm25


def _query_bm25_scores(keywords: list[str], bm25: BM25Okapi) -> np.ndarray:
    query_tokens: list[str] = []
    for keyword in keywords:
        query_tokens.extend(_tokenize_2gram(keyword))
    query_tokens = list(set(query_tokens))

    if not query_tokens:
        return np.zeros(bm25.corpus_size, dtype=np.float32)

    return bm25.get_scores(query_tokens).astype(np.float32)



#===============
def _same_table_neighbor(current: dict[str, Any], neighbor: dict[str, Any]) -> bool:
    """判斷相鄰 metadata 是否屬於同一份表格來源。"""
    if neighbor.get("doc_type") != "table_chunk":
        return False

    current_file = str(current.get("file_name", "") or "")
    neighbor_file = str(neighbor.get("file_name", "") or "")
    if current_file != neighbor_file:
        return False

    current_path = str(current.get("path_text", "") or "")
    neighbor_path = str(neighbor.get("path_text", "") or "")

    # 如果 path_text 兩邊都有值，就要求 path_text 相同，避免跨章節亂併。
    if current_path and neighbor_path:
        return current_path == neighbor_path

    return True


def _expand_table_chunk_text(
    metadata: list[dict[str, Any]],
    idx: int,
    window: int = TABLE_ADJACENT_WINDOW,
    max_chars: int = TABLE_EXPANDED_MAX_CHARS,
) -> str:
    """把 table chunk 與前後相鄰 chunk 合併，提升表格題 context 完整度。"""
    current = metadata[idx]
    collected: list[str] = []

    start = max(0, idx - window)
    end = min(len(metadata), idx + window + 1)

    for neighbor_idx in range(start, end):
        neighbor = metadata[neighbor_idx]
        if not _same_table_neighbor(current, neighbor):
            continue

        text = str(neighbor.get("text", "") or "").strip()
        if not text:
            continue

        relative_pos = neighbor_idx - idx
        if relative_pos == 0:
            label = "[matched chunk]"
        elif relative_pos < 0:
            label = f"[previous chunk {abs(relative_pos)}]"
        else:
            label = f"[next chunk {relative_pos}]"

        collected.append(f"{label}\n{text}")

    expanded_text = "\n\n".join(collected).strip()
    return expanded_text[:max_chars].strip()
#===============


#===============
def _tokenize_for_overlap(text: str) -> set[str]:
    """用簡單 2-gram 做中文 query-chunk overlap，避免依賴特定表格主題。"""
    normalized = str(text or "").strip()
    if not normalized:
        return set()
    return {normalized[i:i + 2] for i in range(len(normalized) - 1)}


def _table_generic_rerank_bonus(question: str, text: str) -> float:
    """通用型 table chunk reranking，不針對特定表格主題。

    加分邏輯：
    1. 與問題的字詞／2-gram 重疊越高，加分越多。
    2. chunk 資訊量越完整，加分越多。
    3. chunk 看起來像完整列、完整句或完整控制敘述，加分。
    4. 純標題、導引句、過短內容扣分。
    """
    q = str(question or "").strip()
    t = str(text or "").strip()

    if not t:
        return -1.0

    bonus = 0.0

    # 1. Query overlap：問題與 chunk 的 2-gram 重疊比例
    q_tokens = _tokenize_for_overlap(q)
    t_tokens = _tokenize_for_overlap(t)
    if q_tokens and t_tokens:
        overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
        bonus += min(overlap * 0.35, 0.25)

    # 2. Content density：避免標題型 chunk，偏好資訊量足夠的 chunk
    text_len = len(t)
    if text_len >= 150:
        bonus += 0.12
    elif text_len >= 80:
        bonus += 0.08
    elif text_len >= 50:
        bonus += 0.04
    elif text_len < 30:
        bonus -= 0.30

    # 3. Structural completeness：完整句、條列、欄位式內容加分
    punctuation_count = sum(t.count(p) for p in ("，", "。", "；", "：", "、", "\n"))
    if punctuation_count >= 4:
        bonus += 0.08
    elif punctuation_count >= 2:
        bonus += 0.04

    if any(marker in t for marker in ("1.", "2.", "3.", "（一）", "（二）", "（三）", "(一)", "(二)", "(三)", "一、", "二、", "三、")):
        bonus += 0.06

    # 4. Penalty：常見導引句或無實質內容的 chunk
    generic_low_value_phrases = (
        "不在此贅述",
        "本單元之作業程序及控制重點",
        "作業程序及控制重點已載於",
        "詳見",
        "如下表",
        "如下",
    )
    if any(phrase in t for phrase in generic_low_value_phrases):
        bonus -= 0.20

    return bonus
#===============

def run_search(
    question: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    bm25: BM25Okapi,
    top_k: int,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    question_b = preprocess({"raw_text": question.strip()})
    combined_query = " ".join(question_b["sub_questions"])
    keywords = question_b["keywords"]

    query_embedding = model.encode(
        [combined_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    vector_scores: np.ndarray = doc_embeddings @ query_embedding
    bm25_scores: np.ndarray = _query_bm25_scores(keywords, bm25)

    valid_mask = vector_scores >= VECTOR_THRESHOLD
    v_max = float(vector_scores[valid_mask].max()) if valid_mask.any() else 1.0
    k_max = float(bm25_scores.max()) or 1.0

    v_norm = vector_scores / v_max
    k_norm = bm25_scores / k_max

    hybrid: np.ndarray = alpha * v_norm + (1 - alpha) * k_norm
    hybrid[~valid_mask] = 0.0


    # ========================
    candidate_k = min(max(top_k * 10, 30), len(hybrid))
    candidate_indices = np.argsort(-hybrid)[:candidate_k]

    reranked_candidates: list[tuple[float, int]] = []

    for idx in candidate_indices:
        if hybrid[idx] <= 0.0:
            continue

        item = metadata[int(idx)]
        text = str(item.get("text", "") or "").strip()

        # 通用 table short chunk filtering
        if item.get("doc_type") == "table_chunk" and len(text) < TABLE_CHUNK_MIN_CHARS:
            continue

        rerank_score = float(hybrid[idx])

        # 通用 table reranking，不針對特定表格主題
        if item.get("doc_type") == "table_chunk":
            rerank_score += _table_generic_rerank_bonus(question, text)

        reranked_candidates.append((rerank_score, int(idx)))

    reranked_candidates.sort(key=lambda pair: pair[0], reverse=True)

    results: list[dict[str, Any]] = []
    rank = 1

    for rerank_score, idx in reranked_candidates:
        item = metadata[int(idx)]

        results.append(
            {
                "rank": rank,
                "score": float(hybrid[idx]),
                "rerank_score": float(rerank_score),
                "vector_score": float(v_norm[idx]),
                "keyword_score": float(k_norm[idx]),
                "preprocessed_query": combined_query,
                "source_index": int(idx),
                **item,
            }
        )

        rank += 1

        if len(results) >= top_k:
            break
    # ========================



    return results


def prepare_prompt_contexts(
    retrieved_contexts: list[dict[str, Any]],
    prompt_top_n: int,
    max_context_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    selected = retrieved_contexts[: max(prompt_top_n, 0)]
    prepared: list[dict[str, Any]] = []
    total_chars = 0

    for item in selected:
        file_name = str(item.get("file_name", "") or "").strip()
        path_text = str(item.get("path_text", "") or "").strip()
        text = str(item.get("text", "") or "").strip()

        header_lines = [
            part for part in (f"file_name: {file_name}" if file_name else "", f"path_text: {path_text}" if path_text else "")
            if part
        ]
        header = "\n".join(header_lines)
        reserved_chars = len(header) + (2 if header else 0)
        remaining_chars = max_context_chars - total_chars - reserved_chars
        if remaining_chars <= 0:
            break

        truncated_text = text[:remaining_chars].strip()
        if not truncated_text:
            continue

        prepared_item = dict(item)
        prepared_item["text"] = truncated_text
        prepared.append(prepared_item)

        block_text = f"{header}\n\n{truncated_text}" if header else truncated_text
        total_chars += len(block_text)
        if total_chars >= max_context_chars:
            break

    return prepared, total_chars


def build_eval_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        context_text = "(no retrieved context)"
    else:
        blocks: list[str] = []
        for idx, item in enumerate(contexts, start=1):
            lines = [f"[Context {idx}]"]
            file_name = str(item.get("file_name", "") or "").strip()
            path_text = str(item.get("path_text", "") or "").strip()
            text = str(item.get("text", "") or "").strip()
            if file_name:
                lines.append(f"file_name: {file_name}")
            if path_text:
                lines.append(f"path_text: {path_text}")
            lines.append("text:")
            lines.append(text)
            blocks.append("\n".join(lines))
        context_text = "\n\n".join(blocks)

    return (
        "你是法規問答助手。請只根據提供的檢索內容回答問題，保持精簡、直接，"
        "不要延伸發揮；若檢索內容不足以支持答案，請明確說明資訊不足。\n\n"
        f"問題：\n{question}\n\n"
        f"檢索內容：\n{context_text}\n\n"
        "請輸出最終答案："
    )
