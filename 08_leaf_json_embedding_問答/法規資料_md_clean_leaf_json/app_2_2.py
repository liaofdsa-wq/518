"""
app_2_2.py： app_2_1.py 改版
- Embedding 模式改為五種（all_node / leaf_with_ancestors /
  table_hierarchy_leaf / table_inner_row / table_inner）
- 移除舊版 hybrid / leaf / table / 800200 分層邏輯
- 混合搜索（向量 + BM25）、正規化、Prompt 工程維持原版不動
- 有新增 prompt_engineering.py 中的指令：加上限制繁體中文輸出，這部份看你最後要不要統一一下指令
"""
from __future__ import annotations

import json
import os
import pickle
import re
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import requests
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from preprocessing import preprocess
from prompt_engineering import build_prompt, generate_answer


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"

# ── 五種新模式 ──────────────────────────────────────────
AVAILABLE_MODES = (
    "all_node",
    "leaf_with_ancestors",
    "table_hierarchy_leaf",
    "table_inner_row",
    "table_inner",
)

# 每種模式對應的資料夾名稱（與 build_embeddings.py 輸出一致）
MODE_DIR: dict[str, str] = {
    "all_node":             "embedding_bge_m3_all_node",
    "leaf_with_ancestors":  "embedding_bge_m3_leaf_with_ancestors",
    "table_hierarchy_leaf": "embedding_bge_m3_table_hierarchy_leaf",
    "table_inner_row":      "embedding_bge_m3_table_inner_row",
    "table_inner":          "embedding_bge_m3_table_inner",
}


# ════════════════════════════════════════════
# Streamlit 啟動邏輯（原版不動）
# ════════════════════════════════════════════

def find_available_port(start_port: int = 8501, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(
        f"No available port found between {start_port} and {start_port + max_attempts - 1}"
    )


def open_browser_delayed(port: int, delay_seconds: float = 1.5) -> None:
    url = f"http://localhost:{port}"

    def _open() -> None:
        try:
            if os.name == "nt":
                os.startfile(url)
            else:
                webbrowser.open(url)
        except Exception:
            webbrowser.open(url)

    threading.Timer(delay_seconds, _open).start()


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = None

    if get_script_run_ctx is None or get_script_run_ctx() is None:
        from streamlit.web import cli as stcli

        port = find_available_port()
        print(f"Opening Streamlit at http://localhost:{port}")
        open_browser_delayed(port)
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.port",
            str(port),
        ]
        raise SystemExit(stcli.main())


# ════════════════════════════════════════════
# 資料載入
# ════════════════════════════════════════════

def load_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_embedding_data(mode: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """
    依模式名稱載入對應的 embeddings.npy / metadata.jsonl / embedding_summary.json。
    資料夾結構由 build_embeddings.py 決定。
    """
    if mode not in AVAILABLE_MODES:
        raise ValueError(f"不支援的模式：{mode}，可選：{AVAILABLE_MODES}")

    embedding_dir = EMBEDDINGS_ROOT / MODE_DIR[mode]
    embedding_path = embedding_dir / "embeddings.npy"
    metadata_path  = embedding_dir / "metadata.jsonl"
    summary_path   = embedding_dir / "embedding_summary.json"

    if not embedding_path.exists():
        raise FileNotFoundError(f"找不到 embeddings 檔：{embedding_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 metadata 檔：{metadata_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"找不到 summary 檔：{summary_path}")

    embeddings = np.load(embedding_path)
    metadata   = load_metadata(metadata_path)
    summary    = json.loads(summary_path.read_text(encoding="utf-8-sig"))

    if len(embeddings) != len(metadata):
        raise RuntimeError(
            f"Embedding 筆數 {len(embeddings)} 與 metadata 筆數 {len(metadata)} 不一致"
        )

    return embeddings, metadata, summary


# ════════════════════════════════════════════
# 工具函式（原版不動）
# ════════════════════════════════════════════

def cosine_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = doc_embeddings @ query_embedding
    top_k = min(top_k, len(scores))
    indices = np.argsort(-scores)[:top_k]
    return indices, scores[indices]


def load_ollama_models() -> list[str]:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    return [item["name"] for item in data.get("models", []) if item.get("name")]


def render_table_location(payload: dict[str, Any]) -> str:
    return (
        f"{payload.get('table_id', '')} / "
        f"r{payload.get('row_index', '')} / "
        f"c{payload.get('col_index', '')} / "
        f"k{payload.get('chunk_index', '')}"
    )


def render_extra_info(item: dict[str, Any]) -> str:
    """顯示各模式的 metadata 欄位（對應五種新模式）"""
    mode     = item.get("record_kind", item.get("doc_type", ""))
    payload  = item.get("payload", {})

    # table_inner / table_inner_row / table_hierarchy_leaf（表格類）
    if mode in ("table_inner", "table_inner_row", "table_hierarchy_leaf"):
        return "\n".join([
            f"法規檔：{item.get('file_name', '')}",
            f"位置：{item.get('position', '')}",
            f"表格位置：{render_table_location(payload)}",
            f"頁碼：{item.get('page_start', '')} - {item.get('page_end', '')}",
            f"原始 Cell：{payload.get('original_cell_text', '')}",
        ])

    # all_node
    if mode == "all_node":
        return "\n".join([
            f"法規檔：{item.get('file_name', '')}",
            f"節點名稱：{payload.get('node_name', '')}",
            f"節點編號：{payload.get('path_key', '')}",
            f"節點路徑：{item.get('path_text', '')}",
            f"頁碼：{item.get('page_start', '')} - {item.get('page_end', '')}",
        ])

    # leaf_with_ancestors（預設）
    context_chain = payload.get("context_chain", [])
    title_chain = " > ".join(
        part
        for row in context_chain
        if isinstance(row, dict)
        for part in [str(row.get("node_name", "")).strip()]
        if part
    )
    return "\n".join([
        f"法規檔：{item.get('file_name', '')}",
        f"標題鏈：{title_chain}",
        f"節點編號：{payload.get('path_key', '')}",
        f"頁碼：{item.get('page_start', '')} - {item.get('page_end', '')}",
    ])


# ════════════════════════════════════════════
# BM25（原版不動，只改 cache key 不含 hybrid_text_mode）
# ════════════════════════════════════════════

def _tokenize_2gram(text: str) -> list[str]:
    return [text[i:i + 2] for i in range(len(text) - 1)]


def _bm25_cache_path(mode: str) -> Path:
    return DATA_ROOT / f"bm25_index_{mode}.pkl"


@st.cache_resource(show_spinner=False)
def load_bm25_index(mode: str, metadata_texts: tuple) -> BM25Okapi:
    """
    優先從 .pkl 讀取 BM25 索引；若不存在則建立並儲存。
    metadata_texts 傳 tuple 讓 Streamlit cache 能正確比對。
    """
    cache_path = _bm25_cache_path(mode)

    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    corpus = [_tokenize_2gram(text) for text in metadata_texts]
    bm25 = BM25Okapi(corpus)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(bm25, f)

    return bm25


def _query_bm25_scores(keywords: list[str], bm25: BM25Okapi) -> np.ndarray:
    query_tokens = []
    for kw in keywords:
        query_tokens.extend(_tokenize_2gram(kw))
    query_tokens = list(set(query_tokens))

    if not query_tokens:
        return np.zeros(bm25.corpus_size, dtype=np.float32)

    return bm25.get_scores(query_tokens).astype(np.float32)


# ════════════════════════════════════════════
# run_search（原版混合搜索邏輯不動）
# ════════════════════════════════════════════

VECTOR_THRESHOLD = 0.2


def run_search(
    question: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    bm25: BM25Okapi,
    top_k: int,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    # Step 1：前處理
    question_b    = preprocess({"raw_text": question.strip()})
    combined_query = " ".join(question_b["sub_questions"])
    keywords       = question_b["keywords"]

    # Step 2：向量搜索
    query_embedding = model.encode(
        [combined_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    vector_scores: np.ndarray = doc_embeddings @ query_embedding

    # Step 3：BM25
    bm25_scores: np.ndarray = _query_bm25_scores(keywords, bm25)

    # Step 4：門檻過濾 + 正規化 + 加權合併
    valid_mask = vector_scores >= VECTOR_THRESHOLD

    v_max = float(vector_scores[valid_mask].max()) if valid_mask.any() else 1.0
    k_max = float(bm25_scores.max()) or 1.0

    v_norm = vector_scores / v_max
    k_norm = bm25_scores / k_max

    hybrid: np.ndarray = alpha * v_norm + (1 - alpha) * k_norm
    hybrid[~valid_mask] = 0.0

    # Step 5：top_k
    top_k_actual = min(top_k, len(hybrid))
    indices = np.argsort(-hybrid)[:top_k_actual]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices, start=1):
        if hybrid[idx] <= 0.0:
            break
        item = metadata[int(idx)]
        results.append({
            "rank": rank,
            "score": float(hybrid[idx]),
            "vector_score": float(v_norm[idx]),
            "keyword_score": float(k_norm[idx]),
            "preprocessed_query": combined_query,
            **item,
        })

    return results


# ════════════════════════════════════════════
# UI
# ════════════════════════════════════════════

if "search_results"  not in st.session_state: st.session_state.search_results  = []
if "search_question" not in st.session_state: st.session_state.search_question = ""
if "ollama_answer"   not in st.session_state: st.session_state.ollama_answer   = ""
if "question_input"  not in st.session_state: st.session_state.question_input  = ""
if "submitted_query" not in st.session_state: st.session_state.submitted_query = ""
if "prompt_used"     not in st.session_state: st.session_state.prompt_used     = ""
if "question_b"      not in st.session_state: st.session_state.question_b      = {}


st.set_page_config(page_title="法規檢索", layout="wide")
st.title("法規檢索")
st.caption("支援五種 embedding 模式：all_node / leaf_with_ancestors / table_hierarchy_leaf / table_inner_row / table_inner")

with st.sidebar:
    model_name = st.text_input("Embedding 模型", value=DEFAULT_MODEL)

    mode = st.selectbox(
        "Embedding 模式",
        options=list(AVAILABLE_MODES),
        index=0,
        help=(
            "all_node：原始所有節點\n"
            "leaf_with_ancestors：葉節點＋祖先路徑\n"
            "table_hierarchy_leaf：表格葉節點＋路徑\n"
            "table_inner_row：表格轉成一段話\n"
            "table_inner：最細表格單元"
        ),
    )

    top_k = st.slider("顯示前幾筆", 1, 20, 5)

    alpha = st.slider(
        "向量 / 關鍵字 權重（越高越偏向量）",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    use_ollama = st.checkbox("用 Ollama 整理回答", value=False)
    ollama_models = load_ollama_models()
    if ollama_models:
        default_model = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in ollama_models else ollama_models[0]
        ollama_model = st.selectbox(
            "Ollama 模型",
            ollama_models,
            index=ollama_models.index(default_model),
            disabled=not use_ollama,
        )
    else:
        ollama_model = st.text_input("Ollama 模型", value=DEFAULT_OLLAMA_MODEL, disabled=not use_ollama)


# ── 載入 embedding 資料 ───────────────────────────────
try:
    doc_embeddings, metadata, summary = load_embedding_data(mode)
except Exception as exc:
    st.error(f"載入 embedding 失敗: {exc}")
    st.stop()

try:
    model = load_model(model_name)
except Exception as exc:
    st.error(f"載入模型失敗: {exc}")
    st.stop()

try:
    with st.spinner("載入 BM25 關鍵字索引..."):
        bm25 = load_bm25_index(
            mode,
            tuple(str(item.get("text", "")) for item in metadata),
        )
except Exception as exc:
    st.error(f"建立 BM25 索引失敗: {exc}")
    st.stop()


# ── 統計指標 ──────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("資料筆數", f"{len(metadata):,}")
col2.metric("向量維度", str(doc_embeddings.shape[1]))
col3.metric("模式", mode)

with st.expander("embedding 摘要"):
    st.json(summary)


# ── 搜尋表單 ──────────────────────────────────────────
with st.form("search_form", clear_on_submit=False):
    question = st.text_area(
        "問題",
        key="question_input",
        height=100,
        placeholder="例如：資訊安全管理有哪些重點？",
    )
    submitted = st.form_submit_button("開始搜尋", type="primary")

if submitted:
    submitted_query = st.session_state.question_input.strip()
    st.session_state.submitted_query = submitted_query
    st.session_state.search_question = submitted_query
    st.session_state.search_results  = []
    st.session_state.ollama_answer   = ""
    st.session_state.prompt_used     = ""
    st.session_state.question_b      = {}

    if submitted_query:
        with st.spinner("搜尋中..."):
            from preprocessing import preprocess as _preprocess
            st.session_state.question_b = _preprocess({"raw_text": submitted_query})

            st.session_state.search_results = run_search(
                submitted_query,
                model,
                doc_embeddings,
                metadata,
                bm25,
                top_k,
                alpha,
            )

        if use_ollama and st.session_state.search_results:
            with st.spinner("組裝 Prompt，呼叫 Ollama 中..."):
                try:
                    result = generate_answer(
                        question_a={"raw_text": submitted_query},
                        question_b=st.session_state.question_b,
                        candidates=st.session_state.search_results,
                        relation_notes="",
                        model_name=ollama_model,
                    )
                    st.session_state.ollama_answer = result["answer"]
                    st.session_state.prompt_used   = result["prompt"]
                except Exception as exc:
                    st.session_state.ollama_answer = ""
                    st.error(f"Ollama 生成失敗: {exc}")
    else:
        st.session_state.search_results = []


# ── 結果顯示 ──────────────────────────────────────────
if st.session_state.search_results:
    st.subheader("搜尋結果")
    st.caption(f"目前問題：{st.session_state.search_question}")
    st.caption(f"實際送出：{st.session_state.submitted_query}")

    preprocessed = st.session_state.search_results[0].get("preprocessed_query", "")
    if preprocessed:
        st.caption(f"前處理後查詢：{preprocessed}")

    query_key = st.session_state.submitted_query or "empty"
    for item in st.session_state.search_results:
        title = (
            f"{item['rank']}. score={item['score']:.4f} | "
            f"{item.get('file_name', '')} | "
            f"{item.get('record_kind', item.get('doc_type', ''))}"
        )
        source_key = str(item.get("source_id", item["rank"]))
        widget_key = f"{query_key}_{item['rank']}_{source_key}"
        with st.expander(title, expanded=item["rank"] == 1):
            st.caption(
                f"向量分數: {item.get('vector_score', 0):.4f}　"
                f"關鍵字分數: {item.get('keyword_score', 0):.4f}　"
                f"混合分數: {item.get('score', 0):.4f}"
            )
            st.text_area(
                f"emb內容 #{item['rank']}",
                value=str(item.get("text", "")),
                height=180,
                key=f"emb_{widget_key}",
            )
            st.text_area(
                f"其他資訊 #{item['rank']}",
                value=render_extra_info(item),
                height=180,
                key=f"meta_{widget_key}",
            )

if st.session_state.ollama_answer:
    st.subheader("Ollama 回答")
    st.write(st.session_state.ollama_answer)

    if st.session_state.prompt_used:
        with st.expander("查看實際送出的 Prompt（debug）"):
            st.text_area(
                "Prompt",
                value=st.session_state.prompt_used,
                height=400,
                key="prompt_display",
            )
