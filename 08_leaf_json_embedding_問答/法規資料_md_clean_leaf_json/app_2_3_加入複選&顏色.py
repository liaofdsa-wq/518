"""
app_2_3.py： app_2_2.py 修改版，新增複選模式、UI 調整、RAG 即時顯示資料來源
  1. 模式選擇改為複選（multiselect），可同時選多種 embedding 合併搜尋
  2. UI 版面：LLM 回答區塊放在資料來源上方（用 st.empty 佔位）
  3. RAG 完成後立即顯示資料來源，不等 Ollama；Ollama 跑完再填回答
  - 有新增 prompt_engineering.py 中的指令：加上限制繁體中文輸出，這部份看你最後要不要統一一下指令

顏色：
  [COLOR-A] 搜尋結果卡片標題列背景色 → CSS .streamlit-expanderHeader
            位置：頁面底部 st.markdown(CUSTOM_CSS) 的 --card-header-bg
  [COLOR-B] 分數列文字顏色（向量/關鍵字/混合）
            位置：st.caption() 字串，可改成 st.markdown 並加 <span style='color:...'>
  [COLOR-C] 「Ollama 回答」標題顏色
            位置：st.markdown() 的 ## Ollama 回答，可加 unsafe_allow_html 改色
  [COLOR-D] 「搜尋結果」標題顏色  →  同上
  [COLOR-E] 模式 badge 顏色（顯示在每張卡片右側）
            位置：render_mode_badge() 函式裡的 background-color / color
"""
from __future__ import annotations

import json
import os
import pickle
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
from prompt_engineering import generate_answer


# ════════════════════════════════════════════
# 路徑 & 常數
# ════════════════════════════════════════════

ROOT            = Path(__file__).resolve().parent
DATA_ROOT       = ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"
DEFAULT_MODEL        = "BAAI/bge-m3"
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"
VECTOR_THRESHOLD     = 0.2
RRF_K                = 60

AVAILABLE_MODES: tuple[str, ...] = (
    "all_node",
    "leaf_with_ancestors",
    "table_hierarchy_leaf",
    "table_inner_row",
    "table_inner",
)

MODE_DIR: dict[str, str] = {
    "all_node":             "embedding_bge_m3_all_node",
    "leaf_with_ancestors":  "embedding_bge_m3_leaf_with_ancestors",
    "table_hierarchy_leaf": "embedding_bge_m3_table_hierarchy_leaf",
    "table_inner_row":      "embedding_bge_m3_table_inner_row",
    "table_inner":          "embedding_bge_m3_table_inner",
}

# [COLOR-E] 每種模式的 badge 背景色 / 文字色 ──────────────
# 格式：(背景色, 文字色)  HEX 或 CSS color 都可
MODE_BADGE_STYLE: dict[str, tuple[str, str]] = {
    "all_node":             ("#4A90D9", "#ffffff"),   # 藍
    "leaf_with_ancestors":  ("#27AE60", "#ffffff"),   # 綠
    "table_hierarchy_leaf": ("#8E44AD", "#ffffff"),   # 紫
    "table_inner_row":      ("#E67E22", "#ffffff"),   # 橘
    "table_inner":          ("#C0392B", "#ffffff"),   # 紅
}

# ════════════════════════════════════════════
# 頁面樣式（[COLOR-A] 在這裡）
# ════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ═══════════════════════════════════════════
   [COLOR-1] 「開始搜尋」按鈕 → 背景 #eccd00、文字白色
   hover 時略深，改 #d4b800 那行
   ─────────────────────────────────────────── */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
div[data-testid="stForm"] button[data-testid="baseButton-primaryFormSubmit"] {
    background-color: #eccd00 !important;
    color: #ffffff !important;
    border: none !important;
}
div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
div[data-testid="stForm"] button[data-testid="baseButton-primaryFormSubmit"]:hover {
    background-color: #d4b800 !important;
    color: #ffffff !important;
}

/* ═══════════════════════════════════════════
   [COLOR-2] multiselect 已選取的模式標籤 → 背景 #eccd00、文字白色
   ─────────────────────────────────────────── */
span[data-baseweb="tag"] {
    background-color: #eccd00 !important;
}
span[data-baseweb="tag"] span {
    color: #ffffff !important;
}
span[data-baseweb="tag"] svg {
    fill: #ffffff !important;
}

/* ═══════════════════════════════════════════
   [COLOR-3] 兩支 slider 的拇指 + 已滑過軌跡 → #eccd00
   未滑過軌跡維持預設灰色
   ─────────────────────────────────────────── */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #eccd00 !important;
    border-color: #eccd00 !important;
}
div[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background-color: #eccd00 !important;
}
div[data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {
    background-color: #eccd00 !important;
    border-color: #eccd00 !important;
}
div[data-testid="stSlider"] input[type="range"]::-moz-range-thumb {
    background-color: #eccd00 !important;
    border-color: #eccd00 !important;
}

/* ═══════════════════════════════════════════
   [COLOR-4] UI 灰色背景 → rgba(240,237,231,0.5)（#f0ede7 透明度50%）
   涵蓋：expander 標題、expander 外框、sidebar、metric 卡片
   ─────────────────────────────────────────── */
div[data-testid="stExpander"] > details > summary {
    background-color: rgba(240, 237, 231, 0.5) !important;
    border-radius: 6px;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(240, 237, 231, 0.8) !important;
    border-radius: 8px;
    margin-bottom: 8px;
}
section[data-testid="stSidebar"] > div:first-child {
    background-color: rgba(240, 237, 231, 0.5) !important;
}
div[data-testid="stMetric"] {
    background-color: rgba(240, 237, 231, 0.5) !important;
    border-radius: 8px;
    padding: 8px 12px;
}

</style>
"""


# ════════════════════════════════════════════
# Streamlit 啟動
# ════════════════════════════════════════════

def find_available_port(start_port: int = 8501, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No available port found between {start_port} and {start_port + max_attempts - 1}")


def open_browser_delayed(port: int, delay_seconds: float = 1.5) -> None:
    url = f"http://localhost:{port}"

    def _open() -> None:
        try:
            os.startfile(url) if os.name == "nt" else webbrowser.open(url)
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
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--server.port", str(port)]
        raise SystemExit(stcli.main())


# ════════════════════════════════════════════
# 資料載入
# ════════════════════════════════════════════

def load_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_single_embedding(mode: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """單一模式載入（結果被 Streamlit cache，不重複讀檔）"""
    if mode not in AVAILABLE_MODES:
        raise ValueError(f"不支援的模式：{mode}")

    d = EMBEDDINGS_ROOT / MODE_DIR[mode]
    emb_path  = d / "embeddings.npy"
    meta_path = d / "metadata.jsonl"
    sum_path  = d / "embedding_summary.json"

    for p in (emb_path, meta_path, sum_path):
        if not p.exists():
            raise FileNotFoundError(f"找不到檔案：{p}")

    embeddings = np.load(emb_path)
    metadata   = load_metadata(meta_path)
    summary    = json.loads(sum_path.read_text(encoding="utf-8-sig"))

    if len(embeddings) != len(metadata):
        raise RuntimeError(f"[{mode}] embedding 筆數 {len(embeddings)} ≠ metadata 筆數 {len(metadata)}")

    return embeddings, metadata, summary


def load_merged_embeddings(
    modes: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """
    合併多個模式的 embedding 與 metadata。
    每筆 metadata 加上 _source_mode 欄位，方便後續辨識來源。
    """
    all_emb:  list[np.ndarray]      = []
    all_meta: list[dict[str, Any]]  = []
    summaries: dict[str, Any]       = {}

    for mode in modes:
        emb, meta, summ = load_single_embedding(mode)
        # 加上來源模式標記
        tagged = [{**row, "_source_mode": mode} for row in meta]
        all_emb.append(emb)
        all_meta.extend(tagged)
        summaries[mode] = {
            "record_count": summ.get("record_count"),
            "embedding_dim": summ.get("embedding_dim"),
        }

    merged_emb = np.concatenate(all_emb, axis=0).astype(np.float32)
    merged_summary = {
        "modes": modes,
        "total_record_count": len(all_meta),
        "embedding_dim": int(merged_emb.shape[1]) if merged_emb.ndim == 2 else None,
        "per_mode": summaries,
    }
    return merged_emb, all_meta, merged_summary


# ════════════════════════════════════════════
# BM25
# ════════════════════════════════════════════

def _tokenize_2gram(text: str) -> list[str]:
    return [text[i:i + 2] for i in range(len(text) - 1)]


def _bm25_cache_path(modes_key: str) -> Path:
    return DATA_ROOT / f"bm25_index_{modes_key}.pkl"


@st.cache_resource(show_spinner=False)
def load_bm25_index(modes_key: str, metadata_texts: tuple) -> BM25Okapi:
    """
    modes_key 是以「-」串接的模式名稱，例如 "all_node-table_inner"，
    用於區分不同複選組合的快取檔。
    """
    cache_path = _bm25_cache_path(modes_key)

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
    query_tokens = list({tok for kw in keywords for tok in _tokenize_2gram(kw)})
    if not query_tokens:
        return np.zeros(bm25.corpus_size, dtype=np.float32)
    return bm25.get_scores(query_tokens).astype(np.float32)


# ════════════════════════════════════════════
# 工具函式
# ════════════════════════════════════════════

def load_ollama_models() -> list[str]:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=15)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def render_table_location(payload: dict[str, Any]) -> str:
    return (
        f"{payload.get('table_id', '')} / "
        f"r{payload.get('row_index', '')} / "
        f"c{payload.get('col_index', '')} / "
        f"k{payload.get('chunk_index', '')}"
    )


def render_extra_info(item: dict[str, Any]) -> str:
    mode    = item.get("record_kind", item.get("doc_type", ""))
    payload = item.get("payload", {})

    if mode in ("table_inner", "table_inner_row", "table_hierarchy_leaf"):
        return "\n".join([
            f"法規檔：{item.get('file_name', '')}",
            f"位置：{item.get('position', '')}",
            f"表格位置：{render_table_location(payload)}",
            f"頁碼：{item.get('page_start', '')} - {item.get('page_end', '')}",
            f"原始 Cell：{payload.get('original_cell_text', '')}",
        ])
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
        str(row.get("node_name", "")).strip()
        for row in context_chain
        if isinstance(row, dict) and str(row.get("node_name", "")).strip()
    )
    return "\n".join([
        f"法規檔：{item.get('file_name', '')}",
        f"標題鏈：{title_chain}",
        f"節點編號：{payload.get('path_key', '')}",
        f"頁碼：{item.get('page_start', '')} - {item.get('page_end', '')}",
    ])


def render_mode_badge(mode: str) -> str:
    """
    回傳 HTML badge 字串，顏色由 MODE_BADGE_STYLE 控制。
    [COLOR-E] → 修改 MODE_BADGE_STYLE dict（檔案頂部）
    """
    bg, fg = MODE_BADGE_STYLE.get(mode, ("#888888", "#ffffff"))
    return (
        f"<span style='"
        f"background-color:{bg};color:{fg};"
        f"padding:2px 8px;border-radius:12px;font-size:0.78em;"
        f"font-weight:600;margin-left:6px;"
        f"'>{mode}</span>"
    )


# ════════════════════════════════════════════
# run_search（混合搜索，支援合併後的 embedding）
# ════════════════════════════════════════════

def run_search(
    question: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    bm25: BM25Okapi,
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    # Step 1：前處理
    question_b     = preprocess({"raw_text": question.strip()})
    combined_query = " ".join(question_b["sub_questions"])
    keywords       = question_b["keywords"]

    # Step 2：向量搜索
    query_emb = model.encode(
        [combined_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    vector_scores: np.ndarray = doc_embeddings @ query_emb

    # Step 3：BM25
    bm25_scores: np.ndarray = _query_bm25_scores(keywords, bm25)

    # Step 4：門檻過濾 + RRF 合併
    valid_mask = vector_scores >= VECTOR_THRESHOLD
    v_max = float(vector_scores[valid_mask].max()) if valid_mask.any() else 1.0
    k_max = float(bm25_scores.max()) or 1.0
    v_norm = vector_scores / v_max
    k_norm = bm25_scores / k_max

    vector_rank_order = np.argsort(-vector_scores)
    bm25_rank_order = np.argsort(-bm25_scores)
    vector_ranks = np.empty(len(vector_scores), dtype=np.int32)
    bm25_ranks = np.empty(len(bm25_scores), dtype=np.int32)
    vector_ranks[vector_rank_order] = np.arange(1, len(vector_scores) + 1, dtype=np.int32)
    bm25_ranks[bm25_rank_order] = np.arange(1, len(bm25_scores) + 1, dtype=np.int32)

    hybrid: np.ndarray = (
        (1.0 / (rrf_k + vector_ranks.astype(np.float32)))
        + (1.0 / (rrf_k + bm25_ranks.astype(np.float32)))
    )
    hybrid[~valid_mask] = 0.0

    # Step 5：top_k
    indices = np.argsort(-hybrid)[:min(top_k, len(hybrid))]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices, start=1):
        if hybrid[idx] <= 0.0:
            break
        item = metadata[int(idx)]
        results.append({
            "rank":             rank,
            "score":            float(hybrid[idx]),
            "vector_score":     float(v_norm[idx]),
            "keyword_score":    float(k_norm[idx]),
            "preprocessed_query": combined_query,
            **item,
        })

    return results


# ════════════════════════════════════════════
# Session state 初始化
# ════════════════════════════════════════════

for _key, _default in [
    ("search_results",  []),
    ("search_question", ""),
    ("ollama_answer",   ""),
    ("question_input",  ""),
    ("submitted_query", ""),
    ("prompt_used",     ""),
    ("question_b",      {}),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ════════════════════════════════════════════
# 渲染函式（定義在 UI 流程之前，避免 NameError）
# ════════════════════════════════════════════

def _render_sources(results: list[dict[str, Any]], query_key: str) -> None:
    """
    渲染資料來源卡片。
    [COLOR-B] 分數文字：修改下方 st.caption() 的字串，
              或換成 st.markdown(..., unsafe_allow_html=True) 加 <span style='color:'>
    [COLOR-D] 「RAG 檢索結果」標題色：把 st.subheader 改成
              st.markdown("## <span style='color:#XXX'>RAG 檢索結果</span>",
                          unsafe_allow_html=True)
    """
    if not results:
        return

    st.subheader("RAG 檢索結果")   # [COLOR-D] 標題色改這行
    st.caption(f"問題：{st.session_state.search_question}")

    preprocessed = results[0].get("preprocessed_query", "")
    if preprocessed:
        st.caption(f"前處理後查詢：{preprocessed}")

    for item in results:
        source_mode = item.get("_source_mode", "")
        badge_html  = render_mode_badge(source_mode)   # [COLOR-E]
        title = (
            f"{item['rank']}. score={item['score']:.4f} | "
            f"{item.get('file_name', '')} | "
            f"{item.get('record_kind', item.get('doc_type', ''))}"
        )
        source_key = str(item.get("source_id", item["rank"]))
        widget_key = f"{query_key}_{item['rank']}_{source_key}"

        with st.expander(title, expanded=item["rank"] == 1):
            st.markdown(f"來源模式：{badge_html}", unsafe_allow_html=True)

            # [COLOR-B] 分數列：修改下方字串或改成 st.markdown + <span style='color:'>
            st.caption(
                f"向量分數: {item.get('vector_score', 0):.4f}　"
                f"關鍵字分數: {item.get('keyword_score', 0):.4f}　"
                f"RRF 分數: {item.get('score', 0):.4f}"
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


def _render_llm_answer(answer: str, prompt: str) -> None:
    """
    渲染 Ollama 回答區塊。
    [COLOR-C] 「Ollama 回答」標題色：
              把 st.subheader 改成
              st.markdown("## <span style='color:#XXX'>Ollama 回答</span>",
                          unsafe_allow_html=True)
    """
    st.subheader("Ollama 回答")   # [COLOR-C] 標題色改這行
    st.write(answer)
    if prompt:
        with st.expander("查看實際送出的 Prompt（debug）"):
            st.text_area("Prompt", value=prompt, height=400, key="prompt_display")


# ════════════════════════════════════════════
# UI 設定
# ════════════════════════════════════════════

st.set_page_config(page_title="法規檢索", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)   # 注入自訂 CSS（[COLOR-A]）
st.title("法規檢索")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    model_name = st.text_input("Embedding 模型", value=DEFAULT_MODEL)

    # ── 複選模式（需求 1）────────────────────────────────
    selected_modes: list[str] = st.multiselect(
        "Embedding 模式（可複選）",
        options=list(AVAILABLE_MODES),
        default=["all_node"],
        help=(
            "all_node：原始所有節點\n"
            "leaf_with_ancestors：葉節點＋祖先路徑\n"
            "table_hierarchy_leaf：表格葉節點＋路徑\n"
            "table_inner_row：表格轉成一段話\n"
            "table_inner：最細表格單元\n\n"
            "複選時會合併所有選定模式的 embedding 一起搜尋。"
        ),
    )
    if not selected_modes:
        st.warning("請至少選擇一種模式。")
        st.stop()

    top_k = st.slider("顯示前幾筆", 1, 20, 5)

    rrf_k = st.slider(
        "RRF k",
        min_value=1, max_value=200, value=RRF_K, step=1,
    )

    use_ollama = st.checkbox("用 Ollama 整理回答", value=False)
    ollama_models = load_ollama_models()
    if ollama_models:
        default_ollama = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in ollama_models else ollama_models[0]
        ollama_model = st.selectbox(
            "Ollama 模型", ollama_models,
            index=ollama_models.index(default_ollama),
            disabled=not use_ollama,
        )
    else:
        ollama_model = st.text_input("Ollama 模型", value=DEFAULT_OLLAMA_MODEL, disabled=not use_ollama)


# ── 載入合併 embedding ────────────────────────────────────
modes_key = "-".join(sorted(selected_modes))   # 唯一字串，用於 BM25 快取

try:
    doc_embeddings, metadata, summary = load_merged_embeddings(selected_modes)
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
            modes_key,
            tuple(str(item.get("text", "")) for item in metadata),
        )
except Exception as exc:
    st.error(f"建立 BM25 索引失敗: {exc}")
    st.stop()


# ── 統計指標 ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("資料筆數（合計）", f"{len(metadata):,}")
col2.metric("向量維度", str(doc_embeddings.shape[1]))
col3.metric("選用模式數", str(len(selected_modes)))

with st.expander("embedding 摘要"):
    st.json(summary)


# ── 搜尋表單 ──────────────────────────────────────────────
with st.form("search_form", clear_on_submit=False):
    question = st.text_area(
        "問題", key="question_input", height=100,
        placeholder="例如：資訊安全管理有哪些重點？",
    )
    submitted = st.form_submit_button("開始搜尋", type="primary")


# ════════════════════════════════════════════
# 搜尋流程（需求 2 & 3）
#
# 版面結構（由上到下）：
#   [llm_placeholder]   ← 先佔位，Ollama 跑完才填入
#   [sources_area]      ← RAG 完成立刻顯示
# ════════════════════════════════════════════

# 固定版面順序：先宣告兩個容器
llm_placeholder  = st.empty()    # Ollama 回答（上方，需求 2）
sources_area     = st.container() # 資料來源（下方）

if submitted:
    submitted_query = st.session_state.question_input.strip()
    st.session_state.submitted_query = submitted_query
    st.session_state.search_question = submitted_query
    st.session_state.search_results  = []
    st.session_state.ollama_answer   = ""
    st.session_state.prompt_used     = ""
    st.session_state.question_b      = {}

    if submitted_query:
        # ── Step 1：RAG 搜尋 ──────────────────────────
        with st.spinner("RAG 搜尋中..."):
            from preprocessing import preprocess as _preprocess
            st.session_state.question_b = _preprocess({"raw_text": submitted_query})
            st.session_state.search_results = run_search(
                submitted_query, model, doc_embeddings, metadata, bm25, top_k, rrf_k,
            )

        # ── Step 2：立刻顯示資料來源（需求 3）────────
        with sources_area:
            _render_sources(st.session_state.search_results, st.session_state.submitted_query)

        # ── Step 3：Ollama（需求 2：填入上方佔位）────
        if use_ollama and st.session_state.search_results:
            with llm_placeholder.container():
                with st.spinner("Ollama 生成回答中..."):
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
                        st.error(f"Ollama 生成失敗: {exc}")
    else:
        st.session_state.search_results = []


# ════════════════════════════════════════════
# 非 submitted 狀態下的顯示（session 保留結果）
# ════════════════════════════════════════════

if not submitted:
    # Ollama 回答（上）
    if st.session_state.ollama_answer:
        with llm_placeholder.container():
            _render_llm_answer(
                st.session_state.ollama_answer,
                st.session_state.prompt_used,
            )
    # 資料來源（下）
    if st.session_state.search_results:
        with sources_area:
            _render_sources(
                st.session_state.search_results,
                st.session_state.submitted_query,
            )
else:
    # submitted 時 Ollama 回答已寫入 session，補上渲染
    if st.session_state.ollama_answer:
        with llm_placeholder.container():
            _render_llm_answer(
                st.session_state.ollama_answer,
                st.session_state.prompt_used,
            )
