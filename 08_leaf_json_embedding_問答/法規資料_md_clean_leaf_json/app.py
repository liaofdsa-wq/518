from __future__ import annotations

import json
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"
AVAILABLE_MODES = ("hybrid", "leaf", "table", "all_nodes", "800200")
HYBRID_TEXT_OPTIONS = ("leaf", "all_nodes")


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


@st.cache_resource(show_spinner=False)
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


def load_ollama_models() -> list[str]:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    return [item["name"] for item in data.get("models", []) if item.get("name")]


def build_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[參考資料 {idx}]",
                    f"類型: {item.get('doc_type', '')}",
                    f"法規檔: {item.get('file_name', '')}",
                    f"路徑: {item.get('path_text', '')}",
                    f"頁碼: {item.get('page_start', '')} - {item.get('page_end', '')}",
                    "內容:",
                    str(item.get("text", "")),
                ]
            )
        )
    return f"""你是法規檢索助理，請只根據提供的內容回答問題，不要自行編造來源。

回答要求:
1. 優先整理與問題最直接相關的資訊。
2. 如果資料不足，直接說明不足，不要硬湊答案。
3. 如有需要，可在答案中引用法規檔名、節點編號或表格位置。

問題:
{question}

參考資料:
{chr(10).join(blocks)}
"""


def generate_with_ollama(prompt: str, model_name: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    if "response" not in data:
        raise RuntimeError(f"Ollama 回傳格式異常: {data}")
    return str(data["response"]).strip()


def render_table_location(payload: dict[str, Any]) -> str:
    return (
        f"{payload.get('table_id', '')} / "
        f"r{payload.get('row_index', '')} / "
        f"c{payload.get('col_index', '')} / "
        f"k{payload.get('chunk_index', '')}"
    )


def render_extra_info(item: dict[str, Any]) -> str:
    payload = item.get("payload", {})
    if item.get("doc_type") == "fixed_chunk_800_200":
        return "\n".join(
            [
                f"file_name: {item.get('file_name', '')}",
                f"chunk_index: {payload.get('chunk_index', '')}",
                f"char_range: {payload.get('char_start', '')} - {payload.get('char_end', '')}",
                f"page_range: {item.get('page_start', '')} - {item.get('page_end', '')}",
            ]
        )
    if item.get("doc_type") == "table_chunk":
        return "\n".join(
            [
                f"法規檔：{payload.get('file_name', '')}",
                f"節點編號：{payload.get('under_path_key', '')}",
                f"表格位置：{render_table_location(payload)}",
                f"頁碼：{item.get('page_start', '')} - {item.get('page_end', '')}",
                f"原始 Cell：{payload.get('original_cell_text', '')}",
            ]
        )

    if item.get("doc_type") == "all_node":
        return "\n".join(
            [
                f"法規檔：{item.get('file_name', '')}",
                f"節點名稱：{payload.get('node_name', '')}",
                f"節點編號：{payload.get('path_key', '')}",
                f"節點路徑：{item.get('path_text', '')}",
                f"頁碼：{item.get('page_start', '')} - {item.get('page_end', '')}",
            ]
        )

    context_chain = payload.get("context_chain", [])
    title_chain = " > ".join(
        part
        for row in context_chain
        if isinstance(row, dict)
        for part in [str(row.get("node_name", "")).strip()]
        if part
    )

    return "\n".join(
        [
            f"法規檔：{item.get('file_name', '')}",
            f"標題鏈：{title_chain}",
            f"節點編號：{payload.get('path_key', '')}",
            f"頁碼：{item.get('page_start', '')} - {item.get('page_end', '')}",
        ]
    )


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


if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_question" not in st.session_state:
    st.session_state.search_question = ""
if "ollama_answer" not in st.session_state:
    st.session_state.ollama_answer = ""
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "submitted_query" not in st.session_state:
    st.session_state.submitted_query = ""


st.set_page_config(page_title="法規檢索", layout="wide")
st.title("法規檢索")
st.caption("支援 leaf / table / hybrid embedding 查詢")

with st.sidebar:
    model_name = st.text_input("Embedding 模型", value=DEFAULT_MODEL)
    mode = st.selectbox("Embedding 模式", options=list(AVAILABLE_MODES), index=0)
    hybrid_text_mode = st.selectbox(
        "隨便模式法條來源",
        options=list(HYBRID_TEXT_OPTIONS),
        index=0,
        disabled=mode != "hybrid",
    )
    top_k = st.slider("顯示前幾筆", 1, 20, 5)
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

try:
    doc_embeddings, metadata, summary = load_embedding_data(mode, hybrid_text_mode)
except Exception as exc:
    st.error(f"載入 embedding 失敗: {exc}")
    st.stop()

try:
    model = load_model(model_name)
except Exception as exc:
    st.error(f"載入模型失敗: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("資料筆數", f"{len(metadata):,}")
col2.metric("向量維度", str(doc_embeddings.shape[1]))
col3.metric("模式", summary.get("mode", mode))

if mode == "hybrid":
    st.caption(f"隨便模式的法條文字來源：{hybrid_text_mode}")

with st.expander("embedding 摘要"):
    st.json(summary)

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
    st.session_state.search_results = []
    st.session_state.ollama_answer = ""

    if submitted_query:
        with st.spinner("搜尋中..."):
            st.session_state.search_results = run_search(
                st.session_state.submitted_query,
                model,
                doc_embeddings,
                metadata,
                top_k,
            )

        if use_ollama and st.session_state.search_results:
            with st.spinner("Ollama 整理中..."):
                try:
                    st.session_state.ollama_answer = generate_with_ollama(
                        build_prompt(st.session_state.submitted_query, st.session_state.search_results[:3]),
                        ollama_model,
                    )
                except Exception as exc:
                    st.session_state.ollama_answer = ""
                    st.error(f"Ollama 生成失敗: {exc}")
    else:
        st.session_state.search_results = []

if st.session_state.search_results:
    st.subheader("搜尋結果")
    st.caption(f"目前問題：{st.session_state.search_question}")
    st.caption(f"實際送出：{st.session_state.submitted_query}")
    query_key = st.session_state.submitted_query or "empty"
    for item in st.session_state.search_results:
        title = f"{item['rank']}. score={item['score']:.4f} | {item.get('file_name', '')} | {item.get('doc_type', '')}"
        source_key = str(item.get("source_id", item["rank"]))
        widget_key = f"{query_key}_{item['rank']}_{source_key}"
        with st.expander(title, expanded=item["rank"] == 1):
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
