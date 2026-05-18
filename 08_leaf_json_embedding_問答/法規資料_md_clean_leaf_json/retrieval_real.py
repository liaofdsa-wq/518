"""
Module 2: Retrieval 混合檢索
─────────────────────────────────────────
向量檢索：SentenceTransformer + .npy（與 app.py 同架構，不需額外向量資料庫）
關鍵字檢索：rank_bm25（BM25Okapi）

安裝套件：
    pip install sentence-transformers rank_bm25 numpy

執行方式：
    互動測試：python retrieval_real.py
    批次測試：python retrieval_real.py --batch

資料夾結構（與 app.py 相同）：
    your_project/
    ├── preprocessing.py
    ├── retrieval_real.py
    └── data/
        └── embeddings/
            └── embedding_bge_m3_{mode}/
                ├── embeddings.npy
                ├── metadata.jsonl
                └── embedding_summary.json
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from preprocessing import preprocess


# ════════════════════════════════════════════
# 設定（與 app.py 一致，可直接修改）
# ════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_MODE = "leaf"           # leaf / all_nodes / table / hybrid / 800200
HYBRID_TEXT_MODE = "leaf"       # hybrid 模式下的法條來源：leaf 或 all_nodes
ALPHA = 0.5                     # 向量權重；1-ALPHA = BM25 權重
TOP_K = 5                       # 預設回傳筆數
VECTOR_THRESHOLD = 0.2          # 原始向量分數低於此值視為不相關，直接排除


# ════════════════════════════════════════════
# 載入 embedding 資料（複用 app.py 相同邏輯）
# ════════════════════════════════════════════

def _load_metadata(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_single_mode(mode: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    emb_dir = EMBEDDINGS_ROOT / f"embedding_bge_m3_{mode}"
    emb_path = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "metadata.jsonl"

    if not emb_path.exists():
        raise FileNotFoundError(f"找不到 embeddings 檔案：{emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到 metadata 檔案：{meta_path}")

    embeddings = np.load(emb_path).astype(np.float32)
    metadata = _load_metadata(meta_path)

    if len(embeddings) != len(metadata):
        raise RuntimeError(
            f"embeddings ({len(embeddings)}) 與 metadata ({len(metadata)}) 數量不符"
        )
    return embeddings, metadata


def load_embedding_data(
    mode: str,
    hybrid_text_mode: str = HYBRID_TEXT_MODE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    對應 app.py 的 load_embedding_data，
    hybrid 模式合併 text + table embeddings。
    """
    if mode != "hybrid":
        return load_single_mode(mode)

    text_emb, text_meta = load_single_mode(hybrid_text_mode)
    table_emb, table_meta = load_single_mode("table")

    merged_emb = np.concatenate([text_emb, table_emb], axis=0)
    merged_meta = [*text_meta, *table_meta]
    return merged_emb, merged_meta


# ════════════════════════════════════════════
# 向量檢索（SentenceTransformer + cosine）
# ════════════════════════════════════════════

def build_vector_index(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """載入 SentenceTransformer 模型（第一次會下載，之後從 cache 讀）"""
    print(f"[向量] 載入模型：{model_name} ...")
    return SentenceTransformer(model_name)


def vector_search(
    query: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """
    回傳 cosine similarity 分數陣列（長度 = len(doc_embeddings)）
    """
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    scores: np.ndarray = doc_embeddings @ query_vec   # cosine（已正規化）
    return scores


# ════════════════════════════════════════════
# 關鍵字檢索（rank_bm25，2-gram tokenize）
# ════════════════════════════════════════════

def _tokenize_2gram(text: str) -> list[str]:
    """
    2-gram 切詞：讓 BM25 具備詞組鑑別力，
    避免單字元對任何中文文件都給高分的問題。
    '資訊安全' → ['資訊', '訊安', '安全']
    """
    return [text[i:i+2] for i in range(len(text) - 1)]


def build_bm25_index(metadata: list[dict[str, Any]]) -> BM25Okapi:
    """
    對 metadata 的 text 欄位建立 BM25 索引。
    第一次建立需要幾秒（視文件量）。
    """
    print("[BM25] 建立關鍵字索引 ...")
    corpus = [
        _tokenize_2gram(str(item.get("text", "")))
        for item in metadata
    ]
    return BM25Okapi(corpus)


def keyword_search(
    keywords: list[str],
    bm25: BM25Okapi,
    n_docs: int,
) -> np.ndarray:
    """
    回傳 BM25 分數陣列（長度 = n_docs）
    """
    query_tokens = []
    for kw in keywords:
        query_tokens.extend(_tokenize_2gram(kw))
    query_tokens = list(set(query_tokens))  # 去重

    if not query_tokens:
        return np.zeros(n_docs, dtype=np.float32)

    scores = bm25.get_scores(query_tokens).astype(np.float32)
    return scores


# ════════════════════════════════════════════
# 去重複
# ════════════════════════════════════════════

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def deduplicate(
    results: list[dict[str, Any]],
    threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    回傳 (保留清單, 移除清單)
    移除清單每筆包含 reason 和 similar_to
    """
    kept, removed_log = [], []
    for c in results:
        dup_of = None
        max_sim = 0.0
        for k in kept:
            sim = _jaccard(str(c.get("text", "")), str(k.get("text", "")))
            if sim > max_sim:
                max_sim = sim
                dup_of = k
        if max_sim >= threshold:
            removed_log.append({
                **c,
                "reason": f"與 rank {dup_of['rank']} 重複（Jaccard={max_sim:.2f}）",
                "similar_to_rank": dup_of["rank"],
            })
        else:
            kept.append(c)
    return kept, removed_log


# ════════════════════════════════════════════
# 去無意義段落
# ════════════════════════════════════════════

import re

MEANINGLESS_PATTERNS = [
    (r'^[\s　。，、…—\-─=＝\.·•]+$', '純符號或空白'),
    (r'^[\d\s\.\-─…]+$', '純數字或頁碼'),
    (r'^（?空白頁）?$', '空白頁標記'),
    (r'^表[一二三四五六七八九十\d]+$', '僅含表格標題'),
]
MIN_LENGTH = 20
MIN_CHINESE_RATIO = 0.2


def _meaningless_reason(text: str) -> str:
    stripped = text.strip()
    if len(stripped) < MIN_LENGTH:
        return f"文字過短（{len(stripped)} 字）"
    for pattern, reason in MEANINGLESS_PATTERNS:
        if re.fullmatch(pattern, stripped):
            return reason
    chinese = re.findall(r'[\u4e00-\u9fff]', stripped)
    if len(chinese) / len(stripped) < MIN_CHINESE_RATIO:
        return f"中文比例過低（{len(chinese)/len(stripped):.0%}）"
    return ""


def remove_meaningless(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, removed_log = [], []
    for c in results:
        reason = _meaningless_reason(str(c.get("text", "")))
        if reason:
            removed_log.append({**c, "reason": reason})
        else:
            kept.append(c)
    return kept, removed_log


# ════════════════════════════════════════════
# 主流程：retrieve()
# ════════════════════════════════════════════

def retrieve(
    question: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    bm25: BM25Okapi,
    top_k: int = TOP_K,
    alpha: float = ALPHA,
    vector_threshold: float = VECTOR_THRESHOLD,
) -> dict[str, Any]:
    """
    輸入：原始問題字串
    輸出 JSON：
    {
        "question_b": {...},           # 前處理結果
        "candidates": [...],           # 最終候選，給動態搜索算法
        "removed_duplicates": [...],   # 去重複移除的，含原因
        "removed_meaningless": [...]   # 去無意義移除的，含原因
    }
    """
    # ── Step 1：前處理 ─────────────────────────────────
    question_b = preprocess({"raw_text": question.strip()})
    combined_query = " ".join(question_b["sub_questions"])
    keywords = question_b["keywords"]

    # ── Step 2：向量分數 ───────────────────────────────
    v_scores = vector_search(combined_query, model, doc_embeddings, top_k)

    # 絕對門檻：原始 cosine < threshold 直接排除
    valid_mask = v_scores >= vector_threshold

    # ── Step 3：BM25 分數 ──────────────────────────────
    k_scores = keyword_search(keywords, bm25, len(metadata))

    # ── Step 4：正規化 + 加權合併 ─────────────────────
    v_max = float(v_scores[valid_mask].max()) if valid_mask.any() else 1.0
    k_max = float(k_scores.max()) or 1.0

    v_norm = v_scores / v_max
    k_norm = k_scores / k_max

    hybrid = alpha * v_norm + (1 - alpha) * k_norm
    hybrid[~valid_mask] = 0.0   # 門檻以下的歸零

    # ── Step 5：取候選（先多取一些，去重去無意義後再截） ─
    fetch_k = min(top_k * 3, len(hybrid))
    indices = np.argsort(-hybrid)[:fetch_k]

    raw_results = []
    for rank, idx in enumerate(indices, start=1):
        if hybrid[idx] <= 0.0:
            break
        item = metadata[int(idx)]
        raw_results.append({
            "rank": rank,
            "score": round(float(hybrid[idx]), 4),
            "vector_score": round(float(v_norm[idx]), 4),
            "keyword_score": round(float(k_norm[idx]), 4),
            "chunk_id": item.get("source_id", str(idx)),
            "source_doc": item.get("file_name", ""),
            "doc_type": item.get("doc_type", ""),
            "text": item.get("text", ""),
            "page_start": item.get("page_start", ""),
            "page_end": item.get("page_end", ""),
            "path_text": item.get("path_text", ""),
            "payload": item.get("payload", {}),
        })

    # ── Step 6：去無意義 → 去重複 → 截 top_k ──────────
    after_meaningful, removed_meaningless = remove_meaningless(raw_results)
    after_dedup, removed_duplicates = deduplicate(after_meaningful)

    # 重新編 rank
    final_candidates = []
    for new_rank, c in enumerate(after_dedup[:top_k], start=1):
        final_candidates.append({**c, "rank": new_rank})

    return {
        "question_b": question_b,
        "candidates": final_candidates,
        "removed_duplicates": removed_duplicates,
        "removed_meaningless": removed_meaningless,
    }


# ════════════════════════════════════════════
# 顯示輔助
# ════════════════════════════════════════════

def _print_result(result: dict):
    qb = result["question_b"]
    print(f"\n  正規化：{qb['normalized']}")
    print(f"  子問題：{qb['sub_questions']}")
    print(f"  關鍵字：{qb['keywords']}")

    print("\n── 最終候選段落 ─────────────────────────────")
    if not result["candidates"]:
        print("  （無符合門檻的結果）")
    for c in result["candidates"]:
        print(f"  [{c['rank']}] {c['chunk_id']} | {c['source_doc']} | {c['doc_type']}")
        print(f"       hybrid={c['score']}  vec={c['vector_score']}  kw={c['keyword_score']}")
        print(f"       {str(c['text'])[:80]}{'…' if len(str(c['text']))>80 else ''}")

    print("\n── 去無意義：移除清單 ───────────────────────")
    if not result["removed_meaningless"]:
        print("  （無移除）")
    for r in result["removed_meaningless"]:
        print(f"  ✗ {r['chunk_id']} | 原因：{r['reason']}")

    print("\n── 去重複：移除清單 ─────────────────────────")
    if not result["removed_duplicates"]:
        print("  （無移除）")
    for r in result["removed_duplicates"]:
        print(f"  ✗ {r['chunk_id']} | 原因：{r['reason']}")

    print("\n── 完整 JSON 輸出 ───────────────────────────")
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ════════════════════════════════════════════
# 互動 / 批次測試
# ════════════════════════════════════════════

def run_interactive(model, doc_embeddings, metadata, bm25):
    print("=" * 60)
    print("  Retrieval 正式版測試介面")
    print("  輸入 'q' 離開")
    print("=" * 60)
    while True:
        print()
        raw = input("請輸入問題 > ").strip()
        if raw.lower() in ("q", "quit", "exit", ""):
            print("離開。")
            break
        result = retrieve(raw, model, doc_embeddings, metadata, bm25)
        _print_result(result)


def run_batch_test(model, doc_embeddings, metadata, bm25):
    test_cases = [
        "根據此規範，內部控制制度的設計應考量哪五個控制因素？",
        "內部控制應考量那些因素？",
        "內部控制包含哪三種因素？",
        "根據此總則，內部稽核的主要目的為何？",
        "內部稽核可以達到何種效果？",
        "內稽的目為何？",
        "零用金管理作業中，對於零用金的設立目的與經管人員的職責有何規範？",
        "零用金因何目的而設立？管理人員應該要做甚麼事情管理零用金？",
        "零用金的設立和經管人員的職責有何相關？",
        "根據第二條，適用本規範的「資訊服務」具體包含哪三種服務形態？",
        "資訊服務有哪幾種型態？",
        "資訊服務包含哪五種服務型態？",
        "採用靜態密碼進行身分驗證時，密碼連續錯誤達幾次後，公司應進行妥善處理？",
        "靜態密碼進行身分驗證時，密碼不能連續錯誤達到幾次？",
        "採用靜態密碼進行身分驗證時，公司應進行妥善處理，代表密碼連續錯了幾次？",
        "金融主管當局制定此安全基準的主要目的為何？",
        "此安全基準由哪三個主要部分組成？其內容重點分別為何？",
        "在設備基準中，資訊中心對於「環境」選址有何具體要求？",
        "根據基準，電腦機房在「火災防範」與「滅火設備」上有何標準？",
        "營運基準如何規定「進出管理」與「人員識別」？",
        "在營運管理中，對於「存取權限」與「密碼管理」有哪些具體要求？",
        "對於「委外管理」，基準規定合約中應包含哪些重要條款？",
        "技術基準中，如何透過「資料保護」來防範洩漏或篡改？",
        "針對「非法存取」與「非法程式」，技術基準提供了哪些偵測對策？",
        "對於無人化服務區（如 ATM）的管理，營運基準有何重點要求？",
        "證卷商在訂定內部控制制度時，其「總則」部分至少應敘明哪些事項（請列舉三項）",
        ".根據此規範，內步控制制度的設計應考量哪五個控制因素？",
        "金融機構在執行「營運充急分析（BIA）」後，應產出哪些關鍵的分析結果？",
    ]
    print("=" * 60)
    print("  批次測試模式")
    print("=" * 60)
    for i, q in enumerate(test_cases, 1):
        print(f"\n{'='*60}\n  【測試 {i}】{q}\n{'='*60}")
        result = retrieve(q, model, doc_embeddings, metadata, bm25)
        _print_result(result)


# ════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════

if __name__ == "__main__":
    mode = DEFAULT_MODE

    print(f"[載入] embedding mode = {mode}")
    doc_embeddings, metadata = load_embedding_data(mode)

    model = build_vector_index(DEFAULT_MODEL)
    bm25 = build_bm25_index(metadata)

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        run_batch_test(model, doc_embeddings, metadata, bm25)
    else:
        run_interactive(model, doc_embeddings, metadata, bm25)
