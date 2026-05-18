"""
build_embeddings.py：五種模式的 embedding 建立腳本
─────────────────────────────────────────────────────
五種模式（all_raw_data 不含）：
  - all_node
  - leaf_with_ancestors
  - table_hierarchy_leaf
  - table_inner_row
  - table_inner

每種模式各自讀取對應的 .jsonl，過濾 text 長度 ≤ 200 字（測試用），
分別存成獨立的資料夾，每個資料夾含：
  - embeddings.npy
  - metadata.jsonl
  - embedding_summary.json

執行方式（VSCode 終端機）：
  全部模式：python build_embeddings.py
  單一模式：python build_embeddings.py --mode all_node
  自訂路徑：python build_embeddings.py --input-dir /path/to/jsonl --output-dir /path/to/output
─────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


# ════════════════════════════════════════════
# 路徑設定（依你的專案結構調整）
# ════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 輸入：embedding_inputs 資料夾（放 .jsonl 的地方）
EMBEDDING_INPUT_DIR = DATA_ROOT / "embedding_inputs"

# 輸出：每種模式各一個子資料夾
EMBEDDINGS_OUTPUT_ROOT = DATA_ROOT / "embeddings"

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 8

# ── 模式 → 對應的 .jsonl 檔名 ──────────────────────
MODE_TO_FILE: dict[str, str] = {
    "all_node":             "all_nodes.jsonl",
    "leaf_with_ancestors":  "leaf_with_ancestors.jsonl",
    "table_hierarchy_leaf": "table_hierarchy_leaves.jsonl",
    "table_inner_row":      "table_inner_rows.jsonl",
    "table_inner":          "table_inner.jsonl",
}

ALL_MODES = list(MODE_TO_FILE.keys())

# 測試用：只 embed text 長度 ≤ 200 字的紀錄（設為 None 表示不過濾）
MAX_TEXT_LEN: int | None = 200


# ════════════════════════════════════════════
# 工具函式
# ════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [警告] {path.name}:{lineno} JSON 解析失敗：{exc}，略過")
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def extract_text(record: dict[str, Any]) -> str:
    """
    從紀錄中取出要 embed 的 text。
    五種模式的 jsonl 都有 "text" 欄位，直接取用。
    """
    return str(record.get("text") or "").strip()


# ════════════════════════════════════════════
# 單一模式 embedding 建立
# ════════════════════════════════════════════

def build_single_mode(
    mode: str,
    input_dir: Path,
    output_root: Path,
    model_name: str,
    batch_size: int,
    max_text_len: int | None,
) -> dict[str, Any]:
    """
    針對單一模式讀取 .jsonl、過濾、embed、儲存。
    回傳 summary dict。
    """
    file_name = MODE_TO_FILE[mode]
    input_path = input_dir / file_name
    output_dir = output_root / f"embedding_bge_m3_{mode}"

    print(f"\n{'='*60}")
    print(f"  模式：{mode}")
    print(f"  來源：{input_path}")
    print(f"  輸出：{output_dir}")
    print(f"{'='*60}")

    if not input_path.exists():
        raise FileNotFoundError(f"找不到輸入檔案：{input_path}")

    # ── 讀取 & 過濾 ──────────────────────────────
    all_records = load_jsonl(input_path)
    print(f"  讀入紀錄數：{len(all_records)}")

    if max_text_len is not None:
        filtered = [r for r in all_records if len(extract_text(r)) <= max_text_len]
        print(f"  過濾後（text ≤ {max_text_len} 字）：{len(filtered)} 筆")
    else:
        filtered = all_records
        print(f"  不過濾，全部 {len(filtered)} 筆")

    if not filtered:
        print("  [警告] 過濾後無資料，跳過此模式。")
        return {"mode": mode, "record_count": 0, "skipped": True}

    texts = [extract_text(r) for r in filtered]

    # ── 載入模型 & Embed ──────────────────────────
    print(f"  載入模型：{model_name} ...")
    model = load_sentence_transformer(model_name)

    print(f"  開始 embedding（batch_size={batch_size}）...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"  Embedding 完成，耗時 {elapsed:.1f}s，維度 {embeddings.shape}")

    # ── 組 metadata ───────────────────────────────
    metadata_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(filtered):
        metadata_rows.append({"index": idx, **record})

    # ── 儲存 ──────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"
    summary_path = output_dir / "embedding_summary.json"

    np.save(embeddings_path, embeddings)
    write_jsonl(metadata_path, metadata_rows)

    summary = {
        "mode": mode,
        "source_file": file_name,
        "model_name": model_name,
        "record_count": len(metadata_rows),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "max_text_len_filter": max_text_len,
        "elapsed_seconds": round(elapsed, 2),
        "files": {
            "embeddings": str(embeddings_path),
            "metadata": str(metadata_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"  ✓ 儲存完成：{output_dir}")
    return summary


# ════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="為五種 embedding 模式各自建立 .npy 索引（測試用：只 embed ≤200 字的 text）"
    )
    parser.add_argument(
        "--mode",
        choices=ALL_MODES + ["all"],
        default="all",
        help="指定要建立的模式，不指定則全部跑（預設 all）",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=EMBEDDING_INPUT_DIR,
        help=f"embedding_inputs 資料夾路徑（預設：{EMBEDDING_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EMBEDDINGS_OUTPUT_ROOT,
        help=f"embeddings 輸出根目錄（預設：{EMBEDDINGS_OUTPUT_ROOT}）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer 模型名稱（預設：{DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size（預設：{DEFAULT_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="不過濾 text 長度，embed 所有紀錄（預設：只 embed ≤200 字）",
    )
    args = parser.parse_args()

    modes_to_run = ALL_MODES if args.mode == "all" else [args.mode]
    max_len = None if args.no_filter else MAX_TEXT_LEN

    print(f"模式清單：{modes_to_run}")
    print(f"輸入目錄：{args.input_dir}")
    print(f"輸出目錄：{args.output_dir}")
    print(f"模型：{args.model}")
    print(f"文字長度過濾：{'不過濾' if max_len is None else f'≤ {max_len} 字'}")

    all_summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    for mode in modes_to_run:
        try:
            summary = build_single_mode(
                mode=mode,
                input_dir=args.input_dir,
                output_root=args.output_dir,
                model_name=args.model,
                batch_size=args.batch_size,
                max_text_len=max_len,
            )
            all_summaries.append(summary)
        except Exception as exc:
            msg = f"[錯誤] 模式 {mode} 失敗：{exc}"
            print(msg)
            errors.append(msg)

    # ── 彙總輸出 ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  所有模式建立結果：")
    for s in all_summaries:
        if s.get("skipped"):
            print(f"  {s['mode']:<25} SKIPPED（無資料）")
        else:
            print(
                f"  {s['mode']:<25} "
                f"{s.get('record_count', 0):>6} 筆  "
                f"dim={s.get('embedding_dim')}  "
                f"耗時 {s.get('elapsed_seconds', 0):.1f}s"
            )
    if errors:
        print("\n  以下模式發生錯誤：")
        for e in errors:
            print(f"  {e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
