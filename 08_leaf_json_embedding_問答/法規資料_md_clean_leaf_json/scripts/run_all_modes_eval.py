"""
run_all_modes_eval.py：一次跑完五種模式的 RAG 產答 + 四種指標評估
─────────────────────────────────────────────────────────────────
執行方式：
  全部跑：
    python run_all_modes_eval.py

  只跑指定模式：
    python run_all_modes_eval.py --modes all_node leaf_with_ancestors

  只跑 RAG 產答（不跑指標）：
    python run_all_modes_eval.py --skip-eval

  只跑指標（已有 eval_outputs.jsonl）：
    python run_all_modes_eval.py --skip-rag

  強制重跑所有問題（忽略已有紀錄）：
    python run_all_modes_eval.py --force-rerun

  限制題數（測試用）：
    python run_all_modes_eval.py --limit 5
─────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT        = SCRIPTS_DIR.parent   # 若此腳本放在 scripts/ 子資料夾，改成 .parent；
                                   # 若放在專案根目錄，改成 SCRIPTS_DIR

AVAILABLE_MODES = (
    "all_node",
    "leaf_with_ancestors",
    "table_hierarchy_leaf",
    "table_inner_row",
    "table_inner",
)

# 輸出資料夾根目錄（每種模式各一個子資料夾）
EVAL_OUTPUT_ROOT  = ROOT / "data" / "evaluation_outputs"
EVAL_RESULTS_ROOT = ROOT / "data" / "evaluation_results"

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_TOP_K        = 5
DEFAULT_PROMPT_TOP_N = 5
DEFAULT_ALPHA        = 0.5


def run_command(cmd: list[str], label: str) -> bool:
    """執行指令，回傳是否成功。"""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    status = "✓ 成功" if result.returncode == 0 else "✗ 失敗"
    print(f"  {status}（耗時 {elapsed:.1f}s）")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一次跑完五種 embedding 模式的 RAG 產答 + 四種指標評估"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(AVAILABLE_MODES),
        default=list(AVAILABLE_MODES),
        metavar="MODE",
        help=f"要跑的模式，可多選（預設全部）。可選：{', '.join(AVAILABLE_MODES)}",
    )
    parser.add_argument("--ollama-model",  default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--top-k",         type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--prompt-top-n",  type=int,   default=DEFAULT_PROMPT_TOP_N)
    parser.add_argument("--alpha",         type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--limit",         type=int,   default=None,  help="每種模式只跑前 N 題（測試用）")
    parser.add_argument("--question-type", default=None, help="只跑指定 question_type")
    parser.add_argument("--force-rerun",   action="store_true", help="忽略已有紀錄，強制重跑")
    parser.add_argument("--skip-rag",      action="store_true", help="跳過 RAG 產答，直接跑指標")
    parser.add_argument("--skip-eval",     action="store_true", help="跳過指標評估，只跑 RAG 產答")
    # 指標的 context 字數上限
    parser.add_argument("--max-context-chars-faithfulness", type=int, default=6000)
    parser.add_argument("--max-context-chars-recall",       type=int, default=6000)
    parser.add_argument("--max-context-chars-precision",    type=int, default=3000)
    args = parser.parse_args()

    py = sys.executable
    rag_script  = str(SCRIPTS_DIR / "run_evaluation_questions.py")
    eval_script = str(SCRIPTS_DIR / "evaluate_all_metrics.py")

    results_summary: list[dict] = []

    for mode in args.modes:
        output_jsonl = EVAL_OUTPUT_ROOT  / mode / "eval_outputs.jsonl"
        output_dir   = EVAL_RESULTS_ROOT / mode

        print(f"\n{'='*60}")
        print(f"  模式：{mode}")
        print(f"  RAG 輸出：{output_jsonl}")
        print(f"  指標輸出：{output_dir}")
        print(f"{'='*60}")

        rag_ok   = True
        eval_ok  = True

        # ── Step 1：RAG 產答 ─────────────────────────────────────
        if not args.skip_rag:
            rag_cmd = [
                py, rag_script,
                "--mode",         mode,
                "--output",       str(output_jsonl),
                "--ollama-model", args.ollama_model,
                "--top-k",        str(args.top_k),
                "--prompt-top-n", str(args.prompt_top_n),
                "--alpha",        str(args.alpha),
            ]
            if args.limit:
                rag_cmd += ["--limit", str(args.limit)]
            if args.question_type:
                rag_cmd += ["--question-type", args.question_type]
            if args.force_rerun:
                rag_cmd.append("--force-rerun")

            rag_ok = run_command(rag_cmd, f"[{mode}] Step 1：RAG 產答")
        else:
            print(f"\n[{mode}] Step 1：RAG 產答（已跳過，--skip-rag）")

        # ── Step 2：四種指標評估 ─────────────────────────────────
        if not args.skip_eval:
            if not output_jsonl.exists():
                print(f"[{mode}] Step 2：找不到 {output_jsonl}，跳過指標評估。")
                eval_ok = False
            else:
                eval_cmd = [
                    py, eval_script,
                    "--input",      str(output_jsonl),
                    "--output-dir", str(output_dir),
                    "--ollama-model", args.ollama_model,
                    "--max-context-chars-faithfulness", str(args.max_context_chars_faithfulness),
                    "--max-context-chars-recall",       str(args.max_context_chars_recall),
                    "--max-context-chars-precision",    str(args.max_context_chars_precision),
                ]
                if args.question_type:
                    eval_cmd += ["--question-type", args.question_type]
                if args.force_rerun:
                    eval_cmd.append("--force-rerun")

                eval_ok = run_command(eval_cmd, f"[{mode}] Step 2：四種指標評估")
        else:
            print(f"\n[{mode}] Step 2：指標評估（已跳過，--skip-eval）")

        results_summary.append({
            "mode":    mode,
            "rag_ok":  rag_ok,
            "eval_ok": eval_ok,
        })

    # ── 最終彙總 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  所有模式執行結果：")
    print(f"{'─'*60}")
    all_ok = True
    for r in results_summary:
        rag_mark  = "✓" if r["rag_ok"]  else "✗"
        eval_mark = "✓" if r["eval_ok"] else "✗"
        if not (r["rag_ok"] and r["eval_ok"]):
            all_ok = False
        print(f"  {r['mode']:<25}  RAG:{rag_mark}  指標:{eval_mark}")

    print(f"{'─'*60}")
    if all_ok:
        print("  全部完成。")
    else:
        print("  部分模式失敗，請查看上方錯誤訊息。")

    print(f"\n  結果資料夾：{EVAL_RESULTS_ROOT}")
    for mode in args.modes:
        output_dir = EVAL_RESULTS_ROOT / mode
        print(f"    {mode}/")
        for summary_file in [
            "relevancy_summary.json",
            "faithfulness_summary.json",
            "context_recall_summary.json",
            "context_precision_summary.json",
        ]:
            p = output_dir / summary_file
            mark = "✓" if p.exists() else "－"
            print(f"      {mark} {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
