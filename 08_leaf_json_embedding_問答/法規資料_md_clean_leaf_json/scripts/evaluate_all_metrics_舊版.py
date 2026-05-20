from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT / "data" / "evaluation_outputs" / "eval_outputs.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "evaluation_results"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all evaluator metrics in sequence.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--max-context-chars-faithfulness", type=int, default=6000)
    parser.add_argument("--max-context-chars-recall", type=int, default=6000)
    parser.add_argument("--max-context-chars-precision", type=int, default=3000)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    common = [
        sys.executable,
    ]

    relevancy_cmd = common + [
        str(SCRIPTS_DIR / "evaluate_response_relevancy.py"),
        "--input",
        str(args.input),
        "--output-dir",
        str(args.output_dir),
        "--ollama-model",
        args.ollama_model,
    ]
    faithfulness_cmd = common + [
        str(SCRIPTS_DIR / "evaluate_faithfulness.py"),
        "--input",
        str(args.input),
        "--output-dir",
        str(args.output_dir),
        "--ollama-model",
        args.ollama_model,
        "--max-context-chars",
        str(args.max_context_chars_faithfulness),
    ]
    recall_cmd = common + [
        str(SCRIPTS_DIR / "evaluate_context_recall.py"),
        "--input",
        str(args.input),
        "--output-dir",
        str(args.output_dir),
        "--ollama-model",
        args.ollama_model,
        "--max-context-chars",
        str(args.max_context_chars_recall),
    ]
    precision_cmd = common + [
        str(SCRIPTS_DIR / "evaluate_context_precision.py"),
        "--input",
        str(args.input),
        "--output-dir",
        str(args.output_dir),
        "--ollama-model",
        args.ollama_model,
        "--max-context-chars",
        str(args.max_context_chars_precision),
    ]

    if args.question_type:
        for cmd in (relevancy_cmd, faithfulness_cmd, recall_cmd, precision_cmd):
            cmd.extend(["--question-type", args.question_type])
    if args.force_rerun:
        for cmd in (relevancy_cmd, faithfulness_cmd, recall_cmd, precision_cmd):
            cmd.append("--force-rerun")

    run_command(relevancy_cmd)
    run_command(faithfulness_cmd)
    run_command(recall_cmd)
    run_command(precision_cmd)

    print(f"relevancy summary: {args.output_dir / 'relevancy_summary.json'}")
    print(f"faithfulness summary: {args.output_dir / 'faithfulness_summary.json'}")
    print(f"context recall summary: {args.output_dir / 'context_recall_summary.json'}")
    print(f"context precision summary: {args.output_dir / 'context_precision_summary.json'}")


if __name__ == "__main__":
    main()
