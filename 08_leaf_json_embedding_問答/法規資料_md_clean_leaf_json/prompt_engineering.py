"""
Module 3: 提示詞工程 (Prompt Engineering)
─────────────────────────────────────────────────────────────
輸入：
    question_a      : dict  → { "raw_text": str }
    question_b      : dict  → preprocessing.py 的輸出
    candidates      : list  → run_search() 的輸出（動態搜索補齊後）
    relation_notes  : str   → 簡稱/指涉/引用關係標記（from組員）>（後來沒有的功能，先變成空字串）

輸出：
    answer          : str   → LLM 的文字回答

執行方式（VSCode 終端機）：
    互動測試：python prompt_engineering.py
    批次測試：python prompt_engineering.py --batch
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from typing import Any

import requests

from preprocessing import preprocess


# ════════════════════════════════════════════
# 設定
# ════════════════════════════════════════════

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 600

# context 最大字數（避免超出 LLM context window）
MAX_CONTEXT_CHARS = 4000

# 候選段落最多取幾筆放進 prompt
MAX_CANDIDATES_IN_PROMPT = 5


# ════════════════════════════════════════════
# Step 1：整理 context 區塊
# ════════════════════════════════════════════

def _format_candidate_block(idx: int, item: dict[str, Any]) -> str:
    """
    將單一候選段落格式化成 prompt 裡的參考資料區塊。
    欄位對應 app_2_1.py 的 run_search() 輸出格式。
    """
    doc_type  = item.get("doc_type", "")
    file_name = item.get("file_name", "")
    path_text = item.get("path_text", "")
    page_s    = item.get("page_start", "")
    page_e    = item.get("page_end", "")
    text      = str(item.get("text", "")).strip()

    lines = [
        f"[參考資料 {idx}]",
        f"類型     : {doc_type}",
        f"法規檔   : {file_name}",
    ]
    if path_text:
        lines.append(f"路徑     : {path_text}")
    if page_s or page_e:
        lines.append(f"頁碼     : {page_s} - {page_e}")
    lines += ["內容     :", text]

    return "\n".join(lines)


def build_context(
    candidates: list[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
    max_items: int = MAX_CANDIDATES_IN_PROMPT,
) -> tuple[str, list[int]]:
    """
    將候選段落組成 context 字串，控制總字數。
    回傳 (context_str, 實際用到的候選索引清單)
    """
    blocks: list[str] = []
    used_indices: list[int] = []
    total_chars = 0

    for i, item in enumerate(candidates[:max_items]):
        block = _format_candidate_block(i + 1, item)
        if total_chars + len(block) > max_chars:
            break
        blocks.append(block)
        used_indices.append(i)
        total_chars += len(block)

    context_str = "\n\n---\n\n".join(blocks)
    return context_str, used_indices


# ════════════════════════════════════════════
# Step 2：組裝完整 Prompt
# ════════════════════════════════════════════

def build_prompt(
    question_a: dict[str, Any],
    question_b: dict[str, Any],
    candidates: list[dict[str, Any]],
    relation_notes: str = "",
) -> str:
    """
    組裝最終送給 LLM 的 prompt。

    輸入：
        question_a     : { "raw_text": str }
        question_b     : preprocessing.py 輸出
        candidates     : run_search() 輸出（已由動態搜索算法補齊）
        relation_notes : 簡稱/指涉/引用關係標記（後來沒有的功能，先變成空字串）

    輸出：
        prompt_str : str
    """
    raw_question  = question_a.get("raw_text", "").strip()
    sub_questions = question_b.get("sub_questions", [raw_question])

    sub_q_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_questions))

    context_str, _ = build_context(candidates)

    relation_section = (
        f"\n## 條文關係說明（簡稱 / 指涉 / 引用）\n{relation_notes.strip()}\n"
        if relation_notes.strip()
        else ""
    )

    prompt = f"""[INST] 你必須只使用繁體中文回答。[/INST]
你是企業資安法規智能問答助手，專門協助解讀台灣金融資安相關法規。


## 使用者問題
{raw_question}

## 拆解後的子問題
{sub_q_str}
{relation_section}
## 參考資料
{context_str}

## 回答要求
1. 使用繁體中文回答。
2. 先給出結論，再逐條說明依據。
3. 引用時請標明法規檔名與條文位置（例如：依據「XXX.md」第X條）。
4. 若參考資料不足以回答，請明確說明「資料不足，無法回答」，勿自行推測。
5. 回答結構：【結論】→【說明】→【引用來源】

## 回答
"""
    return prompt


# ════════════════════════════════════════════
# Step 3：呼叫 LLM（Ollama）
# ════════════════════════════════════════════

def call_ollama(
    prompt: str,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:
    """
    呼叫本機 Ollama，回傳 LLM 文字回答。
    Ollama 必須已在本機啟動（ollama serve）。
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise RuntimeError(f"Ollama 回傳格式異常: {data}")
        return str(data["response"]).strip()
    except requests.exceptions.ConnectionError:
        return "[錯誤] 無法連線到 Ollama，請確認 ollama serve 是否已啟動。"
    except requests.exceptions.Timeout:
        return "[錯誤] Ollama 請求逾時，請稍後再試或縮短 context 長度。"
    except Exception as exc:
        return f"[錯誤] {exc}"


# ════════════════════════════════════════════
# 主流程：generate_answer()
# ════════════════════════════════════════════

def generate_answer(
    question_a: dict[str, Any],
    question_b: dict[str, Any],
    candidates: list[dict[str, Any]],
    relation_notes: str = "",
    model_name: str = DEFAULT_OLLAMA_MODEL,
) -> dict[str, Any]:
    """
    提示詞工程主流程。

    輸入：
        question_a     : { "raw_text": str }
        question_b     : preprocessing.py 的輸出 (dict)
        candidates     : run_search() 的輸出 list[dict]（動態搜索補齊後）
        relation_notes : 簡稱/指涉/引用關係標記，字串，預設空字串
        model_name     : Ollama 模型名稱

    輸出 JSON：
    {
        "question_a"    : { "raw_text": ... },
        "question_b"    : { ... },
        "prompt"        : "...",        # 實際送給 LLM 的完整 prompt
        "answer"        : "...",        # LLM 回答
        "candidates_used": int,         # 實際放進 prompt 的候選數
        "relation_notes": "..."
    }
    """
    prompt = build_prompt(question_a, question_b, candidates, relation_notes)
    _, used_indices = build_context(candidates)

    answer = call_ollama(prompt, model_name)

    return {
        "question_a"     : question_a,
        "question_b"     : question_b,
        "prompt"         : prompt,
        "answer"         : answer,
        "candidates_used": len(used_indices),
        "relation_notes" : relation_notes,
    }


# ════════════════════════════════════════════
# 互動測試介面（單獨執行，不需要 app_2_1.py）
# ════════════════════════════════════════════

def _mock_candidates(question: str) -> list[dict[str, Any]]:
    """
    單獨測試用的假候選段落。
    實際使用時由 app_2_1.py 的 run_search() 提供。
    """
    return [
        {
            "rank": 1,
            "score": 0.85,
            "doc_type": "leaf",
            "file_name": "金融機構資訊系統安全基準.md",
            "path_text": "第三章 > 3.1 存取控制",
            "page_start": 12,
            "page_end": 13,
            "text": (
                "金融機構應依據最小權限原則設定存取控制，"
                "確保人員僅能存取其職務所需之資訊系統與資料。"
                "存取權限之新增、修改及刪除，應有書面申請與主管審核程序。"
            ),
        },
        {
            "rank": 2,
            "score": 0.72,
            "doc_type": "table_chunk",
            "file_name": "金融機構資訊系統安全基準.md",
            "path_text": "附表一",
            "page_start": 45,
            "page_end": 45,
            "text": "存取控制審查頻率：一般帳號每半年一次；特權帳號每季一次。",
        },
    ]


def run_interactive():
    print("=" * 60)
    print("  提示詞工程測試介面")
    print("  問題先經 preprocessing → mock 候選 → 組 prompt → Ollama")
    print(f"  使用模型：{DEFAULT_OLLAMA_MODEL}")
    print("  輸入 'q' 離開")
    print("=" * 60)

    while True:
        print()
        raw = input("請輸入問題 > ").strip()
        if raw.lower() in ("q", "quit", "exit", ""):
            print("離開。")
            break

        question_a = {"raw_text": raw}
        question_b = preprocess(question_a)
        candidates = _mock_candidates(raw)

        print()
        print("── QuestionB（前處理）───────────────────────")
        print(json.dumps(question_b, ensure_ascii=False, indent=2))

        print()
        print("── 組裝 Prompt ──────────────────────────────")
        prompt = build_prompt(question_a, question_b, candidates)
        print(prompt)

        print()
        print("── 呼叫 Ollama ──────────────────────────────")
        result = generate_answer(question_a, question_b, candidates)

        print()
        print("── 回答 ─────────────────────────────────────")
        print(result["answer"])

        print()
        print("── 完整 JSON 輸出 ───────────────────────────")
        # prompt 太長，顯示時截短
        display = {**result, "prompt": result["prompt"][:200] + "...（截短）"}
        print(json.dumps(display, ensure_ascii=False, indent=2))


def run_batch_test():
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

    for i, raw in enumerate(test_cases, 1):
        print(f"\n{'='*60}\n  【測試 {i}】{raw}\n{'='*60}")
        question_a = {"raw_text": raw}
        question_b = preprocess(question_a)
        candidates = _mock_candidates(raw)
        result = generate_answer(question_a, question_b, candidates)
        print("【回答】")
        print(result["answer"])
        print(f"（使用候選段落數：{result['candidates_used']}）")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        run_batch_test()
    else:
        run_interactive()
