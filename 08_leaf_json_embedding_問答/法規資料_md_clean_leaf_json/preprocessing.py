"""
Module 1: Pre-Retrieval 前處理
功能：正規化 → 拆解 → 改寫 → 提取關鍵字
輸入：QuestionA (JSON)
輸出：QuestionB (JSON)
"""

import json
import re
import sys

# ────────────────────────────────────────────
# 資安法規縮寫展開對照表（依專案文件擴充）
# ────────────────────────────────────────────
ABBR_MAP = {
    # 條文格式轉換
#    r'個資法§(\d+)': r'個人資料保護法第\1條',
#    r'個人資料保護法§(\d+)': r'個人資料保護法第\1條',
#    r'資通安全法§(\d+)': r'資通安全管理法第\1條',
#    r'資安法§(\d+)': r'資通安全管理法第\1條',
#    r'金控法§(\d+)': r'金融控股公司法第\1條',
#    r'銀行法§(\d+)': r'銀行法第\1條',
#    r'證交法§(\d+)': r'證券交易法第\1條',
#    r'電子簽章法§(\d+)': r'電子簽章法第\1條',
    # 條文格式轉換
    r'§(\d+)': r'第\1條',
    # 條文縮寫(使用者問題可能會出現的，待擴充)
    r'個資法': '個人資料保護法',
    r'資通安全法': '資通安全管理法',
    r'資安法': '資通安全管理法',
    r'金控法': '金融控股公司法',
    r'證交法': '證券交易法',
    # 機構縮寫(使用者問題可能會出現的，待擴充)
    r'金管會': '金融監督管理委員會',
    r'數位部': '數位發展部',
    r'櫃買中心': '財團法人中華民國證券櫃檯買賣中心',
    r'TPEx': '財團法人中華民國證券櫃檯買賣中心',
}

# 切割子問題的連接詞（待擴充）
SPLIT_CONJUNCTIONS = ['還是', '以及', '另外', '此外', '同時', '並且', '而且', '且']

# 資安法規常見關鍵詞（用於輔助提取，待擴充）
DOMAIN_KEYWORDS = [
    '資通安全', '資訊安全', '個人資料', '個資', '內部控制', '資安事件',
    '通報', '申報', '稽核', '加密', '存取控制', '防火牆', '備份',
    '身分驗證', '滲透測試', '弱點掃描', '資料外洩', '委外', '第三方',
    '金融機構', '證券商', '期貨商', '銀行', '保險', '金控', '安全機制',
    'ATM', '演算法', '金融機構', '金融服務', '期貨業', '證券商', '銀行', 
    '保險業', '金融控股公司', '期貨商',
]


# ────────────────────────────────────────────
# 步驟 1：正規化
# ────────────────────────────────────────────
def normalize(raw_text: str) -> str:
    """
    統一格式、展開縮寫、去除多餘空白
    """
    text = raw_text.strip()

    # 全形數字轉半形
    fullwidth_digits = str.maketrans('０１２３４５６７８９', '0123456789')
    text = text.translate(fullwidth_digits)

    # 全形英文轉半形
    text = ''.join(
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
        for c in text
    )

    # 展開法規縮寫
    for pattern, replacement in ABBR_MAP.items():
        text = re.sub(pattern, replacement, text)

    # 統一標點（中文問號、句號）
    text = text.replace('？', '？').replace('。', '。')

    # 去除多餘空白（保留中文間距）
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '，', text)

    return text.strip()


# ────────────────────────────────────────────
# 步驟 2：拆解複合問題
# ────────────────────────────────────────────
def decompose(normalized_text: str) -> list:
    """
    將複合問題拆成子問題列表
    策略：根據連接詞或問號切割
    """
    # 先試著用連接詞切
    for conj in SPLIT_CONJUNCTIONS:
        if conj in normalized_text:
            parts = normalized_text.split(conj)
            sub_qs = []
            for p in parts:
                p = p.strip().strip('，').strip()
                if len(p) >= 5:
                    # 確保每個子問題有問號結尾
                    if not p.endswith('？') and not p.endswith('?'):
                        p = p + '？'
                    sub_qs.append(p)
            if len(sub_qs) >= 2:
                return sub_qs

    # 用問號切（用在輸入多個問句）
    parts = re.split(r'[？?]', normalized_text)
    sub_qs = [p.strip() + '？' for p in parts if len(p.strip()) >= 5]
    if len(sub_qs) >= 2:
        return sub_qs

    # 無法拆解>整題當一個子問題
    return [normalized_text]


# ────────────────────────────────────────────
# 步驟 3：改寫子問題（去除指涉詞）
# ────────────────────────────────────────────
def rewrite(sub_questions: list) -> list:
    """
    改寫：讓每個子問題語意獨立、更接近法規文件語言
    去除口語指涉詞（它、該、此、前述…）
    """
    rewritten = []
    for q in sub_questions:
        # 去除常見口語指涉詞（待擴充）
        q = re.sub(r'^(那|這|該|此)', '', q).strip()
        q = re.sub(r'(它|其|該規定|前述規定|上述規定)', '相關規定', q)
        q = re.sub(r'(那個|這個|該項)', '', q).strip()

        # 口語轉正式（待擴充）
        q = q.replace('要怎麼', '如何')
        q = q.replace('要怎樣', '如何')
        q = q.replace('是啥', '是什麼')
        q = q.replace('幾天內', '幾日內')

        # 確保結尾有問號
        if q and not q.endswith('？') and not q.endswith('?'):
            q = q + '？'

        if q:
            rewritten.append(q)

    return rewritten if rewritten else sub_questions


# ────────────────────────────────────────────
# 步驟 4：提取關鍵字
# ────────────────────────────────────────────
def extract_keywords(normalized_text: str) -> list:
    """
    提取關鍵字：
    1. 從領域詞庫命中
    2. 提取 2~8 字的中文名詞片段
    """
    keywords = []

    # 先從領域詞庫命中
    for kw in DOMAIN_KEYWORDS:
        if kw in normalized_text:
            keywords.append(kw)

    # 再提取 2~8 字中文詞彙（排除疑問詞）
    stop_words = {'什麼', '如何', '為何', '是否', '有沒有', '可以', '需要',
                  '應該', '要怎', '請問', '規定', '相關', '規範'}
    candidates = re.findall(r'[\u4e00-\u9fff]{2,8}', normalized_text)
    for w in candidates:
        if w not in stop_words and w not in keywords:
            keywords.append(w)

    # 去重、保留順序
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords[:10]  # 最多取 10 個關鍵字


# ────────────────────────────────────────────
# 主流程：QuestionA → QuestionB
# ────────────────────────────────────────────
def preprocess(question_a: dict) -> dict:
    """
    輸入 QuestionA (dict/JSON)
    輸出 QuestionB (dict/JSON)
    """
    raw_text = question_a.get("raw_text", "").strip()

    # 正規化
    normalized = normalize(raw_text)

    # 拆解
    sub_qs = decompose(normalized)

    # 改寫
    sub_qs = rewrite(sub_qs)

    # 提取關鍵字
    keywords = extract_keywords(normalized)

    question_b = {
        "original": raw_text,
        "normalized": normalized,
        "sub_questions": sub_qs,
        "keywords": keywords,
    }

    return question_b


# ────────────────────────────────────────────
# 互動測試介面（在 VSCode 終端機執行）
# ────────────────────────────────────────────
def run_interactive():
    print("=" * 60)
    print("  Pre-Retrieval 前處理測試介面")
    print("  輸入問題後按 Enter，輸入 'q' 離開")
    print("=" * 60)

    while True:
        print()
        raw = input("請輸入問題 > ").strip()

        if raw.lower() in ('q', 'quit', 'exit', ''):
            print("離開測試介面。")
            break

        question_a = {"raw_text": raw}
        question_b = preprocess(question_a)

        print()
        print("── 輸入（QuestionA）──────────────────────")
        print(json.dumps(question_a, ensure_ascii=False, indent=2))
        print()
        print("── 輸出（QuestionB）──────────────────────")
        print(json.dumps(question_b, ensure_ascii=False, indent=2))
        print()

        # 顯示拆解細節
        print(f"  子問題數量：{len(question_b['sub_questions'])}")
        for i, q in enumerate(question_b['sub_questions'], 1):
            print(f"  [{i}] {q}")
        print(f"  關鍵字：{', '.join(question_b['keywords'])}")


# ────────────────────────────────────────────
# 批次測試（自動跑範例，待更改）
# ────────────────────────────────────────────
def run_batch_test():
    test_cases = [
        {"raw_text": "根據此規範，內部控制制度的設計應考量哪五個控制因素？"},
        {"raw_text": "內部控制應考量那些因素？"},
        {"raw_text": "內部控制包含哪三種因素？"},
        {"raw_text": "根據此總則，內部稽核的主要目的為何？"},
        {"raw_text": "內部稽核可以達到何種效果？"},
        {"raw_text": "內稽的目為何？"},
        {"raw_text": "零用金管理作業中，對於零用金的設立目的與經管人員的職責有何規範？"},
        {"raw_text": "零用金因何目的而設立？管理人員應該要做甚麼事情管理零用金？"},
        {"raw_text": "零用金的設立和經管人員的職責有何相關？"},
        {"raw_text": "根據第二條，適用本規範的「資訊服務」具體包含哪三種服務形態？"},
        {"raw_text": "資訊服務有哪幾種型態？"},
        {"raw_text": "資訊服務包含哪五種服務型態？"},
        {"raw_text": "採用靜態密碼進行身分驗證時，密碼連續錯誤達幾次後，公司應進行妥善處理？"},
        {"raw_text": "靜態密碼進行身分驗證時，密碼不能連續錯誤達到幾次？"},
        {"raw_text": "採用靜態密碼進行身分驗證時，公司應進行妥善處理，代表密碼連續錯了幾次？"},
        {"raw_text": "金融主管當局制定此安全基準的主要目的為何？"},
        {"raw_text": "此安全基準由哪三個主要部分組成？其內容重點分別為何？"},
        {"raw_text": "在設備基準中，資訊中心對於「環境」選址有何具體要求？"},
        {"raw_text": "根據基準，電腦機房在「火災防範」與「滅火設備」上有何標準？"},
        {"raw_text": "營運基準如何規定「進出管理」與「人員識別」？"},
        {"raw_text": "在營運管理中，對於「存取權限」與「密碼管理」有哪些具體要求？"},
        {"raw_text": "對於「委外管理」，基準規定合約中應包含哪些重要條款？"},
        {"raw_text": "技術基準中，如何透過「資料保護」來防範洩漏或篡改？"},
        {"raw_text": "針對「非法存取」與「非法程式」，技術基準提供了哪些偵測對策？"},
        {"raw_text": "對於無人化服務區（如 ATM）的管理，營運基準有何重點要求？"},
        {"raw_text": "證卷商在訂定內部控制制度時，其「總則」部分至少應敘明哪些事項（請列舉三項）"},
        {"raw_text": ".根據此規範，內步控制制度的設計應考量哪五個控制因素？"},
        {"raw_text": "金融機構在執行「營運充急分析（BIA）」後，應產出哪些關鍵的分析結果？"},
    ]

    print("=" * 60)
    print("  批次測試模式")
    print("=" * 60)

    for i, qa in enumerate(test_cases, 1):
        print(f"\n【測試 {i}】")
        qb = preprocess(qa)
        print("輸入：", json.dumps(qa, ensure_ascii=False))
        print("輸出：", json.dumps(qb, ensure_ascii=False, indent=2))
        print("-" * 40)


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        run_batch_test()
    else:
        run_interactive()
