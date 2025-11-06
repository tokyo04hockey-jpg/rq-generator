import os
import time
import json
import math
import requests
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from urllib.parse import urlencode

# =========================
# 設定（Secrets を優先）
# =========================
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GOOGLE_CSE_ID  = st.secrets.get("GOOGLE_CSE_ID",  os.getenv("GOOGLE_CSE_ID",  ""))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# =========================
# OpenAI (Responses API)
# =========================
# refs: client.responses.create / Python SDK（公式）
from openai import OpenAI  # pip install openai>=1.0.0
_oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def ask_openai_reasoning(company: str, evidence: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """
    evidence は各カテゴリーごとに [{'title','link','snippet'} ...] の配列
    返り値: カテゴリごとの {label, confidence, reason_ja, reason_en} と、要約ツイート jp/en
    """
    if not _oai:
        return {}

    # 1社ごとプロンプト（日本語で指示＋英語出力も要求）
    sys = (
        "You are an analyst for corporate–startup collaboration. "
        "Return JSON only. Be concise, cite evidence URLs explicitly."
    )

    user = {
        "company": company,
        "tasks": [
            "CVCを立ち上げているか (CVC)",
            "LP投資をしているか (LP)",
            "AI/Roboticsと事業シナジー (AI_Robotics)",
            "Healthcareと事業シナジー (Healthcare)",
            "Climate techと事業シナジー (Climate)"
        ],
        "instruction": (
            "各タスクについて: label を 'Yes' | 'No' | 'Unclear' から、confidence を 0-1、"
            "reason_ja を日本語100字以内、reason_en を英語で1-2文。"
            "必ず根拠URL(見つかった範囲で最大3件)を evidence_urls に含める。"
            "最後に X 投稿ドラフト: jp は全角140字以内（URLなし・本文のみ）、en は英語280文字以内（最重要URLを1つだけ末尾に）。"
            "出力は以下JSONスキーマに厳密準拠:\n"
            "{"
            "  'per_task': {"
            "     'CVC':        {'label':'', 'confidence':0.0, 'reason_ja':'', 'reason_en':'', 'evidence_urls':[]},"
            "     'LP':         {'label':'', 'confidence':0.0, 'reason_ja':'', 'reason_en':'', 'evidence_urls':[]},"
            "     'AI_Robotics':{'label':'', 'confidence':0.0, 'reason_ja':'', 'reason_en':'', 'evidence_urls':[]},"
            "     'Healthcare': {'label':'', 'confidence':0.0, 'reason_ja':'', 'reason_en':'', 'evidence_urls':[]},"
            "     'Climate':    {'label':'', 'confidence':0.0, 'reason_ja':'', 'reason_en':'', 'evidence_urls':[]}"
            "  },"
            "  'x_post': {'jp':'', 'en':''}"
            "}"
        ),
        "evidence": evidence
    }

    # Responses API
    resp = _oai.responses.create(
        model="gpt-4.1-mini",
        input=f"System:\n{sys}\n\nUser:\n{json.dumps(user, ensure_ascii=False)}",
        temperature=0.2,
        max_output_tokens=1200,
    )
    text = resp.output_text  # unified text out
    try:
        data = json.loads(text)
    except Exception:
        # 万が一JSONで返らない場合のフォールバック（簡易）
        data = {"per_task": {}, "x_post": {"jp": "", "en": ""}}

    return data

# =========================
# Google Custom Search
# =========================
def google_search(q: str, num: int = 5) -> List[Dict[str, str]]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": q,
        "num": min(num, 10),
        "hl": "ja",
        "gl": "jp",
        "safe": "off",
    }
    url = f"https://www.googleapis.com/customsearch/v1?{urlencode(params)}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    items = []
    for it in j.get("items", []):
        items.append({
            "title": it.get("title",""),
            "link": it.get("link",""),
            "snippet": it.get("snippet",""),
        })
    return items

def gather_evidence(company: str) -> Dict[str, List[Dict[str, str]]]:
    queries = {
        "CVC":        f"{company} CVC ベンチャー投資 コーポレートベンチャーキャピタル",
        "LP":         f"{company} LP 出資 ベンチャーファンド リミテッドパートナー",
        "AI_Robotics":f"{company} AI ロボティクス 事業 提携 スタートアップ",
        "Healthcare": f"{company} ヘルスケア 医療 デジタルヘルス 提携 スタートアップ",
        "Climate":    f"{company} 脱炭素 クライメートテック 再生可能エネルギー 提携"
    }
    ev = {}
    for k, q in queries.items():
        time.sleep(0.3)  # レート緩和
        ev[k] = google_search(q, num=6)
    return ev

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Corporate–Startup Fit Checker (w/ OpenAI reasons)", layout="wide")
st.title("🏢➡️🤝🚀 Corporate–Startup Fit Checker")
st.caption("ExcelのC列に会社名。Google CSEで証跡を集め、OpenAIで判定理由とX向け要約を生成します。")

with st.expander("🔧 Secrets設定（必須）", expanded=False):
    st.markdown(
        "- `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`, `OPENAI_API_KEY` を **.streamlit/secrets.toml** に設定してください。\n"
        "```toml\n[general]\n# 例:\nGOOGLE_API_KEY = \"xxxxx\"\nGOOGLE_CSE_ID  = \"xxxx:yyyy\"\nOPENAI_API_KEY = \"sk-...\"\n```\n"
    )

uploaded = st.file_uploader("Excel をアップロード（C列=会社名）", type=["xlsx", "xls"])
limit = st.number_input("処理件数の上限（テスト用）", 1, 5000, 50, 10)

run = st.button("解析スタート", type="primary", disabled=uploaded is None)

if run and uploaded:
    df = pd.read_excel(uploaded)
    # 会社列の推定（C列優先）
    if df.shape[1] >= 3:
        companies = df.iloc[:, 2].dropna().astype(str).tolist()
    else:
        companies = df.iloc[:, -1].dropna().astype(str).tolist()
    companies = companies[:int(limit)]

    rows = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, company in enumerate(companies, 1):
        status.info(f"Searching: {company}")
        evidence = gather_evidence(company)

        # OpenAIで理由生成
        reasoning = ask_openai_reasoning(company, evidence) if OPENAI_API_KEY else {}

        per_task = reasoning.get("per_task", {})
        x_post = reasoning.get("x_post", {"jp":"", "en":""})

        def cell(task: str, field: str, default=""):
            return per_task.get(task, {}).get(field, default)

        rows.append({
            "company": company,
            # ラベル
            "CVC":        cell("CVC", "label", "Unclear"),
            "LP":         cell("LP", "label", "Unclear"),
            "AI_Robotics":cell("AI_Robotics", "label", "Unclear"),
            "Healthcare": cell("Healthcare", "label", "Unclear"),
            "Climate":    cell("Climate", "label", "Unclear"),
            # 信頼度
            "CVC_conf":        cell("CVC", "confidence", ""),
            "LP_conf":         cell("LP", "confidence", ""),
            "AI_Robotics_conf":cell("AI_Robotics", "confidence", ""),
            "Healthcare_conf": cell("Healthcare", "confidence", ""),
            "Climate_conf":    cell("Climate", "confidence", ""),
            # 理由（日/英）
            "CVC_reason_ja":        cell("CVC", "reason_ja", ""),
            "LP_reason_ja":         cell("LP", "reason_ja", ""),
            "AI_Robotics_reason_ja":cell("AI_Robotics", "reason_ja", ""),
            "Healthcare_reason_ja": cell("Healthcare", "reason_ja", ""),
            "Climate_reason_ja":    cell("Climate", "reason_ja", ""),
            "CVC_reason_en":        cell("CVC", "reason_en", ""),
            "LP_reason_en":         cell("LP", "reason_en", ""),
            "AI_Robotics_reason_en":cell("AI_Robotics", "reason_en", ""),
            "Healthcare_reason_en": cell("Healthcare", "reason_en", ""),
            "Climate_reason_en":    cell("Climate", "reason_en", ""),
            # URL（最大3件を;区切りで）
            "CVC_urls":        "; ".join(per_task.get("CVC", {}).get("evidence_urls", [])),
            "LP_urls":         "; ".join(per_task.get("LP", {}).get("evidence_urls", [])),
            "AI_Robotics_urls":"; ".join(per_task.get("AI_Robotics", {}).get("evidence_urls", [])),
            "Healthcare_urls": "; ".join(per_task.get("Healthcare", {}).get("evidence_urls", [])),
            "Climate_urls":    "; ".join(per_task.get("Climate", {}).get("evidence_urls", [])),
            # X投稿ドラフト
            "x_post_jp": x_post.get("jp", ""),
            "x_post_en": x_post.get("en", "")
        })

        progress.progress(i/len(companies))
        time.sleep(0.05)

    out = pd.DataFrame(rows)
    st.success("完了！")
    st.dataframe(out, use_container_width=True)

    csv = out.to_csv(index=False)
    st.download_button("CSVをダウンロード", data=csv, file_name="corporate_fit_with_reasons.csv", mime="text/csv")
