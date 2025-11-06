# app.py
import os
import re
import time
import json
import html
import traceback
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlencode, urlparse
from typing import List, Dict, Any
from collections import defaultdict

# =========================
# 設定（Secrets を優先）
# =========================
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GOOGLE_CSE_ID  = st.secrets.get("GOOGLE_CSE_ID",  os.getenv("GOOGLE_CSE_ID",  ""))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# =========================
# OpenAI (Responses API)
# =========================
# pip install openai>=1.0.0
from openai import OpenAI
_oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
MODEL_REASON = os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1-mini")

# =========================
# ユーティリティ
# =========================
SAFE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CorporateStartupFit/1.0)"
}
REQUEST_TIMEOUT = 20

def _strip_html(raw: str) -> str:
    """超軽量の HTML → プレーンテキスト化"""
    if not raw:
        return ""
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def _domain_score(url: str) -> int:
    """公式/IR/PR を優先する簡易スコア"""
    host = urlparse(url).netloc.lower()
    score = 0
    if host.endswith(".co.jp") or host.endswith(".com"):
        score += 1
    if any(k in host for k in ["ir.", "prtimes.jp", "news.", "press"]):
        score += 2
    if any(p in url.lower() for p in ["/ir", "/investor", "/press", "/news", "/release"]):
        score += 2
    return score

def _dedup_urls(items: List[Dict[str, str]], max_per_domain: int = 3) -> List[Dict[str, str]]:
    seen = set()
    domain_counter = defaultdict(int)
    out = []
    for it in items:
        u = it.get("link", "")
        if not u or u in seen:
            continue
        d = urlparse(u).netloc
        if domain_counter[d] >= max_per_domain:
            continue
        seen.add(u)
        domain_counter[d] += 1
        out.append(it)
    return out

# =========================
# Google Custom Search
# =========================
@st.cache_data(show_spinner=False, ttl=60*60)
def google_search(q: str, num: int = 6) -> List[Dict[str, str]]:
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
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    items = []
    for it in j.get("items", []):
        items.append({
            "title": it.get("title",""),
            "link": it.get("link",""),
            "snippet": it.get("snippet",""),
        })
    items.sort(key=lambda x: _domain_score(x["link"]), reverse=True)
    return _dedup_urls(items, max_per_domain=2)

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_text(url: str, max_chars: int = 4000) -> str:
    try:
        r = requests.get(url, headers=SAFE_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        text = _strip_html(r.text)
        text = re.sub(r"(?i)この記事|関連記事|おすすめ|シェア|同社|編集部|注意事項", " ", text)
        return text[:max_chars]
    except Exception:
        return ""

# =========================
# Evidence 収集
# =========================
TASKS = {
    "CVC": "CVCを立ち上げているか",
    "LP": "LP（ファンドのリミテッドパートナー）出資をしているか",
    "AI_Robotics": "AI/Roboticsと事業シナジーがあるか",
    "Healthcare": "Healthcareと事業シナジーがあるか",
    "Climate": "Climate techと事業シナジーがあるか",
}

def _queries_for(company: str) -> Dict[str, List[str]]:
    """日英 + シノニムを含むクエリ束を作成"""
    return {
        "CVC": [
            f"{company} CVC コーポレートベンチャーキャピタル 立ち上げ 投資子会社",
            f"{company} corporate venture capital CVC fund launch investing arm",
        ],
        "LP": [
            f"{company} LP 出資 リミテッドパートナー ベンチャーファンド 出資参画",
            f"{company} limited partner LP commitment venture fund investor",
        ],
        "AI_Robotics": [
            f"{company} AI ロボティクス 提携 出資 共同開発 スタートアップ",
            f"{company} AI robotics partnership investment startup collaboration",
        ],
        "Healthcare": [
            f"{company} ヘルスケア 医療 デジタルヘルス 提携 出資 共同研究",
            f"{company} healthcare medtech digital health partnership investment",
        ],
        "Climate": [
            f"{company} 脱炭素 クライメートテック 再生可能エネルギー 水素 CCS 提携 出資",
            f"{company} climate tech decarbonization renewable hydrogen CCS partnership investment",
        ],
    }

@st.cache_data(show_spinner=False, ttl=60*60)
def gather_evidence(company: str, per_query_limit: int = 6, per_task_urls: int = 6) -> Dict[str, List[Dict[str, str]]]:
    queries = _queries_for(company)
    ev_raw: Dict[str, List[Dict[str, str]]] = {}
    for k, qlist in queries.items():
        bucket = []
        for q in qlist:
            time.sleep(0.25)  # レート緩和
            bucket.extend(google_search(q, num=per_query_limit))
        # 同一 URL をまとめ、上位優先
        seen = set()
        uniq = []
        for it in bucket:
            u = it["link"]
            if u in seen:
                continue
            seen.add(u)
            uniq.append(it)
        ev_raw[k] = uniq[:per_task_urls]
    return ev_raw

@st.cache_data(show_spinner=False, ttl=60*60)
def hydrate_evidence_with_content(evidence: Dict[str, List[Dict[str, str]]], max_sources_per_task: int = 5) -> Dict[str, List[Dict[str, str]]]:
    """各 URL から本文（先頭数千文字）を取得して evidence に埋め込む"""
    out: Dict[str, List[Dict[str, str]]] = {}
    for task, items in evidence.items():
        enriched = []
        for it in items[:max_sources_per_task]:
            url = it["link"]
            body = fetch_text(url, max_chars=5000)
            enriched.append({
                "title": it.get("title",""),
                "link": url,
                "snippet": it.get("snippet",""),
                "body": body,
            })
        out[task] = enriched
    return out

# =========================
# OpenAI Reasoning
# =========================
PROMPT_SYSTEM = (
    "You are an analyst of corporate–startup collaboration. "
    "Return STRICT JSON only with the exact schema requested. "
    "Be conservative: choose 'Unclear' unless there is direct evidence."
)

PROMPT_USER_TEMPLATE = """
You will judge one company across tasks with provided web evidence (titles, snippets, and fetched page body).
Company: {company}

Definitions and decision rules (apply strictly):
- CVC: The company has its own corporate venture capital arm or investment subsidiary (e.g., 'CVC', 'corporate venture capital', 'investment subsidiary', 'capital partners'). One-off venture investments without a dedicated arm → do NOT mark 'Yes'.
- LP: The company has committed capital as a limited partner to an external venture fund (e.g., 'LP', 'limited partner', 'commitment', '出資', 'リミテッドパートナー'). If only investing directly as a CVC without LP commitment, mark 'No' (unless LP is also evidenced).
- Synergy (AI_Robotics / Healthcare / Climate): Mark 'Yes' only if there is concrete evidence of products, partnerships, investments, pilots, or stated strategic focus that clearly connects to the domain. Generic news without relation to the company's business → 'Unclear'.

Hard constraints:
- Output must be valid JSON (UTF-8, no trailing commas, no comments).
- For each task, set 'label' ∈ {{'Yes','No','Unclear'}}, 'confidence' ∈ [0,1].
- 'reason_ja': ≤100 Japanese characters. 'reason_en': 1–2 sentences English.
- 'evidence_urls': include up to 3 URLs, but ONLY from the provided evidence links. Do NOT invent URLs.
- If signals are mixed or decade-old without follow-ups, prefer 'Unclear'.
- If CVC is 'Yes' and LP also 'Yes', ensure reasons clearly distinguish between the two.
- If no relevant evidence, use 'Unclear' with confidence 0.2.

Return JSON in the exact schema:
{{
  "per_task": {{
    "CVC":         {{"label":"","confidence":0.0,"reason_ja":"","reason_en":"","evidence_urls":[]}},
    "LP":          {{"label":"","confidence":0.0,"reason_ja":"","reason_en":"","evidence_urls":[]}},
    "AI_Robotics": {{"label":"","confidence":0.0,"reason_ja":"","reason_en":"","evidence_urls":[]}},
    "Healthcare":  {{"label":"","confidence":0.0,"reason_ja":"","reason_en":"","evidence_urls":[]}},
    "Climate":     {{"label":"","confidence":0.0,"reason_ja":"","reason_en":"","evidence_urls":[]}}
  }},
  "x_post": {{"jp":"","en":""}}
}}

Evidence (grouped by task). Each item has fields: title, link, snippet, body (first kilobytes of fetched page).
{evidence_json}
"""

def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}\s*$", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}

def ask_openai_reasoning(company: str, evidence_enriched: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    if not _oai:
        return {}

    prompt_user = PROMPT_USER_TEMPLATE.format(
        company=company,
        evidence_json=json.dumps(evidence_enriched, ensure_ascii=False)[:120000]
    )

    # 1回目
    resp = _oai.responses.create(
        model=MODEL_REASON,
        temperature=0.0,
        max_output_tokens=1500,
        input=f"System:\n{PROMPT_SYSTEM}\n\nUser:\n{prompt_user}",
    )
    text = resp.output_text
    data = _safe_json_loads(text)

    # パース失敗 or 欠損が大きい場合は 1 回だけ再試行
    if not data or "per_task" not in data:
        resp2 = _oai.responses.create(
            model=MODEL_REASON,
            temperature=0.0,
            max_output_tokens=1500,
            input=(
                "System:\n" + PROMPT_SYSTEM +
                "\n\nUser:\nReturn ONLY valid JSON per the schema. If previous attempt failed, correct and resend the JSON.\n" +
                prompt_user
            ),
        )
        text = resp2.output_text
        data = _safe_json_loads(text)

    # スキーマの穴埋め（最終防御）
    skeleton = {
        "per_task": {k: {"label":"Unclear","confidence":0.2,"reason_ja":"","reason_en":"","evidence_urls":[]} for k in TASKS},
        "x_post": {"jp":"","en":""}
    }
    try:
        merged = skeleton
        if isinstance(data, dict):
            merged["x_post"].update(data.get("x_post", {}))
            pt = data.get("per_task", {})
            for k in TASKS:
                if isinstance(pt.get(k), dict):
                    merged["per_task"][k].update({
                        kk: pt[k].get(kk, merged["per_task"][k][kk])
                        for kk in ["label","confidence","reason_ja","reason_en","evidence_urls"]
                    })
        return merged
    except Exception:
        return skeleton

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Corporate–Startup Fit Checker+", layout="wide")
st.title("🏢➡️🤝🚀 Corporate–Startup Fit Checker+")
st.caption("C列=会社名。Google CSEで証跡を集め、本文取得→OpenAIで CVC/LP/シナジーを厳密判定。Xドラフト付き。")

with st.expander("🔧 Secrets設定（必須）", expanded=False):
    st.markdown(
        "- `.streamlit/secrets.toml` に `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`, `OPENAI_API_KEY` を設定。\n"
        "```toml\n[general]\nGOOGLE_API_KEY=\"xxxxx\"\nGOOGLE_CSE_ID=\"xxxx:yyyy\"\nOPENAI_API_KEY=\"sk-...\"\n```\n"
    )

cols = st.columns(3)
with cols[0]:
    uploaded = st.file_uploader("Excel をアップロード（C列=会社名）", type=["xlsx", "xls"])
with cols[1]:
    limit = st.number_input("処理件数の上限", 1, 5000, 50, 10)
with cols[2]:
    max_sources = st.slider("各タスクの最大参照URL数", 1, 8, 5)

run = st.button("解析スタート", type="primary", disabled=uploaded is None)

if run and uploaded:
    df = pd.read_excel(uploaded)
    # 会社列の推定（C列優先→最終列）
    if df.shape[1] >= 3:
        companies = df.iloc[:, 2].dropna().astype(str).tolist()
    else:
        companies = df.iloc[:, -1].dropna().astype(str).tolist()
    companies = companies[:int(limit)]

    rows = []
    progress = st.progress(0.0)
    status = st.empty()

    tabs = st.tabs(["進捗", "最終テーブル", "詳細ログ"])
    with tabs[0]:
        st.write("検索→本文取得→判定を順に実行します。")

    detail_log = []

    for i, company in enumerate(companies, 1):
        status.info(f"Searching & analyzing: {company}")
        try:
            ev = gather_evidence(company)
            ev_enriched = hydrate_evidence_with_content(ev, max_sources_per_task=max_sources)

            reasoning = ask_openai_reasoning(company, ev_enriched) if OPENAI_API_KEY else {"per_task": {}, "x_post": {"jp":"", "en":""}}
            per_task = reasoning.get("per_task", {})
            x_post = reasoning.get("x_post", {"jp":"", "en":""})

            def cell(task: str, field: str, default=""):
                return per_task.get(task, {}).get(field, default)

            row = {
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
            }
            rows.append(row)

            detail_log.append({
                "company": company,
                "evidence": ev_enriched,
                "result": reasoning
            })
        except Exception as e:
            rows.append({"company": company, "error": str(e)})
            detail_log.append({"company": company, "error": str(e), "trace": traceback.format_exc()})

        progress.progress(i/len(companies))
        time.sleep(0.05)

    out = pd.DataFrame(rows)

    with tabs[1]:
        st.success("完了！")
        st.dataframe(out, use_container_width=True)
        csv = out.to_csv(index=False)
        st.download_button("CSVをダウンロード", data=csv, file_name="corporate_fit_with_reasons.csv", mime="text/csv")

    with tabs[2]:
        for block in detail_log:
            with st.expander(f"🔍 {block.get('company')} の詳細"):
                if "error" in block:
                    st.error(block["error"])
                    st.code(block.get("trace",""))
                else:
                    st.markdown("**参照 Evidence（各タスク 上位）**")
                    for task, items in block["evidence"].items():
                        st.markdown(f"- **{task}**")
                        for it in items:
                            st.write(f"[{it['title']}]({it['link']})")
                            if it.get("snippet"):
                                st.caption(it["snippet"])
                    st.markdown("**LLM 出力（JSON）**")
                    st.code(json.dumps(block["result"], ensure_ascii=False, indent=2))
