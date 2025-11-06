# app.py
import os
import re
import time
import json
import html
import base64
import traceback
import unicodedata
import io
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlencode, urlparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

# =========================
# 設定（Secrets を優先。UIには出さない）
# =========================
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GOOGLE_CSE_ID  = st.secrets.get("GOOGLE_CSE_ID",  os.getenv("GOOGLE_CSE_ID",  ""))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# =========================
# OpenAI (Responses API)
# =========================
from openai import OpenAI  # pip install openai>=1.0.0
_oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
MODEL_REASON = os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1-mini")

# =========================
# ユーティリティ
# =========================
SAFE_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CorporateStartupFit/1.3)"}
REQUEST_TIMEOUT = 20

def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def _domain_score(url: str) -> int:
    host = urlparse(url).netloc.lower()
    score = 0
    if host.endswith(".co.jp") or host.endswith(".com"): score += 1
    if any(k in host for k in ["ir.", "prtimes.jp", "prtimes.co.jp", "news.", "press"]): score += 2
    if any(p in url.lower() for p in ["/ir", "/investor", "/press", "/news", "/release"]): score += 2
    return score

def _dedup_urls(items: List[Dict[str, str]], max_per_domain: int = 3) -> List[Dict[str, str]]:
    seen = set()
    domain_counter = defaultdict(int)
    out = []
    for it in items:
        u = it.get("link", "")
        if not u or u in seen: continue
        d = urlparse(u).netloc
        if domain_counter[d] >= max_per_domain: continue
        seen.add(u); domain_counter[d] += 1; out.append(it)
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
    items = [{"title": it.get("title",""), "link": it.get("link",""), "snippet": it.get("snippet","")} for it in j.get("items", [])]
    items.sort(key=lambda x: _domain_score(x["link"]), reverse=True)
    return _dedup_urls(items, max_per_domain=2)

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_text(url: str, max_chars: int = 5000) -> str:
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
    quoted = f"\"{company.strip()}\""
    return {
        "CVC": [
            f"{quoted} CVC コーポレートベンチャーキャピタル 立ち上げ 投資子会社",
            f"{quoted} corporate venture capital CVC fund launch investing arm",
        ],
        "LP": [
            f"{quoted} LP 出資 リミテッドパートナー ベンチャーファンド 出資参画",
            f"{quoted} limited partner LP commitment venture fund investor",
        ],
        "AI_Robotics": [
            f"{quoted} AI ロボティクス 提携 出資 共同開発 スタートアップ",
            f"{quoted} AI robotics partnership investment startup collaboration",
        ],
        "Healthcare": [
            f"{quoted} ヘルスケア 医療 デジタルヘルス 提携 出資 共同研究",
            f"{quoted} healthcare medtech digital health partnership investment",
        ],
        "Climate": [
            f"{quoted} 脱炭素 クライメートテック 再生可能エネルギー 水素 CCS 提携 出資",
            f"{quoted} climate tech decarbonization renewable hydrogen CCS partnership investment",
        ],
    }

@st.cache_data(show_spinner=False, ttl=60*60)
def gather_evidence(company: str, per_query_limit: int = 6, per_task_urls: int = 6) -> Dict[str, List[Dict[str, str]]]:
    queries = _queries_for(company)
    ev_raw: Dict[str, List[Dict[str, str]]] = {}
    for k, qlist in queries.items():
        bucket = []
        for q in qlist:
            time.sleep(0.2)
            bucket.extend(google_search(q, num=per_query_limit))
        seen = set(); uniq = []
        for it in bucket:
            u = it["link"]
            if u in seen: continue
            seen.add(u); uniq.append(it)
        ev_raw[k] = uniq[:per_task_urls]
    return ev_raw

@st.cache_data(show_spinner=False, ttl=60*60)
def hydrate_evidence_with_content(evidence: Dict[str, List[Dict[str, str]]], max_sources_per_task: int = 5) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for task, items in evidence.items():
        enriched = []
        for it in items[:max_sources_per_task]:
            url = it["link"]
            body = fetch_text(url, max_chars=5000)
            enriched.append({"title": it.get("title",""), "link": url, "snippet": it.get("snippet",""), "body": body})
        out[task] = enriched
    return out

# =========================
# 会社名フィット・スコアリング
# =========================
JP_CORP_SUFFIXES = ["株式会社", "（株）", "(株)", "ホールディングス", "ホールディングス株式会社", "グループ", "グループ株式会社"]
EN_CORP_SUFFIXES = ["Co., Ltd.", "Co.,Ltd.", "Company, Limited", "Inc.", "Incorporated", "Corporation", "Corp.", "Holdings", "Group", "Limited", "Ltd."]

def _normalize_name(n: str) -> str:
    s = unicodedata.normalize("NFKC", n or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _strip_corp_words(n: str) -> str:
    s = n
    for w in JP_CORP_SUFFIXES: s = s.replace(w, "")
    for w in EN_CORP_SUFFIXES: s = s.replace(w, "")
    s = s.replace("Kabushiki Kaisha", "").replace("K.K.", "")
    s = re.sub(r"[.,・／/|｜\\\-‐-–—~〜()\[\]{}＜＞<>]", " ", s)
    s = re.sub(r"\s+", "", s).lower()
    return s

def _variants_for_target(company: str) -> List[str]:
    base = _normalize_name(company)
    v = {base}
    if base.startswith("株式会社"): v.add(base.replace("株式会社", "", 1).strip())
    if base.endswith("株式会社"):   v.add(base.replace("株式会社", "").strip())
    return list({_strip_corp_words(x) for x in v})

COMPANY_PATTERNS = [
    r"株式会社\s*([^\s、。：「」『』()（）【】\n]{1,30})",
    r"([^\s、。：「」『』()（）【】\n]{1,30})\s*株式会社",
    r"（株）\s*([^\s、。：「」『』()（）【】\n]{1,30})",
    r"([A-Z][A-Za-z0-9&.\- ]{1,60})\s+(?:Co\.?,?\s*Ltd\.?|Inc\.|Corporation|Corp\.|Holdings|Group|Limited|Ltd\.)",
]

def _extract_company_like_names(text: str) -> List[str]:
    if not text: return []
    names = []
    for pat in COMPANY_PATTERNS:
        for m in re.findall(pat, text):
            if isinstance(m, tuple): m = m[0]
            nm = _normalize_name(m)
            if 1 <= len(nm) <= 60: names.append(nm)
    return names

def _company_fit_score_for_item(company: str, title: str, snippet: str, body: str) -> Tuple[float, Counter, int]:
    target_vars = _variants_for_target(company)
    title_n = unicodedata.normalize("NFKC", title or "")
    snip_n  = unicodedata.normalize("NFKC", snippet or "")
    body_n  = unicodedata.normalize("NFKC", body or "")

    def count_target(s: str) -> int:
        c = 0
        s_norm = _strip_corp_words(s)
        for tv in target_vars:
            if not tv: continue
            c += len(re.findall(re.escape(tv), s_norm, flags=re.IGNORECASE))
        return c

    title_hit = count_target(title_n) > 0
    snip_hit  = count_target(snip_n)  > 0
    body_cnt  = count_target(body_n)

    names = _extract_company_like_names(body_n + " " + title_n)
    other_counter = Counter()
    for n in names:
        norm = _strip_corp_words(n)
        if norm and norm not in target_vars:
            other_counter[norm] += 1
    max_other = max(other_counter.values()) if other_counter else 0

    score = (2 if title_hit else 0) + (1 if snip_hit else 0) + body_cnt - 2 * max_other
    return score, other_counter, body_cnt

def filter_evidence_by_company(company: str, evidence_enriched: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for task, items in evidence_enriched.items():
        scored = []
        for it in items:
            s, others, tgt_body_cnt = _company_fit_score_for_item(company, it.get("title",""), it.get("snippet",""), it.get("body",""))
            scored.append((s, tgt_body_cnt, it, others))
        scored.sort(key=lambda x: x[0], reverse=True)
        kept = []
        for s, tgt_cnt, it, _ in scored:
            if tgt_cnt >= 1 and s >= 1.0:
                kept.append(it)
        out[task] = kept
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

Decision hygiene (company-specific):
- Ignore any evidence where the target company name does NOT appear in the title or body at least once.
- If multiple specific company names appear, treat the article as valid ONLY if the target's mentions are not fewer than the most frequently mentioned other company name in that article. Otherwise, mark as 'Unclear'.

Definitions and decision rules (apply strictly):
- CVC: The company has its own corporate venture capital arm or investment subsidiary (e.g., 'CVC', 'corporate venture capital', 'investment subsidiary', 'capital partners'). One-off venture investments without a dedicated arm → do NOT mark 'Yes'.
- LP: The company has committed capital as a limited partner to an external venture fund (e.g., 'LP', 'limited partner', 'commitment', '出資', 'リミテッドパートナー'). If only investing directly as a CVC without LP commitment, mark 'No' (unless LP is also evidenced).
- Synergy (AI_Robotics / Healthcare / Climate): Mark 'Yes' only if there is concrete evidence of products, partnerships, investments, pilots, or stated strategic focus that clearly connects to the domain. Generic news unrelated to the company's business → 'Unclear'.

Hard constraints:
- Output must be valid JSON (UTF-8, no trailing commas, no comments).
- For each task, set 'label' ∈ {{'Yes','No','Unclear'}}, 'confidence' ∈ [0,1].
- 'reason_ja': ≤100 Japanese characters. 'reason_en': 1–2 sentences English.
- 'evidence_urls': include up to 3 URLs, but ONLY from the provided evidence links. Do NOT invent URLs.
- If signals conflict or are outdated without follow-ups, prefer 'Unclear'.
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
    resp = _oai.responses.create(
        model=MODEL_REASON,
        temperature=0.0,
        max_output_tokens=1500,
        input=f"System:\n{PROMPT_SYSTEM}\n\nUser:\n{prompt_user}",
    )
    text = resp.output_text
    data = _safe_json_loads(text)
    if not data or "per_task" not in data:
        resp2 = _oai.responses.create(
            model=MODEL_REASON,
            temperature=0.0,
            max_output_tokens=1500,
            input=("System:\n" + PROMPT_SYSTEM + "\n\nUser:\nReturn ONLY valid JSON per the schema. "
                   "If previous attempt failed, correct and resend the JSON.\n" + prompt_user),
        )
        text = resp2.output_text
        data = _safe_json_loads(text)

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
# ダウンロード支援ユーティリティ
# =========================
def _to_b64_csv(df: pd.DataFrame) -> str:
    csv = df.to_csv(index=False)
    return base64.b64encode(csv.encode("utf-8")).decode()

def _auto_download(b64: str, filename: str):
    st.components.v1.html(
        f"""
        <html><body>
        <a id="autodl" href="data:text/csv;base64,{b64}" download="{filename}"></a>
        <script>document.getElementById('autodl').click();</script>
        </body></html>
        """,
        height=0,
    )

# =========================
# Streamlit UI（Secrets セクションなし）
# =========================
st.set_page_config(page_title="Corporate–Startup Fit Checker+", layout="wide")
st.title("🏢➡️🤝🚀 Corporate–Startup Fit Checker+")
st.caption("C列=会社名。証跡→本文取得→会社名スコアで他社記事を除外→OpenAIで判定。中間CSVを自動保存。")

cols = st.columns(4)
with cols[0]:
    uploaded = st.file_uploader("Excel をアップロード（C列=会社名）", type=["xlsx", "xls"])
with cols[1]:
    limit = st.number_input("処理件数の上限", 1, 20000, 200, 50)
with cols[2]:
    max_sources = st.slider("各タスクの最大参照URL数", 1, 8, 5)
with cols[3]:
    checkpoint_every = st.number_input("自動保存（社ごと）", 1, 200, 25, 5)

# ▼ アップロードの永続化（セッション内）
if uploaded is not None:
    st.session_state["uploaded_bytes"] = uploaded.getvalue()
    st.session_state["uploaded_name"] = getattr(uploaded, "name", "input.xlsx")

# 自動DLの一回制御
if "auto_dl_done" not in st.session_state:
    st.session_state.auto_dl_done = False

run = st.button("解析スタート", type="primary", disabled=("uploaded_bytes" not in st.session_state))

if run and ("uploaded_bytes" in st.session_state):
    # セッションから読み直し（途中再実行でも継続可能）
    data_bytes = st.session_state["uploaded_bytes"]
    filelike = io.BytesIO(data_bytes)

    df = pd.read_excel(filelike)
    if df.shape[1] >= 3:
        companies = df.iloc[:, 2].dropna().astype(str).tolist()
    else:
        companies = df.iloc[:, -1].dropna().astype(str).tolist()
    companies = companies[:int(limit)]

    rows = []
    progress = st.progress(0.0)
    status = st.empty()
    tabs = st.tabs(["進捗", "最終テーブル / ダウンロード", "詳細ログ"])
    with tabs[0]:
        st.write("検索→本文取得→会社名フィルタ→判定をループ。一定社数ごとに自動でCSV保存します。")

    detail_log = []
    st.session_state.auto_dl_done = False  # 新規開始ごとにリセット

    for i, company in enumerate(companies, 1):
        status.info(f"Searching & analyzing: {company}")
        try:
            ev = gather_evidence(company)
            ev_enriched = hydrate_evidence_with_content(ev, max_sources_per_task=max_sources)
            ev_enriched = filter_evidence_by_company(company, ev_enriched)
            reasoning = ask_openai_reasoning(company, ev_enriched) if OPENAI_API_KEY else {"per_task": {}, "x_post": {"jp":"", "en":""}}
            per_task = reasoning.get("per_task", {})
            x_post = reasoning.get("x_post", {"jp":"", "en":""})

            def cell(task: str, field: str, default=""):
                return per_task.get(task, {}).get(field, default)

            row = {
                "company": company,
                "CVC":        cell("CVC", "label", "Unclear"),
                "LP":         cell("LP", "label", "Unclear"),
                "AI_Robotics":cell("AI_Robotics", "label", "Unclear"),
                "Healthcare": cell("Healthcare", "label", "Unclear"),
                "Climate":    cell("Climate", "label", "Unclear"),
                "CVC_conf":        cell("CVC", "confidence", ""),
                "LP_conf":         cell("LP", "confidence", ""),
                "AI_Robotics_conf":cell("AI_Robotics", "confidence", ""),
                "Healthcare_conf": cell("Healthcare", "confidence", ""),
                "Climate_conf":    cell("Climate", "confidence", ""),
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
                "CVC_urls":        "; ".join(per_task.get("CVC", {}).get("evidence_urls", [])),
                "LP_urls":         "; ".join(per_task.get("LP", {}).get("evidence_urls", [])),
                "AI_Robotics_urls":"; ".join(per_task.get("AI_Robotics", {}).get("evidence_urls", [])),
                "Healthcare_urls": "; ".join(per_task.get("Healthcare", {}).get("evidence_urls", [])),
                "Climate_urls":    "; ".join(per_task.get("Climate", {}).get("evidence_urls", [])),
                "x_post_jp": x_post.get("jp", ""),
                "x_post_en": x_post.get("en", "")
            }
            rows.append(row)

            detail_log.append({"company": company, "evidence": ev_enriched, "result": reasoning})
        except Exception as e:
            rows.append({"company": company, "error": str(e)})
            detail_log.append({"company": company, "error": str(e), "trace": traceback.format_exc()})

        # 進捗更新（サーバ・ブラウザ双方のアイドル切断回避に有効）
        progress.progress(i/len(companies))
        time.sleep(0.02)

        # ▼ チェックポイント保存・自動DL（一定社数ごと）
        if i % int(checkpoint_every) == 0:
            partial_df = pd.DataFrame(rows)
            b64 = _to_b64_csv(partial_df)
            _auto_download(b64, f"corporate_fit_checkpoint_{i:05d}.csv")
            with tabs[1]:
                st.toast(f"中間CSV（{i}社時点）を自動ダウンロードしました。", icon="✅")
                st.dataframe(partial_df.tail(20), use_container_width=True)

    # ===== 最終結果 =====
    out = pd.DataFrame(rows)
    with tabs[1]:
        st.success("解析完了！")
        st.dataframe(out, use_container_width=True)

        csv_b64 = _to_b64_csv(out)
        # 手動DL（保険）
        st.download_button(
            "最終CSVをダウンロード",
            data=base64.b64decode(csv_b64),
            file_name="corporate_fit_with_reasons.csv",
            mime="text/csv",
        )
        # 自動DL（最終）
        _auto_download(csv_b64, "corporate_fit_with_reasons.csv")
        st.info("最終CSVを自動ダウンロードしました。ブロックされた場合はボタンから保存してください。")

    with tabs[2]:
        for block in detail_log:
            with st.expander(f"🔍 {block.get('company')} の詳細"):
                if "error" in block:
                    st.error(block["error"])
                    st.code(block.get("trace",""))
                else:
                    st.markdown("**参照 Evidence（各タスク）**")
                    for task, items in block["evidence"].items():
                        st.markdown(f"- **{task}**")
                        for it in items:
                            st.write(f"[{it['title']}]({it['link']})")
                            if it.get("snippet"):
                                st.caption(it["snippet"])
                    st.markdown("**LLM 出力（JSON）**")
                    st.code(json.dumps(block["result"], ensure_ascii=False, indent=2))
