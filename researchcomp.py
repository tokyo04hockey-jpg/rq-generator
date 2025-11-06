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
SAFE_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CorporateStartupFit/1.4)"}
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
    s =
