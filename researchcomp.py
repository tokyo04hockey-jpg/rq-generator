import time
import re
import csv
import math
from typing import List, Dict, Any, Tuple
import requests
import pandas as pd
import streamlit as st
from tqdm import tqdm

st.set_page_config(page_title="Company Web Check (CVC / LP / Synergy)", layout="wide")

API_KEY = st.secrets["GOOGLE_API_KEY"]
CSE_ID  = st.secrets["GOOGLE_CSE_ID"]

# -------------------------
# Google Custom Search
# -------------------------
SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

@st.cache_data(show_spinner=False)
def google_search(query: str, num: int = 5, lang: str = "ja") -> List[Dict[str, Any]]:
    """Call Custom Search API. Cached to save quota."""
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": min(max(num, 1), 10),  # APIの1回当たり最大10
        "hl": lang,
        "lr": "lang_ja" if lang == "ja" else None,
        "safe": "off",
    }
    # remove None
    params = {k: v for k, v in params.items() if v is not None}
    r = requests.get(SEARCH_ENDPOINT, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("items", []) or []

# -------------------------
# 判定ロジック（超シンプルなヒューリスティック）
# -------------------------
def hit(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in patterns)

def score_from_results(items: List[Dict[str, Any]], must: List[str], any_of: List[str]) -> Tuple[float, str]:
    """タイトル+スニペットを走査して簡易スコアと根拠URLを返す"""
    best = (0.0, "")
    for it in items:
        title = it.get("title", "")
        snippet = it.get("snippet", "")
        url = it.get("link", "")
        text = f"{title}\n{snippet}"
        must_ok = all(hit(text, [m]) for m in must) if must else True
        any_ok  = hit(text, any_of) if any_of else True
        base = 0.0
        if must_ok and any_ok:
            base = 0.9
        elif must_ok or any_ok:
            base = 0.6
        # ドメインが公式/PR系なら上積み
        if re.search(r"(ir\.|prtimes|newsroom|press|release|investor|corp|company)", url.lower()):
            base += 0.05
        if base > best[0]:
            best = (min(base, 1.0), url)
    return best

def judge_cvc(company: str, lang="ja") -> Tuple[str, int, str]:
    q = f'"{company}" (CVC OR "corporate venture capital" OR Ventures OR ベンチャーズ OR コーポレートベンチャーキャピタル)'
    items = google_search(q, num=8, lang=lang)
    score, url = score_from_results(
        items,
        must=[company],
        any_of=["CVC","corporate venture capital","ventures","ベンチャーズ","コーポレートベンチャー"]
    )
    return ("Yes" if score >= 0.75 else "Maybe" if score >= 0.6 else "No", round(score*100), url)

def judge_lp(company: str, lang="ja") -> Tuple[str, int, str]:
    q = f'"{company}" ("limited partner" OR LP出資 OR LP投資 OR "committed as LP" OR ファンド出資)'
    items = google_search(q, num=8, lang=lang)
    score, url = score_from_results(
        items,
        must=[company],
        any_of=["limited partner","LP出資","LP投資","LPとして","ファンドへ出資","出資を決定","committed as LP"]
    )
    return ("Yes" if score >= 0.75 else "Maybe" if score >= 0.6 else "No", round(score*100), url)

def judge_synergy(company: str, theme: str, lang="ja") -> Tuple[str, int, str]:
    if theme == "AI/Robotics":
        any_of = ["AI","人工知能","ロボティクス","ロボット","オートメーション","生成AI","機械学習","automation","robotics"]
    elif theme == "Healthcare":
        any_of = ["ヘルスケア","医療","メドテック","デジタルヘルス","病院","製薬","医薬","biotech","healthcare","medtech"]
    elif theme == "Climate tech":
        any_of = ["クライメートテック","脱炭素","再生可能エネルギー","水素","バッテリー","CCUS","カーボン","再エネ",
                  "climate tech","decarbonization","renewable","hydrogen","battery","carbon capture","sustainability"]
    else:
        any_of = []

    q = f'"{company}" (partnership OR 提携 OR 協業 OR 共同開発 OR investment OR 出資 OR 買収) ' + " ".join(any_of[:4])
    items = google_search(q, num=8, lang=lang)
    score, url = score_from_results(
        items,
        must=[company],
        any_of=any_of + ["提携","協業","共同","出資","buy","acquire","investment","partnership"]
    )
    return ("Likely" if score >= 0.7 else "Possible" if score >= 0.55 else "Unclear", round(score*100), url)

# -------------------------
# UI
# -------------------------
st.title("🏢 Company Web Check（CVC / LP / Synergy 判定）")

st.markdown(
    "- 入力：Excel（C列=会社名）\n"
    "- 出力：CSV（判定・信頼度・根拠URL）\n"
    "- 検索API：Google Custom Search"
)

uploaded = st.file_uploader("Excel / CSV をアップロード", type=["xlsx","csv"])
lang = st.selectbox("検索言語", ["ja","en"], index=0)
throttle_ms = st.slider("検索スロットリング（ミリ秒/クエリ）", 0, 2000, 200)

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # C列を使う（3列目）。列数不足に備えた保険。
    if df.shape[1] < 3:
        st.error("このファイルにはC列がありません。C列に会社名を入れて再アップロードしてください。")
        st.stop()

    companies = df.iloc[:, 2].dropna().astype(str).tolist()

    st.write(f"読み込み件数：{len(companies)} 件（先頭5件表示）")
    st.dataframe(pd.DataFrame({"company": companies[:5]}))

    if st.button("判定を開始"):
        rows = []
        pbar = st.progress(0)
        total = len(companies)

        for i, company in enumerate(companies, start=1):
            try:
                cvc, cvc_conf, cvc_url = judge_cvc(company, lang=lang)
                lp,  lp_conf,  lp_url  = judge_lp(company, lang=lang)
                ai,  ai_conf,  ai_url  = judge_synergy(company, "AI/Robotics", lang=lang)
                hc,  hc_conf,  hc_url  = judge_synergy(company, "Healthcare",  lang=lang)
                cl,  cl_conf,  cl_url  = judge_synergy(company, "Climate tech", lang=lang)

                rows.append({
                    "Company": company,
                    "CVC": cvc, "CVC_confidence": cvc_conf, "CVC_evidence": cvc_url,
                    "LP_investor": lp, "LP_confidence": lp_conf, "LP_evidence": lp_url,
                    "AI/Robotics_synergy": ai, "AI_confidence": ai_conf, "AI_evidence": ai_url,
                    "Healthcare_synergy": hc, "Healthcare_confidence": hc_conf, "Healthcare_evidence": hc_url,
                    "ClimateTech_synergy": cl, "Climate_confidence": cl_conf, "Climate_evidence": cl_url,
                })
            except requests.HTTPError as e:
                rows.append({
                    "Company": company, "Error": f"HTTPError {e.response.status_code}", 
                })
            except Exception as e:
                rows.append({
                    "Company": company, "Error": str(e),
                })

            pbar.progress(i/total)
            time.sleep(throttle_ms/1000.0)

        result_df = pd.DataFrame(rows)
        st.success("解析完了！CSVをダウンロードできます。")
        st.dataframe(result_df.head(20))

        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")  # Excelで文字化けしないBOM付
        st.download_button(
            "結果CSVをダウンロード",
            data=csv_bytes,
            file_name="company_webcheck_result.csv",
            mime="text/csv",
        )

st.caption("※判定は簡易ヒューリスティックです。重要判断は原典リンクを必ずご確認ください。")
