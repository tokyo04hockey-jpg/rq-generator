# streamlit_app.py — RQ生成（関心領域視点込み）＋CSV出力＋論文リンク
import streamlit as st
import requests
import pandas as pd
from urllib.parse import quote
from openai import OpenAI

st.title("🎓 Research Question Generator")
st.write(
    "パネル/インタビュー要約から4視点（逆張り/飛ばし/トレードオフ幻像/アナロジー）で研究クエスチョンを生成し、"
    "ご関心領域の観点（Entrepreneurship & Innovation / VC & Entrepreneurial Finance / Public Policy & Institutional Design / "
    "Applied Econometrics / Cross-border Investment）を付与。新規性×実用性でスコアリングしてCSVを出力します。"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ========= ユーティリティ =========
@st.cache_data(show_spinner=False, ttl=600)
def openalex_count(query: str) -> int:
    url = f"https://api.openalex.org/works?search={quote(query)}&per_page=1"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return int(r.json().get("meta", {}).get("count", 0))
    except Exception:
        return -1  # 不明

@st.cache_data(show_spinner=False, ttl=600)
def openalex_top_links(query: str, n: int = 3) -> list[dict]:
    """関連上位論文のリンクとタイトルを返す（OpenAlex/DOI）"""
    url = f"https://api.openalex.org/works?search={quote(query)}&per_page={n}&sort=cited_by_count:desc"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        items = r.json().get("results", [])
        out = []
        for it in items:
            oid = it.get("id", "")  # e.g., https://openalex.org/W123...
            title = (it.get("title") or "").strip()
            doi = (it.get("doi") or "").strip()  # e.g., https://doi.org/...
            olink = oid if oid else ""
            dlink = doi if doi else ""
            # 表示は「Title | OpenAlex | DOI(あれば)」
            disp = title
            links = [olink] + ([dlink] if dlink else [])
            out.append({"title": disp, "links": " | ".join([l for l in links if l])})
        return out
    except Exception:
        return []

def novelty_score_from_count(n: int) -> int:
    if n < 0:   return 2
    if n < 50:  return 5
    if n < 150: return 4
    if n < 400: return 3
    if n < 1000:return 2
    return 1

def openalex_search_url(query: str) -> str:
    return f"https://api.openalex.org/works?search={quote(query)}"

def ask_gpt_utility(q: str, context: str) -> dict:
    prompt = f"""
以下の研究クエスチョンについて、ベンチャー投資家・政策立案者の実務にとっての有用性を5点満点で評価し、
短い理由を1〜2文で述べてください。JSONで返してください（keys: score, reason）。

[コンテキスト]
{context}

[研究クエスチョン]
{q}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    txt = resp.choices[0].message.content.strip()
    import json, re
    try:
        m = re.search(r"\{.*\}", txt, re.S)
        data = json.loads(m.group(0)) if m else {}
        score = int(data.get("score", 3))
        reason = str(data.get("reason", "")).strip()[:200]
    except Exception:
        score, reason = 3, txt[:200]
    score = max(1, min(5, score))
    return {"score": score, "reason": reason}

def ask_gpt_perspective_tags(q: str) -> list[str]:
    """関心領域に照らしてどの視点が強いかタグ付け"""
    prompt = f"""
次の研究クエスチョンについて、以下の関心領域のうち該当するものを1〜3個、短い英語タグで返してください。
返答はカンマ区切りのタグのみ（説明不要）。

- Entrepreneurship & Innovation
- Venture Capital & Entrepreneurial Finance
- Public Policy & Institutional Design
- Applied Econometrics
- Cross-border Investment

Question: {q}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    tags = [t.strip() for t in resp.choices[0].message.content.split(",") if t.strip()]
    return tags[:3]

def generate_rqs(context: str) -> dict:
    """4フレーム×各1問。関心領域を前置して生成を誘導。"""
    prompt = f"""
あなたはPhD研究支援アシスタントです。以下の関心領域の観点を常に意識して、議論要約から研究クエスチョンを作ってください：
- Entrepreneurship & Innovation
- Venture Capital & Entrepreneurial Finance
- Public Policy & Institutional Design
- Applied Econometrics
- Cross-border Investment

4つの発想フレームで各1問ずつ、日本語で簡潔に提示：
1. 逆張り（前提を逆に見る）
2. 飛ばし（手段Bを前提とせずAを達成する方法）
3. トレードオフの幻想（AとBを同時達成できる条件）
4. アナロジー（他分野への転用）

出力形式：各行「<フレーム>：<クエスチョン>」のみ（説明文なし）。

[議論要約]
{context}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    out = resp.choices[0].message.content.strip()
    lines = [l.strip("- ").strip() for l in out.splitlines() if l.strip()]
    buckets = {"逆張り": "", "飛ばし": "", "トレードオフの幻想": "", "アナロジー": ""}
    for k in list(buckets.keys()):
        for l in lines:
            if l.startswith(k):
                q = l.split("：", 1)[-1].split(":", 1)[-1].strip()
                buckets[k] = q or l
                break
    i = 0
    for k in list(buckets.keys()):
        if not buckets[k] and i < len(lines):
            buckets[k] = lines[i]
            i += 1
    return buckets

# ========= UI =========
colA, colB = st.columns([3, 2])
with colB:
    st.markdown("**スコア重み**")
    w_n = st.slider("新規性の重み", 0.0, 1.0, 0.6, 0.1)
    w_u = 1.0 - w_n
    st.caption(f"総合点 = 新規性×{w_n:.1f} + 実用性×{w_u:.1f}")
with colA:
    summary = st.text_area("議論の要約を入力してください", height=200)

if st.button("生成 & スコアリング（CSV出力つき）"):
    if not summary.strip():
        st.warning("先に要約を入力してください。")
    else:
        with st.spinner("研究クエスチョンを生成中..."):
            rqs = generate_rqs(summary)

        rows = []
        with st.spinner("スコア算出・リンク収集中..."):
            for frame, q in rqs.items():
                base_query = q if len(q) > 10 else (summary + " " + q)

                # 新規性（OpenAlex件数）
                count = openalex_count(base_query)
                nov = novelty_score_from_count(count)
                search_url = openalex_search_url(base_query)

                # 上位論文リンク（OpenAlex / DOI）
                top = openalex_top_links(base_query, n=3)
                top_links = "; ".join([f"{t['title']} | {t['links']}" for t in top]) if top else ""

                # 実用性（LLM）
                util = ask_gpt_utility(q, summary)

                # 関心領域タグ（LLM）
                tags = ask_gpt_perspective_tags(q)

                score = round(nov * w_n + util["score"] * w_u, 2)

                rows.append({
                    "発想フレーム": frame,
                    "研究クエスチョン": q,
                    "関心領域タグ": ", ".join(tags),
                    "新規性(1-5)": nov,
                    "実用性(1-5)": util["score"],
                    "総合スコア": score,
                    "実用性コメント": util["reason"],
                    "OpenAlex件数(目安)": count if count >= 0 else "N/A",
                    "OpenAlex検索URL": search_url,
                    "関連上位論文": top_links,  # 「Title | OpenAlex | DOI」
                })

        df = pd.DataFrame(rows).sort_values("総合スコア", ascending=False).reset_index(drop=True)
        st.subheader("🏁 ランキング（CSVにダウンロード可能）")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSVダウンロード", data=csv, file_name="rq_ranked_with_links.csv", mime="text/csv")
