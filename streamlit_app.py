# streamlit_app.py — スコアリング & ランキング付き
import streamlit as st
import requests
import pandas as pd
from urllib.parse import quote
from openai import OpenAI

st.title("🎓 Research Question Generator")
st.write("パネル/インタビュー要約から4視点（逆張り/飛ばし/トレードオフ幻像/アナロジー）で研究クエスチョンを生成し、"
         "新規性×実用性でスコアリングしてランキングします。")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ========= ユーティリティ =========
@st.cache_data(show_spinner=False, ttl=600)
def openalex_count(query: str) -> int:
    # OpenAlex works API: 概算件数を返す（最大1万までしか正確に出ないが指標として十分）
    url = f"https://api.openalex.org/works?search={quote(query)}&per_page=1"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        meta = r.json().get("meta", {})
        return int(meta.get("count", 0))
    except Exception:
        return -1  # エラー時

def novelty_score_from_count(n: int) -> int:
    # 件数が少ないほど高スコア（0〜5）
    if n < 0:   return 2  # 不明なら中間寄り
    if n < 50:  return 5
    if n < 150: return 4
    if n < 400: return 3
    if n < 1000:return 2
    return 1

def openalex_link(query: str) -> str:
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
    # ざっくり抽出（厳密にJSONで返らないケースも想定してフォールバック）
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

def generate_rqs(context: str) -> dict:
    prompt = f"""
以下の議論要約に基づき、4つの視点で各1問ずつ、日本語で研究クエスチョンを提案してください。
1. 逆張り（前提を逆に見る）
2. 飛ばし（手段Bを前提とせずAを達成する方法）
3. トレードオフの幻想（AとBを同時達成できる条件）
4. アナロジー（他分野への転用）

それぞれ「見出し：質問文」の形式で簡潔に。
[議論要約]
{context}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    out = resp.choices[0].message.content.strip()
    # シンプルに行単位で拾う
    lines = [l.strip("- ").strip() for l in out.splitlines() if l.strip()]
    buckets = {"逆張り": "", "飛ばし": "", "トレードオフの幻想": "", "アナロジー": ""}
    for k in list(buckets.keys()):
        for l in lines:
            if l.startswith(k):
                q = l.split("：", 1)[-1].split(":", 1)[-1].strip()
                buckets[k] = q or l
                break
    # 見つからなければ先頭から埋める
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

if st.button("生成 & スコアリング"):
    if not summary.strip():
        st.warning("先に要約を入力してください。")
    else:
        with st.spinner("研究クエスチョンを生成中..."):
            rqs = generate_rqs(summary)

        rows = []
        with st.spinner("スコア算出中..."):
            for frame, q in rqs.items():
                # 新規性（OpenAlex件数）
                query = q if len(q) > 10 else (summary + " " + q)
                count = openalex_count(query)
                nov = novelty_score_from_count(count)
                link = openalex_link(query)
                # 実用性（LLM）
                util = ask_gpt_utility(q, summary)
                score = round(nov * w_n + util["score"] * w_u, 2)
                rows.append({
                    "発想フレーム": frame,
                    "研究クエスチョン": q,
                    "新規性(1-5)": nov,
                    "実用性(1-5)": util["score"],
                    "総合スコア": score,
                    "実用性コメント": util["reason"],
                    "OpenAlex検索": link,
                    "件数(目安)": count if count >= 0 else "N/A",
                })

        df = pd.DataFrame(rows).sort_values("総合スコア", ascending=False).reset_index(drop=True)
        st.subheader("🏁 ランキング")
        st.dataframe(df, use_container_width=True)

        # 便利リンクとダウンロード
        st.markdown("**🔗 エビデンス確認（OpenAlex）**：各行のリンクから関連文献を素早く確認できます。")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSVダウンロード", data=csv, file_name="rq_ranked.csv", mime="text/csv")

        # Markdown版（メモ貼り付け用）
        md = df[["発想フレーム","研究クエスチョン","新規性(1-5)","実用性(1-5)","総合スコア"]].to_markdown(index=False)
        st.download_button("⬇️ Markdownダウンロード", data=md, file_name="rq_ranked.md", mime="text/markdown")
