import re
import textwrap
import streamlit as st
import pandas as pd
from urllib.parse import urlparse
from openai import OpenAI

st.set_page_config(page_title="X Post Drafts (Scholar/Google Alerts)", layout="centered")
st.title("🧪 Xポスト下書きジェネレータ（Scholar/Google Alerts対応）")

st.write("貼り付けたアラート本文から **日本語(テキストのみ)** / **英語(リンク付き)** のX投稿案を、無料アカウント上限内で生成します。")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Xの無料投稿制限とURL換算 ---
X_LIMIT = 280
TCO_URL_LEN = 23  # t.co短縮換算（無料でも同じ扱いを想定）

def extract_urls(text: str) -> list[str]:
    url_re = re.compile(r'https?://\S+')
    return url_re.findall(text)

def sanitize_url(u: str) -> str:
    # 余計なトラッキングは簡易に除去
    try:
        p = urlparse(u)
        if not p.scheme:
            return u
        base = f"{p.scheme}://{p.netloc}{p.path}"
        return base
    except Exception:
        return u

def x_count_len(text: str) -> int:
    """URLを23文字換算で合計文字数を概算。"""
    urls = extract_urls(text)
    tmp = text
    for u in urls:
        tmp = tmp.replace(u, " " * TCO_URL_LEN, 1)  # 置換で長さだけ合わせる
    return len(tmp)

def clip_to_limit(text: str, limit: int = X_LIMIT) -> str:
    if x_count_len(text) <= limit:
        return text
    # 末尾に…をつける余地を確保
    ell = "…"
    # URLは壊さない：一旦URLを退避し、本文だけをトリムしてから戻す
    urls = extract_urls(text)
    base = text
    for u in urls:
        base = base.replace(u, "")  # 先に本文だけに
    base = base.strip()
    # できるだけ文を壊さずに短縮
    while x_count_len(base + ((" " + " ".join(urls)) if urls else "")) + len(ell) > limit and len(base) > 0:
        base = base[:-1]
    clipped = base.rstrip() + ell
    final = clipped + ((" " + " ".join(urls)) if urls else "")
    # 念のため最終チェック
    if x_count_len(final) > limit:
        # さらに削る（安全側）
        over = x_count_len(final) - limit
        clipped2 = clipped[:-max(1, over)]
        final = clipped2
    return final.strip()

def summarize_alert(alert_text: str) -> dict:
    """タイトル/要点/英語タイトルなどを抽出"""
    sys = "You are a helpful research assistant for social posts."
    prompt = f"""
From the following Google Scholar/Google Alert text, extract:
1) short Japanese title (<=60 chars),
2) short English title (<=80 chars),
3) 1-2 sentence Japanese summary (<=180 chars),
4) 1 sentence English summary (<=220 chars).
Return strict JSON with keys: ja_title, en_title, ja_sum, en_sum.

Alert:
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":prompt}],
        temperature=0.2,
    )
    import json, re
    txt = resp.choices[0].message.content.strip()
    m = re.search(r"\{.*\}", txt, re.S)
    data = {"ja_title":"", "en_title":"", "ja_sum":"", "en_sum":""}
    if m:
        try:
            data.update(json.loads(m.group(0)))
        except Exception:
            pass
    # フォールバック
    for k in data:
        if not data[k]:
            data[k] = ""
    return data

def build_ja_post(meta: dict, hashtags: list[str]) -> str:
    body = f"{meta['ja_title']} — {meta['ja_sum']}".strip(" —")
    if hashtags:
        body += "\n" + " ".join(hashtags)
    return clip_to_limit(body)

def build_en_post(meta: dict, urls: list[str], hashtags: list[str]) -> str:
    # 出典URLは1本だけ（優先：Scholar本文/論文URL）
    link = urls[0] if urls else ""
    link = sanitize_url(link) if link else ""
    pieces = [f"{meta['en_title']}".strip(), meta['en_sum'].strip(), link.strip()]
    body = " — ".join([p for p in pieces if p])
    if hashtags:
        body += "\n" + " ".join(hashtags)
    return clip_to_limit(body)

st.markdown("**入力**（アラート本文を貼り付け／URLは自動抽出）")
alert = st.text_area("Google Scholar Alert / Google Alert の本文", height=220, placeholder="アラート本文や見出し＋URLを貼り付け")

col1, col2 = st.columns(2)
with col1:
    ja_tags = st.text_input("日本語ハッシュタグ（任意、空白区切り）", value="#研究 #政策 #VC")
with col2:
    en_tags = st.text_input("English hashtags (optional, space-separated)", value="#research #policy #VC")

if st.button("ドラフト生成"):
    if not alert.strip():
        st.warning("まずアラート本文を貼り付けてください。")
    else:
        with st.spinner("抽出・要約中..."):
            urls = extract_urls(alert)
            meta = summarize_alert(alert)

        ja_hashtags = [t for t in ja_tags.split() if t.startswith("#")]
        en_hashtags = [t for t in en_tags.split() if t.startswith("#")]

        ja_post = build_ja_post(meta, ja_hashtags)
        en_post = build_en_post(meta, urls, en_hashtags)

        st.subheader("📌 生成結果（X無料枠 280文字内）")
        st.markdown("**日本語ドラフト**")
        st.code(ja_post, language="markdown")
        st.caption(f"長さ: {len(ja_post)}（URLは含まれていない想定） / 上限 {X_LIMIT}")

        st.markdown("**English draft (w/ source link)**")
        st.code(en_post, language="markdown")
        st.caption(f"概算長さ(x換算): {len(en_post)}（URLは23文字換算） / 上限 {X_LIMIT}")

        # CSV出力（BOM付きUTF-8でExcel対策）
        df = pd.DataFrame([{
            "ja_title": meta.get("ja_title",""),
            "ja_post": ja_post,
            "en_title": meta.get("en_title",""),
            "en_post": en_post,
            "source_url": urls[0] if urls else "",
        }])
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ CSVダウンロード（UTF-8/BOM）", data=csv, file_name="x_post_drafts.csv", mime="text/csv")

st.markdown("---")
st.caption("※ 文字数換算はX無料ユーザーの280文字上限およびURL=23文字想定に準拠（参考: character limit 280 / URLはt.coで23文字扱い）。")
