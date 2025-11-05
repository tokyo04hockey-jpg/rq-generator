# x_news_draft_app_bs.py
# ------------------------------------------------------------
# Google Alerts の HTML を貼付/アップロード → BeautifulSoup で記事抽出
# → LLM で「最も良さそうな記事」を選定 → X 投稿向けドラフト（日/英）を生成
# 環境変数: OPENAI_API_KEY（Streamlit Cloud では Secrets で設定）
# ------------------------------------------------------------

import os
import re
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st
from bs4 import BeautifulSoup

# OpenAI v1 SDK
try:
    from openai import OpenAI
except Exception:
    st.error("openai パッケージが見つかりません。requirements.txt に 'openai>=1.40.0' を追加してください。")
    raise

# -----------------------------
# ユーティリティ
# -----------------------------
def extract_real_url(href: str) -> str:
    """GoogleのリダイレクトURLから実URLを取り出す"""
    if not href:
        return ""
    m = re.search(r"[?&]url=(https?://[^&]+)", href)
    if m:
        return m.group(1)
    return href

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def parse_google_alert_html(html: str) -> List[Dict[str, str]]:
    """
    Google Alerts メールHTMLから記事候補を抽出
    返り値: [{title, url, source, snippet}]
    """
    soup = BeautifulSoup(html, "html.parser")

    # 各 <a> を走査してニュース記事らしいものだけ抽出
    candidates = []
    for a in soup.find_all("a", href=True):
        text = clean_whitespace(a.get_text(" ", strip=True))
        href = a["href"]

        # 除外条件：アラート管理系、共有ボタン等
        if (
            not text
            or "google.com/alerts" in href
            or "alerts/share" in href
            or "facebook" in href.lower()
            or "twitter" in href.lower()
            or "Flag as irrelevant" in text
            or text.lower() in {"facebook", "twitter"}
        ):
            continue

        url = extract_real_url(href)

        # URL が媒体記事っぽくない場合はスキップ（雑に判定）
        if not url.startswith("http"):
            continue
        if "google.com/url" in url:
            # url= が無い稀なケース
            continue

        # 囲っているセルなどからスニペット・媒体名を推測
        td = a.find_parent("td")
        block_text = clean_whitespace(td.get_text(" ", strip=True)) if td else ""
        # 媒体名を " source " として拾いやすい場所から推定（簡易）
        # 例: "Chronicle of Philanthropy ... Venture-Capital-Backed ..."
        source = ""
        # <div>で媒体名が別タグにあることが多いので兄弟要素も見る
        if td:
            source_div = td.find("div", style=re.compile("font-size:12px", re.I))
            if source_div:
                s_txt = clean_whitespace(source_div.get_text(" ", strip=True))
                # 媒体名らしき短い部分を抽出
                if s_txt and len(s_txt) <= 60:
                    source = s_txt

        snippet = block_text
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"

        candidates.append(
            {
                "title": text,
                "url": url,
                "source": source,
                "snippet": snippet,
            }
        )

    # URLで重複排除
    unique = {}
    for c in candidates:
        unique[c["url"]] = c
    results = list(unique.values())

    # タイトル等が極端に短い/ノイズなものを削る
    results = [r for r in results if len(r["title"]) >= 8]

    return results

def safe_trim(text: str, max_len: int) -> str:
    """コードポイント長で280以内に収める簡易トリム"""
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"

def build_ja_post(title: str, source: str, key_points: str, hashtag_hint: Optional[str] = None) -> str:
    """
    日本語ポスト: 文章のみ（リンク無し）、280文字以内
    """
    parts = []
    # タイトルを簡易に和訳済み想定の要約（LLM側で付与）に頼るため、ここは key_points を中心に
    if key_points:
        parts.append(key_points.strip())
    if source:
        parts.append(f"（出所: {source}）")
    if hashtag_hint:
        parts.append(hashtag_hint)
    text = " ".join([p for p in parts if p])
    return safe_trim(text, 280)

def build_en_post(title: str, url: str, source: str, key_points: str, hashtag_hint: Optional[str] = None) -> str:
    """
    英語ポスト: 本文+リンク（URL付き）、280文字以内
    XはURLをt.coに短縮してもカウント上はほぼ固定(約23)だが、実装簡易のため全体280でトリム
    """
    base = f"{key_points.strip()}" if key_points else title
    if source:
        base = f"{base} (via {source})"
    if hashtag_hint:
        base = f"{base} {hashtag_hint}"

    # URLを最後につける。必要なら本文トリム
    # 余裕を少しみて URL 分のスペースも考慮
    max_body = 280 - (len(url) + 1)  # 1はスペース
    body = safe_trim(base, max_body)
    return f"{body} {url}"

# -----------------------------
# OpenAI 呼び出し
# -----------------------------
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.warning("OPENAI_API_KEY が設定されていません。Streamlit Cloud の Secrets で設定してください。")
    return OpenAI()

RATER_SYSTEM_PROMPT = """You are an expert editor for policy and venture capital news. 
Given a list of candidate articles (title, source, snippet, url), pick the SINGLE best one for an X post that would interest a PhD student focused on:
- Entrepreneurship and innovation
- Venture capital and entrepreneurial finance
- Public policy and institutional design
- Applied econometrics
- Cross-border investment

Scoring criteria (0-5 each):
- Relevance to those topics
- Novelty/timeliness (based on text)
- Credibility of source (if known)
- Policy/VC insight density

Return strict JSON with:
{
  "picked_index": <int zero-based>,
  "reason": "<short>",
  "key_points_en": "<1-2 sentence crisp English summary>",
  "key_points_ja": "<1-2 sentence crisp Japanese summary>",
  "hashtags_en": "<up to 3 short hashtags like #VentureCapital #Policy>",
  "hashtags_ja": "<全角なしの短いハッシュタグを最大3つ>"
}
"""

RATER_USER_TEMPLATE = """Candidates:
{items}

Please pick one. Respond JSON only.
"""

def rate_and_pick_article(client: OpenAI, items: List[Dict[str, str]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    # items を番号付きで文字列化
    lines = []
    for i, a in enumerate(items):
        lines.append(
            f"[{i}] title: {a['title']}\nsource: {a.get('source','')}\nurl: {a['url']}\nsnippet: {a.get('snippet','')}\n"
        )
    prompt = RATER_USER_TEMPLATE.format(items="\n".join(lines))

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RATER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    # JSONだけ返す前提
    try:
        data = json.loads(content)
        # バリデーション
        idx = int(data.get("picked_index", 0))
        if idx < 0 or idx >= len(items):
            idx = 0
            data["picked_index"] = 0
        return data
    except Exception:
        # フォールバック：最初を選ぶ
        return {
            "picked_index": 0,
            "reason": "Fallback: JSON parse failed",
            "key_points_en": items[0]["title"],
            "key_points_ja": items[0]["title"],
            "hashtags_en": "#VentureCapital",
            "hashtags_ja": "#VC",
        }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="X News Draft (BeautifulSoup)", page_icon="📰", layout="wide")
st.title("📰 Xニュースドラフト（Google Alerts HTML対応）")

with st.sidebar:
    st.header("設定")
    model = st.selectbox("OpenAIモデル", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    add_hashtags = st.checkbox("ハッシュタグを付与する", value=True)
    st.markdown("---")
    st.markdown("**使い方**")
    st.markdown("1) Google Alerts のメール本文HTMLを貼り付け、または `.html` ファイルをアップロード")
    st.markdown("2) 抽出結果を確認 → 『ドラフト生成』")
    st.markdown("3) 生成された日英ドラフトをコピー（英語はリンク付き）")

tab_input, tab_preview, tab_output = st.tabs(["① 入力", "② 抽出プレビュー", "③ 出力"])

with tab_input:
    st.subheader("HTML を貼り付け")
    html_text = st.text_area(
        "メール本文（HTML）をそのまま貼り付けてください",
        height=240,
        placeholder="ここに <div>... のようなHTMLを貼り付け",
    )
    st.write("または")
    uploaded = st.file_uploader("HTMLファイルをアップロード", type=["html", "htm"])

    html_source = ""
    if uploaded is not None:
        try:
            html_source = uploaded.read().decode("utf-8", errors="ignore")
            st.success("ファイルを読み込みました。")
        except Exception as e:
            st.error(f"ファイルの読み込みでエラー: {e}")
    elif html_text.strip():
        html_source = html_text

    # 抽出ボタン
    if "articles" not in st.session_state:
        st.session_state.articles = []

    if st.button("記事候補を抽出", type="primary", disabled=not html_source.strip()):
        with st.spinner("BeautifulSoupで抽出中..."):
            arts = parse_google_alert_html(html_source)
            st.session_state.articles = arts
        if not st.session_state.articles:
            st.warning("記事候補を抽出できませんでした。HTMLを確認してください。")
        else:
            st.success(f"{len(st.session_state.articles)} 件の候補を抽出しました。上の『② 抽出プレビュー』タブへ。")

with tab_preview:
    st.subheader("抽出プレビュー")
    articles = st.session_state.get("articles", [])
    if not articles:
        st.info("まだ記事候補がありません。『① 入力』でHTMLを読み込み、『記事候補を抽出』を押してください。")
    else:
        for i, a in enumerate(articles):
            with st.container(border=True):
                st.markdown(f"**[{i}] {a['title']}**")
                cols = st.columns([2, 3])
                with cols[0]:
                    st.caption(a.get("source") or "")
                    st.code(a.get("url", ""), language=None)
                with cols[1]:
                    st.write(a.get("snippet") or "")

with tab_output:
    st.subheader("ドラフト生成")
    articles = st.session_state.get("articles", [])
    if not articles:
        st.info("まずは記事候補を抽出してください。")
    else:
        if st.button("LLMで『最も良さそうな記事』を選び、日英ドラフトを生成する", type="primary"):
            client = get_client()
            with st.spinner("記事を評価・選定しています..."):
                picked = rate_and_pick_article(client, articles, model=model)
                idx = picked.get("picked_index", 0)
                best = articles[idx]

            # ハッシュタグ
            h_en = picked.get("hashtags_en", "") if add_hashtags else ""
            h_ja = picked.get("hashtags_ja", "") if add_hashtags else ""

            # キーポイント
            kp_en = picked.get("key_points_en", best["title"])
            kp_ja = picked.get("key_points_ja", best["title"])

            # 日本語ドラフト（リンク無し）
            ja_post = build_ja_post(
                title=best["title"],
                source=best.get("source", ""),
                key_points=kp_ja,
                hashtag_hint=h_ja if h_ja else None,
            )
            # 英語ドラフト（リンク付き）
            en_post = build_en_post(
                title=best["title"],
                url=best["url"],
                source=best.get("source", ""),
                key_points=kp_en,
                hashtag_hint=h_en if h_en else None,
            )

            st.success("ドラフトを生成しました。")

            st.markdown("#### 🏆 選定結果（要約）")
            with st.container(border=True):
                st.markdown(f"**タイトル**: {best['title']}")
                st.caption(best.get("source") or "")
                st.code(best["url"], language=None)
                st.write(f"**LLM理由**: {picked.get('reason','')}")
                st.write(f"**EN Key Points**: {kp_en}")
                st.write(f"**JA Key Points**: {kp_ja}")

            st.markdown("#### 🇯🇵 日本語ドラフト（280文字以内・リンクなし）")
            st.text_area("Japanese Draft", value=ja_post, height=120)
            st.caption(f"文字数: {len(ja_post)} / 280")

            st.markdown("#### 🇺🇸 英語ドラフト（280文字以内・リンク付き）")
            st.text_area("English Draft", value=en_post, height=120)
            st.caption(f"文字数: {len(en_post)} / 280")

            # まとめて JSON ダウンロード（任意）
            payload = {
                "picked_index": idx,
                "picked_article": best,
                "reason": picked.get("reason", ""),
                "drafts": {"ja": ja_post, "en": en_post},
                "key_points": {"ja": kp_ja, "en": kp_en},
                "hashtags": {"ja": h_ja, "en": h_en},
                "generated_at": int(time.time()),
            }
            st.download_button(
                "結果をJSONで保存",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name="x_drafts.json",
                mime="application/json",
            )

# フッタ
st.markdown("---")
st.caption("© X News Draft App — parses Google Alerts HTML with BeautifulSoup and drafts bilingual X posts.")
