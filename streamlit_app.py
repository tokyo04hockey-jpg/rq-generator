# streamlit_app.py
# ------------------------------------------------------------
# Tab1: Interview Notes  (raw memo -> AI生成: Name/Date/Summary/Transcript -> edit -> Notion保存)
# Tab2: RQ Builder       (Transcript -> RQ生成 -> edit -> Notion保存)
#
# Requirements (examples)
#   streamlit==1.39.0
#   openai>=1.30.0     # 1.51+ 推奨（responses API 安定）。古くてもフォールバックで動作
#   notion-client>=2.2.1
#   pydantic>=2.8.0
#
# .streamlit/secrets.toml
#   OPENAI_API_KEY = "sk-..."
#   NOTION_TOKEN = "ntn_..."  # or secret_...
#   NOTION_DATABASE_ID = "Research Questions DB ID（32桁）"
#   NOTION_INTERVIEW_DB_ID = "Interview Notes DB ID（32桁）"
# ------------------------------------------------------------

import json
from typing import List, Optional
from datetime import date

import pandas as pd
import streamlit as st
from notion_client import Client as NotionClient
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, AliasChoices
from pydantic import ConfigDict

# ---------- Page setup ----------
st.set_page_config(page_title="Interview → RQ Builder", page_icon="🗂️", layout="wide")
st.title("📚 Interview Notes → 🧪 Research Question Builder")
st.caption("生メモから議事録を自動生成・保存し、そのTranscriptでRQを生成してNotionへ登録します。")

# ---------- Secrets / clients ----------
def get_secret(key: str, default: Optional[str] = None) -> str:
    try:
        return st.secrets[key]
    except Exception:
        if default is not None:
            return default
        raise KeyError(f"Missing secret: {key}")

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
NOTION_TOKEN = get_secret("NOTION_TOKEN")
NOTION_DATABASE_ID = get_secret("NOTION_DATABASE_ID")              # RQ用DB
NOTION_INTERVIEW_DB_ID = get_secret("NOTION_INTERVIEW_DB_ID")      # Interview Notes用DB

oa_client = OpenAI(api_key=OPENAI_API_KEY)
notion = NotionClient(auth=NOTION_TOKEN)

# ---------- Common helpers ----------
def normalize_keywords_en(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [t.strip() for t in str(v).split(",") if t.strip()]

def call_openai_structured(oa_client: OpenAI, prompt: str, schema: dict, preferred_model: str):
    """
    Fallback order:
      1) responses.create + json_schema
      2) chat.completions.create + json_schema
      3) chat.completions.create + json_object
      4) chat.completions.create (plain) -> json.loads
    Returns: dict
    """
    # 1) Responses API + json_schema
    try:
        resp = oa_client.responses.create(
            model=preferred_model,
            input=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "payload", "schema": schema, "strict": True},
            },
        )
        if hasattr(resp, "output_text") and resp.output_text:
            return json.loads(resp.output_text)
        return json.loads(resp.output[0].content[0].text)
    except TypeError:
        pass
    except Exception:
        pass

    # 2) Chat Completions + json_schema
    try:
        resp = oa_client.chat.completions.create(
            model=preferred_model,
            messages=[
                {"role": "system", "content": "You are a strict JSON generator. Return only JSON that matches the schema."},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "payload", "schema": schema, "strict": True},
            },
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        pass

    # 3) Chat Completions + json_object
    try:
        resp = oa_client.chat.completions.create(
            model=preferred_model,
            messages=[
                {"role": "system", "content": "Return only valid JSON (no extra text)."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        pass

    # 4) Plain JSON
    resp = oa_client.chat.completions.create(
        model=preferred_model,
        messages=[
            {"role": "system", "content": "Return JSON only. No commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)

# ---------- Models ----------
# (A) Interview Notes generation
class INote(BaseModel):
    name_ja: str = Field(..., description="会議の題名（日本語）", validation_alias=AliasChoices("name_ja", "name", "title"))
    date_iso: str = Field(..., description="会議日（YYYY-MM-DD）", validation_alias=AliasChoices("date_iso", "date"))
    summary_ja: str = Field(..., description="日本語概要（200文字以内）", validation_alias=AliasChoices("summary_ja", "summary"))
    transcript_bullets_ja: List[str] = Field(
        default_factory=list,
        description="日本語の箇条書き",
        validation_alias=AliasChoices("transcript_bullets_ja", "transcript_bullets", "bullets"),
    )
    model_config = ConfigDict(extra="ignore")

class INoteResp(BaseModel):
    item: INote
    model_config = ConfigDict(extra="ignore")

# (B) RQ generation
class RQItem(BaseModel):
    title_ja: str = Field(..., description="研究RQ（日本語、1行）",
                          validation_alias=AliasChoices("title_ja", "title", "name", "rq", "question_ja"))
    proposed_approach_ja: str = Field(..., description="方法論案（日本語、2〜4文）",
                          validation_alias=AliasChoices("proposed_approach_ja", "proposed_approach", "approach", "method", "method_ja"))
    keywords_en: List[str] = Field(default_factory=list, description="英語キーワード（3〜7語）",
                          validation_alias=AliasChoices("keywords_en", "keywords", "tags"))
    model_config = ConfigDict(extra="ignore")

class RQResponse(BaseModel):
    items: List[RQItem] = Field(default_factory=list,
                                validation_alias=AliasChoices("items", "research_questions", "rqs"))
    model_config = ConfigDict(extra="ignore")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "OpenAI model",
        ["gpt-4.1-mini", "gpt-4.1", "o4-mini"],
        index=0,
        help="精度↔コストのバランスで選択してください。"
    )
    max_items = st.slider("RQ生成件数（目安）", 3, 8, 6)
    show_debug = st.checkbox("デバッグ表示（受信JSON）", value=False)
    st.markdown("---")
    st.markdown("**Notion DB**")
    st.caption("Interview Notes DB")
    st.code(NOTION_INTERVIEW_DB_ID, language="text")
    st.caption("Research Questions DB")
    st.code(NOTION_DATABASE_ID, language="text")

# ============================================================
# Tabs
tab1, tab2 = st.tabs(["🗂️ Interview Notes", "🧪 Research Question Builder"])

# ============================================================
# Tab1: Interview Notes
with tab1:
    st.header("🗂️ Interview Notes")

    st.subheader("📝 生メモ入力")
    raw_memo = st.text_area(
        "生メモ（日本語・英語どちらでも可）",
        height=220,
        placeholder="話者・論点・数字・仮説などをそのまま貼り付けてください。"
    )
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        upload = st.file_uploader("テキストファイルを読み込む", type=["txt", "md"])
        if upload and not raw_memo:
            try:
                raw_memo = upload.read().decode("utf-8")
            except Exception:
                raw_memo = upload.read().decode("utf-8", errors="ignore")
    with col_u2:
        pass

    gen_btn = st.button("🧠 会議タイトル/日付/概要/Transcript を自動生成", disabled=not bool(raw_memo))

    # 生成結果の表示・編集
    if gen_btn:
        with st.spinner("生成中..."):
            schema = INoteResp.model_json_schema()
            today_iso = date.today().isoformat()
            prompt = f"""
あなたは会議メモの整形アシスタントです。以下の「生メモ」から、
(1) 会議題名（日本語）、(2) 日付（YYYY-MM-DD、文脈から推定。なければ {today_iso} を使用）、
(3) 日本語の要約（200文字以内）、(4) 日本語の箇条書きTranscript（詳しめ、5〜12項目）を生成し、
次のJSON構造のみを返してください（マークダウンやコメント不要）。

構造:
{{
  "item": {{
    "name_ja": "...",
    "date_iso": "YYYY-MM-DD",
    "summary_ja": "... (<=200字)",
    "transcript_bullets_ja": ["...", "..."]
  }}
}}

[生メモ]
{raw_memo}
""".strip()
            try:
                raw_obj = call_openai_structured(oa_client, prompt, schema, model)
                if show_debug:
                    st.subheader("🔎 受信JSON（デバッグ）")
                    st.json(raw_obj)
                note = INoteResp.model_validate(raw_obj).item
                # 保存用にセッションへ
                st.session_state["inote_name"] = note.name_ja
                st.session_state["inote_date"] = note.date_iso
                st.session_state["inote_summary"] = note.summary_ja
                st.session_state["inote_transcript"] = "・" + "\n・".join(note.transcript_bullets_ja)
                st.success("生成しました。下で編集できます。")
            except ValidationError as ve:
                st.error("JSONの構造検証に失敗しました。")
                if show_debug:
                    st.json(raw_obj)
                st.exception(ve)
            except Exception as e:
                st.error("生成に失敗しました。")
                if show_debug:
                    try:
                        st.json(raw_obj)
                    except Exception:
                        pass
                st.exception(e)

    # 編集UI
    st.subheader("✏️ 編集してNotionに保存")
    name_ja = st.text_input("Name（会議の題名・日本語）", value=st.session_state.get("inote_name", ""))
    date_iso = st.text_input("Date（YYYY-MM-DD）", value=st.session_state.get("inote_date", date.today().isoformat()))
    summary_ja = st.text_area("Summary（200字目安・日本語）", value=st.session_state.get("inote_summary", ""), height=100)
    tags_en = st.text_input("Tags（英語・カンマ区切り。任意）", value="")
    transcript_ja = st.text_area(
        "Transcript（日本語：詳しめの箇条書き）",
        value=st.session_state.get("inote_transcript", ""),
        height=240
    )

    if st.button("📤 Notion（Interview Notes DB）に保存"):
        if not name_ja or not transcript_ja:
            st.warning("Name と Transcript は必須です。")
        else:
            try:
                notion.pages.create(
                    parent={"database_id": NOTION_INTERVIEW_DB_ID},
                    properties={
                        "Name": {"title": [{"text": {"content": name_ja}}]},
                        "Date": {"date": {"start": date_iso}},
                        "Summary": {"rich_text": [{"text": {"content": summary_ja[:200]}}]},
                        "Tags": {"multi_select": [{"name": t.strip()} for t in tags_en.split(",") if t.strip()]},
                        "Transcript": {"rich_text": [{"text": {"content": transcript_ja}}]},
                    },
                )
                st.success("Interview Notes に保存しました！")
            except Exception as e:
                st.error(f"保存エラー: {e}")

    st.divider()
    st.subheader("🗂️ 保存済みノート（Notionから取得 → RQタブに転送）")

    try:
        db = notion.databases.query(database_id=NOTION_INTERVIEW_DB_ID)
        for p in db.get("results", []):
            # 安全に取り出し
            props = p.get("properties", {})
            title = ""
            try:
                title = props["Name"]["title"][0]["plain_text"]
            except Exception:
                title = "(No Title)"
            summary_txt = ""
            try:
                summary_txt = "".join([t["plain_text"] for t in props["Summary"]["rich_text"]])
            except Exception:
                pass
            transcript_txt = ""
            try:
                transcript_txt = "".join([t["plain_text"] for t in props["Transcript"]["rich_text"]])
            except Exception:
                pass

            with st.expander(f"📝 {title}"):
                st.write(summary_txt or "_（No Summary）_")
                colb1, colb2 = st.columns(2)
                with colb1:
                    if st.button(f"このTranscriptをRQタブに反映", key=f"use_{p['id']}"):
                        st.session_state["selected_transcript"] = transcript_txt
                        st.success("TranscriptをRQタブに反映しました。👉 次のタブへ")
                with colb2:
                    st.caption(f"ID: {p['id']}")

    except Exception as e:
        st.error(f"Notionからの取得に失敗しました: {e}")

# ============================================================
# Tab2: RQ Builder
with tab2:
    st.header("🧪 Research Question Builder")

    # --- Transcriptの受け渡し（Tab1から） ---
    default_notes = st.session_state.get("selected_transcript", "")
    col1, col2 = st.columns(2)
    with col1:
        notes = st.text_area(
            "議事録（またはInterview NotesのTranscript）",
            height=300,
            placeholder="タブ1の『このTranscriptをRQタブに反映』で自動入力されます。手動で上書きも可。",
            value=default_notes,
        )
    with col2:
        uploaded2 = st.file_uploader("またはテキストファイルをアップロード", type=["txt", "md"], key="rq_upl")
        if uploaded2 and not notes:
            try:
                notes = uploaded2.read().decode("utf-8")
            except Exception:
                notes = uploaded2.read().decode("utf-8", errors="ignore")

    st.divider()

    # ====== RQ生成のプロンプト（興味分野＋ひねり含む） ======
    col_g1, col_g2 = st.columns([1, 3])
    with col_g1:
        gen_btn = st.button("🔮 RQを生成", disabled=not bool(notes))
    with col_g2:
        reset_btn = st.button("🧹 リセット（RQ）")

    if reset_btn:
        for k in list(st.session_state.keys()):
            if k.startswith("rq_") or k in ("rq_items", "rq_editor"):
                del st.session_state[k]
        st.rerun()

    if gen_btn:
        with st.spinner("生成中..."):
            schema = RQResponse.model_json_schema()
            prompt = f"""
あなたは以下の領域に精通した政策研究アシスタントです：
- Entrepreneurship and innovation
- Venture capital and entrepreneurial finance
- Public policy and institutional design
- Applied econometrics
- Cross-border investment

以下の議事録内容をもとに、これらの領域に関連する**質の高い研究リサーチクエスチョン**候補を日本語で生成してください。

各候補は、次の要件を満たす必要があります：

1. 研究テーマとの関連性  
   上記の興味分野のいずれかに明確に関係すること。

2. 議論のひねり（必須：1つ以上を活用）  
   - ① トレードオフ幻想の除去：二者択一を両立させうる条件や設計を探る。  
   - ② 逆張り：否定的に扱われがちなAが特定条件下では有効となる可能性を探る。  
   - ③ スコープ変更：Aを直接ではなくBを介して間接的に解決する発想。  
   - ④ アナロジー：**他分野（例：歴史、文学、物理学など）の概念・制度・理論を、Entrepreneurship/VC/Policyに応用**する視点。

3. 出力形式（必ずこの構造でJSON出力）
{{
  "items": [
    {{
      "title_ja": "（日本語のリサーチクエスチョン）",
      "proposed_approach_ja": "（方法論案：2〜4文。使用データ・識別戦略・分析枠組みを簡潔に）",
      "keywords_en": ["entrepreneurship", "venture capital", "..."]
    }}
  ]
}}

4. 件数：最低3件、最大{max_items}件。  
5. 有効なJSONのみを返す（マークダウン不要）。

[議事録]
{notes}
""".strip()
            try:
                raw_obj = call_openai_structured(oa_client, prompt, schema, model)
                if show_debug:
                    st.subheader("🔎 受信JSON（デバッグ）")
                    st.json(raw_obj)

                data = RQResponse.model_validate(raw_obj)
                items_norm = []
                for it in data.items:
                    d = it.model_dump()
                    d["keywords_en"] = normalize_keywords_en(d.get("keywords_en"))
                    items_norm.append(d)
                st.session_state["rq_items"] = items_norm
                st.success("RQ候補を生成しました。下で編集できます。")
            except ValidationError as ve:
                st.error("JSONの構造検証に失敗しました。")
                if show_debug:
                    st.json(raw_obj)
                st.exception(ve)
            except Exception as e:
                st.error("生成に失敗しました。")
                if show_debug:
                    try:
                        st.json(raw_obj)
                    except Exception:
                        pass
                st.exception(e)

    # ---------- RQ編集＆Notion保存 ----------
    if "rq_items" in st.session_state and st.session_state["rq_items"]:
        st.subheader("📝 RQ候補（編集可能）")

        df = pd.DataFrame([
            {
                "select": True,
                "Name": it["title_ja"],                          # 日本語RQ
                "Proposed Approach": it["proposed_approach_ja"], # 方法論案（日本語）
                "Tags": ", ".join(it.get("keywords_en", [])),    # 英語キーワード（カンマ区切り）
            }
            for it in st.session_state["rq_items"]
        ])

        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "select": st.column_config.CheckboxColumn("選択", default=True),
                "Name": st.column_config.TextColumn("Name（RQ・日本語）"),
                "Proposed Approach": st.column_config.TextColumn("Proposed Approach（日本語）", width="medium"),
                "Tags": st.column_config.TextColumn("Tags（英語・カンマ区切り）"),
            },
            key="rq_editor",
        )

        st.divider()
        st.caption("保存先 Notion Research Questions DB: " + NOTION_DATABASE_ID)

        def to_multi_select_en(s: str):
            tags = [t.strip() for t in (s or "").split(",") if t.strip()]
            return [{"name": t} for t in tags]

        if st.button("📤 選択したRQをNotionに保存"):
            selected = edited[edited["select"] == True]
            if selected.empty:
                st.warning("保存対象がありません。")
            else:
                errors = []
                success_count = 0
                for _, row in selected.iterrows():
                    try:
                        notion.pages.create(
                            parent={"database_id": NOTION_DATABASE_ID},
                            properties={
                                # ---- ご指定のNotionスキーマ ----
                                "Name": {"title": [{"text": {"content": (row["Name"] or "")[:200]}}]},
                                "Gap Identified": {"rich_text": [{"text": {"content": "TBD"}}]},
                                "Priority": {"select": {"name": "Medium"}},
                                "Proposed Approach": {"rich_text": [{"text": {"content": row["Proposed Approach"] or ""}}]},
                                "Rationale / Background": {"rich_text": [{"text": {"content": "TBD"}}]},
                                "Status": {"status": {"name": "New"}},  # ← Status型に合わせる
                                "Tags": {"multi_select": to_multi_select_en(row["Tags"])},
                            },
                        )
                        success_count += 1
                    except Exception as e:
                        errors.append(str(e))

                if errors:
                    st.error("一部保存に失敗しました：\n" + "\n".join(errors))
                if success_count:
                    st.success(f"{success_count}件をNotionに保存しました。")

# ---------- Footer ----------
st.markdown("---")
st.caption("© Interview → RQ Builder — Notes → JSON → Edit → Notion")
