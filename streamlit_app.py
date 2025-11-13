# streamlit_app.py
# ------------------------------------------------------------
# RQ Builder: Meeting notes -> RQ generation -> edit -> Notion save
#
# Requirements (examples)
#   streamlit==1.39.0
#   openai>=1.30.0  # 1.51+ だと responses API が安定。古くてもフォールバックで動作
#   notion-client>=2.2.1
#   pydantic>=2.8.0
#
# .streamlit/secrets.toml
#   OPENAI_API_KEY = "sk-..."
#   NOTION_TOKEN = "ntn_..."  # または secret_...
#   NOTION_DATABASE_ID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# ------------------------------------------------------------

import json
from typing import List, Optional

import pandas as pd
import streamlit as st
from notion_client import Client as NotionClient
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# ---------- Page setup ----------
st.set_page_config(page_title="RQ Builder (Notes → Notion)", page_icon="🧪", layout="wide")
st.title("🧪 Research Question Builder")
st.caption("議事録から研究クエスチョン案を生成し、編集してNotionに保存します。")

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
NOTION_DATABASE_ID = get_secret("NOTION_DATABASE_ID")

oa_client = OpenAI(api_key=OPENAI_API_KEY)
notion = NotionClient(auth=NOTION_TOKEN)

# ---------- Sidebar options ----------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "OpenAI model",
        ["gpt-4.1-mini", "gpt-4.1", "o4-mini"],
        index=0,
        help="精度↔コストのバランスで選択してください。"
    )
    max_items = st.slider("生成件数（目安）", 3, 8, 6, help="実際の件数はモデル出力次第で前後します。")
    st.markdown("---")
    st.markdown("**Notion DB**")
    st.code(NOTION_DATABASE_ID, language="text")

# ---------- Input area ----------
col1, col2 = st.columns(2)
with col1:
    notes = st.text_area(
        "議事録をペースト",
        height=300,
        placeholder="ここに議事録テキストを貼り付けてください（日本語/英語どちらでも可）"
    )
with col2:
    uploaded = st.file_uploader("またはテキストファイルをアップロード", type=["txt", "md"])
    if uploaded and not notes:
        try:
            notes = uploaded.read().decode("utf-8")
        except Exception:
            notes = uploaded.read().decode("utf-8", errors="ignore")

st.divider()

# ---------- Structured output schema ----------
class RQItem(BaseModel):
    title_ja: str = Field(..., description="研究リサーチクエスチョン（日本語、1行）")
    proposed_approach_ja: str = Field(..., description="方法論案（日本語、2〜4文）")
    keywords_en: List[str] = Field(default_factory=list, description="英語キーワード（3〜7語）")

class RQResponse(BaseModel):
    source_summary: Optional[str] = None
    items: List[RQItem]

# ---------- OpenAI call with fallbacks ----------
def call_openai_structured(oa_client: OpenAI, prompt: str, schema: dict, preferred_model: str):
    """
    順に試すフォールバック:
      1) responses.create + json_schema
      2) chat.completions.create + json_schema
      3) chat.completions.create + json_object
      4) chat.completions.create (plain) -> json.loads
    返り値: Python dict
    """
    # 1) Responses API + json_schema
    try:
        resp = oa_client.responses.create(
            model=preferred_model,
            input=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "rq_payload", "schema": schema, "strict": True},
            },
        )
        if hasattr(resp, "output_text") and resp.output_text:
            return json.loads(resp.output_text)
        return json.loads(resp.output[0].content[0].text)
    except TypeError:
        # SDKが response_format 未対応（今回の主因）
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
                "json_schema": {"name": "rq_payload", "schema": schema, "strict": True},
            },
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        pass

    # 3) Chat Completions + json_object（キー整合のみ担保）
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

    # 4) 最終フォールバック：プレーン→json.loads（失敗時は例外を上げる）
    resp = oa_client.chat.completions.create(
        model=preferred_model,
        messages=[
            {"role": "system", "content": "Return JSON only. No commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)

# ---------- Generate UI ----------
col_g1, col_g2 = st.columns([1, 3])
with col_g1:
    gen_btn = st.button("🔮 RQを生成", disabled=not bool(notes))
with col_g2:
    reset_btn = st.button("🧹 リセット")

if reset_btn:
    for k in list(st.session_state.keys()):
        if k.startswith("rq_") or k in ("rq_items", "rq_editor"):
            del st.session_state[k]
    st.rerun()

if gen_btn:
    with st.spinner("生成中..."):
        schema = RQResponse.model_json_schema()
        prompt = f"""
あなたは政策×VC研究のアシスタントです。以下の議事録から、研究リサーチクエスチョン候補を日本語で作成してください。
各候補について以下の項目を必ず埋めて、JSONで返します：

- title_ja：研究クエスチョン（日本語、1行）
- proposed_approach_ja：方法論案（日本語、2〜4文。使用するデータ例・分析枠組み（例：DiD/IV/RD/質的比較等）・識別戦略の方向性をできる範囲で明示）
- keywords_en：分類・検索用のキーワード（英語、3〜7語）

最低3件、最大{max_items}件程度を返してください。

[議事録]
{notes}
""".strip()
        try:
            raw_obj = call_openai_structured(oa_client, prompt, schema, model)
            data = RQResponse.model_validate(raw_obj)  # 厳密検証
            st.session_state["rq_items"] = [it.model_dump() for it in data.items]
            st.success("RQ候補を生成しました。下で編集できます。")
        except ValidationError as ve:
            st.error("JSONの構造検証に失敗しました。もう一度お試しください。")
            st.exception(ve)
        except Exception as e:
            st.error("生成中にエラーが発生しました。")
            st.exception(e)

# ---------- Edit table ----------
if "rq_items" in st.session_state and st.session_state["rq_items"]:
    st.subheader("📝 候補（編集可能）")

    df = pd.DataFrame([
        {
            "select": True,
            "Name": it["title_ja"],                          # 日本語RQ
            "Proposed Approach": it["proposed_approach_ja"], # 日本語の方法論案
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
    st.caption("保存先 Notion DB: " + NOTION_DATABASE_ID)

    # ---------- Save to Notion ----------
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
                            "Status": {"select": {"name": "New"}},
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
st.caption("© RQ Builder — Notes → JSON → Edit → Notion")
