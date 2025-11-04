# streamlit_app.py  — OpenAI SDK v1 用
import streamlit as st
from openai import OpenAI

st.title("🎓 Research Question Generator")
st.write("パネル/インタビュー要約から4視点（逆張り/飛ばし/トレードオフ幻像/アナロジー）で研究クエスチョンを生成します。")

# Secrets に OPENAI_API_KEY を "KEY=VALUE" のTOML形式で設定済みであること
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

summary = st.text_area("議論の要約を入力してください", height=200)

if st.button("クエスチョンを生成"):
    if not summary:
        st.warning("先に要約を入力してください。")
    else:
        prompt = f"""
以下の議論に基づき、各視点で研究クエスチョンを1つずつ日本語で出してください。
1. 逆張り（前提を逆に見る）
2. 飛ばし（手段Bを前提とせずAを達成する方法）
3. トレードオフの幻想（AとBを同時達成できる条件）
4. アナロジー（他分野への転用）
内容: {summary}
"""
        with st.spinner("生成中..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
        st.markdown(resp.choices[0].message.content)
