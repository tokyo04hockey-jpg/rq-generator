import streamlit as st
import openai

st.title("🎓 Research Question Generator")
st.write("パネルディスカッションやインタビューの要約を入力すると、研究クエスチョン案を生成します。")

openai.api_key = st.secrets["OPENAI_API_KEY"]

summary = st.text_area("議論の要約を入力してください", height=200)

if st.button("クエスチョンを生成"):
    if summary:
        prompt = f"""
        以下の議論内容に基づき、研究クエスチョンを4つの視点から提案してください：
        1. 逆張り
        2. 飛ばし
        3. トレードオフの幻想
        4. アナロジー
        出力形式：各視点ごとに1問ずつ、研究テーマとして適切なクエスチョンを日本語で。
        内容: {summary}
        """

        with st.spinner("生成中..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(response.choices[0].message.content)
    else:
        st.warning("先に議論の要約を入力してください。")
