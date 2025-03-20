import streamlit as st
from reSEARCHINGv1p3 import Assistant
from io import BytesIO

assistant = Assistant()

file = st.file_uploader("Drop your file here (PDF only)", accept_multiple_files=False, type=['pdf'])

if file is not None:

    stream = BytesIO(file.read())

    summary = assistant.get_summary(stream)

    st.text(f'Summray:\n{summary}')

    ques = st.chat_input("Enter your question here")
    if ques:
        st.write(f"Question:\n{ques}")

        ans = assistant.ask_llm(ques, stream)
        if ans:
            st.write(f'Answer:\n{ans}')

        context = assistant.getContext(ques, stream)
        if context:
            st.write(f'Context:\n{context}')