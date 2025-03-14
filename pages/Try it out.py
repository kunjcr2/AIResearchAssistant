import streamlit as st
from reSEARCHINGvONEpointTWO import Assistant

assistant = Assistant()

data = st.file_uploader("Drop your file here (PDF only)")

if data is not None:

    summary = "Summary"
    st.text(f'Summray: {summary}')

    ques = st.chat_input("Enter your question here")
    if ques:
        st.write(f"Question: {ques}")

        ans = assistant.ask_llm(ques, "test-1.pdf")
        context = assistant.getContext(ques, "test-1.pdf")
        # ans = "Answer"
        if ans:
            st.write(f'Context: {context}')
            st.write(f'Answer: {ans}')