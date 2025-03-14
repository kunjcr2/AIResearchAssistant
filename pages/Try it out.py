import streamlit as st
from reSEARCHINGvONEpointTWO import Assistant

assistant = Assistant()

data = st.file_uploader("Drop your file here (PDF only)")

if data is not None:
    st.divider()

    summary = "Summary"
    st.text(f'Summray: {summary}')

    st.divider()

    ques = st.chat_input("Enter your question here")
    if ques:
        st.write(f"Question: {ques}")

        ans = assistant.ask_llm(ques, "test-1.pdf")
        # ans = "Answer"
        if ans:
            st.write(f'Answer: {ans}')