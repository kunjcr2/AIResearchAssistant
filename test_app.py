import streamlit as st
from reSEARCHINGvONEpointTWO import Assistant

assistant = Assistant()

pdf_text = assistant.get_text("test-1.pdf")

st.write(assistant.truncate_text(pdf_text))