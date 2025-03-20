import streamlit as st
from reSEARCHINGv1p3 import Assistant
from io import BytesIO

assistant = Assistant()
file = st.file_uploader("Drop your file here (PDF only)", accept_multiple_files=False, type=['pdf'])

if file:
    stream = BytesIO(file.read())
    st.write(assistant.preprocess(stream))