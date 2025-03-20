import streamlit as st

st.set_page_config(page_title="AIResearchAssistant")

st.title("Welcome to theHelper")

st.write("""
### How to Use the AI Research Assistant

This application is designed to assist you in analyzing and summarizing PDF documents, as well as answering questions based on the content of the uploaded PDF. Follow the steps below to use the application effectively:

1. **Upload a PDF File**: Use the file uploader below to upload a PDF document. The application currently supports only PDF files.
2. **View Summary**: Once the file is uploaded, the application will automatically generate a summary of the document and display it below.
3. **Ask Questions**: Use the chat input box at the bottom of the page to ask questions related to the content of the uploaded PDF. The application will provide an answer based on the context of the document.
4. **View Context**: For each question, the application will also display the relevant context from the document that was used to generate the answer.

### Features
- **Document Summarization**: Automatically generates a concise summary of the uploaded PDF.
- **Question Answering**: Answers questions based on the content of the PDF.
- **Context Extraction**: Provides the relevant context from the document for each answer.

### Tips
- Ensure that the uploaded PDF contains readable text (not scanned images) for accurate summarization and question answering.
- Ask specific questions to get the most relevant answers.

For any questions or support, please contact us at kunjcr2@gmail.com.
""")