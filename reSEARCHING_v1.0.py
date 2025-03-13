import PyPDF2
import re
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

os.environ['OPENAI_API_KEY'] = "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"

def get_text(pdf_file):
    try:
        with open(pdf_file, 'rb') as f:
            pdfReader = PyPDF2.PdfReader(f, strict=False)
            pdf_text = []

            for page in pdfReader.pages:
                text = page.extract_text()

                for line in text.split('\n'):
                    if len(line) > 5:
                        line = re.sub(r'[^\x20-\x7E]', ' ', line)
                        line = re.sub(r'\s+', ' ', line).strip()
                        pdf_text.append(line)

            return " ".join(pdf_text)
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def preprocess(pdf_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_text(pdf_text)  # Split the entire text

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.from_texts(docs, embedding)
    retriever = vector_db.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           retriever=retriever
                                        )
    return qa_chain

def streamlitApp(pdf_file):
    pdf_text = get_text(pdf_file)
    if not pdf_text:
        return "No text extracted from the PDF."

    qa_chain = preprocess(pdf_text)
    question = input("Enter the question: ")
    res = qa_chain.run(question)
    return res

print(streamlitApp("test-1.pdf"))