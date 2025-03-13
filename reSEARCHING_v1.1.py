import PyPDF2
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

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

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def truncate_text(text, max_tokens=500):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def preprocess(pdf_file):
    pdf_text = get_text(pdf_file)
    pdf_text = truncate_text(pdf_text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = splitter.split_text(pdf_text)

    documents = [Document(page_content=text) for text in docs]
    vector_db = FAISS.from_documents(documents, embedding_model)

    retriever = vector_db.as_retriever()
    return retriever


def ask_llm(question, pdf_file):
    ret = preprocess(pdf_file)
    llm = pipeline("summarization", model="google/flan-t5-base")

    doc = ret.invoke(question)

    context = "\n".join([d.page_content for d in doc])
    prompt = f'Just repeat: {context}'
    output = llm(prompt, max_length=101)[0]['summary_text']

    return output

print(ask_llm("", "test-1.pdf"))