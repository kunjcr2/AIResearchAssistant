import PyPDF2
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from collections import Counter

class Assistant:
    def __init__(self):
        self.embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def get_text(self, pdf_file):
        try:
            pdfReader = PyPDF2.PdfReader(pdf_file, strict=False)
            pdf_text = []
            for page in pdfReader.pages:
                text = page.extract_text()
                if text:
                    text = re.sub(r'[^\x20-\x7E]', ' ', text)  # Remove non-ASCII characters
                    pdf_text.append(text.strip())
            return " ".join(pdf_text)
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""

    def extract_keywords(self, text, top_n=5):
        words = re.findall(r'\b\w+\b', text.lower())
        common_words = Counter(words).most_common(top_n)
        return [word for word, _ in common_words]

    def preprocess(self, pdf_file):
        pdf_text = self.get_text(pdf_file)
        keywords = self.extract_keywords(pdf_text, top_n=10)

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        docs = splitter.split_text(pdf_text)

        documents = [Document(page_content=text) for text in docs if any(kw in text.lower() for kw in keywords)]

        if not documents:
            documents = [Document(page_content=text) for text in docs]

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.encode(texts)

        text_embeddings_zip = list(zip(texts, embeddings))
        if not text_embeddings_zip:
            raise ValueError("No text embeddings were generated.")

        vector_db = FAISS.from_embeddings(text_embeddings_zip, self.embedding_model)
        return vector_db

    def getContext(self, question, pdf_file):
        vector_db = self.preprocess(pdf_file)
        question_embedding = self.embedding_model.encode([question])[0]
        relevant_docs = vector_db.similarity_search_by_vector(question_embedding, k=5)

        context = " ".join([doc.page_content for doc in relevant_docs])
        keywords = self.extract_keywords(context, top_n=7)

        filtered_sentences = [sent for sent in context.split(". ") if any(kw in sent.lower() for kw in keywords)]
        refined_context = " ".join(filtered_sentences)
        if not refined_context.strip():
            refined_context = context

        return refined_context

    def ask_llm(self, question, pdf_file):
        try:
            refined_context = self.getContext(question, pdf_file)

            prompt = f"Summarize the following text in 4-5 sentences: {refined_context}"
            output = self.summarization_model(prompt, max_length=200, min_length=40, do_sample=False)[0]['summary_text']

            return re.sub(r'\s+', ' ', output).strip()
        except Exception as e:
            print(f"Error in ask_llm: {e}")
            return "An error occurred while processing the question."
