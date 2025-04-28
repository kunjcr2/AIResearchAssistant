# 📄 theHelper – AI Research Assistant

> Upload a PDF, get a clean summary, and ask questions based on its content — fast and intelligently.

---

## 🚀 Features

- 📚 **Extract Text from PDF**  
  Reads and cleans PDF text using `PyPDF2` and regular expressions to remove unwanted characters.

- ✂️ **Smart Text Chunking**  
  Breaks large documents into manageable overlapping chunks for better analysis using `langchain_text_splitters`.

- 🧠 **Keyword Extraction**  
  Identifies the most important keywords from the PDF text to focus on the most relevant sections.

- 🧩 **Semantic Search with FAISS**  
  Embeds text using `SentenceTransformer (BERT)` and builds a FAISS vector store for fast, semantic similarity search.

- ✨ **Summarization**  
  Summarizes entire documents or context passages using `facebook/bart-large-cnn` through Hugging Face’s `transformers` pipeline.

- ❓ **Question Answering**  
  Given a PDF and a question, the system retrieves relevant document chunks and generates a high-quality, context-aware answer.

---

## 🛠 Tech Stack

- **Python 3**
- **PyPDF2** – PDF parsing
- **Hugging Face Transformers** – Summarization model (`BART`)
- **Sentence Transformers** – Embedding with `bert-base-nli-mean-tokens`
- **FAISS** – Fast Approximate Nearest Neighbor search
- **LangChain** – Text splitting and document management
- **Regex, Collections** – Preprocessing and keyword extraction

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

(make sure you have `python3-pip`, and it’s recommended to use a virtual environment)

---

## 🧑‍💻 Usage

**Basic Workflow:**

```python
from assistant import Assistant

helper = Assistant()

# Summarize the whole PDF
summary = helper.get_summary('path_to_pdf.pdf')
print(summary)

# Ask a question based on the PDF content
answer = helper.ask_llm("What are the key points discussed?", 'path_to_pdf.pdf')
print(answer)
```

---

## 📌 Notes

- The QA pipeline currently uses the **summarization model** for simplicity. (Originally prepared for a QA model like `bert-base-cased-squad2`.)
- Works best on documents with **clear structure and readable text**.
- If you get weird results, ensure your PDF is text-based (not image scans without OCR).

---

## 🧠 Future Improvements

- Switch back to a dedicated QA model for more precise question answering.
- Improve context refinement using dynamic keyword weighting.
- Add support for OCR if the PDF is scanned images.

---
