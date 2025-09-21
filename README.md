# AI-Agent
# 📚 AI PDF Q&A Assistant

A Streamlit-based AI-powered application that allows users to **upload a PDF** and **ask questions**. The AI will provide answers based on the content of the uploaded document. The app also supports **chat history** and the ability to **ask previous questions again**.

---

## Features

- Upload PDF files and process their content.
- Ask questions based on the PDF content.
- AI-generated answers using a pre-trained language model (`distilgpt2`).
- Chat history to view previous questions and answers.
- "Ask Again" feature to quickly reuse previous questions.
- Stylish UI with hover effects and chat-like interface.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – Web framework for the interface.
- **LangChain** – For document loading, text splitting, and vector store integration.
- **FAISS** – Fast similarity search for vector embeddings.
- **HuggingFace Transformers** – Pre-trained models for embeddings and text generation.
- **Sentence Transformers** – Model for generating embeddings (`all-MiniLM-L6-v2`).

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/ai-pdf-qna.git
cd ai-pdf-qna
Install dependencies:

pip install streamlit langchain faiss-cpu transformers sentence-transformers

Usage

Run the app:

streamlit run app.py


Open the URL provided (usually http://localhost:8501) in your browser.

Upload a PDF.

Click Process PDF.

Ask questions and view answers.

Use chat history to ask previous questions again.
