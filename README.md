# AI-Powered Multi-Document RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that enables users to upload multiple PDF documents, generate summaries, and ask natural language questions grounded in document content.

## Features

- Upload multiple PDF documents
- OCR support for scanned PDFs
- Automatic document summarization
- Multi-document semantic search
- Context-aware question answering
- Query expansion for improved retrieval
- Broad vs. specific question handling
- Voice-based question input
- Source attribution for generated answers

## Tech Stack

### Frontend
- Streamlit

### AI & NLP
- Sentence Transformers
- Groq API (Llama 3.3)
- Whisper (Speech-to-Text)

### Retrieval
- FAISS Vector Database

### PDF Processing
- PyMuPDF
- Tesseract OCR

## Project Structure

```
project/
│
├── data/
├── saved_index/
├── app.py
├── save_index.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
git clone https://github.com/yourusername/project-name.git

cd project-name

pip install -r requirements.txt
```

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key
```

Run:

```bash
streamlit run app.py
```

## Future Enhancements

- React frontend
- FastAPI backend
- User authentication
- SQLite database
- Persistent chat history
- Workspace-based document organization

