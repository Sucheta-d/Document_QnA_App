# 📄 Enterprise Document Q&A Application

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about them using Large Language Models (LLMs).

The application follows a decoupled architecture:

```
                +----------------------+
                |    Streamlit UI      |
                |      (Frontend)      |
                +----------+-----------+
                           |
                     HTTP REST API
                           |
                           |
                +----------v-----------+
                |      FastAPI         |
                |      (Backend)       |
                +----------+-----------+
                           |
         +-----------------+----------------+
         |                                  |
         |                                  |
+--------v--------+              +----------v---------+
| Google Embedding|              |     Groq LLM       |
|  Gemini Model   |              | Llama-3.1-8B       |
+--------+--------+              +----------+---------+
         |                                  |
         |                                  |
         +-----------------+----------------+
                           |
                     Pinecone Vector DB
```

---

# Features

- Upload PDF documents
- Automatic document summarization
- Semantic document search
- Question Answering using Retrieval-Augmented Generation (RAG)
- FastAPI backend
- Streamlit frontend
- Pinecone cloud vector database
- Google Gemini embeddings
- Groq Llama 3.1 model for summarization and answering questions

---

# Project Structure

```
document_qna_app/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Groq Llama-3.1-8B |
| Embeddings | Google Gemini Embedding |
| Vector Database | Pinecone |
| PDF Loader | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |

---

# Application Workflow

## Step 1: Upload PDF

The user uploads a PDF from the Streamlit interface.

↓

The frontend sends the PDF to

```
POST /ingest
```

↓

The backend:

- saves the PDF temporarily
- extracts text
- generates a summary
- splits the text into chunks
- converts chunks into embeddings
- stores embeddings in Pinecone

↓

The backend returns the document summary.

---

## Step 2: Ask Questions

The user enters a question.

↓

The frontend sends

```
POST /query
```

↓

The backend

- converts the question into an embedding
- searches Pinecone
- retrieves the most relevant chunks
- sends the chunks to the LLM
- generates an answer

↓

The answer and supporting context are displayed.

---

# Backend API

## POST /ingest

Uploads and indexes a PDF.

### Request

Multipart Form Data

```
file=<pdf>
```

### Response

```json
{
  "message": "...",
  "summary": "..."
}
```

---

## POST /query

Queries the indexed document.

### Request

```json
{
    "question":"What is this document about?"
}
```

### Response

```json
{
   "answer":"...",
   "context":[
      {
         "page":2,
         "content":"..."
      }
   ]
}
```

---

# Environment Variables

Create a `.env` file inside the backend folder.

```
GROQ_API_KEY=xxxxxxxx

GOOGLE_API_KEY=xxxxxxxx

PINECONE_API_KEY=xxxxxxxx

PINECONE_INDEX_NAME=document-qna-app
```

---

# Installation

## Backend

```bash
cd backend

pip install -r requirements.txt

uvicorn main:app --reload
```

Runs on

```
http://127.0.0.1:8000
```

---

## Frontend

```bash
cd frontend

pip install -r requirements.txt

streamlit run app.py
```

Runs on

```
http://localhost:8501
```

---

# Retrieval-Augmented Generation (RAG)

This application follows the RAG architecture.

Instead of asking the LLM to answer from its own knowledge, relevant document chunks are first retrieved from Pinecone and supplied as context to the LLM.

This improves:

- Accuracy
- Hallucination reduction
- Domain-specific question answering

---

# Future Improvements

- Multiple document support
- Metadata filtering
- Chat history
- Authentication
- Delete/update indexed documents
- Source highlighting
- Streaming responses
- Docker deployment
- Kubernetes deployment

---

# License

MIT License
