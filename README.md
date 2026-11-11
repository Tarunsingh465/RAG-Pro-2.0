# 📚 RAG-Pro

A production-ready **Retrieval-Augmented Generation** system built with FastAPI, LangChain, Pinecone, and OpenRouter.

## ✨ Features

- **Multi-format document ingestion** — PDF (Mistral OCR), DOCX, PPTX, CSV, XLSX, TXT, MD, and images (PNG, JPG, TIFF, BMP, WEBP)
- **Intelligent query routing** — Supervisor classifies queries into greeting, RAG, guardrail, or out-of-scope
- **Similarity-gated RAG** — Only invokes the LLM when retrieved chunks pass a relevance threshold
- **Streamlit frontend** — Upload documents and chat in a clean web UI
- **Structured logging** — Loguru with console + rotating file logs

## 🏗️ Architecture

```
User → Streamlit UI → FastAPI Backend
                         ├── Supervisor (query routing)
                         ├── Greeting Service (casual chat)
                         ├── RAG Service
                         │     ├── Retriever (Pinecone)
                         │     └── LLM Chain (OpenRouter)
                         └── Ingestion Pipeline
                               ├── Document Loader (multi-format)
                               ├── Text Splitter (recursive)
                               └── Embedder (Google Generative AI)
```

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd RAG-Pro
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env with your actual API keys
```

### 3. Start the backend

```bash
python run.py
```

The API will be available at `http://127.0.0.1:8000`. Swagger docs at `/docs`.

### 4. Start the frontend

```bash
streamlit run frontend/app.py
```

## 📁 Project Structure

```
RAG-Pro/
├── app/
│   ├── api/            # FastAPI routes
│   ├── core/           # Config, constants, logging, prompts
│   ├── ingestion/      # Document loading, splitting, embedding
│   │   └── extractors/ # Format-specific extractors (Mistral OCR, Office, etc.)
│   ├── llm/            # LLM client and RAG chain
│   ├── retrieval/      # Retriever and search utilities
│   ├── schemas/        # Pydantic request/response models
│   ├── services/       # Business logic (chat, RAG, greeting, supervisor)
│   ├── vectorstore/    # Pinecone integration
│   └── main.py         # FastAPI app entry point
├── frontend/
│   └── app.py          # Streamlit UI
├── data/               # Uploads and OCR output
├── .env.example        # Environment variable template
├── requirements.txt    # Python dependencies
└── run.py              # Dev server launcher
```

## 🔑 Required API Keys

| Key | Provider | Purpose |
|-----|----------|---------|
| `NVIDIA_API_KEY` | [NVIDIA Build](https://build.nvidia.com/) | LLM inference (Free 1,000 credits, no card required) |
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/) | Embeddings (100% Free) |
| `PINECONE_API_KEY` | [Pinecone](https://www.pinecone.io/) | Vector database (Free Serverless tier) |
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/) | Alternative LLM inference (Optional) |
| `MISTRAL_API_KEY` | [Mistral AI](https://console.mistral.ai/) | Cloud OCR extraction (Optional; local `pypdf` is used by default) |

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `POST` | `/api/v1/upload` | Upload & index documents |
| `GET` | `/api/v1/documents` | List uploaded documents |
| `DELETE` | `/api/v1/documents/{name}` | Delete a document |
| `POST` | `/api/v1/chat` | Chat with the RAG assistant |

## 📝 License

MIT
