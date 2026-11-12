# 🏭 RAG-Pro 2.0 — Industrial Knowledge Intelligence & Decision Support System

> **A Final-Year B.Tech Computer Science (Data Science & AI) Capstone Project**  
> **Developer:** Tarun Singh  
> **Development Environment:** Windows 10, 8 GB RAM, Integrated Graphics (CPU-first development, Cloud LLM/Embedding inference)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-green.svg)](https://www.pinecone.io/)
[![Tests](https://img.shields.io/badge/Tests-53%20Passed%20(100%25)-brightgreen.svg)]()

---

## 1. Project Overview

**RAG-Pro 2.0** is an academic-level Retrieval-Augmented Generation (RAG) system engineered to provide reliable, evidence-grounded decision support over complex technical and industrial documentation (engineering standards, operating manuals, maintenance procedures, and inspection checklists).

The project was developed to demonstrate core AI/ML engineering competencies:
- **Document intelligence:** Multi-format parsing (PDF, DOCX, PPTX, Excel, CSV, Markdown, TXT) with table structure preservation.
- **Hybrid retrieval:** Dense semantic search (Pinecone + Google Gemini embeddings) combined with sparse lexical search (Okapi BM25 on CPU) fused via Reciprocal Rank Fusion (RRF).
- **Similarity gating & abstention:** Independent thresholds to reject out-of-domain and unanswerable queries, preventing hallucinations.
- **Empirical RAG evaluation:** Ground-truth benchmarking across 50 industrial queries with isolated Development and Held-Out Test splits measuring Precision@K, Recall@K, Hit Rate@K, MRR, nDCG@K, Correct Abstention Rate (CAR), and answer faithfulness.

---

## 2. Problem Statement & Objectives

Industrial documentation presents distinct retrieval challenges compared to general web text:
1. **Dense technical terminology & codes:** Specific standards (e.g., ASME Section VIII, API 610) require exact keyword lookup that dense embeddings alone can miss.
2. **Tabular & multi-format data:** Critical parameters (pressures, clearances, chemical limits) reside in spreadsheets and structured tables.
3. **Severe risk of hallucination:** In industrial operations, fabricated answers can lead to safety hazards; systems must reliably abstain when evidence is absent.

### Project Objectives
- Build a modular, transparent, and easy-to-explain RAG pipeline.
- Preserve low-latency execution on standard consumer hardware (8 GB RAM, CPU).
- Quantitatively measure retrieval quality and trade-offs rather than assuming vector search is always superior.

---

## 3. System Architecture & Data Flow

```
                                  ┌──────────────────────────┐
                                  │   Streamlit Web UI       │
                                  │   (Port 8501)            │
                                  └─────────────┬────────────┘
                                                │ HTTP / REST
                                                ▼
                                  ┌──────────────────────────┐
                                  │   FastAPI Backend API    │
                                  │   (Port 8000)            │
                                  └─────────────┬────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [ Document Ingestion ]                                          [ Query Processing ]
  Multi-Format Document Loader                                     Supervisor Query Router
  (PDF, DOCX, PPTX, CSV, XLSX, MD)                                (Greeting, RAG, Reject)
                 │                                                             │
        Text Splitter                                            Relevance Pre-Filter / Gating
 (Small / Medium / Large Chunks)                                 (τ_dense = 0.65, τ_bm25 = 5.0)
                 │                                                             │
        Embeddings Generation                                                  ▼
     (gemini-embedding-001)                                           [ Hybrid Retrieval ]
                 │                                                 ┌───────────┴───────────┐
                 ▼                                                 ▼                       ▼
      ┌─────────────────────┐                            Dense Vector Search       Sparse Keyword Search
      │ Pinecone Vector DB  │ ◄───────────────────────── (Pinecone Serverless)     (Local Okapi BM25 CPU)
      └─────────────────────┘                                      │                       │
                                                                   └───────────┬───────────┘
                                                                               ▼
                                                                     Reciprocal Rank Fusion
                                                                           (RRF, k=60)
                                                                               │
                                                                               ▼
                                                                     Grounded LLM Chain
                                                                   (NVIDIA NIM / OpenRouter)
                                                                               │
                                                                               ▼
                                                                    Answer + Source Citations
```

---

## 4. Technology Stack

| Layer | Technology | Selection Rationale |
|:---|:---|:---|
| **API Framework** | FastAPI + Uvicorn | High-performance asynchronous REST endpoints, automatic OpenAPI documentation. |
| **Frontend UI** | Streamlit | Lightweight, interactive Python UI for document upload, chat, and live evaluation dashboard. |
| **Vector Database** | Pinecone Serverless | Fully managed cloud vector store with cosine similarity; zero local RAM footprint. |
| **Sparse Retrieval** | Okapi BM25 (Python) | In-memory lexical keyword search executed on CPU in < 0.1 ms with JSON persistence. |
| **Embeddings** | Google Gemini (`models/gemini-embedding-001`) | High-dimensional (3072d) semantic embeddings via Google AI Studio API. |
| **LLM Inference** | NVIDIA NIM / OpenRouter | Cloud inference with NVIDIA Nemotron / Llama models to avoid GPU hardware constraints. |
| **Document Parsers** | `pypdf`, `python-docx`, `python-pptx`, `pandas`, `openpyxl`, `tabulate` | Format-specific local extraction; optional Mistral OCR for scanned images. |
| **Testing** | Python `unittest` | 53 unit and integration tests covering extraction, retrieval, fusion, evaluation, and integrity. |

---

## 5. RAG Pipeline Implementation

### A. Document Extraction & Chunking
- **Extractors:** Dispatches files to dedicated extractors. Excel/CSV files are converted into markdown tables with preserved headers; Word/PowerPoint files preserve section structure.
- **Chunking Strategies:** Implemented in `app/ingestion/splitter.py`:
  - `small`: 500 characters, 100 overlap (granular lookup)
  - `medium`: 1000 characters, 200 overlap (**default**, balanced context)
  - `large`: 1500 characters, 300 overlap (broad procedures)
- **Deterministic Chunk IDs:** Every chunk is assigned a canonical identifier (`{file_name}_{strategy}_{index}`) ensuring 100% reproducible ground-truth evaluation without UUID randomness.

### B. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
To balance semantic understanding and exact keyword precision, queries are sent simultaneously to:
1. **Dense Retrieval:** Pinecone vector search returning top candidates with cosine similarity $\ge 0.65$.
2. **Sparse Retrieval:** Okapi BM25 search returning top candidates with score $\ge 5.0$.

Candidates passing their respective thresholds are merged using Reciprocal Rank Fusion ($k=60$):
$$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + r_m(d)}$$
where $r_m(d)$ is the 1-based rank of document $d$ in retrieval method $m$.

### C. Similarity Gating & Abstention
If no candidate passes the similarity thresholds (or if the supervisor detects an out-of-domain query), the system immediately abstains with an informative message without calling the LLM, eliminating hallucination and saving API credits.

---

## 6. Retrieval & Evaluation Methodology

The evaluation framework (`app/evaluation/`) benchmarks retrieval and generation quality against a 50-question industrial evaluation dataset across 10 authentic technical specifications:

- **Split Isolation:** 35 Development queries (calibrating thresholds and tuning parameters) and 15 Held-Out Test queries (unseen validation).
- **Answerable vs. Unanswerable:** 37 answerable questions test retrieval quality; 13 out-of-corpus questions test abstention safety.

### Benchmark Results Summary (Held-Out Test Split, $N=15$)

| Pipeline | Hit Rate@5 | Recall@5 | Precision@5 | MRR | Correct Abstention (CAR) | Mean Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense Semantic (Pinecone)** | **0.9091** | **0.9091** | 0.1818 | **0.9091** | **100.0% (4/4)** | 861.9 ms |
| **Sparse Lexical (Okapi BM25)** | 0.8182 | 0.8182 | 0.1636 | 0.8182 | **100.0% (4/4)** | **0.08 ms** |
| **Hybrid RRF (Dense + BM25)** | **0.9091** | **0.9091** | 0.1818 | **0.9091** | **100.0% (4/4)** | 862.3 ms |

> *Note on Precision@5:* Each query is annotated with exactly 1 primary ground-truth target chunk; returning $K=5$ chunks yields an expected theoretical maximum precision of $1/5 = 0.2000$.

### Qualitative Answer Generation Pilot (Human Audited)
A 6-question qualitative pilot evaluated generation faithfulness, answer relevance, and citation validity:
- **Faithfulness:** 1.0 (zero factual errors or hallucinations identified)
- **Answer Relevance:** 1.0 (direct responses addressing technical parameters)
- **Citation Validity:** 100% (every citation mapped directly to a verified retrieved document and section)
- **Abstention on Out-of-Domain:** Correctly identified non-corpus questions (e.g. quantum physics) and safely refused to answer.

---

## 7. Installation & Execution Guide

### Prerequisites
- Windows 10 / 11, macOS, or Linux
- Python 3.11 or higher
- 8 GB RAM (runs comfortably on CPU)

### Step 1: Clone Repository & Create Virtual Environment
```powershell
git clone https://github.com/Tarunsingh465/RAG-Pro-2.0.git
cd RAG-Pro-2.0

python -m venv .venv
.\.venv\Scripts\activate       # On Windows
# source .venv/bin/activate    # On Linux/macOS
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy the template and provide your API keys:
```powershell
copy .env.example .env
```
Edit `.env` with:
- `GOOGLE_API_KEY`: For Google Gemini embeddings (free tier from [Google AI Studio](https://aistudio.google.com/)).
- `PINECONE_API_KEY`: For Pinecone vector database (free serverless index from [Pinecone](https://www.pinecone.io/)).
- `NVIDIA_API_KEY` or `OPENROUTER_API_KEY`: For LLM text generation.

### Step 4: Run the Backend Server
```powershell
python run.py
```
- API will start at `http://127.0.0.1:8000`.
- Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### Step 5: Run the Streamlit Frontend
In a separate terminal window:
```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```
- Web UI will launch at `http://localhost:8501`.

### Step 6: Run the Unit Tests
Execute the 53-test verification suite:
```powershell
python -m unittest discover tests
```

---

## 8. Example Queries & System Outputs

| Query Type | User Query | System Behavior & Output |
|:---|:---|:---|
| **Technical Calculation** | *"What is the minimum shell thickness formula under ASME Section VIII?"* | Retrieves ASME standard chunk, computes/explains formula: $t = \frac{P \cdot R}{S \cdot E - 0.6 \cdot P} + C$, citing `pressure_vessel_asme_sec_viii.md`. |
| **Troubleshooting** | *"Why is a 4-20mA process loop saturating above 20.8 mA with 50Hz noise?"* | Identifies ground loop fault from shield grounding at multiple locations, citing `industrial_plc_scada_troubleshooting.md`. |
| **Tabular Data** | *"Compare design pressure and allowable drop between shell and tube sides."* | Parses Excel sheet `heat_exchanger_thermal_rating_spec.xlsx`, contrasts shell (16 bar, 0.65 bar drop) vs tube (25 bar, 0.85 bar drop). |
| **Out-of-Domain / Abstention** | *"What is the quantum decoherence time for superconducting transmon qubits?"* | Cosine scores fall below $\tau = 0.65$; system safely abstains stating documentation does not cover quantum physics (0 citations). |

---

## 9. Limitations & Future Improvements

### Current Limitations
1. **Cloud Embedding Latency:** Dense vector retrieval requires an external API call to Google Generative AI (~689 ms network latency), whereas local BM25 executes in 0.08 ms.
2. **In-Memory BM25 Scale:** The BM25 index is maintained locally in JSON format; suitable for small-to-medium corpora (< 10,000 chunks) but requires persistent disk-backed index engines (like Tantivy or Lucene) for enterprise scale.
3. **Session Persistence:** Chat history is managed in Streamlit session memory rather than an external persistent database (e.g., PostgreSQL or Redis).

### Future Improvements
- **Cross-Encoder Re-Ranking:** Integrate a lightweight cross-encoder (such as `bge-reranker-base`) on top of RRF candidates.
- **Persistent Chat History:** Store conversation threads in SQLite or PostgreSQL with user session tokens.
- **Local Embedding Fallback:** Support quantized ONNX embedding models for completely offline air-gapped environments.

---

## 📄 License & Attribution

Developed by **Tarun Singh** (B.Tech Computer Science - Data Science & AI).  
Academic Capstone & Portfolio Project under MIT License.
