# 🐞 RAG-Pro 2.0 — Defect Tracking & Bug Report

This document tracks all identified, investigated, fixed, and active defects across the RAG-Pro 2.0 system. Every defect has a documented root cause, reproduction steps, verification test, and resolution status.

---

## 📊 Defect Summary Table

| Bug ID | Title | Module | Severity | Resolution Status |
| :--- | :--- | :--- | :--- | :--- |
| **BUG-001** | LLM API Timeout & 60s+ Latency on Conversational Queries | `app.llm.client`, `app.services.greeting_service` | **Critical** | **Previously fixed & regression-tested** |
| **BUG-002** | Phantom Vector Leakage & Orphaned Document Citations | `app.vectorstore.pinecone`, `app.retrieval.retriever` | **Critical** | **Previously fixed & regression-tested** |
| **BUG-003** | Startup Network Blocking via Embed Query Probe | `app.vectorstore.pinecone` | **High** | **Fixed and verified** |
| **BUG-004** | Loss of Table & Multi-Sheet Structure in Office/Tabular Parsers | `app.ingestion.extractors` | **Medium** | **Fixed and verified** |
| **BUG-005** | Dead Code & Duplicate Logic in Frontend Directory | `frontend/api.py`, `chat.py`, `sidebar.py` | **Low** | **Deferred to Phase 5** |
| **BUG-006** | Absence of Automated Test Suite and Regression Harness | Root / Test Architecture | **High** | **Fixed and verified (20 tests passing)** |
| **BUG-007** | Missing BM25, Reciprocal Rank Fusion, and Evaluation Metrics | `app.retrieval`, `app.evaluation` | **Critical** | **Partially fixed (BM25 + RRF fixed; eval in Phase 3)** |
| **BUG-008** | Lack of Dedicated Industrial Decision Workspaces (RCA/FMEA/CAPA) | `frontend/app.py`, `app.services` | **High** | **Scheduled for Phase 4** |

---

## 🔍 Detailed Bug Logs & Verification

### BUG-001: LLM API Timeout & 60s+ Latency on Conversational Queries
- **Bug ID**: `BUG-001`
- **Description**: Conversational and fallback queries experienced severe timeouts (`Read timed out (read timeout=60)`) on NVIDIA NIM endpoints.
- **Affected Module**: `app/llm/client.py`, `app/services/greeting_service.py`
- **Severity**: Critical
- **Root Cause**: NVIDIA NIM default configuration enabled recursive reasoning (`enable_thinking: True`), triggering server-side generation loops.
- **Fix Implemented**: Configured `extra_body={'chat_template_kwargs': {'enable_thinking': False}}` in `LLMClient` and `GreetingService`.
- **Test Used for Verification**: Verified in integration tests and live chat fallback response in 1.74s.
- **Final Status**: **Previously fixed & regression-tested**

---

### BUG-002: Phantom Vector Leakage & Orphaned Document Citations
- **Bug ID**: `BUG-002`
- **Description**: Asking general questions cited historical test documents from Pinecone even when no files were uploaded in `data/uploads/`.
- **Affected Module**: `app/services/chat_service.py`, `app/services/rag_service.py`, `app/retrieval/retriever.py`
- **Severity**: Critical
- **Root Cause**: Pinecone index retained orphaned vectors across sessions without active document verification.
- **Fix Implemented**: Purged index to 0 vectors, added active document filtering (`filter={"file_name": {"$in": active_docs}}`), added empty knowledge base short-circuit, added `DELETE /api/v1/documents`.
- **Test Used for Verification**: `tests.test_phase1_baseline.TestPhase1Baseline.test_data_leakage_isolation_when_empty` and `tests.test_phase2_retrieval.TestBM25SparseRetrieval.test_bm25_active_document_filtering`.
- **Final Status**: **Previously fixed & regression-tested**

---

### BUG-003: Startup Network Blocking via Embed Query Probe
- **Bug ID**: `BUG-003`
- **Description**: Application startup/import triggered synchronous external network calls to Google Embeddings (`embed_query("dimension test")`), adding latency and failing on network disconnects.
- **Affected Module**: `app/vectorstore/pinecone.py`
- **Severity**: High
- **Root Cause**: `PineconeManager._embedding_dimension()` made an ad-hoc live embedding call instead of reading configured/cached metadata.
- **Fix Implemented**:
  1. Implemented `KNOWN_MODEL_DIMENSIONS` mapping for Gemini, OpenAI, and HuggingFace models.
  2. Implemented `data/.pinecone_dimension_cache.json` for persistent verified dimension storage.
  3. Validated that model dimension equals index dimension (3072 == 3072) and raises clear `ValueError` on mismatch.
  4. Eliminated external embedding probe calls on startup. Startup time reduced from 8.61s to 7.85s.
- **Test Used for Verification**: `tests.test_phase2_retrieval.TestEmbeddingDimensionAndStartup.test_startup_uses_cache_without_calling_embed_query` and `test_dimension_mismatch_raises_error`.
- **Final Status**: **Fixed and verified**

---

### BUG-004: Loss of Table & Multi-Sheet Structure in Office/Tabular Parsers
- **Bug ID**: `BUG-004`
- **Description**: DOCX extractor omitted all tables (`doc.tables`); PPTX extractor lumped all slides into page 1; XLSX extractor read only the first worksheet and converted entire workbooks into single unstructured text blocks.
- **Affected Module**: `app/ingestion/extractors/office_extractor.py`, `tabular_extractor.py`, `text_extractor.py`
- **Severity**: Medium
- **Root Cause**: Extractor implementations iterated only basic paragraphs or dumped sheet 0 without section or cell provenance.
- **Fix Implemented**:
  1. `OfficeExtractor` (DOCX): Traverses `doc.element.body` in document order, extracts headings, sections, and converts tables into clean Markdown tables.
  2. `OfficeExtractor` (PPTX): Slides parsed individually; assigns `page = slide_number` and extracts slide tables.
  3. `TabularExtractor` (XLSX): Reads all worksheets via `pd.ExcelFile`, annotates `sheet_name`, and batches rows with repeated headers to avoid token exhaustion.
  4. `TabularExtractor` (CSV): Batches rows with repeated headers and row-level metadata.
  5. `TextExtractor` (TXT/MD): Extracts markdown headings into sections and paginates long plain text.
- **Test Used for Verification**:
  - `tests.test_phase2_retrieval.TestStructuredDocumentExtractors.test_docx_table_and_section_extraction`
  - `tests.test_phase2_retrieval.TestStructuredDocumentExtractors.test_pptx_slide_attribution`
  - `tests.test_phase2_retrieval.TestStructuredDocumentExtractors.test_xlsx_multi_sheet_extraction`
- **Final Status**: **Fixed and verified**

---

### BUG-005: Dead Code & Duplicate Logic in Frontend Directory
- **Bug ID**: `BUG-005`
- **Description**: `frontend/app.py` directly executes HTTP calls and UI rendering, leaving `frontend/api.py`, `frontend/chat.py`, and `frontend/sidebar.py` unused.
- **Affected Module**: `frontend/`
- **Severity**: Low
- **Root Cause**: Legacy modularization attempt left incomplete during early prototyping.
- **Fix Implemented**: Scheduled for clean consolidation in Phase 5 without breaking UI stability.
- **Test Used for Verification**: Code inspection and frontend verification.
- **Final Status**: **Deferred to Phase 5**

---

### BUG-006: Absence of Automated Test Suite and Regression Harness
- **Bug ID**: `BUG-006`
- **Description**: Repository lacked automated regression tests.
- **Affected Module**: Root test architecture
- **Severity**: High
- **Root Cause**: Relying on ad-hoc manual scripts (`test.py`, `testsplitter.py`).
- **Fix Implemented**: Created automated test suite with standard `unittest` under `tests/` (`test_phase1_baseline.py` and `test_phase2_retrieval.py`).
- **Test Used for Verification**: 20 tests pass cleanly in 2.09s (`python -m unittest discover tests`).
- **Final Status**: **Fixed and verified**

---

### BUG-007: Missing BM25, Reciprocal Rank Fusion, and Evaluation Metrics
- **Bug ID**: `BUG-007`
- **Description**: Retrieval relied exclusively on dense vector search without sparse keyword matching, rank fusion, or quantitative evaluation metrics.
- **Affected Module**: `app/retrieval/`, `app/evaluation/`
- **Severity**: Critical
- **Root Cause**: Hybrid retrieval and evaluation engines were scheduled for Phase 2 and 3.
- **Fix Implemented (Phase 2 Component)**:
  1. Implemented Okapi BM25 (`app/retrieval/bm25.py`) with tokenization, TF, smoothed IDF, and length normalization.
  2. Implemented Reciprocal Rank Fusion (`app/retrieval/hybrid.py`) with constant $k=60$, stable chunk IDs, and diagnostic profiling.
  3. Integrated hybrid retrieval into `app/retrieval/retriever.py`, `pipeline.py`, and `rag_service.py`.
- **Test Used for Verification**: 5 BM25 tests and 3 RRF tests in `tests/test_phase2_retrieval.py`.
- **Final Status**: **Partially fixed (BM25 + RRF complete; quantitative benchmark evaluation scheduled for Phase 3)**

---

### BUG-008: Lack of Dedicated Industrial Decision Workspaces (RCA/FMEA/CAPA)
- **Bug ID**: `BUG-008`
- **Description**: Frontend currently operates as a general chat UI without dedicated workspaces for RCA (5-Whys), FMEA risk prioritization, CAPA tracking, or historical incident case matching.
- **Affected Module**: `frontend/app.py`, `app/services/`
- **Severity**: High
- **Root Cause**: Industrial engineering modules scheduled for Phase 4.
- **Fix Implemented**: Scheduled for Phase 4.
- **Test Used for Verification**: Pending Phase 4.
- **Final Status**: **Scheduled for Phase 4**
