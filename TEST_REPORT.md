# 🧪 Test Execution Report — Phase 3.1 & Phase 3.1.1 Evaluation Integrity Check

## 1. Overview
- **Execution Date**: 2026-11-12
- **Environment**: Windows 10 | Python 3.11.0 | Intel Core i5-4440 @ 3.10GHz | 7.69 GB RAM
- **Test Framework**: Python `unittest` standard library (0 external test runner dependencies)
- **Total Tests Discovered**: **41**
- **Total Tests Passed**: **41 (100%)**
- **Failures / Errors**: **0**
- **Execution Time**: **~2.6 seconds**

---

## 2. Test Execution Breakdown

### Phase 1 Regression Suite (`tests/test_phase1_baseline.py` — 6 tests)
| Test Name | Module Covered | Objective | Result |
| :--- | :--- | :--- | :--- |
| `test_zero_api_developer_identity` | `conversational_engine` | Verifies Tarun developer identity match in <0.5ms with 0 API calls | **PASSED** |
| `test_zero_api_greetings` | `conversational_engine` | Verifies small-talk greetings bypass external APIs | **PASSED** |
| `test_engineering_classifier_positive`| `core.engineering` | Verifies detection of technical queries across CS/IT and Mechanical | **PASSED** |
| `test_engineering_classifier_negative`| `core.engineering` | Verifies out-of-domain queries (Gandhi, Napoleon) are not classified as engineering | **PASSED** |
| `test_supervisor_fast_path_routing` | `supervisor_service` | Verifies deterministic routing to 'greeting' and 'rag' | **PASSED** |
| `test_data_leakage_isolation_when_empty`| `chat_service` | Verifies empty knowledge base short-circuits with 0 citations and no leakage | **PASSED** |

### Phase 2 Document Intelligence & Retrieval Suite (`tests/test_phase2_retrieval.py` — 14 tests)
| Test Name | Module Covered | Objective | Result |
| :--- | :--- | :--- | :--- |
| `test_bm25_empty_corpus` | `retrieval.bm25` | Verifies empty BM25 corpus returns empty list gracefully | **PASSED** |
| `test_bm25_unknown_query_terms` | `retrieval.bm25` | Verifies out-of-vocabulary query terms return empty list | **PASSED** |
| `test_bm25_scoring_and_ranking` | `retrieval.bm25` | Verifies TF-IDF scoring and rank ordering on technical chunks | **PASSED** |
| `test_bm25_duplicate_chunks` | `retrieval.bm25` | Verifies updating chunks with identical chunk_id prevents duplication | **PASSED** |
| `test_bm25_active_document_filtering` | `retrieval.bm25` | Verifies active document filtering strictly isolates search to active files | **PASSED** |
| `test_rrf_score_calculation` | `retrieval.hybrid` | Verifies exact formula $1 / (60 + \text{rank})$ calculation | **PASSED** |
| `test_rrf_deduplication_and_score_boost`| `retrieval.hybrid` | Verifies shared chunks in dense & sparse receive fused score boost | **PASSED** |
| `test_rrf_missing_documents_in_one_ranking`| `retrieval.hybrid` | Verifies chunks present in only one ranking are handled gracefully | **PASSED** |
| `test_docx_table_and_section_extraction`| `extractors.office`| Verifies DOCX headings, paragraphs, and markdown table preservation | **PASSED** |
| `test_pptx_slide_attribution` | `extractors.office`| Verifies PPTX slide-by-slide grouping and accurate slide number metadata | **PASSED** |
| `test_xlsx_multi_sheet_extraction` | `extractors.tabular` | Verifies XLSX multi-sheet extraction, sheet names, and batched rows | **PASSED** |
| `test_chunking_presets_and_character_units`| `ingestion.splitter` | Verifies small (500), medium (1000), large (1500) presets & metadata | **PASSED** |
| `test_dimension_mismatch_raises_error` | `vectorstore.pinecone` | Verifies ValueError raised when model and index dimensions mismatch | **PASSED** |
| `test_startup_uses_cache_without_calling_embed_query`| `vectorstore.pinecone`| Verifies startup utilizes dimension cache without calling external embedding API | **PASSED** |

### Phase 3.1 & 3.1.1 Evaluation & Benchmarking Suite (`tests/test_phase3_evaluation.py` — 21 tests)
| Test Name | Module Covered | Objective | Result |
| :--- | :--- | :--- | :--- |
| `test_precision_at_k_standard` | `evaluation.metrics` | Verifies Precision@K calculation on partially relevant retrieved list with deduplication | **PASSED** |
| `test_precision_at_k_perfect_and_zero` | `evaluation.metrics` | Verifies boundary cases: 1.0 on all-relevant and 0.0 on zero-relevant | **PASSED** |
| `test_precision_at_k_raises_on_empty_relevant` | `evaluation.metrics` | Verifies ValueError raised when computing retrieval precision on unanswerable query | **PASSED** |
| `test_precision_at_k_invalid_k` | `evaluation.metrics` | Verifies ValueError raised when $k \le 0$ | **PASSED** |
| `test_recall_at_k_standard` | `evaluation.metrics` | Verifies Recall@K calculation against total ground-truth count with deduplication | **PASSED** |
| `test_recall_at_k_perfect` | `evaluation.metrics` | Verifies 1.0 Recall@K when all relevant items are retrieved | **PASSED** |
| `test_recall_at_k_raises_on_empty_relevant` | `evaluation.metrics` | Verifies ValueError raised when computing recall on unanswerable query | **PASSED** |
| `test_hit_rate_at_k` | `evaluation.metrics` | Verifies binary Hit Rate@K (1.0 if $\ge 1$ relevant, else 0.0) | **PASSED** |
| `test_hit_rate_raises_on_empty_relevant` | `evaluation.metrics` | Verifies ValueError raised on empty relevant set for Hit Rate | **PASSED** |
| `test_reciprocal_rank` | `evaluation.metrics` | Verifies Reciprocal Rank calculation at ranks 1, 2, 4 and miss (0.0) | **PASSED** |
| `test_reciprocal_rank_raises_on_empty_relevant` | `evaluation.metrics` | Verifies ValueError raised on empty relevant set for MRR | **PASSED** |
| `test_ndcg_at_k` | `evaluation.metrics` | Verifies nDCG@K formula against ideal DCG ordering with deduplication | **PASSED** |
| `test_ndcg_raises_on_empty_relevant` | `evaluation.metrics` | Verifies ValueError raised on empty relevant set for nDCG | **PASSED** |
| `test_abstention_metrics` | `evaluation.metrics` | Verifies TN, FP, TP, FN confusion matrix, CAR, FRR, and accuracy | **PASSED** |
| `test_latency_breakdown` | `evaluation.metrics` | Verifies LatencyBreakdown dataclass stage summing (prep, emb, pinecone, bm25, rrf) | **PASSED** |
| `test_bootstrap_confidence_interval` | `evaluation.metrics` | Verifies 95% empirical bootstrap CI across 1,000 resamples | **PASSED** |
| `test_evaluate_single_retrieval` | `evaluation.metrics` | Verifies end-to-end question metric result generation dataclass | **PASSED** |
| `test_aggregate_experiment_metrics` | `evaluation.metrics` | Verifies macro-aggregation, percentiles, abstention, and category breakdowns | **PASSED** |
| `test_corpus_exists` | `evaluation.runner` | Verifies authentic 10-document industrial corpus exists on disk | **PASSED** |
| `test_corpus_chunking_strategies` | `evaluation.runner` | Verifies chunking across small (500), medium (1000), large (1500) presets | **PASSED** |
| `test_bm25_dev_and_test_splits` | `evaluation.runner` | Verifies BM25 execution across isolated dev (35) and held-out test (15) splits | **PASSED** |

---

## 3. Phase 3.1.1 Integrity Check Verification
- **Precision@5 Mathematical Ceiling Audit**:
  - Proved analytically and verified numerically that Precision@5 = 0.2000 is the exact theoretical maximum bound when $|\mathcal{G}|=1$ under 100% Hit Rate.
  - Generated `PER_QUESTION_EVALUATION_AUDIT.csv` containing full 150-row verification across all methods, splits, and queries.
- **Corpus & Pinecone Vector Store Reconciliation**:
  - Enforced deterministic IDs in `app/vectorstore/pinecone.py` (`add_documents(ids=...)`).
  - Cleared and re-indexed Pinecone index to match exactly 30 unique medium chunks (`total_vector_count = 30`), eliminating 2 stale duplicate vectors.
- **Independent BM25 Abstention Protocol**:
  - Calibrated independent BM25 score cutoff $\tau_{\text{BM25}}^* = 5.0$ on the Development Split.
  - Verified on Held-Out Test Split: 100% CAR (4/4 unanswerable) and 100% Hit Rate (11/11 answerable).
- **Threshold Tie-Breaker Rationale & Sample-Size Boundaries**:
  - Documented empirical tie between $\tau = 0.65$ and $\tau = 0.70$ on Dev split ($F_1 = 1.0000$), with selection justified as a buffer against false negatives.
  - Stated explicitly that $N=4$ unanswerable test queries constitutes a diagnostic pilot rather than a population-level safety proof.
- **Regression Invariants Maintained**:
  - BUG-001 (LLM Timeout), BUG-002 (Data Leakage), BUG-003 (Startup Probe), BUG-004 (Structured Extractors) all verified intact.
  - All 41 unit and regression tests pass without errors.
