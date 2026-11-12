# 🧪 Test Execution Report — Phase 3

## 1. Overview
- **Execution Date**: 2026-11-12
- **Environment**: Windows 10 | Python 3.11.0 | Intel Core i5-4440 @ 3.10GHz | 7.69 GB RAM
- **Test Framework**: Python `unittest` standard library (0 external test runner dependencies)
- **Total Tests Discovered**: 35
- **Total Tests Passed**: **35 (100%)**
- **Failures / Errors**: **0**
- **Execution Time**: **2.591 seconds**

---

## 2. Test Execution Breakdown

### Phase 1 Regression Suite (`tests/test_phase1_baseline.py`)
| Test Name | Module Covered | Objective | Result |
| :--- | :--- | :--- | :--- |
| `test_zero_api_developer_identity` | `conversational_engine` | Verifies Tarun developer identity match in <0.5ms with 0 API calls | **PASSED** |
| `test_zero_api_greetings` | `conversational_engine` | Verifies small-talk greetings bypass external APIs | **PASSED** |
| `test_engineering_classifier_positive`| `core.engineering` | Verifies detection of technical queries across CS/IT and Mechanical | **PASSED** |
| `test_engineering_classifier_negative`| `core.engineering` | Verifies out-of-domain queries (Gandhi, Napoleon) are not classified as engineering | **PASSED** |
| `test_supervisor_fast_path_routing` | `supervisor_service` | Verifies deterministic routing to 'greeting' and 'rag' | **PASSED** |
| `test_data_leakage_isolation_when_empty`| `chat_service` | Verifies empty knowledge base short-circuits with 0 citations and no leakage | **PASSED** |

### Phase 2 Document Intelligence & Retrieval Suite (`tests/test_phase2_retrieval.py`)
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

### Phase 3 Evaluation & Benchmarking Suite (`tests/test_phase3_evaluation.py`)
| Test Name | Module Covered | Objective | Result |
| :--- | :--- | :--- | :--- |
| `test_precision_at_k_standard` | `evaluation.metrics` | Verifies Precision@K calculation on partially relevant retrieved list | **PASSED** |
| `test_precision_at_k_perfect_and_zero` | `evaluation.metrics` | Verifies boundary cases: 1.0 on all-relevant and 0.0 on zero-relevant | **PASSED** |
| `test_precision_at_k_unanswerable_abstention`| `evaluation.metrics` | Verifies correct abstention score (1.0 if empty, 0.0 if false retrieval) | **PASSED** |
| `test_precision_at_k_invalid_k` | `evaluation.metrics` | Verifies ValueError raised when $k \le 0$ | **PASSED** |
| `test_recall_at_k_standard` | `evaluation.metrics` | Verifies Recall@K calculation against total ground-truth count | **PASSED** |
| `test_recall_at_k_perfect` | `evaluation.metrics` | Verifies 1.0 Recall@K when all relevant items are retrieved | **PASSED** |
| `test_recall_at_k_unanswerable_abstention` | `evaluation.metrics` | Verifies unanswerable query recall abstention behavior | **PASSED** |
| `test_hit_rate_at_k` | `evaluation.metrics` | Verifies binary Hit Rate@K (1.0 if $\ge 1$ relevant, else 0.0) | **PASSED** |
| `test_reciprocal_rank` | `evaluation.metrics` | Verifies Reciprocal Rank calculation at ranks 1, 2, 4 and miss (0.0) | **PASSED** |
| `test_ndcg_at_k` | `evaluation.metrics` | Verifies nDCG@K formula against ideal DCG ordering | **PASSED** |
| `test_bootstrap_confidence_interval` | `evaluation.metrics` | Verifies 95% empirical bootstrap CI on uniform and varied samples | **PASSED** |
| `test_evaluate_single_retrieval` | `evaluation.metrics` | Verifies end-to-end question metric result generation dataclass | **PASSED** |
| `test_aggregate_experiment_metrics` | `evaluation.metrics` | Verifies macro-aggregation, percentiles, and category breakdowns | **PASSED** |
| `test_corpus_manager_strategies` | `evaluation.runner` | Verifies loading corpus across small (500), medium (1000), large (1500) | **PASSED** |
| `test_bm25_benchmark_runner` | `evaluation.runner` | Verifies BM25 benchmark execution and metric generation on dev split | **PASSED** |

---

## 3. Regression Verification
- **BUG-001 (LLM Timeout)**: Verified that `enable_thinking=False` remains active in NVIDIA NIM client.
- **BUG-002 (Data Leakage)**: Verified that active document filter and short-circuit remain 100% functional.
- **BUG-003 (Startup Probe)**: Verified that cached dimension resolution executes with 0 external API calls.
- **BUG-004 (Structured Extractors)**: Preserved and verified table, slide, and multi-sheet extractors.
- **BUG-007 (Evaluation Framework)**: Complete reproducible framework with zero synthetic metrics. All 35 tests pass.
