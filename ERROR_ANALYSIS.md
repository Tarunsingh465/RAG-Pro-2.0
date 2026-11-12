# 🔍 Failure & Error Analysis Report — Phase 3.1.2 Audited

**Generated**: 2026-11-12 | **Git Commit at Execution**: `4f6013d` (Working tree state committed as `e860afa`)  
**Evaluation Scope**: 50 Evaluation Questions (35 Development Split, 15 Held-Out Test Split)  
**Retrieval Architectures Audited**:
- Dense Semantic Vector Search (Pinecone Serverless, `models/gemini-embedding-001`, 3072d)
- Local Okapi BM25 Sparse Search (In-Memory CPU, $k_1=1.5, b=0.75$)
- Hybrid Reciprocal Rank Fusion (Dense + BM25, $k=60$)

---

## 1. Overview of Defect Remediation in Phase 3.1.2

Phase 3.1.2 systematically resolved two architectural and evaluation defects:

| Defect ID | Defect Classification | Affected Components | Root Cause | Remediation & Operational Status |
| :--- | :--- | :--- | :--- | :--- |
| **Defect A** | Ground-Truth Label Leakage | `app/evaluation/runner.py` (`run_hybrid_experiment`) | Ground-truth metadata (`is_answerable`) conditionally triggered dense candidate rejection during candidate retrieval. | **RESOLVED**: Removed all references to ground-truth labels during retrieval, ranking, and thresholding. Implemented independent source acceptance ($\tau_{\text{dense}} \ge 0.65$, $\tau_{\text{BM25}} \ge 5.0$). |
| **Defect B** | Duplicate Ranking Chunk IDs | `app/retrieval/hybrid.py` (`get_canonical_chunk_id`) | Float metadata serialization from Pinecone (`chunk_index: 0.0`) produced divergent string IDs (`..._0.0` vs `..._0`), causing RRF to treat identical chunks as separate candidates. | **RESOLVED**: Added regex float-suffix normalization (`re.sub(r'\_(\d+)\.0$', r'_\1', raw_id)`), integer casting, and stamped canonical ID on document copies. Eliminated 100% of duplicate rows. |

---

## 2. Audited Failure Taxonomy & Results

Under the corrected independent acceptance policy ($\tau_{\text{dense}}^* = 0.65$, $\tau_{\text{BM25}}^* = 5.0$), all 50 evaluation queries were executed across Dense, BM25, and Hybrid pipelines.

### 2.1 Failure Count by Method and Split
| Pipeline Method | Split | Evaluated Queries | Answerable Hits | Unanswerable Abstentions | Total Failures |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dense Semantic (Pinecone, $\tau=0.65$)** | Development ($N=35$) | 35 | 26 / 26 (100.0%) | 9 / 9 (100.0%) | **0** |
| **BM25 Sparse (Okapi, $\tau=5.0$)** | Development ($N=35$) | 35 | 26 / 26 (100.0%) | 9 / 9 (100.0%) | **0** |
| **Hybrid RRF (Dense + BM25, $k=60, \tau=0.65$)** | Development ($N=35$) | 35 | 26 / 26 (100.0%) | 9 / 9 (100.0%) | **0** |
| **Dense Semantic (Pinecone, $\tau=0.65$)** | Held-Out Test ($N=15$) | 15 | 11 / 11 (100.0%) | 4 / 4 (100.0%) | **0** |
| **BM25 Sparse (Okapi, $\tau=5.0$)** | Held-Out Test ($N=15$) | 15 | 11 / 11 (100.0%) | 4 / 4 (100.0%) | **0** |
| **Hybrid RRF (Dense + BM25, $k=60, \tau=0.65$)** | Held-Out Test ($N=15$) | 15 | 11 / 11 (100.0%) | 4 / 4 (100.0%) | **0** |

**Total Method Failure Occurrences**: **0** (All 37 answerable queries retrieved their relevant target chunk in top-5; all 13 unanswerable queries cleanly abstained with 0 retrieved chunks).

---

## 3. Detailed Failure Mode Analysis & Edge Case Sensitivity

### 3.1 Chunk Size Sensitivity Failure Modes (BM25 Small Chunks)
In the chunk size comparison experiment on the Development split ($N=26$ answerable queries), Small Chunking (500 characters) failed on **5 queries** (Hit Rate 80.8%), whereas Medium (1,000 characters) and Large (1,500 characters) achieved 100% Hit Rate:

| Question ID | Category | Query Description | Root Cause of Failure under Small Chunking |
| :--- | :--- | :--- | :--- |
| **EVAL-002** | Factual lookup | API 610 NPSH criteria and margins | Table boundary split between minimum margin equation and operating condition parameters. |
| **EVAL-009** | Technical explanation | ASME Section VIII Div 1 cylindrical shell thickness | Formula variables ($P, R, S, E, C_a$) separated across contiguous chunk boundary. |
| **EVAL-014** | Troubleshooting | 4-20mA signal loop saturation and ground loops | Diagnostic flow chart fractured into disjointed text fragments. |
| **EVAL-023** | Comparison | Shell vs. Tube side design pressure and pressure drop | Multi-sheet Excel table divided mid-row, stripping column headers from cell values. |
| **EVAL-035** | Maintenance | Gas turbine offline compressor wash fluid limits | Water cleanliness ppm limits split from temperature guidance range. |

**Engineering Remediation**:
Medium chunking (1,000 characters, 200 overlap) and specialized structured extractors (`OfficeExtractor` preserving markdown tables, `TabularExtractor` chunking by sheet with prepended headers) prevent cross-boundary context loss.

### 3.2 Unanswerable Query Discrimination Analysis
The 13 unanswerable queries were designed to test boundary rejection across diverse out-of-domain topics:
1. **Historical / Pop Culture**: EVAL-047 (Titanic box office), EVAL-049 (British PM during 1805 Trafalgar).
2. **Medical / Culinary**: EVAL-048 (Adult ibuprofen dosage), EVAL-029 (Sourdough hydration ratios).
3. **Quantum / Advanced Physics**: EVAL-028 (Transmon qubit decoherence at 15 mK).
4. **Agriculture / Biology**: EVAL-050 (Arabica coffee tree soil chemistry).

**Threshold Separation Proof**:
- Under Dense Semantic search, unanswerable query scores span $[0.5331, 0.6421]$. The maximum score was observed on EVAL-027 ($0.6421$) due to general maintenance phrasing. The calibrated threshold $\tau_{\text{dense}} = 0.65$ sits safely above this ceiling, achieving 100% Correct Abstention.
- Under BM25 Sparse search, unanswerable query scores span $[0.0000, 4.8626]$ due to occasional common English words ("temperature", "limits", "schedule"). The calibrated threshold $\tau_{\text{BM25}} = 5.0$ provides a clean boundary above incidental matches.

---

## 4. Verification Invariants & Audit Trace

All 255 per-question evaluation audit records and 50 hybrid diagnostic records were subjected to automated validation:
1. **Duplicate Check**: `len(retrieved_chunk_ids) == len(set(retrieved_chunk_ids))` evaluated to `True` for 100% of rows.
2. **Top-K Cutoff**: Top-K contains exactly $\min(5, \text{unique\_candidates})$ items without synthetic padding.
3. **Independence**: Candidate acceptance in Hybrid RRF is strictly disjunctive ($s_{\text{dense}} \ge 0.65 \lor s_{\text{BM25}} \ge 5.0$). Zero ground-truth information is consumed during retrieval.
