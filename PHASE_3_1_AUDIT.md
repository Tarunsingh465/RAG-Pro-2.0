# 🛡️ Phase 3.1 Evaluation Methodology Audit and Correction Report

**Project**: RAG-Pro 2.0 — Industrial Knowledge Intelligence & Decision Support System  
**Developer**: Tarun Singh — B.Tech Computer Science (Data Science & AI)  
**Evaluation Scope**: Phase 3 Methodology Audit, Ground-Truth Grounding, Threshold Calibration, and Metric Segregation  
**Date**: November 2026 | **Git Branch**: `main`

---

## 1. Executive Summary & Audit Mandate

During the engineering review of Phase 3 evaluation results, critical methodological limitations were identified that could compromise the scientific defensibility and reproducibility of reported metrics in an industrial engineering portfolio:

1. **Synthetic ID & Virtual Ground Truth Decoupling**: Previous metric calculation utilized transient runtime synthetic identifiers (`item_0`, `item_1`) and dynamically injected virtual ground truth (`unretrieved_corpus_gt_`) to compute metrics, rather than matching against verified, persistent chunk identifiers.
2. **Conflation of Answerable Retrieval with Out-of-Scope Abstention**: The previous implementation assigned a score of `1.0` to Precision@K, Recall@K, Hit Rate@K, and nDCG@K when evaluating unanswerable queries if zero documents were returned. This mathematically conflated document retrieval ranking quality with abstention capability, artificially inflating macro-averaged precision.
3. **Coarse Heuristic Chunk Relevance**: Relevance was determined via keyword presence and page-number heuristics (`_check_chunk_relevance()`), creating a risk that any chunk on the same page sharing a common keyword could be incorrectly labeled as relevant ground truth.
4. **Lump-Sum Latency Masking**: Retrieval latency was reported as an aggregate duration without decomposing local CPU processing from cloud API round-trips, obscuring the dramatic performance contrast between sub-millisecond local BM25 (~0.08 ms) and cloud embedding inference + vector database search (~750 ms).
5. **Absence of Calibrated Abstention Cutoff**: Dense vector search queries out-of-domain prompts returned arbitrary nearest-neighbor chunks without an empirical similarity threshold cutoff.

Phase 3.1 was executed as a **strict corrective iteration** (NOT a system rebuild) to eliminate all synthetic constructs, establish human-verified chunk ID ground truth, calibrate a cosine similarity threshold on a separate Development split, and validate all metrics on a Held-Out Test split.

---

## 2. Replaced vs. Provisional vs. Valid Results Matrix

| Metric / Artifact | Phase 3 Status | Phase 3.1 Audited Status | Reason for Replacement / Audit Finding |
| :--- | :--- | :--- | :--- |
| **Ground Truth Relevance Matching** | Heuristic (Page + Keywords) | **Replaced with Exact Chunk IDs** | Chunks now have deterministic canonical IDs (`{file_name}_{strategy}_{index}`). Verified verbatim evidence spans recorded for all 37 answerable queries. |
| **Synthetic Virtual IDs (`item_0`)** | Used in `_evaluate_retrieved_items` | **Completely Removed** | Evaluation now performs strict set intersection (`retrieved_cids ∩ relevant_cids`) with real persistent IDs. |
| **Unanswerable Scoring in Retrieval Metrics** | Mixed (Scored 1.0 on abstention) | **Strictly Segregated** | Retrieval metrics (Precision@5, Recall@5, Hit Rate@5, MRR, nDCG@5) are computed **exclusively on answerable queries**. Unanswerable queries are evaluated under dedicated **Abstention Metrics** (CAR, FRR, Confusion Matrix). |
| **Similarity Threshold Calibration** | None (Top-K unfiltered) | **Calibrated ($\tau^* = 0.65$)** | Dev split calibration across $[0.50, 0.75]$ established $\tau^* = 0.65$ as optimal cutoff (100% CAR, 100% Hit Rate on Dev split). |
| **Split Segregation (Dev vs Test)** | Unified single average ($N=50$) | **Separated into Dev ($N=35$) & Test ($N=15$)** | Avoids data leakage; threshold calibrated on Dev split is frozen and verified on Held-Out Test split. |
| **Latency Measurement** | Lump sum total latency | **Granular Latency Breakdown** | Explicitly profiles Query Preprocessing, Dense Embedding Generation, Pinecone Search, BM25 Index Search, and RRF Fusion. |
| **BM25 Sparse Tokenization & Inverted Index** | Valid | **Preserved (Valid)** | Tokenization, alphanumeric filtering, and BM25 scoring algorithm preserved without modification. |
| **Reciprocal Rank Fusion (k=60)** | Valid | **Preserved (Valid)** | RRF fusion logic verified; now correctly ingests calibrated threshold outputs. |

---

## 3. Ground Truth Annotation Audit

The complete evaluation dataset (`evaluation/datasets/industrial_eval_questions.jsonl`, $N=50$) was audited and verified:
- **Corpus Coverage**: 10 authentic industrial specifications (Marine Turbochargers, Centrifugal Pumps API 610, Heavy-Duty Gas Turbines, Heat Exchanger TEMA, Boiler Chemistry, Transformer DGA, ASME BPVC Sec VIII, PLC/SCADA).
- **Development Split**: 35 questions (26 Answerable, 9 Unanswerable).
- **Held-Out Test Split**: 15 questions (11 Answerable, 4 Unanswerable).
- **Field Completeness**: Every answerable question records:
  1. `question_id` (e.g. `EVAL-001` through `EVAL-050`).
  2. `relevant_document_ids` (exact source file name).
  3. `relevant_section` (exact section, slide, or sheet heading).
  4. `relevant_evidence_span` (exact verbatim quote from corpus text).
  5. `ground_truth_chunks_medium` (exact chunk IDs for 1000-char strategy).
  6. `ground_truth_chunks_small` (exact chunk IDs for 500-char strategy).
  7. `ground_truth_chunks_large` (exact chunk IDs for 1500-char strategy).
  8. `human_review_status`: `"verified"`.

---

## 4. Similarity Threshold Calibration on Development Split

Pinecone cosine similarity scores ($1 - \text{cosine\_distance}$) were evaluated on the 35 Development Split questions across candidate thresholds $\tau \in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]$:

| Threshold ($\tau$) | Answerable Hit Rate@5 ($N=26$) | Unanswerable CAR ($N=9$) | Unanswerable FRR ($N=9$) | Retrieval Precision | Balanced F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 100.0% (26/26) | 0.0% (0/9) | 100.0% (9/9) | 74.3% | 0.0000 |
| **0.55** | 100.0% (26/26) | 11.1% (1/9) | 88.9% (8/9) | 76.5% | 0.2000 |
| **0.60** | 100.0% (26/26) | 88.9% (8/9) | 11.1% (1/9) | 96.3% | 0.9412 |
| **0.65** ⭐ | **100.0% (26/26)** | **100.0% (9/9)** | **0.0% (0/9)** | **100.0%** | **1.0000** |
| **0.70** | 100.0% (26/26) | 100.0% (9/9) | 0.0% (0/9) | 100.0% | 1.0000 |
| **0.75** | 88.5% (23/26) | 100.0% (9/9) | 0.0% (0/9) | 100.0% | 0.9388 |

### Calibration Conclusion:
- At $\tau < 0.60$, out-of-domain unanswerable queries (e.g. Carbon-14 half-life, sourdough bread recipe) match background engineering vocabulary with cosine similarity between 0.55 and 0.58, producing false positive retrievals (FRR up to 100%).
- At **$\tau^* = 0.65$**, the system achieves **100% Correct Abstention Rate** without sacrificing a single answerable query (100% Hit Rate).
- At $\tau > 0.70$, false negatives emerge as legitimate technical queries (e.g. specialized acronyms and thermal ratings) fall below 0.75.
- Therefore, **$\tau^* = 0.65$** is frozen as the optimal threshold for all subsequent benchmarks.

---

## 5. Granular Latency Accounting: Local vs. Cloud

The audited benchmarking architecture exposes the latency decomposition across all execution stages:

| Stage | Okapi BM25 Sparse | Dense Semantic (Pinecone) | Hybrid RRF ($k=60$) | Infrastructure Context |
| :--- | :---: | :---: | :---: | :--- |
| **Query Preprocessing** | 0.02 ms | 0.02 ms | 0.02 ms | Local CPU string normalization |
| **BM25 Search** | 0.06 ms | — | 0.06 ms | In-memory inverted index traversal |
| **Cloud Embedding Gen** | — | ~620.0 ms | ~620.0 ms | External REST API (`text-embedding-004`) |
| **Pinecone Vector Search** | — | ~120.0 ms | ~120.0 ms | Remote cloud vector index query |
| **RRF Fusion** | — | — | 0.18 ms | Rank reciprocal score merging ($k=60$) |
| **Total End-to-End** | **0.08 ms** | **~740.0 ms** | **~740.3 ms** | Local CPU vs Cloud API round-trip |

> [!IMPORTANT]
> Okapi BM25 provides **9,250x faster retrieval** than Dense Vector search because it executes entirely on the local host CPU with zero network serialization or cloud inference overhead. Hybrid RRF combines the best of both: lexical keyword precision with dense semantic understanding.

---

## 6. Verification on Held-Out Test Split ($N=15$)

Applying the frozen threshold $\tau^* = 0.65$ to the unseen Held-Out Test Split verifies that our retrieval models generalize without overfitting:
- **Okapi BM25**: Achieves 100% Hit Rate@5 on answerable test questions ($MRR=0.9091$, $nDCG@5=0.9322$).
- **Dense Semantic ($\tau^* = 0.65$)**: Achieves 100% Correct Abstention Rate on unanswerable test questions (Titanic box office, adult ibuprofen dosage, British prime minister in 1805, soil nutrients).

---

## 7. Phase 3.1.1 Integrity Check & Methodological Reconciliations

### 7.1 Precision@5 Invariant Reconciliation
An audit of all answerable queries confirmed that mean Precision@5 = 0.2000 is mathematically dictated by single-relevant-document ground truth:
$$\text{Precision@5} = \frac{|\mathcal{A}_5 \cap \mathcal{R}|}{5} = \frac{1}{5} = 0.2000 \quad \text{when } |\mathcal{R}| = 1 \text{ and } \text{Hit Rate@5} = 1.0$$
Relevant annotations were inspected across all 37 answerable queries; each targets a specific, non-redundant engineering specification clause. Ground truth annotations were strictly maintained without artificial inflation.

### 7.2 Vector Store Synchronization & Elimination of Duplicate Vectors
An inspection revealed that Pinecone previously held 32 vectors due to 2 duplicate entries indexed with random UUIDs. The vector index was wiped and repopulated with deterministic vector IDs (`bearing_failure_fmea_and_capa.csv_medium_0`, etc.), ensuring exact 1-to-1 parity between Pinecone's 30 vectors and BM25's 30 chunks.

### 7.3 Independent BM25 Abstention Mechanism
BM25 scores are unbounded TF-IDF scores, incompatible with cosine similarity thresholds. On the Development split ($N=35$), unanswerable queries scored $\le 4.8626$, while answerable queries scored $\ge 6.3266$. A dedicated BM25 threshold $\tau_{\text{BM25}}^* = 5.0$ was calibrated and frozen. On the Held-Out Test split ($N=15$), this rule achieved 100% CAR (4/4) and 100% answerable coverage (11/11).

### 7.4 Threshold Tie & Generalization Boundaries
Both $\tau = 0.65$ and $\tau = 0.70$ achieved $F_1 = 1.0000$ on the Development objective. $\tau^* = 0.65$ was chosen to provide a safety margin against false negatives for technical acronyms (Hit Rate drops at $\tau \ge 0.75$). This threshold is specific to `gemini-embedding-001` and this 10-document corpus. The $N=4$ unanswerable test sample is acknowledged as a pilot diagnostic size rather than an unconstrained safety proof.

### 7.5 Deliverables
Detailed question-by-question evidence is recorded in:
- `PER_QUESTION_EVALUATION_AUDIT.csv` (150 audit rows across 50 questions and 3 methods)
- `PHASE_3_1_1_FINAL_REVIEW.md` (Forensic evaluation review report)
- **Hybrid RRF ($\tau^* = 0.65$)**: Combines 100% Hit Rate@5 on answerable test questions with 100% Correct Abstention Rate on unanswerable test queries.

