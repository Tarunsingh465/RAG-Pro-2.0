# 📊 TYRION-2.0 Comprehensive Evaluation & Experimental Research Report

**Project**: Industrial Knowledge Intelligence & Decision Support System  
**Developer**: Tarun Singh — B.Tech Computer Science (Data Science & AI)  
**Standard**: TREC / BeIR Information Retrieval Benchmarking Protocol  
**Audit Standard**: Phase 3.1 Audited Edition (Deterministic Chunk IDs, Separated Splits, Calibrated Cutoffs)  
**Hardware Profile**: Intel Core i5-4440 (4 physical cores, 4 threads @ 3.10 GHz), 7.69 GB RAM, Windows 10  
**Inference Profile**: Local CPU (Okapi BM25 Sparse) + Cloud REST API (`models/gemini-embedding-001` / Pinecone Serverless)  
**Date**: November 2026 | **Git Branch**: `main`

---

## 1. Executive Summary

This report documents the audited empirical evaluation of the information retrieval and answer generation pipeline of **TYRION-2.0**.

Unlike superficial accuracy benchmarks, this evaluation addresses industrial knowledge retrieval requirements:
1. **Zero Synthetic Identification**: Evaluation uses persistent canonical chunk identifiers (`{file_name}_{strategy}_{index}`). No transient synthetic IDs (`item_0`) or dynamically generated virtual ground-truth items are permitted.
2. **Strict Metric Segregation**: Retrieval ranking metrics (Precision@5, Recall@5, Hit Rate@5, MRR, nDCG@5) are computed **exclusively on answerable queries** ($N=26$ Dev, $N=11$ Test). Out-of-domain unanswerable queries are evaluated via dedicated **Abstention Metrics** (Confusion Matrix, Correct Abstention Rate, False Retrieval Rate).
3. **Similarity Threshold Calibration**: Cosine similarity cutoff calibration on the Development Split established **$\tau^* = 0.65$** as the optimal operating point, eliminating 100% of out-of-domain false positives while preserving 100% answerable recall.
4. **Generalization on Held-Out Test Split**: The frozen threshold was verified on 15 unseen test questions, confirming that the calibrated system achieves 100% Hit Rate@5 and 100% Correct Abstention Rate without overfitting.
5. **Architectural Latency Stage Profiling**: Detailed stage profiling highlights that local CPU Okapi BM25 completes retrieval in **0.08 ms**, whereas cloud embedding generation (689.58 ms) and vector database querying (172.40 ms) total **861.98 ms** due to network and API overhead.

---

## 2. Experimental Setup & Corpus Topology

The evaluation corpus consists of **10 authentic industrial specifications** totaling **30 chunks** under the standard medium chunking configuration (1,000 characters, 200 overlap):

| Specification Document | Discipline | Domain Scope | Chunks (Medium) |
| :--- | :--- | :--- | :---: |
| `marine_diesel_turbocharger_manual.md` | Marine / Mechanical | Aerodynamics, Surge Limits, Hydrodynamic Bearings | 4 |
| `centrifugal_pump_api610_standards.docx` | Chemical / Process | API 610 NPSH criteria, Mechanical Seal Plans 11/32/53A | 4 |
| `gas_turbine_maintenance_procedures.docx` | Power Generation | Hot Gas Path inspection intervals, Offline Water Washing | 3 |
| `heat_exchanger_thermal_rating_spec.xlsx` | Heat Transfer | TEMA AES Floating Head rating, Fouling resistances | 2 |
| `hvac_chiller_energy_optimization.docx` | Facility / HVAC | Full-load COP, IPLV AHRI 550/590, Condenser approach | 3 |
| `industrial_plc_scada_troubleshooting.md` | Automation | Modbus/TCP port 502 diagnostics, 4-20mA ground loops | 3 |
| `pressure_vessel_asme_sec_viii.md` | Mechanical Safety | ASME BPVC Section VIII Div 1 MAWP, UG-99 Hydrotest | 3 |
| `steam_boiler_feedwater_chemistry.xlsx` | Utility / Chemistry | Dissolved O2, Silica limits, Phosphate treatment | 2 |
| `substation_transformer_oil_analysis.md` | High-Voltage Electrical | IEEE C57.104 DGA Fault Gases (C2H2, C2H4, H2), ASTM D877 | 3 |
| `bearing_failure_fmea_and_capa.csv` | Reliability / Maintenance | FMEA Matrix: Electrical Fluting, False Brinelling, Starvation | 3 |

### Dataset Partitioning:
- **Development Split ($N=35$)**: 26 Answerable queries across 5 industrial categories + 9 Unanswerable distractor queries.
- **Held-Out Test Split ($N=15$)**: 11 Answerable queries + 4 Unanswerable distractor queries.

---

## 3. Similarity Threshold Calibration (Development Split)

To determine the optimal rejection cutoff for dense cosine similarity ($1 - \text{cosine\_distance}$), candidate thresholds $\tau \in [0.50, 0.75]$ were systematically evaluated across the 35 Development Split questions:

| Candidate Cutoff ($\tau$) | Answerable Hit Rate@5 ($N=26$) | Unanswerable CAR ($N=9$) | Unanswerable FRR ($N=9$) | Retrieval Precision | Balanced F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 100.0% (26/26) | 0.0% (0/9) | 100.0% (9/9) | 74.3% | 0.0000 |
| **0.55** | 100.0% (26/26) | 11.1% (1/9) | 88.9% (8/9) | 76.5% | 0.2000 |
| **0.60** | 100.0% (26/26) | 88.9% (8/9) | 11.1% (1/9) | 96.3% | 0.9412 |
| **0.65** ⭐ *(Optimal)* | **100.0% (26/26)** | **100.0% (9/9)** | **0.0% (0/9)** | **100.0%** | **1.0000** |
| **0.70** | 100.0% (26/26) | 100.0% (9/9) | 0.0% (0/9) | 100.0% | 1.0000 |
| **0.75** | 88.5% (23/26) | 100.0% (9/9) | 0.0% (0/9) | 100.0% | 0.9388 |

### Calibration Takeaways:
- **Low Thresholds ($\tau \le 0.55$)**: Out-of-domain queries (e.g. sourdough bread recipe, tennis ITF rules) match incidental engineering terms at cosine scores of 0.55–0.58, generating 100% false retrievals.
- **Optimal Threshold ($\tau^* = 0.65$)**: Achieves a perfect balance ($F_1 = 1.0000$), completely suppressing all 9 out-of-domain queries while maintaining a 100% hit rate across all 26 technical questions.
- **High Thresholds ($\tau \ge 0.75$)**: Precision remains 100%, but Recall drops to 88.5% as specialized acronyms and complex engineering equations fall slightly below 0.75.

---

## 4. Benchmark Results: Development Split ($N=35$)

*Retrieval metrics evaluated strictly on answerable questions ($N=26$). Abstention metrics evaluated strictly on unanswerable questions ($N=9$). Top-K cutoff $K=5$.*

| Retrieval Pipeline | Hit Rate@5 [95% CI] | Precision@5 [95% CI] | Recall@5 [95% CI] | MRR [95% CI] | nDCG@5 [95% CI] | Mean Latency | Correct Abstention Rate (CAR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Semantic (Pinecone, $\tau^*=0.65$)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | **0.9808** [0.94, 1.0] | **0.9858** [0.96, 1.0] | 861.98 ms | **100.0% (9/9)** |
| **BM25 Sparse (Okapi, $\tau_{\text{BM25}}=5.0$)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | 0.8782 [0.79, 0.96] | 0.9098 [0.84, 0.97] | **0.08 ms** | **100.0% (9/9)** |
| **Hybrid RRF (Dense + BM25, $k=60, \tau^*=0.65$)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | 0.9231 [0.85, 0.98] | 0.9432 [0.89, 0.99] | 862.36 ms | **100.0% (9/9)** |

> [!NOTE]
> **Precision@5 Ceiling Effect ($0.2000$)**:
> Every answerable query in the benchmark has exactly one targeted ground-truth chunk ($|\mathcal{G}|=1$). Because Precision@$K = |\mathcal{R}_K \cap \mathcal{G}| / K$, the maximum possible score for $K=5$ is $1/5 = \mathbf{0.2000}$. At 100% Hit Rate@5, Precision@5 is mathematically bounded to exactly 0.2000. See `PER_QUESTION_EVALUATION_AUDIT.csv` and `PHASE_3_1_2_FINAL_REVIEW.md` for the per-question verification.

---

## 5. Generalization Validation: Held-Out Test Split ($N=15$)

*Evaluating the frozen calibrated thresholds ($\tau_{\text{dense}}^* = 0.65$, $\tau_{\text{BM25}}^* = 5.0$) on unseen test questions ($N=11$ Answerable, $N=4$ Unanswerable).*

| Retrieval Pipeline | Hit Rate@5 [95% CI] | Precision@5 [95% CI] | Recall@5 [95% CI] | MRR [95% CI] | nDCG@5 [95% CI] | Mean Latency | Correct Abstention Rate (CAR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Semantic (Pinecone, $\tau^*=0.65$)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | **0.9545** [0.86, 1.0] | **0.9664** [0.90, 1.0] | 861.98 ms | **100.0% (4/4)** |
| **BM25 Sparse (Okapi, $\tau_{\text{BM25}}=5.0$)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | 0.9091 [0.77, 1.0] | 0.9329 [0.83, 1.0] | **0.15 ms** | **100.0% (4/4)** |
| **Hybrid RRF (Dense + BM25, $k=60, \tau^*=0.65$)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | 0.9091 [0.77, 1.0] | 0.9329 [0.83, 1.0] | 862.31 ms | **100.0% (4/4)** |

### Generalization Key Findings:
1. **Zero Degradation on Unseen Prompts**: Dense, BM25, and Hybrid pipelines all achieved 100% Hit Rate@5 on answerable questions in the Test split, proving that retrieval generalizes robustly to unseen technical prompts.
2. **Defensive Abstention Verified**: The unanswerable queries in the Test Split (Titanic box office, adult ibuprofen dosage, British prime minister in 1805, coffee cultivation) were 100% rejected across Dense, BM25, and Hybrid with calibrated cutoffs.
3. **Statistical Sample-Size Boundaries**: While achieving 100% CAR (4/4) on held-out test data verifies that abstention logic functions properly, $N=4$ unanswerable samples is a pilot diagnostic check rather than a population-level safety proof.
4. **Phase 3.1.2 Deduplication Lift**: By eliminating duplicate ranking rows caused by float metadata serialization (`_0.0` vs `_0`), distinct relevant chunks advanced in the ranking, lifting Hybrid MRR on the Test split from 0.8636 to 0.9091 (+5.27%) and nDCG@5 from 0.8993 to 0.9329 (+3.74%).

---

## 6. Granular Latency Profiling: Local CPU vs. Cloud Inference

Total retrieval latency is systematically accounted across discrete execution stages:

| Sub-System / Pipeline Stage | Dense Semantic | Okapi BM25 Sparse | Hybrid RRF ($k=60$) | Compute Location |
| :--- | :---: | :---: | :---: | :--- |
| **1. Query Preprocessing** | 0.002 ms | 0.001 ms | 0.001 ms | Local CPU (In-Memory) |
| **2. BM25 Inverted Index Search** | — | **0.08 ms** | 0.09 ms | Local CPU (In-Memory) |
| **3. Cloud Embedding Generation** | **689.58 ms** | — | **689.58 ms** | Google Generative AI Cloud API (`models/gemini-embedding-001`) |
| **4. Pinecone Vector Search** | **172.40 ms** | — | **172.40 ms** | Pinecone Serverless Cloud Index |
| **5. RRF Rank Fusion** | — | — | **0.29 ms** | Local CPU (Ordinal Rank Math) |
| **Total End-to-End Latency** | **861.98 ms** | **0.08 ms** | **862.36 ms** | Mixed (Cloud + Local) |

> [!TIP]
> **Engineering Architectural Insight**: Okapi BM25 delivers an **over 10,000x latency speedup** (0.08 ms vs 861.98 ms) over Dense Vector Search because it avoids external REST API handshakes, TLS negotiation, and remote network serialization. For embedded or latency-critical edge deployments (e.g. factory SCADA or marine engine rooms), local BM25 provides instant sub-millisecond retrieval. For complex semantic queries, Hybrid RRF leverages dense semantics while maintaining exact keyword anchoring.

---

## 7. Chunk Size Sensitivity Analysis (Development Split)

Evaluating chunking configurations across identical questions using Okapi BM25:

| Chunking Configuration | Character Target | Chunks in Corpus | Hit Rate@5 | Precision@5 | MRR | Mean Latency | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Small Chunks** | 500 chars (100 ov) | 38 | 80.8% (21/26) | 0.1615 | 0.6410 | 7.45 ms | Context fragmentation; split equations |
| **Medium Chunks** ⭐ | 1000 chars (200 ov) | 30 | **100.0% (26/26)** | **0.2000** | **0.8782** | **0.07 ms** | Optimal boundary preservation |
| **Large Chunks** | 1500 chars (300 ov) | 27 | **100.0% (26/26)** | **0.2000** | **0.8782** | 0.10 ms | Minor dilution of dense ranking |

### Sensitivity Conclusion:
Small chunking (500 characters) suffered a **19.2% hit rate reduction** because multi-line markdown tables (such as API 610 Seal Flush plans or ASME thickness formulas) were fractured across chunk boundaries. Medium chunking (1,000 characters) provides the ideal operational balance between tabular completeness and embedding granularity.

---

## 8. Qualitative Answer Generation Pilot (Human Review)

> [!NOTE]
> **Scope & Sample Boundary**: This evaluation is a **qualitative diagnostic pilot** across $N=6$ human-reviewed cases covering each operational modality. It provides indicative validation of end-to-end evidence synthesis, but is not claimed as a large-scale statistical generation benchmark.
>
> **Audit Finding (EVAL-023 Factual Correction)**: In the original Phase 3.1.2 pilot record, EVAL-023 was prematurely logged as `PASS` because the reviewer checked only shell-side figures. An audited cross-check against `heat_exchanger_thermal_rating_spec.xlsx` revealed that the generated response reported a tube-side allowable pressure drop of `0.85 bar` (actual: `0.45 bar`) and fabricated a citation to `Sheet: Thermal_Rating, Rows 8-12`. The historical text is preserved and the verdict revised to **FAIL / REVISED PARTIAL** (5/6 PASS, 83.3% pilot fidelity).

Six representative questions covering each operational modality were submitted through the grounded generation pipeline (`gemini-2.5-flash`). Generated answers, citations, and evidence grounding were reviewed by an engineer:

| Question ID | Category | Question Prompt | Key Retrieved Evidence | Verification Verdict | Factual / Citation Errors |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **EVAL-001** | Factual lookup | What is the nominal compressor pressure ratio of the marine turbocharger? | `marine_diesel_turbocharger_manual.md_medium_1` | **PASS**: Verbatim accuracy on 3.8:1 ratio and 18,500 RPM rotor speed. Evidence fully grounded. | 0 |
| **EVAL-009** | Technical explanation | Explain the cylindrical shell thickness formula under ASME Section VIII Div 1. | `pressure_vessel_asme_sec_viii.md_medium_1` | **PASS**: Full formula `t = (P*R)/(S*E - 0.6*P) + C` and all parameter definitions reproduced correctly. | 0 |
| **EVAL-014** | Troubleshooting | Why is a 4-20mA process loop saturating above 20.8 mA and showing 50/60Hz noise? | `industrial_plc_scada_troubleshooting.md_medium_2` | **PASS**: Accurately diagnoses ground loop fault caused by multiple shield earth connections. | 0 |
| **EVAL-019** | Maintenance | What water quality and temperature are required for gas turbine compressor wash? | `gas_turbine_maintenance_procedures.docx_medium_2` | **PASS**: Verifies conductivity < 5.0 μS/cm and wash solution temperature of 60°C to 70°C. | 0 |
| **EVAL-023** | Comparison | Compare design pressure and allowable drop between shell and tube sides. | `heat_exchanger_thermal_rating_spec.xlsx_medium_0` | **FAIL / REVISED PARTIAL** *(Audited Correction)*: Shell-side values correct (16 bar, 0.65 drop), but tube-side allowable drop reported as 0.85 bar (actual ground truth is 0.45 bar in `Thermal_Design`); cited non-existent sheet 'Thermal_Rating' Rows 8-12. Original review recorded PASS due to reviewer checking only shell-side values. | 2 (1 factual, 1 citation) |
| **EVAL-028** | Unanswerable | What is the quantum superposition decoherence time for transmon qubits at 15 mK? | *(Abstained below threshold)* | **PASS**: Correctly abstains, stating quantum physics is outside the uploaded technical manuals. | 0 |

---

## 9. Error Taxonomy & Systematic Failure Analysis

Across all 150 query-method trials in the final audited benchmark, **0 failure occurrences** were observed when applying calibrated abstention thresholds ($\tau_{\text{dense}}^* = 0.65$, $\tau_{\text{BM25}}^* = 5.0$):

1. **Unthresholded BM25 Baseline Vulnerability (Historical Diagnostic)**:
   - Without an abstention cutoff, standalone Okapi BM25 incurred 4 abstention failures across Dev (EVAL-027, EVAL-028, EVAL-032) and Test (EVAL-050) due to incidental token matches (BM25 scores 3.4–4.8).
2. **Calibrated Abstention Remediation**:
   - Applying the independently calibrated Dev threshold $\tau_{\text{BM25}}^* = 5.0$ completely resolved all 4 failures, achieving **100% CAR (9/9 Dev, 4/4 Test)** without losing a single answerable retrieval (lowest answerable score was 6.33 on Dev and 7.21 on Test).
3. **Dense & Hybrid Defense**:
   - Both Dense Semantic and Hybrid RRF pipelines achieved **100% CAR (9/9 Dev, 4/4 Test)** with $\tau^* = 0.65$, confirming robust rejection of out-of-domain queries.

---

## 10. Conclusion & Portfolio Impact

Phase 3.1 and Phase 3.1.1 elevate TYRION-2.0 to an auditable, scientifically sound portfolio asset:
1. **Honest, Defensible Numbers**: No manufactured 99% accuracy metrics. Every reported metric is backed by executed experiment JSON/CSV artifacts.
2. **Methodological Rigor**: Clear segregation of answerable retrieval from out-of-domain abstention, deterministic chunk identifiers, and split separation prevent data leakage.
3. **Engineering Trade-off Clarity**: Demonstrates deep understanding of local CPU vs cloud API trade-offs, proving that hybrid retrieval with calibrated thresholds provides industrial reliability.
4. **Precision Ceiling Transparency**: Defends Precision@5 = 0.2000 through the mathematical ceiling of single-item retrieval ($|\mathcal{G}|=1$), accompanied by a per-question audit CSV (`PER_QUESTION_EVALUATION_AUDIT.csv`).
