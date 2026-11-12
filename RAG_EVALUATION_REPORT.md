# 📊 RAG-Pro 2.0 Comprehensive Evaluation & Experimental Research Report

**Project**: Industrial Knowledge Intelligence & Decision Support System  
**Developer**: Tarun Singh — B.Tech Computer Science (Data Science & AI)  
**Standard**: TREC / BeIR Information Retrieval Benchmarking Protocol  
**Audit Standard**: Phase 3.1 Audited Edition (Deterministic Chunk IDs, Separated Splits, Calibrated Cutoffs)  
**Hardware Profile**: Intel Core i5-4440 (4 physical cores, 4 threads @ 3.10 GHz), 7.69 GB RAM, Windows 10  
**Inference Profile**: Local CPU (Okapi BM25 Sparse) + Cloud REST API (`text-embedding-004` / Pinecone Serverless)  
**Date**: November 2026 | **Git Branch**: `main`

---

## 1. Executive Summary

This report documents the audited empirical evaluation of the information retrieval and answer generation pipeline of **RAG-Pro 2.0**.

Unlike superficial accuracy benchmarks, this evaluation addresses industrial knowledge retrieval requirements:
1. **Zero Synthetic Identification**: Evaluation uses persistent canonical chunk identifiers (`{file_name}_{strategy}_{index}`). No transient synthetic IDs (`item_0`) or dynamically generated virtual ground-truth items are permitted.
2. **Strict Metric Segregation**: Retrieval ranking metrics (Precision@5, Recall@5, Hit Rate@5, MRR, nDCG@5) are computed **exclusively on answerable queries** ($N=26$ Dev, $N=11$ Test). Out-of-domain unanswerable queries are evaluated via dedicated **Abstention Metrics** (Confusion Matrix, Correct Abstention Rate, False Retrieval Rate).
3. **Similarity Threshold Calibration**: Cosine similarity cutoff calibration on the Development Split established **$\tau^* = 0.65$** as the optimal operating point, eliminating 100% of out-of-domain false positives while preserving 100% answerable recall.
4. **Generalization on Held-Out Test Split**: The frozen threshold was verified on 15 unseen test questions, confirming that the calibrated system achieves 100% Hit Rate@5 and 100% Correct Abstention Rate without overfitting.
5. **Architectural Latency Stage Profiling**: Detailed stage profiling highlights that local CPU Okapi BM25 completes retrieval in **0.08 ms**, whereas cloud embedding generation and vector database querying require **~848 ms** due to network and API overhead.

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
| **Dense Semantic (Pinecone, $\tau^*=0.65$)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | **0.9808** [0.94, 1.0] | **0.9858** [0.96, 1.0] | 847.78 ms | **100.0% (9/9)** |
| **BM25 Sparse (Okapi, Medium Chunks)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | 0.8782 [0.77, 0.97] | 0.9098 [0.83, 0.98] | **0.08 ms** | 66.7% (6/9) |
| **Hybrid RRF (Dense + BM25, $k=60, \tau^*=0.65$)** | **1.0000** [1.0, 1.0] | 0.2000 [0.20, 0.20] | **1.0000** [1.0, 1.0] | 0.9231 [0.84, 0.99] | 0.9432 [0.88, 0.99] | 848.30 ms | **100.0% (9/9)** |

---

## 5. Generalization Validation: Held-Out Test Split ($N=15$)

*Evaluating the frozen calibrated threshold ($\tau^* = 0.65$) on unseen test questions ($N=11$ Answerable, $N=4$ Unanswerable).*

| Retrieval Pipeline | Hit Rate@5 | Precision@5 | Recall@5 | MRR | nDCG@5 | Mean Latency | Correct Abstention Rate (CAR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Semantic (Pinecone, $\tau^*=0.65$)** | **1.0000** | 0.2000 | **1.0000** | **0.9545** | **0.9664** | 811.87 ms | **100.0% (4/4)** |
| **BM25 Sparse (Okapi, Medium Chunks)** | **1.0000** | 0.2000 | **1.0000** | 0.9091 | 0.9329 | **0.09 ms** | 75.0% (3/4) |
| **Hybrid RRF (Dense + BM25, $k=60, \tau^*=0.65$)** | **1.0000** | 0.2000 | **1.0000** | 0.9091 | 0.9329 | 812.33 ms | **100.0% (4/4)** |

### Generalization Key Findings:
1. **Zero Degradation on Unseen Prompts**: Both Dense and Hybrid pipelines achieved 100% Hit Rate@5 on answerable questions in the Test split, proving that the semantic embeddings generalize beyond the development queries.
2. **Defensive Abstention Verified**: The unanswerable queries in the Test Split (Titanic box office, adult ibuprofen dosage, British prime minister in 1805, coffee cultivation) were 100% rejected by Dense and Hybrid with $\tau^* = 0.65$. Standalone BM25 failed on coffee cultivation (matching incidental tokens).

---

## 6. Granular Latency Profiling: Local CPU vs. Cloud Inference

Total retrieval latency is systematically accounted across discrete execution stages:

| Sub-System / Pipeline Stage | Dense Semantic | Okapi BM25 Sparse | Hybrid RRF ($k=60$) | Compute Location |
| :--- | :---: | :---: | :---: | :--- |
| **1. Query Preprocessing** | 0.02 ms | 0.02 ms | 0.02 ms | Local CPU (In-Memory) |
| **2. BM25 Inverted Index Search** | — | **0.06 ms** | 0.06 ms | Local CPU (In-Memory) |
| **3. Cloud Embedding Generation** | ~690.0 ms | — | ~690.0 ms | Google Generative AI Cloud API |
| **4. Pinecone Vector Search** | ~157.0 ms | — | ~157.0 ms | Pinecone Serverless Cloud Index |
| **5. RRF Rank Fusion** | — | — | 0.18 ms | Local CPU (Ordinal Rank Math) |
| **Total End-to-End Latency** | **847.78 ms** | **0.08 ms** | **848.30 ms** | Mixed (Cloud + Local) |

> [!TIP]
> **Engineering Architectural Insight**: Okapi BM25 delivers a **10,500x latency speedup** over Dense Vector Search because it avoids external REST API handshakes, TLS negotiation, and remote network serialization. For embedded or latency-critical edge deployments (e.g. factory SCADA or marine engine rooms), local BM25 provides instant sub-millisecond retrieval. For complex semantic queries, Hybrid RRF leverages dense semantics while maintaining exact keyword anchoring.

---

## 7. Chunk Size Sensitivity Analysis (Development Split)

Evaluating chunking configurations across identical questions using Okapi BM25:

| Chunking Configuration | Character Target | Chunks in Corpus | Hit Rate@5 | Precision@5 | MRR | Mean Latency | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Small Chunks** | 500 chars (100 ov) | 38 | 80.8% (21/26) | 0.1615 | 0.6410 | 7.45 ms | Context fragmentation; split equations |
| **Medium Chunks** ⭐ | 1000 chars (200 ov) | 30 | **100.0% (26/26)** | **0.2000** | **0.8782** | **0.08 ms** | Optimal boundary preservation |
| **Large Chunks** | 1500 chars (300 ov) | 27 | **100.0% (26/26)** | **0.2000** | **0.8782** | 0.10 ms | Minor dilution of dense ranking |

### Sensitivity Conclusion:
Small chunking (500 characters) suffered a **19.2% hit rate reduction** because multi-line markdown tables (such as API 610 Seal Flush plans or ASME thickness formulas) were fractured across chunk boundaries. Medium chunking (1,000 characters) provides the ideal operational balance between tabular completeness and embedding granularity.

---

## 8. Qualitative Answer Generation Pilot (Human Review)

Six representative questions covering each operational modality were submitted through the grounded generation pipeline. Answers, citations, and evidence grounding were reviewed by an engineer:

| Question ID | Category | Question Prompt | Key Retrieved Evidence | Verification Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **EVAL-001** | Factual lookup | What is the nominal compressor pressure ratio of the marine turbocharger? | `marine_diesel_turbocharger_manual.md_medium_1` | **PASS**: Verbatim accuracy on 3.8:1 ratio and 18,500 RPM rotor speed. Zero hallucination. |
| **EVAL-009** | Technical explanation | Explain the cylindrical shell thickness formula under ASME Section VIII Div 1. | `pressure_vessel_asme_sec_viii.md_medium_1` | **PASS**: Full formula `t = (P*R)/(S*E - 0.6*P) + C` and all parameter definitions reproduced correctly. |
| **EVAL-014** | Troubleshooting | Why is a 4-20mA process loop saturating above 20.8 mA and showing 50/60Hz noise? | `industrial_plc_scada_troubleshooting.md_medium_2` | **PASS**: Accurately diagnoses ground loop fault caused by multiple shield earth connections. |
| **EVAL-019** | Maintenance | What water quality and temperature are required for gas turbine compressor wash? | `gas_turbine_maintenance_procedures.docx_medium_2` | **PASS**: Verifies conductivity < 5.0 μS/cm and wash solution temperature of 60°C to 70°C. |
| **EVAL-023** | Comparison | Compare design pressure and allowable drop between shell and tube sides. | `heat_exchanger_thermal_rating_spec.xlsx_medium_0` | **PASS**: Exact tabular extraction: Shell (16 bar, 0.65 drop) vs Tube (25 bar, 0.45 drop). |
| **EVAL-028** | Unanswerable | What is the quantum superposition decoherence time for transmon qubits at 15 mK? | *(Abstained below threshold)* | **PASS**: Correctly abstains, stating quantum physics is outside the uploaded technical manuals. |

---

## 9. Error Taxonomy & Systematic Failure Analysis

Out of 150 query-method trials across both splits, only **4 method-level failure occurrences** were recorded, all originating from unanswerable queries in standalone BM25:

| Question ID | Split | Method | Category | Failure Mode | Root Cause Analysis & Engineering Remediation |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **EVAL-027** | `dev` | BM25 Sparse (Medium) | Unanswerable | Abstention Failure | Query terms ("excavators", "mining") matched incidental tokens in heavy-duty gas turbine procedures. Remediation: Calibrated semantic threshold ($\tau^*=0.65$) successfully eliminated this in Dense and Hybrid. |
| **EVAL-028** | `dev` | BM25 Sparse (Medium) | Unanswerable | Abstention Failure | Query token ("superposition") matched incidental mechanical stress terms in bearing manual. Remediation: Calibrated threshold ($\tau^*=0.65$) blocks false retrieval. |
| **EVAL-032** | `dev` | BM25 Sparse (Medium) | Unanswerable | Abstention Failure | Query token ("regulations") matched compliance codes in API 610 manual. Remediation: Semantic distance gating in Hybrid RRF. |
| **EVAL-050** | `test` | BM25 Sparse (Medium) | Unanswerable | Abstention Failure | Query token ("cultivation", "commercial") matched agricultural pump references. Remediation: Calibrated threshold ($\tau^*=0.65$) blocks false retrieval. |

---

## 10. Conclusion & Portfolio Impact

Phase 3.1 elevates RAG-Pro 2.0 to an auditable, scientifically sound portfolio asset:
1. **Honest, Defensible Numbers**: No manufactured 99% accuracy metrics. Every reported metric is backed by executed experiment JSON/CSV artifacts.
2. **Methodological Rigor**: Clear segregation of answerable retrieval from out-of-domain abstention, deterministic chunk identifiers, and split separation prevent data leakage.
3. **Engineering Trade-off Clarity**: Demonstrates deep understanding of local CPU vs cloud API trade-offs, proving that hybrid retrieval with calibrated thresholds provides industrial reliability.
