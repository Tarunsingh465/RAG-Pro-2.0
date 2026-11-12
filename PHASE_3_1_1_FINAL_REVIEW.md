# 🔬 PHASE 3.1.1 — Final Evaluation Integrity Check & Audit Review

**Project**: RAG-Pro 2.0 — Industrial Knowledge Intelligence Platform  
**Developer**: Tarun Singh — B.Tech Computer Science (Data Science & AI)  
**Execution Date**: November 12, 2026  
**Host Environment**: Intel Core i5-4440 (4 cores @ 3.10 GHz), 7.69 GB RAM, Windows 10  
**Embedding Model**: Google `models/gemini-embedding-001` (3072 dimensions)  
**Vector Store**: Pinecone Serverless (AWS `us-east-1`, cosine distance metric)  
**Corpus**: 10 authentic multi-disciplinary industrial engineering specifications (30 medium chunks)  
**Evaluation Dataset**: 50 questions (35 Development split: 26 answerable, 9 unanswerable; 15 Held-Out Test split: 11 answerable, 4 unanswerable)

---

## 1. Executive Summary

Phase 3.1.1 conducts a rigorous, forensic audit of the experimental evaluation pipeline prior to concluding the RAG Evaluation phase. This check verifies:
1. **Mathematical Reconciliation of Precision@5 = 0.2000**: Proves that an invariant Precision@5 of 0.2000 across answerable queries is the exact mathematical ceiling of single-evidence question answering when $k=5$, rather than an implementation bug or hardcoding artifact.
2. **Corpus & Vector Store Synchronization**: Confirms that Pinecone and Okapi BM25 indices operate on identical 30-chunk corpora, verifies document provenance, and documents the elimination of legacy duplicate vectors.
3. **Threshold Calibration Rationale**: Documents the empirical tie between $\tau = 0.65$ and $\tau = 0.70$ on the Development objective ($F_1 = 1.0000$), justifies the selection of $\tau^* = 0.65$, and formalizes domain/model scope limitations.
4. **BM25 Independent Abstention Rule**: Implements a dedicated BM25 score cutoff ($\tau_{\text{BM25}}^* = 5.0$) calibrated strictly on Development data, and validates it frozen on the Held-Out Test split (100% CAR, 0% FRR).
5. **Statistical Rigor & Sample-Size Boundaries**: Reports 95% bootstrap confidence intervals for all test metrics, exact abstention counts, and acknowledges the statistical constraints of $N=4$ held-out negative queries.
6. **Answer-Quality Pilot Auditing**: Formalizes the 6-question human review cards with verbatim generated answers, citations, and error auditing.

---

## 2. Precision@5 Audit & Mathematical Reconciliation

### 2.1 The Precision-at-$k$ Ceiling Phenomenon
A noticeable observation in the Phase 3.1 results was that every answerable retrieval configuration reported a mean Precision@5 of **0.2000** on both the Development ($N=26$) and Held-Out Test ($N=11$) splits.

In standard Information Retrieval literature (Manning, Raghavan, & Schütze, *Introduction to Information Retrieval*, Cambridge University Press, Ch. 8), Precision@$k$ is defined as:
$$\text{Precision@}k = \frac{|\mathcal{A}_k \cap \mathcal{R}|}{k}$$
Where:
- $\mathcal{A}_k$ is the set of top-$k$ unique retrieved documents.
- $\mathcal{R}$ is the set of all genuinely relevant ground-truth documents in the entire corpus.

When a query is an exact technical lookup with a single targeted evidence passage (e.g., maximum continuous rotor speed of a marine turbocharger, or joint efficiency factor under ASME Section VIII), the ground-truth set has size $|\mathcal{R}| = 1$.

Because $|\mathcal{A}_k \cap \mathcal{R}| \le |\mathcal{R}|$, the maximum possible number of relevant documents that *can ever appear* in the top-$k$ retrieved list is $\min(k, |\mathcal{R}|)$.
Therefore, the maximum mathematically achievable Precision@$k$ is bounded by:
$$\text{Precision@}k \le \frac{\min(k, |\mathcal{R}|)}{k} = \frac{1}{5} = \mathbf{0.2000}$$

Whenever a retrieval system achieves a **100% Hit Rate@5** (retrieving the 1 relevant chunk for every query), the number of relevant chunks retrieved is exactly 1 for all queries. Consequently:
$$\text{Precision@5}(q_i) = \frac{1}{5} = 0.2000 \quad \forall i \in \{1, \dots, N\}$$
$$\text{Mean Precision@5} = \frac{1}{N} \sum_{i=1}^N 0.2000 = \mathbf{0.2000}$$

### 2.2 Independent Verification of Held-Out Test Queries
The 11 held-out answerable questions were audited independently. Every retrieved chunk was verified against ground truth, and metrics were recalculated using independent scripts:

| Question ID | Category | Ground Truth Chunk ID | Top-5 Retrieved Chunks (Hybrid RRF) | Rel Count | Prec@5 | Rec@5 | Hit@5 | RR | nDCG@5 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EVAL-036** | Factual lookup | `marine_diesel_turbocharger_manual.md_medium_1` | `marine_diesel_turbocharger_manual.md_medium_1`, `marine_diesel_turbocharger_manual.md_medium_0`, `marine_diesel_turbocharger_manual.md_medium_3`, `substation_transformer_oil_analysis.md_medium_0`, `pressure_vessel_asme_sec_viii.md_medium_1` | 1 | 0.2000 | 1.0000 | 1.0 | 1.0000 | 1.0000 |
| **EVAL-037** | Maintenance | `gas_turbine_maintenance_procedures.docx_medium_2` | `gas_turbine_maintenance_procedures.docx_medium_2`, `gas_turbine_maintenance_procedures.docx_medium_0`, `gas_turbine_maintenance_procedures.docx_medium_1`, `substation_transformer_oil_analysis.md_medium_0`, `hvac_chiller_energy_optimization.docx_medium_1` | 1 | 0.2000 | 1.0000 | 1.0 | 1.0000 | 1.0000 |
| **EVAL-038** | Technical explanation | `pressure_vessel_asme_sec_viii.md_medium_1` | `pressure_vessel_asme_sec_viii.md_medium_0`, `pressure_vessel_asme_sec_viii.md_medium_1`, `pressure_vessel_asme_sec_viii.md_medium_2`, `hvac_chiller_energy_optimization.docx_medium_1`, `hvac_chiller_energy_optimization.docx_medium_0` | 1 | 0.2000 | 1.0000 | 1.0 | 0.5000 | 0.6309 |
| **EVAL-039** | Troubleshooting | `substation_transformer_oil_analysis.md_medium_1` | `substation_transformer_oil_analysis.md_medium_0`, `substation_transformer_oil_analysis.md_medium_1`, `substation_transformer_oil_analysis.md_medium_2`, `marine_diesel_turbocharger_manual.md_medium_2`, `centrifugal_pump_api610_standards.docx_medium_1` | 1 | 0.2000 | 1.0000 | 1.0 | 0.5000 | 0.6309 |
| **EVAL-040** | Troubleshooting | `bearing_failure_fmea_and_capa.csv_medium_1` | `bearing_failure_fmea_and_capa.csv_medium_1`, `bearing_failure_fmea_and_capa.csv_medium_0`, `marine_diesel_turbocharger_manual.md_medium_3`, `industrial_plc_scada_troubleshooting.md_medium_2`, `marine_diesel_turbocharger_manual.md_medium_2` | 1 | 0.2000 | 1.0000 | 1.0 | 1.0000 | 1.0000 |
| **EVAL-041** | Technical explanation | `hvac_chiller_energy_optimization.docx_medium_2` | `hvac_chiller_energy_optimization.docx_medium_2`, `hvac_chiller_energy_optimization.docx_medium_1`, `centrifugal_pump_api610_standards.docx_medium_1`, `marine_diesel_turbocharger_manual.md_medium_3`, `centrifugal_pump_api610_standards.docx_medium_0` | 1 | 0.2000 | 1.0000 | 1.0 | 1.0000 | 1.0000 |
| **EVAL-042** | Factual lookup | `steam_boiler_feedwater_chemistry.xlsx_medium_1` | `steam_boiler_feedwater_chemistry.xlsx_medium_0`, `steam_boiler_feedwater_chemistry.xlsx_medium_1`, `marine_diesel_turbocharger_manual.md_medium_3`, `gas_turbine_maintenance_procedures.docx_medium_2`, `pressure_vessel_asme_sec_viii.md_medium_0` | 1 | 0.2000 | 1.0000 | 1.0 | 0.5000 | 0.6309 |
| **EVAL-043** | Factual lookup | `heat_exchanger_thermal_rating_spec.xlsx_medium_1` | `heat_exchanger_thermal_rating_spec.xlsx_medium_1`, `heat_exchanger_thermal_rating_spec.xlsx_medium_0`, `pressure_vessel_asme_sec_viii.md_medium_0`, `substation_transformer_oil_analysis.md_medium_2`, `marine_diesel_turbocharger_manual.md_medium_0` | 1 | 0.2000 | 1.0000 | 1.0 | 1.0000 | 1.0000 |
| **EVAL-044** | Factual lookup | `marine_diesel_turbocharger_manual.md_medium_3` | `marine_diesel_turbocharger_manual.md_medium_3`, `bearing_failure_fmea_and_capa.csv_medium_1`, `marine_diesel_turbocharger_manual.md_medium_0`, `substation_transformer_oil_analysis.md_medium_2`, `pressure_vessel_asme_sec_viii.md_medium_0` | 1 | 0.2000 | 1.0000 | 1.0 | 1.0000 | 1.0000 |
| **EVAL-045** | Technical explanation | `industrial_plc_scada_troubleshooting.md_medium_2` | `industrial_plc_scada_troubleshooting.md_medium_2`, `industrial_plc_scada_troubleshooting.md_medium_0`, `industrial_plc_scada_troubleshooting.md_medium_1`, `substation_transformer_oil_analysis.md_medium_0`, `centrifugal_pump_api610_standards.docx_medium_1` | 1 | 0.2000 | 1.0000 | 1.0 | 1.0000 | 1.0000 |
| **EVAL-046** | Comparison | `bearing_failure_fmea_and_capa.csv_medium_1` | `bearing_failure_fmea_and_capa.csv_medium_1`, `bearing_failure_fmea_and_capa.csv_medium_0`, `marine_diesel_turbocharger_manual.md_medium_3`, `hvac_chiller_energy_optimization.docx_medium_1`, `marine_diesel_turbocharger_manual.md_medium_2` | 1 | 0.2000 | 1.0000 | 1.0 | 1.0000 | 1.0000 |

### 2.3 Manual Metric Calculation Walkthrough
To verify that evaluation algorithms adhere to textbook definitions:
- **EVAL-036**:
  - Retrieved rank of relevant chunk: 1
  - $\text{RR} = \frac{1}{1} = 1.0000$
  - $\text{DCG@5} = \sum_{i=1}^5 \frac{\text{rel}_i}{\log_2(i+1)} = \frac{1}{\log_2(2)} = 1.0000$
  - $\text{IDCG@5} = \frac{1}{\log_2(2)} = 1.0000 \implies \text{nDCG@5} = 1.0000$.
- **EVAL-038**:
  - Retrieved rank of relevant chunk: 2 (preceded by Section 0 Overview)
  - $\text{RR} = \frac{1}{2} = 0.5000$
  - $\text{DCG@5} = \frac{0}{\log_2(2)} + \frac{1}{\log_2(3)} = \frac{1}{1.58496} = 0.6309$
  - $\text{IDCG@5} = \frac{1}{\log_2(2)} = 1.0000 \implies \text{nDCG@5} = 0.6309$.
- **Aggregate Test Metrics (Hybrid RRF)**:
  - Mean Precision@5: $\frac{11 \times 0.2000}{11} = \mathbf{0.2000}$
  - Mean Recall@5: $\frac{11 \times 1.0000}{11} = \mathbf{1.0000}$
  - Mean Hit Rate@5: $\frac{11 \times 1.0}{11} = \mathbf{1.0000}$
  - Mean MRR: $\frac{8 \times 1.0 + 3 \times 0.5}{11} = \frac{9.5}{11} = \mathbf{0.8636}$
  - Mean nDCG@5: $\frac{8 \times 1.0 + 3 \times 0.6309}{11} = \frac{9.8927}{11} = \mathbf{0.8993}$

All metrics in `EXPERIMENT_RESULTS.json` match these independent calculations to 4 decimal places.

### 2.4 Integrity of Ground-Truth Annotations
A critical question investigated was whether annotations artificially restrict relevance to 1 chunk.
- Inspection of the corpus confirmed that each technical question targets a unique specification clause, equation, or table row located in exactly one chunk.
- For example, in `pressure_vessel_asme_sec_viii.md`, chunk 0 introduces scope, chunk 1 provides the exact MAWP shell thickness equation ($t = \frac{PR}{SE - 0.6P} + C_a$) and joint efficiency $E=1.0$, while chunk 2 covers hydrostatic test pressure ratios ($1.3 \times \text{MAWP}$).
- Assigning relevance to chunk 0 or chunk 2 for a question asking about shell thickness would introduce false ground-truth positives.
- In accordance with instruction guidelines, **ground-truth annotations were not artificially inflated**, preserving the authentic evidentiary nature of technical documentation.

---

## 3. Chunk Identity, Provenance, and Corpus Verification

### 3.1 Deterministic Chunk Naming
Across all three multi-granularity configurations, chunk IDs follow the deterministic canonical format:
$$\text{chunk\_id} = \{\text{file\_name}\}\_\{\text{strategy}\}\_\{\text{chunk\_index}\}$$

| Configuration | Character Window | Overlap | Total Chunks | ID Format Sample | Uniqueness |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Small** | 500 chars | 100 chars | 38 | `centrifugal_pump_api610_standards.docx_small_0` | 100% (38/38) |
| **Medium** | 1000 chars | 200 chars | 30 | `centrifugal_pump_api610_standards.docx_medium_0` | 100% (30/30) |
| **Large** | 1500 chars | 300 chars | 27 | `centrifugal_pump_api610_standards.docx_large_0` | 100% (27/27) |

### 3.2 Document Provenance & Corpus Parity
The experiment corpus in `evaluation/corpus/` consists of 10 authentic industrial specifications spanning 4 major engineering domains:
1. `bearing_failure_fmea_and_capa.csv` (Reliability Engineering / FMEA)
2. `centrifugal_pump_api610_standards.docx` (Rotary Mechanical / API 610 11th Ed)
3. `gas_turbine_maintenance_procedures.docx` (Turbomachinery / Gas Turbine Power Plants)
4. `heat_exchanger_thermal_rating_spec.xlsx` (Thermal / Process Heat Transfer, TEMA Type BEM)
5. `hvac_chiller_energy_optimization.docx` (HVAC / Industrial Refrigeration)
6. `industrial_plc_scada_troubleshooting.md` (Industrial Automation / Control Systems)
7. `marine_diesel_turbocharger_manual.md` (Marine & Stationary Diesel Engines)
8. `pressure_vessel_asme_sec_viii.md` (Pressure Equipment / ASME BPVC Sec VIII Div 1)
9. `steam_boiler_feedwater_chemistry.xlsx` (Chemical / Steam Cycle Water Treatment)
10. `substation_transformer_oil_analysis.md` (High Voltage Electrical / DGA ASTM D3612)

### 3.3 Elimination of Vector Store Leakage
- **Defect Identified**: During earlier iterative testing, Pinecone contained 32 vectors because two chunks from `bearing_failure_fmea_and_capa.csv` had been indexed twice with random UUIDs.
- **Remediation**:
  1. `app/vectorstore/pinecone.py` was refactored so `add_documents()` enforces deterministic vector IDs matching `doc.metadata["chunk_id"]`.
  2. The Pinecone index was purged (`delete_all()`) and repopulated with the exact 30 medium chunks using their canonical IDs.
  3. Index stats confirmed: `total_vector_count = 30`, `dimension = 3072`.
  4. Both Okapi BM25 and Pinecone now operate on the exact same 30 chunks, eliminating duplicate candidate injections and vector leakage.

---

## 4. Threshold Calibration Clarification & Boundaries

### 4.1 Empirical Grid Search Results (Development Split, $N=35$)
The similarity threshold $\tau$ was swept across $[0.50, 0.75]$ in steps of 0.05 on the 35 Development queries:

| Threshold ($\tau$) | Answerable Hit Rate@5 ($N=26$) | Unanswerable CAR ($N=9$) | Unanswerable FRR ($N=9$) | Retrieval Precision | Balanced $F_1$ Score | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50** | 1.0000 (26/26) | 0.0000 (0/9) | 1.0000 (9/9) | 0.7429 (26/35) | 0.0000 | Extreme Over-Retrieval |
| **0.55** | 1.0000 (26/26) | 0.1111 (1/9) | 0.8889 (8/9) | 0.7647 (26/34) | 0.2000 | High False Positives |
| **0.60** | 1.0000 (26/26) | 0.8889 (8/9) | 0.1111 (1/9) | 0.9630 (26/27) | 0.9412 | Near Optimal |
| **0.65** | **1.0000 (26/26)** | **1.0000 (9/9)** | **0.0000 (0/9)** | **1.0000 (26/26)** | **1.0000** | ⭐ **Selected Operating Point** |
| **0.70** | **1.0000 (26/26)** | **1.0000 (9/9)** | **0.0000 (0/9)** | **1.0000 (26/26)** | **1.0000** | ⚖️ **Tied on Objective** |
| **0.75** | 0.8846 (23/26) | 1.0000 (9/9) | 0.0000 (0/9) | 1.0000 (23/23) | 0.9388 | False Negatives on Formulas |

### 4.2 Objective Tie & Rationale for $\tau^* = 0.65$
Both $\tau = 0.65$ and $\tau = 0.70$ achieved identical balanced scores ($F_1 = 1.0000$) on the development split.
$\tau^* = 0.65$ was chosen over $0.70$ for the following engineering reasons:
1. **False Negative Buffer**: At $\tau = 0.75$, 3 answerable queries failed due to cosine scores dropping to 0.71–0.73 on highly condensed engineering acronyms (e.g. Modbus/TCP, API 610 vibration limits). Setting the cutoff at 0.65 provides a 0.05 safety buffer against lexical variation and paraphrase variance.
2. **Sensitivity vs Specificity Trade-off**: In mission-critical industrial decision support, retrieving an incomplete answer or falsely abstaining on a real plant emergency query (false negative) carries higher operational risk than slightly lower score margin, provided unanswerable rejection remains 100%.

### 4.3 Scope Boundaries & Non-Universality
- **Model Specificity**: This cutoff is strictly calibrated for Google's `models/gemini-embedding-001` in 3072-dimensional space under cosine similarity. Other embedding models (e.g., `text-embedding-3-small` at 1536d or `all-MiniLM-L6-v2` at 384d) exhibit different score distributions and require independent calibration.
- **Corpus Specificity**: The threshold reflects the technical density and vocabulary overlap of this 10-document engineering corpus.
- **Negative Sample Limitation**: The held-out negative sample consists of 4 unanswerable questions. While achieving 4/4 correct abstentions demonstrates proof-of-concept abstention behavior, $N=4$ is statistically insufficient to make population-level safety guarantees across unconstrained open-domain inquiries.

---

## 5. Independent BM25 Abstention Mechanism

### 5.1 Rationale for Independent Abstention
Applying a cosine similarity threshold to BM25 scores is mathematically invalid because Okapi BM25 produces unbounded TF-IDF scores ($S_{\text{BM25}} \in [0, \infty)$) that scale with term frequency, document length, and query length, rather than bounded similarity in $[-1, 1]$.

### 5.2 Development Calibration Analysis
BM25 top scores were analyzed across all 35 Development queries:
- **Answerable Queries ($N=26$)**:
  - Score range: $[6.3266, 27.5914]$
  - Mean top score: $14.2930$
  - Minimum score: $6.3266$ (EVAL-022, phosphate treatment concentration)
- **Unanswerable Queries ($N=9$)**:
  - Score range: $[0.0000, 4.8626]$
  - 6 queries produced exactly 0 hits (TopScore = 0.0000)
  - 3 queries matched common incidental words ("maintenance", "time", "rules") with scores 3.47 to 4.86.
  - Maximum score: $4.8626$ (EVAL-027, Boeing 777 airliner landing gear)

There exists a clear separation margin between the highest unanswerable BM25 score ($4.8626$) and the lowest answerable score ($6.3266$).

### 5.3 Implementation & Frozen Evaluation
A minimum BM25 score threshold **$\tau_{\text{BM25}}^* = 5.0$** was calibrated on Development data:
$$\text{If } \max_{d} S_{\text{BM25}}(q, d) < 5.0 \implies \text{Abstain } (\mathcal{A} = \emptyset)$$

**Held-Out Test Set Validation (Frozen $\tau_{\text{BM25}}^* = 5.0$, $N=15$)**:
- **Answerable Queries ($N=11$)**:
  - Minimum BM25 top score: $7.2131$ (EVAL-040)
  - All 11 queries exceeded $\tau_{\text{BM25}}^* = 5.0$ $\implies$ **100.0% Answerable Coverage Preserved (11/11)**.
- **Unanswerable Queries ($N=4$)**:
  - EVAL-047 (Titanic box office): 0 hits ($S = 0.0000 < 5.0$) $\implies$ Abstained.
  - EVAL-048 (Ibuprofen dosage): 0 hits ($S = 0.0000 < 5.0$) $\implies$ Abstained.
  - EVAL-049 (UK Prime Minister): 0 hits ($S = 0.0000 < 5.0$) $\implies$ Abstained.
  - EVAL-050 (Soil nutrients): 1 hit ($S = 2.6402 < 5.0$) $\implies$ Abstained.
  - Correct Abstention Rate: **100.0% (4/4)**, False Retrieval Rate: **0.0% (0/4)**.

---

## 6. Statistical Reporting & Confidence Intervals

### 6.1 Performance Summary Across Splits
All metrics for answerable queries include 95% empirical bootstrap confidence intervals computed over 1,000 resamples:

| Split | Pipeline | Precision@5 [95% CI] | Recall@5 [95% CI] | Hit Rate@5 [95% CI] | MRR [95% CI] | nDCG@5 [95% CI] | Correct Abstention Rate (CAR) | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development**<br>($N=35$: 26 ans, 9 unans) | **Dense Semantic** ($\tau=0.65$) | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.981 [0.942, 1.000] | 0.986 [0.957, 1.000] | **9 / 9 (100.0%)** | 935.4 ms |
| | **BM25 Sparse** ($\tau_{\text{BM25}}=5.0$) | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.878 [0.782, 0.962] | 0.908 [0.835, 0.970] | **9 / 9 (100.0%)** | **0.07 ms** |
| | **Hybrid RRF** ($k=60, \tau=0.65$) | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.885 [0.795, 0.962] | 0.913 [0.842, 0.972] | **9 / 9 (100.0%)** | 935.9 ms |
| **Held-Out Test**<br>($N=15$: 11 ans, 4 unans) | **Dense Semantic** ($\tau=0.65$) | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.955 [0.864, 1.000] | 0.966 [0.899, 1.000] | **4 / 4 (100.0%)** | 862.0 ms |
| | **BM25 Sparse** ($\tau_{\text{BM25}}=5.0$) | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.909 [0.773, 1.000] | 0.932 [0.832, 1.000] | **4 / 4 (100.0%)** | **0.11 ms** |
| | **Hybrid RRF** ($k=60, \tau=0.65$) | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.864 [0.727, 1.000] | 0.899 [0.799, 1.000] | **4 / 4 (100.0%)** | 862.4 ms |

### 6.2 Sample Size Limitation Notice
- With $N=4$ unanswerable questions in the held-out test split, the exact 95% Clopper-Pearson binomial confidence interval for $100\%$ CAR (4/4) is $[0.3976, 1.0000]$.
- While 100% of out-of-domain queries were rejected, this small sample limits statistical power. These results prove that the dual-threshold mechanism functions correctly in code, but larger benchmark sets ($N \ge 100$) are necessary for clinical or mission-critical safety claims.

---

## 7. Qualitative Answer-Quality Pilot Review ($N=6$)

Six representative questions covering factual lookup, mathematical derivation, troubleshooting, and out-of-domain abstention were audited for end-to-end generation quality.

> [!NOTE]
> **Scope Notice**: This 6-question audit is a qualitative diagnostic pilot demonstrating grounded generation, verbatim parameter extraction, and refusal behavior. It is **not** a comprehensive population-scale hallucination benchmark.

```carousel
### Case 1: EVAL-001 (Marine Turbocharger Parameter)
- **Question**: What is the nominal compressor pressure ratio of the marine diesel engine turbocharger?
- **Retrieved Context**: `marine_diesel_turbocharger_manual.md_medium_1` (Section 1: System Overview & Aerodynamics)
- **Generated Answer**: "According to the Marine Diesel Engine Turbocharging System Operational Manual, the compressor operates at a nominal pressure ratio of 3.8:1 with a maximum continuous rotor speed of 18,500 RPM."
- **Citations**: `marine_diesel_turbocharger_manual.md (Section 1, Page 2)`
- **Human Review**: PASS — Exact verbatim extraction of 3.8:1 and 18,500 RPM.
- **Identified Errors**: 0 factual errors, 0 citation hallucinations.
<!-- slide -->
### Case 2: EVAL-009 (ASME Pressure Vessel Formula)
- **Question**: Explain the formula for cylindrical shell minimum thickness under internal pressure in ASME Section VIII Division 1.
- **Retrieved Context**: `pressure_vessel_asme_sec_viii.md_medium_1` (Section 1: MAWP & Cylindrical Shell Thickness)
- **Generated Answer**: "Per ASME Section VIII Division 1 (UG-27), the minimum required thickness of a cylindrical shell under internal design pressure is: $t = \frac{P \cdot R}{S \cdot E - 0.6 \cdot P} + C_a$, where $P$ is internal design pressure, $R$ is inside radius, $S$ is maximum allowable stress from Section II-D, $E$ is joint efficiency factor, and $C_a$ is corrosion allowance."
- **Citations**: `pressure_vessel_asme_sec_viii.md (Section 1, Page 1)`
- **Human Review**: PASS — Perfect LaTeX equation formatting and complete variable definitions.
- **Identified Errors**: 0 factual errors, 0 citation hallucinations.
<!-- slide -->
### Case 3: EVAL-014 (SCADA 4-20mA Loop Saturation)
- **Question**: Why is an analog 4-20mA process loop saturating above 21mA in an industrial SCADA system, and what is the troubleshooting procedure?
- **Retrieved Context**: `industrial_plc_scada_troubleshooting.md_medium_2` (Section 2: Analog Signal Noise & Ground Loops)
- **Generated Answer**: "An analog 4-20mA loop saturating above 21.0 mA indicates either transmitter ADC sensor over-range or ground loop interference from multiple grounding points. Troubleshooting requires measuring loop current with a calibrated multimeter, checking transmitter wiring, and ensuring shields are grounded at the control cabinet only."
- **Citations**: `industrial_plc_scada_troubleshooting.md (Section 2, Page 2)`
- **Human Review**: PASS — Accurately explains ground loop potential differences and step-by-step remediation.
- **Identified Errors**: 0 factual errors, 0 citation hallucinations.
<!-- slide -->
### Case 4: EVAL-019 (Gas Turbine Water Wash Specifications)
- **Question**: What water quality parameters and temperature are required for offline compressor crank washing?
- **Retrieved Context**: `gas_turbine_maintenance_procedures.docx_medium_2` (Section 3: Offline Water Wash Parameters)
- **Generated Answer**: "Offline compressor crank washing on heavy-duty gas turbines requires demineralized water with electrical conductivity below 5.0 $\mu\text{S/cm}$ and total sodium + potassium below 25 ppb. The fluid must be preheated to 60°C to 70°C (140°F to 158°F) to prevent thermal shock and dissolve deposits."
- **Citations**: `gas_turbine_maintenance_procedures.docx (Section 3, Slide 3)`
- **Human Review**: PASS — Retains exact unit standards ($\mu\text{S/cm}$, ppb) and temperature ranges.
- **Identified Errors**: 0 factual errors, 0 citation hallucinations.
<!-- slide -->
### Case 5: EVAL-023 (Heat Exchanger Design Comparison)
- **Question**: Compare the design pressure and allowable pressure drop between the shell side and tube side of the heat exchanger.
- **Retrieved Context**: `heat_exchanger_thermal_rating_spec.xlsx_medium_1` (Sheet: Thermal_Rating, Rows 8-12)
- **Generated Answer**: "Based on the Heat Exchanger Thermal Rating Specification (TEMA Type BEM): Shell Side design pressure is 16.0 bar(g) with allowable pressure drop of 0.65 bar (calculated 0.48 bar). Tube Side design pressure is 25.0 bar(g) with allowable pressure drop of 0.85 bar (calculated 0.62 bar)."
- **Citations**: `heat_exchanger_thermal_rating_spec.xlsx (Sheet: Thermal_Rating, Rows 8-12)`
- **Human Review**: PASS — Accurate tabular contrast between shell side and tube side.
- **Identified Errors**: 0 factual errors, 0 citation hallucinations.
<!-- slide -->
### Case 6: EVAL-028 (Quantum Physics Out-of-Domain Query)
- **Question**: What is the quantum superposition decoherence time in trapped-ion quantum computers?
- **Retrieved Context**: None (Abstained by dual-threshold mechanism; Top cosine score 0.48 < 0.65, Top BM25 score 3.48 < 5.0).
- **Generated Answer**: "I cannot answer this question based on the provided technical documentation. The uploaded documents cover industrial machinery specifications (gas turbines, pumps, chillers, boilers, transformers, and pressure vessels) and do not contain information on quantum superposition or trapped-ion decoherence times."
- **Citations**: None (Correct Abstention).
- **Human Review**: PASS — Zero hallucination; safely declined out-of-domain inquiry.
- **Identified Errors**: 0 factual errors, 0 citation hallucinations.
```

---

## 8. Acceptance Criteria Verification

| Criterion | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :---: |
| **1. Precision@5 Reconciled** | Mathematical proof explaining invariant 0.2000 | Verified single-relevant chunk ceiling $\min(k, R)/k = 1/5$ | **VERIFIED** |
| **2. Relevant Chunk IDs Verified** | No synthetic or virtual IDs in ground truth | Verified against authentic corpus across small, med, large | **VERIFIED** |
| **3. Threshold Tie Documented** | Clarify why $\tau=0.65$ was chosen over $0.70$ | Documented empirical $F_1=1.0000$ tie and safety buffer | **VERIFIED** |
| **4. BM25 Abstention Separated** | Dedicated non-cosine threshold for BM25 | Calibrated $\tau_{\text{BM25}}^* = 5.0$ on Dev, validated on Test | **VERIFIED** |
| **5. Statistical Reporting** | Bootstrap 95% CIs and exact abstention counts | Reported for Dev & Test; documented $N=4$ limitation | **VERIFIED** |
| **6. Answer Pilot Documented** | Verbatim text, citations, and error audit | Documented 6 cases with 0 factual/citation errors | **VERIFIED** |
| **7. Regression Safety** | All existing unit & integration tests pass | Ran complete suite: 41/41 tests passed in 2.61s | **VERIFIED** |
| **8. Zero Invention** | All metrics derived from executed runs | All tables populated directly from benchmark JSON/CSV | **VERIFIED** |

---

## 9. Conclusion & Phase Boundary

Phase 3.1.1 concludes the RAG Evaluation Phase with complete mathematical rigor, full test isolation, transparent confidence reporting, and verified synchronization between vector and lexical indices.

**Phase 4 (Industrial Decision Support / RCA / FMEA) remains paused** until explicitly requested by the user.
