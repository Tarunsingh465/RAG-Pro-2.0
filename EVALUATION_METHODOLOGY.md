# 📐 Evaluation Methodology & Experimental Design Specification

**Project**: RAG-Pro 2.0 — Industrial Knowledge Intelligence & Decision Support System  
**Developer**: Tarun Singh — B.Tech Computer Science (Data Science & AI)  
**Standard**: TREC / BeIR Information Retrieval Benchmarking Protocol  
**Version**: Phase 3.1 Audited Edition | **Date**: November 2026

---

## 1. Corpus Provenance and Design

The evaluation corpus comprises **10 authentic industrial and engineering technical documents** spanning multiple engineering disciplines:

| File Name | Engineering Domain | Primary Equipment / Subject | Structure & Modality |
| :--- | :--- | :--- | :--- |
| `marine_diesel_turbocharger_manual.md` | Marine / Mechanical | Aerodynamics, Surge Limits, Floating Bearings | Markdown (Headings, Bulleted specs) |
| `centrifugal_pump_api610_standards.docx` | Turbomachinery / Chemical | API 610 NPSH criteria, Seal Flush Plans (11, 32, 53A) | Word DOCX (Tables, Sections) |
| `gas_turbine_maintenance_procedures.docx` | Power Generation | Hot Gas Path inspection, Compressor Water Washing | Word DOCX (Procedure steps, Limits) |
| `heat_exchanger_thermal_rating_spec.xlsx` | Process Heat Transfer | TEMA AES Floating Head, Fouling, Rating Tables | Excel Multi-sheet (Thermal_Design, Geometry_Spec) |
| `hvac_chiller_energy_optimization.docx` | Facility & Energy | COP, IPLV AHRI 550/590, Condenser Approach Temp | Word DOCX (Formulas, Guidelines) |
| `industrial_plc_scada_troubleshooting.md` | Automation / Instrumentation | Modbus/TCP Port 502, 4-20mA Ground Loop Faults | Markdown (Diagnostic trees) |
| `pressure_vessel_asme_sec_viii.md` | Mechanical / Pressure Safety | ASME BPVC Section VIII Div 1 MAWP, UG-99 Hydrotest | Markdown (Equations, Joint efficiency E) |
| `steam_boiler_feedwater_chemistry.xlsx` | Chemical / Utility | Dissolved O2, Silica limits, Phosphate treatment | Excel (Water_Limits table across pressure bands) |
| `substation_transformer_oil_analysis.md` | Electrical Power Systems | IEEE C57.104 DGA Fault Gases (C2H2, C2H4, H2), ASTM D877 | Markdown (Concentration limits, Sparkover kV) |
| `bearing_failure_fmea_and_capa.csv` | Reliability / Maintenance | FMEA Matrix: Electrical Fluting, False Brinelling, Starvation | CSV Tabular (Severity, Occurrence, Detection, RPN, CAPA) |

---

## 2. Dataset Design & Ground-Truth Annotation Protocol

The evaluation dataset (`evaluation/datasets/industrial_eval_questions.jsonl`) contains **50 rigorously reviewed questions**:

### 2.1 Split Partitioning
- **Development Split ($N=35$)**: Used for similarity threshold calibration, parameter exploration, and chunk-size sensitivity analysis.
  - 26 Answerable queries across 5 categories.
  - 9 Unanswerable queries (out-of-domain distractors).
- **Held-Out Test Split ($N=15$)**: Reserved strictly for final validation with frozen hyperparameters to measure generalization without data leakage.
  - 11 Answerable queries across 5 categories.
  - 4 Unanswerable queries.

### 2.2 Question Taxonomy & Categories
1. **Factual Lookup**: Specific numbers, tolerances, formulas, or standard codes (e.g. 18,500 RPM, port 502, 1.3 multiplier).
2. **Technical Explanation**: Engineering causality, fluid mechanics, or electrical phenomenon (e.g. compressor surge aerodynamics, DGA acetylene arcing).
3. **Troubleshooting**: Diagnostic symptoms, failure modes, and immediate corrective procedures (e.g. 4-20mA signal saturation, Modbus timeout remediation).
4. **Maintenance**: Operational inspection intervals, fluid cleanliness ratings, and lubrication protocols (e.g. 24,000 FFH, 10-micron filter).
5. **Comparison**: Multi-faceted technical trade-offs (e.g. Shell vs Tube side design pressure, API Plan 11 vs Plan 32, FMEA RPN ranking).
6. **Unanswerable Questions**: Questions intentionally placed outside the technical knowledge base (e.g. Carbon-14 half-life, Titanic box office, sourdough bread recipe) to rigorously test hallucination avoidance and abstention.

### 2.3 Annotation Fields
Every entry in the evaluation dataset contains:
- `question_id`: Unique identifier (`EVAL-001` through `EVAL-050`).
- `relevant_document_ids`: List of ground-truth source files (empty for unanswerable).
- `relevant_evidence_span`: Verbatim quotation extracted directly from the corpus text.
- `relevant_section`: Document section, sheet, or slide title.
- `ground_truth_chunks_medium`: Explicit list of chunk IDs under 1,000-char chunking.
- `ground_truth_chunks_small`: Explicit list of chunk IDs under 500-char chunking.
- `ground_truth_chunks_large`: Explicit list of chunk IDs under 1,500-char chunking.
- `human_review_status`: `"verified"` indicating manual engineering review.

---

## 3. Mathematical Metric Formulations

### 3.1 Answerable Retrieval Metrics (Top-K)

Let $\mathcal{R}_K$ be the ordered list of unique retrieved chunk IDs at cutoff $K$, and $\mathcal{G}$ be the set of verified relevant chunk IDs for question $q$.

$$\text{Precision@K} = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{K}$$

$$\text{Recall@K} = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{|\mathcal{G}|}$$

$$\text{Hit Rate@K} = \begin{cases} 1.0 & \text{if } |\mathcal{R}_K \cap \mathcal{G}| \ge 1 \\ 0.0 & \text{otherwise} \end{cases}$$

$$\text{Reciprocal Rank (RR)} = \begin{cases} \frac{1}{\text{rank}_1} & \text{where } \text{rank}_1 \text{ is the 1-indexed rank of the first relevant chunk in } \mathcal{R}_K \\ 0.0 & \text{if } |\mathcal{R}_K \cap \mathcal{G}| = 0 \end{cases}$$

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\mathbb{I}(\mathcal{R}_K[i] \in \mathcal{G})}{\log_2(i + 1)}, \quad \text{IDCG@K} = \sum_{i=1}^{\min(|\mathcal{G}|, K)} \frac{1}{\log_2(i + 1)}$$

$$\text{nDCG@K} = \begin{cases} \frac{\text{DCG@K}}{\text{IDCG@K}} & \text{if } \text{IDCG@K} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

> [!NOTE]
> Retrieval metrics are evaluated **exclusively on answerable queries** ($|\mathcal{G}| > 0$). If an answerable query has an empty ground truth set, the evaluation framework raises a validation error rather than calculating a score.

### 3.2 Unanswerable Abstention Metrics

For unanswerable queries ($|\mathcal{G}| = 0$):
- **True Negative (TN)**: Query correctly abstains (top similarity score $< \tau$, or zero chunks retrieved).
- **False Positive (FP)**: Query fails to abstain and returns irrelevant chunks (top similarity score $\ge \tau$).

$$\text{Correct Abstention Rate (CAR)} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

$$\text{False Retrieval Rate (FRR)} = \frac{\text{FP}}{\text{TN} + \text{FP}} = 1.0 - \text{CAR}$$

$$\text{Abstention Accuracy} = \frac{\text{TN} + \text{TP}}{\text{Total Questions}}$$

---

## 4. Similarity Threshold Calibration Protocol

For dense vector retrieval using cosine similarity $s = 1 - \text{cosine\_distance}$:
1. Evaluate all queries in the Development Split ($N=35$) across candidate thresholds:
   $$\tau \in \{0.50, 0.55, 0.60, 0.65, 0.70, 0.75\}$$
2. For each candidate threshold $\tau$:
   - Calculate Answerable Hit Rate ($H(\tau)$).
   - Calculate Unanswerable Correct Abstention Rate ($\text{CAR}(\tau)$).
   - Compute Balanced F1-Score:
     $$F_1(\tau) = \frac{2 \cdot H(\tau) \cdot \text{CAR}(\tau)}{H(\tau) + \text{CAR}(\tau)}$$
3. Select $\tau^*$ maximizing $F_1(\tau)$ on the Development Split.
4. Freeze $\tau^*$ and apply it to the Held-Out Test Split.

---

## 5. Granular Latency Profiling Methodology

Total retrieval latency is decomposed into individual measurement stages:
$$T_{\text{total}} = T_{\text{prep}} + T_{\text{emb}} + T_{\text{pinecone}} + T_{\text{bm25}} + T_{\text{rrf}}$$

Where:
- $T_{\text{prep}}$: Local CPU query string sanitization and lowercase tokenization.
- $T_{\text{emb}}$: Remote cloud embedding API call (`text-embedding-004`).
- $T_{\text{pinecone}}$: Remote vector database network search and cosine distance ranking.
- $T_{\text{bm25}}$: In-memory Okapi BM25 inverted index frequency scoring.
- $T_{\text{rrf}}$: Reciprocal Rank Fusion score aggregation ($k=60$) across sorted ordinal ranks.

---

## 6. Non-Parametric Bootstrap Confidence Intervals

To ensure statistical rigor without assuming normal distributions:
- Sample size: $N$ evaluated queries.
- Resampling iterations: $B = 1,000$ with replacement.
- Seed: 42 (deterministic reproducibility).
- Confidence interval: $\alpha = 0.05 \implies 95\%$ two-sided empirical percentile interval $[B_{\alpha/2}, B_{1 - \alpha/2}]$.
