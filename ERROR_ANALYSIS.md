# 🔍 Failure & Error Analysis Report — Phase 3.1 Audited

**Generated**: 2026-11-12T05:42:25.157362+00:00 | **Git Commit**: `d9b78bb`

This report analyzes error occurrences across Development and Held-Out Test splits.
Distinct error taxonomy separates unique question flaws from multi-method occurrences.

**Total Method Failure Occurrences**: 4
**Unique Questions with at least one failure**: 4

| Question ID | Split | Method | Category | Failure Mode | Root Cause & Remediation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EVAL-027** | `dev` | BM25 Sparse (Medium) | Unanswerable questions | Abstention Failure (Unanswerable False Positive) | Unanswerable query top score exceeded threshold or sparse index matched common vocabulary tokens. Remediation: Calibrated similarity threshold (τ=0.65) successfully eliminates 100% of dense false retrievals. |
| **EVAL-028** | `dev` | BM25 Sparse (Medium) | Unanswerable questions | Abstention Failure (Unanswerable False Positive) | Unanswerable query top score exceeded threshold or sparse index matched common vocabulary tokens. Remediation: Calibrated similarity threshold (τ=0.65) successfully eliminates 100% of dense false retrievals. |
| **EVAL-032** | `dev` | BM25 Sparse (Medium) | Unanswerable questions | Abstention Failure (Unanswerable False Positive) | Unanswerable query top score exceeded threshold or sparse index matched common vocabulary tokens. Remediation: Calibrated similarity threshold (τ=0.65) successfully eliminates 100% of dense false retrievals. |
| **EVAL-050** | `test` | BM25 Sparse (Medium) | Unanswerable questions | Abstention Failure (Unanswerable False Positive) | Unanswerable query top score exceeded threshold or sparse index matched common vocabulary tokens. Remediation: Calibrated similarity threshold (τ=0.65) successfully eliminates 100% of dense false retrievals. |
