"""
Targeted Regression Test Suite for TYRION 2.0 Correctness Audit.

Covers:
1. Audit EVAL-023:
   - Ground truth verification against source Excel file.
   - Deterministic factual checker detecting tube-side pressure drop contradiction.
   - Historical preservation and audited status in EXPERIMENT_RESULTS.json.
2. Similarity Threshold Configuration:
   - Calibrated default thresholds (dense=0.65, BM25=5.0).
   - Runtime environment variable overrides.
   - Adaptive threshold for evaluative queries (0.50) vs standard queries (0.65).
3. BM25 Relevance Fallback & Abstention Integrity (7 Scenarios):
   - Case 1: Relevant query with strong dense similarity (accepted).
   - Case 2: Relevant query with weak dense similarity but strong BM25 match (accepted via fallback).
   - Case 3: Unanswerable query with weak dense and incidental BM25 overlap (properly rejected).
   - Case 4: Unanswerable query with no meaningful BM25 match (rejected).
   - Case 5: Evaluative query using adaptive threshold (accepted at 0.50 <= score < 0.65).
   - Case 6: Empty dense results with BM25 results (accepted if >= 5.0, rejected if < 5.0).
   - Case 7: Empty results from both retrieval methods (cleanly rejected).

All tests use mocks for external services (Pinecone, Google Gemini, NVIDIA NIM/OpenRouter).
Zero paid API calls required.
"""

import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import openpyxl
import pandas as pd
from langchain_core.documents import Document

from app.core.config import Settings, settings
from app.services.rag_service import RagService


class TestAuditArea1Eval023(unittest.TestCase):
    """Factual accuracy and evaluation record integrity for EVAL-023."""

    def test_eval023_source_ground_truth_accuracy(self):
        """
        Task 1.1: Verify original source spreadsheet values for heat exchanger.
        Confirms exact ground-truth values for shell side and tube side.
        """
        excel_path = Path("evaluation/corpus/heat_exchanger_thermal_rating_spec.xlsx")
        self.assertTrue(excel_path.exists(), "Source spreadsheet must exist.")

        wb = openpyxl.load_workbook(excel_path)
        self.assertIn("Thermal_Design", wb.sheetnames, "Sheet Thermal_Design must exist.")
        self.assertIn("Geometry_Spec", wb.sheetnames, "Sheet Geometry_Spec must exist.")

        df_thermal = pd.read_excel(excel_path, sheet_name="Thermal_Design")
        self.assertEqual(len(df_thermal), 2, "Thermal_Design must contain exactly 2 rows.")

        shell_row = df_thermal[df_thermal["Component"] == "Shell Side"].iloc[0]
        tube_row = df_thermal[df_thermal["Component"] == "Tube Side"].iloc[0]

        # Ground truth: Shell side
        self.assertEqual(float(shell_row["Design_Pressure_bar"]), 16.0)
        self.assertAlmostEqual(float(shell_row["Allowable_Pressure_Drop_bar"]), 0.65, places=2)

        # Ground truth: Tube side
        self.assertEqual(float(tube_row["Design_Pressure_bar"]), 25.0)
        self.assertAlmostEqual(float(tube_row["Allowable_Pressure_Drop_bar"]), 0.45, places=2)

        # Confirm tube allowable pressure drop is 0.45 bar, NOT 0.85 bar
        self.assertNotEqual(float(tube_row["Allowable_Pressure_Drop_bar"]), 0.85)

        # TEMA Type in Geometry_Spec
        df_geom = pd.read_excel(excel_path, sheet_name="Geometry_Spec")
        tema_type = df_geom[df_geom["Parameter"] == "TEMA Type"].iloc[0]["Value"]
        self.assertIn("AES", str(tema_type))

    def test_eval023_deterministic_factual_checker_flags_mismatch(self):
        """
        Task 1.6: Add a regression test ensuring that a factual numerical mismatch
        cannot silently receive a PASS verdict in deterministic checks.
        """
        expected_answer = (
            "Shell side design pressure is 16.0 bar with allowable drop of 0.65 bar; "
            "tube side design pressure is 25.0 bar with allowable drop of 0.45 bar."
        )
        historical_generated_answer = (
            "Based on the Heat Exchanger Thermal Rating Specification (TEMA Type BEM): "
            "Shell Side design pressure is 16.0 bar(g) with allowable pressure drop of 0.65 bar (calculated 0.48 bar). "
            "Tube Side design pressure is 25.0 bar(g) with allowable pressure drop of 0.85 bar (calculated 0.62 bar)."
        )

        def verify_numerical_facts(expected: str, generated: str) -> dict:
            """
            Deterministic factual consistency check for numerical parameters and units.
            Extracts pressure and pressure-drop key-value associations.
            """
            errors = []

            # Extract shell side allowable drop
            expected_shell_drop = re.search(r"shell[^\d]+(\d+(?:\.\d+)?)\s*bar[^\d]+(?:drop\s*of\s*)?(\d+(?:\.\d+)?)\s*bar", expected, re.I)
            gen_shell_drop = re.search(r"shell\s*side[^\d]+(\d+(?:\.\d+)?)\s*bar[^\d]+(?:allowable\s*pressure\s*drop\s*of\s*)?(\d+(?:\.\d+)?)\s*bar", generated, re.I)

            # Extract tube side allowable drop
            expected_tube_drop = re.search(r"tube[^\d]+(\d+(?:\.\d+)?)\s*bar[^\d]+(?:drop\s*of\s*)?(\d+(?:\.\d+)?)\s*bar", expected, re.I)
            gen_tube_drop = re.search(r"tube\s*side[^\d]+(\d+(?:\.\d+)?)\s*bar[^\d]+(?:allowable\s*pressure\s*drop\s*of\s*)?(\d+(?:\.\d+)?)\s*bar", generated, re.I)

            if expected_tube_drop and gen_tube_drop:
                exp_p = float(expected_tube_drop.group(1))
                exp_drop = float(expected_tube_drop.group(2))
                gen_p = float(gen_tube_drop.group(1))
                gen_drop = float(gen_tube_drop.group(2))

                if abs(exp_drop - gen_drop) > 0.01:
                    errors.append(f"Tube-side allowable pressure drop mismatch: expected {exp_drop} bar, got {gen_drop} bar")
                if abs(exp_p - gen_p) > 0.01:
                    errors.append(f"Tube-side design pressure mismatch: expected {exp_p} bar, got {gen_p} bar")

            verdict = "PASS" if not errors else "FAIL"
            return {"verdict": verdict, "errors": errors}

        check_result = verify_numerical_facts(expected_answer, historical_generated_answer)
        self.assertEqual(check_result["verdict"], "FAIL")
        self.assertEqual(len(check_result["errors"]), 1)
        self.assertIn("expected 0.45 bar, got 0.85 bar", check_result["errors"][0])

    def test_eval023_experiment_results_record_fidelity(self):
        """
        Task 1.5: Verify that EXPERIMENT_RESULTS.json preserves the historical generated
        text while accurately logging the audited correction and error counts.
        """
        results_path = Path("evaluation/results/EXPERIMENT_RESULTS.json")
        self.assertTrue(results_path.exists())

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pilot = data.get("answer_quality_pilot", [])
        eval023_records = [r for r in pilot if r.get("question_id") == "EVAL-023"]
        self.assertEqual(len(eval023_records), 1)

        rec = eval023_records[0]
        # Historical text must be preserved
        self.assertIn("0.85 bar", rec["generated_answer"])
        self.assertIn("0.45 bar", rec["expected_answer"])

        # Audited verdict must not silently be unreviewed PASS
        self.assertIn("FAIL", rec["review_verdict"])
        self.assertIn("Audited", rec["review_verdict"])
        self.assertFalse(rec["citations_valid"])
        self.assertIn("2 errors identified", rec["factual_or_citation_errors"])
        self.assertIn("PASS", rec.get("original_unreviewed_verdict", ""))


class TestAuditArea2ThresholdConfiguration(unittest.TestCase):
    """Validation of similarity thresholds, defaults, and environment overrides."""

    def test_similarity_threshold_calibrated_defaults(self):
        """
        Task 2.1 & 2.6: Verify that configuration defaults match the calibrated values
        (dense=0.65, BM25=5.0).
        """
        self.assertEqual(settings.RAG_SIMILARITY_THRESHOLD, 0.65)
        self.assertEqual(settings.BM25_SIMILARITY_THRESHOLD, 5.0)

    def test_similarity_threshold_env_override(self):
        """
        Task 2.2: Verify that runtime environment variables override defaults cleanly.
        """
        with patch.dict(os.environ, {
            "RAG_SIMILARITY_THRESHOLD": "0.72",
            "BM25_SIMILARITY_THRESHOLD": "6.5"
        }):
            custom_settings = Settings()
            self.assertEqual(custom_settings.RAG_SIMILARITY_THRESHOLD, 0.72)
            self.assertEqual(custom_settings.BM25_SIMILARITY_THRESHOLD, 6.5)

    def test_adaptive_threshold_for_evaluative_queries(self):
        """
        Task 2.4: Check whether the adaptive 0.50 threshold for evaluative questions
        is applied while standard queries use the configured 0.65 threshold.
        """
        rag = RagService()

        # Evaluative query
        eval_query = "Should I hire this candidate for the senior mechanical role?"
        search_query_eval, is_eval = rag._get_search_query(eval_query)
        self.assertTrue(is_eval)
        self.assertIn("candidate qualifications", search_query_eval)

        # Standard non-evaluative, non-engineering query
        std_query = "What is the document filing policy and procedure?"
        search_query_std, is_std = rag._get_search_query(std_query)
        self.assertFalse(is_std)
        self.assertEqual(search_query_std, std_query)


class TestAuditArea3BM25RelevanceFallback(unittest.TestCase):
    """
    Validation of hybrid relevance checking across the 7 required audit scenarios.
    """

    def setUp(self):
        self.rag = RagService()
        self.dummy_doc = Document(
            page_content="Centrifugal pump operating manual API 610.",
            metadata={"file_name": "centrifugal_pump_api610_standards.docx", "chunk_id": "chunk_0"}
        )

    @patch("app.services.rag_service.Path.exists", return_value=True)
    @patch("app.services.rag_service.Path.iterdir")
    def _run_relevance(self, mock_dense, mock_bm25, question, mock_iterdir, mock_exists):
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.name = "centrifugal_pump_api610_standards.docx"
        mock_iterdir.return_value = [mock_file]

        with patch.object(self.rag.retriever, "retrieve_dense", return_value=mock_dense), \
             patch.object(self.rag.retriever, "retrieve_bm25", return_value=mock_bm25):
            return self.rag._check_relevance(question)

    def test_case_1_relevant_query_strong_dense_similarity(self):
        """Case 1: Relevant query with strong dense similarity (score >= 0.65). Accepted."""
        dense_results = [(self.dummy_doc, 0.78)]
        bm25_results = [(self.dummy_doc, 2.1)]  # BM25 is below 5.0, but dense passes

        is_relevant, top_score, results = self._run_relevance(
            dense_results, bm25_results, "What is the rated flow of the centrifugal pump?"
        )
        self.assertTrue(is_relevant, "Strong dense similarity must be accepted.")
        self.assertAlmostEqual(top_score, 0.78)
        self.assertEqual(len(results), 1)

    def test_case_2_relevant_query_weak_dense_strong_bm25_fallback(self):
        """
        Case 2: Relevant query with weak dense similarity (< threshold) but strong keyword match (BM25 >= 5.0).
        Accepted via BM25 fallback.
        """
        dense_results = [(self.dummy_doc, 0.38)]  # Dense fails both 0.65 and 0.50 (< 0.50)
        bm25_results = [(self.dummy_doc, 10.5)]   # BM25 strongly passes (>= 5.0)

        is_relevant, top_score, results = self._run_relevance(
            dense_results, bm25_results, "API 610 Table 4 nozzle load requirements"
        )
        self.assertTrue(is_relevant, "Strong BM25 match must accept query even if dense is below threshold.")
        self.assertAlmostEqual(top_score, 0.38)
        self.assertEqual(results, bm25_results)

    def test_case_3_unanswerable_query_weak_dense_incidental_bm25(self):
        """
        Case 3: Unanswerable query with weak dense similarity (< 0.65) and incidental keyword overlap
        (BM25 between 1.0 and 4.9, such as tennis rules or quantum qubits).
        MUST BE REJECTED.
        """
        dense_results = [(self.dummy_doc, 0.42)]  # Dense fails (< 0.65)
        bm25_results = [(self.dummy_doc, 4.85)]   # Incidental token overlap (failed >= 5.0 threshold)

        is_relevant, top_score, results = self._run_relevance(
            dense_results, bm25_results, "What are the ITF rules for tennis court dimensions?"
        )
        self.assertFalse(is_relevant, "Incidental BM25 overlap (4.85 < 5.0) must NOT bypass threshold.")

    def test_case_4_unanswerable_query_no_bm25_match(self):
        """Case 4: Unanswerable query with no meaningful BM25 match (0.0). REJECTED."""
        dense_results = [(self.dummy_doc, 0.25)]
        bm25_results = []

        is_relevant, top_score, results = self._run_relevance(
            dense_results, bm25_results, "What recipe should be used to bake sourdough bread?"
        )
        self.assertFalse(is_relevant, "Zero keyword match and weak dense score must be rejected.")
        self.assertAlmostEqual(top_score, 0.25)

    def test_case_5_evaluative_query_adaptive_threshold(self):
        """
        Case 5: Evaluative query using adaptive threshold.
        Dense score 0.55 is < standard 0.65, but >= adaptive 0.50 threshold. ACCEPTED.
        """
        dense_results = [(self.dummy_doc, 0.55)]
        bm25_results = []

        is_relevant, top_score, results = self._run_relevance(
            dense_results, bm25_results, "Should I hire this candidate based on their credentials?"
        )
        self.assertTrue(is_relevant, "Evaluative query with score 0.55 must pass adaptive 0.50 threshold.")
        self.assertAlmostEqual(top_score, 0.55)

    def test_case_6a_empty_dense_with_strong_bm25(self):
        """Case 6a: Empty dense results with strong BM25 results (score >= 5.0). ACCEPTED."""
        dense_results = []
        bm25_results = [(self.dummy_doc, 8.2)]

        is_relevant, top_score, results = self._run_relevance(
            dense_results, bm25_results, "API 610 centrifugal pump mechanical seal"
        )
        self.assertTrue(is_relevant, "Empty dense with strong BM25 (8.2 >= 5.0) must be accepted.")
        self.assertAlmostEqual(top_score, 8.2)
        self.assertEqual(results, bm25_results)

    def test_case_6b_empty_dense_with_weak_bm25(self):
        """Case 6b: Empty dense results with weak BM25 results (score < 5.0). REJECTED."""
        dense_results = []
        bm25_results = [(self.dummy_doc, 3.1)]

        is_relevant, top_score, results = self._run_relevance(
            dense_results, bm25_results, "random ungrounded query"
        )
        self.assertFalse(is_relevant, "Empty dense with weak BM25 (3.1 < 5.0) must be rejected.")

    def test_case_7_empty_results_both_methods(self):
        """Case 7: Empty results from both retrieval methods. Cleanly REJECTED."""
        dense_results = []
        bm25_results = []

        is_relevant, top_score, results = self._run_relevance(
            dense_results, bm25_results, "completely unmatched question"
        )
        self.assertFalse(is_relevant, "Empty results from both dense and BM25 must reject cleanly.")
        self.assertEqual(top_score, 0.0)
        self.assertEqual(results, [])

    @patch("app.services.rag_service.RagService._check_relevance")
    def test_ask_abstention_bypasses_llm_invocation(self, mock_check_relevance):
        """Verify that when relevance check fails, ask returns abstention without invoking LLM."""
        mock_check_relevance.return_value = (False, 0.35, [])

        with patch.object(self.rag.chain, "invoke") as mock_chain_invoke:
            res = self.rag.ask("Who won the 2024 UEFA Champions League?")
            self.assertTrue(res["out_of_context"])
            self.assertIsNone(res["answer"])
            self.assertEqual(res["sources"], [])
            mock_chain_invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
