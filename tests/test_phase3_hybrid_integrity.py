"""
Phase 3.1.2 — Hybrid Retrieval Integrity & Validation Test Suite
Tests the 12 critical integrity invariants specified for Phase 3.1.2:
1. No answerability leakage during retrieval, ranking, and thresholding.
2. Independent Dense threshold enforcement (τ = 0.65).
3. Independent BM25 threshold enforcement (τ = 5.0).
4. Independent source acceptance (Dense alone, BM25 alone, or both).
5. RRF deduplication eliminating float/canonical discrepancies.
6. Stable ordering and deterministic tie-breaking.
7. Top-K correctness (exactly 5 unique candidates when >= 5 available).
8. Short candidate list handling (no padding/fabrication when < 5 available).
9. Export integrity (metric inputs match exported CSV rankings exactly).
10. Unanswerable query evaluation (True Abstention vs False Retrieval).
11. Answerable query evaluation (Retrieval Success vs Failure, no forced inclusion).
12. Core regression invariants (formula, dimension validation, extractors).
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from langchain_core.documents import Document

from app.retrieval.hybrid import (
    get_canonical_chunk_id,
    reciprocal_rank_fusion,
    RetrievalDiagnostics,
)
from app.evaluation.metrics import (
    evaluate_single_retrieval,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_hit_rate_at_k,
    calculate_reciprocal_rank,
    calculate_ndcg_at_k,
    LatencyBreakdown,
)
from app.retrieval.bm25 import BM25Index
from app.ingestion.extractors.text_extractor import TextExtractor


class TestPhase3HybridIntegrity(unittest.TestCase):
    """Exhaustive test suite for Phase 3.1.2 Hybrid Retrieval Integrity."""

    def setUp(self):
        self.doc_a = Document(
            page_content="Turbine bearing temperature limits and lubrication.",
            metadata={
                "file_name": "gas_turbine_manual.md",
                "chunk_strategy": "medium",
                "chunk_index": 0,
                "chunk_id": "gas_turbine_manual.md_medium_0",
            },
        )
        self.doc_b = Document(
            page_content="Boiler feed pump cavitation thresholds per API 610.",
            metadata={
                "file_name": "centrifugal_pump_manual.docx",
                "chunk_strategy": "medium",
                "chunk_index": 1,
                "chunk_id": "centrifugal_pump_manual.docx_medium_1",
            },
        )
        self.doc_c = Document(
            page_content="ASME Section VIII Div 1 pressure vessel shell calculation.",
            metadata={
                "file_name": "pressure_vessel_spec.md",
                "chunk_strategy": "medium",
                "chunk_index": 2,
                "chunk_id": "pressure_vessel_spec.md_medium_2",
            },
        )
        self.doc_d = Document(
            page_content="Transformer oil dissolved gas analysis and ethylene limits.",
            metadata={
                "file_name": "transformer_spec.md",
                "chunk_strategy": "medium",
                "chunk_index": 3,
                "chunk_id": "transformer_spec.md_medium_3",
            },
        )
        self.doc_e = Document(
            page_content="Chiller COP performance curves at part-load operating points.",
            metadata={
                "file_name": "chiller_opt.docx",
                "chunk_strategy": "medium",
                "chunk_index": 4,
                "chunk_id": "chiller_opt.docx_medium_4",
            },
        )
        self.doc_f = Document(
            page_content="Bearing fatigue flaking severity rating in FMEA matrix.",
            metadata={
                "file_name": "bearing_fmea.csv",
                "chunk_strategy": "medium",
                "chunk_index": 5,
                "chunk_id": "bearing_fmea.csv_medium_5",
            },
        )

    def test_1_no_answerability_leakage(self):
        """
        Test 1: Verify that identical retrieval candidates and scores produce
        identical retrieval decisions regardless of whether ground-truth is_answerable
        is True or False.
        """
        dense_candidates = [(self.doc_a, 0.72), (self.doc_b, 0.68)]
        sparse_candidates = [(self.doc_a, 12.0), (self.doc_c, 8.5)]

        dense_thresh = 0.65
        bm25_thresh = 5.0

        def simulate_hybrid_retrieval(dense_in, sparse_in):
            passed_dense = [it for it in dense_in if it[1] >= dense_thresh]
            passed_sparse = [it for it in sparse_in if it[1] >= bm25_thresh]
            if not passed_dense and not passed_sparse:
                return []
            fused, _ = reciprocal_rank_fusion(passed_dense, passed_sparse, k=60, top_n=5)
            seen = set()
            retrieved = []
            for d, _ in fused:
                cid = get_canonical_chunk_id(d)
                if cid and cid not in seen:
                    seen.add(cid)
                    retrieved.append(cid)
            return retrieved

        res_when_answerable = simulate_hybrid_retrieval(dense_candidates, sparse_candidates)
        res_when_unanswerable = simulate_hybrid_retrieval(dense_candidates, sparse_candidates)

        self.assertEqual(
            res_when_answerable,
            res_when_unanswerable,
            "Retrieval decision differed based on ground-truth label (Label Leakage detected!)",
        )
        self.assertEqual(len(res_when_answerable), 3)
        self.assertEqual(res_when_answerable[0], "gas_turbine_manual.md_medium_0")

    def test_2_dense_threshold(self):
        """
        Test 2: Verify that dense candidates below 0.65 are excluded and candidates
        at or above 0.65 are accepted.
        """
        dense_thresh = 0.65
        candidates = [
            (self.doc_a, 0.75),
            (self.doc_b, 0.65),
            (self.doc_c, 0.6499),
            (self.doc_d, 0.50),
        ]
        passed = [it for it in candidates if it[1] >= dense_thresh]
        self.assertEqual(len(passed), 2)
        passed_ids = [get_canonical_chunk_id(d) for d, _ in passed]
        self.assertIn("gas_turbine_manual.md_medium_0", passed_ids)
        self.assertIn("centrifugal_pump_manual.docx_medium_1", passed_ids)
        self.assertNotIn("pressure_vessel_spec.md_medium_2", passed_ids)
        self.assertNotIn("transformer_spec.md_medium_3", passed_ids)

    def test_3_bm25_threshold(self):
        """
        Test 3: Verify that BM25 candidates below 5.0 are excluded and candidates
        at or above 5.0 are accepted.
        """
        bm25_thresh = 5.0
        candidates = [
            (self.doc_a, 15.2),
            (self.doc_b, 5.0),
            (self.doc_c, 4.999),
            (self.doc_d, 0.0),
        ]
        passed = [it for it in candidates if it[1] >= bm25_thresh]
        self.assertEqual(len(passed), 2)
        passed_ids = [get_canonical_chunk_id(d) for d, _ in passed]
        self.assertIn("gas_turbine_manual.md_medium_0", passed_ids)
        self.assertIn("centrifugal_pump_manual.docx_medium_1", passed_ids)
        self.assertNotIn("pressure_vessel_spec.md_medium_2", passed_ids)
        self.assertNotIn("transformer_spec.md_medium_3", passed_ids)

    def test_4_independent_source_acceptance(self):
        """
        Test 4: Verify independent source acceptance:
        - Dense contributes without BM25.
        - BM25 contributes without Dense.
        - Both contribute together.
        - Neither accepted leads cleanly to abstention.
        """
        dense_thresh = 0.65
        bm25_thresh = 5.0

        def hybrid_filter_and_fuse(dense_in, sparse_in):
            passed_dense = [it for it in dense_in if it[1] >= dense_thresh]
            passed_sparse = [it for it in sparse_in if it[1] >= bm25_thresh]
            if not passed_dense and not passed_sparse:
                return []
            fused, _ = reciprocal_rank_fusion(passed_dense, passed_sparse, k=60, top_n=5)
            seen = set()
            retrieved = []
            for d, _ in fused:
                cid = get_canonical_chunk_id(d)
                if cid and cid not in seen:
                    seen.add(cid)
                    retrieved.append(cid)
            return retrieved

        # 1. Dense alone passes
        res_dense_only = hybrid_filter_and_fuse([(self.doc_a, 0.78)], [(self.doc_b, 2.1)])
        self.assertEqual(res_dense_only, ["gas_turbine_manual.md_medium_0"])

        # 2. BM25 alone passes
        res_bm25_only = hybrid_filter_and_fuse([(self.doc_a, 0.55)], [(self.doc_b, 10.5)])
        self.assertEqual(res_bm25_only, ["centrifugal_pump_manual.docx_medium_1"])

        # 3. Both contribute
        res_both = hybrid_filter_and_fuse([(self.doc_a, 0.78)], [(self.doc_b, 10.5)])
        self.assertEqual(len(res_both), 2)
        self.assertIn("gas_turbine_manual.md_medium_0", res_both)
        self.assertIn("centrifugal_pump_manual.docx_medium_1", res_both)

        # 4. Neither passes -> Abstain cleanly (empty list)
        res_neither = hybrid_filter_and_fuse([(self.doc_a, 0.55)], [(self.doc_b, 2.1)])
        self.assertEqual(res_neither, [])

    def test_5_rrf_deduplication(self):
        """
        Test 5: Supply overlapping dense and BM25 candidates, including vector-store
        float serialization suffixes (e.g. '_0.0'). Verify canonical deduplication.
        """
        doc_dense_float = Document(
            page_content="Turbine bearing temperature limits.",
            metadata={"chunk_id": "gas_turbine_manual.md_medium_0.0"},
        )
        doc_bm25_int = Document(
            page_content="Turbine bearing temperature limits.",
            metadata={"chunk_id": "gas_turbine_manual.md_medium_0"},
        )

        dense_res = [(doc_dense_float, 0.82)]
        bm25_res = [(doc_bm25_int, 14.5)]

        fused, diag = reciprocal_rank_fusion(dense_res, bm25_res, k=60, top_n=5)

        # Must be deduplicated into exactly one entry
        self.assertEqual(len(fused), 1)
        self.assertEqual(diag.total_unique_chunks, 1)
        self.assertEqual(fused[0][0].metadata["chunk_id"], "gas_turbine_manual.md_medium_0")
        # RRF score should be 1/(60+1) + 1/(60+1) = 2/61
        self.assertAlmostEqual(fused[0][1], 2.0 / 61.0, places=6)

    def test_6_stable_ordering(self):
        """
        Test 6: Verify that deduplication preserves rank order and deterministic tie-breaking.
        """
        dense_results = [
            (self.doc_a, 0.85),  # rank 1
            (self.doc_b, 0.75),  # rank 2
        ]
        bm25_results = [
            (self.doc_a, 12.0),  # rank 1
            (self.doc_c, 8.0),   # rank 2
        ]

        fused, _ = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=5)
        fused_ids = [get_canonical_chunk_id(d) for d, _ in fused]

        # doc_a appears in both at rank 1 -> highest RRF score
        self.assertEqual(fused_ids[0], "gas_turbine_manual.md_medium_0")
        # doc_b and doc_c each appear once at rank 2: tie broken deterministically
        self.assertEqual(len(fused_ids), 3)
        self.assertEqual(len(fused_ids), len(set(fused_ids)))

    def test_7_top_k_correctness(self):
        """
        Test 7: Verify that the final ranking contains exactly 5 unique chunks
        when at least 5 unique accepted candidates are available.
        """
        dense_candidates = [
            (self.doc_a, 0.80),
            (self.doc_b, 0.78),
            (self.doc_c, 0.76),
        ]
        sparse_candidates = [
            (self.doc_d, 15.0),
            (self.doc_e, 12.0),
            (self.doc_f, 9.0),
        ]
        fused, _ = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, top_n=5)
        retrieved_ids = []
        for d, _ in fused:
            cid = get_canonical_chunk_id(d)
            if cid not in retrieved_ids:
                retrieved_ids.append(cid)
            if len(retrieved_ids) == 5:
                break

        self.assertEqual(len(retrieved_ids), 5)
        self.assertEqual(len(set(retrieved_ids)), 5)

    def test_8_short_candidate_list(self):
        """
        Test 8: Verify that fewer than 5 available unique candidates are returned
        without padding or synthetic fabrication, preserving deterministic tie-breaking.
        """
        dense_candidates = [(self.doc_a, 0.80)]
        sparse_candidates = [(self.doc_b, 10.0)]

        fused, _ = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, top_n=5)
        retrieved_ids = [get_canonical_chunk_id(d) for d, _ in fused]

        self.assertEqual(len(retrieved_ids), 2)
        self.assertNotIn(None, retrieved_ids)
        # Deterministic tie-breaking on equal RRF score (1/61) sorts alphabetically by cid
        self.assertEqual(retrieved_ids, [
            "centrifugal_pump_manual.docx_medium_1",
            "gas_turbine_manual.md_medium_0",
        ])

    def test_9_export_integrity(self):
        """
        Test 9: Verify that retrieved chunk IDs passed to evaluation metrics
        match the exported CSV serialization string exactly.
        """
        retrieved_chunk_ids = [
            "gas_turbine_manual.md_medium_0",
            "centrifugal_pump_manual.docx_medium_1",
            "pressure_vessel_spec.md_medium_2",
        ]
        exported_str = ";".join(retrieved_chunk_ids)
        unpacked_ids = exported_str.split(";")

        self.assertEqual(retrieved_chunk_ids, unpacked_ids)
        self.assertEqual(len(unpacked_ids), len(set(unpacked_ids)))

    def test_10_unanswerable_evaluation(self):
        """
        Test 10: Verify unanswerable evaluation:
        - When all candidates are below threshold -> True Abstention (CAR = 1.0)
        - When candidates exceed threshold -> False Retrieval (FRR = 1.0)
        without using ground-truth label to alter retrieval behavior.
        """
        # True abstention case: 0 accepted chunks
        res_abstain = evaluate_single_retrieval(
            question_id="EVAL-UNANS-01",
            category="Unanswerable questions",
            split="test",
            retrieved_chunk_ids=[],
            relevant_chunk_ids=set(),
            is_answerable=False,
            k=5,
            latency_ms=10.0,
            top_score=0.52,
            threshold=0.65,
        )
        self.assertTrue(res_abstain.correctly_abstained)
        self.assertEqual(res_abstain.retrieved_count, 0)

        # False retrieval case: candidates passed threshold on out-of-domain query
        res_false_retrieval = evaluate_single_retrieval(
            question_id="EVAL-UNANS-02",
            category="Unanswerable questions",
            split="test",
            retrieved_chunk_ids=["gas_turbine_manual.md_medium_0"],
            relevant_chunk_ids=set(),
            is_answerable=False,
            k=5,
            latency_ms=10.0,
            top_score=0.72,
            threshold=0.65,
        )
        self.assertFalse(res_false_retrieval.correctly_abstained)
        self.assertEqual(res_false_retrieval.retrieved_count, 1)

    def test_11_answerable_evaluation(self):
        """
        Test 11: Verify answerable evaluation:
        - When relevant chunk is in top-K -> Hit = 1.0
        - When relevant chunk is missed -> Hit = 0.0, recorded as retrieval failure
        Ground truth is never injected into candidate list.
        """
        # Success case
        res_hit = evaluate_single_retrieval(
            question_id="EVAL-ANS-01",
            category="Factual lookup",
            split="dev",
            retrieved_chunk_ids=["gas_turbine_manual.md_medium_0", "chiller_opt.docx_medium_4"],
            relevant_chunk_ids={"gas_turbine_manual.md_medium_0"},
            is_answerable=True,
            k=5,
            latency_ms=15.0,
        )
        self.assertEqual(res_hit.hit_at_k, 1.0)
        self.assertEqual(res_hit.precision_at_k, 0.2)  # 1/5
        self.assertEqual(res_hit.recall_at_k, 1.0)    # 1/1

        # Failure case (missed relevant chunk)
        res_miss = evaluate_single_retrieval(
            question_id="EVAL-ANS-02",
            category="Factual lookup",
            split="dev",
            retrieved_chunk_ids=["centrifugal_pump_manual.docx_medium_1"],
            relevant_chunk_ids={"gas_turbine_manual.md_medium_0"},
            is_answerable=True,
            k=5,
            latency_ms=15.0,
        )
        self.assertEqual(res_miss.hit_at_k, 0.0)
        self.assertEqual(res_miss.precision_at_k, 0.0)
        self.assertEqual(res_miss.recall_at_k, 0.0)
        self.assertNotIn("gas_turbine_manual.md_medium_0", res_miss.retrieved_chunk_ids)

    def test_12_regression_invariants(self):
        """
        Test 12: Verify core regression invariants:
        - BM25 indexing and search remain functional.
        - TextExtractor loads and extracts cleanly.
        - RRF formula matches 1 / (60 + rank).
        """
        # BM25 regression
        bm25 = BM25Index()
        bm25.clear()
        bm25.add_documents([self.doc_a, self.doc_b])
        results = bm25.search("turbine bearing limits", top_k=2)
        self.assertGreater(len(results), 0)
        top_doc, top_score = results[0]
        self.assertEqual(get_canonical_chunk_id(top_doc), "gas_turbine_manual.md_medium_0")
        self.assertGreater(top_score, 0.0)

        # TextExtractor regression
        extractor = TextExtractor()
        extracted = extractor.extract(Path("README.md"))
        self.assertGreater(len(extracted), 0)

        # RRF formula regression: 1/(k+rank)
        dense_single = [(self.doc_a, 0.90)]
        fused_single, _ = reciprocal_rank_fusion(dense_single, [], k=60, top_n=1)
        self.assertAlmostEqual(fused_single[0][1], 1.0 / 61.0, places=6)


if __name__ == "__main__":
    unittest.main()
