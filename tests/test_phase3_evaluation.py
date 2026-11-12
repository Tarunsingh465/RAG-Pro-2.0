"""
Comprehensive Unit and Integration Test Suite for Phase 3 Evaluation.

Tests:
1. Precision@K (standard, unanswerable abstention, boundary checks).
2. Recall@K (standard, partial, unanswerable abstention).
3. Hit Rate@K (binary hit, unanswerable abstention).
4. Mean Reciprocal Rank (MRR at various ranks, zero hit).
5. Normalized Discounted Cumulative Gain (nDCG@K).
6. Non-parametric Bootstrap Confidence Intervals (95% CI).
7. Macro-aggregation across question results and categories.
8. Corpus manager chunking strategies ('small', 'medium', 'large').
9. BM25 benchmark execution and metric generation.
"""

import math
import unittest
from pathlib import Path

from langchain_core.documents import Document

from app.evaluation.metrics import (
    AggregateMetrics,
    QuestionMetricResult,
    aggregate_experiment_metrics,
    bootstrap_confidence_interval,
    calculate_hit_rate_at_k,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    evaluate_single_retrieval,
)
from app.evaluation.runner import EvaluationCorpusManager, ExperimentRunner
from app.retrieval.bm25 import BM25Index


class TestEvaluationMetrics(unittest.TestCase):
    """Unit tests for core information retrieval mathematical formulas."""

    def test_precision_at_k_standard(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = {"c2", "c4"}
        prec = calculate_precision_at_k(retrieved, relevant, k=5)
        self.assertAlmostEqual(prec, 0.4, places=4)

    def test_precision_at_k_perfect_and_zero(self):
        retrieved = ["c1", "c2", "c3"]
        all_rel = {"c1", "c2", "c3"}
        no_rel = {"c9", "c10"}
        self.assertAlmostEqual(calculate_precision_at_k(retrieved, all_rel, k=3), 1.0)
        self.assertAlmostEqual(calculate_precision_at_k(retrieved, no_rel, k=3), 0.0)

    def test_precision_at_k_unanswerable_abstention(self):
        # When ground truth relevant set is empty (unanswerable):
        # Retrieving 0 documents is correct abstention (score = 1.0)
        self.assertAlmostEqual(calculate_precision_at_k([], set(), k=5), 1.0)
        # Retrieving documents for an unanswerable query is a false retrieval (score = 0.0)
        self.assertAlmostEqual(calculate_precision_at_k(["c1"], set(), k=5), 0.0)

    def test_precision_at_k_invalid_k(self):
        with self.assertRaises(ValueError):
            calculate_precision_at_k(["c1"], {"c1"}, k=0)

    def test_recall_at_k_standard(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = {"c2", "c4", "c8", "c9"}  # 4 total relevant in corpus, 2 in top-5
        rec = calculate_recall_at_k(retrieved, relevant, k=5)
        self.assertAlmostEqual(rec, 0.5, places=4)

    def test_recall_at_k_perfect(self):
        retrieved = ["c1", "c2"]
        relevant = {"c1", "c2"}
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, relevant, k=5), 1.0)

    def test_recall_at_k_unanswerable_abstention(self):
        self.assertAlmostEqual(calculate_recall_at_k([], set(), k=5), 1.0)
        self.assertAlmostEqual(calculate_recall_at_k(["c1"], set(), k=5), 0.0)

    def test_hit_rate_at_k(self):
        retrieved = ["c1", "c2", "c3"]
        self.assertAlmostEqual(calculate_hit_rate_at_k(retrieved, {"c2"}, k=3), 1.0)
        self.assertAlmostEqual(calculate_hit_rate_at_k(retrieved, {"c9"}, k=3), 0.0)
        # Unanswerable
        self.assertAlmostEqual(calculate_hit_rate_at_k([], set(), k=3), 1.0)
        self.assertAlmostEqual(calculate_hit_rate_at_k(["c1"], set(), k=3), 0.0)

    def test_reciprocal_rank(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        # Rank 1 hit
        self.assertAlmostEqual(calculate_reciprocal_rank(retrieved, {"c1"}, k=4), 1.0)
        # Rank 2 hit
        self.assertAlmostEqual(calculate_reciprocal_rank(retrieved, {"c2"}, k=4), 0.5)
        # Rank 4 hit
        self.assertAlmostEqual(calculate_reciprocal_rank(retrieved, {"c4"}, k=4), 0.25)
        # Miss
        self.assertAlmostEqual(calculate_reciprocal_rank(retrieved, {"c9"}, k=4), 0.0)

    def test_ndcg_at_k(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        # Ideal: rank 1 and 2 are relevant
        ideal_ndcg = calculate_ndcg_at_k(retrieved, {"c1", "c2"}, k=5)
        self.assertAlmostEqual(ideal_ndcg, 1.0, places=4)

        # Sub-ideal: rank 4 and 5 are relevant
        sub_ndcg = calculate_ndcg_at_k(retrieved, {"c4", "c5"}, k=5)
        self.assertGreater(sub_ndcg, 0.0)
        self.assertLess(sub_ndcg, 1.0)

        # Zero hits
        zero_ndcg = calculate_ndcg_at_k(retrieved, {"c99"}, k=5)
        self.assertAlmostEqual(zero_ndcg, 0.0)

    def test_bootstrap_confidence_interval(self):
        # Uniform data should yield equal low and high
        low, high = bootstrap_confidence_interval([1.0, 1.0, 1.0, 1.0], confidence=0.95)
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(high, 1.0)

        # Varied data
        vals = [0.2, 0.4, 0.6, 0.8, 1.0]
        mean_val = sum(vals) / len(vals)
        low, high = bootstrap_confidence_interval(vals, confidence=0.95, n_resamples=500, seed=42)
        self.assertLessEqual(low, mean_val)
        self.assertGreaterEqual(high, mean_val)

        # Empty data
        self.assertEqual(bootstrap_confidence_interval([]), (0.0, 0.0))

    def test_evaluate_single_retrieval(self):
        res = evaluate_single_retrieval(
            question_id="Q-001",
            category="Factual lookup",
            retrieved_chunk_ids=["c1", "c2", "c3"],
            relevant_chunk_ids={"c2"},
            is_answerable=True,
            k=3,
            latency_ms=12.5,
        )
        self.assertEqual(res.question_id, "Q-001")
        self.assertEqual(res.category, "Factual lookup")
        self.assertAlmostEqual(res.hit_at_k, 1.0)
        self.assertAlmostEqual(res.reciprocal_rank, 0.5)  # rank 2
        self.assertEqual(res.retrieved_count, 3)
        self.assertEqual(res.relevant_count, 1)

    def test_aggregate_experiment_metrics(self):
        r1 = evaluate_single_retrieval("Q1", "Factual lookup", ["c1"], {"c1"}, True, 1, 10.0)
        r2 = evaluate_single_retrieval("Q2", "Factual lookup", ["c2"], {"c3"}, True, 1, 20.0)
        agg = aggregate_experiment_metrics("Test Method", [r1, r2], k=1)

        self.assertEqual(agg.method_name, "Test Method")
        self.assertEqual(agg.total_questions, 2)
        self.assertAlmostEqual(agg.mean_hit_rate_at_k, 0.5)
        self.assertIn("Factual lookup", agg.category_breakdown)
        self.assertEqual(agg.category_breakdown["Factual lookup"]["count"], 2)


class TestEvaluationPipeline(unittest.TestCase):
    """Integration tests for corpus management and evaluation execution."""

    def test_corpus_manager_strategies(self):
        mgr = EvaluationCorpusManager()
        # Test loading strategies
        small_chunks = mgr.load_and_chunk("small")
        medium_chunks = mgr.load_and_chunk("medium")
        large_chunks = mgr.load_and_chunk("large")

        self.assertGreater(len(small_chunks), 0)
        self.assertGreater(len(medium_chunks), 0)
        self.assertGreater(len(large_chunks), 0)

        # Small chunks should outnumber large chunks
        self.assertGreaterEqual(len(small_chunks), len(large_chunks))

        # Check provenance metadata on chunks
        first = medium_chunks[0]
        self.assertIn("chunk_id", first.metadata)
        self.assertIn("file_name", first.metadata)
        self.assertIn("chunk_strategy", first.metadata)
        self.assertEqual(first.metadata["chunk_size_chars"], 1000)

    def test_bm25_benchmark_runner(self):
        runner = ExperimentRunner()
        self.assertEqual(len(runner.questions), 50)

        chunks = runner.corpus_manager.load_and_chunk("medium")
        gt_counts = runner.precompute_ground_truth_counts(chunks)
        self.assertEqual(len(gt_counts), 50)

        # Run BM25 experiment on dev split (35 questions)
        agg, q_res, fails = runner.run_bm25_experiment(chunks, split_filter="dev", gt_counts=gt_counts)
        self.assertEqual(agg.total_questions, 35)
        self.assertGreater(agg.mean_hit_rate_at_k, 0.70)
        self.assertGreater(agg.mrr, 0.70)
        self.assertLess(agg.mean_latency_ms, 5.0)  # BM25 is sub-millisecond local CPU


if __name__ == "__main__":
    unittest.main()
