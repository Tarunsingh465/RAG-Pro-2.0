"""
Comprehensive Unit and Integration Test Suite for Phase 3.1 Audited Evaluation.

Tests:
1. Precision@K (Strictly on answerable queries with real chunk IDs, raises on empty ground truth).
2. Recall@K (Strictly on answerable queries with real chunk IDs, raises on empty ground truth).
3. Hit Rate@K (Binary hit on answerable queries, raises on empty ground truth).
4. Mean Reciprocal Rank (MRR at various ranks, zero hit, raises on empty ground truth).
5. Normalized Discounted Cumulative Gain (nDCG@K).
6. Non-parametric Bootstrap Confidence Intervals (95% CI via 1,000 resamples).
7. Abstention Metrics (Confusion matrix, Correct Abstention Rate, False Retrieval Rate).
8. Granular Latency Breakdown Stages (query prep, embedding, Pinecone, BM25, RRF).
9. Single Retrieval Evaluation (Answerable vs Unanswerable handling).
10. Macro-aggregation across Dev/Test splits with separate retrieval and abstention statistics.
11. Corpus manager chunking strategies ('small', 'medium', 'large') with deterministic IDs.
12. BM25 benchmark execution across separate Dev and Test splits.
"""

import math
import unittest
from pathlib import Path

from langchain_core.documents import Document

from app.evaluation.metrics import (
    AbstentionMetrics,
    AggregateMetrics,
    LatencyBreakdown,
    QuestionMetricResult,
    aggregate_experiment_metrics,
    bootstrap_confidence_interval,
    calculate_abstention_metrics,
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
        retrieved = ["doc1_medium_0", "doc1_medium_1", "doc2_medium_0", "doc2_medium_1", "doc3_medium_0"]
        relevant = {"doc1_medium_1", "doc2_medium_1"}
        prec = calculate_precision_at_k(retrieved, relevant, k=5)
        self.assertAlmostEqual(prec, 0.4, places=4)

    def test_precision_at_k_perfect_and_zero(self):
        retrieved = ["c1", "c2", "c3"]
        all_rel = {"c1", "c2", "c3"}
        no_rel = {"c9", "c10"}
        self.assertAlmostEqual(calculate_precision_at_k(retrieved, all_rel, k=3), 1.0)
        self.assertAlmostEqual(calculate_precision_at_k(retrieved, no_rel, k=3), 0.0)

    def test_precision_at_k_raises_on_empty_relevant(self):
        # Under Phase 3.1 methodology, retrieval metrics must raise on empty relevant_ids
        with self.assertRaises(ValueError):
            calculate_precision_at_k(["c1"], set(), k=5)

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

    def test_recall_at_k_raises_on_empty_relevant(self):
        with self.assertRaises(ValueError):
            calculate_recall_at_k(["c1"], set(), k=5)

    def test_hit_rate_at_k(self):
        retrieved = ["c1", "c2", "c3"]
        self.assertAlmostEqual(calculate_hit_rate_at_k(retrieved, {"c2"}, k=3), 1.0)
        self.assertAlmostEqual(calculate_hit_rate_at_k(retrieved, {"c9"}, k=3), 0.0)

    def test_hit_rate_raises_on_empty_relevant(self):
        with self.assertRaises(ValueError):
            calculate_hit_rate_at_k(["c1"], set(), k=3)

    def test_reciprocal_rank(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        self.assertAlmostEqual(calculate_reciprocal_rank(retrieved, {"c1"}, k=4), 1.0)
        self.assertAlmostEqual(calculate_reciprocal_rank(retrieved, {"c2"}, k=4), 0.5)
        self.assertAlmostEqual(calculate_reciprocal_rank(retrieved, {"c4"}, k=4), 0.25)
        self.assertAlmostEqual(calculate_reciprocal_rank(retrieved, {"c9"}, k=4), 0.0)

    def test_reciprocal_rank_raises_on_empty_relevant(self):
        with self.assertRaises(ValueError):
            calculate_reciprocal_rank(["c1"], set(), k=4)

    def test_ndcg_at_k(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        ideal_ndcg = calculate_ndcg_at_k(retrieved, {"c1", "c2"}, k=5)
        self.assertAlmostEqual(ideal_ndcg, 1.0, places=4)

        sub_ndcg = calculate_ndcg_at_k(retrieved, {"c4", "c5"}, k=5)
        self.assertGreater(sub_ndcg, 0.0)
        self.assertLess(sub_ndcg, 1.0)

        zero_ndcg = calculate_ndcg_at_k(retrieved, {"c99"}, k=5)
        self.assertAlmostEqual(zero_ndcg, 0.0)

    def test_ndcg_raises_on_empty_relevant(self):
        with self.assertRaises(ValueError):
            calculate_ndcg_at_k(["c1"], set(), k=5)

    def test_bootstrap_confidence_interval(self):
        low, high = bootstrap_confidence_interval([1.0, 1.0, 1.0, 1.0], confidence=0.95)
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(high, 1.0)

        vals = [0.2, 0.4, 0.6, 0.8, 1.0]
        mean_val = sum(vals) / len(vals)
        low, high = bootstrap_confidence_interval(vals, confidence=0.95, n_resamples=500, seed=42)
        self.assertLessEqual(low, mean_val)
        self.assertGreaterEqual(high, mean_val)

        self.assertEqual(bootstrap_confidence_interval([]), (0.0, 0.0))

    def test_abstention_metrics(self):
        r_ans_hit = evaluate_single_retrieval(
            "Q1", "Factual", "dev", ["c1"], {"c1"}, True, 1, 10.0, top_score=0.85
        )
        r_ans_miss = evaluate_single_retrieval(
            "Q2", "Factual", "dev", ["c2"], {"c1"}, True, 1, 10.0, top_score=0.80
        )
        r_unans_abstained = evaluate_single_retrieval(
            "Q3", "Unanswerable", "dev", [], set(), False, 1, 10.0, top_score=0.45, threshold=0.65
        )
        r_unans_false_pos = evaluate_single_retrieval(
            "Q4", "Unanswerable", "dev", ["c9"], set(), False, 1, 10.0, top_score=0.72, threshold=0.65
        )

        metrics = calculate_abstention_metrics([r_ans_hit, r_ans_miss, r_unans_abstained, r_unans_false_pos])
        self.assertEqual(metrics.total_unanswerable, 2)
        self.assertEqual(metrics.true_negatives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertAlmostEqual(metrics.correct_abstention_rate, 0.5)
        self.assertAlmostEqual(metrics.false_retrieval_rate, 0.5)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertAlmostEqual(metrics.abstention_accuracy, 0.5)

    def test_latency_breakdown(self):
        lb = LatencyBreakdown(
            query_preprocessing_ms=0.5,
            embedding_generation_ms=620.0,
            pinecone_search_ms=115.0,
            bm25_search_ms=0.1,
            rrf_fusion_ms=0.2,
            total_latency_ms=735.8,
        )
        d = lb.to_dict()
        self.assertEqual(d["embedding_generation_ms"], 620.0)
        self.assertEqual(d["total_latency_ms"], 735.8)

    def test_evaluate_single_retrieval(self):
        breakdown = LatencyBreakdown(query_preprocessing_ms=0.1, bm25_search_ms=0.2, total_latency_ms=0.3)
        res = evaluate_single_retrieval(
            question_id="Q-001",
            category="Factual lookup",
            split="dev",
            retrieved_chunk_ids=["c1", "c2", "c3"],
            relevant_chunk_ids={"c2"},
            is_answerable=True,
            k=3,
            latency_ms=0.3,
            latency_breakdown=breakdown,
            top_score=0.88,
        )
        self.assertEqual(res.question_id, "Q-001")
        self.assertEqual(res.category, "Factual lookup")
        self.assertEqual(res.split, "dev")
        self.assertAlmostEqual(res.hit_at_k, 1.0)
        self.assertAlmostEqual(res.reciprocal_rank, 0.5)
        self.assertEqual(res.retrieved_count, 3)
        self.assertEqual(res.relevant_count, 1)
        self.assertIsNotNone(res.latency_breakdown)

    def test_aggregate_experiment_metrics(self):
        r1 = evaluate_single_retrieval("Q1", "Factual lookup", "dev", ["c1"], {"c1"}, True, 1, 10.0, top_score=0.8)
        r2 = evaluate_single_retrieval("Q2", "Factual lookup", "dev", ["c2"], {"c3"}, True, 1, 20.0, top_score=0.7)
        r3 = evaluate_single_retrieval("Q3", "Unanswerable questions", "dev", [], set(), False, 1, 5.0, top_score=0.4, threshold=0.65)

        agg = aggregate_experiment_metrics("Test Method", [r1, r2, r3], k=1, split="dev")
        self.assertEqual(agg.total_questions, 3)
        self.assertEqual(agg.total_answerable, 2)
        self.assertEqual(agg.total_unanswerable, 1)
        # Precision computed only on answerable: (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(agg.mean_precision_at_k, 0.5)
        self.assertAlmostEqual(agg.mean_hit_rate_at_k, 0.5)
        self.assertAlmostEqual(agg.abstention_metrics.correct_abstention_rate, 1.0)


class TestEvaluationCorpusAndRunner(unittest.TestCase):
    """Integration tests for corpus chunking and experiment execution."""

    @classmethod
    def setUpClass(cls):
        cls.corpus_dir = Path("evaluation/corpus")
        cls.manager = EvaluationCorpusManager(corpus_dir=cls.corpus_dir)

    def test_corpus_exists(self):
        files = list(self.corpus_dir.glob("*.*"))
        self.assertGreaterEqual(len(files), 8, "Expected at least 8 industrial corpus files")

    def test_corpus_chunking_strategies(self):
        chunks_small = self.manager.load_and_chunk(strategy="small")
        chunks_med = self.manager.load_and_chunk(strategy="medium")
        chunks_large = self.manager.load_and_chunk(strategy="large")

        self.assertGreater(len(chunks_small), len(chunks_med))
        self.assertGreaterEqual(len(chunks_med), len(chunks_large))

        for c in chunks_med:
            self.assertIn("chunk_id", c.metadata)
            self.assertIn("_medium_", c.metadata["chunk_id"])

    def test_bm25_dev_and_test_splits(self):
        runner = ExperimentRunner(k=3)
        chunks = runner.corpus_manager.load_and_chunk("medium")

        # Test Dev split execution
        agg_dev, res_dev, _ = runner.run_bm25_experiment(chunks, strategy="medium", split_filter="dev")
        self.assertEqual(agg_dev.split, "dev")
        self.assertEqual(agg_dev.total_questions, 35)
        self.assertEqual(agg_dev.total_answerable, 26)
        self.assertEqual(agg_dev.total_unanswerable, 9)

        # Test Test split execution
        agg_test, res_test, _ = runner.run_bm25_experiment(chunks, strategy="medium", split_filter="test")
        self.assertEqual(agg_test.split, "test")
        self.assertEqual(agg_test.total_questions, 15)
        self.assertEqual(agg_test.total_answerable, 11)
        self.assertEqual(agg_test.total_unanswerable, 4)


if __name__ == "__main__":
    unittest.main()
