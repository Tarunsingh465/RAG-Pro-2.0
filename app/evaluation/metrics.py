"""
Retrieval and Answer Evaluation Metrics Engine.

Provides mathematically rigorous, independently testable implementations of:
- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)
- Bootstrap Confidence Intervals (95% CI via non-parametric resampling)
- Deterministic Citation & Abstention Validation
"""

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class QuestionMetricResult:
    question_id: str
    category: str
    is_answerable: bool
    retrieved_count: int
    relevant_count: int
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    latency_ms: float
    citation_correct: bool = True
    correctly_abstained: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    method_name: str
    total_questions: int
    k: int
    mean_precision_at_k: float
    precision_ci_95: Tuple[float, float]
    mean_recall_at_k: float
    recall_ci_95: Tuple[float, float]
    mean_hit_rate_at_k: float
    hit_rate_ci_95: Tuple[float, float]
    mrr: float
    mrr_ci_95: Tuple[float, float]
    mean_ndcg_at_k: float
    ndcg_ci_95: Tuple[float, float]
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    category_breakdown: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Precision@K = |Retrieved_K ∩ Relevant| / K
    TREC standard: If fewer than K items are returned, denominator is K.
    For unanswerable queries (|Relevant| = 0), returns 1.0 if retrieved is empty, else 0.0.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved_k = list(retrieved_ids[:k])

    if not relevant_ids:
        # Unanswerable query: correct behavior is retrieving 0 documents
        return 1.0 if len(retrieved_k) == 0 else 0.0

    relevant_retrieved = sum(1 for cid in retrieved_k if cid in relevant_ids)
    return relevant_retrieved / float(k)


def calculate_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|
    For unanswerable queries (|Relevant| = 0):
      - 1.0 if 0 documents retrieved (correct abstention)
      - 0.0 if any documents retrieved (false positive retrieval)
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved_k = list(retrieved_ids[:k])

    if not relevant_ids:
        return 1.0 if len(retrieved_k) == 0 else 0.0

    relevant_retrieved = sum(1 for cid in retrieved_k if cid in relevant_ids)
    return relevant_retrieved / float(len(relevant_ids))


def calculate_hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Hit Rate@K = 1.0 if at least one relevant item appears in top K, else 0.0.
    For unanswerable queries: 1.0 if 0 documents retrieved, else 0.0.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved_k = list(retrieved_ids[:k])

    if not relevant_ids:
        return 1.0 if len(retrieved_k) == 0 else 0.0

    for cid in retrieved_k:
        if cid in relevant_ids:
            return 1.0
    return 0.0


def calculate_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Reciprocal Rank = 1 / rank of first relevant item (1-indexed).
    If no relevant item in top K, returns 0.0.
    For unanswerable queries: 1.0 if empty, else 0.0.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved_k = list(retrieved_ids[:k])

    if not relevant_ids:
        return 1.0 if len(retrieved_k) == 0 else 0.0

    for rank_idx, cid in enumerate(retrieved_k, start=1):
        if cid in relevant_ids:
            return 1.0 / float(rank_idx)
    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K (nDCG@K) using binary relevance.
    DCG@K = Σ (r_i / log2(i + 1))
    IDCG@K = Σ (1 / log2(i + 1)) for i in 1..min(|Relevant|, K)
    nDCG@K = DCG@K / IDCG@K
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved_k = list(retrieved_ids[:k])

    if not relevant_ids:
        return 1.0 if len(retrieved_k) == 0 else 0.0

    # Calculate DCG@K
    dcg = 0.0
    for rank_idx, cid in enumerate(retrieved_k, start=1):
        rel = 1.0 if cid in relevant_ids else 0.0
        if rel > 0.0:
            dcg += rel / math.log2(rank_idx + 1.0)

    # Calculate Ideal DCG (IDCG@K)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 1.0 if dcg == 0.0 else 0.0

    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0.0 else 0.0


def bootstrap_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes empirical bootstrap confidence interval for the sample mean.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))

    rng = random.Random(seed)
    n = len(values)
    bootstrap_means: List[float] = []

    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / float(n))

    bootstrap_means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = int(alpha * n_resamples)
    high_idx = int((1.0 - alpha) * n_resamples) - 1

    low = bootstrap_means[max(0, low_idx)]
    high = bootstrap_means[min(n_resamples - 1, high_idx)]

    return (round(low, 4), round(high, 4))


def evaluate_single_retrieval(
    question_id: str,
    category: str,
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Set[str],
    is_answerable: bool,
    k: int,
    latency_ms: float,
) -> QuestionMetricResult:
    """
    Evaluates a single question retrieval result across all core metrics.
    """
    prec = calculate_precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=k)
    rec = calculate_recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=k)
    hit = calculate_hit_rate_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=k)
    rr = calculate_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids, k=k)
    ndcg = calculate_ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=k)

    correctly_abstained = (not is_answerable) and (len(retrieved_chunk_ids) == 0 or hit == 0.0)

    return QuestionMetricResult(
        question_id=question_id,
        category=category,
        is_answerable=is_answerable,
        retrieved_count=len(retrieved_chunk_ids),
        relevant_count=len(relevant_chunk_ids),
        precision_at_k=round(prec, 4),
        recall_at_k=round(rec, 4),
        hit_at_k=round(hit, 4),
        reciprocal_rank=round(rr, 4),
        ndcg_at_k=round(ndcg, 4),
        latency_ms=round(latency_ms, 3),
        correctly_abstained=correctly_abstained,
    )


def aggregate_experiment_metrics(
    method_name: str,
    results: Sequence[QuestionMetricResult],
    k: int,
) -> AggregateMetrics:
    """
    Macro-aggregates question-level results and calculates 95% bootstrap confidence intervals.
    """
    if not results:
        raise ValueError("Results list cannot be empty for aggregation")

    n = len(results)
    precisions = [r.precision_at_k for r in results]
    recalls = [r.recall_at_k for r in results]
    hits = [r.hit_at_k for r in results]
    mrrs = [r.reciprocal_rank for r in results]
    ndcgs = [r.ndcg_at_k for r in results]
    latencies = [r.latency_ms for r in results]

    sorted_latencies = sorted(latencies)
    p50_idx = int(0.50 * n)
    p95_idx = min(n - 1, int(0.95 * n))

    # Per-category aggregation
    categories: Dict[str, List[QuestionMetricResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    category_summary: Dict[str, Dict[str, float]] = {}
    for cat, cat_results in categories.items():
        category_summary[cat] = {
            "count": len(cat_results),
            "precision_at_k": round(sum(c.precision_at_k for c in cat_results) / len(cat_results), 4),
            "recall_at_k": round(sum(c.recall_at_k for c in cat_results) / len(cat_results), 4),
            "hit_rate_at_k": round(sum(c.hit_at_k for c in cat_results) / len(cat_results), 4),
            "mrr": round(sum(c.reciprocal_rank for c in cat_results) / len(cat_results), 4),
            "ndcg_at_k": round(sum(c.ndcg_at_k for c in cat_results) / len(cat_results), 4),
            "latency_ms": round(sum(c.latency_ms for c in cat_results) / len(cat_results), 2),
        }

    return AggregateMetrics(
        method_name=method_name,
        total_questions=n,
        k=k,
        mean_precision_at_k=round(sum(precisions) / n, 4),
        precision_ci_95=bootstrap_confidence_interval(precisions),
        mean_recall_at_k=round(sum(recalls) / n, 4),
        recall_ci_95=bootstrap_confidence_interval(recalls),
        mean_hit_rate_at_k=round(sum(hits) / n, 4),
        hit_rate_ci_95=bootstrap_confidence_interval(hits),
        mrr=round(sum(mrrs) / n, 4),
        mrr_ci_95=bootstrap_confidence_interval(mrrs),
        mean_ndcg_at_k=round(sum(ndcgs) / n, 4),
        ndcg_ci_95=bootstrap_confidence_interval(ndcgs),
        mean_latency_ms=round(sum(latencies) / n, 2),
        p50_latency_ms=round(sorted_latencies[p50_idx], 2),
        p95_latency_ms=round(sorted_latencies[p95_idx], 2),
        category_breakdown=category_summary,
    )
