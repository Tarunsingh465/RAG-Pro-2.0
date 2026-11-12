"""
Retrieval and Answer Evaluation Metrics Engine.

Provides mathematically rigorous, independently testable implementations of:
- Precision@K (Strictly on answerable queries)
- Recall@K (Strictly on answerable queries)
- Hit Rate@K (Strictly on answerable queries)
- Mean Reciprocal Rank (MRR) (Strictly on answerable queries)
- Normalized Discounted Cumulative Gain (nDCG@K) (Strictly on answerable queries)
- Abstention Metrics (Correct Abstention Rate, False Retrieval Rate, Confusion Matrix)
- Granular Latency Breakdown Stage Accounting
- Non-parametric Bootstrap Confidence Intervals (95% CI via 1,000 resamples)
"""

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class LatencyBreakdown:
    query_preprocessing_ms: float = 0.0
    embedding_generation_ms: float = 0.0
    pinecone_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AbstentionMetrics:
    total_unanswerable: int
    true_negatives: int  # Unanswerable query correctly abstained (below threshold or 0 retrieved)
    false_positives: int  # Unanswerable query with false retrieval above threshold
    correct_abstention_rate: float  # TN / (TN + FP)
    false_retrieval_rate: float  # FP / (TN + FP)
    total_answerable: int
    true_positives: int  # Answerable query with >= 1 relevant document retrieved
    false_negatives: int  # Answerable query with 0 relevant documents retrieved
    abstention_accuracy: float  # (TN + TP) / (total_unanswerable + total_answerable)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionMetricResult:
    question_id: str
    category: str
    split: str
    is_answerable: bool
    retrieved_chunk_ids: List[str]
    relevant_chunk_ids: List[str]
    retrieved_count: int
    relevant_count: int
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    hit_at_k: Optional[float] = None
    reciprocal_rank: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    latency_ms: float = 0.0
    latency_breakdown: Optional[LatencyBreakdown] = None
    top_score: float = 0.0
    threshold: Optional[float] = None
    correctly_abstained: Optional[bool] = None
    citation_correct: bool = True

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        if self.latency_breakdown:
            res["latency_breakdown"] = self.latency_breakdown.to_dict()
        return res


@dataclass
class AggregateMetrics:
    method_name: str
    split: str
    total_questions: int
    total_answerable: int
    total_unanswerable: int
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
    latency_breakdown: LatencyBreakdown
    abstention_metrics: AbstentionMetrics
    category_breakdown: Dict[str, Dict[str, float]]
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["latency_breakdown"] = self.latency_breakdown.to_dict()
        res["abstention_metrics"] = self.abstention_metrics.to_dict()
        return res


def calculate_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Precision@K = |Unique_Retrieved_K ∩ Relevant| / K
    TREC standard: If fewer than K items are returned, denominator is K.
    Evaluated strictly on answerable queries with non-empty ground truth.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not relevant_ids:
        raise ValueError("calculate_precision_at_k called with empty relevant_ids. Use abstention metrics for unanswerable queries.")

    seen = set()
    unique_retrieved_k = [cid for cid in retrieved_ids[:k] if not (cid in seen or seen.add(cid))]
    relevant_retrieved = sum(1 for cid in unique_retrieved_k if cid in relevant_ids)
    return min(1.0, relevant_retrieved / float(k))


def calculate_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Recall@K = |Unique_Retrieved_K ∩ Relevant| / |Relevant|
    Evaluated strictly on answerable queries with non-empty ground truth.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not relevant_ids:
        raise ValueError("calculate_recall_at_k called with empty relevant_ids. Use abstention metrics for unanswerable queries.")

    seen = set()
    unique_retrieved_k = [cid for cid in retrieved_ids[:k] if not (cid in seen or seen.add(cid))]
    relevant_retrieved = sum(1 for cid in unique_retrieved_k if cid in relevant_ids)
    return min(1.0, relevant_retrieved / float(len(relevant_ids)))


def calculate_hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Hit Rate@K = 1.0 if at least one relevant item appears in top K, else 0.0.
    Evaluated strictly on answerable queries with non-empty ground truth.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not relevant_ids:
        raise ValueError("calculate_hit_rate_at_k called with empty relevant_ids. Use abstention metrics for unanswerable queries.")

    retrieved_k = list(retrieved_ids[:k])
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
    Reciprocal Rank = 1 / rank of first relevant item in top K (1-indexed).
    If no relevant item in top K, returns 0.0.
    Evaluated strictly on answerable queries with non-empty ground truth.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not relevant_ids:
        raise ValueError("calculate_reciprocal_rank called with empty relevant_ids. Use abstention metrics for unanswerable queries.")

    seen = set()
    unique_retrieved_k = [cid for cid in retrieved_ids[:k] if not (cid in seen or seen.add(cid))]
    for rank_idx, cid in enumerate(unique_retrieved_k, start=1):
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
    Evaluated strictly on answerable queries with non-empty ground truth.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not relevant_ids:
        raise ValueError("calculate_ndcg_at_k called with empty relevant_ids. Use abstention metrics for unanswerable queries.")

    seen = set()
    unique_retrieved_k = [cid for cid in retrieved_ids[:k] if not (cid in seen or seen.add(cid))]

    # Calculate DCG@K
    dcg = 0.0
    for rank_idx, cid in enumerate(unique_retrieved_k, start=1):
        rel = 1.0 if cid in relevant_ids else 0.0
        if rel > 0.0:
            dcg += rel / math.log2(rank_idx + 1.0)

    # Calculate Ideal DCG (IDCG@K)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_hits + 1))
    return min(1.0, dcg / idcg) if idcg > 0.0 else 0.0


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
    split: str,
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Set[str],
    is_answerable: bool,
    k: int,
    latency_ms: float,
    latency_breakdown: Optional[LatencyBreakdown] = None,
    top_score: float = 0.0,
    threshold: Optional[float] = None,
) -> QuestionMetricResult:
    """
    Evaluates a single question retrieval result.
    Separates answerable retrieval metrics from unanswerable abstention results.
    """
    # Deduplicate retrieved IDs preserving rank order
    seen = set()
    unique_retrieved = [cid for cid in retrieved_chunk_ids if not (cid in seen or seen.add(cid))]

    if is_answerable:
        if not relevant_chunk_ids:
            raise ValueError(
                f"Question {question_id} is marked answerable but relevant_chunk_ids is empty."
            )
        prec = calculate_precision_at_k(unique_retrieved, relevant_chunk_ids, k=k)
        rec = calculate_recall_at_k(unique_retrieved, relevant_chunk_ids, k=k)
        hit = calculate_hit_rate_at_k(unique_retrieved, relevant_chunk_ids, k=k)
        rr = calculate_reciprocal_rank(unique_retrieved, relevant_chunk_ids, k=k)
        ndcg = calculate_ndcg_at_k(unique_retrieved, relevant_chunk_ids, k=k)

        correctly_abstained = False
        return QuestionMetricResult(
            question_id=question_id,
            category=category,
            split=split,
            is_answerable=True,
            retrieved_chunk_ids=list(unique_retrieved[:k]),
            relevant_chunk_ids=sorted(list(relevant_chunk_ids)),
            retrieved_count=len(unique_retrieved),
            relevant_count=len(relevant_chunk_ids),
            precision_at_k=round(prec, 4),
            recall_at_k=round(rec, 4),
            hit_at_k=round(hit, 4),
            reciprocal_rank=round(rr, 4),
            ndcg_at_k=round(ndcg, 4),
            latency_ms=round(latency_ms, 3),
            latency_breakdown=latency_breakdown,
            top_score=round(top_score, 4),
            threshold=threshold,
            correctly_abstained=correctly_abstained,
        )
    else:
        if threshold is not None:
            correctly_abstained = bool(top_score < threshold or len(unique_retrieved) == 0)
        else:
            correctly_abstained = bool(len(unique_retrieved) == 0)

        return QuestionMetricResult(
            question_id=question_id,
            category=category,
            split=split,
            is_answerable=False,
            retrieved_chunk_ids=list(unique_retrieved[:k]),
            relevant_chunk_ids=[],
            retrieved_count=len(unique_retrieved),
            relevant_count=0,
            precision_at_k=None,
            recall_at_k=None,
            hit_at_k=None,
            reciprocal_rank=None,
            ndcg_at_k=None,
            latency_ms=round(latency_ms, 3),
            latency_breakdown=latency_breakdown,
            top_score=round(top_score, 4),
            threshold=threshold,
            correctly_abstained=correctly_abstained,
        )


def calculate_abstention_metrics(
    results: Sequence[QuestionMetricResult],
) -> AbstentionMetrics:
    """
    Calculates confusion matrix and abstention rates across all questions in result set.
    """
    unanswerable = [r for r in results if not r.is_answerable]
    answerable = [r for r in results if r.is_answerable]

    tot_unans = len(unanswerable)
    tot_ans = len(answerable)

    tn = sum(1 for r in unanswerable if r.correctly_abstained)
    fp = tot_unans - tn

    tp = sum(1 for r in answerable if (r.hit_at_k is not None and r.hit_at_k > 0.0))
    fn = tot_ans - tp

    car = (tn / float(tot_unans)) if tot_unans > 0 else 1.0
    frr = (fp / float(tot_unans)) if tot_unans > 0 else 0.0
    acc = ((tn + tp) / float(tot_unans + tot_ans)) if (tot_unans + tot_ans) > 0 else 0.0

    return AbstentionMetrics(
        total_unanswerable=tot_unans,
        true_negatives=tn,
        false_positives=fp,
        correct_abstention_rate=round(car, 4),
        false_retrieval_rate=round(frr, 4),
        total_answerable=tot_ans,
        true_positives=tp,
        false_negatives=fn,
        abstention_accuracy=round(acc, 4),
    )


def aggregate_experiment_metrics(
    method_name: str,
    results: Sequence[QuestionMetricResult],
    k: int,
    split: str = "all",
    threshold: Optional[float] = None,
) -> AggregateMetrics:
    """
    Macro-aggregates question-level results and calculates 95% bootstrap confidence intervals.
    Separates answerable retrieval metrics from unanswerable abstention metrics.
    """
    if not results:
        raise ValueError("Results list cannot be empty for aggregation")

    n = len(results)
    answerable = [r for r in results if r.is_answerable]
    unanswerable = [r for r in results if not r.is_answerable]

    n_ans = len(answerable)
    n_unans = len(unanswerable)

    if n_ans > 0:
        precisions = [r.precision_at_k for r in answerable if r.precision_at_k is not None]
        recalls = [r.recall_at_k for r in answerable if r.recall_at_k is not None]
        hits = [r.hit_at_k for r in answerable if r.hit_at_k is not None]
        mrrs = [r.reciprocal_rank for r in answerable if r.reciprocal_rank is not None]
        ndcgs = [r.ndcg_at_k for r in answerable if r.ndcg_at_k is not None]

        mean_p = round(sum(precisions) / n_ans, 4)
        mean_r = round(sum(recalls) / n_ans, 4)
        mean_h = round(sum(hits) / n_ans, 4)
        mean_mrr = round(sum(mrrs) / n_ans, 4)
        mean_ndcg = round(sum(ndcgs) / n_ans, 4)

        p_ci = bootstrap_confidence_interval(precisions)
        r_ci = bootstrap_confidence_interval(recalls)
        h_ci = bootstrap_confidence_interval(hits)
        mrr_ci = bootstrap_confidence_interval(mrrs)
        ndcg_ci = bootstrap_confidence_interval(ndcgs)
    else:
        mean_p = mean_r = mean_h = mean_mrr = mean_ndcg = 0.0
        p_ci = r_ci = h_ci = mrr_ci = ndcg_ci = (0.0, 0.0)

    latencies = [r.latency_ms for r in results]
    sorted_latencies = sorted(latencies)
    p50_idx = int(0.50 * n)
    p95_idx = min(n - 1, int(0.95 * n))

    breakdowns = [r.latency_breakdown for r in results if r.latency_breakdown is not None]
    if breakdowns:
        avg_breakdown = LatencyBreakdown(
            query_preprocessing_ms=round(sum(b.query_preprocessing_ms for b in breakdowns) / len(breakdowns), 3),
            embedding_generation_ms=round(sum(b.embedding_generation_ms for b in breakdowns) / len(breakdowns), 3),
            pinecone_search_ms=round(sum(b.pinecone_search_ms for b in breakdowns) / len(breakdowns), 3),
            bm25_search_ms=round(sum(b.bm25_search_ms for b in breakdowns) / len(breakdowns), 3),
            rrf_fusion_ms=round(sum(b.rrf_fusion_ms for b in breakdowns) / len(breakdowns), 3),
            total_latency_ms=round(sum(b.total_latency_ms for b in breakdowns) / len(breakdowns), 3),
        )
    else:
        avg_breakdown = LatencyBreakdown(total_latency_ms=round(sum(latencies) / n, 3))

    abstention_summary = calculate_abstention_metrics(results)

    categories: Dict[str, List[QuestionMetricResult]] = {}
    for r in answerable:
        categories.setdefault(r.category, []).append(r)

    category_summary: Dict[str, Dict[str, float]] = {}
    for cat, cat_results in categories.items():
        c_n = len(cat_results)
        category_summary[cat] = {
            "count": c_n,
            "precision_at_k": round(sum(c.precision_at_k for c in cat_results if c.precision_at_k is not None) / c_n, 4),
            "recall_at_k": round(sum(c.recall_at_k for c in cat_results if c.recall_at_k is not None) / c_n, 4),
            "hit_rate_at_k": round(sum(c.hit_at_k for c in cat_results if c.hit_at_k is not None) / c_n, 4),
            "mrr": round(sum(c.reciprocal_rank for c in cat_results if c.reciprocal_rank is not None) / c_n, 4),
            "ndcg_at_k": round(sum(c.ndcg_at_k for c in cat_results if c.ndcg_at_k is not None) / c_n, 4),
            "latency_ms": round(sum(c.latency_ms for c in cat_results) / c_n, 2),
        }

    return AggregateMetrics(
        method_name=method_name,
        split=split,
        total_questions=n,
        total_answerable=n_ans,
        total_unanswerable=n_unans,
        k=k,
        mean_precision_at_k=mean_p,
        precision_ci_95=p_ci,
        mean_recall_at_k=mean_r,
        recall_ci_95=r_ci,
        mean_hit_rate_at_k=mean_h,
        hit_rate_ci_95=h_ci,
        mrr=mean_mrr,
        mrr_ci_95=mrr_ci,
        mean_ndcg_at_k=mean_ndcg,
        ndcg_ci_95=ndcg_ci,
        mean_latency_ms=round(sum(latencies) / n, 2),
        p50_latency_ms=round(sorted_latencies[p50_idx], 2),
        p95_latency_ms=round(sorted_latencies[p95_idx], 2),
        latency_breakdown=avg_breakdown,
        abstention_metrics=abstention_summary,
        category_breakdown=category_summary,
        threshold=threshold,
    )
