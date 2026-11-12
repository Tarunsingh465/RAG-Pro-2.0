"""
Reciprocal Rank Fusion (RRF) and Hybrid Retrieval Engine.

Combines Dense Semantic Vectors (Pinecone) with Sparse Keyword Matching (BM25)
using standard Reciprocal Rank Fusion:

    RRF_Score(d) = Σ [ 1.0 / (k + rank_m(d)) ]  for m in {dense, sparse}

Where:
- k = 60 (standard Cormack et al. constant).
- rank_m(d) is the 1-indexed ordinal position of chunk d in retriever m (rank 1, 2, ...).
- Missing chunks receive 0 score contribution from that retriever.
- Raw cosine similarity and BM25 scores are NEVER directly summed; only ordinal
  ranks are fused to prevent scale incompatibility.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from app.core.logger import get_logger

logger = get_logger()

DEFAULT_RRF_K = 60


@dataclass
class ChunkRankInfo:
    chunk_id: str
    source: str
    page: Optional[int]
    section: Optional[str]
    dense_rank: Optional[int]
    dense_score: Optional[float]
    bm25_rank: Optional[int]
    bm25_score: Optional[float]
    rrf_score: float
    content_preview: str


@dataclass
class RetrievalDiagnostics:
    query: str
    k_constant: int
    dense_candidates_count: int
    bm25_candidates_count: int
    total_unique_chunks: int
    dense_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    ranked_chunks: List[ChunkRankInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_canonical_chunk_id(doc: Document) -> str:
    """
    Extracts canonical deterministic chunk_id.
    - If metadata has 'file_name' and 'chunk_index', returns '{file_name}_{chunk_strategy}_{chunk_index}'.
    - Else if metadata has 'chunk_id', returns that 'chunk_id'.
    - Else falls back to source or content hash.
    """
    meta = doc.metadata or {}
    if "file_name" in meta and "chunk_index" in meta:
        fn = meta["file_name"]
        strat = meta.get("chunk_strategy", "medium")
        idx = meta["chunk_index"]
        return f"{fn}_{strat}_{idx}"
    
    if meta.get("chunk_id"):
        return str(meta["chunk_id"])
        
    source = meta.get("source")
    if source and "chunk_index" in meta:
        strat = meta.get("chunk_strategy", "medium")
        idx = meta["chunk_index"]
        return f"{source}_{strat}_{idx}"
        
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:16]


def get_chunk_identifier(doc: Document) -> str:
    """
    Extracts stable canonical chunk_id: {file_name}_{strategy}_{index}.
    """
    return get_canonical_chunk_id(doc)


def reciprocal_rank_fusion(
    dense_results: Sequence[Tuple[Document, float]],
    bm25_results: Sequence[Tuple[Document, float]],
    k: int = DEFAULT_RRF_K,
    top_n: int = 5,
) -> Tuple[List[Tuple[Document, float]], RetrievalDiagnostics]:
    """
    Fuses dense and sparse retrieval result sets into a single ranked list using RRF.
    
    Args:
        dense_results: Ordered list of (Document, cosine_similarity) from Pinecone.
        bm25_results: Ordered list of (Document, bm25_score) from BM25 sparse index.
        k: RRF smoothing constant (default: 60).
        top_n: Number of top fused documents to return.
        
    Returns:
        (fused_documents_with_scores, diagnostics)
    """
    t_start = time.perf_counter()

    chunk_map: Dict[str, Document] = {}
    dense_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, int] = {}
    bm25_scores: Dict[str, float] = {}
    rrf_scores: Dict[str, float] = {}

    # 1. Process Dense Results (1-indexed ranks)
    for rank_idx, (doc, score) in enumerate(dense_results, start=1):
        cid = get_chunk_identifier(doc)
        chunk_map[cid] = doc
        dense_ranks[cid] = rank_idx
        dense_scores[cid] = float(score)
        rrf_scores[cid] = 1.0 / (k + rank_idx)

    # 2. Process BM25 Sparse Results (1-indexed ranks)
    for rank_idx, (doc, score) in enumerate(bm25_results, start=1):
        cid = get_chunk_identifier(doc)
        if cid not in chunk_map:
            chunk_map[cid] = doc
        bm25_ranks[cid] = rank_idx
        bm25_scores[cid] = float(score)
        term = 1.0 / (k + rank_idx)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + term

    # 3. Deterministic Sorting:
    # Primary: Descending RRF score
    # Secondary: Best single rank position
    # Tertiary: Lexicographic chunk_id
    def sort_key(cid: str):
        best_single_rank = min(
            dense_ranks.get(cid, 999999),
            bm25_ranks.get(cid, 999999)
        )
        return (-rrf_scores[cid], best_single_rank, cid)

    sorted_chunk_ids = sorted(chunk_map.keys(), key=sort_key)
    fused_top = sorted_chunk_ids[:top_n]

    # Build final Document collection
    fused_results: List[Tuple[Document, float]] = []
    ranked_infos: List[ChunkRankInfo] = []

    for cid in fused_top:
        orig_doc = chunk_map[cid]
        final_score = rrf_scores[cid]

        # Annotate fused score into metadata
        doc_copy = Document(
            page_content=orig_doc.page_content,
            metadata=dict(orig_doc.metadata),
        )
        doc_copy.metadata["rrf_score"] = final_score
        doc_copy.metadata["dense_rank"] = dense_ranks.get(cid)
        doc_copy.metadata["bm25_rank"] = bm25_ranks.get(cid)

        fused_results.append((doc_copy, final_score))

        # Diagnostic info
        preview = orig_doc.page_content.replace("\n", " ").strip()
        if len(preview) > 100:
            preview = preview[:100] + "..."

        ranked_infos.append(
            ChunkRankInfo(
                chunk_id=cid,
                source=str(orig_doc.metadata.get("source", orig_doc.metadata.get("file_name", "unknown"))),
                page=orig_doc.metadata.get("page"),
                section=orig_doc.metadata.get("section"),
                dense_rank=dense_ranks.get(cid),
                dense_score=dense_scores.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                bm25_score=bm25_scores.get(cid),
                rrf_score=final_score,
                content_preview=preview,
            )
        )

    t_end = time.perf_counter()
    fusion_latency = (t_end - t_start) * 1000.0

    diagnostics = RetrievalDiagnostics(
        query="",
        k_constant=k,
        dense_candidates_count=len(dense_results),
        bm25_candidates_count=len(bm25_results),
        total_unique_chunks=len(chunk_map),
        fusion_latency_ms=fusion_latency,
        ranked_chunks=ranked_infos,
    )

    logger.debug(
        f"RRF complete: fused {len(dense_results)} dense and {len(bm25_results)} sparse "
        f"candidates -> {len(fused_results)} top chunks in {fusion_latency:.2f}ms."
    )

    return fused_results, diagnostics

