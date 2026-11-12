"""
Retriever Module supporting Dense (Pinecone), Sparse (BM25), and Hybrid (RRF) Retrieval.

Features:
- Sub-millisecond active document verification to prevent data leakage.
- Independent Dense, Sparse (BM25), and Hybrid (Reciprocal Rank Fusion) retrieval modes.
- Complete retrieval diagnostics (candidate pools, rank matrices, latency profiling).
- Stable chunk identifier matching across disparate scoring distributions.
"""

from pathlib import Path
import time
from typing import List, Literal, Optional, Tuple

from langchain_core.documents import Document

from app.core.config import settings
from app.core.constants import UPLOAD_FOLDER
from app.core.logger import get_logger
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import RetrievalDiagnostics, reciprocal_rank_fusion
from app.vectorstore.pinecone import PineconeManager

logger = get_logger()

RetrievalMode = Literal["hybrid", "dense", "sparse"]


class Retriever:
    """
    Unified Retriever for RAG-Pro 2.0.
    """

    def __init__(self):
        self.db = PineconeManager()
        self.bm25 = BM25Index()

    def _get_active_documents(self) -> List[str]:
        upload_dir = Path(UPLOAD_FOLDER)
        if not upload_dir.exists():
            return []
        return [
            f.name
            for f in upload_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

    def retrieve_dense(
        self,
        query: str,
        k: int = 5,
        document_id: Optional[str] = None,
        active_docs: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k candidates using dense cosine similarity from Pinecone.
        """
        active = active_docs if active_docs is not None else self._get_active_documents()
        if not active:
            return []

        if document_id:
            metadata_filter = {"document_id": {"$eq": document_id}}
        elif len(active) == 1:
            metadata_filter = {"file_name": {"$eq": active[0]}}
        else:
            metadata_filter = {"file_name": {"$in": active}}

        try:
            results = self.db.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter,
            )
            return results
        except Exception as e:
            logger.error(f"Dense Pinecone retrieval error: {e}")
            return []

    def retrieve_bm25(
        self,
        query: str,
        k: int = 5,
        document_id: Optional[str] = None,
        active_docs: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k candidates using Okapi BM25 sparse keyword matching.
        """
        active = active_docs if active_docs is not None else self._get_active_documents()
        if not active:
            return []

        return self.bm25.search(
            query=query,
            top_k=k,
            active_documents=active,
            document_id=document_id,
        )

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 5,
        document_id: Optional[str] = None,
        rrf_k: int = 60,
    ) -> Tuple[List[Tuple[Document, float]], RetrievalDiagnostics]:
        """
        Retrieves candidates from both Dense and Sparse pipelines and fuses
        them using Reciprocal Rank Fusion (RRF).
        """
        t0 = time.perf_counter()
        active_docs = self._get_active_documents()

        if not active_docs:
            diag = RetrievalDiagnostics(
                query=query,
                k_constant=rrf_k,
                dense_candidates_count=0,
                bm25_candidates_count=0,
                total_unique_chunks=0,
            )
            return [], diag

        # Retrieve a broader pool (2x k) from each retriever for robust fusion
        candidate_k = max(k * 2, 10)

        t_dense_start = time.perf_counter()
        dense_results = self.retrieve_dense(
            query=query,
            k=candidate_k,
            document_id=document_id,
            active_docs=active_docs,
        )
        t_dense_end = time.perf_counter()
        dense_lat = (t_dense_end - t_dense_start) * 1000.0

        t_bm25_start = time.perf_counter()
        bm25_results = self.retrieve_bm25(
            query=query,
            k=candidate_k,
            document_id=document_id,
            active_docs=active_docs,
        )
        t_bm25_end = time.perf_counter()
        bm25_lat = (t_bm25_end - t_bm25_start) * 1000.0

        # Fuse via RRF
        fused_docs, diagnostics = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=rrf_k,
            top_n=k,
        )

        total_lat = (time.perf_counter() - t0) * 1000.0
        diagnostics.query = query
        diagnostics.dense_latency_ms = dense_lat
        diagnostics.bm25_latency_ms = bm25_lat
        diagnostics.total_retrieval_ms = total_lat

        logger.info(
            f"Hybrid retrieval finished in {total_lat:.2f}ms: "
            f"Dense={len(dense_results)} ({dense_lat:.1f}ms), "
            f"BM25={len(bm25_results)} ({bm25_lat:.1f}ms) -> Top {len(fused_docs)} chunks."
        )

        return fused_docs, diagnostics

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        document_id: Optional[str] = None,
        mode: RetrievalMode = "hybrid",
    ) -> List[Document]:
        """
        Unified retrieval interface returning LangChain Document objects.
        Defaults to Hybrid (RRF).
        """
        top_k = k or settings.TOP_K

        if mode == "dense":
            scored = self.retrieve_dense(query=query, k=top_k, document_id=document_id)
            return [doc for doc, _ in scored]
        elif mode == "sparse":
            scored = self.retrieve_bm25(query=query, k=top_k, document_id=document_id)
            return [doc for doc, _ in scored]
        else:
            fused, _ = self.retrieve_hybrid(query=query, k=top_k, document_id=document_id)
            return [doc for doc, _ in fused]

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        document_id: Optional[str] = None,
        mode: RetrievalMode = "hybrid",
    ) -> List[Tuple[Document, float]]:
        """
        Retrieval interface returning (Document, score) tuples.
        """
        top_k = k or settings.TOP_K

        if mode == "dense":
            return self.retrieve_dense(query=query, k=top_k, document_id=document_id)
        elif mode == "sparse":
            return self.retrieve_bm25(query=query, k=top_k, document_id=document_id)
        else:
            fused, _ = self.retrieve_hybrid(query=query, k=top_k, document_id=document_id)
            return fused

    def get_context(
        self,
        query: str,
        k: Optional[int] = None,
        document_id: Optional[str] = None,
        mode: RetrievalMode = "hybrid",
    ) -> str:
        docs = self.retrieve(query=query, k=k, document_id=document_id, mode=mode)
        return "\n\n".join(doc.page_content for doc in docs)

    def get_sources(
        self,
        query: str,
        k: Optional[int] = None,
        document_id: Optional[str] = None,
        mode: RetrievalMode = "hybrid",
    ) -> List[dict]:
        docs = self.retrieve(query=query, k=k, document_id=document_id, mode=mode)
        sources = []
        for doc in docs:
            sources.append(
                {
                    "source": doc.metadata.get("source") or doc.metadata.get("file_name"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "document_id": doc.metadata.get("document_id"),
                    "section": doc.metadata.get("section"),
                    "rrf_score": doc.metadata.get("rrf_score"),
                }
            )
        return sources
