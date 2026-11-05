"""
Document re-ranking module.
Allows secondary re-scoring of retrieved chunks.
"""

from typing import Sequence
from langchain_core.documents import Document

from app.core.logger import get_logger

logger = get_logger()


class DocumentReranker:
    """
    Re-scores and re-ranks retrieved chunks based on exact keyword
    matches and semantic query term frequency.
    """

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int = 5,
    ) -> list[Document]:
        if not documents:
            return []

        query_terms = set(query.lower().split())

        def score_doc(doc: Document) -> float:
            content_lower = doc.page_content.lower()
            # Count how many query terms appear in chunk
            matches = sum(1 for term in query_terms if term in content_lower)
            return float(matches)

        ranked = sorted(documents, key=score_doc, reverse=True)
        logger.info(f"Reranked {len(documents)} chunks -> returning top {top_n}")
        return list(ranked[:top_n])

