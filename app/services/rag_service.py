from app.core.config import settings
from app.core.logger import get_logger
from app.llm.chain import RAGChain
from app.vectorstore.pinecone import PineconeManager

logger = get_logger()

EVALUATIVE_KEYWORDS = [
    "hire",
    "hiring",
    "recruit",
    "fit",
    "candidate",
    "recommend",
    "should i",
    "suitability",
    "qualification",
    "shortlist",
]


class RagService:

    def __init__(self):
        self.chain = RAGChain()
        self.vectorstore = PineconeManager()

    def _get_search_query(self, question: str) -> tuple[str, bool]:
        is_evaluative = any(kw in question.lower() for kw in EVALUATIVE_KEYWORDS)
        if is_evaluative:
            return f"{question} candidate qualifications skills experience projects education achievements", True
        return question, False

    def _check_relevance(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
    ):
        """
        Run a similarity check before invoking the full chain.
        Returns (is_relevant, top_score, results).
        """
        metadata_filter = {"document_id": document_id} if document_id else None
        search_query, is_evaluative = self._get_search_query(question)

        results = self.vectorstore.similarity_search_with_score(
            search_query, k=top_k, filter=metadata_filter
        )

        if not results:
            logger.warning("No matches found in vectorstore for query.")
            return False, 0.0, []

        top_score = results[0][1]  # (Document, score) tuples

        # Evaluative questions (e.g. "Should I hire him?") naturally have slightly lower semantic
        # similarity to raw resume chunks than factual queries, so use adaptive threshold 0.50.
        effective_threshold = 0.50 if is_evaluative else settings.RAG_SIMILARITY_THRESHOLD

        logger.info(
            f"Top similarity score: {top_score:.4f} (threshold: {effective_threshold:.2f}, evaluative={is_evaluative})"
        )

        is_relevant = top_score >= effective_threshold
        return is_relevant, top_score, results

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
        history: list[dict] | None = None,
    ):
        is_relevant, top_score, results = self._check_relevance(
            question, document_id=document_id, top_k=top_k
        )

        if not is_relevant:
            return {
                "answer": None,
                "sources": [],
                "out_of_context": True,
                "similarity_score": top_score,
            }

        response = self.chain.invoke(
            question=question,
            document_id=document_id,
            top_k=top_k,
            history=history,
        )

        sources = []
        for doc in response["documents"]:
            sources.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
            )

        return {
            "answer": response["answer"],
            "sources": sources,
            "out_of_context": False,
            "similarity_score": top_score,
        }
