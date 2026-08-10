from app.core.constants import RAG_SIMILARITY_THRESHOLD
from app.core.logger import get_logger
from app.llm.chain import RAGChain
from app.vectorstore.pinecone import PineconeManager

logger = get_logger()


class RagService:

    def __init__(self):
        self.chain = RAGChain()
        self.vectorstore = PineconeManager()

    def _check_relevance(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
    ):
        """
        Run a cheap similarity check before invoking the full chain.
        Returns (is_relevant, top_score, results).
        """
        filter = {"document_id": document_id} if document_id else None

        results = self.vectorstore.similarity_search_with_score(
            question, k=top_k, filter=filter
        )

        if not results:
            logger.warning("No matches found in vectorstore for query.")
            return False, 0.0, []

        top_score = results[0][1]  # (Document, score) tuples
        logger.info(f"Top similarity score: {top_score:.4f}")

        is_relevant = top_score >= RAG_SIMILARITY_THRESHOLD
        return is_relevant, top_score, results

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
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