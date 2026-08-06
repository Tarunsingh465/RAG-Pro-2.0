from typing import List

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import get_logger
from app.vectorstore.pinecone import PineconeManager

logger = get_logger()


class Retriever:
    """
    Handles retrieval operations from Pinecone.
    """

    def __init__(self):
        self.db = PineconeManager()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Document]:
        """
        Retrieve top-k relevant documents.
        """

        k = k or settings.TOP_K

        logger.info(
            f"Searching for relevant chunks (k={k})..."
        )

        docs = self.db.similarity_search(
            query=query,
            k=k,
        )

        logger.success(
            f"Retrieved {len(docs)} chunks."
        )

        return docs

    def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
    ):
        """
        Retrieve top-k relevant documents
        along with similarity scores.
        """

        k = k or settings.TOP_K

        results = self.db.similarity_search_with_score(
            query=query,
            k=k,
        )

        logger.success(
            f"Retrieved {len(results)} chunks with scores."
        )

        return results

    def get_context(
        self,
        query: str,
        k: int | None = None,
    ) -> str:
        """
        Convert retrieved documents into a single context string.
        """

        docs = self.retrieve(query, k)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        return context

    def get_sources(
        self,
        query: str,
        k: int | None = None,
    ):
        """
        Return document metadata only.
        """

        docs = self.retrieve(query, k)

        sources = []

        for doc in docs:

            sources.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
            )

        return sources