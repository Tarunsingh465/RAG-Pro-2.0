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
        document_id: str | None = None,
    ) -> List[Document]:
        """
        Retrieve top-k relevant documents.

        If document_id is provided, retrieval is restricted
        to chunks belonging to that document.
        """

        k = k or settings.TOP_K

        logger.info(
            f"Searching for relevant chunks (k={k}, "
            f"document_id={document_id})..."
        )

        metadata_filter = None

        if document_id:
            metadata_filter = {
                "document_id": {
                    "$eq": document_id
                }
            }

        docs = self.db.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter,
        )

        logger.success(
            f"Retrieved {len(docs)} chunks."
        )

        return docs

    def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
        document_id: str | None = None,
    ):
        """
        Retrieve top-k relevant documents with similarity scores.
        """

        k = k or settings.TOP_K

        metadata_filter = None

        if document_id:
            metadata_filter = {
                "document_id": {
                    "$eq": document_id
                }
            }

        results = self.db.similarity_search_with_score(
            query=query,
            k=k,
            filter=metadata_filter,
        )

        logger.success(
            f"Retrieved {len(results)} chunks with scores."
        )

        return results

    def get_context(
        self,
        query: str,
        k: int | None = None,
        document_id: str | None = None,
    ) -> str:
        """
        Convert retrieved documents into a single context string.
        """

        docs = self.retrieve(
            query=query,
            k=k,
            document_id=document_id,
        )

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        return context

    def get_sources(
        self,
        query: str,
        k: int | None = None,
        document_id: str | None = None,
    ):
        """
        Return document metadata only.
        """

        docs = self.retrieve(
            query=query,
            k=k,
            document_id=document_id,
        )

        sources = []

        for doc in docs:

            sources.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "document_id": doc.metadata.get("document_id"),
                }
            )

        return sources