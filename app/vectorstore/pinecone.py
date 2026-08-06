from typing import Sequence

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.logger import get_logger
from app.ingestion.embedder import Embedder

logger = get_logger()


class PineconeManager:
    def __init__(self):
        logger.info("Initializing Pinecone...")

        self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME

        self.embedding = Embedder().get_embedding_model()

        self._ensure_index()

        self.index = self.client.Index(self.index_name)

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embedding,
        )

        logger.success(f"Connected to Pinecone index '{self.index_name}'")

    def _embedding_dimension(self) -> int:
        """Detect embedding dimension automatically."""
        return len(self.embedding.embed_query("dimension test"))

    def _ensure_index(self):
        indexes = self.client.list_indexes().names()

        if self.index_name in indexes:
            logger.info(f"Index '{self.index_name}' already exists.")
            return

        dimension = self._embedding_dimension()

        logger.info(
            f"Creating index '{self.index_name}' "
            f"(dimension={dimension})..."
        )

        self.client.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        logger.success("Index created.")

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        return self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5):
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = 5):
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

    def delete(self, ids: Sequence[str]):
        self.vector_store.delete(ids=list(ids))

    def delete_all(self):
        self.index.delete(delete_all=True)