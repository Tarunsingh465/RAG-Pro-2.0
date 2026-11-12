import json
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.logger import get_logger
from app.ingestion.embedder import Embedder

logger = get_logger()

# Known embedding model dimensions to eliminate startup probe calls
KNOWN_MODEL_DIMENSIONS: dict[str, int] = {
    "models/gemini-embedding-001": 3072,
    "gemini-embedding-001": 3072,
    "models/text-embedding-004": 768,
    "text-embedding-004": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

CACHE_FILE_PATH = Path("data/.pinecone_dimension_cache.json")


class PineconeManager:
    """
    Singleton manager for Pinecone vector store connections.
    Reuses connection across loader, retriever, and RAG services.
    Eliminates startup network probe overhead via verified dimension caching.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        logger.info("Initializing Pinecone singleton...")

        self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.model_name = settings.EMBEDDING_MODEL

        self.embedding = Embedder().get_embedding_model()

        # Resolve model dimension without making external probe requests
        self.dimension = self._resolve_model_dimension()

        # Verify index existence and validate dimension match
        self._ensure_index()

        self.index = self.client.Index(self.index_name)

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embedding,
        )

        self._initialized = True
        logger.success(f"Connected to Pinecone index '{self.index_name}' (dimension={self.dimension})")

    def _load_cache(self) -> dict:
        if CACHE_FILE_PATH.exists():
            try:
                return json.loads(CACHE_FILE_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug(f"Failed to read dimension cache: {e}")
        return {}

    def _save_cache(self, cache_data: dict) -> None:
        try:
            CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE_PATH.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to save dimension cache: {e}")

    def _resolve_model_dimension(self) -> int:
        """
        Determines the output dimension of the configured embedding model without making
        network calls during application startup. Checks cache, configuration, and known models.
        """
        cache = self._load_cache()
        cached_model = cache.get("model_name")
        cached_dim = cache.get("model_dimension")

        if cached_model == self.model_name and cached_dim:
            logger.debug(f"Resolved embedding dimension from local cache: {cached_dim} ({self.model_name})")
            return int(cached_dim)

        if settings.EMBEDDING_DIMENSION:
            dim = int(settings.EMBEDDING_DIMENSION)
            logger.debug(f"Resolved embedding dimension from settings: {dim}")
            return dim

        if self.model_name in KNOWN_MODEL_DIMENSIONS:
            dim = KNOWN_MODEL_DIMENSIONS[self.model_name]
            logger.debug(f"Resolved embedding dimension from known mapping: {dim}")
            return dim

        # Fallback only for unmapped/unknown models: one-time probe and cache
        logger.warning(
            f"Model '{self.model_name}' not found in known dimension registry. "
            "Executing one-time probe to determine dimensionality..."
        )
        probe_vec = self.embedding.embed_query("dimension probe")
        dim = len(probe_vec)
        logger.info(f"One-time probe detected dimension={dim}. Caching configuration.")
        return dim

    def _ensure_index(self):
        """
        Ensures index exists and verifies dimension compatibility between model and index.
        Raises ValueError if dimensions mismatch.
        """
        cache = self._load_cache()
        cached_index = cache.get("index_name")
        cached_index_dim = cache.get("index_dimension")
        cached_model = cache.get("model_name")

        # Fast-path: If cache confirms index and model match, skip remote listing
        if (
            cached_index == self.index_name
            and cached_model == self.model_name
            and cached_index_dim == self.dimension
        ):
            logger.debug(
                f"Index '{self.index_name}' and model '{self.model_name}' dimension "
                f"compatibility verified from cache ({self.dimension}-dim)."
            )
            return

        # Check existing index metadata remotely
        indexes = self.client.list_indexes().names()

        if self.index_name in indexes:
            logger.info(f"Index '{self.index_name}' exists in Pinecone. Validating dimensions...")
            desc = self.client.describe_index(self.index_name)
            index_dim = int(desc.dimension)

            if index_dim != self.dimension:
                raise ValueError(
                    f"Dimension mismatch error: Embedding model '{self.model_name}' produces "
                    f"{self.dimension}-dimensional vectors, but Pinecone index '{self.index_name}' "
                    f"is configured for {index_dim}-dimensional vectors. "
                    f"Resolution: Update EMBEDDING_MODEL in .env or recreate index '{self.index_name}' "
                    f"with dimension={self.dimension}."
                )

            # Update cache with verified dimension
            self._save_cache(
                {
                    "model_name": self.model_name,
                    "model_dimension": self.dimension,
                    "index_name": self.index_name,
                    "index_dimension": index_dim,
                    "metric": str(desc.metric),
                }
            )
            logger.success(f"Verified dimension compatibility: {self.dimension} == {index_dim}")
            return

        # Index does not exist: create it with verified dimension
        logger.info(f"Creating index '{self.index_name}' (dimension={self.dimension})...")
        self.client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        self._save_cache(
            {
                "model_name": self.model_name,
                "model_dimension": self.dimension,
                "index_name": self.index_name,
                "index_dimension": self.dimension,
                "metric": "cosine",
            }
        )
        logger.success(f"Index '{self.index_name}' created.")

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        return self.vector_store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ):
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ):
        return self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
        )

    def get_retriever(self, k: int = 5):
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

    def delete(self, ids: Sequence[str]):
        self.vector_store.delete(ids=list(ids))

    def delete_all(self):
        self.index.delete(delete_all=True)
