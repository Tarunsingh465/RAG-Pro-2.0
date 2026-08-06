from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class Embedder:
    """
    Handles embedding generation using Google's embedding model.
    """

    def __init__(self):
        logger.info("Initializing embedding model...")

        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        logger.success("Embedding model initialized.")

    def get_embedding_model(self):
        """
        Returns the embedding model instance.
        """
        return self.embedding_model

    def embed_query(self, query: str):
        """
        Generate embedding for a user query.
        """
        return self.embedding_model.embed_query(query)

    def embed_documents(self, texts: list[str]):
        """
        Generate embeddings for multiple text chunks.
        """
        return self.embedding_model.embed_documents(texts)