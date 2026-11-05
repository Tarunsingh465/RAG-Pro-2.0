from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from .env
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    GOOGLE_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    MISTRAL_API_KEY: str | None = None

    # Pinecone
    PINECONE_INDEX_NAME: str = "rag-pro"
    PINECONE_NAMESPACE: str = "default"

    # Models
    LLM_MODEL: str = "google/gemini-2.0-flash-001"
    EMBEDDING_MODEL: str = "models/embedding-001"

    # Retrieval
    TOP_K: int = Field(default=5)
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.55)

    # Chunking
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # Application
    APP_NAME: str = "RAG-Pro"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()


settings = get_settings()
