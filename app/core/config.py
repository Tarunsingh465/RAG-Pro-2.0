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
    NVIDIA_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    MISTRAL_API_KEY: str | None = None

    # Pinecone
    PINECONE_INDEX_NAME: str = "tyrion-2-0"
    PINECONE_NAMESPACE: str = "default"

    # Models & Endpoints
    LLM_BASE_URL: str | None = None
    LLM_MODEL: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    @property
    def active_llm_api_key(self) -> str:
        return self.NVIDIA_API_KEY or self.OPENROUTER_API_KEY

    @property
    def active_llm_base_url(self) -> str:
        if self.LLM_BASE_URL:
            return self.LLM_BASE_URL
        if self.NVIDIA_API_KEY:
            return "https://integrate.api.nvidia.com/v1"
        return "https://openrouter.ai/api/v1"

    # Retrieval
    TOP_K: int = Field(default=5)
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.55)

    # Chunking
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # Application
    APP_NAME: str = "TYRION-2.0"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()


settings = get_settings()
