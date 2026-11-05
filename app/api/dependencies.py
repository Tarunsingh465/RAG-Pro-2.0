"""
FastAPI dependency injection utilities.
"""

from functools import lru_cache

from app.services.chat_service import ChatService
from app.services.ingest_service import IngestService


@lru_cache
def get_ingest_service() -> IngestService:
    """Provides a cached IngestService instance."""
    return IngestService()


@lru_cache
def get_chat_service() -> ChatService:
    """Provides a cached ChatService instance."""
    return ChatService()

