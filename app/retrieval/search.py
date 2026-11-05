"""
Search helper utilities and metadata filters for retrieval.
"""

from typing import Any


def build_metadata_filter(
    document_id: str | None = None,
    file_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Constructs a Pinecone-compatible metadata filter dictionary.
    """
    conditions = {}

    if document_id:
        conditions["document_id"] = {"$eq": document_id}

    if file_name:
        conditions["file_name"] = {"$eq": file_name}

    return conditions if conditions else None

