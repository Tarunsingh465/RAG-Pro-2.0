from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question",
        examples=["What is Retrieval Augmented Generation?"],
    )

    document_id: str | None = Field(
        default=None,
        description="ID of the document to use for RAG retrieval.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )

    history: list[dict] | None = Field(
        default=None,
        description="Optional recent chat history messages",
    )

    retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'hybrid' (RRF dense+sparse), 'dense' (Pinecone), or 'sparse' (BM25)",
    )


class DeleteDocumentRequest(BaseModel):
    """
    Delete a document from vector database.
    """

    document_name: str
