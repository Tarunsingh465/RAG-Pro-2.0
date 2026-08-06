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

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )


class DeleteDocumentRequest(BaseModel):
    """
    Delete a document from vector database.
    """

    document_name: str