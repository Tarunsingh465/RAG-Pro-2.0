from fastapi import APIRouter, File, UploadFile
from typing import List

from app.core.config import settings
from app.schemas.request import ChatRequest
from app.schemas.response import (
    ChatResponse,
    HealthResponse,
)
from app.services.chat_service import ChatService
from app.services.ingest_service import IngestService
router = APIRouter(
    prefix="/api/v1",
    tags=["RAG API"],
)

ingest_service = IngestService()
chat_service = ChatService()

# =====================================================
# System
# =====================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
)
async def health():
    return HealthResponse(
        status="healthy",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


# =====================================================
# Documents
# =====================================================

@router.post(
    "/upload",
    summary="Upload Documents",
)
async def upload_documents(
    files: List[UploadFile] = File(...),
):
    """
    Upload one or more documents and ingest them.
    """

    return ingest_service.ingest_documents(files)

@router.get(
    "/documents",
    summary="List Uploaded Documents",
)
async def list_documents():
    # TODO: Implement document listing
    return {"documents": []}


@router.delete(
    "/documents/{document_name}",
    summary="Delete Document",
)
async def delete_document(document_name: str):
    # TODO: Delete document from storage + vector DB
    return {
        "message": f"{document_name} deleted successfully."
    }


# =====================================================
# Chat
# =====================================================

from app.services.rag_service import RagService

rag_service = RagService()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with RAG Assistant",
)
async def chat(request: ChatRequest):

    return chat_service.chat(
        question=request.question,
        document_id=request.document_id,
        top_k=request.top_k,
    )