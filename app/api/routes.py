from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.schemas.request import ChatRequest
from app.schemas.response import (
    ChatResponse,
    HealthResponse,
    SourceDocument,
)
from app.services.ingest_service import IngestService

router = APIRouter(
    prefix="/api/v1",
    tags=["RAG API"],
)

ingest_service = IngestService()


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
    summary="Upload PDF",
)
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a PDF and ingest it into the vector database.
    """
    return ingest_service.ingest_document(file)


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


@router.post("/chat")
async def chat(request: ChatRequest):

    return rag_service.ask(
        request.question
    )