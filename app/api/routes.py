from fastapi import APIRouter, File, HTTPException, UploadFile
from typing import List

from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services.chat_service import ChatService
from app.services.ingest_service import IngestService

router = APIRouter(
    prefix="/api/v1",
    tags=["RAG API"],
)

ingest_service = IngestService()
chat_service = ChatService()


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
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    return await ingest_service.ingest_documents(files)


@router.get(
    "/documents",
    summary="List Uploaded Documents",
)
async def list_documents():
    docs = ingest_service.list_documents()
    return {"documents": docs}


@router.delete(
    "/documents",
    summary="Delete All Documents",
)
async def delete_all_documents():
    count = ingest_service.delete_all_documents()
    return {
        "message": f"Successfully deleted {count} document(s) and cleared Pinecone index.",
        "deleted_count": count,
    }


@router.delete(
    "/documents/{document_name}",
    summary="Delete Document",
)
async def delete_document(document_name: str):
    success = ingest_service.delete_document(document_name)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_name}' not found.",
        )
    return {
        "message": f"{document_name} deleted successfully."
    }


# =====================================================
# Chat
# =====================================================

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
        history=request.history,
        retrieval_mode=request.retrieval_mode,
    )
