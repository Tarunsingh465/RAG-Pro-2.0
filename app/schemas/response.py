from pydantic import BaseModel
from typing import List


class SourceDocument(BaseModel):
    source: str
    page: int | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    out_of_context: bool = False
    similarity_score: float | None = None


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str