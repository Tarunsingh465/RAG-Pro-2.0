from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events.
    """
    logger.info("Starting RAG FastAPI Application...")

    # Future:
    # - Initialize Pinecone
    # - Warm embedding model
    # - Check API keys
    # - Database connection

    yield

    logger.info("Shutting down RAG FastAPI Application...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production Ready Retrieval-Augmented Generation API",
    lifespan=lifespan,
)

# CORS — restrict `allow_origins` in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,     # `*` origins + credentials = browser rejection
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Welcome to RAG FastAPI",
        "version": settings.APP_VERSION,
    }


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "online",
        "version": settings.APP_VERSION,
        "app": settings.APP_NAME,
    }


app.include_router(router)
