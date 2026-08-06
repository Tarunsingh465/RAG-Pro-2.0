from pathlib import Path

from app.core.logger import get_logger
from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import DocumentSplitter
from app.vectorstore.pinecone import PineconeManager

logger = get_logger()


class IngestionPipeline:

    def __init__(self):

        self.loader = DocumentLoader()

        self.splitter = DocumentSplitter()

        self.vector_store = PineconeManager()

    def ingest(self, file_path: str | Path):

        logger.info(f"Starting ingestion: {file_path}")

        # Step 1
        documents = self.loader.load(file_path)

        # Step 2
        chunks = self.splitter.split(documents)

        # Step 3
        ids = self.vector_store.add_documents(chunks)

        logger.success(
            f"Ingestion completed. "
            f"Pages={len(documents)} "
            f"Chunks={len(chunks)}"
        )

        return {
            "document_name": Path(file_path).name,
            "pages": len(documents),
            "chunks": len(chunks),
            "vector_ids": ids,
        }