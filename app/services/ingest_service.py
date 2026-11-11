from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.constants import UPLOAD_FOLDER
from app.core.logger import get_logger
from app.ingestion.pipeline import IngestionPipeline

logger = get_logger()


class IngestService:

    def __init__(self):
        self.pipeline = IngestionPipeline()

    async def ingest_documents(
        self,
        files: list[UploadFile],
    ):
        upload_dir = Path(UPLOAD_FOLDER)
        upload_dir.mkdir(parents=True, exist_ok=True)

        paths = []

        for file in files:

            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="Upload contained a file with no filename.",
                )

            file_path = upload_dir / file.filename

            content = await file.read()

            with open(file_path, "wb") as buffer:
                buffer.write(content)

            logger.success(
                f"Saved: {file.filename}"
            )

            paths.append(file_path)

        return self.pipeline.ingest(paths)

    def list_documents(self) -> list[str]:
        upload_dir = Path(UPLOAD_FOLDER)
        if not upload_dir.exists():
            return []
        return sorted(
            [
                f.name
                for f in upload_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
        )

    def delete_document(self, document_name: str) -> bool:
        upload_dir = Path(UPLOAD_FOLDER)
        file_path = upload_dir / document_name

        file_deleted = False
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            file_deleted = True
            logger.info(f"Deleted local file: {document_name}")

        try:
            self.pipeline.vector_store.index.delete(
                filter={"file_name": {"$eq": document_name}}
            )
            # Also purge by source metadata if indexed with full path or basename
            self.pipeline.vector_store.index.delete(
                filter={"source": {"$eq": document_name}}
            )
            logger.info(f"Deleted vectors for '{document_name}' from Pinecone")
        except Exception as e:
            logger.warning(
                f"Could not delete vectors from Pinecone for {document_name}: {e}"
            )

        return file_deleted

    def delete_all_documents(self) -> int:
        """
        Deletes all uploaded documents locally and completely wipes the Pinecone index.
        """
        upload_dir = Path(UPLOAD_FOLDER)
        count = 0
        if upload_dir.exists():
            for f in upload_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    try:
                        f.unlink()
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {f.name}: {e}")
            logger.info(f"Deleted {count} local uploaded file(s).")

        try:
            self.pipeline.vector_store.delete_all()
            logger.success("Purged all vectors from Pinecone index.")
        except Exception as e:
            logger.warning(f"Could not purge Pinecone index: {e}")

        return count
