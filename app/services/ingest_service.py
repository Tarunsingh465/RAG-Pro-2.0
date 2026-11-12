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

            safe_filename = Path(file.filename).name
            file_path = upload_dir / safe_filename

            content = await file.read()

            with open(file_path, "wb") as buffer:
                buffer.write(content)

            logger.success(
                f"Saved: {safe_filename}"
            )

            paths.append(file_path)

        try:
            return self.pipeline.ingest(paths)
        except ValueError as e:
            # Clean up uploaded files on validation error to prevent orphaned files
            for p in paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            logger.error(f"Ingestion failed with validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            for p in paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            logger.error(f"Ingestion failed with unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

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
        safe_name = Path(document_name).name
        upload_dir = Path(UPLOAD_FOLDER)
        file_path = upload_dir / safe_name

        file_deleted = False
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            file_deleted = True
            logger.info(f"Deleted local file: {safe_name}")

        try:
            self.pipeline.vector_store.index.delete(
                filter={"file_name": {"$eq": safe_name}}
            )
            # Also purge by source metadata if indexed with full path or basename
            self.pipeline.vector_store.index.delete(
                filter={"source": {"$eq": safe_name}}
            )
            logger.info(f"Deleted vectors for '{safe_name}' from Pinecone")
        except Exception as e:
            logger.warning(
                f"Could not delete vectors from Pinecone for {safe_name}: {e}"
            )

        try:
            removed_bm25 = self.pipeline.bm25_index.remove_document_by_filename(safe_name)
            logger.info(f"Removed {removed_bm25} chunk(s) from BM25 index for '{safe_name}'.")
        except Exception as e:
            logger.warning(f"Could not remove BM25 chunks for {safe_name}: {e}")

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

        try:
            self.pipeline.bm25_index.clear()
            logger.success("Cleared BM25 sparse index.")
        except Exception as e:
            logger.warning(f"Could not clear BM25 index: {e}")

        return count
