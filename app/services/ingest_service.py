import shutil
from pathlib import Path

from fastapi import UploadFile

from app.core.constants import UPLOAD_FOLDER
from app.core.logger import get_logger
from app.ingestion.pipeline import IngestionPipeline

logger = get_logger()


class IngestService:

    def __init__(self):
        self.pipeline = IngestionPipeline()

    def ingest_documents(
        self,
        files: list[UploadFile],
    ):

        upload_dir = Path(UPLOAD_FOLDER)
        upload_dir.mkdir(parents=True, exist_ok=True)

        paths = []

        for file in files:

            file_path = upload_dir / file.filename

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.success(
                f"Saved: {file.filename}"
            )

            paths.append(file_path)

        return self.pipeline.ingest(paths)