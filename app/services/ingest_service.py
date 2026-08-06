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

    def ingest_document(self, file: UploadFile):

        # Create upload directory
        upload_dir = Path(UPLOAD_FOLDER)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.success(
            f"File saved at {file_path}"
        )

        # Start ingestion
        result = self.pipeline.ingest(file_path)

        return result