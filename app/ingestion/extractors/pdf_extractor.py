from pathlib import Path
import pypdf
from langchain_core.documents import Document

from app.core.logger import get_logger
from app.ingestion.cleaner import TextCleaner

logger = get_logger()


class PdfExtractor:
    """
    Extract text from PDFs locally using pypdf.
    100% free, runs offline, no API key required.
    """

    def __init__(self):
        self.cleaner = TextCleaner()

    def extract(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Extracting text locally from PDF: {file_path.name}")
        reader = pypdf.PdfReader(str(file_path))

        documents = []
        for index, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            text = self.cleaner.clean(raw_text)

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "page": index + 1,
                    },
                )
            )

        logger.info(
            f"Extracted {len(documents)} page(s) from '{file_path.name}' using pypdf"
        )
        return documents

