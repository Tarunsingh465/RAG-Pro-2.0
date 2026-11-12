from pathlib import Path
from typing import List

import pypdf
from langchain_core.documents import Document

from app.core.logger import get_logger
from app.ingestion.cleaner import TextCleaner

logger = get_logger()


class PdfExtractor:
    """
    Extract text from PDFs locally using pypdf.
    100% free, runs offline, no external API key required.
    
    Capabilities:
    - Accurate 1-indexed page attribution.
    - Text sanitization and whitespace normalization.
    - Graceful error recovery for encrypted or partially corrupted PDF pages.
    """

    def __init__(self):
        self.cleaner = TextCleaner()

    def extract(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Extracting text locally from PDF: {file_path.name}")
        try:
            reader = pypdf.PdfReader(str(file_path))
        except Exception as e:
            logger.error(f"Failed to open PDF '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable PDF: {e}")

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                logger.warning(f"Could not decrypt PDF '{file_path.name}': {e}")

        documents: List[Document] = []
        for index, page in enumerate(reader.pages):
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting page {index + 1} of '{file_path.name}': {e}")
                raw_text = ""

            text = self.cleaner.clean(raw_text)

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "page": index + 1,
                        "total_pages": len(reader.pages),
                        "file_type": ".pdf",
                    },
                )
            )

        if not documents:
            documents.append(
                Document(
                    page_content=f"[Scanned/Image-only PDF: No extractable embedded text found in '{file_path.name}']",
                    metadata={
                        "source": file_path.name,
                        "page": 1,
                        "total_pages": len(reader.pages) if hasattr(reader, "pages") else 1,
                        "file_type": ".pdf",
                    },
                )
            )

        logger.info(f"Extracted {len(documents)} page(s) from '{file_path.name}' using pypdf")
        return documents
