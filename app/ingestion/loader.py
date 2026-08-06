from pathlib import Path

from langchain_core.documents import Document

from app.core.logger import get_logger
from app.ingestion.searchable import SearchablePDFExtractor

logger = get_logger()


class DocumentLoader:
    """
    Loads searchable and scanned PDFs and converts them
    into LangChain Document objects.
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self):
        self.extractor = SearchablePDFExtractor()

    def load(self, file_path: str | Path) -> list[Document]:
        """
        Load a PDF and return LangChain Documents.

        Supports:
        - Searchable PDFs
        - Scanned PDFs (OCR)
        - Mixed PDFs

        Returns:
            list[Document]
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"{file_path.suffix} is not supported."
            )

        logger.info(f"Loading '{file_path.name}'")

        pages = self.extractor.extract(file_path)

        documents = []

        for page in pages:

            text = page["text"].strip()

            # Skip completely empty pages
            if not text:
                logger.warning(
                    f"Skipping empty page {page['page'] + 1}"
                )
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "page": page["page"] + 1,
                    },
                )
            )

        logger.success(
            f"Loaded {len(documents)} page(s) successfully."
        )

        return documents