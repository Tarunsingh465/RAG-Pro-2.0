from pathlib import Path

from langchain_core.documents import Document

from app.core.logger import get_logger
from app.ingestion.mistral_ocr import MistralOCRExtractor

logger = get_logger()


class DocumentLoader:
    """
    Loads PDF documents using Mistral OCR and converts them
    into LangChain Document objects.
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self):
        self.extractor = MistralOCRExtractor()

    def load(self, file_path: str | Path) -> list[Document]:
        """
        Load a PDF and return a list of LangChain Documents.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List[Document]
        """

        file_path = Path(file_path)

        self._validate_file(file_path)

        logger.info(f"Loading document: {file_path.name}")

        pages = self.extractor.extract(file_path)

        documents = []

        for page in pages:
            text = page["text"].strip()

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
            f"Loaded {len(documents)} pages from '{file_path.name}'."
        )

        return documents

    @classmethod
    def _validate_file(cls, file_path: Path) -> None:
        """Validate file existence and extension."""

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        if file_path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{file_path.suffix}'. "
                f"Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )