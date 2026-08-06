from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from app.core.logger import get_logger

logger = get_logger()


class DocumentLoader:
    """
    Loads supported documents and converts them
    into LangChain Document objects.
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def load(self, file_path: str | Path) -> list[Document]:
        """
        Load a document and return LangChain Documents.

        Args:
            file_path: Path to document

        Returns:
            list[Document]
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"{file_path.suffix} is not supported.")

        logger.info(f"Loading {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        logger.success(f"{len(documents)} pages loaded successfully.")

        for doc in documents:
            doc.metadata.update(
                {
                    "source": file_path.name,
                    "page": doc.metadata.get("page", 0),
                }
            )

        return documents