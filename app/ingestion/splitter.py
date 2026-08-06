from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class DocumentSplitter:
    """
    Splits LangChain documents into smaller chunks.
    """

    def __init__(self) -> None:
        if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        self.text_splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
                length_function=len,
            )
        )

    def split(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Split LangChain documents into chunks.

        Args:
            documents: Documents to split.

        Returns:
            Chunked documents.
        """

        if not documents:
            logger.warning("No documents to split.")
            return []

        logger.info(
            f"Splitting {len(documents)} document(s)..."
        )

        chunks = self.text_splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_id": str(uuid4()),
                    "chunk_index": index,
                }
            )

        logger.success(
            f"Created {len(chunks)} chunks from {len(documents)} document(s)."
        )

        return chunks