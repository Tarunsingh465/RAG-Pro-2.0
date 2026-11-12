"""
Document Chunking and Splitting Module.

IMPORTANT ARCHITECTURAL NOTE:
The chunk_size and chunk_overlap values in this module represent CHARACTER counts
(via Python's built-in `length_function=len`), NOT token counts.
For typical technical and industrial English text, 1 token ≈ 4 characters:
- Small (500 chars)   ≈ ~125 tokens (ideal for high-precision entity and fact retrieval)
- Medium (1000 chars) ≈ ~250 tokens (balanced default for general technical documentation)
- Large (1500 chars)  ≈ ~375 tokens (suitable for multi-paragraph narrative / procedure context)
"""

from typing import List, Literal, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()

ChunkStrategy = Literal["small", "medium", "large"]

CHUNK_PRESETS: dict[str, dict[str, int]] = {
    "small": {"chunk_size": 500, "chunk_overlap": 100},
    "medium": {"chunk_size": 1000, "chunk_overlap": 200},
    "large": {"chunk_size": 1500, "chunk_overlap": 300},
}


class DocumentSplitter:
    """
    Splits LangChain documents into character-bounded chunks.
    
    Supports:
    - Pre-configured strategies: 'small' (500), 'medium' (1000), 'large' (1500) characters.
    - Custom chunk_size and chunk_overlap overrides.
    - Complete metadata preservation:
      - chunk_id (stable unique UUID)
      - chunk_index (sequential within document)
      - document_id (parent document identifier)
      - document_version_id (version tracking)
      - file_name / source
      - page, slide_number, sheet_name (hierarchical provenance)
      - section (heading context)
      - character_count (explicit unit tracking)
    """

    def __init__(
        self,
        strategy: Optional[ChunkStrategy] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        if strategy and strategy in CHUNK_PRESETS:
            preset = CHUNK_PRESETS[strategy]
            self.chunk_size = preset["chunk_size"]
            self.chunk_overlap = preset["chunk_overlap"]
            self.strategy_name = strategy
        else:
            self.chunk_size = chunk_size or settings.CHUNK_SIZE
            self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
            self.strategy_name = "custom"
            # Reverse lookup preset if matched
            for name, preset in CHUNK_PRESETS.items():
                if preset["chunk_size"] == self.chunk_size and preset["chunk_overlap"] == self.chunk_overlap:
                    self.strategy_name = name
                    break

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be strictly smaller "
                f"than CHUNK_SIZE ({self.chunk_size})."
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
            length_function=len,
        )

        logger.info(
            f"Initialized DocumentSplitter: strategy='{self.strategy_name}', "
            f"chunk_size={self.chunk_size} chars, chunk_overlap={self.chunk_overlap} chars"
        )

    def split(
        self,
        documents: List[Document],
        document_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Split LangChain documents into chunks while preserving full provenance metadata.
        """
        if not documents:
            logger.warning("No documents to split.")
            return []

        doc_version = version_id or str(uuid4())

        logger.info(f"Splitting {len(documents)} document(s) with size={self.chunk_size} chars...")
        raw_chunks = self.text_splitter.split_documents(documents)

        chunks: List[Document] = []
        for index, chunk in enumerate(raw_chunks):
            # Maintain existing metadata from extractors, then augment with chunk provenance
            meta = dict(chunk.metadata)
            meta.update(
                {
                    "chunk_id": str(uuid4()),
                    "chunk_index": index,
                    "chunk_strategy": self.strategy_name,
                    "chunk_size_chars": self.chunk_size,
                    "chunk_overlap_chars": self.chunk_overlap,
                    "character_count": len(chunk.page_content),
                    "document_version_id": doc_version,
                }
            )
            if document_id and "document_id" not in meta:
                meta["document_id"] = document_id

            # Ensure page reference is never completely absent
            if "page" not in meta:
                meta["page"] = meta.get("slide_number", 1)

            chunk.metadata = meta
            chunks.append(chunk)

        logger.success(
            f"Created {len(chunks)} chunks from {len(documents)} document(s) "
            f"using '{self.strategy_name}' configuration."
        )
        return chunks
