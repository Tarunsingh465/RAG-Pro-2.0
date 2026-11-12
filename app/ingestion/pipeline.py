from pathlib import Path
from typing import List, Optional, Union
from uuid import uuid4

from app.core.engineering import classify_engineering_domain
from app.core.logger import get_logger
from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import ChunkStrategy, DocumentSplitter
from app.retrieval.bm25 import BM25Index
from app.vectorstore.pinecone import PineconeManager

logger = get_logger()


class IngestionPipeline:
    """
    Unified Ingestion Pipeline.
    
    Orchestrates:
    1. Multi-format structured document extraction (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD).
    2. Engineering domain classification.
    3. Configurable character-bounded chunking (small=500, medium=1000, large=1500 chars).
    4. Dual indexing: Dense vector embedding into Pinecone and sparse indexing into BM25.
    """

    def __init__(self, chunk_strategy: Optional[ChunkStrategy] = None):
        self.loader = DocumentLoader()
        self.chunk_strategy = chunk_strategy or "medium"
        self.splitter = DocumentSplitter(strategy=self.chunk_strategy)
        self.vector_store = PineconeManager()
        self.bm25_index = BM25Index()

    def ingest(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        chunk_strategy: Optional[ChunkStrategy] = None,
    ):
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        if chunk_strategy and chunk_strategy != self.chunk_strategy:
            self.chunk_strategy = chunk_strategy
            self.splitter = DocumentSplitter(strategy=chunk_strategy)

        all_documents = []
        processed_files = []
        document_ids = []
        disciplines_detected = []

        logger.info(
            f"Starting ingestion of {len(file_paths)} file(s) with chunk_strategy='{self.chunk_strategy}'."
        )

        for file_path in file_paths:
            file_path = Path(file_path)

            logger.info(f"Processing '{file_path.name}'")

            # Clean previous vectors & BM25 entries if file was re-uploaded
            try:
                self.vector_store.index.delete(
                    filter={"file_name": {"$eq": file_path.name}}
                )
                logger.info(f"Cleared previous vectors for '{file_path.name}' from Pinecone.")
            except Exception as e:
                logger.debug(f"Previous vector cleanup notice for '{file_path.name}': {e}")

            try:
                removed_bm25 = self.bm25_index.remove_document_by_filename(file_path.name)
                if removed_bm25 > 0:
                    logger.info(f"Cleared {removed_bm25} stale BM25 chunk(s) for '{file_path.name}'.")
            except Exception as e:
                logger.debug(f"Previous BM25 cleanup notice for '{file_path.name}': {e}")

            document_id = str(uuid4())
            documents = self.loader.load(file_path)

            # Analyze document for engineering domain & document type
            sample_content = " ".join(doc.page_content for doc in documents[:5])
            classification = classify_engineering_domain(sample_content, filename=file_path.name)
            disciplines_detected.append(classification["discipline_name"])
            logger.info(
                f"Classified '{file_path.name}' -> Discipline: {classification['discipline_name']} "
                f"({classification['discipline']}), DocType: {classification['doc_type']}"
            )

            # Attach metadata BEFORE splitting
            for document in documents:
                document.metadata.update(
                    {
                        "document_id": document_id,
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "discipline": classification["discipline"],
                        "discipline_name": classification["discipline_name"],
                        "doc_type": classification["doc_type"],
                        "is_engineering": str(classification["is_engineering"]),
                    }
                )

            all_documents.extend(documents)
            processed_files.append(file_path.name)
            document_ids.append(document_id)

            logger.info(f"Assigned document_id={document_id} to '{file_path.name}'")

        # Split documents into chunks with full metadata retention
        chunks = self.splitter.split(all_documents)

        # Index into Pinecone (Dense)
        ids = self.vector_store.add_documents(chunks)

        # Index into BM25 (Sparse)
        bm25_count = self.bm25_index.add_documents(chunks)

        logger.success(
            f"Ingestion completed. Files={len(processed_files)} "
            f"Documents={len(all_documents)} Chunks={len(chunks)} "
            f"(Dense vectors={len(ids)}, BM25 chunks={bm25_count})"
        )

        return {
            "files": processed_files,
            "documents": len(all_documents),
            "chunks": len(chunks),
            "vector_ids": ids,
            "document_ids": document_ids,
            "chunk_strategy": self.chunk_strategy,
            "disciplines": list(dict.fromkeys(disciplines_detected)),
        }
