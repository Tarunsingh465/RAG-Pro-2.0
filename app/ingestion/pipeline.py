from pathlib import Path
from uuid import uuid4

from app.core.engineering import classify_engineering_domain
from app.core.logger import get_logger
from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import DocumentSplitter
from app.vectorstore.pinecone import PineconeManager

logger = get_logger()


class IngestionPipeline:

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()
        self.vector_store = PineconeManager()

    def ingest(
        self,
        file_paths: str | Path | list[str | Path],
    ):

        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        all_documents = []
        processed_files = []
        document_ids = []
        disciplines_detected = []

        logger.info(
            f"Starting ingestion of {len(file_paths)} file(s)."
        )

        for file_path in file_paths:

            file_path = Path(file_path)

            logger.info(
                f"Processing '{file_path.name}'"
            )

            # Prevent duplicate vectors: remove previous chunks if this file was re-uploaded
            try:
                self.vector_store.index.delete(
                    filter={"file_name": {"$eq": file_path.name}}
                )
                logger.info(f"Cleared previous vectors for '{file_path.name}' before re-indexing.")
            except Exception as e:
                logger.debug(f"Previous vector cleanup notice for '{file_path.name}': {e}")

            # Unique ID for this uploaded document
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

            logger.info(
                f"Assigned document_id={document_id} "
                f"to '{file_path.name}'"
            )

        chunks = self.splitter.split(all_documents)

        ids = self.vector_store.add_documents(chunks)

        logger.success(
            f"Ingestion completed. "
            f"Files={len(processed_files)} "
            f"Documents={len(all_documents)} "
            f"Chunks={len(chunks)}"
        )

        return {
            "files": processed_files,
            "documents": len(all_documents),
            "chunks": len(chunks),
            "vector_ids": ids,
            "document_ids": document_ids,
            "disciplines": list(dict.fromkeys(disciplines_detected)),
        }
