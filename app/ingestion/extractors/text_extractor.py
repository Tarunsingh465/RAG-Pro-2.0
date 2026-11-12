import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.core.logger import get_logger

logger = get_logger()


class TextExtractor:
    """
    Structured extractor for Plain Text (.txt) and Markdown (.md) documents.
    
    Capabilities:
    - Multi-encoding resilience: Tries UTF-8, UTF-8-sig, CP1252, Latin-1.
    - Markdown section awareness: Splits markdown files by top-level/secondary headings,
      attaching the section title to chunk metadata.
    - Text pagination: Partitions long plain text documents into logical pages (~2500 chars)
      to maintain granular page references.
    """

    def extract(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_bytes = file_path.read_bytes()
        text = None
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            text = raw_bytes.decode("utf-8", errors="replace")

        suffix = file_path.suffix.lower()

        if suffix == ".md":
            return self._extract_markdown(text, file_path.name)
        else:
            return self._extract_plaintext(text, file_path.name)

    def _extract_markdown(self, text: str, source_name: str) -> List[Document]:
        # Split on markdown headings: #, ##, ###
        heading_pattern = r"(?=(?:\r?\n|^)#{1,3}\s+)"
        sections = re.split(heading_pattern, text)

        documents: List[Document] = []
        page_counter = 1

        for section in sections:
            clean_sec = section.strip()
            if not clean_sec:
                continue

            # Extract heading line if present
            lines = clean_sec.splitlines()
            first_line = lines[0].strip()
            section_title = "Introduction"
            if first_line.startswith("#"):
                section_title = first_line.lstrip("#").strip()

            documents.append(
                Document(
                    page_content=clean_sec,
                    metadata={
                        "source": source_name,
                        "section": section_title,
                        "page": page_counter,
                        "file_type": ".md",
                    },
                )
            )
            page_counter += 1

        if not documents:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "section": "General",
                        "page": 1,
                        "file_type": ".md",
                    },
                )
            )

        return documents

    def _extract_plaintext(self, text: str, source_name: str) -> List[Document]:
        # If text contains form-feed characters (\x0c / Ctrl+L), respect them as page breaks
        if "\x0c" in text:
            pages = text.split("\x0c")
            documents = []
            for idx, p in enumerate(pages, start=1):
                clean_p = p.strip()
                if clean_p:
                    documents.append(
                        Document(
                            page_content=clean_p,
                            metadata={
                                "source": source_name,
                                "page": idx,
                                "file_type": ".txt",
                            },
                        )
                    )
            if documents:
                return documents

        # Otherwise paginate by paragraphs or ~2500 character boundaries
        paragraphs = text.split("\n\n")
        documents = []
        current_chunk: List[str] = []
        current_len = 0
        page_counter = 1

        for para in paragraphs:
            p_clean = para.strip()
            if not p_clean:
                continue

            current_chunk.append(p_clean)
            current_len += len(p_clean)

            if current_len >= 2200:
                documents.append(
                    Document(
                        page_content="\n\n".join(current_chunk),
                        metadata={
                            "source": source_name,
                            "page": page_counter,
                            "file_type": ".txt",
                        },
                    )
                )
                current_chunk.clear()
                current_len = 0
                page_counter += 1

        if current_chunk:
            documents.append(
                Document(
                    page_content="\n\n".join(current_chunk),
                    metadata={
                        "source": source_name,
                        "page": page_counter,
                        "file_type": ".txt",
                    },
                )
            )

        if not documents:
            documents.append(
                Document(
                    page_content="[Empty text file]",
                    metadata={
                        "source": source_name,
                        "page": 1,
                        "file_type": ".txt",
                    },
                )
            )

        return documents
