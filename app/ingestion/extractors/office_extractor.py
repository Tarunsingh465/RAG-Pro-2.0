from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document
from pptx import Presentation

from app.core.logger import get_logger

logger = get_logger()


class OfficeExtractor:
    """
    Structured extractor for Microsoft Office documents (.docx and .pptx).
    
    DOCX Capabilities:
    - Extracts headings and preserves section hierarchies.
    - Extracts tables in order, converting rows to structured markdown tables.
    - Preserves relationships between table data and parent section headings.
    
    PPTX Capabilities:
    - Extracts slide content slide-by-slide with accurate slide numbers.
    - Extracts tables and text boxes within each slide.
    - Preserves slide index in metadata (page = slide_number).
    """

    def extract(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".docx":
            return self._extract_docx(file_path)
        elif suffix == ".pptx":
            return self._extract_pptx(file_path)
        else:
            raise ValueError(f"Unsupported Office document format: '{suffix}'")

    def _extract_docx(self, file_path: Path) -> List[Document]:
        try:
            doc = DocxDocument(file_path)
        except Exception as e:
            logger.error(f"Failed to open DOCX file '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable DOCX file: {e}")

        documents: List[Document] = []
        current_section = "General / Introduction"
        current_block: List[str] = []
        approx_page = 1
        words_count = 0

        def flush_block(section_name: str, has_tbl: bool = False):
            nonlocal words_count, approx_page
            content = "\n".join(current_block).strip()
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_path.name,
                            "page": approx_page,
                            "section": section_name,
                            "has_table": has_tbl,
                            "file_type": ".docx",
                        },
                    )
                )
                current_block.clear()
                words_count += len(content.split())
                if words_count > 450:
                    approx_page += 1
                    words_count = 0

        # Traverse body elements in document order (paragraphs and tables)
        for child in doc.element.body:
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = (p.style.name if p.style else "").lower()
                is_heading = style_name.startswith("heading") or p.style.name in ["Title", "Subtitle"]

                if is_heading:
                    # Flush current block before starting new section
                    if current_block:
                        flush_block(current_section)
                    current_section = text
                    current_block.append(f"### {text}")
                else:
                    current_block.append(text)

            elif isinstance(child, CT_Tbl):
                # Table encountered in document flow
                table = Table(child, doc)
                table_md = self._format_docx_table(table)
                if table_md:
                    # Flush any pending paragraph text
                    if current_block:
                        flush_block(current_section)
                    # Add table block under current section
                    current_block.append(table_md)
                    flush_block(current_section, has_tbl=True)

        if current_block:
            flush_block(current_section)

        if not documents:
            documents.append(
                Document(
                    page_content="[Empty document: No readable text or tables found]",
                    metadata={
                        "source": file_path.name,
                        "page": 1,
                        "section": current_section,
                        "has_table": False,
                        "file_type": ".docx",
                    },
                )
            )

        logger.info(f"Extracted {len(documents)} structured section(s) from '{file_path.name}'")
        return documents

    def _format_docx_table(self, table: Table) -> str:
        """Converts a docx Table into a clean Markdown representation."""
        rows_data = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            # De-duplicate adjacent identical cells from merged cells
            dedup_cells = []
            for cell in cells:
                if not dedup_cells or cell != dedup_cells[-1]:
                    dedup_cells.append(cell)
            if any(dedup_cells):
                rows_data.append(dedup_cells)

        if not rows_data:
            return ""

        # Normalize column counts
        max_cols = max(len(r) for r in rows_data)
        normalized_rows = [r + [""] * (max_cols - len(r)) for r in rows_data]

        header = normalized_rows[0]
        separator = ["---"] * max_cols
        body = normalized_rows[1:]

        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    def _extract_pptx(self, file_path: Path) -> List[Document]:
        try:
            prs = Presentation(file_path)
        except Exception as e:
            logger.error(f"Failed to open PPTX file '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable PPTX file: {e}")

        documents: List[Document] = []

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_parts: List[str] = []
            has_table = False

            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_parts.append(text)
                elif shape.has_table:
                    has_table = True
                    table_lines = []
                    for row in shape.table.rows:
                        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                        if any(cells):
                            table_lines.append("| " + " | ".join(cells) + " |")
                    if table_lines:
                        slide_parts.append("\n".join(table_lines))

            content = "\n".join(slide_parts).strip()
            if not content:
                content = f"[Slide {slide_idx}: Visual/Diagrammatic slide with no extractable text]"

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "page": slide_idx,
                        "slide_number": slide_idx,
                        "has_table": has_table,
                        "file_type": ".pptx",
                    },
                )
            )

        if not documents:
            documents.append(
                Document(
                    page_content="[Empty presentation: No slides found]",
                    metadata={
                        "source": file_path.name,
                        "page": 1,
                        "slide_number": 1,
                        "has_table": False,
                        "file_type": ".pptx",
                    },
                )
            )

        logger.info(f"Extracted {len(documents)} slide(s) from PPTX '{file_path.name}'")
        return documents
