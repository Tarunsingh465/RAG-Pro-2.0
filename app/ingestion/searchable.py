from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from app.core.logger import get_logger
from app.ingestion.cleaner import TextCleaner
from app.ingestion.ocr import OCRExtractor

logger = get_logger()


class SearchablePDFExtractor:
    """
    Handles both searchable and scanned PDFs.

    - Searchable pages -> PyMuPDF text extraction
    - Scanned pages -> PaddleOCR
    """

    def __init__(self):
        self.cleaner = TextCleaner()
        self.ocr = OCRExtractor()

    def extract(self, pdf_path: str | Path):

        pdf_path = Path(pdf_path)

        pdf = fitz.open(pdf_path)

        pages = []

        logger.info(f"Opened PDF ({len(pdf)} pages)")

        for page_number, page in enumerate(pdf):

            # ---------- Try native text ----------
            text = page.get_text("text").strip()

            # ---------- OCR if needed ----------
            if len(text) < 50:

                logger.info(
                    f"Page {page_number + 1}: OCR required."
                )

                pix = page.get_pixmap(
                    dpi=300,
                    alpha=False,
                )

                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                )

                text = self.ocr.extract_text(image)

            else:

                logger.info(
                    f"Page {page_number + 1}: Searchable."
                )

            # ---------- Clean ----------
            text = self.cleaner.clean(text)

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

        pdf.close()

        logger.success(
            f"Extracted {len(pages)} pages successfully."
        )

        return pages