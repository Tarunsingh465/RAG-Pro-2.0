from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from app.core.logger import get_logger

logger = get_logger()


class OCRExtractor:
    """
    Robust OCR extractor.

    Supported inputs:
    - PIL.Image
    - numpy.ndarray
    - str
    - pathlib.Path
    """

    def __init__(self):
        logger.info("Initializing PaddleOCR...")

        self.ocr = PaddleOCR(lang="en")

        logger.success("PaddleOCR initialized.")

    def _prepare_image(
        self,
        image: Image.Image | np.ndarray | str | Path,
    ) -> np.ndarray | str:
        """
        Convert supported inputs into a format
        accepted by PaddleOCR.
        """

        # File path
        if isinstance(image, (str, Path)):
            path = Path(image)

            if not path.exists():
                raise FileNotFoundError(path)

            return str(path)

        # PIL Image
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        # NumPy image
        if isinstance(image, np.ndarray):
            return image

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    def extract_text(
        self,
        image: Image.Image | np.ndarray | str | Path,
    ) -> str:
        """
        Extract text from an image.

        Returns:
            Extracted text.
        """

        image = self._prepare_image(image)

        logger.info(
            f"OCR Input Type: {type(image).__name__}"
        )

        try:
            result = self.ocr.ocr(image)

        except Exception:
            logger.exception("OCR failed.")
            return ""

        if not result or not result[0]:
            logger.warning("No text detected.")
            return ""

        lines = []

        for line in result[0]:

            try:
                text = line[1][0].strip()

                if text:
                    lines.append(text)

            except Exception:
                continue

        final_text = "\n".join(lines)

        logger.success(
            f"Extracted {len(lines)} line(s), "
            f"{len(final_text)} characters."
        )

        return final_text