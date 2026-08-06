import re

from app.core.logger import get_logger

logger = get_logger()


class TextCleaner:
    """
    Cleans extracted text before chunking.
    """

    def clean(self, text: str) -> str:

        if not text:
            return ""

        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Remove multiple blank lines
        text = re.sub(r"\n{2,}", "\n", text)

        # Remove page numbers
        text = re.sub(
            r"Page\s+\d+(\s+of\s+\d+)?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove standalone page numbers
        text = re.sub(
            r"^\d+$",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Remove duplicate lines
        cleaned = []
        previous = ""

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            if line == previous:
                continue

            previous = line

            cleaned.append(line)

        return "\n".join(cleaned)