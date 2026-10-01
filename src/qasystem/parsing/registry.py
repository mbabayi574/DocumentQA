"""Format dispatch for uploads (plan.md P2).

Extension first, then PDF magic bytes, so a PDF uploaded as ``.txt`` still parses.
The size limit is enforced **before** any parser sees the bytes.
"""

from __future__ import annotations

from qasystem.domain.models import ParsedDocument
from qasystem.errors import EmptyDocumentError, FileTooLargeError, UnsupportedFormatError
from qasystem.parsing.markdown import MarkdownParser
from qasystem.parsing.pdf import PdfParser
from qasystem.parsing.text import TextParser

PDF_MAGIC = b"%PDF-"
SUPPORTED = {
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".txt": "Text",
    ".pdf": "PDF",
}


class ParserRegistry:
    """Chooses a parser and enforces the upload contract."""

    def __init__(self, max_upload_mb: int = 20) -> None:
        self.max_bytes = max_upload_mb * 1024 * 1024
        self._markdown = MarkdownParser()
        self._text = TextParser()
        self._pdf = PdfParser()

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        """Validate size and type, then parse. Raises typed errors, never guesses."""
        if len(data) > self.max_bytes:
            raise FileTooLargeError(
                f"file is {len(data) // (1024 * 1024)} MB, limit is "
                f"{self.max_bytes // (1024 * 1024)} MB"
            )
        if not data.strip():
            raise EmptyDocumentError("uploaded file is empty")

        extension = _extension(filename)
        if data.startswith(PDF_MAGIC):
            return self._pdf.parse(data, filename)
        if extension == ".pdf":
            raise UnsupportedFormatError("file has a .pdf extension but no %PDF- magic bytes")
        kind = SUPPORTED.get(extension)
        if kind is None:
            raise UnsupportedFormatError(
                f"unsupported format {extension or filename!r}; "
                f"supported: {', '.join(sorted(set(SUPPORTED)))}"
            )
        parser = self._markdown if kind == "Markdown" else self._text
        return parser.parse(data, filename)


def _extension(filename: str) -> str:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if "." not in base:
        return ""
    return "." + base.rsplit(".", 1)[1].lower()
