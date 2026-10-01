"""Format dispatch for uploads (plan.md P2).

Extension first, then PDF magic bytes, so a PDF uploaded as ``.txt`` still parses.
The size limit is enforced **before** any parser sees the bytes.
"""

from __future__ import annotations

from qasystem.domain.models import ParsedDocument
from qasystem.domain.ports import DocumentParser
from qasystem.errors import EmptyDocumentError, FileTooLargeError, UnsupportedFormatError
from qasystem.parsing.base import extension
from qasystem.parsing.markdown import MarkdownParser
from qasystem.parsing.pdf import PdfParser
from qasystem.parsing.text import TextParser

PDF_MAGIC = b"%PDF-"


class ParserRegistry:
    """Chooses a parser and enforces the upload contract."""

    def __init__(self, max_upload_mb: int = 20) -> None:
        self.max_bytes = max_upload_mb * 1024 * 1024
        self._by_extension: dict[str, DocumentParser] = {
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".txt": TextParser(),
        }
        self._pdf = PdfParser()

    @property
    def supported_extensions(self) -> list[str]:
        return sorted([*self._by_extension, ".pdf"])

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        """Validate size and type, then parse. Raises typed errors, never guesses."""
        if len(data) > self.max_bytes:
            limit = self.max_bytes // (1024 * 1024)
            raise FileTooLargeError(f"file is over the {limit} MB limit")
        if not data.strip():
            raise EmptyDocumentError("uploaded file is empty")

        ext = extension(filename)
        if data.startswith(PDF_MAGIC):
            return self._pdf.parse(data, filename)
        if ext == ".pdf":
            raise UnsupportedFormatError("file has a .pdf extension but no %PDF- magic bytes")

        parser = self._by_extension.get(ext)
        if parser is None:
            supported = ", ".join(self.supported_extensions)
            raise UnsupportedFormatError(
                f"unsupported format {ext or filename!r}; supported: {supported}"
            )
        return parser.parse(data, filename)
