"""PDF parser (plan.md P2, PyMuPDF).

Page by page with ``get_text("blocks", sort=True)``. Sections are page-based: a PDF's
visual heading structure is not reliably recoverable, and plan.md permits page sections
when font-size heuristics are not dependable.

**No OCR.** A PDF whose pages yield no text raises ``NoTextLayerError`` naming the
reason, rather than returning empty evidence. **No RTL reversal** either: some Persian
PDFs store text in lossy presentation forms that no reversal or normalization can
repair (plan.md §2.3), and mangling a citation is worse than saying so.
"""

from __future__ import annotations

import unicodedata
from typing import Any

import pymupdf

from qasystem.domain.models import ParsedDocument, Section
from qasystem.errors import EmptyDocumentError, NoTextLayerError, ParseError
from qasystem.parsing.base import build_document, stem

MAGIC = b"%PDF-"


class PdfParser:
    """Extracts per-page text with 1-based page numbers."""

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        if not data.strip():
            raise EmptyDocumentError("pdf file is empty")
        if not data.startswith(MAGIC):
            raise ParseError("file does not start with the %PDF- magic bytes")

        title = stem(filename)
        pages = _page_texts(data)
        if not any(pages):
            raise NoTextLayerError(
                "PDF has no text layer (a scanned document); OCR is not supported"
            )
        text, sections = _join(pages, title)
        return build_document(title=title, text=text, sections=sections, format="pdf")


def _open(data: bytes) -> Any:
    """Open bytes as a PDF. PyMuPDF ships no stubs, so Document is held as Any."""
    try:
        return pymupdf.open(stream=data, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise ParseError(f"cannot read PDF: {type(exc).__name__}") from None


def _page_texts(data: bytes) -> list[str]:
    """One entry per page, empty string for a blank page."""
    document = _open(data)
    try:
        if document.needs_pass:
            raise ParseError("PDF is encrypted and cannot be read")
        return [_page_text(document, number) for number in range(document.page_count)]
    finally:
        document.close()


def _page_text(document: Any, number: int) -> str:
    """Blocks sorted into reading order, joined.

    NFKC is applied because a PDF's text is *constructed* by us, not uploaded bytes:
    presentation-form glyphs map onto the letters a reader actually sees. Measured to
    be a no-op on a PDF whose text layer is already clean, and the difference between
    readable and unreadable Persian on one that is not (D14).
    """
    try:
        blocks = document[number].get_text("blocks", sort=True)
    except Exception as exc:
        raise ParseError(f"cannot read page {number + 1}: {type(exc).__name__}") from None
    parts = [
        unicodedata.normalize("NFKC", block[4]).strip()
        for block in blocks
        if len(block) > 4 and isinstance(block[4], str) and block[4].strip()
    ]
    return "\n".join(parts)


def _join(pages: list[str], title: str) -> tuple[str, list[Section]]:
    """Build the document text and its page sections in one pass.

    Pages are joined with ``\\n``, skipping blanks, so each section's offsets index
    that exact string. One loop keeps text and offsets from drifting apart, which is
    what makes I6 checkable.
    """
    text: list[str] = []
    sections: list[Section] = []
    offset = 0
    for page_number, page in enumerate(pages, start=1):
        if not page:
            continue
        if text:
            offset += 1  # the joining newline before this page
        text.append(page)
        sections.append(
            Section(
                char_start=offset,
                char_end=offset + len(page),
                section_path=(title, f"page {page_number}"),
                page_start=page_number,
                page_end=page_number,
            )
        )
        offset += len(page)
    return "\n".join(text), sections
