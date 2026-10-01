"""P2: PDF parser — page numbers, blank pages, encrypted/corrupt/scan handling.

Fixture-wide coverage (every PDF in tests/fixtures/docs/) lives in
test_parsing_pdf_fixtures.py; this file covers behaviour only reachable with a
synthetic PDF.
"""

from __future__ import annotations

import pytest

from qasystem.errors import EmptyDocumentError, NoTextLayerError, ParseError
from qasystem.parsing.pdf import PdfParser

parser = PdfParser()


def test_corrupt_pdf_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        parser.parse(b"%PDF-1.7\nnot really a pdf at all", "broken.pdf")


def test_pdf_without_magic_bytes_is_not_a_pdf() -> None:
    with pytest.raises(ParseError):
        parser.parse(b"just text pretending to be a pdf", "fake.pdf")


def test_scanned_pdf_without_text_layer_raises() -> None:
    """No OCR by design; the message must say so rather than return empty evidence."""
    with pytest.raises(NoTextLayerError, match="no text layer") as exc:
        parser.parse(_blank_pdf(), "scanned.pdf")
    assert "ocr" in str(exc.value).lower()


def test_zero_bytes_pdf_raises_empty() -> None:
    with pytest.raises((EmptyDocumentError, ParseError)):
        parser.parse(b"", "empty.pdf")


def _blank_pdf() -> bytes:
    """A structurally valid PDF whose single page holds no text."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()
    return doc.tobytes()
