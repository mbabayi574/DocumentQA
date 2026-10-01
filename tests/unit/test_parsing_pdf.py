"""P2: PDF parser — pages, blank pages, encrypted/corrupt/scan handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from qasystem.errors import EmptyDocumentError, NoTextLayerError, ParseError
from qasystem.parsing.pdf import PdfParser

parser = PdfParser()
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


def test_persian_pdf_pages_are_numbered_from_one() -> None:
    doc = parser.parse((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")
    assert doc.format == "pdf"
    assert doc.title == "ai-engineer"
    pages = {s.page_start for s in doc.sections} - {None}
    assert min(pages) == 1
    assert max(pages) >= 2


def test_blank_page_does_not_void_the_document() -> None:
    """plan.md 2.3: ai-engineer.pdf page 1 has no text layer; the rest is valid."""
    doc = parser.parse((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")
    assert "پیاده‌سازی" in doc.text.replace(" ", "")


def test_persian_pdf_text_is_readable_not_reversed() -> None:
    """We must not reverse RTL strings; plan.md P2 forbids it without a fixture."""
    doc = parser.parse((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")
    assert "مدل‌هایembedding" in doc.text.replace(" ", "")


def test_section_page_spans_are_consistent() -> None:
    doc = parser.parse((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")
    for section in doc.sections:
        assert section.page_start == section.page_end
        assert section.page_start is not None and section.page_start >= 1


def test_every_section_slice_is_non_empty() -> None:
    doc = parser.parse((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")
    assert all(doc.text[s.char_start : s.char_end].strip() for s in doc.sections)


def test_corrupt_pdf_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        parser.parse(b"%PDF-1.7\nnot really a pdf at all", "broken.pdf")


def test_pdf_without_magic_bytes_is_not_a_pdf() -> None:
    with pytest.raises(ParseError):
        parser.parse(b"just text pretending to be a pdf", "fake.pdf")


def test_scanned_pdf_without_text_layer_raises() -> None:
    """No OCR by design; say so rather than returning empty evidence."""
    with pytest.raises(NoTextLayerError) as exc:
        parser.parse(_blank_pdf(), "scanned.pdf")
    assert "ocr" in str(exc.value).lower()


def test_scanned_pdf_error_says_no_ocr() -> None:
    with pytest.raises(NoTextLayerError, match="no text layer"):
        parser.parse(_blank_pdf(), "scanned.pdf")


def test_zero_bytes_pdf_raises_empty() -> None:
    with pytest.raises((EmptyDocumentError, ParseError)):
        parser.parse(b"", "empty.pdf")


def _blank_pdf() -> bytes:
    """A structurally valid PDF whose single page holds no text."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()
    return doc.tobytes()
