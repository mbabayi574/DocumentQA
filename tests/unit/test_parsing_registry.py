"""P2: registry — format selection, magic bytes, size limit before parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from qasystem.errors import EmptyDocumentError, FileTooLargeError, UnsupportedFormatError
from qasystem.parsing.registry import ParserRegistry

registry = ParserRegistry(max_upload_mb=1)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


def test_md_extension_dispatches_to_markdown() -> None:
    assert registry.parse(b"# Guide\n\nbody", "guide.md").format == "md"


def test_txt_extension_dispatches_to_text() -> None:
    assert registry.parse(b"plain body", "notes.txt").format == "txt"


def test_pdf_extension_dispatches_to_pdf() -> None:
    data = (FIXTURES / "fa" / "ai-engineer.pdf").read_bytes()
    assert registry.parse(data, "sample.pdf").format == "pdf"


def test_pdf_magic_bytes_win_over_a_lying_extension() -> None:
    """A .txt upload that is really a PDF must still parse as a PDF."""
    data = (FIXTURES / "fa" / "ai-engineer.pdf").read_bytes()
    assert registry.parse(data, "sneaky.txt").format == "pdf"


def test_wrong_magic_bytes_for_pdf_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedFormatError):
        registry.parse(b"this is plain text, not a pdf", "claims-to-be.pdf")


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedFormatError) as exc:
        registry.parse(b"<html></html>", "page.html")
    assert "html" in str(exc.value)


def test_extension_is_case_insensitive() -> None:
    assert registry.parse(b"# G\n\nbody", "GUIDE.MD").format == "md"


def test_size_limit_is_enforced_before_parsing() -> None:
    """Over the limit must be rejected without ever decoding the body."""
    with pytest.raises(FileTooLargeError):
        registry.parse(b"x" * (1024 * 1024 + 1), "big.txt")


def test_exactly_at_the_limit_is_allowed() -> None:
    assert registry.parse(b"y" * (1024 * 1024), "exact.txt").text


def test_empty_document_is_rejected() -> None:
    with pytest.raises(EmptyDocumentError):
        registry.parse(b"", "empty.txt")


def test_registry_satisfies_the_document_parser_port() -> None:
    from qasystem.domain.ports import DocumentParser

    assert isinstance(registry, DocumentParser)
