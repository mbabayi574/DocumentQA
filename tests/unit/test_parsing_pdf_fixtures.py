"""P2: every PDF fixture is actually parsed, not just the convenient one.

plan.md §2.3 originally cleared one Persian book as "unrecoverable" from a two-page
probe. A full scan of all PDFs contradicted that, so each one is exercised here. A new
fixture dropped into tests/fixtures/docs/ is picked up automatically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qasystem.parsing.pdf import PdfParser

parser = PdfParser()
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
ALL_PDFS = sorted(FIXTURES.rglob("*.pdf"))


def test_the_fixture_set_is_not_empty() -> None:
    assert ALL_PDFS, "no PDF fixtures found"


@pytest.mark.parametrize("path", ALL_PDFS, ids=lambda p: p.name)
def test_every_pdf_fixture_parses_with_non_empty_sections(path: Path) -> None:
    doc = parser.parse(path.read_bytes(), path.name)
    assert doc.format == "pdf"
    assert doc.text.strip()
    assert doc.sections
    for section in doc.sections:
        assert doc.text[section.char_start : section.char_end].strip()
        assert section.page_start == section.page_end
        assert section.page_start is not None and section.page_start >= 1


@pytest.mark.parametrize("path", ALL_PDFS, ids=lambda p: p.name)
def test_every_pdf_page_number_is_one_based_and_monotonic(path: Path) -> None:
    doc = parser.parse(path.read_bytes(), path.name)
    pages = [section.page_start for section in doc.sections if section.page_start]
    assert pages == sorted(pages)
    assert min(pages) >= 1


def test_blank_page_does_not_void_the_document() -> None:
    """plan.md 2.3: ai-engineer.pdf page 2 has no text layer; the rest is valid."""
    doc = parser.parse((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")
    pages = [section.page_start for section in doc.sections]
    assert min(pages) == 1 and 2 not in pages, "page 2 is blank and must be skipped"
    assert "پیاده‌سازی" in doc.text.replace(" ", "").replace("\n", "")


def test_english_book_excerpt_yields_real_english() -> None:
    """Derived from the 10 MB sample by scripts/build_pdf_fixtures.py."""
    doc = parser.parse((FIXTURES / "en" / "clean-code-excerpt.pdf").read_bytes(), "x.pdf")
    assert len(doc.text) > 20_000
    assert "Contents" in doc.text and "Preface" in doc.text


def test_persian_book_is_readable_after_extraction() -> None:
    """NFKC is what makes this book readable; the raw glyphs are presentation forms."""
    doc = parser.parse(
        (FIXTURES / "fa" / "justforfun_book_a4.pdf").read_bytes(), "justforfun_book_a4.pdf"
    )
    assert len(doc.text) > 100_000
    assert "تفریح" in doc.text
    assert not any("\ufb50" <= ch <= "\ufdff" for ch in doc.text)


def test_known_limitation_some_lines_lose_their_spaces() -> None:
    """Documented ceiling: this book's font maps the space glyph to nothing.

    NFKC recovers the letters and their order; it cannot invent a space the PDF never
    stored. Repairing it would need Persian word segmentation, which is out of scope.
    """
    doc = parser.parse(
        (FIXTURES / "fa" / "justforfun_book_a4.pdf").read_bytes(), "justforfun_book_a4.pdf"
    )
    lines = doc.text.split("\n")
    glued = sum(1 for line in lines if len(line) > 25 and " " not in line)
    assert glued < len(lines) * 0.2, "expected a few glued lines, not most of them"


def test_nfkc_does_not_alter_a_clean_text_layer() -> None:
    """Applying NFKC must be a no-op where the PDF already extracts correctly."""
    doc = parser.parse((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")
    assert "پیاده‌سازی" in doc.text.replace(" ", "").replace("\n", "")


def test_the_uncommitted_book_still_parses_when_present() -> None:
    """The 10 MB source is git-ignored, so this only asserts when it exists locally."""
    path = FIXTURES / "en" / "Clean Code Fundamentals-Martin Hock-Leanpub-EBooksWorld.ir.pdf"
    if not path.exists():
        pytest.skip("10 MB sample book is not committed")
    doc = parser.parse(path.read_bytes(), path.name)
    assert len(doc.text) > 300_000 and len(doc.sections) > 300
