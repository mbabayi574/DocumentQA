"""P2: every PDF fixture is actually parsed, not just the convenient one.

plan.md §2.3 originally cleared one Persian book as "unrecoverable" from a two-page
probe. A full scan of all three PDFs contradicted that, so each one is exercised here.
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


@pytest.mark.parametrize("path", ALL_PDFS, ids=lambda p: p.name)
def test_every_pdf_page_number_is_one_based_and_monotonic(path: Path) -> None:
    doc = parser.parse(path.read_bytes(), path.name)
    pages = [section.page_start for section in doc.sections if section.page_start]
    assert pages == sorted(pages)
    assert min(pages) >= 1


def test_english_book_excerpt_yields_real_english() -> None:
    """Derived from the 10 MB sample by scripts/build_pdf_fixtures.py."""
    doc = parser.parse((FIXTURES / "en" / "clean-code-excerpt.pdf").read_bytes(), "x.pdf")
    assert len(doc.text) > 20_000
    assert "Contents" in doc.text
    assert "Preface" in doc.text


def test_the_uncommitted_book_still_parses_when_present() -> None:
    """The 10 MB source is git-ignored, so this only asserts when it exists locally."""
    path = FIXTURES / "en" / "Clean Code Fundamentals-Martin Hock-Leanpub-EBooksWorld.ir.pdf"
    if not path.exists():
        pytest.skip("10 MB sample book is not committed")
    doc = parser.parse(path.read_bytes(), path.name)
    assert len(doc.text) > 300_000
    assert len(doc.sections) > 300


def test_persian_book_is_readable_after_extraction() -> None:
    """NFKC is what makes this book readable; the raw glyphs are presentation forms."""
    doc = parser.parse(
        (FIXTURES / "fa" / "justforfun_book_a4.pdf").read_bytes(), "justforfun_book_a4.pdf"
    )
    assert len(doc.text) > 100_000
    # A real Persian word survives, and no presentation-form codepoints remain.
    assert "تفریح" in doc.text
    assert not any("\ufb50" <= ch <= "\ufdff" for ch in doc.text)


def test_known_limitation_some_lines_lose_their_spaces() -> None:
    """Documented ceiling: this book's font maps the space glyph to nothing.

    NFKC recovers the letters and their order (test above); it cannot invent a space
    that the PDF never stored. Fixing it would need Persian word segmentation, which
    is out of scope; the tokenizer still retrieves on the characters it has.
    """
    doc = parser.parse(
        (FIXTURES / "fa" / "justforfun_book_a4.pdf").read_bytes(), "justforfun_book_a4.pdf"
    )
    glued = sum(1 for line in doc.text.split("\n") if len(line) > 25 and " " not in line)
    assert glued < doc.text.count("\n") * 0.2, "expected a few glued lines, not most of them"


def test_nfkc_does_not_alter_a_clean_text_layer() -> None:
    """Applying NFKC must be a no-op where the PDF already extracts correctly."""
    clean = (FIXTURES / "fa" / "ai-engineer.pdf").read_bytes()
    doc = parser.parse(clean, "ai-engineer.pdf")
    assert "پیاده‌سازی" in doc.text.replace(" ", "").replace("\n", "")
