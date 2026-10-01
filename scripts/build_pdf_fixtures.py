"""One-time fixture builder: derive a small English PDF from the 10 MB sample book.

The book is too large to commit or to parse in every test run, but its text layer is
clean (348 204 chars, zero mojibake, plan.md §2.3). So we extract a handful of pages
once and commit the result as a small, deterministic fixture.

Run manually:  uv run python scripts/build_pdf_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

SOURCE = Path("tests/fixtures/docs/en/Clean Code Fundamentals-Martin Hock-Leanpub-EBooksWorld.ir.pdf")
TARGET = Path("tests/fixtures/docs/en/clean-code-excerpt.pdf")
PAGES = 12


def main() -> None:
    source = pymupdf.open(SOURCE)
    out = pymupdf.open()
    picked = 0
    for number in range(source.page_count):
        text = source[number].get_text("text").strip()
        # Skip the cover and near-empty pages; keep prose-dense pages only.
        if len(text) < 800:
            continue
        out.insert_pdf(source, from_page=number, to_page=number)
        picked += 1
        if picked == PAGES:
            break
    source.close()
    TARGET.write_bytes(out.tobytes(garbage=4, deflate=True))
    out.close()
    print(f"wrote {TARGET} ({TARGET.stat().st_size // 1024} KB, {picked} pages)")


if __name__ == "__main__":
    main()