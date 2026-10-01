"""P2: Markdown parser — breadcrumbs, fences, Setext, offset fidelity."""

from __future__ import annotations

import pytest

from qasystem.errors import EmptyDocumentError
from qasystem.parsing.markdown import MarkdownParser

parser = MarkdownParser()


def parse(src: str, name: str = "guide.md"):
    return parser.parse(src.encode(), name)


def test_atx_headings_become_breadcrumbs() -> None:
    doc = parse("# Guide\n\nintro\n\n## Install\n\nbody\n\n### Linux\n\nsteps")
    paths = {s.section_path for s in doc.sections}
    assert ("Guide",) in paths
    assert ("Guide", "Install") in paths
    assert ("Guide", "Install", "Linux") in paths


def test_setext_headings_become_breadcrumbs() -> None:
    doc = parse("Title\n=====\n\nintro\n\nSub\n---\n\nbody")
    paths = {s.section_path for s in doc.sections}
    assert ("Title",) in paths
    assert ("Title", "Sub") in paths


def test_horizontal_rules_are_not_sections() -> None:
    """A '---' separator is markup, not prose; it must not become a junk chunk."""
    doc = parse("# Guide\n\nintro\n\n---\n\nmore body")
    assert all(doc.text[s.char_start : s.char_end].strip() != "---" for s in doc.sections)


def test_fenced_code_is_kept_as_content() -> None:
    """Code blocks are real content: config examples must be retrievable."""
    doc = parse("# Guide\n\nintro\n\n```yaml\nlimit: 100\n```")
    assert any("limit: 100" in doc.text[s.char_start : s.char_end] for s in doc.sections)


def test_heading_inside_fenced_code_is_ignored() -> None:
    """A '#' comment inside a code fence must not become a section."""
    doc = parse("# Real\n\ntext\n\n```python\n# fake heading\n```\n\nmore")
    paths = {s.section_path for s in doc.sections}
    assert paths == {("Real",)}
    assert "# fake heading" in doc.text


def test_intro_before_first_heading_uses_document_title_as_root() -> None:
    doc = parse("# The Guide\n\nintro paragraph\n\n## Install\n\nbody")
    intro = doc.sections[0]
    assert intro.section_path == ("The Guide",)
    assert "intro paragraph" in doc.text[intro.char_start : intro.char_end]


def test_document_title_is_the_first_heading() -> None:
    assert parse("# The Guide\n\nbody").title == "The Guide"


def test_title_falls_back_to_filename_without_heading() -> None:
    assert parse("just body text", "readme.md").title == "readme"


def test_line_spans_are_one_based() -> None:
    doc = parse("# Guide\n\nintro\n\n## Install\n\nbody")
    by_path = {s.section_path: s for s in doc.sections}
    assert (by_path[("Guide",)].line_start, by_path[("Guide",)].line_end) == (3, 3)
    assert (by_path[("Guide", "Install")].line_start, by_path[("Guide", "Install")].line_end) == (
        7,
        7,
    )


def test_offsets_index_the_document_text_exactly() -> None:
    """I6 depends on this: every section slice must be a real slice of text."""
    src = "# Guide\n\nintro para\n\n## Install\n\ninstall body\n\n### Linux\n\nlinux steps"
    doc = parse(src)
    for section in doc.sections:
        assert doc.text[section.char_start : section.char_end].strip()


def test_original_source_bytes_are_preserved() -> None:
    """Stored text is never normalized, so headings and fences survive verbatim."""
    src = "# Guide\n\nkeep   my   spacing\tand\ttabs\n"
    doc = parse(src)
    assert doc.text == src


def test_persian_markdown_and_zwnj_survive() -> None:
    src = "# راهنما\n\nمتن آزمایشی\n\n## نصب\n\nمرحله اول"
    doc = parse(src)
    assert doc.text == src
    assert ("راهنما", "نصب") in {s.section_path for s in doc.sections}


def test_empty_markdown_raises() -> None:
    with pytest.raises(EmptyDocumentError):
        parse("   \n\n")
