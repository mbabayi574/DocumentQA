"""P2: TXT parser — UTF-8-SIG/UTF-8 decoding, blank-line sections, line spans."""

from __future__ import annotations

import pytest

from qasystem.errors import EmptyDocumentError, ParseError
from qasystem.parsing.text import TextParser

parser = TextParser()


def test_sections_split_on_blank_lines() -> None:
    doc = parser.parse(b"First para.\n\nSecond para.\n\nThird para.", "notes.txt")
    assert [s.section_path for s in doc.sections] == [("notes",)] * 3
    assert [doc.text[s.char_start : s.char_end] for s in doc.sections] == [
        "First para.",
        "Second para.",
        "Third para.",
    ]


def test_line_spans_are_one_based_and_inclusive() -> None:
    doc = parser.parse(b"First.\n\nSecond.", "notes.txt")
    assert (doc.sections[0].line_start, doc.sections[0].line_end) == (1, 1)
    assert (doc.sections[1].line_start, doc.sections[1].line_end) == (3, 3)


def test_multiline_paragraph_spans_every_line() -> None:
    doc = parser.parse(b"one\ntwo\nthree\n\nnext", "notes.txt")
    assert (doc.sections[0].line_start, doc.sections[0].line_end) == (1, 3)


def test_crlf_is_normalized_to_lf() -> None:
    """Only \r\n -> \n; everything else about the text is preserved verbatim."""
    doc = parser.parse(b"First.\r\n\r\nSecond.", "notes.txt")
    assert "\r" not in doc.text
    assert [doc.text[s.char_start : s.char_end] for s in doc.sections] == [
        "First.",
        "Second.",
    ]


def test_utf8_bom_is_stripped() -> None:
    doc = parser.parse("سلام دنیا".encode("utf-8-sig"), "fa.txt")
    assert doc.text == "سلام دنیا"
    assert not doc.text.startswith("﻿")


def test_persian_text_is_preserved_byte_for_byte() -> None:
    raw = "پشتیبانی از فرمت‌های PDF و Markdown".encode()
    doc = parser.parse(raw, "fa.txt")
    assert doc.text == "پشتیبانی از فرمت‌های PDF و Markdown"


def test_title_comes_from_filename_stem() -> None:
    assert parser.parse(b"body", "release-notes.txt").title == "release-notes"


def test_format_is_txt() -> None:
    assert parser.parse(b"body", "notes.txt").format == "txt"


def test_whitespace_only_document_is_empty() -> None:
    with pytest.raises(EmptyDocumentError):
        parser.parse(b"   \n\n\t\n", "blank.txt")


def test_zero_bytes_is_empty() -> None:
    with pytest.raises(EmptyDocumentError):
        parser.parse(b"", "empty.txt")


def test_undecodable_bytes_raise_parse_error() -> None:
    """No silent mojibake: a typed error beats a garbled citation."""
    with pytest.raises(ParseError):
        parser.parse(b"\xff\xfe\x00\x81broken latin-1 \x81", "broken.txt")


def test_every_section_slice_is_non_empty() -> None:
    doc = parser.parse(b"one\n\ntwo\n\nthree", "notes.txt")
    assert all(doc.text[s.char_start : s.char_end].strip() for s in doc.sections)
