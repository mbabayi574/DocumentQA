"""Plain-text parser (plan.md P2).

Only ``\\r\\n -> \\n`` is normalized. Nothing else about the text changes, because
citations quote it verbatim.
"""

from __future__ import annotations

from qasystem.domain.models import ParsedDocument, Section
from qasystem.errors import EmptyDocumentError, ParseError
from qasystem.parsing.base import build_document, line_span_to_char_span, line_starts, stem


class TextParser:
    """Decodes UTF-8 (with optional BOM) and splits on blank lines."""

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        text = _decode(data)
        starts = line_starts(text)
        sections = [
            _paragraph_section(text, starts, block_start, block_end, stem(filename))
            for block_start, block_end in _blocks(text, len(starts))
        ]
        return build_document(title=stem(filename), text=text, sections=sections, format="txt")


def _decode(data: bytes) -> str:
    """UTF-8-SIG then UTF-8. Anything else is a typed error, not mojibake.

    ``\\r\\n`` becomes ``\\n`` and nothing else changes, so line offsets are stable
    across platforms.
    """
    if not data.strip():
        raise EmptyDocumentError("text file is empty")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError(
            f"file is not valid UTF-8 at byte {exc.start}; convert it to UTF-8"
        ) from None
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _blocks(text: str, total_lines: int) -> list[tuple[int, int]]:
    """Inclusive 1-based (first_line, last_line) for each blank-line-separated block."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            start = line_no if start is None else start
        elif start is not None:
            spans.append((start, line_no - 1))
            start = None
    if start is not None:
        spans.append((start, total_lines))
    return spans


def _paragraph_section(
    text: str, starts: list[int], line_start: int, line_end: int, title: str
) -> Section:
    char_start, char_end = line_span_to_char_span(starts, line_start, line_end, len(text))
    # Trim surrounding whitespace so a citation quotes the paragraph, not its blank line.
    while char_start < char_end and text[char_start].isspace():
        char_start += 1
    while char_end > char_start and text[char_end - 1].isspace():
        char_end -= 1
    return Section(
        char_start=char_start,
        char_end=char_end,
        section_path=(title,),
        line_start=line_start,
        line_end=line_end,
    )
