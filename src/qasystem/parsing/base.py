"""Shared helpers for parsers (plan.md P2).

The one rule every parser obeys: **offsets index the parser's own text**. Chunks
later slice that text directly, so a wrong offset becomes a wrong citation.
"""

from __future__ import annotations

from qasystem.domain.models import ParsedDocument, Section


def line_starts(text: str) -> list[int]:
    """Char offset where each 1-based line begins. ``line_starts(text)[n-1]`` starts line n."""
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def line_to_char(starts: list[int], line: int, total: int) -> int:
    """Char offset for a 1-based line number, clamped to the text length."""
    if line <= 0:
        return 0
    if line > len(starts):
        return total
    return starts[line - 1]


def line_span_to_char_span(
    starts: list[int], line_start: int, line_end: int, total: int
) -> tuple[int, int]:
    """Half-open char span for an inclusive 1-based line range."""
    return line_to_char(starts, line_start, total), line_to_char(starts, line_end + 1, total)


def build_document(
    *,
    title: str,
    text: str,
    sections: list[Section],
    format: str,
) -> ParsedDocument:
    """Assemble a ParsedDocument, dropping sections whose slice would be blank."""
    kept = [section for section in sections if text[section.char_start : section.char_end].strip()]
    if not text.strip() or not kept:
        from qasystem.errors import EmptyDocumentError

        raise EmptyDocumentError("document has no extractable text")
    return ParsedDocument(
        title=title,
        text=text,
        sections=tuple(kept),
        format=format,  # type: ignore[arg-type]
    )


def stem(filename: str) -> str:
    """Filename without directory or extension, used as a title fallback."""
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base
