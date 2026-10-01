"""Shared helpers for parsers (plan.md P2).

The one rule every parser obeys: **offsets index the parser's own text**. Chunks
later slice that text directly, so a wrong offset becomes a wrong citation.
"""

from __future__ import annotations

from qasystem.domain.models import ParsedDocument, Section
from qasystem.errors import EmptyDocumentError


def basename(filename: str) -> str:
    """Filename with any directory path stripped."""
    return filename.replace("\\", "/").rsplit("/", 1)[-1]


def stem(filename: str) -> str:
    """Filename without directory or extension; the title fallback."""
    name = basename(filename)
    return name.rsplit(".", 1)[0] if "." in name else name


def extension(filename: str) -> str:
    """Lowercase extension including the dot, or ``""`` when there is none."""
    name = basename(filename)
    return "." + name.rsplit(".", 1)[1].lower() if "." in name else ""


def line_starts(text: str) -> list[int]:
    """Char offset where each 1-based line begins. ``line_starts(text)[n-1]`` starts line n."""
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def line_span_to_char_span(
    starts: list[int], line_start: int, line_end: int, total: int
) -> tuple[int, int]:
    """Half-open char span for an inclusive 1-based line range, clamped to ``total``."""

    def at(line: int) -> int:
        if line <= 0:
            return 0
        return total if line > len(starts) else starts[line - 1]

    return at(line_start), at(line_end + 1)


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span so it excludes surrounding whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def build_document(
    *,
    title: str,
    text: str,
    sections: list[Section],
    format: str,
) -> ParsedDocument:
    """Assemble a ParsedDocument, dropping sections whose slice would be blank."""
    kept = [s for s in sections if text[s.char_start : s.char_end].strip()]
    if not text.strip() or not kept:
        raise EmptyDocumentError("document has no extractable text")
    return ParsedDocument(
        title=title,
        text=text,
        sections=tuple(kept),
        format=format,  # type: ignore[arg-type]
    )
