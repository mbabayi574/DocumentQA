"""Markdown parser (plan.md P2).

Uses ``markdown-it-py`` token line maps rather than regexes, so ATX and Setext
headings both work and a ``#`` inside a fenced code block is never a heading.
The document text is kept byte-for-byte: headings, fences and odd spacing
survive, because citations quote it.
"""

from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

from qasystem.domain.models import ParsedDocument, Section
from qasystem.parsing.base import build_document, line_span_to_char_span, line_starts, stem


class MarkdownParser:
    """Flattens Markdown into one source text plus breadcrumb-tagged sections."""

    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark")

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        text = data.decode("utf-8-sig")
        starts = line_starts(text)
        title = stem(filename)
        sections, first_heading = _walk(self._md.parse(text), text, starts, title)
        return build_document(
            title=first_heading or title,
            text=text,
            sections=sections,
            format="md",
        )


def _walk(
    tokens: list[Token], text: str, starts: list[int], fallback_title: str
) -> tuple[list[Section], str | None]:
    """Walk block tokens, tracking the heading stack to build breadcrumbs."""
    sections: list[Section] = []
    stack: list[str] = []
    first_heading: str | None = None

    for index, token in enumerate(tokens):
        if token.type == "heading_open" and token.map:
            level = int(token.tag[1:])
            heading = _heading_text(tokens[index + 1])
            del stack[level - 1 :]
            stack.append(heading)
            first_heading = first_heading or heading
            continue

        # ``inline`` carries the prose of its parent block and ``heading_close`` has
        # no map; both fold into the block token that precedes them. ``hr`` is pure
        # markup: a section made of "---" would become a junk chunk.
        if not token.map or token.type == "inline" or token.tag == "hr":
            continue

        char_start, char_end = line_span_to_char_span(
            starts, token.map[0] + 1, token.map[1], len(text)
        )
        char_start, char_end = _trim(text, char_start, char_end)
        if char_start >= char_end:
            continue
        sections.append(
            Section(
                char_start=char_start,
                char_end=char_end,
                section_path=tuple(stack) or (first_heading or fallback_title,),
                line_start=token.map[0] + 1,
                line_end=token.map[1],
            )
        )
    return sections, first_heading


def _heading_text(inline_token: Token) -> str:
    """Plain text of the inline token that follows a heading_open."""
    parts = [
        child.content
        for child in (inline_token.children or [])
        if child.type in ("text", "code_inline")
    ]
    heading = " ".join(part.strip() for part in parts if part.strip())
    return heading or "Untitled"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
