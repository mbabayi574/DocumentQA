"""Deterministic, heading-aware chunking (plan.md P3).

Two invariants shape this module:

* **Every chunk is a contiguous slice** of ``ParsedDocument.text``. Overlap is an
  earlier start offset, never a re-typed copy, so ``text[char_start:char_end] ==
  chunk.text`` holds by construction and I6 is mechanically checkable.
* **Determinism.** Same input, same config, byte-identical output. P6 relies on this
  for the ``unchanged`` fast path and for embedding-cache reuse.

Size is estimated in characters, not real tokens. No tokenizer is available offline,
so ``CHARS_PER_TOKEN`` is the *densest* value measured against BGE-M3 (§0.5), which
keeps the hard cap safe for the worst case rather than the average one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from qasystem.config import Settings
from qasystem.domain.models import Chunk, ParsedDocument, Section
from qasystem.text.language import detect_language
from qasystem.text.normalize import normalize_for_index

# Measured against Bge-m3 (plan.md §0.5): prose 4.1 chars/token, identifiers 2.27,
# markdown tables 1.45. Planning on the *worst* case keeps the hard cap safe for code
# and tables, at the cost of smaller-than-ideal prose chunks.
# ponytail: a real tokenizer would allow larger chunks; revisit if P9 shows size hurts.
CHARS_PER_TOKEN = 1.5

_TERMINATORS = ".!?؟۔"
_ABBREVIATIONS = frozenset(
    [
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "fig",
        "no",
        "inc",
        "ltd",
        "co",
        "al",
        "eq",
    ]
)


@dataclass(frozen=True)
class Sentence:
    """A sentence with offsets into the string it came from."""

    start: int
    end: int
    text: str


def split_sentences(text: str) -> list[Sentence]:
    """Split on ASCII and Persian terminators, ignoring decimals and abbreviations."""
    sentences: list[Sentence] = []
    start = 0
    for index, char in enumerate(text):
        if char not in _TERMINATORS or (char == "." and _is_false_terminator(text, index)):
            continue
        if text[start : index + 1].strip():
            sentences.append(Sentence(start, index + 1, text[start : index + 1]))
        # The gap after a terminator belongs to no sentence.
        start = index + 1
        while start < len(text) and text[start].isspace():
            start += 1
    if text[start:].strip():
        sentences.append(Sentence(start, len(text), text[start:]))
    return sentences


def _is_false_terminator(text: str, index: int) -> bool:
    """True when the '.' at ``index`` is a decimal point or ends an abbreviation."""
    if index + 1 < len(text) and text[index + 1].isdigit():
        return True
    words = text[:index].split()
    return bool(words) and words[-1].lower() in _ABBREVIATIONS


def chunk_hash(section_path: tuple[str, ...], text: str) -> str:
    """sha256 over the canonical breadcrumb plus the normalized text.

    Normalized, so a whitespace-only edit does not invalidate a cached embedding;
    breadcrumb included, so moving a chunk to another section does.
    """
    canonical = " > ".join(section_path)
    payload = f"{canonical}\x00{normalize_for_index(text)}".encode()
    return hashlib.sha256(payload).hexdigest()


class Chunker:
    """Turns a ``ParsedDocument`` into contiguous, overlapping, hashed chunks."""

    def __init__(self, settings: Settings) -> None:
        self.target_chars = max(1, int(settings.chunk_target_tokens * CHARS_PER_TOKEN))
        self.hard_max_chars = max(1, int(settings.chunk_hard_max_tokens * CHARS_PER_TOKEN))
        self.overlap_chars = int(self.target_chars * settings.chunk_overlap_ratio)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk each section independently, then assign dense ordinals.

        Sections are never merged across different breadcrumbs, so a chunk always
        belongs to exactly one section and a citation's section is unambiguous.
        """
        chunks: list[Chunk] = []
        for section in document.sections:
            for start, end in self._windows(document.text, section):
                text = document.text[start:end]
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        ordinal=len(chunks),
                        char_start=start,
                        char_end=end,
                        text=text,
                        section_path=section.section_path,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        line_start=section.line_start,
                        line_end=section.line_end,
                        chunk_hash=chunk_hash(section.section_path, text),
                        language=detect_language(text),
                    )
                )
        return chunks

    def _windows(self, text: str, section: Section) -> list[tuple[int, int]]:
        """Absolute (start, end) spans covering one whole section.

        Sentences are packed up to the target size; anything still over the hard cap
        is cut mechanically. The second pass matters: a table-of-contents page has no
        sentence terminators at all, so packing alone would leave it as one huge chunk.
        """
        base, body = section.char_start, text[section.char_start : section.char_end]
        if not body.strip():
            return []
        if len(body) <= self.hard_max_chars:
            return [(base, base + len(body))]

        sentences = split_sentences(body) or [Sentence(0, len(body), body)]
        spans: list[tuple[int, int]] = []
        start = end = 0
        for sentence in sentences:
            if end > start and end - start + len(sentence.text) > self.target_chars:
                spans.append((start, end))
                start = max(start, end - self.overlap_chars)
            end = sentence.end
        if end > start:
            spans.append((start, end))

        return [(base + start, base + end) for start, end in self._enforce_cap(spans, len(body))]

    def _enforce_cap(self, spans: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
        """Cut any span over the hard cap, keeping the configured overlap."""
        step = max(1, self.hard_max_chars - self.overlap_chars)
        capped: list[tuple[int, int]] = []
        for start, end in spans:
            if end - start <= self.hard_max_chars:
                capped.append((start, end))
                continue
            cursor = start
            while cursor < end:
                capped.append((cursor, min(cursor + self.hard_max_chars, end)))
                cursor += step
        if capped and capped[-1][1] < length:
            capped.append((capped[-1][0], length))
        return capped
