"""Index/query string normalization (plan.md P1).

**Stored and cited text is never normalized.** Only index and query strings pass
through here, so a citation always points at the exact bytes the user uploaded.
Pure functions, no I/O. Invisible characters are written as escapes on purpose:
they are invisible in an editor, which is exactly how they get lost.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Written as escapes: these characters are invisible in an editor, which is
# exactly how they get silently lost from a source file.
ZWNJ = "\u200c"  # keeps می‌رود as one word
ZWJ = "\u200d"  # changes meaning in several scripts, so normalize rather than drop

# Arabic letterforms that must fold onto their Persian equivalents so an Arabic
# keyboard or an Arabic corpus still matches Persian queries.
LETTER_FOLD = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی"})  # yeh, kaf, alef maksura

# Arabic-Indic (U+0660) and Extended Arabic-Indic / Persian (U+06F0) digits -> ASCII.
DIGIT_FOLD = {0x0660 + offset: str(offset) for offset in range(10)}
DIGIT_FOLD.update({0x06F0 + offset: str(offset) for offset in range(10)})

# Zero-width, bidi and other invisible format characters are all category "Cf".
# Two are kept because they change meaning: ZWNJ and ZWJ.
#
# "Cc" (control characters) is deliberately NOT in this set, even though it holds the
# invisible ones. It also holds newline and tab, which *separate* words: deleting them
# fuses the last word of one line onto the first word of the next: the Persian bullet list
# "processing\ndocuments" indexed as one token "processingdocuments", matching no query (D37).
_DROP_CATEGORIES = frozenset({"Cf", "Mn", "Me"})
_KEEP_INVISIBLE = frozenset({ZWNJ, ZWJ})

TATWEEL = "\u0640"  # ـ

# Where Latin letters live. A *script* boundary is a word boundary: the Persian PDFs in this
# corpus write "models-embedding-below" and "documentation-OpenAPI-with" with no space at all,
# so without this the English word becomes part of one giant token and is unmatchable by any
# query -- which defeats the cross-lingual claim and starves the gate's coverage signal (D45).
_LATIN_RANGES = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))
_SCRIPT_TRANSITION = re.compile(r"([A-Za-z\u00C0-\u024F])")


def normalize_for_index(text: str) -> str:
    """NFKC, letter and digit folding, invisible-char removal, whitespace collapse.

    Idempotent: ``normalize_for_index(normalize_for_index(x)) == normalize_for_index(x)``.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(LETTER_FOLD).translate(DIGIT_FOLD)
    kept: list[str] = []
    for char in text:
        if char in _KEEP_INVISIBLE:
            kept.append(char)
        elif unicodedata.category(char) == "Cc":
            kept.append(" ")  # a control character is whitespace, not nothing (D37)
        elif char != TATWEEL and unicodedata.category(char) not in _DROP_CATEGORIES:
            if kept and _is_script_boundary(kept[-1], char):
                kept.append(" ")
            kept.append(char)
    return " ".join("".join(kept).split())


def _is_script_boundary(previous: str, char: str) -> bool:
    """True where a Latin letter touches a non-Latin letter, and vice versa.

    Letters only, never digits or punctuation, so ``ERR-404``, ``v2.3.1``, ``bge-m3`` and
    ``U+06F0`` folding are all unaffected -- they are one identifier, not two words.
    """
    if not previous or not (previous.isalpha() and char.isalpha()):
        return False
    return _is_latin(previous) is not _is_latin(char)


@lru_cache(maxsize=512)
def _is_latin(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _LATIN_RANGES)
