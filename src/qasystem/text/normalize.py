"""Index/query string normalization (plan.md P1).

**Stored and cited text is never normalized.** Only index and query strings pass
through here, so a citation always points at the exact bytes the user uploaded.
Pure functions, no I/O. Invisible characters are written as escapes on purpose:
they are invisible in an editor, which is exactly how they get lost.
"""

from __future__ import annotations

import unicodedata

# Written as escapes: these characters are invisible in an editor, which is
# exactly how they get silently lost from a source file.
ZWNJ = "\u200c"  # keeps می‌رود as one word
ZWSP = "\u200b"  # zero width space, always dropped

# Arabic letterforms that must fold onto their Persian equivalents so an Arabic
# keyboard or an Arabic corpus still matches Persian queries.
LETTER_FOLD = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی"})  # yeh, kaf, alef maksura

# Arabic-Indic (U+0660) and Extended Arabic-Indic / Persian (U+06F0) digits -> ASCII.
DIGIT_FOLD = {0x0660 + offset: str(offset) for offset in range(10)}
DIGIT_FOLD.update({0x06F0 + offset: str(offset) for offset in range(10)})

# Zero-width and bidi control characters. ZWJ (U+200D) is deliberately absent:
# it changes meaning in several scripts, so it is normalized rather than deleted.
INVISIBLE_DROP = frozenset(
    {
        0x00AD,  # soft hyphen
        0x200B,  # zero width space
        0x200E,  # left-to-right mark
        0x200F,  # right-to-left mark
        0x202A,  # bidi embedding
        0x202B,
        0x202C,
        0x202D,
        0x202E,  # bidi override
        0x2066,  # bidi isolates
        0x2067,
        0x2068,
        0x2069,
        0xFEFF,  # BOM / zero width no-break space
    }
)

TATWEEL = "ـ"  # ـ

# Combining marks: vowel marks and diacritics carry no retrieval signal.
_MARKS = frozenset({"Mn", "Me"})
# Control codes, including NUL and DEL.
_CONTROL = frozenset({"Cc"})


def normalize_for_index(text: str) -> str:
    """NFKC, letter and digit folding, invisible-char removal, whitespace collapse.

    Idempotent: ``normalize_for_index(normalize_for_index(x)) == normalize_for_index(x)``.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(LETTER_FOLD).translate(DIGIT_FOLD)
    text = "".join(ch for ch in text if ord(ch) not in INVISIBLE_DROP and ch != TATWEEL)
    # Control codes (Cc) carry no retrieval signal and would corrupt FTS5 output.
    text = "".join(ch for ch in text if unicodedata.category(ch) not in _MARKS | _CONTROL)
    return " ".join(text.split())
