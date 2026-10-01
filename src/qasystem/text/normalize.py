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
ZWJ = "\u200d"  # changes meaning in several scripts, so normalize rather than drop

# Arabic letterforms that must fold onto their Persian equivalents so an Arabic
# keyboard or an Arabic corpus still matches Persian queries.
LETTER_FOLD = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی"})  # yeh, kaf, alef maksura

# Arabic-Indic (U+0660) and Extended Arabic-Indic / Persian (U+06F0) digits -> ASCII.
DIGIT_FOLD = {0x0660 + offset: str(offset) for offset in range(10)}
DIGIT_FOLD.update({0x06F0 + offset: str(offset) for offset in range(10)})

# Zero-width, bidi and other invisible format characters are all category "Cf".
# Two are kept because they change meaning: ZWNJ and ZWJ.
_DROP_CATEGORIES = frozenset({"Cf", "Mn", "Me", "Cc"})
_KEEP_INVISIBLE = frozenset({ZWNJ, ZWJ})

TATWEEL = "\u0640"  # ـ


def normalize_for_index(text: str) -> str:
    """NFKC, letter and digit folding, invisible-char removal, whitespace collapse.

    Idempotent: ``normalize_for_index(normalize_for_index(x)) == normalize_for_index(x)``.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(LETTER_FOLD).translate(DIGIT_FOLD)
    text = "".join(
        ch
        for ch in text
        if ch in _KEEP_INVISIBLE
        or (ch != TATWEEL and unicodedata.category(ch) not in _DROP_CATEGORIES)
    )
    return " ".join(text.split())
