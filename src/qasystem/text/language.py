"""Script-ratio language detection (plan.md P1).

Deliberately simple: the only decision that matters downstream is which language to
label a chunk, and BGE-M3 retrieves across languages regardless (plan.md §2.2a), so
nothing routes on this value.
"""

from __future__ import annotations

from qasystem.domain.models import Language
from qasystem.text.normalize import _LATIN_RANGES  # one definition, shared with the tokenizer

_FA_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
# Below this share of letters, treat a document as not meaningfully in that script.
_MIN_RATIO = 0.2


def _count(text: str, ranges: tuple[tuple[int, int], ...]) -> int:
    return sum(1 for ch in text if any(start <= ord(ch) <= end for start, end in ranges))


def detect_language(text: str) -> Language:
    """``fa`` or ``en`` for a single-script document, ``mixed`` when both are present."""
    fa = _count(text, _FA_RANGES)
    latin = _count(text, _LATIN_RANGES)
    total = fa + latin
    if total == 0:
        return "en"

    fa_ratio, latin_ratio = fa / total, latin / total
    if fa_ratio >= _MIN_RATIO and latin_ratio >= _MIN_RATIO:
        return "mixed"
    return "fa" if fa_ratio >= _MIN_RATIO else "en"
