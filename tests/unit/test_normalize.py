"""P1: normalize_for_index — index/query strings only, never stored text."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from qasystem.text.normalize import normalize_for_index

ARABIC_YEH = "ي"
ARABIC_KEHEH = "ك"
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ZWNJ = "\u200c"
TATWEEL = "\u0640"
FATHA = "َ"
ZWSP = "\u200b"


def test_arabic_letters_fold_to_persian() -> None:
    assert normalize_for_index(ARABIC_KEHEH) == "ک"
    assert normalize_for_index(ARABIC_YEH) == "ی"


def test_arabic_and_persian_digits_become_ascii() -> None:
    assert normalize_for_index(ARABIC_INDIC_DIGITS) == "0123456789"
    assert normalize_for_index(PERSIAN_DIGITS) == "0123456789"


def test_zwnj_is_kept_not_deleted() -> None:
    """می‌رود is one word: deleting ZWNJ would fuse unrelated tokens."""
    assert normalize_for_index("می‌رود") == "می‌رود"


def test_zwnj_variant_matches_spaced_query() -> None:
    """A query typed with a plain space must match a document holding ZWNJ."""
    assert normalize_for_index("می رود") == normalize_for_index("می‌رود").replace(ZWNJ, " ")


def test_tatweel_and_diacritics_are_stripped() -> None:
    assert normalize_for_index("کتابـهاَ") == "کتابها"


def test_control_and_zero_width_chars_are_dropped() -> None:
    assert ZWSP not in normalize_for_index("سلام")
    assert "\x00" not in normalize_for_index("سلام")


def test_whitespace_is_collapsed() -> None:
    assert normalize_for_index("a   b\t\n c") == "a b c"


def test_english_only_changes_whitespace() -> None:
    assert normalize_for_index("Hello   World") == "Hello World"


def test_nfkc_folds_presentation_forms() -> None:
    """Ligature presentation forms must fold so typed text matches indexed text."""
    assert normalize_for_index("ﻣﻘﺪﻣﻪ") == normalize_for_index("مقدمه")


@given(st.text(max_size=200))
def test_is_idempotent(text: str) -> None:
    once = normalize_for_index(text)
    assert normalize_for_index(once) == once


@given(st.text(max_size=200))
def test_output_has_no_control_or_zero_width_chars(text: str) -> None:
    cleaned = normalize_for_index(text)
    assert not any(ch in cleaned for ch in "\x00\r\n\t")
    assert ZWSP not in cleaned
