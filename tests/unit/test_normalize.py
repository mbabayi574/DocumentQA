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


# ---------------------------------------------------------------- D37


def test_a_newline_separates_words_instead_of_fusing_them() -> None:
    """Control characters are whitespace, not nothing.

    Found by P7's Persian-PDF citation test: the document's bullet list indexed as
    ``پردازشاسناد`` because ``\\n`` was in the drop set, so the words matched no query at
    all and the only correct chunk in the document scored zero coverage.
    """
    assert normalize_for_index("پردازش\nاسناد") == "پردازش اسناد"
    assert normalize_for_index("a\nb") == "a b"
    assert normalize_for_index("a\r\nb") == "a b"


def test_every_kind_of_whitespace_collapses_the_same_way() -> None:
    for separator in (" ", "\n", "\r\n", "\t", "\x0b", "\x0c", "  \n  "):
        assert normalize_for_index(f"alpha{separator}beta") == "alpha beta"


def test_invisible_control_characters_become_separators_not_disappearances() -> None:
    """NUL and BEL are dropped as far as the eye is concerned, but they still separate."""
    assert normalize_for_index("a\x00b\x07c") == "a b c"
    assert normalize_for_index("a\u2028b") == "a b", "a line separator is whitespace too"


def test_normalization_is_idempotent_across_line_layouts() -> None:
    layouts = {
        "one\ntwo\nthree": "one two three",
        "one\r\ntwo": "one two",
        "one two": "one two",
        "one\n\n\ntwo": "one two",
    }
    for text, expected in layouts.items():
        once = normalize_for_index(text)
        assert once == expected
        assert normalize_for_index(once) == once


def test_the_zwnj_and_line_break_interaction_is_still_exact() -> None:
    """A ZWNJ keeps its word together while a newline still ends it."""
    assert normalize_for_index("می\u200cرود\nمی رود") == "می\u200cرود می رود"


def test_tokenizing_across_a_line_break_yields_two_tokens() -> None:
    from qasystem.text.tokenize import tokenize

    assert tokenize("●پردازش\nاسناد: پشتیبانی")[-3:] == ["پردازش", "اسناد", "پشتیبانی"]


def test_the_chunk_hash_ignores_line_layout() -> None:
    """P3 documents the hash as whitespace-insensitive; that is only true once \\n survives."""
    from qasystem.chunking.chunker import chunk_hash

    assert chunk_hash(("s",), "alpha\nbeta") == chunk_hash(("s",), "alpha  beta")
    assert chunk_hash(("s",), "alpha\nbeta") != chunk_hash(("s",), "alphabeta")
