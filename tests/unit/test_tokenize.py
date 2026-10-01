"""P1: tokenize + stopwords + language detection."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from qasystem.text.language import detect_language
from qasystem.text.tokenize import fts_query_terms, is_stopword, tokenize


def test_lowercases_latin_and_keeps_persian() -> None:
    assert tokenize("Hello World") == ["hello", "world"]
    assert tokenize("کتاب بزرگ") == ["کتاب", "بزرگ"]


def test_zwnj_compound_emits_both_forms() -> None:
    """می‌رود must yield the compound *and* its parts so recall survives either spelling."""
    tokens = tokenize("می‌رود")
    assert "می‌رود" in tokens
    assert {"می", "رود"} <= set(tokens)


def test_identifiers_are_preserved_whole() -> None:
    assert "err-404" in tokenize("Request ERR-404 returned")
    assert "v2.3.1" in tokenize("Upgrade to v2.3.1 today")


def test_numbers_survive_as_tokens() -> None:
    assert "1024" in tokenize("dimension is 1024")


def test_mixed_script_text_yields_both_languages() -> None:
    tokens = tokenize("Use the tokenizer برای متن فارسی")
    assert {"use", "tokenizer"} <= set(tokens)
    assert {"برای", "متن"} <= set(tokens)


def test_tokens_are_unique_and_non_empty() -> None:
    tokens = tokenize("sqlite sqlite  sqlite")
    assert tokens == ["sqlite"]
    assert all(token.strip() for token in tokens)


def test_fts_query_terms_are_quoted_and_joined_with_or() -> None:
    """User text must never reach FTS5 as raw syntax (plan.md §6)."""
    terms = fts_query_terms('sqlite " OR * ( ) NEAR')
    assert terms.count('"') % 2 == 0
    assert " OR " in terms
    for hostile in ("*", "(", ")", "NEAR"):
        assert hostile not in terms.replace('"', "")


def test_fts_query_terms_are_empty_for_stopword_only_input() -> None:
    assert fts_query_terms("the and of") == ""


def test_stopwords_cover_both_languages() -> None:
    assert is_stopword("the")
    assert is_stopword("و")
    assert not is_stopword("sqlite")
    assert not is_stopword("کتاب")


def test_detects_english() -> None:
    assert detect_language("The quick brown fox jumps over the lazy dog") == "en"


def test_detects_persian() -> None:
    assert detect_language("این یک متن فارسی برای آزمایش است") == "fa"


def test_detects_mixed() -> None:
    assert detect_language("این متن شامل کلمات English هم است") == "mixed"


def test_detects_empty_as_english() -> None:
    assert detect_language("") == "en"
    assert detect_language("12345 !!!") == "en"


@given(st.text(max_size=200))
def test_tokens_never_contain_whitespace(text: str) -> None:
    assert all(" " not in token for token in tokenize(text))


@given(st.text(max_size=200))
def test_tokenize_is_idempotent_on_its_own_output(text: str) -> None:
    tokens = tokenize(text)
    assert tokenize(" ".join(tokens)) == tokens
