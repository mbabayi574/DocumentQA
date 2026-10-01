"""Tokenizer and stopwords (plan.md P1).

Output feeds FTS5 as space-joined tokens, so SQLite's own tokenizer never has to
understand Persian. No stemming: P9 measures whether it helps before it lands.
"""

from __future__ import annotations

import re

from qasystem.text.normalize import ZWNJ, normalize_for_index

# A token is a run of word characters, plus the punctuation that lives *inside*
# identifiers (ERR-404, v2.3.1, some_name) and ZWNJ. Everything else separates.
_TOKEN_RE = re.compile(rf"[^\w{re.escape(ZWNJ)}\-._]+", re.UNICODE)
_VERSION_RE = re.compile(r"v?\d+(?:\.\d+)+")

# Stopwords are used only for coverage and sentence scoring, never for FTS terms.
_EN = "a an and are as at be but by for from has have he her his i in is it its of on or "
_EN += "she that the their them they this to was were what when where which who will with "
_EN += "you your not no do does did can could should would"
# ruff: noqa: RUF001 - Persian letters are not ASCII look-alikes here
_FA = (
    "و در به از که این را با است برای آن یک تا هم بر یا می شود اما های ها بود "
    "گفت کرد کند شد هر دو نیز همه دیگر"
)
EN_STOPWORDS = frozenset(_EN.split())
FA_STOPWORDS = frozenset(_FA.split())
ALL_STOPWORDS = EN_STOPWORDS | FA_STOPWORDS


def _is_version(token: str) -> bool:
    """True for dotted numeric identifiers like ``v2.3.1`` or ``3.14``."""
    return bool(_VERSION_RE.fullmatch(token))


def _split_zwnj(token: str) -> list[str]:
    """A ZWNJ compound yields three forms, so recall survives any spelling.

    ``می‌رود`` -> the compound itself, the fused form ``میرود``, and the parts
    ``می`` / ``رود``. A document typed with ZWNJ and a query typed with a space
    then share ``می`` and ``رود``.
    """
    if ZWNJ not in token:
        return [token]
    forms = [token, token.replace(ZWNJ, ""), *token.split(ZWNJ)]
    return [form for form in forms if form]


def tokenize(text: str) -> list[str]:
    """Normalized, lowercased, de-duplicated tokens in first-appearance order.

    Latin is lowercased; Persian has no case. Identifiers such as ``ERR-404`` and
    ``v2.3.1`` survive intact, so exact-term lookups keep working.
    """
    normalized = normalize_for_index(text)
    tokens: list[str] = []
    seen: set[str] = set()

    for raw in _TOKEN_RE.split(normalized):
        if not raw:
            continue
        # Keep hyphen/underscore/dot inside identifiers, drop them at the edges.
        # A trailing dot is sentence punctuation; v2.3.1 keeps its internal one.
        token = raw.rstrip(".").strip("-_") if not _is_version(raw) else raw.strip("-_")
        if not token:
            continue
        for candidate in _split_zwnj(token):
            lowered = candidate.lower()
            if lowered and lowered not in seen:
                seen.add(lowered)
                tokens.append(lowered)
    return tokens


def is_stopword(token: str) -> str | bool:
    """True when a token carries no retrieval signal (used for coverage scoring)."""
    return token.lower() in ALL_STOPWORDS


def fts_query_terms(text: str) -> str:
    """A quoted ``OR``-joined FTS5 expression. Never pass user text through raw.

    Each token is escaped and wrapped in double quotes, so FTS5 operators in user
    input (``" OR * ( ) NEAR``) are inert. Stopword-only input yields ``""``, which
    the caller must treat as "no lexical candidates".
    """
    terms = [token for token in tokenize(text) if not is_stopword(token)]
    if not terms:
        return ""
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in terms)
