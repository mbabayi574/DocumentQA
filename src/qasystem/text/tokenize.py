"""Tokenizer and stopwords (plan.md P1).

Output feeds FTS5 as space-joined tokens, so SQLite's own tokenizer never has to
understand Persian. No stemming: P9 measures whether it helps before it lands.
"""

from __future__ import annotations

import re

from qasystem.text.normalize import ZWNJ, normalize_for_index

# A token is a run of word characters, plus the punctuation that lives *inside*
# identifiers (ERR-404, v2.3.1, some_name) and ZWNJ. Everything else separates.
# A trailing dot is sentence punctuation; rstrip(".") leaves v2.3.1 and 3.14 intact.
_TOKEN_RE = re.compile(rf"[^\w{re.escape(ZWNJ)}\-._]+", re.UNICODE)

# Stopwords are used only for coverage and sentence scoring, never for FTS terms.
_EN = (
    "a an and are as at be but by for from has have he her his i in is it its of on or "
    "she that the their them they this to was were what when where which who will with "
    "you your not no do does did can could should would"
)
_FA = (
    "و در به از که این را با است برای آن یک تا هم بر یا می شود اما های ها بود "
    "گفت کرد کند شد هر دو نیز همه دیگر"
)
STOPWORDS = frozenset(f"{_EN} {_FA}".split())


def _split_zwnj(token: str) -> list[str]:
    """A ZWNJ compound yields three forms, so recall survives any spelling.

    ``می‌رود`` -> the compound itself, the fused form ``میرود``, and the parts
    ``می`` / ``رود``. A document typed with ZWNJ and a query typed with a space
    then share ``می`` and ``رود``.
    """
    if ZWNJ not in token:
        return [token]
    return [form for form in (token, token.replace(ZWNJ, ""), *token.split(ZWNJ)) if form]


def tokenize(text: str) -> list[str]:
    """Normalized, lowercased, de-duplicated tokens in first-appearance order.

    Latin is lowercased; Persian has no case. Identifiers such as ``ERR-404`` and
    ``v2.3.1`` survive intact, so exact-term lookups keep working.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.split(normalize_for_index(text)):
        token = raw.rstrip(".").strip("-_")
        for candidate in _split_zwnj(token):
            lowered = candidate.lower()
            if lowered and lowered not in seen:
                seen.add(lowered)
                tokens.append(lowered)
    return tokens


def is_stopword(token: str) -> bool:
    """True when a token carries no retrieval signal (used for coverage scoring)."""
    return token.lower() in STOPWORDS


def fts_query_terms(text: str) -> str:
    """A quoted ``OR``-joined FTS5 expression. Never pass user text through raw.

    Each token is wrapped in double quotes, and ``_TOKEN_RE`` cannot produce a quote
    inside a token, so FTS5 operators in user input (``" OR * ( ) NEAR``) are inert.
    Stopword-only input yields ``""``, which the caller must read as "no lexical
    candidates".
    """
    terms = [token for token in tokenize(text) if not is_stopword(token)]
    return " OR ".join(f'"{token}"' for token in terms)
