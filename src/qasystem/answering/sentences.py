"""Sentence selection inside a retrieved chunk (plan.md §7.3).

Pure: sentences in, ranked sentences out. A chunk was retrieved because it matched the
*question*; a chunk is several sentences long, and only some of them do. So this module's
job is to keep a relevant chunk's irrelevant sentences out of the answer -- the failure
mode where a system looks precise because it is quoting, and is wrong because it is
quoting the wrong line.

Scoring is IDF-weighted query-token overlap plus a small term for the chunk's own
retrieval score, so a strong sentence in a strong chunk outranks the same sentence in a
weak one.

ponytail: IDF is computed over the retrieved candidate set, not the corpus. A corpus-wide
document frequency would need a second query and buys almost nothing here, because the
comparison is between sentences from the same handful of chunks. Compute corpus-wide DF
if P9 measures this term as under-discriminating.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from qasystem.chunking.chunker import Sentence
from qasystem.text.tokenize import is_stopword, tokenize

# Deliberately small: this term breaks ties between sentences, it does not pick the chunk.
CHUNK_SCORE_WEIGHT = 0.10


def query_terms(question: str) -> list[str]:
    """The question's signal-bearing tokens, de-duplicated, in order.

    Stopwords go because a question's meaning is not in its function words, and because
    scoring against them lets "how do I do this" match any sentence containing "do".
    """
    return list(dict.fromkeys(t for t in tokenize(question) if not is_stopword(t)))


def token_coverage(question: str, text: str, terms: Sequence[str] | None = None) -> float:
    """Share of the question's signal tokens present in ``text``.

    ``0.0`` for a question with no signal tokens: an undefined ratio is not evidence, and
    defaulting it to 1.0 would make every stopword-only question answerable.

    ``terms`` lets a caller that already tokenized the question reuse that work -- scoring
    thirty candidates should not re-tokenize the question thirty times.
    """
    terms = query_terms(question) if terms is None else terms
    if not terms:
        return 0.0
    present = set(tokenize(text))
    return sum(1 for term in terms if term in present) / len(terms)


def chunk_idf(chunks: Sequence[str], terms: Sequence[str]) -> dict[str, float]:
    """Weight each term by how rare it is across the retrieved chunks.

    ``log(N/df) + 1`` floored at 1.0, so a token present in every candidate chunk is
    discounted to the floor rather than made worthless -- it is still a query term, and
    the ranking should not depend on this estimate being accurate.
    """
    if not terms or not chunks:
        return {}
    counts: Counter[str] = Counter()
    for text in chunks:
        counts.update(set(tokenize(text)))
    total = len(chunks)
    return {
        term: max(1.0, math.log(total / counts[term]) + 1.0)
        for term in terms
        if counts.get(term, 0) > 0
    }


def select_sentences(
    sentences: Sequence[Sentence],
    terms: Sequence[str],
    *,
    idf: Mapping[str, float],
    chunk_score: float = 1.0,
    min_overlap: float = 0.0,
    limit: int,
) -> list[Sentence]:
    """Rank by overlap, drop the weak, truncate, and return **in source order**.

    Source order is not cosmetic: the answer quotes the source, and an answer whose
    sentences jump around reads as reconstructed rather than extracted (I7).
    """
    if not terms or limit <= 0:
        return []
    weights = {term: idf.get(term, 1.0) for term in terms}
    total = sum(weights.values())
    if total <= 0:
        return []

    scored: list[tuple[float, int, Sentence]] = []
    for position, sentence in enumerate(sentences):
        tokens = set(tokenize(sentence.text))
        overlap = sum(w for term, w in weights.items() if term in tokens) / total
        if overlap < min_overlap:
            continue  # never pad the answer with a sentence that does not speak to the question
        scored.append((overlap + CHUNK_SCORE_WEIGHT * chunk_score, position, sentence))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sentence for _, _, sentence in sorted(scored[:limit], key=lambda i: i[1])]
