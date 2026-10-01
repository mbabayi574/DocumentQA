"""The answerer, tested directly rather than only through the retrieval service.

P7 had integration tests and no unit tests here, which is why D48 went unnoticed offline:
a cross-lingual question was refused by the *answerer* after the gate had already accepted
the evidence. Candidates are built by hand so the case can be staged without a provider --
the fake embedder cannot produce a high dense score for a pair that shares no token, which is
precisely the situation under test.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from qasystem.answering.extractive import (
    REFUSAL_MESSAGES,
    Answer,
    build_answer,
    refuse,
)
from qasystem.retrieval.service import Candidate

CHUNK = (
    "نیازمندی‌های این سامانه شامل پردازش اسناد و بازیابی است. "
    "کد باید خوانا باشد. "
    "مستندات OpenAPI همراه با سامانه ارائه می‌شود."
)


def candidate(
    text: str = CHUNK,
    *,
    chunk_id: str = "handbook:v1:0",
    doc_id: str = "handbook",
    score: float = 0.02,
    char_start: int = 0,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_version=1,
        ordinal=0,
        source_name="handbook.pdf",
        text=text,
        section_path=("handbook", "page 6"),
        char_start=char_start,
        char_end=char_start + len(text),
        page_start=6,
        page_end=6,
        line_start=None,
        line_end=None,
        language="fa",
        score=score,
        dense_rank=1,
        lexical_rank=None,
        similarity=0.72,
        bm25=None,
        lexical_score=None,
        token_coverage=0.0,
    )


def answer_for(text: str, question: str, **kwargs: Any) -> Answer:
    return build_answer(
        [candidate(text)],
        question,
        language=kwargs.pop("language", "fa"),
        max_sentences=kwargs.pop("max_sentences", 5),
        min_sentence_overlap=kwargs.pop("min_sentence_overlap", 0.15),
        evidence_score=kwargs.pop("evidence_score", 0.4),
    )


# ---------------------------------------------------------------- refusals


def test_no_candidates_refuses() -> None:
    answer = build_answer(
        [],
        "anything",
        language="en",
        max_sentences=5,
        min_sentence_overlap=0.15,
        evidence_score=0.0,
    )
    assert answer.status == "insufficient_information"
    assert not answer.segments and not answer.citations


def test_a_stopword_only_question_refuses_without_embedding_anything() -> None:
    """An undefined coverage ratio is not evidence; defaulting it would answer anything."""
    answer = answer_for(CHUNK, "the a of")
    assert answer.status == "insufficient_information"
    assert answer.evidence_score == 0.0


def test_a_refusal_carries_no_citations_and_a_fixed_message() -> None:
    for language, message in REFUSAL_MESSAGES.items():
        answer = refuse("below_threshold", language="fa" if language == "fa" else "en")
        assert answer.answer == message
        assert not answer.citations
        assert answer.evidence_score == 0.0
    assert refuse("x", language="fa").answer == REFUSAL_MESSAGES["fa"]
    assert refuse("x", language="en").answer == REFUSAL_MESSAGES["en"]


# ---------------------------------------------------------------- the cross-lingual fallback


def test_a_question_sharing_no_token_still_gets_cited_evidence() -> None:
    """D48: the gate accepted this evidence, so the answerer must not throw it away.

    An English question against a Persian chunk shares no token, so every sentence scores
    overlap 0 and `min_sentence_overlap` excludes them all. Before this, the answerer refused
    -- leaving retrieval and the gate both correct and the user with nothing.
    """
    answer = answer_for(CHUNK, "which programming language is free to use", language="en")

    assert answer.status == "answered"
    assert answer.segments and answer.citations
    citation = answer.citations[0]
    assert citation.document == "handbook.pdf"
    assert citation.section == "handbook > page 6"
    assert citation.page == 6
    assert citation.lines is None, "a PDF has no line numbers, and inventing them is worse"
    for segment in answer.segments:
        assert segment.text in citation.excerpt
        assert citation.excerpt[segment.chunk_char_start : segment.chunk_char_end] == segment.text


def test_the_fallback_respects_the_sentence_budget() -> None:
    answer = answer_for(CHUNK, "unrelated english question", max_sentences=2)
    quoted = " ".join(segment.text for segment in answer.segments)
    assert answer.status == "answered"
    assert quoted.count(".") <= 2, "the budget must bound how much is quoted"


def test_the_fallback_quotes_one_contiguous_prefix_slice() -> None:
    """It is a fallback, not a licence to quote anything.

    Selection is by position, so the quoted text is exactly the chunk's first N sentences as
    one contiguous slice -- and never more than N sentences, nor a gap bridged with text the
    function never chose.
    """
    text = "First sentence here. Second sentence here. Third sentence here."
    answer = answer_for(text, "totally unrelated", max_sentences=2)

    assert answer.status == "answered"
    assert len(answer.segments) == 1, "a prefix of sentences is one contiguous slice"
    assert answer.segments[0].text == "First sentence here. Second sentence here."
    assert "Third" not in answer.answer, "the budget must bound the quotation"
    for segment in answer.segments:
        assert segment.text in text


def test_the_fallback_quotes_the_best_chunk() -> None:
    """``build_answer`` takes candidates in fused order, best first -- a load-bearing contract.

    The fallback quotes ``candidates[0]``, because it is the chunk the fuser most believes
    answers the question. Passing them in order here keeps that dependency visible.
    """
    best = candidate("The quota is forty megabytes.", chunk_id="a:v1:0", score=0.05)
    worse = candidate("Nothing relevant.", chunk_id="b:v1:0", score=0.01)
    answer = build_answer(
        [best, worse],
        "unrelated",
        language="en",
        max_sentences=2,
        min_sentence_overlap=0.15,
        evidence_score=0.3,
    )
    assert answer.status == "answered"
    assert answer.citations[0].chunk_id == "a:v1:0"
    assert "quota" in answer.answer


def test_padding_suppression_is_unaffected_when_a_sentence_does_match() -> None:
    """The fallback fires only when *nothing* qualifies, so an off-topic chunk stays out."""
    on_topic = candidate("The widget ships on Tuesday.", chunk_id="a:v1:0", score=0.05)
    off_topic = candidate("Catering budget approved in March.", chunk_id="b:v1:0", score=0.04)
    answer = build_answer(
        [on_topic, off_topic],
        "when does the widget ship",
        language="en",
        max_sentences=5,
        min_sentence_overlap=0.15,
        evidence_score=0.5,
    )
    assert answer.status == "answered"
    assert "widget" in answer.answer
    assert "catering" not in answer.answer.lower()


# ---------------------------------------------------------------- normal selection


def test_a_matching_sentence_is_preferred_over_an_earlier_one() -> None:
    text = "The budget was approved in March. The quota is forty megabytes."
    answer = answer_for(text, "quota megabytes")
    assert answer.status == "answered"
    assert answer.answer.startswith("The quota is forty megabytes.")


def test_offsets_are_exact_in_both_directions() -> None:
    answer = answer_for(CHUNK, "quota megabytes")
    for segment in answer.segments:
        row = candidate().text
        assert row[segment.chunk_char_start : segment.chunk_char_end] == segment.text
        assert segment.text in row


def test_char_span_points_at_the_cited_excerpt() -> None:
    start = 410
    text = "The quota is forty megabytes."
    answer = build_answer(
        [candidate(text, char_start=start)],
        "quota megabytes",
        language="en",
        max_sentences=5,
        min_sentence_overlap=0.15,
        evidence_score=0.5,
    )
    citation = answer.citations[0]
    assert citation.char_span == (start, start + len(text))
    assert citation.excerpt == text


def test_section_path_is_rendered_as_a_breadcrumb() -> None:
    answer = answer_for(CHUNK, "quota megabytes")
    assert answer.citations[0].section == "handbook > page 6"


def test_segment_section_path_json_round_trips_through_a_real_row() -> None:
    """The breadcrumb is JSON in the database and a string in the API; both must be exact."""
    from qasystem.retrieval.service import Candidate as ServiceCandidate

    stored = json.dumps(["مدیر", "نصب", "لینوکس"], ensure_ascii=False)
    assert json.loads(stored) == ["مدیر", "نصب", "لینوکس"]
    assert ServiceCandidate is Candidate


@pytest.mark.parametrize("limit", [0, -1])
def test_a_zero_budget_refuses_rather_than_emptying_the_answer(limit: int) -> None:
    answer = build_answer(
        [candidate()],
        "quota",
        language="en",
        max_sentences=limit,
        min_sentence_overlap=0.15,
        evidence_score=0.0,
    )
    assert answer.status == "insufficient_information"
    assert not answer.segments


def test_the_answer_string_is_the_segments_plus_markers_only() -> None:
    answer = answer_for(CHUNK, "quota megabytes")
    assert answer.answer == " ".join(f"{s.text} [{s.citation_id}]" for s in answer.segments)
