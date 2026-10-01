"""P7: retrieval, the evidence gate, and the extractive answer.

Real SQLite, a real ``PersistentClient`` on ``tmp_path``, and the deterministic fake
embedder. Thresholds are injected per test rather than read from disk, because the
fake embedder's similarity scale is nothing like BGE-M3's -- the whole point of keying
thresholds on ``model_id`` is that a test cannot borrow the real model's numbers, and
this file is where that shows up.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from qasystem.chunking.chunker import Chunker
from qasystem.domain.models import VectorItem
from qasystem.embeddings.caching import CachingEmbedder
from qasystem.embeddings.fake import FakeEmbedder
from qasystem.errors import InvalidQuestionError, VectorStoreError
from qasystem.ingestion.service import IngestionService
from qasystem.parsing.registry import ParserRegistry
from qasystem.retrieval.gate import Thresholds
from qasystem.retrieval.service import RetrievalService
from qasystem.storage.chroma_store import ChromaStore
from qasystem.storage.embedding_cache import SqliteEmbeddingCache
from qasystem.storage.sqlite_store import SqliteStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
DIM = 32
FAKE_MODEL = "fake-embedder"

HANDBOOK = """# Handbook

## Install

Run the installer on Linux. It checks the kernel version first.

## Usage

The service returns ERR-404 when the token is invalid.
"""

HANDBOOK_V2 = """# Handbook

## Install

Run the installer on Linux. It checks the kernel version first.

## Usage

The service returns ERR-503 when the upstream cache is unreachable.
"""

PERSIAN = """# راهنما

## نص

نصب‌کننده را روی لینوکس اجرا کنید. این ابزار ابتدا نسخه کرنل را بررسی می‌کند.

## استفاده

سرویس در صورت نامعتبر بودن توکن، خطای ERR-404 برمی‌گرداند.
"""


def thresholds(**overrides: Any) -> Thresholds:
    """Gate settings the fake embedder can actually satisfy.

    ``min_dense`` is near zero on purpose: a hashed bag-of-tokens gives ~0.2 cosine between
    a short question and a 500-char chunk, where BGE-M3 gives 0.7+, so raw similarity cannot
    discriminate here and coverage carries the decision. The coverage bar is *measured*, not
    guessed: on this corpus a verbatim question scores 1.0, the realistic paraphrase "What
    does the service return when the token is invalid" scores 0.75 ("return" vs "returns"),
    and the near-topic question that must refuse scores 0.5.
    """
    base: dict[str, Any] = {
        "min_dense": 0.05,
        "min_coverage": 0.70,
        "min_lexical": 0.50,
        "min_coverage_high": 0.90,
        "min_sentence_overlap": 0.15,
        "version": 1,
        "calibrated": False,
        "model_id": FAKE_MODEL,
    }
    base.update(overrides)
    return Thresholds(**base)


class FaultyVectors:
    """A real ``ChromaStore`` with one named method made to fail.

    Faults are injected at the point P7 depends on them not mattering, while everything
    else stays the production adapter, so a failure means the real path broke.
    """

    def __init__(self, inner: ChromaStore, *, fail_on: str) -> None:
        self._inner = inner
        self.fail_on = fail_on

    def _maybe_fail(self, name: str) -> None:
        if self.fail_on == name:
            raise VectorStoreError(f"injected {name} failure")

    def ensure_collection(self) -> None:
        self._maybe_fail("ensure_collection")
        self._inner.ensure_collection()

    def upsert(self, items: Sequence[VectorItem]) -> None:
        self._maybe_fail("upsert")
        self._inner.upsert(items)

    def delete_ids(self, ids: Sequence[str]) -> None:
        self._maybe_fail("delete_ids")
        self._inner.delete_ids(ids)

    def query(self, vector: Sequence[float], n: int) -> list[Any]:
        return self._inner.query(vector, n)

    def get_existing_ids(self, ids: Sequence[str]) -> set[str]:
        return self._inner.get_existing_ids(ids)

    def list_ids(self) -> Iterator[str]:
        return self._inner.list_ids()

    def count(self) -> int:
        return self._inner.count()

    def ping(self) -> None:
        self._inner.ping()


@pytest.fixture
def env(tmp_path: Any, settings_factory: Any) -> Any:
    settings = settings_factory(app_env="test", embedding_provider="fake")
    store = SqliteStore(tmp_path / "qasystem.db")
    inner = ChromaStore(tmp_path / "chroma", model_id=FAKE_MODEL, dimension=DIM)
    fake = FakeEmbedder(dimension=DIM, model_id=FAKE_MODEL)
    embedder = CachingEmbedder(fake, SqliteEmbeddingCache(store))

    class Env:
        def __init__(self) -> None:
            self.settings = settings
            self.store = store
            self.real = inner
            self.vectors = inner
            self.fake = fake
            self.embedder = embedder
            self.chunker = Chunker(settings)
            self.ingest = IngestionService(
                store=store,
                vectors=inner,
                embedder=embedder,
                parser=ParserRegistry(),
                chunker=self.chunker,
            )
            self.retrieval = self.retriever()

        def retriever(self, **overrides: Any) -> RetrievalService:
            return RetrievalService(
                store=store,
                vectors=inner,
                embedder=embedder,
                settings=settings,
                thresholds=thresholds(**overrides),
            )

        def vector_ids(self) -> list[str]:
            return sorted(inner.list_ids())

        def source(self, doc_id: str) -> str:
            version = int(store.get_document(doc_id)["current_version"])
            return store.source_text(doc_id, version) or ""

    yield Env()
    store.close()
    inner.close()


# ---------------------------------------------------------------- plan test 1


async def test_1_an_answerable_question_is_answered_from_the_expected_version(env: Any) -> None:
    published = await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    answer = await env.retrieval.answer("What does the service return when the token is invalid?")

    assert answer.status == "answered"
    assert answer.segments, "an answered response with no evidence is a bug"
    assert answer.citations
    assert "ERR-404" in answer.answer
    # Every cited chunk belongs to the published version, not some other version.
    for citation in answer.citations:
        assert f":v{published.version}:" in citation.chunk_id


async def test_1b_the_cited_text_is_verbatim_from_the_source(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    answer = await env.retrieval.answer("kernel version")

    source = env.source("handbook")
    assert answer.segments
    for segment in answer.segments:
        assert segment.text in source, "a segment is not an exact slice of the source text"
    assert [citation.id for citation in answer.citations] == [1]


# ---------------------------------------------------------------- plan test 3 (I6, I7)


async def test_3_every_segment_is_an_exact_substring_of_its_chunk_and_the_source(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    await env.ingest.ingest(PERSIAN.encode(), "rahnamа.md")
    await env.ingest.ingest((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")

    for question in ("kernel version", "نصب‌کننده لینوکس", "ERR-404 token invalid"):
        answer = await env.retrieval.answer(question)
        if answer.status != "answered":
            continue
        for segment in answer.segments:
            row = env.store.eligible_chunks(chunk_ids=[segment.chunk_id])[0]
            # The mechanical proof of I6, both levels.
            assert row["text"][segment.chunk_char_start : segment.chunk_char_end] == segment.text
            absolute = env.store.source_text(row["doc_id"], int(row["doc_version"])) or ""
            offset = int(row["char_start"]) + segment.chunk_char_start
            assert absolute[offset : offset + len(segment.text)] == segment.text


async def test_3b_the_rendered_answer_is_exactly_the_slices_plus_their_markers(
    env: Any,
) -> None:
    """I7: the answer is the slices in source order plus ``[n]``, and nothing else.

    The rendering is pinned exactly rather than by stripping markers out of the answer.
    Stripping is lossy: an earlier version of this test used it, and because the fixture's
    sentence already ended in a period it could not tell "quoted verbatim" from "quoted,
    with the period moved" -- a paraphrase mutation passed it.
    """
    # A source with no terminal punctuation, so any added full stop is detectable.
    await env.ingest.ingest(
        b"# Notes\n\nThe kernel version is checked first\n\nThe quota is forty megabytes\n",
        "notes.md",
    )
    answer = await env.retrieval.answer("kernel version")

    assert answer.status == "answered"
    assert answer.answer == " ".join(
        f"{segment.text} [{segment.citation_id}]" for segment in answer.segments
    )
    # No full stop was added: the source had none, and a rendering that supplied one would
    # be quoting something the document does not contain.
    assert answer.answer == "The kernel version is checked first [1]"
    # The marker is rendered outside the slice, so the slice stays an exact substring.
    for segment in answer.segments:
        assert segment.text in answer.answer
        assert segment.text in env.source("notes")


async def test_3c_segments_are_ordered_by_source_position(env: Any) -> None:
    document = (
        "# Doc\n\n## One\n\n"
        + "\n\n".join(f"Sentence number {n} mentions the widget." for n in range(8))
        + "\n\n## Two\n\nNothing relevant here.\n"
    )
    await env.ingest.ingest(document.encode(), "ordered.md")

    answer = await env.retrieval.answer("widget")
    assert len(answer.segments) >= 2
    positions = [(s.chunk_id, s.chunk_char_start) for s in answer.segments]
    assert positions == sorted(positions, key=lambda pair: (pair[0], pair[1]))


async def test_3d_adjacent_sentences_are_merged_into_one_slice(env: Any) -> None:
    document = (
        "# Notes\n\nThe widget ships on Tuesday. "
        + ("The widget also ships on Wednesday. The widget never ships on Friday.")
        + "\n"
    )
    await env.ingest.ingest(document.encode(), "merged.md")

    answer = await env.retrieval.answer("widget ships")
    assert len(answer.segments) == 1
    segment = answer.segments[0]
    row = env.store.eligible_chunks(chunk_ids=[segment.chunk_id])[0]
    assert row["text"][segment.chunk_char_start : segment.chunk_char_end] == segment.text
    assert "Wednesday." in segment.text and "Friday." in segment.text


async def test_3f_sentences_separated_by_real_text_are_not_merged(env: Any) -> None:
    """Merging is for contiguity, not tidiness.

    A merge across real text would quote a sentence the selection deliberately left out,
    which is the opposite of extraction: the answer would contain text no signal chose.
    """
    document = (
        "# Schedule\n\nThe widget ships on Tuesday. "
        "The quarterly catering budget was approved in March. "
        "The widget ships on Friday.\n"
    )
    await env.ingest.ingest(document.encode(), "gaps.md")

    answer = await env.retrieval.answer("widget ships")

    assert answer.status == "answered"
    assert len(answer.segments) == 2, "two separate sentences must stay two segments"
    assert all("catering" not in segment.text for segment in answer.segments)
    for segment in answer.segments:
        row = env.store.eligible_chunks(chunk_ids=[segment.chunk_id])[0]
        assert row["text"][segment.chunk_char_start : segment.chunk_char_end] == segment.text
    assert "Tuesday." in answer.segments[0].text
    assert "Friday." in answer.segments[1].text


async def test_3e_an_unrelated_sentence_is_never_padded_in(env: Any) -> None:
    document = (
        "# Notes\n\nThe widget will ship on Tuesday.\n\n"
        "The quarterly budget for catering was approved in March.\n"
    )
    await env.ingest.ingest(document.encode(), "padded.md")

    answer = await env.retrieval.answer("when does the widget ship")
    assert answer.status == "answered"
    assert "catering" not in answer.answer, "a relevant chunk leaked an irrelevant sentence"


# ---------------------------------------------------------------- plan test 2 (gate)


async def test_2_an_off_topic_question_refuses_without_citations(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    answer = await env.retrieval.answer("What is the population of Reykjavik in winter?")

    assert answer.status == "insufficient_information"
    assert not answer.citations
    assert not answer.segments
    assert answer.reason == "below_threshold"


async def test_2b_a_near_topic_question_refuses_rather_than_guessing(env: Any) -> None:
    """Same document family, wrong fact: the case a retrieval system is most tempted by."""
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    answer = await env.retrieval.answer("Which Linux distribution does the installer support?")

    assert answer.status == "insufficient_information"
    assert not answer.citations
    assert "ERR-" not in answer.answer, "it answered with a nearby fact"


async def test_2c_an_empty_knowledge_base_says_so(env: Any) -> None:
    answer = await env.retrieval.answer("kernel version")
    assert answer.status == "insufficient_information"
    assert answer.reason == "empty_knowledge_base"
    assert not answer.citations
    assert answer.evidence_score == 0.0


async def test_2d_the_refusal_message_is_localised_and_never_cited(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    english = await env.retrieval.answer("population of Reykjavik")
    persian = await env.retrieval.answer("جمعیت شهر ریکیاویک چقدر است")

    assert english.answer == "Not enough information in the provided documents."
    assert persian.answer != english.answer
    assert not persian.citations and not english.citations
    # The message is system text, so it can never appear in the evidence.
    for answer in (english, persian):
        assert answer.evidence_score == 0.0
        assert answer.reason


async def test_2e_a_filter_matching_nothing_is_no_relevant_content(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    answer = await env.retrieval.answer("kernel version", doc_ids=["never-ingested"])

    assert answer.status == "insufficient_information"
    assert answer.reason == "no_relevant_content"


async def test_2f_a_strict_threshold_refuses_what_a_loose_one_answers(env: Any) -> None:
    """The gate is load-bearing: same corpus, same question, different threshold."""
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    question = "What does the service return when the token is invalid"
    lenient = await env.retriever(min_coverage=0.50).answer(question)
    strict = await env.retriever(min_coverage=0.90).answer(question)

    assert lenient.status == "answered"
    assert strict.status == "insufficient_information"


async def test_2g_the_gate_runs_before_any_text_is_selected(env: Any) -> None:
    """A refused answer must be empty, not a truncated version of a real one."""
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    answer = await env.retrieval.answer("population of Reykjavik")
    assert answer.answer == "Not enough information in the provided documents."
    assert "installer" not in answer.answer


# ---------------------------------------------------------------- plan test 5 (I1, I2)


async def test_5_superseded_text_never_appears_in_the_answer(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    assert (
        "ERR-404"
        in (
            await env.retrieval.answer("what does the service return when the token is invalid")
        ).answer
    )

    await env.ingest.ingest(HANDBOOK_V2.encode(), "handbook.md")
    answer = await env.retrieval.answer("what does the service return when the token is invalid")

    assert answer.status == "insufficient_information"
    assert not answer.citations
    assert "ERR-404" not in answer.answer
    assert "ERR-503" in (await env.retrieval.answer("upstream cache unreachable")).answer


async def test_5b_deleted_text_never_appears_even_when_the_vectors_survive(env: Any) -> None:
    """I2 through the public entry point, with Chroma cleanup failing."""
    stuck = FaultyVectors(env.real, fail_on="delete_ids")
    service = IngestionService(
        store=env.store,
        vectors=stuck,  # type: ignore[arg-type]
        embedder=env.embedder,
        parser=ParserRegistry(),
        chunker=env.chunker,
    )
    await service.ingest(HANDBOOK.encode(), "handbook.md")
    stale = env.vector_ids()
    assert stale, "precondition: vectors exist"

    await service.delete("handbook")
    assert env.vector_ids() == stale, "precondition: cleanup really did fail"

    retrieval = RetrievalService(
        store=env.store,
        vectors=stuck,  # type: ignore[arg-type]
        embedder=env.embedder,
        settings=env.settings,
        thresholds=thresholds(),
    )
    # Retrieval ran and the vectors are physically present, yet not one is eligible.
    assert await retrieval.retrieve("kernel version installer token") == []
    answer = await retrieval.answer("kernel version installer token")

    assert answer.status == "insufficient_information"
    assert not answer.citations
    assert "installer" not in answer.answer


async def test_5c_a_failed_update_leaves_the_previous_version_answerable(env: Any) -> None:
    """I3 seen from the read side: a publish that never happened is invisible."""
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    broken = IngestionService(
        store=env.store,
        vectors=FaultyVectors(env.real, fail_on="upsert"),  # type: ignore[arg-type]
        embedder=env.embedder,
        parser=ParserRegistry(),
        chunker=env.chunker,
    )
    with pytest.raises(VectorStoreError):
        await broken.ingest(HANDBOOK_V2.encode(), "handbook.md")

    answer = await env.retrieval.answer("kernel version")
    assert answer.status == "answered"
    assert "kernel version" in answer.answer
    for citation in answer.citations:
        assert ":v1:" in citation.chunk_id, "an unpublished version leaked into the answer"


async def test_5d_non_debug_output_never_leaks_a_superseded_or_deleted_chunk_id(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    await env.ingest.ingest(HANDBOOK_V2.encode(), "handbook.md")
    answer = await env.retrieval.answer("kernel version")
    report = answer.as_dict(debug=False)
    assert report["debug"] is None
    assert "handbook:v1:" not in str(report)


# ---------------------------------------------------------------- plan test 6 (Persian)


async def test_6_a_persian_question_cites_a_persian_source(env: Any) -> None:
    await env.ingest.ingest(PERSIAN.encode(), "rahnama.md")

    answer = await env.retrieval.answer("سرویس در صورت نامعتبر بودن توکن")

    assert answer.status == "answered"
    citation = answer.citations[0]
    assert citation.document == "rahnama.md"
    assert citation.doc_id == "rahnama"
    assert citation.section == "راهنما > استفاده"
    assert citation.page is None  # a Markdown source has no pages; a PDF must have them
    assert citation.lines is not None
    assert citation.excerpt
    assert citation.excerpt in env.source("rahnama")
    for segment in answer.segments:
        assert segment.text in env.source("rahnama")


async def test_6b_a_persian_pdf_citation_carries_its_page_and_lines(env: Any) -> None:
    await env.ingest.ingest((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai-engineer.pdf")

    answer = await env.retrieval.answer("پردازش اسناد")

    assert answer.status == "answered", "a Persian PDF must be answerable"
    assert answer.citations
    for citation in answer.citations:
        assert citation.document == "ai-engineer.pdf"
        assert citation.page is not None and citation.page >= 1
        assert citation.section.startswith("ai-engineer > page ")
        assert citation.excerpt in env.source("ai-engineer")
        # A PDF text layer has no line numbers, so `lines` is honestly null rather than
        # invented. Markdown and TXT carry lines (tested above); PDFs carry pages.
        assert citation.lines is None
    for segment in answer.segments:
        assert segment.text in env.source("ai-engineer")


async def test_6c_the_answer_never_needs_a_translation_step(env: Any) -> None:
    """A Persian question and a Persian source, with nothing in between: the answer is Persian."""
    await env.ingest.ingest(PERSIAN.encode(), "rahnama.md")
    answer = await env.retrieval.answer("سرویس در صورت نامعتبر بودن توکن")
    assert answer.status == "answered"
    # The answer is Persian script lifted from the Persian source: no translation happened.
    assert any("\u0600" <= char <= "\u06ff" for char in answer.answer)


async def test_6d_a_language_filter_excludes_the_other_language(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    await env.ingest.ingest(PERSIAN.encode(), "rahnama.md")

    only_fa = await env.retrieval.retrieve("پردازش اسناد", language="fa")
    only_en = await env.retrieval.retrieve("kernel version", language="en")

    assert {c.language for c in only_fa} == {"fa"}
    assert {c.language for c in only_en} == {"en"}


# ---------------------------------------------------------------- plan test 8


@pytest.mark.parametrize(
    "hostile",
    [
        'installer" OR "kernel',
        "kernel NEAR/2 installer",
        "installer *",
        "installer AND kernel OR NOT usage",
        "kernel; DROP TABLE chunks; --",
        "kernel; DROP TABLE chunks",
        "^installer$",
        "{installer}",
        "installer:^",
    ],
)
async def test_8_hostile_fts_input_stays_inert(env: Any, hostile: str) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    answer = await env.retrieval.answer(hostile)

    assert answer.status in {"answered", "insufficient_information"}
    if answer.status == "answered":
        for segment in answer.segments:
            assert segment.text in env.source("handbook")
    assert env.store.fts5_available(), "the FTS table must still be there"
    assert env.store.eligible_chunk_ids()


# ---------------------------------------------------------------- question validation


@pytest.mark.parametrize("question", ["", "   ", "\n\t ", "!!!", "؟؟؟"])
async def test_a_question_with_no_terms_is_a_422(env: Any, question: str) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    with pytest.raises(InvalidQuestionError):
        await env.retrieval.answer(question)


async def test_a_blank_question_costs_no_embedding_call(env: Any) -> None:
    env.fake.reset()
    with pytest.raises(InvalidQuestionError):
        await env.retrieval.answer("   ")
    assert env.fake.texts_embedded == 0


# ---------------------------------------------------------------- retrieval surface


async def test_retrieval_returns_ranked_candidates_with_their_scores(env: Any) -> None:
    published = await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    candidates = await env.retrieval.retrieve("kernel version")

    assert candidates
    assert [c.score for c in candidates] == sorted((c.score for c in candidates), reverse=True)
    for candidate in candidates:
        assert candidate.chunk_id.startswith(f"handbook:v{published.version}:")
        assert candidate.text in env.source("handbook")
        assert 0.0 <= candidate.token_coverage <= 1.0
    best = candidates[0]
    assert best.similarity is not None and best.lexical_score is not None
    assert best.section_path == ("Handbook", "Install")
    # Coverage is per candidate: only a chunk that actually mentions the query has any,
    # and the best-scoring one is the chunk the question's words are in.
    assert best.token_coverage > 0
    assert all(
        candidate.token_coverage == 0.0
        for candidate in candidates
        if candidate.chunk_id != best.chunk_id
    )


async def test_fusion_ranks_within_the_window_never_over_the_corpus(env: Any) -> None:
    """Ranks are positions in the *retrieved list*, not in the corpus (D30).

    Fusing corpus ordinals is the mistake that made hybrid look worse than dense in §2: with
    k=60 a rank-5000 hit contributes almost nothing, so a correctly retrieved lexical
    match gets drowned by corpus size. This corpus is ordered so the answerable chunk is
    *last*, which is exactly where a corpus ordinal would bury it.
    """
    document = (
        "# Manual\n\n"
        + "\n\n".join(f"## Section {n}\n\nRoutine prose about matter {n}." for n in range(39))
        + "\n\n## Quota\n\nThe quota is forty megabytes.\n"
    )
    await env.ingest.ingest(document.encode(), "manual.md")
    answer = env.store.eligible_chunks(chunk_ids=["manual:v1:39"])
    assert answer and "quota" in answer[0]["text"], "precondition: the answerable chunk is last"

    candidates = await env.retrieval.retrieve("quota megabytes", limit=30)
    best = candidates[0]

    assert best.chunk_id == "manual:v1:39"
    assert best.token_coverage == pytest.approx(1.0)
    # Fusion sees ranks 1..3 here, not 37..40, and the score is the window-rank formula.
    assert (best.dense_rank, best.lexical_rank) == (3, 1)
    assert best.score == pytest.approx(0.7 / (60 + 3) + 0.3 / 61)

    # And explicitly *not* the corpus-ordinal score, which is what D30 measured by mistake.
    ordinal = [row["chunk_id"] for row in env.store.eligible_chunks()].index("manual:v1:39") + 1
    assert ordinal >= 35, "precondition: the corpus ordinal really is far away"
    assert best.score != pytest.approx(0.7 / (60 + ordinal) + 0.3 / (60 + ordinal))

    assert (await env.retrieval.answer("quota megabytes")).status == "answered"


async def test_top_k_limits_the_candidate_list(env: Any) -> None:
    document = (
        "# Big\n\n"
        + "\n\n".join(f"## Section {n}\n\nWidget paragraph number {n}." for n in range(10))
        + "\n"
    )
    await env.ingest.ingest(document.encode(), "big.md")
    assert len(await env.retrieval.retrieve("widget", limit=3)) == 3


async def test_doc_ids_filter_restricts_candidates_to_those_documents(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    await env.ingest.ingest(HANDBOOK.encode(), "other.md", doc_id="other")

    candidates = await env.retrieval.retrieve("kernel version", doc_ids=["other"])

    assert candidates
    assert {c.doc_id for c in candidates} == {"other"}


async def test_a_single_arm_baseline_can_be_measured(env: Any) -> None:
    """plan.md §9.2 needs dense-only and lexical-only numbers, so the weights are overridable.

    A weight of 0 must switch an arm *off*, not merely zero its contribution -- otherwise a
    "lexical-only" figure would still be admitting dense-only hits and would not be
    comparable against the hybrid number it is meant to explain.
    """
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    hybrid = await env.retrieval.retrieve("kernel", limit=100)
    dense_ids = {c.chunk_id for c in hybrid if c.dense_rank is not None}
    lexical_ids = {c.chunk_id for c in hybrid if c.lexical_rank is not None}

    dense_only = await env.retrieval.retrieve("kernel", dense_weight=1.0, lexical_weight=0.0)
    lexical_only = await env.retrieval.retrieve("kernel", dense_weight=0.0, lexical_weight=1.0)

    assert {c.chunk_id for c in dense_only} <= dense_ids
    assert {c.chunk_id for c in lexical_only} <= lexical_ids
    # A pure single-arm score is exactly w/(k+1) at rank 1 -- nothing else was added.
    assert dense_only[0].score == pytest.approx(1.0 / 61)
    assert lexical_only[0].score == pytest.approx(1.0 / 61)


async def test_the_gate_sees_beyond_the_answer_size(env: Any) -> None:
    """``top_k`` sizes the *answer*; it must not size the gate's input.

    With BGE-M3 this is a correctness property -- a chunk can be semantically close enough to
    rank high while covering few of the question's words, and one ranked below ``top_k`` can
    cover all of them. That cannot be staged with the fake embedder, because more token
    overlap always means a better dense rank, so the observable claim is made directly: the
    gate is handed the whole retrieved window.
    """
    document = (
        "# Big\n\n"
        + "\n\n".join(f"## Section {n}\n\nUnrelated prose about topic {n}." for n in range(8))
        + "\n\n## Answer\n\nThe quota is forty megabytes.\n"
    )
    await env.ingest.ingest(document.encode(), "big.md")

    answer = await env.retrieval.answer("quota megabytes", top_k=1, debug=True)

    assert answer.status == "answered"
    assert answer.debug is not None
    window = answer.debug["candidates"]
    assert len(window) > 1, "top_k truncated the gate's input; it only saw one chunk"
    # The gate saw coverage it could act on, even though only one chunk may be quoted.
    assert max(entry["token_coverage"] for entry in window) == pytest.approx(1.0)
    assert len(answer.citations) <= env.settings.max_answer_sentences
    for segment in answer.segments:
        assert "quota is forty megabytes" in segment.text


async def test_the_query_vector_is_cached_across_identical_questions(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    await env.retrieval.answer("kernel version")
    env.fake.reset()

    await env.retrieval.answer("kernel version")

    assert env.fake.texts_embedded == 0, "the same question was embedded twice"


async def test_an_empty_collection_is_queried_without_error(env: Any) -> None:
    """L7: never query a zero-length collection; report an empty knowledge base instead."""
    assert await env.retrieval.retrieve("anything") == []
    answer = await env.retrieval.answer("anything")
    assert answer.reason == "empty_knowledge_base"


# ---------------------------------------------------------------- debug output


async def test_debug_reports_the_gate_signals_and_hides_them_by_default(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")

    quiet = (await env.retrieval.answer("kernel version")).as_dict(debug=False)
    loud = (await env.retrieval.answer("kernel version", debug=True)).as_dict(debug=True)

    assert quiet["debug"] is None
    assert loud["debug"] is not None
    gate = loud["debug"]["gate"]
    assert gate["passed"] is True
    assert gate["calibrated"] is False, "P7 ships placeholders and must not pretend otherwise"
    assert gate["model_id"] == FAKE_MODEL
    assert loud["debug"]["candidates"]


async def test_the_evidence_score_is_deterministic_and_bounded(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    runs = [(await env.retrieval.answer("kernel version")).evidence_score for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]
    assert 0.0 <= runs[0] <= 1.0


async def test_a_refusal_still_reports_a_bounded_zero_evidence_score(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    answer = await env.retrieval.answer("population of Reykjavik")
    assert answer.status == "insufficient_information"
    assert answer.evidence_score == 0.0


# ---------------------------------------------------------------- response shape


async def test_the_response_shape_is_the_documented_one(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    report = (await env.retrieval.answer("kernel version", debug=True)).as_dict(debug=True)

    assert report["status"] == "answered"
    assert isinstance(report["answer"], str)
    assert isinstance(report["segments"], list)
    assert isinstance(report["citations"], list)
    assert isinstance(report["evidence_score"], float)
    assert report["reason"] is None
    citation = report["citations"][0]
    assert set(citation) >= {
        "id",
        "doc_id",
        "document",
        "doc_version",
        "section",
        "page",
        "lines",
        "chunk_id",
        "char_span",
        "excerpt",
        "score",
    }
    assert isinstance(citation["char_span"], list) and len(citation["char_span"]) == 2


async def test_a_citation_points_at_real_source_offsets(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    answer = await env.retrieval.answer("kernel version")
    source = env.source("handbook")

    for citation in answer.citations:
        start, end = citation.char_span
        assert source[start:end] == citation.excerpt


async def test_no_segment_ever_cites_a_document_it_does_not_come_from(env: Any) -> None:
    await env.ingest.ingest(HANDBOOK.encode(), "handbook.md")
    await env.ingest.ingest(PERSIAN.encode(), "rahnama.md")
    answer = await env.retrieval.answer("پردازش اسناد")

    for segment in answer.segments:
        citation = next(c for c in answer.citations if c.id == segment.citation_id)
        assert segment.chunk_id.startswith(f"{citation.doc_id}:v{citation.doc_version}:")
