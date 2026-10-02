"""P7 live probe: the real retrieval, gate and answer path against the real provider.

Skipped unless ``RUN_LIVE=1``. This is the only place the P7 code has met real BGE-M3
vectors, and the first live run found four defects that no offline test could see (D45, D46,
D47, and the dead dense branch). Every threshold asserted here comes from a live measurement
recorded in ``config/thresholds.json``.

Design rules for this file:

* Questions are **verified against the corpus** before they are asserted on. A live run of
  invented questions measures nothing: four of the eight "answerable" questions in the first
  run turned out to have no matching text anywhere in the corpus, and the gate was right to
  refuse them.
* It asserts *structure and reachability*, never a tuned number's exact value. Retrieval
  quality is P9's calibration job; this file's job is to catch a provider or pipeline change.
* It never prints the token, and a failure message never contains one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qasystem.chunking.chunker import Chunker
from qasystem.config import load_settings
from qasystem.embeddings.caching import CachingEmbedder
from qasystem.embeddings.client import EmbeddingClient
from qasystem.errors import ConfigError
from qasystem.ingestion.service import IngestionService
from qasystem.parsing.registry import ParserRegistry
from qasystem.retrieval.gate import Thresholds, load_thresholds
from qasystem.retrieval.service import RetrievalService
from qasystem.storage.chroma_store import ChromaStore
from qasystem.storage.sqlite_store import SqliteStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
PERSIAN_PDF = FIXTURES / "fa" / "ai-engineer.pdf"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real provider"
)

# Each question's terms were confirmed present in the corpus by tokenizing it, not guessed.
# `coverage` is what P7's tokenizer computes for the best chunk, measured offline.
ANSWERABLE = [
    ("BGE-M3", 1.0),  # ai-engineer.pdf, p5
    ("پردازش اسناد", 1.0),  # ai-engineer.pdf, p4
    ("مدل embedding", 1.0),  # ai-engineer.pdf, p5 -- needs the D45 script fix
    ("مستندات OpenAPI", 1.0),  # ai-engineer.pdf, p7 -- needs the D45 script fix
]
UNANSWERABLE = [
    "what is the boiling point of mercury at 2000 metres",
    "how do I bake sourdough bread at home",
    "population of Reykjavik in winter",
    "who won the 1998 world cup final",
    "قیمت طلا در سال گذشته",
]
# English questions whose answer exists only in the Persian PDF. Coverage is structurally
# zero for these -- there are no shared tokens -- so only `dense_only` can pass the gate.
CROSS_LINGUAL = [
    "which programming language and tools are free to use",
    "what documentation must accompany the code repository",
]


@pytest.fixture
async def live_env(tmp_path: Path) -> dict[str, object]:
    try:
        settings = load_settings()
    except ConfigError as exc:
        # No .env, so no token. A clean clone must skip rather than error (D31).
        pytest.skip(f"live run needs a configured provider: {exc.code}")
    if settings.is_fake_provider:
        pytest.skip("EMBEDDING_PROVIDER=fake; nothing live to call")
    if settings.embedding_api_key is None:
        pytest.skip("EMBEDDING_API_KEY is not set")
    store = SqliteStore(tmp_path / "qasystem.db")
    async with EmbeddingClient(settings) as client:
        vectors = ChromaStore(
            tmp_path / "chroma", model_id=client.model_id, dimension=client.dimension
        )
        embedder = CachingEmbedder(client, store)
        yield {
            "settings": settings,
            "store": store,
            "vectors": vectors,
            "embedder": embedder,
            "model_id": client.model_id,
            "ingest": IngestionService(
                store=store,
                vectors=vectors,
                embedder=embedder,
                parser=ParserRegistry(),
                chunker=Chunker(settings),
            ),
            "thresholds": load_thresholds(settings.thresholds_path, model_id=client.model_id),
        }
        vectors.close()
    store.close()


def retrieval(env: dict[str, object]) -> RetrievalService:
    return RetrievalService(
        store=env["store"],  # type: ignore[arg-type]
        vectors=env["vectors"],  # type: ignore[arg-type]
        embedder=env["embedder"],  # type: ignore[arg-type]
        settings=env["settings"],  # type: ignore[arg-type]
        thresholds=env["thresholds"],  # type: ignore[arg-type]
    )


async def test_the_shipped_thresholds_load_for_the_real_model(live_env: dict[str, object]) -> None:
    """A real run must not silently fall back to defaults: `model_id` has to match.

    `calibrated` is now True because P9 grid-searched these on the eval dataset's dev split
    (plan.md section 9.3), and the version is asserted so a recalibration is visible here
    rather than only in the git log.
    """
    thresholds = live_env["thresholds"]
    assert thresholds.model_id == live_env["model_id"]
    assert thresholds.calibrated is True, (
        "config/thresholds.json is a measured calibration (P9); `calibrated: false` would "
        "mean the placeholders are shipping"
    )
    assert thresholds.version >= 2, "P9 wrote version 2; a lower version means it was reverted"


async def test_a_verified_question_is_answered_with_real_citations(
    live_env: dict[str, object],
) -> None:
    await live_env["ingest"].ingest(PERSIAN_PDF.read_bytes(), PERSIAN_PDF.name)  # type: ignore[union-attr]

    for question, expected_coverage in ANSWERABLE:
        answer = await retrieval(live_env).answer(question, debug=True)
        assert answer.status == "answered", f"{question!r} refused: {answer.reason}"
        assert answer.segments and answer.citations
        gate = answer.debug["gate"]  # type: ignore[index]
        assert gate["passed"] is True
        # Coverage is measured, so this checks the tokenizer end to end on real text -- it is
        # what proved the D45 script-boundary defect, where "embedding" was not a token.
        assert gate["token_coverage"] >= expected_coverage * 0.999, question


async def test_an_unanswerable_question_is_refused_with_no_citations(
    live_env: dict[str, object],
) -> None:
    await live_env["ingest"].ingest(PERSIAN_PDF.read_bytes(), PERSIAN_PDF.name)  # type: ignore[union-attr]

    for question in UNANSWERABLE:
        answer = await retrieval(live_env).answer(question)
        assert answer.status == "insufficient_information", question
        assert not answer.citations, question
        assert answer.evidence_score == 0.0


async def test_the_gate_separates_answerable_from_unanswerable(live_env: dict[str, object]) -> None:
    """The one property the whole gate exists for, on real vectors.

    Asserted as a *separation*, not as thresholds: that is robust to a provider change while
    still failing if the two distributions collapse into each other.
    """
    await live_env["ingest"].ingest(PERSIAN_PDF.read_bytes(), PERSIAN_PDF.name)  # type: ignore[union-attr]
    service = retrieval(live_env)

    async def best_dense(question: str) -> float:
        answer = await service.answer(question, debug=True)
        gate = (answer.debug or {}).get("gate", {})
        return float(gate.get("max_dense", 0.0))

    answerable = [await best_dense(q) for q, _ in ANSWERABLE]
    unanswerable = [await best_dense(q) for q in UNANSWERABLE]
    assert min(answerable) > max(unanswerable), (
        f"answerable {min(answerable):.3f}-{max(answerable):.3f} must sit above "
        f"unanswerable {min(unanswerable):.3f}-{max(unanswerable):.3f}"
    )


async def test_a_cross_lingual_hit_has_structurally_zero_coverage(
    live_env: dict[str, object],
) -> None:
    """D47, still true and still load-bearing: an English question against a Persian source
    shares no token, so ``token_coverage`` is exactly 0.0 rather than merely unmet.

    This is the property that forced the gate's uncorroborated ``dense_only`` branch, and it
    is asserted without depending on whether the branch currently passes.
    """
    await live_env["ingest"].ingest(PERSIAN_PDF.read_bytes(), PERSIAN_PDF.name)  # type: ignore[union-attr]
    service = retrieval(live_env)

    for question in CROSS_LINGUAL:
        answer = await service.answer(question, debug=True)
        gate = answer.debug["gate"]  # type: ignore[index]
        assert gate["token_coverage"] == 0.0, (  # type: ignore[index]
            f"{question!r} now shares a token with the Persian source; D47's premise has "
            "changed and the uncorroborated branch needs re-justifying"
        )
        assert gate["candidate_count"] > 0, "nothing was retrieved at all"
        assert gate["max_dense"] > 0.4, "a cross-lingual hit should still be close in meaning"


async def test_cross_lingual_answerability_depends_on_the_corpus_not_on_the_model(
    live_env: dict[str, object],
) -> None:
    """Cross-lingual retrieval is not supported, and the guarantee is now unconditional.

    D47's premise: an English question against a Persian source shares no token, so
    `token_coverage` is structurally 0 and the old `dense_only` branch was the only path in. P10
    removed that branch (D71) after measuring that no signal could make it both safe and reachable
    (D69) — the same cross-language chunk scores 0.541 for an unanswerable question and 0.525 for
    an answerable one.

    So the refusal here is no longer "a bar happens to sit below this question". It is structural:
    **a hit with zero coverage is refused at any similarity**, because every path into the gate
    now requires coverage. This test asserts the guarantee rather than a number, which is the
    difference between a fact about the system and a fact about a threshold file.

    P9 also measured that the *corpus* decided whether the same questions were answered at all —
    0.00 coverage on this 7-chunk index, 0.40-0.50 on the 49- and 1225-chunk ones, because an
    unrelated document lent incidental words. That is D61, and it is why those larger-corpus
    answers were luck rather than capability.
    """
    await live_env["ingest"].ingest(PERSIAN_PDF.read_bytes(), PERSIAN_PDF.name)  # type: ignore[union-attr]
    service = retrieval(live_env)
    thresholds = live_env["thresholds"]
    assert isinstance(thresholds, Thresholds)

    for question in CROSS_LINGUAL:
        answer = await service.answer(question, debug=True)
        gate = answer.debug["gate"]  # type: ignore[index]
        assert answer.status == "insufficient_information", (
            f"{question!r} is now answered from the Persian source on a 7-chunk index. That is "
            "the capability section 2 claims, so section 12 question 3 can be closed: update "
            "section 2, README and this test rather than leaving them describing a limitation "
            f"(gate said {gate['reason']}, max_dense {gate['max_dense']})."  # type: ignore[index]
        )
        # A threshold refused it, not a retrieval accident: the chunk was found and it is close.
        assert gate["max_dense"] > 0.4, "the cross-lingual hit should still be retrieved"  # type: ignore[index]
        # And the gate has no path for it at all: zero coverage, whatever the similarity.
        assert gate["token_coverage"] == 0.0  # type: ignore[index]
        assert not (  # type: ignore[index]
            gate["max_dense"] >= thresholds.min_dense
            and gate["token_coverage"] >= thresholds.min_coverage
        )
        assert answer.citations == ()


async def test_the_ingest_log_counts_network_requests_not_calls(
    live_env: dict[str, object],
) -> None:
    """D46: one `embed()` call fans out into batches, and the log must count the batches.

    ``MAX_ITEMS_PER_BATCH`` is 32, so a document of more than 32 chunks cannot be embedded in
    one request. Reading the count off the log is how §9.2 and §2's cost analysis stay true.
    """
    embedder = live_env["embedder"]
    before = embedder.requests
    result = await live_env["ingest"].ingest(PERSIAN_PDF.read_bytes(), PERSIAN_PDF.name)  # type: ignore[union-attr]

    floor = -(-result.chunks_added // live_env["settings"].max_items_per_batch)  # type: ignore[union-attr]
    assert result.embed_requests == embedder.requests - before
    assert result.embed_requests >= floor, (
        f"{result.chunks_added} chunks needs at least {floor} requests at "
        f"{live_env['settings'].max_items_per_batch} per batch (D46)"  # type: ignore[union-attr]
    )
    row = live_env["store"].ingest_log("ai-engineer")[-1]  # type: ignore[union-attr]
    assert row["embed_requests"] == result.embed_requests


async def test_a_repeated_question_costs_no_embedding_request(live_env: dict[str, object]) -> None:
    await live_env["ingest"].ingest(PERSIAN_PDF.read_bytes(), PERSIAN_PDF.name)  # type: ignore[union-attr]
    service = retrieval(live_env)
    await service.answer("BGE-M3")

    before = live_env["embedder"].requests  # type: ignore[union-attr]
    await service.answer("BGE-M3")

    assert live_env["embedder"].requests == before  # type: ignore[union-attr]


async def test_no_secret_appears_in_a_live_error(live_env: dict[str, object]) -> None:
    """I8 against the real service: a refusal path must not echo the credential."""
    from qasystem.config import load_settings as reload_settings

    settings = reload_settings()
    token = settings.embedding_api_key
    if token is None:
        pytest.skip("no token configured, so there is nothing to assert is absent")
    secret = token.get_secret_value()
    assert len(secret) >= 8

    # Nothing in the retrieval surface echoes a credential, so the check is on the values
    # a live run actually produced.
    await live_env["ingest"].ingest(PERSIAN_PDF.read_bytes(), PERSIAN_PDF.name)  # type: ignore[union-attr]
    answer = await retrieval(live_env).answer("BGE-M3", debug=True)
    assert secret not in answer.answer
    assert secret not in str(answer.as_dict(debug=True))
