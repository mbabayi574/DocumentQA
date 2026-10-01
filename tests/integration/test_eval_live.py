"""P9's measurements verified against the real provider, and the gate's own numbers (P9 §9).

``docs/eval_report.md`` and ``config/thresholds.json`` are P9's deliverables, and a
deliverable that cannot be re-derived is a claim. These tests re-derive the parts that
matter and that could silently rot:

* the calibrated thresholds **load** for the real model and are ``calibrated: true``, so
  ``/ready`` and the gate are running on real measurements rather than on placeholders that
  happen to look calibrated;
* the calibration's two headline claims hold on the corpus that produced them -- the
  recalibrated gate refuses the *densest* unanswerable question, which is the one D47 said
  could not be separated, and it still answers the cross-lingual cases the uncorroborated
  branch exists for;
* the metric gap D60 found is still a gap, so a future change cannot close it by accident
  and have the report keep claiming the old number.

Deliberately **not** a second full eval run. ``make eval`` is that, it costs ~2 minutes and
~200 provider requests, and duplicating it here would make ``make live`` a second copy of
the report rather than a check on it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from qasystem.api.deps import Services, build_services
from qasystem.config import Settings, load_settings
from qasystem.errors import ConfigError
from qasystem.ingestion.service import default_doc_id
from qasystem.retrieval.gate import load_thresholds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from runner import load_cases, observe

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "eval" / "corpus"
SHIPPED = ROOT / "config" / "thresholds.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real provider"
)


def _settings(tmp_path: Path) -> Settings:
    try:
        base = load_settings()
    except ConfigError as exc:
        pytest.skip(f"live run needs a configured provider: {exc}")
    if base.is_fake_provider:
        pytest.skip("EMBEDDING_PROVIDER=fake; nothing live to call")
    return base.model_copy(
        update={
            "data_dir": tmp_path,
            "sqlite_path": tmp_path / "qasystem.db",
            "chroma_path": tmp_path / "chroma",
        }
    )


@pytest.fixture
async def graph(tmp_path: Path) -> Any:
    services = await build_services(_settings(tmp_path))
    try:
        for path in sorted(CORPUS.rglob("*")):
            if path.suffix in (".md", ".txt", ".pdf"):
                await services.ingestion.ingest(path.read_bytes(), path.name)
        yield services
    finally:
        await services.aclose()


# ------------------------------------------------------------------ the deliverable


def test_the_shipped_thresholds_are_calibrated_for_the_real_model() -> None:
    """`/ready` reports this flag, so it is a user-visible claim about the gate."""
    base = load_settings()
    if base.is_fake_provider:
        pytest.skip("the shipped thresholds belong to the real model")
    loaded = load_thresholds(SHIPPED, model_id=base.embedding_model or "Bge-m3")
    assert loaded.calibrated is True
    assert loaded.model_id == "Bge-m3"


def test_the_shipped_thresholds_still_refuse_another_models_measurements() -> None:
    """D51 on the real file, which is now a genuine calibration rather than a placeholder."""
    with pytest.raises(ConfigError, match="calibrated for"):
        load_thresholds(SHIPPED, model_id="not-this-model")


def test_the_committed_report_quotes_the_committed_thresholds() -> None:
    """The report and the file are two halves of one claim; if they drift, one is a lie."""
    payload = json.loads(SHIPPED.read_text(encoding="utf-8"))
    report = (ROOT / "tests" / "eval" / "report.md").read_text(encoding="utf-8")
    for name in (
        "min_dense",
        "min_dense_alone",
        "min_coverage",
        "min_lexical",
        "min_coverage_high",
    ):
        assert f'"{name}": {payload[name]}' in report, (
            f"{name} is {payload[name]} in config/thresholds.json but not in the report"
        )
    assert f"questions: **{payload['dataset_size']}**" in report


# ------------------------------------------------------------------ the calibration


async def test_the_recalibrated_gate_refuses_the_densest_unanswerable_question(
    graph: Services,
) -> None:
    """D47 measured one unanswerable question admitted at the same similarity as the real
    cross-lingual ones. The recalibrated `min_dense_alone` is above it, and this is the
    measurement that says so -- the case is the densest unanswerable in the corpus."""
    answer = await graph.retrieval.answer(
        "How do I configure a second listening port for the gateway?", debug=True
    )
    assert answer.status == "insufficient_information", (
        f"the recalibrated gate admits the densest unanswerable case: {answer.answer[:120]}"
    )
    assert answer.citations == ()
    assert answer.debug is not None
    # And the reason is a threshold, not a retrieval accident.
    assert answer.debug["gate"]["candidate_count"] > 0
    assert answer.debug["gate"]["max_dense"] < graph.thresholds.min_dense_alone


async def test_the_recalibrated_gate_answers_two_of_four_cross_lingual_questions(
    graph: Services,
) -> None:
    """The capability the uncorroborated dense branch exists for (D47, D48, section 2) --
    after a calibration that moved ``min_dense_alone`` from 0.50 to 0.66.

    Measured: **2 of 4**. ``x01`` (dense 0.618) and ``m05`` (0.628) are answered through
    dense+coverage. ``m03`` (dense 0.540, coverage 0.19) and ``x08`` (dense 0.482) are refused,
    and both are the weakest cross-lingual cases by any measure, so the 2-of-4 is not two
    arbitrary losses.

    The refusals are asserted to be the *cosine bar* rather than a retrieval accident: the
    chunk is found and it is semantically close, so nothing below the embedding layer is
    broken. That distinction is the whole of section 12 question 3.

    Asserted as a measurement, so the number cannot drift silently. If this reaches 4, the
    trade-off has been resolved and section 2, README and D47's branch all need revisiting.
    """
    cross = [c for c in load_cases() if c.answerable and "cross-lingual" in c.note]
    assert len(cross) == 4, cross
    answered: list[str] = []
    for case in cross:
        answer = await graph.retrieval.answer(case.question, debug=True)
        gate = answer.debug["gate"]  # type: ignore[index]
        if answer.status == "answered":
            answered.append(case.id)
            assert answer.citations, case.id
        else:
            # Refused, and specifically below the bar: the hit was retrieved and is close.
            assert gate["max_dense"] > 0.4, (  # type: ignore[index]
                f"{case.id} was refused because nothing was retrieved, not because of a bar"
            )
            assert gate["max_dense"] < graph.thresholds.min_dense or (
                gate["token_coverage"] < graph.thresholds.min_coverage  # type: ignore[index]
            ), f"{case.id} was refused by some other rule than the corroborated bar"
    assert answered == ["x01", "m05"], (
        f"cross-lingual answerability is now {answered} of {[c.id for c in cross]}; the "
        "calibration's cost or benefit has moved and section 12 question 3 needs the new number"
    )


async def test_cross_lingual_answers_cite_the_wrong_language_document(graph: Services) -> None:
    """D60 on the corpus the report was measured on, asserted so it cannot be forgotten.

    All four cross-lingual questions are answered, and three of them are cited from a
    document that does not contain the answer -- the same-language document, retrieved and
    ranked ahead of the other language's. A caller who cannot read the cited language has no
    way to notice, which makes this worse than a refusal.

    The headline `false answers 0.00` does not see any of this: every one of these is an
    *answerable* question the gate was right to admit. `gold quoted` in the report is the
    number that does, and this test is that number's regression guard.
    """
    from runner import mis_cited, score

    observations = await observe(graph, load_cases(), dense_weight=0.9, lexical_weight=0.1)
    hybrid = score(observations, label="hybrid")
    assert hybrid.evidence < 1.0, (
        "every gold phrase is quoted in a cited chunk; the failure this test pins is gone, so "
        "close D60 and update the report rather than leaving it claiming a defect that no "
        "longer exists"
    )
    wrong_document = [
        (o.case.id, phrase)
        for o in observations
        for phrase, cause in mis_cited(o, graph.settings.top_k)
        if cause == "retrieval"
    ]
    assert wrong_document, (
        "no answer cites a document that lacks the answer; D60's retrieval-kind failures are "
        "all fixed"
    )
    for case_id, phrase in wrong_document:
        assert "cross-lingual" in next(c for c in load_cases() if c.id == case_id).note or (
            case_id in ("x01", "x08", "m05")
        ), f"{case_id} is a new instance of the same failure: {phrase}"


async def test_an_off_topic_question_is_refused_and_offers_no_citations(graph: Services) -> None:
    answer = await graph.retrieval.answer("What is the boiling point of mercury at 2000 metres?")
    assert answer.status == "insufficient_information"
    assert answer.citations == ()
    assert answer.evidence_score == 0.0


async def test_the_eval_corpus_is_the_only_thing_in_this_index(graph: Services) -> None:
    """`make eval` points at its own data dir on purpose (§2's 84%-one-book skew). If the
    fixture corpus leaked in, every number in the report would be about the wrong system."""
    names = {row["source_name"] for row in graph.store.eligible_chunks()}
    assert names == {
        "handbook.md",
        "runbook.txt",
        "limits.pdf",
        "deploy-guide.md",
        "incident-runbook.txt",
        "ai-engineer.pdf",
    }, names
    assert default_doc_id("handbook.md") == "handbook"
    assert len(graph.store.eligible_chunk_ids()) == 49, (
        "49 chunks is the corpus the report was measured on; a different count means the "
        "committed report describes a corpus that no longer exists"
    )


async def test_republishing_a_document_leaks_no_stale_text(graph: Services) -> None:
    """I1/I2 measured on the corpus the report describes, not only on the fixtures.

    This is the same measurement `run()` reports as `stale_leaks: 0`, narrowed to one
    document and one phrase so a regression names itself.
    """
    document = CORPUS / "en" / "handbook.md"
    removed = "0.0.0.0:8443"
    original = document.read_bytes()
    assert removed in original.decode("utf-8")

    cases = [c for c in load_cases() if c.answerable]
    await graph.ingestion.ingest(
        original.decode("utf-8").replace(removed, "REDACTED").encode("utf-8"), document.name
    )
    try:
        for case in cases:
            answer = await graph.retrieval.answer(case.question)
            assert removed not in answer.answer, (
                f"{case.id} still quotes text from a superseded version"
            )
    finally:
        await graph.ingestion.ingest(original, document.name)

    assert (
        "0.0.0.0:8443"
        in (
            await graph.retrieval.answer("What port does the Aurora gateway listen on by default?")
        ).answer
    ), "the restored version is not reachable again"
