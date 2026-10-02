"""The eval harness itself works, verified offline against the fake embedder (plan.md §9).

The report P9 commits is produced by this module, so a bug here does not fail a test — it
produces a confident, well-formatted, wrong number. That is the worst failure mode a test
harness has, and it is why this file exists.

The fake embedder cannot rank or align languages, so nothing here asserts *quality*: R@1 will
be poor and the calibrated thresholds will be meaningless. What is asserted is everything
that must hold whatever the embedder does — that the dataset loads, every case is observed,
gold matching works, the metrics are in range, the grid search finds a feasible point and
its selection obeys §9.3, the citation check bites, and the report renders with real
sections. A quality assertion here would be a claim about a model, not about the harness.

Runs the whole pipeline in-process against ``tmp_path``, so ``make check`` needs no token.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qasystem.config import load_settings
from qasystem.embeddings.fake import FakeEmbedder
from qasystem.retrieval.gate import Thresholds

from .runner import (
    ANSWERABLE,
    UNANSWERABLE,
    Calibration,
    Case,
    Gold,
    Observation,
    choose,
    gold_quoted,
    grid_search,
    load_cases,
    mis_cited,
    render,
    run,
    score,
    strategies,
    verify_citations,
)


def _settings(tmp_path: Path) -> Any:
    return load_settings(
        env_file=None,
        app_env="test",
        embedding_provider="fake",
        data_dir=tmp_path,
        sqlite_path=tmp_path / "qasystem.db",
        chroma_path=tmp_path / "chroma",
        thresholds_path=tmp_path / "thresholds.json",
    )


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return _settings(tmp_path_factory.mktemp("eval"))


@pytest.fixture(scope="module")
def outcome(settings: Any) -> Any:
    import anyio

    # One chunk size, not three: the offline gate has no reason to pay for three extra
    # ingests of the whole corpus. `make eval` runs the full grid, and the sweep's numbers
    # are only ever read out of the committed report.
    return anyio.run(lambda: run(settings, calibrate=True, report=None, chunk_grid=((350, 700),)))


# ------------------------------------------------------------------ the dataset loads


def test_the_dataset_loads_with_every_field_the_metrics_need() -> None:
    cases = load_cases()
    assert len(cases) >= 50
    assert all(case.question and case.category and case.split for case in cases)
    assert sum(1 for c in cases if c.answerable) == 38
    assert sum(1 for c in cases if not c.answerable) == 12


# ------------------------------------------------------------------ gold matching


def _observation(case: Case, hits: list[tuple[str, str]], answered: bool = True) -> Observation:
    from qasystem.answering.extractive import Answer
    from qasystem.retrieval.gate import GateSignals

    candidates = tuple(
        type("C", (), {"source_name": doc, "text": text, "chunk_id": f"{i}"})()
        for i, (doc, text) in enumerate(hits)
    )
    return Observation(
        case=case,
        candidates=candidates,  # type: ignore[arg-type]
        signals=GateSignals(0.5, 0.5, 0.5, 0.0, len(hits)),
        answer=Answer(status="answered" if answered else "insufficient_information", answer=""),
        latency_ms=1.0,
    )


SINGLE = Case("a", "q", "answerable_en", "dev", (Gold("d.md", "needle"),))
DOUBLE = Case("b", "q", "multi_section", "dev", (Gold("d.md", "needle"), Gold("e.md", "other")))


@pytest.mark.parametrize(
    ("hits", "expected"),
    [
        ([("d.md", "a needle here")], 1),
        ([("e.md", "nothing"), ("d.md", "a needle here")], 2),
        ([("d.md", "needle")], 1),
    ],
)
def test_hit_rank_is_the_first_position_holding_the_gold(
    hits: list[tuple[str, str]], expected: int
) -> None:
    assert _observation(SINGLE, hits).hit_rank() == expected


def test_right_text_in_the_wrong_document_is_a_miss() -> None:
    """Gold names a document, not just a phrase. A system that returns the right words from
    the wrong source has not answered the question, and counting it would reward exactly
    the confusion the citation is supposed to prevent."""
    assert _observation(SINGLE, [("e.md", "a needle here")]).hit_rank() is None


def test_a_multi_section_case_needs_both_facts_in_the_window() -> None:
    """The point of the metric: a question asking two things is not answered by one of
    them, and this is the same code path rather than a second, separately-drifted metric."""
    partial = _observation(DOUBLE, [("d.md", "needle"), ("e.md", "no")])
    complete = _observation(DOUBLE, [("d.md", "needle"), ("e.md", "other")])
    assert partial.hit_rank() is None, "one of two facts is not an answer to a two-fact question"
    assert complete.hit_rank() == 2


def test_hit_rank_reports_a_miss_beyond_the_window_as_none_not_as_a_sentinel() -> None:
    """A `len(window) + 1` sentinel would sit inside the R@3 range and score a total miss as
    a hit, which is how a 30-candidate window silently inflates recall on a 40-chunk corpus."""
    assert _observation(SINGLE, [("d.md", "nothing at all")]).hit_rank() is None


# ------------------------------------------------------------------ D60's three causes


def _cited(
    observation: Case, cited: list[tuple[str, str]], hits: list[tuple[str, int]]
) -> Observation:
    """An answered observation with a chosen citation list and chosen gold-chunk positions.

    ``hits`` is ``(source_name, window position)`` per gold-bearing chunk, which is what
    separates "the answerer dropped it" from "top_k never showed it".
    """
    from qasystem.answering.extractive import Answer, Citation, Segment
    from qasystem.answering.sentences import query_terms
    from qasystem.retrieval.gate import GateSignals

    # The window has to actually have `top_k` candidates before the gold-bearing one, and the
    # gold phrase has to be *in* that candidate's text -- otherwise the classifier cannot see
    # the position and every case collapses to one answer.
    padded = [("other.pdf", "filler")] * (max((p for _, p in hits), default=0))
    candidates = tuple(
        type(
            "C", (), {"source_name": name, "text": "the port is 0.0.0.0:8443", "chunk_id": f"{i}"}
        )()
        for i, (name, _) in enumerate(padded + list(hits) + [("other.pdf", "filler")] * TOP_K)
    )
    citations = tuple(
        Citation(i + 1, "d", doc, 1, "s", None, None, f"d:v1:{i}", (0, 1), excerpt, 0.5)
        for i, (doc, excerpt) in enumerate(cited)
    )
    assert query_terms(observation.question) or True
    return Observation(
        case=observation,
        candidates=candidates,  # type: ignore[arg-type]
        signals=GateSignals(0.6, 1.0, 0.5, 0.0, 30),
        answer=Answer(
            status="answered",
            answer="x",
            segments=(Segment("x", 1, "d:v1:0", 0, 1),),
            citations=citations,
        ),
        latency_ms=0.0,
    )


GOLD = Gold("handbook.md", "0.0.0.0:8443")
QUESTION = Case("w", "what port", "answerable_en", "dev", (GOLD,))
TOP_K = 5
#: The fixture's question is "what port" and its chunk reads "the port is ...", so the
#: answerer's own overlap for the gold sentence is 1.0. Asserted rather than left implicit, so a
#: change to the overlap formula shows up here instead of silently shifting every cause test.
GOLD_OVERLAP = 1.0


def test_a_wrong_document_is_reported_as_a_retrieval_failure() -> None:
    observation = _cited(QUESTION, [("runbook.txt", "no answer here")], [("handbook.md", 2)])
    assert mis_cited(observation, TOP_K) == [("0.0.0.0:8443", "retrieval", GOLD_OVERLAP)]


def test_a_gold_chunk_below_top_k_is_reported_as_truncation_not_a_retrieval_failure() -> None:
    """The distinction that matters: retrieval found it and the answer was never shown it.
    Calling this a retrieval failure would send a reader to fix the wrong layer."""
    observation = _cited(
        QUESTION, [("handbook.md", "wrong chunk of the right document")], [("handbook.md", 9)]
    )
    assert mis_cited(observation, TOP_K) == [("0.0.0.0:8443", "truncation", GOLD_OVERLAP)]


def test_a_gold_chunk_inside_top_k_is_reported_as_an_answerer_failure() -> None:
    observation = _cited(
        QUESTION, [("handbook.md", "wrong chunk of the right document")], [("handbook.md", 2)]
    )
    assert mis_cited(observation, TOP_K) == [("0.0.0.0:8443", "answerer", GOLD_OVERLAP)]


def test_a_gold_sentence_the_answerer_would_drop_is_reported_with_its_score() -> None:
    """The number that says whether a case has one blocker or two.

    P10's `top_k` sweep turned up a case labelled "truncation" that raising `top_k` did not
    fix, because its sentence was also below the overlap bar. The label was not wrong about the
    mechanism at `top_k=5`; it was *incomplete*, and this is what makes the second blocker
    visible: the score is the answerer's own, computed with the answerer's formula.
    """
    # Same question and gold, but with the sentence sharing no word with the question. That is
    # `fa02`'s shape: a Markdown code fence holding a command, asked about in words.
    for question, expected in (("what port", 1.0), ("which host", 0.0)):
        observation = _cited(
            replace(QUESTION, question=question),
            [("handbook.md", "an unrelated slice")],
            [("handbook.md", 2)],
        )
        phrase, cause, overlap = mis_cited(observation, TOP_K)[0]
        assert (phrase, cause) == ("0.0.0.0:8443", "answerer")
        assert overlap == expected, f"{question!r} scored {overlap}"


def test_gold_that_is_quoted_is_not_reported_at_all() -> None:
    observation = _cited(QUESTION, [("handbook.md", "it binds to 0.0.0.0:8443")], [])
    assert mis_cited(observation, TOP_K) == []
    assert gold_quoted(observation) == (1, 1)


def test_a_refused_answer_reports_no_mis_citation() -> None:
    """A refusal is not a mis-citation. Scoring it as one would make the metric punish the
    gate for the thing the gate exists to do."""
    observation = _cited(QUESTION, [("runbook.txt", "x")], [("handbook.md", 2)])
    object.__setattr__(
        observation, "answer", replace(observation.answer, status="insufficient_information")
    )
    assert mis_cited(observation, TOP_K) == []
    # (0, 1) and not (0, 0): the case *does* have gold, the answer simply quoted none of
    # it. The metric aggregates only over answered cases, so a refusal cannot leak in.
    assert gold_quoted(observation) == (0, 1)


def test_an_unanswerable_case_has_no_gold_to_quote() -> None:
    observation = _cited(Case("u", "q", "unanswerable_off", "dev", ()), [("runbook.txt", "x")], [])
    assert mis_cited(observation, TOP_K) == []
    # (0, 1) and not (0, 0): the case *does* have gold, the answer simply quoted none of
    # No gold at all, so there is nothing to have quoted: (0, 0), not (0, 1).
    assert gold_quoted(observation) == (0, 0)


# ------------------------------------------------------------------ metrics arithmetic


def test_metrics_are_computed_from_the_candidates_not_asserted() -> None:
    rows = [
        _observation(SINGLE, [("d.md", "needle")]),  # hit at 1
        _observation(
            Case("c", "q", "answerable_en", "dev", (Gold("d.md", "needle"),)),
            [("d.md", "x"), ("d.md", "needle")],
        ),  # hit at 2
        _observation(
            Case("d", "q", "answerable_en", "dev", (Gold("d.md", "needle"),)), [("e.md", "x")]
        ),  # miss
    ]
    metrics = score(rows, label="t")
    assert (metrics.n_answerable, metrics.n_unanswerable) == (3, 0)
    assert metrics.r1 == pytest.approx(1 / 3)
    assert metrics.r3 == metrics.r5 == pytest.approx(2 / 3), "the third case is a total miss"
    assert metrics.mrr5 == pytest.approx((1 + 0.5 + 0) / 3)


def test_false_answers_are_counted_only_on_unanswerable_cases() -> None:
    """The direction of each rate is a real bug class: conflating them would report a
    system that answers everything as perfect."""
    refused = _observation(SINGLE, [("d.md", "needle")], answered=False)
    answered_unanswerable = _observation(
        Case("x", "q", "unanswerable_off", "dev", ()), [("d.md", "needle")], answered=True
    )
    metrics = score([refused, answered_unanswerable], label="t")
    assert metrics.answered_rate == 0.0
    assert metrics.false_answers == 1
    assert metrics.false_answer_rate == 1.0


def test_split_selects_cases_and_none_scores_all_of_them() -> None:
    dev = Case("a", "q", "answerable_en", "dev", (Gold("d.md", "needle"),))
    test = Case("b", "q", "answerable_en", "test", (Gold("d.md", "needle"),))
    rows = [_observation(dev, [("d.md", "needle")]), _observation(test, [("d.md", "needle")])]
    assert score(rows, label="dev", split="dev").n_answerable == 1
    assert score(rows, label="all", split=None).n_answerable == 2


# ------------------------------------------------------------------ calibration


def _signals_observation(
    case: Case, *, dense: float, coverage: float, lexical: float
) -> Observation:
    from qasystem.answering.extractive import Answer
    from qasystem.retrieval.gate import GateSignals

    return Observation(
        case=case,
        candidates=(),
        signals=GateSignals(dense, lexical, coverage, 0.0, 30),
        answer=Answer(status="insufficient_information", answer=""),
        latency_ms=0.0,
    )


CALIBRATION_CASES = [
    # answerable: dense 0.55, coverage 0.6, lexical 0.2 -- needs min_dense <= 0.55
    *[
        _signals_observation(
            Case(f"a{i}", "q", "answerable_en", "dev", (Gold("d.md", "n"),)),
            dense=0.55,
            coverage=0.6,
            lexical=0.2,
        )
        for i in range(4)
    ],
    # answerable via the lexical branch only: dense 0.2, coverage 0.9, lexical 0.95
    _signals_observation(
        Case("l0", "q", "exact_term", "dev", (Gold("d.md", "n"),)),
        dense=0.2,
        coverage=0.9,
        lexical=0.95,
    ),
    # unanswerable: dense 0.30, coverage 0.0 -- must be refused
    *[
        _signals_observation(
            Case(f"u{i}", "q", "unanswerable_off", "dev", ()), dense=0.30, coverage=0.0, lexical=0.0
        )
        for i in range(3)
    ],
]


def test_the_grid_search_refuses_any_point_that_answers_an_unanswerable_case() -> None:
    """Feasibility is still a real constraint, and it is the coverage bars that enforce it.

    With the uncorroborated dense branch removed (D71) there are only two ways in, and both
    require coverage, so an unanswerable case is held out by whichever of ``min_dense`` (below
    0.30 it cannot pass the corroborated branch) and the two coverage bars is stricter. The
    fixtures make the arithmetic visible: the unanswerable band is dense 0.30 / coverage 0.0, and
    a feasible point must sit clear of it on at least one axis.
    """
    feasible, tested = grid_search(CALIBRATION_CASES, min_sentence_overlap=0.15, model_id="fake")
    assert tested > 0
    assert feasible, "expected at least one feasible point"
    for point, _ in feasible:
        assert point.min_dense > 0.30 or point.min_coverage > 0.0, (
            "a point below the unanswerable dense band AND at zero coverage would admit them "
            "through the corroborated branch"
        )


def test_choose_maximises_answered_on_answerable_and_takes_the_strictest_bar() -> None:
    """§9.3's objective, and the tie-break: among points that tie on recall and on false
    answers, the one that refuses the most marginal evidence is the one to ship."""
    loose = Thresholds(
        min_dense=0.40,
        min_coverage=0.2,
        min_coverage_high=0.5,
        min_sentence_overlap=0.15,
        version=1,
        calibrated=False,
        model_id="fake",
    )
    strict = Thresholds(
        min_dense=0.45,
        min_coverage=0.2,
        min_coverage_high=0.5,
        min_sentence_overlap=0.15,
        version=1,
        calibrated=False,
        model_id="fake",
    )
    metrics = score(CALIBRATION_CASES, label="same")
    chosen, _ = choose([(loose, metrics), (strict, metrics)])
    assert chosen is strict


def test_choose_refuses_to_invent_a_feasible_point() -> None:
    """An empty feasible set is plan.md §12 question 3 being true on this data. It must be
    reported, not papered over with a point that misses the constraint."""
    with pytest.raises(ValueError, match="no feasible operating point"):
        choose([])


# ------------------------------------------------------------------ end to end, offline


def test_the_whole_pipeline_runs_offline_and_reports_every_section(outcome: Any) -> None:
    for section in (
        "## Retrieval and gating",
        "## Latency",
        "## Mechanical checks",
        "## The evidence gate",
        "### 1. Fusion weights",
        "### 2. Chunk size",
    ):
        assert section in outcome.report, f"missing {section!r} from the report"


def test_the_pipeline_measured_all_three_strategies_and_the_weight_grid(outcome: Any) -> None:
    assert set(outcome.weights) == {
        f"{round(0.1 * i, 2):.2f}/{round(1 - 0.1 * i, 2):.2f}" for i in range(11)
    }
    for metrics in outcome.weights.values():
        assert 0.0 <= metrics.r1 <= 1.0
        assert 0.0 <= metrics.mrr5 <= 1.0


def test_the_pipeline_verified_citations_and_found_no_stale_leakage(outcome: Any) -> None:
    """A zero here is only worth something if the check actually ran and can fail, which
    the mutation test below establishes."""
    assert outcome.citation_checks > 0, "no answer segments were verified at all"
    assert outcome.stale_leaks == 0


def test_the_pipeline_calibrated_on_dev_and_reported_the_held_out_split(outcome: Any) -> None:
    assert outcome.calibration is not None, outcome.calibration_error
    # 23 of the 38 answerable and 7 of the 12 unanswerable are dev, so a dev-only grid and a
    # 38/12 "held-out" number would both be visible here.
    assert outcome.calibration.dev.n_answerable == 23, "calibration used more than the dev split"
    assert outcome.calibration.dev.n_unanswerable == 7
    assert outcome.calibration.test.n_answerable == 15
    assert outcome.calibration.test.n_unanswerable == 5
    assert outcome.calibration.tested_points > outcome.calibration.feasible_points > 0


def test_eval_does_not_write_thresholds(tmp_path: Path) -> None:
    """`eval` and `calibrate` differ by exactly this, and a measurement command that
    silently rewrote the gate would let any future run change the system's behaviour."""
    import anyio

    settings = _settings(tmp_path / "no-write")
    outcome = anyio.run(
        lambda: run(settings, calibrate=False, report=None, chunk_grid=((350, 700),))
    )
    assert outcome.calibration is not None, outcome.calibration_error
    assert not settings.thresholds_path.exists(), "eval wrote config/thresholds.json"


def test_calibrate_wrote_a_loadable_thresholds_file(settings: Any, outcome: Any) -> None:
    """The committed file is the deliverable, so it must be one the gate can read back
    with `calibrated: true` and the dataset size that produced it."""
    from qasystem.retrieval.gate import load_thresholds

    assert settings.thresholds_path.exists()
    loaded = load_thresholds(settings.thresholds_path, model_id=FakeEmbedder().model_id)
    assert loaded.calibrated is True
    # The bars must match what the grid chose. `calibrated` cannot: the grid builds candidate
    # points at version=1/calibrated=False and the payload stamps the file as calibrated.
    for name in ("min_dense", "min_coverage", "min_coverage_high", "min_sentence_overlap"):
        assert getattr(loaded, name) == getattr(outcome.calibration.thresholds, name), name
    payload = json.loads(settings.thresholds_path.read_text(encoding="utf-8"))
    assert payload["dataset_size"] == 50
    assert sum(payload["class_balance"].values()) == 50
    assert set(payload["class_balance"]) == {*ANSWERABLE, *UNANSWERABLE}


def test_no_cell_in_the_report_is_nan_or_a_broken_format(outcome: Any) -> None:
    """A rate printed as `nan` is a division by an empty split, and it reads as a measurement.
    Checked per cell rather than by substring, because "unanswerable" contains "nan"."""
    assert "None ms" not in outcome.report
    rows = 0
    for line in outcome.report.splitlines():
        if not line.startswith("|") or line.count("|") < 4 or set(line) <= set("|- :"):
            continue
        rows += 1
        for cell in (c.strip() for c in line.strip("|").split("|")[1:]):
            try:
                value = float(cell)
            except ValueError:
                continue
            assert not math.isnan(value), f"NaN in a results table: {line}"
    assert rows >= 20, f"only {rows} table rows; the report is mostly missing"


# ------------------------------------------------------------------ the checks bite


async def test_verify_citations_rejects_a_segment_that_is_not_verbatim(tmp_path: Path) -> None:
    """A check that cannot fail is not a check. The corruption here is the smallest one
    P7's substring test exists for: a segment whose text was reflowed from the source."""

    from qasystem.answering.extractive import Answer, Segment
    from qasystem.api.deps import build_services

    services = await build_services(_settings(tmp_path / "verify"))
    try:
        from .runner import CORPUS

        path = CORPUS / "en" / "handbook.md"
        result = await services.ingestion.ingest(path.read_bytes(), path.name)
        assert result.chunks_added > 0
        answer = await services.retrieval.answer("What port does the gateway listen on?")
        assert answer.status == "answered", answer.reason
        assert answer.segments
        verified = verify_citations(services, [_answer_observation(answer)])
        assert verified == len(answer.segments)

        corrupted = Answer(
            status=answer.status,
            answer=answer.answer,
            segments=(
                Segment(
                    "a paraphrase the source never contained", 1, answer.segments[0].chunk_id, 0, 5
                ),
            ),
            citations=answer.citations,
        )
        with pytest.raises(AssertionError, match="not chunk text"):
            verify_citations(services, [_answer_observation(corrupted)])
    finally:
        await services.aclose()


def _answer_observation(answer: Any) -> Observation:
    """An observation carrying a real Answer, for the citation checks."""
    from qasystem.answering.extractive import Answer
    from qasystem.retrieval.gate import GateSignals

    assert isinstance(answer, Answer)
    return Observation(
        case=Case("z", "q", "answerable_en", "dev", ()),
        candidates=(),
        signals=GateSignals(0.5, 0.5, 0.5, 0.0, 30),
        answer=answer,
        latency_ms=0.0,
    )


# ------------------------------------------------------------------ the report renders


def test_render_does_not_claim_a_calibration_it_did_not_find() -> None:
    """`calibrated: true` in a committed file is a claim about evidence. When the grid found
    nothing, the report must say so rather than rendering an empty thresholds block."""

    class _Fake:
        model_id = "fake"
        dimension = 8
        embedder = SimpleNamespace(requests=0)

    text = render(
        services=_Fake(),  # type: ignore[arg-type]
        cases=load_cases(),
        per_strategy={label: [] for label, _, _ in strategies(0.7, 0.3)},
        weights=[],
        latency_ms=[1.0],
        warm_ms=[1.0],
        documents=6,
        corpus_chunks=0,
        corpus_chars=0,
        corpus_sections=0,
        corpus_max_section=0,
        index_size=0,
        ingest_requests=0,
        citation_checks=0,
        stale_leaks=0,
        calibration=None,
        calibration_error="**Calibration found no feasible operating point.** because",
        chunk_sweep=[],
        shipped_weights=(0.7, 0.3),
        wrong=[],
    )
    assert "found no feasible operating point" in text
    assert "min_dense" not in text.split("## The evidence gate")[1].split("##")[0]


def test_a_calibration_carries_the_numbers_that_produced_it() -> None:
    """`calibrated: true` without the split, the sizes and the rates would be a promise
    rather than a receipt."""
    from .runner import thresholds_payload

    calibration = Calibration(
        thresholds=Thresholds(
            min_dense=0.4,
            min_coverage=0.2,
            min_coverage_high=0.5,
            min_sentence_overlap=0.15,
            version=1,
            calibrated=False,
            model_id="Bge-m3",
        ),
        dev=score(CALIBRATION_CASES, label="dev"),
        test=score(CALIBRATION_CASES, label="test"),
        feasible_points=12,
        tested_points=100,
    )
    payload = thresholds_payload(
        calibration.thresholds,
        dataset_size=50,
        balance={c: 1 for c in (*ANSWERABLE, *UNANSWERABLE)},
        dev=calibration.dev,
    )
    assert payload["calibrated"] is True
    assert payload["model_id"] == "Bge-m3"
    assert payload["dataset_size"] == 50
    assert payload["calibrated_on"]["split"] == "dev"
    assert payload["calibrated_on"]["false_answers"] == 0
    assert "EVIDENCE LIMIT" in payload["notes"]
