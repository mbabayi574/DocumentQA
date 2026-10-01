"""P7 §7.2: the evidence gate and the thresholds it reads.

The gate is the only thing standing between "we found some text" and "we answer from
it", so its boundaries are pinned exactly and its thresholds are proven not to leak
across models (the same guard as I9, one layer up).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qasystem.errors import ConfigError
from qasystem.retrieval.gate import (
    GateSignals,
    Thresholds,
    defaults,
    evaluate,
    load_thresholds,
)

MODEL = "Bge-m3"


def thresholds(**overrides: Any) -> Thresholds:
    base: dict[str, Any] = {
        "min_dense": 0.60,
        "min_coverage": 0.30,
        "min_lexical": 0.80,
        "min_coverage_high": 0.50,
        "min_sentence_overlap": 0.15,
        "version": 1,
        "calibrated": False,
        "model_id": MODEL,
    }
    base.update(overrides)
    return Thresholds(**base)


def signals(**overrides: Any) -> GateSignals:
    base: dict[str, Any] = {
        "max_dense": 0.75,
        "lexical": 0.90,
        "token_coverage": 0.60,
        "overlap": 0.50,
        "candidate_count": 5,
    }
    base.update(overrides)
    return GateSignals(**base)


# ---------------------------------------------------------------- the rule


def test_a_strong_dense_hit_with_coverage_passes() -> None:
    verdict = evaluate(signals(max_dense=0.75, token_coverage=0.60), thresholds())
    assert verdict.passed is True
    assert verdict.reason == "dense+coverage"


def test_a_strong_lexical_hit_with_high_coverage_passes() -> None:
    verdict = evaluate(signals(max_dense=0.10, lexical=0.90, token_coverage=0.60), thresholds())
    assert verdict.passed is True
    assert verdict.reason == "lexical+coverage"


def test_weak_signals_refuse() -> None:
    verdict = evaluate(signals(max_dense=0.10, lexical=0.10, token_coverage=0.10), thresholds())
    assert verdict.passed is False
    assert verdict.reason == "below_threshold"


def test_a_dense_hit_without_coverage_refuses() -> None:
    """The whole point: a semantically close chunk that never contains the query's words."""
    verdict = evaluate(signals(max_dense=0.95, lexical=0.95, token_coverage=0.20), thresholds())
    assert verdict.passed is False
    assert verdict.reason == "below_threshold"


def test_coverage_without_any_similarity_refuses() -> None:
    verdict = evaluate(signals(max_dense=0.05, lexical=0.0, token_coverage=1.0), thresholds())
    assert verdict.passed is False


def test_no_candidates_cannot_pass_however_good_the_signals_look() -> None:
    verdict = evaluate(signals(candidate_count=0, max_dense=0.99, token_coverage=1.0), thresholds())
    assert verdict.passed is False


# ---------------------------------------------------------------- boundaries


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("max_dense", 0.60, True),  # exactly at min_dense: inclusive
        ("max_dense", 0.59, False),
        ("lexical", 0.80, True),
        ("lexical", 0.79, False),
    ],
)
def test_similarity_thresholds_are_inclusive(field: str, value: float, expected: bool) -> None:
    # The arm not under test is pinned below its own threshold, otherwise the other branch
    # would pass regardless and the boundary being tested would prove nothing.
    overrides: dict[str, Any] = {
        "max_dense": value if field == "max_dense" else 0.0,
        "lexical": value if field == "lexical" else 0.0,
        "token_coverage": 0.60,
    }
    assert evaluate(signals(**overrides), thresholds()).passed is expected


@pytest.mark.parametrize(
    ("coverage", "dense", "expected"),
    [
        (0.30, 0.99, True),  # exactly min_coverage: inclusive
        (0.29, 0.99, False),
        (0.50, 0.10, True),  # exactly min_coverage_high on the lexical branch
        (0.49, 0.10, False),
        (0.45, 0.10, False),  # between the two: the lexical branch is stricter
    ],
)
def test_coverage_thresholds_are_inclusive_and_branch_specific(
    coverage: float, dense: float, expected: bool
) -> None:
    verdict = evaluate(
        signals(max_dense=dense, lexical=0.99, token_coverage=coverage), thresholds()
    )
    assert verdict.passed is expected


def test_the_two_branches_are_not_redundant() -> None:
    """Coverage between the two bars passes on dense and fails on lexical only."""
    t = thresholds()
    mid = (t.min_coverage + t.min_coverage_high) / 2
    on_dense = evaluate(signals(max_dense=0.99, lexical=0.0, token_coverage=mid), t)
    on_lexical = evaluate(signals(max_dense=0.0, lexical=0.99, token_coverage=mid), t)
    assert on_dense.passed is True and on_lexical.passed is False


# ---------------------------------------------------------------- the report


def test_the_verdict_carries_every_signal_for_debugging() -> None:
    verdict = evaluate(
        signals(max_dense=0.71, lexical=0.83, token_coverage=0.5, overlap=0.25),
        thresholds(version=7, calibrated=True),
    )
    report = verdict.as_dict()
    assert report["passed"] is True
    assert report["reason"] == "dense+coverage"
    assert report["max_dense"] == pytest.approx(0.71)
    assert report["lexical"] == pytest.approx(0.83)
    assert report["token_coverage"] == pytest.approx(0.5)
    assert report["overlap"] == pytest.approx(0.25)
    assert report["thresholds_version"] == 7
    assert report["calibrated"] is True
    assert report["model_id"] == MODEL


def test_signals_are_clamped_into_the_unit_interval() -> None:
    """A cosine of 1.0000001 from float arithmetic must not leak past 1.0."""
    verdict = evaluate(signals(max_dense=1.0000001, token_coverage=2.0), thresholds())
    assert verdict.signals.max_dense <= 1.0
    assert verdict.signals.token_coverage <= 1.0
    assert verdict.as_dict()["max_dense"] <= 1.0


# ---------------------------------------------------------------- thresholds


def test_the_shipped_defaults_are_flagged_uncalibrated() -> None:
    shipped = defaults(MODEL)
    assert shipped.calibrated is False
    assert shipped.model_id == MODEL


def test_thresholds_load_from_json(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "model_id": MODEL,
                "calibrated": True,
                "min_dense": 0.5,
                "min_coverage": 0.2,
                "min_lexical": 0.7,
                "min_coverage_high": 0.4,
                "min_sentence_overlap": 0.1,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_thresholds(path, model_id=MODEL)
    assert loaded.version == 3
    assert loaded.calibrated is True
    assert loaded.min_dense == 0.5


def test_thresholds_for_another_model_are_refused(tmp_path: Path) -> None:
    """Thresholds from one model must never answer for another (the I9 rule, one layer up)."""
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "model_id": "some-other-model",
                "calibrated": True,
                "min_dense": 0.5,
                "min_coverage": 0.2,
                "min_lexical": 0.7,
                "min_coverage_high": 0.4,
                "min_sentence_overlap": 0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="some-other-model"):
        load_thresholds(path, model_id=MODEL)


def test_a_missing_thresholds_file_falls_back_to_uncalibrated_defaults(tmp_path: Path) -> None:
    loaded = load_thresholds(tmp_path / "absent.json", model_id=MODEL)
    assert loaded == defaults(MODEL)
    assert loaded.calibrated is False


def test_a_corrupt_thresholds_file_falls_back_to_uncalibrated_defaults(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_thresholds(path, model_id=MODEL) == defaults(MODEL)


def test_the_shipped_thresholds_file_is_well_formed_and_uncalibrated() -> None:
    """P7 ships conservative placeholders; P9 overwrites them. Until then it must say so."""
    from qasystem.config import load_settings

    path = load_settings(env_file=None, embedding_provider="fake", app_env="test").thresholds_path
    assert path.exists(), "P7 ships config/thresholds.json; make calibrate overwrites it (P9)"
    shipped = json.loads(path.read_text(encoding="utf-8"))
    assert shipped["calibrated"] is False, "claiming calibration before P9 would be a lie"
    assert shipped["model_id"] == "Bge-m3"
    loaded = load_thresholds(path, model_id=shipped["model_id"])
    assert loaded.calibrated is False
    assert 0.0 <= loaded.min_dense <= 1.0


def test_a_threshold_outside_the_unit_interval_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": MODEL,
                "calibrated": True,
                "min_dense": 1.5,
                "min_coverage": 0.2,
                "min_lexical": 0.7,
                "min_coverage_high": 0.4,
                "min_sentence_overlap": 0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_thresholds(path, model_id=MODEL)


def test_coverage_high_below_coverage_is_refused(tmp_path: Path) -> None:
    """The lexical branch is only stricter if the two bars are ordered."""
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": MODEL,
                "calibrated": True,
                "min_dense": 0.5,
                "min_coverage": 0.6,
                "min_lexical": 0.7,
                "min_coverage_high": 0.2,
                "min_sentence_overlap": 0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_thresholds(path, model_id=MODEL)
