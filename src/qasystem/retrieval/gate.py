"""The evidence gate (plan.md §7.2).

This runs **before** any answer text is chosen. It is the only thing standing between
"retrieval returned some text" and "the system asserts that text answers the question",
so it is deliberately a small, readable, auditable rule rather than a score:

```text
passed = (max_dense >= min_dense           AND coverage >= min_coverage)
      OR (max_dense >= min_dense_alone)                    # uncorroborated, see below
      OR (coverage   >= min_coverage_high)                  # exact terms, see below
```

Three disjuncts, because there are three evidentiary situations, and they are not the same:

* **Dense and lexical agree** -- the query's words are in the chunk *and* the embedding is
  close. The strongest evidence, so the dense bar is the lowest.
* **Dense alone.** ``token_coverage`` counts *shared tokens*, so for the cross-lingual case
  §2 is built around -- an English question against a Persian source -- coverage is not
  merely unmet but **structurally zero**. Requiring it there made the gate unable to
  authorise the system's own headline capability, and the live run refused a correctly
  retrieved cross-lingual answer every time (D47). This branch exists for that case and
  demands a *higher* dense bar, because nothing corroborates it.
* **Exact terms** -- identifiers and numbers, where the match is exact and coverage is the
  evidence. This branch asks for no similarity at all, which is right for ``ERR-404``: a chunk
  containing every word of the question is the answer whatever the embedding thinks.

  It used to read ``lexical >= min_lexical AND coverage >= min_coverage_high``, and
  ``min_lexical`` was removed in P10 because it could not discriminate anything. ``lexical_score``
  is ``bm25 / best_bm25 of the same query``, so the top lexical hit is exactly **1.0 for every
  query FTS matched at all** -- measured at 1.00 for 49 of the 50 eval questions, answerable and
  unanswerable alike (D58). Since FTS indexes the same tokenizer output that ``token_coverage``
  counts, ``coverage > 0`` implies FTS matched, so the clause was always true whenever it could
  be: the branch had one live setting and the threshold was decoration. Deleting it is a
  behaviour-preserving change, and ``test_the_retired_lexical_threshold_cannot_change_a_verdict``
  is what holds that claim.

Thresholds are keyed by ``model_id``: a similarity of 0.6 means something specific about the
model that produced it, so another model's numbers are refused rather than reused (the I9
rule, one layer up).

A missing or corrupt file falls back to conservative **uncalibrated** defaults, because
refusing to start would make the system useless rather than merely cautious. A file whose
``model_id`` does not match the embedder's is a hard ``ConfigError``: those are not
placeholders, they are real numbers that would answer for the wrong model.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qasystem.errors import ConfigError

logger = logging.getLogger(__name__)

THRESHOLD_FIELDS = (
    "min_dense",
    "min_dense_alone",
    "min_coverage",
    "min_coverage_high",
    "min_sentence_overlap",
)


@dataclass(frozen=True)
class Thresholds:
    """The gate's constants, plus the identity of the model they belong to."""

    min_dense: float
    min_dense_alone: float
    min_coverage: float
    min_coverage_high: float
    min_sentence_overlap: float
    version: int
    calibrated: bool
    model_id: str


@dataclass(frozen=True)
class GateSignals:
    """What the gate is allowed to look at. Nothing about the *content* of a candidate."""

    max_dense: float
    lexical: float
    token_coverage: float
    overlap: float
    candidate_count: int

    def clamped(self) -> GateSignals:
        """Keep every score inside ``[0, 1]``.

        Cosine similarity of a unit vector comes back as ``1.0000001`` now and then, and a
        value above 1.0 in a field called ``max_dense`` is a lie the API would publish.
        """

        def unit(value: float) -> float:
            return min(1.0, max(0.0, float(value)))

        return GateSignals(
            max_dense=unit(self.max_dense),
            lexical=unit(self.lexical),
            token_coverage=unit(self.token_coverage),
            overlap=unit(self.overlap),
            candidate_count=self.candidate_count,
        )


@dataclass(frozen=True)
class GateVerdict:
    """The gate's answer, with every signal kept for auditing (``debug=true`` only)."""

    passed: bool
    reason: str
    signals: GateSignals
    thresholds: Thresholds

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "max_dense": round(self.signals.max_dense, 4),
            "lexical": round(self.signals.lexical, 4),
            "token_coverage": round(self.signals.token_coverage, 4),
            "overlap": round(self.signals.overlap, 4),
            "candidate_count": self.signals.candidate_count,
            "thresholds_version": self.thresholds.version,
            "calibrated": self.thresholds.calibrated,
            "model_id": self.thresholds.model_id,
        }


def evaluate(signals: GateSignals, thresholds: Thresholds) -> GateVerdict:
    """Apply the rule. Returns a verdict, never raises: refusing is a normal outcome."""
    signals = signals.clamped()
    t = thresholds
    # No candidates is an unconditional refusal; a high score with nothing behind it is
    # an artifact of a default, not evidence.
    if signals.candidate_count == 0:
        return GateVerdict(False, "below_threshold", signals, t)
    if signals.max_dense >= t.min_dense and signals.token_coverage >= t.min_coverage:
        return GateVerdict(True, "dense+coverage", signals, t)
    if signals.max_dense >= t.min_dense_alone:
        return GateVerdict(True, "dense_only", signals, t)
    if signals.token_coverage >= t.min_coverage_high:
        return GateVerdict(True, "exact_terms", signals, t)
    return GateVerdict(False, "below_threshold", signals, t)


def defaults(model_id: str) -> Thresholds:
    """Placeholders derived from a live measurement, and honestly flagged (D47).

    Measured on the committed corpus with real BGE-M3: ``max_dense`` was 0.501-0.631 for
    answerable questions and 0.359-0.451 for unanswerable ones. The earlier guess of 0.62
    sat *inside* the answerable range, which made the dense branch of the rule unable to be
    the deciding signal at all.

    Both bars clear the measured irrelevant band. ``min_dense_alone`` is the higher of the
    two because that branch has nothing corroborating it. 18 questions is nowhere near
    enough to call this calibrated: P9 grid-searches it, and ``calibrated`` stays false.
    """
    return Thresholds(
        min_dense=0.47,
        min_dense_alone=0.50,
        min_coverage=0.25,
        min_coverage_high=0.50,
        min_sentence_overlap=0.15,
        version=1,
        calibrated=False,
        model_id=model_id,
    )


def load_thresholds(path: str | Path, *, model_id: str) -> Thresholds:
    """Read thresholds for ``model_id``.

    Missing, unreadable, or malformed files give the uncalibrated defaults with a warning;
    a file belonging to a *different* model is an error, because that is a real measured
    set of numbers being offered to the wrong model.
    """
    path = Path(path)
    if not path.exists():
        logger.info("no thresholds at %s; using uncalibrated defaults", path)
        return defaults(model_id)
    try:
        payload: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        found = str(payload["model_id"])
        if found != model_id:
            # Only a *calibrated* file is dangerous to reuse: it holds real measured numbers
            # that belong to another model. An uncalibrated file is a set of placeholders, and
            # placeholders for another model are still just placeholders -- which is what makes
            # `fake` mode startable at all, since no fake-model thresholds will ever exist.
            # The guarantee D42 makes is "another model's *measurements* never answer here",
            # and that still holds: the answer is a flagged default, not a real number (D51).
            if payload.get("calibrated"):
                raise ConfigError(
                    f"thresholds in {path} were calibrated for {found!r}, but the embedder "
                    f"is {model_id!r}. Refusing to reuse another model's measurements: "
                    "re-run `make calibrate`."
                )
            logger.warning(
                "thresholds at %s are for %r and the embedder is %r; using uncalibrated defaults",
                path,
                found,
                model_id,
            )
            return defaults(model_id)
        thresholds = Thresholds(
            version=int(payload.get("version", 1)),
            calibrated=bool(payload.get("calibrated", False)),
            model_id=found,
            **{name: float(payload[name]) for name in THRESHOLD_FIELDS},
        )
    except ConfigError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("thresholds at %s are unusable (%s); using defaults", path, exc)
        return defaults(model_id)
    _validate(thresholds, path)
    return thresholds


def _validate(thresholds: Thresholds, path: Path) -> None:
    for name in THRESHOLD_FIELDS:
        value = getattr(thresholds, name)
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{name} in {path} must be within [0, 1], got {value}")
    if thresholds.min_dense_alone < thresholds.min_dense:
        raise ConfigError(
            f"min_dense_alone ({thresholds.min_dense_alone}) must be at least min_dense "
            f"({thresholds.min_dense}) in {path}: the uncorroborated dense branch has no "
            "coverage to support it, so an equal or lower bar would weaken the gate"
        )
    if thresholds.min_coverage_high < thresholds.min_coverage:
        raise ConfigError(
            f"min_coverage_high ({thresholds.min_coverage_high}) must be at least "
            f"min_coverage ({thresholds.min_coverage}) in {path}: the lexical branch is "
            "only stricter than the dense branch while its coverage bar is higher"
        )
