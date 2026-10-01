"""Maintenance CLI: ``python -m qasystem.cli`` (plan.md P8, P9).

Each command is a thin wrapper over a function that already exists and is already tested
elsewhere. The CLI translates arguments and exit codes; it does not contain logic of its own,
because logic that only runs from a terminal is logic nothing else can check.

Exit codes are the point of a maintenance command: ``0`` when the thing it checked is healthy,
``1`` when it is not, so a cron job or a deploy step can branch on the result without parsing
prose. ``reconcile`` and ``check-storage`` are read-mostly and say what they found; ``rebuild``
changes the index and says how much it restored.

``eval`` and ``calibrate`` delegate to ``tests/eval/runner.py`` — the layout plan.md §7
specifies, and the only one of the seven commands whose implementation is not in ``src/``.
The import is deferred and the failure is reported, not raised, so a checkout without the
eval dataset still gets a diagnostic instead of a traceback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qasystem.config import Settings, load_settings
from qasystem.ingestion.reconcile import rebuild
from qasystem.logging_setup import configure_logging

EXIT_OK = 0
EXIT_FAILED = 1

#: The eval harness's location. Derived from this file rather than from the data directory, so
#: a maintenance command works regardless of where `data/` happens to be.
EVAL_DIR = Path(__file__).resolve().parents[2] / "tests" / "eval"
DATASET = EVAL_DIR / "dataset.jsonl"


def _report(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def cmd_reconcile(settings: Settings, args: argparse.Namespace) -> int:
    """Report and repair index drift: SQLite decides, Chroma is made to match."""
    from qasystem.api.deps import build_services

    async def run() -> int:
        services = await build_services(settings)
        try:
            plan = await services.ingestion.reconcile()
            _report(
                {
                    "in_sync": plan.in_sync,
                    "expected": plan.total_expected,
                    "missing": list(plan.missing),
                    "orphaned": list(plan.orphaned),
                }
            )
            return EXIT_OK if plan.in_sync else EXIT_FAILED
        finally:
            await services.aclose()

    del args
    return asyncio.run(run())


def cmd_rebuild(settings: Settings, args: argparse.Namespace) -> int:
    """Re-upsert every vector from the embedding cache. Never calls the embedding API."""
    from qasystem.api.deps import build_services

    async def run() -> int:
        services = await build_services(settings)
        try:
            restored = rebuild(services.store, services.vectors, services.model_id)
            _report({"restored": restored, "model_id": services.model_id})
            return EXIT_OK
        finally:
            await services.aclose()

    del args
    return asyncio.run(run())


def cmd_check_storage(settings: Settings, args: argparse.Namespace) -> int:
    """Round-trip both stores: upsert a probe vector, query it back, delete it, count.

    A read-only check cannot tell "the index works" from "the index opens": the failure this
    catches is a store that opens and then cannot round-trip, which is exactly what a wrong
    dimension or a stale collection name produces.
    """
    from qasystem.api.deps import build_services
    from qasystem.domain.models import VectorItem

    async def run() -> int:
        services = await build_services(settings)
        probe_id = "__check_storage__"
        try:
            before = services.vectors.count()
            vector = [0.0] * services.dimension
            vector[0] = 1.0
            services.vectors.upsert([VectorItem(probe_id, vector, {"probe": "1"})])
            found = services.vectors.query(vector, 1)
            services.vectors.delete_ids([probe_id])
            after = services.vectors.count()
            report = {
                "sqlite": "ok",
                "schema_version": services.store.schema_version(),
                "fts5": services.store.fts5_available(),
                "chroma": "ok",
                "model_id": services.model_id,
                "dimension": services.dimension,
                "count_before": before,
                "count_after": after,
                "round_trip_ok": found and found[0].id == probe_id,
            }
            _report(report)
            healthy = report["round_trip_ok"] and before == after
            return EXIT_OK if healthy else EXIT_FAILED
        finally:
            await services.aclose()

    del args
    return asyncio.run(run())


def _metrics(metrics: Any) -> dict[str, Any]:
    return {
        "answerable": metrics.n_answerable,
        "unanswerable": metrics.n_unanswerable,
        "r1": round(metrics.r1, 4),
        "r3": round(metrics.r3, 4),
        "r5": round(metrics.r5, 4),
        "mrr5": round(metrics.mrr5, 4),
        "answered_on_answerable": round(metrics.answered_rate, 4),
        "false_answers": metrics.false_answers,
        "false_answer_rate": round(metrics.false_answer_rate, 4),
        "gold_quoted": round(metrics.evidence, 4),
    }


def _run_eval(settings: Settings, *, calibrate: bool) -> int:
    """Shared body of ``eval`` and ``calibrate``; the only difference is the write.

    The harness lives in ``tests/`` because plan.md section 7 puts it there and because it is
    test infrastructure rather than a shipped code path. That makes a missing or broken
    import a checkout problem rather than a package problem, so it is reported as one.
    """
    if not DATASET.exists():
        _report(
            {
                "calibrated": False,
                "reason": "no eval dataset",
                "expected": str(DATASET),
                "next": "author tests/eval/dataset.jsonl (plan.md section 9.1), then re-run",
            }
        )
        return EXIT_FAILED
    if not settings.embedding_api_key and not settings.is_fake_provider:
        _report(
            {
                "error": "no provider configured",
                "message": (
                    "the eval measures the real model; set EMBEDDING_API_KEY, or run with "
                    "EMBEDDING_PROVIDER=fake APP_ENV=demo to check the harness only"
                ),
            }
        )
        return EXIT_FAILED
    try:
        if str(EVAL_DIR) not in sys.path:
            sys.path.insert(0, str(EVAL_DIR))
        import runner  # type: ignore[import-not-found]
    except Exception as exc:  # a broken checkout, reported the way a broken checkout reads
        _report({"error": type(exc).__name__, "message": str(exc), "expected": str(EVAL_DIR)})
        return EXIT_FAILED

    report = EVAL_DIR / "report.md"
    outcome = asyncio.run(runner.run(settings, calibrate=calibrate, report=report))
    _report(
        {
            "report": report.name,
            "calibrated": outcome.calibration is not None,
            "thresholds_written": calibrate and outcome.calibration is not None,
            "stale_leaks": outcome.stale_leaks,
            "citation_segments_verified": outcome.citation_checks,
            "dev": _metrics(outcome.calibration.dev) if outcome.calibration else None,
            "held_out": _metrics(outcome.calibration.test) if outcome.calibration else None,
            "reason": outcome.calibration_error or None,
        }
    )
    return EXIT_OK if outcome.calibration is not None else EXIT_FAILED


def cmd_calibrate(settings: Settings, args: argparse.Namespace) -> int:
    """Grid-search the gate on the eval dataset's dev split, then write it (P9 section 9.3)."""
    del args
    return _run_eval(settings, calibrate=True)


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    """Score the corpus against the eval dataset, without writing thresholds (P9 section 9.2)."""
    del args
    return _run_eval(settings, calibrate=False)


COMMANDS = {
    "reconcile": cmd_reconcile,
    "rebuild": cmd_rebuild,
    "check-storage": cmd_check_storage,
    "calibrate": cmd_calibrate,
    "eval": cmd_eval,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qasystem.cli", description="Maintenance commands."
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="what to do")
    parser.add_argument(
        "--json", action="store_true", help="accepted for scripting; output is always JSON"
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except Exception as exc:
        _report({"error": type(exc).__name__, "message": str(exc)})
        return EXIT_FAILED
    configure_logging(settings)
    try:
        return COMMANDS[args.command](settings, args)
    except Exception as exc:
        _report({"error": type(exc).__name__, "message": str(exc)})
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
