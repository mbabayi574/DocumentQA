"""The maintenance CLI.

Run as a subprocess, not by calling ``main()``: the commands are about what a *terminal*
sees -- exit codes and printed JSON -- and calling them in-process would test the functions
while skipping exactly the part that matters.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DOC = "# Handbook\n\n## Install\n\nRun the installer on Linux. It checks the kernel first.\n"


def run(
    tmp_path: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a subprocess with a tmp data dir and the fake provider."""
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "APP_ENV": "test",
        "EMBEDDING_PROVIDER": "fake",
        "DATA_DIR": str(tmp_path),
        "SQLITE_PATH": str(tmp_path / "qasystem.db"),
        "CHROMA_PATH": str(tmp_path / "chroma"),
        "THRESHOLDS_PATH": str(tmp_path / "thresholds.json"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    (tmp_path / "thresholds.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": "fake-embedder",
                "calibrated": False,
                "min_dense": 0.05,
                "min_coverage": 0.70,
                "min_coverage_high": 0.90,
                "min_sentence_overlap": 0.15,
            }
        ),
        encoding="utf-8",
    )
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "qasystem.cli", *args],
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )


def payload(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return json.loads(result.stdout)  # type: ignore[no-any-return]


# ---------------------------------------------------------------- help


def test_help_lists_every_documented_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "qasystem.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    for command in ("reconcile", "rebuild", "check-storage", "calibrate", "eval"):
        assert command in result.stdout


def test_an_unknown_command_is_a_usage_error(tmp_path: Path) -> None:
    result = run(tmp_path, "teleport")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


# ---------------------------------------------------------------- check-storage


def test_check_storage_round_trips_both_stores(tmp_path: Path) -> None:
    result = run(tmp_path, "check-storage")

    assert result.returncode == 0, result.stderr[-500:]
    body = payload(result)
    assert body["sqlite"] == "ok"
    assert body["fts5"] is True
    assert body["chroma"] == "ok"
    assert body["round_trip_ok"] is True
    assert body["count_before"] == body["count_after"], "the probe vector was left behind"
    assert body["model_id"] == "fake-embedder"
    assert body["dimension"] == 32


def test_check_storage_leaves_no_probe_vector_behind(tmp_path: Path) -> None:
    """A maintenance probe that leaves litter is worse than no probe."""
    run(tmp_path, "check-storage")
    again = payload(run(tmp_path, "check-storage"))
    assert again["count_before"] == 0
    assert again["count_after"] == 0


# ---------------------------------------------------------------- reconcile


def test_reconcile_reports_in_sync_on_an_empty_store(tmp_path: Path) -> None:
    result = run(tmp_path, "reconcile")

    assert result.returncode == 0, result.stderr[-500:]
    assert payload(result) == {
        "in_sync": True,
        "expected": 0,
        "missing": [],
        "orphaned": [],
    }


INGEST = """
import sys
from pathlib import Path
from qasystem.api.deps import build_services
from qasystem.config import load_settings
from qasystem.domain.models import VectorItem
import anyio

data, thresholds, document = sys.argv[1], sys.argv[2], sys.argv[3]
settings = load_settings(
    env_file=None, app_env="test", embedding_provider="fake",
    thresholds_path=Path(thresholds),
)

async def go():
    services = await build_services(settings, data_dir=data)
    await services.ingestion.ingest(document.encode(), "handbook.md")
    # A vector SQLite has no row for: exactly what reconcile exists to remove.
    services.vectors.upsert([
        VectorItem("ghost:v1:0", [1.0] * services.dimension, {"doc_id": "ghost"})
    ])
    await services.aclose()

anyio.run(go)
"""


def _populate_and_ghost(tmp_path: Path) -> None:
    import os

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            INGEST,
            str(tmp_path),
            str(tmp_path / "thresholds.json"),
            DOC,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_reconcile_removes_an_orphan_and_then_reports_in_sync(tmp_path: Path) -> None:
    """The exit code is what a cron job branches on, so drift must be non-zero."""
    _populate_and_ghost(tmp_path)

    drifted = run(tmp_path, "reconcile")
    assert drifted.returncode == 1, drifted.stdout
    body = payload(drifted)
    assert body["in_sync"] is False
    assert body["orphaned"] == ["ghost:v1:0"]
    # The real document's vectors are still expected, so reconcile must not touch them.
    assert body["expected"] > 0

    healed = run(tmp_path, "reconcile")
    assert healed.returncode == 0, healed.stdout
    assert payload(healed)["in_sync"] is True
    assert payload(healed)["orphaned"] == []


# ---------------------------------------------------------------- rebuild


def test_rebuild_on_an_empty_store_restores_nothing(tmp_path: Path) -> None:
    result = run(tmp_path, "rebuild")

    assert result.returncode == 0, result.stderr[-500:]
    body = payload(result)
    assert body["restored"] == 0
    assert body["model_id"] == "fake-embedder"


# ---------------------------------------------------------------- calibrate / eval


# The dataset exists now, so these two cover the branches the CLI itself owns: a missing
# dataset, and a provider that was never configured. Both are reported rather than guessed at,
# because both would otherwise write numbers with no evidence behind them. What a real run
# produces is covered in tests/eval/, where the harness is under test rather than the parser.
def _main(monkeypatch: pytest.MonkeyPatch, dataset: Path, argv: list[str], **env: str) -> int:
    from qasystem import cli

    monkeypatch.setattr(cli, "DATASET", dataset)
    monkeypatch.setattr(cli, "EVAL_DIR", dataset.parent)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["qasystem.cli", *argv])
    return cli.main(argv)


@pytest.mark.parametrize("command", ["calibrate", "eval"])
def test_a_missing_eval_dataset_is_reported_not_guessed_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """`calibrated: false` exists to prevent numbers with no evidence behind them, and the
    way to honour that is to say what is missing and exit non-zero."""
    assert _main(monkeypatch, tmp_path / "absent.jsonl", [command]) == 1


def test_the_eval_commands_refuse_to_run_without_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eval measures the real model. Running it against the fake embedder would produce
    a confident report about a system nobody will ship."""
    assert (
        _main(
            monkeypatch,
            Path(__file__).resolve().parents[1] / "eval" / "dataset.jsonl",
            ["eval"],
            APP_ENV="prod",
            EMBEDDING_PROVIDER="remote",
            EMBEDDING_API_KEY="",
            EMBEDDING_MODEL="",
        )
        == 1
    )


# ---------------------------------------------------------------- failure reporting


def test_a_failure_is_reported_as_json_not_a_traceback(tmp_path: Path) -> None:
    """A maintenance command must not dump a traceback into an operator's terminal."""
    # SQLite cannot open a directory, and it fails in `sqlite3.connect` rather than in our
    # own code -- so this exercises the CLI's catch-all, which is the thing under test.
    result = run(tmp_path, "check-storage", env={"SQLITE_PATH": str(tmp_path)})

    assert result.returncode == 1
    body = payload(result)
    assert "error" in body and body["message"]
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_a_bad_configuration_is_reported_without_a_token(tmp_path: Path) -> None:
    secret = "sk-live-must-not-appear-in-cli-output-0123456789"
    result = run(
        tmp_path,
        "reconcile",
        env={
            "APP_ENV": "prod",
            "EMBEDDING_PROVIDER": "fake",  # forbidden outside test/demo: a real ConfigError
            "EMBEDDING_API_KEY": secret,
        },
    )

    assert result.returncode == 1
    body = payload(result)
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert body["error"] == "ConfigError"
