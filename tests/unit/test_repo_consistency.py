"""The repository must be consistent with its own documentation (D31).

``plan.md`` §2.3 listed ``fa/justforfun_book_a4.pdf`` as committed while ``.gitignore``
excluded it, so four phases of ``make check`` passed against a working tree that had the
file while every clean clone ran two tests into ``FileNotFoundError``. A test that reads
an untracked fixture is not testing anything for the next person.

The check is deliberately narrow: it asks whether any fixture the suite depends on is
ignored or untracked by git. It does not try to parse the suite for path references,
which would be brittle; a fixture that exists locally but is not committed is the
failure mode worth catching.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "tests" / "fixtures" / "docs"

# Read unconditionally by the suite, so each one must be in the repository.
REQUIRED = (
    "en/storyen.md",
    "en/clean-code-excerpt.pdf",
    "fa/storyfa.md",
    "fa/ai-engineer.pdf",
    "fa/justforfun_book_a4.pdf",
)

# plan.md §2.3: parsed once by scripts/build_pdf_fixtures.py, which commits a 12-page
# excerpt instead. The test that would use it skips when the file is absent, so leaving
# it out is a deliberate trade of 10 MB for a redundant check on the excerpt.
DELIBERATELY_UNCOMMITTED = ("en/Clean Code Fundamentals-Martin Hock-Leanpub-EBooksWorld.ir.pdf",)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")


def _git(*args: str) -> list[str]:
    output = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in output.split("\n") if line]


@needs_git
def test_no_required_fixture_is_ignored() -> None:
    ignored = set(
        _git("ls-files", "--others", "--ignored", "--exclude-standard", "--", "tests/fixtures")
    )
    offenders = sorted(
        f"tests/fixtures/docs/{name}"
        for name in REQUIRED
        if f"tests/fixtures/docs/{name}" in ignored
    )
    assert not offenders, (
        "ignored fixtures cannot be read on a clean clone, so the tests that use them "
        f"would fail for everyone but this machine: {offenders}"
    )


@needs_git
def test_no_required_fixture_is_untracked() -> None:
    untracked = set(_git("ls-files", "--others", "--exclude-standard", "--", "tests/fixtures"))
    offenders = sorted(
        f"tests/fixtures/docs/{name}"
        for name in REQUIRED
        if f"tests/fixtures/docs/{name}" in untracked
    )
    assert not offenders, f"untracked fixtures are missing for everyone else: {offenders}"


@needs_git
def test_the_only_ignored_fixture_is_the_one_the_plan_excludes() -> None:
    """A new ignored fixture is a decision someone has not made yet."""
    ignored = {
        line
        for line in _git(
            "ls-files", "--others", "--ignored", "--exclude-standard", "--", "tests/fixtures"
        )
    }
    allowed = {f"tests/fixtures/docs/{name}" for name in DELIBERATELY_UNCOMMITTED}
    assert ignored <= allowed, (
        f"unexpected ignored fixtures, add them to DELIBERATELY_UNCOMMITTED with a "
        f"reason or commit them: {sorted(ignored - allowed)}"
    )


def test_every_required_fixture_is_present() -> None:
    """Named explicitly, so a renamed or deleted fixture fails here rather than six
    files away. This is the check that would have caught the D31 regression directly."""
    missing = [name for name in REQUIRED if not (DOCS / name).exists()]
    assert not missing, f"missing fixtures: {missing}"
