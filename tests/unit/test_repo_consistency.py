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

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "tests" / "fixtures" / "docs"

# The prose a reviewer reads before any code. A broken link here is the first thing
# they hit, and it is invisible to every test that only looks at behaviour.
DOC_FILES = (ROOT / "README.md", ROOT / "plan.md", ROOT / "docs" / "DECISIONS.md")

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


@needs_git
def test_no_runtime_artifact_is_tracked() -> None:
    """A tracked lock file is a runtime artifact that means nothing in a clean clone.

    ``.qasystem.lock`` is L2's single-instance lock: it is created by running the app and
    means nothing until then. Tracked, it invites a reader to reason about ownership of
    ``data/`` from a 0-byte file that every clone inherits.
    """
    tracked = set(_git("ls-files"))
    offenders = sorted(name for name in tracked if name.endswith(".lock") and name != "uv.lock")
    assert not offenders, (
        f"runtime lock files are tracked: {offenders}. `uv.lock` is the dependency lockfile "
        "and is the only one that belongs in source control."
    )


@needs_git
def test_every_relative_markdown_link_resolves() -> None:
    """A link to a file that does not exist is a lie a reviewer pays for.

    D31's failure class, second instance: `docs/eval_report.md` was linked from
    `README.md` three times, named in `plan.md` §7/§9/§11, and `docs/DECISIONS.md`
    contradicted itself about where the report lives -- while `plan.md` §11 carried a
    *ticked* Definition-of-Done box for it. Nothing checked it, because the existing
    test only looks at fixtures.
    """
    broken: list[str] = []
    for source in DOC_FILES:
        for target in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = target.partition("#")[0]
            if path and not (source.parent / path).exists():
                broken.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not broken, f"relative markdown links that do not resolve: {broken}"


def test_every_required_fixture_is_present() -> None:
    """Named explicitly, so a renamed or deleted fixture fails here rather than six
    files away. This is the check that would have caught the D31 regression directly."""
    missing = [name for name in REQUIRED if not (DOCS / name).exists()]
    assert not missing, f"missing fixtures: {missing}"
