"""P0: the domain layer must stay free of framework/storage/network imports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DOMAIN_DIR = Path(__file__).resolve().parents[2] / "src" / "qasystem" / "domain"
FORBIDDEN = {"fastapi", "starlette", "chromadb", "httpx", "sqlite3", "requests", "numpy"}


def _top_level_imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("path", sorted(DOMAIN_DIR.glob("*.py")), ids=lambda p: p.name)
def test_domain_imports_no_framework_or_io_library(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not (_top_level_imports(tree) & FORBIDDEN)


def test_domain_contracts_exist() -> None:
    modules = {
        "models.py": ("Section", "ParsedDocument", "Chunk", "VectorItem", "VectorHit"),
        "ports.py": ("Embedder", "VectorStore", "DocumentParser"),
    }
    for filename, names in modules.items():
        tree = ast.parse((DOMAIN_DIR / filename).read_text(encoding="utf-8"))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert set(names) <= defined, f"{filename} is missing {set(names) - defined}"
