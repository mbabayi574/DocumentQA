"""The dependency direction `api -> services -> domain/ports <- adapters` (plan.md §4, §7).

P0 checked only that `domain/` imports no framework or I/O library. P11 extends it past
`domain/`, because three real violations survived that check and were only found by reading
the import graph: `storage/` imported `embeddings/` for a sha256 one-liner, `ingestion/`
imported a concrete adapter for an integer, and `answering/` and `retrieval/` imported each
other in a cycle.

Each rule below exists because it was violated once. A rule nobody has broken yet is a
preference; these are receipts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "qasystem"
DOMAIN_DIR = SRC / "domain"
FORBIDDEN = {"fastapi", "starlette", "chromadb", "httpx", "sqlite3", "requests", "numpy"}

#: Package -> packages it may not import. Each entry names the violation that earned it.
LAYER_RULES: dict[str, frozenset[str]] = {
    # Pure stdlib: the bottom of the graph.
    "domain": frozenset(
        {
            "api",
            "answering",
            "chunking",
            "embeddings",
            "ingestion",
            "parsing",
            "retrieval",
            "storage",
            "text",
        }
    ),
    # text/ may import domain (it takes the pure `Language` enum and nothing else), never above.
    "text": frozenset(
        {
            "api",
            "answering",
            "chunking",
            "embeddings",
            "ingestion",
            "parsing",
            "retrieval",
            "storage",
        }
    ),
    # A repository is a leaf adapter. It imported embeddings.caching for input_hash (D76).
    "storage": frozenset({"api", "answering", "ingestion", "retrieval", "embeddings"}),
    # Services take ports. reconcile imported the concrete ChromaStore for PAGE_SIZE (D76).
    "ingestion": frozenset({"api", "retrieval", "answering"}),
    # The answerer takes candidates, not the retrieval service that produced them (D76).
    "answering": frozenset({"api", "ingestion", "retrieval", "storage"}),
}


def _top_level_imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _internal_packages(path: Path) -> set[str]:
    """The first segment of every ``qasystem.*`` import this module makes."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    packages: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "qasystem" and len(parts) > 1:
                packages.add(parts[1])
    return packages


def _layer_files(package: str) -> list[Path]:
    return sorted(p for p in (SRC / package).rglob("*.py"))


@pytest.mark.parametrize("path", sorted(DOMAIN_DIR.glob("*.py")), ids=lambda p: p.name)
def test_domain_imports_no_framework_or_io_library(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not (_top_level_imports(tree) & FORBIDDEN)


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [(name, sorted(rules)) for name, rules in sorted(LAYER_RULES.items())],
)
def test_no_layer_imports_a_layer_it_must_not_know_about(
    package: str, forbidden: list[str]
) -> None:
    offenders: dict[str, list[str]] = {}
    for path in _layer_files(package):
        bad = sorted(_internal_packages(path) & set(forbidden))
        if bad:
            offenders[str(path.relative_to(SRC))] = bad
    assert not offenders, (
        f"{package}/ must not import {forbidden}: {offenders}. Every one of these was a real "
        "violation; see the comment on each rule."
    )


def _module_imports(path: Path) -> set[str]:
    """Every ``qasystem.*`` module this file imports, as dotted paths."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            if name.split(".")[0] != "qasystem":
                continue
            parts = name.split(".")
            # `from qasystem.a.b import C` names a.b; a bare `import qasystem.a.b` may name a
            # package, whose module key is the package itself.
            found.add(".".join(parts[1:]))
    return found


def test_the_internal_import_graph_has_no_cycles() -> None:
    """`answering` and `retrieval` imported each other, hidden from mypy by TYPE_CHECKING.

    An import cycle is not a style complaint: it means neither module can be read as the layer
    above the other, and the direction `api -> services -> domain` stops meaning anything.

    D76 records why this needed its own fix. The first version of this test built its edges
    from package names while keying the graph by module path, so every edge pointed at a node
    that did not exist, the walk terminated immediately, and the test passed on a graph that
    did contain the cycle it was written to catch. It is the D25 failure mode wearing a test's
    clothes: green, and proving nothing.
    """
    edges: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        name = ".".join(path.relative_to(SRC).with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        edges[name] = _module_imports(path)

    assert any(edges.values()), "the graph is empty, so this test is asserting nothing"

    cycles: set[str] = set()
    for start in sorted(edges):
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack:
            node, path_so_far = stack.pop()
            for child in sorted(edges[node]):
                if child == start or child in path_so_far:
                    cycles.add(" -> ".join([*path_so_far, child]))
                elif child in edges:
                    stack.append((child, [*path_so_far, child]))
    assert not cycles, f"import cycles among qasystem modules: {sorted(cycles)}"


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
