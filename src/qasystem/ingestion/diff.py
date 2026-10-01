"""Chunk diff for the ingest log (plan.md P6 rule 3).

Pure, so every multiset case is cheap to pin and the "a reorder costs nothing" claim is
directly assertable. It decides only *counts*; what actually gets embedded is the
embedding cache's decision, keyed on the exact embedded string, which is why a diff on
``chunk_hash`` is safe to be approximate about ordering and exact about multiplicity.

Matching is by multiset subtraction, not by index. Two identical chunks are two chunks,
so one disappearing is a real removal -- plan test 12 depends on that.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffPlan:
    """How a new chunk list differs from the currently published one.

    ``added``/``reused`` hold indices into the **new** list, which is what staging needs;
    ``removed`` holds chunk hashes from the old list, which is what cleanup needs.
    """

    added: tuple[int, ...]
    reused: tuple[int, ...]
    removed: tuple[str, ...]

    @property
    def chunks_added(self) -> int:
        return len(self.added)

    @property
    def chunks_reused(self) -> int:
        return len(self.reused)

    @property
    def chunks_removed(self) -> int:
        return len(self.removed)

    @property
    def total_new(self) -> int:
        return len(self.added) + len(self.reused)

    @property
    def is_empty_change(self) -> bool:
        return not self.added and not self.removed


def diff_hashes(prior: Sequence[str], new: Sequence[str]) -> DiffPlan:
    """Match ``new`` against ``prior`` by multiset, so duplicates keep their count."""
    unmatched = Counter(prior)
    added: list[int] = []
    reused: list[int] = []
    for index, chunk_hash in enumerate(new):
        if unmatched[chunk_hash] > 0:
            unmatched[chunk_hash] -= 1
            reused.append(index)
        else:
            added.append(index)
    # What remains in `unmatched` is exactly what prior held and new did not match, so
    # one surviving copy of a duplicated chunk leaves the others as genuine removals.
    removed: list[str] = []
    for chunk_hash, count in unmatched.items():
        if count > 0:
            removed.extend([chunk_hash] * count)
    return DiffPlan(added=tuple(added), reused=tuple(reused), removed=tuple(removed))
