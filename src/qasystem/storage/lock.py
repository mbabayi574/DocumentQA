"""Single-owner data directory lock (plan.md §4.2 L2, P5).

``data/`` holds one SQLite file and one Chroma directory. Two processes writing both
would corrupt Chroma's internal files, so ownership is taken at startup with a
zero-timeout ``FileLock``: a second instance fails immediately with a clear message
rather than starting and racing.

``timeout=0`` is the whole mechanism — a waiting process would be the bug, not the fix.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from filelock import FileLock, Timeout

from qasystem.errors import StorageLockedError

LOCK_NAME = ".qasystem.lock"


class DataLock:
    """Exclusive ownership of a data directory for the life of the process."""

    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / LOCK_NAME
        self._lock: FileLock | None = None

    @property
    def is_held(self) -> bool:
        """True when this object currently owns the directory.

        Deliberately *not* asking ``FileLock.is_locked``: that counter is thread-local, so a
        readiness check running on a request thread saw "not locked" while the process
        plainly held it, and ``/ready`` returned 503 forever (D50). ``acquire()`` succeeding
        and ``release()`` clearing the attribute is the truth about *this process*, which is
        the only question L2 asks.
        """
        return self._lock is not None

    def acquire(self) -> None:
        """Take ownership, or raise ``StorageLockedError`` immediately (L2)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.path, timeout=0)
        try:
            lock.acquire()
        except Timeout:
            raise StorageLockedError(
                f"{self.path.parent} is already owned by another process; "
                "run a single worker (uvicorn --workers 1)"
            ) from None
        except OSError as exc:
            raise StorageLockedError(
                f"cannot take the data-directory lock at {self.path}: {type(exc).__name__}"
            ) from None
        self._lock = lock

    def release(self) -> None:
        """Release ownership. Harmless when nothing is held."""
        lock, self._lock = self._lock, None
        if lock is not None and lock.is_locked:
            lock.release()

    def __enter__(self) -> DataLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
