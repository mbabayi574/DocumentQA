from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest

from qasystem.config import Settings


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """plan.md §0.6: no test may touch the network. Live calls need RUN_LIVE=1."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("tests must not open sockets (set RUN_LIVE=1 to opt out)")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.fixture(autouse=True)
def _isolate_settings_env() -> Iterator[None]:
    """No test may depend on the developer's shell or .env file."""
    saved = {name: os.environ.pop(name) for name in Settings.model_fields if name in os.environ}
    yield
    os.environ.update(saved)


@pytest.fixture
def settings_factory() -> Any:
    """Build settings from explicit values only (no .env, no shell)."""

    def factory(**overrides: Any) -> Settings:
        from qasystem.config import load_settings

        return load_settings(env_file=None, **overrides)

    return factory
