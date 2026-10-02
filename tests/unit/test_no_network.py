"""The no-network guard in tests/conftest.py must actually hold."""

from __future__ import annotations

import os
import socket

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE") == "1",
    reason="RUN_LIVE=1 deliberately turns the guard off, so there is nothing left to assert",
)
def test_network_is_blocked() -> None:
    with pytest.raises(AssertionError, match="must not open sockets"):
        socket.create_connection(("example.com", 80), timeout=1)
