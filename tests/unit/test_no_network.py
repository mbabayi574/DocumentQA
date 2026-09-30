"""The no-network guard in tests/conftest.py must actually hold."""

from __future__ import annotations

import socket

import pytest


def test_network_is_blocked() -> None:
    with pytest.raises(AssertionError, match="must not open sockets"):
        socket.create_connection(("example.com", 80), timeout=1)
