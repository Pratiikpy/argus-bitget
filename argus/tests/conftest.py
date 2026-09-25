"""Suite-wide switches.

``ARGUS_BLOCK_NETWORK=1`` refuses every outbound connection that is not to this machine, so a test
that silently depends on a live venue, feed or filing fails loudly instead of passing on whatever
the network returned that day. It is how the ``network`` marker below was assigned: the suite was
run with the switch on, and every test that failed for want of a connection was marked.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}

# The web console instruments its engines on the first question (`lui/server._traced`). In the
# suite that wrap would outlive the HTTP test that installed it and sit under every later
# monkeypatch, so it is off here; the trace is tested on its own in tests/test_trace.py.
os.environ.setdefault("ARGUS_TRACE", "0")


class NetworkBlocked(ConnectionError):
    """An outbound connection was attempted while ARGUS_BLOCK_NETWORK=1."""


@pytest.fixture(autouse=True, scope="session")
def _block_network_when_asked() -> Iterator[None]:
    if os.environ.get("ARGUS_BLOCK_NETWORK") != "1":
        yield
        return
    original = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOCAL_HOSTS:
            raise NetworkBlocked(f"outbound connection to {host} refused (ARGUS_BLOCK_NETWORK=1)")
        return original(self, address)

    socket.socket.connect = guarded  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _no_live_prediction_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polymarket is never read by the offline suite: research answers call it for fundamentals,
    news, sentiment and directional questions, and a live market list would make their line
    counts depend on the day. `tests/test_prediction.py` passes its own search function."""
    from argus.market import prediction

    monkeypatch.setattr(prediction, "_search", lambda term, **_: [])
