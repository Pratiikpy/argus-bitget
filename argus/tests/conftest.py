"""Suite-wide switches.

``ARGUS_BLOCK_NETWORK=1`` refuses every outbound connection that is not to this machine, so a test
that silently depends on a live venue, feed or filing fails loudly instead of passing on whatever
the network returned that day. It is how the ``network`` marker was assigned: the suite was run
with the switch on, and every test that reached for a connection was marked.

Refusing the connection was not enough on its own. Many live tests catch the error and skip ("venue
unreachable"), so with the network blocked they skipped quietly after minutes of retry back-off and
the offline job looked clean while never running them (audit, 2026-09-26). With the switch on, a
test that attempts any outbound connection and is not marked ``network`` is now failed, whatever it
did with the error.
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import HealthCheck, settings

# Property tests assert what the code does, not how fast this machine is. Hypothesis's timing
# checks failed `test_pause` on 2026-09-27 because drawing one enum value took two seconds while
# three suites shared the CPU; a slow CI runner would fail the same way on an unchanged tree.
settings.register_profile(
    "argus", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("argus")

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}

# The web console instruments its engines on the first question (`lui/server._traced`). In the
# suite that wrap would outlive the HTTP test that installed it and sit under every later
# monkeypatch, so it is off here; the trace is tested on its own in tests/test_trace.py.
os.environ.setdefault("ARGUS_TRACE", "0")


class NetworkBlocked(ConnectionError):
    """An outbound connection was attempted while ARGUS_BLOCK_NETWORK=1."""


_RUNNING: list[pytest.Item | None] = [None]
"""The test whose setup, call or teardown is running, so a refused connection can be charged to
it — including one made by a module-scoped fixture during that test's setup."""

_REACHED: dict[str, set[str]] = {}
"""Hosts each test tried to reach, by node id."""


@pytest.fixture(autouse=True, scope="session")
def _block_network_when_asked() -> Iterator[None]:
    if os.environ.get("ARGUS_BLOCK_NETWORK") != "1":
        yield
        return
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def refuse(address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOCAL_HOSTS:
            item = _RUNNING[0]
            if item is not None:
                _REACHED.setdefault(item.nodeid, set()).add(host)
            raise NetworkBlocked(f"outbound connection to {host} refused (ARGUS_BLOCK_NETWORK=1)")

    def guarded(self: socket.socket, address: Any) -> Any:
        refuse(address)
        return connect(self, address)

    def guarded_ex(self: socket.socket, address: Any) -> Any:
        refuse(address)
        return connect_ex(self, address)

    socket.socket.connect = guarded  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]


@pytest.fixture(autouse=True, scope="session")
def _frozen_contract_list_when_blocked(_block_network_when_asked: None) -> None:
    """With the network blocked, the contract list is the dated snapshot from the start.

    `market/universe.contracts()` fetches Bitget's live list on first use and falls back to the
    snapshot, retrying five minutes later. Symbol lookups sit under a large part of the console, so
    whichever test happened to run first, and one every five minutes after, reached the network
    and was charged with it. The snapshot is what an offline run would get anyway; a test of the
    live fetch itself clears the cache and is marked ``network``."""
    if os.environ.get("ARGUS_BLOCK_NETWORK") != "1":
        return
    from argus.market import universe

    found, frozen_on = universe.contracts_from_snapshot()
    # A stamp in the future keeps the cache fresh for the whole session.
    universe._CACHE = (time.monotonic() + 10 * 365 * 86400, found, f"frozen {frozen_on}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    _RUNNING[0] = item
    yield
    _RUNNING[0] = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[Any]:
    outcome = yield
    if call.when != "teardown":
        return
    hosts = _REACHED.pop(item.nodeid, None)
    if not hosts or item.get_closest_marker("network") is not None:
        return
    report = outcome.get_result()
    report.outcome = "failed"
    report.longrepr = (f"reached the network ({', '.join(sorted(hosts))}) with "
                       "ARGUS_BLOCK_NETWORK=1 but is not marked @pytest.mark.network: mark it, "
                       "or stub the call so it runs offline")


@pytest.fixture(autouse=True)
def _no_live_prediction_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polymarket is never read by the offline suite: research answers call it for fundamentals,
    news, sentiment and directional questions, and a live market list would make their line
    counts depend on the day. `tests/test_prediction.py` passes its own search function."""
    from argus.market import prediction

    monkeypatch.setattr(prediction, "_search", lambda term, **_: [])
