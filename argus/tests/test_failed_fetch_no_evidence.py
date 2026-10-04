"""Build-list 4.2: a failed fetch can never become evidence. With every network read failing, the
desk's evidence gather returns no item and names each source as unavailable — no cached, default
or placeholder item stands in for a source that did not answer."""

from __future__ import annotations

import urllib.request
from datetime import UTC, datetime
from typing import Any

import pytest

from argus.market import evidence


def test_every_source_down_gives_no_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*args: Any, **kwargs: Any) -> Any:
        raise OSError("network unreachable (test)")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    got = evidence.gather("NVDAUSDT", as_of=datetime(2026, 10, 4, 16, tzinfo=UTC),
                          last_price=186.4)
    assert got.evidence == []
    assert got.status, "a dark feed must be named to the desk, not left silent"
    assert all("unavailable" in s or ": 0 " in s or s.endswith(": 0 filings")
               for s in got.status), got.status
