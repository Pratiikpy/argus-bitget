"""ARGUS's book sweep against EGRESS's walk (build-list 5.4), on books built by hand."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from argus.eval import exitcost_parity as ep


def _walk_quote(symbol, size, bids, asks, side="sell"):  # stands in for EGRESS
    """EGRESS's method, restated for an offline test: a base quantity fixed at the mid."""
    mid = (bids[0][0] + asks[0][0]) / 2
    want, got, value = size / mid, 0.0, 0.0
    for price, qty in bids:
        take = min(qty, want - got)
        if take <= 0:
            break
        value += take * price
        got += take
    exhausted = got < want - 1e-12
    vwap = value / got
    return SimpleNamespace(exhausted=exhausted, slippage_bp=(mid - vwap) / mid * 10_000)


def test_the_two_walks_agree_and_agree_on_a_thin_book() -> None:
    deep = {"symbol": "NVDAUSDT", "taken_at": "2026-09-14T10:54:53+00:00",
            "bids": [[100.0, 500.0], [99.9, 5000.0]], "asks": [[100.1, 500.0], [100.2, 5000.0]]}
    thin = {**deep, "bids": [[100.0, 10.0]], "asks": [[100.1, 10.0]]}
    got = ep.compare([deep, thin], _walk_quote)
    small = got["10000"]
    assert small["compared"] == 1 and small["median_abs_gap_bps"] == pytest.approx(0, abs=0.05)
    assert small["too_thin"] == {"egress": 1, "argus": 1}
