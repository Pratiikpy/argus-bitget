"""The rToken overnight hedge head-to-head against Ballast: the held-out protocol and the scorer.

The live comparison reads Ballast's frozen per-night series and runs with
`python -m argus.eval.copilot_hedge`. Pinned here: the ratio is fitted on the first 70% and never
refitted, the same-name arm is ARGUS's own function and equals Ballast's OLS by construction, and
the verdict states the tie and the index hedge that lost.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import pytest

from argus.eval.copilot_hedge import FIT_SHARE, compare, evaluate, hedged, verdict


def _series(n: int, slope: float, noise: float, phase: float) -> tuple[list[float], list[float]]:
    leg = [0.01 * math.sin(i * 0.7 + phase) for i in range(n)]
    spot = [slope * x + noise * math.cos(i * 1.3) for i, x in enumerate(leg)]
    return spot, leg


def test_a_perfect_hedge_removes_all_held_out_variance() -> None:
    spot, leg = _series(100, 1.0, 0.0, 0.0)
    got = evaluate(spot, leg, "X")
    assert got["ratio"] == pytest.approx(-1.0)
    assert got["held_out_variance_removed"] == pytest.approx(1.0)
    assert got["fit_nights"] == int(100 * FIT_SHARE)
    assert hedged([1.0, 2.0], [1.0, 2.0], -1.0) == [0.0, 0.0]


def test_the_ratio_is_not_refitted_on_the_held_out_nights() -> None:
    spot, leg = _series(100, 1.0, 0.0, 0.0)
    spot = spot[:70] + [2 * x for x in leg[70:]]  # the relationship doubles after the split
    got = evaluate(spot, leg, "X")
    assert got["ratio"] == pytest.approx(-1.0)
    assert got["held_out_variance_removed"] == pytest.approx(0.75)


def _nights(spot: list[float], perp: list[float]) -> list[dict[str, Any]]:
    start = date(2026, 1, 5)
    return [{"date": (start + timedelta(days=i)).isoformat(), "spot_overnight_log_return": s,
             "perp_overnight_log_return": p} for i, (s, p) in enumerate(zip(spot, perp,
                                                                            strict=True))]


def test_the_same_name_hedge_ties_ballast_and_beats_the_index() -> None:
    index_spot, index_leg = _series(60, 1.0, 0.0, 2.0)
    series: dict[str, Any] = {"QQQ": {"spot_symbol": "RQQQUSDT", "perp_symbol": "QQQUSDT",
                                      "nights": _nights(index_spot, index_leg)}}
    for k, name in enumerate(("AAA", "BBB", "CCC")):
        spot, perp = _series(60, 0.98, 0.0005, 0.3 * k)
        series[name] = {"spot_symbol": f"R{name}USDT", "perp_symbol": f"{name}USDT",
                        "nights": _nights(spot, perp)}
    result = compare(series)
    assert result["names"] == 3
    for body in result["per_name"].values():
        # The console's live ratio is fitted on every night, Ballast's on 70%; the held-out
        # evidence both are scored on comes from the same split and must agree.
        for key in ("held_out_variance_removed", "held_out_tail_cut", "held_out_nights"):
            assert body["argus_same_name"][key] == body["ballast_same_name"][key]
    comp = result["comparison"]["argus_index vs ballast_same_name"]
    assert comp["names_better"] == 0
    text = verdict(result)
    assert text.startswith("TIED") and "index hedge" in text
