"""Tests for `desk/crossasset.py`, the cross-asset hedge router, on hand-built inputs."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.desk.crossasset import (
    OPEN,
    SHUT,
    CostCurve,
    RouterConfig,
    RouterInput,
    _argmin_1d,
    _ce,
    estimate,
    hours_to_boundary,
    phase_calendar,
    phase_of_bar,
    route,
    same_underlying,
    settlements_between,
)
from argus.eval.xa_arena import _case_model, adversarial_cases

HOUR = 3_600_000


def _ms(y: int, m: int, d: int, h: int) -> int:
    return int(datetime(y, m, d, h, tzinfo=UTC).timestamp() * 1000)


def test_phase_calendar_marks_cash_session_open_and_weekend_shut() -> None:
    # 2026-09-23 is a Wednesday; NYSE opens 13:30 UTC (EDT) and closes 20:00 UTC.
    assert phase_of_bar(_ms(2026, 9, 23, 13)) == SHUT  # the bar containing the open is shut
    assert phase_of_bar(_ms(2026, 9, 23, 14)) == OPEN
    assert phase_of_bar(_ms(2026, 9, 23, 19)) == OPEN
    assert phase_of_bar(_ms(2026, 9, 23, 20)) == SHUT
    assert phase_of_bar(_ms(2026, 9, 26, 15)) == SHUT  # Saturday
    hours = [_ms(2026, 9, 23, 0) + j * HOUR for j in range(48)]
    assert phase_calendar(hours) == [phase_of_bar(h) for h in hours]


def test_hours_to_boundary_counts_the_run_and_respects_the_cap() -> None:
    phases = [SHUT] * 5 + [OPEN] * 3
    assert hours_to_boundary(phases, 0, cap=72) == 5
    assert hours_to_boundary(phases, 5, cap=72) == 3
    assert hours_to_boundary(phases, 0, cap=2) == 2


def test_settlements_between_counts_the_8h_grid() -> None:
    start = _ms(2026, 9, 23, 1)
    assert settlements_between(start, 6) == 0  # 01:00 -> 07:00
    assert settlements_between(start, 7) == 1  # includes 08:00
    assert settlements_between(start, 24) == 3  # 08:00, 16:00, 00:00


def test_cost_curve_interpolates_and_extends_proportionally() -> None:
    c = CostCurve(6.0, ((1_000.0, 1.0), (5_000.0, 3.0)))
    assert c.bps(0.0) == pytest.approx(7.0)
    assert c.bps(3_000.0) == pytest.approx(8.0)
    assert c.bps(-3_000.0) == pytest.approx(8.0)
    assert c.bps(10_000.0) == pytest.approx(6.0 + 3.0 * 2)
    assert CostCurve(6.0, ()).bps(1e9) == 6.0


def test_argmin_1d_matches_a_brute_force_grid() -> None:
    a, b, x0, base, c = 1e-6, -0.05, 1_000.0, 0.0, 0.001
    got = _argmin_1d(a, b, x0, base, c, c, 1e6)

    def f(x: float) -> float:
        return a * x * x + b * x + c * abs(x - x0) + c * max(0.0, abs(x - base) - abs(x0 - base))

    grid = min((x * 10.0 for x in range(-100_000, 100_001)), key=f)
    assert f(got) <= f(grid) + 1e-9


def test_same_underlying_maps_rtoken_to_its_perpetual() -> None:
    assert same_underlying("spot:RNVDAUSDT", "perp:NVDAUSDT")
    assert not same_underlying("spot:RNVDAUSDT", "perp:BTCUSDT")


def test_every_constructed_adversarial_case_passes() -> None:
    cases = adversarial_cases()
    assert set(cases) == {"normal_risk_normal_carry", "risk_shock", "funding_spike",
                          "prohibitive_costs", "calm_forecast", "thin_history"}
    failed = {k: v for k, v in cases.items() if not v["passed"]}
    assert not failed, failed


def test_router_never_returns_a_point_worse_than_holding() -> None:
    keys = ("spot:RNVDAUSDT", "perp:NVDAUSDT", "perp:BTCUSDT")
    corr = ((1.0, 0.9, 0.3), (0.9, 1.0, 0.3), (0.3, 0.3, 1.0))
    cost = CostCurve(6.0, ((1_000.0, 1.0), (40_000.0, 5.0)))
    cfg = RouterConfig()
    for scale in (0.25, 1.0, 2.0, 6.0):
        for f_now in (-0.002, 0.0001, 0.003, 0.01):
            m = _case_model(keys, (0.02, 0.02, 0.025), corr, scale=scale)
            inp = RouterInput(
                nav=100_000.0, spot_usd={"spot:RNVDAUSDT": 30_000.0},
                perp_usd={"perp:NVDAUSDT": 0.0, "perp:BTCUSDT": 25_000.0},
                base_perp_usd={"perp:BTCUSDT": 25_000.0},
                funding_now={"perp:NVDAUSDT": 0.0001, "perp:BTCUSDT": f_now},
                funding_normal={"perp:NVDAUSDT": 0.0001, "perp:BTCUSDT": 0.0001},
                settlements_ahead=3, costs={"perp:NVDAUSDT": cost, "perp:BTCUSDT": cost})
            d = route(m, inp, cfg, tradeable=["perp:NVDAUSDT", "perp:BTCUSDT"])
            hold = _ce(m, cfg, inp, dict(inp.perp_usd))
            assert _ce(m, cfg, inp, d.target_usd) >= hold - 1e-9
            assert d.ce_gain_usd >= 0.0
            assert all(abs(v) <= cfg.max_abs_notional_nav * inp.nav for v in d.target_usd.values())


def test_estimate_refuses_with_too_little_history_and_reads_only_the_past() -> None:
    n = 400
    hours = [_ms(2026, 6, 1, 0) + j * HOUR for j in range(n)]
    phases = phase_calendar(hours)
    closes = {"perp:BTCUSDT": [100.0 + (j % 7) for j in range(n)],
              "perp:NVDAUSDT": [50.0 + (j % 5) for j in range(n)]}
    keys = ("perp:BTCUSDT", "perp:NVDAUSDT")
    thin = estimate(closes, keys, phases, 30, RouterConfig())
    assert not thin.ok and "required" in thin.reason
    i = 300
    m1 = estimate(closes, keys, phases, i, RouterConfig(min_blocks=5))
    future = {k: v[: i + 1] + [v[i] * 3.0] * (n - i - 1) for k, v in closes.items()}
    m2 = estimate(future, keys, phases, i, RouterConfig(min_blocks=5))
    assert m1.ok and m1.cov_h == m2.cov_h and m1.cov_normal_h == m2.cov_normal_h
