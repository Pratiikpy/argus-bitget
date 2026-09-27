"""Weekend-gap forecasts scored walk-forward (`eval/weekend_quantiles.py`), offline."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from argus.eval import weekend_quantiles as wq
from argus.market.equity_history import Day, Gap


def test_quantile_interpolates_like_numpy_type_7() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert wq.quantile(values, 0.0) == 1.0
    assert wq.quantile(values, 1.0) == 4.0
    assert wq.quantile(values, 0.5) == pytest.approx(2.5)
    assert wq.quantile(values, 0.1) == pytest.approx(1.3)


def test_pinball_charges_each_side_by_its_quantile() -> None:
    assert wq.pinball(1.0, 0.0, 0.9) == pytest.approx(0.9)
    assert wq.pinball(-1.0, 0.0, 0.9) == pytest.approx(0.1)
    assert wq.pinball(0.0, 0.0, 0.5) == 0.0


def _days(n_weeks: int, seed: int = 7) -> list[Day]:
    """Trading days whose volatility switches regime every 40 weeks; each weekend gap is drawn
    with the week's volatility, so a volatility-aware forecast has something real to find."""
    rng = random.Random(seed)
    days: list[Day] = []
    price = 100.0
    start = date(2000, 1, 3)  # a Monday
    for week in range(n_weeks):
        sigma = 0.005 if (week // 40) % 2 == 0 else 0.03
        for d in range(5):
            day = start + timedelta(days=7 * week + d)
            opened = price * (1 + (rng.gauss(0, 2 * sigma) if d == 0 and week else 0.0))
            price = opened * (1 + rng.gauss(0, sigma))
            days.append(Day(day=day, open=opened, close=price))
    return days


def test_volatility_scaling_wins_where_volatility_moves_and_nothing_sees_the_future() -> None:
    result = wq.evaluate(_days(700), warmup=200)
    assert result is not None
    loss = result["mean_pinball_bps"]
    # the median forecast is near zero for every method, so the gain is in the two tails
    assert loss["vol_scaled"] < 0.95 * loss["unconditional"]
    assert loss["vol_matched"] < 0.95 * loss["unconditional"]
    assert loss["vol_scaled_ewma"] < 0.95 * loss["unconditional"]
    by_half = result["mean_pinball_bps_by_half"]
    assert all(by_half["vol_scaled"][h] < by_half["unconditional"][h] for h in (0, 1))


def test_a_forecast_reads_only_the_weekends_before_it() -> None:
    friday = wq.Friday(vol=0.01, trend="flat", five_day=0.0)
    history = [(Gap(date(2020, 1, 3) + timedelta(days=7 * i), date(2020, 1, 6)
                    + timedelta(days=7 * i), move), friday)
               for i, move in enumerate([-0.02, -0.01, 0.0, 0.01, 0.02] * 10)]
    got = wq.forecasts(history, friday)
    band, fell_back = got["unconditional"]
    assert band == pytest.approx((-0.02, 0.0, 0.02))
    assert not fell_back
    later = [(Gap(date(2021, 1, 1) + timedelta(days=7 * i), date(2021, 1, 4)
                  + timedelta(days=7 * i), 0.5), friday) for i in range(20)]
    assert wq.forecasts(history + later, friday)["unconditional"][0][2] > band[2]


def test_too_few_matched_weekends_fall_back_and_are_counted() -> None:
    friday = wq.Friday(vol=0.01, trend="up", five_day=0.0)
    other = wq.Friday(vol=0.02, trend="down", five_day=0.05)
    history = [(Gap(date(2020, 1, 3) + timedelta(days=7 * i), date(2020, 1, 6)
                    + timedelta(days=7 * i), 0.001 * i), other) for i in range(40)]
    band, fell_back = wq.forecasts(history, friday)["regime_matched"]
    assert fell_back and band == wq.forecasts(history, friday)["unconditional"][0]


def test_the_summary_pools_by_weekends_and_counts_both_halves() -> None:
    loss = {m: 9.5 for m in wq.METHODS} | {"unconditional": 10.0, "regime_matched": 10.5,
                                          "vol_scaled": 9.0}
    halves = {m: [9.5, 9.5] for m in wq.METHODS} | {
        "unconditional": [10.0, 10.0], "regime_matched": [10.5, 9.9], "vol_scaled": [9.0, 9.0],
        "vol_matched": [9.5, 10.5]}
    row = {"weekends_scored": 100, "mean_pinball_bps": loss,
           "mean_pinball_bps_by_half": halves,
           "band_coverage": {m: 0.8 for m in wq.METHODS}, "fallbacks": {}}
    summary = wq.summarise({"A": row, "B": row, "C": {"error": "history unreadable"}})
    assert summary["stocks"] == 2 and summary["weekends_scored"] == 200
    assert summary["pooled_mean_pinball_bps"]["vol_scaled"] == 9.0
    both = summary["beats_unconditional_in_both_halves"]
    assert both["regime_matched"] == 0 and both["vol_matched"] == 0 and both["vol_scaled"] == 2
    assert summary["best_by_stock"] == {"A": "vol_scaled", "B": "vol_scaled"}


def test_the_shipped_band_walks_exactly_as_the_evaluation_scored_it() -> None:
    """`equity_history.weekend_band` is what the weekend answer prints; its own record of how
    often its bands covered must equal the evaluation's adaptive arm on the same days."""
    from argus.market.equity_history import weekend_band

    days = _days(700)
    band = weekend_band(days)
    scored = wq.evaluate(days, warmup=wq.WARMUP)
    assert band is not None and scored is not None
    assert band.tracked == scored["weekends_scored"]
    assert band.covered == pytest.approx(scored["band_coverage"]["vol_scaled_ewma_adaptive"],
                                         abs=1e-4)
    assert band.p10 < band.p50 < band.p90
    assert 0.6 < band.covered < 0.95


def test_too_short_a_history_has_no_band() -> None:
    from argus.market.equity_history import weekend_band

    assert weekend_band(_days(100)) is None
