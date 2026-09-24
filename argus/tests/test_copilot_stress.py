"""The Portfolio Copilot stress head-to-head: the scorer that ranks ARGUS's stress sentence against
skfolio's vine copula and Entropy Pooling.

The live run needs skfolio in its own interpreter and is exercised by
`python -m argus.eval.copilot_stress`. Pinned here: stale closes are removed for every arm, a day a
rival cannot answer is dropped from that pair only, and the quantile and test arithmetic.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, timedelta
from typing import Any

import pytest

from argus.eval.copilot_rivals import wilcoxon_exact
from argus.eval.copilot_stress import (
    BENCHMARK,
    events,
    history,
    pinball,
    residual_quantile,
    score,
    stale,
    weighted_quantile,
    wilcoxon,
)


def test_a_day_with_any_unchanged_close_is_stale() -> None:
    rets = {"A": [0.01, 0.0, 0.02], BENCHMARK: [0.01, 0.01, -0.03]}
    assert [stale(rets, i) for i in range(3)] == [False, True, False]
    assert history(rets, 3) == [0, 2]


def test_events_need_history_and_a_traded_down_day() -> None:
    n = 70
    rets = {"A": [0.01] * n, BENCHMARK: [0.005] * n}
    rets[BENCHMARK][65] = -0.015
    rets[BENCHMARK][66] = -0.02
    rets["A"][66] = 0.0  # stale: not an event
    rets[BENCHMARK][10] = -0.03  # too early: not an event
    stamps = [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]
    assert events(stamps, rets) == [65]


def test_pinball_charges_misses_asymmetrically() -> None:
    assert pinball(pred=-0.02, real=-0.03, q=0.1) == pytest.approx(0.9 * 0.01)
    assert pinball(pred=-0.02, real=0.0, q=0.1) == pytest.approx(0.1 * 0.02)


def test_weighted_quantile_follows_the_weights() -> None:
    assert weighted_quantile([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.1) == 1.0
    assert weighted_quantile([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 0.1) == 3.0


def test_the_residual_quantile_is_centred_and_below_zero() -> None:
    qqq = [0.01 * math.sin(i) for i in range(100)]
    noise = [0.002 * math.cos(3 * i) for i in range(100)]
    daily = {"A": [1.5 * q + e for q, e in zip(qqq, noise, strict=True)], BENCHMARK: qqq}
    got = residual_quantile({"A": 1.0}, daily, {"A": 1.5})
    assert -0.0025 < got < 0


def test_the_normal_approximation_agrees_with_the_exact_test_at_the_boundary() -> None:
    diffs = [(-1) ** (i % 3 == 0) * (i + 1) * 0.1 for i in range(20)]
    exact = wilcoxon_exact(diffs)
    assert wilcoxon(diffs) == exact
    bigger = [*diffs, 2.5]
    assert abs(wilcoxon(bigger) - exact) < 0.25


def _row(day: str, shock: float, real: float, pred: dict[str, float | None]) -> dict[str, Any]:
    q10 = {"skfolio_vine": real - 0.01, "skfolio_ep": real - 0.01, "argus_band": real - 0.02}
    return {"day": day, "shock": shock, "real": real, "pred": pred, "q10": q10}


def test_a_day_the_rival_cannot_answer_is_dropped_from_that_pair_only() -> None:
    arms = ["argus_beta", "argus_blended", "daily_beta", "skfolio_vine", "skfolio_ep"]
    rows = []
    for k in range(6):
        pred: dict[str, float | None] = {a: -0.012 for a in arms}
        pred["argus_beta"] = -0.011
        if k == 0:
            pred["skfolio_ep"] = None
        rows.append(_row(f"d{k}", -0.025 if k == 0 else -0.012, -0.011, pred))
    scored = score(rows, arms)
    block = scored["down_1pct"]
    assert block["days_answered"]["skfolio_ep"] == 5
    assert block["comparisons"]["argus_beta vs skfolio_vine"]["days"] == 6
    assert block["comparisons"]["argus_beta vs skfolio_ep"]["days"] == 5
    assert block["mean_abs_error_pp"]["argus_beta"] == pytest.approx(0.0)
    assert scored["down_2pct"]["days"] == 1
    vine = block["pinball_q10_pp"]["skfolio_vine"]
    assert vine == pytest.approx(statistics.fmean([0.1 * 0.01 * 100] * 6))
