"""Volatility-targeted sizing (`backtest/vol_target.py`, build-list 1.6), offline: the series are
simulated here from a known GARCH(1,1) with Student-t shocks, so the fit has a truth to find."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from argus.backtest import vol_target as vt


def _simulate(n: int, omega: float = 0.2, alpha: float = 0.08, beta: float = 0.9,
              nu: float = 4.0, seed: int = 7) -> list[float]:
    """Closes from percent returns r_t = sigma_t * z_t, z_t Student-t scaled to unit variance."""
    rng = random.Random(seed)
    s2 = omega / (1 - alpha - beta)
    closes, price = [100.0], 100.0
    for _ in range(n):
        chi = sum(rng.gauss(0, 1) ** 2 for _ in range(int(nu)))
        z = rng.gauss(0, 1) / math.sqrt(chi / nu) * math.sqrt((nu - 2) / nu)
        r = math.sqrt(s2) * z
        price *= 1 + r / 100
        closes.append(price)
        s2 = omega + alpha * r * r + beta * s2
    return closes


def test_the_student_t_fit_finds_heavy_tails() -> None:
    closes = _simulate(3000)
    returns = [100 * (b / a - 1) for a, b in itertools.pairwise(closes)]
    got = vt.fit(returns)
    assert got.alpha + got.beta == pytest.approx(0.98, abs=0.03)
    assert got.beta == pytest.approx(0.9, abs=0.06)
    assert 3.0 < got.nu < 6.5


def test_forecasts_use_only_the_past() -> None:
    """The truncation test: the forecast made at close i is the same whether or not later closes
    exist (freqtrade's lookahead method, applied to the forecast)."""
    closes = _simulate(900)
    full = vt.forecasts(closes, min_train=500, refit_every=21)
    for cut in (560, 701, 899):
        assert vt.forecasts(closes[:cut], min_train=500, refit_every=21) == full[:cut]
    assert full[499] is None and full[500] is not None


def test_size_is_target_over_forecast_clipped() -> None:
    assert vt.size(50.0, 25.0) == 0.5
    assert vt.size(10.0, 60.0) == vt.MAX_SIZE
    assert vt.size(400.0, 20.0) == vt.MIN_SIZE
    assert vt.size(None, 20.0) == vt.MIN_SIZE


def test_the_default_target_is_known_before_the_first_sized_day() -> None:
    closes = _simulate(900)
    _, target = vt.sizes(closes, 365)
    _, again = vt.sizes(closes[:700], 365)
    assert target == again  # read from the training window only, so no later day moves it
