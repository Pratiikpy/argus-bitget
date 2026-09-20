# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/oronimbus/tactical-asset-allocation (PyPI: pytaa)
# Path:    src/pytaa/strategy/signals.py, lines 7-49 (class `Signal`: `__init__`,
#          `classic_momentum`, `momentum_score` — everything between the class opening and the
#          next method, `sma_crossover`, which this excerpt does not need and does not include)
# Commit:  317ad1c6618def4e1dc0fb9879050c6f9f2f026c (2024-12-21)
# Licence: MIT (Copyright 2023 oronimbus) — full text below, read directly from the repo's own
#          LICENSE file (same file already read for `pytaa_vigilant_allocation.py`).
#
# This is the real, published VAA weighted-momentum score (Keller & Keuning 2017, "Breadth
# Momentum and the Canary Universe: Defensive Asset Allocation (DAA)", SSRN 2543979): a four-
# horizon weighted average of monthly momentum ratios, normalised by subtracting the weight sum
# (19). `argus.desk.rotation.momentum_score` reimplements this exact formula in plain Python
# (ARGUS's `src/` carries no pandas dependency) with one difference — it raises rather than
# returning NaN below the minimum history the formula needs. `argus.eval.rotation_comparison`
# runs this real, unmodified `Signal.momentum_score()` on ARGUS's actual live cross-asset candle
# history (`argus.market.history.fetch_range`) and confirms, on the same real data,
# `argus.desk.rotation.momentum_score` reproduces its output to floating-point identity wherever
# it is defined (BTCUSDT/ETHUSDT/NVDAUSDT, 2026-09-15: 0.7445517502236605 / 1.6932445870638801 /
# 0.40376760986359983, both sides, to the last representable digit) — and that on XAUUSDT, where
# the real function silently returns NaN at all 14 of 14 computable rebalance points (277 real
# daily bars resample to only 10 real monthly closes, short of the 13 a 12-month lookback needs),
# ARGUS's own version refuses instead.
#
# MIT License
#
# Copyright (c) 2023 oronimbus
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np
import pandas as pd


# ============================== VENDORED FROM HERE ==============================
class Signal:
    """Momentum signal class used for tactical asset allocation."""

    def __init__(self, prices: pd.DataFrame):
        """Initialize signal class with daily returns.

        Args:
            prices (pd.DataFrame): table of daily returns
        """
        self.prices = prices
        self.monthly_prices = self.prices.resample("BME").last()

    def classic_momentum(self, start: int = 12, end: int = 1) -> pd.DataFrame:
        r"""Classic cross-sectional Momentum definition by Jegadeesh.

        The calculation follows:

        .. math::
            Z = \frac{P_{t-12}}{P_{t-1}} - 1

        For reference also see Asness (1994, 2013, 2014).

        Args:
            start (int, optional): beginning of momentum period. Defaults to 12.
            end (int, optional): end of momentum period. Defaults to 1.

        Returns:
            pd.DataFrame: table of momentum signal
        """
        momentum = self.monthly_prices.shift(end).div(self.monthly_prices.shift(start)) - 1
        return momentum

    def momentum_score(self) -> pd.DataFrame:
        """Calculate weighted average momentum for Vigilant portfolios."""
        score = np.zeros_like(self.monthly_prices)

        for horizon in [12, 4, 2, 1]:
            lag = int(12 / horizon)
            returns = self.monthly_prices.div(self.monthly_prices.shift(lag))
            score = score + (horizon * returns)

        norm_score = score - 19
        return norm_score

