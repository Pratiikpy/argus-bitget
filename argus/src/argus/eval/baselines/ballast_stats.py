# ruff: noqa
# mypy: ignore-errors
#
# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/Ritapossible/Ballast (an S2 Track 3 entry)
# Path:    ballast/stats.py (the whole file)
# Commit:  5cf675904bbb27ea338377b4cfe01a79d880086f (2026-09-24)
# Blob:    c3397f893d6aa89ad013f07999bea3f59403b9ad
# SHA256 of the vendored body: a731201da6aa63163569f39ae363993bf90a5fa26042aa19c93a599c18d9fa21
# Licence: MIT (Copyright (c) 2026 Ballast contributors), full text below.
#
# `t_stat` is the significance test behind Ballast's published "Gate 1" and "Gate 1b" event
# verdicts (docs/RESEARCH.md sections 5 and 6b): a one-sample t on returns pooled across names.
# Their own text says the pooled t = 3.08 "comes from stacking 12 names that move together on the
# same nights, which inflates significance by ignoring cross-sectional correlation" — stated,
# not corrected. `argus.eval.eventdriven_rivals` runs this function unmodified on the same
# clustered events as ARGUS's Kolari-Pynnonen-deflated tests. Loaded by
# `eval/baselines/vibe_trading_eventstudy_loader.py`, which re-checks the SHA256 above.
#
# MIT License
#
# Copyright (c) 2026 Ballast contributors
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
# ============================== VENDORED FROM HERE ==============================
"""Minimal statistics helpers. No dependencies, so `make research` needs no install."""
from __future__ import annotations

import math
import statistics as st


def ols(x: list[float], y: list[float]) -> tuple[float, float, list[float]]:
    """Univariate OLS. Returns (beta, r_squared, residuals)."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("need at least 3 paired observations")
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((u - mx) ** 2 for u in x)
    if sxx == 0:
        raise ValueError("zero variance in regressor")
    beta = sum((u - mx) * (v - my) for u, v in zip(x, y, strict=True)) / sxx
    resid = [v - my - beta * (u - mx) for u, v in zip(x, y, strict=True)]
    vy = st.pvariance(y)
    r2 = 1 - st.pvariance(resid) / vy if vy else float("nan")
    return beta, r2, resid


def t_stat(a: list[float]) -> tuple[float, float]:
    """(mean, t) for H0: mean == 0."""
    if len(a) < 8:
        return float("nan"), float("nan")
    m, s = st.mean(a), st.stdev(a)
    return m, m / (s / math.sqrt(len(a))) if s else float("nan")


def percentile(a: list[float], p: float) -> float:
    s = sorted(a)
    return s[min(int(p * len(s)), len(s) - 1)]


def bp(x: float) -> float:
    return x * 1e4
