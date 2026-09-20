"""Regime detection, run against its two real references and against the two-line rule it claims
to beat — and it loses to both, on numbers this module produced rather than asserted.

`desk/regime.py` is ARGUS's offline FLUSS (Gharghabi et al., ICDM 2017). Its module docstring cites
`matrix-profile-foundation/matrixprofile/algorithms/regimes.py` and `stumpy/floss.py` by file:line
and explains, correctly, which choice it took from which. What it never did was **run** either
reference on the same series and compare, and the single published run (`data/regimes.json`:
NVDAUSDT, 1,439 bars) reported `threshold_agrees: true` — the module's own output saying it found
nothing the incumbent had not. One symbol, no numerical parity, no chance baseline. This module
closes all three, across the full 12-symbol rToken universe.

**Finding 1 — the incumbent it compared itself against was not the incumbent.** `desk/regime.py`'s
`_threshold_flips` says in its own docstring that it reimplements
`strategies/track1_suite.py:206-215` and that "the arithmetic is the same and the comparison would
be meaningless if it were not". Read side by side, the arithmetic is **not** the same:
`track1_suite.py:51-63`'s `_vol` is a sample **standard deviation of bar returns**, while
`regime.py:306-312`'s `_volatility_bps` — the function `_threshold_flips` calls — is the **median
absolute bar-to-bar move**. Different statistics, different flip sets, confirmed by running both:
across the 12 rTokens the real `rotation_regime_switch` changes state 400 times and the proxy 559,
and only 71 of the 400 real flips (17.8%) fall within 2 bars of a proxy flip. `data/regimes.json`'s
`threshold_agrees` field was therefore computed against something that is not the rule it names.
This module drives the **real** `rotation_regime_switch` over real `Bar` objects instead, and keeps
the proxy alongside it only to measure the gap.

**Finding 2 — the number this capability owed and never produced.** Across the 12 rTokens FLUSS
places 23 boundaries, 6 of them inside the incumbent's own 240-bar warmup where the volatility rule
cannot flip at all and a "discovery" would only be the rule's silence. Of the 17 boundaries in the
region where both methods are free to answer (`comparable_region`), **3 (17.6%) have no real
incumbent flip within 24 bars**. That looks like an argument for FLUSS until the null is computed:
the real rule flips 400 times across the 12 symbols (23 to 39 each), so its +/-24-bar
neighbourhoods already cover 66.9% of that region, and a boundary dropped at a uniformly random
comparable bar would look "novel" 33.1% of the time. FLUSS's 17.6% is **well below** that —
one-sided binomial p for novelty above the null is 0.954. **There is no evidence, on 12 symbols,
that FLUSS finds regime boundaries the two-line volatility rule misses; it finds fewer than
throwing a dart would.** Scored on the same incumbent at the same tolerance, ruptures' budgeted
boundaries are novel 7 of 18 times (38.9%) and stumpy's 3 of 19 (15.8%) — ruptures is the only one
of the three above its own null, which cuts against ARGUS rather than for it. That is the honest
answer to the question `desk/regime.py` exists to answer, and it is a demotion.
(Scored over the whole series instead, the same run reads 8 of 23,
34.8%, against a 40.5% null — the same verdict, weaker, and 5 of those 8 "novel" boundaries were
only novel because the incumbent had not warmed up yet. `comparable_region` explains the fix.)

**Finding 3 — exact numerical parity with stumpy on the part that is shared.** ARGUS's own
`matrix_profile_index` (pure-Python brute force, correlation identity) and the real, installed
`stumpy.stump()` return the **identical** nearest-neighbour index on all 12 symbols: 16,992 windows
compared, zero disagreements, agreement 1.0. Boundaries then land within 1 bar on 18 of 23 and
within one window on 20 of 23; the residual disagreement is not the profile, it is the idealised
arc curve (Finding 4).

**Finding 4 — the documented departure from stumpy is load-bearing, and it costs boundaries.**
`desk/regime.py` deliberately takes matrixprofile's analytic parabola over stumpy's fitted beta
distribution (`floss.py:111`). Isolating exactly that one choice — feeding the SAME ARGUS arc
counts through the real `stumpy.floss._cac` twice, once with its own beta IAC and once with ARGUS's
parabola as `custom_iac` — moved 5 of 24 boundaries on the published run, two of them by more
than 300 bars. On GOOGLUSDT the beta IAC finds a second boundary the parabola flattens away
entirely, which is also why the two arms do not even produce the same NUMBER of boundaries.
The parabola is cheaper and deterministic, as claimed; it is not equivalent, which the
original docstring implied by not saying otherwise.

**Finding 5 — one narrow place ARGUS is better than its reference, and one where all three fail.**
Fed a corrected arc curve with no dip in it (all ones — a series with no regime boundary), the real
`stumpy.floss._rea` returns `[0, 0]`: two fabricated boundaries, the same index twice, at the very
edge its own exclusion zone was supposed to pin. ARGUS's `find_boundaries` returns `[]`, because of
its explicit `if working[best] >= 1.0: break`. The same defect is reachable on real data: with the
parabola IAC, GOOGLUSDT drives `_rea` to report `[0, 191]` while ARGUS reports `[192]` alone.
**The win stops at the curve.** Fed a constant PRICE series, ARGUS invents two boundaries of its
own (indices 335 and 456), stumpy invents two others (291, 411) and `ruptures.KernelCPD` two more
(26, 574). Every z-normalised window of a flat series is zeros, so every correlation is exactly 0
and the nearest neighbour is pure tie-breaking; all three then read structure into that artefact.
ARGUS's arc curve at least bottoms out at ~0.63 against stumpy's ~0.03 — much less confident in its
own fabrication — but a constant-input guard is a real, open defect in `desk/regime.py` that this
comparison found and deliberately did not fix, because patching the module under test from inside
its own comparison would make the comparison unfalsifiable.

**Finding 6 — ruptures, cloned here since the corpus was built and never run until now, answers a
different question and answers it more coherently.** `deepcharles/ruptures` (BSD-2-Clause) is run
two ways on the same series: `Pelt(model="normal")` at the textbook BIC penalty `2*log(n)`
(CostNormal is `|I| log det(Sigma_I)`, i.e. Gaussian -2 log-likelihood, so two free parameters per
segment), and `KernelCPD(kernel="rbf")` asked for exactly the same number of boundaries FLUSS was
asked for, the only like-for-like comparison available. PELT-at-BIC is not usable here and this says
so rather than scoring it: it returns 44 to 50 boundaries per symbol (552 in all), and they pile
onto the weekly market-hours calendar — one hour-of-week bucket collects 38 of them against a
uniform expectation of 3.3 — which is the rToken session cycle, not a regime.
The fixed-count kernel run is the real result, and it wins the one
objective test available without ground truth — see Finding 7.

**Finding 7 — the coherence test, and why it is the fairest referee here.** Nobody knows where a
real regime boundary is, so "who is right" is normally unanswerable. QQQUSDT, TQQQUSDT and SQQQUSDT
escape that: they are the same NASDAQ-100 exposure at 1x, 3x and -3x. Measured on this data, hourly
log-return correlation QQQ-TQQQ is 0.983 at beta 2.94 and QQQ-SQQQ is -0.975 at beta -2.93. A
regime change in the underlying must therefore appear at the same hour in all three, so the spread
of a method's k-th boundary across the family is a pure error measurement. ARGUS and stumpy both
spread by 50 and 122 bars; ruptures' rbf kernel spreads by 0 and 1 bars on price and 0 and 0 on log
returns. **Stated against it honestly: FLUSS's z-normalised distance matches shape and is not
sign-invariant, so SQQQ's inverted path is a genuinely different shape to it — but QQQ vs TQQQ
alone, same sign and pure scale, which z-normalisation IS invariant to, still spreads 50 and 122
bars.** And a shared calendar can manufacture coherence, so that control is measured too rather
than waved away. It does not clear ruptures outright and is not reported as if it did: on the
hour-of-week histogram PELT-at-BIC is extreme (fullest bucket 38 against a uniform expectation of
3.3, z=19.2), and the budgeted methods sit lower but not equal (ruptures' fullest bucket 4 against
an expectation of 0.14, z=10.2; ARGUS's 2, z=5.1). Read directly, all four boundaries in ruptures'
fullest bucket ARE the QQQ family landing together — which is the coherence being measured, not an
independent calendar effect — so the control neither proves nor refutes Finding 7 and is published
as a number rather than a verdict.

**Cost.** ARGUS's pure-Python profile takes 3.7-5.6 s per 1,439-bar symbol. Warm `stumpy.stump`
takes 0.017 s for the bit-identical answer — a measured 310x — and ruptures' PELT 0.43 s. Two
orders of magnitude, for output that agrees exactly.

**Every figure above is from the published `data/regime_comparison.json`, and every figure moves.**
The series are fetched live on each run, so boundary counts and the ablation's own totals shift
between passes (the same ablation read 4-of-16 one pass and 5-of-24 the next). What has not moved
across passes is every direction: parity exact, novelty below its null, family spread worse than
ruptures'. The artefact is the source of truth; this docstring is a reading of it.

**Verdict: the baseline wins.** ruptures is better where a ground-truth-free test can be run,
stumpy is bit-identical where the machinery is shared and two orders of magnitude faster, and
against the two-line incumbent this capability has no measurable edge at all. The one thing that is
ARGUS's own is the narrow degenerate-curve guard in Finding 5, which does not extend to degenerate
price input. Per this project's own standing rule that a weak capability is demoted rather than
shipped as filler, `desk/regime.py` should be recorded as LOST.

**What would change the verdict**, stated in advance so it cannot be moved afterwards: FLUSS's
novelty rate clearing the null at 5% on the comparable region (Finding 2), or ARGUS's family spread
falling at or below ruptures' (Finding 7). Both are asserted by `tests/test_regime_comparison.py`
in their current direction, so either would surface as a test failure rather than as silence.

    python -m argus.eval.regime_comparison
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import ruptures
import stumpy
from scipy.stats import binomtest
from stumpy.floss import _cac, _rea

from argus.backtest.engine import Bar
from argus.desk.regime import (
    EXCLUSION_FACTOR,
    corrected_arc_curve,
    find_boundaries,
    idealised_arc,
    matrix_profile_index,
)

# `_threshold_flips` is private to `desk/regime.py` and is imported here on purpose: auditing it
# against the rule it claims to reimplement IS this module's Finding 1, and a fourth copy of the
# arithmetic would audit the copy rather than the original.
from argus.desk.regime import _threshold_flips as regime_proxy_flips
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range
from argus.strategies.track1_suite import rotation_regime_switch

DAYS = 60
"""Sixty days of hourly bars is 1,439 per symbol — the same size as `data/regimes.json`'s single
published run, so Finding 2 is measured on the window the original claim was made on."""

WINDOW = 24
REGIMES = 3
TOLERANCE_BARS = 24
"""How close an incumbent flip must be to count as having found the same boundary. One window, the
same tolerance `desk/regime.py`'s own `segment()` uses for `threshold_agrees`."""

INCUMBENT_SLOW = 240
"""`track1_suite.py:210`'s slow lookback. `_vol` returns 0.0 until `i - lookback >= 1`, so the real
rule has no opinion before bar 241 and a "flip" counted there would be a warmup artefact."""

COMPARABLE_FROM = INCUMBENT_SLOW + 1
"""The first bar at which the incumbent has an opinion. Lower edge of `comparable_region`.

**This is a correction to this module's own first measurement, not a refinement.** FLUSS's pinned
head ends at bar 120 (`window * EXCLUSION_FACTOR`) but the incumbent's slow leg needs 240 bars, so
bars 121-240 are a window where FLUSS can place a boundary and the volatility rule structurally
cannot flip. Scoring that window counts the incumbent's warmup as ARGUS's discovery. Run without
the restriction the module reported 8 of 23 boundaries "novel" (34.8%) — and 5 of those 8 sat in
the warmup. Restricted, the same run gives 3 of 17 (17.6%) against a 33.1% null: a stronger result
in the same direction, and an honest one. Boundaries outside the region are reported and counted,
never silently dropped."""

BIC_PARAMETERS_PER_SEGMENT = 2
"""Mean and variance. `CostNormal` is `|I| log det(Sigma_I)`, i.e. Gaussian -2 log-likelihood up to
a constant, so the BIC penalty for one extra segment is `k * log(n)` with `k = 2`."""

FAMILY = ("QQQUSDT", "TQQQUSDT", "SQQQUSDT")
"""Three tokens over one underlying index at 1x / 3x / -3x. Finding 7's referee."""

HOURS_PER_WEEK = 168
"""The calendar control's bucket count. rTokens track US equities, so an hourly series carries a
hard weekly structure and a method that has merely found the session cycle will concentrate here."""


FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = 4.0
"""Retry policy for the venue's rate limiter.

**Added because a real run lost half its sample to it, not as defensive boilerplate.** This module
fetches the 12-symbol universe four separate times (base case, ablation, OOS, reproducibility) and
the fourth pass drew `HTTP 429 Too Many Requests` on 6 of 12 symbols. Nothing crashed — they landed
in `failures` exactly as designed — but the out-of-sample check then ran on 6 symbols while
reporting itself as the out-of-sample check, which is the quiet version of the sample-shrinkage
this project has been caught by before. Backoff is 4s, 8s, 16s; a symbol that still fails is still
named in `failures` rather than retried forever.
"""


def fetch_series(symbol: str, *, days: int = DAYS) -> list[tuple[datetime, float]]:
    """One symbol's real hourly closes, retried through the venue's rate limiter.

    Errors are swallowed only between attempts — the last one propagates, so a symbol that is
    genuinely unavailable still reaches the caller as a failure rather than as an empty list that
    would look like a short series.
    """
    last: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            bars = fetch_range(
                symbol, days=days, interval="1H", candle_type=CandleType.MARKET
            )
            return [(c.ts, float(c.close)) for c in bars]
        except Exception as exc:
            last = exc
            if attempt < FETCH_ATTEMPTS - 1:
                time.sleep(FETCH_BACKOFF_SECONDS * (2 ** attempt))
    assert last is not None
    raise last


def fetch_universe(
    symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = DAYS,
) -> tuple[dict[str, list[tuple[datetime, float]]], dict[str, str]]:
    """One real fetch per symbol. A symbol that fails, or is too short to segment, is named in
    `failures` rather than dropped silently — a comparison run on 6 of 12 symbols that reads as 12
    is the failure mode this project has already corrected elsewhere, and is exactly what the
    venue's rate limiter produced here before `fetch_series` grew its backoff."""
    series: dict[str, list[tuple[datetime, float]]] = {}
    failures: dict[str, str] = {}
    for symbol in symbols:
        try:
            rows = fetch_series(symbol, days=days)
        except Exception as exc:
            failures[symbol] = str(exc)[:120]
            continue
        if len(rows) < INCUMBENT_SLOW + WINDOW * EXCLUSION_FACTOR * 2:
            failures[symbol] = f"{len(rows)} bars is too short for both sides of the comparison"
            continue
        series[symbol] = rows
    return series, failures


# ---------------------------------------------------------------------------------------------
# The four segmenters, each run on the SAME values
# ---------------------------------------------------------------------------------------------

def argus_profile(values: Sequence[float], *, window: int = WINDOW) -> tuple[list[int], float]:
    start = time.perf_counter()
    index = matrix_profile_index(values, window)
    return index, time.perf_counter() - start


def argus_boundaries(
    index: Sequence[int], *, window: int = WINDOW, regimes: int = REGIMES,
) -> tuple[list[int], list[float]]:
    curve = corrected_arc_curve(index, window, exclusion_factor=EXCLUSION_FACTOR)
    cuts = find_boundaries(
        curve, count=max(0, regimes - 1), window=window, exclusion_factor=EXCLUSION_FACTOR,
    )
    return cuts, curve


def stumpy_run(
    values: Sequence[float], *, window: int = WINDOW, regimes: int = REGIMES,
) -> tuple[list[int], list[float], list[int], float]:
    """The real, installed `stumpy.stump` + `stumpy.fluss`, its own public API, own defaults.

    Returns the nearest-neighbour index, the corrected arc curve, the boundaries, and the wall
    clock for `stump` alone. `stumpy` is numba-JIT'd, so the caller must warm it once before
    timing means anything — `measure_costs` does; `run_symbol` reports the number it got and says
    nothing about whether it was warm.
    """
    series = np.asarray(values, dtype=np.float64)
    start = time.perf_counter()
    profile = stumpy.stump(series, m=window)
    elapsed = time.perf_counter() - start
    index = [int(v) for v in profile[:, 1]]
    cac, regime_locations = stumpy.fluss(
        profile[:, 1], L=window, n_regimes=regimes, excl_factor=EXCLUSION_FACTOR,
    )
    curve = [float(v) for v in np.nan_to_num(cac, nan=1.0)]
    return index, curve, sorted(int(v) for v in regime_locations), elapsed


def ruptures_bic(values: Sequence[float], *, window: int = WINDOW) -> tuple[list[int], float]:
    """PELT with a Gaussian mean-and-variance cost at the textbook BIC penalty.

    **The penalty is derived from the cost function's own source, not copied from a tutorial.**
    `src/ruptures/costs/costnormal.py:70-80` returns `slogdet(cov)[1] * (end - start)`, i.e.
    `|I| log det(Sigma_I)` — Gaussian -2 log-likelihood up to an additive constant. BIC charges
    `k log n` per extra segment and a univariate Gaussian segment has two free parameters (mean,
    variance), so the penalty is `2 log n`. Using `model="rbf"` here instead would have left the
    penalty with no principled value at all, which is why the kernel is confined to the
    fixed-count arm (`ruptures_fixed`).

    `min_size=window` so the two methods are allowed the same minimum segment; `jump=1` so the
    change-point grid is every bar, matching FLUSS's per-window resolution rather than handing
    ruptures a coarser search and then calling its answer imprecise. `_seg` appends `n_samples` to
    its admissible list (`src/ruptures/detection/pelt.py:68`) so the returned list always ends at
    the series end, which is a terminator and not a boundary — stripped here rather than counted.
    """
    signal = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    penalty = BIC_PARAMETERS_PER_SEGMENT * math.log(len(values))
    start = time.perf_counter()
    found = ruptures.Pelt(model="normal", min_size=window, jump=1).fit(signal).predict(pen=penalty)
    elapsed = time.perf_counter() - start
    return [int(b) for b in found if int(b) < len(values)], elapsed


def ruptures_fixed(
    values: Sequence[float], *, window: int = WINDOW, regimes: int = REGIMES,
) -> list[int]:
    """`KernelCPD` with an rbf kernel, asked for exactly `regimes - 1` boundaries.

    This is the only like-for-like configuration: FLUSS is given a boundary budget and returns
    that many, so ruptures must be too. Scoring a penalty-driven method that chooses its own
    count against a budgeted one measures the choice of penalty, not the method.

    `KernelCPD.predict(n_bkps=...)` (`src/ruptures/detection/kernelcpd.py:89-137`) dispatches an
    rbf kernel to its own compiled `ekcpd_Gaussian` exact solver and caches by `n_bkps`, so this
    is ruptures' optimal segmentation at that budget rather than a greedy approximation — the
    fairest possible version of the baseline, not the fastest.
    """
    signal = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    algo = ruptures.KernelCPD(kernel="rbf", min_size=window).fit(signal)
    found = algo.predict(n_bkps=max(1, regimes - 1))
    return [int(b) for b in found if int(b) < len(values)]


def incumbent_flips(series: Sequence[tuple[datetime, float]]) -> list[int]:
    """Where the REAL `strategies/track1_suite.py:205-214` rule changes its mind.

    Driven over real `Bar` objects through the real exported function, not reimplemented — that
    reimplementation is the defect this module found (Finding 1). `rotation_regime_switch` returns
    a weight (1.0 long, 0.0 flat), so a flip is a change in that weight.
    """
    bars = [Bar(ts=ts, close=Decimal(str(price))) for ts, price in series]
    weights = [rotation_regime_switch(bars, i) for i in range(len(bars))]
    return [
        i for i in range(INCUMBENT_SLOW + 1, len(bars)) if weights[i] != weights[i - 1]
    ]


# ---------------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------------

def _agreement(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """For each boundary on the left, the distance to the nearest boundary on the right."""
    if not right:
        return []
    return [min(abs(a - b) for b in right) for a in left]


def _exact_index_agreement(left: Sequence[int], right: Sequence[int]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(1 for a, b in zip(left, right, strict=True) if a == b) / len(left)


def comparable_region(n: int, *, window: int = WINDOW) -> tuple[int, int]:
    """The half-open bar range `[lo, hi)` in which BOTH methods can answer.

    Two constraints, each from a different side, and both are needed or the null is wrong:

    * `lo = COMPARABLE_FROM` — the incumbent's slow leg is silent before bar 241.
    * `hi = n - window + 1 - window * EXCLUSION_FACTOR` — the arc curve has `n - window + 1`
      entries and `find_boundaries` pins the last `window * EXCLUSION_FACTOR` of them to 1.0, so
      FLUSS cannot place a boundary there however the series behaves. On the real 1,439-bar
      window that is bar 1296, and the largest boundary any symbol produced is 1295.

    Leaving the tail in was the milder of the two errors but the same mistake: it credits the
    incumbent's flips in a region FLUSS was never allowed to answer in, inflating the null by
    about 1.7 points (31.6% to 33.2% on the real run). Both are corrected rather than the
    convenient one.
    """
    pin = window * EXCLUSION_FACTOR
    lo = max(COMPARABLE_FROM, pin)
    hi = max(lo, n - window + 1 - pin)
    return lo, hi


def _covered_fraction(
    flips: Sequence[int], n: int, *, tolerance: int = TOLERANCE_BARS, window: int = WINDOW,
) -> float:
    """What share of the comparable bar positions sit within `tolerance` of an incumbent flip.

    This is the null Finding 2 turns on. A rule that flips 35 times in 1,439 bars puts a 49-bar
    window around each flip; unless those windows are computed, "the FLUSS boundary agrees with a
    flip" is a statement about how often the rule flips, not about where the boundary is.
    """
    lo, hi = comparable_region(n, window=window)
    if hi <= lo:
        return 0.0
    covered: set[int] = set()
    for flip in flips:
        covered.update(range(max(lo, flip - tolerance), min(hi, flip + tolerance + 1)))
    return len(covered) / (hi - lo)


def _comparable(cuts: Sequence[int], n: int, *, window: int = WINDOW) -> list[int]:
    """The boundaries that fall where both methods were free to answer."""
    lo, hi = comparable_region(n, window=window)
    return [c for c in cuts if lo <= c < hi]


def _novel(
    cuts: Sequence[int], flips: Sequence[int], n: int, *, tolerance: int = TOLERANCE_BARS,
) -> list[int]:
    """Comparable boundaries with no incumbent flip nearby. Others are not eligible either way."""
    return [
        c for c in _comparable(cuts, n) if not any(abs(f - c) <= tolerance for f in flips)
    ]


@dataclass(frozen=True, slots=True)
class SymbolRun:
    """Every segmenter's answer for one symbol, on one fetched series."""

    symbol: str
    bars: int
    argus_cuts: tuple[int, ...]
    stumpy_cuts: tuple[int, ...]
    ruptures_bic_cuts: tuple[int, ...]
    ruptures_fixed_cuts: tuple[int, ...]
    real_flips: tuple[int, ...]
    proxy_flips: tuple[int, ...]
    profile_index_agreement: float
    cac_correlation: float
    covered_fraction: float
    argus_seconds: float
    stumpy_seconds: float
    ruptures_seconds: float

    @property
    def argus_novel(self) -> tuple[int, ...]:
        return tuple(_novel(self.argus_cuts, self.real_flips, self.bars))

    @property
    def stumpy_novel(self) -> tuple[int, ...]:
        return tuple(_novel(self.stumpy_cuts, self.real_flips, self.bars))

    @property
    def ruptures_novel(self) -> tuple[int, ...]:
        return tuple(_novel(self.ruptures_fixed_cuts, self.real_flips, self.bars))

    @property
    def argus_comparable(self) -> tuple[int, ...]:
        """ARGUS boundaries the incumbent could have matched — the novelty question's sample."""
        return tuple(_comparable(self.argus_cuts, self.bars))

    @property
    def argus_outside_region(self) -> tuple[int, ...]:
        """ARGUS boundaries where one side or the other was structurally silent. Reported, never
        scored either way — in practice all of them are the incumbent's own warmup, because
        `find_boundaries` already refuses to place anything past `hi`."""
        comparable = set(self.argus_comparable)
        return tuple(c for c in self.argus_cuts if c not in comparable)

    @property
    def real_flips_matched_by_proxy(self) -> int:
        """How many of the real rule's flips the `desk/regime.py` proxy places within 2 bars.

        Two bars, not one window: this asks whether the proxy reproduces the rule, and a
        reimplementation that claims identical arithmetic should land on the same bar.
        """
        return sum(1 for f in self.real_flips if any(abs(f - p) <= 2 for p in self.proxy_flips))

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bars": self.bars,
            "argus_cuts": list(self.argus_cuts),
            "stumpy_cuts": list(self.stumpy_cuts),
            "ruptures_bic_n_cuts": len(self.ruptures_bic_cuts),
            "ruptures_fixed_cuts": list(self.ruptures_fixed_cuts),
            "n_real_incumbent_flips": len(self.real_flips),
            "n_regime_py_proxy_flips": len(self.proxy_flips),
            "real_flips_matched_by_proxy_within_2_bars": self.real_flips_matched_by_proxy,
            "profile_index_agreement_with_stumpy": round(self.profile_index_agreement, 6),
            "cac_correlation_with_stumpy": round(self.cac_correlation, 6),
            "incumbent_covered_fraction": round(self.covered_fraction, 6),
            "argus_comparable_boundaries": list(self.argus_comparable),
            "argus_boundaries_outside_comparable_region": list(self.argus_outside_region),
            "argus_novel_boundaries": list(self.argus_novel),
            "argus_vs_stumpy_offsets": _agreement(self.argus_cuts, self.stumpy_cuts),
            "argus_vs_ruptures_fixed_offsets": _agreement(
                self.argus_cuts, self.ruptures_fixed_cuts
            ),
            "argus_seconds": round(self.argus_seconds, 4),
            "stumpy_seconds": round(self.stumpy_seconds, 4),
            "ruptures_seconds": round(self.ruptures_seconds, 4),
        }


def run_symbol(symbol: str, series: Sequence[tuple[datetime, float]]) -> SymbolRun:
    """All four segmenters plus both incumbents, on one already-fetched series. No network."""
    values = [price for _, price in series]
    argus_index, argus_seconds = argus_profile(values)
    argus_cuts, argus_curve = argus_boundaries(argus_index)
    stumpy_index, stumpy_curve, stumpy_cuts, stumpy_seconds = stumpy_run(values)
    bic_cuts, ruptures_seconds = ruptures_bic(values)
    fixed_cuts = ruptures_fixed(values)
    real = incumbent_flips(series)
    proxy = regime_proxy_flips(values)
    correlation = float(
        np.corrcoef(np.asarray(argus_curve), np.asarray(stumpy_curve))[0, 1]
    ) if len(argus_curve) == len(stumpy_curve) else float("nan")
    return SymbolRun(
        symbol=symbol,
        bars=len(values),
        argus_cuts=tuple(argus_cuts),
        stumpy_cuts=tuple(stumpy_cuts),
        ruptures_bic_cuts=tuple(bic_cuts),
        ruptures_fixed_cuts=tuple(fixed_cuts),
        real_flips=tuple(real),
        proxy_flips=tuple(proxy),
        profile_index_agreement=_exact_index_agreement(argus_index, stumpy_index),
        cac_correlation=correlation,
        covered_fraction=_covered_fraction(real, len(values)),
        argus_seconds=argus_seconds,
        stumpy_seconds=stumpy_seconds,
        ruptures_seconds=ruptures_seconds,
    )


def _novelty_summary(runs: Sequence[SymbolRun]) -> dict[str, Any]:
    """Finding 2, pooled: the owed number, its null, and whether the gap survives a binomial test.

    Two binomial tests, both one-sided, because they answer different questions and reporting only
    one of them would hide half the result. `binomial_p_novelty_above_chance` is the claim
    `desk/regime.py` actually makes — FLUSS lands away from the incumbent MORE often than a random
    bar would — and `beats_chance` is that same comparison as a bare flag. The agreement-side test
    is its mirror: if FLUSS agreed with the incumbent more than chance it would be actively
    redundant, which is a different (and worse) finding than merely having no edge.

    **The chance rate must be weighted by boundary count, not averaged over symbols.** A symbol
    that produced one boundary rather than two contributes one trial to the binomial, so its own
    coverage must contribute one trial's worth of null. Averaging the 12 per-symbol coverages
    equally would mis-state the null whenever `find_boundaries` returns short, which it does
    (GOOGLUSDT yields one boundary, not two).

    **The sample is `argus_comparable`, not `argus_cuts`** — see `COMPARABLE_FROM`. Boundaries in
    the incumbent's 240-bar warmup are counted and published as their own field, because dropping
    them without saying so would be indistinguishable from cherry-picking the sample that gave the
    answer this module wanted.
    """
    total = sum(len(r.argus_comparable) for r in runs)
    novel = sum(len(r.argus_novel) for r in runs)
    chance_agreement = (
        sum(r.covered_fraction * len(r.argus_comparable) for r in runs) / total if total else 0.0
    )
    agreed = total - novel
    testable = bool(total) and 0.0 < chance_agreement < 1.0
    agreement_test = (
        binomtest(agreed, total, chance_agreement, alternative="greater") if testable else None
    )
    novelty_test = (
        binomtest(novel, total, 1.0 - chance_agreement, alternative="greater")
        if testable
        else None
    )
    return {
        "n_argus_boundaries": total,
        "n_argus_boundaries_all": sum(len(r.argus_cuts) for r in runs),
        "n_argus_boundaries_outside_region": sum(len(r.argus_outside_region) for r in runs),
        "comparable_region_example": list(comparable_region(runs[0].bars)) if runs else None,
        "n_novel_vs_real_incumbent": novel,
        "novelty_rate": novel / total if total else None,
        "chance_novelty_rate": 1.0 - chance_agreement,
        "agreement_rate": agreed / total if total else None,
        "chance_agreement_rate": chance_agreement,
        # FLUSS "beats chance" only by being MORE novel than a randomly placed bar, never less.
        # The first version of this line had the inequality the wrong way round and reported a
        # win for a capability that had just measured 34.8% novelty against a 40.5% null.
        "beats_chance": bool(novel / total > 1.0 - chance_agreement) if total else None,
        "binomial_p_novelty_above_chance": (
            float(novelty_test.pvalue) if novelty_test is not None else None
        ),
        "binomial_p_agreement_above_chance": (
            float(agreement_test.pvalue) if agreement_test is not None else None
        ),
        # Comparable counts, matching the novel counts beside them — a denominator that included
        # warmup boundaries against a numerator that excluded them would understate both
        # baselines' novelty and quietly flatter ARGUS by comparison.
        "n_stumpy_boundaries": sum(len(_comparable(r.stumpy_cuts, r.bars)) for r in runs),
        "n_stumpy_novel": sum(len(r.stumpy_novel) for r in runs),
        "n_ruptures_fixed_boundaries": sum(
            len(_comparable(r.ruptures_fixed_cuts, r.bars)) for r in runs
        ),
        "n_ruptures_novel": sum(len(r.ruptures_novel) for r in runs),
        "total_real_incumbent_flips": sum(len(r.real_flips) for r in runs),
    }


def _parity_summary(runs: Sequence[SymbolRun]) -> dict[str, Any]:
    """Finding 3: the numerical parity `tests/test_regime.py`'s 21 property tests never ran."""
    windows = sum(r.bars - WINDOW + 1 for r in runs)
    offsets = [o for r in runs for o in _agreement(r.argus_cuts, r.stumpy_cuts)]
    return {
        "symbols": len(runs),
        "windows_compared": windows,
        "symbols_with_exact_profile_index_match": sum(
            1 for r in runs if r.profile_index_agreement == 1.0
        ),
        "min_profile_index_agreement": min(
            (r.profile_index_agreement for r in runs), default=None
        ),
        "cac_correlation_min": min((r.cac_correlation for r in runs), default=None),
        "cac_correlation_max": max((r.cac_correlation for r in runs), default=None),
        "boundary_offsets": offsets,
        "boundaries_within_1_bar": sum(1 for o in offsets if o <= 1),
        "boundaries_within_one_window": sum(1 for o in offsets if o <= WINDOW),
        "n_boundaries": len(offsets),
    }


def _proxy_summary(runs: Sequence[SymbolRun]) -> dict[str, Any]:
    """Finding 1: `desk/regime.py`'s reimplementation of the incumbent is not the incumbent."""
    real = sum(len(r.real_flips) for r in runs)
    matched = sum(r.real_flips_matched_by_proxy for r in runs)
    return {
        "real_incumbent_flips": real,
        "regime_py_proxy_flips": sum(len(r.proxy_flips) for r in runs),
        "real_flips_matched_by_proxy_within_2_bars": matched,
        "match_rate": matched / real if real else None,
        "real_statistic": "sample standard deviation of bar returns (track1_suite.py:51-63)",
        "proxy_statistic": "median absolute bar-to-bar move (regime.py:306-312)",
    }


def _calendar_control(runs: Sequence[SymbolRun]) -> dict[str, Any]:
    """Finding 6/7's control: are a method's boundaries just the weekly session cycle?

    rTokens track US equities, so hourly bars carry a hard weekly structure (closes, weekends).
    Bucketing every boundary by hour-of-week and reporting the fullest bucket separates "found a
    regime" from "found Friday".

    **The fullest bucket's raw SHARE is not comparable between methods and is not used for the
    comparison.** PELT-at-BIC drops 552 boundaries into 168 buckets and its fullest holds 38 — a
    6.9% share. The budgeted methods drop 24 into 168 and their fullest holds 4 — a 16.7% share,
    which naively reads as "more clustered" when it is the opposite: under a uniform null, 552
    draws expect 3.3 per bucket and 24 expect 0.14, so 38 and 4 are not the same kind of number.
    The first version of this comparison asserted `bic_share > fixed_share` and was simply wrong.
    What is reported instead is `concentration_z`, the fullest bucket's excess over its own
    uniform expectation in units of that expectation's standard deviation, which is comparable
    across counts. With 24 draws the normal approximation behind it is coarse, so it is published
    as a number to read rather than as a threshold to pass.
    """
    def buckets(get: str) -> dict[str, Any]:
        hours: list[int] = []
        for run in runs:
            cuts: Sequence[int] = getattr(run, get)
            for cut in cuts:
                # Bar index is hours since the series start; the series is contiguous hourly, so
                # index mod 168 is a stable hour-of-week bucket without re-reading timestamps.
                hours.append(cut % HOURS_PER_WEEK)
        counts = Counter(hours)
        top = counts.most_common(1)
        n = len(hours)
        fullest = top[0][1] if top else 0
        expected = n / HOURS_PER_WEEK
        sd = (expected * (1.0 - 1.0 / HOURS_PER_WEEK)) ** 0.5
        return {
            "n_boundaries": n,
            "distinct_hour_of_week_buckets": len(counts),
            "fullest_bucket": fullest,
            "expected_per_bucket_if_uniform": expected,
            "concentration_z": (fullest - expected) / sd if sd > 0 else None,
        }

    return {
        "argus": buckets("argus_cuts"),
        "stumpy": buckets("stumpy_cuts"),
        "ruptures_fixed": buckets("ruptures_fixed_cuts"),
        "ruptures_bic": buckets("ruptures_bic_cuts"),
    }


def run_family_coherence(
    runs: Sequence[SymbolRun], series: dict[str, list[tuple[datetime, float]]],
) -> dict[str, Any]:
    """Finding 7. The only ground-truth-free referee available here.

    QQQ/TQQQ/SQQQ are one underlying at 1x/3x/-3x, so a real boundary is the same hour in all
    three and the spread of a method's k-th boundary across the family is pure error. The measured
    correlations and betas are reported alongside, because the test is only valid if the three
    really are the same series — that is checked here, not assumed from their tickers.
    """
    by_symbol = {r.symbol: r for r in runs}
    present = [s for s in FAMILY if s in by_symbol and s in series]
    if len(present) < 2:
        return {"available": False, "symbols": present}

    base = FAMILY[0]
    relations: dict[str, dict[str, float]] = {}
    if base in series:
        ref = np.diff(np.log(np.asarray([p for _, p in series[base]], dtype=np.float64)))
        for symbol in present:
            other = np.diff(np.log(np.asarray([p for _, p in series[symbol]], dtype=np.float64)))
            if len(other) != len(ref):
                continue
            relations[symbol] = {
                "log_return_correlation_vs_" + base: float(np.corrcoef(ref, other)[0, 1]),
                "beta_vs_" + base: float(np.polyfit(ref, other, 1)[0]),
            }

    def spread(attr: str, members: Sequence[str]) -> list[int] | None:
        sets = [sorted(getattr(by_symbol[s], attr)) for s in members]
        k = min((len(x) for x in sets), default=0)
        if k == 0:
            return None
        return [max(x[i] for x in sets) - min(x[i] for x in sets) for i in range(k)]

    same_sign = [s for s in ("QQQUSDT", "TQQQUSDT") if s in by_symbol]
    return {
        "available": True,
        "symbols": present,
        "relations": relations,
        "all_three": {
            attr: spread(attr, present)
            for attr in ("argus_cuts", "stumpy_cuts", "ruptures_fixed_cuts")
        },
        "same_sign_pair_only": {
            attr: spread(attr, same_sign)
            for attr in ("argus_cuts", "stumpy_cuts", "ruptures_fixed_cuts")
        } if len(same_sign) == 2 else None,
    }


_TIMING_FIELDS = frozenset({"argus_seconds", "stumpy_seconds", "ruptures_seconds"})


def _without_timings(report: dict[str, Any]) -> str:
    """The report with every wall-clock field removed, serialised for comparison.

    Only the top-level `per_symbol` rows carry timings, so this strips exactly those keys rather
    than walking the whole structure looking for anything that smells like a duration — a blanket
    rule would silently drop a future field whose name happened to end in `_seconds` and whose
    value was a real result.
    """
    stripped = dict(report)
    stripped["per_symbol"] = [
        {k: v for k, v in row.items() if k not in _TIMING_FIELDS}
        for row in report.get("per_symbol", [])
    ]
    return json.dumps(stripped, sort_keys=True, default=str)


def _compute_base_case(
    series: dict[str, list[tuple[datetime, float]]],
) -> dict[str, Any]:
    """Pure computation on already-fetched real series — no network, so it can be run twice for
    the reproducibility check without asking whether real time stood still in between."""
    runs = [run_symbol(symbol, rows) for symbol, rows in sorted(series.items())]
    return {
        "per_symbol": [r.as_dict() for r in runs],
        "profile_parity": _parity_summary(runs),
        "novelty_vs_incumbent": _novelty_summary(runs),
        "incumbent_proxy_defect": _proxy_summary(runs),
        "calendar_control": _calendar_control(runs),
        "family_coherence": run_family_coherence(runs, series),
        "ruptures_bic_cuts_per_symbol": [len(r.ruptures_bic_cuts) for r in runs],
    }


def run_base_case(symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = DAYS) -> dict[str, Any]:
    series, failures = fetch_universe(symbols, days=days)
    return {"days": days, "window": WINDOW, "failures": failures, **_compute_base_case(series)}


# ---------------------------------------------------------------------------------------------
# Ablation, adversarial, OOS, cost, reproducibility
# ---------------------------------------------------------------------------------------------

def ablate_idealised_curve(index: Sequence[int], *, window: int = WINDOW) -> dict[str, Any]:
    """Finding 4: isolate the ONE documented departure from stumpy.

    The SAME ARGUS nearest-neighbour index goes through the real `stumpy.floss._cac` twice — once
    with stumpy's own fitted-beta IAC and once with ARGUS's analytic parabola passed as
    `custom_iac`. Everything else (arc counting, the clip to 1, the head/tail pin, `_rea`) is
    stumpy's own code in both arms, so any difference in the boundaries is attributable to the
    idealised curve alone and to nothing else.
    """
    profile = np.asarray(index, dtype=np.int64)
    parabola = np.asarray(
        [idealised_arc(len(index), i) for i in range(len(index))], dtype=np.float64
    )
    beta_curve = _cac(profile.copy(), window, excl_factor=EXCLUSION_FACTOR)
    parabola_curve = _cac(
        profile.copy(), window, excl_factor=EXCLUSION_FACTOR, custom_iac=parabola.copy()
    )
    beta_cuts = sorted(
        int(v) for v in _rea(beta_curve, REGIMES, window, excl_factor=EXCLUSION_FACTOR)
    )
    parabola_cuts = sorted(
        int(v) for v in _rea(parabola_curve, REGIMES, window, excl_factor=EXCLUSION_FACTOR)
    )
    return {
        "beta_iac_cuts": beta_cuts,
        "parabola_iac_cuts": parabola_cuts,
        "offsets": _agreement(beta_cuts, parabola_cuts),
        "curve_correlation": float(
            np.corrcoef(
                np.nan_to_num(beta_curve, nan=1.0), np.nan_to_num(parabola_curve, nan=1.0)
            )[0, 1]
        ),
    }


def run_ablation(symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = DAYS) -> dict[str, Any]:
    series, failures = fetch_universe(symbols, days=days)
    per_symbol: dict[str, dict[str, Any]] = {}
    for symbol, rows in sorted(series.items()):
        index, _ = argus_profile([price for _, price in rows])
        per_symbol[symbol] = ablate_idealised_curve(index)
    offsets = [o for v in per_symbol.values() for o in v["offsets"]]
    return {
        "days": days,
        "failures": failures,
        "per_symbol": per_symbol,
        "n_boundaries": len(offsets),
        "unchanged_within_1_bar": sum(1 for o in offsets if o <= 1),
        "moved_more_than_300_bars": sum(1 for o in offsets if o > 300),
        "idealised_curve_is_load_bearing": any(o > WINDOW for o in offsets),
    }


def run_adversarial() -> dict[str, Any]:
    """Finding 5, plus the degenerate inputs both sides claim to handle. Constructed, then RUN.

    Two degenerate inputs, and they do NOT give the same answer — which is why both are here.

    **The flat corrected arc curve is the one ARGUS wins.** Handed a curve with no dip in it, the
    real `stumpy.floss._rea` returns index 0 twice: two boundaries, the same index, inside the
    head zone its own `excl_factor` had just pinned to 1.0. ARGUS's `find_boundaries` returns
    nothing, because of its explicit `if working[best] >= 1.0: break`.

    **The constant PRICE series is one all three lose, including ARGUS, and this module says so
    rather than stopping at the win above.** Every z-normalised window of a flat series is zeros,
    so every pairwise correlation is exactly 0 and the nearest neighbour is decided entirely by
    tie-breaking — ARGUS takes the first candidate that beats its `-2.0` sentinel, stumpy's
    vectorised scan takes a different one (8 distinct profile entries against stumpy's 570 on the
    same input). Both then find structure in that tie-breaking artefact: ARGUS reports two
    boundaries, stumpy reports two others, and `ruptures.KernelCPD` reports two more. ARGUS's arc
    curve at least stays shallow (minimum ~0.63 against stumpy's ~0.03, i.e. it is much less
    confident in its own fabrication) but shallow-and-wrong is still wrong. **A constant-input
    guard is a real, open defect in `desk/regime.py`, found by this comparison and not fixed by
    it** — fixing the module under test from inside its own comparison would make the comparison
    unfalsifiable.
    """
    flat = [1.0] * 500
    stumpy_flat = [
        int(v)
        for v in _rea(
            np.ones(500, dtype=np.float64), REGIMES, WINDOW, excl_factor=EXCLUSION_FACTOR
        )
    ]
    argus_flat = find_boundaries(flat, count=REGIMES - 1, window=WINDOW)

    constant = [100.0] * 600
    constant_error: str | None = None
    constant_cuts: list[int] = []
    constant_curve_min: float | None = None
    try:
        constant_index, _ = argus_profile(constant)
        constant_cuts, curve = argus_boundaries(constant_index)
        constant_curve_min = min(curve)
    except Exception as exc:
        constant_error = type(exc).__name__

    constant_stumpy_cuts: list[int] = []
    constant_stumpy_min: float | None = None
    try:
        _, stumpy_curve, constant_stumpy_cuts, _ = stumpy_run(constant)
        constant_stumpy_min = min(stumpy_curve)
    except Exception as exc:
        constant_stumpy_cuts = []
        constant_stumpy_min = None
        constant_error = constant_error or f"stumpy:{type(exc).__name__}"

    try:
        constant_ruptures_cuts = ruptures_fixed(constant)
    except Exception:
        constant_ruptures_cuts = []

    try:
        argus_profile([100.0, 101.0, 102.0])
        short_error = None
    except Exception as exc:
        short_error = type(exc).__name__

    return {
        "flat_curve_stumpy_rea_boundaries": stumpy_flat,
        "flat_curve_argus_boundaries": argus_flat,
        "stumpy_fabricates_a_boundary_on_a_flat_curve": bool(stumpy_flat),
        "stumpy_repeats_the_same_index": len(stumpy_flat) > len(set(stumpy_flat)),
        "argus_refuses_a_flat_curve": argus_flat == [],
        "constant_series_argus_boundaries": constant_cuts,
        "constant_series_stumpy_boundaries": constant_stumpy_cuts,
        "constant_series_ruptures_boundaries": constant_ruptures_cuts,
        "constant_series_argus_curve_min": constant_curve_min,
        "constant_series_stumpy_curve_min": constant_stumpy_min,
        # Not an ARGUS win and not scored as one: every method invents a boundary where no series
        # structure exists at all. Recorded as an open defect in all three.
        "all_three_fabricate_on_a_constant_series": bool(
            constant_cuts and constant_stumpy_cuts and constant_ruptures_cuts
        ),
        "argus_is_less_confident_in_its_fabrication": (
            None
            if constant_curve_min is None or constant_stumpy_min is None
            else constant_curve_min > constant_stumpy_min
        ),
        "constant_series_argus_error": constant_error,
        "too_short_series_argus_error": short_error,
    }


SIGNIFICANCE = 0.05


def _no_significant_edge(novelty: dict[str, Any]) -> bool:
    """True when FLUSS's novelty rate is not distinguishable from the null at 5%.

    A missing p-value (too few boundaries, or a degenerate null) counts as no demonstrated edge,
    never as one: this project's rule is that an unproven capability is not an owned one.
    """
    p = novelty.get("binomial_p_novelty_above_chance")
    return p is None or p > SIGNIFICANCE


def run_oos_check(symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = DAYS) -> dict[str, Any]:
    """Chronological midpoint split, never random: does the verdict hold in both halves?

    A random split would put the same regime on both sides and guarantee agreement. Each half must
    still clear the incumbent's own 240-bar warmup plus FLUSS's pinned head and tail, which is why
    the base window is 60 days rather than shorter — half of 1,439 bars is 719, and the rule has no
    opinion until bar 241 of that half.
    """
    series, failures = fetch_universe(symbols, days=days)
    first: dict[str, list[tuple[datetime, float]]] = {}
    second: dict[str, list[tuple[datetime, float]]] = {}
    skipped: dict[str, str] = {}
    floor = INCUMBENT_SLOW + WINDOW * EXCLUSION_FACTOR * 2
    for symbol, rows in series.items():
        mid = len(rows) // 2
        if mid < floor or len(rows) - mid < floor:
            skipped[symbol] = "half is shorter than the incumbent warmup plus the pinned ends"
            continue
        first[symbol] = rows[:mid]
        second[symbol] = rows[mid:]
    first_half = _compute_base_case(first)
    second_half = _compute_base_case(second)
    return {
        "days": days,
        "failures": failures,
        "skipped": skipped,
        "first_half": {
            "novelty": first_half["novelty_vs_incumbent"],
            "parity": first_half["profile_parity"],
        },
        "second_half": {
            "novelty": second_half["novelty_vs_incumbent"],
            "parity": second_half["profile_parity"],
        },
        # The verdict under test is "FLUSS shows no measurable edge over the incumbent". It holds
        # out of sample when NEITHER half produces novelty significantly above the null — the flag
        # alone is too brittle at 11 or 12 boundaries per half, where one boundary moving 25 bars
        # swings the rate by 8 points. Significance is the honest reading of that sample size.
        "verdict_holds_in_both_halves": bool(
            _no_significant_edge(first_half["novelty_vs_incumbent"])
            and _no_significant_edge(second_half["novelty_vs_incumbent"])
        ),
    }


def measure_costs(
    symbol: str = "NVDAUSDT", *, days: int = DAYS, repeats: int = 3,
) -> dict[str, Any]:
    """Real wall clock, all three, same series. stumpy is numba-JIT'd, so it is warmed on a throw-
    away series first — timing a compile instead of a computation would flatter ARGUS by ~30s."""
    rows = fetch_series(symbol, days=days)
    values = [price for _, price in rows]
    stumpy.stump(np.arange(300, dtype=np.float64) ** 1.01, m=WINDOW)

    argus_total = 0.0
    stumpy_total = 0.0
    ruptures_total = 0.0
    for _ in range(repeats):
        argus_total += argus_profile(values)[1]
        stumpy_total += stumpy_run(values)[3]
        ruptures_total += ruptures_bic(values)[1]
    return {
        "symbol": symbol,
        "bars": len(values),
        "repeats": repeats,
        "argus_seconds": argus_total / repeats,
        "stumpy_seconds_warm": stumpy_total / repeats,
        "ruptures_pelt_seconds": ruptures_total / repeats,
        "stumpy_speedup": (argus_total / stumpy_total) if stumpy_total > 0 else None,
    }


def run_reproducibility_check(
    symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = DAYS,
) -> dict[str, Any]:
    """Fetch once, compute twice. Re-fetching would test whether the venue moved, not determinism.

    `ruptures.KernelCPD` and `stumpy.fluss` both have stochastic-looking internals (`_iac` samples
    a beta fit; the kernel solver is iterative) so this is a real question about the baselines, not
    only about ARGUS's own arithmetic.

    **Wall-clock fields are stripped before comparing, and that is not a convenience.** The first
    version of this function compared the whole report and reported `identical: False` — not
    because any boundary moved but because `argus_seconds` differed in the fourth decimal between
    two passes over identical data. A reproducibility check that fails on its own stopwatch tests
    the machine's scheduler, not the algorithm, and would have buried a real determinism finding
    under noise.
    """
    series, _ = fetch_universe(symbols, days=days)
    first = _compute_base_case(series)
    second = _compute_base_case(series)
    return {
        "identical": _without_timings(first) == _without_timings(second),
        "timing_fields_excluded": sorted(_TIMING_FIELDS),
    }


SCOPE_STATEMENT = (
    "ARGUS's own offline FLUSS (desk/regime.py), the real installed stumpy (stump + fluss, its own "
    "public API), the real installed ruptures (Pelt with a Gaussian mean-and-variance cost at the "
    "BIC penalty 2*log(n), and KernelCPD with an rbf kernel budgeted to the same boundary count), "
    "and the REAL incumbent two-line rule (strategies/track1_suite.py's own "
    "rotation_regime_switch, driven over real Bar objects) "
    "are all run on the SAME real 60-day hourly Bitget MARKET series "
    "for all 12 rTokens. "
    "CLAIMED, and this is the decisive finding: the capability LOSES. (1) FLUSS places 23 "
    "boundaries, 6 of them inside the incumbent's own 240-bar warmup where that rule cannot flip "
    "at all; of the 17 genuinely comparable boundaries, 3 (17.6%) have no real incumbent flip "
    "within 24 bars. The incumbent flips 400 times across the 12 symbols, so its +/-24-bar "
    "neighbourhoods already cover 66.9% of the comparable region and a boundary at a uniformly "
    "random comparable bar would look novel 33.1% of the time. FLUSS's 17.6% is WELL BELOW that; "
    "the one-sided binomial p for novelty above the null is 0.954. There is no evidence FLUSS "
    "finds "
    "regime boundaries the two-line volatility rule misses -- it finds fewer than chance would. "
    "NOT hidden: the unrestricted measurement is 8 of 23 (34.8%) against a 40.5% null, the same "
    "verdict and a weaker one, and both are published. "
    "(2) data/regimes.json's "
    "threshold_agrees flag was computed against a proxy that is not the incumbent: "
    "track1_suite.py:51-63 uses a sample standard deviation of bar returns, regime.py:306-312 uses "
    "a median absolute bar-to-bar move; the proxy fires 559 times to the real rule's 400 and "
    "reproduces only 17.8% of the real flips within 2 bars. (3) ARGUS's matrix profile index is "
    "EXACTLY stumpy's on all 12 symbols (16,992 windows, zero disagreements) while taking 5.25s "
    "per 1,439-bar symbol against warm stumpy's 0.017s, a measured 310x. "
    "(4) On the QQQ/TQQQ/SQQQ family -- one "
    "underlying at 1x/3x/-3x, verified here at log-return correlation 0.983/-0.975 and beta "
    "2.94/-2.93 -- ruptures' boundaries spread 0 and 1 bars across the three while ARGUS's and "
    "stumpy's spread 50 and 122, and 50/122 persists on the same-sign QQQ-vs-TQQQ pair alone, "
    "which z-normalisation IS invariant to. "
    "NOT claimed that ruptures is better at regime detection in general: PELT at the "
    "BIC penalty returns 44-50 boundaries per symbol that pile onto the weekly market-hours "
    "calendar (one hour-of-week bucket holds 38 of 552 against a uniform expectation of 3.3), and "
    "this module reports that as unusable rather "
    "than scoring it. NOT claimed the family-coherence test settles correctness -- it is a "
    "necessary condition with no ground truth behind it, and the calendar control beside it does "
    "NOT clear ruptures: its budgeted boundaries are MORE hour-of-week-clustered than ARGUS's "
    "(z=10.2 against 5.1) and all four in its fullest bucket are the QQQ family itself, so the "
    "control is published as a number to read rather than as a verdict. "
    "NOT hidden either: scored against the SAME incumbent on the same tolerance, ruptures' own "
    "boundaries are novel 7 of 18 times (38.9%) and stumpy's 3 of 19 (15.8%) -- ruptures is the "
    "only one of the three sitting above its own null, though not significantly at this sample "
    "size, and that cuts against ARGUS rather than for it. "
    "NOT claimed ARGUS loses everywhere: fed a corrected "
    "arc curve with no dip, the real stumpy.floss._rea fabricates two boundaries at index 0 while "
    "ARGUS's find_boundaries correctly returns none, and that same defect is reachable on real "
    "GOOGLUSDT data. NOT claimed that guard extends to degenerate PRICE input, and this is a "
    "defect found here and left unfixed on purpose: on a constant series ARGUS fabricates two "
    "boundaries (335, 456), stumpy two others (291, 411) and ruptures two more (26, 574) -- all "
    "three read structure into pure tie-breaking, ARGUS only less confidently (arc-curve minimum "
    "0.63 against stumpy's 0.03). NOT claimed these boundary locations are permanent -- fetched "
    "live and will move with the venue."
)


def render(report: dict[str, Any]) -> str:
    base = report["base_case"]
    novelty = base["novelty_vs_incumbent"]
    parity = base["profile_parity"]
    proxy = base["incumbent_proxy_defect"]
    family = base["family_coherence"]
    lines = ["REGIME DETECTION vs stumpy FLOSS, ruptures, and the two-line incumbent\n"]
    lines.append(
        f"  THE OWED NUMBER: {novelty['n_novel_vs_real_incumbent']}/"
        f"{novelty['n_argus_boundaries']} comparable FLUSS boundaries "
        f"({novelty['novelty_rate']:.1%}) have no real incumbent flip within "
        f"{TOLERANCE_BARS} bars "
        f"({novelty['n_argus_boundaries_outside_region']} of "
        f"{novelty['n_argus_boundaries_all']} excluded: outside the comparable region)"
    )
    lines.append(
        f"  chance novelty for a random bar: {novelty['chance_novelty_rate']:.1%} "
        f"(the rule flips {novelty['total_real_incumbent_flips']} times) "
        f"-> beats chance: {novelty['beats_chance']}, "
        f"binomial p (novelty above null)={novelty['binomial_p_novelty_above_chance']}, "
        f"p (redundant with the incumbent)={novelty['binomial_p_agreement_above_chance']}"
    )
    lines.append(
        f"\n  PROXY DEFECT: regime.py's reimplementation fires {proxy['regime_py_proxy_flips']} "
        f"times to the real rule's {proxy['real_incumbent_flips']}, matching "
        f"{proxy['match_rate']:.1%} of real flips within 2 bars"
    )
    lines.append(
        f"\n  PARITY with stumpy: {parity['symbols_with_exact_profile_index_match']}/"
        f"{parity['symbols']} symbols exact on the profile index "
        f"({parity['windows_compared']} windows); boundaries within 1 bar "
        f"{parity['boundaries_within_1_bar']}/{parity['n_boundaries']}, within one window "
        f"{parity['boundaries_within_one_window']}/{parity['n_boundaries']}"
    )
    if family.get("available"):
        lines.append(
            f"\n  QQQ-FAMILY SPREAD (bars, lower is better): argus "
            f"{family['all_three']['argus_cuts']}, stumpy {family['all_three']['stumpy_cuts']}, "
            f"ruptures {family['all_three']['ruptures_fixed_cuts']}"
        )
    adversarial = report["adversarial"]
    lines.append(
        f"\n  FLAT CURVE: stumpy._rea returns "
        f"{adversarial['flat_curve_stumpy_rea_boundaries']}, ARGUS returns "
        f"{adversarial['flat_curve_argus_boundaries']}"
    )
    costs = report["costs"]
    lines.append(
        f"\n  COST on {costs['bars']} bars: argus {costs['argus_seconds']:.2f}s, "
        f"stumpy {costs['stumpy_seconds_warm']:.3f}s (warm), "
        f"ruptures {costs['ruptures_pelt_seconds']:.3f}s"
    )
    ablation = report["ablation"]
    lines.append(
        f"  ABLATION (parabola IAC vs stumpy's beta IAC, same arc counts): "
        f"{ablation['unchanged_within_1_bar']}/{ablation['n_boundaries']} unchanged, "
        f"load-bearing={ablation['idealised_curve_is_load_bearing']}"
    )
    holds = report["oos_check"]["verdict_holds_in_both_halves"]
    lines.append(f"  out-of-sample verdict holds: {holds}")
    lines.append(f"  reproducible: {report['reproducibility']['identical']}")
    lines.append(f"\n  WHO WINS: {report['who_wins']}")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base = run_base_case()
    novelty = base["novelty_vs_incumbent"]
    costs = measure_costs()
    # The speedup in the verdict is read from the run that just happened, never written in as a
    # remembered constant — a stale multiplier in a sentence claiming to report a measurement is
    # exactly the kind of unevidenced number this module was built to catch.
    speedup = costs["stumpy_speedup"]
    report: dict[str, Any] = {
        "base_case": base,
        "ablation": run_ablation(),
        "adversarial": run_adversarial(),
        "oos_check": run_oos_check(),
        "costs": costs,
        "reproducibility": run_reproducibility_check(),
        # A demotion needs no significance test — "we could not show an edge" is the default and
        # the honest one. Promotion does: FLUSS only wins here if its novelty clears the null at
        # 5%, not merely by sitting a point or two above a null computed from 23 boundaries.
        "who_wins": (
            "argus — FLUSS finds boundaries the incumbent misses at better than chance "
            f"(p={novelty['binomial_p_novelty_above_chance']})"
            if not _no_significant_edge(novelty)
            else "baseline — ruptures is more coherent on the one ground-truth-free test, stumpy "
            f"is bit-identical and {speedup:.0f}x faster, and FLUSS shows no measurable edge "
            "over the two-line incumbent"
        ),
        "scope_statement": SCOPE_STATEMENT,
    }
    print(render(report))
    out = Path(__file__).resolve().parents[3] / "data" / "regime_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BIC_PARAMETERS_PER_SEGMENT",
    "COMPARABLE_FROM",
    "DAYS",
    "FAMILY",
    "HOURS_PER_WEEK",
    "INCUMBENT_SLOW",
    "REGIMES",
    "SCOPE_STATEMENT",
    "SIGNIFICANCE",
    "TOLERANCE_BARS",
    "WINDOW",
    "SymbolRun",
    "ablate_idealised_curve",
    "argus_boundaries",
    "argus_profile",
    "comparable_region",
    "fetch_series",
    "fetch_universe",
    "incumbent_flips",
    "main",
    "measure_costs",
    "render",
    "run_ablation",
    "run_adversarial",
    "run_base_case",
    "run_family_coherence",
    "run_oos_check",
    "run_reproducibility_check",
    "run_symbol",
    "ruptures_bic",
    "ruptures_fixed",
    "stumpy_run",
]
