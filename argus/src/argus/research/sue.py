"""Standardized Unexpected Earnings (SUE) — cross-sectional earnings surprise, refusing rather
than silently going infinite.

`agents/earnings.py`'s seven-way decomposition scores one print's surprise qualitatively, in
[-1, 1], from an Earnings Analyst's judgment — it has no statistically-standardized number
anywhere, and no cross-sectional ranking across a universe. SUE is the field's own answer to
"how surprising, compared to what": a quarter's EPS change from a year ago, standardized by the
volatility of that same company's own historical year-over-year EPS changes. It is the textbook
post-earnings-announcement-drift (PEAD) proxy this project's real, published reference implements.

**Read before written.** The reference is QuantConnect's own tutorial library (Apache-2.0),
`research/repos-t3/Tutorials/04 Strategy Library/355 Standardized Unexpected Earnings/03 Method.
html` (the real `FineSelectionAndSueSorting` method, vendored verbatim at
`eval/baselines/quantconnect_sue.py`): ``SUE = (EPS_q - EPS_q-4) / stdev(eight historical
EPS_q - EPS_q-4 deltas)``. :func:`sue_from_quarters` reproduces that exact formula in plain Python
(matching `population` standard deviation — `numpy.std`'s default `ddof=0` — via
`statistics.pstdev`, not the sample `stdev`, which would silently disagree in the last few
digits). Verified to reproduce the real vendored function's real output to floating-point
identity on real quarterly EPS pulled live from SEC EDGAR (`market/fundamentals.py`, already
built, already point-in-time and restatement aware) for all nine real rToken anchor companies —
`eval/earnings_comparison.py`.

**What this module adds that the reference does not.** The real vendored formula's denominator —
the standard deviation of eight real deltas — has no guard anywhere in the real source. Run
directly (`eval/earnings_comparison.py`), a constructed but entirely plausible smoothly-growing
EPS path (a constant year-over-year delta every quarter — zero variance by construction) produces
a real, silent ``inf`` from the real code, with nothing louder than a `RuntimeWarning:
divide by zero` that a real `sorted()` cross-sectional ranking would never see — an `inf` SUE
sorts first, unconditionally, ahead of every genuine surprise in the universe. An all-flat EPS
history (real for a pre-revenue or loss-making issuer reporting $0.00 every quarter) produces a
real, silent ``nan`` instead, which corrupts `sorted()`'s ordering guarantee outright (Python's
sort treats NaN comparisons as always-false). :func:`sue_from_quarters` raises
:class:`SueError` in both cases instead of returning a number.

**Two corrections taken from general-purpose tools (2026-09-25, `eval/general_sue_comparison.py`).**

1. *Year-over-year means a year, not four list positions.* The positional formula assumes the
   list holds consecutive quarters. SEC XBRL does not: a fiscal fourth quarter is reported only
   inside the 10-K's annual figure, so a company's quarterly EPS facts run Q3, Q2, Q1, Q3, Q2, …
   and ``quarters[i + 4]`` lands five fiscal quarters back — a different quarter of a different
   year. Measured on 3,997 real SEC filers with twelve or more quarterly EPS frames, 3,587 had
   *none* of the eight positional deltas year-over-year and only 159 had all eight. pandas gets
   this right by construction: a ``PeriodIndex`` shift pairs periods by calendar, not by position,
   and yields NaN where the partner quarter was never reported. :func:`read_dated` adopts that
   principle — each quarter is paired with the one whose period ended a year earlier
   (:data:`YOY_DAY_RANGE`, by day distance rather than pandas' calendar-quarter bucket, so a
   52/53-week fiscal calendar whose quarter ends drift across a calendar boundary still pairs) —
   and refuses, with the reason, where no such partner exists.

2. *"Zero" variance is relative to the numbers being subtracted.* The first guard here was an
   absolute ``eps_std <= 1e-9``. scikit-learn's ``_is_constant_feature``
   (`sklearn/preprocessing/_data.py:83-98`, BSD-3-Clause, after Chan, Golub & LeVeque) and SciPy's
   ``zmap`` (`scipy/stats/_stats_py.py:2936-2940`, BSD-3-Clause, ``std <= |eps * mean|``) both
   express the threshold in machine epsilon relative to the data's own magnitude, which makes the
   refusal decision invariant to units — the absolute threshold was not (the same real NVDA
   history re-expressed in units 1e-9 smaller was refused as degenerate). Their reference
   magnitude was *not* taken: both scale by the deltas' own mean/variance, and a constant
   year-over-year delta on a high EPS level (a $10 EPS growing $0.10 a year) leaves rounding noise
   proportional to the *level*, not the delta, which both libraries were measured to let through
   as a finite SUE of order 1e13. The noise enters in the subtraction ``EPS_q - EPS_q-4``
   (`statistics.pstdev` itself is exact over rationals), so :func:`noise_floor` bounds it by the
   largest operand: each float delta sits within ``2 * eps * max|EPS|`` of its exact decimal
   value, and :data:`NOISE_ULPS` leaves an eight-fold margin over that bound.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from statistics import pstdev
from typing import Any

LAG_QUARTERS = 4
"""SUE compares a quarter's EPS to the same quarter one year (four quarters) earlier."""

N_DELTAS = 8
"""The real reference's own choice: eight historical year-over-year deltas set the volatility
denominator (`quantconnect_sue.py`'s real `new_eps_list`/`old_eps_list` slicing, both length 8)."""

MIN_QUARTERS = LAG_QUARTERS + N_DELTAS
"""12 — the earliest delta needs a value four quarters before the eighth-oldest delta point."""

NOISE_ULPS = 16
"""The refusal threshold, in machine epsilons of the largest EPS the deltas were computed from.

Derivation (see the module docstring, correction 2): EPS arrives as decimals correctly rounded to
binary, so each operand carries at most half an epsilon of relative error and the subtraction
adds at most half an epsilon more; a delta whose exact value is ``c`` therefore lands within
``2 * eps * M`` of ``c`` (``M`` the largest operand magnitude), and eight such deltas cannot have a
population standard deviation above that. 16 is that bound with an eight-fold margin for values
that passed through a few float operations upstream (a unit conversion, say).

A genuine history cannot get near it: EPS is reported to at most four decimals, and the smallest
non-zero spread eight deltas on a 0.0001 grid can have is about 3.3e-5 — a false refusal would
need an EPS level above ~9e9 per share. The largest diluted EPS in the SEC's own quarterly frames
(2022-2026, 6,729 filers) is 6.1e6.

This replaced an absolute ``ZERO_VARIANCE_TOLERANCE = 1e-9``, which was a unit choice disguised as
a numerical one — measured in `eval/general_sue_comparison.py`'s unit sweep."""

YOY_DAY_RANGE: tuple[int, int] = (350, 380)
"""Two quarterly periods are the same quarter a year apart when their end dates are this many days
apart. Calendar quarters sit 365-366 days apart and 52/53-week fiscal quarters 364 or 371; the
nearest *wrong* pairing (the adjacent quarter) sits about 91 days away on either side, so any
window strictly between 320 and 410 classifies identically — this one is tight enough to exclude
a fiscal-year change that shifted a quarter by a month."""

MAX_DELTA_SPAN_DAYS = 1096
"""The eight year-over-year changes must all belong to quarters that ended within three years of
the newest one. With every fiscal Q4 missing (the SEC XBRL norm, see correction 1) eight changes
span eleven fiscal quarters, about 830 days; three years allows one further missing quarter and no
more, so a filer with gap years cannot get a "trailing" volatility reaching back a decade."""


MIN_QUARTER_GAP_DAYS = 60
"""Two distinct quarters end at least ~84 days apart (a 12-week fiscal quarter). Two quarterly EPS
values whose periods end closer than this overlap: one quarter reported twice (NVDA's real EDGAR
history carries both a 2010-07-31 and a 2010-08-01 quarter) or a fiscal-year change (KalVista's
2025-01-31 and 2025-03-31 quarters share January). SUE refuses a window that would count
overlapping periods as two quarters rather than guessing which to keep."""


class SueError(ValueError):
    """Raised rather than returning a SUE computed from insufficient or degenerate history."""


@dataclass(frozen=True, slots=True)
class SueRead:
    """One symbol's SUE, with the inputs that produced it — never just the scalar."""

    symbol: str
    sue: float
    eps_change: float
    eps_std: float
    quarters_used: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sue": round(self.sue, 6),
            "eps_change": round(self.eps_change, 6),
            "eps_std": round(self.eps_std, 6),
            "quarters_used": self.quarters_used,
        }


@dataclass(frozen=True, slots=True)
class YoyDelta:
    """One genuine year-over-year EPS change: the quarter ending ``end`` minus the one ending
    ``prior_end``, a year earlier."""

    end: date
    prior_end: date
    eps: float
    prior_eps: float

    @property
    def delta(self) -> float:
        return self.eps - self.prior_eps


def noise_floor(magnitude: float) -> float:
    """The largest standard deviation float rounding alone can give eight deltas computed from
    operands no larger than ``magnitude`` in absolute value — see :data:`NOISE_ULPS`."""
    return NOISE_ULPS * sys.float_info.epsilon * abs(magnitude)


def standardise(deltas: Sequence[float], magnitude: float) -> tuple[float, float, float]:
    """``(sue, eps_change, eps_std)`` from year-over-year deltas, newest first, computed from EPS
    no larger than ``magnitude`` in absolute value. Raises :class:`SueError` — never returns
    ``inf``/``nan`` — when their spread is indistinguishable from rounding noise."""
    eps_change = deltas[0]
    eps_std = pstdev(deltas)
    if eps_std <= noise_floor(magnitude):
        raise SueError(
            "the eight real year-over-year EPS deltas have zero variance (or only float rounding "
            f"noise: {eps_std:.3g}, at or below {noise_floor(magnitude):.3g} for EPS of magnitude "
            f"{abs(magnitude):.4g}); SUE is undefined, not infinite — the real reference silently "
            "returns inf/nan here instead of refusing"
        )
    return eps_change / eps_std, eps_change, eps_std


def _compute(quarters: Sequence[float]) -> tuple[float, float, float]:
    """Returns ``(sue, eps_change, eps_std)``. Raises :class:`SueError` — never returns
    ``inf``/``nan`` — below :data:`MIN_QUARTERS` or when the eight real year-over-year deltas
    have zero variance within :func:`noise_floor`, which is the module docstring's own measured,
    real divergence from the reference."""
    if len(quarters) < MIN_QUARTERS:
        raise SueError(
            f"{len(quarters)} quarter(s) is below the {MIN_QUARTERS} SUE needs "
            f"({LAG_QUARTERS} lag quarters plus {N_DELTAS} historical deltas)"
        )
    deltas = [quarters[i] - quarters[i + LAG_QUARTERS] for i in range(N_DELTAS)]
    magnitude = max(abs(q) for q in quarters[:MIN_QUARTERS])
    return standardise(deltas, magnitude)


def sue_from_quarters(quarters: Sequence[float]) -> float:
    """The real SUE formula (`eval/baselines/quantconnect_sue.py`), reproduced in plain Python.

    ``quarters`` must be real quarterly EPS, newest first — ``quarters[0]`` is the most recent
    quarter — and **consecutive**: ``quarters[i + 4]`` is taken to be the same quarter a year
    earlier. SEC XBRL quarterly facts are not consecutive (fiscal Q4 lives only in the 10-K), so a
    caller holding dated facts must use :func:`read_dated` instead.
    """
    sue, _eps_change, _eps_std = _compute(quarters)
    return sue


def read(symbol: str, quarters: Sequence[float]) -> SueRead:
    """Full :class:`SueRead`, with the intermediate values a caller would otherwise have to
    recompute to audit the scalar. Positional, like :func:`sue_from_quarters` — consecutive
    quarters only."""
    sue, eps_change, eps_std = _compute(quarters)
    return SueRead(
        symbol=symbol, sue=sue, eps_change=eps_change, eps_std=eps_std,
        quarters_used=len(quarters),
    )


def yoy_deltas(points: Sequence[tuple[date, float]]) -> list[YoyDelta]:
    """Every genuine year-over-year change in ``points`` (``(period_end, eps)`` pairs, any order),
    newest first. A quarter whose same-quarter-last-year value was never reported contributes
    nothing — it is not silently paired with a neighbouring quarter.

    Duplicate period ends are a caller error (`market.fundamentals.latest_per_period` resolves
    restatements before this is reached) and raise :class:`SueError` rather than being guessed
    between.
    """
    ordered = sorted(points, key=lambda p: p[0], reverse=True)
    ends = [end for end, _ in ordered]
    if len(set(ends)) != len(ends):
        raise SueError("two EPS values share one period end; resolve restatements first")
    low, high = YOY_DAY_RANGE
    out: list[YoyDelta] = []
    for i, (end, eps) in enumerate(ordered):
        for prior_end, prior_eps in ordered[i + 1:]:
            gap = (end - prior_end).days
            if gap > high:
                break
            if gap >= low:
                out.append(YoyDelta(end=end, prior_end=prior_end, eps=eps, prior_eps=prior_eps))
                break
    return out


def yoy_window(points: Sequence[tuple[date, float]]) -> list[YoyDelta]:
    """The :data:`N_DELTAS` year-over-year changes SUE is computed from, newest first — the newest
    quarter's own change first. Raises :class:`SueError`, naming what is missing, when the newest
    quarter has no year-earlier partner, when fewer than eight changes exist, or when the eight
    reach back further than :data:`MAX_DELTA_SPAN_DAYS`."""
    if not points:
        raise SueError("no quarterly EPS to compute SUE from")
    newest = max(end for end, _ in points)
    deltas = yoy_deltas(points)
    if not deltas or deltas[0].end != newest:
        raise SueError(
            f"the newest quarter (ending {newest.isoformat()}) has no same-quarter-last-year EPS "
            "to compare against, so its surprise is undefined"
        )
    if len(deltas) < N_DELTAS:
        raise SueError(
            f"only {len(deltas)} genuine year-over-year EPS change(s) in {len(points)} quarter(s); "
            f"SUE needs {N_DELTAS}"
        )
    window = deltas[:N_DELTAS]
    used = sorted({d.end for d in window} | {d.prior_end for d in window})
    for earlier, later in pairwise(used):
        if (later - earlier).days < MIN_QUARTER_GAP_DAYS:
            raise SueError(
                f"EPS is reported for quarters ending {earlier.isoformat()} and "
                f"{later.isoformat()}, only {(later - earlier).days} days apart — overlapping "
                "periods (one quarter reported twice, or a fiscal-year change); SUE refuses rather "
                "than guess which figure is the quarter"
            )
    span = (newest - window[-1].end).days
    if span > MAX_DELTA_SPAN_DAYS:
        raise SueError(
            f"the {N_DELTAS} most recent year-over-year EPS changes span {span} days, beyond the "
            f"{MAX_DELTA_SPAN_DAYS} a trailing volatility may reach back"
        )
    return window


def read_dated(symbol: str, points: Sequence[tuple[date, float]]) -> SueRead:
    """SUE from dated quarterly EPS, pairing each quarter with the same quarter a year earlier by
    date — the form SEC XBRL facts actually arrive in (see the module docstring, correction 1).

    On a history of consecutive quarters this equals :func:`read` on the same values to the last
    bit; on a history with fiscal fourth quarters missing, it is the only one of the two whose
    deltas are year-over-year at all.
    """
    window = yoy_window(points)
    magnitude = max(max(abs(d.eps), abs(d.prior_eps)) for d in window)
    sue, eps_change, eps_std = standardise([d.delta for d in window], magnitude)
    used = {d.end for d in window} | {d.prior_end for d in window}
    return SueRead(
        symbol=symbol, sue=sue, eps_change=eps_change, eps_std=eps_std, quarters_used=len(used),
    )


def _ranked(
    reader: Any, inputs: Mapping[str, Any],
) -> tuple[list[SueRead], dict[str, str]]:
    reads: list[SueRead] = []
    skipped: dict[str, str] = {}
    for symbol, history in inputs.items():
        try:
            reads.append(reader(symbol, history))
        except SueError as exc:
            skipped[symbol] = str(exc)
    return sorted(reads, key=lambda r: r.sue, reverse=True), skipped


def rank_universe(
    quarters_by_symbol: Mapping[str, Sequence[float]],
) -> tuple[list[SueRead], dict[str, str]]:
    """Every symbol with enough real history, ranked by SUE descending — the cross-sectional
    ranking `agents/earnings.py` has no equivalent of. Returns ``(ranked, skipped)`` rather than
    silently dropping symbols that cannot be scored: ``skipped`` names every excluded symbol and
    why, the same discipline `eval/rotation_comparison.py`'s own `_real_dataframe` applies to a
    real fetch that cannot cover every requested symbol. Positional — see :func:`read`."""
    return _ranked(read, quarters_by_symbol)


def rank_universe_dated(
    points_by_symbol: Mapping[str, Sequence[tuple[date, float]]],
) -> tuple[list[SueRead], dict[str, str]]:
    """:func:`rank_universe` over dated EPS, through :func:`read_dated`."""
    return _ranked(read_dated, points_by_symbol)


__all__ = [
    "LAG_QUARTERS",
    "MAX_DELTA_SPAN_DAYS",
    "MIN_QUARTERS",
    "MIN_QUARTER_GAP_DAYS",
    "NOISE_ULPS",
    "N_DELTAS",
    "YOY_DAY_RANGE",
    "SueError",
    "SueRead",
    "YoyDelta",
    "noise_floor",
    "rank_universe",
    "rank_universe_dated",
    "read",
    "read_dated",
    "standardise",
    "sue_from_quarters",
    "yoy_deltas",
    "yoy_window",
]
