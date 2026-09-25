"""General-purpose rivals for the SUE capability, run on the same real SEC EPS.

`research/sue.py` does three things that are not specific to trading at all, and each has a
general-purpose tool that is far more widely used than any finance repository:

1. **Lagged differencing on a gappy quarterly series** — "this quarter minus the same quarter a
   year earlier". The general tool is **pandas** (BSD-3-Clause): a ``PeriodIndex`` pairs by
   calendar period, so a missing quarter becomes NaN instead of silently shifting the lag.
2. **Refusing to standardise by a degenerate scale** — a standard deviation that is zero, or only
   rounding noise. The general tools are **scikit-learn** (``StandardScaler``, whose
   ``_is_constant_feature`` bounds rounding error after Chan, Golub & LeVeque, and
   ``VarianceThreshold``, which raises on a constant feature) and **SciPy** (``zscore``/``zmap``,
   which return NaN when ``std <= |eps * mean|``).
3. **Cross-sectional ranking** of the standardised score, where one fabricated value sorts to the
   top of the list.

All three are run here on the same input the capability is measured on — live SEC EDGAR diluted
EPS: the nine rToken anchors through `market/fundamentals.py` (exactly what the desk reads), and
every US filer in the SEC's own quarterly XBRL frames for 2022Q1-2026Q2 (6,729 filers, values kept
as the exact decimals the SEC serves). Ground truth is exact rational arithmetic on those decimals,
so "degenerate" and "correct" are facts, not tolerances.

**What was found, and what it changed.** pandas beat the ARGUS that existed this morning on (1):
SEC XBRL has no standalone fiscal Q4, so the positional ``quarters[i + 4]`` was five fiscal
quarters back for almost every filer, including every anchor — the "year-over-year" SUE the desk
was reading was not year-over-year. `research.sue.read_dated` now pairs by date; this module
measures both, and :func:`pandas_agreement` checks the fix against pandas filer by filer: where
the two select the same quarter pairs the values must agree to 1e-9, and where they select
different pairs the report says which side's pairs are a year apart. On (2) the general tools
lose: both scale their threshold by the deltas' own mean, and let a constant yearly step on a real
EPS level through as a finite SUE between about 1e7 and 1e16 (the measured range is in the
report's ``constant_step_fabricated_magnitudes``), while ``StandardScaler`` answers a genuinely
constant history with a plausible-looking number — the step itself — rather than a refusal. Their
one advantage over the old absolute ``1e-9`` threshold — invariance to units — was taken
(`research.sue.noise_floor`), measured in the unit sweep below.

SCOPE, stated explicitly: :data:`SCOPE_STATEMENT`.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
import urllib.request
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np

from argus.eval import artefact
from argus.market.evidence import _UA
from argus.market.fundamentals import FundamentalsSource
from argus.research.sue import (
    MAX_DELTA_SPAN_DAYS,
    MIN_QUARTERS,
    N_DELTAS,
    SueError,
    read,
    read_dated,
    standardise,
    yoy_window,
)

ARTEFACT = Path(__file__).resolve().parents[3] / "data" / "general_sue_comparison.json"

FRAMES_URL = (
    "https://data.sec.gov/api/xbrl/frames/us-gaap/EarningsPerShareDiluted/USD-per-shares/"
    "CY{year}Q{quarter}.json"
)
FRAME_QUARTERS: tuple[tuple[int, int], ...] = tuple(
    (year, quarter) for year in range(2022, 2027) for quarter in range(1, 5)
    if (year, quarter) <= (2026, 2)
)
"""Eighteen calendar quarters, the SEC's own frames (one fact per filer per calendar quarter)."""

ANCHORS: tuple[str, ...] = (
    "NVDA", "TSLA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "COIN", "MSTR",
)
"""The same nine anchors `eval/earnings_comparison.py` measures."""

TRUTH_YOY_DAYS: tuple[int, int] = (320, 410)
"""Ground truth for "these two quarters are the same quarter a year apart". Deliberately wider than
`research.sue.YOY_DAY_RANGE` and set independently of it: a same-quarter pair is 364-371 days
apart, the nearest wrong pair about 91 days further either way, so every threshold in this open
interval classifies every real pair identically — the verdict cannot depend on where it sits."""

LEGACY_ABSOLUTE_TOLERANCE = 1e-9
"""The absolute refusal threshold `research.sue` used until 2026-09-25, kept here only as the
named ablation the unit sweep measures."""

VALUE_REL_TOL = 1e-9
"""A returned SUE counts as correct when it matches the exact-arithmetic value this closely."""

TOP_K = 50

UNIT_EXPONENTS: tuple[int, ...] = (-12, -9, -6, -3, 0, 2, 3, 6, 9, 12)
"""Units the anchors' EPS is re-expressed in. 10^2 is cents; 10^3 covers yen/won-scale EPS;
10^-3 and 10^-6 are thousands/millions-per-share mislabelled feeds; the rest map the envelope."""

STEP_LEVELS_FROM_PANEL: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 1.0)
"""Quantiles of the real panel's EPS magnitude used as levels for the constant-step family."""

STEPS: tuple[str, ...] = ("0.01", "0.10", "1.00")
SEASONAL: tuple[str, ...] = ("0.00", "0.03", "-0.02", "0.01")


# --- data ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point:
    """One quarterly EPS fact: the period end, and the exact decimal the SEC filed."""

    end: date
    value: Decimal

    @property
    def as_float(self) -> float:
        return float(self.value)


Panel = dict[str, list[Point]]


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body: bytes = response.read()
    loaded: dict[str, Any] = json.loads(body, parse_float=Decimal)
    return loaded


def fetch_frames_panel() -> tuple[Panel, dict[str, str]]:
    """Every filer's quarterly diluted EPS across :data:`FRAME_QUARTERS`, newest first, as the
    exact decimals the SEC serves (``parse_float=Decimal`` — never through a float). Returns
    ``(panel, names)`` keyed by CIK."""
    panel: Panel = {}
    names: dict[str, str] = {}
    for year, quarter in FRAME_QUARTERS:
        payload = _get_json(FRAMES_URL.format(year=year, quarter=quarter))
        for row in payload.get("data", []):
            cik = str(row["cik"])
            names.setdefault(cik, str(row.get("entityName", "")))
            panel.setdefault(cik, []).append(
                Point(end=date.fromisoformat(row["end"]), value=Decimal(str(row["val"])))
            )
        time.sleep(0.12)  # the SEC's fair-access limit is 10 requests a second
    for points in panel.values():
        points.sort(key=lambda p: p.end, reverse=True)
    return panel, names


def fetch_anchor_panel() -> Panel:
    """The nine anchors' quarterly diluted EPS through `FundamentalsSource.facts` — the exact
    path the desk's SUE evidence takes. Values arrive as floats parsed from SEC JSON, whose
    shortest ``repr`` is the filed decimal."""
    source = FundamentalsSource()
    panel: Panel = {}
    for ticker in ANCHORS:
        facts, _status = source.facts(ticker, concept="eps_diluted", as_of=datetime.now(UTC))
        panel[ticker] = [Point(end=f.end, value=Decimal(repr(f.value))) for f in facts]
    return panel


def panel_digest(panel: Mapping[str, Sequence[Point]]) -> str:
    canonical = json.dumps(
        {k: [[p.end.isoformat(), str(p.value)] for p in v] for k, v in sorted(panel.items())},
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# --- ground truth ---------------------------------------------------------------------------------


def is_true_yoy(end: date, prior_end: date) -> bool:
    low, high = TRUTH_YOY_DAYS
    return low <= (end - prior_end).days <= high


def exact_sue(deltas: Sequence[Decimal]) -> Decimal | None:
    """Exact SUE over exact deltas (newest first): ``None`` when their variance is exactly zero —
    the only honest answer — else the ratio, to 60 significant digits."""
    spread = exact_spread(deltas)
    if spread == 0:
        return None
    with localcontext() as ctx:
        ctx.prec = 60
        return deltas[0] / spread


def exact_spread(deltas: Sequence[Decimal]) -> Decimal:
    """Population standard deviation in exact decimal arithmetic (square root to 60 digits)."""
    with localcontext() as ctx:
        ctx.prec = 60
        n = Decimal(len(deltas))
        mean = sum(deltas, Decimal(0)) / n
        return (sum(((d - mean) ** 2 for d in deltas), Decimal(0)) / n).sqrt()


@dataclass(frozen=True, slots=True)
class Window:
    """Eight year-over-year changes for one filer, newest first, as both exact decimals and the
    floats every method under test receives."""

    key: str
    exact: tuple[Decimal, ...]
    floats: tuple[float, ...]
    magnitude: float

    @property
    def truth(self) -> Decimal | None:
        return exact_sue(self.exact)


def _dated_window(key: str, points: Sequence[Point]) -> Window | None:
    """The eight genuine year-over-year pairs `research.sue.yoy_window` selects, or ``None``."""
    by_end = {p.end: p for p in points}
    try:
        pairs = yoy_window([(p.end, p.as_float) for p in points])
    except SueError:
        return None
    exact = tuple(by_end[d.end].value - by_end[d.prior_end].value for d in pairs)
    floats = tuple(d.delta for d in pairs)
    magnitude = max(max(abs(d.eps), abs(d.prior_eps)) for d in pairs)
    return Window(key=key, exact=exact, floats=floats, magnitude=magnitude)


# --- part 1: lagged differencing ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reading:
    """One method's end-to-end answer for one filer: a value from named pairs, or a refusal."""

    value: float | None
    pairs: tuple[tuple[date, date], ...]
    refusal: str = ""
    crashed: str = ""


def run_argus_positional(points: Sequence[Point]) -> Reading:
    """The desk's path until 2026-09-25: the newest twelve facts, ``quarters[i + 4]``."""
    window = list(points[:MIN_QUARTERS])
    pairs = tuple((window[i].end, window[i + 4].end) for i in range(N_DELTAS)) if len(
        window) == MIN_QUARTERS else ()
    try:
        return Reading(read("X", [p.as_float for p in window]).sue, pairs)
    except SueError as exc:
        return Reading(None, pairs, refusal=str(exc))


def run_argus_dated(points: Sequence[Point]) -> Reading:
    history = [(p.end, p.as_float) for p in points]
    try:
        pairs = tuple((d.end, d.prior_end) for d in yoy_window(history))
    except SueError as exc:
        return Reading(None, (), refusal=str(exc))
    try:
        return Reading(read_dated("X", history).sue, pairs)
    except SueError as exc:
        return Reading(None, pairs, refusal=str(exc))


def _pandas_deltas(points: Sequence[Point]) -> tuple[Any, dict[Any, date]]:
    """pandas' own idiom: a quarterly ``PeriodIndex`` built from the period ends, reindexed onto a
    complete ``period_range`` so a missing quarter is an explicit NaN, then ``diff(4)``."""
    import pandas as pd

    ordered = sorted(points, key=lambda p: p.end)
    index = pd.PeriodIndex([pd.Period(p.end, freq="Q") for p in ordered])
    series = pd.Series([p.as_float for p in ordered], index=index)
    full = series.reindex(pd.period_range(index.min(), index.max(), freq="Q"))
    end_of = {period: p.end for period, p in zip(index, ordered, strict=True)}
    return full.diff(4), end_of


def _pandas_pairs(deltas: Any, end_of: Mapping[Any, date]) -> tuple[tuple[date, date], ...]:
    return tuple((end_of[period], end_of[period - 4]) for period in deltas.index[::-1])


def run_pandas_strict(points: Sequence[Point]) -> Reading:
    """``diff(4).rolling(8).std(ddof=0)`` — the textbook pandas SUE. NaN (a refusal with no
    reason attached) whenever any of the last eight calendar quarters lacks a same-quarter-last-
    year partner."""
    try:
        deltas, end_of = _pandas_deltas(points)
    except ValueError as exc:
        return Reading(None, (), crashed=f"{type(exc).__name__}: {exc}")
    std = deltas.rolling(N_DELTAS, min_periods=N_DELTAS).std(ddof=0)
    last, spread = float(deltas.iloc[-1]), float(std.iloc[-1])
    tail = deltas.iloc[-N_DELTAS:]
    pairs = _pandas_pairs(tail.dropna(), end_of) if not math.isnan(spread) else ()
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(np.float64(last) / np.float64(spread))
    if math.isnan(value):
        return Reading(None, pairs, refusal="NaN")
    return Reading(value, pairs)


def run_pandas_lenient(points: Sequence[Point]) -> Reading:
    """``diff(4).dropna().tail(8)`` — pandas' NaN-skipping variant, the closest pandas idiom to
    `read_dated`. No refusal beyond NaN: a zero spread divides."""
    try:
        deltas, end_of = _pandas_deltas(points)
    except ValueError as exc:
        return Reading(None, (), crashed=f"{type(exc).__name__}: {exc}")
    if math.isnan(float(deltas.iloc[-1])):
        return Reading(None, (), refusal="NaN (newest quarter has no partner)")
    tail = deltas.dropna().iloc[-N_DELTAS:]
    if len(tail) < N_DELTAS:
        return Reading(None, (), refusal="fewer than eight changes")
    values = tail.to_numpy(dtype=float)[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(values[0] / values.std())
    return Reading(value, _pandas_pairs(tail, end_of))


ALIGNMENT_METHODS: dict[str, Callable[[Sequence[Point]], Reading]] = {
    "argus_positional_before": run_argus_positional,
    "pandas_period_strict": run_pandas_strict,
    "pandas_period_lenient": run_pandas_lenient,
    "argus_dated_after": run_argus_dated,
}


def read_all(panel: Panel, method: Callable[[Sequence[Point]], Reading]) -> dict[str, Reading]:
    return {key: method(points) for key, points in panel.items()}


def _alignment_row(readings: Mapping[str, Reading]) -> dict[str, Any]:
    produced = {k: r for k, r in readings.items() if r.value is not None}
    pair_total = sum(len(r.pairs) for r in produced.values())
    pair_true = sum(is_true_yoy(a, b) for r in produced.values() for a, b in r.pairs)
    all_true = sum(
        1 for r in produced.values() if r.pairs and all(is_true_yoy(a, b) for a, b in r.pairs)
    )
    return {
        "filers": len(readings),
        "readings": len(produced),
        "non_finite_readings": sum(1 for r in produced.values() if not math.isfinite(r.value or 0)),
        "refused": sum(1 for r in readings.values() if r.value is None and not r.crashed),
        "crashed": sum(1 for r in readings.values() if r.crashed),
        "pairs_used": pair_total,
        "pairs_genuinely_year_over_year": pair_true,
        "share_of_pairs_year_over_year": pair_true / pair_total if pair_total else None,
        "readings_with_all_pairs_year_over_year": all_true,
        "readings_with_no_pair_year_over_year": sum(
            1 for r in produced.values()
            if r.pairs and not any(is_true_yoy(a, b) for a, b in r.pairs)
        ),
    }


def _anchor_detail(panel: Panel) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ticker, points in panel.items():
        row: dict[str, Any] = {"symbol": ticker, "facts": len(points)}
        for name, method in ALIGNMENT_METHODS.items():
            reading = method(points)
            row[name] = {
                "sue": reading.value,
                "refusal": reading.refusal or reading.crashed or None,
                "pair_gaps_days": [(a - b).days for a, b in reading.pairs],
            }
        out.append(row)
    return out


def pandas_collisions(panel: Panel, names: Mapping[str, str], limit: int = 8) -> dict[str, Any]:
    """Filers whose real period ends put two different fiscal quarters into one calendar quarter
    — the 52/53-week calendars pandas' ``PeriodIndex`` cannot hold without duplicate labels."""
    import pandas as pd

    examples: list[dict[str, Any]] = []
    count = 0
    for key, points in panel.items():
        buckets: dict[Any, list[date]] = {}
        for p in points:
            buckets.setdefault(pd.Period(p.end, freq="Q"), []).append(p.end)
        clashes = [sorted(v) for v in buckets.values() if len(v) > 1]
        if clashes:
            count += 1
            if len(examples) < limit:
                examples.append({
                    "filer": key, "name": names.get(key, key),
                    "period_ends_sharing_one_calendar_quarter": [
                        [d.isoformat() for d in c] for c in clashes
                    ],
                })
    return {"filers": count, "examples": examples}


AGREEMENT_REL_TOL = 1e-9
"""Two readings built from the same quarter pairs agree when they match this closely (relative to
the larger of 1 and the value): both are one float64 division of the same eight differences, so
anything beyond rounding would be a defect in one of them."""


def pandas_agreement(
    pandas_readings: Mapping[str, Reading], argus_readings: Mapping[str, Reading],
    limit: int = 8,
) -> dict[str, Any]:
    """ARGUS's dated pairing checked against pandas' calendar-period pairing, filer by filer.

    Over every filer where both return a finite SUE: those whose two readings use the same set of
    quarter pairs must agree to :data:`AGREEMENT_REL_TOL` — a disagreement there is a bug, not a
    convention. Where the pair sets differ, each side's pairs are checked against the
    year-over-year truth, so the report says who picked the wrong quarters rather than only that
    they differ. Also counted: filers pandas answers with ``inf``/``nan`` (a zero spread divided)
    and what ARGUS did with the same filer."""
    both = sorted(
        k for k in pandas_readings.keys() & argus_readings.keys()
        if pandas_readings[k].value is not None and argus_readings[k].value is not None
        and math.isfinite(pandas_readings[k].value or 0.0)
    )
    same = [k for k in both if set(pandas_readings[k].pairs) == set(argus_readings[k].pairs)]
    disagree = [
        k for k in same
        if abs((pandas_readings[k].value or 0.0) - (argus_readings[k].value or 0.0))
        > AGREEMENT_REL_TOL * max(1.0, abs(argus_readings[k].value or 0.0))
    ]
    different = [k for k in both if k not in set(same)]

    def all_yoy(reading: Reading) -> bool:
        return all(is_true_yoy(a, b) for a, b in reading.pairs)

    non_finite = [
        k for k, r in pandas_readings.items()
        if r.value is not None and not math.isfinite(r.value)
    ]
    return {
        "both_finite": len(both),
        "identical_pair_sets": len(same),
        "identical_pair_sets_values_disagree": disagree[:limit],
        "identical_pair_sets_values_disagree_count": len(disagree),
        "different_pair_sets": len(different),
        "different_pair_sets_pandas_has_a_non_year_over_year_pair": sum(
            1 for k in different if not all_yoy(pandas_readings[k])),
        "different_pair_sets_argus_has_a_non_year_over_year_pair": sum(
            1 for k in different if not all_yoy(argus_readings[k])),
        "different_pair_sets_examples": [
            {
                "filer": k,
                "pandas_pair_gaps_days": [(a - b).days for a, b in pandas_readings[k].pairs],
                "argus_pair_gaps_days": [(a - b).days for a, b in argus_readings[k].pairs],
            }
            for k in different[:limit]
        ],
        "pandas_non_finite": len(non_finite),
        "pandas_non_finite_argus_refused": sum(
            1 for k in non_finite if k in argus_readings and argus_readings[k].value is None),
    }


def run_alignment(
    anchors: Panel, frames: Panel, names: Mapping[str, str],
    frames_readings: Mapping[str, Mapping[str, Reading]],
) -> dict[str, Any]:
    """Part 1: which methods' "year-over-year" changes are actually a year apart. ``frames`` is
    the filers with at least :data:`MIN_QUARTERS` quarters; ``frames_readings`` each end-to-end
    method's readings of them (computed once, shared with :func:`run_ranking`)."""
    return {
        "pandas_calendar_collisions_anchors": pandas_collisions(anchors, {a: a for a in anchors}),
        "pandas_calendar_collisions_frames": pandas_collisions(frames, names),
        "anchors": {
            name: _alignment_row(read_all(anchors, m)) for name, m in ALIGNMENT_METHODS.items()
        },
        "anchor_detail": _anchor_detail(anchors),
        "frames_filers_with_12_or_more_quarters": len(frames),
        "frames": {name: _alignment_row(frames_readings[name]) for name in ALIGNMENT_METHODS},
        "argus_dated_vs_pandas_lenient": pandas_agreement(
            frames_readings["pandas_period_lenient"], frames_readings["argus_dated_after"]),
    }


# --- part 2: degenerate scale ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    value: float | None
    refused: bool


def _reference(window: Window) -> Verdict:
    """QuantConnect's vendored denominator (``np.std``), no guard."""
    deltas = np.array(window.floats)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Verdict(float(deltas[0] / np.std(deltas)), refused=False)


def _sklearn_standard_scaler(window: Window) -> Verdict:
    """``StandardScaler(with_mean=False)``: SUE is the newest change over the fitted ``scale_``.
    A feature its ``_is_constant_feature`` calls constant gets ``scale_ = 1.0`` — it never
    refuses."""
    from sklearn.preprocessing import StandardScaler

    deltas = np.array(window.floats).reshape(-1, 1)
    scaler = StandardScaler(with_mean=False).fit(deltas)
    return Verdict(float(deltas[0, 0] / scaler.scale_[0]), refused=False)


def _sklearn_variance_threshold(window: Window) -> Verdict:
    """``VarianceThreshold(0.0)``: its ``fit`` raises ``ValueError`` when no feature has variance
    above zero (with a peak-to-peak check for exactly-constant columns) — a real refusal."""
    from sklearn.feature_selection import VarianceThreshold

    deltas = np.array(window.floats).reshape(-1, 1)
    try:
        VarianceThreshold(threshold=0.0).fit(deltas)
    except ValueError:
        return Verdict(None, refused=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Verdict(float(deltas[0, 0] / np.std(deltas)), refused=False)


def _scipy_zscore(window: Window) -> Verdict:
    """SciPy's own degeneracy rule: ``zscore`` returns all-NaN when ``std <= |eps * mean|``
    (`scipy/stats/_stats_py.py`, ``zmap``); treated as a refusal. Otherwise the population
    standard deviation from ``scipy.stats.tstd(ddof=0)``."""
    from scipy import stats

    deltas = np.array(window.floats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = stats.zscore(deltas)
        if bool(np.all(np.isnan(z))):
            return Verdict(None, refused=True)
        spread = float(stats.tstd(deltas, ddof=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return Verdict(float(np.float64(deltas[0]) / np.float64(spread)), refused=False)


def _argus_absolute_before(window: Window) -> Verdict:
    """The ablation: ARGUS's rule until 2026-09-25, ``pstdev <= 1e-9``."""
    spread = statistics.pstdev(window.floats)
    if spread <= LEGACY_ABSOLUTE_TOLERANCE:
        return Verdict(None, refused=True)
    return Verdict(window.floats[0] / spread, refused=False)


def _argus_after(window: Window) -> Verdict:
    """`research.sue.standardise` as it now ships: refusal relative to the operands' magnitude."""
    try:
        sue, _change, _std = standardise(window.floats, window.magnitude)
    except SueError:
        return Verdict(None, refused=True)
    return Verdict(sue, refused=False)


DEGENERACY_METHODS: dict[str, Callable[[Window], Verdict]] = {
    "quantconnect_reference": _reference,
    "sklearn_standard_scaler": _sklearn_standard_scaler,
    "sklearn_variance_threshold": _sklearn_variance_threshold,
    "scipy_zscore_guard": _scipy_zscore,
    "argus_absolute_before": _argus_absolute_before,
    "argus_relative_after": _argus_after,
}

OUTCOMES = (
    "correct_refusal", "fabricated_non_finite", "fabricated_finite", "false_refusal",
    "correct_value", "wrong_value",
)


def value_tolerance(window: Window, truth: Decimal) -> float:
    """How far a float64 method may land from the exact SUE and still be right.

    The inputs themselves are the limit: each float delta can sit ``2 * eps * M`` from its exact
    decimal (``M`` the largest EPS magnitude), which moves ``d0 / sigma`` by up to
    ``2 * eps * M * (1 + |SUE|) / sigma``. Four times that, plus :data:`VALUE_REL_TOL` of the
    value — so a one-cent surprise on a $6.1 million EPS level, where no float64 method can be
    closer than about 1e-7, is judged by what float64 can carry rather than failed for everyone
    alike."""
    sigma = float(exact_spread(window.exact))
    sue = abs(float(truth))
    return VALUE_REL_TOL * sue + 8 * sys.float_info.epsilon * window.magnitude * (1 + sue) / sigma


def classify(window: Window, verdict: Verdict) -> str:
    """Against exact arithmetic: a refusal is right only when the exact variance is zero; any
    number returned for such a window is fabricated — an ``inf``/``nan`` a ranking mis-sorts, or
    worse, a finite number nothing downstream can tell from a real one."""
    truth = window.truth
    if truth is None:
        if verdict.refused or verdict.value is None:
            return "correct_refusal"
        return "fabricated_finite" if math.isfinite(verdict.value) else "fabricated_non_finite"
    if verdict.refused or verdict.value is None:
        return "false_refusal"
    if not math.isfinite(verdict.value):
        return "wrong_value"
    close = abs(verdict.value - float(truth)) <= value_tolerance(window, truth)
    return "correct_value" if close else "wrong_value"


def _score(windows: Iterable[Window]) -> dict[str, dict[str, int]]:
    table = {name: dict.fromkeys(OUTCOMES, 0) for name in DEGENERACY_METHODS}
    for window in windows:
        for name, method in DEGENERACY_METHODS.items():
            table[name][classify(window, method(window))] += 1
    return table


def _window_from_decimals(key: str, oldest_first: Sequence[Decimal]) -> Window:
    """Eight positional year-over-year changes from twelve consecutive quarters (oldest first)."""
    newest = list(reversed(oldest_first))
    exact = tuple(newest[i] - newest[i + 4] for i in range(N_DELTAS))
    floats_q = [float(v) for v in newest]
    floats = tuple(floats_q[i] - floats_q[i + 4] for i in range(N_DELTAS))
    return Window(key=key, exact=exact, floats=floats,
                  magnitude=max(abs(v) for v in floats_q[:MIN_QUARTERS]))


CONSTRUCTED_OLDEST_FIRST: dict[str, tuple[Decimal, ...]] = {
    "whole_dollar_linear": tuple(Decimal(i) for i in range(1, 13)),
    "all_flat": (Decimal(0),) * MIN_QUARTERS,
    "decimal_linear": tuple(Decimal(repr(round(1.2 - 0.1 * i, 2))) for i in range(11, -1, -1)),
}
"""`eval/earnings_comparison.py`'s own three degenerate cases (its ``CONSTRUCTED_LINEAR_QUARTERS``,
its all-flat history, and `tests/test_sue.py`'s float-noise decimal path), oldest first."""


def constructed_windows(scale: Decimal = Decimal(1)) -> list[Window]:
    return [
        _window_from_decimals(
            key if scale == 1 else f"{key} x{scale:.0E}", [v * scale for v in values]
        )
        for key, values in CONSTRUCTED_OLDEST_FIRST.items()
    ]


def step_windows(levels: Sequence[Decimal]) -> tuple[list[Window], list[Window]]:
    """A seasonal EPS path growing by an exact constant step each year, at real EPS levels —
    degenerate by construction — and its one-cent-surprise twin, which is not."""
    degenerate: list[Window] = []
    genuine: list[Window] = []
    for level in levels:
        for step in STEPS:
            path = [
                level + Decimal(SEASONAL[t % 4]) + Decimal(step) * (t // 4) for t in range(12)
            ]
            degenerate.append(_window_from_decimals(f"step level={level} step={step}", path))
            twin = [*path[:-1], path[-1] + Decimal("0.01")]
            genuine.append(_window_from_decimals(f"twin level={level} step={step}", twin))
    return degenerate, genuine


def _scaled(window: Window, exponent: int, raw: Sequence[Point]) -> Window | None:
    scale = Decimal(10) ** exponent
    return _dated_window(
        f"{window.key} x1e{exponent}", [Point(end=p.end, value=p.value * scale) for p in raw]
    )


def unit_sweep(anchors: Panel) -> dict[str, Any]:
    """Every anchor's genuine window re-expressed in :data:`UNIT_EXPONENTS` units. SUE is a
    ratio: the exact answer is identical at every scale, so a method that changes its verdict
    with the unit is wrong at some unit."""
    per_method: dict[str, dict[str, int]] = {
        name: dict.fromkeys(OUTCOMES, 0) for name in DEGENERACY_METHODS
    }
    flips: dict[str, list[str]] = {name: [] for name in DEGENERACY_METHODS}
    for ticker, points in anchors.items():
        base = _dated_window(ticker, points)
        if base is None:
            continue
        for exponent in UNIT_EXPONENTS:
            window = _scaled(base, exponent, points)
            if window is None:
                continue
            for name, method in DEGENERACY_METHODS.items():
                outcome = classify(window, method(window))
                per_method[name][outcome] += 1
                if outcome != "correct_value":
                    flips[name].append(f"{ticker} x1e{exponent}: {outcome}")
    for exponent in UNIT_EXPONENTS:
        for window in constructed_windows(Decimal(10) ** exponent):
            for name, method in DEGENERACY_METHODS.items():
                outcome = classify(window, method(window))
                per_method[name][outcome] += 1
                if outcome != "correct_refusal":
                    flips[name].append(f"{window.key}: {outcome}")
    return {"outcomes": per_method, "non_correct_cases": flips}


def _levels(frames: Panel) -> list[Decimal]:
    magnitudes = sorted(
        max(abs(p.value) for p in points[:MIN_QUARTERS])
        for points in frames.values() if len(points) >= MIN_QUARTERS
    )
    out: list[Decimal] = []
    for q in STEP_LEVELS_FROM_PANEL:
        value = magnitudes[min(len(magnitudes) - 1, int(q * len(magnitudes)))]
        out.append(value.quantize(Decimal("0.01")))
    return out


def run_degeneracy(
    anchors: Panel, frames: Panel, windows: Mapping[str, Window | None],
) -> dict[str, Any]:
    """Part 2: who refuses exactly when the exact variance is zero, and scores correctly
    otherwise, on the same eight float deltas. ``windows`` is every filer's genuine year-over-year
    window (``None`` where it has none)."""
    real = [w for w in windows.values() if w is not None]
    levels = _levels(frames)
    degenerate_steps, genuine_twins = step_windows(levels)
    real_degenerate = [w.key for w in real if w.truth is None]
    return {
        "real_frames_windows": len(real),
        "real_frames_degenerate_windows": len(real_degenerate),
        "real_frames_degenerate_filers": sorted(real_degenerate, key=int)[:12],
        "real_frames": _score(real),
        "constructed_from_earnings_comparison": _score(constructed_windows()),
        "constant_step_levels_from_panel": [str(v) for v in levels],
        "constant_step_degenerate": _score(degenerate_steps),
        "constant_step_genuine_twins": _score(genuine_twins),
        "constant_step_cases": {
            w.key: {name: _verdict_json(m(w)) for name, m in DEGENERACY_METHODS.items()}
            for w in degenerate_steps
        },
        "constant_step_fabricated_magnitudes": fabricated_magnitudes(degenerate_steps),
        "unit_sweep": unit_sweep(anchors),
    }


def fabricated_magnitudes(windows: Sequence[Window]) -> dict[str, dict[str, Any]]:
    """Per method, the smallest and largest ``|SUE|`` it returned as a finite number for a window
    whose exact variance is zero — the size of the number a downstream ranking would have taken
    as real. ``None`` where the method fabricated no finite value."""
    out: dict[str, dict[str, Any]] = {}
    for name, method in DEGENERACY_METHODS.items():
        values = [
            abs(v.value) for w in windows if w.truth is None
            for v in (method(w),)
            if not v.refused and v.value is not None and math.isfinite(v.value)
        ]
        out[name] = {
            "count": len(values),
            "min_abs": min(values) if values else None,
            "max_abs": max(values) if values else None,
        }
    return out


def _verdict_json(verdict: Verdict) -> Any:
    if verdict.refused:
        return "refused"
    return verdict.value


# --- part 3: ranking --------------------------------------------------------------------------


def _ranking(readings: Mapping[str, Reading]) -> list[tuple[str, float, Reading]]:
    rows = [
        (key, r.value, r) for key, r in readings.items()
        if r.value is not None and not math.isnan(r.value)
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _sklearn_end_to_end(points: Sequence[Point]) -> Reading:
    """What a general-purpose pipeline gives: pandas' lenient alignment for the changes, then
    scikit-learn's ``StandardScaler`` for the scale."""
    from sklearn.preprocessing import StandardScaler

    aligned = run_pandas_lenient(points)
    if aligned.value is None:
        return aligned
    deltas, _end_of = _pandas_deltas(points)
    values = deltas.dropna().iloc[-N_DELTAS:].to_numpy(dtype=float)[::-1].reshape(-1, 1)
    scaler = StandardScaler(with_mean=False).fit(values)
    return Reading(float(values[0, 0] / scaler.scale_[0]), aligned.pairs)


def run_ranking(
    names: Mapping[str, str],
    frames_readings: Mapping[str, Mapping[str, Reading]],
    truth: Mapping[str, Decimal | None],
) -> dict[str, Any]:
    """Part 3: the top of each method's cross-sectional ranking, checked against exact arithmetic
    over genuine year-over-year pairs (``truth``: filer -> exact SUE, ``None`` when degenerate)."""
    out: dict[str, Any] = {}
    for name in RANKING_METHODS:
        ranked = _ranking(frames_readings[name])
        top = ranked[:TOP_K]
        out[name] = {
            "ranked": len(ranked),
            f"top_{TOP_K}_non_year_over_year": sum(
                1 for _k, _v, r in top if not all(is_true_yoy(a, b) for a, b in r.pairs)
            ),
            f"top_{TOP_K}_exactly_degenerate": sum(
                1 for k, _v, _r in top if k in truth and truth[k] is None
            ),
            f"top_{TOP_K}_non_finite": sum(1 for _k, v, _r in top if not math.isfinite(v)),
            "top_5": [
                {"cik": k, "name": names.get(k, ""), "sue": v} for k, v, _r in ranked[:5]
            ],
            "spearman_vs_exact": _spearman(
                {k: v for k, v, _r in ranked if math.isfinite(v)},
                {k: float(t) for k, t in truth.items() if t is not None},
            ),
        }
    return out


def _reference_end_to_end(points: Sequence[Point]) -> Reading:
    """QuantConnect's vendored formula on the newest twelve facts, as the desk fed it."""
    window = list(points[:MIN_QUARTERS])
    if len(window) < MIN_QUARTERS:
        return Reading(None, (), refusal="short")
    values = [p.as_float for p in window]
    deltas = np.array([values[i] - values[i + 4] for i in range(N_DELTAS)])
    pairs = tuple((window[i].end, window[i + 4].end) for i in range(N_DELTAS))
    with np.errstate(divide="ignore", invalid="ignore"):
        return Reading(float(deltas[0] / np.std(deltas)), pairs)


RANKING_METHODS: dict[str, Callable[[Sequence[Point]], Reading]] = {
    "quantconnect_positional": _reference_end_to_end,
    "argus_positional_before": run_argus_positional,
    "pandas_period_lenient": run_pandas_lenient,
    "pandas_then_sklearn_scaler": _sklearn_end_to_end,
    "argus_dated_after": run_argus_dated,
}

END_TO_END: dict[str, Callable[[Sequence[Point]], Reading]] = {
    **ALIGNMENT_METHODS, **RANKING_METHODS,
}


def _spearman(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return {"n": len(common), "rho": None}
    from scipy.stats import spearmanr

    rho = spearmanr([a[k] for k in common], [b[k] for k in common]).statistic
    return {"n": len(common), "rho": float(rho)}


# --- costs, reproducibility, report ---------------------------------------------------------------


def measure_costs(frames: Panel, sample: int = 300) -> dict[str, float]:
    """Wall-clock seconds per end-to-end reading on the same real filers."""
    keys = sorted(k for k, v in frames.items() if len(v) >= MIN_QUARTERS)[:sample]
    out: dict[str, float] = {}
    for name, method in ALIGNMENT_METHODS.items():
        start = time.perf_counter()
        for key in keys:
            method(frames[key])
        out[f"{name}_seconds_per_reading"] = (time.perf_counter() - start) / max(1, len(keys))
    return out


def compute(anchors: Panel, frames: Panel, names: Mapping[str, str]) -> dict[str, Any]:
    """Everything deterministic, given fetched data — run twice for the reproducibility check."""
    eligible = {k: v for k, v in frames.items() if len(v) >= MIN_QUARTERS}
    readings = {name: read_all(eligible, method) for name, method in END_TO_END.items()}
    windows = {k: _dated_window(k, v) for k, v in frames.items()}
    truth = {k: w.truth for k, w in windows.items() if w is not None and k in eligible}
    return {
        "alignment": run_alignment(anchors, eligible, names, readings),
        "degeneracy": run_degeneracy(anchors, frames, windows),
        "degenerate_filer_names": {
            k: names.get(k, "") for k, w in windows.items() if w is not None and w.truth is None
        },
        "ranking": run_ranking(names, readings, truth),
    }


SCOPE_STATEMENT = (
    "Measured on live SEC EDGAR diluted EPS: the nine rToken anchors through the desk's own "
    "fetch, and every filer in the SEC's quarterly XBRL frames 2022Q1-2026Q2, against exact "
    "rational arithmetic on the filed decimals. Claimed: (1) pandas' calendar-period differencing "
    "beat ARGUS's positional lag, which paired different fiscal quarters for almost every filer "
    "because XBRL has no standalone fiscal Q4; ARGUS now pairs by date, agrees with pandas to "
    "1e-9 on every filer where the two select the same quarter pairs, and also pairs 52/53-week "
    "fiscal quarters that straddle a calendar boundary, where pandas' PeriodIndex crashes on a "
    "duplicate label or pairs quarters that are not a year apart. (2) On refusing a degenerate "
    "scale, scikit-learn and SciPy lose: their delta-relative thresholds pass a constant yearly "
    "step on a real EPS level as a finite SUE between about 1e7 and 1e16, and StandardScaler "
    "returns a number for a constant history instead of refusing; their unit-invariance was "
    "adopted. On the real SEC panel VarianceThreshold, SciPy and ARGUS all refuse every "
    "exactly-degenerate filer (StandardScaler returns a number for each, the reference inf or "
    "nan), so ARGUS's margin over VarianceThreshold and SciPy rests on constructed constant-step "
    "windows at real EPS levels, not on a real filer. NOT claimed: that SUE predicts returns; that "
    "eight changes spanning eleven fiscal quarters (Q4 excluded) is the same estimator as the "
    "reference's eight consecutive quarters — it is the closest honest one XBRL allows; that the "
    "frames panel is point-in-time (it carries each period's latest filed value)."
)


def main() -> int:  # pragma: no cover - CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    anchors = fetch_anchor_panel()
    frames, names = fetch_frames_panel()
    first = compute(anchors, frames, names)
    second = compute(anchors, frames, names)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "frames_quarters": [f"CY{y}Q{q}" for y, q in FRAME_QUARTERS],
            "frames_filers": len(frames),
            "frames_digest_sha256": panel_digest(frames),
            "anchors_digest_sha256": panel_digest(anchors),
            "truth_yoy_days": list(TRUTH_YOY_DAYS),
            "max_delta_span_days": MAX_DELTA_SPAN_DAYS,
        },
        **first,
        "costs": measure_costs(frames),
        "reproducibility": {
            "identical_on_same_fetched_inputs": artefact.dumps(first) == artefact.dumps(second),
        },
        "scope_statement": SCOPE_STATEMENT,
    }
    undefined = artefact.write(ARTEFACT, report)
    print(render(report))
    print(f"\nwritten to {ARTEFACT}" + (f" ({len(undefined)} non-finite -> null)" if undefined
                                      else ""))
    return 0


def render(report: Mapping[str, Any]) -> str:
    lines = ["SUE vs general-purpose rivals (pandas / scikit-learn / SciPy)\n"]
    for universe in ("anchors", "frames"):
        lines.append(f"  alignment on {universe}:")
        for name, row in report["alignment"][universe].items():
            share = row["share_of_pairs_year_over_year"]
            lines.append(
                f"    {name:26s} readings={row['readings']:5d}  YoY pairs="
                f"{'n/a' if share is None else f'{share:.1%}'}"
            )
    lines.append("  degenerate-scale detection (constant yearly step at real EPS levels):")
    for name, row in report["degeneracy"]["constant_step_degenerate"].items():
        lines.append(
            f"    {name:28s} refused={row['correct_refusal']:3d} fabricated="
            f"{row['fabricated_finite'] + row['fabricated_non_finite']:3d}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "AGREEMENT_REL_TOL",
    "ALIGNMENT_METHODS",
    "ANCHORS",
    "DEGENERACY_METHODS",
    "END_TO_END",
    "RANKING_METHODS",
    "SCOPE_STATEMENT",
    "Point",
    "Reading",
    "Verdict",
    "Window",
    "classify",
    "compute",
    "constructed_windows",
    "exact_spread",
    "exact_sue",
    "fabricated_magnitudes",
    "fetch_anchor_panel",
    "fetch_frames_panel",
    "is_true_yoy",
    "main",
    "measure_costs",
    "pandas_agreement",
    "pandas_collisions",
    "panel_digest",
    "read_all",
    "render",
    "run_alignment",
    "run_argus_dated",
    "run_argus_positional",
    "run_degeneracy",
    "run_pandas_lenient",
    "run_pandas_strict",
    "run_ranking",
    "step_windows",
    "unit_sweep",
    "value_tolerance",
]
