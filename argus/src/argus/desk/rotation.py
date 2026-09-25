"""Cross-asset breadth-momentum rotation — switching between risk and safety by asset class.

`desk/allocation.py` answers "how should risk be split *within* the book we already hold" — twelve
tokenized US equities, all one asset class, all correlated above 0.9. It has no opinion on the
question the handbook's own "Cross-Asset Allocation / Rotation" sub-theme actually names: risk-on/
off rotation between **US stocks, Crypto and commodities** — three genuinely different asset
classes, not twelve correlated slices of one. This module answers that question, on real Bitget
instruments: the twelve rTokens (`market/bitget.RTOKEN_SYMBOLS`), real crypto majors (BTCUSDT,
ETHUSDT — confirmed live on the same public futures book, same endpoint, same `fetch_tickers()`),
and a real commodity token (XAUUSDT, gold — confirmed live, price sanity-checked against spot gold
at ~$4,290/oz on 2026-09-15; `FOXAUSDT` was checked and excluded, at $67 with $299 of 24h volume it
is not a gold-tracking instrument despite the ticker, the same class of trap `RTOKEN_SYMBOLS`'s own
comment already documents for SPXUSDT).

**Read before written.** The reference is pytaa (MIT),
`research/repos-t3/pytaa/src/pytaa/backtest/positions.py:154-190` (function `vigilant_allocation`,
vendored verbatim at `eval/baselines/pytaa_vigilant_allocation.py`) and
`.../strategy/signals.py:39-49` (`Signal.momentum_score`) — the real, published Vigilant Asset
Allocation rule (Keller & Keuning 2017, "Breadth Momentum and the Canary Universe: Defensive Asset
Allocation (DAA)", SSRN 2543979). :func:`momentum_score` reproduces its weighted four-horizon
formula exactly (weights 12/4/2/1 at lag ratios 1/3/6/12 months, normalised by subtracting the
weight sum of 19). :func:`breadth_allocation` reproduces its discrete safe/risk switching rule
exactly: count how many of the risk-and-safe universe carry a negative score, scale the highest-
ranked safe asset's weight by ``step`` per negative count, and split the rest equally across the
top-``top_k`` risk assets by rank. Verified to reproduce the real vendored function's numeric output
on identical input — `eval/rotation_comparison.py`'s ``run_baseline_reproduced_cases``.

**Two things ARGUS's src/ cannot do that the reference does, and one thing it deliberately refuses
to do that the reference does not.** ARGUS's `src/` carries no pandas/numpy dependency (project
convention — pydantic and python-dateutil only), so both formulas above are reimplemented in plain
Python here rather than imported; :func:`monthly_closes_from_bars` replaces pandas'
``resample("BME").last()`` with the equivalent plain-Python group-by-month-take-last. And where
the reference is silent, this module is loud: **a momentum score that cannot be computed for
every asset in play raises :class:`RotationError` rather than proposing a partial allocation.**
That difference is not cosmetic. Run on ARGUS's own real, live candle history
(`eval/rotation_comparison.py`,
2026-09-15), gold's real `momentum_score()` is NaN on every one of the 14 monthly rebalance points
computable from its available 277 real daily bars — nine months of listing history, short of the
thirteen a twelve-month lookback needs. Fed that NaN, the real vendored `vigilant_allocation` does
not refuse: `NaN < 0` is `False` in numpy, so a data-starved asset is silently counted as
"not distressed" in the breadth tally, and `.rank()` on a NaN-valued safe-asset series never equals
rank 1, so the safe asset assigned the flight-to-safety weight receives **zero** regardless of what
the breadth count says. The comparison's own measured case: three of three risk assets negative
(the sharpest "flee to safety" signal the rule can raise) with gold's score NaN — the real vendored
function allocates a grand total of **25% of the book** and silently leaves the other 75% nowhere,
worst exactly when the signal is loudest. `breadth_allocation` here raises instead.

**A full data contract, adopted from a general-purpose validator (2026-09-25).** Refusing a NaN
score covered one clause of what "data-honest" has to mean. Running the general-purpose data-
validation stack — pandera (MIT) and Great Expectations (Apache-2.0), wrapped around the same pytaa
pipeline on the same real Bitget history (`eval/general_rotation_comparison.py`) — showed the rest:
the pre-contract version of this module silently accepted a calendar-month gap (its positional
month list then read the wrong month at every later anchor), a series that had stopped updating
weeks before the rest of the universe, a negative or infinite close, ``top_k`` larger than the
risk universe (a book that quietly sums to less than one), a negative or NaN ``step``, and the same
asset named as both risk and safe. Every one of those is now a named clause —
:data:`CONTRACT_CLAUSES` — checked on every call. Two further ideas are taken from pandera, pattern
only, no code copied: *lazy* validation, which collects every violation across every symbol before
raising (`pandera/api/base/error_handler.py:52-140`, `ErrorHandler(lazy=True)`; surfaced as
`SchemaErrors`, `pandera/errors.py:165`), so a refusal names all three short-history symbols of
today's default universe rather than the first one it meets; and an output check on the finished
allocation (`pandera/decorators.py:322`, ``check_output``), so a book that does not sum to one is
refused even if a future edit opens a path no input clause anticipated.

**Bar timestamps are close times.** Bitget's ``1D`` candle ``ts`` is the candle's *open* time,
16:00 UTC (00:00 UTC+8), and its close prints 24 hours later — verified 2026-09-25 by matching each
daily close to the close of the hourly candle opened 23 hours after it. Grouping by the open time
files the candle opened 31 August 16:00 UTC, which closes on 1 September, under August, so every
"month-end close" was one bar into the next month. :func:`propose_rotation_from_bars` takes close
times explicitly and the CLI passes ``ts + 1 day``.

**Calendar months, not business months — a deliberate departure from the reference.** pytaa bins
with ``resample("BME")``, the equity-ETF business-month-end convention: a bar on a Saturday or
Sunday month-end rolls into the *next* month's bin. Bitget's instruments trade seven days a week, so
this module bins by calendar month. The two agree exactly only when every anchor month ends on a
weekday; on real BTCUSDT history they diverged at four of six evaluation dates checked
(2025-12-15, 2026-02-10, 2026-03-10, 2026-06-10: score differences of 0.39 to 0.87) and agreed at
the two whose anchors all end on weekdays (2025-10-10, 2026-09-23). The float-identity agreement
`eval/rotation_comparison.py` reports is therefore a property of the evaluation date, not of the
formula.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from argus.desk.allocation import TAKER_BPS, Trade

MOMENTUM_HORIZONS: tuple[tuple[int, int], ...] = ((12, 1), (4, 3), (2, 6), (1, 12))
"""(horizon_weight, lag_months) pairs, exactly pytaa's real `Signal.momentum_score`
(`strategy/signals.py:39-49`): ``lag = 12 // horizon_weight`` for horizon in (12, 4, 2, 1)."""

MOMENTUM_WEIGHT_SUM = sum(w for w, _ in MOMENTUM_HORIZONS)
"""19 — the real formula's own normalising constant (`score - 19`, `signals.py:48`): at zero
momentum on every horizon every ratio is 1, so the raw sum is exactly this and the normalised score
is exactly zero."""

MIN_MONTHLY_OBSERVATIONS = max(lag for _, lag in MOMENTUM_HORIZONS) + 1
"""13 — one more than the longest lag. `closes[-1 - lag]` needs at least ``lag + 1`` points; below
that the real formula's own longest-horizon term is unconditionally undefined, and NaN propagates
through the summed score in the reference. Verified on ARGUS's real data: XAUUSDT's 277 real daily
bars resample to fewer than 13 real monthly closes, and its real `momentum_score()` is NaN at all
14 of 14 computable rebalance points as of 2026-09-15 (`eval/rotation_comparison.py`)."""

TRADING_DAYS_PER_MONTH = 21
"""pytaa's own stated business-month convention (`strategy/signals.py:67`, the `days` default of
`sma_crossover`) — used only to size how many real daily bars a `--days` CLI request should fetch
to reach `MIN_MONTHLY_OBSERVATIONS` months, never to annualise anything."""


BAR_FRESHNESS_DAYS = 3.0
"""How far a series' last bar may trail the freshest series in the universe before the rotation
refuses. Every instrument here prints a daily bar seven days a week, so three days is already two
missed bars; a series further behind than that is being compared, as of an older date, against
assets priced today."""

BAR_INTERVAL = timedelta(days=1)
"""The duration of the ``1D`` candles the CLI fetches: a candle opened at ``ts`` closes at
``ts + BAR_INTERVAL``, and that close time is the timestamp the monthly binning must use."""

CONTRACT_CLAUSES: tuple[str, ...] = (
    "completeness",  # every asset in play has a series / a score
    "validity",      # every close and every score is a finite number; closes are positive
    "ordering",      # every bar timestamp is unique (reordered input is sorted, not refused)
    "timeliness",    # no series trails the freshest one by more than BAR_FRESHNESS_DAYS
    "sufficiency",   # the evaluation month and every anchor month the formula reads are present
    "parameters",    # non-empty, disjoint, duplicate-free universes; 1 <= top_k <= risk count;
                     # step finite and positive
    "output",        # the finished book sums to one and every weight lies in [0, 1]
)
"""The data contract every rotation is checked against — see the module docstring for which
clauses the general-purpose validators surfaced and the pre-contract version lacked."""

WEIGHT_SUM_TOLERANCE = 1e-9
"""Float slack for the output clause. ``(1 - s) / k`` summed ``k`` times can land a few ulps off one
(``1/3 * 3``); anything further off is a real under- or over-allocation."""


@dataclass(frozen=True, slots=True)
class RotationViolation:
    """One broken clause of :data:`CONTRACT_CLAUSES`, attributed to a symbol where it has one."""

    clause: str
    symbol: str | None
    detail: str

    def __str__(self) -> str:
        where = f"{self.symbol}: " if self.symbol else ""
        return f"[{self.clause}] {where}{self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {"clause": self.clause, "symbol": self.symbol, "detail": self.detail}


class RotationError(ValueError):
    """Raised rather than proposing a rotation from a signal that could not be computed for every
    asset the rotation would touch. See the module docstring for the real, measured failure mode
    in the reference this refuses to reproduce.

    ``violations`` carries every broken clause found in one pass (empty for the older single-cause
    raises), so a caller can act on all of them rather than fixing one and meeting the next."""

    def __init__(self, message: str, violations: Sequence[RotationViolation] = ()) -> None:
        super().__init__(message)
        self.violations: tuple[RotationViolation, ...] = tuple(violations)

    @classmethod
    def from_violations(cls, violations: Sequence[RotationViolation]) -> RotationError:
        count = len(violations)
        head = f"refused: {count} contract violation{'s' if count != 1 else ''}"
        return cls(f"{head} — " + "; ".join(str(v) for v in violations), violations)


def parameter_violations(
    risk_assets: Sequence[str], safe_assets: Sequence[str], *, top_k: int, step: float,
) -> list[RotationViolation]:
    """The ``parameters`` clause, collected rather than raised. The reference accepts all of these
    silently: ``top_k`` above the risk count or at zero leaves part of the book unallocated, a
    negative ``top_k`` or ``step`` writes negative weights, and a NaN ``step`` resolves through
    ``min(1, nan) == 1`` to a book that is 100% safe whatever the breadth count says."""
    found: list[RotationViolation] = []
    if not risk_assets:
        found.append(RotationViolation(
            "parameters", None, "breadth allocation needs at least one risk asset"))
    if not safe_assets:
        found.append(RotationViolation(
            "parameters", None, "breadth allocation needs at least one safe asset"))
    for label, group in (("risk", risk_assets), ("safe", safe_assets)):
        repeated = sorted({a for a in group if list(group).count(a) > 1})
        for asset in repeated:
            found.append(RotationViolation(
                "parameters", asset, f"listed more than once among the {label} assets"))
    for asset in sorted(set(risk_assets) & set(safe_assets)):
        found.append(RotationViolation(
            "parameters", asset, "listed as both a risk asset and a safe asset"))
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        found.append(RotationViolation("parameters", None, f"top_k={top_k!r} is not an integer"))
    elif top_k < 1:
        found.append(RotationViolation(
            "parameters", None, f"top_k={top_k} holds no risk asset; it must be at least 1"))
    elif risk_assets and top_k > len(set(risk_assets)):
        found.append(RotationViolation(
            "parameters", None,
            f"top_k={top_k} exceeds the {len(set(risk_assets))} risk asset(s) supplied — the "
            f"unfilled slots would leave part of the book unallocated"))
    if not isinstance(step, (int, float)) or isinstance(step, bool) or not math.isfinite(step) \
            or step <= 0:
        found.append(RotationViolation(
            "parameters", None, f"step={step!r} must be a finite number above zero"))
    return found


# --- signal ----------------------------------------------------------------------------------


def momentum_score(monthly_closes: Sequence[float]) -> float:
    """The real VAA weighted momentum score (`signals.py:39-49`), reproduced in plain Python.

    ``monthly_closes`` must be chronological, oldest first, real month-end closes (see
    :func:`monthly_closes_from_bars`). Raises :class:`RotationError` — never returns NaN — when
    there are fewer than :data:`MIN_MONTHLY_OBSERVATIONS` points, which is the module's one
    deliberate departure from the reference (see the module docstring).
    """
    if len(monthly_closes) < MIN_MONTHLY_OBSERVATIONS:
        raise _refusal(
            "sufficiency",
            f"{len(monthly_closes)} monthly close(s) is below the "
            f"{MIN_MONTHLY_OBSERVATIONS} a {max(lag for _, lag in MOMENTUM_HORIZONS)}-month "
            f"lookback needs",
        )
    last = monthly_closes[-1]
    _require_usable_close(last, "latest")
    raw = 0.0
    for weight, lag in MOMENTUM_HORIZONS:
        anchor = monthly_closes[-1 - lag]
        if anchor == 0:
            raise _refusal("validity", "a zero monthly close cannot anchor a momentum ratio")
        _require_usable_close(anchor, f"{lag}-month anchor")
        raw += weight * (last / anchor)
    return raw - MOMENTUM_WEIGHT_SUM


def _require_usable_close(value: float, label: str) -> None:
    """A close the formula reads must be a finite positive price. NaN here is also how
    :func:`monthly_closes_from_bars` marks a calendar month with no bar at all, so a missing
    anchor month is refused by name instead of propagating NaN into the score as the reference
    does; a negative or infinite close would otherwise yield a finite, meaningless ratio."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refusal("validity", f"the {label} monthly close {value!r} is not a number")
    if math.isnan(value):
        raise _refusal(
            "sufficiency", f"the {label} monthly close is missing (no bar in that calendar month)")
    if not math.isfinite(value) or value <= 0:
        raise _refusal(
            "validity",
            f"the {label} monthly close is {value!r}; a momentum ratio needs a finite positive "
            f"price",
        )


def _refusal(clause: str, detail: str, symbol: str | None = None) -> RotationError:
    """A single-violation :class:`RotationError`, message unchanged from the pre-contract text so
    existing callers matching on it still match."""
    return RotationError(detail, [RotationViolation(clause, symbol, detail)])


def monthly_closes_from_bars(bars: Sequence[tuple[datetime, float]]) -> list[float]:
    """Plain-Python counterpart of pytaa's ``prices.resample("BME").last()``: the last real close
    recorded in each **calendar** month (see the module docstring for why not business months),
    in chronological order, one entry per month from the first to the last.

    A calendar month with no bar at all is kept as ``NaN`` rather than skipped — exactly what the
    reference's resample does with an empty bin. Skipping it, as this function did before
    2026-09-25, shifted every earlier month one position closer, so ``closes[-1 - 12]`` silently
    read the thirteen-month-old close as the twelve-month anchor. With the ``NaN`` kept in place,
    :func:`momentum_score` computes the calendar-correct score when the gap falls between anchors
    and refuses by name when it falls on one.

    ``bars`` must be chronological, oldest first, with no repeated timestamp; anything else raises
    :class:`RotationError` (``ordering``) rather than letting "last in the month" mean "last in
    the input". Pass each bar's *close* time — see :data:`BAR_INTERVAL`.
    """
    by_month: dict[tuple[int, int], float] = {}
    previous: datetime | None = None
    for ts, close in bars:
        if previous is not None and ts <= previous:
            raise _refusal(
                "ordering",
                f"bars are not strictly chronological: {ts.isoformat()} follows "
                f"{previous.isoformat()}",
            )
        previous = ts
        by_month[(ts.year, ts.month)] = close
    if not by_month:
        return []
    first, last = min(by_month), max(by_month)
    out: list[float] = []
    year, month = first
    while (year, month) <= last:
        out.append(by_month.get((year, month), math.nan))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _month_index(ts: datetime) -> int:
    return ts.year * 12 + ts.month - 1


def _month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def bar_violations(
    bars_by_symbol: Mapping[str, Sequence[tuple[datetime, float]]],
    universe: Sequence[str],
    *,
    freshness_days: float = BAR_FRESHNESS_DAYS,
) -> list[RotationViolation]:
    """Every ``completeness``/``validity``/``ordering``/``timeliness``/``sufficiency`` violation
    in the supplied bars, collected across the whole universe before anything is raised (pandera's
    lazy mode — see the module docstring). ``bars`` carry close times.

    The evaluation month is the calendar month of the freshest bar in the universe; every symbol
    must have a bar in that month and in each anchor month the formula reads (1, 3, 6 and 12
    months back). A gap between anchors is not a violation — the score does not read it."""
    found: list[RotationViolation] = []
    usable: dict[str, Sequence[tuple[datetime, float]]] = {}
    for symbol in dict.fromkeys(universe):
        bars = bars_by_symbol.get(symbol)
        if bars is None:
            found.append(RotationViolation("completeness", symbol, "no bar series supplied"))
            continue
        if not bars:
            found.append(RotationViolation("completeness", symbol, "the bar series is empty"))
            continue
        clean = True
        repeated = sorted(ts for ts, n in Counter(ts for ts, _ in bars).items() if n > 1)
        if repeated:
            found.append(RotationViolation(
                "ordering", symbol,
                f"{len(repeated)} timestamp(s) carry more than one bar (first: "
                f"{repeated[0].isoformat()}) — which close is real cannot be decided"))
            clean = False
        bars = sorted(bars, key=lambda bar: bar[0])
        bad = [(ts, close) for ts, close in bars
               if isinstance(close, bool) or not isinstance(close, (int, float))
               or not math.isfinite(close) or close <= 0]
        if bad:
            ts, close = bad[0]
            found.append(RotationViolation(
                "validity", symbol,
                f"{len(bad)} close(s) are not finite positive prices (first: {close!r} at "
                f"{ts.isoformat()})"))
            clean = False
        if clean:
            usable[symbol] = bars

    if not usable:
        return found
    freshest = max(bars[-1][0] for bars in usable.values())
    slack = timedelta(days=freshness_days)
    for symbol, bars in usable.items():
        behind = freshest - bars[-1][0]
        if behind > slack:
            found.append(RotationViolation(
                "timeliness", symbol,
                f"last bar closes {bars[-1][0].isoformat()}, {behind.total_seconds() / 86400:.1f} "
                f"days behind the freshest series ({freshest.isoformat()}); limit "
                f"{freshness_days:g} days"))

    evaluation = _month_index(freshest)
    for symbol, bars in usable.items():
        months = {_month_index(ts) for ts, _ in bars}
        needed = (0, *(lag for _, lag in MOMENTUM_HORIZONS))
        absent = [lag for lag in needed if evaluation - lag not in months]
        if not absent:
            continue
        held = evaluation - min(months) + 1
        if held < MIN_MONTHLY_OBSERVATIONS:
            detail = (f"{held} month(s) of history to {_month_label(evaluation)}, below the "
                      f"{MIN_MONTHLY_OBSERVATIONS} a 12-month lookback needs")
        else:
            detail = "no bar in " + ", ".join(
                f"{_month_label(evaluation - lag)} "
                f"({'evaluation month' if lag == 0 else f'{lag}-month anchor'})"
                for lag in absent)
        found.append(RotationViolation("sufficiency", symbol, detail))
    return found


# --- breadth rule ------------------------------------------------------------------------------


def breadth_allocation(
    scores: Mapping[str, float],
    risk_assets: Sequence[str],
    safe_assets: Sequence[str],
    *,
    top_k: int = 2,
    step: float = 0.25,
) -> dict[str, float]:
    """The real VAA breadth rule (`vigilant_allocation`, `positions.py:154-190`), reproduced in
    plain Python, with one addition: every asset in ``risk_assets``/``safe_assets`` must have a
    real score in ``scores`` or this raises :class:`RotationError`. The reference does not — see
    the module docstring for the measured cost of that silence.

    Matches the reference exactly otherwise: ``is_neg`` counts negative scores across *both*
    universes; the single highest-ranked safe asset receives ``min(1, step * is_neg)``; the
    remaining weight splits equally across the top-``top_k`` risk assets by rank.
    """
    universe = tuple(dict.fromkeys((*risk_assets, *safe_assets)))
    found = parameter_violations(risk_assets, safe_assets, top_k=top_k, step=step)
    for a in universe:
        value = scores.get(a)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            found.append(RotationViolation(
                "completeness", a, "cannot allocate: no usable momentum score "
                "(insufficient history — see MIN_MONTHLY_OBSERVATIONS)"))
        elif math.isnan(value):
            found.append(RotationViolation(
                "completeness", a, "cannot allocate: no usable momentum score (NaN — "
                "insufficient history, see MIN_MONTHLY_OBSERVATIONS)"))
        elif not math.isfinite(value):
            found.append(RotationViolation(
                "validity", a, f"momentum score {value!r} is not finite"))
    if found:
        raise RotationError.from_violations(found)

    is_neg = sum(1 for a in universe if scores[a] < 0)
    safe_weight_total = min(1.0, step * is_neg)

    weights = dict.fromkeys(universe, 0.0)
    ranked_safe = sorted(safe_assets, key=lambda a: -scores[a])
    weights[ranked_safe[0]] = safe_weight_total

    ranked_risk = sorted(risk_assets, key=lambda a: -scores[a])
    risk_weight_each = (1.0 - safe_weight_total) / top_k
    for a in ranked_risk[:top_k]:
        weights[a] = risk_weight_each
    _check_output(weights)
    return weights


def _check_output(weights: Mapping[str, float]) -> None:
    """The ``output`` clause, pandera's ``check_output`` idea: whatever path produced the book,
    it is refused unless fully invested and long-only. Unreachable through the input clauses
    today; it is here so the next edit that opens a new path cannot open a silent one."""
    found = [RotationViolation("output", a, f"weight {w!r} lies outside [0, 1]")
             for a, w in weights.items() if not (0.0 <= w <= 1.0)]
    total = sum(weights.values())
    if not abs(total - 1.0) <= WEIGHT_SUM_TOLERANCE:
        found.append(RotationViolation(
            "output", None, f"weights sum to {total!r}, not 1 — part of the book is unallocated "
            f"or over-allocated"))
    if found:
        raise RotationError.from_violations(found)


# --- pricing -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RotationPlan:
    """The move from the book you hold to the breadth rule's target, priced. Mirrors
    `allocation.TradePlan`'s shape deliberately — both answer "is this rebalance worth its own
    cost", just on a different partition of the book (risk-parity-within-a-class vs.
    risk-on/off-across-classes)."""

    trades: tuple[Trade, ...]
    is_neg: int
    universe_size: int
    turnover: float
    cost_bps: float
    scores: dict[str, float] = field(default_factory=dict)
    """The momentum score behind every weight — the signal the regime label is counted from."""

    @property
    def regime(self) -> str:
        if self.is_neg == 0:
            return "fully risk-on — no asset in the universe carries a negative score"
        if self.is_neg >= self.universe_size:
            return "fully risk-off — every asset in the universe carries a negative score"
        return f"partial risk-off — {self.is_neg} of {self.universe_size} assets negative"

    @property
    def verdict(self) -> str:
        if not self.trades:
            return f"{self.regime}. The book is already at the breadth-rule allocation; no trade"
        moved = ", ".join(
            f"{t.symbol} {t.weight_before:.1%}->{t.weight_after:.1%}" for t in self.trades[:4]
        )
        return (
            f"{self.regime}. {len(self.trades)} leg(s) ({moved}"
            f"{'...' if len(self.trades) > 4 else ''}), turnover {self.turnover:.1%} costing "
            f"{self.cost_bps:.1f}bps"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades": [t.as_dict() for t in self.trades],
            "is_neg": self.is_neg,
            "universe_size": self.universe_size,
            "regime": self.regime,
            "turnover": round(self.turnover, 6),
            "cost_bps": round(self.cost_bps, 4),
            "scores": {a: round(v, 6) for a, v in self.scores.items()},
            "verdict": self.verdict,
        }


def propose_rotation(
    current: Mapping[str, float],
    monthly_closes: Mapping[str, Sequence[float]],
    risk_assets: Sequence[str],
    safe_assets: Sequence[str],
    *,
    taker_bps: float = TAKER_BPS,
    top_k: int = 2,
    step: float = 0.25,
    min_leg: float = 0.01,
) -> RotationPlan:
    """The breadth-rule rotation from ``current`` toward the target, priced against real cost.

    Raises :class:`RotationError` for any asset without enough real history — never proposes a
    partial rotation, unlike the reference. Every refusal in the universe is collected first and
    raised together (``violations``), so a universe with three short-history assets names all
    three, not whichever the loop met first.

    ``monthly_closes`` carries no timestamps, so a series that stopped updating months ago cannot
    be told apart from a current one here; :func:`propose_rotation_from_bars` checks that from the
    bars themselves and is the entry point for live data.
    """
    universe = tuple(dict.fromkeys((*risk_assets, *safe_assets)))
    found = parameter_violations(risk_assets, safe_assets, top_k=top_k, step=step)
    scores: dict[str, float] = {}
    for a in universe:
        if a not in monthly_closes:
            found.append(RotationViolation(
                "completeness", a, "no monthly close series supplied"))
            continue
        try:
            scores[a] = momentum_score(monthly_closes[a])
        except RotationError as exc:
            clause = exc.violations[0].clause if exc.violations else "validity"
            found.append(RotationViolation(clause, a, str(exc)))
    if found:
        raise RotationError.from_violations(found)

    target = breadth_allocation(scores, risk_assets, safe_assets, top_k=top_k, step=step)

    trades = tuple(
        Trade(symbol=a, weight_before=current.get(a, 0.0), weight_after=target[a])
        for a in universe
        if abs(target[a] - current.get(a, 0.0)) >= min_leg
    )
    turnover = sum(abs(t.delta) for t in trades)
    is_neg = sum(1 for a in universe if scores[a] < 0)
    return RotationPlan(
        trades=trades, is_neg=is_neg, universe_size=len(universe),
        turnover=turnover, cost_bps=turnover * taker_bps, scores=scores,
    )


def propose_rotation_from_bars(
    current: Mapping[str, float],
    bars_by_symbol: Mapping[str, Sequence[tuple[datetime, float]]],
    risk_assets: Sequence[str],
    safe_assets: Sequence[str],
    *,
    taker_bps: float = TAKER_BPS,
    top_k: int = 2,
    step: float = 0.25,
    min_leg: float = 0.01,
    freshness_days: float = BAR_FRESHNESS_DAYS,
) -> RotationPlan:
    """The live-data entry point: :func:`propose_rotation` from raw ``(close_time, close)`` bars,
    with the whole :data:`CONTRACT_CLAUSES` contract checked first and every violation across
    the universe raised together.

    Every series is cut at the evaluation month (the month of the freshest bar), so all scores
    are computed as of the same month; a series that cannot reach that month is already refused
    by the ``timeliness``/``sufficiency`` clauses.
    """
    universe = tuple(dict.fromkeys((*risk_assets, *safe_assets)))
    found = parameter_violations(risk_assets, safe_assets, top_k=top_k, step=step)
    found += bar_violations(bars_by_symbol, universe, freshness_days=freshness_days)
    if found:
        raise RotationError.from_violations(found)

    # Reordered input is sorted, not refused: with every timestamp unique (checked above) the
    # order carries no information the timestamps do not.
    ordered = {a: sorted(bars_by_symbol[a], key=lambda bar: bar[0]) for a in universe}
    evaluation = max(_month_index(ordered[a][-1][0]) for a in universe)
    monthly = {
        a: monthly_closes_from_bars(
            [(ts, close) for ts, close in ordered[a] if _month_index(ts) <= evaluation])
        for a in universe
    }
    return propose_rotation(current, monthly, risk_assets, safe_assets, taker_bps=taker_bps,
                            top_k=top_k, step=step, min_leg=min_leg)


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import json
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range

    parser = argparse.ArgumentParser(description="risk-on/off across US stocks, crypto, gold")
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--risk", default="NVDAUSDT,AAPLUSDT,QQQUSDT,BTCUSDT,ETHUSDT")
    parser.add_argument("--safe", default="XAUUSDT")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    risk_assets = tuple(args.risk.split(","))
    safe_assets = tuple(args.safe.split(","))
    for symbol in (*risk_assets, *safe_assets):
        if symbol not in RTOKEN_SYMBOLS and not symbol.endswith("USDT"):
            print(f"  {symbol}: not a recognised USDT-margined symbol")
            return 1

    bars_by_symbol: dict[str, list[tuple[datetime, float]]] = {}
    for symbol in (*risk_assets, *safe_assets):
        try:
            bars = fetch_range(symbol, days=args.days, interval="1D",
                                candle_type=CandleType.MARKET)
        except Exception as exc:
            print(f"  {symbol}: no history ({type(exc).__name__})")
            continue
        # Bitget's `ts` is the candle's open time; its close prints BAR_INTERVAL later.
        bars_by_symbol[symbol] = [(b.ts + BAR_INTERVAL, float(b.close)) for b in bars]

    book = dict.fromkeys((*risk_assets, *safe_assets), 1.0 / (len(risk_assets) + len(safe_assets)))
    try:
        plan = propose_rotation_from_bars(book, bars_by_symbol, risk_assets, safe_assets,
                                          top_k=args.top_k)
    except RotationError as exc:
        print("REFUSED:")
        for violation in exc.violations or ():
            print(f"  {violation}")
        if not exc.violations:
            print(f"  {exc}")
        return 1

    print(f"CROSS-ASSET ROTATION — {len(risk_assets)} risk, {len(safe_assets)} safe\n")
    for trade in plan.trades:
        print(f"  {trade.symbol:12} {trade.weight_before:7.2%} -> {trade.weight_after:7.2%}")
    print(f"\n  {plan.verdict}")
    out = Path(__file__).resolve().parents[3] / "data" / "rotation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BAR_FRESHNESS_DAYS",
    "BAR_INTERVAL",
    "CONTRACT_CLAUSES",
    "MIN_MONTHLY_OBSERVATIONS",
    "MOMENTUM_HORIZONS",
    "MOMENTUM_WEIGHT_SUM",
    "TRADING_DAYS_PER_MONTH",
    "WEIGHT_SUM_TOLERANCE",
    "RotationError",
    "RotationPlan",
    "RotationViolation",
    "bar_violations",
    "breadth_allocation",
    "momentum_score",
    "monthly_closes_from_bars",
    "parameter_violations",
    "propose_rotation",
    "propose_rotation_from_bars",
]
