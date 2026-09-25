"""Session-aware execution: a schedule that knows where the anchor market opens and closes, solves
each side of that boundary with that side's own liquidity and volatility, and refuses to solve at
all when it has not measured a side.

**The problem.** A USDT-margined stock perpetual and a spot rToken trade around the clock, but the
market that prices them does not: the anchor's open and close are where the token's volatility and
book depth are expected to change. How much they change on Bitget's books is what
`eval/session_arena.py` measures per segment (its ``segment_table``) once it has been run; that
study has not produced a published artefact yet, so no size of the change is claimed here (NOT
VERIFIED). Almgren & Chriss (2000) derive the optimal schedule *assuming the parameters are
constant over the horizon*. `execution/schedule.py` ports that closed form exactly and could only
flag a crossed boundary (``Trajectory.crosses_session_boundary``, set by the caller); the console
(`lui/research._optimal_schedule`) truncates the schedule at the first boundary and tells the
trader to plan the rest later. Both avoid the wrong answer. Neither gives the right one.

**What this module does instead.** The discrete Almgren-Chriss objective with *time-varying*
parameters is still a convex quadratic in the holdings, and its first-order conditions are still a
tridiagonal system — Almgren's own continuous-time treatment of time-dependent liquidity and
volatility (Almgren, "Optimal trading with stochastic liquidity and volatility", SIAM J. Financial
Math. 3, 2012) reduces, on a grid, to exactly that. With per-interval half-spread cost ``e_k``,
temporary-impact coefficient ``a_k`` and risk weight ``r_k``::

    J(x) = sum_k [ e_k n_k + a_k n_k^2 ]  +  sum_{k=1}^{N-1} r_k x_k^2,     n_k = x_{k-1} - x_k
    r_k  = lambda * step * sigma_{k+1}^2

    FOC: -a_k x_{k-1} + (a_k + a_{k+1} + r_k) x_k - a_{k+1} x_{k+1} = (e_k - e_{k+1}) / 2

With every parameter constant this is Almgren-Chriss's own recurrence, ``cosh(kappa*tau) = 1 +
r/(2a)``, and the solution is their hyperbolic-sine trajectory — :func:`solve` reproduces
`execution.schedule.trajectory` (gamma = 0) to float precision, which is the baseline check in
``tests/test_session_schedule.py``. When the parameters change at a boundary the same system puts
more of the order where it is cheap and holds less inventory where it is risky, and no closed form
is needed: the solver is a primal active-set method over the chain (the no-round-trip constraint
``n_k >= 0`` is enforced exactly, not clipped afterwards), pure Python, O(N) per step.

**Where the boundaries come from.** The anchor calendar is `market/rtoken_spot.py`'s NYSE calendar
by rule (holidays including Good Friday, 13:00 early closes, Ballast MIT, vendored there), resolved
in America/New_York so daylight saving moves the UTC open and close. It is checked minute by minute
against QuantConnect LEAN's market-hours database (`research/repos/Lean-upstream/Data/
market-hours/market-hours-database.json`, Apache-2.0) in the tests. Three rival calendars read at
source get it wrong in ways that move a boundary: `truth/clocks.DualClock` (and so the console)
has no early closes; Egress (`egress/sessions.py:19-40`) has holidays but no early closes;
zz-0816's `session_of` (`common/market_calendar.py`) checks neither holidays nor early closes.

**The refusal is the contract.** If any segment the horizon touches has not been calibrated —
too few observations, or never seen (a holiday the model was not fitted on, say) — :func:`solve`
raises :class:`SessionRefusal` naming the segment and the instant, rather than borrowing the
arrival segment's parameters. Borrowing them is precisely the constant-parameter assumption this
module exists to stop making silently; ``parameters="arrival"`` makes it explicitly, for the
ablation that measures what it costs.

What is NOT modelled, stated rather than assumed away: permanent impact (constant gamma is
path-independent and drops out; a gamma that changes across a boundary would not), the auction
prints at the open and close themselves (the perpetual has none; its anchor does), and any change
in the parameters within a segment beyond what the segment's median captures.
"""

from __future__ import annotations

import functools
import math
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from argus.execution.schedule import ScheduleError, Slice, Trajectory
from argus.market.rtoken_spot import NEW_YORK, close_utc, holidays, is_trading_day, open_utc

OPEN_WINDOW = timedelta(minutes=30)
"""The first half hour of the anchor's regular session is its own segment: the open's volatility
is not the rest of the day's, and averaging them understates the risk of holding into it."""

CLOSE_WINDOW = timedelta(minutes=30)
"""The last half hour, for the same reason in reverse."""

PRE_MARKET_START = time(4, 0)
"""US pre-market opens 04:00 ET (LEAN's ``Equity-usa-[*]`` entry, ``premarket`` 04:00-09:30)."""

POST_MARKET_LENGTH = timedelta(hours=4)
"""US post-market runs four hours past the close: 16:00-20:00 ET, or 13:00-17:00 on an early
close."""

MIN_OBSERVATIONS = 20
"""A segment measured on fewer one-minute observations than this is treated as unmeasured. Twenty
is the floor, not a target: a single training day contributes 30 to each of the half-hour open and
close segments."""

BPS = 1e-4


class SessionRefusal(ScheduleError):
    """The horizon touches a segment this model has not measured, so there is no honest schedule.

    Carries the segment and the first instant it is reached, so the caller can say exactly what is
    missing ("the post-market after an early close has never been calibrated") rather than a
    generic failure.
    """

    def __init__(self, segment: Segment, at: datetime, reason: str) -> None:
        self.segment = segment
        self.at = at
        super().__init__(f"refusing to schedule through {segment.value} at "
                         f"{at.isoformat(timespec='minutes')}: {reason}")


class Segment(StrEnum):
    """Where the anchor market is, for the purpose of pricing a child order on the token.

    Finer than `truth.clocks.SessionPhase`, whose ``EXTENDED`` merges pre- and post-market and
    whose ``RTH`` does not separate the open and close half-hours: each of those pairs is kept apart
    so that a calibration can price them differently, and a model that averages them solves
    through a boundary it cannot see.
    """

    OVERNIGHT = "overnight"
    PRE = "pre"
    OPEN = "open"
    RTH = "rth"
    CLOSE = "close"
    POST = "post"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"

    @property
    def anchor_open(self) -> bool:
        return self in (Segment.OPEN, Segment.RTH, Segment.CLOSE)


# --- the calendar --------------------------------------------------------------------------------


@functools.cache
def _unscheduled_closures() -> frozenset[date]:
    """Closures no rule produces, from LEAN's vendored holiday list (1998-2028): the national
    days of mourning, 2025-01-09 among them. A calendar by rule alone calls those days open —
    found by the minute-by-minute check against LEAN in ``tests/test_session_schedule.py``.
    An unreadable file degrades to the rules, never to "every day is open"."""
    try:
        from argus.eval.baselines.lean_market_holidays_loader import load_usa_equity_holidays

        return load_usa_equity_holidays()
    except Exception:
        return frozenset()


def is_closed_all_day(day: date) -> bool:
    """A weekday on which the anchor does not trade: a rule holiday or a LEAN-listed closure."""
    return day in holidays(day.year) or day in _unscheduled_closures()


def session_bounds(day: date) -> tuple[datetime, datetime] | None:
    """The regular session of ``day`` in UTC, or ``None`` when the anchor does not trade.

    Early closes end at 13:00 ET (`market.rtoken_spot.early_closes`); daylight saving is resolved
    per day by the America/New_York zone, not by a fixed offset.
    """
    if not is_trading_day(day) or is_closed_all_day(day):
        return None
    return open_utc(day), close_utc(day)


def segment_at(instant: datetime) -> Segment:
    """The anchor segment in force at ``instant``. Naive datetimes are refused, not guessed."""
    if instant.tzinfo is None:
        raise ValueError("naive datetime: a session segment needs an instant, not a wall clock")
    local = instant.astimezone(NEW_YORK)
    day = local.date()
    if day.weekday() >= 5:
        return Segment.WEEKEND
    if is_closed_all_day(day):
        return Segment.HOLIDAY
    bounds = session_bounds(day)
    if bounds is None:  # pragma: no cover - holidays and weekends are both handled above
        return Segment.HOLIDAY
    opens, closes = bounds
    if opens <= instant < closes:
        if instant < opens + OPEN_WINDOW:
            return Segment.OPEN
        if instant >= closes - CLOSE_WINDOW:
            return Segment.CLOSE
        return Segment.RTH
    pre_start = datetime.combine(day, PRE_MARKET_START, tzinfo=NEW_YORK)
    if pre_start <= instant < opens:
        return Segment.PRE
    if closes <= instant < closes + POST_MARKET_LENGTH:
        return Segment.POST
    return Segment.OVERNIGHT


@dataclass(frozen=True, slots=True)
class Boundary:
    """A change of segment inside a horizon, at the first minute the new segment is in force."""

    at: datetime
    before: Segment
    after: Segment

    @property
    def is_anchor_open_or_close(self) -> bool:
        """An open or a close of the anchor's regular session, not e.g. pre -> overnight."""
        return self.before.anchor_open != self.after.anchor_open

    def as_dict(self) -> dict[str, object]:
        return {"at": self.at.isoformat(timespec="minutes"), "before": self.before.value,
                "after": self.after.value, "anchor_open_or_close": self.is_anchor_open_or_close}


Segmenter = Callable[[datetime], Segment]
"""Any clock that labels an instant. :func:`segment_at` is the real one; the arena passes a fixed
UTC time-of-day clock through the same code to measure what the wrong calendar costs."""


def segments_over(start: datetime, minutes: int, step_minutes: int = 1, *,
                  segmenter: Segmenter | None = None) -> list[Segment]:
    """The segment in force at the start of each ``step_minutes`` interval of the horizon."""
    if minutes < 1 or step_minutes < 1 or minutes % step_minutes:
        raise ScheduleError(f"horizon {minutes}min is not a positive multiple of the "
                            f"{step_minutes}min step")
    label = segmenter or segment_at
    return [label(start + timedelta(minutes=k * step_minutes))
            for k in range(minutes // step_minutes)]


def boundaries(start: datetime, minutes: int, step_minutes: int = 1, *,
               segmenter: Segmenter | None = None) -> list[Boundary]:
    """Every segment change inside ``[start, start + minutes)``, at step resolution."""
    segs = segments_over(start, minutes, step_minutes, segmenter=segmenter)
    return [Boundary(start + timedelta(minutes=k * step_minutes), segs[k - 1], segs[k])
            for k in range(1, len(segs)) if segs[k] is not segs[k - 1]]


# --- the model -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentParams:
    """What one segment costs and risks, measured rather than assumed.

    ``impact_bps_per_usd`` is the slope ``b`` of a child's walk cost beyond the half-spread,
    ``cost_bps(q) ~= half_spread_bps + b * q`` for a child of ``q`` USDT — which makes the dollar
    cost ``e*q + a*q^2`` with ``a = b * 1e-4``, Almgren-Chriss's linear temporary impact.
    ``sigma_bps`` is the one-minute volatility of the mid, in bps.
    """

    half_spread_bps: float
    impact_bps_per_usd: float
    sigma_bps: float
    observations: int

    def __post_init__(self) -> None:
        for name in ("half_spread_bps", "sigma_bps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ScheduleError(f"{name}={value} must be finite and non-negative")
        if not math.isfinite(self.impact_bps_per_usd) or self.impact_bps_per_usd <= 0:
            raise ScheduleError(
                f"impact_bps_per_usd={self.impact_bps_per_usd} must be positive: with no temporary "
                "impact nothing trades off against risk and the optimum is one block order")

    def as_dict(self) -> dict[str, float | int]:
        return {"half_spread_bps": round(self.half_spread_bps, 4),
                "impact_bps_per_kusd": round(self.impact_bps_per_usd * 1000, 6),
                "sigma_bps": round(self.sigma_bps, 4), "observations": self.observations}


@dataclass(frozen=True, slots=True)
class SessionModel:
    """Per-segment parameters for one instrument, with where they were measured."""

    symbol: str
    venue: str
    fee_bps: float
    params: Mapping[Segment, SegmentParams]
    calibrated_on: tuple[str, ...] = ()

    def lookup(self, segment: Segment, at: datetime) -> SegmentParams:
        """The segment's parameters, or a :class:`SessionRefusal` saying why there are none."""
        found = self.params.get(segment)
        if found is None:
            raise SessionRefusal(segment, at, f"{self.symbol} has never been measured in the "
                                              f"{segment.value} segment")
        if found.observations < MIN_OBSERVATIONS:
            raise SessionRefusal(segment, at, f"{self.symbol} has {found.observations} "
                                              f"observation(s) of {segment.value}, fewer than "
                                              f"{MIN_OBSERVATIONS}")
        return found

    def as_dict(self) -> dict[str, object]:
        return {"symbol": self.symbol, "venue": self.venue, "fee_bps": self.fee_bps,
                "calibrated_on": list(self.calibrated_on),
                "segments": {s.value: p.as_dict() for s, p in sorted(
                    self.params.items(), key=lambda kv: list(Segment).index(kv[0]))}}


# --- the schedule --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionSchedule:
    """An executable schedule: one child per step, each labelled with the segment it trades in."""

    start: datetime
    step_minutes: int
    quantity_usd: float
    trades: tuple[float, ...]
    segments: tuple[Segment, ...]
    boundaries: tuple[Boundary, ...]
    risk_aversion: float
    expected_cost_usd: float
    variance_usd2: float
    parameters: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def children(self) -> list[tuple[int, float]]:
        """``(minutes after start, notional)`` for every non-empty child, in order."""
        return [(k * self.step_minutes, q) for k, q in enumerate(self.trades) if q > 1e-9]

    def share_by_segment(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for seg, q in zip(self.segments, self.trades, strict=True):
            out[seg.value] = out.get(seg.value, 0.0) + q / self.quantity_usd
        return {k: round(v, 6) for k, v in out.items()}

    @property
    def front_loading(self) -> float:
        """Share done in the first half of the horizon; 0.5 is TWAP."""
        half = len(self.trades) / 2
        return sum(q for k, q in enumerate(self.trades) if k < half) / self.quantity_usd

    @property
    def crosses_anchor_boundary(self) -> bool:
        return any(b.is_anchor_open_or_close for b in self.boundaries)

    def as_trajectory(self, price: Decimal) -> Trajectory:
        """The schedule in shares, as `execution.schedule.Trajectory`, so the venue's lot rules
        apply through `execution.schedule.quantise_trajectory` unchanged — the Nautilus-derived
        floor-and-carry quantiser, not a second one. ``kappa`` is recorded as zero because a
        time-varying schedule has no single one; ``is_twap`` must not be read off it."""
        if price <= 0:
            raise ScheduleError(f"price={price} must be positive")
        shares = [Decimal(str(q)) / price for q in self.trades]
        total = sum(shares, Decimal(0))
        remaining = total
        slices = []
        for k, q in enumerate(shares):
            remaining -= q
            slices.append(Slice(index=k + 1, start=Decimal(k * self.step_minutes), quantity=q,
                                remaining=max(remaining, Decimal(0))))
        return Trajectory(
            total_quantity=total, horizon=Decimal(len(shares) * self.step_minutes),
            slices=tuple(slices), expected_cost=Decimal(str(self.expected_cost_usd)) / price,
            variance=Decimal(str(self.variance_usd2)) / (price * price), kappa=Decimal(0),
            risk_aversion=Decimal(str(self.risk_aversion)),
            crosses_session_boundary=self.crosses_anchor_boundary)

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.start.isoformat(timespec="minutes"), "step_minutes": self.step_minutes,
            "quantity_usd": self.quantity_usd, "parameters": self.parameters,
            "risk_aversion": self.risk_aversion,
            "expected_cost_bps": round(self.expected_cost_usd / self.quantity_usd / BPS, 4),
            "risk_bps": round(math.sqrt(max(self.variance_usd2, 0.0)) / self.quantity_usd / BPS,
                              4),
            "front_loading": round(self.front_loading, 4),
            "share_by_segment": self.share_by_segment(),
            "boundaries": [b.as_dict() for b in self.boundaries],
            "notes": list(self.notes),
        }


def decay_to_risk_aversion(decay: float, horizon_steps: int, params: SegmentParams,
                           step_minutes: int = 1) -> float:
    """The lambda at which a constant-parameter schedule under ``params`` decays inventory by
    ``exp(-decay)`` over the horizon — the console's own way of stating risk aversion
    (`lui/research.SCHEDULE_DECAYS`: e-fold by default, e-cubed when urgent).

    From the discrete recurrence ``cosh(kappa*tau) = 1 + r / (2a)`` with ``kappa*tau = decay / N``:
    ``r = 2a(cosh(decay/N) - 1)`` and ``r = lambda * step * (sigma * 1e-4)^2``.
    """
    if decay < 0:
        raise ScheduleError(f"decay={decay} cannot be negative")
    if decay == 0 or params.sigma_bps == 0:
        return 0.0
    a = params.impact_bps_per_usd * BPS
    r = 2.0 * a * (math.cosh(decay / horizon_steps) - 1.0)
    return r / (step_minutes * (params.sigma_bps * BPS) ** 2)


def solve(
    quantity_usd: float,
    start: datetime,
    horizon_minutes: int,
    model: SessionModel,
    *,
    step_minutes: int = 1,
    decay: float | None = None,
    risk_aversion: float | None = None,
    parameters: Literal["session", "arrival"] = "session",
    segmenter: Segmenter | None = None,
) -> SessionSchedule:
    """The optimal schedule for ``quantity_usd`` over the horizon, one child per step.

    ``parameters="session"`` prices each step with its own segment's parameters and refuses
    (:class:`SessionRefusal`) if any segment is unmeasured. ``"arrival"`` prices every step with
    the arrival segment's — the constant-parameter assumption, kept only so the ablation can
    measure it. Risk is stated either as ``decay`` (converted with the *arrival* segment's
    parameters, so both modes share one lambda and differ only in the parameters) or as an
    absolute ``risk_aversion`` in 1/USD.
    """
    if not math.isfinite(quantity_usd) or quantity_usd <= 0:
        raise ScheduleError(f"quantity_usd={quantity_usd} must be positive")
    if start.tzinfo is None:
        raise ScheduleError("start must be timezone-aware")
    if (decay is None) == (risk_aversion is None):
        raise ScheduleError("state risk as exactly one of decay or risk_aversion")
    segs = segments_over(start, horizon_minutes, step_minutes, segmenter=segmenter)
    n = len(segs)
    arrival = model.lookup(segs[0], start)
    if parameters == "session":
        per_step = [model.lookup(seg, start + timedelta(minutes=k * step_minutes))
                    for k, seg in enumerate(segs)]
    elif parameters == "arrival":
        per_step = [arrival] * n
    else:
        raise ScheduleError(f"parameters={parameters!r} is not 'session' or 'arrival'")

    lam = (decay_to_risk_aversion(decay, n, arrival, step_minutes) if decay is not None
           else float(risk_aversion or 0.0))
    if lam < 0 or not math.isfinite(lam):
        raise ScheduleError(f"risk_aversion={lam} must be finite and non-negative")

    e = [p.half_spread_bps * BPS for p in per_step]
    a = [p.impact_bps_per_usd * BPS for p in per_step]
    # Holding x_k over interval k+1 carries that interval's volatility.
    r = [lam * step_minutes * (per_step[k + 1].sigma_bps * BPS) ** 2 for k in range(n - 1)]
    holdings = optimal_holdings(quantity_usd, e, a, r)
    raw = [max(0.0, holdings[k] - holdings[k + 1]) for k in range(n)]
    total = sum(raw)
    trades = tuple(q * quantity_usd / total for q in raw)
    fee = model.fee_bps * BPS
    cost = sum((e[k] + fee) * trades[k] + a[k] * trades[k] ** 2 for k in range(n))
    variance = sum(step_minutes * (per_step[k + 1].sigma_bps * BPS) ** 2 * holdings[k + 1] ** 2
                   for k in range(n - 1))
    crossed = boundaries(start, horizon_minutes, step_minutes, segmenter=segmenter)
    notes: list[str] = []
    if parameters == "arrival" and any(b.is_anchor_open_or_close for b in crossed):
        notes.append("constant arrival-segment parameters used across an anchor open or close")
    return SessionSchedule(
        start=start, step_minutes=step_minutes, quantity_usd=quantity_usd, trades=trades,
        segments=tuple(segs), boundaries=tuple(crossed), risk_aversion=lam,
        expected_cost_usd=cost, variance_usd2=variance, parameters=parameters,
        notes=tuple(notes))


# --- the solver ----------------------------------------------------------------------------------


def optimal_holdings(quantity: float, e: Sequence[float], a: Sequence[float],
                     r: Sequence[float], *, tol: float = 1e-10,
                     max_iterations: int | None = None) -> list[float]:
    """Minimise ``sum e_k n_k + a_k n_k^2 + sum r_k x_k^2`` over holdings ``x_0 = quantity >=
    x_1 >= ... >= x_N = 0`` (every trade non-negative: no buying back what was just sold).

    A primal active-set method, started from the even split (always feasible): solve the
    equality problem with the working set's trades pinned at zero, step toward it until a trade
    would turn negative and pin that one, and when the equality optimum is reached release the
    pinned trade with the most negative multiplier. For a strictly convex quadratic (every
    ``a_k > 0``) this terminates at the exact constrained optimum in finitely many steps (Nocedal &
    Wright, *Numerical Optimization*, 2nd ed., Algorithm 16.3). Each equality solve merges pinned
    intervals into a single level and is one tridiagonal (Thomas) solve.
    """
    n = len(e)
    if n < 1 or len(a) != n or len(r) != n - 1:
        raise ScheduleError(f"need N costs and impacts and N-1 risk weights, got "
                            f"{len(e)}, {len(a)}, {len(r)}")
    if any(not math.isfinite(v) or v <= 0 for v in a):
        raise ScheduleError("every temporary-impact coefficient must be positive and finite")
    if any(not math.isfinite(v) or v < 0 for v in r):
        raise ScheduleError("risk weights must be finite and non-negative")
    if n == 1:
        return [quantity, 0.0]
    scale = quantity
    x = [scale * (n - k) / n for k in range(n + 1)]
    pinned: set[int] = set()  # interval k (1-based) pinned at zero trade
    limit = max_iterations or 20 * n + 50
    tol_q = tol * scale
    for _ in range(limit):
        target = _equality_optimum(quantity, e, a, r, pinned)
        d = [target[i] - x[i] for i in range(n + 1)]
        if max(abs(v) for v in d) <= tol_q:
            x = target
            mu = _multipliers(x, e, a, r, pinned)
            worst = min(pinned, key=lambda k: mu[k], default=None)
            if worst is None or mu[worst] >= -tol * max(1.0, max(abs(v) for v in e)):
                return [max(0.0, v) if abs(v) > tol_q else 0.0 for v in x]
            pinned.discard(worst)
            continue
        alpha, block = 1.0, None
        for k in range(1, n + 1):
            if k in pinned:
                continue
            dn = d[k - 1] - d[k]
            if dn < -tol_q:
                step = (x[k - 1] - x[k]) / -dn
                if step < alpha:
                    alpha, block = step, k
        x = [x[i] + alpha * d[i] for i in range(n + 1)]
        x[0], x[n] = quantity, 0.0
        if block is not None:
            pinned.add(block)
    raise ScheduleError(f"active-set solver did not converge in {limit} iterations")


def _levels(n: int, pinned: set[int]) -> list[list[int]]:
    """Nodes 0..N grouped into levels joined by pinned (zero-trade) intervals."""
    levels: list[list[int]] = [[0]]
    for node in range(1, n + 1):
        if node in pinned:  # interval `node` joins node-1 and node
            levels[-1].append(node)
        else:
            levels.append([node])
    return levels


def _equality_optimum(quantity: float, e: Sequence[float], a: Sequence[float],
                      r: Sequence[float], pinned: set[int]) -> list[float]:
    n = len(e)
    levels = _levels(n, pinned)
    if len(levels) < 2:
        raise ScheduleError("every interval is pinned at zero; the order could not trade")
    fixed: dict[int, float] = {}
    for j, members in enumerate(levels):
        if 0 in members:
            fixed[j] = quantity
        if n in members:
            if j in fixed:
                raise ScheduleError("start and end holdings are tied; the order could not trade")
            fixed[j] = 0.0
    # Edge j joins level j-1 and level j through interval levels[j][0].
    edge_a = [0.0] + [a[levels[j][0] - 1] for j in range(1, len(levels))]
    edge_e = [0.0] + [e[levels[j][0] - 1] for j in range(1, len(levels))]
    risk = [sum(r[node - 1] for node in members if 1 <= node <= n - 1) for members in levels]
    free = [j for j in range(len(levels)) if j not in fixed]
    y = [fixed.get(j, 0.0) for j in range(len(levels))]
    if free:
        # Tridiagonal system over the free levels (they are contiguous: level 0 and the last level
        # are the fixed ones).
        sub, diag, sup, rhs = [], [], [], []
        for j in free:
            left, right = edge_a[j], edge_a[j + 1]
            b = (edge_e[j] - edge_e[j + 1]) / 2.0
            if j - 1 in fixed:
                b += left * fixed[j - 1]
            if j + 1 in fixed:
                b += right * fixed[j + 1]
            sub.append(-left)
            diag.append(left + right + risk[j])
            sup.append(-right)
            rhs.append(b)
        solved = _thomas(sub, diag, sup, rhs)
        for j, v in zip(free, solved, strict=True):
            y[j] = v
    out = [0.0] * (n + 1)
    for j, members in enumerate(levels):
        for node in members:
            out[node] = y[j]
    return out


def _thomas(sub: list[float], diag: list[float], sup: list[float],
            rhs: list[float]) -> list[float]:
    m = len(diag)
    c = [0.0] * m
    d = [0.0] * m
    c[0] = sup[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, m):
        denom = diag[i] - sub[i] * c[i - 1]
        c[i] = sup[i] / denom
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / denom
    out = [0.0] * m
    out[-1] = d[-1]
    for i in range(m - 2, -1, -1):
        out[i] = d[i] - c[i] * out[i + 1]
    return out


def _multipliers(x: Sequence[float], e: Sequence[float], a: Sequence[float],
                 r: Sequence[float], pinned: set[int]) -> dict[int, float]:
    """KKT multipliers of the pinned constraints ``x_k - x_{k-1} <= 0`` at the equality optimum.

    Stationarity at every free node ``i`` is ``G_i + mu_i - mu_{i+1} = 0`` with ``mu = 0`` on
    unpinned intervals, so inside a level the multipliers are running sums of the gradient, started
    from whichever end of the level is an unpinned interval.
    """
    n = len(e)

    def grad(i: int) -> float:
        return (e[i] - e[i - 1] + 2 * a[i - 1] * (x[i] - x[i - 1])
                + 2 * a[i] * (x[i] - x[i + 1]) + 2 * r[i - 1] * x[i])

    mu: dict[int, float] = {}
    for members in _levels(n, pinned):
        if len(members) == 1:
            continue
        if members[0] == 0:
            # Anchored at the fixed start: run backwards from the unpinned interval after it.
            nxt = 0.0
            for node in reversed(members[1:]):
                value = nxt - grad(node)
                mu[node] = value
                nxt = value
        else:
            prev = 0.0
            # Unpinned interval enters the level at its first node; the last node's equation is
            # the level's own optimality (or node N, which is fixed and has none).
            for node in members[:-1]:
                value = prev + grad(node)
                mu[node + 1] = value
                prev = value
    return mu


# --- calibration ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookObservation:
    """One book, as a list of ``(price, quantity)`` levels best-first per side."""

    at: datetime
    bids: Sequence[tuple[float, float]]
    asks: Sequence[tuple[float, float]]

    @property
    def mid(self) -> float:
        return (self.bids[0][0] + self.asks[0][0]) / 2.0


CALIBRATION_SIZES = (1_000.0, 5_000.0, 25_000.0)
"""Child sizes the impact slope is fitted over, in USDT: a one-minute child of a $250k order over
an hour is ~$4k, and one over four hours ~$1k."""


def walk_cost_bps(levels: Sequence[tuple[float, float]], notional: float,
                  mid: float) -> float | None:
    """What taking ``notional`` from these levels costs against the mid, in bps; ``None`` if the
    visible book cannot fill it."""
    left, shares, spent = notional, 0.0, 0.0
    for price, qty in levels:
        take = min(qty, left / price)
        shares += take
        spent += take * price
        left -= take * price
        if left <= 1e-9:
            break
    if left > 1e-6 or shares <= 0:
        return None
    return abs(spent / shares - mid) / mid / BPS


def calibrate(observations: Iterable[BookObservation], *, symbol: str, venue: str,
              fee_bps: float, calibrated_on: tuple[str, ...] = (),
              sizes: Sequence[float] = CALIBRATION_SIZES,
              segmenter: Segmenter | None = None) -> SessionModel:
    """Measure every segment's half-spread, impact slope and one-minute volatility.

    Half-spread and slope are medians over the segment's minutes (robust to the odd crossed or
    stale book); the slope at each minute is the least-squares fit through the origin of
    ``walk_cost - half_spread`` on child size, averaged over both sides. Volatility is the root mean
    square of one-minute log mid changes, counted in the segment of the later minute — so the jump
    at the open belongs to the open. A segment with fewer than :data:`MIN_OBSERVATIONS` minutes is
    left out, which makes :func:`solve` refuse it rather than guess.
    """
    spreads: dict[Segment, list[float]] = {}
    slopes: dict[Segment, list[float]] = {}
    moves: dict[Segment, list[float]] = {}
    previous: BookObservation | None = None
    for obs in observations:
        if not obs.bids or not obs.asks or obs.bids[0][0] >= obs.asks[0][0]:
            previous = None
            continue
        seg = (segmenter or segment_at)(obs.at)
        mid = obs.mid
        half = (obs.asks[0][0] - obs.bids[0][0]) / 2.0 / mid / BPS
        spreads.setdefault(seg, []).append(half)
        fits = []
        for side in (obs.asks, obs.bids):
            pts = [(q, c - half) for q in sizes
                   if (c := walk_cost_bps(side, q, mid)) is not None]
            denom = sum(q * q for q, _ in pts)
            if pts and denom > 0:
                fits.append(max(sum(q * y for q, y in pts) / denom, 0.0))
        if fits:
            slopes.setdefault(seg, []).append(sum(fits) / len(fits))
        if previous is not None and obs.at - previous.at == timedelta(minutes=1):
            moves.setdefault(seg, []).append(math.log(mid / previous.mid) / BPS)
        previous = obs
    params: dict[Segment, SegmentParams] = {}
    for seg, values in spreads.items():
        seg_moves = moves.get(seg, [])
        seg_slopes = [s for s in slopes.get(seg, []) if s > 0]
        count = min(len(values), len(seg_moves), len(seg_slopes))
        if count < 2:
            continue
        params[seg] = SegmentParams(
            half_spread_bps=statistics.median(values),
            impact_bps_per_usd=statistics.median(seg_slopes),
            sigma_bps=math.sqrt(sum(m * m for m in seg_moves) / len(seg_moves)),
            observations=count)
    return SessionModel(symbol=symbol, venue=venue, fee_bps=fee_bps, params=params,
                        calibrated_on=calibrated_on)


def load_model(blob: Mapping[str, Any]) -> SessionModel:
    """Inverse of :meth:`SessionModel.as_dict`."""
    segments = blob.get("segments")
    if not isinstance(segments, Mapping):
        raise ScheduleError("a session model needs a 'segments' mapping")
    params: dict[Segment, SegmentParams] = {}
    for name, row in segments.items():
        if not isinstance(row, Mapping):
            raise ScheduleError(f"segment {name!r} is not a mapping")
        params[Segment(str(name))] = SegmentParams(
            half_spread_bps=float(row["half_spread_bps"]),
            impact_bps_per_usd=float(row["impact_bps_per_kusd"]) / 1000.0,
            sigma_bps=float(row["sigma_bps"]), observations=int(row["observations"]))
    calibrated = blob.get("calibrated_on", ())
    fee = blob.get("fee_bps")
    if not isinstance(fee, (int, float)):
        raise ScheduleError("a session model needs a numeric 'fee_bps'")
    return SessionModel(
        symbol=str(blob["symbol"]), venue=str(blob["venue"]), fee_bps=float(fee), params=params,
        calibrated_on=tuple(str(d) for d in calibrated) if isinstance(calibrated, list) else ())


__all__ = [
    "CALIBRATION_SIZES",
    "MIN_OBSERVATIONS",
    "BookObservation",
    "Boundary",
    "Segment",
    "SegmentParams",
    "Segmenter",
    "SessionModel",
    "SessionRefusal",
    "SessionSchedule",
    "boundaries",
    "calibrate",
    "decay_to_risk_aversion",
    "is_closed_all_day",
    "load_model",
    "optimal_holdings",
    "segment_at",
    "segments_over",
    "session_bounds",
    "solve",
    "walk_cost_bps",
]
