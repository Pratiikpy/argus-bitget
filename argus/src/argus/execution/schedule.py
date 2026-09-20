"""Optimal execution trajectories — Almgren & Chriss (2000), *Optimal Execution of Portfolio
Transactions*, Journal of Risk 3(2), 5-39.

The goal file names this paper as the reference for Track 3's Execution Assistance sub-theme, and no
repository in the corpus owns it. What it settles is a question a fixed slice schedule cannot even
ask: **how fast should a large order be worked?**

Trading fast pays impact. Trading slow pays risk — the price wanders while you are still holding
inventory. Almgren and Chriss show the efficient frontier between the two has a closed form, and
that the optimal trajectory is a *hyperbolic sine* decay of remaining inventory, not a straight
line. A linear schedule (TWAP) is the special case at zero risk aversion, and it is what almost
every execution tool ships as its default.

The model, in the paper's own notation:

    permanent impact   g(v) = gamma * v            (moves the price for everyone, permanently)
    temporary impact   h(v) = epsilon*sgn(v) + eta*v   (the price *you* pay, spread plus slippage)
    eta_tilde = eta - gamma*tau/2
    kappa_tilde^2 = lambda * sigma^2 / eta_tilde
    cosh(kappa*tau) = 1 + kappa_tilde^2 * tau^2 / 2

    holdings   x_j = sinh(kappa*(T - t_j)) / sinh(kappa*T) * X
    trades     n_j = x_{j-1} - x_j

    E[cost] = gamma*X^2/2 + epsilon*sum|n_j| + (eta_tilde/tau)*sum n_j^2
    V[cost] = sigma^2 * tau * sum x_j^2

**Where this bites on our venue specifically.** Almgren-Chriss assumes volatility and liquidity are
constant over the trading horizon. On an rToken they are emphatically not: the anchor market opens
and closes inside a plausible execution window, and depth changes by roughly 3x across that
boundary. A trajectory computed through a session change is solving the wrong problem, so
:func:`trajectory` takes the horizon it is given and :class:`Trajectory` reports whether that
horizon crosses one — refusing to hide the assumption that makes the answer valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import acosh, sinh

_ZERO = Decimal("0")


class ScheduleError(ValueError):
    """The trajectory was asked for something the model cannot answer."""


@dataclass(frozen=True, slots=True)
class ImpactParameters:
    """The four parameters Almgren-Chriss needs, in units of price per share traded.

    Naming follows the paper rather than our cost model, because two different ``gamma``s in one
    file is how the wrong one gets used: here ``gamma`` is *permanent impact*, while
    :attr:`argus.cost.model.CostModel.gamma` is the *impact exponent*. They are unrelated.
    """

    sigma: Decimal
    """Volatility per unit time, in price units. Same clock as the horizon."""

    gamma: Decimal
    """Permanent impact per share. The part of the move that does not come back."""

    eta: Decimal
    """Temporary impact per share per unit rate. The part you pay and others do not."""

    epsilon: Decimal
    """Fixed cost per share — half the spread plus fees. Paid on every slice regardless of size."""

    def __post_init__(self) -> None:
        for name in ("sigma", "gamma", "eta", "epsilon"):
            if getattr(self, name) < 0:
                raise ScheduleError(f"{name} cannot be negative")
        if self.eta <= 0:
            raise ScheduleError(
                "eta must be positive: with no temporary impact there is nothing to trade off "
                "against risk, and the optimal schedule degenerates to trading everything at once"
            )


@dataclass(frozen=True, slots=True)
class Slice:
    """One interval of the trajectory."""

    index: int
    start: Decimal
    """Time at the start of the interval, in the same units as the horizon."""

    quantity: Decimal
    """Shares traded during this interval."""

    remaining: Decimal
    """Shares still held at the end of it."""

    @property
    def rate(self) -> Decimal:
        return self.quantity


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A complete execution schedule, with both halves of the trade-off priced."""

    total_quantity: Decimal
    horizon: Decimal
    slices: tuple[Slice, ...]
    expected_cost: Decimal
    variance: Decimal
    kappa: Decimal
    risk_aversion: Decimal
    crosses_session_boundary: bool = False
    """Whether the horizon spans an anchor open or close.

    When true the constant-volatility and constant-liquidity assumptions behind this trajectory do
    not hold, and the schedule should be recomputed per session rather than trusted across the
    boundary. Reported rather than silently corrected, because correcting it would require a
    liquidity forecast we do not have.
    """

    @property
    def is_twap(self) -> bool:
        """Is this effectively a straight line?

        At zero risk aversion Almgren-Chriss reduces to TWAP. Naming that case makes the difference
        between "we chose TWAP" and "we shipped the default" visible.
        """
        return self.kappa < Decimal("1e-9")

    @property
    def front_loading(self) -> Decimal:
        """Share of the order done in the first half of the horizon.

        0.5 is TWAP. Above it means risk aversion is pulling the trade forward, and how far above
        is the whole content of the risk-aversion choice.
        """
        if self.total_quantity <= 0 or not self.slices:
            return _ZERO
        half = self.horizon / 2
        done = sum((s.quantity for s in self.slices if s.start < half), _ZERO)
        return done / self.total_quantity

    def cost_per_share(self) -> Decimal:
        if self.total_quantity <= 0:
            return _ZERO
        return self.expected_cost / self.total_quantity

    def as_dict(self) -> dict[str, object]:
        return {
            "total_quantity": str(self.total_quantity),
            "horizon": str(self.horizon),
            "intervals": len(self.slices),
            "kappa": str(round(self.kappa, 6)),
            "risk_aversion": str(self.risk_aversion),
            "is_twap": self.is_twap,
            "front_loading": str(round(self.front_loading, 4)),
            "expected_cost": str(round(self.expected_cost, 6)),
            "variance": str(round(self.variance, 6)),
            "crosses_session_boundary": self.crosses_session_boundary,
            "slices": [
                {"index": s.index, "start": str(s.start), "quantity": str(round(s.quantity, 6))}
                for s in self.slices
            ],
        }


def trajectory(
    *,
    quantity: Decimal,
    horizon: Decimal,
    intervals: int,
    impact: ImpactParameters,
    risk_aversion: Decimal = Decimal("1e-6"),
    crosses_session_boundary: bool = False,
) -> Trajectory:
    """The Almgren-Chriss optimal trajectory.

    ``risk_aversion`` is the paper's ``lambda``: the rate at which variance of execution cost is
    traded against its mean. Zero gives TWAP. Larger values front-load the order, paying more impact
    to hold inventory for less time.

    Everything is computed in :class:`~decimal.Decimal` except the hyperbolic functions themselves,
    which have no exact decimal form — those cross to float and come straight back, so the
    conversion is one bounded step rather than a float pipeline.
    """
    if quantity <= 0:
        raise ScheduleError(f"quantity={quantity} must be positive")
    if horizon <= 0:
        raise ScheduleError(f"horizon={horizon} must be positive")
    if intervals < 1:
        raise ScheduleError(f"intervals={intervals} must be at least 1")
    if risk_aversion < 0:
        raise ScheduleError(
            f"risk_aversion={risk_aversion} cannot be negative: a negative lambda would *seek* "
            "execution risk, which is not a preference this model can express"
        )

    tau = horizon / Decimal(intervals)

    # eta_tilde = eta - gamma*tau/2. The paper's correction for the half-interval of permanent
    # impact already absorbed into the temporary term.
    eta_tilde = impact.eta - impact.gamma * tau / 2
    if eta_tilde <= 0:
        raise ScheduleError(
            f"eta_tilde={eta_tilde} is not positive: permanent impact over one interval exceeds "
            "temporary impact, so the model's decomposition has broken down. Use more intervals "
            "(shorter tau) or re-estimate the impact parameters."
        )

    kappa = _kappa(risk_aversion=risk_aversion, impact=impact, eta_tilde=eta_tilde, tau=tau)
    holdings = _holdings(quantity, horizon, tau, intervals, kappa)

    slices: list[Slice] = []
    for j in range(1, intervals + 1):
        slices.append(
            Slice(
                index=j,
                start=tau * Decimal(j - 1),
                quantity=holdings[j - 1] - holdings[j],
                remaining=holdings[j],
            )
        )

    # E = gamma*X^2/2 + epsilon*sum|n| + (eta_tilde/tau)*sum n^2
    expected = (
        impact.gamma * quantity * quantity / 2
        + impact.epsilon * sum((abs(s.quantity) for s in slices), _ZERO)
        + (eta_tilde / tau) * sum((s.quantity * s.quantity for s in slices), _ZERO)
    )
    # V = sigma^2 * tau * sum x_j^2, over holdings *during* each interval.
    variance = (
        impact.sigma * impact.sigma * tau * sum((h * h for h in holdings[1:]), _ZERO)
    )

    return Trajectory(
        total_quantity=quantity,
        horizon=horizon,
        slices=tuple(slices),
        expected_cost=expected,
        variance=variance,
        kappa=kappa,
        risk_aversion=risk_aversion,
        crosses_session_boundary=crosses_session_boundary,
    )


def _kappa(
    *, risk_aversion: Decimal, impact: ImpactParameters, eta_tilde: Decimal, tau: Decimal
) -> Decimal:
    """Solve ``cosh(kappa*tau) = 1 + kappa_tilde^2 * tau^2 / 2`` for kappa.

    Almgren-Chriss give the exact inversion, so there is no root-finding here: taking ``arccosh``
    of the right-hand side and dividing by tau is the answer.
    """
    if risk_aversion == 0 or impact.sigma == 0:
        return _ZERO  # TWAP: no reason to hurry.

    kappa_tilde_sq = risk_aversion * impact.sigma * impact.sigma / eta_tilde
    rhs = 1 + float(kappa_tilde_sq * tau * tau) / 2
    if rhs <= 1:
        return _ZERO
    return Decimal(str(acosh(rhs))) / tau


def _holdings(
    quantity: Decimal, horizon: Decimal, tau: Decimal, intervals: int, kappa: Decimal
) -> list[Decimal]:
    """``x_j = sinh(kappa*(T - t_j)) / sinh(kappa*T) * X``, with the TWAP limit handled exactly.

    The limit matters: as kappa goes to zero both sinh terms go to zero and the ratio is 0/0. Taken
    as a limit it is the straight line ``(T - t)/T``, which is TWAP — so the degenerate case is
    computed rather than guarded against with an epsilon that would quietly bias small-kappa
    schedules.
    """
    out = [quantity]
    if kappa == 0:
        for j in range(1, intervals + 1):
            out.append(quantity * (horizon - tau * Decimal(j)) / horizon)
        return out

    # CPython's math.sinh *raises* OverflowError rather than returning inf, so a risk aversion high
    # enough to saturate the float range would crash rather than degrade. It is a legitimate
    # preference — "do not hold this inventory for any measurable time" — and the answer is
    # immediate execution, not an exception.
    try:
        denom = sinh(float(kappa * horizon))
    except OverflowError:
        return [quantity] + [_ZERO] * intervals
    if denom == 0 or not _is_finite(denom):
        return [quantity] + [_ZERO] * intervals

    for j in range(1, intervals + 1):
        t = tau * Decimal(j)
        try:
            numer = sinh(float(kappa * (horizon - t)))
        except OverflowError:  # pragma: no cover - denom would have overflowed first
            numer = float("inf")
        out.append(quantity * Decimal(str(numer / denom)))
    return out


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def twap(*, quantity: Decimal, horizon: Decimal, intervals: int) -> Trajectory:
    """The straight line, for comparison.

    Provided so a trajectory can be scored *against* the default rather than merely described. A
    schedule that does not beat TWAP on the chosen risk preference is not worth its complexity.
    """
    return trajectory(
        quantity=quantity,
        horizon=horizon,
        intervals=intervals,
        impact=ImpactParameters(
            sigma=_ZERO, gamma=_ZERO, eta=Decimal("1"), epsilon=_ZERO
        ),
        risk_aversion=_ZERO,
    )


@dataclass(frozen=True, slots=True)
class QuantizedTrajectory:
    """A :class:`Trajectory` re-cut to real, tradeable lot sizes.

    Closes two real gaps `eval.schedule_comparison` found reading
    ``nautechsystems/nautilus_trader``'s real ``TwapAlgorithm`` (``crates/trading/src/algorithm/
    twap.rs``), not silently: (1) it floors every child order to the instrument's size increment
    and schedules the shortfall as an extra final slice (``twap.rs:220-309``) — generalised here
    from TWAP's equal slices to Almgren-Chriss's unequal ones; (2) when a slice would floor to
    **zero or below the venue's own minimum order size**, it does not schedule an unplaceable
    order — it merges that quantity forward rather than emitting it (``twap.rs``'s own tests name
    this exactly: ``test_twap_submits_entire_size_when_qty_per_interval_below_size_increment``,
    ``..._below_min_quantity``). **Found by testing an extreme case, not assumed away**: an early
    version of this function passed every unit test yet still emitted 18 zero-quantity "slices"
    for a heavily front-loaded, high-risk-aversion schedule — a nonsensical, unplaceable order
    that summed correctly but was not a real answer. Fixed with the carry-forward merge below.
    """

    slices: tuple[Decimal, ...]
    """Tradeable quantities. Every entry is an exact multiple of ``quantity_multiplier`` and
    (when ``min_order_qty`` is supplied) at or above it — never zero, never emitted."""

    quantity_multiplier: Decimal
    min_order_qty: Decimal
    remainder_slice_added: bool
    """Whether the final slice is a carried-forward merge of quantity that could not stand on its
    own, rather than the trajectory's own last interval unchanged."""

    @property
    def total(self) -> Decimal:
        return sum(self.slices, _ZERO)

    def as_dict(self) -> dict[str, object]:
        return {
            "slices": [str(s) for s in self.slices],
            "quantity_multiplier": str(self.quantity_multiplier),
            "min_order_qty": str(self.min_order_qty),
            "remainder_slice_added": self.remainder_slice_added,
            "total": str(self.total),
        }


def quantise_trajectory(
    traj: Trajectory, *, quantity_multiplier: Decimal, min_order_qty: Decimal = _ZERO,
) -> QuantizedTrajectory:
    """Floor every slice to a real tradeable size, carrying forward anything too small to stand
    on its own — matching Nautilus's real TWAP algorithm's own approach (see
    :class:`QuantizedTrajectory`'s docstring for the two gaps this closes and the file:line each
    is read from), generalised from TWAP's equal slices to Almgren-Chriss's unequal ones.

    Requires ``traj.total_quantity`` to already be an exact multiple of ``quantity_multiplier``.
    That is not a limitation invented for this function: a real order's own TOTAL size is
    validated against the same instrument rule (``execution.guard.validate``) before it is ever
    scheduled, so a total that is not already a clean multiple is a bug further up the pipeline,
    not something this function should silently repair by inventing a rounding rule nobody asked
    for. Raising here is what makes that upstream bug visible instead of absorbed.

    **The algorithm, in words — and the bug an earlier version of it had.** A first draft floored
    each raw slice independently, *then* carried the floored (already-truncated) values forward to
    consolidate too-small ones. That silently discarded each slice's own fractional remainder at
    the moment it was floored, before the carry ever saw it — caught by testing the total-
    preservation property on the SAME extreme case that exposed the zero-slice bug, not by
    inspection. The fix tracks the running RAW (un-truncated) cumulative total instead: at each
    interval, ``available`` is however much has accumulated since the last emitted slice, at full
    precision; it is floored to ``quantity_multiplier`` only at the moment of checking whether it
    now reaches ``min_order_qty``. Emitting resets what counts as "since the last slice", but never
    discards the fractional part that was not emitted — it simply stays inside ``available`` for
    the next interval to pick up. A run of intervals too small to trade on their own therefore
    becomes exactly one correctly-sized order once their sum crosses the floor, and nothing is
    ever lost to a truncation nobody carried forward. If nothing in the whole trajectory ever
    reaches the floor, the entire order goes out as a single slice — Nautilus's own stated
    fallback for both cases its test suite names.
    """
    from argus.execution.guard import quantise_down

    if quantity_multiplier <= 0:
        raise ScheduleError(f"quantity_multiplier={quantity_multiplier} must be positive")
    if min_order_qty < 0:
        raise ScheduleError(f"min_order_qty={min_order_qty} cannot be negative")
    floored_total = quantise_down(traj.total_quantity, quantity_multiplier)
    if floored_total != traj.total_quantity:
        raise ScheduleError(
            f"total_quantity={traj.total_quantity} is not an exact multiple of "
            f"quantity_multiplier={quantity_multiplier}; a real order's own total size must "
            f"already be quantised (execution.guard.validate) before it reaches a schedule"
        )

    floor = max(quantity_multiplier, min_order_qty)
    slices: list[Decimal] = []
    cumulative_raw = _ZERO
    cumulative_emitted = _ZERO
    for s in traj.slices:
        cumulative_raw += s.quantity
        available = quantise_down(cumulative_raw - cumulative_emitted, quantity_multiplier)
        if available >= floor:
            slices.append(available)
            cumulative_emitted += available

    # Whatever never crossed the floor — trailing too-small intervals, or the last slice's own
    # flooring shortfall. Guaranteed an exact multiple of `quantity_multiplier`: the total is
    # (checked above) and `cumulative_emitted` is a sum of quantities that were each already a
    # multiple, so their difference must be too.
    final_remainder = traj.total_quantity - cumulative_emitted
    if final_remainder > 0:
        if slices:
            slices[-1] = slices[-1] + final_remainder
        else:
            slices.append(final_remainder)

    return QuantizedTrajectory(
        slices=tuple(slices), quantity_multiplier=quantity_multiplier,
        min_order_qty=min_order_qty, remainder_slice_added=len(slices) < len(traj.slices),
    )


__all__ = [
    "ImpactParameters",
    "QuantizedTrajectory",
    "ScheduleError",
    "Slice",
    "Trajectory",
    "quantise_trajectory",
    "trajectory",
    "twap",
]
