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


__all__ = [
    "ImpactParameters",
    "ScheduleError",
    "Slice",
    "Trajectory",
    "trajectory",
    "twap",
]
