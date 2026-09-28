"""The execution cost-risk frontier, a loss-tolerance way to choose a point on it, and a simulation
that checks the closed forms against the price process they assume.

**Why this exists.** `execution/schedule.py` solves one Almgren-Chriss trajectory for one risk
aversion, and the console chose that aversion from a fixed table (``SCHEDULE_DECAYS``). The paper's
own headline result is the whole trade-off — every optimal trajectory, cost against variance, the
*efficient frontier* (Almgren and Chriss 2000, §2, eq. 14-15, Figure 1) — and its §2.5 reads one
point off it from a stated confidence: minimise ``E + lambda_v * sqrt(V)`` (eq. 22), where
``lambda_v`` is the normal quantile of that confidence, so "keep the 95% worst-case execution cost
as low as it can be" becomes a schedule (research/harvest/47-almgren-chriss.md, decisions 1-2).
Field names follow AshJha0/electronic-trading's ``FrontierPoint`` (MIT); nothing else is taken.

**The simulation** (decision 3) does not re-use the closed forms it checks. It walks the
arithmetic random walk the model assumes, slice by slice —
``S_k = S_{k-1} + sigma*sqrt(tau)*xi_k - gamma*n_k``, each slice sold at
``S_{k-1} - epsilon - (eta/tau)*n_k`` — and measures the implementation shortfall
``X*S_0 - sum(n_k * price_k)`` on each path. Its mean and variance must agree with the
trajectory's ``expected_cost`` and ``variance`` to within sampling error; if the closed forms
drift from the process, this is what says so.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from statistics import NormalDist, fmean, pvariance

from argus.execution.schedule import ImpactParameters, ScheduleError, Trajectory, trajectory

KAPPA_T_GRID = tuple(Decimal(k) / 4 for k in range(0, 41))
"""The frontier is swept in ``kappa*T`` from 0 (TWAP) to 10 (almost everything in the first
interval); ``kappa*T`` is the shape of the schedule, independent of units, so one grid serves any
order."""


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One optimal trajectory, priced both ways."""

    risk_aversion: Decimal
    kappa: Decimal
    expected_cost: Decimal
    std: Decimal
    front_loading: Decimal

    def as_dict(self) -> dict[str, str]:
        return {"risk_aversion": f"{self.risk_aversion:.6g}", "kappa": f"{self.kappa:.6g}",
                "expected_cost": f"{self.expected_cost:.6g}", "std": f"{self.std:.6g}",
                "front_loading": f"{self.front_loading:.4f}"}


def _aversion_for(kappa_t: Decimal, *, horizon: Decimal, impact: ImpactParameters) -> Decimal:
    """The risk aversion giving roughly ``kappa*T``, from ``kappa^2 ~ lambda*sigma^2/eta``
    (the continuous-time limit of eq. 16); the trajectory then solves the exact discrete kappa."""
    if kappa_t == 0 or impact.sigma == 0:
        return Decimal(0)
    kappa = kappa_t / horizon
    return kappa * kappa * impact.eta / (impact.sigma * impact.sigma)


def efficient_frontier(*, quantity: Decimal, horizon: Decimal, intervals: int,
                       impact: ImpactParameters,
                       grid: Sequence[Decimal] = KAPPA_T_GRID) -> tuple[FrontierPoint, ...]:
    """Every optimal trajectory along the grid, cheapest-and-riskiest (TWAP) first."""
    points: list[FrontierPoint] = []
    for kappa_t in grid:
        try:
            path = trajectory(quantity=quantity, horizon=horizon, intervals=intervals,
                              impact=impact,
                              risk_aversion=_aversion_for(kappa_t, horizon=horizon,
                                                          impact=impact))
        except ScheduleError:
            continue
        points.append(_point(path))
    return tuple(points)


def _point(path: Trajectory) -> FrontierPoint:
    return FrontierPoint(risk_aversion=path.risk_aversion, kappa=path.kappa,
                         expected_cost=path.expected_cost,
                         std=Decimal(str(math.sqrt(float(max(path.variance, Decimal(0)))))),
                         front_loading=path.front_loading)


@dataclass(frozen=True, slots=True)
class VarChoice:
    """The frontier point with the lowest ``E + z*std`` at a stated confidence (eq. 22)."""

    confidence: float
    z: float
    point: FrontierPoint
    value_at_risk: Decimal
    """The execution cost this schedule stays under with the stated probability, if costs are
    normal — the paper's own assumption for eq. 22."""
    twap_value_at_risk: Decimal


def choose_by_confidence(frontier: Sequence[FrontierPoint], confidence: float) -> VarChoice:
    if not frontier:
        raise ScheduleError("an empty frontier has no point to choose")
    if not 0.5 < confidence < 1:
        raise ScheduleError(f"confidence {confidence} must be above 0.5 and below 1")
    z = NormalDist().inv_cdf(confidence)
    zd = Decimal(str(z))

    def var(p: FrontierPoint) -> Decimal:
        return p.expected_cost + zd * p.std

    best = min(frontier, key=var)
    return VarChoice(confidence=confidence, z=z, point=best, value_at_risk=var(best),
                     twap_value_at_risk=var(frontier[0]))


@dataclass(frozen=True, slots=True)
class Simulation:
    paths: int
    mean: float
    variance: float
    standard_error: float
    """Of the simulated mean: the closed-form cost should sit within a few of these."""


def simulate(path: Trajectory, impact: ImpactParameters, *, paths: int = 20_000,
             seed: int = 20260928) -> Simulation:
    """Implementation shortfall of selling ``path`` over simulated price paths (module note)."""
    n = len(path.slices)
    if n == 0:
        raise ScheduleError("a trajectory with no slices trades nothing")
    tau = float(path.horizon) / n
    sigma, gamma = float(impact.sigma), float(impact.gamma)
    eta, epsilon = float(impact.eta), float(impact.epsilon)
    sells = [float(s.quantity) for s in path.slices]
    rng = random.Random(seed)
    costs: list[float] = []
    step = sigma * math.sqrt(tau)
    for _ in range(paths):
        price = 0.0  # S_0 = 0: shortfall is independent of the level
        received = 0.0
        for sold in sells:
            # Interval k sells n_k at S_{k-1} less its temporary impact (eq. 5); the price then
            # moves by its diffusion and the permanent impact of what was sold (eq. 1).
            sign = 1 if sold > 0 else -1 if sold < 0 else 0
            received += sold * (price - epsilon * sign - eta / tau * sold)
            price += step * rng.gauss(0.0, 1.0) - gamma * sold
        costs.append(-received)
    mean = fmean(costs)
    variance = pvariance(costs, mu=mean)
    return Simulation(paths=paths, mean=mean, variance=variance,
                      standard_error=math.sqrt(variance / paths))


__all__ = ["KAPPA_T_GRID", "FrontierPoint", "Simulation", "VarChoice", "choose_by_confidence",
           "efficient_frontier", "simulate"]
