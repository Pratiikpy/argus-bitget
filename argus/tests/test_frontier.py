"""The execution frontier, the confidence choice on it, and a simulation of the price process the
closed forms assume (research/harvest/47-almgren-chriss.md). Parameters are the paper's own worked
example: 10^6 shares over 5 days in 5 intervals, sigma 0.95, gamma 2.5e-7, eta 2.5e-6,
epsilon 0.0625 (Almgren and Chriss 2000, Table 1)."""

from __future__ import annotations

from decimal import Decimal as D

import pytest

from argus.execution.frontier import choose_by_confidence, efficient_frontier, simulate
from argus.execution.schedule import ImpactParameters, ScheduleError, trajectory

PAPER = ImpactParameters(sigma=D("0.95"), gamma=D("2.5e-7"), eta=D("2.5e-6"),
                         epsilon=D("0.0625"))
ORDER = {"quantity": D(1_000_000), "horizon": D(5), "intervals": 5, "impact": PAPER}


def test_the_frontier_trades_cost_for_risk_monotonically() -> None:
    frontier = efficient_frontier(**ORDER)  # type: ignore[arg-type]
    assert frontier[0].kappa == 0  # TWAP first
    costs = [p.expected_cost for p in frontier]
    risks = [p.std for p in frontier]
    assert costs == sorted(costs) and risks == sorted(risks, reverse=True)
    assert frontier[-1].front_loading > D("0.9")


def test_a_higher_confidence_chooses_a_more_front_loaded_schedule() -> None:
    frontier = efficient_frontier(**ORDER)  # type: ignore[arg-type]
    mild = choose_by_confidence(frontier, 0.80)
    strict = choose_by_confidence(frontier, 0.99)
    assert strict.point.front_loading >= mild.point.front_loading
    assert strict.value_at_risk <= strict.twap_value_at_risk


def test_the_confidence_must_mean_something() -> None:
    frontier = efficient_frontier(**ORDER)  # type: ignore[arg-type]
    for bad in (0.5, 1.0, 0.2):
        with pytest.raises(ScheduleError):
            choose_by_confidence(frontier, bad)
    with pytest.raises(ScheduleError):
        choose_by_confidence((), 0.95)


@pytest.mark.parametrize("aversion", [D(0), D("1e-6")])
def test_the_closed_forms_agree_with_a_simulated_price_process(aversion: D) -> None:
    path = trajectory(risk_aversion=aversion, **ORDER)  # type: ignore[arg-type]
    sim = simulate(path, PAPER, paths=40_000, seed=7)
    assert abs(sim.mean - float(path.expected_cost)) < 4 * sim.standard_error
    # The variance of a sample variance of normal draws is about 2*V^2/n: 4 of those, ~2.8%.
    assert abs(sim.variance / float(path.variance) - 1) < 4 * (2 / sim.paths) ** 0.5


def test_twap_matches_the_papers_expected_cost() -> None:
    """E for the straight-line schedule: gamma X^2/2 + epsilon X + (eta_tilde/tau) X^2/N."""
    path = trajectory(risk_aversion=D(0), **ORDER)  # type: ignore[arg-type]
    assert float(path.expected_cost) == pytest.approx(662_500, rel=1e-9)
