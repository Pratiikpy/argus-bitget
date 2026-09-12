"""Almgren-Chriss trajectory tests.

The paper's results are checkable against their own limits rather than against this code's output:
zero risk aversion must give TWAP exactly, infinite risk aversion must give immediate execution,
and everything between must sit on a monotone frontier of mean against variance. A test suite that
only asserted internal consistency would pass on a wrong sign.
"""

from __future__ import annotations

from decimal import Decimal
from itertools import pairwise

import pytest

from argus.execution.schedule import (
    ImpactParameters,
    ScheduleError,
    trajectory,
    twap,
)

D = Decimal
IMPACT = ImpactParameters(sigma=D("0.3"), gamma=D("0.0001"), eta=D("0.01"), epsilon=D("0.02"))


def _run(lam: str, *, intervals: int = 10, quantity: str = "100000"):  # type: ignore[no-untyped-def]
    return trajectory(
        quantity=D(quantity), horizon=D("1"), intervals=intervals,
        impact=IMPACT, risk_aversion=D(lam),
    )


class TestTheLimits:
    """Both ends of the frontier have known closed forms. They are the strongest available check."""

    def test_zero_risk_aversion_is_exactly_twap(self) -> None:
        t = _run("0")
        assert t.is_twap
        assert t.kappa == 0
        # Ten equal slices, and the equality is exact because the limit is computed, not
        # approximated with an epsilon.
        assert all(s.quantity == D("10000") for s in t.slices)
        assert t.front_loading == D("0.5")

    def test_the_twap_limit_is_approached_continuously(self) -> None:
        """A vanishing lambda must converge to TWAP, not jump to it at a guard."""
        nearly = _run("1e-12")
        assert not nearly.is_twap or nearly.kappa == 0
        for s in nearly.slices:
            assert abs(s.quantity - D("10000")) < D("1")

    def test_extreme_risk_aversion_trades_immediately(self) -> None:
        t = _run("1e9")
        assert t.front_loading == 1
        assert t.slices[0].quantity > D("99999")
        assert t.variance < D("1")

    def test_a_saturating_kappa_degrades_to_immediate_not_an_error(self) -> None:
        """CPython's math.sinh *raises* OverflowError rather than returning inf.

        A risk aversion that saturates the float range is a legitimate preference - "do not hold
        this inventory for any measurable time" - and the answer is immediate execution. Before
        this was handled, the call crashed.
        """
        t = _run("1e40")
        assert t.slices[0].quantity == D("100000")
        assert all(s.quantity == 0 for s in t.slices[1:])

    def test_below_saturation_the_tail_is_negligible_not_absent(self) -> None:
        """Just under the overflow the schedule is essentially immediate, with a real residue.

        Asserted as negligible rather than zero, because rounding it to zero would be the model
        lying about having liquidated.
        """
        t = _run("1e30")
        tail = sum((s.quantity for s in t.slices[1:]), D("0"))
        assert 0 < tail < D("1e-6")
        assert sum((s.quantity for s in t.slices), D("0")) == t.total_quantity

    def test_inventory_is_fully_liquidated(self) -> None:
        for lam in ("0", "1e-6", "1", "100"):
            t = _run(lam)
            assert t.slices[-1].remaining == pytest.approx(0, abs=1e-6)
            assert sum(s.quantity for s in t.slices) == pytest.approx(float(t.total_quantity),
                                                                     rel=1e-9)


class TestTheFrontier:
    """More risk aversion buys less variance at the price of more expected cost. Both directions
    must hold, or the model is not an optimiser."""

    def test_variance_falls_monotonically_in_risk_aversion(self) -> None:
        runs = [_run(lam) for lam in ("0", "1e-4", "1e-2", "1", "100")]
        variances = [t.variance for t in runs]
        assert variances == sorted(variances, reverse=True)

    def test_expected_cost_rises_monotonically(self) -> None:
        runs = [_run(lam) for lam in ("0", "1e-4", "1e-2", "1", "100")]
        costs = [t.expected_cost for t in runs]
        assert costs == sorted(costs)

    def test_the_trade_off_is_real_not_free(self) -> None:
        """If a schedule reduced variance at no cost, the model would be wrong."""
        cheap, safe = _run("0"), _run("1")
        assert safe.variance < cheap.variance
        assert safe.expected_cost > cheap.expected_cost

    def test_holdings_decay_hyperbolically_not_linearly(self) -> None:
        """The paper's central result: the optimal path is sinh, not a straight line."""
        t = _run("1")
        linear_midpoint = t.total_quantity / 2
        midpoint = t.slices[len(t.slices) // 2 - 1].remaining
        assert midpoint < linear_midpoint, "an optimal path front-loads; a linear one does not"

    def test_front_loading_increases_with_risk_aversion(self) -> None:
        assert _run("0").front_loading < _run("1").front_loading <= _run("100").front_loading

    def test_holdings_never_increase(self) -> None:
        """A liquidation schedule that buys back is solving a different problem."""
        for lam in ("0", "1e-3", "1", "1000"):
            t = _run(lam)
            remaining = [t.total_quantity] + [s.remaining for s in t.slices]
            assert all(a >= b for a, b in pairwise(remaining))
            assert all(s.quantity >= 0 for s in t.slices)


class TestGuards:
    def test_a_negative_risk_aversion_is_refused(self) -> None:
        with pytest.raises(ScheduleError, match="seek"):
            _run("-1")

    def test_zero_quantity_is_refused(self) -> None:
        with pytest.raises(ScheduleError, match="quantity"):
            _run("0", quantity="0")

    def test_zero_intervals_is_refused(self) -> None:
        with pytest.raises(ScheduleError, match="intervals"):
            trajectory(quantity=D("100"), horizon=D("1"), intervals=0, impact=IMPACT)

    def test_zero_horizon_is_refused(self) -> None:
        with pytest.raises(ScheduleError, match="horizon"):
            trajectory(quantity=D("100"), horizon=D("0"), intervals=5, impact=IMPACT)

    def test_zero_temporary_impact_is_refused(self) -> None:
        """Without eta there is nothing to trade against risk and the model degenerates."""
        with pytest.raises(ScheduleError, match="eta must be positive"):
            ImpactParameters(sigma=D("0.3"), gamma=D("0.0001"), eta=D("0"), epsilon=D("0.02"))

    def test_permanent_impact_swamping_temporary_is_refused(self) -> None:
        """eta_tilde <= 0 means the decomposition has broken down, not that the answer is zero."""
        bad = ImpactParameters(sigma=D("0.3"), gamma=D("10"), eta=D("0.01"), epsilon=D("0"))
        with pytest.raises(ScheduleError, match="eta_tilde"):
            trajectory(quantity=D("100"), horizon=D("1"), intervals=2, impact=bad)

    def test_negative_impact_parameters_are_refused(self) -> None:
        with pytest.raises(ScheduleError, match="sigma"):
            ImpactParameters(sigma=D("-1"), gamma=D("0"), eta=D("1"), epsilon=D("0"))


class TestSessionHonesty:
    """Almgren-Chriss assumes constant volatility and liquidity. Our venue does not provide them."""

    def test_a_boundary_crossing_is_reported_not_corrected(self) -> None:
        t = trajectory(
            quantity=D("100000"), horizon=D("1"), intervals=10, impact=IMPACT,
            risk_aversion=D("1"), crosses_session_boundary=True,
        )
        assert t.crosses_session_boundary
        assert t.as_dict()["crosses_session_boundary"] is True

    def test_it_defaults_to_not_crossing(self) -> None:
        assert _run("1").crosses_session_boundary is False


class TestTwapBaseline:
    """A schedule that cannot be compared to the default is not a result."""

    def test_twap_is_the_straight_line(self) -> None:
        t = twap(quantity=D("1000"), horizon=D("1"), intervals=4)
        assert t.is_twap
        assert [s.quantity for s in t.slices] == [D("250")] * 4

    def test_twap_carries_no_variance_term(self) -> None:
        """Its sigma is zero by construction, so it is a schedule not a risk claim."""
        assert twap(quantity=D("1000"), horizon=D("1"), intervals=4).variance == 0

    def test_cost_per_share_is_reported(self) -> None:
        t = _run("1")
        assert t.cost_per_share() == t.expected_cost / t.total_quantity

    def test_the_record_serialises(self) -> None:
        payload = _run("1").as_dict()
        assert payload["intervals"] == 10
        assert len(payload["slices"]) == 10  # type: ignore[arg-type]
        assert payload["is_twap"] is False
