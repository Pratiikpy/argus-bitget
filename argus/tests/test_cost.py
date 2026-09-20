"""Cost-model tests.

Cost-blindness is the most widespread defect class across the 103 catalogued in our teardowns.
These tests assert that the specific failures observed in real systems cannot occur here.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.cost.model import (
    CostModel,
    Fill,
    FrictionlessCostError,
    Liquidity,
    frictionless_for_research,
    net_edge_bps,
)


class TestZeroFeeIsUnreachable:
    """NautilusTrader backtests at zero fees unless configured. Here that is not constructible."""

    def test_zero_taker_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="most common defect"):
            CostModel(taker_bps=Decimal("0"), maker_bps=Decimal("0"))

    def test_negative_taker_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            CostModel(taker_bps=Decimal("-1"), maker_bps=Decimal("0"))

    def test_there_is_no_zero_classmethod(self) -> None:
        """Guards against a future convenience constructor quietly reintroducing the defect."""
        assert not hasattr(CostModel, "zero")
        assert not hasattr(CostModel, "free")

    def test_non_convex_impact_exponent_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="convex"):
            CostModel(taker_bps=Decimal("6"), maker_bps=Decimal("2"), gamma=Decimal("0.5"))


class TestFrictionlessIsPoisoned:
    """alphaagent gates on without_cost metrics while with_cost sits computed and unread.
    RD-Agent shows without_cost to proposer and evaluator while feedback reads with_cost."""

    def test_frictionless_requires_a_written_reason(self) -> None:
        with pytest.raises(ValueError, match="written reason"):
            frictionless_for_research("")

    def test_frictionless_cannot_gate_a_decision(self) -> None:
        m = frictionless_for_research("isolating the sentiment signal from execution drag")
        with pytest.raises(FrictionlessCostError, match="cannot justify a decision"):
            m.assert_gateable()

    def test_frictionless_cannot_reach_net_edge(self) -> None:
        m = frictionless_for_research("ablation: signal only")
        with pytest.raises(FrictionlessCostError):
            net_edge_bps(Decimal("50"), m)

    def test_frictionless_still_computes_so_research_is_possible(self) -> None:
        """The escape hatch must actually work, or people route around it."""
        m = frictionless_for_research("ablation: signal only")
        got = m.charge(Fill(notional=Decimal("100000"), liquidity=Liquidity.TAKER))
        assert got.total == Decimal("0")

    def test_a_real_model_gates_without_complaint(self) -> None:
        CostModel.bitget_perp().assert_gateable()


class TestBitgetEconomics:
    model = CostModel.bitget_perp()

    def test_round_trip_is_twelve_bps(self) -> None:
        """0.06% per side, 0.12% round trip — the measured Bitget taker fee."""
        assert self.model.round_trip_bps() == Decimal("12")

    def test_intraday_edge_does_not_clear_the_fee(self) -> None:
        """~0.00% measured intraday edge against a 0.12% round trip. Anything round-tripping
        daily loses before it begins; 360+ variants were negative in our own testing."""
        assert net_edge_bps(Decimal("0"), self.model) == Decimal("-12")

    def test_overnight_drift_also_does_not_clear_the_fee(self) -> None:
        """~0.10% overnight drift = 10bps, still short of the 12bps round trip."""
        assert net_edge_bps(Decimal("10"), self.model) < 0

    def test_a_genuine_edge_survives(self) -> None:
        assert net_edge_bps(Decimal("45"), self.model) == Decimal("33")

    def test_multiple_round_trips_compound_the_fee(self) -> None:
        """Turnover is the mechanism by which a real edge is destroyed."""
        assert net_edge_bps(Decimal("45"), self.model, round_trips=4) == Decimal("-3")


class TestCharging:
    model = CostModel.bitget_perp()

    def test_taker_pays_commission_and_crosses_the_spread(self) -> None:
        got = self.model.charge(
            Fill(notional=Decimal("100000"), liquidity=Liquidity.TAKER, spread_bps=Decimal("4"))
        )
        assert got.commission == Decimal("60")   # 6bps of 100k
        assert got.spread == Decimal("40")       # 4bps of 100k
        assert got.total == Decimal("100")

    def test_maker_is_not_credited_the_spread(self) -> None:
        """Deliberate: resting limit orders are adversely selected. A local sim showed +90%
        assuming limit fills; the real replay gave -0.45%. We never model a maker rebate as
        free money."""
        got = self.model.charge(
            Fill(notional=Decimal("100000"), liquidity=Liquidity.MAKER, spread_bps=Decimal("4"))
        )
        assert got.spread == Decimal("0")
        assert got.commission == Decimal("20")

    def test_impact_is_superlinear_in_participation(self) -> None:
        """Square-root law, gamma=1.5: doubling participation more than doubles impact."""
        small = self.model.charge(
            Fill(Decimal("100000"), Liquidity.TAKER, adv_participation=Decimal("0.01"))
        )
        big = self.model.charge(
            Fill(Decimal("100000"), Liquidity.TAKER, adv_participation=Decimal("0.02"))
        )
        assert big.impact > small.impact * 2

    def test_zero_participation_means_zero_impact(self) -> None:
        got = self.model.charge(Fill(Decimal("100000"), Liquidity.TAKER))
        assert got.impact == Decimal("0")

    def test_borrow_accrues_only_while_held(self) -> None:
        m = CostModel(
            taker_bps=Decimal("6"), maker_bps=Decimal("2"), borrow_bps_annual=Decimal("365")
        )
        f = Fill(notional=Decimal("100000"), liquidity=Liquidity.TAKER)
        assert m.charge(f, holding_days=Decimal("0")).borrow == Decimal("0")
        assert m.charge(f, holding_days=Decimal("10")).borrow == Decimal("100")

    def test_breakdown_keeps_components_separate(self) -> None:
        """A single total hides which assumption is load-bearing."""
        got = self.model.charge(
            Fill(Decimal("50000"), Liquidity.TAKER, Decimal("3"), Decimal("0.005"))
        )
        assert got.commission > 0
        assert got.spread > 0
        assert got.impact > 0
        assert got.total == got.commission + got.spread + got.impact + got.borrow

    def test_bps_of_requires_positive_notional(self) -> None:
        got = self.model.charge(Fill(Decimal("100000"), Liquidity.TAKER))
        with pytest.raises(ValueError, match="positive"):
            got.bps_of(Decimal("0"))


class TestFundingIsChargedPerSettlement:
    """Perpetual funding was fetched from the venue and thrown away before it reached a cost.

    `Ticker.funding_rate` has existed since the beginning (`market/bitget.py:82`) and no cost model
    read it. Measured from Bitget's own history on 2026-09-14 over 100 settlements per symbol:
    NVDAUSDT non-zero on 14, TSLAUSDT 23, COINUSDT 30; means 0.23/0.36/0.58bps, worst single
    settlement 5.8bps. A 24-hour hold crosses three of them.
    """

    @staticmethod
    def _model(rate_bps: str = "0.36") -> CostModel:
        return CostModel(
            taker_bps=Decimal("6"), maker_bps=Decimal("2"),
            funding_bps_per_interval=Decimal(rate_bps),
        )

    @staticmethod
    def _fill() -> Fill:
        return Fill(notional=Decimal("10000"), liquidity=Liquidity.TAKER, spread_bps=Decimal("9"))

    def test_a_hold_shorter_than_one_interval_pays_nothing(self) -> None:
        """Funding settles on a clock, not by the hour. Charging a fraction would invent a cost the
        venue never takes."""
        got = self._model().charge(self._fill(), holding_days=Decimal("0.1"))
        assert got.funding == Decimal("0")

    def test_twenty_four_hours_crosses_three_settlements(self) -> None:
        got = self._model().charge(self._fill(), holding_days=Decimal("1"))
        assert got.funding == Decimal("10000") * Decimal("0.36") / Decimal("10000") * 3

    def test_a_short_receives_funding_rather_than_paying_it(self) -> None:
        long = self._model().charge(self._fill(), holding_days=Decimal("1"), long=True)
        short = self._model().charge(self._fill(), holding_days=Decimal("1"), long=False)
        assert long.funding > 0 > short.funding
        assert long.funding == -short.funding

    def test_funding_reaches_the_total(self) -> None:
        """The bug this class exists for: the field was added and left out of `total`, so the
        charge was computed correctly and then discarded."""
        priced = self._model().charge(self._fill(), holding_days=Decimal("1"))
        free = self._model("0").charge(self._fill(), holding_days=Decimal("1"))
        assert priced.total > free.total
        assert priced.total - free.total == priced.funding

    def test_it_reaches_the_bps_figure_too(self) -> None:
        notional = Decimal("10000")
        priced = self._model().charge(self._fill(), holding_days=Decimal("1"))
        free = self._model("0").charge(self._fill(), holding_days=Decimal("1"))
        assert priced.bps_of(notional) > free.bps_of(notional)

    def test_zero_is_the_default_and_is_honest_not_frictionless(self) -> None:
        """Most rToken settlements really are zero, so a zero default is accurate — and it must not
        trip the frictionless guard, which exists for a different failure."""
        model = CostModel(taker_bps=Decimal("6"), maker_bps=Decimal("2"))
        assert model.funding_bps_per_interval == Decimal("0")
        model.assert_gateable()

    def test_a_rate_passed_as_a_fraction_is_refused(self) -> None:
        """The API returns 0.000341; this field wants 3.41. Passing 500+ means the units are wrong,
        and the difference is a 3.41bps charge versus a 34,100bps one."""
        with pytest.raises(ValueError, match="check the units"):
            CostModel(
                taker_bps=Decimal("6"), maker_bps=Decimal("2"),
                funding_bps_per_interval=Decimal("600"),
            )

    def test_a_negative_rate_is_allowed(self) -> None:
        """Funding is the one channel here that can be a credit; clamping it would delete that."""
        model = CostModel(
            taker_bps=Decimal("6"), maker_bps=Decimal("2"),
            funding_bps_per_interval=Decimal("-0.4"),
        )
        got = model.charge(self._fill(), holding_days=Decimal("1"), long=True)
        assert got.funding < 0

    def test_the_venue_factory_converts_the_fraction_for_you(self) -> None:
        model = CostModel.bitget_perp(funding_rate=Decimal("0.000036"))
        assert model.funding_bps_per_interval == Decimal("0.36")

    def test_the_venue_factory_defaults_to_no_funding(self) -> None:
        assert CostModel.bitget_perp().funding_bps_per_interval == Decimal("0")
