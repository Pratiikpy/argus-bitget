"""`eval/schedule_comparison.py` — Almgren-Chriss vs the same-cost straight line, the run
comparison Track 3's Execution Assistance sub-theme needs. The theorem is already proven in the
paper; what these tests check is that THIS port computes it correctly, by verifying the guarantee
computationally rather than assuming the implementation matches the math."""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.eval.schedule_comparison import (
    HORIZONS,
    IMPACT_GRID,
    QUANTITIES,
    RISK_AVERSIONS,
    OosResult,
    ScheduleComparisonError,
    _price_volatility_per_bar,
    compare,
    out_of_sample_test,
    report,
    sweep,
    verify_quantisation,
)
from argus.execution.schedule import ImpactParameters

IMPACT = ImpactParameters(
    sigma=Decimal("0.01"), gamma=Decimal("0.0001"), eta=Decimal("0.001"), epsilon=Decimal("0.001"),
)


class TestCompareAtOneInput:
    def test_ac_never_loses_to_the_straight_line_at_positive_risk_aversion(self) -> None:
        c = compare(
            quantity=Decimal("10000"), horizon=Decimal("6"), intervals=12, impact=IMPACT,
            risk_aversion=Decimal("1"),
        )
        assert c.ac_wins_or_ties

    def test_exact_zero_risk_aversion_ties_itself_exactly(self) -> None:
        """At risk_aversion=0 the 'AC' trajectory and the straight-line trajectory are computed
        from the identical inputs (both at risk_aversion=0) — they must be the exact same
        schedule, not merely tie on the objective."""
        c = compare(
            quantity=Decimal("5000"), horizon=Decimal("6"), intervals=12, impact=IMPACT,
            risk_aversion=Decimal("0"),
        )
        assert c.ac.is_twap
        assert c.ac_objective == c.straight_objective
        assert c.savings_pct == Decimal("0")

    def test_higher_risk_aversion_front_loads_more_than_the_straight_line(self) -> None:
        c = compare(
            quantity=Decimal("10000"), horizon=Decimal("6"), intervals=12, impact=IMPACT,
            risk_aversion=Decimal("5"),
        )
        assert c.ac.front_loading > c.straight.front_loading
        assert c.straight.front_loading == Decimal("0.5")

    def test_savings_pct_is_nonnegative_when_ac_wins(self) -> None:
        c = compare(
            quantity=Decimal("10000"), horizon=Decimal("6"), intervals=12, impact=IMPACT,
            risk_aversion=Decimal("2"),
        )
        assert c.savings_pct >= Decimal("0")


class TestSweepAndReport:
    def test_the_default_grid_is_large_enough_to_be_a_real_sweep(self) -> None:
        assert len(QUANTITIES) * len(HORIZONS) * len(IMPACT_GRID) * len(RISK_AVERSIONS) >= 100

    def test_ac_never_loses_anywhere_in_the_default_sweep(self) -> None:
        """The decisive check: the paper's guarantee, verified over every point in the real
        default grid this module ships, not a hand-picked subset."""
        comparisons = sweep()
        rpt = report(comparisons)
        assert rpt["ac_never_loses"] is True
        assert rpt["losses"] == []

    def test_exact_zero_risk_aversion_matches_twap_shape_everywhere(self) -> None:
        comparisons = sweep()
        rpt = report(comparisons)
        assert rpt["exact_zero_risk_aversion_matches_twap_shape"] is True

    def test_near_zero_risk_aversion_converges_close_to_half(self) -> None:
        comparisons = sweep()
        rpt = report(comparisons)
        assert rpt["near_zero_front_loading_converges_to_half"] is not None
        assert float(rpt["near_zero_front_loading_converges_to_half"]) < 0.01

    def test_positive_risk_aversion_produces_real_positive_mean_savings(self) -> None:
        comparisons = sweep()
        rpt = report(comparisons)
        assert float(rpt["mean_savings_pct_at_positive_risk_aversion"]) > 0

    def test_report_on_an_empty_sweep_does_not_crash(self) -> None:
        rpt = report([])
        assert rpt["n"] == 0
        assert rpt["ac_never_loses"] is True
        assert rpt["exact_zero_risk_aversion_matches_twap_shape"] is None
        assert rpt["near_zero_front_loading_converges_to_half"] is None


class TestVerifyQuantisation:
    """Closes the one real gap `failure_cases_documented` named against Nautilus's real TWAP
    (twap.rs:220-309): every schedule in the real default sweep, both AC and the straight line,
    must quantise to a real lot size with its total preserved exactly."""

    def test_every_schedule_in_the_default_sweep_preserves_its_total_when_quantised(self) -> None:
        comparisons = sweep()
        qrpt = verify_quantisation(comparisons, quantity_multiplier=Decimal("1"))
        assert qrpt["all_totals_preserved"] is True
        assert qrpt["failures"] == []

    def test_it_checks_both_the_ac_and_the_straight_schedule_per_point(self) -> None:
        comparisons = sweep()
        qrpt = verify_quantisation(comparisons, quantity_multiplier=Decimal("1"))
        assert qrpt["checked"] == 2 * len(comparisons)

    def test_an_empty_sweep_reports_honestly_rather_than_crashing(self) -> None:
        qrpt = verify_quantisation([], quantity_multiplier=Decimal("1"))
        assert qrpt["checked"] == 0
        assert qrpt["all_totals_preserved"] is True
        assert qrpt["genuinely_consolidated"] == 0

    def test_the_real_sweep_genuinely_exercises_the_merge_path_not_just_the_happy_case(
        self,
    ) -> None:
        """A zero here would mean the sweep's own grid never produced a slice too small to trade
        on its own, and the total-preservation pass above would be proving nothing about the
        merge logic specifically — checked directly rather than assumed from the grid's size."""
        comparisons = sweep()
        qrpt = verify_quantisation(comparisons, quantity_multiplier=Decimal("1"))
        assert qrpt["genuinely_consolidated"] > 0


class TestPriceVolatilityPerBar:
    """The one parameter `out_of_sample_test` genuinely calibrates from real data — checked in
    isolation, no live fetch needed, before trusting it inside the network-dependent test."""

    def test_it_scales_with_the_actual_spread_of_returns(self) -> None:
        calm = [Decimal("100"), Decimal("100.1")] * 20
        volatile = [Decimal("100"), Decimal("110")] * 20
        calm_sigma = _price_volatility_per_bar(calm)
        volatile_sigma = _price_volatility_per_bar(volatile)
        assert volatile_sigma > calm_sigma

    def test_it_is_in_price_units_not_a_bare_fraction(self) -> None:
        """A 1% move on a $100 close should read close to $1, not close to 0.01 — matching
        `ImpactParameters.sigma`'s own documented units."""
        closes = [Decimal("100"), Decimal("101")] * 20
        sigma = _price_volatility_per_bar(closes)
        assert Decimal("0.1") < sigma < Decimal("10")

    def test_too_few_observations_raises_rather_than_returning_noise(self) -> None:
        with pytest.raises(ScheduleComparisonError):
            _price_volatility_per_bar([Decimal("100"), Decimal("101"), Decimal("99")])


class TestOosResult:
    """The pure arithmetic on a result, independent of how it was produced."""

    def _result(
        self, *, in_sigma: str, out_sigma: str, ac_in: bool = True, ac_out: bool = True,
    ) -> OosResult:
        return OosResult(
            symbol="TESTUSDT", days=90, bars_in_sample=100, bars_out_of_sample=100,
            in_sample_sigma=Decimal(in_sigma), out_of_sample_sigma=Decimal(out_sigma),
            epsilon=Decimal("0.1"), ac_wins_in_sample=ac_in, ac_wins_out_of_sample=ac_out,
            in_sample_savings_pct=Decimal("10"), out_of_sample_savings_pct=Decimal("10"),
        )

    def test_sigma_ratio_is_out_over_in(self) -> None:
        r = self._result(in_sigma="2", out_sigma="1")
        assert r.sigma_ratio == Decimal("0.5")

    def test_advantage_holds_only_when_both_windows_win(self) -> None:
        assert self._result(in_sigma="1", out_sigma="1", ac_in=True, ac_out=True) \
            .advantage_holds_out_of_sample
        assert not self._result(in_sigma="1", out_sigma="1", ac_in=True, ac_out=False) \
            .advantage_holds_out_of_sample
        assert not self._result(in_sigma="1", out_sigma="1", ac_in=False, ac_out=True) \
            .advantage_holds_out_of_sample

    def test_as_dict_serialises(self) -> None:
        payload = self._result(in_sigma="2", out_sigma="1").as_dict()
        assert payload["advantage_holds_out_of_sample"] is True
        assert Decimal(str(payload["sigma_ratio"])) == Decimal("0.5")


class TestOutOfSampleTestFailsHonestly:
    def test_too_few_candles_raises_rather_than_guessing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import argus.market.history as history_module

        def _fake_fetch_range(*args: object, **kwargs: object) -> list[object]:
            return []

        monkeypatch.setattr(history_module, "fetch_range", _fake_fetch_range)
        with pytest.raises(ScheduleComparisonError):
            out_of_sample_test(symbol="NVDAUSDT")


