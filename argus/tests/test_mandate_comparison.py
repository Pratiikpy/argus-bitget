"""The ARGUS-vs-Vibe-Trading comparison: real verdicts from both real systems, on real scenarios."""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.desk.workbench import TraderProfile
from argus.eval.baselines.loader import VibeTradingSymbols, load_baseline
from argus.eval.mandate_comparison import (
    SCOPE_STATEMENT,
    Scenario,
    ablation_cases,
    blind_spot_costs,
    compare,
    designed_scenarios,
    main,
    render,
    run_argus,
    run_designed,
    run_sweep,
    run_vibe_trading,
    swept_scenarios,
)

_EXPECTED_SWEEP_SIZE = 2 * 4 * 3 * 3 * 3 * 3 * 3 * 5 * 2
"""profiles x notional_fractions x horizons x hedge_states x confidences x open_positions_states
x leverages x daily_pairs x exposure_pairs, kept in sync with `swept_scenarios()`'s own grids by
hand — a size this test pins so a silent grid shrink (accidentally testing less than intended)
shows up as a failure rather than a smaller, quieter sweep nobody notices."""


@pytest.fixture(scope="module")
def baseline() -> VibeTradingSymbols:
    return load_baseline()


class TestRunArgus:
    def test_a_clean_order_is_allowed(self) -> None:
        result = run_argus(
            Scenario(
                name="x", profile=TraderProfile.aggressive(), symbol="NVDAUSDT", side="buy",
                notional_usd=Decimal("5000"), thesis_horizon_hours=4.0,
            )
        )
        assert result.allowed and not result.reasons

    def test_a_horizon_breach_is_refused_with_a_reason(self) -> None:
        result = run_argus(
            Scenario(
                name="x", profile=TraderProfile.aggressive(), symbol="NVDAUSDT", side="buy",
                notional_usd=Decimal("5000"), thesis_horizon_hours=500.0,
            )
        )
        assert not result.allowed
        assert any("different trader's trade" in r for r in result.reasons)


class TestRunVibeTrading:
    def test_a_clean_order_is_allowed(self, baseline: VibeTradingSymbols) -> None:
        result = run_vibe_trading(
            Scenario(
                name="x", profile=TraderProfile.aggressive(), symbol="NVDAUSDT", side="buy",
                notional_usd=Decimal("5000"), thesis_horizon_hours=4.0,
            ),
            baseline,
        )
        assert result.allowed and not result.reasons

    def test_a_leverage_breach_is_refused_with_a_reason(
        self, baseline: VibeTradingSymbols
    ) -> None:
        result = run_vibe_trading(
            Scenario(
                name="x", profile=TraderProfile.aggressive(), symbol="NVDAUSDT", side="buy",
                notional_usd=Decimal("5000"), thesis_horizon_hours=4.0, max_leverage=0.01,
            ),
            baseline,
        )
        assert not result.allowed
        assert "max_leverage" in result.reasons[0]

    def test_a_horizon_only_breach_is_invisible_to_vibe_trading(
        self, baseline: VibeTradingSymbols
    ) -> None:
        """The structural point of this whole module: Vibe-Trading has no horizon field at all."""
        result = run_vibe_trading(
            Scenario(
                name="x", profile=TraderProfile.aggressive(), symbol="NVDAUSDT", side="buy",
                notional_usd=Decimal("5000"), thesis_horizon_hours=100_000.0,
            ),
            baseline,
        )
        assert result.allowed


class TestDesignedScenarios:
    def test_the_design_matches_its_own_stated_intent(
        self, baseline: VibeTradingSymbols
    ) -> None:
        """Every hand-built scenario does what this module's own module-level comment says it
        does — the same "check before trusting the design" discipline as the sweep-grid bug this
        module's own history records (see `swept_scenarios()`'s docstring)."""
        run = run_designed(baseline)
        assert run.design_is_sound, run.mismatches

    def test_there_is_at_least_one_scenario_per_stated_category(self) -> None:
        names = {s.name for s in designed_scenarios()}
        assert "clean_allow" in names
        assert any("argus_only" in n for n in names)
        assert any("vibe_trading_only" in n for n in names)
        assert any("both_should_refuse" in n for n in names)

    def test_compare_reports_agreement_correctly_on_the_clean_scenario(
        self, baseline: VibeTradingSymbols
    ) -> None:
        clean = next(s for s in designed_scenarios() if s.name == "clean_allow")
        result = compare(clean, baseline)
        assert result.agree_on_allow
        assert not result.only_argus_refuses
        assert not result.only_vibe_trading_refuses


class TestSweptScenarios:
    def test_the_grid_size_matches_what_this_test_expects(self) -> None:
        assert len(swept_scenarios()) == _EXPECTED_SWEEP_SIZE

    def test_scenario_names_are_unique(self) -> None:
        names = [s.name for s in swept_scenarios()]
        assert len(names) == len(set(names))

    def test_the_sweep_actually_exercises_every_vibe_trading_dimension_it_claims_to(
        self, baseline: VibeTradingSymbols
    ) -> None:
        """Regression test for a real bug this module's own history records: a first version of
        the grid held `max_trades_per_day`, `max_total_exposure_usd` and every `max_leverage`
        value at non-triggering levels, so only `max_order_notional_usd` ever tripped — silently
        understating Vibe-Trading's real coverage. This pins the fix: every one of the four
        dimensions the grid is designed to stress must show up at least once."""
        _, summary = run_sweep(baseline)
        for expected in (
            "max_order_notional_usd", "max_total_exposure_usd",
            "max_leverage", "max_trades_per_day",
        ):
            assert expected in summary.vibe_trading_limits_seen, summary.vibe_trading_limits_seen

    def test_the_sweep_produces_disagreements_in_both_directions(
        self, baseline: VibeTradingSymbols
    ) -> None:
        """Both systems must have SOME non-trivial blind spot on this grid — a sweep where only
        one side ever disagrees is either a genuinely lopsided pair of systems or, as it was
        before the fix above, an under-stressed grid. Real, run numbers, not asserted equal."""
        results, _ = run_sweep(baseline)
        assert any(r.only_argus_refuses for r in results)
        assert any(r.only_vibe_trading_refuses for r in results)


class TestAblation:
    def test_every_check_on_both_sides_is_independently_load_bearing(
        self, baseline: VibeTradingSymbols
    ) -> None:
        cases = ablation_cases(baseline)
        assert len(cases) >= 10
        for case in cases:
            assert case.check_is_load_bearing, (case.dimension, case.system)

    def test_both_systems_are_represented_in_the_ablation(
        self, baseline: VibeTradingSymbols
    ) -> None:
        systems = {c.system for c in ablation_cases(baseline)}
        assert systems == {"argus", "vibe_trading"}


class TestBlindSpotCosts:
    def test_rates_are_proper_fractions_and_counts_are_consistent(
        self, baseline: VibeTradingSymbols
    ) -> None:
        results, _ = run_sweep(baseline)
        costs = blind_spot_costs(results)
        assert costs.total_scenarios == len(results)
        assert 0.0 <= costs.argus_blind_spot_rate <= 1.0
        assert 0.0 <= costs.vibe_trading_blind_spot_rate <= 1.0
        assert costs.argus_blind_spot_count == pytest.approx(
            costs.argus_blind_spot_rate * costs.total_scenarios, abs=1
        )

    def test_an_empty_result_set_reports_zero_not_a_division_error(self) -> None:
        costs = blind_spot_costs(())
        assert costs.total_scenarios == 0
        assert costs.argus_blind_spot_rate == 0.0
        assert costs.vibe_trading_blind_spot_rate == 0.0


class TestScopeStatement:
    def test_the_scope_statement_names_the_real_separate_capability_it_defers_to(self) -> None:
        """The `no_specialist_capability_superior` claim is scoped, not total — this pins that
        the scoping actually names `ConstitutionPolicy`, the real module checked to exist before
        this claim was written, rather than silently drifting into an unscoped claim later."""
        assert "ConstitutionPolicy" in SCOPE_STATEMENT
        assert "max_gross_exposure_notional" in SCOPE_STATEMENT
        assert "NOT claimed" in SCOPE_STATEMENT


class TestMain:
    def test_main_runs_end_to_end_and_render_produces_readable_text(self) -> None:
        report = main()
        assert report["designed"]["design_is_sound"] is True
        assert report["sweep_summary"]["total"] == _EXPECTED_SWEEP_SIZE
        assert report["ablation_all_load_bearing"] is True
        text = render(report)
        assert "MANDATE COMPARISON" in text
        assert "blind spots on the sweep" in text
