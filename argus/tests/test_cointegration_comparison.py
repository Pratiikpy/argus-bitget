"""Tests for the ARGUS-vs-statsmodels/Lean/FinceptTerminal cointegration comparison.

Every case runs real code: the real, installed `statsmodels.tsa.stattools.adfuller`; the real,
vendored, unmodified Lean `rank_pairs_by_correlation()`; and a clean-room reimplementation of
FinceptTerminal's real formula, independently verified (in this file) to reproduce their real
function's real output exactly. No live network calls. See
`eval/cointegration_comparison.py`'s module docstring for the three findings these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.baselines.lean_pairs_ranking_loader import load_pairs_ranking_module
from argus.eval.cointegration_comparison import (
    SCOPE_STATEMENT,
    fincept_style_zscore,
    main,
    measure_costs,
    render,
    run_adf_cases,
    run_failure_cases,
    run_multiple_testing_comparison,
    run_reproducibility_check,
    run_self_inclusion_cases,
    run_window_size_ablation,
)


@pytest.fixture(scope="module")
def pairs_module():
    return load_pairs_ranking_module()


class TestNumericAgreement:
    def test_every_series_shape_agrees_with_real_statsmodels(self) -> None:
        cases = run_adf_cases()
        assert len(cases) == 3
        for c in cases:
            assert c.agrees, f"{c.name} diverged: argus={c.argus.statistic} sm={c.statsmodels_stat}"


class TestSelfInclusionBias:
    def test_the_reimplementation_matches_finceptterminals_real_function(self) -> None:
        """Independently verifies this module's `fincept_style_zscore` against the real,
        unmodified FinceptTerminal function's real output — run once, locally, never vendored —
        before that reimplementation is trusted anywhere else in this comparison."""
        import importlib.util
        import sys
        import types

        root = (
            r"research\corpus\repos\FinceptTerminal\fincept-qt\scripts"
            r"\agents\hedgeFundAgents\renaissance_technologies_hedge_fund_agent\strategies"
        )
        for name in (
            "renaissance_technologies_hedge_fund_agent",
            "renaissance_technologies_hedge_fund_agent.strategies",
        ):
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)

        def load(dotted: str, path: str):
            spec = importlib.util.spec_from_file_location(dotted, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mod
            spec.loader.exec_module(mod)
            return mod

        load(
            "renaissance_technologies_hedge_fund_agent.strategies.mean_reversion",
            root + r"\mean_reversion.py",
        )
        stat_arb = load(
            "renaissance_technologies_hedge_fund_agent.strategies.statistical_arbitrage",
            root + r"\statistical_arbitrage.py",
        )

        a = [100.0, 99.5, 101.2, 98.7, 100.3, 99.9, 101.5, 100.1, 99.4, 100.8] * 4
        b = [50.0, 49.6, 50.8, 49.2, 50.1, 49.9, 50.6, 50.0, 49.5, 50.3] * 4
        result = stat_arb.analyze_pairs_trading(list(a), list(b))

        hedge = result.hedge_ratio
        spread = [a[i] - hedge * b[i] for i in range(len(a))]
        assert fincept_style_zscore(spread) == pytest.approx(result.spread_z_score, abs=1e-9)

    def test_a_designed_outlier_is_understated_by_self_inclusion(self) -> None:
        cases = {c.name: c for c in run_self_inclusion_cases()}
        outlier = cases["designed_outlier"]
        assert outlier.relative_understatement > 0.5
        assert abs(outlier.argus_z) > abs(outlier.fincept_z)


class TestWindowSizeAblation:
    def test_the_understatement_shrinks_as_the_window_grows(self) -> None:
        points = run_window_size_ablation()
        rates = [p.relative_understatement for p in points]
        assert rates == sorted(rates, reverse=True), (
            "self-inclusion bias must monotonically shrink as window size grows"
        )
        assert rates[0] > rates[-1]


class TestMultipleTestingCorrection:
    def test_a_naive_screen_selects_false_positives_argus_corrections_reject(
        self, pairs_module
    ) -> None:
        result = run_multiple_testing_comparison(pairs_module)
        assert result.naive_p05_selected > 0
        assert result.argus_bonferroni_survivors == 0
        assert result.argus_fdr_survivors == 0
        assert result.naive_selection_exceeds_the_corrected_ones

    def test_lean_style_threshold_and_naive_screen_are_both_measured(self, pairs_module) -> None:
        result = run_multiple_testing_comparison(pairs_module)
        assert result.n_pairs > 0
        assert result.lean_style_selected >= 0


class TestFailureCases:
    def test_both_real_systems_reject_a_too_short_series(self) -> None:
        cases = run_failure_cases()
        argus_case = next(c for c in cases if c.system == "argus")
        sm_case = next(c for c in cases if c.system == "statsmodels")
        assert "CointegrationError" in argus_case.outcome
        assert "NO EXCEPTION" not in sm_case.outcome


class TestCosts:
    def test_costs_are_measured_on_both_real_sides(self) -> None:
        costs = measure_costs()
        assert costs["argus_adf_seconds_per_call"] > 0
        assert costs["statsmodels_adfuller_seconds_per_call"] > 0


class TestReproducibility:
    def test_both_are_reproducible_on_identical_input(self) -> None:
        result = run_reproducibility_check()
        assert result["argus_reproducible"]
        assert result["fincept_style_reproducible"]


class TestMainAndRender:
    def test_main_returns_a_complete_serialisable_report(self) -> None:
        report = main()
        assert report["all_adf_cases_agree"]
        assert report["lean_has_no_correction"]
        assert report["multiple_testing"]["naive_selection_exceeds_the_corrected_ones"]
        assert report["scope_statement"] == SCOPE_STATEMENT
        import json

        json.dumps(report)  # must not raise — numpy scalar types must be cast to plain Python

    def test_render_produces_readable_text(self) -> None:
        report = main()
        text = render(report)
        assert "COINTEGRATION COMPARISON" in text
        assert "multiple-testing correction" in text
