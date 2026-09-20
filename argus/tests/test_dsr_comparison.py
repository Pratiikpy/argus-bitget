"""ARGUS's deflated_sharpe vs. vectorbt's real DSR math: real verdicts from both real systems."""

from __future__ import annotations

import math

from argus.eval.baselines.vectorbt_loader import (
    VectorbtDsrSymbols,
    load_vectorbt_baseline,
    vectorbt_var_sharpe,
)
from argus.eval.dsr_comparison import (
    SCOPE_STATEMENT,
    DsrCase,
    ablation_cases,
    compare,
    designed_cases,
    main,
    render,
    run_argus_dsr,
    run_designed,
    run_sweep,
    run_vectorbt_dsr,
    silent_nan_cost,
    swept_cases,
)


def _baseline() -> VectorbtDsrSymbols:
    return load_vectorbt_baseline()


class TestVectorbtVarSharpe:
    def test_a_single_trial_is_silently_nan(self) -> None:
        """The exact defect at accessors.py:596 — reproduced, not described."""
        result = vectorbt_var_sharpe([1.2])
        assert result != result  # NaN != NaN

    def test_multiple_trials_produce_a_real_variance(self) -> None:
        result = vectorbt_var_sharpe([1.0, 2.0, 3.0])
        assert result == result
        assert result > 0


class TestRunArgusDsr:
    def test_a_normal_case_produces_a_value(self) -> None:
        value, raised = run_argus_dsr(
            DsrCase("x", observed=1.2, n=500, trials=10, variance_of_trials=0.3)
        )
        assert raised is None
        assert value is not None
        assert 0.0 <= value <= 1.0

    def test_a_nan_variance_raises_rather_than_returning_nan(self) -> None:
        value, raised = run_argus_dsr(
            DsrCase(
                "x", observed=1.2, n=500, trials=1,
                variance_of_trials=vectorbt_var_sharpe([1.2]),
            )
        )
        assert value is None
        assert raised is not None
        assert "negative or NaN" in raised


class TestRunVectorbtDsr:
    def test_a_normal_case_produces_a_value(self) -> None:
        value, is_nan = run_vectorbt_dsr(
            DsrCase("x", observed=1.2, n=500, trials=10, variance_of_trials=0.3), _baseline()
        )
        assert not is_nan
        assert value is not None

    def test_a_nan_variance_produces_a_silent_nan(self) -> None:
        """The real, run-verified divergence this whole comparison exists to surface."""
        value, is_nan = run_vectorbt_dsr(
            DsrCase(
                "x", observed=1.2, n=500, trials=1,
                variance_of_trials=vectorbt_var_sharpe([1.2]),
            ),
            _baseline(),
        )
        assert is_nan
        assert value is None


class TestCompare:
    def test_agrees_exactly_on_a_normal_case(self) -> None:
        """"Agree" allows for ULP-level float noise between two independent computation paths
        (Python's `statistics.NormalDist` vs. numpy/scipy) — `DsrResult.agree` already applies
        the same `math.isclose` tolerance `main()`'s own sweep summary reports as "max abs diff",
        so this asserts through that property rather than re-testing raw `==` equality, which a
        real agreement can legitimately fail on the last significant digit."""
        result = compare(
            DsrCase("x", observed=1.2, n=500, trials=10, variance_of_trials=0.3), _baseline()
        )
        assert result.agree
        assert result.argus_value is not None and result.vectorbt_value is not None
        assert math.isclose(result.argus_value, result.vectorbt_value, rel_tol=1e-9)

    def test_agrees_by_argus_refusing_where_vectorbt_goes_nan(self) -> None:
        result = compare(
            DsrCase(
                "x", observed=1.2, n=500, trials=1,
                variance_of_trials=vectorbt_var_sharpe([1.2]),
            ),
            _baseline(),
        )
        assert result.agree
        assert result.argus_raised is not None
        assert result.vectorbt_produced_nan


class TestDesignedCases:
    def test_the_design_matches_its_own_stated_intent(self) -> None:
        run = run_designed(_baseline())
        assert run.design_is_sound, run.mismatches

    def test_the_single_trial_edge_case_is_included(self) -> None:
        names = {c.name for c in designed_cases()}
        assert "single_trial_var_computed_the_way_vectorbt_does" in names


class TestSweptCases:
    def test_the_grid_size_matches_what_this_test_expects(self) -> None:
        assert len(swept_cases()) == 5 * 3 * 3 * 3 * 3 * 3

    def test_every_swept_case_agrees(self) -> None:
        _, summary = run_sweep(_baseline())
        assert summary.agree == summary.total
        assert summary.max_abs_diff < 1e-9


class TestAblation:
    def test_both_guards_are_load_bearing(self) -> None:
        cases = ablation_cases()
        assert len(cases) == 2
        for case in cases:
            assert case.tripped_raises, case.dimension


class TestSilentNanCost:
    def test_the_real_shape_scenario_shows_vectorbt_going_nan_and_argus_refusing(self) -> None:
        cost = silent_nan_cost(_baseline())
        assert "NaN" in cost.vectorbt_verdict
        assert "MetricError" in cost.argus_verdict


class TestScopeStatement:
    def test_names_the_self_found_fix(self) -> None:
        assert "NOT claimed" in SCOPE_STATEMENT
        assert "fixed the same session" in SCOPE_STATEMENT


class TestMain:
    def test_main_runs_end_to_end_and_render_produces_readable_text(self) -> None:
        report = main()
        assert report["designed"]["design_is_sound"] is True
        assert report["sweep_summary"]["agree"] == report["sweep_summary"]["total"]
        assert report["ablation_all_load_bearing"] is True
        text = render(report)
        assert "DSR COMPARISON" in text
        assert "real single-trial shape" in text
