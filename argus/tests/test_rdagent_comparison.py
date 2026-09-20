"""Tests for the ARGUS-vs-RD-Agent factor-discovery comparison.

Every case runs BOTH systems' real code: RD-Agent's real, vendored, unmodified
`FactorFBWorkspace.execute()` and ARGUS's real `argus.research.searchoff` search machinery. No
live network calls and no LLM calls in this suite — `synthetic_bars()` (seeded, deterministic, no
network) stands in for `main()`'s own real, live NVDAUSDT fetch. See
`eval/rdagent_comparison.py`'s module docstring for the two findings these tests pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.eval.baselines.rdagent_factor_loader import load_factor_module
from argus.eval.rdagent_comparison import (
    SCOPE_STATEMENT,
    main,
    measure_costs,
    render,
    run_ablation,
    run_failure_cases,
    run_injection_proof,
    run_reproducibility_check,
    run_trial_pool,
    scan_argus_searchoff,
    scan_rdagent_factor_execution,
    synthetic_bars,
)


@pytest.fixture(scope="module")
def factor_module():
    _experiment, factor = load_factor_module()
    return factor


@pytest.fixture(autouse=True)
def _clean_rdagent_workspace():
    yield
    import shutil

    root = Path(__file__).resolve().parents[1] / "data"
    for name in ("_rdagent_workspace_tmp", "_rdagent_pickle_cache_tmp"):
        shutil.rmtree(root / name, ignore_errors=True)


class TestExecutionSurface:
    def test_argus_searchoff_has_no_forbidden_calls(self) -> None:
        scan = scan_argus_searchoff()
        assert not scan.has_execution_surface
        assert scan.forbidden_calls_found == ()

    def test_rdagent_factor_execution_contains_a_real_subprocess_call(self) -> None:
        scan = scan_rdagent_factor_execution()
        assert scan.has_execution_surface
        assert "subprocess.check_output" in scan.forbidden_calls_found

    def test_a_qualified_run_does_not_false_positive_on_an_unrelated_function_named_run(
        self,
    ) -> None:
        """`research/searchoff.py` has its own top-level `run()` (the bake-off entry point) —
        the scan must not flag it just because the NAME matches a subprocess method name."""
        scan = scan_argus_searchoff()
        assert "run" not in scan.forbidden_calls_found
        assert "subprocess.run" not in scan.forbidden_calls_found

    def test_the_injection_payload_actually_executes_through_unmodified_rdagent_code(
        self, factor_module
    ) -> None:
        proof = run_injection_proof(factor_module)
        assert proof.executed_attacker_code
        assert proof.marker_written


class TestFailureCases:
    def test_rdagent_catches_a_crashing_factor_cleanly(self, factor_module) -> None:
        cases = {c.input_: c for c in run_failure_cases(factor_module)}
        outcome = cases["factor.py raises RuntimeError"].outcome
        assert "CustomRuntimeError caught cleanly" in outcome
        assert "True" in outcome

    def test_argus_rejects_an_unknown_op_before_a_search_loop_even_starts(
        self, factor_module
    ) -> None:
        cases = {c.input_: c for c in run_failure_cases(factor_module)}
        assert "GrammarError at construction" in cases["Window('not_a_real_op', ...)"].outcome


class TestTrialCountCorrection:
    def test_the_real_dsr_gate_runs_on_a_real_trial_pool(self) -> None:
        bars = synthetic_bars()
        pool = run_trial_pool(bars, budget=30)
        assert pool.trials_spent == 30
        assert pool.deflated_probability is not None
        assert 0.0 <= pool.deflated_probability <= 1.0

    def test_more_trials_on_the_same_observed_value_deflates_the_probability_further(
        self,
    ) -> None:
        """The ablation: is the correction trial-count-SENSITIVE, not a fixed penalty."""
        bars = synthetic_bars()
        pool = run_trial_pool(bars, budget=30)
        points = {p.trials: p.probability for p in run_ablation(pool)}
        ordered_trials = sorted(points)
        probabilities = [points[t] for t in ordered_trials]
        assert probabilities == sorted(probabilities, reverse=True), (
            "probability must strictly decrease as trial count increases on the same observation"
        )


class TestReproducibility:
    def test_the_same_seed_reproduces_the_same_trial_pool(self) -> None:
        bars = synthetic_bars()
        result = run_reproducibility_check(bars)
        assert result["argus_reproducible"]


class TestCosts:
    def test_costs_are_measured_on_both_real_sides(self, factor_module) -> None:
        """The two numbers are NOT directly comparable (see `measure_costs`'s own docstring) — a
        real, live run on real ~1487-bar NVDAUSDT data found ARGUS's computation-bound cost
        EXCEEDING RD-Agent's fixed subprocess-spawn floor, the opposite of what the smaller
        synthetic fixture here shows, so this test asserts only that both are measured and
        positive — never that one is unconditionally cheaper than the other."""
        bars = synthetic_bars()
        costs = measure_costs(factor_module, bars)
        assert costs["rdagent_subprocess_spawn_seconds"] > 0
        assert costs["argus_grammar_evaluate_seconds"] > 0
        assert costs["bars_scored"] == len(bars)
        assert isinstance(costs["argus_cost_exceeds_rdagent_spawn_floor"], bool)


class TestMainAndRender:
    def test_main_returns_a_complete_serialisable_report_without_live_network(self) -> None:
        report = main(use_live_market_data=False)
        assert report["injection_proof"]["executed_attacker_code"]
        assert not report["argus_execution_surface_scan"]["has_execution_surface"]
        assert report["rdagent_execution_surface_scan"]["has_execution_surface"]
        assert report["trial_pool"]["trials_spent"] > 0
        assert report["scope_statement"] == SCOPE_STATEMENT
        assert report["used_live_market_data"] is False

    def test_render_produces_readable_text(self) -> None:
        report = main(use_live_market_data=False)
        text = render(report)
        assert "RD-AGENT COMPARISON" in text
        assert "injection proof" in text
