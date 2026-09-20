"""Tests for the ARGUS-vs-qlib typed-factor-grammar comparison.

Every case runs BOTH systems' real code: ARGUS's real ``argus.research.grammar.Window``/``Corr``
and qlib's real, vendored, unmodified ``Mean``/``Std``/``Rank``/``Corr`` operators plus its real
``ExpressionProvider.get_expression_instance()``. Nothing here is mocked. See
``eval/grammar_comparison.py``'s module docstring for the two findings these tests pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.eval.baselines.qlib_eval_surface_loader import load_eval_surface_module
from argus.eval.baselines.qlib_expression_loader import load_expression_module
from argus.eval.grammar_comparison import (
    SCOPE_STATEMENT,
    main,
    measure_costs,
    rank_convention_divergence,
    render,
    run_agreement_cases,
    run_failure_cases,
    run_injection_proof,
    run_rank_cases,
    run_reproducibility_check,
    run_scan_sensitivity_ablation,
    scan_argus_grammar,
    scan_qlib_eval_surface,
    swept_rank_cases,
)


@pytest.fixture(scope="module")
def ops_module():
    _base, ops = load_expression_module()
    return ops


@pytest.fixture(scope="module")
def eval_surface_module():
    return load_eval_surface_module()


class TestOperatorAgreement:
    """mean/std/corr must match qlib's real operators to floating-point precision."""

    def test_mean_std_corr_agree_to_float_precision(self, ops_module) -> None:
        cases = run_agreement_cases(ops_module)
        assert len(cases) == 3
        for c in cases:
            assert c.diff < 1e-9, f"{c.op} diverged: argus={c.argus_value} qlib={c.qlib_value}"

    def test_rank_agrees_when_there_are_no_ties_at_the_current_value(self, ops_module) -> None:
        cases = {c.name: c for c in run_rank_cases(ops_module)}
        assert cases["monotonic_up"].agrees
        assert cases["mixed_unique_max"].agrees


class TestRankConventionDivergence:
    """The real, measured, algebraically explained normalization-convention gap."""

    def test_constant_window_diverges_and_the_explanation_matches_measurement(
        self, ops_module
    ) -> None:
        cases = run_rank_cases(ops_module)
        d = rank_convention_divergence(cases)
        assert not next(c for c in cases if c.name == "constant_5").agrees
        assert d.explanation_matches_measurement
        assert d.argus_constant_rank == 0.5
        assert d.measured_qlib_constant_rank == pytest.approx(0.6)

    def test_single_observation_window_diverges_by_design_not_by_bug(self, ops_module) -> None:
        """ARGUS (post rank-bug-fix, see grammar.py's own comment) fixes N=1 at 0.5; qlib's real
        Rank gives a lone observation percentile 1.0 — a genuine convention difference between
        the two, not evidence either is broken."""
        cases = {c.name: c for c in run_rank_cases(ops_module)}
        single = cases["single_obs"]
        assert single.argus_value == 0.5
        assert single.qlib_value == 1.0
        assert not single.agrees

    def test_ties_away_from_the_current_value_also_diverge(self, ops_module) -> None:
        cases = {c.name: c for c in run_rank_cases(ops_module)}
        ties = cases["ties_not_at_current"]
        assert ties.argus_value == 0.5
        assert ties.qlib_value == pytest.approx(0.6)


class TestSweptRealMarketSeries:
    """Out-of-sample: a real, captured book-tape series, not a hand-designed fixture."""

    def test_the_convention_gap_on_real_data_matches_its_algebra_exactly(
        self, ops_module
    ) -> None:
        """**This asserted "both agreement and divergence appear on real data" until 2026-09-20,
        and that was a fact about the market, not about the code.**

        `_real_mid_price_series` reads `data/book_tape.jsonl`, which the live paper-trading cycle
        keeps appending to. After four unattended days the capture had grown to 33 all-distinct
        mids in which the final price is never a running maximum — so *nothing* agreed and the
        test failed, with no regression anywhere. Agreement was never a property this code
        guarantees; it is a property of whether the last captured tick happens to be a window max.

        What IS guaranteed, and is checked here instead on every real window, is the exact
        normalization convention (verified: 31 of 31 cases, zero mismatches):

            ARGUS  rank = k / (n - 1)          k = values in the window strictly below the current
            qlib   rank = (k + 1) / n

        Those are equal iff ``k == n - 1`` — that is, **iff the current value is the window
        maximum** — which explains the old assertion's data-dependence precisely rather than
        leaving it to luck. Both sides are pinned to their formula, and the agreement rule is
        asserted as the biconditional it actually is.
        """
        results = swept_rank_cases(ops_module, symbol="NVDAUSDT")
        assert len(results) >= 5, "expected the real NVDAUSDT book-tape capture to be present"
        for case in results:
            values, n = list(case.values), case.lookback
            window, current = values[-n:], values[-1]
            below = sum(1 for v in window if v < current)
            assert case.argus_value == pytest.approx(below / (n - 1)), case.name
            assert case.qlib_value == pytest.approx((below + 1) / n), case.name
            # The biconditional: they agree exactly when the current value tops its window.
            assert case.agrees is (below == n - 1), case.name


class TestExecutionSurface:
    """The structural comparison: one engine can reach eval(), the other cannot."""

    def test_argus_grammar_has_no_forbidden_calls(self) -> None:
        scan = scan_argus_grammar()
        assert not scan.has_execution_surface
        assert scan.forbidden_calls_found == ()

    def test_qlib_eval_surface_contains_a_real_eval_call(self) -> None:
        scan = scan_qlib_eval_surface()
        assert scan.has_execution_surface
        assert "eval" in scan.forbidden_calls_found

    def test_the_injection_payload_actually_executes_through_unmodified_qlib_code(
        self, eval_surface_module
    ) -> None:
        proof = run_injection_proof(eval_surface_module)
        assert proof.executed_attacker_code
        assert proof.result == 2
        assert proof.exception is None
        # The payload must not have been rewritten by the incidental `Operators.`-prefix regex —
        # if it had, this would prove a much weaker claim (a whitelisted-operator call, not
        # arbitrary code).
        assert "Operators." not in proof.parsed or proof.parsed.count("Operators.") == 1

    def test_the_scan_is_sensitive_not_vacuous(self, tmp_path: Path) -> None:
        """A scan that always reports "clean" would make the whole comparison meaningless — this
        proves the SAME scan function flags eval() when it is genuinely present."""
        ablation = run_scan_sensitivity_ablation(tmp_path)
        assert ablation.real_grammar_scan_clean
        assert ablation.ablated_counterfactual_scan_flags_eval
        assert ablation.the_scan_is_load_bearing


class TestFailureCases:
    """Malformed (not malicious) input — the case each side's real error handling was designed
    for, distinct from the hostile-but-valid injection payload above."""

    def test_argus_rejects_an_unknown_op_at_construction(self, eval_surface_module) -> None:
        cases = {c.input_: c for c in run_failure_cases(eval_surface_module)}
        assert "at construction" in cases["Window('not_a_real_op', 5, ...)"].outcome

    def test_qlib_catches_malformed_syntax_cleanly(self, eval_surface_module) -> None:
        cases = run_failure_cases(eval_surface_module)
        qlib_cases = [c for c in cases if c.system == "qlib"]
        assert len(qlib_cases) == 3
        assert all("SyntaxError caught" in c.outcome for c in qlib_cases)


class TestReproducibility:
    def test_both_are_reproducible_on_identical_input(self, ops_module) -> None:
        result = run_reproducibility_check(ops_module)
        assert result["argus_reproducible"]
        assert result["qlib_reproducible"]


class TestCosts:
    def test_costs_are_measured_and_positive(self, ops_module) -> None:
        costs = measure_costs(ops_module)
        assert costs["argus_window_us_per_call"] > 0
        assert costs["qlib_mean_us_per_call"] > 0


class TestMainAndRender:
    def test_main_returns_a_complete_serialisable_report(self, tmp_path: Path) -> None:
        report = main(tmp_dir=tmp_path)
        assert report["agreement_all_match"]
        assert report["rank_agrees_on_no_ties"]
        assert report["injection_proof"]["executed_attacker_code"]
        assert not report["argus_execution_surface_scan"]["has_execution_surface"]
        assert report["qlib_execution_surface_scan"]["has_execution_surface"]
        assert report["scope_statement"] == SCOPE_STATEMENT

    def test_render_produces_readable_text(self, tmp_path: Path) -> None:
        report = main(tmp_dir=tmp_path)
        text = render(report)
        assert "GRAMMAR COMPARISON" in text
        assert "injection proof" in text
