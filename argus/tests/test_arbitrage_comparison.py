"""Tests for the ARGUS-vs-maxme/bitcoin-arbitrage comparison.

Every case runs real code: `maxme/bitcoin-arbitrage`'s real, vendored, unmodified profit-detection
methods, and ARGUS's real `research/arbitrage_study.py` decomposition. No live network calls — all
cases use constructed order-book depth and `BasisPoint`s. See
`eval/arbitrage_comparison.py`'s module docstring for the finding these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.arbitrage_comparison import (
    SCOPE_STATEMENT,
    main,
    measure_costs,
    render,
    run_designed_cases,
    run_failure_cases,
    run_fee_ablation,
    run_reproducibility_check,
    run_swept_cases,
)
from argus.eval.baselines.maxme_arbitrer_loader import load_arbitrer_module


@pytest.fixture(scope="module")
def arbitrer_module():
    return load_arbitrer_module()


class TestDesignedCases:
    def test_the_real_maxme_detector_reports_profit_on_the_median_real_spread(self) -> None:
        cases = {c.name: c for c in run_designed_cases()}
        median = cases["median_real_basis"]
        assert median.maxme_reports_opportunity
        assert median.maxme_profit > 0

    def test_argus_refuses_the_median_and_p95_real_spreads_as_not_monetizable(self) -> None:
        cases = {c.name: c for c in run_designed_cases()}
        assert not cases["median_real_basis"].argus_decomposition.is_monetizable
        assert not cases["p95_real_basis"].argus_decomposition.is_monetizable
        assert cases["median_real_basis"].disagreement
        assert cases["p95_real_basis"].disagreement

    def test_the_largest_observed_spread_genuinely_clears_costs_on_both_sides(self) -> None:
        """Not every case is a disagreement — a large enough real spread clears ARGUS's own real
        cost stack too, and this must be reported honestly, not glossed over as a third
        disagreement to make the finding look more sweeping than it is."""
        cases = {c.name: c for c in run_designed_cases()}
        largest = cases["max_observed_real_basis"]
        assert largest.maxme_reports_opportunity
        assert largest.argus_decomposition.is_monetizable
        assert not largest.disagreement


class TestSweptCases:
    def test_the_swept_distribution_shows_a_real_disagreement_rate(self) -> None:
        summary = run_swept_cases(n=40)
        assert summary.n == 40
        assert summary.n_disagreements > 0
        assert 0.0 < summary.disagreement_rate <= 1.0


class TestCostStackAblation:
    def test_fee_alone_does_not_explain_the_disagreement(self) -> None:
        """The real result of running this ablation, not the one originally planned — spread-
        crossing and slippage alone already exceed the median real spread even at zero fee."""
        points = {p.stage: p for p in run_fee_ablation()}
        assert points["all_real_costs"].argus_monetizable is False
        assert points["fee_zeroed_only"].argus_monetizable is False

    def test_the_full_cost_stack_zeroed_reproduces_maxmes_naive_verdict(self) -> None:
        points = {p.stage: p for p in run_fee_ablation()}
        assert points["all_costs_zeroed"].argus_monetizable is True


class TestFailureCases:
    def test_maxmes_real_code_crashes_on_an_empty_order_book(self, arbitrer_module) -> None:
        """The real, verified behaviour — get_profit_for()'s own first line indexes
        `self.depths[kask]["asks"][mi]` with no length guard, and `get_max_depth()`'s own guards
        still leave `mi=mj=0` when both sides are empty, so index 0 is read from an empty list.
        A genuine crash in the real, unmodified competitor code on a plausible real input (an
        exchange with a temporarily empty book side), not an assumption about how it "should"
        behave — this test originally assumed it would return cleanly and was wrong; fixed to
        assert the real, observed outcome instead of the guessed one."""
        cases = {c.system: c for c in run_failure_cases()}
        assert "IndexError" in cases["maxme"].outcome

    def test_argus_handles_a_zero_index_basis_point(self) -> None:
        cases = {c.system: c for c in run_failure_cases()}
        assert "basis_bps=0" in cases["argus"].outcome


class TestCosts:
    def test_costs_are_measured_on_both_real_sides(self) -> None:
        costs = measure_costs()
        assert costs["maxme_arbitrage_depth_opportunity_seconds_per_call"] > 0
        assert costs["argus_decompose_seconds_per_call"] > 0


class TestReproducibility:
    def test_both_are_reproducible_on_identical_input(self) -> None:
        result = run_reproducibility_check()
        assert result["maxme_reproducible"]
        assert result["argus_reproducible"]


class TestMainAndRender:
    def test_main_returns_a_complete_serialisable_report(self) -> None:
        report = main()
        assert report["any_designed_disagreement"]
        assert report["cost_stack_ablation_confirms_the_full_stack_is_the_mechanism"]
        assert report["fee_alone_does_not_explain_the_disagreement"]
        assert report["scope_statement"] == SCOPE_STATEMENT
        import json

        json.dumps(report)

    def test_render_produces_readable_text(self) -> None:
        report = main()
        text = render(report)
        assert "ARBITRAGE COMPARISON" in text
        assert "cost stack ablation" in text
