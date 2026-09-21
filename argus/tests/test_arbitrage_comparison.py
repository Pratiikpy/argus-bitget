"""Tests for the ARGUS-vs-maxme/bitcoin-arbitrage comparison.

Every case runs real code: `maxme/bitcoin-arbitrage`'s real, vendored, unmodified profit-detection
methods, and ARGUS's real `research/arbitrage_study.py` decomposition. No live network calls — all
cases use constructed order-book depth and `BasisPoint`s. See
`eval/arbitrage_comparison.py`'s module docstring for the finding these tests pin.
"""

from __future__ import annotations

from types import ModuleType

import pytest

from argus.eval.arbitrage_comparison import (
    main,
    measure_costs,
    render,
    run_designed_cases,
    run_failure_cases,
    run_fee_ablation,
    run_reproducibility_check,
    run_swept_cases,
    scope_statement,
)
from argus.eval.baselines.maxme_arbitrer_loader import load_arbitrer_module


@pytest.fixture(scope="module")
def arbitrer_module() -> ModuleType:
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
    def test_full_costs_refuse_and_zeroing_everything_reproduces_maxmes_verdict(self) -> None:
        """The two ends of the ablation are stable regardless of which basis figures feed it:
        with every real cost applied ARGUS refuses, and with the whole stack zeroed it agrees
        with maxme's fee-blind verdict — this is what the ablation exists to bracket."""
        points = {p.stage: p for p in run_fee_ablation()}
        assert points["all_real_costs"].argus_monetizable is False
        assert points["all_costs_zeroed"].argus_monetizable is True

    def test_whether_the_fee_alone_flips_it_is_read_from_the_live_median_not_assumed(self) -> None:
        """This assertion used to read `is False` unconditionally: at the study's old, stale
        median (1.12bps) spread-crossing + slippage (2.6bps) alone exceeded the spread, so
        zeroing the fee term in isolation never flipped the verdict. The study since moved to an
        absolute basis and the real median is 3.16bps — above that same 2.6bps — so on the live
        data fee_zeroed_only genuinely IS monetizable now. Asserted against `real_basis()`'s own
        current number rather than re-pinned to a literal, so this test cannot go stale the same
        way the code it tests just did."""
        from argus.eval.arbitrage_comparison import real_basis

        median, _, _ = real_basis()
        points = {p.stage: p for p in run_fee_ablation()}
        non_fee_costs_bps = 0.6 + 2.0  # spread-crossing + slippage, both fixed policy constants
        assert points["fee_zeroed_only"].argus_monetizable is (median > non_fee_costs_bps)


class TestFailureCases:
    def test_maxmes_real_code_crashes_on_an_empty_order_book(
        self, arbitrer_module: ModuleType
    ) -> None:
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
        # NOT asserted here whether fee-alone flips the verdict: that is read from the live
        # median in TestCostStackAblation and is not stable across a study re-run, unlike the
        # two ends of the ablation this method already checks.
        import json

        json.dumps(report)

    def test_the_reports_scope_statement_matches_what_scope_statement_would_build_now(self) -> None:
        """`scope_statement` takes the live designed cases and ablation as arguments, so the only
        way this could drift from `main()`'s own report is `main()` passing something other than
        its own live results into it — this is the guard against that."""
        from argus.eval.arbitrage_comparison import real_basis

        report = main()
        rebuilt = scope_statement(
            run_designed_cases(), run_fee_ablation(), *real_basis(),
        )
        assert report["scope_statement"] == rebuilt

    def test_render_produces_readable_text(self) -> None:
        report = main()
        text = render(report)
        assert "ARBITRAGE COMPARISON" in text
        assert "cost stack ablation" in text
