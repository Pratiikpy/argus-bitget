"""ARGUS's review.evaluate() vs. TradingAgents' real TradingMemoryLog: real, not asserted."""

from __future__ import annotations

from argus.desk.review import Status
from argus.eval.baselines.tradingagents_loader import load_trading_memory_log_class
from argus.eval.review_comparison import (
    SCOPE_STATEMENT,
    main,
    render,
    run_lifecycle,
    run_lifecycle_cases,
    run_threshold_ablations,
    run_unconditional_reuse_case,
)


def _memory_log_class() -> type:
    return load_trading_memory_log_class()


class TestUnconditionalReuse:
    def test_tradingagents_reinjects_a_demonstrably_wrong_call(self) -> None:
        """The real, run-verified finding this comparison exists to surface."""
        result = run_unconditional_reuse_case(_memory_log_class())
        assert result.wrong_call_reinjected is True
        assert result.right_call_reinjected is True
        assert result.no_precision_gate is True


class TestLifecycleCases:
    def test_every_designed_case_lands_on_its_intended_status(self) -> None:
        cases = {c.name: c.status for c in run_lifecycle_cases()}
        assert cases["below_min_decisions"] == Status.PROPOSED
        assert cases["never_fires"] == Status.DEAD_WEIGHT
        assert cases["fires_on_almost_everything"] == Status.NO_DISCRIMINATION
        assert cases["fires_selectively_mostly_wrong"] == Status.MISLEADING
        assert cases["fires_selectively_always_right_below_min_firings"] == Status.EARNING
        assert cases["fires_selectively_always_right_at_min_firings"] == Status.ACTIVE

    def test_the_run_reports_itself_sound(self) -> None:
        run = run_lifecycle()
        assert run.design_is_sound, run.mismatches


class TestThresholdAblations:
    def test_all_four_thresholds_are_independently_load_bearing(self) -> None:
        """NEVER_FIRES and MIN_FIRINGS were added 2026-09-21 — the function used to return 2,
        docstring claiming 4, an overclaim caught by an AUDIT sweep and closed as a queued BUILD
        task rather than rushed. This asserts the real count, not a re-typed literal, would be
        the more thorough guard — but the specific dimension names matter more here, since a
        length-4 result with a wrong or duplicated dimension would pass a bare count check."""
        results = run_threshold_ablations()
        by_dim = {r.dimension: r for r in results}
        assert set(by_dim) == {
            "min_precision", "always_fires_ceiling", "never_fires_floor", "min_firings_floor",
        }
        for r in results:
            assert r.differs, r.dimension
            assert r.real_status == Status.ACTIVE, r.dimension

    def test_the_real_thresholds_are_restored_after_ablation(self) -> None:
        """The monkeypatches must not leak into any other test in the same process."""
        from argus.desk.review import ALWAYS_FIRES, MIN_FIRINGS, MIN_PRECISION, NEVER_FIRES

        run_threshold_ablations()
        assert MIN_PRECISION == 0.5
        assert ALWAYS_FIRES == 0.9
        assert NEVER_FIRES == 0.0
        assert MIN_FIRINGS == 5


class TestScopeStatement:
    def test_names_what_is_and_is_not_claimed(self) -> None:
        assert "NOT claimed" in SCOPE_STATEMENT
        assert "zero matches" in SCOPE_STATEMENT

    def test_claims_all_four_thresholds_not_two(self) -> None:
        """This used to say 'Two of ARGUS's four' — accurate then, wrong now that all four are
        genuinely ablated. Regression guard for the overclaim this replaced."""
        assert "All four of ARGUS's" in SCOPE_STATEMENT
        assert "Two of ARGUS's four" not in SCOPE_STATEMENT


class TestMain:
    def test_main_runs_end_to_end_and_render_produces_readable_text(self) -> None:
        report = main()
        assert report["unconditional_reuse"]["no_precision_gate"] is True
        assert report["lifecycle"]["design_is_sound"] is True
        assert report["ablation_all_load_bearing"] is True
        text = render(report)
        assert "REVIEW COMPARISON" in text
        assert "lifecycle cases" in text
