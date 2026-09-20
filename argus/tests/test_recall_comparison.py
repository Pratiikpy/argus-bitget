"""ARGUS's recall() vs. TradingAgents' real TradingMemoryLog: real verdicts, both real systems."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from argus.eval.baselines.tradingagents_loader import load_trading_memory_log_class
from argus.eval.recall_comparison import (
    SCOPE_STATEMENT,
    LedgerEntry,
    main,
    render,
    run_ablation,
    run_argus_recall,
    run_boundary_cases,
    run_no_floor_case,
    run_pending_inclusion_case,
    run_tradingagents_context,
)


def _memory_log_class() -> type:
    return load_trading_memory_log_class()


class TestRunArgusRecall:
    def test_a_settled_episode_before_now_is_graded(self) -> None:
        entry = LedgerEntry(
            seq=1, symbol="NVDA", decided_at="2026-01-01T00:00:00+00:00",
            settled_at="2026-01-02T00:00:00+00:00", verdict="buy", session_phase="rth",
            stated_confidence=0.8, thesis="x",
        )
        result = run_argus_recall([entry], symbol="NVDA", now=datetime(2026, 1, 3, tzinfo=UTC))
        assert len(result.graded) == 1


class TestRunTradingagentsContext:
    def test_a_resolved_entry_before_as_of_is_visible(self) -> None:
        context = run_tradingagents_context(
            ticker="NVDA", trade_date="2026-01-01", decision_text="buy",
            resolution_date="2026-01-02", as_of="2026-01-03",
            memory_log_class=_memory_log_class(),
        )
        assert context != ""

    def test_a_resolved_entry_after_as_of_is_hidden(self) -> None:
        context = run_tradingagents_context(
            ticker="NVDA", trade_date="2026-01-01", decision_text="buy",
            resolution_date="2026-01-05", as_of="2026-01-02",
            memory_log_class=_memory_log_class(),
        )
        assert context == ""


class TestBoundaryCases:
    def test_both_agree_on_the_two_clear_cut_cases(self) -> None:
        cases = run_boundary_cases(_memory_log_class())
        clear = {c.name: c for c in cases if c.name != "exact_resolution_instant"}
        assert clear["well_after_resolution"].argus_visible
        assert clear["well_after_resolution"].tradingagents_visible
        assert not clear["well_before_resolution"].argus_visible
        assert not clear["well_before_resolution"].tradingagents_visible

    def test_the_exact_instant_diverges_by_granularity_not_by_bug(self) -> None:
        """The real, run-verified structural finding this comparison exists to surface."""
        cases = run_boundary_cases(_memory_log_class())
        exact = next(c for c in cases if c.name == "exact_resolution_instant")
        assert exact.argus_visible is False
        assert exact.tradingagents_visible is True


class TestPendingInclusion:
    def test_argus_shows_a_pending_episode_and_tradingagents_does_not(self) -> None:
        result = run_pending_inclusion_case(_memory_log_class())
        assert result.argus_shows_pending is True
        assert result.tradingagents_shows_pending is False


class TestNoFloorCase:
    def test_argus_refuses_a_pattern_from_one_and_tradingagents_shows_content(self) -> None:
        result = run_no_floor_case(_memory_log_class())
        assert result.argus_states_a_pattern_from_one is False
        assert result.tradingagents_shows_content_from_one is True


class TestAblation:
    def test_the_floor_is_load_bearing(self) -> None:
        """Regression test for a real bug this module's own history records: a first version
        read the ablated result AFTER the monkeypatched floor had already been restored, so it
        silently got the unablated (real) answer back and reported the floor as not load-bearing
        for the wrong reason. Fixed by reading `.lessons()` while the patch is still active."""
        result = run_ablation()
        assert result.with_floor_states_pattern is False
        assert result.without_floor_would_state_pattern is True
        assert result.floor_is_load_bearing is True

    def test_the_real_floor_is_restored_after_the_ablation_runs(self) -> None:
        """The monkeypatch must not leak into any other test in the same process."""
        from argus.agents.recall import MIN_EPISODES_FOR_A_LESSON

        run_ablation()
        assert MIN_EPISODES_FOR_A_LESSON == 5


class TestScopeStatement:
    def test_names_both_real_structural_differences(self) -> None:
        assert "NOT claimed" in SCOPE_STATEMENT
        assert "pending" in SCOPE_STATEMENT.lower()
        assert "MIN_EPISODES_FOR_A_LESSON" in SCOPE_STATEMENT


class TestMain:
    def test_main_runs_end_to_end_and_render_produces_readable_text(self) -> None:
        report = main()
        assert report["boundary_agreement_on_clear_cases"] is True
        assert report["ablation"]["floor_is_load_bearing"] is True
        text = render(report)
        assert "RECALL COMPARISON" in text
        assert "pending inclusion" in text


class TestRecallHurdle:
    def test_the_abstention_lesson_arithmetic_is_reproducible(self) -> None:
        """Sanity check that the fixtures used throughout this module produce a real, readable
        lesson, not just a truthy non-empty tuple."""
        entries = [
            LedgerEntry(
                seq=i, symbol="MSFT", decided_at=f"2026-01-0{i}T00:00:00+00:00",
                settled_at=f"2026-01-0{i + 1}T00:00:00+00:00", verdict="no_trade",
                session_phase="rth", stated_confidence=0.5, thesis="x",
                counterfactual_move_bps="15",
            )
            for i in range(1, 6)
        ]
        result = run_argus_recall(entries, symbol="MSFT", now=datetime(2026, 1, 10, tzinfo=UTC))
        lessons = result.lessons(hurdle_bps=Decimal("5"))
        assert lessons
        assert "MSFT" in lessons[0]
