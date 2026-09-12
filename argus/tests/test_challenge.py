"""Challenge Mode tests.

A control that cannot fail is not a test. These assert both that each defence fires AND that the
control is actually exercising the thing it claims to.
"""

from __future__ import annotations

from argus.eval.challenge import CONTROLS, run_all


class TestAllControls:
    def test_thirteen_controls_exist(self) -> None:
        assert len(CONTROLS) == 13

    def test_every_defence_holds(self) -> None:
        got = run_all()
        assert got["all_defences_held"] is True, f"failed: {got['failed']}"

    def test_nine_are_original_to_argus(self) -> None:
        """TraderBench covers 1 outright and 3 partially; the rest are ours."""
        assert run_all()["original_to_argus"] == 9

    def test_each_control_names_its_expected_defence(self) -> None:
        """A control with no stated expectation cannot be judged."""
        for r in run_all()["results"]:
            assert r["expected_defence"], f"{r['control']} has no expectation"
            assert r["detail"], f"{r['control']} produced no evidence"


class TestControlsActuallyExercise:
    """Guard against a control that passes because it does nothing."""

    def test_poison_memory_really_had_something_to_reject(self) -> None:
        got = next(r for r in run_all()["results"] if r["control"] == "poison_memory")
        assert "of 1 poisoned" in got["detail"]

    def test_partial_fill_leaves_a_real_residual(self) -> None:
        got = next(r for r in run_all()["results"] if r["control"] == "force_partial_fill")
        assert "residual 60" in got["detail"]

    def test_holiday_actually_changes_the_session(self) -> None:
        got = next(r for r in run_all()["results"] if r["control"] == "move_the_market_open")
        assert "rth -> holiday" in got["detail"]

    def test_restatement_shows_both_views(self) -> None:
        got = next(r for r in run_all()["results"] if r["control"] == "alter_one_source")
        assert "4.20bn" in got["detail"] and "3.95bn" in got["detail"]
