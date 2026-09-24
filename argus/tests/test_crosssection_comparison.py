"""ARGUS's crossrank vs. qlib's real CSRankNorm: real verdicts from both real systems."""

from __future__ import annotations

from argus.eval.baselines.qlib_loader import QlibCrossSectionSymbols, load_qlib_baseline
from argus.eval.crosssection_comparison import (
    ablation_cases,
    compare_ordering,
    designed_panels,
    main,
    render,
    run_argus,
    run_qlib,
    scope_statement,
    single_name_distortion_cost,
    swept_panels,
)


def _baseline() -> QlibCrossSectionSymbols:
    return load_qlib_baseline()


class TestRunArgus:
    def test_produces_zero_to_one_scale(self) -> None:
        out = run_argus([1.0, 2.0, 3.0])
        assert out == [0.0, 0.5, 1.0]

    def test_single_instrument_is_neutral(self) -> None:
        assert run_argus([9.0]) == [0.5]

    def test_empty_is_empty(self) -> None:
        assert run_argus([]) == []


class TestRunQlib:
    def test_produces_the_unit_std_scale(self) -> None:
        out = run_qlib([1.0, 2.0, 3.0], _baseline().cs_rank_norm)
        assert out[0] < 0 < out[2]

    def test_single_instrument_is_an_extreme_not_neutral(self) -> None:
        """The real, run-verified divergence this whole comparison exists to surface."""
        out = run_qlib([9.0], _baseline().cs_rank_norm)
        assert out[0] == 1.73

    def test_empty_is_empty(self) -> None:
        assert run_qlib([], _baseline().cs_rank_norm) == []


class TestCompareOrdering:
    def test_they_agree_on_order_despite_different_scales(self) -> None:
        result = compare_ordering([5.0, 3.0, 3.0, 1.0], _baseline().cs_rank_norm)
        assert result.same_order
        assert result.argus_output != result.qlib_output  # different scale, same order

    def test_ties_are_grouped_identically(self) -> None:
        result = compare_ordering([1.0, 1.0, 1.0], _baseline().cs_rank_norm)
        assert result.same_order


class TestDesignedPanels:
    def test_every_designed_panel_agrees_on_order(self) -> None:
        baseline = _baseline()
        for panel in designed_panels():
            result = compare_ordering(list(panel), baseline.cs_rank_norm)
            assert result.same_order, panel

    def test_the_single_instrument_panel_is_included(self) -> None:
        assert (0.0,) in designed_panels()

    def test_the_empty_panel_is_included(self) -> None:
        assert () in designed_panels()


class TestSweptPanels:
    def test_the_grid_size_matches_what_this_test_expects(self) -> None:
        assert len(swept_panels()) == 6 * 3 * 3

    def test_every_swept_panel_agrees_on_order(self) -> None:
        baseline = _baseline()
        mismatches = [
            p for p in swept_panels()
            if not compare_ordering(list(p), baseline.cs_rank_norm).same_order
        ]
        assert not mismatches, mismatches


class TestAblation:
    def test_both_design_choices_are_load_bearing(self) -> None:
        cases = ablation_cases()
        assert len(cases) == 2
        for case in cases:
            assert case.differs, case.dimension

    def test_midrank_case_uses_a_real_tie(self) -> None:
        case = next(c for c in ablation_cases() if c.dimension == "midrank_tie_handling")
        assert case.real_output[1] == case.real_output[2]  # the tied pair really did tie


class TestSingleNameDistortionCost:
    def test_the_distortion_is_large_relative_to_neutral(self) -> None:
        cost = single_name_distortion_cost(_baseline().cs_rank_norm)
        assert cost.distortion_magnitude > 1.0
        assert cost.qlib_single_name_value == 1.73
        assert cost.argus_single_name_value == 0.5


class TestScopeStatement:
    def test_names_the_real_divergence_and_the_real_design_boundary(self) -> None:
        text = scope_statement(54)
        assert "CrossScale" in text
        assert "NOT claimed" in text
        assert "1.73" in text

    def test_carries_the_live_sweep_count(self) -> None:
        assert "77" in scope_statement(77)
        assert "78" not in scope_statement(77)


class TestMain:
    def test_main_runs_end_to_end_and_render_produces_readable_text(self) -> None:
        report = main()
        assert report["designed_all_agree"] is True
        assert report["swept_all_agree"] is True
        assert report["ablation_all_load_bearing"] is True
        text = render(report)
        assert "CROSS-SECTION COMPARISON" in text
        assert "single-name distortion" in text
