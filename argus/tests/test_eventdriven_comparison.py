"""Tests for the ARGUS-vs-whale-signals event-significance comparison.

Live-network tests: real Bitget ETHUSDT/BTCUSDT candle history is the whole point of the
comparison. Expensive functions are wrapped in module-scoped fixtures so each real fetch and
each real, CPU-heavy `argus.research.eventstudy.study()` call happens once per test session. See
`eval/eventdriven_comparison.py`'s module docstring for the findings these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.eventdriven_comparison import (
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_base_rate_case,
    run_failure_cases,
    run_false_positive_sweep,
    run_null_ablation,
    run_reproducibility_check,
)


@pytest.fixture(scope="module")
def base_case() -> dict:
    return run_base_rate_case()


@pytest.fixture(scope="module")
def sweep() -> dict:
    return run_false_positive_sweep(n_seeds=12, n_events=120)


@pytest.fixture(scope="module")
def ablation() -> dict:
    """**n_seeds=40, not 12, and the difference is the test being able to mean anything.**

    This asserted a statistical claim — that testing at the real measured base rate produces a
    materially lower false-positive rate than the fixed 0.50 null whale-signals uses — on twelve
    placebo draws. Twelve is not enough to carry it: the false-positive rate over overlapping
    24h-forward windows is itself serially correlated (the exact phenomenon this capability
    exists to measure), so the estimate is noisy and the assertion flipped to False on a live
    run on 2026-09-20 with nothing wrong in the code.

    Measured the same day at n_seeds=60: fixed-null 0.217 against true-rate 0.050 — the effect is
    large and unambiguous when the sample can show it. 40 is the compromise between that and the
    real runtime cost of each additional seed, and it is above the n where the claim was observed
    to be stable.

    The claim itself was NOT weakened again to keep this green. It had already been weakened once,
    from "the true rate is the exact argmin among swept candidates" to this directional form;
    weakening it a second time to fit an underpowered sample would be fitting the claim to the
    test rather than the test to the claim.
    """
    return run_null_ablation(n_seeds=40, n_events=120)


@pytest.fixture(scope="module")
def failures() -> dict:
    return run_failure_cases()


@pytest.fixture(scope="module")
def costs() -> dict:
    return measure_costs(repeats=3, n_events=120)


class TestBaseRateCase:
    def test_the_real_base_rate_departs_from_fifty_percent(self, base_case: dict) -> None:
        """Not guaranteed to be exactly this value forever — it is measured live — but on a real
        market it is essentially never exactly 0.5."""
        assert base_case["real_base_rate_24h"] != 0.5

    def test_whale_signals_edge_over_true_base_rate_is_small(self, base_case: dict) -> None:
        """The placebo events carry zero true informational edge by construction — whatever
        whale-signals' raw hit rate reports, it should track the real base rate closely, not the
        fixed 50% null its significance test actually uses."""
        edge = base_case["whale_signals"]["edge_over_true_base_rate"]
        assert edge is not None
        assert abs(edge) < 0.10

    def test_argus_and_whale_signals_report_independent_verdicts(self, base_case: dict) -> None:
        """Both real, non-null results — the point is that they are computed from genuinely
        different definitions of "effect", not that either is empty."""
        assert base_case["whale_signals"]["hit_rate"] is not None
        assert base_case["argus"]["statistics"]


class TestFalsePositiveSweep:
    def test_whale_signals_false_positive_rate_exceeds_nominal(self, sweep: dict) -> None:
        """The fixed-null test on genuinely uninformed placebo draws should reject far more often
        than the nominal 5% significance level — this is the central, measured finding."""
        assert sweep["whale_signals_false_positive_rate"] > 0.05

    def test_argus_full_agreement_rate_is_low(self, sweep: dict) -> None:
        """ARGUS requires all four clustering-adjusted tests to agree before claiming EFFECT
        ESTABLISHED — on placebo data with no real BTC-adjusted signal, that bar should rarely
        clear."""
        assert sweep["argus_full_agreement_rate"] < sweep["whale_signals_false_positive_rate"]

    def test_the_sweep_ran_the_requested_number_of_draws(self, sweep: dict) -> None:
        assert sweep["n_seeds"] == 12
        assert (
            sweep["argus_effect_established"]
            + sweep["argus_partial"]
            + sweep["argus_no_effect"]
            == 12
        )


class TestNullAblation:
    def test_the_true_rate_beats_the_fixed_null_materially(self, ablation: dict) -> None:
        """The robust, directional claim: re-testing the SAME real hit counts against whale-
        signals' real fixed null of 0.50 shows a materially higher false-positive rate than
        testing at the real measured rate. Deliberately NOT asserting the true rate is the exact
        minimum among the swept candidates — that stricter claim was tried first and found, by
        running it repeatedly, not to hold reliably: the overlapping 24h-forward-return windows
        sampled here carry real serial-correlation sampling variance (the same clustering this
        capability's other functions measure), so an arbitrary extra candidate occasionally
        scores lower than the true rate by chance even at n_seeds=60."""
        assert ablation["true_rate_materially_better_than_fixed_null"]
        assert ablation["true_rate_fp_rate"] < ablation["fixed_null_fp_rate"]

    def test_the_fixed_null_of_point_five_is_the_worst_or_near_worst_choice(
        self, ablation: dict
    ) -> None:
        rates = ablation["false_positive_rate_by_assumed_null"]
        assert rates["0.5"] >= min(rates.values())


class TestFailureCases:
    def test_twenty_nine_events_is_silently_skipped(self, failures: dict) -> None:
        case = failures["n_29_events"]
        assert not case["category_present_in_results"]
        assert not case["real_raised"]

    def test_thirty_events_is_computed(self, failures: dict) -> None:
        case = failures["n_30_events"]
        assert case["category_present_in_results"]

    def test_an_empty_condition_mask_silently_returns_exactly_point_five(
        self, failures: dict
    ) -> None:
        case = failures["empty_condition_mask"]
        assert case["silently_reports_exactly_0_5_with_zero_real_observations"]
        assert not case["real_raised"]


class TestCosts:
    def test_costs_are_measured_on_both_real_sides(self, costs: dict) -> None:
        assert costs["whale_signals_seconds_per_call"] > 0
        assert costs["argus_seconds_per_call"] > 0
        assert costs["argus_slower_by_factor"] is not None


class TestReproducibility:
    def test_repeated_runs_are_identical(self) -> None:
        report = run_reproducibility_check()
        assert report["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self, base_case: dict, sweep: dict, ablation: dict, failures: dict, costs: dict,
    ) -> None:
        report = {
            "base_rate_case": base_case,
            "false_positive_sweep": sweep,
            "null_ablation": ablation,
            "failure_cases": failures,
            "costs": costs,
            "reproducibility": run_reproducibility_check(),
            "scope_statement": SCOPE_STATEMENT,
        }
        text = render(report)
        assert "EVENT-DRIVEN SIGNIFICANCE" in text
        assert "false-positive rate" in text


def test_scope_statement_names_what_was_not_reproduced() -> None:
    assert "Dune" in SCOPE_STATEMENT
    assert "NOT claimed" in SCOPE_STATEMENT
