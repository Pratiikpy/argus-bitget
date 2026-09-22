"""Tests for the ARGUS-HRP-vs-Riskfolio-Lib-and-cvxportfolio allocation comparison.

Live-network tests: real Bitget hourly candle history for the real 12-symbol rToken universe is
the same input both baselines must see, and that is the whole point of the comparison. The fetch
is expensive, so it happens once per session in a module-scoped fixture and every derived run
reuses those columns -- the same fetch-once-compute-twice discipline the module itself follows.

These tests deliberately do NOT pin which allocator wins. The result of this comparison is a
measurement of live market data and it is allowed to move; what is pinned is that the comparison
is honest -- that both baselines really ran, that the covariance really is shared, that the parity
control really behaves as a null, and that the verdict fields agree with the numbers they claim to
summarise. A test asserting "ARGUS wins" would be exactly the flattering comparison this module
exists to replace. See `eval/allocation_comparison.py`'s own docstring for what each run means.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from argus.desk.allocation import MIN_OBSERVATIONS, hrp_weights
from argus.desk.portfolio import covariance_matrix
from argus.eval.allocation_comparison import (
    RISKFOLIO_VARIANTS,
    SCOPE_STATEMENT,
    SHARPE_BAND,
    convex_first_step,
    load_returns,
    measure_costs,
    render,
    riskfolio_weights,
    run_adversarial_covariance,
    run_convex_rebalance,
    run_failure_cases,
    run_oos_variance,
    run_parity,
    run_rebalance_sweep,
    run_reproducibility_check,
    run_sharpe_sensitivity,
)


@pytest.fixture(scope="module")
def market() -> tuple[list[str], dict[str, list[float]]]:
    names, columns, failures = load_returns()
    if len(names) < 3 or len(columns[names[0]]) < MIN_OBSERVATIONS:
        pytest.skip(f"insufficient live rToken history to compare allocators: {failures}")
    return names, columns


@pytest.fixture(scope="module")
def columns(market: tuple[list[str], dict[str, list[float]]]) -> dict[str, list[float]]:
    return market[1]


@pytest.fixture(scope="module")
def parity(columns: dict[str, list[float]]) -> dict[str, Any]:
    return run_parity(columns)


@pytest.fixture(scope="module")
def oos(columns: dict[str, list[float]]) -> dict[str, Any]:
    return run_oos_variance(columns)


@pytest.fixture(scope="module")
def convex(columns: dict[str, list[float]]) -> dict[str, Any]:
    return run_convex_rebalance(columns)


@pytest.fixture(scope="module")
def band(columns: dict[str, list[float]]) -> dict[str, Any]:
    return run_sharpe_sensitivity(columns)


@pytest.fixture(scope="module")
def sweep(columns: dict[str, list[float]]) -> dict[str, Any]:
    return run_rebalance_sweep(columns)


class TestTheBaselinesActuallyRan:
    """The failure this module exists to correct was a baseline that was cloned and never run.
    A comparison that silently degrades to "the library would not import, so ARGUS wins by
    default" would reproduce that failure with a green test suite on top of it."""

    def test_riskfolio_produces_real_weights(self, columns: dict[str, list[float]]) -> None:
        weights, _noise = riskfolio_weights(
            columns, model="HRP", linkage="single", leaf_order=False
        )
        assert len(weights) == len(columns)
        assert all(w >= 0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_most_riskfolio_variants_run(self, parity: dict[str, Any]) -> None:
        ran = [k for k, v in parity["variants"].items() if "error" not in v]
        assert len(ran) >= len(RISKFOLIO_VARIANTS) - 1

    def test_herc_is_reachable_through_the_documented_shim(self, parity: dict[str, Any]) -> None:
        # Riskfolio 7.3.0's own `optimization()` cannot call its own HERC bisection (see the
        # module docstring). If this ever passes without the shim, the upstream bug was fixed and
        # `_herc_compatible` can be retired — which is worth knowing, so it is asserted rather
        # than assumed.
        assert "error" not in parity["variants"]["riskfolio_herc_ward"]

    def test_cvxportfolio_solves_and_returns_a_fully_invested_long_only_book(
        self, columns: dict[str, list[float]]
    ) -> None:
        built = covariance_matrix(columns)
        assert built is not None
        names, cov = built
        step = convex_first_step(
            dict.fromkeys(names, 1.0 / len(names)), names, cov, gamma=5.0, horizon_bars=24,
        )
        after: dict[str, float] = step["weights_after"]
        assert sum(after.values()) == pytest.approx(1.0, abs=1e-4)
        assert min(after.values()) >= -1e-6


class TestParity:
    def test_both_sides_see_the_same_covariance(self, parity: dict[str, Any]) -> None:
        # Not a formality: if the two libraries disagreed about the covariance, every weight
        # difference below would be a data-handling artefact rather than an allocation result.
        assert parity["covariance_max_abs_diff"] < 1e-15

    def test_argus_reproduces_riskfolio_under_riskfolios_own_configuration(
        self, parity: dict[str, Any]
    ) -> None:
        twin = parity["variants"]["riskfolio_hrp_single_no_leaf_order"]
        assert twin["max_abs_weight_diff_vs_argus"] < 1e-12
        assert parity["argus_reproduces_riskfolio_exactly"] is True

    def test_riskfolios_shipped_default_genuinely_diverges(self, parity: dict[str, Any]) -> None:
        # `leaf_order=True` is the single keyword between the two. If this ever collapsed to zero,
        # the divergence claim in the scope statement would be false and must be withdrawn.
        default = parity["variants"]["riskfolio_hrp_default"]
        assert default["max_abs_weight_diff_vs_argus"] > 1e-6

    def test_argus_weights_are_a_long_only_simplex(self, parity: dict[str, Any]) -> None:
        weights: dict[str, float] = parity["argus_weights"]
        assert all(w >= 0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)


class TestOutOfSampleVariance:
    def test_the_walk_forward_produced_several_held_out_windows(
        self, oos: dict[str, Any]
    ) -> None:
        assert oos["n_origins"] >= 5

    def test_the_parity_twin_is_a_null_control(self, oos: dict[str, Any]) -> None:
        """The twin is ARGUS's own weights under another name, so the harness must find no
        difference in any window. A harness that scores its own control as a winner is measuring
        itself — which is exactly what happened before ties were handled: one-ULP gaps fell the
        same way every window and the sign test called two identical allocations different at
        p=0.004."""
        control = oos["parity_twin_null_control"]
        assert control["oos_vol_ratio_vs_argus"] == pytest.approx(1.0, abs=1e-9)
        assert control["windows_tied_with_argus"] == control["windows_compared"]
        assert control["sign_test_p"] is None
        assert control["behaves_as_a_null"] is True

    def test_the_verdict_fields_match_the_numbers_they_summarise(
        self, oos: dict[str, Any]
    ) -> None:
        results = oos["results"]
        argus = results["argus_hrp"]["mean_oos_vol_bps"]
        # The same relative tolerance the module uses. A bare `<` here would demand that the
        # parity twin — ARGUS's own weights, whose mean differs in the last bit — be listed as a
        # baseline that beats ARGUS, which is the very misreading the tolerance exists to stop.
        recomputed = sorted(
            k for k, v in results.items()
            if k != "argus_hrp" and v["mean_oos_vol_bps"] < argus * (1.0 - 1e-9)
        )
        assert sorted(oos["baselines_with_lower_mean_oos_vol"]) == recomputed
        assert oos["argus_wins"] == (oos["winner"] == "argus_hrp")
        assert oos["ranking_by_mean_oos_vol"][0] == oos["winner"]

    def test_every_allocator_is_scored_on_the_same_windows(self, oos: dict[str, Any]) -> None:
        counts = {v["n_windows"] for v in oos["results"].values()}
        assert len(counts) == 1

    def test_wins_losses_and_ties_account_for_every_window(self, oos: dict[str, Any]) -> None:
        for label, row in oos["results"].items():
            total = (
                row["windows_beating_argus"]
                + row["windows_losing_to_argus"]
                + row["windows_tied_with_argus"]
            )
            assert total == row["windows_compared"], label

    def test_equal_weight_is_the_sanity_floor(self, oos: dict[str, Any]) -> None:
        # Twelve tokenized US equities correlate above 0.9, so a risk-aware allocation that could
        # not beat naive equal weighting out of sample would mean the covariance carries nothing.
        assert (
            oos["results"]["argus_hrp"]["mean_oos_vol_bps"]
            < oos["results"]["equal_weight"]["mean_oos_vol_bps"]
        )

    def test_concentration_is_reported_beside_every_volatility(
        self, oos: dict[str, Any]
    ) -> None:
        for label, row in oos["results"].items():
            assert 1.0 <= row["mean_effective_positions"] <= oos["n_instruments"] + 1e-9, label
            assert 0.0 < row["mean_largest_weight"] <= 1.0 + 1e-9, label
            assert 0.0 <= row["mean_inverse_pair_overlap"] <= 0.5 + 1e-9, label

    def test_equal_weight_holds_the_whole_universe_evenly(self, oos: dict[str, Any]) -> None:
        # The one allocator whose concentration is known in advance, so it pins the metric. Read
        # off the run's own instrument count rather than hard-coded at twelve: a venue 429 can
        # drop a symbol, and a test that fails for that reason would hide the real failure.
        size = oos["n_instruments"]
        assert oos["results"]["equal_weight"]["mean_effective_positions"] == pytest.approx(
            float(size), abs=1e-9
        )
        assert oos["results"]["equal_weight"]["mean_inverse_pair_overlap"] == pytest.approx(
            1.0 / size, abs=1e-9
        )

    def test_holm_correction_is_applied_to_every_comparison(self, oos: dict[str, Any]) -> None:
        tested = [k for k, v in oos["results"].items() if v["sign_test_p"] is not None]
        assert all("significant_after_holm" in oos["results"][k] for k in tested)
        assert set(oos["baselines_significant_after_holm"]) <= set(tested)


class TestConvexRebalance:
    def test_every_sharpe_in_the_band_was_solved(self, convex: dict[str, Any]) -> None:
        assert len(convex["rows"]) == len(SHARPE_BAND)
        assert all("error" not in row for row in convex["rows"])

    def test_agreement_is_computed_from_the_rows(self, convex: dict[str, Any]) -> None:
        rows = convex["rows"]
        expected = sum(1 for r in rows if r["agree_on_go_no_go"]) / len(rows)
        assert convex["agreement_rate"] == pytest.approx(expected)

    def test_the_two_sides_are_compared_on_the_same_rebalance(
        self, convex: dict[str, Any]
    ) -> None:
        # ARGUS's move is to a fixed HRP target and does not depend on the Sharpe assumption, so
        # its turnover must be constant across the band while only the verdict moves. If it ever
        # varied, the comparison would be conflating two different trades.
        turnovers = {round(r["argus_one_way_turnover"], 12) for r in convex["rows"]}
        assert len(turnovers) == 1

    def test_the_convex_solution_responds_to_risk_aversion(
        self, convex: dict[str, Any]
    ) -> None:
        gammas = [r["implied_gamma"] for r in convex["rows"]]
        assert gammas == sorted(gammas)
        turnovers = [r["convex_one_way_turnover"] for r in convex["rows"]]
        assert turnovers == sorted(turnovers)

    def test_the_rebalance_actually_lowers_modelled_volatility(
        self, convex: dict[str, Any]
    ) -> None:
        assert convex["argus_vol_after_bps"] < convex["argus_vol_before_bps"]


class TestSharpeSensitivity:
    def test_the_whole_band_is_reported(self, band: dict[str, Any]) -> None:
        assert [r["assumed_sharpe_annual"] for r in band["rows"]] == list(SHARPE_BAND)

    def test_break_even_is_inversely_proportional_to_the_assumption(
        self, band: dict[str, Any]
    ) -> None:
        """The single closed-form fact the flip point is solved from, checked against the
        function rather than trusted. `break_even_bars` divides by the assumed Sharpe once and
        nowhere else, so doubling the assumption must halve the payback exactly."""
        rows = {r["assumed_sharpe_annual"]: r["break_even_bars"] for r in band["rows"]}
        assert rows[0.5] == pytest.approx(rows[1.0] * 2.0, rel=1e-9)
        assert rows[2.0] == pytest.approx(rows[1.0] / 2.0, rel=1e-9)

    def test_the_flip_point_is_verified_either_side(self, band: dict[str, Any]) -> None:
        assert band["flip_sharpe_annual"] is not None
        assert band["flip_check"]["closed_form_verified"] is True
        assert band["flip_check"]["worth_doing_just_below_flip"] is False
        assert band["flip_check"]["worth_doing_just_above_flip"] is True


class TestRebalanceSweep:
    def test_the_sweep_covers_books_and_dates(self, sweep: dict[str, Any]) -> None:
        assert sweep["n_origins"] >= 4
        assert len(sweep["by_book"]) >= 6
        assert sweep["n_cells"] == sum(v["cells"] for v in sweep["by_book"].values())

    def test_a_book_already_at_the_target_never_trades(self, sweep: dict[str, Any]) -> None:
        # The one case where "do not rebalance" is true by construction. If it ever trades, either
        # `hrp_weights` is not deterministic or `min_leg` is being applied to a zero move.
        target = sweep["by_book"]["hrp_target_itself"]
        assert target["do_not_rebalance"] == target["cells"]

    def test_the_trivial_case_is_excluded_from_the_headline(self, sweep: dict[str, Any]) -> None:
        assert sweep["non_trivial_cells"] == sweep["n_cells"] - 32
        assert sweep["non_trivial_do_not_rebalance_share"] < sweep["do_not_rebalance_share"]

    def test_a_lower_assumed_sharpe_never_makes_a_rebalance_more_attractive(
        self, sweep: dict[str, Any]
    ) -> None:
        counts = [sweep["by_assumed_sharpe"][str(s)]["do_not_rebalance"] for s in SHARPE_BAND]
        assert counts == sorted(counts, reverse=True)

    def test_the_docstring_verdict_matches_the_count(self, sweep: dict[str, Any]) -> None:
        share = sweep["non_trivial_do_not_rebalance_share"]
        assert sweep["docstring_claim_holds"] == (share > 0.5)


class TestSupportingEvidence:
    def test_degenerate_inputs_are_refused_rather_than_answered(
        self, columns: dict[str, list[float]]
    ) -> None:
        cases = run_failure_cases(columns)
        assert cases["two_assets_below_min"]["argus_raises"] is True
        assert cases["zero_variance_column"]["argus_raises"] is True

    def test_a_duplicated_column_does_not_split_the_two_implementations(
        self, columns: dict[str, list[float]]
    ) -> None:
        # A perfectly collinear pair is where a naive correlation-distance implementation produces
        # a negative radicand. Both sides clip; this checks they still agree afterwards.
        cases = run_failure_cases(columns)
        duplicate = cases["duplicated_column"]
        assert "error" not in duplicate
        assert duplicate["max_abs_weight_diff_vs_argus"] < 1e-12

    def test_costs_are_measured_on_both_sides(self, columns: dict[str, list[float]]) -> None:
        costs = measure_costs(columns, repeats=1)
        assert costs["argus_hrp_seconds"] > 0
        assert costs["riskfolio_hrp_seconds"] > 0

    def test_the_comparison_is_deterministic_on_fixed_data(
        self, columns: dict[str, list[float]]
    ) -> None:
        assert run_reproducibility_check(columns)["identical"] is True


class TestAdversarialCovariance:
    """Near-singular covariance vs ARGUS's real `nco_weights` and Riskfolio's real NCO -- the
    adversarial input `TestSupportingEvidence` above never reaches, because HRP (what those tests
    exercise) never inverts a matrix at all. NCO's minimum-variance step does, which is exactly
    where a singular covariance can bite, and until `run_adversarial_covariance` existed nothing
    had ever tried it there. Added 2026-09-22 to close this capability's own `adversarial_test`
    condition in `eval/standing.py`, previously unproven."""

    def test_every_trial_produces_a_covariance_that_was_actually_built(
        self, columns: dict[str, list[float]]
    ) -> None:
        result = run_adversarial_covariance(columns, trials=20)
        assert result["n_trials"] == 20
        assert len(result["trials"]) == 20

    def test_riskfolio_never_refuses_even_on_an_exactly_singular_input(
        self, columns: dict[str, list[float]]
    ) -> None:
        """The load-bearing finding: across 20 trials spanning exact duplication (noise=0.0)
        through merely-ill-conditioned (noise=1e-4), Riskfolio's real NCO answers every time --
        it is not softened by any specific noise level, it is the library's behaviour on this
        whole family of adversarial input."""
        result = run_adversarial_covariance(columns, trials=20)
        assert result["riskfolio_refused_or_crashed"] == 0

    def test_riskfolio_flags_the_covariance_as_untrustworthy_every_time_it_proceeds(
        self, columns: dict[str, list[float]]
    ) -> None:
        """Not "Riskfolio doesn't notice" -- its own code detects the same condition ARGUS
        refuses on, prints a warning about it, and returns an answer anyway. Pinned so a future
        Riskfolio version that starts refusing outright (or stops warning) is caught rather than
        silently assumed unchanged."""
        result = run_adversarial_covariance(columns, trials=20)
        assert (
            result["riskfolio_proceeded_despite_flagging_not_positive_definite"]
            == result["n_trials"] - result["riskfolio_refused_or_crashed"]
        )

    def test_argus_refuses_on_at_least_some_of_the_sweep(
        self, columns: dict[str, list[float]]
    ) -> None:
        """Not asserting a specific count -- the RNG-drawn mix of noise levels varies which
        trials are genuinely singular -- only that the refusal mechanism actually fires on this
        real sweep rather than being dead code that never triggers."""
        result = run_adversarial_covariance(columns, trials=20)
        assert result["argus_refused"] > 0

    def test_when_both_succeed_the_weight_difference_is_reported_not_hidden(
        self, columns: dict[str, list[float]]
    ) -> None:
        """Not claiming the two allocators are close on this family -- claiming the difference,
        whatever it is, is measured and surfaced rather than only reporting the trials where
        ARGUS refused (which would flatter the "ARGUS is more careful" reading by hiding the
        cases where careful wasn't necessary and the two still diverge)."""
        result = run_adversarial_covariance(columns, trials=20)
        both = result["both_succeeded"]
        if both["n"] > 0:
            assert both["mean_max_weight_diff"] is not None
            assert both["worst_max_weight_diff"] >= both["mean_max_weight_diff"]

    def test_the_sweep_is_reproducible_for_a_fixed_seed(
        self, columns: dict[str, list[float]]
    ) -> None:
        first = run_adversarial_covariance(columns, trials=10, seed=7)
        second = run_adversarial_covariance(columns, trials=10, seed=7)
        assert [t["argus_status"] for t in first["trials"]] == [
            t["argus_status"] for t in second["trials"]
        ]


class TestReport:
    def test_render_names_every_section_and_both_baselines(
        self, parity: dict[str, Any], oos: dict[str, Any], convex: dict[str, Any],
        band: dict[str, Any], sweep: dict[str, Any],
    ) -> None:
        text = render({
            "parity": parity, "oos_variance": oos, "convex_rebalance": convex,
            "sharpe_sensitivity": band, "rebalance_sweep": sweep,
        })
        for fragment in (
            "Riskfolio-Lib", "cvxportfolio", "REALISED OUT-OF-SAMPLE VOLATILITY",
            "CONVEX REBALANCE", "SHARPE SENSITIVITY", "DO NOT REBALANCE",
        ):
            assert fragment in text

    def test_render_survives_a_report_without_the_dense_run(
        self, parity: dict[str, Any], oos: dict[str, Any], convex: dict[str, Any],
        band: dict[str, Any], sweep: dict[str, Any],
    ) -> None:
        text = render({
            "parity": parity, "oos_variance": oos, "convex_rebalance": convex,
            "sharpe_sensitivity": band, "rebalance_sweep": sweep,
        })
        assert "robustness" not in text

    def test_the_scope_statement_says_who_wins_and_what_is_not_claimed(self) -> None:
        """Was "does NOT win" until 2026-09-22, when ARGUS's own NCO (built and verified against
        the real specialist) closed the walk-forward loss to a tie -- restoring the old phrase
        would misreport a real result as an unchanged loss."""
        assert "TIED" in SCOPE_STATEMENT
        assert "nco_weights" in SCOPE_STATEMENT
        assert SCOPE_STATEMENT.count("NOT CLAIMED") >= 4
        assert "GPL-3.0" in SCOPE_STATEMENT


class TestHelpers:
    def test_realised_volatility_matches_a_hand_computation(self) -> None:
        from argus.eval.allocation_comparison import _realised_vol

        series: dict[str, Sequence[float]] = {"A": [0.01, -0.01, 0.01, -0.01]}
        # A single asset at full weight must return that asset's own sample standard deviation.
        assert _realised_vol({"A": 1.0}, series) == pytest.approx(0.0115470053837925, rel=1e-9)

    def test_effective_positions_counts_what_is_actually_held(self) -> None:
        from argus.eval.allocation_comparison import _effective_positions

        assert _effective_positions(dict.fromkeys("abcd", 0.25)) == pytest.approx(4.0)
        assert _effective_positions({"a": 1.0}) == pytest.approx(1.0)
        assert _effective_positions({"a": 0.5, "b": 0.5, "c": 0.0}) == pytest.approx(2.0)

    def test_the_inverse_pair_overlap_measures_size_not_presence(self) -> None:
        from argus.eval.allocation_comparison import INVERSE_PAIR, _inverse_pair_overlap

        long_leg, short_leg = INVERSE_PAIR
        assert _inverse_pair_overlap({long_leg: 0.7, short_leg: 0.2}) == pytest.approx(0.2)
        assert _inverse_pair_overlap({long_leg: 0.7}) == pytest.approx(0.0)
        assert _inverse_pair_overlap({}) == pytest.approx(0.0)

    def test_the_binomial_p_value_is_exact(self) -> None:
        from argus.eval.allocation_comparison import _two_sided_binomial_p

        # All nine one way is 2 * 0.5**9 under the two-sided convention used.
        assert _two_sided_binomial_p(9, 9) == pytest.approx(2 * 0.5 ** 9)
        assert _two_sided_binomial_p(0, 9) == pytest.approx(2 * 0.5 ** 9)
        # A dead heat cannot be evidence of anything.
        assert _two_sided_binomial_p(5, 10) == pytest.approx(1.0)
        # Every window a tie leaves no trials, which must read as "no test", not as "p=1".
        assert _two_sided_binomial_p(0, 0) is None

    def test_the_implied_risk_aversion_equates_the_two_benefit_models(self) -> None:
        import math

        from argus.desk.allocation import TRADING_HOURS_PER_YEAR
        from argus.eval.allocation_comparison import _implied_risk_aversion

        before, after, sharpe = 0.002, 0.0015, 1.0
        gamma = _implied_risk_aversion(before, after, sharpe)
        per_bar = sharpe / math.sqrt(TRADING_HOURS_PER_YEAR)
        argus_gain = per_bar * before * (before / after - 1.0)
        convex_gain = gamma * (before ** 2 - after ** 2)
        assert convex_gain == pytest.approx(argus_gain, rel=1e-12)

    def test_no_improvement_means_no_finite_risk_aversion(self) -> None:
        from argus.eval.allocation_comparison import _implied_risk_aversion

        assert _implied_risk_aversion(0.002, 0.002, 1.0) == float("inf")

    def test_hrp_on_the_live_covariance_is_a_valid_allocation(
        self, columns: dict[str, list[float]]
    ) -> None:
        built = covariance_matrix(columns)
        assert built is not None
        weights = hrp_weights(built[0], built[1])
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
        assert min(weights.values()) > 0
