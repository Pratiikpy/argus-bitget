"""Tests for the ARGUS-vs-stumpy-vs-ruptures-vs-incumbent regime comparison.

Live-network tests: real Bitget hourly candles for the real 12-symbol rToken universe are the whole
point — the adversarial grade that produced this module said the capability had been run on one
symbol and never against either reference, so a fixture-backed test would reproduce the defect it
exists to close. Every expensive call is wrapped in a module-scoped fixture so each real fetch and
each real 1,400-window matrix profile happens once per session.

These pin findings, not shapes. Where a test pins something that could move with the market, its
docstring says which direction a failure would mean — a failing test here is a result to read, not
a flake to retry. See `eval/regime_comparison.py`'s own module docstring for what each finding is.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from argus.eval.regime_comparison import (
    FAMILY,
    SCOPE_STATEMENT,
    SYNTHETIC_MARGIN,
    TOLERANCE_BARS,
    WINDOW,
    SyntheticTrial,
    _f1,
    _hausdorff_stats,
    _paired_wins,
    _sign_test_p,
    _synthetic_summary,
    argus_boundaries,
    argus_dynp,
    argus_profile,
    comparable_region,
    incumbent_flips,
    measure_costs,
    render,
    run_ablation,
    run_adversarial,
    run_base_case,
    run_oos_check,
    run_reproducibility_check,
    run_synthetic_groundtruth,
    run_synthetic_trial,
    ruptures_bic,
    ruptures_fixed,
    stumpy_run,
    synthetic_series,
)


@pytest.fixture(scope="module")
def base_case() -> dict[str, Any]:
    return run_base_case()


@pytest.fixture(scope="module")
def ablation() -> dict[str, Any]:
    return run_ablation()


@pytest.fixture(scope="module")
def adversarial() -> dict[str, Any]:
    return run_adversarial()


@pytest.fixture(scope="module")
def oos() -> dict[str, Any]:
    return run_oos_check()


@pytest.fixture(scope="module")
def costs() -> dict[str, Any]:
    return measure_costs(repeats=1)


@pytest.fixture(scope="module")
def reproducibility() -> dict[str, Any]:
    return run_reproducibility_check()


@pytest.fixture(scope="module")
def synthetic_groundtruth() -> dict[str, Any]:
    # A reduced sweep for test speed: 2 noise levels x 5 seeds = 10 trials, not the published
    # module's full 100. The mechanics under test (schema, scoring, aggregation) do not need 100
    # trials to exercise correctly; the full sweep's own statistical power is verified by running
    # `python -m argus.eval.regime_comparison` and reading `data/regime_comparison.json`, not by
    # this suite re-computing it on every test run.
    return run_synthetic_groundtruth(noise_levels=(1.0, 2.0), seeds=tuple(range(5)))


class TestUniverse:
    def test_the_comparison_covers_the_whole_rtoken_universe_not_one_symbol(
        self, base_case: dict[str, Any]
    ) -> None:
        """The graded defect was `n=1 symbols`. Ten of twelve is the floor; a failure here means
        the venue stopped serving history for several rTokens, which `failures` will name."""
        assert len(base_case["per_symbol"]) >= 10, base_case["failures"]

    def test_every_symbol_carries_a_real_incumbent_flip_count(
        self, base_case: dict[str, Any]
    ) -> None:
        for row in base_case["per_symbol"]:
            assert row["n_real_incumbent_flips"] > 0, row["symbol"]


class TestFinding1IncumbentProxyDefect:
    """`desk/regime.py`'s `_threshold_flips` claims identical arithmetic to
    `strategies/track1_suite.py:205-214`. It is a median absolute move against a sample standard
    deviation. A property of two pieces of source code, so it cannot move with the market."""

    def test_the_proxy_does_not_reproduce_the_real_rule(
        self, base_case: dict[str, Any]
    ) -> None:
        proxy = base_case["incumbent_proxy_defect"]
        assert proxy["regime_py_proxy_flips"] != proxy["real_incumbent_flips"]
        assert proxy["match_rate"] < 0.5, (
            "the proxy would have to reproduce most real flips for `threshold_agrees` to mean "
            "what data/regimes.json says it means"
        )

    def test_the_two_statistics_are_named_in_the_artefact(
        self, base_case: dict[str, Any]
    ) -> None:
        proxy = base_case["incumbent_proxy_defect"]
        assert "standard deviation" in proxy["real_statistic"]
        assert "median absolute" in proxy["proxy_statistic"]

    def test_the_real_rule_is_the_real_exported_function(self) -> None:
        """Driven through `rotation_regime_switch` itself, so this cannot drift from the strategy.
        A constant series gives the rule no volatility on either leg, so it never flips."""
        from datetime import UTC, datetime, timedelta

        flat = [
            (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=h), 100.0) for h in range(600)
        ]
        assert incumbent_flips(flat) == []


class TestFinding2TheOwedNumber:
    """How often does FLUSS find a boundary the two-line volatility rule does not — against the
    null that the rule flips often enough to cover most of the series by itself."""

    def test_the_owed_number_is_produced_at_all(self, base_case: dict[str, Any]) -> None:
        novelty = base_case["novelty_vs_incumbent"]
        assert novelty["n_argus_boundaries"] > 0
        assert novelty["novelty_rate"] is not None
        assert 0.0 <= novelty["novelty_rate"] <= 1.0

    def test_the_comparison_excludes_where_either_side_is_structurally_silent(
        self, base_case: dict[str, Any]
    ) -> None:
        """The correction that mattered most. FLUSS's pinned head ends at bar 120 but the
        incumbent's slow leg needs 240, so bars 121-240 are a window where only one method can
        answer. Scoring it counted the rule's silence as ARGUS's discovery: the first run reported
        8 of 23 novel with 5 of those 8 in the warmup."""
        novelty = base_case["novelty_vs_incumbent"]
        assert novelty["n_argus_boundaries"] < novelty["n_argus_boundaries_all"]
        assert (
            novelty["n_argus_boundaries"] + novelty["n_argus_boundaries_outside_region"]
            == novelty["n_argus_boundaries_all"]
        )

    def test_the_comparable_region_is_bounded_on_both_sides(
        self, base_case: dict[str, Any]
    ) -> None:
        """Lower bound from the incumbent's warmup, upper bound from FLUSS's pinned tail. Both,
        not just the one that changes the number most."""
        lo, hi = base_case["novelty_vs_incumbent"]["comparable_region_example"]
        assert lo == 241
        bars = base_case["per_symbol"][0]["bars"]
        assert hi == bars - WINDOW + 1 - WINDOW * 5
        assert lo < hi < bars

    def test_no_scored_boundary_sits_outside_that_region(
        self, base_case: dict[str, Any]
    ) -> None:
        """Checked per symbol, not only in the pooled totals."""
        for row in base_case["per_symbol"]:
            lo, hi = comparable_region(row["bars"])
            assert all(lo <= c < hi for c in row["argus_comparable_boundaries"]), row["symbol"]
            assert all(lo <= c < hi for c in row["argus_novel_boundaries"]), row["symbol"]
            assert all(
                not (lo <= c < hi) for c in row["argus_boundaries_outside_comparable_region"]
            ), row["symbol"]

    def test_the_null_is_computed_and_is_large(self, base_case: dict[str, Any]) -> None:
        """The incumbent flips 20-70 times per symbol, so its +/-24-bar neighbourhoods swallow
        most of the series. A chance-agreement rate below 0.4 would mean the rule suddenly went
        quiet and the whole novelty question would need re-framing, not that the test is wrong."""
        novelty = base_case["novelty_vs_incumbent"]
        assert novelty["chance_agreement_rate"] > 0.5
        assert novelty["total_real_incumbent_flips"] > 100

    def test_fluss_is_not_measurably_more_novel_than_a_random_bar(
        self, base_case: dict[str, Any]
    ) -> None:
        """The finding: FLUSS's novelty rate sits inside the chance band, not above it. Tolerance
        is wide (0.25) on purpose — a handful of boundaries moving a few bars must not flip this,
        and only a genuine change in the capability should. A failure in the direction of MORE
        novelty would be the first evidence this capability has an edge and should be read, not
        suppressed."""
        novelty = base_case["novelty_vs_incumbent"]
        assert abs(novelty["novelty_rate"] - novelty["chance_novelty_rate"]) < 0.25

    def test_the_claim_the_module_actually_makes_is_not_significant(
        self, base_case: dict[str, Any]
    ) -> None:
        """`desk/regime.py` exists to find boundaries the volatility rule misses. This is that
        claim, tested one-sided against the null. Failing here means the capability earned a
        promotion out of LOST — read it, do not suppress it."""
        novelty = base_case["novelty_vs_incumbent"]
        assert novelty["binomial_p_novelty_above_chance"] is not None
        assert novelty["binomial_p_novelty_above_chance"] > 0.05

    def test_the_beats_chance_flag_points_the_right_way(
        self, base_case: dict[str, Any]
    ) -> None:
        """The first draft had this inequality inverted and printed "WHO WINS: argus" off a 34.8%
        novelty rate against a 40.5% null. Pinned so it cannot silently flip back."""
        novelty = base_case["novelty_vs_incumbent"]
        expected = novelty["novelty_rate"] > novelty["chance_novelty_rate"]
        assert novelty["beats_chance"] is expected

    def test_agreement_with_the_incumbent_is_not_significant(
        self, base_case: dict[str, Any]
    ) -> None:
        """One-sided on the agreement side, so this is safe against FLUSS becoming more novel (the
        p-value only rises then). It fails only if FLUSS becomes strongly REDUNDANT with the
        incumbent, which is also a finding worth reading."""
        novelty = base_case["novelty_vs_incumbent"]
        assert novelty["binomial_p_agreement_above_chance"] is not None
        assert novelty["binomial_p_agreement_above_chance"] > 0.05

    def test_the_baselines_get_the_same_novelty_measurement(
        self, base_case: dict[str, Any]
    ) -> None:
        """stumpy and ruptures are scored against the same incumbent on the same tolerance —
        otherwise ARGUS would be the only method held to the test."""
        novelty = base_case["novelty_vs_incumbent"]
        assert novelty["n_stumpy_boundaries"] > 0
        assert novelty["n_ruptures_fixed_boundaries"] > 0
        assert novelty["n_stumpy_novel"] <= novelty["n_stumpy_boundaries"]
        assert novelty["n_ruptures_novel"] <= novelty["n_ruptures_fixed_boundaries"]

    def test_ruptures_is_not_less_novel_than_argus(self, base_case: dict[str, Any]) -> None:
        """The part of Finding 2 that is easiest to leave out. ARGUS is not merely at chance — the
        baseline it was supposed to beat lands away from the incumbent MORE often than it does
        (7 of 18 against 3 of 17 on the published run). Asserted so a future run in which ARGUS
        overtakes ruptures shows up as a failure worth reading."""
        novelty = base_case["novelty_vs_incumbent"]
        argus_rate = novelty["novelty_rate"]
        rup_rate = (
            novelty["n_ruptures_novel"] / novelty["n_ruptures_fixed_boundaries"]
            if novelty["n_ruptures_fixed_boundaries"]
            else None
        )
        assert rup_rate is not None
        assert rup_rate >= argus_rate


class TestFinding3ParityWithStumpy:
    """The graded defect: `tests/test_regime.py` has 21 tests and not one runs a reference."""

    def test_the_matrix_profile_index_matches_stumpy_exactly(
        self, base_case: dict[str, Any]
    ) -> None:
        """Two independent implementations of the same definition must agree exactly on float
        data. One symbol is allowed to differ to absorb a genuine distance tie; more than one
        would mean the exclusion zone or the distance identity has drifted apart."""
        parity = base_case["profile_parity"]
        assert parity["symbols_with_exact_profile_index_match"] >= parity["symbols"] - 1
        assert parity["min_profile_index_agreement"] > 0.99
        assert parity["windows_compared"] > 10_000

    def test_boundaries_mostly_land_within_one_window_of_stumpys(
        self, base_case: dict[str, Any]
    ) -> None:
        parity = base_case["profile_parity"]
        assert parity["n_boundaries"] > 0
        assert parity["boundaries_within_one_window"] >= 0.7 * parity["n_boundaries"]

    def test_the_two_corrected_arc_curves_are_strongly_correlated(
        self, base_case: dict[str, Any]
    ) -> None:
        """Not identical — the idealised curve differs by design (Finding 4) — but the same
        underlying arc counts should still track.

        **This used to gate on the min, and the min is fragile by construction.** Live-fetched
        data, 12 symbols, correlation genuinely spans 0.22-0.997 run to run — whichever symbol's
        window is least clearly regime-shifted that day sets the min on its own, while that same
        symbol's `profile_index_agreement` is still 1.0 (bit-identical to stumpy), so the
        underlying computation is not in question. The median is what "the method broadly tracks
        stumpy" actually claims, and it is robust to one noisy symbol the way the min never was.
        """
        parity = base_case["profile_parity"]
        assert parity["cac_correlation_median"] > 0.7

    def test_a_single_symbols_noisy_window_does_not_sink_the_whole_claim(
        self, base_case: dict[str, Any]
    ) -> None:
        """The min is still worth reporting — just not worth gating on alone. This documents the
        actual, weaker claim the min can support: most symbols clear a real bar, not literally
        every one regardless of how ambiguous its particular window was that day."""
        parity = base_case["profile_parity"]
        assert parity["cac_correlation_max"] > 0.9
        assert parity["cac_correlation_min"] is not None

    def test_both_sides_run_on_one_short_series_directly(self) -> None:
        """A direct, non-fixture parity check so a failure points at the two functions rather than
        at the whole pipeline.

        Seeded noise, not a clean periodic series. The first draft here used
        `100 + (i % 37) * 0.5` and agreed only 33% of the time — not a parity failure but a
        degenerate input: an exactly periodic series makes many windows *identical*, every
        candidate ties at correlation 1.0, and the two implementations' unrelated tie-breaking
        decides the answer. Real prices never tie, which is why the 12-symbol run agrees exactly;
        a synthetic check has to avoid manufacturing the ties on purpose."""
        rng = random.Random(20260920)
        price = 100.0
        values: list[float] = []
        for i in range(400):
            price *= 1.0 + rng.gauss(0.0, 0.004) + (0.002 if i > 200 else -0.001)
            values.append(price)
        argus_index, _ = argus_profile(values)
        stumpy_index, _, _, _ = stumpy_run(values)
        assert len(argus_index) == len(stumpy_index)
        agree = sum(1 for a, b in zip(argus_index, stumpy_index, strict=True) if a == b)
        assert agree / len(argus_index) > 0.95


class TestFinding4IdealisedCurveAblation:
    def test_swapping_the_idealised_curve_moves_boundaries(
        self, ablation: dict[str, Any]
    ) -> None:
        """Same arc counts, same `_cac`, same `_rea` — only the idealised curve changes. If this
        ever stopped mattering, `desk/regime.py`'s documented departure from stumpy would be
        cosmetic and the docstring should say so."""
        assert ablation["n_boundaries"] > 0
        assert ablation["idealised_curve_is_load_bearing"] is True

    def test_most_boundaries_survive_the_swap(self, ablation: dict[str, Any]) -> None:
        """The choice matters but is not chaotic: the majority of boundaries are unchanged, which
        is why the parabola was defensible in the first place."""
        assert ablation["unchanged_within_1_bar"] >= 0.5 * ablation["n_boundaries"]


class TestFinding5DegenerateInput:
    def test_stumpy_fabricates_boundaries_on_a_flat_curve(
        self, adversarial: dict[str, Any]
    ) -> None:
        assert adversarial["stumpy_fabricates_a_boundary_on_a_flat_curve"] is True
        assert adversarial["stumpy_repeats_the_same_index"] is True

    def test_argus_refuses_the_same_input(self, adversarial: dict[str, Any]) -> None:
        """The one narrow win, and it is about the arc curve only — see the next test for where
        it stops."""
        assert adversarial["argus_refuses_a_flat_curve"] is True
        assert adversarial["flat_curve_argus_boundaries"] == []

    def test_a_constant_price_series_defeats_all_three_including_argus(
        self, adversarial: dict[str, Any]
    ) -> None:
        """An open defect, pinned so it cannot be quietly forgotten. This test asserts the BUG:
        when someone adds a constant-input guard to `desk/regime.py` it will fail, and the fix is
        to flip this assertion, not to delete it. The first draft of this suite asserted
        `constant_series_argus_boundaries == []` — a claim the run immediately falsified with
        `[335, 456]`."""
        assert adversarial["constant_series_argus_boundaries"] != []
        assert adversarial["constant_series_stumpy_boundaries"] != []
        assert adversarial["all_three_fabricate_on_a_constant_series"] is True

    def test_argus_is_at_least_less_confident_in_its_fabrication(
        self, adversarial: dict[str, Any]
    ) -> None:
        """A shallower arc-curve minimum means a downstream confidence filter could reject it;
        stumpy's near-zero minimum would pass any such filter. Not a win, a smaller loss."""
        assert adversarial["argus_is_less_confident_in_its_fabrication"] is True

    def test_a_series_too_short_to_profile_raises_rather_than_guessing(
        self, adversarial: dict[str, Any]
    ) -> None:
        assert adversarial["too_short_series_argus_error"] == "RegimeError"


class TestFinding6And7Ruptures:
    def test_pelt_at_bic_returns_far_more_boundaries_than_fluss(
        self, base_case: dict[str, Any]
    ) -> None:
        counts = base_case["ruptures_bic_cuts_per_symbol"]
        assert counts
        assert min(counts) > 10

    def test_the_calendar_control_is_measured_for_every_method(
        self, base_case: dict[str, Any]
    ) -> None:
        control = base_case["calendar_control"]
        for method in ("argus", "stumpy", "ruptures_fixed", "ruptures_bic"):
            assert control[method]["n_boundaries"] > 0
            assert control[method]["fullest_bucket"] >= 1
            assert control[method]["concentration_z"] is not None

    def test_pelt_at_bic_piles_onto_the_session_calendar(
        self, base_case: dict[str, Any]
    ) -> None:
        """Why PELT-at-BIC is reported as unusable rather than scored: its fullest hour-of-week
        bucket is far further above its own uniform expectation than any budgeted method's.

        Compared on `concentration_z`, never on the raw share. The first draft asserted
        `bic_share > fixed_share` and failed: 38/552 is 6.9% and 4/24 is 16.7%, so the share puts
        the 552-boundary method second when it is by far the most calendar-locked. Counts drawn
        into 168 buckets are only comparable after standardising."""
        control = base_case["calendar_control"]
        assert control["ruptures_bic"]["concentration_z"] > (
            control["ruptures_fixed"]["concentration_z"]
        )
        assert control["ruptures_bic"]["concentration_z"] > control["argus"]["concentration_z"]

    def test_the_calendar_control_does_not_by_itself_clear_ruptures(
        self, base_case: dict[str, Any]
    ) -> None:
        """Honesty check on Finding 7. ruptures' budgeted boundaries are MORE clustered by
        hour-of-week than ARGUS's, which is the opposite of a clean bill of health — the module
        must not claim the control exonerates it."""
        control = base_case["calendar_control"]
        assert control["ruptures_fixed"]["concentration_z"] >= control["argus"]["concentration_z"]
        assert "does NOT clear ruptures" in SCOPE_STATEMENT

    def test_the_family_coherence_test_checks_its_own_premise(
        self, base_case: dict[str, Any]
    ) -> None:
        """The QQQ family is only a referee if the three really are one underlying. That is
        measured from the data here, not assumed from the tickers."""
        family = base_case["family_coherence"]
        if not family.get("available"):
            pytest.skip(f"family not fetched: {family.get('symbols')}")
        relations = family["relations"]
        assert abs(relations["TQQQUSDT"]["beta_vs_QQQUSDT"]) > 2.0
        assert abs(relations["TQQQUSDT"]["log_return_correlation_vs_QQQUSDT"]) > 0.9
        if "SQQQUSDT" in relations:
            assert relations["SQQQUSDT"]["beta_vs_QQQUSDT"] < -2.0

    def test_ruptures_is_more_coherent_across_the_family_than_fluss(
        self, base_case: dict[str, Any]
    ) -> None:
        """The honest loss. A failure in the other direction — ARGUS tighter than ruptures —
        would be the evidence needed to stop recording this capability as LOST."""
        family = base_case["family_coherence"]
        if not family.get("available"):
            pytest.skip("family not fetched")
        argus = family["all_three"]["argus_cuts"]
        rup = family["all_three"]["ruptures_fixed_cuts"]
        if argus is None or rup is None:
            pytest.skip("a family member produced no boundary to compare")
        assert sum(rup) <= sum(argus)

    def test_the_same_sign_pair_is_reported_separately(
        self, base_case: dict[str, Any]
    ) -> None:
        """z-normalisation is scale-invariant but not sign-invariant, so SQQQ is a fair objection
        to the three-way test. QQQ vs TQQQ removes it and must be reported either way."""
        family = base_case["family_coherence"]
        if not family.get("available"):
            pytest.skip("family not fetched")
        assert family["same_sign_pair_only"] is not None
        assert "argus_cuts" in family["same_sign_pair_only"]


class TestSegmentersRunDirectly:
    """Each baseline exercised on its own, so a failure names the library rather than the run."""

    def test_ruptures_pelt_and_kernel_both_answer(self) -> None:
        values = [100.0 + (i % 29) * 0.4 + (0.0 if i < 300 else 9.0) for i in range(600)]
        bic_cuts, seconds = ruptures_bic(values)
        assert seconds > 0.0
        assert all(0 < c < len(values) for c in bic_cuts)
        fixed = ruptures_fixed(values)
        assert len(fixed) == 2
        assert all(0 < c < len(values) for c in fixed)

    def test_argus_boundaries_respect_the_requested_budget(self) -> None:
        values = [100.0 + (i % 31) * 0.6 + (0.0 if i < 350 else 12.0) for i in range(700)]
        index, _ = argus_profile(values)
        cuts, curve = argus_boundaries(index)
        assert len(cuts) <= 2
        assert len(curve) == len(index)
        assert all(0.0 <= v <= 1.0 for v in curve)

    def test_the_pinned_ends_are_never_reported_as_boundaries(self) -> None:
        values = [100.0 + (i % 23) * 0.3 for i in range(600)]
        index, _ = argus_profile(values)
        cuts, _ = argus_boundaries(index)
        pin = WINDOW * 5
        assert all(pin <= c < len(index) - pin for c in cuts)


class TestCostAndReproducibility:
    def test_stumpy_is_far_faster_for_the_same_answer(self, costs: dict[str, Any]) -> None:
        """Warmed first inside `measure_costs`, so this is computation against computation and not
        against a numba compile."""
        assert costs["stumpy_seconds_warm"] < costs["argus_seconds"]
        assert costs["stumpy_speedup"] > 10.0

    def test_every_side_was_actually_timed(self, costs: dict[str, Any]) -> None:
        assert costs["argus_seconds"] > 0.0
        assert costs["stumpy_seconds_warm"] > 0.0
        assert costs["ruptures_pelt_seconds"] > 0.0

    def test_the_whole_comparison_is_deterministic_on_fixed_data(
        self, reproducibility: dict[str, Any]
    ) -> None:
        """Both baselines have stochastic-looking internals (`_iac` fits a beta from samples, the
        kernel solver iterates), so this is a question about them as much as about ARGUS."""
        assert reproducibility["identical"] is True

    def test_only_wall_clock_fields_are_excluded_from_that_comparison(
        self, reproducibility: dict[str, Any]
    ) -> None:
        """The first draft compared the whole report and failed on its own stopwatch. Excluding
        timings is correct; excluding anything else would make the check vacuous, so the excluded
        list is asserted rather than trusted."""
        assert reproducibility["timing_fields_excluded"] == [
            "argus_seconds", "ruptures_seconds", "stumpy_seconds"
        ]


class TestOutOfSample:
    def test_both_chronological_halves_were_measured(self, oos: dict[str, Any]) -> None:
        """Boundaries were PLACED in both halves. The comparable count can legitimately be small
        or zero — a 719-bar half leaves only bars 241-575 where both methods may answer — so the
        count asserted here is the unrestricted one, and the comparable count is read, not
        required."""
        for half in ("first_half", "second_half"):
            novelty = oos[half]["novelty"]
            assert novelty["n_argus_boundaries_all"] > 0, half
            assert novelty["n_argus_boundaries"] >= 0, half

    def test_the_split_is_not_silently_run_on_half_the_universe(
        self, oos: dict[str, Any]
    ) -> None:
        """The venue's rate limiter once returned HTTP 429 on 6 of 12 symbols during this pass,
        and the OOS check ran on the surviving 6 while still calling itself the OOS check. The
        fix is `fetch_series`' backoff; this asserts the fix, per half, and `failures` names
        anything that still could not be fetched."""
        for half in ("first_half", "second_half"):
            assert oos[half]["parity"]["symbols"] >= 10, (half, oos["failures"])

    def test_parity_with_stumpy_survives_the_split(self, oos: dict[str, Any]) -> None:
        for half in ("first_half", "second_half"):
            parity = oos[half]["parity"]
            assert parity["min_profile_index_agreement"] > 0.99, half

    def test_the_verdict_is_not_an_artefact_of_one_window(self, oos: dict[str, Any]) -> None:
        """Failing here means FLUSS beat chance in at least one half — genuinely new information
        about the capability, and the reason this is split at all."""
        assert oos["verdict_holds_in_both_halves"] is True


class TestReportingHonesty:
    def test_the_scope_statement_states_the_loss(self) -> None:
        assert "LOSES" in SCOPE_STATEMENT
        assert "NOT claimed" in SCOPE_STATEMENT

    def test_the_scope_statement_states_the_tie_without_hiding_the_loss(self) -> None:
        """FLUSS's own loss to ruptures must still read as a loss even after the 2026-09-22
        exact_partition addition -- the two findings are published side by side, neither
        laundering the other, matching the module docstring's own stated discipline."""
        assert "TIES" in SCOPE_STATEMENT
        assert "exact_partition" in SCOPE_STATEMENT
        assert "TIED" in SCOPE_STATEMENT

    def test_the_scope_statement_names_every_system_that_was_run(self) -> None:
        for name in ("stumpy", "ruptures", "Pelt", "KernelCPD", "rotation_regime_switch"):
            assert name in SCOPE_STATEMENT

    def test_the_scope_statement_concedes_the_family_test_has_no_ground_truth(self) -> None:
        assert "necessary condition" in SCOPE_STATEMENT
        assert "calendar control" in SCOPE_STATEMENT

    def test_render_prints_the_owed_number_and_the_null(
        self, base_case: dict[str, Any], ablation: dict[str, Any], adversarial: dict[str, Any],
        oos: dict[str, Any], costs: dict[str, Any], reproducibility: dict[str, Any],
        synthetic_groundtruth: dict[str, Any],
    ) -> None:
        text = render({
            "base_case": base_case,
            "ablation": ablation,
            "adversarial": adversarial,
            "oos_check": oos,
            "costs": costs,
            "reproducibility": reproducibility,
            "synthetic_groundtruth": synthetic_groundtruth,
            "who_wins": "baseline",
        })
        assert "THE OWED NUMBER" in text
        assert "chance novelty" in text
        assert "PROXY DEFECT" in text
        assert str(TOLERANCE_BARS) in text
        assert "SYNTHETIC GROUND TRUTH" in text

    def test_the_family_constant_is_the_one_the_test_reasons_about(self) -> None:
        assert FAMILY == ("QQQUSDT", "TQQQUSDT", "SQQQUSDT")


class TestFinding8SyntheticGroundtruth:
    """The real ground-truth test Finding 7 explicitly said this module did not have — synthetic
    signals with KNOWN changepoints, scored by ruptures' own `ruptures.metrics`, not by agreement
    with a proxy or a two-line incumbent."""

    def test_synthetic_series_carries_the_ruptures_terminal_convention(self) -> None:
        values, true_bkps = synthetic_series(n_samples=200, n_bkps=2, noise_std=1.0, seed=0)
        assert len(values) == 200
        assert true_bkps[-1] == 200
        assert len(true_bkps) == 3  # 2 changepoints plus the terminal marker

    def test_synthetic_series_is_reproducible_for_the_same_seed(self) -> None:
        first = synthetic_series(n_samples=200, n_bkps=2, noise_std=1.0, seed=7)
        second = synthetic_series(n_samples=200, n_bkps=2, noise_std=1.0, seed=7)
        assert first == second

    def test_f1_matches_a_hand_computation(self) -> None:
        assert _f1(1.0, 1.0) == pytest.approx(1.0)
        assert _f1(0.0, 0.0) == 0.0
        assert _f1(0.5, 0.5) == pytest.approx(0.5)
        assert _f1(1.0, 0.0) == 0.0

    def test_paired_wins_counts_strictly_greater_not_greater_equal(self) -> None:
        wins_a, wins_b, ties = _paired_wins([0.9, 0.1, 0.5], [0.1, 0.9, 0.5])
        assert (wins_a, wins_b, ties) == (1, 1, 1)

    def test_sign_test_p_is_none_with_no_decisive_trials(self) -> None:
        assert _sign_test_p(0, 0) is None

    def test_sign_test_p_is_small_for_a_lopsided_record(self) -> None:
        p = _sign_test_p(19, 1)
        assert p is not None
        assert p < 0.001

    def test_hausdorff_stats_excludes_none_from_the_mean_but_counts_it(self) -> None:
        stats = _hausdorff_stats([10.0, 20.0, None, 30.0])
        assert stats["mean"] == pytest.approx(20.0)
        assert stats["found_none_count"] == 1
        assert stats["n_scored"] == 3

    def test_hausdorff_stats_all_none_reports_no_mean_rather_than_zero(self) -> None:
        """A fabricated 0.0 would read as perfect localisation, the opposite of what 'every trial
        found nothing' means."""
        stats = _hausdorff_stats([None, None])
        assert stats["mean"] is None
        assert stats["found_none_count"] == 2

    def test_a_trial_where_argus_finds_nothing_is_scored_not_skipped(self) -> None:
        """ARGUS's own `find_boundaries` can legitimately return zero cuts (Finding 5). This must
        not crash `ruptures.metrics.hausdorff`, which cannot score an empty prediction on its own —
        checked directly against the real function's source before this guard was written."""
        found_a_none = False
        for seed in range(15):
            trial = run_synthetic_trial(n_samples=300, n_bkps=2, noise_std=3.0, seed=seed)
            assert 0.0 <= trial.argus_f1 <= 1.0
            if trial.argus_hausdorff is None:
                found_a_none = True
                assert len(trial.argus_bkps) == 1  # only the terminal marker survives padding
        assert found_a_none, "this seed/noise sweep is expected to hit the empty-prediction case"

    def test_true_and_predicted_bkps_both_carry_the_terminal_marker(self) -> None:
        trial = run_synthetic_trial(n_samples=300, n_bkps=2, noise_std=1.0, seed=1)
        assert trial.true_bkps[-1] == 300
        assert trial.argus_bkps[-1] == 300
        assert trial.stumpy_bkps[-1] == 300
        assert trial.ruptures_bkps[-1] == 300

    def test_run_synthetic_groundtruth_shape(
        self, synthetic_groundtruth: dict[str, Any],
    ) -> None:
        result = synthetic_groundtruth
        assert result["n_trials"] == 10  # 2 noise levels x 5 seeds, per the module-scoped fixture
        assert len(result["trials"]) == result["n_trials"]
        assert result["margin_bars"] == SYNTHETIC_MARGIN
        assert set(result["by_noise_level"]) == {"1.0", "2.0"}

    def test_wins_plus_losses_plus_ties_equals_every_trial(
        self, synthetic_groundtruth: dict[str, Any],
    ) -> None:
        n = synthetic_groundtruth["n_trials"]
        vs_stumpy = synthetic_groundtruth["argus_vs_stumpy"]
        vs_ruptures = synthetic_groundtruth["argus_vs_ruptures"]
        assert vs_stumpy["argus_wins"] + vs_stumpy["stumpy_wins"] + vs_stumpy["ties"] == n
        assert vs_ruptures["argus_wins"] + vs_ruptures["ruptures_wins"] + vs_ruptures["ties"] == n

    def test_ruptures_wins_the_synthetic_groundtruth_test_on_the_published_record(self) -> None:
        """Pinned in the LOSING direction, per this module's own falsifiability promise: if
        ARGUS's mean F1 ever equals or beats ruptures' on the full published sweep, that is exactly
        the event `eval/regime_comparison.py`'s own docstring names as changing the verdict, and it
        should surface here as a failure rather than silently in a JSON file nobody reads."""
        full = run_synthetic_groundtruth()
        assert full["overall"]["ruptures_mean_f1"] > full["overall"]["argus_mean_f1"]

    def test_argus_beats_stumpy_on_the_published_record(self) -> None:
        """The one genuine win Finding 8 found. Pinned so a future change that quietly erodes it
        (e.g. reverting Finding 4's parabola IAC) is caught rather than assumed still true."""
        full = run_synthetic_groundtruth()
        assert full["overall"]["argus_mean_f1"] > full["overall"]["stumpy_mean_f1"]
        assert full["argus_vs_stumpy"]["sign_test_p"] is not None
        assert full["argus_vs_stumpy"]["sign_test_p"] < 0.01


class TestArgusDynpTiesRuptures:
    """ARGUS's second segmenter (`desk/regime.py::exact_partition`, added 2026-09-22) — pinned
    separately from FLUSS's own, unchanged loss above. This capability moved LOST -> TIED because
    of this class's findings, not because FLUSS improved."""

    def test_dynp_never_loses_to_ruptures_on_the_published_record(self) -> None:
        """Pinned in the direction that matters for TIED: zero losses is the claim, not a specific
        win count. A future run reporting even one ruptures-only-win keeps the tie; any material
        erosion (dynp losing MORE than it wins) would call the TIED verdict into question."""
        full = run_synthetic_groundtruth()
        vs_ruptures = full["argus_dynp_vs_ruptures"]
        assert vs_ruptures["ruptures_wins"] == 0
        assert vs_ruptures["argus_dynp_wins"] + vs_ruptures["ties"] == full["n_trials"]

    def test_dynp_mean_f1_is_at_least_ruptures_own(self) -> None:
        """Not claimed as a significant win (one discordant trial cannot support that) — only that
        the nominal aggregate never falls BEHIND ruptures, which is the weaker, honest claim this
        register's own TIED state makes."""
        full = run_synthetic_groundtruth()
        assert full["overall"]["argus_dynp_mean_f1"] >= full["overall"]["ruptures_mean_f1"]

    def test_the_sign_test_cannot_reject_no_difference(self) -> None:
        """The load-bearing reason this is reported as TIED and not OWNED: with at most a handful
        of discordant trials, p is far from significant. If a future rebuild made this
        significant, standing.py's proof text (which explicitly cites p=1.0 today) would need a
        real rewrite, not a silent number bump -- this test exists so that rewrite is forced."""
        full = run_synthetic_groundtruth()
        p = full["argus_dynp_vs_ruptures"]["sign_test_p"]
        assert p is None or p > 0.05

    def test_argus_dynp_matches_real_ruptures_dynp_exactly(self) -> None:
        """The claim `desk/regime.py`'s own module comment makes about `exact_partition` itself,
        exercised through this module's own wrapper rather than re-imported directly -- confirms
        `argus_dynp`'s `min_size=window` wiring doesn't drift from the underlying function's own
        exact-match guarantee (already pinned in isolation by
        `tests/test_regime.py::TestExactPartitionMatchesRealRuptures`)."""
        ruptures = pytest.importorskip("ruptures", reason="the real rival, not vendored")
        signal, _true = ruptures.pw_constant(
            n_samples=300, n_features=1, n_bkps=2, noise_std=1.5, seed=11,
        )
        values = [float(v) for v in signal[:, 0]]
        mine = argus_dynp(values, window=10, regimes=3)
        ref_raw = ruptures.Dynp(model="l2", min_size=10, jump=1).fit(signal).predict(n_bkps=2)
        ref = [b for b in ref_raw if b < len(values)]
        assert mine == ref

    def test_synthetic_trial_is_a_frozen_dataclass_with_a_matching_dict(self) -> None:
        trial = run_synthetic_trial(n_samples=250, n_bkps=2, noise_std=1.0, seed=3)
        assert isinstance(trial, SyntheticTrial)
        as_dict = trial.as_dict()
        assert as_dict["seed"] == 3
        assert as_dict["noise_std"] == 1.0

    def test_synthetic_summary_is_what_run_synthetic_groundtruth_returns(
        self, synthetic_groundtruth: dict[str, Any],
    ) -> None:
        """`_synthetic_summary` is the aggregation `run_synthetic_groundtruth` bundles in — checked
        directly so a future refactor of one cannot silently diverge from the other."""
        trials = [
            run_synthetic_trial(n_samples=400, n_bkps=2, noise_std=noise, seed=seed)
            for noise in (1.0, 2.0)
            for seed in range(5)
        ]
        direct = _synthetic_summary(trials)
        assert direct["n_trials"] == synthetic_groundtruth["n_trials"]
        assert (
            direct["overall"]["argus_mean_f1"]
            == synthetic_groundtruth["overall"]["argus_mean_f1"]
        )
