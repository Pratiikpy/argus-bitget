"""Tests for the tradeable-hedge comparison against Blackout Desk's real `src/blackout/hedges.py`.

Two kinds of test here, deliberately separated.

The pure ones pin the statistics against closed-form answers a reader can check by hand — a Fisher
interval, a sign-flip count, a percentile, the exact round trip through `_price_path` that lets
`risk/effectiveness.measure` see a phase-restricted change sequence without splicing across the
gaps. Those need no network and no baseline.

The live ones fetch the real panel once, in a module-scoped fixture, because the whole claim is
that the pairs measured are the ones Bitget actually lists and quotes. They also run Blackout's
own unmodified code, loaded from the clone — see `eval/baselines/blackout_hedges_loader.py` for why
it is loaded rather than vendored (their repository publishes no licence). A machine without that
clone skips those tests rather than silently comparing against a reimplementation, which would be
the one outcome worse than not running the comparison at all.

Structural assertions throughout, not pinned decimals: the panel is live market data and every
figure moves with it. What must not move is the direction of each finding, and that is what these
pin.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from argus.eval.baselines.blackout_hedges_loader import (
    BlackoutLoadError,
    load_hedges_module,
    provenance,
)
from argus.eval.hedge_tradeable import (
    CANDIDATES,
    EXPOSURES,
    MIN_ROLLING_WINDOWS,
    ROLLING_WINDOW,
    SCOPE_STATEMENT,
    SIGN_AGREEMENT_FLOOR,
    STABILITY_FLOOR,
    UNIVERSE,
    CostModel,
    Panel,
    TradeableError,
    compare_pair,
    contiguous_runs,
    count_sign_flips,
    decide,
    fetch_cost_model,
    fetch_panel,
    hold_hours_for,
    hold_matched_window,
    measure_from_changes,
    ols_ratio,
    pearson,
    percentile,
    phase_changes,
    render,
    rolling_correlations,
    run_blackout_native,
    run_failure_cases,
    run_gap_leak,
    run_oos_split,
    run_pair_table,
    run_placebo,
    run_reproducibility_check,
    run_sign_asymmetry,
    run_stability_ablation,
    run_window_sensitivity,
    sign_agreement,
    straddling_windows,
    summarise,
    variance_reduction,
)
from argus.risk.effectiveness import MIN_PAIRS, measure

# --- pure statistics ------------------------------------------------------------------------------


class TestPearson:
    def test_a_perfect_line_correlates_at_one(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        got = pearson(xs, [2.0 * x + 1 for x in xs])
        assert got is not None
        assert math.isclose(got, 1.0, abs_tol=1e-12)

    def test_a_flat_leg_returns_none_rather_than_nan(self) -> None:
        # The whole reason this helper exists: a NaN correlation compares False against every
        # threshold, which is how the baseline reports "unstable" for a window it never measured.
        assert pearson([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]) is None

    def test_too_few_points_returns_none(self) -> None:
        assert pearson([1.0, 2.0], [3.0, 4.0]) is None


class TestSignFlips:
    def test_a_constant_sign_never_flips(self) -> None:
        assert count_sign_flips([0.9, 0.5, 0.3, 0.8]) == 0
        assert count_sign_flips([-0.9, -0.5, -0.3]) == 0

    def test_each_crossing_counts_once(self) -> None:
        assert count_sign_flips([0.5, -0.4, 0.6]) == 2

    def test_an_exact_zero_is_not_a_crossing(self) -> None:
        # Blackout's np.sign(...).diff() scores two transitions here because sign(0) is 0. A
        # correlation of exactly zero is the absence of evidence, not a change of direction.
        assert count_sign_flips([0.5, 0.0, 0.4]) == 0

    def test_nan_is_skipped_rather_than_counted(self) -> None:
        assert count_sign_flips([0.5, math.nan, 0.4]) == 0


class TestSignAgreement:
    def test_full_agreement_with_a_negative_reference(self) -> None:
        assert sign_agreement([-0.9, -0.5, -0.3], -0.7) == 1.0

    def test_partial_agreement_is_a_share(self) -> None:
        assert sign_agreement([0.9, -0.5, 0.3, 0.8], 0.7) == 0.75

    def test_no_usable_values_reports_nan_not_a_default(self) -> None:
        assert math.isnan(sign_agreement([], 0.5))
        assert math.isnan(sign_agreement([0.5], 0.0))


class TestPercentile:
    def test_the_fifth_percentile_sits_near_the_bottom(self) -> None:
        values = [float(i) for i in range(101)]
        assert math.isclose(percentile(values, 0.05), 5.0)

    def test_it_interpolates_rather_than_snapping(self) -> None:
        assert math.isclose(percentile([0.0, 10.0], 0.5), 5.0)

    def test_an_empty_sample_is_nan_not_an_exception(self) -> None:
        assert math.isnan(percentile([], 0.5))

    def test_a_percentile_is_never_below_the_minimum(self) -> None:
        values = [0.4, 0.9, 0.2, 0.7, 0.55]
        assert percentile(values, 0.05) >= min(values)


class TestRollingCorrelations:
    def test_window_count_is_sample_minus_window_plus_one(self) -> None:
        xs = [float(i % 7) for i in range(100)]
        ys = [float((i * 3) % 11) for i in range(100)]
        assert len(rolling_correlations(xs, ys, window=24)) == 100 - 24 + 1

    def test_a_window_longer_than_the_sample_yields_nothing(self) -> None:
        # The baseline returns NaN here and then reports "unstable" from it; we return no windows
        # and the MIN_ROLLING_WINDOWS floor turns that into "not testable".
        assert rolling_correlations([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], window=24) == []


class TestStraddlingWindows:
    def test_a_contiguous_index_run_straddles_nothing(self) -> None:
        assert straddling_windows(list(range(100)), window=24) == 0

    def test_a_break_is_counted_not_hidden(self) -> None:
        indices = list(range(30)) + list(range(100, 130))
        assert straddling_windows(indices, window=24) > 0


class TestPricePathRoundTrip:
    def test_measure_sees_exactly_the_changes_it_was_given(self) -> None:
        # This is the load-bearing step: `measure` takes prices and differences them itself, so the
        # phase-restricted changes must survive the trip intact or the whole comparison is scoring
        # something other than what it filtered.
        changes = [0.001 * ((i % 13) - 6) for i in range(MIN_PAIRS + 20)]
        hedge = [0.0008 * ((i % 7) - 3) for i in range(MIN_PAIRS + 20)]
        got = measure_from_changes(
            changes, hedge, spot="A", hedge="B", phase="test", window_days=1,
        )
        assert got.observations == len(changes)
        direct = pearson(changes, hedge)
        assert direct is not None
        assert math.isclose(got.correlation, direct, abs_tol=1e-12)

    def test_the_fisher_lower_bound_is_below_the_point_estimate(self) -> None:
        changes = [0.001 * ((i % 13) - 6) for i in range(MIN_PAIRS + 20)]
        hedge = [c * 0.8 + 0.0002 * ((i % 5) - 2) for i, c in enumerate(changes)]
        got = measure_from_changes(
            changes, hedge, spot="A", hedge="B", phase="test", window_days=1,
        )
        assert got.correlation_low < got.correlation


class TestRatioAndVarianceReduction:
    def test_a_unit_relationship_gives_a_unit_ratio_and_full_reduction(self) -> None:
        hedge = [0.01 * ((i % 11) - 5) for i in range(200)]
        spot = list(hedge)
        assert math.isclose(ols_ratio(spot, hedge), 1.0, abs_tol=1e-12)
        assert math.isclose(variance_reduction(spot, hedge, 1.0), 1.0, abs_tol=1e-12)

    def test_the_wrong_ratio_can_add_variance(self) -> None:
        hedge = [0.01 * ((i % 11) - 5) for i in range(200)]
        spot = list(hedge)
        assert variance_reduction(spot, hedge, -1.0) < 0

    def test_an_unmoving_hedge_raises_rather_than_returning_a_ratio(self) -> None:
        with pytest.raises(TradeableError):
            ols_ratio([0.1, -0.1, 0.2], [1.0, 1.0, 1.0])


class TestContiguousRuns:
    def test_runs_are_split_at_every_change(self) -> None:
        runs = contiguous_runs(["a", "a", "b", "b", "b", "a"])
        assert runs == {"a": [2, 1], "b": [3]}

    def test_an_empty_sequence_has_no_runs(self) -> None:
        assert contiguous_runs([]) == {}


class TestCostModel:
    def test_the_round_trip_is_two_taker_legs_plus_absolute_funding(self) -> None:
        model = CostModel(
            taker_fee_rate={"X": 0.0006}, funding_rate={"X": -0.0001},
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC), source="unit test",
        )
        # 48h at an 8h settlement interval is 6 funding periods; funding is charged at |rate|
        # because which side of the hedge we end up on decides whether it is a credit.
        assert math.isclose(model.round_trip("X", 48.0), 0.0006 * 2 + 0.0001 * 6)

    def test_an_unknown_symbol_refuses_rather_than_assuming_a_fee(self) -> None:
        model = CostModel(
            taker_fee_rate={"X": 0.0006}, funding_rate={},
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC), source="unit test",
        )
        with pytest.raises(TradeableError):
            model.round_trip("Y", 48.0)


class TestPanelGuards:
    def test_a_ragged_panel_is_refused_at_construction(self) -> None:
        with pytest.raises(TradeableError):
            Panel(
                timestamps=[datetime(2026, 1, 1, tzinfo=UTC)],
                phases=["rth"],
                closes={"A": [1.0, 2.0]},
                days=1,
            )


def test_the_universe_is_only_instruments_bitget_lists() -> None:
    from argus.market.bitget import RTOKEN_SYMBOLS

    assert set(EXPOSURES) == set(RTOKEN_SYMBOLS)
    assert "BTCUSDT" in CANDIDATES and "ETHUSDT" in CANDIDATES
    assert "SQQQUSDT" in CANDIDATES, "the inverse instrument is the point of the exercise"
    # No anchor-index leg anywhere: that is the pair this module exists to stop measuring.
    assert not any("index" in name.lower() for name in UNIVERSE)


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert "no hedge was actually placed" in SCOPE_STATEMENT
    assert "snapshot" in SCOPE_STATEMENT
    assert "order statistics" in SCOPE_STATEMENT


# --- the real baseline ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline() -> object:
    try:
        return load_hedges_module()
    except BlackoutLoadError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"Blackout Desk clone unavailable: {exc}")


class TestBaselineProvenance:
    def test_the_real_file_is_the_one_that_ran(self, baseline: object) -> None:
        where = provenance()
        assert where["file"].replace("\\", "/").endswith("blackout/hedges.py")
        assert where["licence"].startswith("none published")

    def test_the_gate_we_borrowed_is_the_gate_they_published(self, baseline: object) -> None:
        # Pinned so a clone at a different commit fails loudly here rather than silently changing
        # what this comparison is comparing against.
        assert baseline.MIN_WINDOWS == 12  # type: ignore[attr-defined]
        assert set(baseline.COSTS) == {"BTC-USD", "ETH-USD", "rtoken"}  # type: ignore[attr-defined]


class TestFailureCases:
    @pytest.fixture(scope="class")
    def failures(self, baseline: object) -> dict:
        return run_failure_cases()

    def test_their_risk_reduction_field_is_volatility_not_variance(
        self, failures: dict
    ) -> None:
        row = failures["risk_reduction_is_vol_not_variance"]
        assert row["matches_one_minus_sqrt_one_minus_r_squared"]
        assert row["their_risk_reduction_field"] < row["variance_share_removed_r_squared"]
        assert row["their_verdict_calls_it_variance"]

    def test_their_cost_lookup_falls_through_on_a_bitget_symbol(self, failures: dict) -> None:
        row = failures["cost_lookup_falls_through_on_bitget_symbols"]
        assert "BTCUSDT" not in row["keys_their_costs_dict_has"]
        assert row["btcusdt_charged"] > row["bitget_real_round_trip_taker"]
        assert row["silent"]

    def test_a_window_longer_than_the_sample_still_yields_a_stability_verdict(
        self, failures: dict
    ) -> None:
        row = failures["rolling_window_longer_than_sample"]
        assert row["corr_min_is_nan"]
        assert row["claims_unstable_from_zero_estimates"]

    def test_a_two_row_frame_is_declined_but_still_priced(self, failures: dict) -> None:
        row = failures["n_below_3"]
        assert row["verdict"] == "insufficient history to quote"
        assert row["cost_pct_still_quoted"] > 0


# --- live market data -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel() -> Panel:
    return fetch_panel(days=90)


@pytest.fixture(scope="module")
def costs() -> CostModel:
    return fetch_cost_model()


@pytest.fixture(scope="module")
def rows(panel: Panel, costs: CostModel, baseline: object) -> list:
    return run_pair_table(panel, costs)


class TestPanel:
    def test_every_instrument_is_aligned_on_the_same_timestamps(self, panel: Panel) -> None:
        assert len(panel.timestamps) > MIN_PAIRS * 2
        for series in panel.closes.values():
            assert len(series) == len(panel.timestamps)

    def test_the_weekend_hold_is_measured_not_assumed(self, panel: Panel) -> None:
        hours = hold_hours_for(panel, "weekend")
        assert 24.0 <= hours <= 80.0, "a weekend run should be roughly two days of hourly bars"

    def test_phase_changes_never_splice_across_a_closure(self, panel: Panel) -> None:
        _, indices = phase_changes(panel, "NVDAUSDT", "weekend")
        for i in indices:
            assert panel.phases[i] == "weekend"
            assert panel.phases[i - 1] == "weekend"

    def test_both_legs_of_a_pair_share_the_same_bar_indices(self, panel: Panel) -> None:
        _, spot_idx = phase_changes(panel, "NVDAUSDT", "weekend")
        _, hedge_idx = phase_changes(panel, "BTCUSDT", "weekend")
        assert spot_idx == hedge_idx


class TestCostsAreReal:
    def test_the_taker_fee_comes_from_the_venue_for_every_instrument(
        self, costs: CostModel
    ) -> None:
        for symbol in UNIVERSE:
            assert symbol in costs.taker_fee_rate
            assert 0 < costs.taker_fee_rate[symbol] < 0.01

    def test_the_real_round_trip_is_far_below_the_baselines_assumption(
        self, costs: CostModel, baseline: object
    ) -> None:
        real = costs.round_trip("BTCUSDT", 48.0)
        assumed = baseline.COSTS["rtoken"]  # type: ignore[attr-defined]
        assert real < assumed


class TestPairTable:
    def test_every_tradeable_pair_in_every_measured_phase_was_scored(self, rows: list) -> None:
        assert len(rows) > 100
        for row in rows:
            assert row.observations >= MIN_PAIRS
            assert row.spot != row.hedge

    def test_both_systems_agree_on_the_full_sample_correlation(self, rows: list) -> None:
        # The same-input contract: if these ever diverge, the two systems are not being fed the
        # same rows and nothing else in the comparison means anything.
        for row in rows:
            assert math.isclose(row.correlation, row.baseline_correlation, abs_tol=1e-9)

    def test_their_risk_reduction_is_our_gross_vol_reduction(self, rows: list) -> None:
        for row in rows:
            assert math.isclose(row.gross_vol_reduction, row.baseline_risk_reduction, abs_tol=1e-9)

    def test_the_published_effectiveness_is_net_of_cost(self, rows: list) -> None:
        for row in rows:
            assert row.cost_fraction > 0
            assert row.net_vol_reduction < row.gross_vol_reduction

    def test_a_stable_row_carries_enough_rolling_windows_to_have_asked(self, rows: list) -> None:
        for row in rows:
            if row.stable:
                assert row.rolling_windows >= MIN_ROLLING_WINDOWS
                assert row.sign_flips == 0
                assert row.abs_rolling_corr_min > STABILITY_FLOOR

    def test_a_robust_stable_row_agrees_on_sign_almost_always(self, rows: list) -> None:
        for row in rows:
            if row.robust_stable:
                assert row.sign_agreement >= SIGN_AGREEMENT_FLOOR
                assert row.abs_rolling_corr_p05 > STABILITY_FLOOR

    def test_every_row_serialises_to_json_safe_primitives(self, rows: list) -> None:
        import json

        json.dumps([row.as_dict() for row in rows])


class TestTheAnswer:
    def test_the_weekend_is_the_hard_phase(self, rows: list) -> None:
        summary = summarise(rows)
        weekend = summary["weekend"]
        overnight = summary["overnight"]
        # The finding: a tight full-sample bound is not a hedge. Many weekend pairs clear the
        # Fisher floor; far fewer survive the stability test.
        assert weekend["fisher_low_above_floor"] > weekend["stable"]
        assert overnight["stable"] >= weekend["stable"]

    def test_the_verdict_answers_per_phase_and_names_its_caveat(self, rows: list) -> None:
        verdict = decide(summarise(rows))
        assert "overnight" in verdict and "weekend" in verdict
        assert isinstance(verdict["overnight"]["can_hedge"], bool)
        assert "out of sample" in verdict["caveat"]

    def test_render_reads_as_a_report(self, panel: Panel, rows: list) -> None:
        summary = summarise(rows)
        report = {
            "baseline": provenance(),
            "universe": {
                "bars": len(panel.timestamps), "days": panel.days,
                "exposures": list(EXPOSURES), "candidates": list(CANDIDATES),
            },
            "summary": summary,
            "verdict": decide(summary),
            "blackout_native": {
                "closure_windows_found": 12, "every_candidate_unstable": True,
                "argus_min_pairs": MIN_PAIRS, "argus_refused": True,
            },
            "sign_asymmetry": run_sign_asymmetry(rows),
            "gap_leak": {"max_absolute_r_squared_error": 0.1},
            "out_of_sample": {"degraded_out_of_sample": 1, "pairs": 2, "median_decay": 0.0},
            "window_sensitivity": {
                "phase": "weekend", "strict_gate_range": [0, 1], "robust_gate_range": [0, 1],
                "by_window": [{"rolling_window_bars": ROLLING_WINDOW}],
            },
            "reproducibility": {"identical": True},
        }
        text = render(report)
        assert "TRADEABLE HEDGE EFFECTIVENESS" in text
        assert "ANSWER:" in text


@pytest.fixture(scope="module")
def tautology(baseline: object) -> dict:
    """Fetched once. Three paged series per symbol against a venue that rate-limits, so a
    per-test call would spend the budget the live panel needs."""
    from argus.eval.hedge_tradeable import run_tautology_check

    return run_tautology_check()


class TestTautologyAndBaselineNative:
    def test_the_anchor_index_pair_is_marked_untradeable(self, tautology: dict) -> None:
        got = tautology
        for symbol in ("NVDAUSDT", "TQQQUSDT"):
            assert got[symbol]["tradeable_on_bitget"] is False
            assert got[symbol]["listed_instrument"] is None
        assert "no order can be sent to it" in got["note"]

    def test_the_index_pair_measures_far_higher_than_any_tradeable_one(
        self, tautology: dict, rows: list
    ) -> None:
        rth = tautology["NVDAUSDT"]["by_phase"].get("rth")
        if rth is None:
            pytest.skip("not enough regular-hours basis history in the fetched window")
        best_tradeable = max(
            (r.r_squared for r in rows if r.spot == "NVDAUSDT" and r.phase == "rth"), default=0.0
        )
        assert rth["r_squared"] > best_tradeable

    def test_the_baseline_runs_at_its_own_granularity_where_we_refuse(
        self, panel: Panel, baseline: object
    ) -> None:
        got = run_blackout_native(panel)
        assert got["closure_windows_found"] >= got["blackout_min_windows"]
        assert got["closure_windows_found"] < got["argus_min_pairs"]
        assert got["argus_refused"] is True
        assert len(got["baseline_menu"]) > 0
        # Their COSTS dict does not know a Bitget symbol, so every leg is charged the DEX default.
        assert all(row["cost_pct"] == 0.006 for row in got["baseline_menu"])


class TestAblations:
    def test_the_stability_test_rejects_pairs_a_fisher_bound_alone_would_pass(
        self, rows: list
    ) -> None:
        got = run_stability_ablation(rows)
        weekend = got["weekend"]
        assert weekend["fisher_bound_alone_would_pass"] > 0
        assert weekend["stability_test_rejects"] > 0

    def test_the_baseline_never_credits_an_inverse_hedge(self, rows: list) -> None:
        got = run_sign_asymmetry(rows)
        assert got["inverse_instrument_pairs"] > 0
        assert got["baseline_ever_called_an_inverse_pair_stable"] is False

    def test_the_strict_gate_is_shown_to_depend_on_the_window_length(
        self, panel: Panel, costs: CostModel, baseline: object
    ) -> None:
        got = run_window_sensitivity(panel, costs, windows=(12, 48, 168))
        assert got["hold_matched_window_bars"] == hold_matched_window(panel, "weekend")
        assert got["at_hold_matched_window"] is not None
        counts = [row["strict_stable"] for row in got["by_window"]]
        # Not an assertion that it MUST vary — an assertion that the sweep is real and reported.
        assert len(counts) >= 3
        assert got["strict_gate_range"] == [min(counts), max(counts)]

    def test_the_out_of_sample_ratio_is_scored_on_rows_it_was_not_fitted_on(
        self, panel: Panel, costs: CostModel
    ) -> None:
        got = run_oos_split(panel, costs)
        assert got["pairs"] > 0
        for row in got["worst"]:
            # 1e-5, not 1e-9: all three figures are rounded to six places independently before they
            # reach the report, so the difference of two of them can miss the third by one unit in
            # the last place. Rounding the report is deliberate — this tolerance is the price.
            assert math.isclose(
                float(row["decay"]),
                float(row["variance_reduction_in_sample"])
                - float(row["variance_reduction_out_of_sample"]),
                abs_tol=1e-5,
            )
            assert -1.0 <= float(row["variance_reduction_out_of_sample"]) <= 1.0

    def test_a_shuffled_hedge_is_rejected_by_both_systems(
        self, panel: Panel, costs: CostModel, baseline: object
    ) -> None:
        got = run_placebo(panel, costs)
        assert got["both_systems_reject_every_placebo"]
        for row in got["rows"]:
            assert row["r_squared"] < 0.1


class TestGapLeakInOurOwnModule:
    def test_bucketing_prices_by_phase_changes_the_answer(self, panel: Panel) -> None:
        got = run_gap_leak(panel)
        assert got["max_absolute_r_squared_error"] is not None
        assert got["max_absolute_r_squared_error"] > 0
        for finding in got["findings"]:
            # Every extra "observation" is the splice between two runs of the phase.
            assert finding["spliced_observations"] > 0

    def test_the_spliced_count_matches_the_number_of_runs(self, panel: Panel) -> None:
        runs = contiguous_runs(panel.phases)
        got = run_gap_leak(panel)
        for finding in got["findings"]:
            expected = len(runs[finding["phase"]]) - 1
            assert finding["spliced_observations"] == expected

    def test_the_within_phase_estimate_is_what_this_module_publishes(self, panel: Panel) -> None:
        bucketed = [
            panel.closes["NVDAUSDT"][i]
            for i in range(len(panel.timestamps))
            if panel.phases[i] == "weekend"
        ]
        bucketed_hedge = [
            panel.closes["QQQUSDT"][i]
            for i in range(len(panel.timestamps))
            if panel.phases[i] == "weekend"
        ]
        leaky = measure(bucketed, bucketed_hedge, spot="NVDAUSDT", hedge="QQQUSDT",
                        phase="weekend")
        spot_changes, _ = phase_changes(panel, "NVDAUSDT", "weekend")
        assert leaky.observations > len(spot_changes)


class TestReproducibility:
    def test_the_same_panel_scores_identically_twice(
        self, panel: Panel, costs: CostModel, baseline: object
    ) -> None:
        assert run_reproducibility_check(panel, costs)["identical"]

    def test_a_single_pair_is_deterministic(
        self, panel: Panel, costs: CostModel, baseline: object
    ) -> None:
        fixed = datetime(2026, 1, 1, tzinfo=UTC)
        first = compare_pair(panel, costs, spot="NVDAUSDT", hedge="BTCUSDT", phase="weekend",
                             now=fixed)
        second = compare_pair(panel, costs, spot="NVDAUSDT", hedge="BTCUSDT", phase="weekend",
                              now=fixed)
        assert first.as_dict() == second.as_dict()
