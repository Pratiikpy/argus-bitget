"""Tests for `eval/general_sue_comparison.py` — SUE against pandas, scikit-learn and SciPy.

Two tiers. The offline tests pin each rival's measured behaviour on constructed inputs whose exact
answer is known. The live tests fetch the same real SEC EDGAR data the comparison runs on (the nine
anchors through `market/fundamentals.py`, and the SEC's quarterly XBRL frames) once per session
and pin the headline findings; they fail loudly rather than skip if SEC is unreachable, like
`tests/test_earnings_comparison.py`.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from argus.eval import artefact
from argus.eval.general_sue_comparison import (
    DEGENERACY_METHODS,
    SCOPE_STATEMENT,
    Point,
    Reading,
    Verdict,
    Window,
    classify,
    compute,
    constructed_windows,
    exact_sue,
    fabricated_magnitudes,
    fetch_anchor_panel,
    fetch_frames_panel,
    is_true_yoy,
    pandas_agreement,
    pandas_collisions,
    render,
    run_argus_dated,
    run_argus_positional,
    run_pandas_lenient,
    run_pandas_strict,
    step_windows,
    unit_sweep,
)

NVDA_SHAPED = ["2.46", "1.91", "1.5", "0.89", "0.81", "0.78", "0.61", "0.52", "0.4", "0.35",
               "0.27", "0.18", "0.15", "0.09", "0.11", "0.07"]


def _points(ends: list[date], values: list[str]) -> list[Point]:
    return [Point(end=e, value=Decimal(v)) for e, v in zip(ends, values, strict=False)]


def _quarter_ends(newest: date, count: int, *, skip_month: int | None = None) -> list[date]:
    out: list[date] = []
    year, month = newest.year, newest.month
    while len(out) < count:
        if month != skip_month:
            out.append(date(year, month, 28))
        month -= 3
        if month <= 0:
            month += 12
            year -= 1
    return out


class TestGroundTruth:
    def test_exact_sue_is_none_only_for_exactly_zero_variance(self) -> None:
        assert exact_sue([Decimal("0.1")] * 8) is None
        assert exact_sue([Decimal("0.1")] * 7 + [Decimal("0.1000001")]) is not None

    def test_year_over_year_truth_is_unambiguous(self) -> None:
        assert is_true_yoy(date(2026, 1, 3), date(2024, 12, 28))  # 371 days, a 53-week year
        assert not is_true_yoy(date(2026, 6, 28), date(2025, 3, 28))  # 457 days, five quarters


class TestAlignmentOffline:
    def test_positional_lag_pairs_the_wrong_quarter_when_q4_is_missing(self) -> None:
        points = _points(_quarter_ends(date(2026, 6, 28), 16, skip_month=12), NVDA_SHAPED)
        positional = run_argus_positional(points)
        assert positional.value is not None
        assert not any(is_true_yoy(a, b) for a, b in positional.pairs)
        dated = run_argus_dated(points)
        assert dated.value is not None
        assert all(is_true_yoy(a, b) for a, b in dated.pairs)

    def test_pandas_lenient_agrees_with_argus_dated_on_a_clean_calendar(self) -> None:
        points = _points(_quarter_ends(date(2026, 6, 28), 16, skip_month=12), NVDA_SHAPED)
        assert run_pandas_lenient(points).value == pytest.approx(
            run_argus_dated(points).value, rel=1e-12
        )

    def test_pandas_strict_returns_nan_whenever_a_q4_is_missing(self) -> None:
        points = _points(_quarter_ends(date(2026, 6, 28), 16, skip_month=12), NVDA_SHAPED)
        assert run_pandas_strict(points).value is None

    def test_agreement_is_checked_filer_by_filer_on_identical_pairs(self) -> None:
        points = _points(_quarter_ends(date(2026, 6, 28), 16, skip_month=12), NVDA_SHAPED)
        pandas_reading, argus_reading = run_pandas_lenient(points), run_argus_dated(points)
        report = pandas_agreement({"X": pandas_reading}, {"X": argus_reading})
        assert report["both_finite"] == report["identical_pair_sets"] == 1
        assert report["identical_pair_sets_values_disagree_count"] == 0

    def test_a_value_that_drifts_on_identical_pairs_is_reported_as_disagreement(self) -> None:
        points = _points(_quarter_ends(date(2026, 6, 28), 16, skip_month=12), NVDA_SHAPED)
        argus_reading = run_argus_dated(points)
        assert argus_reading.value is not None
        drifted = Reading(argus_reading.value * (1 + 1e-6), argus_reading.pairs)
        report = pandas_agreement({"X": drifted}, {"X": argus_reading})
        assert report["identical_pair_sets_values_disagree"] == ["X"]

    def test_different_pairs_are_attributed_to_the_side_that_is_not_year_over_year(self) -> None:
        points = _points(_quarter_ends(date(2026, 6, 28), 16, skip_month=12), NVDA_SHAPED)
        argus_reading = run_argus_dated(points)
        wrong = Reading(1.0, run_argus_positional(points).pairs)
        report = pandas_agreement({"X": wrong}, {"X": argus_reading})
        assert report["different_pair_sets"] == 1
        assert report["different_pair_sets_pandas_has_a_non_year_over_year_pair"] == 1
        assert report["different_pair_sets_argus_has_a_non_year_over_year_pair"] == 0

    def test_a_non_finite_pandas_reading_is_counted_with_what_argus_did(self) -> None:
        report = pandas_agreement(
            {"X": Reading(float("inf"), ())}, {"X": Reading(None, (), refusal="degenerate")})
        assert report["both_finite"] == 0
        assert report["pandas_non_finite"] == report["pandas_non_finite_argus_refused"] == 1

    def test_a_52_53_week_calendar_crashes_pandas_but_pairs_by_date(self) -> None:
        """Fiscal quarters ending the Saturday nearest a calendar quarter end: 2023-07-01 and
        2023-09-30 both fall in calendar 2023Q3, so pandas' PeriodIndex holds a duplicate label."""
        ends = [date(2026, 6, 27), date(2026, 3, 28), date(2025, 9, 27), date(2025, 6, 28),
                date(2025, 3, 29), date(2024, 9, 28), date(2024, 6, 29), date(2024, 3, 30),
                date(2023, 9, 30), date(2023, 7, 1), date(2023, 4, 1), date(2022, 10, 1),
                date(2022, 7, 2)]
        points = _points(ends, NVDA_SHAPED)
        assert run_pandas_lenient(points).crashed
        assert pandas_collisions({"X": points}, {"X": "X"})["filers"] == 1
        dated = run_argus_dated(points)
        assert dated.value is not None
        assert all(is_true_yoy(a, b) for a, b in dated.pairs)


class TestDegeneracyOffline:
    def test_the_earnings_comparison_cases_split_the_rivals(self) -> None:
        outcomes = {
            name: [classify(w, m(w)) for w in constructed_windows()]
            for name, m in DEGENERACY_METHODS.items()
        }
        assert outcomes["argus_relative_after"] == ["correct_refusal"] * 3
        assert outcomes["scipy_zscore_guard"] == ["correct_refusal"] * 3
        assert "fabricated_finite" in outcomes["sklearn_variance_threshold"]
        assert outcomes["sklearn_standard_scaler"] == ["fabricated_finite"] * 3
        assert "fabricated_non_finite" in outcomes["quantconnect_reference"]

    def test_a_constant_step_on_a_real_eps_level_beats_both_libraries(self) -> None:
        """$8.80 and $680.23 are the real panel's 90th and 99th percentile EPS magnitudes."""
        degenerate, genuine = step_windows([Decimal("8.80"), Decimal("680.23")])
        for window in degenerate:
            assert classify(window, DEGENERACY_METHODS["argus_relative_after"](window)) == (
                "correct_refusal"
            ), window.key
        scipy = [classify(w, DEGENERACY_METHODS["scipy_zscore_guard"](w)) for w in degenerate]
        sklearn = [classify(w, DEGENERACY_METHODS["sklearn_variance_threshold"](w))
                   for w in degenerate]
        assert "fabricated_finite" in scipy
        assert "fabricated_finite" in sklearn
        for window in genuine:
            for name, method in DEGENERACY_METHODS.items():
                assert classify(window, method(window)) == "correct_value", (name, window.key)

    def test_fabricated_magnitudes_report_the_size_a_ranking_would_have_taken(self) -> None:
        degenerate, _genuine = step_windows([Decimal("8.80")])
        ranges = fabricated_magnitudes(degenerate)
        assert ranges["argus_relative_after"] == {"count": 0, "min_abs": None, "max_abs": None}
        scipy = ranges["scipy_zscore_guard"]
        assert scipy["count"] > 0
        assert scipy["min_abs"] > 1e6  # a real SUE is single digits

    def test_a_fabricated_value_is_never_counted_as_correct(self) -> None:
        window = constructed_windows()[0]
        assert classify(window, Verdict(4.0, refused=False)) == "fabricated_finite"
        assert classify(window, Verdict(float("inf"), refused=False)) == "fabricated_non_finite"
        assert classify(window, Verdict(None, refused=True)) == "correct_refusal"

    def test_the_legacy_absolute_threshold_was_not_unit_invariant(self) -> None:
        anchors = {"NVDA": _points(_quarter_ends(date(2026, 6, 28), 16), NVDA_SHAPED)}
        sweep = unit_sweep(anchors)
        assert sweep["outcomes"]["argus_absolute_before"]["false_refusal"] > 0
        assert sweep["outcomes"]["argus_relative_after"]["false_refusal"] == 0
        assert sweep["outcomes"]["argus_relative_after"]["fabricated_finite"] == 0
        assert not sweep["non_correct_cases"]["argus_relative_after"]

    def test_window_is_exact_where_the_floats_are_not(self) -> None:
        window = constructed_windows()[2]  # decimal_linear
        assert isinstance(window, Window)
        assert window.truth is None
        assert len(set(window.floats)) > 1  # float noise the exact deltas do not have


def test_scope_statement_names_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert "predicts returns" in SCOPE_STATEMENT


# --- live: the same real SEC EDGAR data the comparison runs on ------------------------------------


@pytest.fixture(scope="module")
def live() -> dict:
    anchors = fetch_anchor_panel()
    frames, names = fetch_frames_panel()
    return {"anchors": anchors, "frames": frames, "names": names,
            "report": compute(anchors, frames, names)}


class TestLiveAlignment:
    def test_the_old_positional_lag_was_not_year_over_year_for_any_anchor(self, live: dict) -> None:
        row = live["report"]["alignment"]["anchors"]["argus_positional_before"]
        assert row["readings_with_no_pair_year_over_year"] >= 7

    def test_argus_dated_is_year_over_year_for_every_pair_on_every_anchor(
        self, live: dict
    ) -> None:
        row = live["report"]["alignment"]["anchors"]["argus_dated_after"]
        assert row["readings"] == 9
        assert row["share_of_pairs_year_over_year"] == 1.0

    def test_on_the_whole_sec_panel_dated_pairing_is_exact_and_covers_more_than_pandas(
        self, live: dict
    ) -> None:
        frames = live["report"]["alignment"]["frames"]
        assert frames["argus_positional_before"]["share_of_pairs_year_over_year"] < 0.2
        assert frames["argus_dated_after"]["share_of_pairs_year_over_year"] == 1.0
        assert frames["argus_dated_after"]["crashed"] == 0
        assert frames["pandas_period_lenient"]["crashed"] > 0
        assert frames["argus_dated_after"]["readings"] > frames["pandas_period_lenient"][
            "readings"] - frames["pandas_period_lenient"]["non_finite_readings"]

    def test_on_identical_pairs_argus_and_pandas_agree_on_every_filer(self, live: dict) -> None:
        agreement = live["report"]["alignment"]["argus_dated_vs_pandas_lenient"]
        assert agreement["identical_pair_sets"] > 3000
        assert agreement["identical_pair_sets_values_disagree_count"] == 0
        assert agreement["different_pair_sets_argus_has_a_non_year_over_year_pair"] == 0
        assert agreement["pandas_non_finite_argus_refused"] == agreement["pandas_non_finite"]


class TestLiveDegeneracy:
    def test_real_degenerate_filers_exist_and_argus_refuses_every_one(self, live: dict) -> None:
        deg = live["report"]["degeneracy"]
        assert deg["real_frames_degenerate_windows"] > 0
        argus = deg["real_frames"]["argus_relative_after"]
        assert argus["fabricated_finite"] == argus["fabricated_non_finite"] == 0
        assert argus["false_refusal"] == argus["wrong_value"] == 0
        assert deg["real_frames"]["sklearn_standard_scaler"]["fabricated_finite"] == deg[
            "real_frames_degenerate_windows"]

    def test_constant_step_at_real_levels_only_argus_refuses_all(self, live: dict) -> None:
        steps = live["report"]["degeneracy"]["constant_step_degenerate"]
        total = sum(steps["argus_relative_after"].values())
        assert steps["argus_relative_after"]["correct_refusal"] == total
        for rival in ("sklearn_standard_scaler", "sklearn_variance_threshold",
                      "scipy_zscore_guard", "quantconnect_reference"):
            assert steps[rival]["correct_refusal"] < total, rival


class TestLiveRanking:
    def test_positional_ranking_is_scrambled_and_dated_matches_exact(self, live: dict) -> None:
        ranking = live["report"]["ranking"]
        assert ranking["argus_dated_after"]["spearman_vs_exact"]["rho"] > 0.9999
        assert ranking["argus_positional_before"]["spearman_vs_exact"]["rho"] < 0.9
        assert ranking["argus_dated_after"]["top_50_non_year_over_year"] == 0
        assert ranking["argus_positional_before"]["top_50_non_year_over_year"] > 25


class TestLiveReproducibility:
    def test_the_deterministic_step_is_byte_identical_on_the_same_inputs(
        self, live: dict
    ) -> None:
        frames = dict(list(live["frames"].items())[:400])
        first = compute(live["anchors"], frames, live["names"])
        second = compute(live["anchors"], frames, live["names"])
        assert artefact.dumps(first) == artefact.dumps(second)

    def test_render_is_readable(self, live: dict) -> None:
        text = render(live["report"])
        assert "alignment on anchors" in text
        assert "degenerate-scale detection" in text
