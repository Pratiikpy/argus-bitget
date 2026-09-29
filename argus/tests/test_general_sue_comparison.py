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

from argus.eval.general_sue_comparison import (
    ALIGNMENT_METHODS,
    ARGUS_ARM,
    DEGENERACY_METHODS,
    SCOPE_STATEMENT,
    CompanyFact,
    Point,
    Reading,
    Verdict,
    Window,
    argus_desk_points,
    classify,
    companyfacts_rows,
    compute,
    constructed_windows,
    correct_summary,
    draw_sample,
    edgartools_fiscal_year_end,
    edgartools_fiscal_year_label,
    edgartools_quarterize,
    exact_sue,
    fabricated_magnitudes,
    fetch_anchor_panel,
    fetch_frames_panel,
    filer_rows,
    is_correct,
    is_true_yoy,
    mcnemar_exact,
    paired_counts,
    pandas_agreement,
    pandas_collisions,
    parse_companyfacts_rows,
    reading_status,
    render,
    run_argus_dated,
    run_argus_positional,
    run_edgartools_fiscal,
    run_pandas_lenient,
    run_pandas_strict,
    run_raw_fiscal_labels,
    step_windows,
    unit_sweep,
)
from argus.truth import artefact

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


class TestFilerRowsOffline:
    def _points(self) -> list[Point]:
        return _points(_quarter_ends(date(2026, 6, 28), 16, skip_month=12), NVDA_SHAPED)

    def test_every_reading_has_exactly_one_status(self) -> None:
        assert reading_status(Reading(1.0, ())) == "value"
        assert reading_status(Reading(float("inf"), ())) == "non_finite"
        assert reading_status(Reading(float("nan"), ())) == "non_finite"
        assert reading_status(Reading(None, (), refusal="short")) == "refused"
        assert reading_status(Reading(None, (), crashed="ValueError: x")) == "crashed"

    def test_correct_needs_a_finite_value_and_every_pair_year_over_year(self) -> None:
        points = self._points()
        dated, positional = run_argus_dated(points), run_argus_positional(points)
        assert is_correct(dated)
        assert not is_correct(positional)  # finite, but five quarters back
        assert not is_correct(Reading(float("inf"), dated.pairs))
        assert not is_correct(Reading(1.0, ()))  # no pairs cannot be "every pair"
        one_wrong = Reading(dated.value, (*dated.pairs[:-1], positional.pairs[0]))
        assert not is_correct(one_wrong)

    def test_rows_carry_status_pairs_truth_and_verdict_per_method(self) -> None:
        panel = {"42": self._points()}
        readings = {name: {"42": m(panel["42"])} for name, m in ALIGNMENT_METHODS.items()}
        (row,) = filer_rows(panel, readings)
        assert row["cik"] == "42"
        assert row["truth"] == "value"
        assert row["exact_sue"] == pytest.approx(readings["argus_dated_after"]["42"].value,
                                                 rel=1e-12)
        dated = row["arms"]["argus_dated_after"]
        assert dated["status"] == "value" and dated["correct"] and dated["all_pairs_yoy"]
        assert all(365 <= gap <= 366 for gap in dated["pair_gaps_days"])
        assert row["arms"]["argus_positional_before"]["correct"] is False
        assert row["arms"]["pandas_period_strict"]["status"] == "refused"

    def test_a_degenerate_filer_is_marked_and_a_refusal_is_not_counted_correct(self) -> None:
        flat = _points(_quarter_ends(date(2026, 6, 28), 16, skip_month=12), ["0.5"] * 16)
        readings = {ARGUS_ARM: {"7": run_argus_dated(flat)}}
        (row,) = filer_rows({"7": flat}, readings)
        assert row["truth"] == "degenerate" and row["exact_sue"] is None
        assert row["arms"][ARGUS_ARM]["status"] == "refused"
        assert row["arms"][ARGUS_ARM]["correct"] is False

    def test_summary_and_paired_counts_add_up(self) -> None:
        rows = [{"arms": {"a": {"correct": a, "status": "value", "all_pairs_yoy": a},
                          "r": {"correct": r, "status": "value", "all_pairs_yoy": r}}}
                for a, r in [(True, True), (True, False), (True, False), (False, True),
                             (False, False)]]
        summary = correct_summary(rows, ("a", "r"))
        assert summary["a"]["correct"] == 3 and summary["r"]["correct"] == 2
        assert summary["r"]["value_with_a_non_year_over_year_pair"] == 3
        paired = paired_counts(rows, "a", "r")
        assert (paired["both_correct"], paired["only_argus_correct"],
                paired["only_rival_correct"], paired["neither_correct"]) == (1, 2, 1, 1)
        assert paired["difference_in_correct_share"] == pytest.approx(0.2)

    def test_mcnemar_exact_matches_the_binomial_tail(self) -> None:
        assert mcnemar_exact(0, 0)["p"] == 1.0
        assert mcnemar_exact(5, 0)["p"] == pytest.approx(2 / 32)
        assert mcnemar_exact(3, 3)["p"] == 1.0
        from scipy.stats import binomtest

        assert mcnemar_exact(40, 12)["p"] == pytest.approx(binomtest(12, 52, 0.5).pvalue,
                                                           rel=1e-9)
        huge = mcnemar_exact(3000, 0)
        assert huge["log10_p"] < -900  # finite where p itself underflows


def _fact(start: date, end: date, val: str, fy: int, fp: str, form: str = "10-Q",
          filed: date | None = None) -> CompanyFact:
    return CompanyFact(start=start, end=end, value=Decimal(val), fy=fy, fp=fp, form=form,
                       filed=filed or date(end.year + (end.month + 1) // 12,
                                           (end.month + 1) % 12 + 1, 1))


def _calendar_filer(years: range, values: dict[tuple[int, int], str]) -> list[CompanyFact]:
    """A December-year-end filer: Q1-Q3 as 10-Q quarters (fp Q1-Q3), the year as a 10-K FY fact,
    no standalone Q4 — the SEC XBRL norm."""
    facts: list[CompanyFact] = []
    for year in years:
        for q, (start_month, end_month, end_day) in enumerate(
                [(1, 3, 31), (4, 6, 30), (7, 9, 30)], start=1):
            facts.append(_fact(date(year, start_month, 1), date(year, end_month, end_day),
                               values[(year, q)], year, f"Q{q}"))
        facts.append(_fact(date(year, 1, 1), date(year, 12, 31), "9.99", year, "FY",
                           form="10-K", filed=date(year + 1, 2, 20)))
    return facts


class TestEdgartoolsPairingOffline:
    def test_fiscal_year_label_matches_edgartools_docstring_examples(self) -> None:
        """`edgar/entity/enhanced_statement.py:1375-1397`."""
        assert edgartools_fiscal_year_label(date(2025, 10, 26), 1) == 2026  # NVIDIA Q3
        assert edgartools_fiscal_year_label(date(2026, 1, 26), 1) == 2026  # NVIDIA Q4
        assert edgartools_fiscal_year_label(date(2023, 12, 30), 9) == 2024  # Apple Q1
        assert edgartools_fiscal_year_label(date(2024, 6, 28), 9) == 2024  # Apple Q3
        assert edgartools_fiscal_year_label(date(2023, 1, 1), 12) == 2022  # 52/53-week edge

    def test_fiscal_year_end_is_the_most_common_fy_month(self) -> None:
        facts = _calendar_filer(range(2020, 2023), {(y, q): "1" for y in range(2020, 2023)
                                                    for q in (1, 2, 3)})
        assert edgartools_fiscal_year_end(facts) == 12
        assert edgartools_fiscal_year_end([f for f in facts if f.fp != "FY"]) == 12  # default

    def test_quarterize_keeps_quarter_durations_and_the_latest_periodic_filing(self) -> None:
        q = _fact(date(2024, 1, 1), date(2024, 3, 31), "1.00", 2024, "Q1",
                  filed=date(2024, 5, 1))
        restated = _fact(date(2024, 1, 1), date(2024, 3, 31), "1.10", 2025, "Q1",
                         filed=date(2025, 5, 1))
        proxy = _fact(date(2024, 1, 1), date(2024, 3, 31), "9.00", 0, "", form="DEF 14A",
                      filed=date(2026, 1, 1))
        ytd = _fact(date(2024, 1, 1), date(2024, 6, 30), "2.00", 2024, "Q2")
        kept = edgartools_quarterize([q, restated, proxy, ytd])
        assert [f.value for f in kept] == [Decimal("1.10")]  # periodic over later non-periodic

    def test_clean_calendar_pairs_year_over_year_and_equals_argus_dated(self) -> None:
        years = range(2020, 2025)
        values = {(y, q): str(Decimal("0.10") * (y - 2019) + Decimal(q) / 100
                              + (Decimal("0.03") if (y, q) == (2024, 3) else 0))
                  for y in years for q in (1, 2, 3)}
        facts = _calendar_filer(years, values)
        labelled = run_edgartools_fiscal(facts)
        assert labelled.label_collisions == 0
        assert is_correct(labelled.reading)
        argus = run_argus_dated(argus_desk_points(facts))
        assert set(labelled.reading.pairs) == set(argus.pairs)
        assert labelled.reading.value == pytest.approx(argus.value, rel=1e-12)

    def test_quarters_relabelled_fy_by_a_10k_collide_and_are_not_all_year_over_year(
        self,
    ) -> None:
        """A filer whose 10-K also tags its quarters: the latest filing of each quarter is the
        10-K (fp FY), so three quarters share one label and the fp pairing picks the wrong one."""
        years = range(2020, 2025)
        values = {(y, q): str(Decimal("0.10") * (y - 2019) + Decimal(q) / 100
                              + (Decimal("0.03") if (y, q) == (2024, 3) else 0))
                  for y in years for q in (1, 2, 3)}
        facts = _calendar_filer(years, values)
        for year in years:
            for q, (start_month, end_month, end_day) in enumerate(
                    [(1, 3, 31), (4, 6, 30), (7, 9, 30)], start=1):
                facts.append(_fact(date(year, start_month, 1), date(year, end_month, end_day),
                                   values[(year, q)], year, "FY", form="10-K",
                                   filed=date(year + 1, 2, 20)))
        labelled = run_edgartools_fiscal(facts)
        assert labelled.label_collisions > 0
        assert not is_correct(labelled.reading)
        assert is_correct(run_argus_dated(argus_desk_points(facts)))

    def test_raw_fy_labels_break_on_comparatives_the_derived_label_does_not(self) -> None:
        """Comparative re-filing: each quarter's latest filing is next year's 10-Q, tagged with
        next year's fy — raw (fy, fp) then pairs a quarter with itself-a-year-later's label."""
        years = range(2020, 2025)
        values = {(y, q): str(Decimal("0.10") * (y - 2019) + Decimal(q) / 100
                              + (Decimal("0.03") if (y, q) == (2024, 3) else 0))
                  for y in years for q in (1, 2, 3)}
        facts = _calendar_filer(years, values)
        for year in years[:-1]:
            for q, (start_month, end_month, end_day) in enumerate(
                    [(1, 3, 31), (4, 6, 30), (7, 9, 30)], start=1):
                facts.append(_fact(date(year, start_month, 1), date(year, end_month, end_day),
                                   values[(year, q)], year + 1, f"Q{q}",
                                   filed=date(year + 1, end_month + 1, 10)))
        assert is_correct(run_edgartools_fiscal(facts).reading)
        assert not is_correct(run_raw_fiscal_labels(facts).reading)

    def test_newest_quarter_without_a_labelled_partner_is_refused(self) -> None:
        values = {(y, q): "1" for y in range(2020, 2025) for q in (1, 2, 3)}
        facts = [f for f in _calendar_filer(range(2020, 2025), values)
                 if f.end != date(2023, 9, 30)]
        reading = run_edgartools_fiscal(facts).reading
        assert reading.value is None and "newest" in reading.refusal

    def test_companyfacts_rows_round_trip_exact_decimals(self) -> None:
        payload = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [
            {"start": "2024-01-01", "end": "2024-03-31", "val": Decimal("0.1"), "fy": 2024,
             "fp": "Q1", "form": "10-Q", "filed": "2024-05-01", "accn": "a"},
        ]}}}}}
        rows = companyfacts_rows(payload)
        assert rows == [["2024-01-01", "2024-03-31", "0.1", "2024", "Q1", "10-Q", "2024-05-01",
                         "a"]]
        (fact,) = parse_companyfacts_rows(rows)
        assert fact.value == Decimal("0.1") and fact.fy == 2024
        assert companyfacts_rows({"facts": {}}) is None

    def test_sample_is_seeded_and_excludes_anchors(self) -> None:
        panel = {str(i): [Point(end=date(2020, 1, 1), value=Decimal(1))] * 12
                 for i in range(1, 600)}
        first, second = draw_sample(panel, ["5"]), draw_sample(panel, ["5"])
        assert first == second and len(first) == 300 and "5" not in first


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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
class TestLiveRanking:
    def test_positional_ranking_is_scrambled_and_dated_matches_exact(self, live: dict) -> None:
        ranking = live["report"]["ranking"]
        assert ranking["argus_dated_after"]["spearman_vs_exact"]["rho"] > 0.9999
        assert ranking["argus_positional_before"]["spearman_vs_exact"]["rho"] < 0.9
        assert ranking["argus_dated_after"]["top_50_non_year_over_year"] == 0
        assert ranking["argus_positional_before"]["top_50_non_year_over_year"] > 25


@pytest.mark.network
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
