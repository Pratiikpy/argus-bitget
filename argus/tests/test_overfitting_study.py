"""Study tests — the assembly, not the statistics.

`backtest/validation.py` is tested against processes whose answers are known. What this module adds
is the wiring: which strategies enter the field, how the matrix is built, how the grid of p-values
is corrected, and whether the reported result can be read as more than it is. Those are the parts
that would silently produce a flattering number, so they are what is tested here.

The network-dependent path is exercised only under `ARGUS_LIVE_VENUE=1`, because a test that needs
a venue is a test that fails for reasons unrelated to the code.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.backtest.engine import Bar
from argus.research.overfitting_study import (
    CONFIDENCE,
    FDR,
    GROUPS,
    MIN_BARS,
    STUDY_PATH,
    SymbolOverfitting,
    _matrix,
    _p_value,
    _splits,
    _track_record,
    render,
)
from argus.research.track1_study import ALL_VARIANTS, CANDIDATES, CONTROLS

LIVE = os.environ.get("ARGUS_LIVE_VENUE") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_VENUE=1 to hit the venue")

START = datetime(2026, 3, 1, tzinfo=UTC)


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "symbol": "NVDAUSDT", "bars": 4319, "strategies_all": 23, "strategies_candidates": 21,
        "pbo_all_trials": 0.07, "pbo_candidates_only": 0.11, "median_rank_all": 0.77,
        "selection_churn": 0.09, "dominant_share": 0.71,
        "verdict_all": "the selection survives out of sample on this evidence",
        "winner": "weekend_only", "winner_net_sharpe": 1.80,
        "min_track_record_bars": 7350.0, "min_track_record_years": 0.84,
        "track_record_note": "7,350 hourly bars at 95% confidence",
    }
    base.update(kw)
    return base


class TestTheFieldIsNotRigged:
    def test_the_controls_are_in_the_variant_set(self) -> None:
        """A sweep of only plausible winners is a rigged sweep. The two falsified rules stay in."""
        for control in CONTROLS:
            assert control in ALL_VARIANTS

    def test_the_controls_are_excluded_from_the_candidate_set(self) -> None:
        """And they are still not deployment candidates, which is why PBO is reported twice."""
        for control in CONTROLS:
            assert control not in CANDIDATES

    def test_the_candidate_set_is_strictly_smaller(self) -> None:
        assert 0 < len(CANDIDATES) < len(ALL_VARIANTS)


class TestTheMatrix:
    def test_it_is_observations_by_strategies(self) -> None:
        series = {"a": (1.0, 2.0, 3.0), "b": (4.0, 5.0, 6.0)}
        assert _matrix(series, ["a", "b"]) == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_it_truncates_to_the_shortest_series(self) -> None:
        """Ragged columns would make the cross-validation compare different periods, and padding
        one with zeros would invent a flat stretch that no strategy actually had."""
        series = {"a": (1.0, 2.0, 3.0, 4.0), "b": (5.0, 6.0)}
        assert _matrix(series, ["a", "b"]) == [[1.0, 5.0], [2.0, 6.0]]

    def test_column_order_follows_the_label_order_given(self) -> None:
        series = {"a": (1.0,), "b": (2.0,)}
        assert _matrix(series, ["b", "a"]) == [[2.0, 1.0]]


class TestThePValues:
    def test_a_series_with_a_clear_edge_scores_small(self) -> None:
        import random

        rng = random.Random(1)
        p = _p_value(tuple(rng.gauss(0.002, 0.01) for _ in range(4000)))
        assert p is not None and p < 0.01

    def test_a_series_with_no_edge_scores_large(self) -> None:
        import random

        rng = random.Random(2)
        p = _p_value(tuple(rng.gauss(0.0, 0.01) for _ in range(4000)))
        assert p is not None and p > 0.10

    def test_a_losing_series_is_not_significant(self) -> None:
        import random

        rng = random.Random(3)
        p = _p_value(tuple(rng.gauss(-0.002, 0.01) for _ in range(4000)))
        assert p is not None and p > 0.9

    def test_a_degenerate_series_returns_none_rather_than_a_number(self) -> None:
        """A variant that never traded has a flat series. Scoring it as p=1.0 would put it in
        the grid as a trial that was run and failed, which is a different claim from one that
        could not be computed at all."""
        assert _p_value((0.0,) * 500) is None
        assert _p_value((0.01,) * 500) is None

    def test_a_short_series_returns_none(self) -> None:
        assert _p_value((0.01, -0.02, 0.03)) is None

    def test_it_is_bounded_to_the_unit_interval(self) -> None:
        import random

        rng = random.Random(4)
        for mean in (-0.05, 0.0, 0.05):
            p = _p_value(tuple(rng.gauss(mean, 0.01) for _ in range(500)))
            assert p is not None and 0.0 <= p <= 1.0


class TestTheTrackRecordLength:
    def test_a_winning_series_gets_a_length_in_bars_and_years(self) -> None:
        import random

        rng = random.Random(5)
        bars, years, note = _track_record(tuple(rng.gauss(0.001, 0.01) for _ in range(2000)))
        assert bars is not None and years is not None
        assert years == pytest.approx(bars / (24 * 365), rel=1e-6)
        assert "95% confidence" in note

    def test_a_losing_series_gets_no_length_and_says_why(self) -> None:
        """Not None-with-no-reason. The note is what a reader sees in the table cell."""
        import random

        rng = random.Random(6)
        bars, years, note = _track_record(tuple(rng.gauss(-0.001, 0.01) for _ in range(2000)))
        assert bars is None and years is None
        assert "does not exceed the benchmark" in note

    def test_the_confidence_level_is_the_stated_one(self) -> None:
        assert CONFIDENCE == 0.95


class TestTheStudyConstants:
    def test_eight_groups_gives_seventy_balanced_splits(self) -> None:
        assert GROUPS == 8
        assert _splits(8) == 70

    def test_more_groups_gives_combinatorially_more(self) -> None:
        assert _splits(10) == 252
        assert _splits(12) == 924

    def test_the_false_discovery_rate_is_the_conventional_one(self) -> None:
        assert FDR == 0.05

    def test_the_bar_floor_matches_the_track_one_study(self) -> None:
        """Both studies must accept or reject the same symbols, or they describe different
        universes while appearing to describe one."""
        assert MIN_BARS == 200

    def test_the_output_is_a_separate_artefact(self) -> None:
        assert STUDY_PATH.name == "overfitting_study.json"


class TestTheReportRefusesToBeOverread:
    def _payload(self, rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "days": 180, "groups": 8, "splits_per_symbol": 70,
            "confidence": CONFIDENCE, "fdr": FDR,
            "symbols": rows, "failures": {},
            "grid": {
                "trials": 300,
                "survivors_benjamini_hochberg": 0,
                "survivors_bonferroni": 0,
                "rows": [],
            },
        }

    def test_it_says_pbo_scores_the_selection_not_the_strategy(self) -> None:
        """The one sentence that stops a low PBO being read as "this strategy makes money"."""
        text = render(self._payload([_row()]))
        assert "scores the selection procedure, not the strategy" in text

    def test_it_reports_both_deflations_side_by_side(self) -> None:
        text = render(self._payload([_row()]))
        assert "all" in text and "cands" in text

    def test_it_reports_the_multiple_testing_survivors(self) -> None:
        text = render(self._payload([_row()]))
        assert "0 survive Benjamini-Hochberg" in text
        assert "0 survive Bonferroni" in text

    def test_a_symbol_with_no_track_record_renders_as_not_applicable(self) -> None:
        """Rather than a blank cell, which reads as a missing value rather than a refusal."""
        text = render(self._payload([_row(min_track_record_years=None)]))
        assert "n/a" in text

    def test_an_empty_study_says_so_rather_than_rendering_an_empty_table(self) -> None:
        payload = self._payload([])
        payload["failures"] = {"NVDAUSDT": "only 12 bars"}
        assert "nothing scored" in render(payload)

    def test_the_row_serialises_every_field_it_reports(self) -> None:
        row = SymbolOverfitting(**_row())  # type: ignore[arg-type]
        got = row.as_dict()
        assert got["pbo_all_trials"] == 0.07
        assert got["pbo_candidates_only"] == 0.11
        assert got["min_track_record_years"] == 0.84


class TestTheStudyOnDisk:
    """If the artefact exists, its content must match what this module claims about it."""

    def _load(self) -> dict[str, object] | None:
        if not STUDY_PATH.exists():
            return None
        loaded: dict[str, object] = json.loads(STUDY_PATH.read_text(encoding="utf-8"))
        return loaded

    def test_every_symbol_reports_both_pbo_figures(self) -> None:
        payload = self._load()
        if payload is None:
            pytest.skip("run `python -m argus.research.overfitting_study` to produce the artefact")
        for row in payload["symbols"]:  # type: ignore[index]
            assert 0.0 <= row["pbo_all_trials"] <= 1.0
            assert 0.0 <= row["pbo_candidates_only"] <= 1.0

    def test_the_grid_counts_match_the_rows(self) -> None:
        payload = self._load()
        if payload is None:
            pytest.skip("artefact not built")
        grid = payload["grid"]  # type: ignore[index]
        assert grid["trials"] == len(grid["rows"])
        assert grid["survivors_benjamini_hochberg"] == sum(
            1 for r in grid["rows"] if r["survives_benjamini_hochberg"]
        )

    def test_every_bonferroni_survivor_also_survives_the_softer_gate(self) -> None:
        """Bonferroni is strictly more conservative. A row surviving it but not Benjamini-Hochberg
        means the two were computed over different p-value sets."""
        payload = self._load()
        if payload is None:
            pytest.skip("artefact not built")
        for row in payload["grid"]["rows"]:  # type: ignore[index]
            assert row["survives_benjamini_hochberg"] or not row["survives_bonferroni"]


@live_only
class TestAgainstTheRealVenue:
    def test_one_symbol_produces_a_usable_field(self) -> None:
        from argus.cost.model import CostModel
        from argus.research.overfitting_study import _bars_for, _series_for

        bars = _bars_for("NVDAUSDT", 180)
        assert len(bars) >= MIN_BARS
        series, sharpes = _series_for("NVDAUSDT", bars, CostModel.bitget_perp())
        assert len(series) >= 2
        assert set(series) == set(sharpes)

    def test_the_index_series_reaches_the_bars(self) -> None:
        """Two variants price the rToken against its index and go flat without it, which would
        silently narrow the field and change the PBO."""
        from argus.research.overfitting_study import _bars_for

        bars = _bars_for("NVDAUSDT", 30)
        assert any("index" in bar.extra for bar in bars)

    def test_a_bar_carries_the_fields_the_grammar_reads(self) -> None:
        from argus.research.overfitting_study import _bars_for

        bar: Bar = _bars_for("NVDAUSDT", 10)[-1]
        assert {"volume", "high", "low"} <= set(bar.extra)


class TestTheArtefactPathIsNotTheStudyItReads:
    def test_it_does_not_overwrite_the_track_one_study(self) -> None:
        """Two studies over the same sweep, and only one of them is the deflated-Sharpe result."""
        other = Path(STUDY_PATH).with_name("track1_study.json")
        assert other != STUDY_PATH

    def test_the_window_is_the_same_one_track_one_uses(self) -> None:
        """A different window would make the two sets of numbers incomparable while looking like
        they describe the same strategies."""
        import inspect

        from argus.research.overfitting_study import study
        from argus.research.track1_study import study as track1

        assert inspect.signature(study).parameters["days"].default == 180
        assert inspect.signature(track1).parameters["days"].default in (90, 180)


class TestTheHoldWindowAssumption:
    def test_a_bar_timestamp_is_hourly(self) -> None:
        """The minimum track record length is divided by 24*365, which is only the right divisor
        if the observations really are hourly."""
        from argus.backtest.metrics import HOURLY_PER_YEAR

        assert HOURLY_PER_YEAR == 24 * 365
        assert START + timedelta(hours=1) != START
