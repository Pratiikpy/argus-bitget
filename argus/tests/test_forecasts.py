"""The policy forecast record: point-in-time, honestly scored, honestly sized."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from argus.eval.forecasts import (
    MIN_OBSERVATIONS,
    WINDOW,
    Forecast,
    Scorecard,
    design_effect,
    forecast_series,
    intraclass_correlation,
    probability_of_clearing,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Bar:
    ts: datetime
    close: float


def _bars(closes: list[float]) -> list[Bar]:
    return [Bar(ts=START + timedelta(hours=i), close=c) for i, c in enumerate(closes)]


def _forecast(prob: float, realised: float, *, hour: int = 0, symbol: str = "X") -> Forecast:
    return Forecast(symbol=symbol, at=START + timedelta(hours=hour), hurdle_bps=12.0,
                    probability=prob, realised_bps=realised)


class TestTheProbabilityIsEmpiricalNotFitted:
    def test_it_is_the_share_of_history_that_cleared(self) -> None:
        moves = [100.0] * 30 + [1.0] * 30
        assert probability_of_clearing(moves, 12.0) == pytest.approx(0.5)

    def test_size_is_absolute_so_a_crash_counts_as_a_move(self) -> None:
        """A -200bp move pays a round trip exactly as well as a +200bp one."""
        assert probability_of_clearing([-200.0] * MIN_OBSERVATIONS, 12.0) == 1.0

    def test_too_little_history_refuses_rather_than_guessing(self) -> None:
        assert probability_of_clearing([100.0] * (MIN_OBSERVATIONS - 1), 12.0) < 0

    def test_the_floor_is_stated(self) -> None:
        assert MIN_OBSERVATIONS == 60


class TestNoLookAhead:
    def test_the_last_bar_produces_no_forecast(self) -> None:
        """The loop stops one short, so every forecast is graded against a bar it could not see."""
        bars = _bars([100.0 + (i % 7) for i in range(WINDOW + 50)])
        got = forecast_series("X", bars, hurdle_bps=12.0)
        assert got[-1].at < bars[-1].ts

    def test_the_first_window_bars_produce_nothing(self) -> None:
        bars = _bars([100.0 + (i % 7) for i in range(WINDOW + 10)])
        got = forecast_series("X", bars, hurdle_bps=12.0)
        assert got[0].at >= bars[WINDOW].ts

    def test_truncating_the_future_does_not_change_the_past(self) -> None:
        """The decisive test. If any later bar reached an earlier forecast, cutting the series
        short would move it."""
        closes = [100.0 * (1 + 0.003 * ((i * 7) % 11 - 5)) for i in range(WINDOW + 120)]
        full = forecast_series("X", _bars(closes), hurdle_bps=12.0)
        short = forecast_series("X", _bars(closes[: WINDOW + 60]), hurdle_bps=12.0)
        assert [f.probability for f in short] == pytest.approx(
            [f.probability for f in full[: len(short)]]
        )

    def test_a_zero_price_bar_is_skipped_not_divided_by(self) -> None:
        closes = [100.0 + (i % 5) for i in range(WINDOW + 20)]
        closes[WINDOW + 5] = 0.0
        got = forecast_series("X", _bars(closes), hurdle_bps=12.0)
        assert all(f.realised_bps == f.realised_bps for f in got)  # no NaN


class TestTheScoringIsTheStandardShape:
    def test_a_perfectly_calibrated_record_has_zero_calibration_error(self) -> None:
        """The bug this pins: an earlier version passed correct=True on every row, which made
        every prediction correct by construction and produced an ECE of 0.46 that measured only
        the mean confidence."""
        forecasts = [
            _forecast(0.3, 99.0 if (i % 10) < 3 else 1.0, hour=i) for i in range(1000)
        ]
        card = Scorecard(forecasts=tuple(forecasts), hurdle_bps=12.0)
        assert card.base_rate == pytest.approx(0.3)
        assert card.ece == pytest.approx(0.0, abs=0.01)

    def test_a_confident_and_wrong_record_is_punished(self) -> None:
        forecasts = [_forecast(0.95, 1.0, hour=i) for i in range(200)]
        card = Scorecard(forecasts=tuple(forecasts), hurdle_bps=12.0)
        assert card.brier > 0.8
        assert card.ece > 0.9

    def test_the_base_rate_is_reported_beside_the_hit_rate(self) -> None:
        """A hit rate without the base rate cannot be judged: always saying the majority beats
        most forecasters."""
        card = Scorecard(
            forecasts=tuple(_forecast(0.9, 99.0, hour=i) for i in range(50)), hurdle_bps=12.0
        )
        got = card.as_dict()
        assert got["base_rate"] is not None and got["hit_rate"] is not None

    def test_an_empty_record_reports_nothing_rather_than_zero(self) -> None:
        card = Scorecard(forecasts=(), hurdle_bps=12.0)
        assert card.brier is None and card.ece is None and card.hit_rate is None
        assert "none produced" in card.render()


class TestTheSampleIsSizedHonestly:
    def test_identical_observations_in_a_cluster_give_a_high_icc(self) -> None:
        groups = [[0.9] * 12, [0.1] * 12, [0.5] * 12]
        assert intraclass_correlation(groups) > 0.9

    def test_uncorrelated_observations_give_an_icc_near_zero(self) -> None:
        groups = [[0.1, 0.9, 0.5, 0.3] for _ in range(20)]
        assert intraclass_correlation(groups) == pytest.approx(0.0, abs=0.05)

    def test_a_negative_icc_is_clamped_to_zero(self) -> None:
        """A clustered sample can never be MORE informative than an independent one."""
        assert intraclass_correlation([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]) >= 0.0

    def test_the_design_effect_is_kishs_formula(self) -> None:
        assert design_effect(0.14, 12.0) == pytest.approx(1.0 + 11 * 0.14)

    def test_no_clustering_means_the_raw_count_stands(self) -> None:
        assert design_effect(0.0, 12.0) == 1.0

    def test_the_effective_count_is_never_above_the_raw_count(self) -> None:
        forecasts = [
            _forecast(0.9 if s == "A" else 0.1, 99.0, hour=h, symbol=s)
            for h in range(50) for s in ("A", "B", "C")
        ]
        card = Scorecard(forecasts=tuple(forecasts), hurdle_bps=12.0)
        assert 0 < card.effective_count <= card.count

    def test_distinct_instants_is_reported_beside_the_raw_count(self) -> None:
        """A teardown found a competing entry reporting 2,304 forecasts resting on 119 distinct
        timestamps, with intervals computed as though all 2,304 were independent."""
        forecasts = [
            _forecast(0.5, 99.0, hour=h, symbol=s) for h in range(10) for s in ("A", "B", "C")
        ]
        card = Scorecard(forecasts=tuple(forecasts), hurdle_bps=12.0)
        assert card.count == 30 and card.distinct_instants == 10

    def test_the_report_names_the_effective_sample(self) -> None:
        forecasts = [
            _forecast(0.4, 99.0 if h % 2 else 1.0, hour=h, symbol=s)
            for h in range(40) for s in ("A", "B")
        ]
        text = Scorecard(forecasts=tuple(forecasts), hurdle_bps=12.0).render()
        assert "EFFECTIVE sample" in text
        assert "too narrow by the root of the design effect" in text


class TestTheSplitIsChronological:
    def test_the_halves_do_not_overlap_in_time(self) -> None:
        """A random split on a time series leaks: a shuffled test bar sits between two training
        bars that bracket it."""
        forecasts = [_forecast(0.5, 99.0, hour=h) for h in range(100)]
        early, late = Scorecard(forecasts=tuple(forecasts), hurdle_bps=12.0).split()
        assert max(f.at for f in early.forecasts) < min(f.at for f in late.forecasts)

    def test_both_halves_are_reported(self) -> None:
        forecasts = [_forecast(0.4, 99.0 if h % 3 else 1.0, hour=h) for h in range(200)]
        got = Scorecard(forecasts=tuple(forecasts), hurdle_bps=12.0).as_dict()
        assert got["in_sample_ece"] is not None and got["out_of_sample_ece"] is not None


def test_the_record_says_what_it_is_not() -> None:
    """It measures whether the tape pays a round trip. It is not the desk's judgement, and adding
    the two counts together would inflate a judgement record with an arithmetic one."""
    card = Scorecard(forecasts=(), hurdle_bps=12.0)
    assert "DETERMINISTIC POLICY LAYER" in card.what_this_is_not
    assert "paper_ledger.jsonl" in card.what_this_is_not
    assert card.what_this_is_not in Scorecard(
        forecasts=(_forecast(0.5, 99.0),), hurdle_bps=12.0
    ).render()
