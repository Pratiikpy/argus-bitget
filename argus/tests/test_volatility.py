"""VIX: percentiles not levels, absence not a neutral 20, and the units are right."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from argus.market.volatility import (
    MIN_HISTORY,
    TRADING_DAYS,
    Reading,
    Regime,
    VolatilityError,
    classify,
    evidence,
    fetch,
    history,
    parse_history,
    percentile_of,
    status,
)

LIVE = os.environ.get("ARGUS_LIVE_VENUE") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_VENUE=1 to hit CBOE")

AT = datetime(2026, 9, 13, 9, 34, 15, tzinfo=UTC)

CSV = """DATE,OPEN,HIGH,LOW,CLOSE
01/02/1990,17.240000,17.240000,17.240000,17.240000
01/03/1990,18.190000,18.190000,18.190000,18.190000
09/11/2026,16.120000,16.200000,15.590000,15.840000
"""


def _reading(level: float = 15.84, percentile: float | None = 0.38,
             observations: int = 9271) -> Reading:
    return Reading(as_of=AT, level=level, change=-2.0, day_high=16.2, day_low=15.59,
                   percentile=percentile, observations=observations)


class TestTheCsvIsParsedNotGuessed:
    def test_it_reads_the_real_column_names(self) -> None:
        got = parse_history(CSV)
        assert len(got) == 3 and got[0] == (date(1990, 1, 2), 17.24)

    def test_rows_come_back_in_date_order(self) -> None:
        got = parse_history(CSV)
        assert [d for d, _ in got] == sorted(d for d, _ in got)

    def test_an_unparseable_row_is_skipped_not_zeroed(self) -> None:
        """A zero close would sit at the bottom of every percentile it touched."""
        broken = CSV + "not-a-date,1,1,1,1\n09/12/2026,1,1,1,not-a-number\n"
        assert len(parse_history(broken)) == 3

    def test_a_row_missing_its_close_is_skipped(self) -> None:
        assert len(parse_history(CSV + "09/12/2026,1,1,1,\n")) == 3

    def test_an_empty_file_is_empty_not_an_error(self) -> None:
        assert parse_history("DATE,OPEN,HIGH,LOW,CLOSE\n") == []

    def test_identical_early_rows_are_kept(self) -> None:
        """1990 rows carry one value in every column because the index was reconstructed. That is
        real data, not a corrupt row, and filtering it would quietly shorten the history."""
        assert (date(1990, 1, 2), 17.24) in parse_history(CSV)


class TestPercentilesNotLevels:
    def test_a_level_below_everything_is_the_bottom(self) -> None:
        assert percentile_of(1.0, [10.0] * MIN_HISTORY) == 0.0

    def test_a_level_above_everything_is_the_top(self) -> None:
        assert percentile_of(99.0, [10.0] * MIN_HISTORY) == 1.0

    def test_too_little_history_refuses_rather_than_ranking(self) -> None:
        assert percentile_of(20.0, [10.0] * (MIN_HISTORY - 1)) is None

    def test_the_minimum_history_is_about_a_year(self) -> None:
        assert MIN_HISTORY == 250

    def test_an_unranked_reading_is_unknown_not_normal(self) -> None:
        """Reporting an unplaced level as a normal regime would invent context."""
        assert _reading(percentile=None).regime is Regime.UNKNOWN

    def test_the_bands_are_percentile_based(self) -> None:
        assert classify(0.99) is Regime.STRESSED
        assert classify(0.85) is Regime.ELEVATED
        assert classify(0.50) is Regime.NORMAL
        assert classify(0.05) is Regime.CALM

    def test_the_band_edges_are_pinned(self) -> None:
        assert classify(0.95) is Regime.STRESSED
        assert classify(0.80) is Regime.ELEVATED
        assert classify(0.20) is Regime.CALM
        assert classify(0.21) is Regime.NORMAL

    def test_none_stays_unknown(self) -> None:
        assert classify(None) is Regime.UNKNOWN


class TestTheUnitsAreRight:
    def test_the_daily_move_is_the_annual_figure_descaled(self) -> None:
        """VIX is quoted annualised. Reporting it as a daily move would be a 16x error."""
        got = _reading(level=16.0)
        assert got.implied_daily_move_pct == pytest.approx(16.0 / TRADING_DAYS**0.5)

    def test_basis_points_are_a_hundred_times_percent(self) -> None:
        got = _reading(level=15.84)
        assert got.implied_daily_move_bps == pytest.approx(
            100 * got.implied_daily_move_pct
        )

    def test_a_normal_regime_implies_a_move_far_above_the_cost_hurdle(self) -> None:
        """The finding this makes sayable: the desk's abstentions are not because the tape is too
        quiet to pay a round trip. At VIX 15.84 a one-sigma day is about 100bps against an 18.8bps
        hurdle, so the binding constraint is direction, not size."""
        assert _reading(level=15.84).clears_hurdle(18.8)

    def test_a_dead_tape_would_not_clear_it(self) -> None:
        assert not _reading(level=2.0).clears_hurdle(18.8)

    def test_the_rendered_claim_says_what_the_index_measures(self) -> None:
        text = _reading().render()
        assert "US equities" in text and "these tokens track" in text

    def test_an_unranked_reading_says_so_in_words(self) -> None:
        assert "too few to rank" in _reading(percentile=None, observations=10).render()


class TestEvidence:
    def test_it_is_dated_by_the_quote_not_by_now(self) -> None:
        got = evidence(_reading(), as_of=datetime(2027, 1, 1, tzinfo=UTC))[0]
        assert got.available_at == AT

    def test_its_credibility_is_higher_than_the_crypto_sentiment_feed(self) -> None:
        """The whole point of the distinction: an index measuring the market these tokens track
        is worth more than one measuring a neighbouring market."""
        from argus.market.macro import FearGreed, fear_greed_evidence

        vix = evidence(_reading(), as_of=AT)[0]
        fng = fear_greed_evidence(
            FearGreed(as_of=AT, value=61, classification="Greed"), as_of=AT
        )[0]
        assert vix.credibility > fng.credibility

    def test_the_regime_travels_as_an_attribute(self) -> None:
        got = evidence(_reading(percentile=0.99), as_of=AT)[0]
        assert got.attributes["regime"] == "stressed"

    def test_ids_are_stable_per_observation_date(self) -> None:
        a = evidence(_reading(), as_of=AT)[0]
        b = evidence(_reading(), as_of=AT)[0]
        assert a.id == b.id and "2026-09-13" in a.id

    def test_it_serialises(self) -> None:
        got = _reading().as_dict()
        assert got["regime"] == "normal" and got["level"] == 15.84
        assert got["implied_daily_move_bps"] > 0


class TestFailureIsNamedNotNeutralised:
    @staticmethod
    def _break_the_feed(monkeypatch: pytest.MonkeyPatch) -> None:
        """Deterministically dead. An earlier version of this test used a one-second timeout and
        passed only when the CDN happened to be slow, which is a test that reports the network's
        mood rather than the code's behaviour."""
        import argus.market.volatility as vol

        def _dead(url: str, *, timeout: int) -> bytes:
            raise VolatilityError("connection refused")

        monkeypatch.setattr(vol, "_get", _dead)

    def test_a_missing_cache_and_a_dead_feed_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is no neutral VIX. Returning 20 would be inventing a regime."""
        self._break_the_feed(monkeypatch)
        with pytest.raises(VolatilityError):
            history(path=tmp_path / "absent.csv", refresh=True)

    def test_a_stale_cache_is_used_when_the_feed_is_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An old history is almost as good as a new one; an exception here would take down a
        decision cycle over a CDN hiccup."""
        cache = tmp_path / "vix.csv"
        cache.write_text(CSV, encoding="utf-8")
        self._break_the_feed(monkeypatch)
        assert len(history(path=cache, refresh=True)) == 3

    def test_a_fresh_cache_is_not_refetched_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The history is a megabyte and nearly static; refetching it every cycle would be a
        self-inflicted rate limit."""
        cache = tmp_path / "vix.csv"
        cache.write_text(CSV, encoding="utf-8")
        self._break_the_feed(monkeypatch)
        assert len(history(path=cache)) == 3

    def test_a_quote_without_a_level_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import argus.market.volatility as vol

        monkeypatch.setattr(
            vol, "_get",
            lambda url, *, timeout: b'{"timestamp": "2026-09-13 09:34:15", "data": {}}',
        )
        with pytest.raises(VolatilityError, match="no usable level"):
            fetch()

    def test_an_unreadable_timestamp_raises_rather_than_claiming_now(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stamping the reading with the current time would claim a freshness CBOE did not give."""
        import argus.market.volatility as vol

        monkeypatch.setattr(
            vol, "_get",
            lambda url, *, timeout: b'{"timestamp": "whenever", "data": {"current_price": 15.8}}',
        )
        with pytest.raises(VolatilityError, match="unreadable timestamp"):
            fetch()

    def test_the_status_line_names_the_absence(self) -> None:
        assert "unavailable" in status(None, "connection refused")

    def test_the_status_line_reports_the_regime_when_present(self) -> None:
        assert "normal" in status(_reading())


@live_only
class TestAgainstCboe:
    def test_the_quote_answers_with_a_plausible_level(self) -> None:
        got = fetch()
        assert 5.0 < got.level < 100.0

    def test_the_history_goes_back_decades(self) -> None:
        got = history(refresh=True)
        assert len(got) > 8000
        assert got[0][0].year <= 1990

    def test_the_live_reading_is_placed_in_its_history(self) -> None:
        got = fetch()
        assert got.percentile is not None
        assert got.regime is not Regime.UNKNOWN
