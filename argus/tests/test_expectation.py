"""Expectation gap tests.

Track 3's Information Extraction sub-theme names "expectation gap detection" as its example. The
properties pinned here are the ones that keep a gap honest:

* **a surprise is refused, not manufactured.** No keyless source publishes the consensus that
  existed for an already-reported quarter, so comparing a reported quarter to a forward estimate
  that was never about it would invent the most important number on the page;
* **quarters are matched on period end, never on ordering.** Fiscal calendars drift — NVDA's
  year-ago comparable sits 370 days back — and pairing the wrong quarters yields a growth rate that
  looks precise and describes nothing;
* **a negative or zero base yields no growth rate.** Going from -1.00 to +0.50 is not 150% growth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from argus.desk.expectation import (
    MATERIAL_DECELERATION_PP,
    YEAR_MATCH_DAYS,
    detect,
)


@dataclass(frozen=True)
class Fact:
    concept: str
    value: float
    end: date
    filed: date
    is_quarterly: bool = True


@dataclass(frozen=True)
class Consensus:
    period: str
    end_date: str
    eps_avg: float | None
    analysts: int | None


# NVDA's real shape: quarter ends drift, and the year-ago comparable is 370 days back.
NVDA = [
    Fact("eps_diluted", 2.46, date(2026, 7, 26), date(2026, 8, 26)),
    Fact("eps_diluted", 2.39, date(2026, 4, 26), date(2026, 5, 20)),
    Fact("eps_diluted", 1.30, date(2025, 10, 26), date(2025, 11, 19)),
    Fact("eps_diluted", 1.08, date(2025, 7, 27), date(2025, 8, 28)),
    Fact("eps_diluted", 0.76, date(2025, 4, 27), date(2025, 5, 22)),
]
FORWARD = [Consensus("0q", "2026-10-31", 2.47269, 42), Consensus("+1q", "2027-01-31", 2.74, 40)]


def _gap(**kw: object):
    defaults = dict(
        ticker="NVDA", reported=NVDA, consensus=FORWARD,
        revision_direction="up", revisions_up_30d=35, revisions_down_30d=1,
    )
    defaults.update(kw)
    return detect(**defaults)  # type: ignore[arg-type]


class TestASurpriseIsRefusedNotManufactured:
    def test_the_surprise_is_always_none(self) -> None:
        assert _gap().surprise is None

    def test_the_reason_travels_with_the_record(self) -> None:
        assert "unavailable rather than zero" in _gap().surprise_reason

    def test_the_reason_is_rendered_not_buried_in_a_field(self) -> None:
        assert any("unavailable rather than zero" in line for line in _gap().render())

    def test_it_survives_serialisation(self) -> None:
        payload = _gap().as_dict()
        assert payload["surprise"] is None and payload["surprise_reason"]


class TestQuartersAreMatchedOnPeriodEnd:
    def test_the_year_ago_comparable_is_found_across_a_calendar_drift(self) -> None:
        """2026-10-31 against 2025-10-26 is 370 days, not 365."""
        got = _gap()
        assert got.comparable_period_end == "2025-10-26"
        assert got.comparable_value == pytest.approx(1.30)

    def test_the_implied_growth_uses_that_comparable(self) -> None:
        got = _gap()
        assert got.implied_growth_pct == pytest.approx((2.47269 - 1.30) / 1.30 * 100, abs=0.1)

    def test_the_adjacent_quarter_is_never_mistaken_for_the_comparable(self) -> None:
        """The previous quarter sits ~90 days back and must fall outside the band."""
        low, high = YEAR_MATCH_DAYS
        assert not low <= 90 <= high
        assert not low <= 275 <= high

    def test_no_comparable_in_the_band_yields_no_implied_growth(self) -> None:
        lonely = [Fact("eps_diluted", 2.46, date(2026, 7, 26), date(2026, 8, 26))]
        got = _gap(reported=lonely)
        assert got.comparable_period_end is None
        assert got.implied_growth_pct is None

    def test_an_unparseable_period_end_is_refused_rather_than_guessed(self) -> None:
        got = _gap(consensus=[Consensus("0q", "not-a-date", 2.4, 42)])
        assert got.comparable_period_end is None

    def test_annual_rows_are_ignored(self) -> None:
        """A cumulative row sits beside the quarterly one under the same fiscal period."""
        polluted = [*NVDA, Fact("eps_diluted", 9.9, date(2025, 10, 26), date(2025, 11, 19), False)]
        assert _gap(reported=polluted).comparable_value == pytest.approx(1.30)

    def test_a_different_concept_is_not_compared(self) -> None:
        revenue = [Fact("revenue", 1e10, d.end, d.filed) for d in NVDA]
        assert _gap(reported=revenue).implied_growth_pct is None


class TestGrowthRefusesMeaninglessBases:
    def test_a_zero_base_gives_no_growth_rate(self) -> None:
        zeroed = [
            Fact("eps_diluted", 2.46, date(2026, 7, 26), date(2026, 8, 26)),
            Fact("eps_diluted", 0.0, date(2025, 7, 27), date(2025, 8, 28)),
        ]
        got = _gap(reported=zeroed, consensus=[Consensus("0q", "2026-07-26", 2.4, 10)])
        assert got.implied_growth_pct is None

    def test_a_negative_base_gives_no_growth_rate(self) -> None:
        """-1.00 to +0.50 is not 150% growth in any sense a reader would accept."""
        negative = [
            Fact("eps_diluted", 0.50, date(2026, 7, 26), date(2026, 8, 26)),
            Fact("eps_diluted", -1.00, date(2025, 7, 27), date(2025, 8, 28)),
        ]
        got = _gap(reported=negative, consensus=[Consensus("0q", "2026-07-26", 0.5, 10)])
        assert got.implied_growth_pct is None


class TestTheDirectionOfTravel:
    def test_slowing_growth_reads_as_decelerating(self) -> None:
        assert _gap().direction_of_travel == "decelerating"

    def test_the_delivered_trail_is_reported_newest_first(self) -> None:
        got = _gap()
        assert got.delivered_growth_pct[0][0] == "2026-07-26"

    def test_a_small_change_is_steady_rather_than_a_trend(self) -> None:
        delivered = _gap().latest_delivered_pct or 0.0
        nudged = [Consensus("0q", "2026-10-31", 1.30 * (1 + (delivered + 1) / 100), 42)]
        assert _gap(consensus=nudged).direction_of_travel == "steady"

    def test_the_threshold_is_the_named_constant(self) -> None:
        assert MATERIAL_DECELERATION_PP > 0

    def test_unknown_when_there_is_nothing_to_compare(self) -> None:
        lonely = [Fact("eps_diluted", 2.46, date(2026, 7, 26), date(2026, 8, 26))]
        assert _gap(reported=lonely).direction_of_travel == "unknown"

    def test_rising_estimates_into_deceleration_are_named(self) -> None:
        """The configuration that most often ends in a de-rating."""
        got = _gap()
        assert got.expectations_rising_into_deceleration
        assert any("raising their numbers" in line for line in got.render())

    def test_falling_estimates_into_deceleration_are_not_that_configuration(self) -> None:
        got = _gap(revision_direction="down", revisions_up_30d=1, revisions_down_30d=35)
        assert got.direction_of_travel == "decelerating"
        assert not got.expectations_rising_into_deceleration


class TestWhenThereIsNoConsensus:
    def test_a_missing_period_yields_an_empty_gap_rather_than_an_error(self) -> None:
        got = _gap(consensus=[])
        assert got.expected_value is None

    def test_a_consensus_with_no_eps_is_treated_as_absent(self) -> None:
        got = _gap(consensus=[Consensus("0q", "2026-10-31", None, 42)])
        assert got.expected_value is None

    def test_it_still_renders(self) -> None:
        assert _gap(consensus=[]).render()

    def test_the_rendered_line_says_which_ticker_has_no_consensus(self) -> None:
        assert "NVDA" in _gap(consensus=[]).render()[0]


class TestTheRecordIsReadable:
    def test_it_names_how_many_analysts_stand_behind_the_number(self) -> None:
        assert "42 analysts" in _gap().render()[0]

    def test_it_states_the_revision_trail_both_ways(self) -> None:
        text = " ".join(_gap().render())
        assert "35 up" in text and "1 down" in text

    def test_it_serialises_every_field_a_reader_needs(self) -> None:
        payload = _gap().as_dict()
        for key in ("implied_growth_pct", "delivered_growth_pct", "direction_of_travel",
                    "revision_direction", "comparable_period_end", "surprise_reason"):
            assert key in payload
