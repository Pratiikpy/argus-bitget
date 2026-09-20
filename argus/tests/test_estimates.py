"""Consensus estimate tests.

The module exists because the desk asked for it: ledger seq 41 ended *"fundamentals cannot be
assessed against expectations without consensus estimates."* So the properties under test are the
ones that keep an expectation honest:

* **no expectation is not a zero surprise.** "In line" and "nobody had a view" are different states,
  and returning 0.0 for the second invents a consensus;
* **the figure is dated at the fetch and never earlier.** Yahoo does not say when a consensus was
  formed, so back-dating it would manufacture the look-ahead `argus.truth` exists to prevent;
* **a revision trail is described, never forecast.** Up, down or mixed is a statement about what
  analysts did, not about what the price will do;
* an instrument with no earnings at all (an index ETF) is an **absence**, not a failure.

Payloads are trimmed from real `quoteSummary` responses, so the shape under test is the shape that
actually arrives — including Yahoo's habit of wrapping every number as ``{"raw": x, "fmt": "..."}``
and returning ``{}`` for a field it does not have.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from argus.market.estimates import (
    PERIOD_LABELS,
    STRONG_REVISION_RATIO,
    Consensus,
    EstimatesError,
    Revisions,
    parse,
)

AT = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _payload(
    *,
    period: str = "0q",
    eps_avg: Any = 2.47269,
    analysts: Any = 42,
    up30: Any = 35,
    down30: Any = 1,
) -> dict[str, Any]:
    """The real response shape, trimmed."""
    return {
        "quoteSummary": {
            "result": [{
                "earningsTrend": {
                    "trend": [{
                        "period": period,
                        "endDate": "2026-10-31",
                        "earningsEstimate": {
                            "avg": {"raw": eps_avg, "fmt": str(eps_avg)},
                            "low": {"raw": 2.3, "fmt": "2.3"},
                            "high": {"raw": 2.7, "fmt": "2.7"},
                            "numberOfAnalysts": {"raw": analysts, "fmt": str(analysts)},
                        },
                        "revenueEstimate": {"avg": {"raw": 108988683310}},
                        "epsRevisions": {
                            "upLast7days": {"raw": 34},
                            "downLast7days": {},
                            "upLast30days": {"raw": up30},
                            "downLast30days": {"raw": down30},
                        },
                    }],
                },
            }],
            "error": None,
        }
    }


def _one(**kw: Any) -> Consensus:
    return parse(_payload(**kw), ticker="NVDA", fetched_at=AT)[0]


class TestTheShapeThatActuallyArrives:
    def test_a_wrapped_number_is_unwrapped(self) -> None:
        assert _one().eps_avg == pytest.approx(2.47269)

    def test_an_absent_field_arrives_as_an_empty_dict_and_counts_as_zero(self) -> None:
        """Yahoo sends `{}` rather than omitting the key or sending null."""
        assert _one().revisions.down_7d == 0

    def test_the_analyst_count_is_read(self) -> None:
        assert _one().analysts == 42

    def test_no_analysts_is_none_rather_than_zero(self) -> None:
        """Zero analysts and an unreported count are the same absence; neither is a real zero."""
        assert _one(analysts=0).analysts is None

    def test_a_missing_eps_leaves_the_period_on_the_record(self) -> None:
        """Dropping it would hide that a period exists and nobody has published a number."""
        record = _one(eps_avg=None)
        assert record.eps_avg is None and record.period == "0q"

    def test_a_boolean_is_not_read_as_a_number(self) -> None:
        assert _one(eps_avg=True).eps_avg is None

    def test_a_string_is_not_coerced(self) -> None:
        assert _one(eps_avg="n/a").eps_avg is None

    def test_an_empty_result_is_refused_rather_than_returned_as_nothing(self) -> None:
        with pytest.raises(EstimatesError, match="no quoteSummary result"):
            parse({"quoteSummary": {"result": []}}, ticker="NVDA", fetched_at=AT)

    def test_a_response_with_no_trend_yields_no_records_without_raising(self) -> None:
        payload = {"quoteSummary": {"result": [{"earningsTrend": {"trend": []}}]}}
        assert parse(payload, ticker="NVDA", fetched_at=AT) == []

    def test_the_period_code_is_translated_for_a_human_reader(self) -> None:
        assert _one(period="0q").period_label == PERIOD_LABELS["0q"]

    def test_an_unknown_period_code_is_labelled_not_dropped(self) -> None:
        assert _one(period="+9q").period_label


class TestNoExpectationIsNotAZeroSurprise:
    def test_a_beat_is_positive(self) -> None:
        assert (_one().surprise_vs(3.0) or 0) > 0

    def test_a_miss_is_negative(self) -> None:
        assert (_one().surprise_vs(2.0) or 0) < 0

    def test_it_is_a_fraction_of_the_expectation(self) -> None:
        record = _one(eps_avg=2.0)
        assert record.surprise_vs(2.2) == pytest.approx(0.1)

    def test_no_consensus_gives_no_surprise_rather_than_zero(self) -> None:
        assert _one(eps_avg=None).surprise_vs(3.0) is None

    def test_a_zero_consensus_gives_no_surprise_rather_than_dividing_by_zero(self) -> None:
        assert _one(eps_avg=0).surprise_vs(3.0) is None

    def test_a_negative_consensus_still_reads_a_beat_as_a_beat(self) -> None:
        """COIN's current-quarter consensus is negative; the sign must come from the direction."""
        assert (_one(eps_avg=-0.14).surprise_vs(-0.10) or 0) > 0


class TestTheRevisionTrailIsDescribedNotForecast:
    def test_one_sided_up_reads_up(self) -> None:
        assert Revisions(0, 0, 35, 1).direction == "up"

    def test_one_sided_down_reads_down(self) -> None:
        assert Revisions(0, 0, 1, 35).direction == "down"

    def test_a_split_field_is_mixed(self) -> None:
        """Forty raising and thirty cutting is disagreement, not a direction."""
        assert Revisions(0, 0, 40, 30).direction == "mixed"

    def test_no_revisions_at_all_is_mixed_rather_than_up(self) -> None:
        assert Revisions(0, 0, 0, 0).direction == "mixed"

    def test_revisions_with_no_cuts_are_up(self) -> None:
        assert Revisions(0, 0, 4, 0).direction == "up"

    def test_the_threshold_is_the_named_constant(self) -> None:
        assert Revisions(0, 0, int(STRONG_REVISION_RATIO * 10), 10).direction == "up"
        assert Revisions(0, 0, int(STRONG_REVISION_RATIO * 10) - 5, 10).direction == "mixed"

    def test_the_net_is_reported_so_a_reader_can_disagree(self) -> None:
        assert Revisions(0, 0, 35, 1).net_30d == 34

    def test_the_rendered_line_states_both_sides_not_just_the_verdict(self) -> None:
        text = _one().render()
        assert "35 up" in text and "1 down" in text


class TestPointInTimeHonesty:
    def test_the_record_is_dated_at_the_fetch(self) -> None:
        assert _one().fetched_at == AT.isoformat()

    def test_the_period_end_is_not_used_as_the_knowledge_date(self) -> None:
        """The quarter ends in October; we did not know this consensus in October."""
        record = _one()
        assert record.end_date == "2026-10-31"
        assert record.fetched_at != record.end_date

    def test_every_record_from_one_response_shares_the_fetch_stamp(self) -> None:
        records = parse(_payload(), ticker="NVDA", fetched_at=AT)
        assert {r.fetched_at for r in records} == {AT.isoformat()}


class TestWhatTheRecordCarries:
    def test_dispersion_is_the_analyst_range(self) -> None:
        assert _one().dispersion == pytest.approx(0.4)

    def test_dispersion_is_unknown_when_the_range_is_missing(self) -> None:
        payload = _payload()
        payload["quoteSummary"]["result"][0]["earningsTrend"]["trend"][0][
            "earningsEstimate"
        ]["low"] = {}
        assert parse(payload, ticker="NVDA", fetched_at=AT)[0].dispersion is None

    def test_it_serialises_everything_a_claim_could_be_checked_against(self) -> None:
        payload = _one().as_dict()
        for key in ("eps_avg", "analysts", "revenue_avg", "revision_direction",
                    "net_30d", "end_date", "fetched_at"):
            assert key in payload

    def test_the_rendered_line_names_the_ticker_and_the_period(self) -> None:
        text = _one().render()
        assert "NVDA" in text and PERIOD_LABELS["0q"] in text

    def test_an_unknown_consensus_renders_as_unknown_not_as_a_number(self) -> None:
        assert "unknown" in _one(eps_avg=None).render()
