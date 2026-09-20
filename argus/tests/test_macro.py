"""The Treasury yield curve: dated at the issuer, never zero for missing, right units."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from argus.market.macro import (
    MAX_AGE_DAYS,
    Curve,
    FearGreed,
    MacroError,
    evidence,
    fear_greed_evidence,
    fetch,
    fetch_fear_greed,
    latest,
    parse,
    status,
)

LIVE = os.environ.get("ARGUS_LIVE_VENUE") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_VENUE=1 to hit the Treasury")

# Verbatim from the live feed on 2026-09-13, trimmed to two entries.
XML = """<?xml version="1.0" encoding="utf-8"?>
<feed>
<entry><content><m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-09-10T00:00:00</d:NEW_DATE>
<d:BC_1MONTH m:type="Edm.Double">3.90</d:BC_1MONTH>
<d:BC_3MONTH m:type="Edm.Double">4.05</d:BC_3MONTH>
<d:BC_2YEAR m:type="Edm.Double">4.60</d:BC_2YEAR>
<d:BC_10YEAR m:type="Edm.Double">4.90</d:BC_10YEAR>
<d:BC_30YEAR m:type="Edm.Double">5.30</d:BC_30YEAR>
</m:properties></content></entry>
<entry><content><m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-09-11T00:00:00</d:NEW_DATE>
<d:BC_1MONTH m:type="Edm.Double">3.93</d:BC_1MONTH>
<d:BC_3MONTH m:type="Edm.Double">4.07</d:BC_3MONTH>
<d:BC_2YEAR m:type="Edm.Double">4.63</d:BC_2YEAR>
<d:BC_10YEAR m:type="Edm.Double">4.96</d:BC_10YEAR>
<d:BC_30YEAR m:type="Edm.Double">5.35</d:BC_30YEAR>
</m:properties></content></entry>
</feed>"""

AT = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _curve(**over: object) -> Curve:
    yields = {
        "3m": Decimal("4.07"), "2y": Decimal("4.63"), "10y": Decimal("4.96"),
    }
    yields.update(over.pop("yields", {}))  # type: ignore[arg-type]
    return Curve(as_of=over.pop("as_of", date(2026, 9, 11)), yields=yields)  # type: ignore[arg-type]


class TestParsing:
    def test_every_dated_entry_is_read(self) -> None:
        got = parse(XML)
        assert len(got) == 2
        assert [c.as_of for c in got] == [date(2026, 9, 10), date(2026, 9, 11)]

    def test_curves_come_back_oldest_first(self) -> None:
        got = parse(XML)
        assert got[0].as_of < got[1].as_of

    def test_the_issuers_own_field_names_are_read(self) -> None:
        got = parse(XML)[-1]
        assert got.get("10y") == Decimal("4.96")
        assert got.get("1m") == Decimal("3.90") or got.get("1m") == Decimal("3.93")

    def test_an_entry_with_no_date_is_skipped_not_guessed(self) -> None:
        broken = XML.replace('<d:NEW_DATE m:type="Edm.DateTime">2026-09-10T00:00:00</d:NEW_DATE>',
                             "")
        assert len(parse(broken)) == 1

    def test_an_unparseable_date_does_not_abort_the_year(self) -> None:
        broken = XML.replace("2026-09-10T00:00:00", "not-a-date")
        assert len(parse(broken)) == 1

    def test_a_missing_tenor_is_simply_absent(self) -> None:
        thin = XML.replace('<d:BC_30YEAR m:type="Edm.Double">5.35</d:BC_30YEAR>', "")
        got = parse(thin)[-1]
        assert got.get("30y") is None
        assert got.get("10y") is not None

    def test_empty_input_yields_nothing(self) -> None:
        assert parse("") == []


class TestSpreadsAndUnits:
    def test_the_spread_is_reported_in_basis_points(self) -> None:
        """4.96 - 4.63 = 0.33 percentage points = 33bps. An early version said 0.33bps."""
        got = _curve()
        assert got.spread_10y2y == Decimal("0.33")
        assert got.spread_10y2y_bps == Decimal("33.00")

    def test_the_rendered_line_uses_basis_points(self) -> None:
        assert "+33.00bps" in _curve().render()

    def test_a_missing_leg_gives_none_not_zero(self) -> None:
        """Zero is a real state — a flat curve. Absence must not look like flatness."""
        got = Curve(as_of=date(2026, 9, 11), yields={"10y": Decimal("4.96")})
        assert got.spread_10y2y is None
        assert got.spread_10y2y_bps is None
        assert got.is_inverted is None

    def test_an_inverted_curve_is_flagged(self) -> None:
        got = Curve(
            as_of=date(2026, 9, 11),
            yields={"2y": Decimal("5.00"), "10y": Decimal("4.50"), "3m": Decimal("5.20")},
        )
        assert got.is_inverted is True
        assert got.spread_10y2y_bps == Decimal("-50.00")
        assert "INVERTED" in got.render()

    def test_a_flat_curve_is_not_inverted(self) -> None:
        got = Curve(
            as_of=date(2026, 9, 11),
            yields={"2y": Decimal("4.50"), "10y": Decimal("4.50")},
        )
        assert got.is_inverted is False and got.spread_10y2y_bps == Decimal("0.00")

    def test_the_ten_year_three_month_spread_is_available_too(self) -> None:
        assert _curve().spread_10y3m == Decimal("0.89")

    def test_it_serialises_both_units(self) -> None:
        got = _curve().as_dict()
        assert got["spread_10y2y_pct_points"] == "0.33"
        assert got["spread_10y2y_bps"] == "33.00"


class TestStaleness:
    def test_age_is_measured_from_the_issuers_date(self) -> None:
        assert _curve().age_days(now=date(2026, 9, 13)) == 2

    def test_a_recent_curve_is_not_stale(self) -> None:
        assert not _curve().is_stale(now=date(2026, 9, 13))

    def test_a_curve_older_than_the_limit_is_stale(self) -> None:
        old = _curve(as_of=date(2026, 8, 1))
        assert old.is_stale(now=date(2026, 9, 13))
        assert old.age_days(now=date(2026, 9, 13)) > MAX_AGE_DAYS

    def test_a_stale_curve_produces_no_evidence_at_all(self) -> None:
        """An old rate reads exactly like a current one, so it is withheld rather than dated."""
        assert evidence(_curve(as_of=date(2026, 1, 2)), as_of=AT) == []

    def test_the_status_line_names_the_staleness(self) -> None:
        text = status(_curve(as_of=date(2026, 1, 2)), now=AT.date())
        assert "too stale to use" in text

    def test_the_status_line_reports_a_healthy_curve(self) -> None:
        """Pinned to `AT`, not to the wall clock — this asserted `ok` against a fixed
        2026-09-11 fixture while `status()` read `date.today()`, so it passed when written and
        started failing four days later as real time carried the fixture past `MAX_AGE_DAYS`.
        A test that only passes near the date it was written measures the calendar."""
        assert "ok" in status(_curve(), now=AT.date())

    def test_an_unavailable_curve_says_why(self) -> None:
        assert "unavailable" in status(None, "network down")

    def test_an_unavailable_curve_with_no_reason_still_says_something(self) -> None:
        assert "no reason given" in status(None)


class TestEvidence:
    def test_a_healthy_curve_produces_the_curve_and_the_spread(self) -> None:
        got = evidence(_curve(), as_of=AT)
        assert len(got) == 2
        assert "par curve" in got[0].claim and "spread" in got[1].claim

    def test_it_is_dated_at_the_issuer_not_at_the_fetch(self) -> None:
        """A Friday curve read on a Sunday is Friday's information."""
        got = evidence(_curve(as_of=date(2026, 9, 11)), as_of=AT)
        assert got[0].available_at.date() == date(2026, 9, 11)
        assert got[0].available_at.date() != AT.date()

    def test_the_issuer_is_maximally_credible_on_its_own_instrument(self) -> None:
        assert evidence(_curve(), as_of=AT)[0].credibility == 1.0

    def test_the_source_channel_is_macro(self) -> None:
        assert all(e.source == "macro" for e in evidence(_curve(), as_of=AT))

    def test_the_tenors_are_carried_as_structured_attributes(self) -> None:
        got = evidence(_curve(), as_of=AT)[0]
        assert got.attributes["10y"] == "4.96"

    def test_an_inversion_is_explained_rather_than_merely_flagged(self) -> None:
        inverted = Curve(
            as_of=date(2026, 9, 11),
            yields={"2y": Decimal("5.00"), "10y": Decimal("4.50")},
        )
        got = evidence(inverted, as_of=AT)
        assert "inverted" in got[1].claim
        assert "not a trading signal" in got[1].claim

    def test_a_curve_missing_a_leg_still_produces_the_curve_itself(self) -> None:
        thin = Curve(as_of=date(2026, 9, 11), yields={"10y": Decimal("4.96")})
        got = evidence(thin, as_of=AT)
        assert len(got) == 1

    def test_evidence_ids_are_stable_per_observation_date(self) -> None:
        a = evidence(_curve(), as_of=AT)[0]
        b = evidence(_curve(), as_of=AT)[0]
        assert a.id == b.id and "2026-09-11" in a.id


@live_only
class TestAgainstTheIssuer:
    def test_the_treasury_publishes_a_readable_curve(self) -> None:
        got = latest()
        assert got.get("10y") is not None
        assert not got.is_stale()

    def test_a_year_of_curves_comes_back(self) -> None:
        assert len(fetch()) > 100

    def test_an_impossible_year_raises_rather_than_returning_zeros(self) -> None:
        with pytest.raises(MacroError):
            fetch(year=1700)


class TestRiskAppetiteIsLabelledForWhatItMeasures:
    """The one free sentiment feed that answers. It must never read as a per-symbol equity view."""

    def _reading(self) -> FearGreed:
        return FearGreed(as_of=datetime(2026, 9, 13, tzinfo=UTC), value=61,
                         classification="Greed")

    def test_the_rendered_claim_says_it_is_not_a_single_stock_view(self) -> None:
        text = self._reading().render()
        assert "not a view on any single equity" in text

    def test_the_claim_carries_the_number_and_the_date(self) -> None:
        text = self._reading().render()
        assert "61/100" in text and "2026-09-13" in text

    def test_its_credibility_is_low_and_deliberate(self) -> None:
        """A published number about a related market is weaker than a filing about this company,
        and the credibility must say so rather than the docstring."""
        got = fear_greed_evidence(self._reading(), as_of=AT)[0]
        assert got.credibility == 0.35

    def test_the_evidence_is_tagged_crypto_wide(self) -> None:
        got = fear_greed_evidence(self._reading(), as_of=AT)[0]
        assert got.attributes["scope"] == "crypto-wide"

    def test_it_is_dated_by_the_reading_not_by_now(self) -> None:
        """Point-in-time: a two-day-old index must not present as fresh at decision time."""
        got = fear_greed_evidence(self._reading(), as_of=AT)[0]
        assert got.available_at == datetime(2026, 9, 13, tzinfo=UTC)

    def test_the_id_is_stable_for_one_observation_date(self) -> None:
        a = fear_greed_evidence(self._reading(), as_of=AT)[0]
        b = fear_greed_evidence(self._reading(), as_of=AT)[0]
        assert a.id == b.id and "2026-09-13" in a.id

    def test_it_serialises(self) -> None:
        got = self._reading().as_dict()
        assert got["value"] == 61 and got["classification"] == "Greed"


@live_only
class TestTheFearGreedFeedAnswers:
    def test_the_only_working_free_sentiment_feed_still_answers(self) -> None:
        """If this ever fails, the last live sentiment input is gone and the analyst reasons over
        nothing at all — which is a finding, not a flaky test."""
        got = fetch_fear_greed()
        assert 0 <= got.value <= 100 and got.classification


def test_the_risk_appetite_index_does_not_route_to_the_sentiment_analyst() -> None:
    """It is a venue-level index, not a narrative about a company.

    Routing it to `social` had a measurable cost, found by ablating the selection layer rather
    than by reading the code: `agents/selection.py` sends the social channel to the sentiment
    analyst, so one market-wide reading made a DEMOTED analyst run on all eleven symbols and
    charged deliberation for it on every one.
    """
    from argus.agents.selection import SOURCES

    got = fear_greed_evidence(
        FearGreed(as_of=AT, value=61, classification="Greed"), as_of=AT
    )[0]
    assert got.source == "macro"
    assert got.source not in SOURCES["sentiment"]
