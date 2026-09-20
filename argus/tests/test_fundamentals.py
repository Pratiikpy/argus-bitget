"""XBRL fundamentals tests.

The decisive property is the duration discriminator. SEC returns, from one Q2 10-Q, a 181-day
cumulative figure and a 90-day quarterly figure with the same ``end`` date and the same ``fp: Q2``.
For NVDA those are 177.8B and 96.2B: reading the wrong one overstates quarterly revenue by 85%, and
nothing in ``fp``, ``form`` or ``frame`` reliably separates them — ``frame`` is *absent* on exactly
the cumulative row that most needs flagging.

The second property is point-in-time. Facts carry the date they were filed, and a restatement of a
two-year-old quarter filed last week is not something a backtest of that quarter may see.

Fixtures use the real response shape and the real NVDA numbers.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from argus.agents.claims import check as check_claims
from argus.market.fundamentals import (
    CONCEPTS,
    Fact,
    FundamentalsError,
    FundamentalsSource,
    latest_per_period,
    parse_concept,
)
from argus.truth.evidence import Evidence

AS_OF = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)

# Verbatim shape from https://data.sec.gov/api/xbrl/companyconcept/CIK0001045810/us-gaap/Revenues.json
NVDA_REVENUES = {
    "cik": 1045810,
    "tag": "Revenues",
    "units": {
        "USD": [
            {"start": "2025-01-27", "end": "2026-01-25", "val": 215900000000,
             "fy": 2026, "fp": "FY", "form": "10-K", "filed": "2026-02-26", "frame": "CY2025"},
            {"start": "2026-01-26", "end": "2026-04-26", "val": 81615000000,
             "fy": 2027, "fp": "Q1", "form": "10-Q", "filed": "2026-05-20", "frame": "CY2026Q1"},
            # The trap: same end date, same fp, 181 days, and no frame to warn you.
            {"start": "2026-01-26", "end": "2026-07-26", "val": 177837000000,
             "fy": 2027, "fp": "Q2", "form": "10-Q", "filed": "2026-08-26", "frame": None},
            {"start": "2026-04-27", "end": "2026-07-26", "val": 96221000000,
             "fy": 2027, "fp": "Q2", "form": "10-Q", "filed": "2026-08-26", "frame": "CY2026Q2"},
        ]
    },
}


def _facts() -> list[Fact]:
    return parse_concept(NVDA_REVENUES, concept="revenue")


class TestTheDurationTrap:
    """The failure this module exists to prevent."""

    def test_the_cumulative_and_the_quarter_both_arrive(self) -> None:
        same_end = [f for f in _facts() if f.end == date(2026, 7, 26)]
        assert len(same_end) == 2
        assert {f.value for f in same_end} == {177837000000.0, 96221000000.0}

    def test_only_the_ninety_day_row_is_quarterly(self) -> None:
        quarterly = [f for f in _facts() if f.is_quarterly]
        assert {f.value for f in quarterly} == {81615000000.0, 96221000000.0}
        assert 177837000000.0 not in {f.value for f in quarterly}

    def test_the_fiscal_period_field_cannot_separate_them(self) -> None:
        """Both are fp=Q2 on the same 10-Q. Trusting `fp` is how the 85% error happens."""
        q2 = [f for f in _facts() if f.fiscal_period == "Q2"]
        assert len(q2) == 2
        assert len({f.is_quarterly for f in q2}) == 2, "only the duration tells them apart"

    def test_the_frame_field_is_absent_where_it_would_be_most_useful(self) -> None:
        cumulative = next(f for f in _facts() if f.duration_days == 181)
        assert cumulative.frame is None

    def test_the_period_kind_names_an_odd_duration_rather_than_guessing(self) -> None:
        assert next(f for f in _facts() if f.duration_days == 181).period_kind == "181-day"

    def test_an_annual_row_is_recognised_as_annual(self) -> None:
        annual = next(f for f in _facts() if f.fiscal_period == "FY")
        assert annual.is_annual and not annual.is_quarterly

    def test_a_balance_sheet_item_has_no_duration_and_is_an_instant(self) -> None:
        """A snapshot is not a flow, and subtracting two of them is not growth."""
        payload = {"tag": "Assets", "units": {"USD": [
            {"end": "2026-07-26", "val": 1, "form": "10-Q", "filed": "2026-08-26"}
        ]}}
        fact = parse_concept(payload, concept="assets")[0]
        assert fact.is_instant and fact.period_kind == "instant"
        assert fact.duration_days is None


class TestRestatementsSupersede:
    def _restated(self) -> list[Fact]:
        payload = {"tag": "Revenues", "units": {"USD": [
            {"start": "2026-04-27", "end": "2026-07-26", "val": 96221000000,
             "fy": 2027, "fp": "Q2", "form": "10-Q", "filed": "2026-08-26"},
            {"start": "2026-04-27", "end": "2026-07-26", "val": 95000000000,
             "fy": 2027, "fp": "Q2", "form": "10-Q/A", "filed": "2026-09-10"},
        ]}}
        return parse_concept(payload, concept="revenue")

    def test_the_latest_filing_wins(self) -> None:
        kept, superseded = latest_per_period(self._restated())
        assert len(kept) == 1
        assert kept[0].value == 95000000000.0
        assert kept[0].form == "10-Q/A"
        assert superseded == 1

    def test_the_supersede_count_is_returned_rather_than_dropped_silently(self) -> None:
        """A restatement is itself news about a company."""
        _, superseded = latest_per_period(self._restated())
        assert superseded == 1

    def test_distinct_periods_are_never_merged(self) -> None:
        kept, superseded = latest_per_period([f for f in _facts() if f.is_quarterly])
        assert len(kept) == 2
        assert superseded == 0

    def test_a_cumulative_row_does_not_supersede_a_quarterly_one(self) -> None:
        """They share an end date; the period key must keep them apart."""
        kept, _ = latest_per_period(_facts())
        ends = [(f.period_kind, f.end) for f in kept]
        assert len(ends) == len(set(ends))
        assert any(k == "quarter" and e == date(2026, 7, 26) for k, e in ends)
        assert any(k == "181-day" and e == date(2026, 7, 26) for k, e in ends)

    def test_results_come_back_newest_first(self) -> None:
        kept, _ = latest_per_period(_facts())
        assert [f.end for f in kept] == sorted((f.end for f in kept), reverse=True)


class TestParsingIsRobust:
    def test_a_row_without_a_value_is_skipped(self) -> None:
        payload = {"tag": "X", "units": {"USD": [{"end": "2026-07-26", "filed": "2026-08-26"}]}}
        assert parse_concept(payload, concept="revenue") == []

    def test_a_row_without_a_filed_date_is_skipped(self) -> None:
        """Without it the fact cannot be placed on the clock, so it cannot be used point-in-time."""
        payload = {"tag": "X", "units": {"USD": [{"end": "2026-07-26", "val": 1}]}}
        assert parse_concept(payload, concept="revenue") == []

    def test_a_malformed_date_is_skipped_rather_than_raising(self) -> None:
        payload = {"tag": "X", "units": {"USD": [
            {"end": "not-a-date", "val": 1, "filed": "2026-08-26"}
        ]}}
        assert parse_concept(payload, concept="revenue") == []

    def test_a_non_numeric_value_is_skipped(self) -> None:
        payload = {"tag": "X", "units": {"USD": [
            {"end": "2026-07-26", "val": "n/a", "filed": "2026-08-26"}
        ]}}
        assert parse_concept(payload, concept="revenue") == []

    def test_an_empty_payload_yields_nothing(self) -> None:
        assert parse_concept({}, concept="revenue") == []

    def test_every_unit_is_read_not_just_usd(self) -> None:
        """EPS arrives as USD/shares; a USD-only reader would silently return nothing."""
        payload = {"tag": "EarningsPerShareDiluted", "units": {"USD/shares": [
            {"start": "2026-04-27", "end": "2026-07-26", "val": 2.46,
             "form": "10-Q", "filed": "2026-08-26"}
        ]}}
        facts = parse_concept(payload, concept="eps_diluted")
        assert len(facts) == 1 and facts[0].unit == "USD/shares"


class TestTagSynonyms:
    def test_revenue_lists_the_tags_issuers_actually_use(self) -> None:
        """NVDA uses `Revenues`; many issuers use the contract-with-customer tag instead."""
        assert "Revenues" in CONCEPTS["revenue"]
        assert any("ContractWithCustomer" in t for t in CONCEPTS["revenue"])

    def test_every_concept_has_at_least_one_tag(self) -> None:
        assert all(tags for tags in CONCEPTS.values())

    def test_an_unknown_concept_is_refused_by_name(self) -> None:
        from argus.market.fundamentals import FundamentalsSource

        with pytest.raises(FundamentalsError, match="unknown concept"):
            FundamentalsSource().facts("NVDA", concept="vibes", as_of=AS_OF)

    def test_a_naive_clock_is_refused(self) -> None:
        from argus.market.fundamentals import FundamentalsSource

        with pytest.raises(FundamentalsError, match="timezone-aware"):
            FundamentalsSource().facts(
                "NVDA", concept="revenue", as_of=datetime(2026, 9, 12, 12, 0)
            )


class TestRendering:
    def test_a_large_figure_is_rendered_in_billions(self) -> None:
        fact = next(f for f in _facts() if f.value == 96221000000.0)
        assert "96.22B" in fact.render()

    def test_the_render_states_the_period_kind_and_the_filing(self) -> None:
        fact = next(f for f in _facts() if f.value == 96221000000.0)
        text = fact.render()
        assert "quarter ending 2026-07-26" in text
        assert "filed 2026-08-26 on 10-Q" in text

    def test_a_small_figure_is_not_forced_into_billions(self) -> None:
        payload = {"tag": "EarningsPerShareDiluted", "units": {"USD/shares": [
            {"start": "2026-04-27", "end": "2026-07-26", "val": 2.46,
             "form": "10-Q", "filed": "2026-08-26"}
        ]}}
        assert "2.46" in parse_concept(payload, concept="eps_diluted")[0].render()

    def test_a_fact_serialises_with_its_duration(self) -> None:
        payload = next(f for f in _facts() if f.value == 96221000000.0).as_dict()
        assert payload["duration_days"] == 90
        assert payload["period_kind"] == "quarter"


class TestTheEvidenceCarriesItsRecord:
    """Direction is the one property of a filed figure that a thesis can contradict.

    A desk writing "revenue growth supports the long case" against a filed sequential decline has
    made an error that no numeric check can see, because it quoted no number. The quarter-on-quarter
    item carries ``growing`` so `argus.agents.claims` can settle it; the per-period facts carry no
    such field, and a claim about direction is left unexamined rather than guessed at.
    """

    class _Stub(FundamentalsSource):
        def __init__(self, facts: list[Fact]) -> None:  # no network
            self._facts = facts

        def facts(  # type: ignore[override]
            self, ticker: str, *, concept: str, as_of: datetime, quarterly_only: bool = True
        ) -> tuple[list[Fact], list[str]]:
            return (self._facts if concept == "revenue" else []), []

    @staticmethod
    def _quarter(end: date, value: float, filed: date) -> Fact:
        return Fact(
            concept="revenue", tag="Revenues", value=value, unit="USD",
            start=end - timedelta(days=91), end=end, filed=filed, form="10-Q",
            fiscal_year=2027, fiscal_period="Q2", frame=None,
        )

    def _evidence(self, *, newest: float, previous: float) -> list[Evidence]:
        facts = [
            self._quarter(date(2026, 7, 26), newest, date(2026, 8, 20)),
            self._quarter(date(2026, 4, 26), previous, date(2026, 5, 20)),
        ]
        got, _ = self._Stub(facts).evidence("NVDA", as_of=AS_OF)
        return got

    def _qoq(self, **kw: float) -> Evidence:
        return next(e for e in self._evidence(**kw) if e.id.endswith("-qoq"))

    def test_the_quarter_on_quarter_item_states_its_direction(self) -> None:
        assert self._qoq(newest=100.0, previous=80.0).attributes["growing"] is True
        assert self._qoq(newest=80.0, previous=100.0).attributes["growing"] is False

    def test_the_change_is_carried_as_a_number_not_only_as_prose(self) -> None:
        assert self._qoq(newest=110.0, previous=100.0).attributes["change_pct"] == pytest.approx(10)

    def test_a_per_period_fact_claims_no_direction(self) -> None:
        facts = [e for e in self._evidence(newest=100.0, previous=80.0) if not e.id.endswith("qoq")]
        assert facts and all("growing" not in e.attributes for e in facts)

    def test_the_checker_fires_on_evidence_built_the_real_way(self) -> None:
        evidence = self._evidence(newest=80.0, previous=100.0)
        report = check_claims(
            "Revenue growth supports the long case.",
            records=[(e.id, e.attributes) for e in evidence],
        )
        assert not report.sound
        assert report.contradictions[0].rule == "fundamental_growth"
