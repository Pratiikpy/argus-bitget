"""Point-in-time SEC facts (`market/pit.py`), the deployed source's acceptance gate
(`market/fundamentals.py`), and the rival scoring (`eval/pit_rivals.py`), on constructed filings.

The case every test turns on: a 10-Q dated 2026-08-26 that EDGAR accepted at 20:36 UTC. Asked at
14:00 UTC that day, it did not exist yet; a filed-date gate shows it anyway."""

from __future__ import annotations

import json
import urllib.error
from datetime import UTC, date, datetime, timedelta
from email.message import Message
from pathlib import Path
from typing import Any

from argus.eval import pit_rivals
from argus.market.fundamentals import FundamentalsSource
from argus.market.pit import (
    AcceptanceIndex,
    PitFact,
    availability,
    edgar_close_utc,
    resolve,
)

ACCEPTED = datetime(2026, 8, 26, 20, 36, tzinfo=UTC)
BEFORE = datetime(2026, 8, 26, 14, 0, tzinfo=UTC)
AFTER = datetime(2026, 8, 26, 21, 0, tzinfo=UTC)


def fact(end: date, value: float, accn: str, when: datetime, *, rank: int = 0,
         start: date | None = None) -> PitFact:
    return PitFact(concept="revenue", tag="Revenues", tag_rank=rank, value=value, unit="USD",
                   start=start or end - timedelta(days=90), end=end, filed=when.date(),
                   form="10-Q", accn=accn, available_at=when, basis="accepted")


Q1 = fact(date(2026, 4, 26), 81.6, "a-q1", datetime(2026, 5, 28, 20, 20, tzinfo=UTC))
Q2 = fact(date(2026, 7, 26), 96.2, "a-q2", ACCEPTED)
Q1_RESTATED = fact(date(2026, 4, 26), 80.9, "a-q2", ACCEPTED)


class TestTheGate:
    def test_an_hour_before_acceptance_the_quarter_does_not_exist(self) -> None:
        view = resolve([Q1, Q2], concept="revenue", as_of=BEFORE)
        latest = view.latest()
        assert latest is not None
        assert latest.end == Q1.end
        assert view.withheld == 1

    def test_a_filed_date_gate_leaks_it_the_same_morning(self) -> None:
        view = resolve([Q1, Q2], concept="revenue", as_of=BEFORE, gate="filed-date")
        latest = view.latest()
        assert latest is not None
        assert latest.end == Q2.end

    def test_a_restatement_is_served_once_public_and_not_before(self) -> None:
        facts = [Q1, Q1_RESTATED]
        before = resolve(facts, concept="revenue", as_of=BEFORE).by_end()[Q1.end]
        after = resolve(facts, concept="revenue", as_of=AFTER).by_end()[Q1.end]
        first = resolve(facts, concept="revenue", as_of=AFTER, supersede="first").by_end()[Q1.end]
        assert (before.value, after.value, first.value) == (81.6, 80.9, 81.6)

    def test_an_unknown_accession_is_bounded_to_edgars_close_or_the_fetch(self) -> None:
        index = AcceptanceIndex(cik=1, accepted={}, fetched_at=AFTER)
        when, basis = availability("x", date(2026, 8, 26), index, observed_at=None)
        assert (when, basis) == (edgar_close_utc(date(2026, 8, 26)), "edgar-close")
        live, basis = availability("x", date(2026, 8, 26), index, observed_at=BEFORE)
        assert (live, basis) == (BEFORE, "observed")


class _Sec:
    """SEC's two endpoints the deployed source reads, served from constructed payloads."""

    def __init__(self) -> None:
        self.concept = {"tag": "Revenues", "units": {"USD": [
            {"start": "2026-01-26", "end": "2026-04-26", "val": 81.6, "accn": "a-q1",
             "filed": "2026-05-28", "form": "10-Q", "fy": 2027, "fp": "Q1"},
            {"start": "2026-04-27", "end": "2026-07-26", "val": 96.2, "accn": "a-q2",
             "filed": "2026-08-26", "form": "10-Q", "fy": 2027, "fp": "Q2"},
        ]}}
        self.submissions = {"filings": {"recent": {
            "accessionNumber": ["a-q1", "a-q2"],
            "acceptanceDateTime": ["2026-05-28T20:20:00.000Z", "2026-08-26T20:36:00.000Z"],
        }, "files": []}}

    def get(self, url: str) -> dict[str, Any]:
        if "companyconcept" in url:
            return self.concept
        if "submissions" in url:
            return self.submissions
        raise AssertionError(url)


class _Edgar:
    def cik_for(self, ticker: str) -> int:
        return 1045810


def _source(sec: _Sec) -> FundamentalsSource:
    source = FundamentalsSource(edgar=_Edgar())  # type: ignore[arg-type]
    source._get = sec.get  # type: ignore[method-assign]
    return source


class TestTheDeployedSource:
    def test_a_past_cutoff_the_morning_of_a_filing_does_not_see_it(self) -> None:
        facts, status = _source(_Sec()).facts("NVDA", concept="revenue", as_of=BEFORE)
        assert [f.end for f in facts] == [date(2026, 4, 26)]
        assert any("accepted after as_of withheld" in s for s in status)

    def test_after_acceptance_it_does(self) -> None:
        facts, _ = _source(_Sec()).facts("NVDA", concept="revenue", as_of=AFTER)
        assert {f.end for f in facts} == {date(2026, 4, 26), date(2026, 7, 26)}

    def test_without_the_index_the_bound_errs_late_never_early(self) -> None:
        sec = _Sec()
        sec.submissions = {}
        facts, status = _source(sec).facts("NVDA", concept="revenue", as_of=AFTER)
        assert [f.end for f in facts] == [date(2026, 4, 26)]   # 22:00 New York has not come
        assert any("no acceptance time in EDGAR's index" in s for s in status)

    def test_a_filer_that_changed_tags_is_answered_from_the_live_one(self) -> None:
        # AAPL's `Revenues` stops in 2018; its later quarters sit under the contract-with-customer
        # tag. The first tag with any rows used to answer alone.
        sec = _Sec()
        old = {"tag": "Revenues", "units": {"USD": [
            {"start": "2018-07-01", "end": "2018-09-29", "val": 62.9, "accn": "a-old",
             "filed": "2018-11-05", "form": "10-K", "fy": 2018, "fp": "FY"},
            {"start": "2018-04-01", "end": "2018-06-30", "val": 53.3, "accn": "a-old2",
             "filed": "2018-08-01", "form": "10-Q", "fy": 2018, "fp": "Q3"}]}}
        new = dict(sec.concept, tag="RevenueFromContractWithCustomerExcludingAssessedTax")

        def get(url: str) -> dict[str, Any]:
            if "/Revenues.json" in url:
                return old
            if "RevenueFromContractWithCustomerExcludingAssessedTax" in url:
                return new
            if "companyconcept" in url:
                raise urllib.error.HTTPError(url, 404, "no such tag", Message(), None)
            return sec.get(url)

        source = _source(sec)
        source._get = get  # type: ignore[method-assign]
        facts, status = source.facts("AAPL", concept="revenue", as_of=AFTER)
        assert max(f.end for f in facts) == date(2026, 7, 26)
        assert date(2018, 6, 30) in {f.end for f in facts}
        assert any("Revenues (2)" in s and "RevenueFromContract" in s for s in status), status

    def test_a_live_question_sees_every_row_just_fetched(self) -> None:
        now = datetime.now(UTC)
        facts, _ = _source(_Sec()).facts("NVDA", concept="revenue", as_of=now)
        assert len(facts) == 2


class TestTheRivalScoring:
    def test_a_date_aligned_rival_leaks_the_morning_of_a_filing(self) -> None:
        probes = [pit_rivals.Probe("T", "a-q2", ACCEPTED, "before"),
                  pit_rivals.Probe("T", "a-q2", ACCEPTED, "after")]
        outputs: dict[str, Any] = {"vibe": {"tickers": {"T": {"pit": {"series": {"revenue": [
            ["2026-04-26", "2026-05-28", 81.6], ["2026-07-26", "2026-08-26", 96.2]]}}}}},
            "openbb": {"tickers": {}}, "langalpha": {"tickers": {}}}
        out = pit_rivals.score_rival("vibe_trading_pit", "T", [Q1, Q2], probes, outputs)
        assert (out["leak"], out["right"]) == (1, 1)

    def test_a_first_reported_value_after_a_restatement_is_stale(self) -> None:
        outputs: dict[str, Any] = {"openbb": {"tickers": {"T": {"pit": {"rows": [
            {"period_ending": "2026-04-26", "total_revenue": 81.6}]}}}},
            "vibe": {"tickers": {}}, "langalpha": {"tickers": {}}}
        out = pit_rivals.score_rival_restatements("openbb_pit", "T", [Q1, Q1_RESTATED], outputs)
        assert (out["stale_after_restatement"], out["right"]) == (1, 0)

    def test_a_month_end_label_is_the_fiscal_quarter_it_reports(self) -> None:
        # Yahoo labels NVDA's quarter ending 2026-07-26 as 2026-07-31; that is the same quarter,
        # not a quarter that never became public.
        outputs: dict[str, Any] = {"langalpha": {"tickers": {"T": {"response": {"data": {
            "Total Revenue": {"2026-07-31": 96.2, "2026-04-30": 81.6}}}}}}}
        probes = [pit_rivals.Probe("T", "a-q2", ACCEPTED, "after")]
        out = pit_rivals.score_rival("langalpha_yfinance", "T", [Q1, Q2], probes, outputs)
        assert (out["right"], out["leak"]) == (1, 0)


class TestTheCommittedRun:
    run = json.loads((Path(__file__).resolve().parents[1] / "data" / "pit_rivals.json")
                     .read_text(encoding="utf-8"))

    def test_argus_answers_every_question_without_a_leak(self) -> None:
        argus = self.run["argus"]
        assert argus["latest_quarter"]["right"] == argus["latest_quarter"]["questions"] == 619
        assert argus["latest_quarter"]["leak"] == 0
        assert argus["restated_quarter"]["right_after"] == argus["restated_quarter"][
            "questions_after"]

    def test_the_rivals_scores_are_their_own_outputs(self) -> None:
        rivals = self.run["rivals_from_their_own_outputs"]["latest_quarter"]
        assert rivals["vibe_trading_pit"]["leak"] > 0
        assert rivals["openbb_pit"]["leak"] > 0

    def test_every_rival_loses_paired_and_significantly(self) -> None:
        for rival, pair in self.run["significance"]["paired"].items():
            assert pair["rival_only_right"] == 0, rival
            assert pair["p_value"] < 1e-6, rival

    def test_the_held_out_tickers_exclude_the_design_case(self) -> None:
        held = self.run["held_out"]
        assert held["design_ticker"] not in held["held_out_tickers"]
        assert held["argus"]["right"] == held["argus"]["n"] > 0
        assert held["argus"]["leak"] == 0


class TestTheStatistics:
    def test_exact_mcnemar_is_the_binomial_tail(self) -> None:
        assert pit_rivals.exact_mcnemar(0, 0) == 1.0
        assert pit_rivals.exact_mcnemar(4, 0) == 0.125
        assert pit_rivals.exact_mcnemar(3, 3) == 1.0

    def test_wilson_stays_inside_the_unit_interval(self) -> None:
        low, high = pit_rivals.wilson(619, 619)
        assert 0.99 < low < high <= 1.0
