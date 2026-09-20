"""Official Bitget Skill integration tests.

Track 3 scores "Skill integration count and effectiveness". Count is easy to fake — wire in nineteen
tools, let the dead ones return nothing, and the integration looks five times bigger than it
behaves. So the properties under test here are the ones that make the count honest:

* a tool that **timed out** is never recorded the same way as a tool that **answered with nothing**,
  because the first is an unknown and the second may be a real "no news";
* an upstream that replies with only an error envelope is empty, not healthy;
* only a call that answered becomes evidence, and a failure contributes none at all rather than
  entering the desk's evidence list as the finding "no data";
* the Skill's own numbers are recomputed from Bitget candles before being believed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from argus.market.skills import (
    PROBES,
    RSI_AGREEMENT_POINTS,
    SKILLS,
    Health,
    Probe,
    SkillReport,
    ToolHealth,
    cross_check_rsi,
    evidence,
    hollow,
    probe,
    rsi,
)

AT = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


class FakeClient:
    """Answers from a table keyed by tool name. Anything unlisted times out."""

    def __init__(self, table: dict[str, tuple[Any, str]]) -> None:
        self.table = table
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(
        self, tool: str, args: dict[str, Any], *, timeout: int = 20
    ) -> tuple[Any, str]:
        self.calls.append((tool, args))
        return self.table.get(tool, (None, f"bitget:{tool}: unavailable (TimeoutError)"))


OK_RSI = ({"symbol": "NVDAUSDT", "timeframe": "4h", "rsi": 28.47, "signal": "oversold"},
          "bitget:technical_analysis: ok")


def _one(skill: str = "technical-analysis", tool: str = "technical_analysis") -> tuple[Probe, ...]:
    return (Probe(skill, tool, "rsi", {}, "momentum: RSI(14)", symbol_key="symbol"),)


class TestHealthIsNeverAmbiguous:
    def test_a_real_payload_is_ok(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}),
                       symbol="NVDAUSDT", at=AT, probes=_one())
        assert report.results[0].health is Health.OK

    def test_a_timeout_is_a_timeout_not_an_empty_market(self) -> None:
        report = probe(FakeClient({}), symbol="NVDAUSDT", at=AT, probes=_one())
        assert report.results[0].health is Health.TIMEOUT

    def test_a_tool_error_is_its_own_state(self) -> None:
        table = {"cross_asset": (None, "bitget:cross_asset: tool reported an error (boom)")}
        probes = _one("macro-analyst", "cross_asset")
        assert probe(FakeClient(table), symbol="X", at=AT,
                     probes=probes).results[0].health is Health.TOOL_ERROR

    def test_a_mismatched_reply_is_unavailable_not_data(self) -> None:
        """The live defect: one tool's answer arriving under another tool's name."""
        table = {"sentiment_index": (None, "bitget:sentiment_index: response correlation failed")}
        probes = _one("sentiment-analyst", "sentiment_index")
        assert probe(FakeClient(table), symbol="X", at=AT,
                     probes=probes).results[0].health is Health.UNAVAILABLE

    def test_an_upstream_error_envelope_is_empty_not_healthy(self) -> None:
        """Seen live: {"error": "", "url": "..."} — the tool worked, the data did not arrive."""
        table = {"network_status": ({"error": "", "url": "https://example/api"},
                                    "bitget:network_status: ok")}
        probes = _one("market-intel", "network_status")
        got = probe(FakeClient(table), symbol="X", at=AT, probes=probes).results[0]
        assert got.health is Health.EMPTY
        assert "error envelope" in got.detail

    def test_the_other_live_envelope_shape_is_also_empty(self) -> None:
        table = {"sentiment_index": ({"alt_me_error": ""}, "bitget:sentiment_index: ok")}
        probes = _one("sentiment-analyst", "sentiment_index")
        assert probe(FakeClient(table), symbol="X", at=AT,
                     probes=probes).results[0].health is Health.EMPTY

    def test_a_payload_carrying_data_beside_an_error_key_is_still_ok(self) -> None:
        """Only an envelope of *nothing but* error keys is empty."""
        table = {"technical_analysis": ({"error": "", "rsi": 41.2}, "bitget:x: ok")}
        assert probe(FakeClient(table), symbol="X", at=AT,
                     probes=_one()).results[0].health is Health.OK

    def test_a_zero_filled_yield_curve_built_from_error_legs_is_empty(self) -> None:
        """Seen live 2026-09-13: every tenor an error, spread 0.0, inverted false; was OK."""
        curve = {"yield_curve": {k: {"error": ""} for k in ("t3m", "t2y", "t10y", "t30y")},
                 "spread_10y2y": 0.0, "inverted": False, "note": "derived"}
        table = {"rates_yields": (curve, "bitget:rates_yields: ok")}
        got = probe(FakeClient(table), symbol="X", at=AT,
                    probes=_one("macro-analyst", "rates_yields")).results[0]
        assert got.health is Health.EMPTY
        assert "hollow" in got.detail

    def test_a_news_day_of_empty_feeds_is_empty_not_quiet(self) -> None:
        feeds = [{"feed": n, "error": "", "items": []} for n in ("a", "b", "c")]
        table = {"news_feed": (feeds, "bitget:news_feed: ok")}
        got = probe(FakeClient(table), symbol="X", at=AT,
                    probes=_one("news-briefing", "news_feed")).results[0]
        assert got.health is Health.EMPTY

    def test_a_real_number_beside_error_legs_keeps_the_payload_substantive(self) -> None:
        curve = {"yield_curve": {"t2y": {"error": ""}, "t10y": {"value": 4.12}},
                 "spread_10y2y": 0.0}
        assert not hollow(curve)
        table = {"rates_yields": (curve, "bitget:rates_yields: ok")}
        assert probe(FakeClient(table), symbol="X", at=AT,
                     probes=_one("macro-analyst", "rates_yields")).results[0].health is Health.OK

    def test_hollow_rules(self) -> None:
        assert hollow(None) and hollow({}) and hollow([]) and hollow("")
        assert hollow({"error": "", "url": "x"})
        assert hollow({"symbol": "NVDAUSDT", "timeframe": "4h", "supports": [], "resistances": []})
        assert hollow([{"error": ""}, {"items": []}])
        assert not hollow({"rsi": 28.4})
        assert not hollow({"signal": "oversold"})
        assert not hollow([{"error": ""}, {"value": 1}])
        assert hollow(0) and hollow(False) and not hollow(1) and not hollow("x")

    def test_only_ok_counts_as_answered(self) -> None:
        assert Health.OK.answered
        for state in (Health.EMPTY, Health.TIMEOUT, Health.TOOL_ERROR, Health.UNAVAILABLE):
            assert not state.answered


class TestTheCountIsHonest:
    def test_skills_reached_counts_only_skills_that_answered(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        assert report.skills_reached == ("technical-analysis",)

    def test_reachable_is_wider_than_reached_and_never_quoted_as_integration(self) -> None:
        table = {"technical_analysis": OK_RSI,
                 "sentiment_index": ({"alt_me_error": ""}, "bitget:sentiment_index: ok"),
                 "crypto_market": (None, "bitget:crypto_market: tool reported an error (x)")}
        report = probe(FakeClient(table), symbol="NVDAUSDT", at=AT)
        assert report.skills_reached == ("technical-analysis",)
        assert set(report.skills_reachable) == {"technical-analysis", "sentiment-analyst",
                                               "market-intel"}

    def test_a_dead_endpoint_reaches_no_skills(self) -> None:
        report = probe(FakeClient({}), symbol="NVDAUSDT", at=AT)
        assert report.skills_reached == ()
        assert report.answered == ()

    def test_the_report_states_both_numbers(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        text = " ".join(report.render())
        assert f"of {len(PROBES)} official-Skill calls answered" in text
        assert f"of {len(SKILLS)} Skills reached" in text

    def test_failures_are_reported_as_absent_data_not_as_a_quiet_market(self) -> None:
        text = " ".join(probe(FakeClient({}), symbol="X", at=AT).render())
        assert "never as a quiet market" in text

    def test_every_probe_names_a_real_skill(self) -> None:
        assert all(p.skill in SKILLS for p in PROBES)

    def test_every_probe_says_what_it_yields(self) -> None:
        assert all(p.yields and p.tool and p.action for p in PROBES)

    def test_all_five_official_skills_are_covered_by_the_catalogue(self) -> None:
        assert {p.skill for p in PROBES} == set(SKILLS)

    def test_the_per_symbol_calls_carry_the_symbol(self) -> None:
        per_symbol = [p for p in PROBES if p.symbol_key]
        assert per_symbol
        for item in per_symbol:
            assert item.for_symbol("NVDAUSDT")[item.symbol_key or ""] == "NVDAUSDT"

    def test_a_call_without_a_symbol_key_is_not_given_one(self) -> None:
        item = next(p for p in PROBES if p.symbol_key is None)
        assert "symbol" not in item.for_symbol("NVDAUSDT") or item.args.get("symbol")


class TestOnlyAnswersBecomeEvidence:
    def test_an_answer_becomes_one_piece_of_evidence(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        got = evidence(report, as_of=AT)
        assert len(got) == len(report.answered)

    def test_a_dead_sweep_produces_no_evidence_at_all(self) -> None:
        """Not a piece of evidence saying "no data" — a desk would reason about that."""
        assert evidence(probe(FakeClient({}), symbol="X", at=AT), as_of=AT) == []

    def test_the_evidence_names_the_skill_it_came_from(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        assert "[technical-analysis]" in evidence(report, as_of=AT)[0].claim

    def test_the_structured_payload_is_carried_for_claim_checking(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        assert evidence(report, as_of=AT)[0].attributes["rsi"] == 28.47

    def test_a_vendor_indicator_sits_below_a_filing_and_above_a_headline(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        assert 0.7 <= evidence(report, as_of=AT)[0].credibility < 1.0

    def test_it_is_dated_at_the_call_not_earlier(self) -> None:
        """A snapshot carries no earlier stamp; back-dating it would invent a PIT guarantee."""
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        assert evidence(report, as_of=AT)[0].available_at == AT


class TestTheNumbersAreCheckedNotTrusted:
    RISING: ClassVar[list[float]] = [float(i) for i in range(1, 40)]
    FALLING: ClassVar[list[float]] = [float(i) for i in range(40, 1, -1)]

    def test_an_unbroken_rise_is_the_maximum(self) -> None:
        assert rsi(self.RISING) == 100.0

    def test_an_unbroken_fall_is_the_minimum(self) -> None:
        assert rsi(self.FALLING) == pytest.approx(0.0, abs=1e-9)

    def test_too_few_bars_is_unknown_rather_than_a_number(self) -> None:
        assert rsi([1.0, 2.0, 3.0]) is None

    def test_a_flat_series_has_no_losses_and_reads_at_the_maximum(self) -> None:
        assert rsi([100.0] * 30) == 100.0

    def test_agreement_is_reported_with_both_numbers(self) -> None:
        client = FakeClient({"technical_analysis": ({"rsi": 30.0}, "ok")})
        got = cross_check_rsi(client, symbol="NVDAUSDT", closes=self.RISING)
        assert got.theirs == 30.0 and got.ours == 100.0

    def test_a_wide_gap_is_a_disagreement(self) -> None:
        client = FakeClient({"technical_analysis": ({"rsi": 30.0}, "ok")})
        assert cross_check_rsi(client, symbol="X", closes=self.RISING).agrees is False

    def test_a_close_reading_agrees(self) -> None:
        client = FakeClient({"technical_analysis": ({"rsi": 100.0 - 1.0}, "ok")})
        assert cross_check_rsi(client, symbol="X", closes=self.RISING).agrees is True

    def test_the_boundary_is_the_named_tolerance(self) -> None:
        client = FakeClient({"technical_analysis": ({"rsi": 100.0 - RSI_AGREEMENT_POINTS}, "ok")})
        assert cross_check_rsi(client, symbol="X", closes=self.RISING).agrees is True

    def test_an_unanswered_skill_is_unknown_not_a_disagreement(self) -> None:
        """False would read as "the Skill is wrong"; it did not speak."""
        got = cross_check_rsi(FakeClient({}), symbol="X", closes=self.RISING)
        assert got.agrees is None and "did not answer" in got.note

    def test_too_few_bars_of_our_own_is_also_unknown(self) -> None:
        client = FakeClient({"technical_analysis": ({"rsi": 50.0}, "ok")})
        got = cross_check_rsi(client, symbol="X", closes=[1.0, 2.0])
        assert got.agrees is None and "too few bars" in got.note

    def test_a_non_numeric_reading_is_not_coerced(self) -> None:
        client = FakeClient({"technical_analysis": ({"rsi": "n/a"}, "ok")})
        assert cross_check_rsi(client, symbol="X", closes=self.RISING).agrees is None

    def test_a_boolean_is_not_mistaken_for_a_number(self) -> None:
        client = FakeClient({"technical_analysis": ({"rsi": True}, "ok")})
        assert cross_check_rsi(client, symbol="X", closes=self.RISING).theirs is None

    def test_the_check_serialises_for_the_record(self) -> None:
        client = FakeClient({"technical_analysis": ({"rsi": 99.0}, "ok")})
        payload = cross_check_rsi(client, symbol="X", closes=self.RISING).as_dict()
        for key in ("symbol", "skill_rsi", "our_rsi", "agrees", "note"):
            assert key in payload


class TestTheReportIsAnArtefact:
    def test_it_serialises_every_probe_not_only_the_successes(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        assert len(report.as_dict()["results"]) == len(PROBES)

    def test_it_records_the_full_health_breakdown(self) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        counts = report.as_dict()["by_health"]
        assert sum(counts.values()) == len(PROBES)

    def test_it_names_the_skills_that_exist_as_well_as_those_reached(self) -> None:
        payload = probe(FakeClient({}), symbol="X", at=AT).as_dict()
        assert payload["skills_available"] == sorted(SKILLS)
        assert payload["skills_reached"] == []

    def test_a_failed_call_carries_no_sample(self) -> None:
        rows = probe(FakeClient({}), symbol="X", at=AT).as_dict()["results"]
        assert all(r["sample"] is None for r in rows)

    def test_it_writes_to_disk_and_reads_back_as_json(self, tmp_path: Path) -> None:
        report = probe(FakeClient({"technical_analysis": OK_RSI}), symbol="NVDAUSDT", at=AT)
        path = report.save(tmp_path / "health.json")
        assert json.loads(path.read_text(encoding="utf-8"))["symbol"] == "NVDAUSDT"

    def test_an_empty_report_still_renders(self) -> None:
        assert SkillReport(symbol="X", checked_at=AT.isoformat(), results=()).render()

    def test_a_tool_health_row_says_what_the_call_was_for(self) -> None:
        row = ToolHealth(probe=PROBES[0], health=Health.OK, detail="ok",
                         payload={"rsi": 1}, checked_at=AT.isoformat())
        assert row.as_dict()["yields"] == PROBES[0].yields
