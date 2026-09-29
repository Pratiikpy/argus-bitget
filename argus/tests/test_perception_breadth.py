"""The perception-breadth comparison's own logic, offline, and the published artefact read back
(eval/perception_breadth.py; capability 19)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from argus.eval import perception_breadth as pb


def _raw(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"as_of_utc": "2026-09-28T20:44:11+00:00", "openbb_commit": "c", "per_symbol": rows,
            "installed_openbb_packages": [], "keyless_tested_providers": [],
            "excluded_keyed_providers": [], "keyless_not_tested_no_per_symbol_endpoint": []}


def _ob(endpoint: str, ticker: str = "NVDA", answered: bool = True,
        provider: str = "yfinance") -> dict[str, Any]:
    return {"provider": provider, "endpoint": endpoint, "ticker": ticker, "answered": answered,
            "rows": 3 if answered else 0, "error": None}


def _note(at: str, evidence: list[str], symbol: str = "NVDAUSDT", seq: int = 1) -> dict[str, Any]:
    return {"seq": seq, "symbol": symbol, "at": at, "evidence": evidence}


class TestCounting:
    def test_a_category_counts_once_however_many_endpoints_serve_it(self) -> None:
        raw = _raw([_ob("equity.fundamental.income"), _ob("equity.fundamental.balance"),
                    _ob("equity.compare.company_facts", provider="sec"),
                    _ob("news.company")])
        notes = [_note("2026-09-28T19:40:00+00:00", ["[mkt-NVDAUSDT] (news) x",
                                                     "[twitter-1] (social) y"])]
        report = pb.run(raw, notes)
        (row,) = report["per_symbol"]
        assert row["openbb_categories"] == ["fundamentals", "news"]
        assert row["argus_categories"] == ["price_quote", "social"]
        assert row["category_difference"] == 0 and report["summary"]["verdict"] == "mixed"

    def test_an_endpoint_that_returned_nothing_does_not_count(self) -> None:
        raw = _raw([_ob("derivatives.options.chains", answered=False), _ob("news.company")])
        notes = [_note("2026-09-28T19:40:00+00:00", ["[rss-yahoo-nvda-1] (news) z"])]
        (row,) = pb.run(raw, notes)["per_symbol"]
        assert row["openbb_categories"] == ["news"]

    def test_market_wide_and_derived_evidence_is_listed_not_scored(self) -> None:
        raw = _raw([_ob("news.company")])
        notes = [_note("2026-09-28T19:40:00+00:00", [
            "[vix-2026-09] (macro) v", "[mirror-news_feed-latest] (news) m",
            "[skill-technical_analysis-rsi] (news) r", "[feeds-NVDAUSDT] (news) f"])]
        report = pb.run(raw, notes)
        assert report["per_symbol"][0]["argus_categories"] == []
        cats = {c["category"] for c in report["cells"] if c["system"] == "argus"}
        assert cats == {"market_wide", "derived", "meta"}


class TestTheTwoSidesAreReadOnTheSameDay:
    def test_the_last_cycle_before_the_rival_run_is_used(self) -> None:
        notes = [_note("2026-09-28T13:40:00+00:00", ["[mkt-NVDAUSDT] a"], seq=1),
                 _note("2026-09-28T19:40:00+00:00", ["[mkt-NVDAUSDT] a"], seq=2),
                 _note("2026-09-28T21:40:00+00:00", ["[mkt-NVDAUSDT] a"], seq=3)]
        report = pb.run(_raw([_ob("news.company")]), notes)
        assert report["argus_cycles"]["NVDA"]["seq"] == 2

    def test_no_cycle_that_day_refuses(self) -> None:
        with pytest.raises(pb.BreadthError, match="same day"):
            pb.run(_raw([_ob("news.company")]),
                   [_note("2026-09-27T19:40:00+00:00", ["[mkt-NVDAUSDT] a"])])


class TestNothingIsSilentlyDropped:
    def test_an_unmapped_openbb_endpoint_refuses(self) -> None:
        with pytest.raises(pb.BreadthError, match="no category"):
            pb.openbb_cells(_raw([_ob("equity.something.new")]))

    def test_an_unmapped_argus_evidence_id_refuses(self) -> None:
        with pytest.raises(pb.BreadthError, match="no source mapping"):
            pb.argus_cells({"NVDAUSDT": _note("2026-09-28T19:40:00+00:00",
                                               ["[brand-new-feed-1] x"])})


def test_the_published_artefact_matches_a_rerun() -> None:
    path = pb.REPORT_PATH
    if not (path.exists() and pb.RAW_PATH.exists()):
        pytest.skip("perception_breadth.json is not on this machine")
    published = json.loads(path.read_text(encoding="utf-8"))
    notes = [json.loads(x) for x in pb.NOTES_PATH.read_text(encoding="utf-8").splitlines()
             if x.strip()]
    workbench = (json.loads(pb.WORKBENCH_PATH.read_text(encoding="utf-8"))
                 if pb.WORKBENCH_PATH.exists() else None)
    rerun = pb.run(json.loads(pb.RAW_PATH.read_text(encoding="utf-8")), notes, workbench)
    assert rerun["summary"] == published["summary"]
    assert rerun["workbench_summary"] == published["workbench_summary"]
    assert rerun["per_symbol"] == published["per_symbol"]


class TestTheWorkbenchArm:
    def test_its_cited_sources_map_to_categories(self) -> None:
        raw = {"as_of_utc": "t", "rows": [{"symbol": "NVDA", "kind": "sentiment", "sources": [
            {"ref": "cboe delayed_quotes/options", "detail": "NVDA listed chain"},
            {"ref": "finra otcMarket weeklySummary", "detail": "NVDA ATS volume"},
            {"ref": "RSS + Yahoo Finance + SEC EDGAR", "detail": "13 headline(s), 2 filing(s)"},
            {"ref": "bitget-mcp-server consensus", "detail": "scraped 2024-11-21; withheld as "
                                                            "too old"},
        ]}]}
        cats = {c.category for c in pb.workbench_cells(raw)}
        assert cats == {"options", "dark_pool", "news", "filings", "meta"}

    def test_a_news_answer_with_no_filings_is_not_a_filings_answer(self) -> None:
        raw = {"as_of_utc": "t", "rows": [{"symbol": "NVDA", "kind": "news", "sources": [
            {"ref": "bitget + RSS + SEC EDGAR", "detail": "13 headline(s), 0 filing(s)"}]}]}
        assert {c.category for c in pb.workbench_cells(raw)} == {"news"}

    def test_an_unmapped_workbench_source_refuses(self) -> None:
        raw = {"as_of_utc": "t", "rows": [{"symbol": "NVDA", "kind": "quote", "sources": [
            {"ref": "a brand new feed", "detail": ""}]}]}
        with pytest.raises(pb.BreadthError, match="no mapping"):
            pb.workbench_cells(raw)


def test_the_runner_names_no_machine_path() -> None:
    runner = Path(__file__).resolve().parents[1] / "scripts" / "pit_runners" / \
        "openbb_breadth_runner.py"
    assert "Users" not in runner.read_text(encoding="utf-8")
