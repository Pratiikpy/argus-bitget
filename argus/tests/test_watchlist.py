"""The weekly watchlist: routing, the official-schedule parsers, the book, the ranking, closed days.

Offline: every live source (earnings calendar, EDGAR) is injected, the calendar and the event study
are written to a temporary directory, and no model is called.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from argus.lui import watchlist as wl
from argus.lui.provenance import label

DATA = Path(__file__).resolve().parents[1] / "data"
NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)  # a Saturday

ASKS = [
    "what should I watch this week?",
    "what earnings and macro events should I watch this week for my book?",
    "what's on the calendar this week for my portfolio",
    "any catalysts coming up for my holdings this week",
    "upcoming events for my book",
    "what macro events are coming up next week",
    "economic calendar",
    "macro calendar for the next 10 days",
    "which of my names report earnings this week",
    "earnings this week for NVDA, AAPL and MSFT",
    "week ahead for my book",
    "key dates to watch for NVDA and TSLA over the next 10 days",
    "give me a watchlist for this week",
    "what market-moving events are scheduled this week",
    "anything I should keep an eye on next week for 40% NVDA 60% BTC",
    "这周我的持仓要关注什么",
    "本周有哪些财报和宏观数据要关注",
    "下周的经济日历",
    "未来10天我的组合有什么重要事件",
]
NEAR_MISSES = [
    "what movie should I watch tonight",
    "what should I watch on netflix this week",
    "when does coin report earnings",
    "does nvda report earnings next week",
    "will bitcoin go crazy when the fed meets next week",
    "what's driving the macro narrative this week",
    "what's the cheapest way to hedge my spy exposure this week",
    "what decisions has the desk made this week",
    "where will bitcoin be next week",
    "how does qqq typically behave the week of cpi",
    "protective hedge for meta going into cpi week",
    "gimme a quick read on crowd sentiment for tech stocks this week",
    "where's the key support level on QQQ into next week",
    "did you guys actually trade anything this week",
    "what's the latest CPI print",
    "when is the next FOMC meeting",
    "watch out for liquidation on my 20x BTC long",
    "英伟达在CPI公布当天通常怎么波动",
    "这周比特币会涨吗",
]


@pytest.mark.parametrize("text", ASKS)
def test_watchlist_questions_route_here(text: str) -> None:
    assert wl.asks_for_watchlist(text), text


@pytest.mark.parametrize("text", NEAR_MISSES)
def test_near_misses_do_not(text: str) -> None:
    assert not wl.asks_for_watchlist(text), text


def test_none_of_the_held_out_corpora_is_captured() -> None:
    captured = []
    for path in sorted(DATA.glob("lui_*_2026-09-25.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                row = json.loads(raw)
                if wl.asks_for_watchlist(row["text"]):
                    captured.append((path.name, row["text"], row.get("expected")))
    assert captured == []


def test_bls_rows_parse_from_the_reader_markdown_and_from_html() -> None:
    markdown = ("| September 2026 | Oct. 02, 2026 | 08:30 AM |\n"
                "| October 2026 | Nov. 06, 2026 | 08:30 AM |")
    rows = wl.parse_bls(markdown, "NFP")
    assert [(r["date"], r["time_et"]) for r in rows] == [("2026-10-02", "08:30"),
                                                         ("2026-11-06", "08:30")]
    assert rows[0]["title"].endswith("September 2026")
    html = "<tr><td>August 2026</td><td>Sep. 10, 2026</td><td>08:30 AM</td></tr>"
    assert wl.parse_bls(html, "PPI")[0]["date"] == "2026-09-10"


def test_bea_census_and_fed_parsers() -> None:
    bea = ("<h2>Year 2026</h2><table><tr><td>September 30</td><td>8:30 AM</td><td>News</td>"
           "<td>Personal Income and Outlays, August 2026</td></tr><tr><td>October 6</td>"
           "<td>8:30 AM</td><td>News</td><td>U.S. International Trade, August 2026</td></tr>"
           "<tr><td>October 29</td><td>8:30 AM</td><td>News</td><td>GDP (Advance Estimate), 3rd "
           "Quarter 2026</td></tr></table>")
    rows = wl.parse_bea(bea)
    assert [(r["kind"], r["date"]) for r in rows] == [("PCE", "2026-09-30"), ("GDP", "2026-10-29")]
    census = ("Advance Monthly Retail Trade Report | Data Month | Release Date at 8:30 am | "
              "September 2026 | | October 15, 2026 | Monthly Retail Trade Report | August 2026 | "
              "October 15, 2026")
    assert [r["date"] for r in wl.parse_census(census)] == ["2026-10-15"]
    fed = json.dumps({"events": [
        {"title": "FOMC Meeting", "type": "FOMC", "month": "2026-10", "days": "28",
         "time": "2:00 p.m.", "description": "&lt;p&gt;Two-day meeting, October 27 - 28&lt;/p&gt;"},
        {"title": " FOMC Minutes", "type": "FOMC", "month": "2026-10", "days": "7",
         "time": "2:00 p.m.", "description": "&lt;p&gt;Meeting of September 15-16&lt;/p&gt;"},
        {"title": "FOMC Press Conference", "type": "FOMC", "month": "2026-10", "days": "28"},
        {"title": "Speech", "type": "Speeches", "month": "2026-10", "days": "1"}]})
    fed_rows = wl.parse_fed(fed)
    assert [(r["kind"], r["date"], r["time_et"]) for r in fed_rows] == [
        ("MINUTES", "2026-10-07", "14:00"), ("FOMC", "2026-10-28", "14:00")]


def test_the_frozen_calendar_names_its_official_sources() -> None:
    calendar = wl.load_calendar()
    assert calendar is not None
    kinds = {r["kind"] for r in calendar["releases"]}
    assert kinds == set(wl.KIND_NAMES)
    for kind, meta in calendar["sources"].items():
        assert meta["url"].startswith(("https://www.bls.gov", "https://www.bea.gov",
                                       "https://www.census.gov", "https://www.federalreserve.gov"))
        assert meta.get("rows", 0) > 0 or "note" in meta, kind
    assert all(r["source"] for r in calendar["releases"])


def test_window_rules() -> None:
    start, end, stated = wl.window("what should I watch this week", NOW)
    assert (start, (end - start).days, stated) == (NOW, 7, False)
    start, end, stated = wl.window("next week's calendar", NOW)
    assert start.astimezone(wl.NEW_YORK).date() == date(2026, 9, 28) and stated
    assert (end - start).days == 7
    assert (wl.window("macro events over the next 10 days", NOW)[1] - NOW).days == 10


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    calendar = tmp_path / "cal.json"
    calendar.write_text(json.dumps({"retrieved_at": "2026-09-25T20:00:00+00:00", "sources": {},
                                    "releases": [
        {"kind": "PCE", "date": "2026-09-30", "time_et": "08:30",
         "title": "Personal Income and Outlays, August 2026", "source": wl.BEA_URL},
        {"kind": "CPI", "date": "2026-10-01", "time_et": "08:30", "title": "CPI, September 2026",
         "source": wl.BLS_PAGES["CPI"]},
        {"kind": "NFP", "date": "2026-10-02", "time_et": "08:30",
         "title": "nonfarm payrolls (Employment Situation), September 2026",
         "source": wl.BLS_PAGES["NFP"]},
        {"kind": "CPI", "date": "2026-10-14", "time_et": "08:30", "title": "CPI, later",
         "source": wl.BLS_PAGES["CPI"]}]}), encoding="utf-8")
    reactions = tmp_path / "event_reactions.json"
    reactions.write_text(json.dumps({"generated_at": "2026-09-25T13:00:00+00:00", "reactions": [
        {"symbol": "NVDAUSDT", "kind": "CPI", "events": 11, "average_car_bps": 34.6,
         "size_ratio": 1.3, "verdict": "NO EFFECT ESTABLISHED", "dates": ["2025-10-24"]},
        {"symbol": "TSLAUSDT", "kind": "CPI", "events": 11, "average_car_bps": 10.0,
         "size_ratio": 1.2, "verdict": "NO EFFECT ESTABLISHED", "dates": ["2025-10-24"]},
        {"symbol": "TSLAUSDT", "kind": "earnings", "events": 8, "average_car_bps": -120.0,
         "size_ratio": 3.22, "verdict": "PARTIAL", "dates": ["2025-10-22"]}]}), encoding="utf-8")
    return calendar, reactions


def _earnings(ticker: str, today: date) -> wl.Report | None:
    return {"TSLA": wl.Report("TSLA", date(2026, 9, 30), "after the close", "test calendar"),
            "NVDA": wl.Report("NVDA", date(2026, 11, 18), "", "test calendar")}.get(ticker)


def _filings(tickers: object, since: datetime) -> dict[str, list[wl.Filed]]:
    return {"NVDA": [wl.Filed("NVDA", date(2026, 9, 24), "8.01 Other Events")]}


def test_a_measured_earnings_event_leads_the_book(tmp_path: Path) -> None:
    calendar, reactions = _setup(tmp_path)
    lines, sources, data = wl.watchlist(
        "what should I watch this week?", "60% NVDA, 30% TSLA, 10% BTC", now=NOW,
        calendar_path=calendar, reactions_path=reactions, earnings_lookup=_earnings,
        filings_lookup=_filings)
    assert lines[0].startswith("Actionable: the biggest event for this book")
    assert "TSLA reports" in lines[0]  # 0.30 x (3.22 - 1) = 0.67 beats CPI's 0.6x0.3+0.3x0.2
    assert data["ranked"][:2] == ["TSLA earnings", "CPI, September 2026"]
    text = "\n".join(lines)
    assert "Remembered: your saved book" in text
    assert "Sat 26 Sep to Sun 27 Sep (weekend)" in text
    assert "Filed: NVDA 8-K on Thu 24 Sep" in text
    assert "Next report: NVDA Wed 18 Nov" in text
    assert "CPI, later" not in text  # outside the window
    assert "payrolls" in text and "was not measured" in text
    assert "BTC" in text and sources
    for line in lines:
        if line.startswith(("Scheduled:", "Assumed:", "Remembered:", "Filed:")):
            assert label(line) is not None, line


def test_unmeasured_week_leads_by_the_stated_convention(tmp_path: Path) -> None:
    calendar, reactions = _setup(tmp_path)
    lines, _, data = wl.watchlist(
        "what macro events should I watch this week", "", now=NOW, days=6,
        calendar_path=calendar, reactions_path=reactions, earnings_lookup=_earnings,
        filings_lookup=_filings)
    assert "CPI" in lines[0] and "stated convention" in lines[0]
    assert any(line.startswith("Assumed: no holdings were named") for line in lines)
    assert data["book"] == {}


def test_names_asked_about_are_equal_weight_and_said_so(tmp_path: Path) -> None:
    calendar, reactions = _setup(tmp_path)
    lines, _, data = wl.watchlist(
        "key dates to watch for NVDA and TSLA over the next 10 days", "90% AAPL", now=NOW,
        calendar_path=calendar, reactions_path=reactions, earnings_lookup=_earnings,
        filings_lookup=_filings)
    assert set(data["book"]) == {"NVDAUSDT", "TSLAUSDT"}
    assert any("read at 50% each" in line for line in lines)
    assert "CPI, later" not in "\n".join(lines)  # 26 Sep + 10 days ends 6 Oct, before 14 Oct


def test_missing_calendar_is_said_not_estimated(tmp_path: Path) -> None:
    lines, _, data = wl.watchlist(
        "what should I watch this week", "", now=NOW, calendar_path=tmp_path / "absent.json",
        reactions_path=tmp_path / "absent2.json", earnings_lookup=_earnings,
        filings_lookup=_filings)
    assert any("could not be read" in line for line in lines)
    assert data["events"] == []
    assert lines[0].startswith("Actionable: nothing on the official US macro calendar")
