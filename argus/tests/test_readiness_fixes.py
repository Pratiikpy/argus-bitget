"""Fixes from the 2026-09-25 readiness audit of the hosted console, offline."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from argus.lui.answer import Answer
from argus.lui.question import classify
from argus.lui.research import ResearchKind, ResearchRequest, with_book
from argus.lui.research.dispatch import _honest_data_line
from argus.lui.research.parse import _idea_request
from argus.truth.coverage import Record


def _stress(book: dict[str, float], shock: float | None) -> ResearchRequest:
    return ResearchRequest(kind=ResearchKind.STRESS, symbols=tuple(book), book=book,
                           shock_pct=shock)


def test_an_ideas_size_is_not_a_market_shock() -> None:
    got = _idea_request("stress test my idea: long NVDA 20% of book into earnings",
                        _stress({"NVDAUSDT": 0.2}, 20.0))
    assert got.kind is ResearchKind.IMPACT and got.symbols[0] == "NVDAUSDT"
    assert got.size == 0.2 and got.shock_pct is None and not got.book


def test_the_idea_joins_the_saved_book() -> None:
    got = with_book(_stress({"NVDAUSDT": 0.2}, 20.0), "40% MSFT, 30% META, 30% GOOGL",
                    "go long 20% of my book in NVDA and stress test it")
    assert got is not None and got.kind is ResearchKind.IMPACT
    assert got.symbols[0] == "NVDAUSDT" and set(got.book) == {"MSFTUSDT", "METAUSDT",
                                                               "GOOGLUSDT"}


def test_a_stated_market_shock_stays_a_stress() -> None:
    request = _stress({"NVDAUSDT": 0.6, "AAPLUSDT": 0.4}, -10.0)
    assert _idea_request("what if the nasdaq drops 10%? I hold 60% NVDA 40% AAPL",
                         request) is request


def _answer(lines: list[str]) -> Answer:
    question = classify("when does NVDA report earnings", now=datetime(2026, 9, 25, tzinfo=UTC))
    return Answer(question=question, lines=lines)


def test_a_server_that_did_not_answer_is_not_credited() -> None:
    record = Record()
    for tool in ("equity_calendar", "equity_price_quote"):
        record.note(f"bitget-mcp-server {tool}", False, "upstream 5xx")
    record.note("SEC EDGAR", True)
    answer = _answer(["Next report: 17 Nov (Yahoo Finance's earnings calendar).",
                      "Data: Bitget's bitget-mcp-server (US equity data) and the company's SEC "
                      "filings, live. This is analysis, not advice — you make the call."])
    _honest_data_line(answer, record)
    data = answer.lines[-1]
    assert data.startswith("Data: SEC EDGAR, Yahoo Finance, live;")
    assert "bitget-mcp-server did not answer this time" in data
    assert data.endswith("you make the call.")
    line = record.line() or ""
    assert "bitget-mcp-server (2 tools: upstream 5xx)" in line and "without it." in line


def test_a_server_that_answered_keeps_its_credit() -> None:
    record = Record()
    record.note("bitget-mcp-server equity_calendar", True)
    record.note("bitget-mcp-server equity_price_quote", False, "upstream 5xx")
    before = "Data: Bitget's bitget-mcp-server (US equity data), live."
    answer = _answer(["x", before])
    _honest_data_line(answer, record)
    assert answer.lines[-1] == before


def test_an_as_of_answer_keeps_nothing_it_could_not_have_known() -> None:
    from argus.lui.research.dispatch import _point_in_time

    answer = _answer([
        "Bottom line: NVDA's net income for the quarter ending 26 Oct 2025 was $31.91bn.",
        "NVDA reports in 53 day(s), on 17 Nov — no earnings gap inside 7 days.",
        "Next report: 2026-11-17 — 53 day(s) away.",
        "Analyst price targets (59 analysts, Yahoo Finance): mean $327.70.",
        "Earnings surprise: the quarter ending 2026-07-26 was a large beat (SEC filing, filed "
        "2026-08-26).",
        "Earnings surprise: the quarter ending 2025-10-26 was a beat (SEC filing, filed "
        "2025-11-19).",
        "Data: SEC XBRL, live."])
    _point_in_time(answer, datetime(2026, 3, 1, 23, 59, tzinfo=UTC))
    text = "\n".join(answer.lines)
    assert "53 day" not in text and "Analyst price targets" not in text
    assert "filed 2026-08-26" not in text and "filed 2025-11-19" in text
    assert any(line.startswith("Point in time: 4 line(s)") for line in answer.lines)
    assert answer.lines[-1].startswith("Data:")


@pytest.mark.network
def test_a_hedge_leg_without_history_does_not_sink_the_answer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """With SMH's history unreadable, the hedge plan answers with the legs it can measure
    instead of raising out of the console (trace audit, 2026-09-26). The legs it does measure
    are priced on live order books, so this reaches the venue."""
    from decimal import Decimal

    from argus.desk.portfolio import PortfolioError
    from argus.lui import research

    real = research.load

    def load(symbols: Sequence[str], **kwargs: Any) -> Any:
        if "SMHUSDT" in symbols:
            raise PortfolioError("SMHUSDT is not in the frozen history either")
        return real(symbols, **kwargs)

    for _module in (research.book, research.data, research.dispatch, research.news):
        monkeypatch.setattr(_module, "load", load)
    lines, _sources, _data = research.book._hedge_plan({"AAPLUSDT": 1.0}, Decimal(10_000),
                                                  "best way to hedge a long AAPL book")
    assert lines and not any("SMH" in line and "beta" in line for line in lines)


def test_a_ticker_outage_does_not_raise_out_of_the_hedge_plan(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The bulk ticker call sat outside the per-leg guard, so one 429 from Bitget raised out of
    every hedge question (found running the suite with the network blocked, 2026-09-27). Now each
    leg is reported as not answering and the caller refuses honestly."""
    from decimal import Decimal
    from types import SimpleNamespace

    from argus.lui import research
    from argus.market import bitget, depth

    hours = range(200)
    wiggle = [0.001 * ((h * 7) % 11 - 5) for h in hours]
    series = {h: w for h, w in zip(hours, wiggle, strict=True)}

    def load(symbols: Sequence[str], **kwargs: Any) -> Any:
        return SimpleNamespace(raw={s: dict(series) for s in symbols})

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise bitget.BitgetError("transport failure for /api/v2/mix/market/tickers: 429")

    def no_book(*args: Any, **kwargs: Any) -> Any:
        raise depth.DepthError("no book")

    for _module in (research.book, research.data, research.dispatch, research.news):
        monkeypatch.setattr(_module, "load", load)
    for _module in (research.book, research.dispatch, research.execution, research.news,
                    research.parse, research.quote, research.sentiment, research.venue):
        monkeypatch.setattr(_module, "is_us_equity", lambda symbol: True)  # reads the live list
    monkeypatch.setattr(bitget, "fetch_tickers", refuse)
    monkeypatch.setattr(depth, "fetch_orderbook", no_book)
    assert research.book._hedge_plan({"AAPLUSDT": 1.0}, Decimal(10_000),
                                "best way to hedge a long AAPL book") == ([], [], {})


@pytest.mark.parametrize(("module", "attr", "by"), [
    ("argus.lui.journal", "review_trades", "journal"),
    ("argus.lui.watchlist", "asks_for_watchlist", "watchlist"),
    ("argus.lui.exposures", "answer", "exposures")])
def test_the_trader_book_questions_reach_their_engines(
        monkeypatch: pytest.MonkeyPatch, module: str, attr: str, by: str) -> None:
    """The console consults the journal, the watchlist and the exposures readers before any
    research reader (readiness audit, 2026-09-25: all three questions were refused)."""
    import importlib

    from argus.lui import server

    target = importlib.import_module(module)
    if by == "journal":
        monkeypatch.setattr(target, attr, lambda text, now=None: (["Bottom line: journal"], [], {}))
    elif by == "watchlist":
        monkeypatch.setattr(target, attr, lambda text: True)
        monkeypatch.setattr(target, "watchlist",
                            lambda text, book="", now=None: (["Bottom line: watch"], [], {}))
    else:
        monkeypatch.setattr(target, attr, lambda text, book="": Answer(
            question=classify(text, now=datetime(2026, 9, 25, tzinfo=UTC)),
            lines=["Bottom line: exposures"]))
    payload = server.handle_ask("what should I look at for my book", [], visitor="t")
    assert payload["classified_by"] == by
