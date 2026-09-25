"""The trader's own trades: parsed, paired, checked and reviewed, offline (no network, no model)."""
# ruff: noqa: RUF001 - Chinese test inputs carry full-width punctuation on purpose

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from argus.lui import journal
from argus.lui.journal import (
    Leg,
    pair,
    parse_fills,
    parse_text,
    reaction_day,
    read_journal,
    review_trades,
)
from argus.lui.provenance import label

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)
DATA = Path(__file__).resolve().parents[1] / "data"

SHOULD_REACH = [
    "bought NVDA at 188 sold at 176, bought COIN at 330 sold at 301 ... what bad patterns do I "
    "have and give me a checklist",
    "bought NVDA at 188 sold at 176, bought COIN at 330 sold at 301",
    "long NVDA from 188 to 176, short TSLA 250 -> 270",
    "bought 10 NVDA at 188 on 2026-08-20, held it over earnings, sold at 176 on 2026-08-28. "
    "what did I do wrong?",
    "shorted TSLA at 250, covered at 270; bought AAPL at 220 and sold at 231",
    "bought MSTR @ 350 on Sep 4, stopped out at 330 on Sep 8. bought COIN at 300, sold at 320",
    "went long QQQ at 560, exited at 548, then bought NVDA at 170 and sold at 181",
    "bought 0.5 btc at 110000 and sold at 104000, bought eth at 4200 sold at 3900",
    "FOMO bought NVDA at 195 after the pop, sold at 181. what mistakes am I making?",
    "Here are my trades: bought AMZN at 225 sold at 219; bought GOOGL at 250 sold at 262; "
    "bought META at 760 sold at 721. review please",
    "我在188买了英伟达，176卖了；做空特斯拉250，270止损。帮我复盘一下",
    "9月1日在330买入COIN，9月5日301卖出，8月3日做多比特币 60000，8月9日平仓 58000",
    "188买入英伟达，176卖出，330买入COIN，301卖出，我有什么坏习惯？",
    "做多苹果 220到231，做空特斯拉 250到270",
    "上周在180买的NVDA，拿过了财报，172卖了，帮我总结一下问题出在哪",
    "symbol,side,tradeSide,execPrice,execQty,createdTime\n"
    "NVDAUSDT,buy,open,188,2,1755680400000\nNVDAUSDT,sell,close,176,2,1756371600000",
]

MUST_NOT_REACH = [
    "I bought NVDA at 188, should I sell now?",
    "just bought 0.1 btc more on top of what i had, is my account too risky now",
    "if I buy NVDA at 180 and sell at 200 what's my profit",
    "should I sell my TSLA at 250 or hold",
    "show me the desk's past trades and how they performed",
    "what's the max drawdown on the desk's own trades",
    "我想卖出全部amzn持仓(约80股),怎么执行比较好",
    "卖出30万股MSTR，建议用TWAP还是VWAP，时间窗口多长合适？",
    "I sold my NVDA at 176 yesterday, where is it now?",
    "bought NVDA at 188, set a stop at 176 — is that too tight?",
    "what did the desk buy today",
    "did you sell TSLA?",
    "NVDA went from 188 to 176 this week, why?",
    "if NVDA drops from 188 to 176 what happens to my book",
    "帮我直接下单卖出所有TSLA持仓",
    "I'm long NVDA from 188, where should my stop be?",
    "my stop got hit on NVDA, is it a good time to rebuy?",
    "I bought NVDA at 188 and sold at 176, where is it now?",
]


@pytest.mark.parametrize("text", SHOULD_REACH)
def test_reaches(text: str) -> None:
    assert read_journal(text, now=NOW.date()) is not None, text


@pytest.mark.parametrize("text", MUST_NOT_REACH)
def test_does_not_reach(text: str) -> None:
    assert read_journal(text, now=NOW.date()) is None, text
    assert review_trades(text, now=NOW, history=None) is None, text


def test_no_held_out_question_is_captured() -> None:
    """None of the 680 held-out questions is a journal, and none may be read as one."""
    captured: list[str] = []
    for path in sorted(DATA.glob("lui_*_2026-09-25.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            if review_trades(text, now=NOW, history=None) is not None:
                captured.append(text)
    assert captured == []


def test_the_audit_question_reads_two_losing_longs() -> None:
    legs, _ = parse_text(SHOULD_REACH[0], now=NOW.date())
    trades, still_open, dropped = pair(legs)
    assert [(t.symbol, t.side, t.entry, t.exit) for t in trades] == [
        ("NVDAUSDT", "long", 188.0, 176.0), ("COINUSDT", "long", 330.0, 301.0)]
    assert still_open == [] and dropped == []
    assert round(trades[0].return_pct, 2) == -6.38
    assert round(trades[1].return_pct, 2) == -8.79


def test_chinese_short_and_stop() -> None:
    legs, _ = parse_text(SHOULD_REACH[10], now=NOW.date())
    trades, _, _ = pair(legs)
    tsla = next(t for t in trades if t.symbol == "TSLAUSDT")
    assert tsla.side == "short" and tsla.entry == 250.0 and tsla.exit == 270.0
    assert tsla.return_pct == pytest.approx(-8.0)
    assert "stopped" in tsla.flags


def test_dates_sizes_and_flags() -> None:
    legs, _ = parse_text(SHOULD_REACH[3], now=NOW.date())
    trades, _, _ = pair(legs)
    assert len(trades) == 1
    t = trades[0]
    assert t.opened == date(2026, 8, 20) and t.closed == date(2026, 8, 28)
    assert t.qty == 10.0 and "earnings" in t.flags


def test_year_less_dates_are_the_most_recent_past() -> None:
    legs, notes = parse_text("bought NVDA at 188 on Dec 3, sold at 190 on Dec 5. review",
                             now=NOW.date())
    assert legs[0].day == date(2025, 12, 3)
    assert any("without a year" in n for n in notes)


def test_fills_fifo_partial_and_close_side() -> None:
    table = ("symbol,side,tradeSide,execPrice,execQty,createdTime,feeDetail,execPnl\n"
             "NVDAUSDT,buy,open,100,2,1755680400000,\"[{\"\"feeCoin\"\":\"\"USDT\"\",\"\"fee\"\":"
             "\"\"-0.12\"\"}]\",\n"
             "NVDAUSDT,buy,open,90,2,1755766800000,,\n"
             "NVDAUSDT,buy,close,110,3,1756371600000,,15\n")
    parsed = parse_fills(table)
    assert parsed is not None
    legs, notes = parsed
    assert "execPrice" in notes[0]
    trades, still_open, _ = pair(legs)
    # hedge mode: a "close" row reduces the open long whatever its side says
    assert len(trades) == 1 and trades[0].side == "long"
    assert trades[0].qty == pytest.approx(3.0)
    assert trades[0].entry == pytest.approx((100 * 2 + 90 * 1) / 3)
    assert trades[0].averaged_down
    assert trades[0].fees == pytest.approx(0.12)
    assert still_open == ["NVDA long from 90"]


def test_reaction_day_places_releases_around_the_close() -> None:
    holidays = frozenset({date(2026, 9, 7)})
    after_close = datetime(2026, 8, 26, 20, 21, tzinfo=UTC)  # 16:21 New York
    assert reaction_day(after_close, holidays) == date(2026, 8, 27)
    pre_market = datetime(2026, 8, 26, 11, 0, tzinfo=UTC)  # 07:00 New York
    assert reaction_day(pre_market, holidays) == date(2026, 8, 26)
    friday_after = datetime(2026, 9, 4, 20, 30, tzinfo=UTC)  # Labor Day Monday follows
    assert reaction_day(friday_after, holidays) == date(2026, 9, 8)


def _history(symbol: str) -> tuple[list[tuple[date, float]], str]:
    start = date(2025, 6, 2)
    closes: list[tuple[date, float]] = []
    price = 100.0
    k = 0
    d = start
    while d < date(2026, 9, 25):
        if d.weekday() < 5:
            k += 1
            price *= 1.0 + (0.02 if k % 2 else -0.019)
            if d == date(2026, 8, 19):
                price *= 1.10  # a top-decile day the session before an entry
            closes.append((d, price))
        d += timedelta(days=1)
    return closes, f"{symbol} synthetic closes"


def _releases(symbol: str, since: date) -> tuple[list[datetime], str]:
    return [datetime(2026, 8, 26, 20, 21, tzinfo=UTC)], "synthetic 8-K"


def test_review_computes_checks_and_leads_with_a_counted_habit() -> None:
    text = ("bought NVDA at 180 on 2026-08-20, sold at 170 on 2026-08-28; "
            "bought NVDA at 175 on 2026-09-01, sold at 177 on 2026-09-02; "
            "bought NVDA at 172 on 2026-09-03, sold at 150 on 2026-09-09. what patterns do I have?")
    out = review_trades(text, now=NOW, history=_history, releases=_releases)
    assert out is not None
    lines, sources, data = out
    assert lines[0].startswith("Actionable: the habit most worth fixing is ")
    assert " of " in lines[0]
    assert lines[1].startswith("Too few trades to call any of this a statistic")
    first = data["journal"]["trades"][0]
    assert first["earnings"] is True and first["earnings_day"] == "2026-08-27"
    assert first["closures"] == 2 and first["chased"] is True
    keys = {p["key"] for p in data["journal"]["patterns"]}
    assert {"earnings_losers", "closure_losers", "chasing", "losers_bigger"} <= keys
    assert any(line.startswith("Checklist") for line in lines)
    assert data["journal"]["anecdotal"] is True
    assert lines[-1].startswith("Data: ")
    assert any(s.ref == "argus.desk.odds:directional_odds" for s in sources)


def test_labels_lead_assumed_missing() -> None:
    out = review_trades(SHOULD_REACH[0], now=NOW, history=None)
    assert out is not None
    lines = out[0]
    assert label(lines[0]) == "computed"
    assert all(label(x) == "assumed" for x in lines if x.startswith("Assumed:"))
    assert all(label(x) == "missing" for x in lines if x.startswith("Missing:"))
    assert label(lines[-1]) is None  # the Data: line is meta


def test_review_request_without_trades_asks_for_them() -> None:
    out = review_trades("can you review my trades and tell me my bad habits?", now=NOW,
                        history=None)
    assert out is not None
    assert out[0][0].startswith("Missing: no trades were found")
    assert review_trades("复盘一下我的交易", now=NOW, history=None) is not None
    assert review_trades("复盘一下今天的市场", now=NOW, history=None) is None


def test_gate_floor_matches_the_regression_gate() -> None:
    from argus.eval.regression_gate import MIN_JOURNAL_TRADES

    assert journal.GATE_MIN_TRADES == MIN_JOURNAL_TRADES


def test_an_unpriced_leg_is_reported_not_guessed() -> None:
    trades, _, dropped = pair([Leg("buy", "NVDAUSDT", None, text="bought NVDA"),
                               Leg("sell", "NVDAUSDT", 176.0, text="sold at 176")])
    assert trades == []
    assert any("carries no price" in d for d in dropped)
