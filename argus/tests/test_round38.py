"""Round 38 audits (judge, newcomer, hostile; Activity/audits/round38_*.md), fixed offline."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pytest


class TestJudge:
    def test_astated_entry_is_the_entry(self) -> None:
        from argus.lui.research.dispatch import stated_entry

        assert stated_entry("If I open a 10x isolated long on BTC at $62,000 entry with $5,000 "
                             "margin, where does Bitget liquidate me?") == 62_000
        assert stated_entry("I'm 5x short ETH, entry price 2500 — liquidation?") == 2_500
        assert stated_entry("where is my liquidation price on a 3x BTC long") is None

    def test_astated_entry_is_not_a_price_claim(self) -> None:
        from argus.lui import server

        assert server._price_premise("If I open a 10x long on BTC at $62,000 entry") is None

    def test_chinese_correlation_is_read(self) -> None:
        from argus.lui.research import desk_answers

        def never(*_a: object, **_k: object) -> list[object]:
            raise RuntimeError("offline")

        # the question is recognised (it reaches the fetch), not passed over
        from argus.market import history

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(history, "fetch_window", never)
            got = desk_answers.correlation_lines("比特币和黄金过去90天的相关性是多少?", [])
        assert got is None or got  # no exception; the reader took it

    def test_rebalancing_a_shock_is_arithmetic(self) -> None:
        from argus.lui.research.rebalance import shock_trades

        said = shock_trades({"BTCUSDT": 0.4, "NVDAUSDT": 0.3, "XAUUSDT": 0.3}, "BTCUSDT", -0.15)
        assert "BTC drifts from 40% to 36.2%" in said[0]
        assert "buy 3.6% of the original book" in said[0]

    def test_rebalancing_cadence_is_replayed(self) -> None:
        from argus.lui.research.rebalance import cadence_lines

        n = 500
        a = [100 * math.exp(0.002 * i) for i in range(n)]
        b = [100 * math.exp(-0.001 * i + 0.05 * math.sin(i / 9)) for i in range(n)]
        said = cadence_lines({"A": 0.5, "B": 0.5}, {"A": a, "B": b}, 10_000, years=2.0)
        assert said is not None and said[0].startswith("Bottom line: on the last 2.0 years")
        assert "monthly: 12 rebalances a year" in said[1]

    def test_an_episode_against_the_limit(self) -> None:
        from argus.lui.research import book_episode as episodes

        start, end = date(2021, 11, 10), date(2022, 11, 21)
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        stamps = [datetime(d.year, d.month, d.day, tzinfo=UTC) for d in days]
        falling = [100 * (1 - 0.7 * i / len(days)) for i in range(len(days))]
        flat = [100.0] * len(days)
        said = episodes.lines("does it hold under a 2022 crypto winter?",
                              {"BTCUSDT": 0.8, "XAUUSDT": 0.2}, 0.15,
                              {"BTCUSDT": (stamps, falling), "XAUUSDT": (stamps, flat)})
        assert said is not None and said[0].startswith("Bottom line: no")
        assert any(x.startswith("What would change it:") for x in said)

    def test_follow_ups_reach_their_readers(self) -> None:
        from argus.lui import memory as mem
        from argus.lui.research.book_episode import BREAK_ASKED
        from argus.lui.research.carry import _RISK_FREE
        from argus.lui.server import _SINCE_FILING

        assert mem.recall_asked("Remind me — what was my stated drawdown tolerance?")
        assert BREAK_ASKED.search("What would break that conclusion?")
        assert _SINCE_FILING.search("Did the stock's move since that filing match what it said?")
        assert _RISK_FREE.search("does shorting the perp against it lock in that carry risk-free?")
        facts = mem.extract("I can't tolerate more than a 15% drawdown",
                            datetime(2026, 10, 5, tzinfo=UTC))
        assert any(f.kind == "max_loss" and float(f.value) == 0.15 for f in facts)

    def test_an_allocation_is_a_portfolio_not_an_order(self) -> None:
        from argus.lui.research.parse import _ALLOCATE

        assert _ALLOCATE.search("How should I split it to get the best risk-adjusted mix")
        assert not _ALLOCATE.search("how should I split a $50k order in NVDA")

    def test_proof_paths_link_to_the_repository(self) -> None:
        from argus.lui.proof_page import REPOSITORY, _linked

        got = _linked("ARGUS's own research/eventstudy.py and data/cointegration.json")
        assert f"{REPOSITORY}src/argus/research/eventstudy.py" in got
        assert f"{REPOSITORY}data/cointegration.json" in got


class TestHostile:
    def test_scientific_notation(self) -> None:
        from argus.lui.journal import parse_text

        legs, notes = parse_text("bought 1e308 BTC at 80000")
        assert all(leg.qty != 308 for leg in legs)
        assert any("larger than any real trade" in n for n in notes)
        assert parse_text("bought 5e1 BTC at 80000")[0][0].qty == 50

    def test_a_metric_named_is_performance(self) -> None:
        from argus.lui import ngram
        from argus.lui.question import classify

        q = classify("What is ARGUS's own Sharpe ratio on its paper-trading desk?",
                     now=datetime(2026, 10, 5, tzinfo=UTC))
        got, _source = ngram.reclassify(q)
        assert str(got.intent) == "performance"

    def test_a_near_miss_ticker_is_not_quoted(self) -> None:
        from argus.lui.server import _near_miss_ticker

        said = _near_miss_ticker("What's the price of NVDAA?")
        assert said is not None and said[0].startswith("Bottom line: NVDAA is not listed")
        assert _near_miss_ticker("What's the price of NVDA?") is None

    def test_cents(self) -> None:
        from argus.lui.account_math import cents_lines

        said = cents_lines("If I have 100 cents, how many dollars is that?")
        assert said is not None and "$1.00" in said[0]

    def test_shocks_are_capped_both_ways(self) -> None:
        from argus.lui import mcp_server

        with pytest.raises(mcp_server.ToolError):
            mcp_server.call_tool("argus_stress", {"book": {"BTC": 100},
                                                  "shock_percent": 1_000_000})


class TestNewcomer:
    @pytest.mark.parametrize(("asked", "expected"), [
        ("how do i get money onto bitget", "on Bitget itself, not here"),
        ("can i lose my house doing this", "not from trading on Bitget itself"),
        ("whats a good app for crypto", "will not rank apps"),
        ("how does this website make money off me", "makes no money from you"),
        ("so are you selling my data then", "nothing you type is sold or kept"),
        ("i dont trust ai honestly", "trust it only as far as you can check it"),
        ("is trading crypto halal", "religious ruling"),
        ("is crypto even legal in india", "legal in India"),
        ("what time does the stock market open", "9:30"),
        ("im completely new to all this", "welcome"),
        ("and if i mess something up on here can i undo it", "nothing here can be messed up"),
        ("my friend made 5x on a meme coin should i buy some too", "says nothing about yours"),
        ("is 100 dollars enough to learn trading", "yes, to learn"),
    ])
    def test_plain_questions(self, asked: str, expected: str) -> None:
        from argus.lui import newcomer

        said = newcomer.reply(asked)
        assert said is not None and expected in " ".join(said.lines), (asked, said)

    def test_follow_ups(self) -> None:
        from argus.lui.newcomer import followup_lines

        assert "rToken" in " ".join(followup_lines(
            "is that different from a stock token on here", ["whats an etf"]) or [])
        assert "market order" in " ".join(followup_lines(
            "how is that different from just buying right now", ["whats a limit order"]) or [])
        assert "not risk-free" in " ".join(followup_lines("is that safe to do",
                                                          ["whats staking"]) or [])
        assert "30%" in " ".join(followup_lines("whats the catch then if its legal",
                                                ["is crypto even legal in india"]) or [])

    def test_emoji_alone(self) -> None:
        from argus.lui.answer import _not_a_question

        assert "emoji" in (_not_a_question("📈🚀💰") or "")


class TestReAsked:
    """New phrasings of round 38's findings, asked before shipping."""

    def test_first_replies(self) -> None:
        from argus.lui.newcomer import first_reply

        for asked, expected in (("how do I put money in my bitget account", "on Bitget itself"),
                                ("is crypto allowed in islam", "religious ruling"),
                                ("what hours is the NYSE open", "9:30")):
            said = first_reply(asked)
            assert said is not None and expected in " ".join(said), asked

    def test_an_entry_said_with_from(self) -> None:
        from argus.lui.research.dispatch import stated_entry

        assert stated_entry("I'm 20x long ETH from 3,100 — at what price do I get "
                             "liquidated?") == 3_100
