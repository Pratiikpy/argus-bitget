"""Round 25: a first-time user's questions that were declined or misread, each read correctly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.lui import newcomer, server
from argus.lui.research import performance


@dataclass(frozen=True)
class _Candle:
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


class TestFirstTimeUserRound25:
    def test_a_pick_between_two_names_shows_both_without_picking(self) -> None:
        said = server._restated("should i buy bitcoin or solana", [])
        assert said is not None
        question, lead = said if isinstance(said, tuple) else (said, "")
        assert question == "how have bitcoin and solana done over the last year"
        assert "will not pick" in lead

    def test_follow_ups_carry_the_names_said_before(self) -> None:
        def _q(said: object) -> object:
            return said[0] if isinstance(said, tuple) else said

        assert _q(server._restated("with $200 how many can i get", ["what is the price of AAPL"])) \
            == "how many AAPL is $200"
        assert _q(server._restated("which one drops less when stuff crashes",
                                ["should i buy BTC or SOL"])) == \
            "which drops less on stock-market selloff days, BTC or SOL"
        assert _q(server._restated("matlab kitne dollar ka girna padega?",
                                ["im long btc 20x, where do i get liquidated"])) == \
            "where is my liquidation price on a 20x BTC long"
        assert _q(server._restated("is that good compared to the stock market",
                                ["bitcoin this year"])) == "how have BTC and SPY done this year"

    def test_the_selloff_comparison_ranks_by_the_move_on_those_days(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        moves = {"BTC": ("BTC", -0.031, 11, 2), "SOL": ("SOL", -0.051, 11, 1)}
        monkeypatch.setattr(server, "_hedge_claim_lines",
                            lambda name, days=365, numbers=False: moves[name])
        lines = server._selloff_compare_lines(["SOL", "BTC"])
        assert lines is not None
        assert lines[0].startswith("Bottom line: BTC has dropped less")
        assert "BTC averaged -3.1% and SOL -5.1%" in lines[0]

    def test_period_phrasings_a_beginner_uses(self) -> None:
        for asked in ("bitcoin this year", "how much did doge drop from its high",
                      "how far did sol move yesterday",
                      "What would $10,000 put in a year ago be worth now?"):
            assert performance.PERFORMANCE_Q.search(asked), asked
        now = datetime(2026, 10, 3, 12, tzinfo=UTC)
        day = performance.asked_period("how far did sol move yesterday", now)
        assert day is not None
        assert day.start == datetime(2026, 10, 2, tzinfo=UTC)
        # yesterday's candle closes at today's 00:00, so the period must reach it
        assert day.end == datetime(2026, 10, 3, tzinfo=UTC)
        high = performance.asked_period("how much did doge drop from its high", now)
        assert high is not None and "last year" in high.said

    def test_one_day_is_answered_with_its_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import history

        candle = _Candle(datetime(2026, 10, 2, tzinfo=UTC), Decimal("100"), Decimal("110"),
                         Decimal("95"), Decimal("104"))
        monkeypatch.setattr(history, "fetch_window", lambda *a, **k: [candle])
        now = datetime(2026, 10, 3, 12, tzinfo=UTC)
        period = performance.asked_period("yesterday", now)
        assert period is not None
        lines = performance.performance_lines(("SOLUSDT",), period)
        assert lines is not None
        assert "SOL moved +4.0%" in lines[0]
        assert "15.8% from its low to its high" in lines[0]

    @pytest.mark.parametrize("asked", ["can I buy part of one with $20?",
                                       "Is gold a safe place for my retirement savings?"])
    def test_beginner_questions_get_a_plain_answer(self, asked: str) -> None:
        said = newcomer.reply(asked)
        assert said is not None and (said.lines or said.reask), asked


class TestHostileReviewRound25:
    def test_the_saved_book_is_priced_at_its_own_marks(self) -> None:
        said = server._statement_lines(
            "what's my P&L on the book?", [],
            "Long 100 AAPL @ 200 (mark 220) / Short 20 ETH @ 3,000 (mark 2,700) / "
            "Long 2 MES @ 6,000 (mark 6,050)")
        assert said is not None and said[0].startswith("Bottom line: +$8,500.00 in all")

    def test_a_correction_reprices_the_statement_before(self) -> None:
        size = server._statement_lines("sorry, it was 20 SOL not 10",
                                       ["Long 10 SOL at 150, now 160"], "")
        assert size is not None and size[0].startswith("Bottom line: +$200.00 in all")
        side = server._statement_lines("actually it was short, not long",
                                       ["what's my P&L on the book?"],
                                       "long 50 TSLA at 400 mark 370")
        assert side is not None and side[0].startswith("Bottom line: +$1,500.00 in all")

    def test_a_statement_with_marks_is_not_prefaced_as_an_order(self) -> None:
        from argus.lui.honesty import order_prefix

        assert order_prefix("long 1,000 KO at 60, short 600 PEP at 170. KO goes to 62, "
                            "PEP to 175") is None
