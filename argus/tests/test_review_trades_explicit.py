"""The MCP review tool reads the one-trade example it documents; the console still needs a review
cue for a single trade, so a chat about one past trade is not taken for a review (QA inventory,
2026-09-28)."""

from __future__ import annotations

from datetime import date

from argus.lui.journal import read_journal
from argus.lui.mcp_server import TOOLS

ONE_TRADE = "bought 10 NVDA at 180 on 2 Sep, sold at 172 on 9 Sep"


def test_the_tool_documents_the_example_it_now_reads() -> None:
    tool = next(t for t in TOOLS if t["name"] == "argus_review_trades")
    assert ONE_TRADE in str(tool["inputSchema"])


def test_one_round_trip_is_a_journal_when_the_caller_asked_for_a_review() -> None:
    journal = read_journal(ONE_TRADE, now=date(2026, 9, 28), explicit=True)
    assert journal is not None and len(journal.legs) == 2


def test_the_console_still_wants_a_cue_for_a_single_trade() -> None:
    assert read_journal(ONE_TRADE, now=date(2026, 9, 28)) is None
    assert read_journal(ONE_TRADE + " — what did I do wrong?", now=date(2026, 9, 28)) is not None
