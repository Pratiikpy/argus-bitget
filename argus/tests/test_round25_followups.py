"""Round 25, the last of a first-time user's and a judge's follow-ups: each read as it was meant."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from argus.lui import server
from argus.lui.research import hindsight, research_symbols


def test_a_sum_a_week_is_each_weekly_buy() -> None:
    start = date(2025, 10, 1)
    days = [SimpleNamespace(day=start + timedelta(days=i), open=100.0, close=100.0)
            for i in range(400)]
    lines, _sources, data = hindsight.dca(
        "if I put $100 a week into bitcoin for a year what would I have?", "BTCUSDT",
        today=date(2026, 10, 3), daily=lambda ticker: days)
    weeks = data["dca"]["buys"]
    assert 50 <= weeks <= 54
    assert lines[0].startswith("Bottom line: $100 a week into BTC from ")
    assert f"{weeks} buys, ${weeks * 100:,.0f} in all" in lines[0]


def test_holding_follow_ups_carry_the_move_and_add_the_holdings() -> None:
    first = "i have 1 btc, what if it drops 10%?"
    second = "and if eth drop same and i have 2 eth?"
    assert server._leaning_follow_up(second, [first]) == "i have 2 eth, what if eth drops 10%?"
    both = server._leaning_follow_up("total both?", [first, second])
    assert both == "i have 1 btc and 2 eth, what if they all drop 10%?"


def test_a_stated_loss_is_judged_against_the_names_own_windows(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.market import history

    base = datetime(2025, 9, 1, tzinfo=UTC)
    # every week alternately -5% and +5%: a 3.5% weekly fall is ordinary
    closes = [100.0 * (0.95 if (i // 7) % 2 else 1.0) for i in range(400)]
    monkeypatch.setattr(history, "fetch_range", lambda *a, **k: [
        SimpleNamespace(ts=base + timedelta(days=i), close=c) for i, c in enumerate(closes)])
    lines = server._own_move_judged("I lost 3.5% on NVDA this week, is that bad?")
    assert lines is not None and lines[0].startswith("Bottom line: ordinary for NVDA")


def test_ordinary_words_are_not_companies() -> None:
    assert research_symbols("is that bullish?")[0] == ()
    assert research_symbols("When does Micron report next")[0] == ("MUUSDT",)


def test_the_agent_page_states_the_next_trades_effect() -> None:
    from argus.lui import agent_page

    html = agent_page._metrics({"n_closed_trades": 7, "win_rate": 2 / 7}, {})
    assert "the next win takes the win rate to 37.5% and the next loss to 25.0%" in html
