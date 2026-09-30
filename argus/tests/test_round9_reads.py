"""How questions are read, as the round-9 role audits found them misread (2026-09-30)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from argus.lui.research import detect, research_symbols
from argus.lui.research.kinds import ResearchKind

NOW = datetime(2026, 9, 30, 12, tzinfo=UTC)


def _agent_summary(name: str) -> dict[str, Any]:
    return {
        "generated_at": "2026-09-29T23:00:52Z",
        "metrics": {"n_hours": 43, "n_closed_trades": 3, "total_return": 0.00024,
                    "sharpe_ann": 1.8788, "sharpe_se_ann": 14.27,
                    "ci90": {"sharpe_ann": [-28.9, 27.5]}, "max_drawdown": -0.00135,
                    "win_rate": 0.0},
        "counts": {"decisions": 20, "orders_sent": 84, "fills": 10, "protective_rulings": 71},
        "expected_envelope": {"sharpe_ann": "median -0.5, p05 -12.3, p95 +11.1",
                              "closed_trades": "median 23"},
    }


def test_the_track_2_agent_is_answered_from_its_own_record() -> None:
    """A judge's audit: the agent's Sharpe and win rate were answered from the research desk's
    ledger, a different system, without a word."""
    from argus.lui import agent_answer

    assert agent_answer.asks_about_the_agent(
        "The agent page shows a Sharpe of 1.88 but a win rate of 0% on 3 closed trades. How?")
    assert not agent_answer.asks_about_the_agent("what is your track record")
    lines, _, data = agent_answer.answer(fetch=_agent_summary)
    assert "the Track 2 agent — a separate project" in lines[0]
    assert "Sharpe 1.88 with a standard error of 14.3" in lines[0]
    assert "win rate 0% on 3 closed trades" in lines[0]
    assert data["agent"]["closed_trades"] == 3


def test_an_unreadable_agent_record_gives_no_figure() -> None:
    from argus.lui import agent_answer

    lines, _, _ = agent_answer.answer(fetch=lambda name: None)
    assert "did not answer just now" in lines[0]


@pytest.mark.parametrize(("text", "symbols"), [
    ("What is the live price of BRK.B and its beta to QQQ?", ("BRKBUSDT", "QQQUSDT")),
    ("BRK/B price", ("BRKBUSDT",)),
    ("the U.S. economy and NVDA", ("NVDAUSDT",)),
])
def test_a_share_class_with_its_dot_is_one_ticker(text: str, symbols: tuple[str, ...]) -> None:
    assert research_symbols(text)[0] == symbols


def test_goog_says_it_was_read_as_googl() -> None:
    symbols, notes = research_symbols("GOOG price")
    assert symbols == ("GOOGLUSDT",)
    assert any("GOOG read as GOOGL" in n for n in notes)


def test_a_beta_to_the_market_is_the_names_profile() -> None:
    request = detect("What is the live price of BRK.B and its beta to QQQ?")
    assert request is not None and request.kind is ResearchKind.IMPACT
    assert request.symbols == ("BRKBUSDT",)


def test_risk_exactly_a_sum_is_a_sizing_question() -> None:
    from argus.lui.research import sizing

    text = "I want to risk exactly $500 with a 2% stop on TSLA. How big a position should I take?"
    assert sizing.asks_for_size(text)
    _, _, data = sizing.answer(text)
    assert data["sizing"]["position_net"] == pytest.approx(500 / (0.02 + sizing.ROUND_TRIP))


def test_a_long_form_holding_is_a_position() -> None:
    from argus.lui.journal import position_and_pnl

    held = position_and_pnl("I am long 50 NVDA perpetual contracts at an average entry price of "
                            "205.00. What is my unrealized P&L right now?", now=NOW,
                            price=lambda s: 228.76)
    assert held is not None and "+$1,188.00 unrealised" in held[0][0]


@pytest.mark.parametrize(("text", "rewritten"), [
    ("so what would you do with a 60/40 NVDA/AMD book?",
     "so what would you do with a 60% NVDA, 40% AMD book?"),
    ("50/30/20 BTC/ETH/SOL", "50% BTC, 30% ETH, 20% SOL"),
    ("a 24/7 market", "a 24/7 market"),
])
def test_a_split_written_with_slashes_is_weights(text: str, rewritten: str) -> None:
    from argus.lui.research.parse import split_notation

    assert split_notation(text) == rewritten


def test_a_compare_follow_up_keeps_the_first_name() -> None:
    from argus.lui.research.parse import follow_up

    request = follow_up("how does that compare with AMD?",
                        ["Nvidia's most recent quarterly revenue?", "and its earnings?"])
    assert request is not None and request.kind is ResearchKind.FUNDAMENTALS
    assert request.symbols == ("NVDAUSDT", "AMDUSDT")


def test_holdings_stated_in_dollars_are_a_priced_book() -> None:
    text = "I have $5k in NVDA and $5k in an S&P fund, what does that mean for me?"
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.BOOK
    assert request.book == pytest.approx({"NVDAUSDT": 0.5, "SP500USDT": 0.5})
    assert float(request.notional or 0) == pytest.approx(10_000)


def test_given_that_asks_the_previous_question_again() -> None:
    from argus.lui.research.parse import follow_up

    earlier = "I have $5k in NVDA and $5k in an S&P fund, what does that mean for me?"
    request = follow_up("what should I do differently given that", [earlier])
    assert request is not None and request.kind is ResearchKind.BOOK
    assert set(request.book) == {"NVDAUSDT", "SP500USDT"}


def test_a_mandate_says_which_limits_are_defaults_and_reads_a_retirement_horizon() -> None:
    from argus.lui.research.parse import _hours_said, stated_profile

    profile = stated_profile("I'm 60, retiring in 3 years, and can't afford to lose more than "
                             "10% — what does adding 15% TSLA do to my risk?")
    assert profile is not None
    assert profile.holding_horizon_hours == 3 * 8760
    assert "max_position_pct" in profile.defaulted
    assert "loss_tolerance_pct" not in profile.defaulted
    assert _hours_said(3 * 8760) == "3-year" and _hours_said(168) == "1-week"


def test_the_research_task_names_what_it_did_not_research() -> None:
    from argus.lui.task import _also_named

    notes = _also_named("research on COIN's competitor risk and MSTR's bitcoin exposure",
                        "COINUSDT", {})
    assert notes and "MSTR" in notes[0] and "one name at a time" in notes[0]


@pytest.mark.parametrize("text", ["What's WTI crude oil doing and does it matter for BTC?",
                                  "how does the oil price affect bitcoin"])
def test_whether_one_market_moves_another_is_their_co_movement(text: str) -> None:
    from argus.lui.research.fundamentals import pattern_reading_wins

    request = detect(text)
    assert request is not None and request.kind is ResearchKind.COMPARE
    assert pattern_reading_wins(request, text)


def test_the_console_keeps_answers_in_the_order_asked_and_shows_a_saved_book() -> None:
    from argus.lui.server import ALLOWANCE_NOTE, PAGE

    assert "pending-${++askSeq}" in PAGE and "id=\"clearbook\"" in PAGE.replace("'", '"')
    assert "counted per network address" in ALLOWANCE_NOTE


def test_the_proof_page_explains_its_words() -> None:
    from pathlib import Path

    from argus.lui.proof_page import collect, render

    wins, counts = collect(Path(__file__).resolve().parents[1] / "data")
    page = render(wins, counts)
    for term in ("Ablation", "Out-of-sample", "SUE", "Euler risk decomposition"):
        assert f"<dt>{term}</dt>" in page


def test_rates_named_beside_a_contract_are_the_macro_engines() -> None:
    """Live, 2026-09-30: "Where's the 10-year Treasury yield and gold right now?" was planned as a
    gold quote and the yield went unanswered."""
    from argus.lui.research.fundamentals import pattern_reading_wins

    text = "Where's the 10-year Treasury yield and gold right now?"
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.MACRO
    assert pattern_reading_wins(request, text)
