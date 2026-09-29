"""Position and profit from typed fills (`lui/journal.position_and_pnl`)."""

from __future__ import annotations

from argus.lui.journal import parse_text, position_and_pnl

JUDGE = ("I bought 200 TSLA at 245, 100 at 260, then sold 150 at 255, what's my current position "
         "and pnl")


def test_the_judges_example_by_average_cost_and_fifo() -> None:
    """A judge's audit, 2026-09-29: this was answered with an execution plan for a new order."""
    found = position_and_pnl(JUDGE, price=lambda symbol: 300.0)
    assert found is not None
    lines, _, data = found
    assert lines[0] == ("Bottom line: you hold 150 TSLA at an average cost of $250.00; the 150 "
                        "sold realised +$750.00 at average cost (+$1,500.00 first in, first "
                        "out); at today's prices that is +$7,500.00 unrealised in all.")
    row = data["position"]["TSLAUSDT"]
    assert row["fifo_average_cost"] == 255.0
    assert ("TSLA unrealised at Bitget's live 300.00: +$7,500.00 at average cost, +$6,750.00 "
            "first in, first out.") in lines


def test_the_fills_parse_with_a_repeated_verb_and_a_bare_size() -> None:
    legs, _ = parse_text(JUDGE)
    assert [(leg.action, leg.qty, leg.price) for leg in legs] == [
        ("buy", 200.0, 245.0), ("buy", 100.0, 260.0), ("sell", 150.0, 255.0)]


def test_a_sale_larger_than_the_holding_is_said_not_guessed() -> None:
    found = position_and_pnl("bought 10 NVDA at 180 and sold 40 at 190 whats my pnl")
    assert found is not None and "would open a short" in found[0][0]


def test_a_question_that_is_not_about_the_position_is_left_alone() -> None:
    assert position_and_pnl("bought 10 NVDA at 180 and sold 4 at 190, what bad habits") is None


def test_fills_joined_by_and_and_a_price_the_trader_gives() -> None:
    """A judge's audit, 2026-09-29: answered with the desk's own statistics."""
    found = position_and_pnl("I bought 100 shares of AAPL at 150 and 50 more at 160, what's my "
                             "average cost and P&L if it's at 175 now")
    assert found is not None
    lines, _, _ = found
    assert lines[0] == ("Bottom line: you hold 150 AAPL at an average cost of $153.33; at "
                        "today's prices that is +$3,250.00 unrealised in all.")
    assert "AAPL unrealised at the 175.00 you gave: +$3,250.00." in lines


def test_typos_are_repaired_but_a_changed_letter_never_is() -> None:
    """A hostile review, 2026-09-29: "boght ... sodl ... pattrens" was not read as a journal."""
    from argus.lui.journal import repair_typos

    assert repair_typos("boght AAPL at 150, sodl at 145, waht bad pattrens") == (
        "bought AAPL at 150, sold at 145, what bad patterns")
    assert repair_typos("bought gold at 2400") == "bought gold at 2400"


def test_a_typed_fee_is_read_onto_the_fill_before_it() -> None:
    legs, _ = parse_text("bought NVDA at 200, sold at 210, fee was 25 dollars")
    assert legs[-1].fee == 25.0


def test_a_sale_typed_as_a_share_of_the_holding() -> None:
    """First-user audit, 2026-09-29: "sold half ... how am I doing" was answered with sizing."""
    found = position_and_pnl("I bought 1 BTC at 60000 and sold half at 70000, how am I doing",
                             price=lambda symbol: 80000.0)
    assert found is not None
    lines, _, data = found
    assert lines[0].startswith("Bottom line: you hold 0.5 BTC at an average cost of $60,000.00; "
                               "the 0.5 sold realised +$5,000.00")
    assert data["position"]["BTCUSDT"]["held"] == 0.5


def test_holdings_written_with_their_price_are_read_as_buys() -> None:
    """A judge's audit, 2026-09-29: "P&L on 100 COIN@$180, 50 MSTR@$320" computed no P&L."""
    found = position_and_pnl("what's my P&L on 100 COIN@$180, 50 MSTR@$320",
                             price=lambda symbol: {"COINUSDT": 190.0, "MSTRUSDT": 150.0}[symbol])
    assert found is not None
    lines, _, _ = found
    assert lines[0] == ("Bottom line: you hold 100 COIN at an average cost of $180.00; you hold "
                        "50 MSTR at an average cost of $320.00; at today's prices that is "
                        "-$7,500.00 unrealised in all.")
