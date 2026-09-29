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
                        "out).")
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
