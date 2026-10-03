"""`research/statement.py`: a trader's written legs, fills and marks, priced as a blotter would.

Every case is one a hostile review in round 25 typed and got a wrong figure for; the expected
total is the reviewer's hand computation, checked again here line by line in the comments.
"""

from __future__ import annotations

import pytest

from argus.lui.research.statement import is_statement, price_statement, share_of_book

CASES = [
    # +3,000 + 8,000 - 1,000
    ("Long 10 ETH at 3,000 mark 3,300; short 2 BTC at 70,000 mark 66,000; long 100 NVDA at 150 "
     "mark 140. Total P&L?", "+$10,000.00"),
    # 2,000 - 3,000
    ("long 1,000 KO at 60, short 600 PEP at 170. KO goes to 62, PEP to 175", "-$1,000.00"),
    ("Long 100 AAPL at 200 now 220, short 50 MSFT at 400 now 380", "+$3,000.00"),
    # the hedge is a short: -10,000 + 15,000
    ("I'm long 1 BTC at 60k. I hedge by shorting 1.5 BTC perp at 60k. BTC goes to 50k. Net P&L?",
     "+$5,000.00"),
    # spot and perp are two instruments with two marks: +8,000 - 7,400 - 2,000
    ("long 2 BTC spot at 60,000, short 2 BTC perp at 60,600, long 10 ETH at 3,000. BTC now "
     "64,000 spot, perp 64,300; ETH 2,800.", "-$1,400.00"),
    ("Short 1 BTC perp at 70,000, long 1 spot at 69,500, converges at 72,000", "+$500.00"),
    ("Long 2 BTC at 60,000, short 2 at 62,000, BTC now 61,000", "+$4,000.00"),
    ("long 10 ETH at 3,000, short 10 ETH at 3,000, ETH at 3,300. also long 1 BTC at 60k, now 66k",
     "+$6,000.00"),
    # 7,500 - 75 - 82.50
    ("Bought 500 SOL at $150, paid 0.1% fee each side, sold at $165.", "+$7,342.50"),
    # MNQ is $2 a point: +400 then -400
    ("Short 3 MNQ at 20,000, cover 1 at 19,800, the rest at 20,100.", "$0.00"),
    # MES $5, ES $50: +2,000 - 2,500
    ("long 4 MES at 6,000, short 1 ES at 6,050, index to 6,100", "-$500.00"),
    ("Long 1 NQ at 20,000 mark 20,250, long 2 MNQ at 20,000 mark 20,250", "+$6,000.00"),
    # realised +500, then 2 at an average 3,100 marked 3,400
    ("Bought 2 ETH at 3,000, sold 1 at 3,500, bought 1 at 3,200, now 3,400.", "+$1,100.00"),
    # (3 - 10) x 100 x 5
    ("Sold 5 AAPL 220 puts at $3.00. At expiry AAPL is 210. P&L?", "-$3,500.00"),
    ("Long 1,000 TSLA at 300 marked at 3.5e2", "+$50,000.00"),
    ("long 2.5e3 SOL at 1.5e2, now 1.6e2", "+$25,000.00"),
    ("Long 10 BTC at 1e5, now 9.5e4", "-$50,000.00"),
    # the fee is charged on the opening fill: 3,000 - 18
    ("short 100 COIN at 300, mark 270, paid 0.06% taker fee", "+$2,982.00"),
    ("Long 10 ETH at 3,000 now 3,300, paid $40 in fees and $25 in funding", "+$2,935.00"),
    # a negative oil price is a price: (20 - -37) x 10
    ("WTI: bought 10 barrels at -37 and sold at 20", "+$570.00"),
    # average 45, marked 44
    ("XYZ 1,000 at 50, +1,000 at 40, +2,000 at 45, now 44", "-$4,000.00"),
    ("long 3e9 PEPE at $0.0001, now $0.0003, short 1 BTC at 60k, now 70k", "+$590,000.00"),
    ("long 100 NVDA at 180, drops 5% from 180", "-$900.00"),
]


@pytest.mark.parametrize(("text", "total"), CASES)
def test_each_statement_is_priced_from_the_traders_own_figures(text: str, total: str) -> None:
    def no_live_price(name: str) -> float:
        raise AssertionError(f"{name} was priced live although a mark was typed")

    priced = price_statement(text, price=no_live_price)
    assert priced is not None, text
    assert priced.lines[0].startswith(f"Bottom line: {total} in all"), priced.lines[0]


def test_fifo_and_lifo_are_both_given_and_the_unmarked_lot_is_priced_live() -> None:
    priced = price_statement("Bought 100 AAPL at 150 in Jan, 100 at 200 in March. Sold 100 at 220 "
                             "in June. FIFO vs LIFO?", price=lambda name: 250.0)
    assert priced is not None
    said = " ".join(priced.lines)
    assert "FIFO (oldest lots sold first): +$7,000.00 realised" in said
    assert "LIFO (newest first): +$2,000.00 realised" in said
    assert "Marked at Bitget's last price, because no price was typed for it: AAPL." in said


def test_a_spread_without_an_expiry_price_gives_its_payoff_not_a_zero() -> None:
    priced = price_statement("Bull call spread 200/210 at $6/$2, 5 lots")
    assert priced is not None
    assert "most it can make +$3,000.00, most it can lose -$2,000.00, breakeven 204" in \
        priced.lines[0]


def test_worth_is_led_with_when_it_is_asked() -> None:
    priced = price_statement("1.5e2 AAPL at 2.1e2, value at 2.3e2")
    assert priced is not None
    assert priced.lines[0].startswith("Bottom line: worth $34,500.00 at the mark")


def test_a_share_of_the_book_moves_only_that_share() -> None:
    lost = share_of_book("$50,000. 30% in NVDA. NVDA falls 20%")
    assert lost is not None and "a loss of $3,000.00 (-6.00% of the $50,000)" in lost[0]
    gained = share_of_book("10% of my $1,000,000 in ETH; ETH +50%. New value?")
    assert gained is not None and "the book becomes $1,050,000" in gained[0]


@pytest.mark.parametrize("text", ["should i buy 10 ETH at 3,000?", "what is the price of BTC",
                                  "Long 10 SOL at 150", "I have $10,000, what should I do"])
def test_what_is_not_a_priced_statement_is_left_alone(text: str) -> None:
    assert not is_statement(text)


@pytest.mark.parametrize(("text", "total"), [
    # ticker first: +1,500 + 4,000
    ("ETH long 5 @ 2,800 marked 3,100; BTC short 1 @ 90k marked 86k. Net P&L?", "+$5,500.00"),
    # the long written as a holding with its entry, hedged by an equal short: flat
    ("I hedged my 2 BTC long (entry 70k) by shorting 2 BTC perps at 70k. BTC is now 60k. Total?",
     "$0.00"),
    # (8 - 20) x 100 x 3
    ("Wrote 3 TSLA 300 puts for $8. TSLA closes at 280 on expiry. P&L?", "-$3,600.00"),
])
def test_round25_reask_forms(text: str, total: str) -> None:
    def no_live_price(name: str) -> float:
        raise AssertionError(f"{name} was priced live although a mark was typed")

    priced = price_statement(text, price=no_live_price)
    assert priced is not None, text
    assert priced.lines[0].startswith(f"Bottom line: {total} in all"), priced.lines[0]


@pytest.mark.parametrize(("text", "total"), [
    # the clock times are times: 2 settlements are not a $1m fill
    ("Long 100 AMD at 100. Sold 150 at 110. Mark 105.", "+$1,250.00"),
    ("Short 100 AAPL at 150, covered 150 at 140, mark 145.", "+$1,250.00"),
    ("AAPL 100 @150 now 160, MSFT -50 @300 now 290, NVDA 10 @ 500 now 450", "+$1,000.00"),
    ("Long 100 AAPL at 150 mark 160, fees $12 total", "+$988.00"),
    ("Long 2 ES at 5000, closed at 5010, commission $2.25 per contract per side", "+$991.00"),
    ("Long 100 SAP at 200,50 mark 210,25", "+$975.00"),
    ("Long 100 AAPL at $150 and 200 MSFT at $300; marks 160 and 290 respectively", "-$1,000.00"),
    ("At 250 I shorted 100 TSLA; it's 240 now.", "+$1,000.00"),
    ("Sold 100 AAPL at 170 that I bought at 150", "+$2,000.00"),
    ("Long 100 AAPL at 150, it's up 10% since", "+$1,500.00"),
    ("Bought 5 AAPL 200 calls at 3.20, now 1.10", "-$1,050.00"),
    ("Sold 3 TSLA 250 puts at 5.00, now 7.50", "-$750.00"),
    ("Bought 2 SPY 500/510 call spreads for 4.00, mark 6.50. P&L and max profit?", "+$500.00"),
    ("Bought 100 AAPL at 10, bought 100 more at 20, sold 150 at 25, LIFO. Realised P&L?",
     "+$1,250.00"),
    ("Long 100 SAP at €200 mark €210, EURUSD 1.08. P&L in dollars?", "+$1,080.00"),
    # 1,900 x 1.05 - 1,800 x 1.10
    ("Long 10 SAP shares at 180 EUR, mark 190 EUR; EUR/USD 1.10 at entry and 1.05 now. P&L in "
     "USD?", "+$15.00"),
])
def test_round26_hostile_forms(text: str, total: str) -> None:
    priced = price_statement(text, price=lambda name: 333.0)
    assert priced is not None, text
    assert priced.lines[0].startswith(f"Bottom line: {total} in all"), priced.lines[0]


def test_a_live_mark_far_from_the_traders_prices_is_not_used() -> None:
    priced = price_statement("Bought 100 AAPL at 10, bought 100 more at 20, sold 150 at 25, "
                             "FIFO. Realised and what's left?", price=lambda name: 333.0)
    assert priced is not None and priced.lines[0].startswith("Bottom line: +$1,750.00 in all")
    assert any(line.startswith("Still open and not priced") for line in priced.lines)


def test_a_spread_quoted_at_its_net_has_its_limits() -> None:
    priced = price_statement("Bought 2 SPY 500/510 call spreads for 4.00, mark 6.50")
    assert priced is not None
    assert "most it can make +$1,200.00, most it can lose -$800.00, breakeven 504" in \
        priced.lines[1]
