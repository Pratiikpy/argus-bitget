"""`research/desk_math.py` and the round-25 statement additions: worked on the trader's figures.

Each expected figure is the hostile reviewer's hand computation from round 25.
"""

from __future__ import annotations

from datetime import date

from argus.lui.research import desk_math
from argus.lui.research.statement import price_statement


def _no_live(name: str) -> float:
    raise AssertionError(f"{name} was priced live")


class TestDeskMath:
    def test_basis_and_its_annualised_yield(self) -> None:
        lines = desk_math.basis_lines("BTC spot is 64,000 and the quarterly future is 65,600 with "
                                      "90 days to expiry. What's the basis and annualised yield?")
        assert lines is not None
        assert "the basis is 1,600.00 (2.50% of spot)" in lines[0]
        assert "about 10.1% a year annualised simply" in lines[0]

    def test_funding_counts_the_settlements_held_through(self) -> None:
        across = desk_math.funding_window_lines(
            "Funding is paid at 00:00, 08:00, 16:00 UTC. I opened a short at 07:59 UTC and closed "
            "at 08:01 UTC with $1,000,000 notional and +0.05% funding. Did I pay or receive and "
            "how much?")
        assert across is not None and across[0].startswith(
            "Bottom line: the short receives 1 funding payment")
        assert "$500.00" in across[0]
        weekend = desk_math.funding_window_lines(
            "Holding a BTC perp long of $50,000 from Friday 16:00 UTC to Monday 16:00 UTC, "
            "funding +0.01% every 8h. How many funding payments and total cost?")
        assert weekend is not None and "pays 9 funding payments" in weekend[0]
        assert "$45.00" in weekend[0]

    def test_time_zones_use_the_offset_on_that_date(self) -> None:
        lines = desk_math.time_zone_lines("The Fed decision is at 2pm New York time on 28 Oct. "
                                          "What time is that in UTC and in Mumbai?",
                                          today=date(2026, 10, 3))
        assert lines is not None
        assert "UTC 18:00 on 28 Oct" in lines[0] and "Mumbai 23:30 on 28 Oct" in lines[0]

    def test_a_stated_beta_sizes_the_hedge(self) -> None:
        lines = desk_math.stated_beta_hedge_lines(
            "I'm long $500k of SOL. Its beta to BTC is 1.4. How much BTC do I short to hedge?")
        assert lines is not None and "short about $700,000 of BTC" in lines[0]


class TestStatementRound25:
    def test_euro_legs_are_converted_at_the_stated_rate(self) -> None:
        priced = price_statement("Long €20,000 of SAP at 200 EUR, now 180 EUR, and long "
                                 "$10,000 of AAPL at 200, now 220. Total P&L in dollars at "
                                 "EURUSD 1.10?", price=_no_live)
        assert priced is not None and priced.lines[0].startswith("Bottom line: -$1,200.00 in all")

    def test_breakeven_after_doubling(self) -> None:
        priced = price_statement("I bought 50 MSTR at 400 and the price is now 300. What do I need "
                                 "for breakeven if I double my position at 300?", price=_no_live)
        assert priced is not None
        assert "breakeven after doubling at 300 is 350" in priced.lines[0]

    def test_the_remaining_shares_and_a_share_of_the_account(self) -> None:
        sold = price_statement("Bought 200 NVDA at 120 on Friday, sold 100 at 130 on Monday, sold "
                               "the remaining 100 at 110 on Tuesday. Realised P&L?",
                               price=_no_live)
        assert sold is not None and sold.lines[0].startswith("Bottom line: $0.00 in all")
        share = price_statement("long 100 NVDA at 180, drops 5% from 180. What is that as a % of "
                                "my $90,000 account?", price=_no_live)
        assert share is not None and "-1.00% of the $90,000 account" in share.lines[0]

    def test_more_shares_and_a_pronoun_mark(self) -> None:
        priced = price_statement("Long 1,000 shares XYZ at 50, average down by buying 1,000 more "
                                 "at 40, then average up by buying 2,000 at 45. Stock at 44. Avg "
                                 "cost and P&L?", price=_no_live)
        assert priced is not None and priced.lines[0].startswith("Bottom line: -$4,000.00 in all")


def test_an_owned_position_with_a_stated_beta() -> None:
    lines = desk_math.stated_beta_hedge_lines(
        "I own $200k of ETH and its beta to BTC is 1.2. How much BTC short hedges it?")
    assert lines is not None and "short about $240,000 of BTC" in lines[0]
