"""Is there an arbitrage between a stock's spot rToken and its perpetual on Bitget right now?

The console used to answer no arbitrage question (the capability register said so, blocker 4 of
`eval/capabilities/08-*.toml`), while the study that did exist, `research/arbitrage_study.
decompose()`, subtracted flat constants from a mid-to-mid gap and, scored against a general LP
solver on real simultaneous Bitget books, accepted 49 losing trades out of 52 (precision 5.8%,
`data/general_arb_comparison.json`). This answer uses the method that tied the solver on every
snapshot instead: `research/executable_arb.best_arbitrage`, which walks both live books level by
level in both directions, charges each venue its own published taker fee, and reports the most
money the two books leave — very often none (Activity/27_CAPABILITY_CLOSE_PLAN.md §3).

What a trader gets: whether the touch clears both fees, by how many basis points, the best size and
what it would net, what that is worth once a leg can fail, and the two quotes it was read from, so a
reader can see why the answer is usually no. Nothing is placed: the console sends no orders.
"""

from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal
from typing import Any

from argus.lui.answer import Source

ASKS = re.compile(
    r"\barb(?:itrage)?s?\b|\bbasis\s+trade\b|\bspot[\s-]*(?:vs\.?|versus|and|against|to)[\s-]*"
    r"(?:the\s+)?perp(?:etual)?s?\b|\bperp(?:etual)?s?[\s-]*(?:vs\.?|versus|and|against|to)"
    r"[\s-]*(?:the\s+)?(?:spot|rtoken)\b|\bgap\s+between\s+(?:the\s+)?(?:spot|rtoken)", re.I)
WORTH_SENDING = Decimal("1")
"""Below this net, in USDT, the trade is reported as not worth two orders rather than as a yes."""
LEVELS = 50
"""Book depth read on each side: enough for any size the console sizes (up to ~$100k)."""


def asks_for_arbitrage(text: str) -> bool:
    """An arbitrage question that names a listed name."""
    from argus.lui.research import research_symbols

    return bool(ASKS.search(text)) and bool(research_symbols(text)[0])


def _spot_fee(spot: str) -> Decimal | None:
    """The spot rToken's taker fee as a fraction, or None when Bitget lists no such spot token."""
    from argus.market.bitget import BitgetError, public_get

    try:
        rows = public_get("/api/v2/spot/public/symbols", {"symbol": spot}) or []
    except BitgetError:
        return None  # Bitget answers an unlisted spot symbol with HTTP 400, not an empty list
    if not isinstance(rows, list) or not rows or rows[0].get("status") != "online":
        return None
    return Decimal(str(rows[0]["takerFeeRate"]))


def answer(text: str) -> tuple[list[str], list[Source], dict[str, Any]]:
    """Lines, sources and data for the console, like every research engine."""
    from argus.lui.research import research_symbols
    from argus.market.crossasset_feed import fetch_taker_bps
    from argus.market.depth import fetch_orderbook
    from argus.market.universe import is_equity
    from argus.research.executable_arb import best_arbitrage

    perp = research_symbols(text)[0][0]
    ticker = perp.removesuffix("USDT").removesuffix("STOCK")
    # A company's spot leg is its rToken (RNVDAUSDT); a coin's is the same symbol on spot.
    stock = is_equity(perp)
    spot = f"R{ticker}USDT" if stock else f"{ticker}USDT"
    spot_fee = _spot_fee(spot)
    if spot_fee is None:
        what = "spot rToken" if stock else "spot market"
        return ([f"Bottom line: Bitget lists no {what} for {ticker} ({spot}), so there are not "
                 f"two books to trade against each other — the perpetual is the only {ticker} "
                 f"instrument here."],
                [Source("venue", "bitget /api/v2/spot/public/symbols", f"{spot}: not online")],
                {"arbitrage": None})
    perp_fee = Decimal(str(fetch_taker_bps(perp))) / Decimal(10_000)
    spot_book = fetch_orderbook(spot, limit=LEVELS, category="SPOT")
    perp_book = fetch_orderbook(perp, limit=LEVELS, category="USDT-FUTURES")
    # Named by leg, so a coin whose spot and perpetual share one symbol (BTCUSDT) reads clearly.
    spot_book = replace(spot_book, symbol=f"{spot} spot")
    perp_book = replace(perp_book, symbol=f"{perp} perpetual")
    arb = best_arbitrage(spot_book, perp_book, fee_a=spot_fee, fee_b=perp_fee)
    best = arb.best
    edge = best.top_of_book_edge_bps
    fees_bps = (spot_fee + perp_fee) * 10_000
    buy, sell = best.buy_venue, best.sell_venue
    lines: list[str] = []
    if arb.monetizable and best.net >= WORTH_SENDING:
        lines.append(
            f"Bottom line: yes — buying {best.quantity:,.4f} on the {buy} and selling on the "
            f"{sell} nets about ${best.net:,.2f} after both taker fees ({best.net_bps:.1f}bps of "
            f"the bought notional), about ${arb.expected_net:,.2f} once a leg can fail to fill.")
    elif arb.monetizable:
        lines.append(
            f"Bottom line: technically, by ${best.net:,.4f} — buying on the {buy} and selling on "
            f"the {sell} clears both fees on {best.quantity:,.4f} units only, less than "
            f"${WORTH_SENDING} for two orders and the risk that one of them does not fill.")
    else:
        lines.append(
            f"Bottom line: no — at the touch the gap between {spot} spot and {perp} perpetual "
            f"does not cover the "
            f"two taker fees ({fees_bps:.0f}bps together); the best first unit loses "
            f"{abs(edge or Decimal(0)):.1f}bps." if edge is not None else
            "Bottom line: no — one of the two books is empty, so there is nothing to trade "
            "across.")
    for book, fee in ((spot_book, spot_fee), (perp_book, perp_fee)):
        if book.bids and book.asks:
            bid, ask = book.bids[0].price, book.asks[0].price
            spread = (ask - bid) / ((ask + bid) / 2) * 10_000
            shown = f"{spread:.1f}" if spread >= 1 else f"{spread:.3f}"
            lines.append(f"{book.symbol}: bid {bid} / ask {ask} ({shown}bps spread), "
                         f"taker fee {fee * 10_000:.0f}bps.")
    if spot_book.bids and spot_book.asks and perp_book.bids and perp_book.asks:
        mid_spot = (spot_book.bids[0].price + spot_book.asks[0].price) / 2
        mid_perp = (perp_book.bids[0].price + perp_book.asks[0].price) / 2
        lines.append(
            f"The mid-to-mid gap is {(mid_perp / mid_spot - 1) * 10_000:+.1f}bps (perpetual over "
            f"spot). A gap between mids is not money: each side must cross its own spread and pay "
            f"its own fee, which is where a mid-to-mid reading goes wrong.")
    lines.append(
        "Method: both books walked level by level in both directions, each venue charged its own "
        "published taker fee — the exact optimum, which matched a general linear-programming "
        "solver on every one of 640 real Bitget book snapshots. The console places no orders.")
    sources = [
        Source("venue", "bitget /api/v3/market/orderbook", f"{spot} and {perp}, {LEVELS} levels"),
        Source("venue", "bitget /api/v2/spot/public/symbols + /api/v3/market/instruments",
               "each venue's taker fee"),
        Source("computation", "argus.research.executable_arb.best_arbitrage",
               "exact two-book optimum, both directions"),
    ]
    return lines, sources, {"arbitrage": arb.as_dict(), "spot": spot, "perp": perp}
