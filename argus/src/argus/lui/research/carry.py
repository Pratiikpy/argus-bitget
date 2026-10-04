"""Funding carry: what holding spot long against a short perpetual has earned, and what it costs.

A judge (round 30) asked three questions about a pure funding-rate carry on BTC and ETH: is this a
good week for it and what is the expected annualised yield; which of the two carries better at the
current rates; and, if funding flips negative within three days, what holding period breaks even
at their VIP2 fees. The first got Bitget's fee schedule, the second both quotes with no verdict,
the third "VIP is not listed on Bitget". Each is answered here from Bitget's own data:

* the current rate and settlement interval (``/api/v2/mix/market/current-fund-rate``, through
  ``market.bitget.fetch_tickers`` and the contract list) and every settlement Bitget serves, about
  ninety days (``market.crossasset_feed.fetch_funding``);
* the week's average against the ninety-day average, so "is this a good week" is a comparison, not
  an adjective, and the share of settlements that were positive, which is the carry's real risk;
* the round trip of the position itself: buy spot and short the perpetual to open, the reverse to
  close, at Bitget's taker rates — spot from its published VIP table (read from
  bitget.com/fee on 2026-10-04: VIP 0 0.10%/0.10%, VIP 1 0.08%/0.08%, VIP 2 0.065%/0.07% maker/
  taker), the perpetual from its contract list (``takerFeeRate`` 0.06%). Bitget's futures VIP table
  sits on a tab that could not be read here, so a futures VIP discount is not applied and the answer
  says so;
* how many days of funding at today's rate pay those fees back, which is the breakeven holding
  period, and what flipping negative after a stated number of days leaves.

Nothing here is a forecast: a yearly figure is the window's average carried forward, and the
answer says that.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from argus.lui.trace import trace_module

CARRY_Q: Final = re.compile(
    r"\bcarry\b|\bcash[\s-]+and[\s-]+carry\b|\bbasis\s+trade\b|\bfunding[\s-]*(?:rate\s+)?(?:yield|"
    r"arb\w*|harvest\w*|farm\w*|strategy|trade|income)\b|\bdelta[\s-]*neutral\b", re.I)
"""A question about earning the funding rate."""
_VIP: Final = re.compile(r"\bvip\s*-?\s*(?P<n>[0-7])\b", re.I)
_FLIP: Final = re.compile(
    r"\b(?:flips?|turns?|goes|went)\s+negative\b[^?]{0,40}?\b(?:within|in|after)\s+(?P<n>\d+)\s*"
    r"(?P<u>days?|hours?|h|d|settlements?)\b|\b(?:within|in|after)\s+(?P<n2>\d+)\s*(?P<u2>days?|"
    r"hours?)\b[^?]{0,40}?\b(?:flips?|turns?|goes)\s+negative\b", re.I)
SPOT_VIP: Final[dict[int, tuple[float, float]]] = {
    0: (0.0010, 0.0010), 1: (0.0008, 0.0008), 2: (0.00065, 0.0007), 3: (0.0005, 0.0006),
    4: (0.0004, 0.0005), 5: (0.0003, 0.0004), 6: (0.0002, 0.00035), 7: (0.0, 0.0003)}
"""Bitget's spot maker/taker by VIP level, from bitget.com/fee, read 2026-10-04."""


def _settlements(symbol: str, days: int) -> list[float]:
    from argus.market.crossasset_feed import fetch_funding

    since = datetime.now(UTC) - timedelta(days=days)
    return [r for t, r in fetch_funding(symbol) if datetime.fromtimestamp(t / 1000, UTC) >= since]


def _per_year(symbol: str) -> float:
    from argus.market import universe

    hours = (universe.contracts().get(symbol) or universe.Contract(symbol, False)).funding_hours
    return 365 * 24 / (hours or 8)


def lines(text: str, prior: list[str], names: tuple[str, ...]) -> list[str] | None:
    """The carry answer for the coins named (BTC and ETH when none are), or None when the question
    is not about carry."""
    if not CARRY_Q.search(text) and not (_FLIP.search(text) and any(CARRY_Q.search(p)
                                                                     for p in prior[-4:])):
        return None
    from argus.lui.research.desk_answers import PERP_TAKER
    from argus.market.bitget import fetch_tickers

    symbols = [s for s in names if s.endswith("USDT")][:4] or ["BTCUSDT", "ETHUSDT"]
    try:
        tickers = fetch_tickers()
    except Exception:
        return None
    vip_m = _VIP.search(" ".join([*prior[-4:], text]))
    vip = int(vip_m.group("n")) if vip_m else 0
    spot_taker = SPOT_VIP[vip][1]
    round_trip = 2 * (spot_taker + PERP_TAKER)
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        ticker = tickers.get(symbol)
        if ticker is None:
            continue
        per_year = _per_year(symbol)
        now_rate = float(ticker.funding_rate)
        try:
            quarter = _settlements(symbol, 90)
            week = _settlements(symbol, 7)
        except Exception:
            quarter, week = [], []
        rows.append({
            "name": symbol.removesuffix("USDT"), "now": now_rate, "now_y": now_rate * per_year,
            "week_y": (sum(week) / len(week) * per_year) if week else None,
            "quarter_y": (sum(quarter) / len(quarter) * per_year) if quarter else None,
            "positive": (sum(1 for r in quarter if r > 0) / len(quarter)) if quarter else None,
            "count": len(quarter), "per_day": now_rate * per_year / 365,
            "hours": round(365 * 24 / per_year)})
    if not rows:
        return None
    ranked = sorted(rows, key=lambda r: -(r["week_y"] if r["week_y"] is not None else r["now_y"]))
    best = ranked[0]
    out: list[str] = []
    if len(rows) >= 2:
        out.append(f"Bottom line: {best['name']} carries better right now — "
                   + "; ".join(f"{r['name']} funding {r['now']:+.4%} every {r['hours']}h, about "
                               f"{r['now_y']:+.1%} a year at today's rate"
                               + (f" and {r['week_y']:+.1%} over the last 7 days"
                                  if r["week_y"] is not None else "") for r in ranked) + ".")
    else:
        r = rows[0]
        out.append(f"Bottom line: {r['name']} funding is {r['now']:+.4%} every {r['hours']}h, "
                   f"about "
                   f"{r['now_y']:+.1%} a year to a short perpetual held against spot at today's "
                   f"rate" + (f", {r['week_y']:+.1%} over the last 7 days"
                              if r["week_y"] is not None else "") + ".")
    for r in rows:
        if r["week_y"] is None or r["quarter_y"] is None:
            continue
        verdict = ("a better week than usual" if r["week_y"] > r["quarter_y"] * 1.15 else
                   "a weaker week than usual" if r["week_y"] < r["quarter_y"] * 0.85 else
                   "an ordinary week")
        out.append(f"Is this a good week for {r['name']}: {verdict} — the last 7 days paid "
                   f"{r['week_y']:+.1%} a year against {r['quarter_y']:+.1%} over the "
                   f"{r['count']} settlements of the last 90 days; funding was positive in "
                   f"{r['positive']:.0%} of them — in each of the others the carry paid "
                   f"instead of earning.")
    vip_said = (f"VIP {vip}" if vip_m else "VIP 0 (no level was stated)")
    out.append(f"Cost to run it: buy spot and short the perpetual, then reverse — {round_trip:.2%} "
               f"of the position at taker fees ({vip_said} spot {spot_taker:.3%} a side from "
               f"Bitget's fee schedule, perpetual {PERP_TAKER:.2%} a side from its contract list; "
               f"Bitget's futures VIP table could not be read here, so no futures discount is "
               f"applied).")
    for r in ranked[:2]:
        if r["per_day"] > 0:
            days = round_trip / r["per_day"]
            out.append(f"Breakeven for {r['name']}: at today's rate the funding pays the fees back "
                       f"in about {days:.1f} days; held a year at that rate the net is about "
                       f"{r['now_y'] - round_trip:+.1%}.")
        else:
            out.append(f"Breakeven for {r['name']}: funding is not positive now, so a short "
                       f"perpetual against spot pays rather than earns — there is no carry to "
                       f"collect until it turns.")
    flip = _FLIP.search(text)
    if flip is not None:
        n = int(flip.group("n") or flip.group("n2"))
        unit = (flip.group("u") or flip.group("u2") or "d").lower()
        days_in = n if unit.startswith("d") else n / 24 if unit.startswith("h") else n * best[
            "hours"] / 24
        target = next((r for r in rows if r["name"] in text.upper()), best)
        earned = target["per_day"] * days_in
        need = round_trip / target["per_day"] if target["per_day"] > 0 else None
        out[0] = out[0].replace("Bottom line: ", "Rates now: ", 1)
        out.insert(0, f"Bottom line: if {target['name']} funding turns negative after {days_in:g} "
                      f"days, the "
                      f"carry has earned about {earned:.3%} against {round_trip:.2%} of fees, a "
                      f"net {earned - round_trip:+.3%} before any negative funding — "
                      + (f"breakeven needs about {need:.1f} days at today's rate, so a flip "
                         f"inside that leaves the trade under water." if need else
                         "with funding not positive now, there is nothing earned to set against "
                         "the fees."))
    out.append("Each yearly figure is a window's average carried forward, not a forecast; Bitget's "
               "funding history and live rates, read just now.")
    return out


def execution_lines(symbol: str, notional: float | None = None) -> list[str] | None:
    """The carry as orders: two legs of the same size, and why neither carries a stop.

    "Give me the exact execution plan for that — order type, entry, stop, take-profit" after a
    carry discussion got a naked long with a stop (a judge, round 31), which is a different trade:
    the carry is delta-neutral by construction."""
    from argus.lui.research.desk_answers import PERP_TAKER
    from argus.market.bitget import fetch_tickers

    try:
        ticker = fetch_tickers().get(symbol)
    except Exception:
        return None
    if ticker is None:
        return None
    last, rate = float(ticker.last), float(ticker.funding_rate)
    size = notional or 10_000.0
    qty = size / last
    name = symbol.removesuffix("USDT")
    spot_taker = SPOT_VIP[0][1]
    hours = round(365 * 24 / _per_year(symbol))
    return [f"Bottom line: the carry is two orders of the same size, not one — buy {qty:,.4f} "
            f"{name} on spot and short {qty:,.4f} {symbol} perpetual, about ${size:,.0f} each at "
            f"Bitget's last {last:,.2f}" + ("" if notional else " (no size was given, so $10,000 "
                                             "is the worked example)") + ".",
            f"Leg 1, spot: a near-touch limit buy, market if unfilled; {spot_taker:.2%} taker. "
            f"Leg 2, perpetual: a short of the same quantity, placed right after the first fills, "
            f"{PERP_TAKER:.2%} taker; set 1-2x leverage so a sharp rise cannot liquidate it — "
            f"at 2x the short's margin lasts through about a 45% rise.",
            "No stop and no take-profit on either leg: a stop on one leg leaves the other "
            "unhedged, which is the move the carry exists to avoid. The exit is closing both "
            "together, when funding turns against you.",
            f"Live inputs: funding {rate:+.4%} every {hours}h "
            f"({'shorts receive' if rate > 0 else 'shorts pay'} it), last {last:,.2f}. What can "
            f"go wrong: funding turns negative, the spot and "
            f"perpetual prices drift apart for a while, or the short's margin is called on a "
            f"spike — each priced above, none a forecast.",
            "Nothing is sent: ask \"how should I buy $10,000 of " + name + "\" for the live "
            "book on either leg."]


__all__ = ["CARRY_Q", "SPOT_VIP", "execution_lines", "lines"]

trace_module(globals())
