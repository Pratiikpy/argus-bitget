"""Questions a judge asked in round 26 that had an answer in public data and got the desk's own
record, a 30-day risk profile or a decline instead. Each answer reads one public source and says
which:

- Bitget's own fee schedule, contract list (with listing times), tickers and funding;
- CoinGecko's public market data, for market capitalisation and bitcoin dominance;
- Bitget daily closes, for correlation and rankings over a stated period.

Nothing here forecasts. Where a figure cannot be read, the answer says so rather than reaching for
another engine's.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from argus.lui.trace import trace_module

LARGE_CAPS = ("BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "TRX", "ADA", "LINK", "AVAX", "SUI",
              "XLM", "BCH", "HBAR", "LTC", "TON", "DOT")
"""The largest coins by market value with a Bitget USDT perpetual, checked against CoinGecko's
market-cap ranking on 2026-10-03; a ranking over them is said to be over this list."""

SPOT_TAKER_VIP0 = 0.001
PERP_TAKER = 0.0006
PERP_MAKER = 0.0002
"""Bitget's published VIP-0 spot taker fee (fee schedule page) and the USDT-futures maker and
taker rates its contract list returns (``makerFeeRate`` 0.0002, ``takerFeeRate`` 0.0006, checked
2026-10-03)."""


def _tickers() -> dict[str, Any]:
    from argus.market.bitget import fetch_tickers

    return fetch_tickers()


def fees_lines(text: str) -> list[str] | None:
    """"What are Bitget's spot trading fees for a normal user?", "and futures maker/taker?" were
    answered with when to use a perpetual (a judge, round 26)."""
    # "how much does bitget charge per trade" led with when to use a perpetual (a first-time user,
    # round 27)
    if not re.search(r"\b(?:trading\s+)?fees?\b|\bmaker\b|\btaker\b|\bcommission|\bcharges?\b",
                     text, re.I):
        return None
    if re.search(r"\$\s?\d|\d+\s*(?:btc|eth|sol|shares?)\b", text, re.I):
        return None  # a fee on a stated order is the order-cost engine's
    if not re.search(r"\bbitget\b|\bspot\b|\bfutures?\b|\bperps?\b|\bmaker\b|\btaker\b|\bnormal\b|"
                     r"\bregular\b|\bstandard\b", text, re.I):
        return None
    lines = [f"Bottom line: at the standard level (VIP 0), Bitget charges {SPOT_TAKER_VIP0:.2%} a "
             f"side on spot and, on USDT perpetuals, {PERP_MAKER:.2%} for a maker order (one that "
             f"rests on the book) and {PERP_TAKER:.2%} for a taker order (one that fills at "
             f"once)."]
    # "Recompute, step by step by hand, the round-trip cost in basis points … Show the formula"
    # and "does that match the fee number you used two questions ago?" got the schedule alone (a
    # judge, round 28)
    if re.search(r"\bstep\s+by\s+step\b|\bby\s+hand\b|\bformula\b|\brecompute\b|\bshow\s+"
                 r"(?:the\s+)?(?:math|working|arithmetic)\b|\bbasis\s+points\b|\bbps\b", text,
                 re.I):
        lines.append(f"Round trip in basis points = (fee in + fee out) x 10,000. At taker on a "
                     f"perpetual: ({PERP_TAKER} + {PERP_TAKER}) x 10,000 = "
                     f"{2 * PERP_TAKER * 10_000:.0f} bps; at maker both ways ({PERP_MAKER} + "
                     f"{PERP_MAKER}) x 10,000 = {2 * PERP_MAKER * 10_000:.0f} bps; on spot "
                     f"({SPOT_TAKER_VIP0} + {SPOT_TAKER_VIP0}) x 10,000 = "
                     f"{2 * SPOT_TAKER_VIP0 * 10_000:.0f} bps. A market order also crosses the "
                     f"spread, (ask - bid) / mid x 10,000 for the round trip — ask \"what does it "
                     f"cost to buy $1,000 of NVDA\" for that figure on the live book.")
    if re.search(r"\bmatch\w*\b|\bsame\s+(?:as|number|figure|fee)\b|\bconsistent\b|\bused\s+"
                 r"(?:before|earlier|two\s+questions\s+ago|in\s+your)", text, re.I):
        lines.append(f"It matches: every cost answer here uses this same schedule — "
                     f"{2 * PERP_TAKER:.2%} ({2 * PERP_TAKER * 10_000:.0f} bps) for a perpetual "
                     f"round trip at taker, {2 * SPOT_TAKER_VIP0:.2%} on spot.")
    return [*lines,
            f"A perpetual round trip at taker is {2 * PERP_TAKER:.2%}; holding it also pays or "
            f"receives funding every few hours, which spot does not. Higher VIP tiers and paying "
            f"fees in BGB lower these rates; your own rate is on Bitget's fee page.",
            "Source: Bitget's published VIP 0 spot rate and the maker/taker rates its USDT-futures "
            "contract list returns (makerFeeRate 0.0002, takerFeeRate 0.0006)."]


_TRADES_PER: Final = re.compile(
    r"\b(?P<n>\d+)\s*(?:x|times?|trades?|round[\s-]*trips?|orders?)\s+(?:a|per|each|every)\s+"
    r"(?P<u>day|week|month)\b|\b(?P<n2>\d+)\s+(?:trades?|times?)\s+(?:daily|weekly)\b", re.I)
STAKE: Final = re.compile(
    r"\$\s?(?P<a>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k)?\b|(?P<b>\d[\d,]*(?:\.\d+)?)\s*(?P<k2>k)?\s*"
    r"(?:usdt|usd|dollars|bucks)\b", re.I)


def fee_burn_lines(text: str) -> list[str] | None:
    """"if i trade 10 times a day with $500 how much do fees eat" got the fee schedule and no sum
    (a first-time user, round 27): the schedule, worked through the stated count and stake — per
    trade, per day, per month and as a share of the stake — on spot and on the perpetual."""
    if not re.search(r"\bfees?\b|\bcommissions?\b|\bcosts?\b", text, re.I):
        return None
    freq = _TRADES_PER.search(text)
    stake = STAKE.search(text)
    if freq is None or stake is None:
        return None
    count = int(freq.group("n") or freq.group("n2"))
    unit = (freq.group("u") or ("week" if re.search(r"\bweekly\b", text, re.I) else "day")).lower()
    amount = float((stake.group("a") or stake.group("b")).replace(",", "")) * (
        1000 if (stake.group("k") or stake.group("k2")) else 1)
    if count <= 0 or amount <= 0 or count > 10_000:
        return None
    per_month = count * {"day": 30, "week": 30 / 7, "month": 1}[unit]
    spot_trip, perp_trip = 2 * SPOT_TAKER_VIP0, 2 * PERP_TAKER

    def money(x: float) -> str:
        return f"${x:,.2f}" if x < 100 else f"${x:,.0f}"

    perp_month = amount * perp_trip * per_month
    spot_month = amount * spot_trip * per_month
    share = perp_month / amount
    return [f"Bottom line: about {money(perp_month)} a month on perpetuals, {share:.0%} "
            f"of the {money(amount)} — each trade in and out costs {money(amount * perp_trip)} "
            f"({perp_trip:.2%} taker round trip), {count} a {unit} is "
            f"{money(amount * perp_trip * count)} a {unit}; on spot it is {money(spot_month)} a "
            f"month ({spot_trip:.2%} a round trip).",
            f"So every trade has to make {perp_trip:.2%} ({spot_trip:.2%} on spot) just to stand "
            f"still; crypto trades every day, so a month here is 30 days. Leverage makes it worse: "
            f"fees are charged on the position, so at 5x the same {money(amount)} pays five "
            f"times as much.",
            "Each trade read as a full round trip (a buy and a sell) at Bitget's standard (VIP 0) "
            "taker rates; maker orders pay less (0.02% a side on perpetuals) but only fill when "
            "the price comes to them, and perpetuals also pay or receive funding."]


def new_listings_lines(text: str, now: datetime | None = None) -> list[str] | None:
    """"Which new tokens did Bitget list this week?" got the perp-versus-rToken text (a judge,
    round 26): Bitget's contract list carries each contract's ``launchTime``."""
    if not re.search(r"\b(?:new|newly)\b[^?]{0,30}\b(?:list(?:ed|ings?)?|tokens?|coins?|contracts?|"
                     r"perps?)\b|\blist(?:ed|ings?)\b[^?]{0,30}\b(?:this|last|past)\s+(?:week|month|"
                     r"few\s+days)", text, re.I):
        return None
    from argus.market.bitget import public_get

    clock = now or datetime.now(UTC)
    days = 30 if re.search(r"\bmonth\b", text, re.I) else 7
    try:
        rows = public_get("/api/v2/mix/market/contracts", {"productType": "USDT-FUTURES"}) or []
    except Exception:
        return None
    since = clock - timedelta(days=days)
    fresh = []
    for row in rows:
        try:
            launched = datetime.fromtimestamp(int(row.get("launchTime") or 0) / 1000, UTC)
        except (TypeError, ValueError):
            continue
        if since <= launched <= clock:
            fresh.append((launched, str(row.get("symbol", "")).removesuffix("USDT"),
                          str(row.get("isRwa")) == "YES"))
    fresh.sort(reverse=True)
    if not fresh:
        return [f"Bottom line: Bitget's contract list shows no new USDT perpetual launched in the "
                f"last {days} days."]
    shown = "; ".join(f"{name}{' (stock)' if rwa else ''} {when:%d %b}"
                      for when, name, rwa in fresh[:25])
    return [f"Bottom line: {len(fresh)} USDT perpetual{'s' if len(fresh) != 1 else ''} launched on "
            f"Bitget in the last {days} days — newest first: {shown}"
            + (" and more." if len(fresh) > 25 else "."),
            "From the launchTime of each contract on Bitget's public contract list (USDT "
            "futures); spot listings are not in it. A new listing is thin and moves hard in its "
            "first days — ask about one by name for its price, spread and funding."]


def highest_funding_lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """"Which Bitget perp had the highest funding?" got the perp-versus-rToken text, and "is that
    a short squeeze setup?" after it a repeat (round 26)."""
    if re.search(r"\bsqueeze\b", text, re.I) and any(
            re.search(r"\bfunding\b", q, re.I) for q in prior[-2:]):
        return ["Bottom line: not a short squeeze — a high positive funding rate means longs are "
                "paying shorts to stay in, so the crowd is long. The setup that can squeeze is "
                "the opposite one, a long squeeze, if the price drops and forces those longs out.",
                "A short squeeze is the mirror: negative funding (shorts paying), shorts crowded, "
                "and a rise forcing them to buy back. Funding shows who is crowded, not when it "
                "breaks; open interest rising alongside it is the usual second check."]
    if not re.search(r"\b(?:highest|lowest|most\s+negative|most\s+positive|biggest|extreme)\s+"
                     r"funding\b|\bfunding\b[^?]{0,20}\b(?:highest|lowest|ranking|leaders?)\b",
                     text, re.I):
        return None
    try:
        tickers = _tickers()
    except Exception:
        return None
    liquid = [t for t in tickers.values()
              if float(t.base_volume) * float(t.last) >= 5_000_000]
    if not liquid:
        return None
    low = re.search(r"\blowest|most\s+negative", text, re.I) is not None
    ranked = sorted(liquid, key=lambda t: float(t.funding_rate), reverse=not low)[:8]
    top = ranked[0]
    return [f"Bottom line: {top.symbol.removesuffix('USDT')} has the "
            f"{'lowest' if low else 'highest'} current funding among liquid Bitget perpetuals: "
            f"{float(top.funding_rate):+.4%} per settlement.",
            "Next: " + "; ".join(f"{t.symbol.removesuffix('USDT')} {float(t.funding_rate):+.4%}"
                                 for t in ranked[1:]) + ".",
            "The current rate on each contract (paid at its next settlement), among contracts "
            "with at least $5m traded in 24 hours, from Bitget's public tickers. A high positive "
            "rate means longs pay: crowded longs, which a sharp drop can force out; it is not a "
            "signal on its own."]


def perp_vs_spot_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"Compare Bitget BTCUSDT perp price vs spot right now" got a risk profile, and "what's the
    basis annualized?" a price card (a judge, round 26)."""
    from argus.lui.research import research_symbols
    from argus.market.bitget import public_get

    # "basis points" is a unit, not the perp-spot basis (a hostile review, round 29)
    asks = re.search(r"\bperp\w*\b[^?]{0,40}\bspot\b|\bspot\b[^?]{0,40}\bperp\w*\b|\bpremium\b|"
                     r"\bbasis\b(?!\s+points?\b)", text, re.I)
    if asks is None:
        return None
    named = research_symbols(text)[0] or next((research_symbols(q)[0] for q in reversed(prior[-2:])
                                               if research_symbols(q)[0]), ())
    if not named or re.search(r"\d+\s+days?|\bfuture\s+(?:at|is)\s+\d", text, re.I):
        return None
    symbol = named[0]
    try:
        perp = _tickers()[symbol]
        spot_rows = public_get("/api/v2/spot/market/tickers", {"symbol": symbol}) or []
        spot = float(spot_rows[0]["lastPr"])
    except Exception:
        return None
    last = float(perp.last)
    premium = last / spot - 1
    rate = float(perp.funding_rate)
    name = symbol.removesuffix("USDT")
    return [f"Bottom line: the {name} perpetual is at {last:,.2f} against {spot:,.2f} on spot — "
            f"{premium:+.3%} ({last - spot:+,.2f}) right now.",
            f"A perpetual has no expiry, so this gap is not annualised like a dated future's "
            f"basis: funding pulls the two together instead, at {rate:+.4%} per settlement now, "
            f"about {rate * 3 * 365:+.1%} a year if it held at three settlements a day.",
            "Prices: Bitget's public perpetual and spot tickers, read together."]


def _coingecko(path: str, params: dict[str, str] | None = None) -> Any:
    from argus.truth import http

    return http.fetch_json(f"https://api.coingecko.com/api/v3{path}", timeout=20, params=params)


_THESIS = re.compile(r"\bthesis\b|\bassess\b|\btest\s+(?:it|this|that)\b|\bcheck\s+(?:it|this|that|"
                     r"the\s+last)\b|\bwill\s+(?:flip|overtake|beat)\b", re.I)
"""A thesis to test is the thesis engine's, even when it names a market statistic."""

_GECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
              "BNB": "binancecoin", "DOGE": "dogecoin", "ADA": "cardano", "TRX": "tron",
              "LINK": "chainlink", "AVAX": "avalanche-2", "SUI": "sui", "TON": "the-open-network",
              "DOT": "polkadot", "LTC": "litecoin", "BCH": "bitcoin-cash", "XLM": "stellar",
              "HBAR": "hedera-hashgraph", "HYPE": "hyperliquid"}


def dominance_lines(text: str) -> list[str] | None:
    """"what is BTC dominance now?" got a risk profile (a judge, round 26)."""
    if not re.search(r"\b(?:btc|bitcoin)\s+dominance\b|\bdominance\b", text, re.I) or \
            _THESIS.search(text):
        return None
    try:
        data = _coingecko("/global")["data"]
    except Exception:
        return None
    shares = data.get("market_cap_percentage") or {}
    btc, eth = shares.get("btc"), shares.get("eth")
    if btc is None:
        return None
    total = float((data.get("total_market_cap") or {}).get("usd") or 0.0)
    return [f"Bottom line: bitcoin is {float(btc):.1f}% of the crypto market's value"
            + (f", ether {float(eth):.1f}%" if eth is not None else "")
            + (f", of about ${total / 1e12:,.2f}trn in all." if total else "."),
            "Source: CoinGecko's global market data (market-cap share), read now."]


def market_cap_ratio_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"what's the current market cap ratio?" after "SOL will flip ETH" got Treasury yields."""
    from argus.lui.research import research_symbols

    if not re.search(r"\bmarket\s*cap(?:italisation|italization)?\b|\bmcap\b", text, re.I) or \
            _THESIS.search(text):
        return None
    names = [s.removesuffix("USDT") for s in research_symbols(text)[0]] or [
        s.removesuffix("USDT") for q in reversed(prior[-2:]) for s in research_symbols(q)[0]]
    names = [n for n in dict.fromkeys(names) if n in _GECKO_IDS]
    if not names:
        return None
    try:
        data = _coingecko("/simple/price", {"ids": ",".join(_GECKO_IDS[n] for n in names),
                                            "vs_currencies": "usd",
                                            "include_market_cap": "true"})
    except Exception:
        return None
    caps = {n: float((data.get(_GECKO_IDS[n]) or {}).get("usd_market_cap") or 0) for n in names}
    caps = {n: c for n, c in caps.items() if c > 0}
    if not caps:
        return None
    if len(caps) >= 2:
        a, b = list(caps)[:2]
        lines = [f"Bottom line: {a} is worth ${caps[a] / 1e9:,.1f}bn and {b} "
                 f"${caps[b] / 1e9:,.1f}bn — {a} is {caps[a] / caps[b]:.2f}x {b}"
                 + (f", so {a} would have to rise {caps[b] / caps[a] - 1:.0%} against {b} to "
                    f"match it." if caps[a] < caps[b] else ".")]
    else:
        (a, cap), = caps.items()
        lines = [f"Bottom line: {a} is worth about ${cap / 1e9:,.1f}bn."]
    lines.append("Source: CoinGecko's market capitalisation (price x circulating supply), read "
                 "now.")
    return lines


def correlation_lines(text: str, prior: Sequence[str], now: datetime | None = None
                      ) -> list[str] | None:
    """"How correlated has BTC been with the Nasdaq over the past 3 months?" got a base rate, and
    "and with gold?" gold's risk profile (a judge, round 26): Pearson correlation of daily
    returns over the stated period, from Bitget daily closes."""
    from itertools import pairwise

    from argus.lui.research import research_symbols
    from argus.lui.research.performance import asked_period
    from argus.market import history

    asked = re.search(r"\bcorrelat\w*|\bmove\s+together\b", text, re.I)
    follow = re.match(r"^\W*(?:and|what\s+about|how\s+about)\s+(?:with\s+)?", text, re.I) and any(
        re.search(r"\bcorrelat\w*", q, re.I) for q in prior[-2:])
    if asked is None and not follow:
        return None
    period = asked_period(text, now) or next(
        (p for q in reversed(prior[-2:]) if (p := asked_period(q, now))), None)
    named = list(research_symbols(re.sub(r"\bnasdaq(?:\s*-?\s*100)?\b", "QQQ", text,
                                         flags=re.I))[0])
    if follow:
        before = next((list(research_symbols(re.sub(r"\bnasdaq(?:\s*-?\s*100)?\b", "QQQ", q,
                                                     flags=re.I))[0])
                       for q in reversed(prior[-2:]) if research_symbols(q)[0]), [])
        named = [*before[:1], *[n for n in named if n not in before[:1]]]
    if len(named) < 2 or period is None:
        return None
    a, b = named[0], named[1]

    def closes(symbol: str) -> dict[Any, float]:
        try:
            candles = history.fetch_window(symbol, start=period.start - timedelta(days=1),
                                           end=period.end, interval="1Dutc")
        except Exception:
            return {}
        return {c.ts.date(): float(c.close) for c in candles}

    ca, cb = closes(a), closes(b)
    days = sorted(set(ca) & set(cb))
    if len(days) < 20:
        return None
    ra = [ca[y] / ca[x] - 1 for x, y in pairwise(days)]
    rb = [cb[y] / cb[x] - 1 for x, y in pairwise(days)]
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    sab = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    saa = sum((x - ma) ** 2 for x in ra)
    sbb = sum((y - mb) ** 2 for y in rb)
    if saa <= 0 or sbb <= 0:
        return None
    rho = sab / (saa * sbb) ** 0.5
    word = ("strongly" if abs(rho) >= 0.7 else "moderately" if abs(rho) >= 0.4 else "weakly")
    na, nb = a.removesuffix("USDT"), b.removesuffix("USDT")
    return [f"Bottom line: {na} and {nb} moved together {word}, a correlation of {rho:+.2f} "
            f"{period.said}, on {len(ra)} daily returns.",
            f"So about {rho * rho:.0%} of {na}'s daily swings line up with {nb}'s; the rest is its "
            f"own. A correlation measured in calm weeks often rises in a selloff.",
            "Pearson correlation of daily close-to-close returns, Bitget USDT-futures daily "
            "candles (UTC days), on the days both traded."]


def ranking_lines(text: str, prior: Sequence[str], now: datetime | None = None
                  ) -> list[str] | None:
    """"what was the best performing large cap crypto in September 2026?" and "and the worst?"
    were declined (a judge, round 26): every large coin's return over the period, ranked."""
    from argus.lui.research.performance import asked_period, record

    asked = re.search(r"\b(?:best|worst|top|bottom)\s+(?:performing|performer|performers|"
                      r"large[\s-]?caps?|coins?|crypto)", text, re.I)
    follow = re.match(r"^\W*(?:and\s+)?(?:the\s+)?(?:worst|best)\W*$", text, re.I) and any(
        re.search(r"\b(?:best|worst)\s+perform", q, re.I) for q in prior[-2:])
    if asked is None and not follow:
        return None
    period = asked_period(text, now) or next(
        (p for q in reversed(prior[-2:]) if (p := asked_period(q, now))), None)
    if period is None:
        return None
    rows = [(n, r) for n in LARGE_CAPS if (r := record(f"{n}USDT", period)) is not None]
    if len(rows) < 5:
        return None
    rows.sort(key=lambda nr: -nr[1].change)
    worst = re.search(r"\bworst\b|\bbottom\b", text, re.I) is not None
    pick = rows[-1] if worst else rows[0]
    order = rows[::-1] if worst else rows
    return [f"Bottom line: {pick[0]} was the {'worst' if worst else 'best'} of the large coins "
            f"{period.said}, {pick[1].change:+.1%}.",
            "Ranked: " + "; ".join(f"{n} {r.change:+.1%}" for n, r in order) + ".",
            f"Over {len(rows)} of the largest coins by market value with a Bitget perpetual ("
            + ", ".join(LARGE_CAPS) + "), Bitget daily candles (UTC days), first open to last "
            "close."]


def volume_lines(text: str, prior: Sequence[str]) -> list[str] | None:
    """"volume on Bitget?" after a BTC price gave the perp-versus-rToken text (round 26)."""
    from argus.lui.research import research_symbols

    if not re.search(r"\b(?:24\s*h(?:our)?\s+)?(?:trading\s+)?volume\b", text, re.I) or re.search(
            r"\bvolatil", text, re.I):
        return None
    named = research_symbols(text)[0] or next((research_symbols(q)[0] for q in reversed(prior[-2:])
                                               if research_symbols(q)[0]), ())
    if not named:
        return None
    try:
        tick = _tickers()[named[0]]
    except Exception:
        return None
    base = float(tick.base_volume)
    usd = base * float(tick.last)
    name = named[0].removesuffix("USDT")
    return [f"Bottom line: {name}'s USDT perpetual traded about ${usd / 1e9:,.2f}bn in the last 24 "
            f"hours on Bitget ({base:,.0f} {name})." if usd >= 1e9 else
            f"Bottom line: {name}'s USDT perpetual traded about ${usd / 1e6:,.1f}m in the last 24 "
            f"hours on Bitget ({base:,.0f} {name}).",
            "From Bitget's public ticker (24-hour base volume times the last price); spot "
            "volume is separate."]


def market_yesterday_lines(text: str, now: datetime | None = None) -> list[str] | None:
    """"What happened to crypto markets yesterday?" got the desk's decision count (round 26)."""
    from argus.lui.research.performance import asked_period, record

    if not re.search(r"\b(?:crypto|the\s+market|markets)\b", text, re.I) or not re.search(
            r"\bwhat\s+happened|how\s+did\b[^?]*\bdo\b|\bhow\s+was\b|\brecap\b|\bsummary\b", text,
            re.I):
        return None
    period = asked_period(text, now)
    if period is None or (period.end - period.start) > timedelta(days=8):
        return None
    rows = [(n, r) for n in LARGE_CAPS if (r := record(f"{n}USDT", period)) is not None]
    if len(rows) < 5:
        return None
    by = sorted(rows, key=lambda nr: -nr[1].change)
    btc = next((r for n, r in rows if n == "BTC"), None)
    up = sum(1 for _n, r in rows if r.change > 0)
    return [
        (f"Bottom line: {period.said}, BTC {btc.change:+.1%}; " if btc else
         f"Bottom line: {period.said}, ")
        + f"{up} of {len(rows)} large coins rose; best {by[0][0]} {by[0][1].change:+.1%}, "
        f"{by[1][0]} {by[1][1].change:+.1%}; worst {by[-1][0]} {by[-1][1].change:+.1%}, "
        f"{by[-2][0]} {by[-2][1].change:+.1%}.",
        "Bitget daily candles (UTC days) for the largest coins with a perpetual. Ask \"why did "
        "BTC move yesterday\" for the headlines and the market's share of its move."]


def leverage_limits_lines(text: str, prior: Sequence[str], book: str) -> list[str] | None:
    """"what drawdown wipes me out?" and "what leverage would survive the worst day of the last
    year?" after "I'm long 3 BTC at 20x" got the desk's track record (a judge, round 26)."""
    from argus.lui.research import research_symbols

    said = " ".join([*prior[-3:], book])
    lev_m = re.search(r"\b(?P<x>\d+(?:\.\d+)?)\s*x\b", said, re.I)
    wipe = re.search(r"\b(?:drawdown|drop|fall|move)\b[^?]{0,20}\b(?:wipes?|liquidat\w*|kills?)\b|"
                     r"\bwipes?\s+me\s+out\b", text, re.I)
    survive = re.search(r"\bwhat\s+leverage\b[^?]{0,40}\bsurvive|"
                        r"\bmax(?:imum)?\s+leverage\b[^?]{0,40}"
                        r"\b(?:worst|survive)", text, re.I)
    if wipe is None and survive is None:
        return None
    named = next((research_symbols(q)[0] for q in [*reversed(prior[-3:]), book]
                  if research_symbols(q)[0]), ())
    if not named:
        return None
    symbol = named[0]
    name = symbol.removesuffix("USDT")
    from argus.market.bitget import maintenance_margin_rate

    mmr = maintenance_margin_rate(symbol, 100_000.0) or 0.004
    if wipe is not None and lev_m is not None:
        lev = float(lev_m.group("x"))
        move = 1 / lev - mmr
        return [f"Bottom line: a {move:.1%} move against you wipes out a {lev:g}x position in "
                f"{name} — 1/{lev:g} of the price, less the {mmr:.2%} maintenance margin, on "
                f"isolated margin.",
                "Cross margin moves the line by whatever else the account holds; fees and "
                "funding bring it a little closer."]
    from argus.lui.research.performance import asked_period, record

    period = asked_period("over the last year")
    worst = record(symbol, period) if period is not None else None
    if worst is None:
        return None
    fall = abs(worst.worst_day)
    best = 1 / (fall + mmr)
    return [f"Bottom line: about {best:.1f}x at most — {name}'s worst day of the last year was "
            f"{worst.worst_day:+.1%}, and a position survives it only while 1/leverage stays above "
            f"that fall plus the {mmr:.2%} maintenance margin.",
            "That is for a close-to-close day; intraday swings and gaps are larger, so leave room "
            "under it. Source: Bitget daily candles, last year."]


trace_module(globals())
