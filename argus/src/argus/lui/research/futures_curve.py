"""Commodity futures term structure: backwardation or contango, how steep, and what it pays.

Round 43's judge asked "is crude in backwardation or contango and what does the curve say?" and
"is the gold futures curve in contango?". The console had no futures reader: it answered with the
spot-like perpetual price, or with a Treasury curve, and never with the shape of the futures chain.

**Source.** Yahoo Finance's daily chart (``query1.finance.yahoo.com/v8/finance/chart/{symbol}``,
keyless) answers one symbol per contract month: ``{root}{month code}{yy}.{exchange}``, with the
month codes ``F G H J K M N Q U V X Z`` for January to December. Checked live on 2026-10-06
(every root, every month of 2026 to 2028):

* WTI ``CL`` and Brent ``BZ`` on ``NYM``, natural gas ``NG`` on ``NYM``, copper ``HG``, gold
  ``GC`` and silver ``SI`` on ``CMX``. ``meta.instrumentType`` is ``FUTURE``.
* An expired contract is simply gone (``CLV26.NYM``, ``BZX26.NYM``: "No data found") or, worse,
  still answers with a price from the day it died (``CLQ26.NYM`` 84.99 stamped 21 Jul). Thin far
  months answer too, with a last trade days or weeks old (``HGJ27`` 1 Oct, ``GCX27`` 16 Sep,
  ``SIQ27`` 16 Jul), which would put a stale print beside live ones and invent a kink. So a contract
  counts only if its last trade is within one calendar day of the newest contract's.
* Figures that day: WTI Nov 2026 89.18, Dec 87.95, Jan 2027 86.86, Dec 2027 76.16; Brent Dec 2026
  100.26 (Nov is gone, Brent expires two months ahead); natural gas Nov 3.075, Dec 3.363, Jan 3.735,
  Apr 2027 2.705; copper Oct 6.58, Dec 6.638; gold Oct 4134.8, Dec 4168.1; silver Dec 61.42.

**Method.** The front is the nearest contract that is fresh. The 2nd, 3rd, 6th and 12th months are
the contracts that many calendar months after the front's delivery month (the nearest fresh one
within a month when that exact month is not traded: gold and silver trade thinly in off months).
Spread is deferred minus front, in dollars and percent. Annualised roll yield is
``(front / deferred) ** (12 / months) - 1`` with ``months`` the gap between delivery months: what a
long holder who rolls each month earns (backwardation, positive) or pays (contango, negative) if the
curve stays where it is. Delivery months differ from expiry dates by up to a month, so it is an
estimate and the answer calls it one. A month ago is read from the same two contracts' history
(three-month daily chart), so "steeper" never compares different contracts.

**Bitget.** The USDT-futures tickers (``/api/v2/mix/market/tickers``) give price and funding rate;
the contract list (``argus.market.universe``) gives the funding interval (4 hours on all of these).
Checked 2026-10-06: ``CLUSDT`` 89.22, ``BZUSDT``, ``NATGASUSDT`` 3.07, ``COPPERUSDT`` 6.65,
``XAUUSDT`` 4143.3, ``XAUTUSDT`` and ``PAXGUSDT`` (gold-backed tokens), ``XAGUSDT`` 61.1. Bitget's
``GASUSDT`` is the Neo Gas token, not natural gas, and is deliberately not matched. A perpetual
never expires, so it does not carry the roll the curve shows; its funding rate plays that role.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

from argus.truth import http
from argus.truth.bounded import BoundedDict
from argus.truth.endpoints import BITGET_API

CHART: Final = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=3mo&interval=1d"
TICKERS: Final = BITGET_API + "/api/v2/mix/market/tickers?productType=USDT-FUTURES"
MINUS: Final = chr(0x2212)
CODES: Final = "FGHJKMNQUVXZ"
MONTH_NAMES: Final = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                      "Nov", "Dec")
HORIZON_MONTHS: Final = 14
"""Delivery months tried from the current one: enough to reach the 12th month after a front that
is itself a month or two out."""
FRESH_DAYS: Final = 1
FLAT: Final = 0.001
"""A spread under 0.1% of the front is called flat."""
STEADY_POINTS: Final = 0.3
"""A change of the spread under 0.3 percentage points in a month is called little changed."""
CACHE_SECONDS: Final = 600.0
TENORS: Final = ((2, 1), (3, 2), (6, 5), (12, 11))
"""(ordinal month, calendar months after the front's delivery month)."""


@dataclass(frozen=True, slots=True)
class Commodity:
    key: str
    name: str
    root: str
    exchange: str
    unit: str
    perps: tuple[str, ...]


COMMODITIES: Final = {
    "wti": Commodity("wti", "WTI crude", "CL", "NYM", "$/bbl", ("CLUSDT",)),
    "brent": Commodity("brent", "Brent crude", "BZ", "NYM", "$/bbl", ("BZUSDT",)),
    "natgas": Commodity("natgas", "natural gas", "NG", "NYM", "$/MMBtu", ("NATGASUSDT",)),
    "copper": Commodity("copper", "copper", "HG", "CMX", "$/lb", ("COPPERUSDT",)),
    "gold": Commodity("gold", "gold", "GC", "CMX", "$/oz", ("XAUUSDT", "XAUTUSDT", "PAXGUSDT")),
    "silver": Commodity("silver", "silver", "SI", "CMX", "$/oz", ("XAGUSDT",)),
}
PERP_NOTES: Final = {
    "XAUTUSDT": "perpetual on the gold-backed XAUT token", "PAXGUSDT":
        "perpetual on the gold-backed PAXG token", "BZUSDT": "Brent",
    "CLUSDT": "WTI",
}

_VETO: Final = re.compile(
    r"\b(?:yields?|treasur\w*|bonds?|bunds?|gilts?|t-?bills?|vix|volatil\w*|bitcoin|btc|ether|"
    r"eth|sol|options?|swaps?|fed funds|interest rates?)\b", re.I)
_CURVE: Final = re.compile(
    r"\b(?:backwardat\w*|contango|term[- ]structure|(?:futures?|forward|forwards|commodity) "
    r"curves?|curves?|roll[- ](?:yield|cost|return)|forward prices?|futures? chain)\b", re.I)
_ROLL: Final = re.compile(r"roll[- ]yields?", re.I)
_CONTRACTS: Final = re.compile(
    r"\b(?:contracts?|perps?|perpetuals?|futures|markets?|instruments?)\b", re.I)
_TRADE: Final = re.compile(r"\b(?:trade|trading|exposure|bitget|play|bet on|get long|go long)\b",
                           re.I)
_WTI: Final = re.compile(r"\bwti\b|\bwest texas\b", re.I)
_BRENT: Final = re.compile(r"\bbrent\b", re.I)
_OIL: Final = re.compile(r"\b(?:oil|crude)\b", re.I)
_GAS: Final = re.compile(r"\bnat(?:ural)?[- ]?gas\b|\bnatgas\b", re.I)
_COPPER: Final = re.compile(r"\bcopper\b", re.I)
_GOLD: Final = re.compile(r"\bgold\b", re.I)
_SILVER: Final = re.compile(r"\bsilver\b", re.I)
_BROAD: Final = re.compile(r"\bcommodit(?:y|ies)\b", re.I)


@dataclass(frozen=True, slots=True)
class Series:
    symbol: str
    price: float
    last: date
    closes: tuple[tuple[date, float], ...]


@dataclass(frozen=True, slots=True)
class Leg:
    year: int
    month: int
    series: Series

    @property
    def index(self) -> int:
        return self.year * 12 + self.month - 1

    @property
    def label(self) -> str:
        return f"{MONTH_NAMES[self.month - 1]} {self.year}"


@dataclass(frozen=True, slots=True)
class Perp:
    symbol: str
    price: float
    funding: float
    hours: int | None


_cache: BoundedDict[str, tuple[float, Series | None]] = BoundedDict(512)
_lock = threading.Lock()


def commodities_asked(text: str) -> list[Commodity]:
    """The commodities a question names, in the order a reader expects (oil before metals)."""
    keys: list[str] = []
    if _WTI.search(text):
        keys.append("wti")
    if _BRENT.search(text):
        keys.append("brent")
    if _OIL.search(text):
        keys += [k for k in ("wti", "brent") if k not in keys]
    for pattern, key in ((_GAS, "natgas"), (_COPPER, "copper"), (_GOLD, "gold"),
                         (_SILVER, "silver")):
        if pattern.search(text):
            keys.append(key)
    if not keys and _BROAD.search(text):
        keys = list(COMMODITIES)
    return [COMMODITIES[k] for k in keys]


def asks_for_contracts(text: str) -> bool:
    """True for "which Bitget contracts could I use to trade oil?" rather than a curve question."""
    return bool(_CONTRACTS.search(text) and _TRADE.search(text))


def asked(text: str) -> bool:
    """The wiring condition: a commodity is named and the question is about its futures curve, or
    about which contracts trade it; rate, bond, volatility and crypto curves are never ours."""
    if _VETO.search(_ROLL.sub("", text)) or not commodities_asked(text):
        return False
    return bool(_CURVE.search(text) or asks_for_contracts(text))


def symbol_for(commodity: Commodity, year: int, month: int) -> str:
    return f"{commodity.root}{CODES[month - 1]}{year % 100:02d}.{commodity.exchange}"


def _chart(symbol: str) -> dict[str, Any]:
    body: dict[str, Any] = http.fetch_json(
        CHART.format(symbol=symbol), timeout=15.0,
        headers={"User-Agent": "Mozilla/5.0 argus-research"})
    return body


def _parse(symbol: str, body: dict[str, Any]) -> Series | None:
    try:
        result = body["chart"]["result"][0]
        meta = result["meta"]
        price = float(meta["regularMarketPrice"])
        last = datetime.fromtimestamp(int(meta["regularMarketTime"]), tz=UTC).date()
        stamps = result.get("timestamp") or []
        raw = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if price <= 0:
        return None
    closes = tuple((datetime.fromtimestamp(int(t), tz=UTC).date(), float(c))
                   for t, c in zip(stamps, raw, strict=False) if c)
    return Series(symbol, price, last, closes)


def series(symbol: str) -> Series | None:
    """One contract's price, last-trade date and daily closes; None when Yahoo has none (expired
    or never listed) or did not answer."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(symbol)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    try:
        found = _parse(symbol, _chart(symbol))
    except http.RpcError:
        found = None
    with _lock:
        _cache[symbol] = (now, found)
    return found


def _months_from(today: date) -> list[tuple[int, int]]:
    base = today.year * 12 + today.month - 1
    return [divmod(base + i, 12) for i in range(HORIZON_MONTHS)]


def chain(commodity: Commodity, today: date) -> list[Leg]:
    """Every contract month that answers with a current price, nearest first."""
    wanted = [(y, m + 1) for y, m in _months_from(today)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda ym: series(symbol_for(commodity, *ym)), wanted))
    legs = [Leg(y, m, s) for (y, m), s in zip(wanted, got, strict=True) if s is not None]
    if not legs:
        return []
    newest = max(leg.series.last for leg in legs)
    return [leg for leg in legs if (newest - leg.series.last).days <= FRESH_DAYS]


def pick(legs: Sequence[Leg], front: Leg, ahead: int) -> Leg | None:
    """The leg ``ahead`` delivery months after the front, else the nearest within one month (the
    earlier on a tie), never the front itself."""
    target = front.index + ahead
    options = [leg for leg in legs if leg.index > front.index and abs(leg.index - target) <= 1]
    if not options:
        return None
    return min(options, key=lambda leg: (abs(leg.index - target), leg.index))


@dataclass(frozen=True, slots=True)
class Step:
    ordinal: int
    leg: Leg
    months: int
    dollars: float
    pct: float
    roll: float
    exact: bool


def steps(legs: Sequence[Leg]) -> tuple[Leg, list[Step]]:
    front = legs[0]
    out: list[Step] = []
    used = {front.index}
    for ordinal, ahead in TENORS:
        leg = pick(legs, front, ahead)
        if leg is None or leg.index in used:
            continue
        used.add(leg.index)
        months = leg.index - front.index
        f, d = front.series.price, leg.series.price
        out.append(Step(ordinal, leg, months, d - f, d / f - 1.0, (f / d) ** (12 / months) - 1.0,
                        leg.index == front.index + ahead))
    return front, out


def shape(pct: float) -> str:
    if abs(pct) < FLAT:
        return "flat"
    return "backwardation" if pct < 0 else "contango"


def overall(out: Sequence[Step]) -> str:
    """One word for the whole chain, or "mixed" when its near and far ends disagree."""
    kinds = {shape(s.pct) for s in out}
    kinds.discard("flat")
    if not kinds:
        return "flat"
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _close_near(s: Series, day: date) -> float | None:
    """The close on ``day`` or the latest one up to four days before it."""
    earlier = [(d, c) for d, c in s.closes if day - timedelta(days=4) <= d <= day]
    return earlier[-1][1] if earlier else None


def month_ago(front: Leg, step: Step) -> float | None:
    """The same two contracts' spread, in percent of the front, a month before the front's last
    trade; None when either has no close then."""
    day = front.series.last - timedelta(days=30)
    then_f, then_d = _close_near(front.series, day), _close_near(step.leg.series, day)
    if not then_f or not then_d:
        return None
    return (then_d / then_f - 1.0) * 100


def _money(value: float) -> str:
    places = 3 if abs(value) < 20 else 2
    return f"{value:,.{places}f}"


def _signed(value: float, places: int = 2) -> str:
    return f"{value:+,.{places}f}".replace("-", MINUS)


def _pct(value: float, places: int = 1) -> str:
    return f"{value * 100:+.{places}f}%".replace("-", MINUS)


def _ordinal(n: int) -> str:
    return {2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def curve_lines(c: Commodity, legs: Sequence[Leg]) -> tuple[str, list[str]]:
    """(the bottom-line clause for this commodity, its supporting lines)."""
    front, out = steps(legs)
    unit = c.unit
    if not out:
        return (f"{c.name} has only one fresh futures month ({front.label}, "
                f"{_money(front.series.price)} {unit}), so no curve can be drawn", [])
    word = overall(out)
    rolled = out[0] if word == "mixed" else out[2] if len(out) > 2 else out[-1]
    path = ", ".join(f"{_pct(s.pct)} to {s.leg.label}" for s in out)
    opening = (f"{c.name} is in {word}" if word != "mixed"
               else f"{c.name} is not one shape (near and far months disagree)")
    head = (f"{opening}: front {front.label} {_money(front.series.price)} {unit}, {path}; "
            f"roll yield about {_pct(rolled.roll, 0)} a year to {rolled.leg.label}")
    chain_bits = [f"front {front.label} {_money(front.series.price)}"]
    chain_bits += [f"{_ordinal(s.ordinal)} {s.leg.label} {_money(s.leg.series.price)} "
                   f"({_signed(s.dollars)}, {_pct(s.pct)}"
                   + ("" if s.exact else f"; {s.months} months out, nearest traded month") + ")"
                   for s in out]
    lines = [f"{c.name} curve ({c.root} on {c.exchange}, {unit}, {front.series.last:%d %b %Y}): "
             + "; ".join(chain_bits) + "."]
    rolls = "; ".join(f"{_ordinal(s.ordinal)} month {_pct(s.roll)}" for s in out)
    lines.append(f"{c.name} roll yield, annualised from front to each month (estimate, delivery "
                 f"months stand in for expiry dates): {rolls}.")
    if word == "mixed" and c.key == "natgas":
        lines.append("Natural gas prices winter highest, so a hump over the heating months is "
                     "the normal shape, not a sign of stress.")
    anchor = out[2] if len(out) > 2 else out[-1]
    then = month_ago(front, anchor)
    if then is None:
        lines.append(f"A month ago: the history of {front.label} and {anchor.leg.label} did not "
                     "give a close then, so no steepening reading is made.")
    else:
        now = anchor.pct * 100
        sentence = (f"A month ago the {front.label} to {anchor.leg.label} spread was "
                    f"{_signed(then, 1)}% (same two contracts), now {_signed(now, 1)}%")
        if then * now < 0 and abs(then) >= FLAT * 100 and abs(now) >= FLAT * 100:
            sentence += f": the curve has flipped from {shape(then)} to {shape(now)}."
        elif abs(abs(now) - abs(then)) < STEADY_POINTS:
            sentence += ": little changed."
        else:
            deeper = abs(now) > abs(then)
            name = shape(now) if shape(now) != "flat" else shape(then)
            sentence += f": {name} is {'steepening' if deeper else 'flattening'}."
        lines.append(sentence)
    return head, lines


def perps_for(symbols: Sequence[str]) -> dict[str, Perp]:
    """Price, funding rate and funding interval of each listed Bitget perpetual in ``symbols``."""
    body = http.fetch_json(TICKERS, timeout=10.0)
    rows = {str(r.get("symbol")): r for r in body.get("data") or []}
    hours: dict[str, int | None] = {}
    try:
        from argus.market import universe

        listed = universe.contracts()
        hours = {s: listed[s].funding_hours for s in symbols if s in listed}
    except (OSError, ValueError, RuntimeError):
        hours = {}
    found: dict[str, Perp] = {}
    for symbol in symbols:
        row = rows.get(symbol)
        if row is None:
            continue
        try:
            found[symbol] = Perp(symbol, float(row["lastPr"]), float(row["fundingRate"]),
                                 hours.get(symbol))
        except (KeyError, TypeError, ValueError):
            continue
    return found


def _perp_text(p: Perp) -> str:
    note = f" ({PERP_NOTES[p.symbol]})" if p.symbol in PERP_NOTES else ""
    if p.hours:
        yearly = p.funding * (8760 / p.hours)
        fund = (f"funding {_pct(p.funding, 4)} per {p.hours}h, {_pct(yearly)} a year")
    else:
        fund = f"funding {_pct(p.funding, 4)} per interval"
    return f"{p.symbol}{note} {_money(p.price)}, {fund}"


def lines(text: str, *, today: date | None = None) -> list[str] | None:
    """The futures-curve answer for the commodities a question names, or None when it is not a
    commodity futures-curve or "which contracts trade it" question."""
    if not asked(text):
        return None
    wanted = commodities_asked(text)
    day = today or datetime.now(UTC).date()
    heads: list[str] = []
    body: list[str] = []
    fronts: dict[str, Leg] = {}
    for c in wanted:
        legs = chain(c, day)
        if not legs:
            heads.append(f"the {c.name} futures curve could not be read just now")
            continue
        fronts[c.key] = legs[0]
        head, more = curve_lines(c, legs)
        heads.append(head)
        body += more
    perp_symbols = [s for c in wanted for s in c.perps]
    try:
        perps = perps_for(perp_symbols)
    except http.RpcError:
        perps = {}
    contracts_first = asks_for_contracts(text) and not _CURVE.search(text)
    bitget: list[str] = []
    for c in wanted:
        mine = [perps[s] for s in c.perps if s in perps]
        if not mine:
            bitget.append(f"Bitget lists no USDT perpetual for {c.name} that could be read.")
            continue
        text_ = "; ".join(_perp_text(p) for p in mine)
        front = fronts.get(c.key)
        basis = ""
        if front is not None:
            gap = mine[0].price / front.series.price - 1.0
            basis = (f" {mine[0].symbol} sits {_pct(gap)} from the {front.label} future, so it "
                     "tracks the front month.")
        bitget.append(f"Bitget perpetuals for {c.name}: {text_}.{basis}")
    out: list[str] = []
    if contracts_first:
        names = [p.symbol for c in wanted for p in (perps[s] for s in c.perps if s in perps)]
        listing = ", ".join(names) if names else "none that could be read"
        out.append(f"Bottom line: Bitget lists these USDT perpetuals for {_join(wanted)}: "
                   f"{listing}; " + "; ".join(heads) + ".")
    else:
        out.append("Bottom line: " + "; ".join(heads) + ".")
    out += body
    out += bitget
    if perps:
        out.append("A perpetual has no expiry, so it does not carry the roll the futures curve "
                   "shows; its funding rate plays that role (positive means longs pay shorts, "
                   "settled each funding interval). Backwardation pays a long holder of futures "
                   "who rolls, contango costs one.")
    out.append("Data: Yahoo Finance daily charts of each contract month (contracts whose last "
               "trade is older than a day are left out); Bitget USDT-futures tickers and "
               "contract list. Roll yield is an estimate from delivery months, not a promise. "
               "Not advice.")
    return out


def _join(items: Sequence[Commodity]) -> str:
    names = [c.name for c in items]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


__all__ = ["COMMODITIES", "asked", "chain", "commodities_asked", "lines", "perps_for", "series"]
