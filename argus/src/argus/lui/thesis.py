"""The trader's own reasons, each tested against live data.

"long SOL, ecosystem activity strengthening, this pullback looks temporary" was read by the research
task as "add 20% SOL" and nothing else: both stated reasons went untested, although one engine
answers the second exactly (research/h2h/RESULTS.md §4, 2026-09-28). sniperchief/killmythesis
(MIT, `research/repos-s2new/sniperchief~killmythesis`), a Season-2 entrant, is built around this:
an LLM splits the thesis into assumptions (`src/server/research/parser.ts:60-120`), another LLM call
labels every finding supporting or challenging with a confidence and a relevance, and
`src/lib/scoring.ts:1-100` turns those labels into a verdict by fixed ratio thresholds (0.75
supported, 0.25 contradicted) and hand-set weights (0.35/0.25/0.20/0.20).

What is taken: the unit (one stated reason, one verdict), the verdict set (supported,
contradicted, and an honest middle), and the rule that an unreadable or untestable reason is said
to be so rather than guessed at. What is not: no model decides a direction here. Each reason is
matched by pattern to the one measurement that bears on it, and the verdict is that measurement's
test —

* **a move reversing** ("this pullback is temporary", "the rally will fade"): whether the move
  described exists — over 20 days, or as a dip or a pop of 1% inside the last 24 hours — then
  whether states like today's 20-day one were followed by the direction the reason expects more
  often than an ordinary day, by two standard errors on independent episodes (`research/analogue.
  _long_run`); inside that band the reason is *not measurable*, which is the usual answer;
* **network activity** ("ecosystem activity strengthening", "adoption growing"): the chain's
  daily fees, DEX volume and TVL on DeFiLlama, the last 30 days against the 30 before, placed in
  the distribution of every such 30-day change over the past three years;
* **a demand driver** ("runs on AI capex", "demand is slowing"): the company's revenue and, for
  AI or cloud capex, the four largest US cloud builders' capital spending, quarter by quarter from
  SEC XBRL (`lui/drivers.py`) — whether the driver and the company are growing today, and which
  part of a forward claim no filing can test;
* **valuation**: the trailing P/E against the sector fund's (a measurement) and the analysts' mean
  target against the price (an opinion); a verdict only where they agree or one alone leans;
* **momentum**, **positioning** (funding) and **sentiment** (Fear & Greed) read the figure the task
  already holds and say whether it agrees — and where agreeing is not an edge, the line says so.

A reason no engine reads is listed as *not tested*, with what would test it.

**Evidence beside each verdict** (the second pass against killmythesis, which showed 21 findings to
our two): every reason also carries the other figures that bear on it — returns at 7, 30 and ~90
days, the place in the 90-day range, the gap to BTC and the week's volume against the month's
from Bitget's daily candles; Bitget's account and position long/short split and 24 hours of taker
buy/sell flow — each with a link a reader can open. They inform; they do
not vote. And one assumption no trader states is tested whenever a direction is given: that the
crowd is not already in the trade (:func:`implied_crowding`).
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from argus.truth import http
from argus.truth.endpoints import BITGET_API
from argus.truth.failures import RpcError


class Kind(StrEnum):
    REVERSION = "a move reversing"
    ACTIVITY = "network activity"
    MOMENTUM = "momentum"
    POSITIONING = "positioning"
    SENTIMENT = "sentiment"
    VALUATION = "valuation"
    DRIVER = "a demand driver"
    EARNINGS = "earnings"
    MACRO = "macro"
    OTHER = "other"


class Result(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_MEASURABLE = "not measurable"
    NOT_TESTED = "not tested"


_KINDS: tuple[tuple[Kind, re.Pattern[str]], ...] = (
    (Kind.REVERSION, re.compile(
        r"pull\s?back|\bdip\b|correction|temporary|bounce|rebound|recover|oversold|bottom(ed| is)|"
        r"overdone|(rally|pump|move|run)\b.*\b(fade|reverse|unwind|won'?t last)|overbought", re.I)),
    (Kind.ACTIVITY, re.compile(
        r"ecosystem|activity|adoption|usage|\busers?\b|\btvl\b|on-?chain|network (growth|use)|"
        r"developers?|\bfees?\b|transactions?|\bdapps?\b", re.I)),
    (Kind.POSITIONING, re.compile(
        r"funding|squeeze|open interest|positioning|crowded|shorts? (are|is|pay)|longs? (are|pay)|"
        r"over-?leveraged", re.I)),
    (Kind.MOMENTUM, re.compile(
        r"momentum|breakout|break(ing)? out|uptrend|downtrend|trend|higher highs|lower lows|"
        r"strength|has further to run|keeps? (going|rising|falling)", re.I)),
    (Kind.SENTIMENT, re.compile(
        r"sentiment|fear|greed|panic|euphori|capitulat|hype|everyone|the crowd", re.I)),
    (Kind.VALUATION, re.compile(
        r"cheap|undervalued|overvalued|expensive|valuation|\bp/?e\b|price target|fair value",
        re.I)),
    # "NVDA runs on AI capex" matched no kind and was not even listed (judge audit, 2026-09-30);
    # tested against the filings by `lui/drivers.py`. Before EARNINGS, which "revenue" would take.
    (Kind.DRIVER, re.compile(
        r"\bcapex\b|capital\s+spend|data[\s-]*cent(?:er|re)|hyperscal|\bai\s+(?:spend|demand|"
        r"build|boom|infrastructure|investment|chips?|servers?)|cloud\s+spend|\bdemand\b|"
        r"runs?\s+on\b|driven\s+by|(?:revenue|sales)\s+(?:growth|keeps?\s+growing|(?:is|are)\s+"
        r"(?:growing|accelerating))", re.I)),
    (Kind.EARNINGS, re.compile(r"earnings|revenue|guidance|profit|margins?|\beps\b|beat", re.I)),
    (Kind.MACRO, re.compile(
        r"\bfed\b|rates?\b|rate cuts?|\bcpi\b|inflation|dollar|\bdxy\b|macro|liquidity|yields?",
        re.I)),
)

_SPLIT = re.compile(r"\s*(?:[,;?]|\bbecause\b|\bsince\b|\bas\b|\band\b|\bgiven\b|\bplus\b)\s*",
                    re.I)
_STANCE = re.compile(r"^(?:i(?:'m| am)?\s+)?(?:going\s+|thinking of going\s+|considering\s+"
                     r"(?:going\s+)?)?(?:long|short|buy(?:ing)?|sell(?:ing)?|bullish|bearish)"
                     r"(?:\s+on)?\s+\S+$", re.I)

_ASK_WORDS = re.compile(
    r"^\s*(?:i\s+(?:think|believe|reckon|expect|feel)\s+(?:that\s+)?|my\s+(?:thesis|view|take|"
    r"idea)\s+(?:is\s+)?(?:that\s+)?:?\s*|here'?s\s+my\s+(?:thesis|view|take|idea):?\s*)|"
    r"\s*[-\u2013\u2014:,.]?\s*(?:please\s+|can\s+you\s+)?(?:test|check|challenge|"
    r"stress[\s-]*test|pressure[\s-]*test|poke\s+holes\s+in|kill|critique|evaluate|assess|validate)\s+"
    r"(?:(?:my|this|the|that)\s+(?:thesis|idea|view|take|theory|call)|it|this|that)\b[\s?.!]*$",
    re.I)
"""The asking around a stated thesis ("I think ...", "... test my thesis"): not a reason."""

_ASKED = re.compile(r"^(?:should|would|could|can|do|does|is|are)\s+(?:i|we|it|you)\b", re.I)
"""The question itself ("should I buy NVDA") is not a reason."""

CHAINS: Mapping[str, str] = {
    "SOL": "Solana", "ETH": "Ethereum", "BNB": "BSC", "AVAX": "Avalanche", "SUI": "Sui",
    "APT": "Aptos", "ARB": "Arbitrum", "OP": "Optimism", "TRX": "Tron", "TON": "TON",
    "NEAR": "Near", "ADA": "Cardano", "DOT": "Polkadot", "SEI": "Sei", "INJ": "Injective",
    "BTC": "Bitcoin", "HYPE": "Hyperliquid L1", "BERA": "Berachain", "S": "Sonic",
    "POL": "Polygon", "MNT": "Mantle", "STX": "Stacks", "TIA": "Celestia",
}
"""Tickers whose own chain DeFiLlama tracks by this name (`api.llama.fi/v2/chains`)."""

LLAMA = "https://api.llama.fi"
WINDOW_DAYS = 30
HISTORY_DAYS = 3 * 365


@dataclass(frozen=True)
class Reason:
    text: str
    kind: Kind


@dataclass(frozen=True)
class Finding:
    """One figure behind a verdict, where it came from, and where a reader can check it."""

    text: str
    source: str
    url: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "source": self.source, "url": self.url}


@dataclass(frozen=True)
class Tested:
    reason: str
    kind: Kind
    result: Result
    line: str
    evidence: tuple[Finding, ...] = ()
    """What else bears on the reason. Shown beside the verdict, never deciding it: the verdict is
    the one test in :attr:`line`."""
    implied: bool = False
    """True for an assumption the trader did not state but every such position carries."""

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "kind": self.kind.value, "result": self.result.value,
                "line": self.line, "implied": self.implied,
                "evidence": [f.as_dict() for f in self.evidence]}


def reasons(text: str) -> tuple[Reason, ...]:
    """The reasons a thesis states, each with the kind of evidence that could test it. The
    stance and name ("long SOL") are not a reason; a fragment of two words or fewer that names no
    kind is not one either. Empty when the text states no reason."""
    out: list[Reason] = []
    text = _ASK_WORDS.sub("", text.strip()).strip()
    for part in _SPLIT.split(text.rstrip(".?!")):
        part = part.strip(" .")
        if not part or _STANCE.match(part) or _ASKED.match(part):
            continue
        kind = next((k for k, pattern in _KINDS if pattern.search(part)), None)
        if kind is None:
            if len(part.split()) <= 2 or not re.search(r"\b(is|are|will|looks?|has|have|"
                                                         r"getting|keeps?)\b", part, re.I):
                continue
            kind = Kind.OTHER
        out.append(Reason(text=part, kind=kind))
    return tuple(out)


# --- network activity -------------------------------------------------------------------------

def _daily(series: Sequence[Any]) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for row in series:
        if isinstance(row, Mapping):
            stamp, value = row.get("date"), row.get("tvl")
        else:
            stamp, value = (row[0], row[1]) if len(row) >= 2 else (None, None)
        try:
            rows.append((int(stamp), float(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return sorted(rows)


def month_change(rows: Sequence[tuple[int, float]], *, window: int = WINDOW_DAYS,
                 history: int = HISTORY_DAYS) -> dict[str, Any] | None:
    """The last ``window`` days' mean against the ``window`` before, and where that change sits
    among every such change (stepped by ``window`` days) over the last ``history`` days."""
    values = [v for _, v in rows[-(history + 2 * window):]]
    if len(values) < 4 * window:
        return None

    def change(end: int) -> float | None:
        recent, prior = values[end - window:end], values[end - 2 * window:end - window]
        base = sum(prior) / len(prior)
        return (sum(recent) / len(recent)) / base - 1 if base > 0 else None

    now = change(len(values))
    past = [c for end in range(2 * window, len(values) - window + 1, window)
            if (c := change(end)) is not None]
    if now is None or len(past) < 6:
        return None
    rank = sum(1 for c in past if c < now) / len(past)
    return {"change": now, "percentile": rank, "months": len(past),
            "as_of": datetime.fromtimestamp(rows[-1][0], UTC).date().isoformat()}


def chain_activity(symbol: str, *, timeout: float = 15.0) -> dict[str, Any] | None:
    """Daily fees and TVL for the chain ``symbol`` is the token of, read from DeFiLlama, each as a
    :func:`month_change`. None for a name that is not a chain's token."""
    chain = CHAINS.get(symbol.removesuffix("USDT").upper())
    if chain is None:
        return None
    out: dict[str, Any] = {"chain": chain}
    try:
        fees = http.fetch_json(f"{LLAMA}/overview/fees/{chain.lower()}",
                               params={"excludeTotalDataChart": "false",
                                       "excludeTotalDataChartBreakdown": "true"},
                               timeout=timeout)
        out["fees"] = month_change(_daily((fees or {}).get("totalDataChart") or []))
    except RpcError as exc:
        out["fees_error"] = exc.kind.value
    try:
        dex = http.fetch_json(f"{LLAMA}/overview/dexs/{chain.lower()}",
                              params={"excludeTotalDataChart": "false",
                                      "excludeTotalDataChartBreakdown": "true"},
                              timeout=timeout)
        out["dex"] = month_change(_daily((dex or {}).get("totalDataChart") or []))
    except RpcError as exc:
        out["dex_error"] = exc.kind.value
    try:
        tvl = http.fetch_json(f"{LLAMA}/v2/historicalChainTvl/{chain}", timeout=timeout)
        out["tvl"] = month_change(_daily(tvl if isinstance(tvl, list) else []))
    except RpcError as exc:
        out["tvl_error"] = exc.kind.value
    return out


# --- context read once for every reason -------------------------------------------------------

CANDLES = "/api/v2/mix/market/candles"
TAKER = "/api/v2/mix/market/taker-buy-sell"
CROWDED_LONG = 0.70
"""A share of Bitget accounts long at or above this is a crowd already in the trade."""
UNCROWDED = 0.50


def _closes(symbol: str) -> list[tuple[float, float, float, float]]:
    """Up to 90 daily (high, low, close, quote volume) rows for ``symbol``'s perpetual, oldest
    first (Bitget serves 90 at most for this granularity, checked 2026-09-28)."""
    from argus.market.bitget import public_get

    rows = public_get(CANDLES, {"productType": "USDT-FUTURES", "symbol": symbol,
                                "granularity": "1D", "limit": "100"}) or []
    ordered = sorted(rows, key=lambda r: int(r[0]))
    return [(float(r[2]), float(r[3]), float(r[4]), float(r[6])) for r in ordered]


def _taker_flow(symbol: str) -> float | None:
    """Taker buy volume over taker sell volume on Bitget's perpetual across the last 24 hours
    (six 4-hour rows of ``/api/v2/mix/market/taker-buy-sell``)."""
    from argus.market.bitget import public_get

    rows = public_get(TAKER, {"symbol": symbol, "period": "4h"}) or []
    recent = sorted(rows, key=lambda r: int(r["ts"]))[-6:]
    sold = sum(float(r["sellVolume"]) for r in recent)
    return sum(float(r["buyVolume"]) for r in recent) / sold if recent and sold > 0 else None


def context(symbol: str) -> dict[str, Any]:
    """The figures several reasons draw on: 7/30/90-day returns and the place in the 90-day
    range (Bitget daily candles), the same 30 days for BTC, and Bitget's account long/short
    split. Each part is left out when its source does not answer."""
    from argus.market import long_short

    out: dict[str, Any] = {"symbol": symbol}
    try:
        rows = _closes(symbol)
        if len(rows) >= 31:
            last = rows[-1][2]
            # Bitget serves at most 90 daily rows, so the longest return is over what it gave.
            out["returns"] = {n: last / rows[-1 - n][2] - 1
                              for n in (7, 30, len(rows) - 1) if len(rows) > n}
            window = rows[-90:]
            high, low = max(r[0] for r in window), min(r[1] for r in window)
            out["range"] = {"high": high, "low": low, "days": len(window),
                            "place": (last - low) / (high - low) if high > low else None}
            week, month = [r[3] for r in rows[-7:]], [r[3] for r in rows[-30:]]
            if sum(month) > 0:
                out["volume"] = {"week": sum(week) / len(week), "month": sum(month) / len(month)}
        if symbol != "BTCUSDT":
            btc = _closes("BTCUSDT")
            if len(btc) >= 31:
                out["btc_30d"] = btc[-1][2] / btc[-31][2] - 1
    except Exception:
        out["candles_error"] = True
    try:
        flow = _taker_flow(symbol)
        if flow is not None:
            out["taker"] = flow
    except Exception:
        out["taker_error"] = True
    reading = long_short.read(symbol)
    if reading is not None:
        out["crowd"] = {"accounts_long": reading.accounts_long,
                        "day_ago": reading.accounts_long_day_ago,
                        "position_long": reading.position_long}
    return out


def _candles_url(symbol: str) -> str:
    return (f"{BITGET_API}{CANDLES}?productType=USDT-FUTURES&symbol={symbol}&granularity=1D"
            f"&limit=100")


def _price_findings(ctx: Mapping[str, Any], name: str) -> tuple[Finding, ...]:
    symbol = str(ctx.get("symbol") or f"{name}USDT")
    out: list[Finding] = []
    returns = ctx.get("returns") or {}
    if returns:
        text = ", ".join(f"{n}-day {r:+.1%}" for n, r in sorted(returns.items()))
        btc = ctx.get("btc_30d")
        if btc is not None and 30 in returns:
            text += (f"; over 30 days {(returns[30] - btc) * 100:+.1f} points against BTC's "
                     f"{btc:+.1%}")
        out.append(Finding(f"{name} returns from daily closes: {text}.", "Bitget daily candles",
                           _candles_url(symbol)))
    volume = ctx.get("volume") or {}
    if volume.get("month"):
        ratio = volume["week"] / volume["month"]
        out.append(Finding(f"{name}'s average daily volume over the last 7 days is {ratio:.2f}x "
                           f"the 30-day average (${volume['week'] / 1e6:,.1f}M against "
                           f"${volume['month'] / 1e6:,.1f}M).", "Bitget daily candles",
                           _candles_url(symbol)))
    span = ctx.get("range") or {}
    if span.get("place") is not None:
        out.append(Finding(f"{name} sits at {span['place']:.0%} of its {span['days']}-day range "
                           f"({span['low']:g} to {span['high']:g}).", "Bitget daily candles",
                           _candles_url(symbol)))
    return tuple(out)


def _taker_finding(ctx: Mapping[str, Any], name: str) -> Finding | None:
    flow = ctx.get("taker")
    if flow is None:
        return None
    symbol = str(ctx.get("symbol") or f"{name}USDT")
    return Finding(f"Over the last 24 hours takers bought {flow:.2f}x what they sold on Bitget's "
                   f"{name} perpetual.", "Bitget taker buy/sell, 4-hourly",
                   f"{BITGET_API}{TAKER}?symbol={symbol}&period=4h")


def _crowd_finding(ctx: Mapping[str, Any], name: str) -> Finding | None:
    crowd = ctx.get("crowd")
    if not crowd:
        return None
    symbol = str(ctx.get("symbol") or f"{name}USDT")
    text = f"{crowd['accounts_long']:.0%} of Bitget {name} futures accounts are long"
    if crowd.get("day_ago") is not None:
        text += f" ({(crowd['accounts_long'] - crowd['day_ago']) * 100:+.1f} points on a day ago)"
    if crowd.get("position_long") is not None:
        text += f"; by position size {crowd['position_long']:.0%} is long"
    return Finding(text + ".", "Bitget account long/short, hourly",
                   f"{BITGET_API}/api/v2/mix/market/account-long-short?symbol={symbol}&period=1h")


_STATED_SIDE = re.compile(r"^\s*(?:i(?:'m| am)?\s+)?(?:going\s+|considering\s+(?:going\s+)?)?"
                          r"(long|short|buy|sell|bullish|bearish)\b", re.I)


def stated_side(text: str) -> str | None:
    """``long`` or ``short`` when the thesis opens with its direction, else None."""
    found = _STATED_SIDE.match(text)
    if found is None:
        return None
    return "short" if found.group(1).lower() in ("short", "sell", "bearish") else "long"


def implied_crowding(ctx: Mapping[str, Any], name: str, side: str) -> Tested:
    """The assumption every position carries and no trader states: the crowd is not already in
    it. Tested on Bitget's own two splits, because they disagree often and the disagreement is the
    reading: the share of *accounts* on the trader's side and the share of *position size*.

    Crowded means both at 70% or more; clear means both at 50% or less. Many small accounts on
    one side while the larger money is on the other is neither: it is the split a contrarian read
    starts from, and the line says so rather than calling it crowded (SOL on 2026-09-28: 82% of
    accounts long, 36% of position size)."""
    reason = ("the move is not already crowded" if side == "long" else
              "the fall is not already crowded")
    crowd = ctx.get("crowd")
    finding = _crowd_finding(ctx, name)
    if not crowd or finding is None:
        return Tested(reason, Kind.POSITIONING, Result.NOT_TESTED,
                      f"Bitget's long/short split did not answer for {name}.", implied=True)
    def mine(share: float) -> float:
        return share if side == "long" else 1 - share

    accounts = mine(float(crowd["accounts_long"]))
    size = (mine(float(crowd["position_long"])) if crowd.get("position_long") is not None
            else None)
    split = f"{accounts:.0%} of Bitget {name} accounts" + (
        f" and {size:.0%} of position size" if size is not None else "") + f" are {side}"
    if accounts >= CROWDED_LONG and (size is None or size >= CROWDED_LONG):
        result, lead = Result.CONTRADICTED, "The crowd is already there"
    elif accounts <= UNCROWDED and (size is None or size <= UNCROWDED):
        result, lead = Result.SUPPORTED, "The crowd is not there yet"
    elif size is not None and (accounts >= CROWDED_LONG) != (size >= CROWDED_LONG) \
            and abs(accounts - size) >= 0.2:
        result = Result.NOT_MEASURABLE
        lead = ("Split: most accounts are with you, the larger money is not" if accounts > size
                else "Split: the larger money is with you, most accounts are not")
    else:
        result, lead = Result.NOT_MEASURABLE, "The crowd leans but is not one-sided"
    return Tested(reason, Kind.POSITIONING, result,
                  f"{lead}: {split}. A crowd measure, not an edge.",
                  evidence=tuple(f for f in (finding, _taker_finding(ctx, name)) if f),
                  implied=True)


# --- the tests --------------------------------------------------------------------------------

SHORT_MOVE_PCT = 1.0
"""A move this large inside the last 24 hours (from the 24-hour high or low, or over the day)
counts as the dip or the pop a trader means by "this pullback" inside a longer trend."""


def _recent(quote: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """The 24-hour change and the distance from the 24-hour high and low, in percent."""
    ticker: Mapping[str, Any] = next(iter((quote.get("quotes") or {}).values()), {})
    try:
        last, high, low = (float(ticker["last"]), float(ticker["high_24h"]),
                           float(ticker["low_24h"]))
        change = float(ticker["change_24h"]) * 100
    except (KeyError, TypeError, ValueError):
        return None
    if min(last, high, low) <= 0:
        return None
    return change, (last / high - 1) * 100, (last / low - 1) * 100


def _reversion(reason: Reason, long_run: Mapping[str, Any], quote: Mapping[str, Any],
               name: str) -> Tested:
    """Whether the move the reason describes exists — over 20 days, or as a dip or a pop inside
    the last 24 hours — and then whether states like today's 20-day one were followed by the
    direction the reason expects more often than an ordinary day was.

    A "temporary pullback" expects the price to rise from here, whether the pullback is a 20-day
    fall or a day's dip inside a rise; a "rally that will fade" expects it to fall. So the test is
    the same up-rate either way, read in the direction the reason expects."""
    trailing = long_run.get("trailing_return_pct")
    rows = [r for r in long_run.get("horizons") or [] if int(r.get("episodes") or 0) >= 8]
    window = long_run.get("window", 20)
    if trailing is None or not rows:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"No long-run daily history answered for {name}, so whether moves like "
                      f"this one reversed could not be counted.")
    trailing = float(trailing)
    fade = bool(re.search(r"rally|pump|\brun\b|overbought|fade|won'?t last", reason.text, re.I))
    recent = _recent(quote)
    if fade:
        premise = trailing > 0 or (recent is not None and (
            recent[0] >= SHORT_MOVE_PCT or recent[2] >= SHORT_MOVE_PCT))
        where = (f"{name} is {trailing:+.1f}% over {window} days"
                 + (f" and {recent[0]:+.1f}% over 24 hours" if recent else ""))
    else:
        premise = trailing < 0 or (recent is not None and (
            recent[0] <= -SHORT_MOVE_PCT or recent[1] <= -SHORT_MOVE_PCT))
        where = (f"{name} is {trailing:+.1f}% over {window} days"
                 + (f" and {abs(recent[1]):.1f}% below its 24-hour high" if recent else ""))
    if not premise:
        return Tested(reason.text, reason.kind, Result.CONTRADICTED,
                      f"The premise does not hold: {where}, so there is no "
                      f"{'rally' if fade else 'pullback'} to reverse.")
    sign = -1 if fade else 1  # the direction the reason expects next
    best = max(rows, key=lambda r: sign * float(r.get("z") or 0))
    worst = min(rows, key=lambda r: sign * float(r.get("z") or 0))
    z = float(best.get("z") or 0)

    def detail(row: Mapping[str, Any]) -> str:
        days = int(row["days"])
        return (f"after {window}-day moves like this one, {name} was up {float(row['up']):.0%} of "
                f"the time {days} day{'' if days == 1 else 's'} later against "
                f"{float(row['base_up']):.0%} for any day, on {row['episodes']} independent "
                f"episodes")

    expects = "a rise" if sign > 0 else "a fall"
    if sign * z >= 2:
        return Tested(reason.text, reason.kind, Result.SUPPORTED,
                      f"{where}. History backs {expects} from here: {detail(best)} — "
                      f"{abs(z):.1f} standard errors.")
    if sign * float(worst.get("z") or 0) <= -2:
        return Tested(reason.text, reason.kind, Result.CONTRADICTED,
                      f"{where}. History points the other way: {detail(worst)} — "
                      f"{abs(float(worst['z'])):.1f} standard errors.")
    return Tested(reason.text, reason.kind, Result.NOT_MEASURABLE,
                  f"{where}. History neither backs nor refutes {expects} from here: "
                  f"{detail(best)}, within two standard errors ({z:+.1f}).")


def _activity(reason: Reason, activity: Mapping[str, Any] | None, name: str) -> Tested:
    if activity is None:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"{name} is not a chain's own token, so there is no network whose activity "
                      f"would test this.")
    parts = [(label, activity.get(key)) for key, label in (("fees", "daily fees"),
                                                           ("dex", "DEX volume"),
                                                           ("tvl", "value locked (TVL)"))]
    read = [(label, m) for label, m in parts if m]
    if not read:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"DeFiLlama did not answer for {activity.get('chain')} just now, so the "
                      f"claim was not tested.")
    chain = str(activity["chain"])
    lines = [f"{label} {m['change']:+.1%} ({m['percentile']:.0%} percentile)" for label, m in read]
    evidence = tuple(
        Finding(f"{chain} {label}: {m['change']:+.1%} over the last {WINDOW_DAYS} days against the "
                f"{WINDOW_DAYS} before, higher than {m['percentile']:.0%} of the {m['months']} "
                f"monthly changes in three years (to {m['as_of']}).", "DeFiLlama",
                f"https://defillama.com/chain/{urllib.parse.quote(chain)}")
        for label, m in read)
    ups = [m["percentile"] >= 0.6 and m["change"] > 0 for _, m in read]
    downs = [m["percentile"] <= 0.4 and m["change"] < 0 for _, m in read]
    weakening = bool(re.search(r"weak|slow|declin|fall|drop|shrink", reason.text, re.I))
    note = (" All are in dollars, so a falling token price lowers them without any change in "
            "use.")
    if all(ups):
        result = Result.CONTRADICTED if weakening else Result.SUPPORTED
    elif all(downs):
        result = Result.SUPPORTED if weakening else Result.CONTRADICTED
    else:
        result = Result.NOT_MEASURABLE
    lead = {Result.SUPPORTED: "The data agrees", Result.CONTRADICTED: "The data disagrees",
            Result.NOT_MEASURABLE: "Within the usual month-to-month swing"}[result]
    return Tested(reason.text, reason.kind, result,
                  f"{lead}: {chain} " + ", ".join(lines) + f", month on month against three "
                  f"years of monthly changes (DeFiLlama, to {read[0][1]['as_of']})." + note,
                  evidence=evidence)


def _momentum(reason: Reason, tech: Mapping[str, Any], name: str) -> Tested:
    hist = tech.get("macd_histogram")
    if hist is None:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"No technical reading answered for {name} this time.")
    down_claim = bool(re.search(r"down|lower lows|falling|weak", reason.text, re.I))
    agrees = (float(hist) < 0) == down_claim
    return Tested(reason.text, reason.kind,
                  Result.SUPPORTED if agrees else Result.CONTRADICTED,
                  f"The tape {'agrees' if agrees else 'disagrees'}: MACD histogram "
                  f"{float(hist):+.3f} on {tech.get('timeframe') or '4h'}"
                  + (f", RSI {float(tech['rsi']):.0f}" if tech.get("rsi") is not None else "")
                  + ". Agreement is not an edge: the technical signals tested on these names did "
                    "not clear costs.")


def _positioning(reason: Reason, quote: Mapping[str, Any], name: str) -> Tested:
    ticker: Mapping[str, Any] = next(iter((quote.get("quotes") or {}).values()), {})
    try:
        funding = float(ticker["funding_rate"]) * 100
    except (KeyError, TypeError, ValueError):
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"No funding rate answered for {name} this time.")
    shorts_claim = bool(re.search(r"short|squeeze", reason.text, re.I))
    if abs(funding) < 0.005:
        return Tested(reason.text, reason.kind, Result.NOT_MEASURABLE,
                      f"Funding is {funding:+.4f}% per interval: neither side is paying to hold, "
                      f"so no crowding shows in it.")
    agrees = (funding < 0) == shorts_claim
    side = "shorts pay longs" if funding < 0 else "longs pay shorts"
    return Tested(reason.text, reason.kind, Result.SUPPORTED if agrees else Result.CONTRADICTED,
                  f"Funding is {funding:+.4f}% per interval on Bitget — {side}, so the crowded "
                  f"side is the {'short' if funding < 0 else 'long'}.")


def _sentiment(reason: Reason, fear_greed: Mapping[str, Any] | None) -> Tested:
    if not fear_greed:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      "The crypto Fear & Greed index did not answer, and no text-sentiment "
                      "engine runs in this task.")
    value = int(fear_greed["value"])
    fearful_claim = bool(re.search(r"fear|panic|capitulat|bearish|hated", reason.text, re.I))
    if 40 <= value <= 60:
        result = Result.NOT_MEASURABLE
    else:
        result = Result.SUPPORTED if (value < 40) == fearful_claim else Result.CONTRADICTED
    return Tested(reason.text, reason.kind, result,
                  f"Crypto Fear & Greed is {value} ({fear_greed.get('classification')}), "
                  f"alternative.me. It describes the whole crypto market, not this name.")


PE_RICH = 1.5
PE_CHEAP = 0.9
"""Trailing P/E against the sector fund's: above 1.5 times reads as priced richly, below 0.9 as
cheaply. Wider on the rich side because growth companies carry a premium as a matter of course."""


def _valuation(reason: Reason, fund: Mapping[str, Any], name: str) -> Tested:
    """Two reads, each named for what it is: the trailing P/E against the sector's (a
    measurement) and the analysts' mean target against the price (an opinion). The verdict is
    theirs when they agree or only one answered; when they point opposite ways it is not
    measurable, and the line says why.

    Before 2026-09-30 only the target was read, and only when Yahoo supplied it — with Bitget's
    targets present it said "no analyst target answered", and it called TSLA "not overvalued" on
    an 11% target gap while its P/E stood at 14.6 times its sector's."""
    cheap_claim = not re.search(r"overvalued|over-valued|expensive|rich|pricey|bubble|frothy",
                                reason.text, re.I)
    target, price = fund.get("target_mean"), fund.get("price")
    ratio = fund.get("pe_vs_sector")
    reads: list[tuple[str, bool | None]] = []
    if ratio:
        ratio = float(ratio)
        cheap = False if ratio > PE_RICH else True if ratio < PE_CHEAP else None
        sector = fund.get("sector") or "its sector"
        reads.append((f"trailing P/E {float(fund['pe']):.1f} against {sector}'s "
                      f"{float(fund['sector_pe']):.1f} ({fund.get('sector_fund')}), "
                      f"{ratio:.1f} times — a measurement, and one a growth company can justify",
                      cheap))
    if target and price:
        gap = float(target) / float(price) - 1
        cheap = True if gap > 0.10 else False if gap < -0.10 else None
        reads.append((f"analysts' mean target ${float(target):,.2f}, {abs(gap):.0%} "
                      f"{'above' if gap > 0 else 'below'} the last close — an opinion", cheap))
    if not reads:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"Neither {name}'s sector multiple nor an analyst target answered; ARGUS "
                      f"does not value companies itself.")
    said = "; ".join(r for r, _ in reads)
    leans = {c for _, c in reads if c is not None}
    if len(leans) == 1:
        result = Result.SUPPORTED if leans == {cheap_claim} else Result.CONTRADICTED
        how = ("both reads agree" if len(reads) == 2 and all(c is not None for _, c in reads)
               else "the one read that leans either way")
    elif len(leans) == 2:
        result, how = Result.NOT_MEASURABLE, "the two reads point opposite ways"
    else:
        result, how = Result.NOT_MEASURABLE, "neither read leans far enough to call"
    return Tested(reason.text, reason.kind, result,
                  f"{said[:1].upper()}{said[1:]}. Verdict: {how}.")


def _driver(reason: Reason, found: Mapping[str, Any] | None, name: str) -> Tested:
    """A demand driver against the filings (`lui/drivers.py`)."""
    from argus.lui import drivers

    if found is None:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"{name}'s filings did not answer in time, so the driver is not tested.")
    result, line, evidence = drivers.test(reason.text, found)
    return Tested(reason.text, reason.kind, Result(result), line,
                  evidence=tuple(Finding(text, source, url) for text, source, url in evidence))


_UNTESTED: Mapping[Kind, str] = {
    Kind.EARNINGS: "The earnings step reports the last surprise and the next date, but no engine "
                   "tests a claim about future earnings.",
    Kind.MACRO: "No engine in this task tests a macro claim; the console's macro answers "
                "(`/ask`, FRED) do, one question at a time.",
    Kind.OTHER: "No engine reads this kind of claim, so it is yours to judge.",
}


def check(stated: Sequence[Reason], *, name: str, data: Mapping[str, Mapping[str, Any]],
         activity: Mapping[str, Any] | None = None,
         fear_greed: Mapping[str, Any] | None = None,
         ctx: Mapping[str, Any] | None = None, side: str | None = None,
         driver: Mapping[str, Any] | None = None) -> tuple[Tested, ...]:
    """Each stated reason against the measurement that bears on it. ``data`` is the research
    task's step data by kind, as :func:`argus.lui.weigh.weigh` takes it."""
    analogue = data.get("analogue") or {}
    tech = (data.get("technicals") or {}).get("technicals") or {}
    fund = (data.get("fundamentals") or {}).get("fundamentals") or {}
    out: list[Tested] = []
    for reason in stated:
        if reason.kind is Kind.REVERSION:
            out.append(_reversion(reason, analogue.get("long_run") or {},
                                  data.get("quote") or {}, name))
        elif reason.kind is Kind.ACTIVITY:
            out.append(_activity(reason, activity, name))
        elif reason.kind is Kind.MOMENTUM:
            out.append(_momentum(reason, tech, name))
        elif reason.kind is Kind.POSITIONING:
            out.append(_positioning(reason, data.get("quote") or {}, name))
        elif reason.kind is Kind.SENTIMENT:
            out.append(_sentiment(reason, fear_greed))
        elif reason.kind is Kind.VALUATION:
            out.append(_valuation(reason, fund, name))
        elif reason.kind is Kind.DRIVER:
            out.append(_driver(reason, driver, name))
        else:
            out.append(Tested(reason.text, reason.kind, Result.NOT_TESTED, _UNTESTED[reason.kind]))
    ctx = ctx or {}
    price = _price_findings(ctx, name)
    crowd = tuple(f for f in (_crowd_finding(ctx, name), _taker_finding(ctx, name)) if f)
    extra: dict[Kind, tuple[Finding, ...]] = {
        Kind.REVERSION: price, Kind.MOMENTUM: price, Kind.POSITIONING: crowd,
    }
    if fear_greed:
        extra[Kind.SENTIMENT] = (Finding(
            f"Crypto Fear & Greed {int(fear_greed['value'])} "
            f"({fear_greed.get('classification')}), market-wide.", "alternative.me",
            "https://alternative.me/crypto/fear-and-greed-index/"),)
    out = [replace(t, evidence=t.evidence + extra.get(t.kind, ())) for t in out]
    if side is not None and stated:
        out.append(implied_crowding(ctx, name, side))
    return tuple(out)


def fear_greed_now() -> dict[str, Any] | None:
    """The crypto Fear & Greed reading, through the Skill mirror's own reader."""
    from argus.market import skill_mirror

    try:
        return skill_mirror.fear_greed({})
    except Exception:
        return None

