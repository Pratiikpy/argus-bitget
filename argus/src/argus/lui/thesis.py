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
* **one name against another** ("SOL will outrun ETH"): both names' returns over 7, 30 and ~90
  days from Bitget's daily candles — the record so far, never the claim about what comes next;
* **macro** ("Fed rate cuts will boost NVDA"): the name's sensitivity to the 10-year yield and its
  event-study reaction around Fed decisions or CPI releases (`lui/macro_thesis.py`);
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
    RELATIVE = "one name against another"
    DIRECTION = "a direction"
    FLOWS = "fund flows"
    EARNINGS = "earnings"
    MACRO = "macro"
    OTHER = "other"


class Result(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_MEASURABLE = "not measurable"
    NOT_TESTED = "not tested"


_KINDS: tuple[tuple[Kind, re.Pattern[str]], ...] = (
    # "ETF inflows are accelerating" was not read as anything (a judge, round 12); the spot ETF
    # flows the console already keeps (data/etf_flows.json, SoSoValue) test it.
    (Kind.FLOWS, re.compile(r"\betfs?\b.{0,30}\b(?:in|out)flows?\b|\b(?:in|out)flows?\b.{0,30}"
                            r"\betfs?\b|\betf\s+(?:demand|buying|selling)\b", re.I)),
    (Kind.REVERSION, re.compile(
        r"pull\s?back|\bdip\b|correction|temporary|bounce|rebound|recover|oversold|bottom(ed| is)|"
        r"overdone|(rally|pump|move|run)\b.*\b(fade|reverse|unwind|won'?t last)|overbought|"
        # "SOL has run too far above its 20-day average", "should mean-revert": read as nothing
        # and answered "no engine reads this" although the console tests it (a judge, round 14)
        r"too\s+far\s+(?:above|below|from)|mean[\s-]*revert|revert\s+to\s+(?:the\s+)?mean|"
        r"stretched\s+(?:above|below|from)", re.I)),
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

_SPLIT = re.compile(
    # a comma, but not the one inside "$200,000" (a hostile review, round 12: the number was split
    # into two "reasons"), and a sentence's end
    r"\s*(?:,(?!\d{3}(?!\d))|[;?]|(?<=[a-z%)])\.\s+(?=[A-Za-z])|\bbecause\b|\bsince\b|\bas\b|"
    r"\band\b|\bgiven\b|\bplus\b)\s*", re.I)
_OPENER = re.compile(r"^i\s+(?:think|believe|reckon|expect|feel)\s+(?:that\s+)?", re.I)
_PRICE_STATEMENT = re.compile(r"(?:[$€£]\s*\d|\d\s*(?:k|usd|dollars?|euros?)\b)", re.I)
"""A fragment that only states a price ("Bitcoin is at $200,000 right now"): the premise check
reads it (:func:`price_premise`), and it is not a reason of its own."""
_STANCE = re.compile(r"^(?:i(?:'m| am)?\s+)?(?:going\s+|thinking of going\s+|considering\s+"
                     r"(?:going\s+)?)?(?:long|short|buy(?:ing)?|sell(?:ing)?|bullish|bearish)"
                     r"(?:\s+on)?\s+\S+$", re.I)

_ASK_WORDS = re.compile(
    r"^\s*(?:i\s+(?:think|believe|reckon|expect|feel)\s+(?:that\s+)?|my\s+(?:thesis|view|take|"
    r"idea)(?:\s+is)?(?:\s+that)?\s*:?\s*|here'?s\s+my\s+(?:thesis|view|take|idea):?\s*|"
    # "My bull case for BTC: ..." (a judge, round 12)
    r"(?:my|the)\s+(?:bull|bear)(?:ish)?\s+case(?:\s+(?:for|on)\s+[\w$.&/-]+)?\s*[:,-]?\s*)|"
    r"\s*[-\u2013\u2014:,.]?\s*(?:please\s+|can\s+you\s+)?(?:test|check|challenge|"
    r"stress[\s-]*test|pressure[\s-]*test|poke\s+holes\s+in|kill|critique|evaluate|assess|validate)\s+"
    r"(?:(?:my|this|the|that)\s+(?:thesis|idea|view|take|theory|call)|it|this|that)\b[\s?.!]*$|"
    # a closing question about the thesis: "... Is that thesis right?", "am I wrong?" (round 11)
    r"\s*[.?!]?\s*(?:is\s+(?:that|this|my|the)\s+(?:thesis|view|idea|take|theory|reasoning)"
    r"(?:\s+\w+){1,2}|am\s+i\s+(?:right|wrong)|what\s+do\s+you\s+think|thoughts|"
    r"which\s+(?:of\s+(?:these|them|those)\s+)?(?:one\s+)?(?:is|are)\s+(?:the\s+)?(?:strongest|"
    r"weakest|best|worst|most\s+\w+)(?:\s+(?:one|reason|point))?|true\s+or\s+false)[\s?.!]*$",
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


_RATIO = re.compile(r"\b[A-Za-z]{2,6}\s*/\s*[A-Za-z]{2,6}\b")


def reasons(text: str) -> tuple[Reason, ...]:
    """The reasons a thesis states, each with the kind of evidence that could test it. The
    stance and name ("long SOL") are not a reason; a fragment of two words or fewer that names no
    kind is not one either. Empty when the text states no reason."""
    out: list[Reason] = []
    text = text.strip()
    while (stripped := _ASK_WORDS.sub("", text).strip()) != text:
        # more than one asking phrase: "... Test my thesis. Which of these is strongest?"
        text = stripped
    for part in _SPLIT.split(text.rstrip(".?!")):
        # "because of institutional adoption" leaves "of ..." after the split (round 11)
        part = re.sub(r"^(?:of|to|on|that|the\s+fact\s+that)\s+", "", part.strip(" ."), flags=re.I)
        part = _OPENER.sub("", part)
        if not part or _STANCE.match(part) or _ASKED.match(part):
            continue
        kind = next((k for k, pattern in _KINDS if pattern.search(part)), None)
        if _RELATIVE.search(part) and len(_two_names(part)) == 2:
            # "SOL will outrun ETH" named two contracts and was left "not tested" (round 11).
            kind = Kind.RELATIVE
        if kind is Kind.REVERSION and _RATIO.search(part):
            # "ETH/BTC should mean-revert" is about a ratio; the price test reads one name's own
            # history and answered it with the other claim's numbers (a judge, round 14).
            kind = Kind.OTHER
        if kind is None and _DIRECTION.search(part):
            # "stocks will go up" was listed as nothing any engine reads (round 11)
            kind = Kind.DIRECTION
        if kind is None:
            # Kept unless it is two words or fewer, or only a stated price: "the halving cut new
            # supply" was dropped for want of a listed verb, and the answer's count left it out
            # without a word (a hostile review, round 12).
            if len(part.split()) <= 2 or _PRICE_STATEMENT.search(part):
                continue
            kind = Kind.OTHER
        out.append(Reason(text=part, kind=kind))
    return tuple(out)


_RELATIVE = re.compile(
    r"\b(?:outrun|outperform|outpace|beat|overtake|flip|do(?:es)?\s+better\s+than|lead|lag|"
    r"underperform)\w*\b", re.I)
"""A claim about one name against another."""


def _two_names(text: str) -> tuple[str, ...]:
    from argus.lui.research import research_symbols

    return research_symbols(text)[0][:2]


def relative_facts(reason: str) -> dict[str, Any] | None:
    """Both names' returns over 7, 30 and ~90 days from Bitget's daily candles, in the order
    named."""
    names = _two_names(reason)
    if len(names) != 2:
        return None
    out: dict[str, Any] = {"names": list(names)}
    for symbol in names:
        rows = _closes(symbol)
        if len(rows) < 31:
            return None
        last = rows[-1][2]
        out[symbol] = {n: last / rows[-1 - n][2] - 1 for n in (7, 30, len(rows) - 1)
                       if len(rows) > n}
    return out


def _is_coin(name: str) -> bool:
    from argus.lui.research.parse import is_us_equity
    from argus.market import universe

    symbol = name if name.endswith("USDT") else f"{name}USDT"
    return not is_us_equity(symbol) and universe.NOT_EQUITY.get(symbol, "crypto") == "crypto"


_NAMED_WINDOW = re.compile(r"\b(?:over|in|during)\s+the\s+(?:past|last)\s+(?:(?P<n>\d+)\s+)?"
                           r"(?P<unit>days?|weeks?|months?)\b|\bthis\s+(?P<unit2>week|month)\b",
                           re.I)
"""The window a backward-looking claim names; a month when it names none."""

_PAST_WINDOW = re.compile(
    r"\b(?:has|have)\s+been\b|\b(?:did|was|were|outperformed|underperformed|beat|lagged)\b|"
    r"\b(?:over|in|during)\s+the\s+(?:past|last)\s+(?:(?P<n>\d+)\s+)?(?P<unit>days?|weeks?|"
    r"months?)\b|\bthis\s+(?:week|month)\b", re.I)
"""A claim about what already happened ("SOL has been outperforming ETH over the past month"),
which the record settles, unlike one about what comes next."""


def _relative(reason: Reason, found: Mapping[str, Any] | None) -> Tested:
    """Whether the first name has been beating the second over the last 30 days and the longest
    window Bitget serves; the claim itself is about what comes next, which no data tests."""
    if not found:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      "The two names' daily candles did not answer, so the comparison is not "
                      "tested.")
    first, second = found["names"]
    a, b = found[first], found[second]
    spans = sorted(set(a) & set(b))
    parts = [f"over {n} days {first.removesuffix('USDT')} {a[n]:+.1%} against "
             f"{second.removesuffix('USDT')} {b[n]:+.1%}" for n in spans]
    ahead = [a[n] > b[n] for n in spans if n >= 30]
    claims_behind = bool(re.search(r"\b(?:lag|underperform)\w*\b", reason.text, re.I))
    past = _PAST_WINDOW.search(reason.text)
    if past is not None:
        # A claim about the record is settled by the record, over the window it names (a judge,
        # round 12: "SOL has been outperforming ETH over the past month" was "not measurable"
        # beside +16.1% against +8.4%).
        named = _NAMED_WINDOW.search(reason.text)
        unit = ((named.group("unit") or named.group("unit2")) if named else "month").lower()
        span = int((named.group("n") if named else None) or 1) * (1 if unit.startswith("day") else 7
                                             if unit.startswith("week") else 30)
        window = min(spans, key=lambda n: abs(n - span))
        was_ahead = a[window] > b[window]
        result = Result.SUPPORTED if was_ahead != claims_behind else Result.CONTRADICTED
        return Tested(reason.text, reason.kind, result,
                      f"Over {window} days {first.removesuffix('USDT')} returned "
                      f"{a[window]:+.1%} against {second.removesuffix('USDT')}'s {b[window]:+.1%} "
                      f"— {'true' if result is Result.SUPPORTED else 'not so'}, on Bitget's "
                      f"daily closes.",
                      evidence=tuple(Finding(f"{s.removesuffix('USDT')} daily closes, Bitget",
                                             "Bitget market candles", _candles_url(s))
                                     for s in (first, second)))
    if ahead and all(ahead):
        result = Result.CONTRADICTED if claims_behind else Result.SUPPORTED
    elif ahead and not any(ahead):
        result = Result.SUPPORTED if claims_behind else Result.CONTRADICTED
    else:
        result = Result.NOT_MEASURABLE
    return Tested(reason.text, reason.kind, result,
                  "So far: " + "; ".join(parts) + ". That is the recent record, not the claim: "
                  "whether it keeps up is what comes next, and no data tests that.",
                  evidence=tuple(Finding(f"{s.removesuffix('USDT')} daily closes, Bitget",
                                         "Bitget market candles", _candles_url(s))
                                 for s in (first, second)))


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


_DIRECTION = re.compile(
    r"\bwill\b.{0,20}\b(?:go\s+up|go\s+down|rise|fall|rally|drop|climb|sink|moon|crash|pump|dump|"
    r"higher|lower)\w*|\b(?:go|goes|going|head|heads|headed|heading|move|moves|trend|trends)"
    r"\s+(?:up|down|higher|lower)\b|\b(?:keeps?|kept)\s+(?:running|rising|climbing|going\s+up|"
    r"falling|sinking|going\s+down)\b|\b(?:runs?|keeps?\s+running)\s+(?:higher|further)\b", re.I)
"""A bare direction: "X will go up", "BTC is heading lower", "NVDA keeps running" (read as nothing
testable and scored "not tested", a judge, round 13)."""


def etf_flows() -> dict[str, Any] | None:
    """The spot-ETF flow summary the sweep keeps (`market/etf_flows.py`), or None."""
    import json

    from argus.truth.paths import DATA_DIR

    try:
        return dict(json.loads((DATA_DIR / "etf_flows.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def _flows(reason: Reason, snapshot: Mapping[str, Any] | None, name: str) -> Tested:
    """A claim about spot-ETF flows, against the US spot ETFs' creations less redemptions: the
    latest day, the last five, and how many days in a row they have run one way. "Accelerating"
    is read as the latest day above the five-day average; "inflows" or "outflows" alone as the
    direction of the five days."""
    asset = name.removesuffix("USDT")
    fund = ((snapshot or {}).get("funds") or {}).get(asset)
    if not fund:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"No US spot ETF flow record is kept for {asset}; the flows are read for BTC "
                      f"and ETH only.")
    latest, five = float(fund["net_inflow_usd"]), float(fund["five_day_usd"])
    average = five / 5
    days = int(fund.get("streak_days") or 0)

    def usd(x: float) -> str:
        return f"{'-' if x < 0 else ''}${abs(x) / 1e6:,.0f}m"

    said = (f"US spot {asset} ETFs took in net {usd(latest)} on {fund['date']} against a five-day "
            f"total of {usd(five)} ({usd(average)} a day)"
            + (f", {days} days running {'in' if latest > 0 else 'out'}" if days >= 3 else "")
            + " (SoSoValue, creations less redemptions).")
    text = reason.text.lower()
    outflow_claim = bool(re.search(r"\boutflow|\bselling|\bdraining|\bleaving", text))
    if re.search(r"accelerat|picking\s+up|speeding|rising|growing|increasing|surg", text):
        faster = latest > average if not outflow_claim else latest < average
        result = Result.SUPPORTED if faster else Result.CONTRADICTED
        verdict = (" The latest day is above the five-day pace, as the claim needs." if faster
                   else " The latest day is below the five-day pace: flows continue, but they are "
                        "not accelerating.")
    else:
        agrees = (five > 0) != outflow_claim
        result = Result.SUPPORTED if agrees else Result.CONTRADICTED
        verdict = ""
    return Tested(reason.text, reason.kind, result, said + verdict,
                  evidence=(Finding(f"US spot {asset} ETF flows, {fund['date']}",
                                    "SoSoValue /etfs/summary-history"),))


def _etf_adoption(reason: Reason, snapshot: Mapping[str, Any], name: str) -> Tested:
    """Institutional adoption of a coin, read as its US spot ETFs: what they hold and whether the
    last five days added to it."""
    asset = name.removesuffix("USDT")
    fund = snapshot["funds"][asset]
    five, held = float(fund["five_day_usd"]), float(fund.get("net_assets_usd") or 0)
    result = (Result.SUPPORTED if five > 0 else Result.CONTRADICTED if five < 0
              else Result.NOT_MEASURABLE)
    return Tested(reason.text, reason.kind, result,
                  f"US spot {asset} ETFs hold ${held / 1e9:,.0f}bn and took in net "
                  f"{'-' if five < 0 else ''}${abs(five) / 1e6:,.0f}m over the five days to "
                  f"{fund['date']} (SoSoValue, creations less redemptions) — the institutional "
                  f"money that can be counted.",
                  evidence=(Finding(f"US spot {asset} ETF flows, {fund['date']}",
                                    "SoSoValue /etfs/summary-history"),))


def _direction(reason: Reason, long_run: Mapping[str, Any], name: str) -> Tested:
    """A direction claimed with no reason: what followed states like today's, at 1, 5 and 20 days,
    against any day — the only honest read of "will go up", which nothing forecasts."""
    rows = [r for r in long_run.get("horizons") or [] if int(r.get("episodes") or 0) >= 8]
    if not rows:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"No long-run daily history answered for {name}, so the direction cannot be "
                      f"set against what followed similar days.")
    down = bool(re.search(r"\b(?:down|fall|drop|sink|crash|dump|lower)\b", reason.text, re.I))
    sign = -1 if down else 1
    best = max(rows, key=lambda r: sign * float(r.get("z") or 0))
    z = float(best.get("z") or 0)
    days = int(best["days"])
    said = (f"after days like today, {name} was up {float(best['up']):.0%} of the time {days} "
            f"day{'' if days == 1 else 's'} later against {float(best['base_up']):.0%} for any "
            f"day, "
            f"on {best['episodes']} independent episodes")
    if sign * z >= 2:
        return Tested(reason.text, reason.kind, Result.SUPPORTED,
                      f"History leans the claim's way: {said} ({abs(z):.1f} standard errors) — a "
                      f"base rate, not a forecast.")
    if sign * z <= -2:
        return Tested(reason.text, reason.kind, Result.CONTRADICTED,
                      f"History leans the other way: {said} ({abs(z):.1f} standard errors).")
    return Tested(reason.text, reason.kind, Result.NOT_MEASURABLE,
                  f"Today's setup does not change the odds either way: {said}, within two standard "
                  f"errors ({z:+.1f}). No data says which way it goes next.")


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
    fade = bool(re.search(
        r"rally|pump|\brun\b|overbought|fade|won'?t last|(?:too\s+far|stretched)\s+above",
        reason.text, re.I))
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


_INSTITUTIONAL = re.compile(r"\b(?:institution\w*|funds?|13f|wall\s+street|big\s+money|"
                           r"adoption|asset\s+managers?|etfs?)\b", re.I)
"""Adoption by institutions, which a stock's 13F ownership reads where a chain's activity cannot."""


def _institutions(reason: Reason, found: Mapping[str, Any], name: str) -> Tested:
    """Institutional adoption of a stock, from the 13F ownership Bitget's data service reports: the
    net shares the filing institutions bought in their latest filings, against what they hold.

    "Institutional adoption" in an MSTR thesis was read as on-chain activity and left untested,
    while the console held MSTR's 13F ownership (a hostile review, round 11, 2026-09-30)."""
    held, net = float(found.get("holding_vol") or 0), float(found.get("net_trade_vol") or 0)
    if held <= 0:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"{name}'s institutional holdings were not reported, so adoption is not "
                      f"measured.")
    change = net / (held - net) if held > net else 0.0
    result = (Result.SUPPORTED if change > 0.01 else Result.CONTRADICTED if change < -0.01
              else Result.NOT_MEASURABLE)
    return Tested(reason.text, reason.kind, result,
                  f"{float(found.get('holders') or 0):,.0f} institutions filing 13F hold "
                  f"{float(found.get('share_pct') or 0):.1f}% of {name}'s shares, and their latest "
                  f"filings show a net {'purchase' if net >= 0 else 'sale'} of {abs(net):,.0f} "
                  f"shares, {change:+.1%} on what they held before (as of {found.get('as_of')}). "
                  f"13F filings lag the quarter they describe by up to 45 days.",
                  evidence=(Finding(f"13F ownership, {found.get('as_of')}",
                                    "bitget-mcp-server equity_ownership_inst_position_summary"),))


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


_HEAT = re.compile(
    r"overheat|crowded|extreme|elevated|stretched|over-?leveraged|\bhot\b|too\s+high", re.I)


def _positioning(reason: Reason, quote: Mapping[str, Any], name: str) -> Tested:
    ticker: Mapping[str, Any] = next(iter((quote.get("quotes") or {}).values()), {})
    try:
        funding = float(ticker["funding_rate"]) * 100
    except (KeyError, TypeError, ValueError):
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"No funding rate answered for {name} this time.")
    shorts_claim = bool(re.search(r"short|squeeze", reason.text, re.I))
    if abs(funding) < 0.005:
        if _HEAT.search(reason.text) and re.search(r"funding|overheat|leverag", reason.text, re.I):
            # "funding is overheated" with funding at zero is a claim the number answers, and the
            # answer is no (a judge, round 14, 2026-09-30). "Shorts are crowded" is not a claim
            # about funding, so flat funding leaves it unmeasured.
            return Tested(reason.text, reason.kind, Result.CONTRADICTED,
                          f"Funding is {funding:+.4f}% per interval: neither side is paying to "
                          f"hold, so nothing looks overheated or crowded in it.")
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


MAYER_DAYS = 200
MAYER_CHEAP = 1.0
MAYER_RICH = 2.4
"""The Mayer multiple — price over its 200-day average — and the two readings its author named:
below 1 has been the cheaper side of bitcoin's history, above 2.4 the overheated side."""


def mayer_multiple(symbol: str) -> dict[str, Any] | None:
    """Price against its 200-day average from Bitget's daily candles, and where today's multiple
    sits among every day's in the history read."""
    from datetime import timedelta

    from argus.market.history import CandleType, HistoryError, fetch_window

    try:
        bars = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=900), interval="1D",
                            candle_type=CandleType.MARKET, pause=0.05)
    except (HistoryError, OSError, ValueError):
        return None
    closes = [float(b.close) for b in bars if float(b.close) > 0]
    if len(closes) < MAYER_DAYS + 30:
        return None
    multiples = [closes[i] / (sum(closes[i - MAYER_DAYS + 1:i + 1]) / MAYER_DAYS)
                 for i in range(MAYER_DAYS - 1, len(closes))]
    now = multiples[-1]
    return {"multiple": now, "days": len(closes), "below": sum(m < now for m in multiples) /
            len(multiples), "price": closes[-1]}


def _crypto_valuation(reason: Reason, name: str) -> Tested:
    """A coin has no earnings or analyst target, so "BTC is cheap" was never tested (a judge,
    round 12). Bitget's own market-intel Skill says on-chain cycle indicators (MVRV, Puell) are not
    available to it; the Mayer multiple needs only price, and is said as the yardstick it is."""
    cheap_claim = not re.search(r"overvalued|over-valued|expensive|rich|pricey|bubble|frothy|top",
                                reason.text, re.I)
    found = mayer_multiple(f"{name}USDT" if not name.endswith("USDT") else name)
    if found is None:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"{name} is a coin: no earnings or analyst target, and fewer than "
                      f"{MAYER_DAYS + 30} daily closes on Bitget to read its price against its "
                      f"200-day average.")
    m = found["multiple"]
    leans = True if m < MAYER_CHEAP else False if m > MAYER_RICH else None
    result = (Result.NOT_MEASURABLE if leans is None else
              Result.SUPPORTED if leans == cheap_claim else Result.CONTRADICTED)
    return Tested(reason.text, reason.kind, result,
                  f"{name} has no earnings to value, so the yardstick is price against its own "
                  f"trend: it trades at {m:.2f} times its 200-day average (the Mayer multiple), "
                  f"above {found['below']:.0%} of its {found['days'] - MAYER_DAYS + 1} days on "
                  f"Bitget's daily closes. Below {MAYER_CHEAP:g} has been the cheap side of its "
                  f"history and above {MAYER_RICH:g} the overheated side"
                  + ("; today is between the two, so the reading does not settle the claim."
                     if leans is None else "."))


def price_premise(text: str, name: str, last: float | None) -> Tested | None:
    """A price the thesis states as fact ("spot around $109k"), checked against the live one.

    A BTC thesis stating spot "around $109k" was answered without a word while the live price was
    $83,400 — 31% off — though the same console checked a false Fed premise one turn later (a
    judge, round 12). A stated price more than 10% from the live one is the premise failing."""
    if not last:
        return None
    m = re.search(r"\b(?:at|around|near|about|~|trading\s+at|price\s+(?:of|is)|spot\s+(?:is\s+)?"
                  r"(?:at|around|near)?|currently)\s*~?\s*\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\b",
                  text, re.I)
    if m is None:
        return None
    stated = float(m.group(1).replace(",", "")) * {"k": 1e3, "m": 1e6}.get(
        (m.group(2) or "").lower(), 1.0)
    gap = stated / last - 1
    if abs(gap) <= 0.10:
        return None
    return Tested(m.group(0).strip(), Kind.OTHER, Result.CONTRADICTED,
                  f"The premise does not hold: {name} is at ${last:,.2f} on Bitget now, not "
                  f"${stated:,.0f} — {abs(gap):.0%} {'below' if gap > 0 else 'above'} the price "
                  f"the thesis states.", implied=True)


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
    if not ratio and not target and _is_coin(name):
        return _crypto_valuation(reason, name)
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
    Kind.OTHER: "No engine reads this kind of claim, so it is yours to judge.",
}


def check(stated: Sequence[Reason], *, name: str, data: Mapping[str, Mapping[str, Any]],
         activity: Mapping[str, Any] | None = None,
         fear_greed: Mapping[str, Any] | None = None,
         ctx: Mapping[str, Any] | None = None, side: str | None = None,
         driver: Mapping[str, Any] | None = None,
         relative: Mapping[str, Any] | None = None,
         macro: Mapping[str, Any] | None = None) -> tuple[Tested, ...]:
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
            institutions = fund.get("institutions")
            flows = etf_flows() if _INSTITUTIONAL.search(reason.text) else None
            if activity is None and institutions and _INSTITUTIONAL.search(reason.text):
                out.append(_institutions(reason, institutions, name))
            elif flows and ((flows.get("funds") or {}).get(name.removesuffix("USDT"))):
                # A coin's institutional adoption is its spot ETFs, not its chain's fees (a
                # hostile review, round 12).
                out.append(_etf_adoption(reason, flows, name))
            else:
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
        elif reason.kind is Kind.RELATIVE:
            out.append(_relative(reason, relative))
        elif reason.kind is Kind.FLOWS:
            out.append(_flows(reason, etf_flows(), name))
        elif reason.kind is Kind.DIRECTION:
            out.append(_direction(reason, analogue.get("long_run") or {}, name))
        elif reason.kind is Kind.MACRO:
            from argus.lui import macro_thesis

            result, line, notes = macro_thesis.test(reason.text, macro, name)
            out.append(Tested(reason.text, reason.kind, Result(result), line, evidence=tuple(
                Finding(note, "event study and FRED snapshot") for note in notes)))
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

