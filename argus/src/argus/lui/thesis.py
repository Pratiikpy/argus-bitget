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
  daily fees and TVL on DeFiLlama, the last 30 days against the 30 before, placed in the
  distribution of every such 30-day change over the past three years;
* **momentum**, **positioning** (funding), **sentiment** (Fear & Greed) and **valuation**
  (analysts' mean target) read the figure the task already holds and say whether it agrees — and
  where agreeing is not an edge (technicals, a target), the line says so.

A reason no engine reads is listed as *not tested*, with what would test it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from argus.truth import http
from argus.truth.failures import RpcError


class Kind(StrEnum):
    REVERSION = "a move reversing"
    ACTIVITY = "network activity"
    MOMENTUM = "momentum"
    POSITIONING = "positioning"
    SENTIMENT = "sentiment"
    VALUATION = "valuation"
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
class Tested:
    reason: str
    kind: Kind
    result: Result
    line: str

    def as_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "kind": self.kind.value, "result": self.result.value,
                "line": self.line}


def reasons(text: str) -> tuple[Reason, ...]:
    """The reasons a thesis states, each with the kind of evidence that could test it. The
    stance and name ("long SOL") are not a reason; a fragment of two words or fewer that names no
    kind is not one either. Empty when the text states no reason."""
    out: list[Reason] = []
    for part in _SPLIT.split(text.strip().rstrip(".?!")):
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
        tvl = http.fetch_json(f"{LLAMA}/v2/historicalChainTvl/{chain}", timeout=timeout)
        out["tvl"] = month_change(_daily(tvl if isinstance(tvl, list) else []))
    except RpcError as exc:
        out["tvl_error"] = exc.kind.value
    return out


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
                                                           ("tvl", "value locked (TVL)"))]
    read = [(label, m) for label, m in parts if m]
    if not read:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"DeFiLlama did not answer for {activity.get('chain')} just now, so the "
                      f"claim was not tested.")
    lines = [f"{activity['chain']} {label}: {m['change']:+.1%} over the last {WINDOW_DAYS} days "
             f"against the {WINDOW_DAYS} before, higher than {m['percentile']:.0%} of the "
             f"{m['months']} monthly changes in three years" for label, m in read]
    ups = [m["percentile"] >= 0.6 and m["change"] > 0 for _, m in read]
    downs = [m["percentile"] <= 0.4 and m["change"] < 0 for _, m in read]
    weakening = bool(re.search(r"weak|slow|declin|fall|drop|shrink", reason.text, re.I))
    note = (" Both are in dollars, so a falling token price lowers them without any change in "
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
                  f"{lead}: " + "; ".join(lines) + f" (DeFiLlama, to {read[0][1]['as_of']})."
                  + note)


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


def _valuation(reason: Reason, fund: Mapping[str, Any], name: str) -> Tested:
    target, price = fund.get("target_mean"), fund.get("price")
    if not target or not price:
        return Tested(reason.text, reason.kind, Result.NOT_TESTED,
                      f"No analyst target answered for {name}; ARGUS does not value companies "
                      f"itself.")
    gap = float(target) / float(price) - 1
    cheap_claim = not re.search(r"overvalued|expensive|rich", reason.text, re.I)
    if abs(gap) < 0.10:
        result = Result.NOT_MEASURABLE
    else:
        result = Result.SUPPORTED if (gap > 0) == cheap_claim else Result.CONTRADICTED
    return Tested(reason.text, reason.kind, result,
                  f"Analysts' mean target is ${float(target):,.2f}, {abs(gap):.0%} "
                  f"{'above' if gap > 0 else 'below'} the last close — an opinion, not a "
                  f"measurement, and the only valuation read in this task.")


_UNTESTED: Mapping[Kind, str] = {
    Kind.EARNINGS: "The earnings step reports the last surprise and the next date, but no engine "
                   "tests a claim about future earnings.",
    Kind.MACRO: "No engine in this task tests a macro claim; the console's macro answers "
                "(`/ask`, FRED) do, one question at a time.",
    Kind.OTHER: "No engine reads this kind of claim, so it is yours to judge.",
}


def check(stated: Sequence[Reason], *, name: str, data: Mapping[str, Mapping[str, Any]],
         activity: Mapping[str, Any] | None = None,
         fear_greed: Mapping[str, Any] | None = None) -> tuple[Tested, ...]:
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
        else:
            out.append(Tested(reason.text, reason.kind, Result.NOT_TESTED, _UNTESTED[reason.kind]))
    return tuple(out)


def fear_greed_now() -> dict[str, Any] | None:
    """The crypto Fear & Greed reading, through the Skill mirror's own reader."""
    from argus.market import skill_mirror

    try:
        return skill_mirror.fear_greed({})
    except Exception:
        return None

