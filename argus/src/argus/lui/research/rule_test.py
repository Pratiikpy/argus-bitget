"""A trading rule in plain words, run as a backtest, with fees, out of sample and by regime.

"Backtest buying BTC when RSI drops below 30 and selling above 70" was answered with today's RSI,
"how would a 50/200 moving-average crossover on ETH have done?" with today's moving averages, and
"buy NVDA when it falls 3% in a day, hold a week" was declined (a check of the console, 2026-10-05).
The engine to answer them was already here (`backtest/engine.py`: fees cannot be omitted, a signal
from bar *t* trades from *t* to *t+1*, a chronological out-of-sample split); nothing read the rule.

**Rules read** (long by default; "short when" or "long/short" adds the other side):

- RSI thresholds — "buy when RSI(14) is below 30, sell above 70";
- moving-average crossover — "50/200 day moving average crossover", "20 and 50 EMA cross";
- price against a moving average — "hold BTC when it is above its 200-day average";
- after a move — "buy NVDA when it falls 3% in a day, hold a week" (a dip) or "when it rises 5%"
  (momentum), held the stated days or five if none is said, and the answer says it assumed five;
- breakout — "buy the 20-day high, sell the 10-day low".

**What every answer carries.** Net of Bitget's taker fee on every change of position; against
buy-and-hold over the same days; the out-of-sample stretch on its own and the engine's decay alert;
the share of trades that made money with a Wilson interval (`backtest/proportion.py`) and whether
the first and second halves agree; and **the regime split** (build-list 3.5): every day labelled
bull, bear or chop by the market's own trailing 90-day move (above +20%, below -20%, between),
known on that day, never after it; the rule's return and buy-and-hold's within each, and a plain
sentence when the edge lives in one regime. A rule that only beats holding in a bear market is a
bear-market rule, and saying so is the point of the split.

Prices: Bitget's daily closes; a US stock is tested on the stock's own daily closes (Yahoo,
split-adjusted) because Bitget's perpetual on it is often a year old, and the answer says which.
Funding on a held perpetual is not in the engine's cost; a line estimates it from the last 30 days
of settlements for the share of days the rule was in the market.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:backtest\w*|back[\s-]test\w*|test\s+(?:a|this|my|the)\s+(?:strategy|rule|system)|"
    r"how\s+(?:would|did|does|has)\b[^?]{0,60}\b(?:have\s+)?(?:done|do|performed|perform|worked|work|fared)|"
    r"does\s+(?:buying|selling|shorting|a|the)\b[^?]{0,80}\bwork|would\s+(?:have\s+)?made\s+money|"
    r"(?:strategy|rule)\s*:)", re.I)
_RSI: Final = re.compile(r"\brsi\s*(?:\(\s*(?P<n>\d{1,3})\s*\))?[^.;]{0,40}?\b(?:below|under|<|"
                         r"drops?\s+(?:below|under|to)|falls?\s+(?:below|under|to))\s*(?P<lo>\d{1,2})"
                         r"(?:[^.;]{0,60}?\b(?:above|over|>|rises?\s+(?:above|over|to)|crosses\s+"
                         r"above)\s*(?P<hi>\d{1,2}))?", re.I)
_CROSS: Final = re.compile(r"\b(?P<fast>\d{1,3})\s*(?:/|and|-|vs\.?)\s*(?P<slow>\d{1,3})[\s-]*"
                           r"(?:day\s+|d\s+)?(?P<kind>e?ma|sma|moving\s+averages?|exponential)"
                           r"[^.?]{0,30}\bcross\w*|\b(?P<fast2>\d{1,3})\s*(?:/|and|-|vs\.?)\s*"
                           r"(?P<slow2>\d{1,3})\s*(?:day\s+)?(?P<kind2>e?ma|sma)?\s*cross\w*", re.I)
_ABOVE_MA: Final = re.compile(r"\babove\s+(?:its\s+|the\s+)?(?P<n>\d{1,3})[\s-]*(?:day|d|week)?"
                              r"[\s-]*(?P<kind>e?ma|sma|moving\s+average|average)\b", re.I)
_MOVE: Final = re.compile(r"\b(?P<verb>fall(?:s|en)?|fell|drop(?:s|ped)?|dip(?:s|ped)?|down|"
                          r"los(?:e|es|t)|rise[sn]?|rose|jump(?:s|ed)?|gain(?:s|ed)?|up|"
                          r"rall(?:y|ies|ied))\s+(?:by\s+)?(?:more\s+than\s+|at\s+least\s+|over\s+)?"
                          r"(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?:or\s+more\s+)?(?:in\s+(?:a|one)\s+day|"
                          r"on\s+the\s+day|in\s+a\s+session|daily|that\s+day)?", re.I)
_HOLD: Final = re.compile(r"\bhold(?:ing|s)?\s+(?:it\s+|for\s+)?(?P<n>\d{1,3}|a|one|two|three)"
                          r"\s*(?P<unit>days?|weeks?|months?|sessions?)\b", re.I)
_BREAKOUT: Final = re.compile(r"\b(?P<n>\d{1,3})[\s-]*day\s+(?:high|highs|breakout)\b(?:[^.?]{0,40}"
                              r"\b(?P<m>\d{1,3})[\s-]*day\s+low)?", re.I)
_SHORT: Final = re.compile(r"\bshort\s+(?:when|if|it|on)|\blong[\s/-]*short\b|\bboth\s+sides\b",
                           re.I)
VOL_SIZED: Final = re.compile(r"\b(?:vol(?:atility)?[\s-]*(?:target\w*|sized?|sizing|scaled?|"
                              r"adjusted)|siz\w+\s+(?:it\s+)?by\s+vol\w*|garch|risk[\s-]*parity)\b",
                              re.I)
"""Asked to size the rule by its volatility, as well as to test it."""
DEFAULT_HOLD: Final = 5
DAYS_BACK: Final = 1825
REGIME_DAYS: Final = 90
REGIME_EDGE: Final = 0.20


@dataclass(frozen=True)
class Rule:
    kind: str
    label: str
    params: dict[str, float]
    short: bool = False
    assumed: tuple[str, ...] = ()


def _count(n: str) -> int:
    return int(n) if n.isdigit() else {"a": 1, "one": 1, "two": 2, "three": 3}[n.lower()]


def read_rule(text: str) -> Rule | None:
    """The rule a question states, or None when it states none this module reads."""
    short = _SHORT.search(text) is not None
    assumed: tuple[str, ...] = ()
    rsi = _RSI.search(text)
    if rsi is not None:
        n = int(rsi.group("n") or 14)
        lo = float(rsi.group("lo"))
        hi = float(rsi.group("hi") or 100 - lo)
        assumed = () if rsi.group("hi") else (f"exit when RSI rises above {hi:g} (not stated)",)
        return Rule("rsi", f"long when RSI({n}) is below {lo:g}, out above {hi:g}"
                    + (f"; short above {hi:g}, out below {lo:g}" if short else ""),
                    {"n": n, "lo": lo, "hi": hi}, short, assumed)
    cross = _CROSS.search(text)
    if cross is not None:
        fast = int(cross.group("fast") or cross.group("fast2"))
        slow = int(cross.group("slow") or cross.group("slow2"))
        fast, slow = min(fast, slow), max(fast, slow)
        kind = (cross.group("kind") or cross.group("kind2") or "").lower()
        ema = kind.startswith("e")
        name = "EMA" if ema else "simple moving average"
        return Rule("cross", f"long when the {fast}-day {name} is above the {slow}-day"
                    + ("; short when below" if short else ", flat when below"),
                    {"fast": fast, "slow": slow, "ema": float(ema)}, short)
    breakout = _BREAKOUT.search(text)
    if breakout is not None:
        n = int(breakout.group("n"))
        m = int(breakout.group("m") or max(2, n // 2))
        assumed = () if breakout.group("m") else (f"exit at a {m}-day low (not stated)",)
        return Rule("breakout", f"long on a close above the prior {n}-day high, out on a close "
                    f"below the prior {m}-day low", {"n": n, "m": m}, False, assumed)
    above = _ABOVE_MA.search(text)
    if above is not None and not _MOVE.search(text):
        n = int(above.group("n"))
        ema = (above.group("kind") or "").lower().startswith("e")
        return Rule("above", f"long when the close is above its {n}-day "
                    f"{'EMA' if ema else 'average'}" + ("; short below" if short else
                                                        ", flat below"),
                    {"n": n, "ema": float(ema)}, short)
    move = _MOVE.search(text)
    if move is not None:
        down = re.match(r"fall|fell|drop|dip|down|los", move.group("verb"), re.I) is not None
        pct = float(move.group("pct")) / 100
        held = _HOLD.search(text)
        if held is not None:
            count = _count(held.group("n"))
            unit = held.group("unit").lower()
            days = count * (7 if unit.startswith("week") else 30 if unit.startswith("month")
                            else 1)
            sessions = count * (5 if unit.startswith("week") else 21 if unit.startswith("month")
                                else 1)
            assumed = ()
        else:
            days = sessions = DEFAULT_HOLD
            assumed = (f"held {DEFAULT_HOLD} days (no holding period stated)",)
        side = "short" if short and not down else "long"
        return Rule("move", f"{'buy' if side == 'long' else 'short'} at the close of a day it "
                    f"{'fell' if down else 'rose'} {pct:.1%} or more, hold {days} days",
                    {"pct": pct, "down": float(down), "days": days, "sessions": sessions,
                     "short": float(short)},
                    False, assumed)
    return None


# --- the rule as a position, built causally -------------------------------------------------------

def _sma(closes: Sequence[float], n: int) -> list[float | None]:
    out: list[float | None] = []
    total = 0.0
    for i, c in enumerate(closes):
        total += c
        if i >= n:
            total -= closes[i - n]
        out.append(total / n if i >= n - 1 else None)
    return out


def _ema(closes: Sequence[float], n: int) -> list[float | None]:
    out: list[float | None] = []
    k, value = 2 / (n + 1), None
    for i, c in enumerate(closes):
        value = c if value is None else value + k * (c - value)
        out.append(value if i >= n - 1 else None)
    return out


def _rsi(closes: Sequence[float], n: int) -> list[float | None]:
    """Wilder's RSI: the first average a simple mean of n changes, then smoothed by 1/n."""
    out: list[float | None] = [None] * len(closes)
    gain = loss = 0.0
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        up, down = max(change, 0.0), max(-change, 0.0)
        if i <= n:
            gain += up / n
            loss += down / n
            if i < n:
                continue
        else:
            gain = (gain * (n - 1) + up) / n
            loss = (loss * (n - 1) + down) / n
        out[i] = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    return out


def positions(rule: Rule, closes: Sequence[float]) -> list[float]:
    """The weight to hold from each close to the next, from data up to that close only."""
    p = rule.params
    weights: list[float] = []
    held, hold_left = 0.0, 0
    if rule.kind == "rsi":
        rsi = _rsi(closes, int(p["n"]))
        for value in rsi:
            if value is not None:
                if held == 0 and value < p["lo"]:
                    held = 1.0
                elif held == 0 and rule.short and value > p["hi"]:
                    held = -1.0
                elif held > 0 and value > p["hi"]:
                    held = -1.0 if rule.short else 0.0
                elif held < 0 and value < p["lo"]:
                    held = 1.0
            weights.append(held)
    elif rule.kind == "cross":
        average = _ema if p["ema"] else _sma
        fast, slow = average(closes, int(p["fast"])), average(closes, int(p["slow"]))
        for f, s in zip(fast, slow, strict=True):
            weights.append(0.0 if f is None or s is None else
                           1.0 if f > s else (-1.0 if rule.short else 0.0))
    elif rule.kind == "above":
        line = (_ema if p["ema"] else _sma)(closes, int(p["n"]))
        for c, m in zip(closes, line, strict=True):
            weights.append(0.0 if m is None else 1.0 if c > m else (-1.0 if rule.short else 0.0))
    elif rule.kind == "breakout":
        n, m = int(p["n"]), int(p["m"])
        for i, c in enumerate(closes):
            if i >= n and held == 0 and c > max(closes[i - n:i]):
                held = 1.0
            elif i >= m and held > 0 and c < min(closes[i - m:i]):
                held = 0.0
            weights.append(held)
    elif rule.kind == "move":
        side = -1.0 if p["short"] and not p["down"] else 1.0
        for i, c in enumerate(closes):
            if hold_left > 0:
                hold_left -= 1
                if hold_left == 0:
                    held = 0.0
            if i >= 1 and held == 0:
                change = c / closes[i - 1] - 1
                if (change <= -p["pct"]) if p["down"] else (change >= p["pct"]):
                    held, hold_left = side, int(p["days"])
            weights.append(held)
    return weights


# --- regimes ----------------------------------------------------------------------------------

def regimes(closes: Sequence[float], days: int = REGIME_DAYS,
            edge: float = REGIME_EDGE) -> list[str | None]:
    """Each day's regime from the trailing ``days`` move, known on that day."""
    out: list[str | None] = []
    for i, c in enumerate(closes):
        if i < days:
            out.append(None)
            continue
        move = c / closes[i - days] - 1
        out.append("bull" if move >= edge else "bear" if move <= -edge else "chop")
    return out


@dataclass(frozen=True)
class RegimeRow:
    regime: str
    days: int
    rule: float
    hold: float

    @property
    def excess(self) -> float:
        return self.rule - self.hold


def regime_split(closes: Sequence[float], net: Sequence[float],
                 labels: Sequence[str | None]) -> list[RegimeRow]:
    """The rule's compounded net return and buy-and-hold's within each regime's days."""
    rows = []
    for name in ("bull", "chop", "bear"):
        days = [i for i in range(len(net)) if labels[i] == name]
        if not days:
            continue
        rule = math.prod(1 + net[i] for i in days) - 1
        hold = math.prod(closes[i + 1] / closes[i] for i in days) - 1
        rows.append(RegimeRow(name, len(days), rule, hold))
    return rows


def regime_verdict(rows: Sequence[RegimeRow]) -> str:
    """A plain sentence on where the rule beat holding, if anywhere. Beating holding by losing
    less is said as that, not as an edge."""
    if not rows:
        return "too little history to label regimes"
    beat = [r for r in rows if r.excess > 0]
    if not beat:
        return "it did not beat holding in any regime"

    def how(r: RegimeRow) -> str:
        return (f"in {r.regime} markets ({r.rule:+.0%} against {r.hold:+.0%})" if r.rule > 0
                else f"in {r.regime} markets only by losing less ({r.rule:+.0%} against "
                     f"{r.hold:+.0%})")

    if len(beat) == len(rows):
        return "it beat holding in every regime it met"
    trailed = " and ".join(r.regime for r in rows if r not in beat)
    if len(beat) == 1 and beat[0].rule > 0:
        one = beat[0]
        return (f"the edge lives in the {one.regime} regime alone ({one.rule:+.0%} against "
                f"{one.hold:+.0%}); in {trailed} markets it trailed holding")
    return f"it beat holding {' and '.join(how(r) for r in beat)}, and trailed in {trailed} markets"


# --- the answer ---------------------------------------------------------------------------------

YAHOO_SERIES: Final = {"XAUUSDT": ("GC=F", "gold futures"), "XAGUSDT": ("SI=F", "silver futures"),
                       "CLUSDT": ("CL=F", "WTI crude futures"),
                       "SP500USDT": ("^GSPC", "the S&P 500 index"),
                       "NDX100USDT": ("^NDX", "the Nasdaq-100 index")}
"""Markets whose Bitget perpetual is about a year old, tested on the long series it tracks: a
200-day rule on gold met 290 days of Bitget history and made one trade (2026-10-05)."""


def _closes(symbol: str) -> tuple[list[datetime], list[float], str]:
    from argus.lui.research.parse import is_us_equity

    if is_us_equity(symbol) or symbol in YAHOO_SERIES:
        from argus.market.equity_history import daily

        ticker, what = YAHOO_SERIES.get(symbol, (symbol.removesuffix("USDT"), ""))
        days = daily(ticker)[-int(DAYS_BACK * 252 / 365):]
        return ([datetime(d.day.year, d.day.month, d.day.day, tzinfo=UTC) for d in days],
                [d.close for d in days],
                f"the daily closes of {what} ({ticker}, Yahoo Finance) that Bitget's perpetual "
                f"tracks" if what else
                "the stock's own daily closes (Yahoo Finance, split-adjusted)")
    from argus.market.history import fetch_window

    bars = fetch_window(symbol, start=datetime.now(UTC) - timedelta(days=DAYS_BACK),
                        interval="1Dutc", pause=0.05)
    pairs = [(b.ts, float(b.close)) for b in bars if float(b.close) > 0]
    return [t for t, _ in pairs], [c for _, c in pairs], "Bitget's daily closes"


def lines(text: str) -> list[str] | None:
    """The backtest of the rule a question states; None when it is not one."""
    if not ASKED.search(text):
        return None
    rule = read_rule(text)
    if rule is None:
        return None
    from argus.lui.research import research_symbols

    named = research_symbols(text)[0]
    if not named:
        return None
    symbol = named[0]
    name = symbol.removesuffix("USDT")
    try:
        stamps, closes, source = _closes(symbol)
    except Exception:
        return [f"Bottom line: {name}'s daily history did not load, so the rule cannot be tested "
                f"right now."]
    if len(closes) < 250:
        return [f"Bottom line: only {len(closes)} days of {name} history are available, too few "
                f"to test a rule on — a year is the least this console tests."]
    if rule.kind == "move" and "Yahoo" in source:
        # a week on a market that shuts is five sessions, not seven rows (2026-10-05)
        rule = replace(rule, params={**rule.params, "days": rule.params["sessions"]},
                       label=rule.label.replace(f"hold {int(rule.params['days'])} days",
                                                f"hold {int(rule.params['sessions'])} sessions"))
    warmup = int(max([rule.params.get(k, 0) for k in ("n", "slow", "m")] + [0]))
    if len(closes) - warmup < 250:
        return [f"Bottom line: {name} has {len(closes)} days of history here and the rule needs "
                f"{warmup} of them to start, leaving too few to test on — a year is the least "
                f"this console tests."]
    from argus.backtest.engine import BacktestResult, Bar, extract_trades, run
    from argus.backtest.proportion import rate_phrase, stability_phrase
    from argus.cost.model import CostModel

    weights = positions(rule, closes)
    bars = [Bar(ts=t, close=Decimal(str(c))) for t, c in zip(stamps, closes, strict=True)]
    yearly = 252 if "Yahoo" in source else 365
    result = run(rule.label, symbol, bars, lambda _bars, i: weights[i],
                 cost=CostModel.bitget_perp(), periods_per_year=yearly)
    hold = closes[-1] / closes[0] - 1
    net = result.net
    trades = extract_trades(bars, list(result.weights), symbol=symbol)
    won = [t.return_pct > 0 for t in trades]
    in_market = sum(1 for w in result.weights if w) / max(1, len(result.weights))
    years = (stamps[-1] - stamps[0]).days / 365.25
    labels = regimes(closes)
    split = regime_split(closes, list(result.net_returns), labels)
    beat = net.total_return > hold
    lead = (f"Bottom line: the rule ({rule.label}) returned {net.total_return:+.0%} on {name} "
            f"after fees over {years:.1f} years, against {hold:+.0%} for simply holding — "
            + ("it beat holding" if beat else "it trailed holding")
            + f"; {regime_verdict(split)}.")
    out = [lead]
    out.append(f"The rule: Sharpe {net.sharpe:.2f}, worst drawdown {net.max_drawdown:.0%}, in the "
               f"market {in_market:.0%} of days, {len(trades)} trades; trades that made money: "
               + (rate_phrase(sum(won), len(won), noun="trades") if trades else "none") + ".")
    stable = stability_phrase(won) if trades else None
    if stable is not None:
        out.append(f"Stability: {stable}.")
    if result.out_of_sample is not None and result.in_sample is not None:
        decayed = bool((result.decay or {}).get("alert"))
        out.append(f"Out of sample (the last 35% of the days, never used to choose anything): "
                   f"{result.out_of_sample.total_return:+.0%}, Sharpe "
                   f"{result.out_of_sample.sharpe:.2f}, against {result.in_sample.sharpe:.2f} "
                   f"before it" + (" — the edge decayed" if decayed else "") + ".")
    if split:
        out.append("By regime (each day labelled by the trailing 90-day move: bull above +20%, "
                   "bear below -20%, chop between, known on the day): " + "; ".join(
                       f"{r.regime} {r.days} days, rule {r.rule:+.0%} vs hold {r.hold:+.0%}"
                       for r in split) + ".")
    if VOL_SIZED.search(text) and len(closes) > 560:
        # build-list 1.6: the same rule sized by a walk-forward GARCH forecast, both after fees
        from argus.backtest import vol_target

        multipliers, target = vol_target.sizes(closes, yearly)
        first = next(k for k, m in enumerate(multipliers) if m is not None)
        # both arms over the same days: the sized arm has no forecast for its first 500, which
        # held the 2022 bear, and a comparison across different days measures the days
        span = bars[first:]
        fixed_w, sized_w = weights[first:], [w * (m or 0.0) for w, m in
                                             zip(weights[first:], multipliers[first:],
                                                 strict=True)]
        def arm(name: str, held: list[float]) -> BacktestResult:
            return run(name, symbol, span, lambda _bars, i: held[i],
                       cost=CostModel.bitget_perp(), periods_per_year=yearly,
                       max_weight=vol_target.MAX_SIZE)

        arms = [arm("fixed", fixed_w), arm("sized by volatility", sized_w)]
        fx, vt = arms[0].net, arms[1].net
        out.append(f"Sized by forecast volatility (GARCH(1,1) walk-forward, aiming at "
                   f"{target:.0f}% a year, sizes 0.25x to 2x), compared from "
                   f"{stamps[first]:%d %b %Y} when its first forecast exists: Sharpe "
                   f"{vt.sharpe:.2f} against {fx.sharpe:.2f} at fixed size, worst drawdown "
                   f"{vt.max_drawdown:.0%} against {fx.max_drawdown:.0%}, fees "
                   f"{arms[1].total_cost_bps:.0f}bps against {arms[0].total_cost_bps:.0f}bps — "
                   + ("better risk-adjusted here" if vt.sharpe > fx.sharpe else
                      "no better risk-adjusted here")
                   + ". On the console's own test (an EMA crossover on BTC, ETH and SOL) it lost "
                     "to fixed size after fees: register #52.")
    if result.cost_destroyed_the_edge:
        out.append(f"Fees turned it from a gain ({result.gross.total_return:+.0%} before them) "
                   f"into a loss.")
    try:
        if "Bitget" in source:
            from argus.market.crossasset_feed import fetch_funding

            since = (datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000
            rates = [r for t, r in fetch_funding(symbol) if t >= since]
            if rates:
                yearly_funding = sum(rates) / len(rates) * 3 * 365 * in_market
                out.append(f"Funding is not in the figures: on the perpetual, at the last 30 "
                           f"days' rate, being in the market {in_market:.0%} of days would have "
                           f"cost about {yearly_funding:.1%} a year more.")
    except Exception:
        pass
    assumed = "; ".join(rule.assumed)
    out.append(f"How: {source}, {len(closes)} days to {stamps[-1]:%d %b %Y}; a signal from a "
               f"day's close is traded at that close and earns from there (the engine never uses "
               f"a later price), Bitget's taker fee of 0.06% on every change"
               + (f"; assumed: {assumed}" if assumed else "")
               + ". A past test, not a forecast; it runs the rule as written, with no tuning.")
    return out
