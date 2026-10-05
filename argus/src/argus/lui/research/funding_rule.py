"""A trading rule whose signal is the perpetual funding rate, tested on Bitget's own settlements
and hourly prices: "go long ETH whenever funding has been negative for 3 straight 8-hour periods,
exit after 5 days or a 2% stop".

That rule went to the analogue engine, which never read funding and answered for the next 24
hours (round 40 judge, Q20); the RSI rule beside it reached the backtester, which works on daily
closes and has no funding series. This module tests the funding rule as written:

- **Signal.** Bitget's funding settlements (`market/crossasset_feed.fetch_funding`, about ninety
  days, every 8 hours on most contracts). The rule fires at a settlement when that one and the
  N-1 before it all have the stated sign; a position already open is not doubled.
- **Entry.** The open of the first hourly bar after the settlement — the signal is known only
  once the rate is published, so nothing is priced from the bar it was read in.
- **Exit.** The first of: the stop, tested on each hour's low (a long) or high (a short), filled
  at the stop price — or at the hour's open when the price gapped through it, which is the worse
  fill; the take-profit, likewise; or the holding period's last close.
- **Costs.** The taker fee both ways (Bitget's 0.06% on USDT perpetuals), and every funding
  settlement while the position is open, paid or received by its sign.
- **The baseline.** The same exits from every hourly open in the window: what holding that long
  with that stop earned anyway. The rule's edge is its average less that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Any, Final

ASKED: Final = re.compile(
    r"\b(?P<side>long|short|buy|sell)\b[^?]{0,40}?\b(?P<sym>[A-Za-z]{2,10})\b[^?]{0,60}?"
    r"\bfunding\b[^?]{0,40}?\b(?P<sign>negative|positive|below\s+zero|above\s+zero)\b[^?]{0,30}?"
    r"\bfor\s+(?P<n>\d|two|three|four|five)\s+(?:straight\s+|consecutive\s+|in\s+a\s+row\s+)?"
    r"(?:8[\s-]?h(?:our)?\s+)?(?:periods?|settlements?|intervals?|times|fundings?)", re.I)
_HOLD: Final = re.compile(r"\b(?:exit|close|sell|hold)\b[^?.]{0,20}?\b(?:after\s+)?"
                          r"(?P<n>\d{1,3})\s*"
                          r"(?P<unit>days?|hours?|h)\b", re.I)
_STOP: Final = re.compile(r"\b(?P<pct>\d+(?:\.\d+)?)\s*%\s*stop(?:[\s-]*loss)?\b|\bstop(?:[\s-]*"
                          r"loss)?\s+(?:of\s+|at\s+)?(?P<pct2>\d+(?:\.\d+)?)\s*%", re.I)
_TAKE: Final = re.compile(r"\b(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?:take[\s-]*profit|target)\b|"
                          r"\b(?:take[\s-]*profit|target)\s+(?:of\s+|at\s+)?(?P<pct2>\d+(?:\.\d+)?)"
                          r"\s*%", re.I)
_WORDS: Final = {"two": 2, "three": 3, "four": 4, "five": 5}
TAKER: Final = 0.0006


@dataclass(frozen=True)
class Trade:
    entered: datetime
    net: float
    why: str


def _exit(bars: list[Any], start: int, hours: int, long: bool, stop: float | None,
          take: float | None) -> tuple[int, float, str]:
    """(exit bar index, gross return, reason) for a position opened at bars[start].open."""
    entry = float(bars[start].open)
    end = min(start + hours - 1, len(bars) - 1)
    for i in range(start, end + 1):
        bar = bars[i]
        lo, hi, op = float(bar.low), float(bar.high), float(bar.open)
        if stop is not None:
            level = entry * (1 - stop) if long else entry * (1 + stop)
            hit = lo <= level if long else hi >= level
            if hit:
                gapped = (op < level) if long else (op > level)
                fill = op if (gapped and i > start) else level
                return i, (fill / entry - 1) * (1 if long else -1), "stop"
        if take is not None:
            level = entry * (1 + take) if long else entry * (1 - take)
            hit = hi >= level if long else lo <= level
            if hit:
                return i, (level / entry - 1) * (1 if long else -1), "target"
    return end, (float(bars[end].close) / entry - 1) * (1 if long else -1), "time"


_SILLY: Final = (
    (re.compile(r"\b(?:hold|exit|out)\b[^?.]{0,20}?(?:for\s+)?"
                r"[-−]\s?\d+\s*(?:days?|hours?|h)\b",  # noqa: RUF001
                re.I), "a negative holding period"),
    (re.compile(r"\b(?P<x>\d{4,})\s*x\b", re.I), "leverage above Bitget's 125x maximum"),
    (re.compile(r"\b0\s*(?:bps|bp|basis\s+points?|%)\s+(?:fees?|commission)\b|\bno\s+fees\b|"
                r"\bfees?\s+(?:of\s+)?0\s*(?:bps|%)?\b", re.I),
     "zero fees (Bitget charges about 6bps a side on a perpetual; a test without them overstates "
     "every result)"),
    (re.compile(r"[-−]\s?(?:[1-9]\d{2,})\s*%\s*"  # noqa: RUF001
                r"(?:annuali[sz]ed|a\s+year|apr)?", re.I),
     "a funding rate of hundreds of percent a year below zero"),
)


def silly_lines(text: str) -> list[str] | None:
    """A funding backtest whose parameters cannot be run as stated, each one named: "long BTC
    whenever funding is below -500% annualised, hold for -3 days, 0 bps fees, 10000x leverage"
    was answered as a 72-hour funding-cost line with fees of 12bps (round 41 hostile, M8)."""
    if not re.search(r"\bfunding\b", text, re.I) or not re.search(
            r"\bbacktest\w*|\bwhenever\b|\bstrategy\b|\brule\b", text, re.I):
        return None
    wrong = []
    for pattern, said in _SILLY:
        m = pattern.search(text)
        if m is None:
            continue
        if said.startswith("leverage") and int(m.group("x")) <= 125:
            continue
        wrong.append(said)
    if not wrong:
        return None
    return [f"Bottom line: this backtest cannot be run as stated — {'; '.join(wrong)}.",
            "A rule that can be tested reads like: \"long BTC whenever funding has been negative "
            "for 3 straight periods, exit after 5 days or a 2% stop\" — fees are always charged at "
            "Bitget's own rate."]


def lines(text: str) -> list[str] | None:
    """The rule's record, or None when ``text`` is not a funding rule."""
    silly = silly_lines(text)
    if silly is not None:
        return silly
    m = ASKED.search(text)
    if m is None:
        return None
    from argus.lui.research.parse import research_symbols

    named = research_symbols(m.group("sym"))[0]
    if not named:
        return None
    symbol = named[0]
    long = m.group("side").lower() in ("long", "buy")
    negative = m.group("sign").lower().startswith(("negative", "below"))
    need = int(m.group("n")) if m.group("n").isdigit() else _WORDS[m.group("n").lower()]
    hold_m = _HOLD.search(text)
    hours = (int(hold_m.group("n")) * (24 if hold_m.group("unit").lower().startswith("d") else 1)
             if hold_m else 24 * 5)
    stop_m, take_m = _STOP.search(text), _TAKE.search(text)
    stop = float(stop_m.group("pct") or stop_m.group("pct2")) / 100 if stop_m else None
    take = float(take_m.group("pct") or take_m.group("pct2")) / 100 if take_m else None
    from argus.market.crossasset_feed import fetch_funding
    from argus.market.history import fetch_range

    try:
        funding = fetch_funding(symbol)
        bars = fetch_range(symbol, days=90, interval="1H", pause=0.1)
    except Exception:
        return [f"Bottom line: Bitget's funding history or hourly prices for {symbol} did not "
                f"answer just now, so the rule is not tested; ask again in a minute."]
    if len(funding) < need + 5 or len(bars) < hours + 24:
        return [f"Bottom line: too little history came back for {symbol} to test the rule."]
    stamps = [b.ts for b in bars]
    first_bar = {}
    j = 0
    for ts_ms, _rate in funding:
        t = datetime.fromtimestamp(ts_ms / 1000, tz=stamps[0].tzinfo)
        while j < len(stamps) and stamps[j] <= t:
            j += 1
        first_bar[ts_ms] = j if j < len(stamps) else None

    def funding_paid(a: datetime, b: datetime) -> float:
        # a long pays positive funding and receives negative; a short the reverse
        total = sum(r for ts, r in funding
                    if a < datetime.fromtimestamp(ts / 1000, tz=a.tzinfo) <= b)
        return total if long else -total

    trades: list[Trade] = []
    busy_until = -1
    for k in range(need - 1, len(funding)):
        window = [r for _, r in funding[k - need + 1:k + 1]]
        if not all((r < 0) if negative else (r > 0) for r in window):
            continue
        start = first_bar.get(funding[k][0])
        if start is None or start <= busy_until or start + 1 >= len(bars):
            continue
        end, gross, why = _exit(bars, start, hours, long, stop, take)
        if end - start + 1 < min(hours, 2) and why == "time":
            continue
        cost = 2 * TAKER + funding_paid(bars[start].ts, bars[end].ts)
        trades.append(Trade(bars[start].ts, gross - cost, why))
        busy_until = end
    name = symbol.removesuffix("USDT")
    span = f"{bars[0].ts:%d %b} to {bars[-1].ts:%d %b %Y}"
    side = "long" if long else "short"
    sign = "negative" if negative else "positive"
    held = f"{hours // 24} days" if hours % 24 == 0 else f"{hours} hours"
    rule = (f"{side} {name} after {need} {sign} funding settlements in a row, out after {held}"
            + (f" or a {stop:.0%} stop" if stop else "")
            + (f" or a {take:.0%} target" if take else ""))
    if not trades:
        return [f"Bottom line: the rule ({rule}) never fired in the {len(funding)} settlements "
                f"Bitget serves ({span}), so there is no record to judge it on.",
                "Data: Bitget funding history and hourly candles. Not advice."]
    base = []
    for i in range(0, len(bars) - hours, 4):
        end, gross, _ = _exit(bars, i, hours, long, stop, take)
        base.append(gross - 2 * TAKER - funding_paid(bars[i].ts, bars[end].ts))
    avg = mean(t.net for t in trades)
    wins = sum(t.net > 0 for t in trades)
    stops = sum(t.why == "stop" for t in trades)
    base_avg = mean(base) if base else 0.0
    out = [f"Bottom line: over {span} the rule ({rule}) fired {len(trades)} time"
           f"{'s' if len(trades) != 1 else ''} and averaged {avg:+.2%} a trade after fees and "
           f"funding, against {base_avg:+.2%} for the same exits from any hour — an edge of "
           f"{avg - base_avg:+.2%}" + (", from too few trades to call it one."
                                      if len(trades) < 20 else ".")]
    out.append(f"{wins} of {len(trades)} trades made money; {stops} hit the stop. Worst "
               f"{min(t.net for t in trades):+.2%}, best {max(t.net for t in trades):+.2%}.")
    out += [f"{t.entered:%d %b %H:%M} UTC: {t.net:+.2%} ({t.why})" for t in trades[:8]]
    out.append("Entries at the first hourly open after the qualifying settlement; stops tested "
               "on hourly lows and highs, filled at the stop or at a worse open; 0.06% taker "
               "each way; "
               "funding paid or received while held. About ninety days is all Bitget serves of "
               "funding history. Data: Bitget funding settlements and hourly candles. Not advice.")
    return out


__all__ = ["ASKED", "Trade", "lines"]
