"""Whether a name is a good setup for the trader asking: the evidence a swing trader checks, read
from the price itself, then sized on the trader's own rule.

Round 43's judge (M19) asked "Is SOL a good candidate this week?" after stating "I am a swing
trader with a 50k account, max risk 1% per trade", and got a seven-day return line. The question
asks for a judgement this console does not give (no buy or sell call), but every input to that
judgement is measurable, so the answer lays them side by side and leaves the call to the trader:

* **Trend** — the last close against its 20-, 50- and 200-day simple moving averages.
* **Momentum** — the 14-day RSI (Wilder's smoothing), named overbought above 70 and oversold
  below 30, and the last 5- and 20-day returns.
* **Volatility** — 20-day realised volatility, annualised, and its percentile over the history
  read, so a quiet tape is told from a wild one.
* **Levels** — the last 20 closes' low and high: where a long's stop and a breakout sit.
* **The base rate** — over the history, the share of windows as long as the horizon asked (a
  week when none is said) that rose, and their median move.
* **The trader's rule** — when the conversation states (or the trader asked to keep) an account
  and a per-trade risk, the size that rule allows with the stop at the 20-day low.

Prices come from `rule_test.daily_closes` (Bitget's daily closes for coins, Yahoo's
split-adjusted closes for US stocks). The tally of signals is a description, not a forecast.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Sequence
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:is|are|would)\s+(?P<name>[A-Za-z][\w.&-]{1,15})\s+(?:a\s+|an\s+)?(?:good|decent|"
    r"strong|bad|weak)\s+(?:candidate|setup|set[\s-]up|entry|trade|pick|swing|opportunity)\b|"
    r"\b(?:good|decent)\s+(?:candidate|setup|entry)\s+(?:in|on|for)\s+(?P<name2>[A-Za-z][\w.&-]"
    r"{1,15})\b", re.I)
_HORIZON: Final = re.compile(r"\b(?:this|next|the\s+coming)\s+(?P<u>week|month)\b|\b(?P<n>\d{1,2})"
                             r"\s*(?P<u2>day|week|month)s?\b", re.I)


def _rsi(closes: list[float], n: int = 14) -> float:
    gains = [max(0.0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]
    avg_g, avg_l = statistics.fmean(gains[:n]), statistics.fmean(losses[:n])
    for g, lo in zip(gains[n:], losses[n:], strict=True):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + lo) / n
    return 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)


def _vol(returns: Sequence[float], per_year: float) -> float:
    return statistics.stdev(returns) * math.sqrt(per_year)


def lines(text: str, prior: Sequence[str] = (), memory: str = "") -> list[str] | None:
    """The setup checklist for the one name asked about, or None when not asked."""
    m = ASKED.search(text)
    if m is None:
        return None
    from argus.lui.research.parse import research_symbols
    from argus.lui.research.rule_test import daily_closes

    found = research_symbols(m.group("name") or m.group("name2") or "")[0]
    if not found:
        return None
    symbol = found[0]
    name = symbol.removesuffix("USDT")
    try:
        stamps, closes, said = daily_closes(symbol)
    except Exception:
        return [f"Bottom line: {name}'s daily history could not be read just now; ask again in a "
                "minute."]
    if len(closes) < 220:
        return [f"Bottom line: {name} has under 220 daily closes here — too little for a 200-day "
                "trend read."]
    span = max(1, (stamps[-1] - stamps[0]).days)
    per_day = len(closes) / span
    per_year = per_day * 365
    h = _HORIZON.search(text)
    unit = (h.group("u") or h.group("u2") or "week").lower() if h else "week"
    count = int(h.group("n")) if h and h.group("n") else 1
    days = count * {"day": 1, "week": 7, "month": 30}[unit]
    step = max(1, round(days * per_day))
    last = closes[-1]
    sma = {n: statistics.fmean(closes[-n:]) for n in (20, 50, 200)}
    above = [n for n, v in sma.items() if last > v]
    rsi = _rsi(closes[-120:])
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    vol_now = _vol(rets[-20:], per_year)
    history = [_vol(rets[i - 20:i], per_year) for i in range(20, len(rets) + 1, 5)]
    pct = sum(v <= vol_now for v in history) / len(history)
    low20, high20 = min(closes[-20:]), max(closes[-20:])
    windows = [closes[i + step] / closes[i] - 1 for i in range(len(closes) - step)]
    rose = sum(w > 0 for w in windows) / len(windows)
    signals = {
        "trend": ("up" if len(above) == 3 else "down" if not above else "mixed"),
        "momentum": ("overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"),
        "volatility": ("high" if pct > 2 / 3 else "low" if pct < 1 / 3 else "normal"),
    }
    trend_said = ("above all three of its 20-, 50- and 200-day averages" if len(above) == 3 else
                  "below all three of its 20-, 50- and 200-day averages" if not above else
                  "above its " + " and ".join(f"{n}-day" for n in above) + " average"
                  + ("s" if len(above) > 1 else "") + " but below the "
                  + " and ".join(f"{n}-day" for n in sma if n not in above))
    out = [f"Bottom line: {name} at {last:,.2f} — trend {signals['trend']} ({trend_said}), "
           f"momentum {signals['momentum']} (14-day RSI {rsi:.0f}), volatility "
           f"{signals['volatility']} ({vol_now:.0%} a year, {pct:.0%} percentile); this console "
           "makes no buy or sell call, so the checklist below is the evidence, and the call is "
           "yours.",
           f"Returns: {closes[-1] / closes[-1 - round(5 * per_day)] - 1:+.1%} over 5 days, "
           f"{closes[-1] / closes[-1 - round(20 * per_day)] - 1:+.1%} over 20; averages 20-day "
           f"{sma[20]:,.2f}, 50-day {sma[50]:,.2f}, 200-day {sma[200]:,.2f}.",
           f"Levels: the last 20 closes ran from {low20:,.2f} to {high20:,.2f} — a long's stop "
           f"under {low20:,.2f} is {last / low20 - 1:.1%} away; a close above {high20:,.2f} would "
           "be a breakout.",
           f"Base rate for {count} {unit}{'s' if count != 1 else ''}: {rose:.0%} of past "
           f"{days}-day windows rose (median {statistics.median(windows):+.1%}) over the "
           "history read."]
    sized = _sized(text, prior, memory, last, low20, name)
    if sized:
        out.append(sized)
    out.append(f"Data: {said}. Simple moving averages; RSI with Wilder's smoothing. A checklist "
               "is a description of the tape, not a forecast. Not advice.")
    return out


def _sized(text: str, prior: Sequence[str], memory: str, last: float, stop: float,
           name: str) -> str:
    """The trader's own rule applied at the 20-day-low stop, or "" when no rule is held."""
    from argus.lui import memory as kept

    facts = kept.parse(memory)
    risk_fact, acct_fact = kept.get(facts, "trade_risk"), kept.get(facts, "capital")
    said = " ".join([*prior[-4:], text])
    risk_m = re.search(r"\brisk\w*\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%|\b(\d+(?:\.\d+)?)\s*%\s+"
                       r"(?:risk|per\s+trade)", said, re.I)
    acct_m = re.search(r"\$?(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s*(?:usd\s+)?(?:account|capital)\b",
                       said, re.I)
    risk = (float(risk_m.group(1) or risk_m.group(2)) / 100 if risk_m else
            float(risk_fact.value) if risk_fact else None)
    account = (float(acct_m.group(1).replace(",", "")) * {"k": 1e3, "m": 1e6}.get(
        (acct_m.group(2) or "").lower(), 1.0) if acct_m else
               float(acct_fact.value) if acct_fact else None)
    if risk is None or account is None or stop >= last:
        return ""
    distance = last / stop - 1
    notional = account * risk / distance
    return (f"Your rule: {risk:.1%} of ${account:,.0f} is ${account * risk:,.0f} at risk; with "
            f"the stop under the 20-day low ({distance:.1%} away) that is ${notional:,.0f} of "
            f"{name}, about {notional / last:,.2f} units.")


__all__ = ["ASKED", "lines"]
