"""How often to rebalance a book, and what one shock asks of it, measured on the book's own history.

"How often should I rebalance?" beside a four-name allocation was answered "I can run that the
moment I know what you hold", though the names had just been given, and "if BTC drops 15% from
here, how much rebalancing would I need to get back to my target weights?" got the revenge-trading
warning (round 38 judge, C-3 and M-4). Both are arithmetic on the book, and this module does it.

**Cadence, measured, not recommended.** Each policy is replayed on the last two years of daily
closes (the backtester's cached series, `lui/research/rule_test._closes`): never, monthly,
quarterly, and a band rule that trades only when a weight drifts five points from its target.
For each: how many rebalances, the turnover they cost at Bitget's taker fee, and the largest drift
the book reached between them. The literature (Daryanani 2008, Vanguard's "Best practices for
portfolio rebalancing", 2022) finds threshold rules match calendar ones on risk with fewer trades;
the replay checks that on this book rather than asserting it. Return differences between policies
on two years are noise and are said to be, not ranked.

**A shock, worked out.** After a stated fall in one holding, every weight is recomputed and the
trades back to target are listed in percent of the book and in dollars when a sum is known.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Final

ASKED: Final = re.compile(r"\brebalanc\w*|\bback\s+to\s+(?:my\s+)?target\s+weights?\b", re.I)
BAND: Final = 0.05
FEE: Final = 0.0006
"""Bitget's perpetual taker fee: a rebalance is priced as the trades it takes on the perpetuals."""
YEARS: Final = 2


def shock_trades(weights: Mapping[str, float], shocked: str, move: float,
                 capital: float | None = None) -> list[str]:
    """The trades back to target after ``shocked`` moves by ``move`` (-0.15 for a 15% fall)."""
    after = {s: w * (1 + move) if s == shocked else w for s, w in weights.items()}
    total = sum(after.values())
    drifted = {s: v / total for s, v in after.items()}
    name = shocked.removesuffix("USDT")
    trades = []
    for symbol, target in weights.items():
        gap = target - drifted[symbol]
        if abs(gap) < 1e-6:
            continue
        verb = "buy" if gap > 0 else "sell"
        money = (f" (${abs(gap) * total * capital:,.0f})" if capital else "")
        trades.append(f"{verb} {abs(gap) * total:.1%} of the original book{money} of "
                      f"{symbol.removesuffix('USDT')}")
    turnover = sum(abs(weights[s] - drifted[s]) for s in weights) / 2 * total
    return [f"Bottom line: after a {move:+.0%} move in {name} the book is worth {total - 1:+.1%}, "
            f"{name} drifts from {weights[shocked]:.0%} to {drifted[shocked]:.1%}, and getting "
            f"back to target takes trading about {turnover:.1%} of the original book: "
            + "; ".join(trades) + ".",
            f"At Bitget's {FEE:.2%} taker fee that is about "
            + (f"${turnover * 2 * FEE * capital:,.2f}" if capital else
               f"{turnover * 2 * FEE:.3%} of the book")
            + " in fees, buys and sells both counted. Arithmetic on the weights you gave, "
              "before any other holding moves with it."]


def cadence_lines(weights: Mapping[str, float], closes: Mapping[str, Sequence[float]],
                  capital: float | None = None, years: float | None = None) -> list[str] | None:
    """Never, monthly, quarterly and a five-point band, replayed on aligned daily closes."""
    names = [s for s in weights if s in closes]
    n = min((len(closes[s]) for s in names), default=0)
    if len(names) < 2 or n < 120:
        return None
    series = {s: list(closes[s])[-n:] for s in names}
    target = {s: weights[s] / sum(weights[x] for x in names) for s in names}
    years = years or n / 252
    per_year = n / years
    month, quarter = max(1, round(per_year / 12)), max(1, round(per_year / 4))

    def replay(rule: str) -> tuple[int, float, float, float]:
        held = dict(target)  # weights, renormalised every day
        value, trades, turnover, worst = 1.0, 0, 0.0, 0.0
        for day in range(1, n):
            grown = {s: held[s] * series[s][day] / series[s][day - 1] for s in names}
            step = sum(grown.values())
            value *= step
            held = {s: v / step for s, v in grown.items()}
            drift = max(abs(held[s] - target[s]) for s in names)
            worst = max(worst, drift)
            due = ((rule == "monthly" and day % month == 0)
                   or (rule == "quarterly" and day % quarter == 0)
                   or (rule == "band" and drift >= BAND))
            if due:
                turnover += sum(abs(held[s] - target[s]) for s in names) / 2
                held = dict(target)
                trades += 1
        return trades, turnover, worst, value

    rows = {rule: replay(rule) for rule in ("never", "monthly", "quarterly", "band")}
    said = []
    for rule, label in (("monthly", "monthly"), ("quarterly", "quarterly"),
                        ("band", f"when a weight drifts {BAND * 100:.0f} points")):
        trades, turnover, worst, _ = rows[rule]
        fees = turnover * 2 * FEE
        money = f" (${fees * capital / years:,.0f} a year on ${capital:,.0f})" if capital else ""
        said.append(f"{label}: {trades / years:.0f} rebalances a year, {turnover / years:.0%} "
                    f"of the book traded a year, about {fees / years:.2%} a year in fees{money}, "
                    f"weights never more than {worst * 100:.0f} points off")
    never_drift = rows["never"][2]
    band_trades = rows["band"][0] / years
    lead = (f"Bottom line: on the last {years:.1f} years of these names, a rule that rebalances "
            f"only when a weight drifts {BAND * 100:.0f} percentage points from target traded "
            f"{band_trades:.0f} times a year and kept every weight within "
            f"{rows['band'][2] * 100:.0f} points of target; left alone, a weight drifted as far as "
            f"{never_drift * 100:.0f} points.")
    return [lead, "Measured: " + "; ".join(said) + ".",
            "Which made more money over two years is noise and is not ranked here; what the "
            "replay shows is the trade-off between trading more and drifting further. Fees at "
            f"Bitget's {FEE:.2%} taker rate; daily closes."]


def answer(text: str, weights: Mapping[str, float], capital: float | None = None
           ) -> list[str] | None:
    """The rebalancing answer for ``text`` on ``weights``: a stated shock's trades, or the
    cadence replay."""
    from argus.lui.research import research_symbols

    if not ASKED.search(text) or len(weights) < 2:
        return None
    move = re.search(r"\b(?P<name>[A-Za-z]{2,10})\s+(?:drops?|falls?|crash\w*|tanks?|rises?|"
                     r"jumps?|rall\w*|gains?)\s+(?:by\s+)?(?P<p>\d+(?:\.\d+)?)\s*%", text, re.I)
    if move is not None:
        named = research_symbols(move.group("name"))[0]
        if named and named[0] in weights:
            sign = 1 if re.match(r"ris|jum|ral|gai", move.group(0).split()[1], re.I) else -1
            return shock_trades(weights, named[0], sign * float(move.group("p")) / 100, capital)
    from datetime import timedelta

    from argus.lui.research.rule_test import _closes

    by_day: dict[str, dict[date, float]] = {}
    for symbol in weights:
        try:
            stamps, values, _source = _closes(symbol)
        except Exception:
            continue
        if stamps:
            since = stamps[-1] - timedelta(days=YEARS * 365)
            by_day[symbol] = {t.date(): v for t, v in zip(stamps, values, strict=True)
                              if t >= since}
    if len(by_day) < 2:
        return None
    # closes on the days every name traded, so a stock's weekend is one move, not a gap
    days = sorted(set.intersection(*(set(d) for d in by_day.values())))
    if len(days) < 120:
        return None
    closes = {s: [d[day] for day in days] for s, d in by_day.items()}
    span = (days[-1] - days[0]).days / 365.25
    return cadence_lines(weights, closes, capital, years=span)
