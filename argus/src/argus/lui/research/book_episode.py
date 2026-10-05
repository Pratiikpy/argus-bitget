"""A book through a named crypto episode, against the trader's own drawdown limit.

"If I do, does my drawdown tolerance still hold under a 2022-style crypto winter?" was refused at
0.45 confidence, and "What would break that conclusion?" after it found no kind of question (round
38 judge, M-1 and M-2). A stress of the beta kind needs a shock size; an episode carries its own:
the book is held from the episode's start to its end at the weights stated (buy and hold, no
rebalancing), on daily closes from `lui/research/rule_test._closes` (Bitget's for crypto, the long
Yahoo series a stock or commodity perpetual tracks), and its worst fall from a peak is read off the
path. Set against the trader's stated drawdown limit, the answer is yes or no, and the line after
it says what would change it: the largest share the episode's worst name could have had with the
book still inside the limit.

The episode dates are the market's, named in the module so a reader can check them: the 2022
crypto bear from bitcoin's all-time close (10 Nov 2021) to its cycle low (21 Nov 2022), the Terra
collapse (5 to 18 May 2022) and the FTX collapse (6 to 21 Nov 2022). Stock-market episodes (March
2020, 2008) are `lui/research/episodes.py`'s, which sizes a stress shock from the Nasdaq-100's own
fall; this module is for a book held through a crypto one, where the backtester's daily closes
reach.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Final

EPISODES: Final[tuple[tuple[re.Pattern[str], str, date, date], ...]] = (
    (re.compile(r"\b(?:ftx)\b", re.I), "the FTX collapse", date(2022, 11, 6), date(2022, 11, 21)),
    (re.compile(r"\b(?:luna|terra|ust)\b", re.I), "the Terra collapse", date(2022, 5, 5),
     date(2022, 5, 18)),
    # crypto named, so a "2022-style selloff" on a stock book stays the stress engine's, whose
    # episode is the Nasdaq-100's (`lui/research/episodes.py`)
    (re.compile(r"\bcrypto\s+(?:winter|bear|crash)\b|\b2022\b[^?.]{0,30}\bcrypto\b|"
                r"\bcrypto\b[^?.]{0,30}\b2022\b", re.I),
     "the 2022 crypto bear", date(2021, 11, 10), date(2022, 11, 21)),
)
BREAK_ASKED: Final = re.compile(
    r"\bwhat\s+would\s+(?:break|change|invalidate|flip|undo)\s+(?:that|this|it|the)\b|"
    r"\bwhat\s+(?:could|would)\s+make\s+(?:that|this|it)\s+(?:wrong|fail|not\s+hold)\b", re.I)


def episode_named(text: str) -> tuple[str, date, date] | None:
    return next(((name, a, b) for pattern, name, a, b in EPISODES if pattern.search(text)), None)


def book_path(weights: Mapping[str, float], series: Mapping[str, Mapping[date, float]],
              start: date, end: date) -> tuple[float, dict[str, float], date, date] | None:
    """(worst fall from a peak, each name's move over the episode, peak day, trough day) for a book
    bought at ``start`` and held to ``end``."""
    days = sorted(set.intersection(*(set(d for d in s if start <= d <= end)
                                     for s in series.values())))
    if len(days) < 5:
        return None
    first = days[0]
    value = [sum(w * series[s][d] / series[s][first] for s, w in weights.items()) for d in days]
    peak, worst, peak_day, low_day, top = value[0], 0.0, days[0], days[0], days[0]
    for d, v in zip(days, value, strict=True):
        if v > peak:
            peak, top = v, d
        fall = v / peak - 1
        if fall < worst:
            worst, peak_day, low_day = fall, top, d
    moves = {s: series[s][days[-1]] / series[s][first] - 1 for s in weights}
    return worst, moves, peak_day, low_day


def lines(text: str, weights: Mapping[str, float], limit: float | None,
          closes: Mapping[str, tuple[Sequence[datetime], Sequence[float]]]) -> list[str] | None:
    """The answer for a book through the episode ``text`` names; ``limit`` the trader's drawdown
    limit as a fraction, when one was said."""
    named = episode_named(text)
    if named is None or len(weights) < 1:
        return None
    label, start, end = named
    series = {s: {t.date(): v for t, v in zip(closes[s][0], closes[s][1], strict=True)}
              for s in weights if s in closes}
    missing = [s for s in weights if s not in series]
    if missing:
        return None
    path = book_path(weights, series, start, end)
    if path is None:
        return None
    worst, moves, top, low = path
    book = ", ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in weights.items())
    lead = (f"Bottom line: through {label} ({start:%d %b %Y} to {end:%d %b %Y}) a book of {book}, "
            f"bought at the start and held, fell {-worst:.0%} at its worst ({top:%d %b %Y} to "
            f"{low:%d %b %Y})")
    if limit is not None:
        holds = -worst <= limit
        lead = (f"Bottom line: {'yes' if holds else 'no'} — " + lead.removeprefix("Bottom line: ")
                + f", {'inside' if holds else 'past'} your {limit:.0%} limit.")
    else:
        lead += "."
    out = [lead, "Each over the whole episode: " + "; ".join(
        f"{s.removesuffix('USDT')} {m:+.0%}" for s, m in moves.items()) + "."]
    if limit is not None:
        # the largest share of the episode's worst name that keeps the book inside the limit,
        # the rest in the best one, found by stepping the weight
        worst_name = min(moves, key=lambda s: moves[s])
        best_name = max(moves, key=lambda s: moves[s])
        fitted = None
        if worst_name != best_name:
            for step in range(100, -1, -1):
                w = step / 100
                trial = {worst_name: w, best_name: 1 - w}
                got = book_path(trial, {s: series[s] for s in trial}, start, end)
                if got is not None and -got[0] <= limit:
                    fitted = w
                    break
        alone = book_path({best_name: 1.0}, {best_name: series[best_name]}, start, end)
        if fitted is not None:
            out.append(f"What would change it: with {best_name.removesuffix('USDT')} beside it, "
                       f"{worst_name.removesuffix('USDT')} could have been at most {fitted:.0%} of "
                       f"the book for the fall to stay inside {limit:.0%} in this episode; a "
                       f"deeper or longer episode than this one would break that too.")
        elif alone is not None:
            out.append(f"What would change it: nothing in this book would have — even "
                       f"{best_name.removesuffix('USDT')} alone fell {-alone[0]:.0%} at its worst "
                       f"in this episode, past {limit:.0%}; only cash or a hedge sized to the "
                       f"book would have kept the fall inside it.")
    out.append("Daily closes, buy and hold from the episode's first day, no rebalancing and no "
               "fees; a past episode, not a forecast of the next.")
    return out
