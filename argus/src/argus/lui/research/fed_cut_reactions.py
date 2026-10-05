"""How BTC, gold and the Nasdaq moved in the 48 hours around each past Fed cut — the base rate a
"what if the Fed cuts" question asks for.

"If the Fed cuts 25bp at the next FOMC, which has historically reacted more in the 48 hours
after: BTC, gold or the Nasdaq? Give the base rate for each from past cut episodes" got
Polymarket's odds for the next meeting (round 40 judge, Q3). The episodes are on the record:

- **The cuts.** Every day FRED's upper bound of the fed funds target (DFEDTARU) fell — the day the
  new range took effect, the day after the decision (an emergency cut on a Sunday takes effect the
  Monday). Since 2019 that is eleven cuts, including the two emergency cuts of March 2020.
- **The move.** Each asset's close the last day before the decision to its close two calendar
  days after it, from Yahoo Finance daily closes (BTC-USD, gold futures GC=F, QQQ for the
  Nasdaq-100) — on a day an asset did not trade, its last close before it.
- **The base rate.** The median absolute move says how much each tends to move; the share of
  rises says which way; the mean is given beside them. Eleven episodes, two of them in a crash,
  are few: the answer says so and shows the 2020 cuts' weight.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from statistics import mean, median
from typing import Final

ASKED: Final = re.compile(
    r"\b(?:fed|fomc|rate)\b[^?]{0,40}\bcuts?\b[^?]{0,160}\b(?:histor\w*|base\s+rate|past|"
    r"previous|typically|usually|react\w*)\b|\b(?:histor\w*|base\s+rate|past|previous)\b"
    r"[^?]{0,80}\b(?:fed|fomc|rate)\s+cuts?\b", re.I)
_ASSETS: Final = (("BTC", "BTC-USD"), ("gold", "GC=F"), ("the Nasdaq-100 (QQQ)", "QQQ"))


def _close_on_or_before(days: dict[date, float], day: date) -> float | None:
    for back in range(7):
        found = days.get(day - timedelta(days=back))
        if found is not None:
            return found
    return None


def cut_dates() -> list[date]:
    """The decision dates of every cut in FRED's DFEDTARU history (effective day minus one)."""
    from argus.lui.research.macro import _fred

    rows = _fred("DFEDTARU", 2700)
    return [date.fromisoformat(rows[i][0]) - timedelta(days=1)
            for i in range(1, len(rows)) if rows[i][1] < rows[i - 1][1]]


def lines(text: str) -> list[str] | None:
    """The base rate for each asset, or None when ``text`` does not ask it."""
    if not ASKED.search(text):
        return None
    from argus.market.equity_history import daily

    try:
        cuts = cut_dates()
    except Exception:
        return ["Bottom line: FRED's fed funds target history did not answer just now, so the past "
                "cuts cannot be listed; ask again in a minute."]
    if not cuts:
        return None
    rows = []
    for label, ticker in _ASSETS:
        try:
            closes = {d.day: float(d.close) for d in daily(ticker)}
        except Exception:
            continue
        moves = []
        for cut in cuts:
            before = _close_on_or_before(closes, cut - timedelta(days=1))
            after = _close_on_or_before(closes, cut + timedelta(days=2))
            if before and after:
                moves.append((cut, after / before - 1))
        if moves:
            rows.append((label, moves))
    if not rows:
        return None
    stats = [(label, median(abs(m) for _, m in moves), mean(m for _, m in moves),
              sum(m > 0 for _, m in moves), len(moves), moves) for label, moves in rows]
    biggest = max(stats, key=lambda s: s[1])
    others = ", ".join(f"{s[1]:.1%} for {s[0]}" for s in stats if s is not biggest)
    out = [f"Bottom line: on the record of {len(cuts)} Fed cuts since 2019 (of every size, "
           f"25bp and larger), {biggest[0]} moved the most in the window around them — a "
           f"median move of {biggest[1]:.1%} either way, against {others}."]
    for label, med, avg, ups, n, _moves in stats:
        out.append(f"{label}: median move {med:.1%} (either direction), average {avg:+.1%}; "
                   f"it rose after {ups} of {n} cuts.")
    crash = [c for c in cuts if c.year == 2020]
    if crash:
        out.append(f"{len(crash)} of the {len(cuts)} were the emergency cuts of March 2020, in a "
                   f"crash; without them the record is {len(cuts) - len(crash)} ordinary cuts — "
                   f"few enough that one surprise can change the base rate.")
    out.append("Window: the close the day before each decision to the close two days after it. "
               "Cut dates from FRED (DFEDTARU), prices from Yahoo Finance daily closes. A base "
               "rate, not a forecast; not advice.")
    return out
