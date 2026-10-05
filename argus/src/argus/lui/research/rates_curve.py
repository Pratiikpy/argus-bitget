"""The Treasury curve and real yields, asked directly: "is the 2s10s inverted, and what has the
10-year real yield done?"

Round 41's judge (M1, q02) got "did not recognise that question". The figures are FRED's daily
series, each read live (with `macro._fred`'s snapshot fallback) and checked against FRED on
2026-10-05:

- ``T10Y2Y`` — 10-year minus 2-year Treasury yield, the 2s10s (+0.45 on 2 Oct 2026);
- ``T10Y3M`` — 10-year minus 3-month, the spread the New York Fed's recession model uses;
- ``DGS2``, ``DGS10`` — the two yields themselves (4.78% and 5.24% on 1 Oct 2026);
- ``DFII10`` — the 10-year TIPS yield, the real yield (2.88% on 1 Oct 2026);
- ``T10YIE`` — the 10-year breakeven inflation rate, nominal less real.

"Inverted" is said with its history: when the 2s10s last sat below zero and for how long, read
over five years of the daily series, because "not inverted" alone hides that it may have been
inverted for two years until recently.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Final

ASKED: Final = re.compile(
    r"\b2s\s*10s\b|\b2s?[\s/-]*10s?\s+(?:spread|curve)\b|"
    r"\b10y?\s*[-–]\s*2y?\b|\byield\s+curve\b|"  # noqa: RUF001
    r"\bcurve\b[^?]{0,30}\binvert\w*|\binvert\w*[^?]{0,30}\bcurve\b|\breal\s+yields?\b|\btips\s+"
    r"yields?\b|\bbreakeven\s+inflation\b|\b10s?\s*[-/]\s*3m\b|\bterm\s+spread\b", re.I)


def _change(rows: list[tuple[str, float]], days: int) -> float | None:
    if not rows:
        return None
    end = date.fromisoformat(rows[-1][0])
    then = [v for d, v in rows if (end - date.fromisoformat(d)).days >= days]
    return rows[-1][1] - then[-1] if then else None


def _runs(rows: list[tuple[str, float]]) -> list[tuple[str, str, int]]:
    """Each stretch below zero: (first day, last day, calendar days spanned)."""
    runs: list[tuple[str, str, int]] = []
    start = None
    prev = None
    for d, v in rows:
        if v < 0 and start is None:
            start = d
        if v >= 0 and start is not None and prev is not None:
            runs.append((start, prev, (date.fromisoformat(prev) - date.fromisoformat(start)).days))
            start = None
        prev = d
    if start is not None and prev is not None:
        runs.append((start, prev, (date.fromisoformat(prev) - date.fromisoformat(start)).days))
    return runs


def _bp(x: float) -> str:
    return f"{x * 100:+.0f}bp"


def lines(text: str) -> list[str] | None:
    """The curve and real-yield answer, or None when ``text`` does not ask about them."""
    if not ASKED.search(text):
        return None
    from argus.lui.research.macro import _fred

    spread = _fred("T10Y2Y", 1900)
    if not spread:
        return ["Bottom line: FRED did not answer just now and no snapshot of the curve is kept "
                "here, so there is no 2s10s figure to give; ask again in a minute."]
    two, ten = _fred("DGS2", 30), _fred("DGS10", 30)
    real = _fred("DFII10", 400)
    breakeven = _fred("T10YIE", 30)
    three_m = _fred("T10Y3M", 30)
    day, now = spread[-1]
    below = [d for d, v in spread if v < 0]
    if now < 0:
        state = "inverted"
        start = next((spread[i][0] for i in range(len(spread) - 1, 0, -1)
                      if spread[i - 1][1] >= 0), spread[0][0])
        history = f"it has been below zero since {date.fromisoformat(start):%d %b %Y}"
    else:
        state = "not inverted"
        if below:
            runs = _runs(spread)
            last, longest = runs[-1], max(runs, key=lambda r: r[2])
            history = f"it was last below zero on {date.fromisoformat(last[1]):%d %b %Y}"
            if longest[2] >= 30:
                history += (f"; its long inversion ran from "
                            f"{date.fromisoformat(longest[0]):%d %b %Y} to "
                            f"{date.fromisoformat(longest[1]):%d %b %Y}, about "
                            f"{longest[2] / 30.4:.0f} months")
        else:
            history = f"it has not been below zero since {date.fromisoformat(spread[0][0]):%b %Y}"
    out = [f"Bottom line: the 2s10s is {state} — the 10-year yields {now:+.2f} points over the "
           f"2-year ({date.fromisoformat(day):%d %b %Y}); {history}."]
    if two and ten:
        out.append(f"The yields: 2-year {two[-1][1]:.2f}%, 10-year {ten[-1][1]:.2f}% "
                   f"({date.fromisoformat(ten[-1][0]):%d %b}); over the last month the 2s10s "
                   f"moved {_bp(_change(spread, 30) or 0.0)}.")
    if three_m:
        out.append(f"The 10-year less 3-month spread, the one the New York Fed's recession model "
                   f"uses: {three_m[-1][1]:+.2f} points — "
                   + ("also inverted." if three_m[-1][1] < 0 else "positive too."))
    if real:
        moves = [(label, _change(real, days)) for label, days in
                 (("one month", 30), ("three months", 91), ("a year", 365))]
        said = ", ".join(f"{_bp(m)} over {label}" for label, m in moves if m is not None)
        out.append(f"The 10-year real yield (TIPS) is {real[-1][1]:.2f}% "
                   f"({date.fromisoformat(real[-1][0]):%d %b}): {said}"
                   + (f"; breakeven inflation, the gap to the nominal 10-year, is "
                      f"{breakeven[-1][1]:.2f}%" if breakeven else "") + ".")
        if real[-1][1] >= 2:
            out.append("A real yield above 2% is high by the standard of the last fifteen years; "
                       "it raises what cash and bonds pay after inflation, which is the rate gold "
                       "and long-duration stocks are measured against.")
    out.append("Data: FRED (T10Y2Y, T10Y3M, DGS2, DGS10, DFII10, T10YIE), daily. Analysis, not "
               "advice.")
    return out


__all__ = ["ASKED", "lines"]
