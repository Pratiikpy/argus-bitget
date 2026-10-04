"""Tracking error: how closely one Bitget market follows another, hour by hour.

"Compare rSOL and rETH against their perpetuals. What is the tracking error on each over the last
month?" was answered with SOL's monthly return (a judge, round 35). Two things were missed: Bitget's
rTokens are tokenized US stocks — there is no rSOL or rETH — and the measure asked for was never
computed. This reader does both:

- an rToken the text names (``rTSLA``) is read against its stock perpetual (``TSLAUSDT``);
- a coin written as an rToken (``rSOL``) is said not to exist, and the nearest real pair is read
  instead: the coin's spot market against its USDT perpetual, which is what "tracking" can mean for
  it on Bitget.

For each pair, from Bitget's hourly closes over the window (default 30 days): the level gap (the
spot or rToken's close against the perpetual's, in basis points: median, mean, range), and the
tracking error proper — the standard deviation of the difference between the two hourly returns,
given per hour and annualised over 24 x 365 hours. The annualised figure follows the usual
definition (the standard deviation of active return, e.g. Grinold & Kahn, *Active Portfolio
Management*, ch. 5); hourly returns are used because a month of daily ones (30 points) is too few
to read a standard deviation from, which is the same reason `anchor._rtoken_tracking` reads hours.
"""

from __future__ import annotations

import math
import re
import statistics
from itertools import pairwise
from typing import Final

from argus.lui.trace import trace_module

ASKED: Final = re.compile(r"\btracking\s+(?:error|difference|gap)\b|\bhow\s+(?:closely|well|"
                          r"tightly)\s+(?:does|do|did)\s+\S+(?:\s+\S+)?\s+track\b", re.I)
_R_NAME: Final = re.compile(r"\br([A-Z]{2,6})\b")
_DAYS: Final = re.compile(r"\b(?:last|past)\s+(?:(?P<n>\d{1,3})\s+)?(?P<u>days?|weeks?|months?)\b",
                          re.I)


def _window(text: str) -> int:
    m = _DAYS.search(text)
    if m is None:
        return 30
    n = int(m.group("n") or 1)
    unit = m.group("u").lower()
    return max(2, min(90, n * (30 if unit.startswith("month") else 7 if unit.startswith("week")
                               else 1)))


def _pair_lines(label: str, spot: str, perp: str, days: int) -> str | None:
    from argus.market.rtoken_spot import hourly_bars

    try:
        a = hourly_bars(spot, spot=True, days=days)
        b = hourly_bars(perp, spot=False, days=days)
    except Exception:
        return None
    hours = sorted(t for t in a if t in b and b[t][1] > 0 and a[t][1] > 0)
    if len(hours) < 48:
        return None
    gaps = sorted((a[t][1] / b[t][1] - 1) * 10_000 for t in hours)
    diffs = [(a[t2][1] / a[t1][1]) - (b[t2][1] / b[t1][1])
             for t1, t2 in pairwise(hours) if t2 - t1 == 3_600_000]
    if len(diffs) < 24:
        return None
    hourly = statistics.pstdev(diffs)
    yearly = hourly * math.sqrt(24 * 365)
    return (f"{label}: tracking error {hourly * 10_000:.1f}bps an hour, {yearly:.1%} a year "
            f"annualised; the level sat {gaps[len(gaps) // 2]:+.1f}bps from the perpetual at the "
            f"median and {sum(gaps) / len(gaps):+.1f}bps on average, from {gaps[0]:+.1f} to "
            f"{gaps[-1]:+.1f}bps ({len(hours):,} matched hours).")


def lines(text: str) -> list[str] | None:
    if not ASKED.search(text):
        return None
    from argus.lui.research import research_symbols
    from argus.lui.research.parse import rtokens_named

    days = _window(text)
    found: list[str] = []
    missing: list[str] = []
    for spot, perp in rtokens_named(text)[:4]:
        label = f"r{perp.removesuffix('USDT')} against the {perp.removesuffix('USDT')} perpetual"
        said = _pair_lines(label, spot, perp, days)
        (found.append(said) if said else missing.append(label))
    coins: list[str] = []
    for m in _R_NAME.finditer(text):
        base = m.group(1)
        if any(perp == f"{base}USDT" for _, perp in rtokens_named(text)):
            continue
        named = research_symbols(base)[0]
        if named and named[0] == f"{base}USDT":
            coins.append(base)
    if not coins and not found and not missing:
        for symbol in research_symbols(text)[0][:3]:
            # "tracking error of BTC spot vs its perpetual"
            coins.append(symbol.removesuffix("USDT"))
    for base in dict.fromkeys(coins):
        said = _pair_lines(f"{base} spot against the {base} perpetual", f"{base}USDT",
                           f"{base}USDT", days)
        (found.append(said) if said else missing.append(f"{base} spot against its perpetual"))
    if not found and not missing:
        return None
    fake = [f"r{b}" for b in dict.fromkeys(coins) if re.search(rf"\br{b}\b", text)]
    lead = ("Bottom line: " + (f"{' and '.join(fake)} {'is' if len(fake) == 1 else 'are'} not on "
                               f"Bitget — its rTokens are tokenized US stocks (rTSLA, rNVDA and "
                               f"the like), not coins — so the nearest real pair is read instead: "
                               f"each coin's spot market against its USDT perpetual. "
                               if fake else "")
            + ((f"Over the last {days} days, " if fake else f"over the last {days} days, ")
               if found else "")
            + ("; ".join(x.split(":")[0] + " —" + x.split(":", 1)[1].split(";")[0]
                         for x in found) + "." if found else
               "no hourly history answered for the pairs asked about."))
    out = [lead, *found]
    if missing:
        out.append("Not read: " + "; ".join(missing) + " — Bitget's hourly candles did not answer "
                   "for the whole window.")
    out.append("Tracking error is the standard deviation of the difference between the two hourly "
               "returns (annualised over 24 x 365 hours); the level gap is the close of one "
               "against the other. Data: Bitget spot and USDT-futures hourly candles, read just "
               "now.")
    return out


trace_module(globals())
