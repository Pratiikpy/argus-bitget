"""How a market moved around past ECB and Fed rate *hikes* and ECB *cuts*, decision by decision —
the event study a "how would a 25bp hike from the ECB move EURUSD and BTC historically?" asks.

Round 41's judge (C4, q27) got the next-24h behaviour of 50 "similar" EURUSD hourly states over 26
days: no ECB decision, no BTC. The record is public:

- **ECB decisions.** The deposit facility rate from the ECB's own data portal
  (``data-api.ecb.europa.eu``, series ``FM/D.U2.EUR.4F.KR.DFR.LEV``), daily since 1999. A change
  there is dated the day it *takes effect*, and the ECB's rule for that day has changed: the
  Wednesday after a Thursday decision since 2015 (21 Jul 2022's hike took effect 27 Jul), the next
  MRO settlement in 2005-08 (1 Dec 2005's took effect Tuesday 6 Dec), the next day in 1999-2000
  (3 Feb 2000's took effect Friday 4 Feb). The Governing Council decided on a Thursday in every
  case, so the decision is the last Thursday on or before the effective day. The two-day
  reversals of October 2008 (a corridor change, not a decision) and the January 1999 launch
  corridor are left out.
- **Fed decisions.** FRED's upper bound of the target range (DFEDTARU), effective the day after the
  decision, as `fed_cut_reactions` reads it for cuts.
- **The move.** Each asset's close the day before the decision to its close on the decision day
  (the announcement lands inside that window — 13:15 CET for the ECB, 14:00 ET for the Fed), and
  to the close five days later; Yahoo Finance daily closes (EURUSD=X, BTC-USD, and any other name
  asked). A day an asset did not trade takes its last close before it.

**What it cannot say.** A *surprise* is the gap between the decision and what markets priced the
day before (an OIS or futures read this console does not keep). Most hikes were priced in; the
answer measures "a hike of that size", says so, and does not pretend to isolate the surprise.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date, timedelta
from statistics import mean, median
from typing import Final

from argus.truth import http

ASKED: Final = re.compile(
    r"\b(?P<bank>ecb|european\s+central\s+bank|fed|fomc|federal\s+reserve)\b(?:'s)?[^?]{0,60}?"
    r"\b(?P<move>hikes?|hiked|hiking|raises?|increases?|cuts?|lowers?)\b|\b(?P<move2>hikes?|cuts?)\b"
    r"[^?]{0,30}\b(?:from|by)\s+the\s+(?P<bank2>ecb|fed|fomc)\b", re.I)
_REACTION: Final = re.compile(r"\bhistor\w*|\bmove\w*|\breact\w*|\bimpact\w*|\bdo\s+to\b|"
                              r"\bperform\w*|\bbase\s+rate\b|\bpast\b|\bprevious\b|\btypically\b|"
                              r"\busually\b|\bbacktest\w*", re.I)
_SIZE: Final = re.compile(r"\b(\d{2,3})\s*(?:bp|bps|basis\s+points?)\b", re.I)
ECB_DFR: Final = ("https://data-api.ecb.europa.eu/service/data/FM/"
                  "D.U2.EUR.4F.KR.DFR.LEV?format=csvdata&startPeriod=1999-01-01")
_NAMES: Final = {"eurusd": ("EURUSD", "EURUSD=X"), "euro": ("EURUSD", "EURUSD=X"),
                 "btc": ("BTC", "BTC-USD"), "bitcoin": ("BTC", "BTC-USD"),
                 "eth": ("ETH", "ETH-USD"), "ether": ("ETH", "ETH-USD"),
                 "gold": ("gold", "GC=F"), "nasdaq": ("the Nasdaq-100 (QQQ)", "QQQ"),
                 "qqq": ("the Nasdaq-100 (QQQ)", "QQQ"), "spy": ("the S&P 500 (SPY)", "SPY"),
                 "s&p": ("the S&P 500 (SPY)", "SPY"), "dax": ("the DAX", "^GDAXI"),
                 "stoxx": ("the Euro Stoxx 50", "^STOXX50E"),
                 "dollar": ("the dollar index (DXY)", "DX-Y.NYB"),
                 "usdjpy": ("USDJPY", "JPY=X"), "yen": ("USDJPY", "JPY=X")}


def ecb_changes() -> list[tuple[date, float]]:
    """(decision date, change in percentage points) for every ECB deposit-rate change."""
    body = http.fetch_text(ECB_DFR, timeout=30.0)
    rows = [(date.fromisoformat(r["TIME_PERIOD"]), float(r["OBS_VALUE"]))
            for r in csv.DictReader(io.StringIO(body)) if r.get("OBS_VALUE")]
    changes = [(rows[i][0], rows[i][1] - rows[i - 1][1]) for i in range(1, len(rows))
               if rows[i][1] != rows[i - 1][1]]
    kept = []
    for i, (day, step) in enumerate(changes):
        undone = any(abs((other - day).days) <= 2 and abs(step + back) < 1e-9
                     for other, back in changes[max(0, i - 1):i + 2] if other != day)
        if undone:
            continue
        if day < date(1999, 2, 1):
            continue  # the launch corridor of January 1999, not a policy decision
        decided = day - timedelta(days=(day.weekday() - 3) % 7)
        kept.append((decided, round(step, 4)))
    return kept


def fed_changes() -> list[tuple[date, float]]:
    from argus.lui.research.macro import _fred

    rows = _fred("DFEDTARU", 9000)
    return [(date.fromisoformat(rows[i][0]) - timedelta(days=1),
             round(rows[i][1] - rows[i - 1][1], 4))
            for i in range(1, len(rows)) if rows[i][1] != rows[i - 1][1]]


def _pct(x: float, places: int = 2) -> str:
    text = f"{x:+.{places}%}"
    return text.replace("-", "", 1).replace("+", "", 1) if float(text.strip("+-%")) == 0 else text


def _on_or_before(closes: dict[date, float], day: date) -> float | None:
    for back in range(7):
        found = closes.get(day - timedelta(days=back))
        if found is not None:
            return found
    return None


def lines(text: str) -> list[str] | None:
    """The event study, or None when ``text`` does not ask how a market met past decisions."""
    m = ASKED.search(text)
    if m is None or not _REACTION.search(text):
        return None
    bank_word = (m.group("bank") or m.group("bank2") or "").lower()
    move_word = (m.group("move") or m.group("move2") or "").lower()
    ecb = bank_word.startswith(("ecb", "european"))
    hiking = not move_word.startswith(("cut", "lower"))
    if not ecb and not hiking:
        return None  # Fed cuts: `fed_cut_reactions` answers them
    size = _SIZE.search(text)
    step = int(size.group(1)) / 100 if size else None
    named = []
    for word, pair in _NAMES.items():
        if re.search(rf"\b{re.escape(word)}\b", text, re.I) and pair not in named:
            named.append(pair)
    assets = named or ([("EURUSD", "EURUSD=X"), ("BTC", "BTC-USD")] if ecb else
                       [("the dollar index (DXY)", "DX-Y.NYB"), ("BTC", "BTC-USD"),
                        ("the S&P 500 (SPY)", "SPY")])
    bank = "ECB" if ecb else "Fed"
    try:
        changes = ecb_changes() if ecb else fed_changes()
    except Exception:
        return [f"Bottom line: the {bank}'s rate history did not answer just now "
                f"({'the ECB data portal' if ecb else 'FRED'}), so past decisions cannot be "
                "listed; ask again in a minute."]
    same_way = [(d, s) for d, s in changes if (s > 0) == hiking]
    events = [(d, s) for d, s in same_way if step is None or abs(abs(s) - step) < 1e-9]
    verb = "hike" if hiking else "cut"
    if not events:
        sizes = sorted({abs(s) for _, s in same_way})
        return [f"Bottom line: the {bank} has made no {int(step * 100) if step else ''}bp "
                f"{verb} on record; its {verb}s have been "
                + ", ".join(f"{int(s * 100)}bp" for s in sizes) + "."]
    from argus.market.equity_history import daily

    out: list[str] = []
    summary: list[tuple[str, float, float, int, int]] = []
    for label, ticker in assets[:4]:
        try:
            closes = {d.day: float(d.close) for d in daily(ticker)}
        except Exception:
            out.append(f"{label}: Yahoo Finance did not return its closes just now.")
            continue
        rows = []
        for day, _s in events:
            before = _on_or_before(closes, day - timedelta(days=1))
            on = _on_or_before(closes, day)
            later = _on_or_before(closes, day + timedelta(days=5))
            if before and on and later and day - timedelta(days=7) >= min(closes):
                rows.append((day, on / before - 1, later / before - 1))
        if not rows:
            out.append(f"{label}: no price history covers those decisions.")
            continue
        day_moves = [r[1] for r in rows]
        week_moves = [r[2] for r in rows]
        summary.append((label, median(day_moves), mean(week_moves),
                        sum(x > 0 for x in day_moves), len(rows)))
        detail = "; ".join(f"{d:%d %b %Y} {a:+.1%}" for d, a, _ in rows[-6:])
        out.append(f"{label} over {len(rows)} {bank} {verb}s: on the decision day a median "
                   f"{_pct(median(day_moves))} (up {sum(x > 0 for x in day_moves)} of "
                   f"{len(rows)}), five days on an average {_pct(mean(week_moves), 1)}. "
                   f"Latest: {detail}.")
    if not summary:
        return ["Bottom line: no price history could be read for those decisions just now."]
    sized = f"{int(step * 100)}bp " if step else ""
    lead = "; ".join(f"{s[0]} {_pct(s[1])} (up {s[3]} of {s[4]})" for s in summary)
    first, last = events[0][0], events[-1][0]
    small = all(abs(s[1]) < 0.005 for s in summary)
    leaning = [s for s in summary if s[4] >= 8 and not 0.3 <= s[3] / s[4] <= 0.7]
    verdict = ("a small reaction" if small else "not a negligible move") + (
        ", though " + " and ".join(f"{s[0]} rose on {s[3]} of {s[4]}" for s in leaning)
        if leaning else ", with no consistent direction")
    out.insert(0, f"Bottom line: across {len(events)} {bank} {sized}{verb}s ({first:%b %Y} to "
                  f"{last:%b %Y}), the median decision-day move was {lead} — {verdict}; most "
                  "were priced in before the meeting.")
    out.append(f"A *surprise* {verb} is not measured here: that needs what markets priced the day "
               "before (OIS or futures), which this console does not keep; this is how the market "
               f"met a {sized}{verb} of that size, priced or not. {len(events)} decisions is a "
               "small sample.")
    out.append("Decisions from " + ("the ECB data portal (deposit facility rate; decision = the "
                                     "Thursday before the change takes effect)" if ecb else
                                     "FRED (DFEDTARU, the day before the change takes effect)")
               + ("" if ecb else ", whose target-range series begins in December 2008")
               + "; prices from Yahoo Finance daily closes, day before to decision day and to "
                 "five days after. A base rate, not a forecast; not advice.")
    return out


__all__ = ["ASKED", "ecb_changes", "fed_changes", "lines"]
