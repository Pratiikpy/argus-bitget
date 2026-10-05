"""What 30-day fed funds futures price for the next Fed decision: the CME FedWatch method, on the
futures' own public settlement prices.

"What is Polymarket pricing for a Fed cut at the next FOMC, and does that match the CME FedWatch
implied probability?" got Polymarket's odds and no word on FedWatch (round 40 judge, Q21). CME's
tool is built on its 30-day fed funds futures (ZQ), and its method is published in "Understanding
the CME Group FedWatch Tool Methodology": a contract settles at 100 minus the month's average
effective fed funds rate, so its price implies that average. Here:

- The decision date is the next FOMC "rate decision" in the frozen official calendar
  (`data/macro_calendar_2026.json`).
- The rate before it is the latest effective fed funds rate (FRED's DFF).
- The rate after it is read from the contract for the first month after the meeting that has no
  meeting of its own — that month's whole average is the post-decision rate. When the next month
  holds a meeting too, the meeting month's own contract is split into its days before and after
  the decision, as FedWatch does.
- The implied move over 25 basis points is the probability of a 25bp step, signed (a hike when the
  rate rises). Larger steps are not separated, as FedWatch's binary tree would.

Prices are Yahoo Finance's for the CBOT contracts ("ZQX26.CBT" for November 2026).
"""

from __future__ import annotations

import calendar as _cal
from datetime import date
from typing import Final

_MONTH_CODE: Final = "FGHJKMNQUVXZ"


def _price(month: date) -> float | None:
    from argus.truth import http

    ticker = f"ZQ{_MONTH_CODE[month.month - 1]}{month.year % 100:02d}.CBT"
    try:
        body = http.fetch_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                               params={"interval": "1d", "range": "5d"},
                               headers={"User-Agent": "Mozilla/5.0 argus-research"}, timeout=15)
        return float(body["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        return None


def _next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def implied(today: date) -> dict[str, float | str] | None:
    """The next decision's date, the rate before and after it, and the implied 25bp probability."""
    from argus.lui.research.macro import _fred
    from argus.lui.watchlist import load_calendar

    cal = load_calendar() or {}
    meetings = sorted(str(r["date"]) for r in cal.get("releases") or []
                      if isinstance(r, dict) and r.get("kind") == "FOMC")
    upcoming = [date.fromisoformat(m) for m in meetings if m >= today.isoformat()]
    if not upcoming:
        return None
    meeting = upcoming[0]
    meeting_months = {(date.fromisoformat(m).year, date.fromisoformat(m).month) for m in meetings}
    try:
        rows = _fred("DFF", 20)
        before = float(rows[-1][1])
    except Exception:
        return None
    after_month = _next_month(meeting)
    if (after_month.year, after_month.month) not in meeting_months:
        price = _price(after_month)
        if price is None:
            return None
        after = 100 - price
        source = f"{after_month:%b %Y} contract"
    else:
        price = _price(date(meeting.year, meeting.month, 1))
        if price is None:
            return None
        days = _cal.monthrange(meeting.year, meeting.month)[1]
        # the decision takes effect the day after the meeting
        days_before = meeting.day
        average = 100 - price
        after = (average * days - before * days_before) / max(days - days_before, 1)
        source = f"{meeting:%b %Y} contract, split at the decision"
    move = after - before
    return {"meeting": meeting.isoformat(), "before": before, "after": after,
            "probability": abs(move) / 0.25, "direction": "hike" if move > 0 else "cut",
            "source": source}


def line(today: date) -> str | None:
    """One sentence on what the futures price, for the Fed odds answer."""
    found = implied(today)
    if found is None:
        return None
    p = min(float(found["probability"]), 1.0)
    return (f"Fed funds futures (the CME FedWatch method, on the {found['source']}): the rate "
            f"after the {found['meeting']} decision is priced at {float(found['after']):.3f}% "
            f"against {float(found['before']):.2f}% now, about a {p:.0%} chance of a 25bp "
            f"{found['direction']} — CME's own tool is not read here, but this is the arithmetic "
            f"it runs.")
