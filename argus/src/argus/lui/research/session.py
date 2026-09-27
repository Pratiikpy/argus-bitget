"""Which session the US anchor market is in — the dual clock, holidays, and the session answer."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import trace_module

# --- answering ------------------------------------------------------------------------------


def dual_clock() -> Any:
    """The session clock with the real US equity holiday calendar.

    Until 2026-09-25 the console built ``DualClock()`` with no holidays, so on Thanksgiving it
    would have called the anchor open and every open-session figure would have read a closed day
    as live. The calendar is QuantConnect Lean's (Apache-2.0, vendored with its provenance in
    `eval/baselines/lean_market_holidays_loader.py`, 1998-2028). If it cannot be read the clock
    still runs and says so: `session_status` names the missing calendar."""
    global _CLOCK
    if _CLOCK is None:
        from argus.truth.clocks import DualClock

        try:
            from argus.eval.baselines.lean_market_holidays_loader import (
                load_usa_equity_holidays,
            )

            _CLOCK = (DualClock(holidays=load_usa_equity_holidays()), True)
        except Exception:
            _CLOCK = (DualClock(), False)
    return _CLOCK


_CLOCK: tuple[Any, bool] | None = None


def anchor_is_open() -> Any:
    clock, _ = dual_clock()
    return lambda t: clock.phase(t).has_price_discovery


SESSION_QUESTION = re.compile(
    r"\b(?:is|are)\s+(?:the\s+)?(?:(?:us|u\.s\.|american|stock|equity|nyse|nasdaq|wall\s+street)"
    r"\s+)*(?:stock\s+)?(?:market|exchange|session)s?\s+(?:open|closed|shut|trading)\b"
    r"|\bwhen\s+(?:does|do|will)\s+(?:the\s+)?(?:(?:us|stock|nyse|nasdaq)\s+)*markets?\s+"
    r"(?:open|close|reopen)\b|\bmarket\s+hours\b"
    # "is today a US holiday" was declined (2026-09-25 audit); the session clock carries Lean's
    # holiday calendar
    r"|\bis\s+(?:today|tomorrow|(?:this\s+)?(?:monday|tuesday|wednesday|thursday|friday))\s+a\s+"
    r"(?:us\s+|u\.s\.\s+|stock\s+|market\s+|trading\s+)*(?:holiday|trading\s+day)\b"
    r"|\b(?:us|u\.s\.|stock|nyse|market)\s+(?:market\s+)?holidays?\b"
    # "session state right now" reached the rates dashboard (answer audit, round 3)
    r"|\bsession\s+(?:state|status|right\s+now|now|today)\b|\bwhat\s+session\b"
    r"|\bwhich\s+session\b"
    r"|\u7f8e\u80a1.{0,6}(?:\u5f00\u76d8|\u6536\u76d8|\u4f11\u5e02|\u4ea4\u6613)"
    r"|^\W*(?:the\s+)?(?:us|u\.s\.|american|nyse|nasdaq)\s+(?:stock\s+)?(?:market|session)\s+"
    r"is\s+(?:now\s+|still\s+|currently\s+)?(?:open|closed|shut)\b[^?]*$",
    re.IGNORECASE)
"""A question about whether the US market is trading. It named no instrument, so the kind model
read "is the US market open right now?" as a sentiment question (2026-09-25)."""


def holiday_line(text: str, now: datetime | None = None) -> str | None:
    """A direct answer to "is today (or Friday) a US market holiday", from Lean's calendar. The
    session answer only implied it (2026-09-25 audit)."""
    from datetime import time as clock_time
    from zoneinfo import ZoneInfo

    from argus.truth.clocks import SessionPhase

    if not re.search(r"\bholiday|trading\s+day\b", text, re.I):
        return None
    new_york = ZoneInfo("America/New_York")
    clock, with_holidays = dual_clock()
    day = (now or datetime.now(UTC)).astimezone(new_york).date()
    names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    asked = next((i for i, n in enumerate(names) if re.search(rf"\b{n}\b", text, re.I)), None)
    if re.search(r"\btomorrow\b", text, re.I):
        day += timedelta(days=1)
    elif asked is not None:
        day += timedelta(days=(asked - day.weekday()) % 7)
    phase = clock.phase(datetime.combine(day, clock_time(12), tzinfo=new_york))
    when = f"{day:%a %d %b}"
    if not with_holidays:
        return (f"The US holiday calendar did not load, so whether {when} is a market holiday "
                f"cannot be said here.")
    if phase is SessionPhase.HOLIDAY:
        return f"Bottom line: {when} is a US market holiday — NYSE and Nasdaq are shut all day."
    if phase is SessionPhase.WEEKEND:
        return f"Bottom line: {when} is a weekend day, so the US market is shut — not a holiday."
    return (f"Bottom line: {when} is not a US market holiday — a regular session, 09:30 to 16:00 "
            f"New York time.")


def session_status(now: datetime | None = None) -> tuple[list[str], list[Source]]:
    """Whether the NYSE regular session is open, when it next opens, and what that means here."""
    clock, with_holidays = dual_clock()
    instant = now or datetime.now(UTC)
    phase = clock.phase(instant)
    is_open = phase.has_price_discovery
    lines: list[str] = []
    if is_open:
        lines.append(f"Bottom line: the US stock market (NYSE regular session) is open at "
                     f"{instant:%H:%M} UTC, so a stock perpetual and its stock are both pricing.")
    else:
        reopen = clock.next_discovery(instant).astimezone(UTC)
        wait = reopen - instant
        hours, minutes = divmod(int(wait.total_seconds() // 60), 60)
        why = {"weekend": "it is the weekend", "holiday": "it is a US market holiday"}.get(
            str(getattr(phase, "value", phase)).lower(), "it is outside regular hours")
        lines.append(f"Bottom line: the US stock market (NYSE regular session) is closed at "
                     f"{instant:%H:%M} UTC — {why}. It opens {reopen:%a %d %b %H:%M} UTC, in "
                     f"{hours}h {minutes:02d}m.")
        lines.append("Bitget's stock perpetuals trade around the clock, so while the market is "
                     "shut they are the only price moving; the stock's own last price is its "
                     "last close.")
    if not with_holidays:
        lines.append("The US holiday calendar could not be read just now, so a holiday would be "
                      "shown as a normal day — check the exchange calendar before relying on it.")
    lines.append("Data: NYSE regular hours (09:30 to 16:00 New York time) and the US equity "
                 "holiday calendar from QuantConnect Lean. This is analysis, not advice — you "
                 "make the call.")
    return lines, [Source(kind="computation", ref="argus.truth.clocks.DualClock",
                          detail="regular session, weekends and US equity holidays")]


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
