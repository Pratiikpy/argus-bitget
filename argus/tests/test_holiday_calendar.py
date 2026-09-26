"""The two-clock model knows US market holidays without being told (`truth/clocks.py`)."""

from __future__ import annotations


def test_the_real_holiday_calendar_is_the_default() -> None:
    """Every clock built without an argument used to call Thanksgiving a session (judge audit,
    2026-09-26); the live paper cycle builds one that way."""
    from datetime import UTC, datetime

    from argus.truth.clocks import DualClock, SessionPhase

    assert DualClock().phase(datetime(2026, 11, 26, 20, 0, tzinfo=UTC)) is SessionPhase.HOLIDAY
    assert DualClock().phase(datetime(2026, 11, 25, 20, 0, tzinfo=UTC)) is SessionPhase.RTH
    assert DualClock(holidays=frozenset()).phase(
        datetime(2026, 11, 26, 20, 0, tzinfo=UTC)) is SessionPhase.RTH
