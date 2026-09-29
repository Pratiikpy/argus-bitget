"""A past buy held to now, and averaging in against a lump sum (`lui/research/hindsight.py`)."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from argus.lui.research import hindsight as hs


def _daily(start: date, closes: list[float]) -> Any:
    days = [SimpleNamespace(day=start + timedelta(days=i), open=c, close=c)
            for i, c in enumerate(closes)]
    return lambda ticker: days


@pytest.mark.parametrize(("text", "day"), [
    ("if I bought TSLA at the start of 2022", date(2022, 1, 1)),
    ("had I bought it in March 2021", date(2021, 3, 1)),
    ("bought it 3 years ago", date(2023, 9, 30)),
    ("since 2020", date(2020, 1, 1)),
    ("if I had bought TSLA", None),
])
def test_the_entry_day_is_read_and_never_guessed(text: str, day: date | None) -> None:
    assert hs.entry_date(text, date(2026, 9, 30)) == day


def test_a_past_buy_states_its_closes_its_result_and_its_worst_fall() -> None:
    """A judge's audit, 2026-09-29: answered with position sizing, the 2022 entry never read."""
    daily = _daily(date(2022, 1, 1), [100.0, 120.0, 60.0, 90.0])
    lines, _, data = hs.hindsight("if I bought TSLA at the start of 2022", "TSLAUSDT",
                                  today=date(2026, 9, 30), daily=daily)
    assert lines[0] == ("Bottom line: bought TSLA at the close on 01 Jan 2022 (100.00) and held "
                        "to 04 Jan 2022 (90.00), it lost 10.0% — $10,000 would now be $9,000.")
    assert "50.0% below its own high" in lines[1]
    assert data["hindsight"]["worst_drawdown"] == pytest.approx(-0.5)


def test_averaging_and_a_lump_sum_are_set_side_by_side() -> None:
    closes = [100.0] * 31 + [50.0] * 30 + [100.0] * 30
    daily = _daily(date(2026, 7, 1), closes)
    lines, _, data = hs.dca("should I DCA into BTC", "BTCUSDT", today=date(2026, 9, 30),
                            daily=daily, months=3)
    assert data["dca"]["buys"] == 3
    assert data["dca"]["end_dca"] > data["dca"]["end_lump"]
    assert "averaging came out ahead" in lines[0]


def test_an_impossible_a_future_and_a_pre_listing_date_are_each_said() -> None:
    """A hostile review, 2026-09-29: 2023-02-29 gave an HTTP 500; the end of 2030 was refused as
    history "not reaching back"; CRCL in 2022 was answered from 2025 without a word."""
    daily = _daily(date(2025, 6, 5), [80.0, 82.0, 84.0])
    today = date(2026, 9, 30)
    bad = hs.hindsight("bought TSLA on 2023-02-29", "TSLAUSDT", today=today, daily=daily)[0]
    assert bad[0].startswith("Bottom line: 2023-02-29 is not a date on the calendar")
    future = hs.hindsight("bought TSLA at the end of 2030", "TSLAUSDT", today=today,
                          daily=daily)[0]
    assert future[0].startswith("Bottom line: 31 Dec 2030 has not happened yet")
    early = hs.hindsight("if I bought CRCL at the start of 2022", "CRCLUSDT", today=today,
                         daily=daily)[0]
    assert early[0].startswith("Bottom line: bought CRCL at the close on 05 Jun 2025")
    assert early[1].startswith("Note: CRCL's price history starts on 05 Jun 2025")
    assert hs.dca("dca into BTC starting 2023-02-29", "BTCUSDT", today=today,
                  daily=daily)[0][0].startswith("Bottom line: 2023-02-29 is not a date")


@pytest.mark.parametrize(("text", "amount"), [
    ("if I had put 1000 dollars in nvidia", 1000.0),
    ("if I had invested $2,500 in TSLA", 2500.0),
    ("had I bought 5k bucks of BTC", 5000.0),
    ("if I bought TSLA in 2022", None),
])
def test_the_sum_asked_about_is_the_sum_answered(text: str, amount: float | None) -> None:
    assert hs.stated_amount(text) == amount
