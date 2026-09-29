"""Sector rotation against SPY (`lui/research/sector_rotation.py`)."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from argus.lui.research import sector_rotation as sr
from argus.lui.research.sector import SECTOR_FUND


def _daily(month: dict[str, float], quarter: dict[str, float]) -> Any:
    """Closes that give each fund the stated 21- and 63-day returns."""
    def daily(ticker: str) -> list[Any]:
        m, q = month.get(ticker, 0.0), quarter.get(ticker, 0.0)
        start = date(2026, 6, 1)
        closes = [100.0] * 100
        closes[-1 - sr.QUARTER] = 100.0 / (1 + q)
        closes[-1 - sr.MONTH] = 100.0 / (1 + m)
        return [SimpleNamespace(day=start + timedelta(days=i), open=c, close=c)
                for i, c in enumerate(closes)]
    return daily


def test_leaders_are_only_the_sectors_ahead_of_spy() -> None:
    month = {"SPY": 0.01, "XLK": 0.05, "XLV": 0.02, "XLE": 0.0, "XLB": -0.06}
    quarter = {"SPY": 0.02, "XLK": 0.01, "XLV": 0.06, "XLE": 0.1}
    lines, _, data = sr.answer("which sectors lead", daily=_daily(month, quarter))
    lead = lines[0]
    assert lead.startswith("Bottom line: over the last month relative strength has favoured "
                           "Technology (+5.0%, +4.0 points against SPY), Healthcare")
    assert "Energy" not in lead.split("; ")[0].split("and left")[0]
    assert "Healthcare also leads over three months" in lead
    assert len(data["sectors"]) == len(SECTOR_FUND)
    assert any("not fund flows" in line for line in lines)


@pytest.mark.parametrize(("text", "asked"), [
    ("which sectors are money rotating into this month", True),
    ("sector rotation right now", True),
    ("which sectors are leading", True),
    ("what sector is NVDA in", False),
    ("is NVDA a buy", False),
])
def test_only_a_question_about_sectors_leading_reaches_it(text: str, asked: bool) -> None:
    assert sr.asks_for_sector_rotation(text) is asked
