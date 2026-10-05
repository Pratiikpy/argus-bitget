"""Macro indicator answers (`lui/research/macro_indicators.py`): the five round-42 questions that
were refused or answered with the wrong series, read from fake FRED series, offline."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import pytest

from argus.lui.research import macro, macro_indicators


def _daily(end: date, days: int, fn: Callable[[int], float]) -> list[tuple[str, float]]:
    rows = []
    for back in range(days, -1, -1):
        d = end - timedelta(days=back)
        if d.weekday() < 5:
            rows.append((d.isoformat(), float(fn(back))))
    return rows


def _weekly(end: date, weeks: int, fn: Callable[[int], float]) -> list[tuple[str, float]]:
    return [((end - timedelta(weeks=w)).isoformat(), float(fn(w))) for w in range(weeks, -1, -1)]


def _monthly(end: date, months: int, fn: Callable[[int], float]) -> list[tuple[str, float]]:
    rows = []
    for m in range(months, -1, -1):
        index = end.year * 12 + end.month - 1 - m
        year, month = divmod(index, 12)
        rows.append((date(year, month + 1, 1).isoformat(), float(fn(m))))
    return rows


_DAILY_END = date(2026, 10, 1)
_WEEK_END = date(2026, 9, 26)
_MONTH_END = date(2026, 9, 1)

SERIES: dict[str, list[tuple[str, float]]] = {
    "ICSA": _weekly(_WEEK_END, 120, lambda w: 197_000 + 1_000 * w),
    "IC4WSA": _weekly(_WEEK_END, 120, lambda w: 200_000 + 1_000 * w),
    "DFF": _daily(_DAILY_END, 800, lambda b: 3.88),
    "DTB3": _daily(_DAILY_END, 800, lambda b: 4.00 + 0.001 * b),
    "DGS2": _daily(_DAILY_END, 800, lambda b: 4.78 - 0.002 * b),
    "DGS5": _daily(_DAILY_END, 800, lambda b: 5.01),
    "DGS10": _daily(_DAILY_END, 800, lambda b: 5.24),
    "T10YIE": _daily(_DAILY_END, 800, lambda b: 2.36 - 0.001 * b),
    "PAYEMS": _monthly(_MONTH_END, 30, lambda m: 159_044 - 29 * m),
    "CPIAUCSL": _monthly(_MONTH_END, 30, lambda m: 334.0 / (1.03 ** (m / 12))),
    "DFEDTARU": _daily(_DAILY_END, 800, lambda b: 4.00 if b < 100 else 4.25),
    "DFEDTARL": _daily(_DAILY_END, 800, lambda b: 3.75 if b < 100 else 4.00),
}


@pytest.fixture(autouse=True)
def fake_fred(monkeypatch: pytest.MonkeyPatch) -> None:
    def fred(series: str, days: int = 45) -> list[tuple[str, float]]:
        return list(SERIES.get(series, []))

    monkeypatch.setattr(macro, "_fred", fred)
    macro._FRED_USED_SNAPSHOT.clear()


def _text(out: list[str] | None) -> str:
    assert out is not None
    return "\n".join(out)


def test_jobless_claims_and_four_week_average() -> None:
    out = macro_indicators.lines(
        "What were the latest US initial jobless claims and the 4-week moving average?")
    assert out is not None
    assert out[0].startswith("Bottom line: Initial jobless claims 197,000 for the week ending "
                             "26 Sep 2026")
    body = _text(out)
    assert "4-week moving average of weekly claims 200,000" in body
    assert "a month ago (" in body
    assert out[-1].startswith("Data: FRED series ICSA, IC4WSA")
    assert out[-1].endswith("Not advice.")


def test_fed_funds_and_three_month_bill_lead_with_the_first_asked() -> None:
    out = macro_indicators.lines(
        "What is the effective fed funds rate right now and the 3-month Treasury bill yield?")
    assert out is not None
    assert out[0].startswith("Bottom line: Effective federal funds rate 3.88% on 1 Oct 2026")
    assert out[1].startswith("3-month Treasury bill rate 4.00% on 1 Oct 2026")
    assert "DGS10" not in _text(out)
    assert "DGS3MO is the constant-maturity" in _text(out)
    swapped = macro_indicators.lines(
        "3-month Treasury bill yield and the effective fed funds rate?")
    assert swapped is not None
    assert swapped[0].startswith("Bottom line: 3-month Treasury bill rate")


def test_breakeven_against_a_month_ago_and_nothing_else() -> None:
    out = macro_indicators.lines(
        "What is the 10-year breakeven inflation rate today and how does it compare with a "
        "month ago?")
    assert out is not None
    assert out[0].startswith("Bottom line: 10-year breakeven inflation rate 2.36%")
    assert "a month ago (1 Sep 2026) 2.33%, change +3bp" in out[0]
    assert "year ago" not in out[0]
    assert "DGS10" not in _text(out)


def test_two_year_yield_then_follow_up_year_ago_from_prior() -> None:
    first = macro_indicators.lines("What is the 2-year Treasury yield today?")
    assert first is not None
    assert first[0].startswith("Bottom line: 2-year Treasury yield 4.78%")
    assert "10-year" not in first[0]
    follow = macro_indicators.lines(
        "And how does that compare with the same date a year ago?",
        ["What is the 2-year Treasury yield today?", "\n".join(first)])
    assert follow is not None
    assert follow[0].startswith("Bottom line: 2-year Treasury yield 4.78%")
    assert "a year ago (1 Oct 2025) 4.05%" in follow[0]
    assert "month ago" not in follow[0]


def test_follow_up_naming_a_new_tenor_inherits_the_comparison() -> None:
    out = macro_indicators.lines(
        "And the 5-year?", ["How does the 2-year yield compare with a year ago?"])
    assert out is not None
    assert out[0].startswith("Bottom line: 5-year Treasury yield 5.01%")
    assert "a year ago" in out[0]


def test_not_macro_questions_return_none() -> None:
    assert macro_indicators.lines("what is NVDA's price") is None
    assert macro_indicators.lines("how did BTC do last week") is None
    assert macro_indicators.lines("") is None
    prior = ["What is the 2-year Treasury yield today?"]
    assert macro_indicators.lines("how did BTC do last week", prior) is None
    assert macro_indicators.lines("what about NVDA a year ago", prior) is None
    assert macro_indicators.lines("what is the 5 year return of NVDA") is None


def test_payroll_change_is_a_monthly_difference() -> None:
    out = macro_indicators.lines("What did nonfarm payrolls add last month?")
    assert out is not None
    assert out[0].startswith("Bottom line: Nonfarm payrolls, monthly change +29k for Sep 2026")
    assert "159" not in out[0]


def test_cpi_year_on_year_is_computed() -> None:
    out = macro_indicators.lines("What is CPI inflation right now?")
    assert out is not None
    assert "CPI inflation (year on year) 3.00% for Sep 2026" in out[0]
    assert out[-1].startswith("Data: FRED series CPIAUCSL")


def test_target_range_prints_both_bounds_and_the_old_range() -> None:
    out = macro_indicators.lines("What is the fed funds target range, versus a year ago?")
    assert out is not None
    assert "Fed funds target range 3.75%-4.00% on 1 Oct 2026" in out[0]
    assert "a year ago (1 Oct 2025) 4.00%-4.25%, change -25bp" in out[0]
    assert "DFEDTARU, DFEDTARL" in out[-1]


def test_german_question_about_the_ten_year_over_four_weeks() -> None:
    out = macro_indicators.lines(
        "Wie hat sich die Rendite 10-jähriger US-Staatsanleihen in den letzten vier Wochen "
        "verändert?")
    assert out is not None
    assert out[0].startswith("Bottom line: 10-year Treasury yield 5.24%")
    assert "4 weeks ago (3 Sep 2026)" in out[0]


def test_since_january_and_default_windows() -> None:
    out = macro_indicators.lines("Where is the 10-year yield since January?")
    assert out is not None
    assert "since January (1 Jan 2026)" in out[0]
    default = macro_indicators.lines("What is the 10-year Treasury yield?")
    assert default is not None
    assert "a month ago" in default[0]
    assert "a year ago" in default[0]


def test_no_series_says_so_and_gives_no_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macro, "_fred", lambda series, days=45: [])
    out = macro_indicators.lines("What is the 2-year Treasury yield today?")
    assert out is not None
    assert "FRED did not answer and no copy is kept here" in out[0]
    assert "%" not in out[0]


def test_weekly_comparison_uses_the_nearest_observation_at_or_before() -> None:
    out = macro_indicators.lines("Initial jobless claims versus a month ago?")
    assert out is not None
    # a month before 2026-09-26 is 2026-08-26; the Saturday at or before it is 2026-08-22
    assert "a month ago (22 Aug 2026) 202,000" in out[0]


def test_snapshot_is_declared_for_every_series() -> None:
    assert set(SERIES) <= set(macro_indicators.SNAPSHOT_DAYS)
    assert all(d >= 800 for d in macro_indicators.SNAPSHOT_DAYS.values())
    assert {"ICSA", "IC4WSA", "DTB3", "T10YIE", "DFEDTARL"} <= set(macro_indicators.SNAPSHOT_DAYS)
