"""A stock's own weekends: split-adjusted gaps, closures only, and the record from a side's view."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from argus.lui.research import ResearchKind, _detect
from argus.market import equity_history as eh


def _payload(rows: list[tuple[date, float, float, float]]) -> dict[str, Any]:
    stamps = [int(datetime(d.year, d.month, d.day, 14, 30, tzinfo=UTC).timestamp())
              for d, *_ in rows]
    return {"chart": {"result": [{
        "timestamp": stamps,
        "indicators": {"quote": [{"open": [r[1] for r in rows], "close": [r[2] for r in rows]}],
                       "adjclose": [{"adjclose": [r[3] for r in rows]}]}}]}}


def test_a_split_weekend_is_not_a_crash() -> None:
    # A 4-for-1 split between Friday and Monday: raw 400 -> 101 is -75%, adjusted it is +1%.
    days = eh.parse(_payload([
        (date(2024, 6, 7), 398.0, 400.0, 100.0),   # Friday, adjclose already split-adjusted
        (date(2024, 6, 10), 101.0, 102.0, 102.0),  # Monday
    ]))
    [gap] = eh.closure_gaps(days)
    assert gap.move == pytest.approx(0.01)


def test_only_closures_that_span_a_saturday_count_as_weekends() -> None:
    days = [eh.Day(date(2026, 9, d), 100, 100) for d in (21, 22, 23, 24)] + [
            eh.Day(date(2026, 9, 25), 99, 100),   # Friday
            eh.Day(date(2026, 9, 28), 95, 96),    # Monday: -5%
            eh.Day(date(2026, 11, 25), 100, 100),  # Wednesday before Thanksgiving
            eh.Day(date(2026, 11, 27), 98, 99)]   # Friday after: a holiday gap, no Saturday
    # (28 Sep to 25 Nov is a hole in this toy series, longer than any real closure: skipped)
    weekends = eh.closure_gaps(days)
    assert [(g.closed, g.reopened) for g in weekends] == [(date(2026, 9, 25), date(2026, 9, 28))]
    assert len(eh.closure_gaps(days, weekend_only=False)) == 2


def test_the_record_is_read_from_the_positions_side() -> None:
    gaps = [eh.Gap(date(2020, 1, 3), date(2020, 1, 6), m)
            for m in (0.02, 0.005, -0.03, -0.12, -0.40)]
    long = eh.record(gaps, side="long", adverse=0.327)
    assert (long.up, long.flat, long.down_1_5, long.down_5) == (0.2, 0.2, 0.2, 0.4)
    assert long.worst.move == -0.40 and long.beyond == 1
    short = eh.record(gaps, side="short", adverse=0.195)
    assert short.worst.move == 0.02 and short.beyond == 0


def test_too_little_history_is_an_error_not_an_answer() -> None:
    with pytest.raises(eh.HistoryError):
        eh.daily("ZZZZ", fetch=lambda ticker: _payload([(date(2026, 9, 25), 1.0, 1.0, 1.0)]))


@pytest.mark.parametrize(("text", "symbol", "leverage", "side", "weekend"), [
    ("long rNVDA over the weekend at 3x with 5,000 USDT", "NVDAUSDT", 3.0, "long", True),
    ("short TSLA at 5x over the weekend", "TSLAUSDT", 5.0, "short", True),
    ("10x long MSTR overnight", "MSTRUSDT", 10.0, "long", False),
])
def test_a_leveraged_hold_across_a_closure_is_a_leverage_question(
        text: str, symbol: str, leverage: float, side: str, weekend: bool) -> None:
    request = _detect(text)
    assert request is not None and request.kind is ResearchKind.LEVERAGE
    assert (request.symbols[0], request.leverage, request.side, request.weekend) == (
        symbol, leverage, side, weekend)


def test_a_hedge_request_on_a_spot_token_is_still_a_hedge() -> None:
    request = _detect("I hold RNVDAUSDT, protect it over the weekend")
    assert request is not None and request.kind is ResearchKind.HEDGE
