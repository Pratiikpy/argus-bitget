"""The console's own calls, recorded and graded (`eval/call_record.py`), offline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from argus.eval import call_record as cr
from argus.lui.task import IMPACT_TITLE

T0 = datetime(2026, 9, 27, 12, tzinfo=UTC)


@dataclass
class _Step:
    title: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Verdict:
    call: str


@dataclass
class _Task:
    name: str
    steps: list[_Step]
    verdict: _Verdict | None


def _task(question: str) -> _Task:
    impact = _Step(IMPACT_TITLE, {"report": {
        "symbol": "TSLAUSDT", "benchmark": "QQQUSDT",
        "weights_before": {"NVDAUSDT": 1.0},
        "weights_after": {"NVDAUSDT": 0.85, "TSLAUSDT": 0.15},
        "impact": {"beta_after": 1.2, "risk_share_after": 0.2}}})
    analogue = _Step("Has it been here before?", {"long_run": {"horizons": [
        {"days": 1, "up": 0.6, "base_up": 0.5}, {"days": 5, "up": 0.7, "base_up": 0.55}]}})
    return _Task("TSLA", [impact, analogue], _Verdict(f"Add, at 15% ({question[:5]})"))


def test_a_day_is_recorded_once_and_chained(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    first = cr.record(T0, path=path, run=_task)
    assert len(first) == len(cr.QUESTIONS)
    assert cr.record(T0 + timedelta(hours=3), path=path, run=_task) == []
    later = cr.record(T0 + timedelta(days=1), path=path, run=_task)
    assert later[0]["prev"] == first[-1]["hash"]
    rows = cr.read(path)
    cr.verify(rows)
    assert rows[0]["predictions"]["beta_after"] == 1.2
    assert rows[0]["predictions"]["direction"][1] == {"days": 5, "up": 0.7, "base_up": 0.55}


def test_an_edited_or_dropped_row_breaks_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    cr.record(T0, path=path, run=_task)
    rows = cr.read(path)
    edited = [dict(r) for r in rows]
    edited[1]["verdict"] = "Do not add"
    with pytest.raises(cr.CallRecordError, match="changed after it was written"):
        cr.verify(edited)
    with pytest.raises(cr.CallRecordError, match="does not follow"):
        cr.verify([rows[0], rows[2]])


def _returns(start: datetime, hours: int) -> dict[str, dict[datetime, float]]:
    rng = random.Random(3)
    out: dict[str, dict[datetime, float]] = {"QQQUSDT": {}, "NVDAUSDT": {}, "TSLAUSDT": {}}
    for i in range(hours):
        t = start + timedelta(hours=i)
        market = rng.gauss(0, 0.004)
        out["QQQUSDT"][t] = market
        out["NVDAUSDT"][t] = 1.5 * market + rng.gauss(0, 0.003)
        out["TSLAUSDT"][t] = 2.0 * market + rng.gauss(0, 0.006)
    return out


def test_a_call_is_graded_only_after_its_horizon(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    cr.record(T0, path=path, run=_task)
    row = cr.read(path)[0]
    raw = _returns(T0 - timedelta(days=2), 24 * 32)

    def loader(symbols: Any, days: int) -> Any:
        return raw

    assert cr.grade_risk(row, T0 + timedelta(days=10), loader) is None
    graded = cr.grade_risk(row, T0 + timedelta(days=cr.HORIZON_DAYS, hours=1), loader)
    assert graded is not None and "beta_error" in graded
    assert graded["beta_error"] == pytest.approx(abs(graded["beta_after"] - 1.2))


def test_an_unreadable_window_is_ungraded_and_says_why(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    cr.record(T0, path=path, run=_task)

    def failing(symbols: Any, days: int) -> Any:
        raise TimeoutError("no candles")

    graded = cr.grade_risk(cr.read(path)[0], T0 + timedelta(days=40), failing)
    assert graded == {"ungraded": "candles unreadable (TimeoutError)"}


def test_direction_is_scored_against_the_base_rate(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    cr.record(T0, path=path, run=_task)
    closes = {date(2026, 9, 27): 100.0, date(2026, 9, 28): 101.0, date(2026, 9, 29): 99.0}
    scored = cr.grade_direction(cr.read(path)[0], lambda symbol: closes)
    # one close after the call: only the 1-day horizon can be graded, and it went up
    assert scored == [{"days": 1, "up": True, "brier": pytest.approx(0.16),
                       "brier_base": pytest.approx(0.25)}]


def test_the_totals_count_what_was_graded(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    cr.record(T0, path=path, run=_task)
    closes = {date(2026, 9, 27): 100.0, date(2026, 9, 28): 99.0}
    grades = cr.grade(T0 + timedelta(days=2), path=path,
                      loader=lambda s, d: {}, closes=lambda symbol: closes)
    assert grades["calls"] == len(cr.QUESTIONS)
    assert grades["risk_graded"] == 0
    assert grades["direction_graded"] == len(cr.QUESTIONS)
    assert grades["brier"] == pytest.approx(0.36)
    json.dumps(grades)


def test_unreadable_closes_are_named_not_dropped(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    cr.record(T0, path=path, run=_task)

    def failing(symbol: str) -> Any:
        raise TimeoutError("no closes")

    grades = cr.grade(T0 + timedelta(days=2), path=path, loader=lambda s, d: {}, closes=failing)
    assert grades["direction_graded"] == 0
    assert all(g["direction_ungraded"] == "closes unreadable (TimeoutError)"
               for g in grades["calls_graded"])


# --- weekend bands, recorded before the reopen and graded by it ---------------------------------

from argus.market.equity_history import Day  # noqa: E402


def _trading_days(weeks: int, *, reopen: float | None = None) -> list[Day]:
    """Weekdays ending on a Friday; with ``reopen``, a Monday opening at that move."""
    rng = random.Random(5)
    start = date(2015, 1, 5)  # a Monday
    days, price = [], 100.0
    for week in range(weeks):
        for d in range(5):
            opened = price * (1 + (rng.gauss(0, 0.01) if d == 0 and week else 0.0))
            price = opened * (1 + rng.gauss(0, 0.01))
            days.append(Day(day=start + timedelta(days=7 * week + d), open=opened, close=price))
    if reopen is not None:
        monday = days[-1].day + timedelta(days=3)
        days.append(Day(day=monday, open=days[-1].close * (1 + reopen), close=days[-1].close))
    return days


SATURDAY = datetime(2026, 9, 26, 12, tzinfo=UTC)


def test_a_weekend_band_is_recorded_once_and_only_while_the_market_is_shut(
        tmp_path: Path) -> None:
    path = tmp_path / "weekend.jsonl"
    fridays = _trading_days(400)
    assert fridays[-1].day.weekday() == 4
    assert cr.record_weekend(datetime(2026, 9, 23, 12, tzinfo=UTC), path=path,
                             days_of=lambda t: fridays) == []
    added = cr.record_weekend(SATURDAY, path=path, days_of=lambda t: fridays)
    assert [r["ticker"] for r in added] == list(cr.WEEKEND_TICKERS)
    assert added[0]["band"]["p10"] < added[0]["band"]["p90"]
    assert cr.record_weekend(SATURDAY + timedelta(hours=20), path=path,
                             days_of=lambda t: fridays) == []
    cr.verify(cr.read(path))


def test_a_band_is_graded_by_the_open_that_follows_it(tmp_path: Path) -> None:
    path = tmp_path / "weekend.jsonl"
    cr.record_weekend(SATURDAY, path=path, days_of=lambda t: _trading_days(400))
    assert cr.grade_weekend(path, days_of=lambda t: _trading_days(400))["pending"] == 5
    calm = cr.grade_weekend(path, days_of=lambda t: _trading_days(400, reopen=0.0))
    assert calm["graded"] == 5 and calm["covered"] == 1.0
    crash = cr.grade_weekend(path, days_of=lambda t: _trading_days(400, reopen=-0.2))
    assert crash["covered"] == 0.0
    assert crash["mean_pinball_bps"] > calm["mean_pinball_bps"]


class TestThesisVerdicts:
    """Build-list 2.4: the thesis engine's verdicts kept before the week and graded after it."""

    def test_recorded_once_a_day_and_chained(self, tmp_path: Path) -> None:
        path = tmp_path / "theses.jsonl"
        said = {"BTC": "Supported — tape agrees", "ETH": "Contradicted — tape disagrees"}

        def ask(claim: str) -> list[str]:
            return ["Bottom line: your thesis.", said.get(claim.split()[0], "Not measurable — x")]

        now = datetime(2026, 10, 4, 12, tzinfo=UTC)
        first = cr.record_theses(now, path=path, ask=ask)
        assert len(first) == len(cr.THESIS_CLAIMS)
        assert [r["verdict"] for r in first[:2]] == ["Supported", "Contradicted"]
        assert cr.record_theses(now, path=path, ask=ask) == []
        cr.verify(cr.read(path))

    def test_graded_after_the_week(self, tmp_path: Path) -> None:
        path = tmp_path / "theses.jsonl"
        verdicts = iter(["Supported", "Contradicted"] + ["Not measurable"] * 10)
        cr.record_theses(datetime(2026, 10, 1, 12, tzinfo=UTC), path=path,
                         ask=lambda claim: [next(verdicts)])
        days = [date(2026, 10, 1) + timedelta(days=i) for i in range(10)]

        def closes(symbol: str) -> dict[date, float]:
            # BTC rises over the week, ETH falls; the rest rise
            step = -1.0 if symbol == "ETHUSDT" else 1.0
            return {d: 100 + step * i for i, d in enumerate(days)}

        got = cr.grade_theses(path, closes=closes)
        assert got["graded"] == len(cr.THESIS_CLAIMS) and got["pending"] == 0
        assert got["by_verdict"]["Supported"] == {"graded": 1, "came_true": 1}
        assert got["by_verdict"]["Contradicted"] == {"graded": 1, "came_true": 0}
        # a week not yet over is pending, not graded
        short = {d: 100.0 for d in days[:5]}
        assert cr.grade_theses(path, closes=lambda s: short)["pending"] == len(cr.THESIS_CLAIMS)
