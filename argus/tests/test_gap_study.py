"""Gap-study tests — this module had no test file at all, and it publishes research numbers.

`research/gap_study.py` produces `data/gap_study.json`, whose medians are quoted in the project's
documents. It carried a hand-rolled median that returned the **upper** of the two middle values on
every even-length input — `[1, 2, 3, 4]` gave 3 rather than 2.5 — so every figure it produced was
biased upward, and nothing in the suite could have noticed because nothing tested this file.

The bug was found by an audit, not by the tests. This file exists so the next one is found here.
"""

from __future__ import annotations

import statistics
from datetime import date, timedelta
from decimal import Decimal

import pytest

from argus.research.gap_study import _median, raw_sessions, study
from argus.truth.clocks import SessionPhase


def _dec(values: list[float]) -> list[Decimal]:
    return [Decimal(str(v)) for v in values]


class TestTheMedian:
    @pytest.mark.parametrize("values", [
        [1, 2, 3, 4],
        [1, 2, 3],
        [1, 2],
        [5, 1, 4, 2],
        [10],
        [3, 1, 4, 1, 5, 9, 2, 6],
        [-4, -2, -1, 0],
    ])
    def test_it_agrees_with_the_standard_library(self, values: list[float]) -> None:
        """The reference every other median in this codebase uses. A hand-rolled one that disagrees
        with it is not a stylistic difference, it is a different statistic."""
        assert float(_median(_dec(values))) == pytest.approx(statistics.median(values))

    def test_an_even_list_averages_the_two_middles(self) -> None:
        """The exact defect: the upper middle was returned instead of the average of both."""
        assert _median(_dec([1, 2, 3, 4])) == Decimal("2.5")

    def test_it_stays_in_decimal(self) -> None:
        """The reason it was hand-rolled at all — `statistics.median` returns a float, and these
        values are Decimal. The fix had to keep the type as well as fix the answer."""
        assert isinstance(_median(_dec([1, 2, 3, 4])), Decimal)

    def test_an_empty_list_is_zero_not_an_error(self) -> None:
        assert _median([]) == Decimal("0")

    def test_the_input_is_not_mutated(self) -> None:
        values = _dec([3, 1, 2])
        before = list(values)
        _median(values)
        assert values == before


def _recent_weekday(days_back: int) -> date:
    """A real date guaranteed to fall inside a short trailing fetch window, walked back off a
    weekend if `days_back` lands on one — so the constructed "holiday" below is unambiguous
    (`DualClock.phase` checks `d in holidays` before the weekday check, so a weekend date would
    already read HOLIDAY without ever exercising that first branch).

    Tuesday to Thursday only (2026-09-26): a Monday or Friday holiday joins the weekend it touches
    and is correctly read as one long WEEKEND closure, so a Monday landing here (as 26 Sep minus 5
    did) failed the test on the calendar, not on the code."""
    d = date.today() - timedelta(days=days_back)
    while d.weekday() not in (1, 2, 3):
        d -= timedelta(days=1)
    return d


class TestRealHolidayCalendarThreading:
    """`study()`/`raw_sessions()` never accepted a `holidays` argument until 2026-09-16 — see
    both functions' own docstrings for the real, previously-dead `SessionPhase.HOLIDAY` branch
    this unlocks. `eval/afterhours_comparison.py` already exercises this live, end to end, but
    through its own parallel `_classify()` helper, not through `research.gap_study`'s actual
    production entry points — this class hits `raw_sessions`/`study` directly, on real live
    Bitget data, so the fix is proven on the function that ships, not only its comparison-side
    mirror. A real (not simulated) date is declared a holiday here — `DualClock.phase()` is a
    pure calendar model that does not consult Bitget's own real feed, so marking any real recent
    weekday a "holiday" deterministically produces a real `SessionPhase.HOLIDAY` session without
    needing an actual market holiday to fall inside the live window this test happens to run in."""

    SYMBOL = ("NVDAUSDT",)

    def test_raw_sessions_never_classifies_a_holiday_when_none_is_given(self) -> None:
        sessions, failures = raw_sessions(self.SYMBOL, days=14)
        assert not failures, failures
        assert not any(s.phase is SessionPhase.HOLIDAY for s in sessions)

    def test_raw_sessions_classifies_a_real_declared_holiday(self) -> None:
        holiday = _recent_weekday(5)
        sessions, failures = raw_sessions(self.SYMBOL, days=14, holidays=frozenset({holiday}))
        assert not failures, failures
        holiday_sessions = [s for s in sessions if s.phase is SessionPhase.HOLIDAY]
        assert holiday_sessions, (
            f"declaring {holiday} a holiday produced no HOLIDAY session in "
            f"{[(s.phase, s.start, s.end) for s in sessions]}"
        )

    def test_study_threads_holidays_through_to_by_phase(self) -> None:
        holiday = _recent_weekday(5)
        result = study(self.SYMBOL, days=14, holidays=frozenset({holiday}))
        assert "holiday" in result["by_phase"]
        assert result["by_phase"]["holiday"]["sessions"] >= 1

    def test_study_with_no_holidays_keeps_its_old_behaviour(self) -> None:
        result = study(self.SYMBOL, days=14)
        assert "holiday" not in result["by_phase"]


class TestThePublishedArtefact:
    def test_the_published_medians_are_well_formed(self) -> None:
        """A deliberately weak assertion: the point is that the artefact parses and every median it
        publishes is a non-negative number, which is all that can be checked without recomputing the
        study. The artefact is keyed by phase and by a headline block, not by symbol — asserting a
        shape it does not have would be a test of my assumption rather than of the file."""
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "data" / "gap_study.json"
        if not path.exists():
            pytest.skip("no gap study on this machine")
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert blob["headline"], "the gap study published no headline"

        found = 0
        def walk(node: object) -> None:
            nonlocal found
            if isinstance(node, dict):
                for key, value in node.items():
                    if "median" in key and value is not None:
                        found += 1
                        assert float(value) >= 0, f"{key} is negative: {value}"
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(blob)
        assert found >= 2, f"expected several medians in the artefact, found {found}"
