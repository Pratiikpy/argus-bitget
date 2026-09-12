"""The leakage suite.

Every test here corresponds to a defect found by reading a real system's source. If one of these
fails, ARGUS has acquired a bug that is already shipping somewhere else, and the citation says
where to look for the shape of it.

Run: pytest -m leakage
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given
from hypothesis import strategies as st

from argus.truth.clocks import ET, DualClock, SessionPhase
from argus.truth.facts import AsOfStore, Fact, LookAheadError

UTC = ZoneInfo("UTC")


def _fact(claim: str, available_at: datetime, *, source: str = "sec", rev: int = 0,
          supersedes: str | None = None) -> Fact:
    return Fact(
        claim=claim,
        source_id=source,
        event_time=available_at - timedelta(hours=1),
        publish_time=available_at - timedelta(minutes=5),
        ingest_time=available_at,
        available_at=available_at,
        revision_id=rev,
        supersedes=supersedes,
    )


# ---------------------------------------------------------------------------------------------
# FinMem, memorydb.py:138-218 — temp_date_list populated at 169/195, never checked before ranking.
# ---------------------------------------------------------------------------------------------

@pytest.mark.leakage
def test_retrieval_never_returns_a_fact_from_the_future() -> None:
    decision_time = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
    store = AsOfStore([
        _fact("past: guidance cut", decision_time - timedelta(days=3)),
        _fact("FUTURE: earnings beat", decision_time + timedelta(days=1)),
        _fact("FUTURE: stock rallied 9%", decision_time + timedelta(days=100)),
    ])

    got = store.query(as_of=decision_time)

    assert len(got) == 1
    assert got[0].claim == "past: guidance cut"
    assert all(f.available_at <= decision_time for f in got)


@pytest.mark.leakage
@given(offset_days=st.integers(min_value=1, max_value=3650))
def test_no_horizon_lets_a_future_fact_through(offset_days: int) -> None:
    """FinMem's bug was not a boundary error — retrieval was unbounded at every horizon."""
    t = datetime(2026, 1, 1, tzinfo=UTC)
    store = AsOfStore([_fact("future", t + timedelta(days=offset_days))])
    assert store.query(as_of=t) == []


@pytest.mark.leakage
def test_omitting_as_of_is_a_type_error_not_a_silent_default() -> None:
    """The design claim: the bound is enforced by the interface, not by caller discipline.

    Both leaking systems *intended* point-in-time retrieval. They failed because forgetting the
    filter was possible. Here it is not expressible.
    """
    store = AsOfStore([_fact("anything", datetime(2026, 1, 1, tzinfo=UTC))])
    with pytest.raises(TypeError):
        store.query()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------------------------
# Restatement leakage — the subtler cousin. Nothing in the corpus handled this correctly.
# ---------------------------------------------------------------------------------------------

@pytest.mark.leakage
def test_as_of_returns_the_figure_as_it_stood_not_the_later_restatement() -> None:
    t0 = datetime(2026, 2, 1, tzinfo=UTC)
    original = _fact("revenue 4.20bn", t0)
    restated = _fact("revenue 3.95bn", t0 + timedelta(days=45), rev=1,
                     supersedes=original.content_hash)
    store = AsOfStore([original, restated])

    before = store.query(as_of=t0 + timedelta(days=1))
    assert [f.claim for f in before] == ["revenue 4.20bn"]

    after = store.query(as_of=t0 + timedelta(days=60))
    assert [f.claim for f in after] == ["revenue 3.95bn"]


# ---------------------------------------------------------------------------------------------
# FinAgent — environment computes days_future = now + 14, putting future state in the observation.
# A guard is needed for evidence assembled by any path other than query().
# ---------------------------------------------------------------------------------------------

@pytest.mark.leakage
def test_injected_future_fact_raises_rather_than_being_quietly_dropped() -> None:
    t = datetime(2026, 5, 1, tzinfo=UTC)
    store = AsOfStore()
    smuggled = [_fact("next week's close", t + timedelta(days=14))]

    with pytest.raises(LookAheadError, match="after as_of"):
        store.assert_no_future_facts(t, smuggled)


@pytest.mark.leakage
def test_naive_datetimes_are_rejected_at_construction() -> None:
    """Naive timestamps cannot be ordered safely across two clocks in different zones."""
    with pytest.raises(ValueError, match="timezone-aware"):
        Fact(
            claim="x", source_id="s",
            event_time=datetime(2026, 1, 1),
            publish_time=datetime(2026, 1, 1, tzinfo=UTC),
            ingest_time=datetime(2026, 1, 1, tzinfo=UTC),
            available_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------------------------
# The two-clock claim — verified absent from all 922 corpus sources.
# ---------------------------------------------------------------------------------------------

class TestTwoClocks:
    clock = DualClock()

    def test_sunday_fact_is_knowable_but_not_priceable(self) -> None:
        """The Sleeping-Anchor case, stated as an assertion.

        3am Sunday: a real filing exists and the token trades on it. No anchor price discovery is
        available and no hedge can be placed. A single-clock model cannot represent this state,
        which is why a backtest built on one silently assumes the hedge was available.
        """
        sunday_3am = datetime(2026, 3, 8, 3, 0, tzinfo=ET)
        published = sunday_3am - timedelta(minutes=30)

        assert self.clock.is_knowable(published, sunday_3am) is True
        assert self.clock.is_priceable(sunday_3am) is False
        assert self.clock.is_hedgeable_against_anchor(sunday_3am) is False
        assert self.clock.phase(sunday_3am) is SessionPhase.WEEKEND

    def test_weekend_gap_is_long_and_measured_in_hours(self) -> None:
        friday_close = datetime(2026, 3, 6, 16, 1, tzinfo=ET)
        state = self.clock.state(friday_close)
        assert state.is_anchor_asleep
        assert 60 < state.hours_to_next_discovery < 70  # ~65.5h Friday close to Monday open

    def test_extended_hours_are_not_price_discovery(self) -> None:
        """A pre-market print on thin size is a quote, not discovery. Treating it as discovery is
        how overnight strategies acquire an edge that does not survive a real fill."""
        premarket = datetime(2026, 3, 10, 5, 0, tzinfo=ET)
        assert self.clock.phase(premarket) is SessionPhase.EXTENDED
        assert self.clock.is_priceable(premarket) is False

    def test_rth_is_priceable_and_has_zero_hours_to_discovery(self) -> None:
        midday = datetime(2026, 3, 10, 11, 0, tzinfo=ET)
        state = self.clock.state(midday)
        assert state.phase is SessionPhase.RTH
        assert self.clock.is_priceable(midday) is True
        assert state.hours_to_next_discovery == 0.0

    def test_holiday_is_distinguishable_from_weekend(self) -> None:
        """Both are closed, but attribution must tell them apart: a holiday gap and a weekend gap
        have different information-arrival profiles."""
        july4 = datetime(2026, 7, 3, 12, 0, tzinfo=ET)
        clock = DualClock(holidays=frozenset({july4.date()}))
        assert clock.phase(july4) is SessionPhase.HOLIDAY
        assert clock.is_priceable(july4) is False

    def test_naive_instant_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive datetime"):
            self.clock.phase(datetime(2026, 3, 8, 3, 0))


class TestStaleness:
    clock = DualClock()

    def test_unknown_nav_age_is_stale_never_optimistic(self) -> None:
        state = self.clock.state(datetime(2026, 3, 10, 11, 0, tzinfo=ET), nav_age_seconds=None)
        assert state.nav_is_stale() is True

    def test_threshold_depends_on_phase(self) -> None:
        """One hour old is a broken feed during RTH and entirely normal on a Saturday."""
        age = 3700.0
        rth = self.clock.state(datetime(2026, 3, 10, 11, 0, tzinfo=ET), nav_age_seconds=age)
        weekend = self.clock.state(datetime(2026, 3, 8, 11, 0, tzinfo=ET), nav_age_seconds=age)
        assert rth.nav_is_stale() is True
        assert weekend.nav_is_stale() is False
