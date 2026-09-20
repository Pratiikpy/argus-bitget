"""Session-volatility tests — including the one that says the common intuition is backwards.

Every tokenized-equity system in the sweep throttles *because the anchor is asleep*. The measurement
this module is built on says the shut window is the calmest part of the week and the danger is the
single bar where discovery resumes. So the tests here are not only "does the arithmetic work" — they
pin the shape of the answer, because a future refactor that quietly restored the intuitive behaviour
would look correct and be wrong.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.risk.session_risk import (
    MIN_BARS_PER_PHASE,
    MIN_REOPENS,
    STALE_AFTER_HOURS,
    SessionRisk,
    SessionRiskError,
    expected_path_bps,
    load,
    lookup,
    measure,
    throttle,
)
from argus.truth.clocks import DualClock

NOW = datetime(2026, 9, 15, 12, tzinfo=UTC)  # a Tuesday, before the 13:30 UTC open


def _profile(**kwargs: object) -> SessionRisk:
    base: dict[str, object] = {
        "symbol": "NVDAUSDT",
        "phase_bps": {"rth": 31.5, "extended": 14.1, "overnight": 12.8, "weekend": 5.2},
        "phase_counts": {"rth": 251, "extended": 419, "overnight": 335, "weekend": 430},
        "reopen_bps": 52.1,
        "reopens": 64,
        "measured_at": NOW,
        "window_days": 90,
    }
    base.update(kwargs)
    return SessionRisk(**base)  # type: ignore[arg-type]


def _bars(hours: int, *, move: float = 0.001) -> list[tuple[datetime, float]]:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    price = 100.0
    out: list[tuple[datetime, float]] = []
    for i in range(hours):
        price *= 1 + (move if i % 2 else -move)
        out.append((start + timedelta(hours=i), price))
    return out


class TestTheMeasurement:
    def test_it_reports_one_number_per_phase(self) -> None:
        got = measure(_bars(24 * 40), symbol="X", clock=DualClock(), window_days=40)
        assert "rth" in got.phase_bps
        assert set(got.phase_bps) <= {"rth", "extended", "overnight", "weekend", "holiday"}
        assert all(count >= MIN_BARS_PER_PHASE for count in got.phase_counts.values() if count)

    def test_a_phase_with_too_few_bars_is_dropped_not_estimated(self) -> None:
        got = measure(_bars(24 * 40), symbol="X", clock=DualClock())
        for phase, value in got.phase_bps.items():
            assert got.phase_counts[phase] >= MIN_BARS_PER_PHASE
            assert value >= 0

    def test_no_regular_hours_sample_is_refused(self) -> None:
        """Without a baseline there is nothing to throttle against, and a throttle with no baseline
        would be a multiplier chosen rather than derived."""
        # Ten weekends' worth of Saturday/Sunday bars: long enough to clear the length floor,
        # and containing no regular-hours bar at all.
        weekend_only: list[tuple[datetime, float]] = []
        price = 100.0
        for week in range(10):
            base = datetime(2026, 6, 6, tzinfo=UTC) + timedelta(days=7 * week)
            for hour in range(48):
                price *= 1.0005 if hour % 2 else 0.9995
                weekend_only.append((base + timedelta(hours=hour), price))
        with pytest.raises(SessionRiskError, match="regular-hours"):
            measure(weekend_only, symbol="X", clock=DualClock())

    def test_a_short_series_is_refused(self) -> None:
        with pytest.raises(SessionRiskError, match="cannot support"):
            measure(_bars(10), symbol="X")

    def test_a_thin_reopen_sample_is_none_rather_than_an_anecdote(self) -> None:
        """**This test was vacuous.** It guarded the assertion on `reopens < MIN_REOPENS` and then
        handed `measure` forty days of bars, which contain 30 reopens — so the condition was always
        False and the assertion never ran. A guarded assertion that never fires is a test that
        cannot fail.

        Fixed by constructing the case it claims to test: eight days of bars, which contain fewer
        than the ten reopens the estimate requires, asserted unconditionally."""
        thin = measure(_bars(24 * 8), symbol="X", clock=DualClock())
        assert thin.reopens < MIN_REOPENS, (
            f"the fixture was meant to be thin and has {thin.reopens} reopens"
        )
        assert thin.reopen_bps is None
        assert thin.reopen_multiple is None

    def test_a_full_sample_does_produce_a_reopen_estimate(self) -> None:
        """The other side of the same boundary, so the floor is shown to be a floor and not a
        function that always returns None."""
        thick = measure(_bars(24 * 40), symbol="X", clock=DualClock())
        assert thick.reopens >= MIN_REOPENS
        assert thick.reopen_bps is not None


class TestTheShapeOfTheAnswer:
    """These pin the finding, not just the arithmetic."""

    def test_the_reopen_bar_is_the_violent_one(self) -> None:
        profile = _profile()
        assert profile.reopen_bps is not None
        assert profile.reopen_bps > profile.phase_bps["rth"]
        assert profile.reopen_bps > profile.phase_bps["weekend"] * 5

    def test_the_shut_window_is_quieter_than_regular_hours(self) -> None:
        profile = _profile()
        assert profile.phase_bps["weekend"] < profile.phase_bps["rth"]
        assert profile.phase_bps["overnight"] < profile.phase_bps["rth"]

    def test_a_long_horizon_dilutes_the_jump_and_a_short_one_does_not(self) -> None:
        """The measured result, asserted: a two-hour horizon straddling the open is throttled and a
        twenty-four-hour one is not, because the jump is one bar out of twenty-four."""
        profile = _profile()
        short = throttle(profile, start=NOW, horizon_bars=2)
        long = throttle(profile, start=NOW, horizon_bars=24)
        assert short.multiplier < 1
        assert long.multiplier == 1
        assert short.reopens_crossed == long.reopens_crossed == 1

    def test_a_horizon_inside_a_quiet_weekend_is_never_sized_up(self) -> None:
        """The invariant that makes this a risk layer: it may reduce and never add, even when the
        arithmetic says the coming hours are calmer than the baseline."""
        saturday = datetime(2026, 9, 12, 6, tzinfo=UTC)
        got = throttle(_profile(), start=saturday, horizon_bars=12)
        assert got.multiplier == Decimal("1")
        assert "may not size up" in got.reason


class TestThePathArithmetic:
    def test_variances_add_not_volatilities(self) -> None:
        """Summing medians would understate a path that mixes a quiet weekend with a violent
        reopen, which is precisely the path this module exists to price."""
        profile = _profile()
        expected, reopens = expected_path_bps(profile, start=NOW, horizon_bars=2)
        assert reopens == 1
        # One extended bar at 14.1 then the reopen at 52.1: sqrt((14.1^2 + 52.1^2)/2) = 38.2.
        assert expected == pytest.approx(38.2, abs=0.3)

    def test_an_unmeasured_reopen_falls_back_to_the_phase_volatility(self) -> None:
        quiet = _profile(reopen_bps=None, reopens=3)
        expected, reopens = expected_path_bps(quiet, start=NOW, horizon_bars=2)
        assert reopens == 0
        assert expected < 38.0

    def test_a_zero_horizon_is_refused(self) -> None:
        with pytest.raises(SessionRiskError, match="at least one bar"):
            expected_path_bps(_profile(), start=NOW, horizon_bars=0)

    def test_the_multiplier_is_floored(self) -> None:
        """A violent reopen must not shrink a position to dust and call that risk management."""
        violent = _profile(reopen_bps=100_000.0)
        got = throttle(violent, start=NOW, horizon_bars=2, floor=Decimal("0.25"))
        assert got.multiplier == Decimal("0.25")


class TestItRefusesRatherThanGuessing:
    def test_an_unmeasured_symbol_is_not_throttled(self) -> None:
        got = throttle(None, start=NOW)
        assert got.multiplier == Decimal("1")
        assert "not measured" in got.reason

    def test_a_stale_profile_is_absent_not_weaker(self) -> None:
        table = {"NVDAUSDT": _profile()}
        assert lookup("NVDAUSDT", table=table, now=NOW) is not None
        later = NOW + timedelta(hours=STALE_AFTER_HOURS + 1)
        assert lookup("NVDAUSDT", table=table, now=later) is None

    def test_a_missing_file_is_an_empty_table(self, tmp_path: Path) -> None:
        assert load(tmp_path / "absent.json") == {}

    def test_a_malformed_row_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "session.json"
        path.write_text(
            json.dumps({"measurements": [_profile().as_dict(), {"symbol": "BAD"}]}),
            encoding="utf-8",
        )
        assert list(load(path)) == ["NVDAUSDT"]

    def test_it_round_trips(self) -> None:
        again = SessionRisk.from_dict(_profile().as_dict())
        assert again.reopen_bps == _profile().reopen_bps
        assert again.phase_bps == _profile().phase_bps


class TestTheConstitutionGate:
    """The wiring: a measured throttle that narrows a real intent, in the live rulebook."""

    def _ruling(self, horizon: int, quantity: str = "10000") -> object:
        from argus.agents.desk import ConstitutionPolicy
        from argus.decision.verdicts import Intent, Side, Verdict
        from argus.risk.hedgeability import HedgeabilitySurface, open_market_candidate

        policy = ConstitutionPolicy(session_risk=_profile(), session_horizon_bars=horizon)
        intent = Intent(
            symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal(quantity), verdict=Verdict.TRADE,
            stated_confidence=0.8, thesis="t", invalidation=("i",),
        )
        hedges = HedgeabilitySurface(
            (open_market_candidate(
                "NVDA", Decimal("0.9"), execution_probability=Decimal("1"),
            ),),
        )
        return policy.rule(
            intent,
            session=DualClock().state(NOW, nav_age_seconds=60, oracle_age_seconds=60),
            hedges=hedges,
        )

    def test_a_short_horizon_across_the_open_is_resized(self) -> None:
        ruling = self._ruling(2)
        assert ruling.binding_constraint == "session_volatility"  # type: ignore[attr-defined]
        assert ruling.resulting_intent.quantity < Decimal("10000")  # type: ignore[attr-defined]

    def test_a_day_long_horizon_is_allowed(self) -> None:
        ruling = self._ruling(24)
        assert ruling.binding_constraint == "none"  # type: ignore[attr-defined]

    def test_the_gate_only_reduces(self) -> None:
        ruling = self._ruling(2)
        assert ruling.resulting_intent.quantity <= Decimal("10000")  # type: ignore[attr-defined]

    def test_an_unmeasured_policy_leaves_the_gate_inert(self) -> None:
        from argus.agents.desk import ConstitutionPolicy

        assert ConstitutionPolicy().session_risk is None

    def test_the_autopsy_knows_about_the_gate(self) -> None:
        """`eval/autopsy.py` walks the chain in source order and subtracts each gate's hits. A new
        rule the autopsy does not know about would be attributed to whichever gate follows it."""
        from argus.eval.autopsy import CHAIN

        names = [gate.name for gate in CHAIN]
        assert "session_volatility" in names
        assert names.index("session_volatility") < names.index("max_position")
        assert names.index("unhedgeable_gap") < names.index("session_volatility")


class TestTheLiveProfile:
    def test_the_stored_profiles_are_usable(self) -> None:
        from argus.risk.session_risk import REPORT_PATH

        if not REPORT_PATH.exists():
            pytest.skip("no session-risk measurement on this machine")
        table = load()
        assert table
        for symbol, profile in table.items():
            assert symbol.endswith("USDT")
            assert profile.baseline_bps > 0
            if profile.reopen_multiple is not None:
                # The finding, on live data: the reopen bar is the violent one.
                assert profile.reopen_multiple > 1.0
