"""Session-aware scheduling tests.

The solver is checked against things it did not produce: Almgren-Chriss's own closed form (the
constant-parameter limit), an independent quadratic-programming solver on random instances with the
no-round-trip constraint active, and QuantConnect LEAN's market-hours database for the calendar.
A suite that only checked the schedule against itself would pass on a wrong sign.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.desk.session_schedule import (
    MIN_OBSERVATIONS,
    BookObservation,
    Segment,
    SegmentParams,
    SessionModel,
    SessionRefusal,
    boundaries,
    calibrate,
    decay_to_risk_aversion,
    is_closed_all_day,
    load_model,
    optimal_holdings,
    segment_at,
    session_bounds,
    solve,
    walk_cost_bps,
)
from argus.execution.schedule import (
    ImpactParameters,
    ScheduleError,
    quantise_trajectory,
    trajectory,
)

LEAN_DB = (Path(__file__).resolve().parents[2] / "research" / "repos" / "Lean-upstream" / "Data"
           / "market-hours" / "market-hours-database.json")


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def lean_sessions(db: dict[str, Any], years: tuple[int, ...]) -> list[tuple[datetime, datetime]]:
    """Every regular session in ``years`` per LEAN's market-hours database, as UTC intervals: the
    weekday's ``market`` window in America/New_York, minus ``holidays``, cut at ``earlyCloses``.
    LEAN's date strings are unpadded (``1/9/2025``). Written here, independently of the module under
    test, so the reference is not derived from the calendar it checks."""
    from zoneinfo import ZoneInfo

    entry = db["entries"]["Equity-usa-[*]"]
    tz = ZoneInfo(entry["exchangeTimeZone"])
    closed = set(entry["holidays"])
    early_closes = entry.get("earlyCloses", {})
    names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    out = []
    day = date(min(years), 1, 1)
    while day <= date(max(years), 12, 31):
        stamp = f"{day.month}/{day.day}/{day.year}"
        if stamp not in closed:
            for seg in entry[names[day.weekday()]]:
                if seg["state"] != "market":
                    continue
                start = datetime.strptime(seg["start"], "%H:%M:%S").time()
                end = datetime.strptime(seg["end"], "%H:%M:%S").time()
                if stamp in early_closes:
                    end = min(end, datetime.strptime(early_closes[stamp], "%H:%M:%S").time())
                out.append((datetime.combine(day, start, tzinfo=tz).astimezone(UTC),
                            datetime.combine(day, end, tzinfo=tz).astimezone(UTC)))
        day += timedelta(days=1)
    return out


def params(half: float = 0.5, impact: float = 1e-5, sigma: float = 3.0,
           n: int = 100) -> SegmentParams:
    return SegmentParams(half_spread_bps=half, impact_bps_per_usd=impact, sigma_bps=sigma,
                         observations=n)


def model(**overrides: SegmentParams) -> SessionModel:
    base = {seg: params() for seg in Segment}
    base.update({Segment(k): v for k, v in overrides.items()})
    return SessionModel("NVDAUSDT", "bitget-futures", 6.0, base, ("test",))


class TestCalendar:
    def test_daylight_time_boundaries(self) -> None:
        assert segment_at(utc(2026, 9, 1, 13, 29)) is Segment.PRE
        assert segment_at(utc(2026, 9, 1, 13, 30)) is Segment.OPEN
        assert segment_at(utc(2026, 9, 1, 14, 0)) is Segment.RTH
        assert segment_at(utc(2026, 9, 1, 19, 30)) is Segment.CLOSE
        assert segment_at(utc(2026, 9, 1, 20, 0)) is Segment.POST
        assert segment_at(utc(2026, 9, 2, 0, 0)) is Segment.OVERNIGHT
        assert segment_at(utc(2026, 9, 1, 8, 0)) is Segment.PRE

    def test_standard_time_moves_the_open_an_hour(self) -> None:
        # 2025-12-01 is in US standard time: 09:30 ET is 14:30 UTC.
        assert segment_at(utc(2025, 12, 1, 13, 30)) is Segment.PRE
        assert segment_at(utc(2025, 12, 1, 14, 30)) is Segment.OPEN
        assert segment_at(utc(2025, 12, 1, 20, 59)) is Segment.CLOSE
        assert segment_at(utc(2025, 12, 1, 21, 0)) is Segment.POST

    def test_early_close_moves_the_close(self) -> None:
        # 2026-11-27, the day after Thanksgiving, closes 13:00 ET = 18:00 UTC (standard time).
        assert segment_at(utc(2026, 11, 27, 17, 30)) is Segment.CLOSE
        assert segment_at(utc(2026, 11, 27, 18, 0)) is Segment.POST
        opens, closes = session_bounds(date(2026, 11, 27)) or (None, None)
        assert closes == utc(2026, 11, 27, 18, 0)
        assert opens == utc(2026, 11, 27, 14, 30)

    def test_holidays_including_an_unscheduled_closure(self) -> None:
        assert segment_at(utc(2026, 11, 26, 16, 0)) is Segment.HOLIDAY
        assert segment_at(utc(2026, 1, 1, 15, 0)) is Segment.HOLIDAY
        # The national day of mourning for President Carter: no rule produces it, LEAN lists it.
        assert is_closed_all_day(date(2025, 1, 9))
        assert segment_at(utc(2025, 1, 9, 16, 0)) is Segment.HOLIDAY
        assert session_bounds(date(2025, 1, 9)) is None

    def test_weekend(self) -> None:
        assert segment_at(utc(2026, 8, 1, 16, 0)) is Segment.WEEKEND

    def test_naive_datetimes_are_refused(self) -> None:
        with pytest.raises(ValueError):
            segment_at(datetime(2026, 9, 1, 14, 0))  # a naive datetime is the point of the test

    def test_boundaries_inside_a_horizon(self) -> None:
        found = boundaries(utc(2026, 9, 1, 13, 0), 60)
        assert [(b.at, b.before, b.after) for b in found] == [
            (utc(2026, 9, 1, 13, 30), Segment.PRE, Segment.OPEN)]
        assert found[0].is_anchor_open_or_close
        close = boundaries(utc(2026, 9, 1, 19, 0), 120)
        assert [b.after for b in close] == [Segment.CLOSE, Segment.POST]
        assert [b.is_anchor_open_or_close for b in close] == [False, True]

    @pytest.mark.skipif(not LEAN_DB.exists(), reason="LEAN market-hours database not cloned")
    def test_regular_session_matches_lean_every_minute_of_2025_to_2027(self) -> None:
        db = json.loads(LEAN_DB.read_text(encoding="utf-8"))
        sessions = lean_sessions(db, (2025, 2026, 2027))
        cursor, wrong = 0, []
        t = utc(2025, 1, 1)
        end = utc(2028, 1, 1)
        while t < end:
            while cursor < len(sessions) and sessions[cursor][1] <= t:
                cursor += 1
            truth = cursor < len(sessions) and sessions[cursor][0] <= t < sessions[cursor][1]
            if segment_at(t).anchor_open != truth:
                wrong.append(t)
            t += timedelta(minutes=5)
        assert wrong == []


class TestSolverAgainstTheClosedForm:
    """With every parameter constant the time-varying system is Almgren-Chriss's own recurrence."""

    @pytest.mark.parametrize("lam", ["0", "0.01", "1", "50"])
    def test_constant_parameters_reproduce_execution_schedule(self, lam: str) -> None:
        eta, sigma, n = 0.001, 0.01, 12
        impact = ImpactParameters(sigma=Decimal(str(sigma)), gamma=Decimal(0),
                                  eta=Decimal(str(eta)), epsilon=Decimal("0.001"))
        ref = trajectory(quantity=Decimal(1000), horizon=Decimal(n), intervals=n, impact=impact,
                         risk_aversion=Decimal(lam))
        expected = [1000.0] + [float(s.remaining) for s in ref.slices]
        ours = optimal_holdings(1000.0, [0.001] * n, [eta] * n,
                                [float(lam) * sigma ** 2] * (n - 1))
        assert max(abs(a - b) for a, b in zip(expected, ours, strict=True)) < 1e-8

    def test_decay_reproduces_the_console_sinh_shape(self) -> None:
        p = params()
        lam = decay_to_risk_aversion(1.0, 240, p)
        sched = solve(100_000.0, utc(2026, 9, 1, 15, 0), 240,
                      model(), risk_aversion=lam, parameters="arrival")
        held = [math.sinh(1.0 * (1 - j / 240)) / math.sinh(1.0) for j in range(241)]
        shape = [held[j] - held[j + 1] for j in range(240)]
        got = [q / 100_000.0 for q in sched.trades]
        assert max(abs(a - b) for a, b in zip(shape, got, strict=True)) < 1e-9


class TestSolverAgainstAnIndependentQP:
    def test_random_instances_with_active_constraints(self) -> None:
        np = pytest.importorskip("numpy")
        optimize = pytest.importorskip("scipy.optimize")
        rng = random.Random(7)
        active = 0
        for _ in range(40):
            n = rng.randint(2, 20)
            x0 = rng.uniform(1e3, 3e5)
            e = np.array([rng.uniform(0, 5) * 1e-4 for _ in range(n)])
            a = np.array([rng.uniform(1e-10, 1e-7) for _ in range(n)])
            r = np.array([rng.uniform(0, 1e-8) * rng.choice([0, 1]) for _ in range(n - 1)])
            ours = optimal_holdings(x0, list(e), list(a), list(r))
            trades = np.array([ours[k] - ours[k + 1] for k in range(n)])
            assert (trades >= -1e-9).all()
            assert abs(trades.sum() - x0) < 1e-6 * x0
            scale = x0 * 1e-4
            tril = np.tril(np.ones((n, n)))

            def cost(u: Any, e: Any = e, a: Any = a, r: Any = r, x0: float = x0,
                     scale: float = scale) -> float:
                q = x0 * u
                x = x0 - np.cumsum(q)
                return float(e @ q + a @ (q * q) + r @ (x[:-1] ** 2)) / scale

            def grad(u: Any, e: Any = e, a: Any = a, r: Any = r, x0: float = x0,
                     scale: float = scale, tril: Any = tril, n: int = n) -> Any:
                q = x0 * u
                x = x0 - np.cumsum(q)
                gx = np.zeros(n)
                gx[:-1] = 2 * r * x[:-1]
                return (e + 2 * a * q - tril.T @ gx) * x0 / scale

            res = optimize.minimize(
                cost, np.full(n, 1 / n), jac=grad, method="SLSQP", bounds=[(0, 1)] * n,
                constraints=[{"type": "eq", "fun": lambda u: u.sum() - 1}],
                options={"ftol": 1e-15, "maxiter": 5000})
            mine = cost(trades / x0)
            assert mine <= res.fun + 1e-9 * abs(res.fun)
            active += bool((trades <= 1e-9).any())
        assert active >= 10, "the constraint must actually bind in a good share of instances"

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ScheduleError):
            optimal_holdings(1.0, [0.0, 0.0], [1e-8, 0.0], [0.0])
        with pytest.raises(ScheduleError):
            optimal_holdings(1.0, [0.0, 0.0], [1e-8, 1e-8], [-1.0])
        with pytest.raises(ScheduleError):
            optimal_holdings(1.0, [0.0], [1e-8, 1e-8], [])


class TestSessionAwareBehaviour:
    def test_riskier_after_the_open_trades_more_before_it(self) -> None:
        m = model(open=params(sigma=12.0), rth=params(sigma=8.0), close=params(sigma=8.0))
        start = utc(2026, 9, 1, 12, 30)
        aware = solve(100_000.0, start, 120, m, decay=1.0)
        blind = solve(100_000.0, start, 120, m, decay=1.0, parameters="arrival")
        assert aware.share_by_segment()["pre"] > blind.share_by_segment()["pre"]
        assert aware.crosses_anchor_boundary
        assert blind.notes and not aware.notes

    def test_dearer_after_the_close_moves_volume_before_it(self) -> None:
        m = model(post=params(half=3.0), overnight=params(half=3.0))
        start = utc(2026, 9, 1, 19, 0)
        aware = solve(25_000.0, start, 120, m, risk_aversion=0.0)
        assert aware.share_by_segment().get("post", 0.0) < 0.2
        blind = solve(25_000.0, start, 120, m, risk_aversion=0.0, parameters="arrival")
        assert blind.share_by_segment()["post"] == pytest.approx(0.5, abs=1e-9)

    def test_refuses_an_unmeasured_segment(self) -> None:
        m = SessionModel("X", "v", 6.0, {Segment.PRE: params(), Segment.OPEN: params()})
        with pytest.raises(SessionRefusal) as err:
            solve(10_000.0, utc(2026, 9, 1, 13, 0), 120, m, decay=1.0)
        assert err.value.segment is Segment.RTH
        assert err.value.at == utc(2026, 9, 1, 14, 0)

    def test_refuses_a_thinly_measured_segment(self) -> None:
        m = model(open=params(n=MIN_OBSERVATIONS - 1))
        with pytest.raises(SessionRefusal, match="fewer than"):
            solve(10_000.0, utc(2026, 9, 1, 13, 0), 60, m, decay=1.0)

    def test_blind_mode_does_not_look_past_arrival(self) -> None:
        m = SessionModel("X", "v", 6.0, {Segment.PRE: params()})
        blind = solve(10_000.0, utc(2026, 9, 1, 13, 0), 120, m, decay=1.0, parameters="arrival")
        assert sum(blind.trades) == pytest.approx(10_000.0)

    def test_children_sum_and_quantise(self) -> None:
        sched = solve(100_000.0, utc(2026, 9, 1, 19, 0), 60, model(), decay=1.0)
        assert sum(q for _, q in sched.children()) == pytest.approx(100_000.0)
        traj = sched.as_trajectory(Decimal("200"))
        lot = Decimal("0.01")
        total = (traj.total_quantity / lot).to_integral_value(rounding="ROUND_FLOOR") * lot
        rounded = type(traj)(total_quantity=total, horizon=traj.horizon, slices=traj.slices,
                             expected_cost=traj.expected_cost, variance=traj.variance,
                             kappa=traj.kappa, risk_aversion=traj.risk_aversion)
        q = quantise_trajectory(rounded, quantity_multiplier=lot, min_order_qty=Decimal("0.1"))
        assert q.total == total
        assert all(s >= Decimal("0.1") for s in q.slices)

    def test_input_validation(self) -> None:
        with pytest.raises(ScheduleError):
            solve(0.0, utc(2026, 9, 1), 60, model(), decay=1.0)
        with pytest.raises(ScheduleError):
            solve(1.0, datetime(2026, 9, 1), 60, model(), decay=1.0)
        with pytest.raises(ScheduleError):
            solve(1.0, utc(2026, 9, 1), 60, model())
        with pytest.raises(ScheduleError):
            solve(1.0, utc(2026, 9, 1), 60, model(), decay=1.0, risk_aversion=1.0)
        with pytest.raises(ScheduleError):
            SegmentParams(0.1, 0.0, 1.0, 50)


def _book(mid: float, half_bps: float, depth_per_level: float) -> tuple[list[tuple[float, float]],
                                                                       list[tuple[float, float]]]:
    tick = mid * half_bps * 1e-4
    bids = [(mid - tick * (1 + 2 * i), depth_per_level) for i in range(50)]
    asks = [(mid + tick * (1 + 2 * i), depth_per_level) for i in range(50)]
    return bids, asks


class TestCalibrate:
    def test_recovers_what_each_segment_was_built_with(self) -> None:
        rng = random.Random(3)
        obs = []
        mid = 200.0
        t = utc(2026, 9, 1, 12, 0)
        for _ in range(360):
            seg = segment_at(t)
            vol = 6.0 if seg.anchor_open else 2.0
            half = 0.5 if seg.anchor_open else 1.5
            mid *= math.exp(rng.gauss(0, vol) * 1e-4)
            bids, asks = _book(mid, half, 20.0)
            obs.append(BookObservation(t, bids, asks))
            t += timedelta(minutes=1)
        m = calibrate(obs, symbol="X", venue="v", fee_bps=6.0)
        assert m.params[Segment.RTH].half_spread_bps == pytest.approx(0.5, rel=1e-6)
        assert m.params[Segment.PRE].half_spread_bps == pytest.approx(1.5, rel=1e-6)
        assert m.params[Segment.RTH].sigma_bps == pytest.approx(6.0, rel=0.2)
        assert m.params[Segment.PRE].sigma_bps == pytest.approx(2.0, rel=0.25)
        assert m.params[Segment.RTH].impact_bps_per_usd > 0
        assert load_model(json.loads(json.dumps(m.as_dict()))).params.keys() == m.params.keys()

    def test_walk_cost(self) -> None:
        bids, asks = _book(100.0, 1.0, 10.0)
        assert walk_cost_bps(asks, 100.0, 100.0) == pytest.approx(1.0, rel=1e-9)
        assert walk_cost_bps(bids, 100.0, 100.0) == pytest.approx(1.0, rel=1e-9)
        # Two whole levels: 10 shares at 100.01 then 10 at 100.03, average 100.02, 2bps.
        assert walk_cost_bps(asks, 10 * 100.01 + 10 * 100.03, 100.0) == pytest.approx(
            2.0, rel=1e-9)
        assert walk_cost_bps(asks, 1e9, 100.0) is None
