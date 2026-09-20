"""The attenuation thesis, and the reopening gap it predicts.

This module tests the corrected version of ARGUS's core claim. The original specification said
price discovery *stops* while the anchor market is shut. Measured on Bitget's published index
series, that is false: the index updates in 100% of hours, weekends included.

What is true, and is a sharper claim, is **attenuation**. Median absolute hourly index move on
NVDAUSDT over 90 days:

    RTH        32.17 bps
    extended   15.59 bps
    overnight  11.97 bps
    weekend     4.53 bps    <- roughly one seventh of RTH

The token tracks that damped reference closely (market/index move ratio 0.92-1.16 across every
phase), so the *basis* stays narrow — which is why the naive arbitrage does not pay and why only
7.72% of 28,067 observations were monetizable.

The thesis this leaves is falsifiable and worth testing:

    If information arrives while the anchor is shut, it is priced into a 7x-damped reference.
    The reopening auction then prices it fully. The gap is therefore not visible in the basis —
    it is visible in what happens at the reopen.

:func:`study` measures exactly that: for every closed session in the sample, how far the token
moved *during* the closure versus how far it moved *across the reopen*, and whether the reopening
move is predictable from the closed-session move. A ratio near 1.0 means the closed session priced
the information correctly and there is nothing to trade. A ratio well above 1.0 means the closed
session under-reacted, which is the effect the After-Hours cell is built on.

Costs are applied before any claim of edge. The measured round trip is 12bps and the whole point of
this system is that the fee is larger than most effects in it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from argus.cost.model import CostModel
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import BasisPoint, fetch_basis
from argus.truth.clocks import DualClock, SessionPhase

ROUND_TRIP_BPS = CostModel.bitget_perp().round_trip_bps()


@dataclass(frozen=True, slots=True)
class ClosedSession:
    """One period during which the anchor was shut, and what happened when it reopened."""

    symbol: str
    phase: SessionPhase
    start: datetime
    end: datetime
    hours: float

    price_at_close: Decimal
    price_at_reopen: Decimal
    price_after_reopen: Decimal
    """The token one hour after discovery resumed — the reopening move lands here."""

    @property
    def closed_move_bps(self) -> Decimal:
        """How far the token travelled while the anchor was asleep."""
        if self.price_at_close <= 0:
            return Decimal("0")
        return (self.price_at_reopen - self.price_at_close) / self.price_at_close * Decimal("10000")

    @property
    def reopen_move_bps(self) -> Decimal:
        """How far it travelled in the first hour of restored price discovery."""
        if self.price_at_reopen <= 0:
            return Decimal("0")
        return (
            (self.price_after_reopen - self.price_at_reopen) / self.price_at_reopen
            * Decimal("10000")
        )

    @property
    def continued_in_same_direction(self) -> bool:
        """Did the reopen extend the closed-session move, or reverse it?

        Continuation is the signature of under-reaction — the closed session moved part of the way
        and the reopen completed it. Reversal is the signature of over-reaction, which is the
        opposite trade and equally worth knowing.
        """
        a, b = self.closed_move_bps, self.reopen_move_bps
        return (a > 0 and b > 0) or (a < 0 and b < 0)

    @property
    def net_of_fee_bps(self) -> Decimal:
        """The reopening move less the round trip needed to capture it.

        Almost always negative. That is the finding, not a failure of the measurement.
        """
        return abs(self.reopen_move_bps) - ROUND_TRIP_BPS


def _closed_sessions(
    points: list[BasisPoint], symbol: str, clock: DualClock
) -> list[ClosedSession]:
    """Split the series into runs where the anchor was shut, bounded by real observations.

    A run is only usable if we hold a price before it, at its end, and an hour after — a session
    at the edge of the sample is dropped rather than extrapolated.
    """
    sessions: list[ClosedSession] = []
    i = 0
    n = len(points)

    while i < n:
        phase = clock.phase(points[i].ts)
        if phase.has_price_discovery:
            i += 1
            continue

        start_idx = i
        # Classify by the DEEPEST closure reached, not the first bar. A run beginning Friday
        # 16:00 starts in EXTENDED and becomes a weekend; labelling it "extended" collapsed all
        # 832 sessions into one bucket and hid the weekend/overnight distinction entirely.
        depth = {SessionPhase.EXTENDED: 1, SessionPhase.OVERNIGHT: 2,
                 SessionPhase.WEEKEND: 3, SessionPhase.HOLIDAY: 3}
        deepest = SessionPhase.EXTENDED
        while i < n and not clock.phase(points[i].ts).has_price_discovery:
            ph = clock.phase(points[i].ts)
            if depth.get(ph, 0) > depth.get(deepest, 0):
                deepest = ph
            i += 1

        # Need a price before the closure, at its end, and one bar after discovery resumes.
        if start_idx == 0 or i >= n:
            continue

        before = points[start_idx - 1]
        at_reopen = points[i - 1]
        after = points[i]
        hours = (at_reopen.ts - before.ts).total_seconds() / 3600.0
        if hours <= 0:
            continue

        sessions.append(ClosedSession(
            symbol=symbol,
            phase=deepest,
            start=before.ts,
            end=at_reopen.ts,
            hours=round(hours, 2),
            price_at_close=before.market,
            price_at_reopen=at_reopen.market,
            price_after_reopen=after.market,
        ))

    return sessions


def _median(values: list[Decimal]) -> Decimal:
    """The middle value, averaging the two middles on an even-length list.

    **This returned the upper of the two middles and was wrong on every even-length input** —
    ``[1, 2, 3, 4]`` gave 3 rather than 2.5. Every figure this function produced was biased upward
    by half an inter-quantile step, including the published `median_closed_move_bps` and
    `median_reopen_move_bps` in `data/gap_study.json`. Six other medians in this codebase were
    correct; this one was hand-rolled because the values are `Decimal` and `statistics.median`
    returns a float. Averaging in `Decimal` keeps both the type and the answer.
    """
    if not values:
        return Decimal("0")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def raw_sessions(
    symbols: tuple[str, ...] = RTOKEN_SYMBOLS, *, days: int = 90,
    holidays: frozenset[date] | None = None,
) -> tuple[list[ClosedSession], dict[str, str]]:
    """Every real closed session across `symbols`, unaggregated — the fetch-and-classify step
    `study()` itself uses, exposed separately so a caller can inspect individual real sessions
    (e.g. `eval/afterhours_comparison.py`'s per-session real P&L check) rather than only the
    aggregated `by_phase` statistics `study()` returns."""
    clock = DualClock(holidays)
    all_sessions: list[ClosedSession] = []
    failures: dict[str, str] = {}

    for symbol in symbols:
        try:
            points = fetch_basis(symbol, days=days, interval="1H")
        except Exception as exc:
            failures[symbol] = str(exc)[:120]
            continue
        if len(points) < 24:
            failures[symbol] = "insufficient history"
            continue
        all_sessions.extend(_closed_sessions(points, symbol, clock))

    return all_sessions, failures


def study(
    symbols: tuple[str, ...] = RTOKEN_SYMBOLS, *, days: int = 90,
    holidays: frozenset[date] | None = None,
) -> dict[str, object]:
    """Measure closed-session behaviour and reopening moves across every rToken.

    ``holidays`` did not exist as a parameter here until 2026-09-16: `study()` always
    constructed `DualClock()` bare, so the `SessionPhase.HOLIDAY` branch `_closed_sessions()`
    already classifies for had never once fired in a real run — real market holidays were
    silently absorbed into whichever other phase the day's hours happened to match (see
    `eval/afterhours_comparison.py` for the real, measured consequence and a real holiday
    calendar to pass here). Omitting it keeps this function's exact prior behaviour.
    """
    all_sessions, failures = raw_sessions(symbols, days=days, holidays=holidays)

    if not all_sessions:
        return {"error": "no closed sessions measured", "failures": failures}

    by_phase: dict[str, object] = {}
    for phase in (SessionPhase.WEEKEND, SessionPhase.HOLIDAY, SessionPhase.OVERNIGHT,
                  SessionPhase.EXTENDED):
        rows = [s for s in all_sessions if s.phase is phase]
        if not rows:
            continue
        continued = sum(1 for s in rows if s.continued_in_same_direction)
        reopen_abs = [abs(s.reopen_move_bps) for s in rows]
        by_phase[str(phase)] = {
            "sessions": len(rows),
            "median_hours_closed": round(sorted(s.hours for s in rows)[len(rows) // 2], 2),
            "median_closed_move_bps": str(
                round(_median([abs(s.closed_move_bps) for s in rows]), 2)
            ),
            "median_reopen_move_bps": str(round(_median(reopen_abs), 2)),
            # The headline: does the reopen extend the closed move, or reverse it?
            "continuation_rate_pct": round(100 * continued / len(rows), 1),
            "reopen_clears_fee_pct": round(
                100 * sum(1 for s in rows if s.net_of_fee_bps > 0) / len(rows), 1
            ),
            "median_net_of_fee_bps": str(round(_median([s.net_of_fee_bps for s in rows]), 2)),
        }

    continued_all = sum(1 for s in all_sessions if s.continued_in_same_direction)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "days": days,
        "round_trip_fee_bps": str(ROUND_TRIP_BPS),
        "total_closed_sessions": len(all_sessions),
        "symbols_failed": failures,
        "headline": {
            "continuation_rate_pct": round(100 * continued_all / len(all_sessions), 1),
            "coin_flip_is": 50.0,
            "median_reopen_move_bps": str(
                round(_median([abs(s.reopen_move_bps) for s in all_sessions]), 2)
            ),
            "reopen_clears_fee_pct": round(
                100 * sum(1 for s in all_sessions if s.net_of_fee_bps > 0) / len(all_sessions), 1
            ),
        },
        "by_phase": by_phase,
    }


def main() -> int:
    result = study()
    out = Path(__file__).resolve().parents[3] / "data" / "gap_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nfull report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
