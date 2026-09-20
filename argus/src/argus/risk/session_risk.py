"""Volatility by session phase, and the position throttle that follows from it.

The intuition every tokenized-equity system encodes is *the overnight is dangerous, so hold less
into it*. One rival freezes its mark and pauses rebalancing when the anchor market shuts. That
intuition is half right, and its half-wrongness matters: it throttles the wrong hours.

**Measured on 90 days of hourly rTokens, median absolute one-hour move in bps:**

| symbol | regular | extended | overnight | weekend | first bar after discovery resumes |
|---|---|---|---|---|---|
| NVDAUSDT | 31.5 | 14.1 | 12.8 | 5.2 | **52.1** |
| TSLAUSDT | 36.9 | 15.5 | 12.7 | 6.4 | **75.7** |
| AAPLUSDT | 23.7 | 10.8 | 8.0 | 5.3 | **46.1** |
| MSFTUSDT | 26.6 | 13.7 | 9.2 | 5.9 | **43.4** |
| METAUSDT | 34.2 | 14.6 | 10.1 | 6.4 | **67.9** |
| GOOGLUSDT | 23.5 | 13.6 | 9.3 | 5.6 | **51.7** |

The shut window is the **quietest** part of the week — a weekend hour moves a fifth as much as a
regular-hours one, which is what you would expect of a token whose anchor has no price to discover.
The risk is not in the darkness. It is in the **discontinuity at the end of it**: the first bar
after price discovery resumes moves 1.5 to 2.2 times a regular-hours bar on every instrument, and
5.5 to 11.8 times a weekend bar depending on the name, over 64 reopens each.

So a throttle keyed to "is the anchor asleep" reduces risk during the calmest hours of the week and
leaves it untouched for the one bar that actually carries the jump. This module throttles the
horizon that **crosses a reopen** instead, which is a different set of positions.

**The rule is volatility targeting, not a multiplier somebody chose.** For a position meant to be
held ``horizon`` bars from now, the expected path variance is the sum of the phase-specific
variances of the bars it will live through, plus the reopen variance for each transition it
crosses.
The baseline is the same number of regular-hours bars. The throttle is the ratio of the two
standard deviations, and it is **capped at one**: this is a risk layer, and `agents/desk.py`'s
Constitution may only reduce. A horizon that sits entirely inside a quiet weekend would justify
sizing *up* by this arithmetic, and the arithmetic is not permitted to say so.

Everything here is measured per symbol and refuses when it has not been. :func:`throttle` on an
unmeasured symbol returns 1.0 and says "not measured" — it does not invent a cautious-looking
number, because a cautious-looking number with no sample behind it is the defect this project keeps
finding in its own risk layer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from argus.truth.clocks import DualClock, SessionPhase

MIN_BARS_PER_PHASE = 30
"""Fewest observations before a phase's volatility is reported.

A median over ten bars is a statement about those ten bars. Phases are unequal by construction —
regular hours are 32.5 of a week's 168 — so the floor has to be low enough that the scarce phase
still qualifies and high enough that it means something.
"""

MIN_REOPENS = 10
"""Fewest discovery transitions before the reopen jump is used to throttle anything.

At one reopen per trading day, 90 days of history gives about 64. Below ten the jump estimate is
anecdote and the throttle falls back to the phase volatilities alone.
"""

STALE_AFTER_HOURS = 36
"""Same rule as `risk/effectiveness.py`: past this a measurement counts as absent, not as weaker."""

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "session_risk.json"


class SessionRiskError(ValueError):
    """Raised rather than returning a volatility profile built from too little data."""


@dataclass(frozen=True, slots=True)
class SessionRisk:
    """Per-phase volatility for one instrument, and the jump that ends each shut window."""

    symbol: str
    phase_bps: dict[str, float]
    """Median absolute one-bar move, by phase, in basis points."""

    phase_counts: dict[str, int]
    reopen_bps: float | None
    """Median absolute move on the first bar after price discovery resumes."""

    reopens: int
    measured_at: datetime
    window_days: int

    @property
    def baseline_bps(self) -> float:
        """The regular-hours bar, which is what a position is implicitly sized against."""
        return self.phase_bps.get(SessionPhase.RTH.value, 0.0)

    @property
    def reopen_multiple(self) -> float | None:
        """How many regular-hours bars of risk one reopen bar carries."""
        if self.reopen_bps is None or self.baseline_bps <= 0:
            return None
        return self.reopen_bps / self.baseline_bps

    def is_fresh(self, *, now: datetime | None = None) -> bool:
        return (
            (now or datetime.now(UTC)) - self.measured_at
        ).total_seconds() <= STALE_AFTER_HOURS * 3600

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "phase_bps": {k: round(v, 4) for k, v in sorted(self.phase_bps.items())},
            "phase_counts": dict(sorted(self.phase_counts.items())),
            "reopen_bps": None if self.reopen_bps is None else round(self.reopen_bps, 4),
            "reopens": self.reopens,
            "reopen_multiple": (
                None if self.reopen_multiple is None else round(self.reopen_multiple, 4)
            ),
            "measured_at": self.measured_at.isoformat(),
            "window_days": self.window_days,
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> SessionRisk:
        return cls(
            symbol=str(blob["symbol"]),
            phase_bps={str(k): float(v) for k, v in dict(blob["phase_bps"]).items()},
            phase_counts={str(k): int(v) for k, v in dict(blob["phase_counts"]).items()},
            reopen_bps=None if blob.get("reopen_bps") is None else float(blob["reopen_bps"]),
            reopens=int(blob["reopens"]),
            measured_at=datetime.fromisoformat(str(blob["measured_at"])),
            window_days=int(blob["window_days"]),
        )

    def render(self) -> str:
        parts = " ".join(
            f"{phase}={self.phase_bps[phase]:.1f}" for phase in sorted(self.phase_bps)
        )
        jump = (
            "no reopen sample" if self.reopen_multiple is None
            else f"reopen={self.reopen_bps:.1f} ({self.reopen_multiple:.1f}x rth, n={self.reopens})"
        )
        return f"{self.symbol}: {parts} | {jump}"


def measure(
    bars: Sequence[tuple[datetime, float]], *, symbol: str, clock: DualClock | None = None,
    window_days: int = 0, now: datetime | None = None,
) -> SessionRisk:
    """Median absolute move by phase, plus the reopen jump, from aligned hourly closes.

    The median rather than the standard deviation: these series carry occasional 5% prints, and a
    standard deviation over 90 days of hourly data is largely a statement about three of them. The
    throttle wants a typical bar, and the tail is `desk/stress.py`'s question rather than this one.
    """
    if len(bars) < MIN_BARS_PER_PHASE * 2:
        raise SessionRiskError(
            f"{len(bars)} bar(s) cannot support a per-phase volatility profile"
        )
    dual = clock or DualClock()
    by_phase: dict[str, list[float]] = {}
    reopen: list[float] = []
    previous: SessionPhase | None = None
    for (_, before), (stamp, after) in pairwise(bars):
        if before <= 0 or after <= 0:
            continue
        move = abs(after / before - 1.0) * 10_000
        phase = dual.phase(stamp)
        by_phase.setdefault(phase.value, []).append(move)
        if previous is not None and not previous.has_price_discovery and phase.has_price_discovery:
            reopen.append(move)
        previous = phase

    kept = {
        phase: median(values)
        for phase, values in by_phase.items()
        if len(values) >= MIN_BARS_PER_PHASE
    }
    if SessionPhase.RTH.value not in kept:
        raise SessionRiskError(
            "no regular-hours sample; without a baseline there is nothing to throttle against"
        )
    return SessionRisk(
        symbol=symbol,
        phase_bps=kept,
        phase_counts={phase: len(values) for phase, values in by_phase.items()},
        reopen_bps=median(reopen) if len(reopen) >= MIN_REOPENS else None,
        reopens=len(reopen),
        measured_at=now or datetime.now(UTC),
        window_days=window_days,
    )


@dataclass(frozen=True, slots=True)
class Throttle:
    """How much the position is scaled, and the arithmetic that produced the number."""

    multiplier: Decimal
    reason: str
    expected_bps: float
    baseline_bps: float
    reopens_crossed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "multiplier": str(self.multiplier),
            "reason": self.reason,
            "expected_bps": round(self.expected_bps, 4),
            "baseline_bps": round(self.baseline_bps, 4),
            "reopens_crossed": self.reopens_crossed,
        }


def expected_path_bps(
    profile: SessionRisk, *, start: datetime, horizon_bars: int, clock: DualClock | None = None,
) -> tuple[float, int]:
    """Expected one-bar-equivalent volatility over the next ``horizon_bars``, and reopens crossed.

    Variances add along a path; volatilities do not. So each bar contributes the square of its
    phase's typical move, a reopen contributes the square of the jump, and the total is turned back
    into a per-bar figure at the end. Summing the medians instead would understate a path that mixes
    a quiet weekend with a violent reopen — which is exactly the path this module exists to price.
    """
    if horizon_bars < 1:
        raise SessionRiskError("the horizon must be at least one bar")
    dual = clock or DualClock()
    total = 0.0
    reopens = 0
    previous = dual.phase(start)
    for step in range(1, horizon_bars + 1):
        stamp = start + timedelta(hours=step)
        phase = dual.phase(stamp)
        typical = profile.phase_bps.get(phase.value, profile.baseline_bps)
        if (
            previous is not None
            and not previous.has_price_discovery
            and phase.has_price_discovery
            and profile.reopen_bps is not None
        ):
            typical = profile.reopen_bps
            reopens += 1
        total += typical * typical
        previous = phase
    return math.sqrt(total / horizon_bars), reopens


def throttle(
    profile: SessionRisk | None, *, start: datetime, horizon_bars: int = 24,
    clock: DualClock | None = None, floor: Decimal = Decimal("0.25"),
) -> Throttle:
    """Scale a position so its expected path volatility matches a regular-hours one.

    Capped at 1.0 — a risk layer may reduce and never add — and floored at ``floor`` so a violent
    reopen cannot shrink a position to dust and call that risk management. An unmeasured symbol
    returns 1.0 with "not measured" as the reason: the alternative, a cautious-looking constant with
    no sample behind it, is the defect this module was built to remove from a different file.
    """
    if profile is None:
        return Throttle(
            multiplier=Decimal("1"), reason="session volatility not measured for this instrument",
            expected_bps=0.0, baseline_bps=0.0, reopens_crossed=0,
        )
    baseline = profile.baseline_bps
    if baseline <= 0:
        return Throttle(
            multiplier=Decimal("1"), reason="no regular-hours baseline to compare against",
            expected_bps=0.0, baseline_bps=0.0, reopens_crossed=0,
        )
    expected, reopens = expected_path_bps(
        profile, start=start, horizon_bars=horizon_bars, clock=clock,
    )
    if expected <= baseline:
        return Throttle(
            multiplier=Decimal("1"),
            reason=(
                f"the next {horizon_bars} bar(s) are expected to move {expected:.1f}bps per bar "
                f"against a {baseline:.1f}bps regular-hours bar; no reduction, and this layer may "
                f"not size up"
            ),
            expected_bps=expected, baseline_bps=baseline, reopens_crossed=reopens,
        )
    raw = Decimal(str(baseline / expected))
    bounded = max(floor, min(Decimal("1"), raw))
    detail = (
        f" crossing {reopens} reopen(s) at {profile.reopen_bps:.1f}bps"
        if reopens and profile.reopen_bps is not None else ""
    )
    return Throttle(
        multiplier=bounded,
        reason=(
            f"the next {horizon_bars} bar(s){detail} are expected to move {expected:.1f}bps per "
            f"bar against a {baseline:.1f}bps regular-hours bar, so the position is scaled to "
            f"{bounded:.2f} to hold risk at the regular-hours level"
        ),
        expected_bps=expected, baseline_bps=baseline, reopens_crossed=reopens,
    )


def load(path: Path | None = None) -> dict[str, SessionRisk]:
    """Measurements by symbol. An unreadable file is an empty map, never a default."""
    target = path or REPORT_PATH
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, SessionRisk] = {}
    for row in blob.get("measurements", []):
        try:
            got = SessionRisk.from_dict(row)
        except (KeyError, TypeError, ValueError):
            continue
        out[got.symbol] = got
    return out


def lookup(
    symbol: str, *, table: dict[str, SessionRisk] | None = None, now: datetime | None = None,
) -> SessionRisk | None:
    """The fresh profile for this symbol, or ``None``. Staleness is absence, as everywhere else."""
    found = (table if table is not None else load()).get(symbol)
    if found is None or not found.is_fresh(now=now):
        return None
    return found


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range

    parser = argparse.ArgumentParser(description="how much does each session phase actually move?")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or list(
        RTOKEN_SYMBOLS
    )
    clock = DualClock()
    now = datetime.now(UTC)
    rows: list[SessionRisk] = []
    for symbol in wanted:
        try:
            bars = fetch_range(symbol, days=args.days, interval="1H",
                               candle_type=CandleType.MARKET)
        except Exception as exc:
            print(f"  {symbol}: no history ({type(exc).__name__})")
            continue
        try:
            got = measure(
                [(c.ts, float(c.close)) for c in bars], symbol=symbol, clock=clock,
                window_days=args.days, now=now,
            )
        except SessionRiskError as exc:
            print(f"  {symbol}: {exc}")
            continue
        rows.append(got)
        scaled = throttle(got, start=now, horizon_bars=args.horizon, clock=clock)
        print(f"  {got.render()}")
        print(f"      throttle now x{scaled.multiplier} — {scaled.reason}")

    if not rows:
        print("nothing measurable; no file written")
        return 1
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "generated_at": now.isoformat(), "window_days": args.days,
                "measurements": [r.as_dict() for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(rows)} profile(s) written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "MIN_BARS_PER_PHASE",
    "MIN_REOPENS",
    "REPORT_PATH",
    "STALE_AFTER_HOURS",
    "SessionRisk",
    "SessionRiskError",
    "Throttle",
    "expected_path_bps",
    "load",
    "lookup",
    "measure",
    "throttle",
]
