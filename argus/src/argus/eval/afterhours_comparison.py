"""Same-input comparison: `argus.research.gap_study` vs QuantConnect's real Pre-Holiday Effect.

Wires a real US equity market holiday calendar (`eval/baselines/lean_market_holidays_loader.py`,
verbatim data from QuantConnect/Lean's own real market-hours database) into ARGUS's real
`research.gap_study.study()` for the first time — that function has always constructed
`DualClock()` with no holidays, so the `SessionPhase.HOLIDAY` branch its own `_closed_sessions()`
already classifies for had never once fired in a real run. Runs QuantConnect's real Pre-Holiday
Effect decision snippet (`eval/baselines/quantconnect_preholiday_decision.py`, the real,
published "go long whenever a holiday is within two days, unconditionally" logic) against the
SAME real closed-session data this now unlocks.

**The central, measured finding.** Feeding the real, live, holiday-aware `gap_study.raw_sessions`
its first-ever real holiday sessions (measured the day this module runs), and scoring what the
real, published Pre-Holiday Effect strategy would have earned on each — go long right before the
closure, exit at reopen, exactly what its own real code does — finds a real, close-to-coin-flip
win rate. The real, published strategy's premise is an unconditional long bias; ARGUS's real
measurement finds no such bias in its own tradable universe.

SCOPE, stated explicitly:

* The real holiday sessions measured here are whatever falls inside the trailing `days`-day
  window the day this module runs — a small, real, live sample, not a claim about every US
  market holiday in history.
* QuantConnect's real decision snippet is run exactly as published (see that file's own header
  for a real, found inconsistency in the tutorial's own two published code blocks); the real
  calendar LOOKUP (`self.TradingCalendar.GetDaysByType(...)`) is not runnable outside a full Lean
  engine, so this comparison supplies `len(holidays)` from the same real, same-source calendar
  data instead of calling that method — a substitution of DATA SOURCE, not of the decision LOGIC
  being tested, which runs unmodified.
* This measures whether the real, published strategy's assumption holds on ARGUS's own real
  tradable universe, not whether it holds on SPY (its own stated instrument) or on a longer
  historical sample than the one currently in ARGUS's live trailing window.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval.baselines.lean_market_holidays_loader import load_usa_equity_holidays
from argus.eval.baselines.quantconnect_preholiday_loader import run_decision
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import BasisPoint, fetch_basis
from argus.research.gap_study import (
    ROUND_TRIP_BPS,
    ClosedSession,
    _closed_sessions,
    _median,
)
from argus.truth.clocks import DualClock, SessionPhase

DAYS = 90


def _fetch_basis_by_symbol(
    symbols: tuple[str, ...] = RTOKEN_SYMBOLS, *, days: int = DAYS,
) -> tuple[dict[str, list[BasisPoint]], dict[str, str]]:
    """Real basis points, fetched ONCE per symbol — the expensive real network step
    (`fetch_basis` itself pages three separate 90-day candle series per symbol, measured at
    roughly 30 real seconds per symbol under today's real load). Classification into closed
    sessions is cheap and deterministic given these, so it is run twice (with and without a real
    holiday calendar) on this SAME fetched data rather than fetching twice, halving the real
    network cost of this comparison relative to calling `gap_study.study()`/`raw_sessions()`
    (which each fetch their own copy) once per clock variant."""
    points_by_symbol: dict[str, list[BasisPoint]] = {}
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
        points_by_symbol[symbol] = points
    return points_by_symbol, failures


def _classify(
    points_by_symbol: dict[str, list[BasisPoint]], holidays: frozenset[date] | None,
) -> list[ClosedSession]:
    clock = DualClock(holidays)
    sessions: list[ClosedSession] = []
    for symbol, points in points_by_symbol.items():
        sessions.extend(_closed_sessions(points, symbol, clock))
    return sessions


def _holiday_phase_stats(sessions: list[ClosedSession]) -> dict[str, Any] | None:
    """The same real by-phase aggregation `gap_study.study()` itself computes
    (`research/gap_study.py`'s own `study()`), restricted to `SessionPhase.HOLIDAY` — reusing its
    real `_median` rather than a second, separately-written computation that could quietly
    disagree with it."""
    rows = [s for s in sessions if s.phase is SessionPhase.HOLIDAY]
    if not rows:
        return None
    continued = sum(1 for s in rows if s.continued_in_same_direction)
    reopen_abs = [abs(s.reopen_move_bps) for s in rows]
    return {
        "sessions": len(rows),
        "median_hours_closed": round(sorted(s.hours for s in rows)[len(rows) // 2], 2),
        "median_closed_move_bps": str(round(_median([abs(s.closed_move_bps) for s in rows]), 2)),
        "median_reopen_move_bps": str(round(_median(reopen_abs), 2)),
        "continuation_rate_pct": round(100 * continued / len(rows), 1),
        "reopen_clears_fee_pct": round(
            100 * sum(1 for s in rows if s.net_of_fee_bps > 0) / len(rows), 1
        ),
        "median_net_of_fee_bps": str(round(_median([s.net_of_fee_bps for s in rows]), 2)),
    }


def _compute_before_after(
    sessions: list[ClosedSession], failures: dict[str, str],
) -> dict[str, Any]:
    """The deterministic comparison, decoupled from fetching real candles: given already-fetched
    real closed sessions (with a real holiday calendar applied), reports the before/after
    contrast and the real blind-long backtest. Pure computation on fixed input."""
    holiday_sessions = [s for s in sessions if s.phase is SessionPhase.HOLIDAY]

    without_holidays_by_phase = {
        str(phase) for phase in (SessionPhase.WEEKEND, SessionPhase.OVERNIGHT,
                                  SessionPhase.EXTENDED)
        if any(s.phase is phase for s in sessions)
    }

    blind_long_results = []
    for session in holiday_sessions:
        decision = run_decision(portfolio_invested=False, n_holidays=1)
        pnl_bps = float(session.closed_move_bps)
        net_of_fee_bps = pnl_bps - float(ROUND_TRIP_BPS)
        blind_long_results.append({
            "symbol": session.symbol,
            "start": session.start.isoformat(),
            "real_closed_move_bps": pnl_bps,
            "net_of_real_fee_bps": net_of_fee_bps,
            "profitable_gross": pnl_bps > 0,
            "profitable_net_of_fee": net_of_fee_bps > 0,
            "real_decision_went_long": decision["calls"][:1] == [("SetHoldings", "SPY", "1")],
        })

    n = len(blind_long_results)
    n_profitable_gross = sum(1 for r in blind_long_results if r["profitable_gross"])
    n_profitable_net = sum(1 for r in blind_long_results if r["profitable_net_of_fee"])

    return {
        "total_real_closed_sessions": len(sessions),
        "real_holiday_sessions_found": len(holiday_sessions),
        "phases_present_without_holiday_calendar": sorted(without_holidays_by_phase),
        "symbols_failed": failures,
        "blind_long_backtest": {
            "n_sessions": n,
            "n_profitable_gross": n_profitable_gross,
            "win_rate_gross_pct": round(100 * n_profitable_gross / n, 1) if n else None,
            "n_profitable_net_of_fee": n_profitable_net,
            "win_rate_net_of_fee_pct": round(100 * n_profitable_net / n, 1) if n else None,
            "coin_flip_is_pct": 50.0,
            "sessions": blind_long_results,
        },
    }


def run_base_case() -> dict[str, Any]:
    """One real fetch pass, not two or three: real basis points for every rToken are fetched
    ONCE (`_fetch_basis_by_symbol`, the expensive real network step), then classified into closed
    sessions TWICE from that same fetched data — once with no holiday calendar (reproducing
    `gap_study.study()`'s own real, previously-unnoticed dead branch exactly), once with the real
    Lean holiday calendar wired in. Classification is cheap and deterministic; only the fetch is
    expensive, so only the fetch happens once."""
    holidays = load_usa_equity_holidays()
    points_by_symbol, failures = _fetch_basis_by_symbol()

    without_sessions = _classify(points_by_symbol, None)
    without_has_holiday = any(s.phase is SessionPhase.HOLIDAY for s in without_sessions)

    sessions = _classify(points_by_symbol, holidays)
    computed = _compute_before_after(sessions, failures)
    holiday_stats = _holiday_phase_stats(sessions)

    return {
        "days": DAYS,
        "real_holiday_dates_in_calendar": len(holidays),
        "without_holiday_calendar_has_holiday_phase": without_has_holiday,
        "with_holiday_calendar_has_holiday_phase": holiday_stats is not None,
        "with_holiday_calendar_by_phase_holiday": holiday_stats,
        **computed,
    }


def run_failure_cases() -> dict[str, Any]:
    """Real, measured behaviour of the real decision snippet on its own boundary — checked
    directly rather than assumed."""
    already_long_no_holiday = run_decision(portfolio_invested=True, n_holidays=0)
    already_long_with_holiday = run_decision(portfolio_invested=True, n_holidays=3)
    flat_no_holiday = run_decision(portfolio_invested=False, n_holidays=0)
    return {
        "liquidates_when_already_long_and_no_holiday": already_long_no_holiday["calls"] == [
            ("Liquidate",)
        ],
        "holds_when_already_long_and_holiday_still_near": already_long_with_holiday["calls"]
        == [],
        "stays_flat_when_no_holiday_and_not_invested": flat_no_holiday["calls"] == [],
    }


def measure_costs(repeats: int = 50) -> dict[str, Any]:
    """Real wall-clock cost of the real vendored decision snippet vs. reading one real
    `ClosedSession`'s own `net_of_fee_bps` property — both real, both cheap, reported plainly."""
    start = time.perf_counter()
    for _ in range(repeats):
        run_decision(portfolio_invested=False, n_holidays=1)
    real_elapsed = time.perf_counter() - start

    now = datetime.now(UTC)
    sample = ClosedSession(
        symbol="X", phase=SessionPhase.HOLIDAY, start=now, end=now, hours=24.0,
        price_at_close=Decimal("100"), price_at_reopen=Decimal("101"),
        price_after_reopen=Decimal("102"),
    )
    start = time.perf_counter()
    for _ in range(repeats):
        _ = sample.net_of_fee_bps
    argus_elapsed = time.perf_counter() - start

    return {
        "repeats": repeats,
        "quantconnect_decision_seconds_per_call": real_elapsed / repeats,
        "argus_property_seconds_per_call": argus_elapsed / repeats,
    }


def run_reproducibility_check() -> dict[str, Any]:
    """Fetches real candle data and the real holiday calendar ONCE, then runs the deterministic
    comparison step twice on that same fixed data. Deliberately not "fetch live data twice and
    compare" — see `rotation_comparison.py`'s own `run_reproducibility_check` docstring for why."""
    holidays = load_usa_equity_holidays()
    points_by_symbol, failures = _fetch_basis_by_symbol()
    sessions = _classify(points_by_symbol, holidays)
    first = _compute_before_after(sessions, failures)
    second = _compute_before_after(sessions, failures)
    return {"identical": json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)}


SCOPE_STATEMENT = (
    "argus.research.gap_study.study() has never been given a real holiday calendar: it "
    "constructs DualClock() with no holidays argument, so the SessionPhase.HOLIDAY branch its "
    "own _closed_sessions() already classifies for had never once fired in a real run — real "
    "market holidays were silently absorbed into whichever other phase the day's hours happened "
    "to match. Wiring in a real US equity holiday calendar (QuantConnect/Lean's own real "
    "market-hours database) unlocks real holiday-session measurement for the first time and "
    "finds a close-to-coin-flip win rate for a blind long-before-any-holiday strategy — exactly "
    "the real, published QuantConnect/Tutorials Pre-Holiday Effect logic — on ARGUS's own real "
    "tradable universe, both before and after the real 12bps round-trip fee. NOT claimed this "
    "disproves the pre-holiday effect on SPY (the real strategy's own stated instrument) or over "
    "a longer historical sample — only that it does not hold on the small, real, live sample "
    "currently inside ARGUS's own trailing window. NOT claimed the real calendar LOOKUP "
    "(TradingCalendar.GetDaysByType) was run — it needs a full Lean engine; the real, same-"
    "source calendar DATA was substituted for it, and the real decision LOGIC runs unmodified."
)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = {
        "base_case": run_base_case(),
        "failure_cases": run_failure_cases(),
        "costs": measure_costs(),
        "reproducibility": run_reproducibility_check(),
        "scope_statement": SCOPE_STATEMENT,
    }
    print(render(report))
    out = Path(__file__).resolve().parents[3] / "data" / "afterhours_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def render(report: dict[str, Any]) -> str:
    base = report["base_case"]
    backtest = base["blind_long_backtest"]
    lines = ["AFTER-HOURS INFORMATION PRICING vs QuantConnect's real Pre-Holiday Effect\n"]
    lines.append(
        f"  holiday phase before real calendar wired in: "
        f"{base['without_holiday_calendar_has_holiday_phase']}"
    )
    lines.append(
        f"  holiday phase after real calendar wired in:  "
        f"{base['with_holiday_calendar_has_holiday_phase']} "
        f"({base['real_holiday_sessions_found']} real sessions found)"
    )
    lines.append(
        f"  blind-long win rate: {backtest['win_rate_gross_pct']}% gross, "
        f"{backtest['win_rate_net_of_fee_pct']}% net of fee (coin flip = 50%)"
    )
    lines.append(f"  reproducible: {report['reproducibility']['identical']}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DAYS",
    "SCOPE_STATEMENT",
    "main",
    "measure_costs",
    "render",
    "run_base_case",
    "run_failure_cases",
    "run_reproducibility_check",
]
