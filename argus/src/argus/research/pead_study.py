"""Does the already-verified SUE ranking carry forward-return information on this venue?

The zero-settled-trades gap traces to a real finding this project already made and never acted
on. `eval/hurdle.py` proved the desk's abstention is not fee-driven — break-even directional
accuracy is 55%, not the near-100% a fee explanation would need. `research/overfitting_study.py`
then swept 25 systematic session-based variants across all 12 symbols, 300 trials, and found
**0 of 12 survive deflated Sharpe**: no systematic session-boundary signal on this venue is
distinguishable from noise once the trial count is priced in. And `research/sue.py`, while
verifying the Standardized Unexpected Earnings computation to floating-point identity against
QuantConnect's own reference, explicitly declined to test it as a signal: *"no return or edge
claim is made for SUE as a trading signal here."* This module is that untested claim, run for the
first time, with the same rigor that found the session sweep wanting.

**Why earnings drift and not another session variant.** Post-earnings-announcement drift is one of
finance's most replicated anomalies (Ball & Brown 1968 onward) and, unlike a session-boundary
microstructure rule invented for this specific synthetic venue, it has an economic mechanism that
does not depend on this venue existing: under-reaction to a genuine information shock. If anything
here has a better chance of surviving deflated Sharpe than a 12bps-round-trip session rule, the
literature would nominate this candidate, not a guess dressed up as one.

**Method.** For each of the nine real rToken anchors with a real ticker-to-EPS mapping
(`market.bitget.ANCHOR_OF`), every historical quarter with 12 trailing quarters of real EDGAR EPS
(`research.sue.MIN_QUARTERS`) gets a real SUE reading, dated by the real SEC filing date
(`Fact.filed` — point-in-time by construction, never the fiscal-period end). Events are pooled
across anchors: a single-name result is not a result here, the same lesson this project already
paid for once (`CLAUDE.md`: single-symbol trend "looked strong only because one instrument
produced the whole return"). A long-short rule — long positive SUE, short negative, the sign the
PEAD literature itself predicts rather than one fit to this sample — is tested at several holding
periods, each one "trial" in exactly the sense `research.track1_study` already counts trials: the
observed best is deflated for having looked at more than one.

**What is deliberately excluded, named rather than hidden.**

* Entry is delayed to the first bar dated strictly after the filing date. SEC filings carry no
  intraday timestamp on this feed (`Fact.filed` is a date, not a datetime), so same-day entry would
  assume pre-market availability this module cannot verify.
* Traditional PEAD windows run 60-90 calendar days; this venue's perpetual funding makes a hold
  that long expensive by construction, so the sweep stays inside 10 days and does not claim to
  reproduce the textbook-length effect — a different, shorter-horizon claim.
* The primary figures are commission-only (12bps round trip), matching
  `research.overfitting_study`'s own methodology so the two studies are directly comparable. A
  funding-adjusted secondary figure, using each anchor's real fetched funding rate at study time
  and the actual number of funding settlements each hold crosses, is reported alongside it rather
  than silently substituted in — funding was not measured at the time of each historical event,
  only at the time this module ran, so it is a same-order-of-magnitude estimate, not a historical
  fact, and is labelled as one.
* Sample size is whatever nine real anchors' real earnings calendars produced inside this venue's
  real listing history — likely a few dozen events, not the hundreds a systematic sweep gets from
  slicing one continuous price series. `MIN_EVENTS` and the validation layer's own refusals
  (`MetricError` from `min_track_record_length`, `probability_of_overfitting`) are what stand
  between that and reporting a Sharpe with no sample behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.backtest.metrics import MetricError, Performance, deflated_sharpe, evaluate
from argus.backtest.validation import (
    OverfittingResult,
    min_track_record_length,
    probability_of_overfitting,
)
from argus.cost.model import FUNDING_INTERVAL_HOURS, CostModel
from argus.market.bitget import ANCHOR_OF, fetch_rtokens
from argus.market.fundamentals import FundamentalsSource
from argus.market.history import Candle, fetch_range
from argus.research.sue import MIN_QUARTERS, SueError, read_dated, yoy_window

STUDY_PATH = Path(__file__).resolve().parents[3] / "data" / "pead_study.json"

HOLD_HOURS: tuple[int, ...] = (24, 72, 120, 240)
"""Holding periods swept, in hours. Each is one deflated-Sharpe trial. Capped at 240h (10 days):
Bitget perpetual funding, excluded from the primary commission-only figure, makes a longer hold
expensive on a real book — see the module docstring."""

PRICE_HISTORY_DAYS = 500
"""Long enough to clear the longest real listing history measured on this venue (NVDAUSDT:
2025-08-19, about 400 days as of 2026-09-23) with margin, without hardcoding a per-symbol listing
date that would go stale."""

ENTRY_DELAY_DAYS = 1
"""Enter on the first bar dated strictly after the filing date — see the module docstring."""

MIN_EVENTS = 10
"""Below this, a pooled return series is reported INSUFFICIENT rather than scored — the same floor
`eval/hurdle.py` applies to its own instant count, for the same reason: a Sharpe computed from a
handful of events has no sample behind it."""

PBO_GROUPS = 4
"""Fewer blocks than `overfitting_study`'s 8: pooled PEAD events number in the dozens, not the
thousands of hourly bars a session sweep gets, and each PBO block must itself hold enough
observations to compute a Sharpe. Named explicitly because it is a deliberate, weaker-power
tradeoff, not the same setting reused without thought."""


class PeadStudyError(RuntimeError):
    """Raised rather than reporting a study that could not gather real data."""


@dataclass(frozen=True, slots=True)
class PeadEvent:
    """One real, point-in-time earnings-surprise reading."""

    rtoken: str
    anchor: str
    filed: date
    period_end: date
    sue: float


@dataclass(frozen=True, slots=True)
class PeadTrade:
    """One realised round trip from one event, at one holding period."""

    event: PeadEvent
    hold_hours: int
    entry_at: datetime
    exit_at: datetime
    side: int
    """+1 (long) if SUE > 0, -1 (short) if SUE < 0 — the literature's own predicted sign, never
    fit to this sample."""

    gross_return: float
    net_return: float
    """Net of the 12bps commission round trip. Funding excluded — see :attr:`funding_adjusted`."""

    funding_adjusted: float
    """``net_return`` less an estimated funding cost from a rate fetched at study time, not at the
    historical event — a same-order-of-magnitude robustness figure, not a historical fact."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rtoken": self.event.rtoken,
            "anchor": self.event.anchor,
            "filed": self.event.filed.isoformat(),
            "sue": round(self.event.sue, 4),
            "hold_hours": self.hold_hours,
            "entry_at": self.entry_at.isoformat(),
            "exit_at": self.exit_at.isoformat(),
            "side": self.side,
            "gross_return_bps": round(self.gross_return * 10_000, 2),
            "net_return_bps": round(self.net_return * 10_000, 2),
            "funding_adjusted_bps": round(self.funding_adjusted * 10_000, 2),
        }


def _fetch_sue_events(
    anchors: Sequence[tuple[str, str]],
    *,
    as_of: datetime,
    source: FundamentalsSource | None = None,
) -> tuple[list[PeadEvent], dict[str, str]]:
    """Every real, PIT-computable SUE reading for these anchors, as of ``as_of``.

    ``anchors`` is ``(rtoken_symbol, ticker)`` pairs. Returns ``(events, skipped)`` — the same
    never-silently-drop discipline as :func:`argus.research.sue.rank_universe`. Walking the full
    fetched history rather than only the most recent window is deliberate: EDGAR often carries a
    decade or more of quarterly EPS for these anchors, and letting each event's own filing date
    decide whether it falls inside the tradeable price history (:func:`_entry_index`) is simpler
    and less error-prone than pre-computing a cutoff per symbol here.
    """
    src = source or FundamentalsSource()
    events: list[PeadEvent] = []
    skipped: dict[str, str] = {}
    for rtoken, ticker in anchors:
        facts, status = src.facts(ticker, concept="eps_diluted", as_of=as_of, quarterly_only=True)
        if len(facts) < MIN_QUARTERS:
            skipped[rtoken] = (
                f"{len(facts)} quarter(s) on EDGAR, below the {MIN_QUARTERS} SUE needs"
                + (f" ({'; '.join(status)})" if status else "")
            )
            continue
        made = 0
        for i in range(len(facts) - MIN_QUARTERS + 1):
            # Paired by date, not by position (corrected 2026-09-25): SEC XBRL has no
            # standalone fiscal Q4, so `facts[i + 4]` is not the same quarter a year earlier —
            # see `research.sue`'s module docstring and `eval/general_sue_comparison.py`. Only
            # quarters ending on or before `facts[i]` are visible to the reading.
            history = [(f.end, f.value) for f in facts[i:]]
            try:
                window = yoy_window(history)
                sue = read_dated(ticker, history)
            except SueError:
                continue
            used = {d.end for d in window} | {d.prior_end for d in window}
            events.append(PeadEvent(
                rtoken=rtoken, anchor=ticker,
                filed=max(f.filed for f in facts[i:] if f.end in used),
                period_end=facts[i].end, sue=sue.sue,
            ))
            made += 1
        if made == 0:
            skipped[rtoken] = f"{len(facts)} quarter(s) fetched, no window produced a valid SUE"
    return events, skipped


def _entry_index(bars: Sequence[Candle], filed: date) -> int | None:
    """First bar strictly after ``filed`` — see :data:`ENTRY_DELAY_DAYS`.

    ``None`` both when the event postdates every fetched bar (too recent to hold) and when it
    predates the first one. The second case matters: EDGAR carries years of history and a naive
    "first bar on or after cutoff" scan would clamp a decade-old filing to bar 0 and silently
    test it as though the news broke the instant this venue's price history begins — a real,
    found-by-running defect, not a hypothetical one (see `tests/test_pead_study.py`).
    """
    if not bars:
        return None
    cutoff = filed + timedelta(days=ENTRY_DELAY_DAYS)
    if cutoff < bars[0].ts.date():
        return None
    for i, bar in enumerate(bars):
        if bar.ts.date() >= cutoff:
            return i
    return None


def _build_trade(
    event: PeadEvent,
    bars: Sequence[Candle],
    *,
    hold_hours: int,
    cost: CostModel,
    funding_rate_bps: float,
) -> PeadTrade | None:
    """One event's outcome at one holding period, or ``None`` if the price history cannot cover
    it — either the event predates the fetched bars or the hold would run past the last one."""
    entry_i = _entry_index(bars, event.filed)
    if entry_i is None:
        return None
    exit_i = entry_i + hold_hours
    if exit_i >= len(bars):
        return None
    entry_px, exit_px = float(bars[entry_i].close), float(bars[exit_i].close)
    if entry_px <= 0:
        return None
    side = 1 if event.sue > 0 else -1
    gross = side * (exit_px - entry_px) / entry_px
    commission = float(cost.round_trip_bps()) / 10_000.0
    net = gross - commission
    hours_held = (bars[exit_i].ts - bars[entry_i].ts).total_seconds() / 3600.0
    settlements = int(hours_held // float(FUNDING_INTERVAL_HOURS))
    funding_cost = abs(funding_rate_bps) / 10_000.0 * settlements
    return PeadTrade(
        event=event, hold_hours=hold_hours,
        entry_at=bars[entry_i].ts, exit_at=bars[exit_i].ts, side=side,
        gross_return=gross, net_return=net, funding_adjusted=net - funding_cost,
    )


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - 1)


def _performance_or_none(returns: list[float], *, periods_per_year: int) -> Performance | None:
    if len(returns) < MIN_EVENTS:
        return None
    try:
        return evaluate(returns, [1.0] * len(returns), periods_per_year=periods_per_year)
    except MetricError:
        return None


@dataclass(frozen=True, slots=True)
class PeadStudy:
    generated_at: datetime
    anchors_used: tuple[str, ...]
    skipped_anchors: dict[str, str]
    events_found: int
    trades: tuple[PeadTrade, ...]
    performance_by_hold: dict[int, Performance | None]
    best_hold_hours: int | None
    best_sharpe: float | None
    deflated_sharpe: float | None
    """P(true Sharpe > the expected max of this many trials). Absent when too few holds scored to
    deflate, or when the winning Sharpe is non-positive (no length of record helps a losing
    strategy, and deflating one is not informative)."""
    pbo: OverfittingResult | None = None
    min_track_record_years: float | None = None
    verdict: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "anchors_used": list(self.anchors_used),
            "skipped_anchors": self.skipped_anchors,
            "hold_hours_swept": list(HOLD_HOURS),
            "events_found": self.events_found,
            "trades_realised": len(self.trades),
            "performance_by_hold": {
                str(h): (p.as_dict() if p else None) for h, p in self.performance_by_hold.items()
            },
            "best_hold_hours": self.best_hold_hours,
            "best_sharpe": round(self.best_sharpe, 4) if self.best_sharpe is not None else None,
            "deflated_sharpe": (
                round(self.deflated_sharpe, 4) if self.deflated_sharpe is not None else None
            ),
            "pbo": (
                {
                    "pbo": round(self.pbo.pbo, 4),
                    "splits": self.pbo.splits,
                    "strategies": self.pbo.strategies,
                    "observations": self.pbo.observations,
                }
                if self.pbo is not None else None
            ),
            "min_track_record_years": (
                round(self.min_track_record_years, 2)
                if self.min_track_record_years is not None else None
            ),
            "verdict": self.verdict,
            "trades": [t.as_dict() for t in self.trades],
        }

    def render(self) -> str:
        lines = [
            f"PEAD/SUE STUDY — {len(self.anchors_used)} anchor(s), {self.events_found} PIT "
            f"SUE event(s), {len(self.trades)} realised trade(s) across {len(HOLD_HOURS)} "
            f"holding period(s)",
            "",
        ]
        for h in HOLD_HOURS:
            p = self.performance_by_hold.get(h)
            if p is None:
                lines.append(f"  {h:>4}h   INSUFFICIENT (fewer than {MIN_EVENTS} events)")
            else:
                lines.append(
                    f"  {h:>4}h   n={p.trades:<4} sharpe={p.sharpe:>7.3f}  "
                    f"win_rate={100 * p.win_rate:>5.1f}%  "
                    f"total_return={100 * p.total_return:>7.2f}%"
                )
        lines.append("")
        if self.best_hold_hours is not None:
            lines.append(
                f"  best: {self.best_hold_hours}h, Sharpe {self.best_sharpe:.3f}, "
                f"deflated {self.deflated_sharpe:.4f}"
                if self.deflated_sharpe is not None
                else f"  best: {self.best_hold_hours}h, Sharpe {self.best_sharpe:.3f} "
                     f"(not deflated)"
            )
        if self.pbo is not None:
            lines.append(f"  PBO: {self.pbo.pbo:.3f} over {self.pbo.splits} splits")
        if self.min_track_record_years is not None:
            lines.append(
                f"  min track record to trust this Sharpe: "
                f"{self.min_track_record_years:.2f} years"
            )
        lines += ["", f"  VERDICT: {self.verdict}"]
        return "\n".join(lines)


def build(
    *,
    as_of: datetime | None = None,
    price_history_days: int = PRICE_HISTORY_DAYS,
    cost: CostModel | None = None,
) -> PeadStudy:
    now = as_of or datetime.now(UTC)
    etfs = {"QQQ", "TQQQ", "SQQQ"}
    anchors = [(rtoken, ticker) for rtoken, ticker in ANCHOR_OF.items() if ticker not in etfs]
    cost_model = cost or CostModel.bitget_perp()

    events, skipped = _fetch_sue_events(anchors, as_of=now)
    if not events:
        raise PeadStudyError(
            f"no PIT-computable SUE event for any of {len(anchors)} anchor(s): {skipped}"
        )

    funding_by_rtoken: dict[str, float] = {}
    try:
        for symbol, ticker in fetch_rtokens().items():
            funding_by_rtoken[symbol] = float(ticker.funding_rate) * 10_000.0
    except Exception:
        funding_by_rtoken = {}

    rtokens_with_events = {e.rtoken for e in events}
    bars_by_rtoken: dict[str, list[Candle]] = {}
    trades: list[PeadTrade] = []
    for rtoken, _ticker in anchors:
        if rtoken not in rtokens_with_events:
            continue
        try:
            bars_by_rtoken[rtoken] = fetch_range(rtoken, days=price_history_days, interval="1H")
        except Exception as exc:  # pragma: no cover - network
            skipped.setdefault(rtoken, f"price history fetch failed: {exc}")

    for event in events:
        bars = bars_by_rtoken.get(event.rtoken)
        if not bars:
            continue
        funding_bps = funding_by_rtoken.get(event.rtoken, 0.0)
        for hold in HOLD_HOURS:
            trade = _build_trade(
                event, bars, hold_hours=hold, cost=cost_model, funding_rate_bps=funding_bps,
            )
            if trade is not None:
                trades.append(trade)

    performance_by_hold: dict[int, Performance | None] = {}
    sharpes_by_hold: dict[int, float] = {}
    for hold in HOLD_HOURS:
        returns = [t.net_return for t in trades if t.hold_hours == hold]
        perf = _performance_or_none(returns, periods_per_year=365 * 24 // hold)
        performance_by_hold[hold] = perf
        if perf is not None:
            sharpes_by_hold[hold] = perf.sharpe

    best_hold = max(sharpes_by_hold, key=lambda h: sharpes_by_hold[h]) if sharpes_by_hold else None
    best_sharpe = sharpes_by_hold.get(best_hold) if best_hold is not None else None

    deflated: float | None = None
    if best_hold is not None and best_sharpe is not None and best_sharpe > 0:
        all_sharpes = list(sharpes_by_hold.values())
        n_best = performance_by_hold[best_hold].trades  # type: ignore[union-attr]
        try:
            deflated = deflated_sharpe(
                best_sharpe, n=n_best, trials=max(1, len(all_sharpes)),
                variance_of_trials=_sample_variance(all_sharpes),
            )
        except MetricError:
            deflated = None

    # PBO needs a complete matrix: every "row" (event) must have a return at every hold length,
    # which only events far enough from the end of the fetched history satisfy — see the
    # module docstring's note on ragged coverage.
    pbo_result: OverfittingResult | None = None
    by_event: dict[tuple[str, date], dict[int, float]] = {}
    for t in trades:
        by_event.setdefault((t.event.rtoken, t.event.filed), {})[t.hold_hours] = t.net_return
    complete_rows = [
        [row[h] for h in HOLD_HOURS] for row in by_event.values()
        if all(h in row for h in HOLD_HOURS)
    ]
    if len(complete_rows) >= PBO_GROUPS * 3:
        try:
            pbo_result = probability_of_overfitting(complete_rows, groups=PBO_GROUPS)
        except MetricError:
            pbo_result = None

    min_years: float | None = None
    if best_hold is not None and best_sharpe is not None and best_sharpe > 0:
        returns = [t.net_return for t in trades if t.hold_hours == best_hold]
        try:
            min_years = min_track_record_length(
                returns, benchmark_sharpe=0.0,
            ) / (365 * 24 // best_hold)
        except MetricError:
            min_years = None

    verdict = _verdict(
        trades=trades, best_hold=best_hold, best_sharpe=best_sharpe,
        deflated=deflated, pbo=pbo_result,
    )

    return PeadStudy(
        generated_at=now,
        anchors_used=tuple(sorted({e.rtoken for e in events} & set(bars_by_rtoken))),
        skipped_anchors=skipped,
        events_found=len(events),
        trades=tuple(trades),
        performance_by_hold=performance_by_hold,
        best_hold_hours=best_hold,
        best_sharpe=best_sharpe,
        deflated_sharpe=deflated,
        pbo=pbo_result,
        min_track_record_years=min_years,
        verdict=verdict,
    )


def _verdict(
    *,
    trades: list[PeadTrade],
    best_hold: int | None,
    best_sharpe: float | None,
    deflated: float | None,
    pbo: OverfittingResult | None,
) -> str:
    if not trades:
        return "NO DATA — no event fell inside a tradeable price history; nothing was tested"
    if best_hold is None or best_sharpe is None:
        return (
            f"INSUFFICIENT — {len(trades)} trade(s) realised but none reached {MIN_EVENTS} at a "
            f"single holding period; no Sharpe is reportable"
        )
    if best_sharpe <= 0:
        return (
            f"NO EDGE — best net Sharpe across {len(HOLD_HOURS)} holding period(s) is "
            f"{best_sharpe:.3f} (hold {best_hold}h), not positive; SUE sign does not predict "
            f"forward returns net of cost on this sample"
        )
    if deflated is None:
        return (
            f"UNVERIFIED — best net Sharpe {best_sharpe:.3f} (hold {best_hold}h) is positive but "
            f"could not be deflated for the trial count; treat as noise until it can be"
        )
    if deflated < 0.95:
        return (
            f"DOES NOT SURVIVE DEFLATION — best net Sharpe {best_sharpe:.3f} (hold {best_hold}h) "
            f"deflates to P={deflated:.4f} of being real given {len(HOLD_HOURS)} trials; "
            f"indistinguishable from the best of several noisy holding-period guesses, the exact "
            f"failure mode `research.overfitting_study` already found in 12 of 12 session-based "
            f"symbols"
        )
    if pbo is not None and pbo.pbo >= 0.5:
        return (
            f"SURVIVES DEFLATION BUT FAILS PBO — deflated Sharpe P={deflated:.4f} looks real, but "
            f"the selection procedure itself scores PBO={pbo.pbo:.3f} (>= 0.5 is indistinguishable "
            f"from picking at random); do not trust which holding period was chosen"
        )
    return (
        f"SURVIVES — best net Sharpe {best_sharpe:.3f} (hold {best_hold}h) deflates to "
        f"P={deflated:.4f} across {len(HOLD_HOURS)} trials"
        + (f", PBO={pbo.pbo:.3f}" if pbo is not None
           else ", PBO not computable at this sample size")
        + f", on {len(trades)} real trade(s) pooled across {len({t.event.anchor for t in trades})} "
        f"anchor(s) — the first quantitative signal on this venue to clear this bar"
    )


def main() -> int:  # pragma: no cover - CLI
    import sys

    from argus.eval.artefact import write as write_artefact

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    study = build()
    undefined = write_artefact(STUDY_PATH, study.as_dict())
    print(study.render())
    if undefined:
        print(f"\nnon-finite (written as null): {', '.join(undefined)}")
    print(f"\nwritten to {STUDY_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ENTRY_DELAY_DAYS",
    "HOLD_HOURS",
    "MIN_EVENTS",
    "PBO_GROUPS",
    "PRICE_HISTORY_DAYS",
    "PeadEvent",
    "PeadStudy",
    "PeadStudyError",
    "PeadTrade",
    "build",
]
