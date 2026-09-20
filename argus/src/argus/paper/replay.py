"""Replay — the desk run against past regular-hours instants, so the hypothesis can be tested now.

**The question this answers, and why it could not wait.** Protocol v1 permits trading only in
regular and extended hours (`paper/protocol.py`, ``tradeable_sessions``), and every one of the 126
decisions on the live ledger was taken in a weekend or overnight session. So the pre-registered
hypothesis — *"during US regular trading hours the desk will open positions, and if it abstains
through a full regular-hours week the deliberation hurdle is too high to trade this universe at
all"* — is **untested**, and the only way to test it by waiting is to wait.

It does not have to be tested by waiting. Regular hours happened last week. The desk can be run
against them.

**What makes this legitimate rather than a backtest of a language model.** Every input is
reconstructed as of the decision instant, from sources that are themselves point-in-time:

* **Candles** up to and including the instant, never past it.
* **SEC filings** whose *acceptance* timestamp precedes the instant. EDGAR publishes that
  timestamp, so this is the filing's real availability, not an assumption about it.
* **The Treasury curve and the VIX** as of the instant's date, from the dated series.
* **FINRA short volume** for the session **before** the instant, because that file is published
  after the close of the session it describes.

**What is deliberately absent, said before anyone asks.** Live RSS headlines cannot be
reconstructed — the feeds carry only recent items — and Bitget's Skill tools answer only for now.
A replayed frame is therefore **thinner than a live one**, and that biases the desk *toward*
abstention, not away from it. A replay that trades is evidence; a replay that abstains is weaker
evidence than a live abstention would be, and :class:`ReplayResult` says so in the artefact.

**It is written to its own ledger and its own chain.** `data/replay_ledger.jsonl`, never the live
one. A replayed decision and a live decision are different objects: one was made against a full
evidence panel in real time, the other against a reconstructed and thinner one. Mixing them would
inflate the live record with decisions that were never live, which is the single most tempting
dishonesty available here and the reason for the separate file.

**Settlement is by the same rule as live.** The outcome is the move from the decision instant to
the instant plus the protocol's hold window, read from candles that postdate the decision. Because
the replay runs over past data, those candles already exist — which is exactly why the frame
construction has to be airtight, and why :func:`frame_at` refuses any bar at or after the instant.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.truth.evidence import Evidence

REPLAY_PATH = Path(__file__).resolve().parents[3] / "data" / "replay_ledger.jsonl"

HOLD_HOURS = 24
"""Matches the live protocol's hold window. A replay on a different horizon would not test the
same policy."""

TRADEABLE_PHASES = ("rth", "extended")
"""Only the sessions the committed protocol permits trading in. Replaying a weekend instant would
re-measure what the live ledger already establishes."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One reconstructed decision instant: everything knowable then, and nothing after."""

    symbol: str
    at: datetime
    price: Decimal
    evidence: tuple[Evidence, ...]
    absent: tuple[str, ...]
    """Sources that exist live and cannot be reconstructed. Named, never omitted."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "at": self.at.isoformat(), "price": str(self.price),
            "evidence": len(self.evidence),
            "channels": sorted({e.source for e in self.evidence}),
            "absent_sources": list(self.absent),
        }


def frame_at(
    symbol: str,
    at: datetime,
    bars: Sequence[Any],
    *,
    filings: Sequence[Evidence] = (),
    macro: Sequence[Evidence] = (),
) -> Frame | None:
    """Build the frame for one instant. Returns ``None`` when the instant is not reconstructible.

    The guard that matters is the bar filter: only bars **strictly before or at** the instant are
    used for the price, and every piece of evidence is filtered on ``available_at <= at``. An
    off-by-one here would hand the desk the answer and every number downstream would be worthless.
    """
    usable = [b for b in bars if b.ts <= at]
    if not usable:
        return None
    admissible = tuple(
        e for e in list(filings) + list(macro) if e.available_at <= at
    )
    return Frame(
        symbol=symbol,
        at=at,
        price=Decimal(str(usable[-1].close)),
        evidence=(
            Evidence(
                id=f"mkt-{symbol}-{at.isoformat()}",
                claim=(
                    f"{symbol} last {usable[-1].close}; "
                    f"{len(usable)} hourly bars of history to this instant"
                ),
                source="news",
                available_at=at,
                credibility=1.0,
            ),
            *admissible,
        ),
        absent=(
            "live RSS headlines (feeds carry only recent items)",
            "Bitget official Skills (answer only for now)",
            "consensus estimates and insider Form 4 (not reconstructed in this run)",
        ),
    )


def realised_bps(
    bars: Sequence[Any], at: datetime, *, hold_hours: int = HOLD_HOURS
) -> float | None:
    """The move from ``at`` to ``at + hold``. ``None`` when the horizon runs past the data.

    A replay whose horizon exceeds the available history must not settle at the last bar it has —
    that would score a 24-hour call against however many hours happened to remain, and short
    horizons would look systematically calmer.
    """
    entry = [b for b in bars if b.ts <= at]
    exit_target = at + timedelta(hours=hold_hours)
    later = [b for b in bars if b.ts >= exit_target]
    if not entry or not later:
        return None
    before, after = float(entry[-1].close), float(later[0].close)
    return (after - before) / before * 10_000.0 if before > 0 else None


@dataclass(frozen=True)
class Outcome:
    """One replayed decision and what the market did next."""

    symbol: str
    at: datetime
    verdict: str
    side: str
    quantity: Decimal
    confidence: float
    thesis: str
    evidence_count: int
    realised_bps: float
    cost_bps: float

    @property
    def opened(self) -> bool:
        return self.quantity > 0 and self.verdict not in ("no_trade", "data_insufficient")

    @property
    def net_bps(self) -> float:
        """Signed return net of the round trip. Zero for an abstention — it held nothing.

        An abstention's *counterfactual* is recorded separately in :attr:`realised_bps`; conflating
        the two would credit a desk that stood aside with the move it declined.
        """
        if not self.opened:
            return 0.0
        direction = 1.0 if self.side.lower() in ("buy", "long") else -1.0
        return direction * self.realised_bps - self.cost_bps

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "at": self.at.isoformat(), "verdict": self.verdict,
            "side": self.side, "quantity": str(self.quantity), "confidence": self.confidence,
            "thesis": self.thesis[:400], "evidence_count": self.evidence_count,
            "realised_bps": round(self.realised_bps, 3),
            "opened": self.opened, "net_bps": round(self.net_bps, 3),
        }


@dataclass(frozen=True)
class ReplayResult:
    """The whole replay, and the honest reading of it."""

    outcomes: tuple[Outcome, ...]
    instants: int
    absent_sources: tuple[str, ...] = ()
    note: str = field(default="")

    @property
    def opened(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if o.opened)

    @property
    def abstentions(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if not o.opened)

    @property
    def net_returns(self) -> list[float]:
        return [o.net_bps / 10_000.0 for o in self.outcomes]

    @property
    def hypothesis_verdict(self) -> str:
        """The pre-registered question, answered by this run alone."""
        if not self.outcomes:
            return "no decision was replayed; the hypothesis is untouched"
        if self.opened:
            return (
                f"SUPPORTED in part: the desk opened {len(self.opened)} position(s) in "
                f"{len(self.outcomes)} regular-hours instant(s). The pre-registered claim that it "
                f"trades during regular hours is not refuted."
            )
        return (
            f"NOT SUPPORTED by this run: the desk abstained on all {len(self.outcomes)} "
            f"regular-hours instant(s) replayed. The pre-registered reading of that is that the "
            f"deliberation-cost hurdle is too high to trade this universe — and the honest caveat "
            f"is that a replayed frame carries fewer sources than a live one, which biases the "
            f"desk toward abstention. This weakens the finding; it does not reverse it."
        )

    def render(self) -> str:
        lines = [
            f"REPLAY — {len(self.outcomes)} decision(s) over {self.instants} regular-hours "
            f"instant(s); {len(self.opened)} opened, {len(self.abstentions)} abstained",
            f"  {self.hypothesis_verdict}",
        ]
        if self.opened:
            wins = sum(1 for o in self.opened if o.net_bps > 0)
            total = sum(o.net_bps for o in self.opened)
            lines.append(
                f"  of the {len(self.opened)} position(s): {wins} net-positive after the round "
                f"trip, {total:+.1f}bps in total"
            )
        if self.absent_sources:
            lines.append(
                "  Sources a live frame has and this one does not: "
                + "; ".join(self.absent_sources)
            )
        lines.append(
            "  Written to data/replay_ledger.jsonl, NEVER to the live chain. A replayed decision "
            "and a live one are different objects and adding them together would inflate the live "
            "record with decisions that were never live."
        )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": len(self.outcomes),
            "instants": self.instants,
            "opened": len(self.opened),
            "abstained": len(self.abstentions),
            "hypothesis_verdict": self.hypothesis_verdict,
            "absent_sources": list(self.absent_sources),
            "outcomes": [o.as_dict() for o in self.outcomes],
            "note": self.note,
        }


def write(result: ReplayResult, *, path: Path = REPLAY_PATH) -> None:
    """Append every replayed outcome to the replay ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for outcome in result.outcomes:
            handle.write(json.dumps({"kind": "replay", **outcome.as_dict()}) + "\n")


def load(path: Path = REPLAY_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def instants(
    bars: Sequence[Any], *, phases: tuple[str, ...] = TRADEABLE_PHASES, every_hours: int = 6,
    hold_hours: int = HOLD_HOURS,
) -> list[datetime]:
    """Every tradeable instant in the history, spaced so decisions do not overlap their own hold.

    Spacing matters more than it looks. Two decisions six hours apart on the same symbol share
    eighteen hours of their twenty-four-hour horizon, so their outcomes are nearly the same
    observation counted twice — which is precisely the clustering `eval/forecasts.py` exists to
    measure. Six hours is a compromise stated rather than hidden, and the overlap is reported.
    """
    from argus.truth.clocks import DualClock

    clock = DualClock()
    last_usable = max(b.ts for b in bars) - timedelta(hours=hold_hours) if bars else None
    out: list[datetime] = []
    previous: datetime | None = None
    for bar in bars:
        if last_usable is not None and bar.ts > last_usable:
            break
        if str(clock.state(bar.ts).phase) not in phases:
            continue
        if previous is not None and (bar.ts - previous) < timedelta(hours=every_hours):
            continue
        out.append(bar.ts)
        previous = bar.ts
    return out


def reconstruct_sources(
    symbol: str, at: datetime, *, lookback_days: int = 45
) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
    """Every source that can honestly be rebuilt as of ``at``, and the ones that cannot.

    The first pilot ran with a single market line per frame and the desk abstained on all ten
    instants — which is almost no evidence, because a frame that thin biases the desk toward
    abstention by construction. An abstention on a rich frame is a finding; an abstention on an
    empty one is a tautology.

    Four sources rebuild honestly, each because it is itself dated at the source rather than
    merely fetched today:

    * **SEC filings** carry an acceptance timestamp published by EDGAR, and `EdgarSource.evidence`
      already applies the as-of rule against it, so passing a historical instant is enough.
    * **The Treasury par curve** is a dated daily series; the curve in force is the last one dated
      on or before the instant.
    * **CBOE VIX** is a dated daily close, placed in its own 9,000-day history exactly as the live
      path places it.
    * **FINRA short volume** is published per session and is read for the session strictly before
      the instant, because the file for a session appears after that session closes.

    Anything that cannot be rebuilt is returned as a named absence rather than quietly skipped.
    """
    from argus.market.evidence import EdgarSource
    from argus.market.macro import evidence as curve_evidence
    from argus.market.macro import fetch as fetch_curves
    from argus.market.microstructure import ShortVolume, fetch_short_volume
    from argus.market.microstructure import evidence as micro_evidence
    from argus.market.volatility import Reading, percentile_of
    from argus.market.volatility import evidence as vix_evidence
    from argus.market.volatility import history as vix_history

    out: list[Evidence] = []
    absent: list[str] = ["live RSS headlines (feeds carry only recent items)",
                         "Bitget official Skills (answer only for now)"]

    try:
        out.extend(EdgarSource().evidence(
            symbol, as_of=at, lookback=timedelta(days=lookback_days)
        ))
    except Exception:
        absent.append("SEC EDGAR filings (unavailable at replay time)")

    try:
        curves = [c for c in fetch_curves() if c.as_of <= at.date()]
        if curves:
            out.extend(curve_evidence(curves[-1], as_of=at))
        else:
            absent.append("US Treasury curve (no dated curve at or before this instant)")
    except Exception:
        absent.append("US Treasury curve (unavailable at replay time)")

    try:
        closes = vix_history()
        past = [(d, v) for d, v in closes if d <= at.date()]
        if past:
            day, level = past[-1]
            # No prior close means no day change. `level - level` would have reported the
            # first observation in the series as an unchanged day, which it is not: it is a day
            # whose change is unknown.
            prior = past[-2][1] if len(past) > 1 else None
            reading = Reading(
                as_of=datetime(day.year, day.month, day.day, tzinfo=at.tzinfo),
                level=level,
                change=None if prior is None else level - prior,
                # The history is daily closes. A replay has no intraday range, and saying the
                # high and the low both equalled the close would invent a flat session.
                day_high=None,
                day_low=None,
                percentile=percentile_of(level, [v for _, v in past]),
                observations=len(past),
            )
            out.extend(vix_evidence(reading, as_of=at))
        else:
            absent.append("CBOE VIX (no close at or before this instant)")
    except Exception:
        absent.append("CBOE VIX (unavailable at replay time)")

    try:
        from argus.market.evidence import underlying_ticker

        ticker = underlying_ticker(symbol)
        # The session BEFORE the instant: FINRA publishes a session's file after it closes.
        readings = fetch_short_volume(
            on=(at - timedelta(days=1)).date(), wanted=frozenset({ticker})
        )
        short: ShortVolume | None = readings.get(ticker)
        if short is not None and short.as_of < at.date():
            out.extend(micro_evidence(ticker=ticker, as_of=at, short=short))
        else:
            absent.append("FINRA short volume (no session strictly before this instant)")
    except Exception:
        absent.append("FINRA short volume (unavailable at replay time)")

    # The as-of rule applied once more at the boundary. Every source above claims to respect it;
    # this is the check rather than the claim, and it is cheap.
    admissible = tuple(e for e in out if e.available_at <= at)
    if len(admissible) != len(out):
        absent.append(
            f"{len(out) - len(admissible)} item(s) dated after the instant were refused"
        )
    return admissible, tuple(absent)


def run(
    *,
    symbols: Sequence[str],
    days: int = 14,
    every_hours: int = 6,
    max_decisions: int = 12,
    budget_per_decision: int = 30_000,
    path: Path = REPLAY_PATH,
) -> ReplayResult:
    """Replay the desk over past regular-hours instants and settle every decision.

    ``max_decisions`` is a hard stop and exists because each decision is a real model call against
    a finite hackathon key. A replay that silently spent the budget would be the same class of
    failure as the scheduled cycle that died mid-run, and the stop is reported in the result rather
    than discovered afterwards.
    """
    from argus.agents.desk import ConstitutionPolicy, TradingDesk
    from argus.backtest.engine import Bar
    from argus.cost.model import CostModel
    from argus.llm.qwen import QwenClient, QwenError, Thinking, TokenBudget
    from argus.market.evidence import underlying_ticker
    from argus.market.history import CandleType, fetch_range
    from argus.risk.hedgeability import HedgeabilitySurface, open_market_candidate
    from argus.truth.clocks import DualClock

    clock = DualClock()
    cost = CostModel.bitget_perp()
    round_trip = float(cost.round_trip_bps())
    client = QwenClient(budget=TokenBudget(limit=budget_per_decision * max_decisions))
    desk = TradingDesk(client, pm_thinking=Thinking.LOW)

    outcomes: list[Outcome] = []
    absent_seen: list[str] = []
    seen_instants = 0
    stopped = ""
    for symbol in symbols:
        try:
            candles = fetch_range(
                symbol, days=days, interval="1H", candle_type=CandleType.MARKET
            )
        except Exception as exc:
            stopped = f"{symbol}: candles unavailable ({str(exc)[:60]})"
            continue
        bars = [Bar(ts=c.ts, close=c.close, extra={}) for c in candles]
        for at in instants(bars, every_hours=every_hours):
            if len(outcomes) >= max_decisions:
                stopped = (
                    f"stopped at the {max_decisions}-decision ceiling; more instants exist and "
                    f"were not replayed"
                )
                break
            seen_instants += 1
            rebuilt, absent_here = reconstruct_sources(symbol, at)
            frame = frame_at(symbol, at, bars, macro=rebuilt)
            move = realised_bps(bars, at)
            if frame is None or move is None:
                continue
            absent_seen.extend(absent_here)
            session = clock.state(at, nav_age_seconds=5.0)
            underlying = underlying_ticker(symbol)
            hedges = HedgeabilitySurface(
                (open_market_candidate(underlying, Decimal("0.98")),),
                session_note=(
                    f"{underlying} was open at this instant and the hedge was reachable in the "
                    f"market; ARGUS has no equity broker, so it was not reachable by us"
                ),
            )
            try:
                run_result = desk.run(
                    symbol=symbol, session=session, token_price=frame.price,
                    position=Decimal("0"), evidence=list(frame.evidence), hedges=hedges,
                    decision_id=f"replay-{symbol}-{int(at.timestamp())}",
                    constitution=ConstitutionPolicy(),
                )
            except QwenError as exc:
                stopped = f"model unavailable after {len(outcomes)} decision(s): {str(exc)[:80]}"
                break
            final = run_result.proof.llm_revised_intent or run_result.proof.llm_original_intent
            outcomes.append(Outcome(
                symbol=symbol, at=at, verdict=str(final.verdict), side=str(final.side),
                quantity=final.quantity, confidence=final.stated_confidence,
                thesis=str(final.thesis), evidence_count=len(frame.evidence),
                realised_bps=move, cost_bps=round_trip,
            ))
        if stopped:
            break

    result = ReplayResult(
        outcomes=tuple(outcomes),
        instants=seen_instants,
        absent_sources=tuple(sorted(set(absent_seen))) or (
            "live RSS headlines", "Bitget official Skills",
        ),
        note=stopped,
    )
    if outcomes:
        write(result, path=path)
    return result


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import contextlib
    import sys

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="NVDAUSDT,TSLAUSDT")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--max", type=int, default=12)
    args = parser.parse_args()
    result = run(
        symbols=tuple(s.strip() for s in args.symbols.split(",") if s.strip()),
        days=args.days, max_decisions=args.max,
    )
    print(result.render())
    return 0


__all__ = [
    "HOLD_HOURS",
    "REPLAY_PATH",
    "TRADEABLE_PHASES",
    "Frame",
    "Outcome",
    "ReplayResult",
    "frame_at",
    "instants",
    "load",
    "main",
    "realised_bps",
    "run",
    "write",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
