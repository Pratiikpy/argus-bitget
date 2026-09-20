"""Every vetted primitive through the four anti-overfit gates, then through the fee.

**Why this module exists.** `data/overfit_gates.json` is cited in `README.md` for the project's
strongest negative finding — *four primitives are indistinguishable from shuffled data and the other
four still lose money after the 12bps fee* — and **nothing in the codebase could produce it.** It
was written by a script that no longer exists, so a reader who wanted to check the claim had no
command to run. A negative result nobody can reproduce is worth no more than a positive one nobody
can reproduce.

**It is not the same measurement as `factor_lab.json`, and the difference is the point.** The lab
runs its own funnel — a deflated-Sharpe gate over the whole trial set, then the fee — and on the
live record it kills all eight, seven on fees and one on DSR. This module runs the *four
anti-overfit gates* of `research/overfit.py` first, which separates two populations the lab's
summary merges:

* **rejected as noise** — the factor fails a gate, most damningly the placebo. There is no edge to
  cost, so the fee never gets a say.
* **cleared every gate and still unprofitable** — real, stable, out-of-sample structure that the
  round trip eats anyway.

Reporting those separately is the whole finding. *Statistical structure and tradeable edge are
different things*, and a single "rejected" bucket cannot say so.

**What it deliberately does not do.** It does not rank, score or average the gates.
`OverfitReport.verdict` refuses a 0-100 score for a stated reason — a factor that fails the placebo
is indistinguishable from noise, and no number of other passes redeems that — and this module keeps
that refusal rather than flattening it into a league table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar
from argus.research.factor_lab import PRIMITIVES
from argus.research.overfit import MIN_OBSERVATIONS, Observation, OverfitReport, run_all

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "overfit_gates.json"

FEE_BPS = 12.0
"""The round trip on rToken perps: taker in and taker out.

The same figure `eval/hurdle.py` and `research/factor_lab.py` charge. Hardcoding it separately here
would let three modules drift into three different answers about the same venue.
"""

HORIZON_BARS = 1
"""Hold one hourly bar. The primitives are session-structure signals evaluated bar to bar, which is
what the artefact this replaces measured."""

PERIOD_BARS = 55
"""Bars per IC period.

**This is the one number in this module that is inferred rather than recovered, and it is flagged
as such.** `research/overfit.py:period_ic` computes a *rank* IC across the rows sharing a period, so
one observation per period makes every IC undefined and every gate INCONCLUSIVE — the factor must be
bucketed. The orphaned artefact does not record its bucket size, but it does record
``positive_rate: 0.5128`` for `long_while_closed`, which is exactly **20/39**; and 2,158
observations over 39 periods is ~55 bars each. That is the arithmetic this constant comes from.

⚠️ **NOT VERIFIED against the original code, which no longer exists.** The window has also moved on
since 2026-09-12, so the stored figures cannot be reproduced exactly even with the right bucket
size. What is restored is the *method and the shape*; the numbers are re-measured, and the artefact
says when.
"""


class GatesError(RuntimeError):
    """Raised rather than publishing a verdict computed from too little history."""


@dataclass(frozen=True, slots=True)
class FactorOutcome:
    """One primitive, its gate report, and what the fee did to it."""

    factor: str
    report: OverfitReport
    gross_sharpe: float
    net_sharpe: float
    observations: int

    @property
    def cleared_every_gate(self) -> bool:
        return self.report.verdict == "clears every gate"

    @property
    def profitable_after_fees(self) -> bool:
        return self.net_sharpe > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "anti_overfit": self.report.as_dict(),
            "gross_sharpe": round(self.gross_sharpe, 4),
            "net_sharpe": round(self.net_sharpe, 4),
            "observations": self.observations,
        }


def _sharpe(returns: list[float]) -> float:
    """Per-bar Sharpe, unannualised.

    Unannualised deliberately: the comparison here is between primitives on the same bars, and an
    annualisation factor is a constant that cancels — while making the number look like a
    portfolio result it is not.
    """
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return mean / (var ** 0.5) if var > 0 else 0.0


def observe(bars: list[Bar], factor: str, *, horizon: int = HORIZON_BARS) -> list[Observation]:
    """Signal now against the return realised over the next ``horizon`` bar(s).

    Strictly forward: the signal at ``i`` is scored against ``close[i+horizon] / close[i]``, never
    against a return that overlaps the bar the signal was read from.

    Bars are grouped into periods of :data:`PERIOD_BARS` because the gates compute a **rank** IC
    across the rows sharing a period — see that constant for why, and for what is inferred about it.
    """
    if factor not in PRIMITIVES:
        raise GatesError(f"{factor} is not a vetted primitive; known: {', '.join(PRIMITIVES)}")
    signal = PRIMITIVES[factor]

    out: list[Observation] = []
    for i in range(len(bars) - horizon):
        close = float(bars[i].close)
        later = float(bars[i + horizon].close)
        if close <= 0:
            raise GatesError(
                f"{factor}: a close of {close} at {bars[i].ts} is not a price, and a zero return "
                f"read from it would be indistinguishable from a real flat bar"
            )
        out.append(
            Observation(
                period=i // PERIOD_BARS,
                name=bars[i].ts.isoformat(),
                factor=float(signal(bars, i)),
                forward_return=later / close - 1.0,
            )
        )
    return out


def screen(bars: list[Bar], factor: str) -> FactorOutcome:
    """One primitive through the four gates and then through the fee."""
    observations = observe(bars, factor)
    if len(observations) < MIN_OBSERVATIONS:
        raise GatesError(
            f"{factor}: {len(observations)} observation(s) is below the {MIN_OBSERVATIONS} the "
            f"gates require. A verdict from fewer would be a statement about the sample size"
        )

    report = run_all(observations)

    # The strategy return: hold the signal's sign for the horizon, pay the round trip whenever the
    # position changes. A factor that flips every bar pays every bar, which is exactly the cost
    # structure that kills session-frequency signals on this venue.
    gross: list[float] = []
    net: list[float] = []
    previous = 0.0
    for ob in observations:
        position = 1.0 if ob.factor > 0 else (-1.0 if ob.factor < 0 else 0.0)
        raw = position * ob.forward_return
        turnover = abs(position - previous)
        gross.append(raw)
        net.append(raw - turnover * FEE_BPS / 10_000)
        previous = position

    return FactorOutcome(
        factor=factor,
        report=report,
        gross_sharpe=_sharpe(gross),
        net_sharpe=_sharpe(net),
        observations=len(observations),
    )


def report_of(
    outcomes: list[FactorOutcome], *, instrument: str, bars: int, interval: str
) -> dict[str, Any]:
    """The artefact, in the shape `README.md` already cites."""
    # **Three buckets, not two.** A factor the gates could not evaluate has not been rejected — it
    # has not been measured, and folding it in with the refuted ones would be the
    # absence-is-not-zero error, in the module written to honour that rule. `docclaims`
    # caught exactly this: the first
    # version reported 8 noise primitives when 6 were refuted and 2 were unevaluated.
    noise = [o.factor for o in outcomes if o.report.verdict.startswith("rejected")]
    not_proven = [o.factor for o in outcomes if o.report.verdict.startswith("not proven")]
    cleared = [o for o in outcomes if o.cleared_every_gate]
    unprofitable = [o.factor for o in cleared if not o.profitable_after_fees]
    net_negative = [o.factor for o in outcomes if o.net_sharpe <= 0]

    finding = (
        f"{len(noise)} of {len(outcomes)} primitives are rejected by the anti-overfit gates as "
        f"indistinguishable from shuffled data, and {len(not_proven)} return no verdict at all for "
        f"want of data — which is not a pass. {len(cleared)} clear every gate, of which "
        f"{len(unprofitable)} still lose money after the {FEE_BPS:.0f}bps round trip. "
        f"{len(net_negative)} of {len(outcomes)} are net-negative after fees regardless of gate. "
        f"Statistical structure and tradeable edge are different things, and on this venue the "
        f"difference is the fee."
    )
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "instrument": instrument,
        "bars": bars,
        "interval": interval,
        "fee_bps": FEE_BPS,
        "period_bars": PERIOD_BARS,
        "rejected_as_noise": noise,
        "not_proven_insufficient_data": not_proven,
        "net_negative_after_fees": net_negative,
        "cleared_every_gate": [o.factor for o in cleared],
        "cleared_but_unprofitable_after_fees": unprofitable,
        "finding": finding,
        "detail": [o.as_dict() for o in outcomes],
    }


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from argus.market.history import CandleType, fetch_range

    parser = argparse.ArgumentParser(
        description="every vetted primitive through the four anti-overfit gates, then the fee"
    )
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--save", action="store_true", help=f"write {REPORT_PATH.name}")
    args = parser.parse_args()

    candles = fetch_range(args.symbol, days=args.days, interval="1H",
                          candle_type=CandleType.MARKET)
    bars = [
        Bar(
            ts=c.ts,
            close=Decimal(str(c.close)),
            extra={"volume": float(c.volume), "high": float(c.high), "low": float(c.low)},
        )
        for c in candles
    ]
    print(f"{args.symbol}: {len(bars)} hourly bars\n")

    outcomes = [screen(bars, name) for name in PRIMITIVES]
    for got in outcomes:
        mark = "CLEARS" if got.cleared_every_gate else "noise "
        print(
            f"  {mark}  {got.factor:<22} gross {got.gross_sharpe:+7.3f}  "
            f"net {got.net_sharpe:+7.3f}  {got.report.verdict[:52]}"
        )

    blob = report_of(outcomes, instrument=args.symbol, bars=len(bars), interval="1H")
    print(f"\n  {blob['finding']}")

    if args.save:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"\n  written to {REPORT_PATH}")
    else:
        print("\n  (not saved — pass --save)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "FEE_BPS",
    "HORIZON_BARS",
    "PERIOD_BARS",
    "REPORT_PATH",
    "FactorOutcome",
    "GatesError",
    "main",
    "observe",
    "report_of",
    "screen",
]
