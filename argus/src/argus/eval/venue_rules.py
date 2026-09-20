"""Is any simple rule profitable on this venue at all? Asked over the whole history, not our record.

`eval/incremental.py` compares the desk against five fixed rules over the 42 instants the desk
actually faced. At 25 effective instants that comparison cannot settle a small difference and says
so. **The obvious way to fix it would be to fabricate desk decisions at instants the desk never
saw, and that is not available** — re-deciding today and calling it last month is the worst kind of
backtest, and the replay harness exists precisely to avoid it.

So this asks the neighbouring question, which needs no decision at all and can therefore use the
whole history: **do these rules make money on this venue?** Every tradeable instant across every
rToken, each rule scored on the realised move net of the same hurdle the desk faces. No model is
called, nothing is re-decided, and the desk does not appear in the result.

It bears directly on whether abstaining is right. The hurdle frontier established that trading
beats abstaining at 56% directional accuracy. If no simple rule reaches that over thousands of
instants, then a desk declining to trade this universe is declining something that does not pay —
and if one does reach it, the desk has a case to answer.

**Every interval here is dependence-aware.** Instants six hours apart share eighteen hours of their
twenty-four-hour horizon, so the observations are not independent and an ordinary standard error
would be too narrow. `backtest/dependence.py` supplies the correction: a stationary bootstrap with
the Politis-White block length, and Lo's HAC interval beside it. Where the two disagree, the
disagreement is reported rather than resolved — the closed form assumes the statistic is
asymptotically normal and the bootstrap does not.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Any

from argus.backtest.dependence import bootstrap_sharpe, hac_sharpe, optimal_block_length
from argus.backtest.metrics import MetricError
from argus.eval.incremental import BASELINES, DIRECTION_NONE, Instant, Policy

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "venue_rules.json"

MIN_INSTANTS = 200
"""Fewest instants before a venue-wide claim is made.

Far above `eval/incremental.MIN_INSTANTS` on purpose. That floor governs a paired comparison where
both arms see identical instants and the pairing removes most of the variance; this is an absolute
claim about profitability, which needs a sample that could actually refute it.
"""


class VenueRulesError(ValueError):
    """Raised rather than reporting a venue-wide claim from too little."""


@dataclass(frozen=True, slots=True)
class RuleResult:
    """One rule over the whole venue."""

    name: str
    instants: int
    trades: int
    net_bps: tuple[float, ...]
    accuracy: float | None
    """Directional accuracy over instants the rule traded. None when it never traded."""

    @property
    def total_bps(self) -> float:
        return sum(self.net_bps)

    @property
    def mean_bps(self) -> float:
        return fmean(self.net_bps) if self.net_bps else 0.0

    @property
    def interval(self) -> dict[str, Any]:
        """A dependence-aware interval on the per-instant net return, both ways round.

        Reported as a dict rather than a number because the two methods can disagree and the
        disagreement is informative: the closed form assumes the Sharpe estimator is asymptotically
        normal, and on a fat-tailed net-return series it is not.
        """
        traded = [n for n in self.net_bps if n != 0.0]
        if len(traded) < 30:
            return {"available": False, "why": f"only {len(traded)} traded instant(s)"}
        out: dict[str, Any] = {"available": True, "traded": len(traded)}
        try:
            hac = hac_sharpe(traded)
            out["hac"] = {
                "sharpe": round(hac.sharpe, 5),
                "low": round(hac.low, 5),
                "high": round(hac.high, 5),
                "eta": round(hac.eta, 4),
                "widening": round(hac.widening, 4),
                "excludes_zero": hac.excludes_zero,
            }
        except MetricError as exc:
            out["hac"] = {"error": str(exc)[:120]}
        try:
            block = optimal_block_length(traded)[0]
            boot = bootstrap_sharpe(traded, block_length=block, resamples=1_000)
            out["bootstrap"] = {
                "sharpe": round(boot.statistic, 5),
                "low": round(boot.low, 5),
                "high": round(boot.high, 5),
                "block_length": round(block, 2),
                "excludes_zero": boot.excludes_zero,
            }
        except MetricError as exc:
            out["bootstrap"] = {"error": str(exc)[:120]}
        return out

    @property
    def profitable(self) -> bool | None:
        """True only when a dependence-aware interval excludes zero on the profitable side.

        None when no interval could be formed. A positive total with an interval covering zero is
        **not** profitable here, which is the whole point of computing the interval.
        """
        interval = self.interval
        if not interval.get("available"):
            return None
        for method in ("bootstrap", "hac"):
            row = interval.get(method, {})
            if row.get("excludes_zero") and self.mean_bps > 0:
                return True
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instants": self.instants,
            "trades": self.trades,
            "total_bps": round(self.total_bps, 2),
            "mean_bps": round(self.mean_bps, 4),
            "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
            "interval": self.interval,
            "profitable": self.profitable,
        }


def score_rule(name: str, policy: Policy, instants: Sequence[Instant]) -> RuleResult:
    directions = [policy(i) for i in instants]
    nets = [i.net_bps(d) for i, d in zip(instants, directions, strict=True)]
    traded = [
        (i, d) for i, d in zip(instants, directions, strict=True) if d != DIRECTION_NONE
    ]
    accuracy = (
        sum(1 for i, d in traded if d * i.realised_bps > 0) / len(traded) if traded else None
    )
    return RuleResult(
        name=name,
        instants=len(instants),
        trades=len(traded),
        net_bps=tuple(nets),
        accuracy=accuracy,
    )


@dataclass(frozen=True, slots=True)
class VenueSurvey:
    """What the simple rules earn on this venue, and what it means for abstaining."""

    instants: int
    symbols: tuple[str, ...]
    break_even_accuracy: float
    results: tuple[RuleResult, ...]

    @property
    def best(self) -> RuleResult:
        return max(self.results, key=lambda r: r.total_bps)

    @property
    def any_profitable(self) -> bool:
        return any(r.profitable for r in self.results)

    @property
    def verdict(self) -> str:
        traded = [r for r in self.results if r.accuracy is not None]
        best_accuracy = max((r.accuracy or 0.0) for r in traded) if traded else 0.0
        head = (
            f"Five fixed rules over {self.instants:,} tradeable instant(s) across "
            f"{len(self.symbols)} rToken(s). Best directional accuracy {best_accuracy:.1%} "
            f"against a break-even of {self.break_even_accuracy:.1%}."
        )
        if self.any_profitable:
            names = ", ".join(r.name for r in self.results if r.profitable)
            return (
                f"{head} {names} is profitable with a dependence-aware interval that excludes "
                f"zero. The desk declining to trade this universe therefore has a case to answer, "
                f"and that case is now a measured one rather than a suspicion."
            )
        return (
            f"{head} NO RULE IS PROFITABLE once the interval accounts for overlapping horizons: "
            f"every one covers zero or is negative. A desk declining to trade this universe with "
            f"these rules is declining something that does not pay, which is the strongest "
            f"available defence of abstention short of the desk trading and showing it."
        )

    def render(self) -> str:
        lines = [
            f"VENUE RULES — {self.instants:,} instant(s), {len(self.symbols)} symbol(s)",
            "",
            f"{'rule':>18}{'trades':>8}{'total bps':>12}{'mean':>9}{'accuracy':>10}"
            f"{'profitable':>12}",
        ]
        for result in sorted(self.results, key=lambda r: -r.total_bps):
            accuracy = "  n/a" if result.accuracy is None else f"{result.accuracy:.1%}"
            profitable = {True: "yes", False: "no", None: "n/a"}[result.profitable]
            lines.append(
                f"{result.name:>18}{result.trades:>8}{result.total_bps:>12.0f}"
                f"{result.mean_bps:>9.2f}{accuracy:>10}{profitable:>12}"
            )
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "instants": self.instants,
            "symbols": list(self.symbols),
            "break_even_accuracy": round(self.break_even_accuracy, 5),
            "any_profitable": self.any_profitable,
            "results": [r.as_dict() for r in self.results],
            "verdict": self.verdict,
        }


def survey(
    instants: Sequence[Instant],
    *,
    rules: dict[str, Policy] | None = None,
    minimum: int = MIN_INSTANTS,
) -> VenueSurvey:
    if len(instants) < minimum:
        raise VenueRulesError(
            f"{len(instants)} instant(s) is below the {minimum} a venue-wide claim needs; this is "
            f"an absolute statement about profitability, not a paired comparison"
        )
    from argus.eval.hurdle import break_even_accuracy

    magnitudes = [abs(i.realised_bps) for i in instants]
    hurdle = fmean([i.hurdle_bps for i in instants])
    break_even = break_even_accuracy(fmean(magnitudes), hurdle)
    chosen = BASELINES if rules is None else rules
    return VenueSurvey(
        instants=len(instants),
        symbols=tuple(sorted({i.symbol for i in instants})),
        break_even_accuracy=0.5 if break_even is None else break_even,
        results=tuple(score_rule(name, policy, instants) for name, policy in chosen.items()),
    )


def collect(*, days: int = 180, every_hours: int = 6) -> list[Instant]:  # pragma: no cover
    """Every tradeable instant across the universe, with its trailing window and realised move.

    No decision is reconstructed and no model is called. The instants are generated the same way
    `paper/replay.instants` generates them — tradeable sessions only, spaced so a decision does not
    overlap its own hold — so this and the replay harness describe the same moments.
    """
    from argus.backtest.engine import Bar
    from argus.eval.collect import MIN_TRAILING, TRAILING_BARS, trailing_returns
    from argus.eval.hurdle import default_hurdle_bps
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range
    from argus.paper.replay import HOLD_HOURS
    from argus.paper.replay import instants as tradeable_instants

    hurdle = default_hurdle_bps()
    out: list[Instant] = []
    for symbol in RTOKEN_SYMBOLS:
        try:
            candles = fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.MARKET)
        except Exception:
            continue
        bars = [Bar(ts=c.ts, close=c.close, extra={}) for c in candles]
        closes = {b.ts: float(b.close) for b in bars}
        stamps = [b.ts for b in bars[1:]]
        returns = [
            0.0 if float(a.close) == 0 else float(b.close) / float(a.close) - 1.0
            for a, b in pairwise(bars)
        ]
        index = {b.ts: i for i, b in enumerate(bars)}
        for at in tradeable_instants(bars, every_hours=every_hours):
            start = index.get(at)
            if start is None or start + HOLD_HOURS >= len(bars):
                continue
            entry = closes[at]
            exit_price = float(bars[start + HOLD_HOURS].close)
            if entry <= 0:
                continue
            trailing = trailing_returns(stamps, returns, at, bars=TRAILING_BARS)
            if len(trailing) < MIN_TRAILING:
                continue
            out.append(Instant(
                symbol=symbol, at=at, trailing=trailing,
                realised_bps=(exit_price / entry - 1.0) * 10_000, hurdle_bps=hurdle,
            ))
    return out


def main() -> int:  # pragma: no cover - CLI
    instants = collect()
    result = survey(instants)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(result.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "MIN_INSTANTS",
    "RuleResult",
    "VenueRulesError",
    "VenueSurvey",
    "collect",
    "score_rule",
    "survey",
]
