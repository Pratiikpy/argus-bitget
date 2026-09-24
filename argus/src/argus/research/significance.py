"""Is the weekend effect real, or is it 156 coin flips that happened to land?

The gap study measured a 57.7% continuation rate across weekend closures against 50.6% overnight.
That is the most promising number this project has produced, which is exactly why it gets the
harshest treatment. Three checks, all of them from our own standing rules:

1. **Significance.** An exact binomial test against p=0.5. A hit rate is not a finding until the
   sample can carry it.
2. **Train/test split.** Our own backtest rules record that cross-sectional momentum "looked
   excellent full-sample and went negative in both halves once split". Any effect that does not
   survive a chronological split is not an effect.
3. **Per-symbol contribution.** The same rules record a single-name result that "looked strong only
   because one instrument produced the whole return from four trades". An aggregate that hides its
   distribution is not evidence.

A result that fails any of these is reported as failed. The PRD's standard is that an honest
negative is more credible than an unexplained positive, and this module is where that is enforced
rather than asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from math import comb, sqrt
from pathlib import Path
from typing import Any

from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import fetch_basis
from argus.research.gap_study import ClosedSession, _closed_sessions
from argus.truth.clocks import DualClock, SessionPhase


def binomial_p_value(successes: int, trials: int, p_null: float = 0.5) -> float:
    """Exact one-sided binomial test: P(X >= successes | p = p_null).

    Exact rather than normal-approximated because the samples here are small enough for the
    approximation to flatter a marginal result, and a marginal result is precisely what we have.
    """
    if trials <= 0:
        return 1.0
    if successes <= p_null * trials:
        return 1.0
    total = 0.0
    for k in range(successes, trials + 1):
        total += comb(trials, k) * (p_null ** k) * ((1 - p_null) ** (trials - k))
    return min(1.0, total)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — well behaved at small n, unlike the normal approximation.

    If the lower bound sits below 0.5, the effect is not distinguishable from a coin flip whatever
    the point estimate says.
    """
    if trials == 0:
        return (0.0, 1.0)
    phat = successes / trials
    denom = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    margin = z * sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True, slots=True)
class Verdict:
    label: str
    trials: int
    successes: int
    rate: float
    p_value: float
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        """Both tests must agree: p < 0.05 *and* the interval excludes a coin flip."""
        return self.p_value < 0.05 and self.ci_low > 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "trials": self.trials,
            "successes": self.successes,
            "rate_pct": round(100 * self.rate, 1),
            "p_value": round(self.p_value, 4),
            "ci95": [round(100 * self.ci_low, 1), round(100 * self.ci_high, 1)],
            "significant": self.significant,
        }


def _verdict(label: str, rows: list[ClosedSession]) -> Verdict:
    trials = len(rows)
    successes = sum(1 for s in rows if s.continued_in_same_direction)
    return Verdict(
        label=label,
        trials=trials,
        successes=successes,
        rate=successes / trials if trials else 0.0,
        p_value=binomial_p_value(successes, trials),
        **dict(zip(("ci_low", "ci_high"), wilson_interval(successes, trials), strict=True)),
    )


def study(*, days: int = 90) -> dict[str, Any]:
    clock = DualClock()
    sessions: list[ClosedSession] = []
    for symbol in RTOKEN_SYMBOLS:
        try:
            points = fetch_basis(symbol, days=days, interval="1H")
        except Exception:
            continue
        if len(points) >= 24:
            sessions.extend(_closed_sessions(points, symbol, clock))

    weekend = sorted(
        (s for s in sessions if s.phase in (SessionPhase.WEEKEND, SessionPhase.HOLIDAY)),
        key=lambda s: s.start,
    )
    overnight = sorted(
        (s for s in sessions if s.phase is SessionPhase.OVERNIGHT), key=lambda s: s.start
    )

    # 1 — full sample
    full = _verdict("weekend, full sample", weekend)
    control = _verdict("overnight, full sample (control)", overnight)

    # 2 — chronological split. Not random: a random split leaks the future into the past.
    mid = len(weekend) // 2
    train = _verdict("weekend, first half (train)", weekend[:mid])
    test = _verdict("weekend, second half (out-of-sample)", weekend[mid:])

    # 3 — per symbol
    per_symbol: dict[str, dict[str, Any]] = {}
    for symbol in sorted({s.symbol for s in weekend}):
        rows = [s for s in weekend if s.symbol == symbol]
        if len(rows) >= 8:
            per_symbol[symbol] = _verdict(symbol, rows).as_dict()

    positive = [s for s, v in per_symbol.items() if float(v["rate_pct"]) > 50]
    leave_one_out = {}
    for symbol in per_symbol:
        rest = [s for s in weekend if s.symbol != symbol]
        leave_one_out[f"without_{symbol}"] = round(
            100 * sum(1 for s in rest if s.continued_in_same_direction) / max(len(rest), 1), 1
        )

    median_net = sorted(s.net_of_fee_bps for s in weekend)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "days": days,
        "test_1_significance": {
            "weekend": full.as_dict(),
            "overnight_control": control.as_dict(),
        },
        "test_2_train_test_split": {
            "train": train.as_dict(),
            "out_of_sample": test.as_dict(),
            "survives_split": test.rate > 0.5 and train.rate > 0.5,
        },
        "test_3_per_symbol": {
            "symbols_tested": len(per_symbol),
            "symbols_above_coin_flip": len(positive),
            "detail": per_symbol,
            "leave_one_out_rates_pct": leave_one_out,
        },
        "economics": {
            "median_net_of_fee_bps": str(
                round(median_net[len(median_net) // 2], 2) if median_net else Decimal("0")
            ),
            "round_trip_fee_bps": "12",
        },
    }


def main() -> int:
    result = study()
    out = Path(__file__).resolve().parents[3] / "data" / "weekend_significance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    t1 = result["test_1_significance"]["weekend"]
    t2 = result["test_2_train_test_split"]
    t3 = result["test_3_per_symbol"]
    print("\n" + "=" * 70)
    print(f"  significance   {'PASS' if t1['significant'] else 'FAIL'}  "
          f"(p={t1['p_value']}, 95% CI {t1['ci95']})")
    print(f"  train/test     {'PASS' if t2['survives_split'] else 'FAIL'}")
    print(f"  per-symbol     {t3['symbols_above_coin_flip']}/{t3['symbols_tested']} "
          f"above coin flip")
    print("=" * 70)
    print(f"full report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
