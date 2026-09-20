"""Hedge effectiveness, measured from the venue's own series instead of asserted.

`risk/hedgeability.py` multiplies five factors to price a hedge, and three of them — the risk it
would remove, how much we trust the correlation, and how well the basis behaves — were **constants
typed into the source**: ``correlation_confidence=Decimal("0.98")``, ``basis_stability=0.95``,
``risk_reduction=0.95``. Nothing measured them. They were plausible, they were never wrong in a way
anyone could see, and a system whose standing rule is *never guess, read the source* had three
guesses inside its risk layer.

This module replaces them with three quantities that come from data and carry their sample size.

**1. Risk reduction is Ederington's hedging effectiveness.** Regress the change in the thing you
hold on the change in the thing you would hedge with; the minimum-variance hedge ratio is the slope
``Cov(spot, hedge) / Var(hedge)`` and the share of variance removed is the regression's R²
(Ederington 1979, *The Hedging Performance of the New Futures Markets*). That is exactly the field
``risk_reduction`` claims to be — "fraction of the position's risk this instrument would remove if
it executed perfectly" — so it should be that number and not a round one near it.

**2. Basis stability is the same measure at a unit ratio.** Holding one token against one unit of
its anchor leaves the basis as the residual, so the share of variance that survives is
``1 - Var(spot - hedge) / Var(spot)``. The gap between this and the optimal-ratio R² *is* the basis
problem: when the two coincide the relationship is one-for-one, and when unit-ratio effectiveness
collapses while optimal-ratio effectiveness holds, the hedge works only if you size it correctly.

**3. Correlation confidence is the lower bound of a confidence interval, not the estimate.** The
field's own docstring says it is "how much we trust the correlation estimate *in the current
regime*" — a statement about sampling error, which has a standard answer: Fisher's z-transform,
``atanh(r) ± z / sqrt(n - 3)``, transformed back. Sixty observations and a correlation of 0.98
give a tight bound; twelve give a wide one, and the wide one should shrink the hedge's credit.
A point estimate cannot express that difference, which is why the constant could not either.

**Measured per session phase, because the anchor sleeps.** During regular hours the anchor equity
trades and its index moves with it. Overnight and at weekends the index is still published but the
market behind it is shut, so the same correlation means something different. `truth/clocks.py`
already draws that line; this module measures each side of it separately and will not pool them.

**It refuses rather than defaults.** Below :data:`MIN_PAIRS` observations no number is returned —
the caller gets an exception, not a conservative-looking constant. That is the whole point: the
previous behaviour was a constant that looked conservative and had no sample behind it at all.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

MIN_PAIRS = 60
"""Fewest change-pairs before an effectiveness estimate is reported at all.

The Fisher interval is asymptotic and ``sqrt(n - 3)`` is brutal at small n: at twelve observations a
0.9 correlation's 95% lower bound is roughly 0.68, which is honest but barely usable, and below that
the interval is wider than the range it is trying to describe.
"""

CONFIDENCE_Z = 1.959963984540054
"""Two-sided 95%. Stated to full precision so the interval is reproducible rather than approximately
reproducible; 1.96 shifts the bound in the third decimal at these sample sizes."""

STALE_AFTER_HOURS = 36
"""How old a measurement may be before the live path treats it as absent.

Longer than a weekend gap on purpose — a Monday cycle should still be able to read Friday's
measurement — and short enough that a fortnight-old correlation can never silently price a hedge.
"""


class EffectivenessError(ValueError):
    """Raised rather than returning an effectiveness computed from too little data."""


class Provenance(StrEnum):
    """Where a hedge factor came from. Carried so a reader never has to assume."""

    MEASURED = "measured"
    ASSUMED = "assumed"


@dataclass(frozen=True, slots=True)
class HedgeEffectiveness:
    """One measurement of how well one instrument would hedge another, with its sample."""

    spot: str
    hedge: str
    phase: str
    observations: int
    correlation: float
    correlation_low: float
    """Lower bound of the 95% Fisher interval. The number the risk layer is given."""

    hedge_ratio: float
    """Minimum-variance ratio: units of hedge per unit of spot."""

    r_squared: float
    """Share of spot variance removed at the optimal ratio — Ederington effectiveness."""

    unit_ratio_effectiveness: float
    """Share removed at a one-for-one ratio: what survives the basis without sizing."""

    measured_at: datetime
    window_days: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "spot": self.spot, "hedge": self.hedge, "phase": self.phase,
            "observations": self.observations,
            "correlation": round(self.correlation, 6),
            "correlation_low": round(self.correlation_low, 6),
            "hedge_ratio": round(self.hedge_ratio, 6),
            "r_squared": round(self.r_squared, 6),
            "unit_ratio_effectiveness": round(self.unit_ratio_effectiveness, 6),
            "measured_at": self.measured_at.isoformat(),
            "window_days": self.window_days,
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> HedgeEffectiveness:
        return cls(
            spot=str(blob["spot"]), hedge=str(blob["hedge"]), phase=str(blob["phase"]),
            observations=int(blob["observations"]),
            correlation=float(blob["correlation"]),
            correlation_low=float(blob["correlation_low"]),
            hedge_ratio=float(blob["hedge_ratio"]),
            r_squared=float(blob["r_squared"]),
            unit_ratio_effectiveness=float(blob["unit_ratio_effectiveness"]),
            measured_at=datetime.fromisoformat(str(blob["measured_at"])),
            window_days=int(blob["window_days"]),
        )

    def is_fresh(self, *, now: datetime | None = None) -> bool:
        age = (now or datetime.now(UTC)) - self.measured_at
        return age.total_seconds() <= STALE_AFTER_HOURS * 3600

    @property
    def risk_reduction(self) -> Decimal:
        return _unit_decimal(self.r_squared)

    @property
    def correlation_confidence(self) -> Decimal:
        """The Fisher lower bound, floored at zero.

        A negative lower bound means the data cannot rule out the two instruments being unrelated,
        and the honest credit for that is none — not the absolute value, which would treat "we
        cannot tell" as "strongly anticorrelated, which also hedges".
        """
        return _unit_decimal(max(0.0, self.correlation_low))

    @property
    def basis_stability(self) -> Decimal:
        return _unit_decimal(self.unit_ratio_effectiveness)

    def render(self) -> str:
        return (
            f"{self.spot} hedged with {self.hedge} ({self.phase}): r={self.correlation:+.3f} "
            f"[95% low {self.correlation_low:+.3f}], optimal ratio {self.hedge_ratio:.3f} removes "
            f"{self.r_squared:.1%} of variance, one-for-one removes "
            f"{self.unit_ratio_effectiveness:.1%}, n={self.observations}"
        )


def _unit_decimal(value: float) -> Decimal:
    """Clamp into [0, 1] and quantise. The risk layer validates this range and rejects anything
    outside it, so a -0.0000001 from floating-point noise must not become an exception."""
    bounded = min(1.0, max(0.0, value))
    return Decimal(str(round(bounded, 6)))


def _changes(values: Sequence[float]) -> list[float]:
    """Log changes. Used rather than simple differences so the two legs are on a common scale when
    their price levels differ by an order of magnitude, which they do: an rToken quotes near its
    anchor but the index and the mark are not the same number."""
    out: list[float] = []
    for previous, current in pairwise(values):
        if previous <= 0 or current <= 0:
            raise EffectivenessError("prices must be positive to take log changes")
        out.append(math.log(current / previous))
    return out


def fisher_interval(r: float, n: int, *, z: float = CONFIDENCE_Z) -> tuple[float, float]:
    """Fisher z-transform confidence interval for a correlation.

    ``atanh`` is undefined at ±1, which happens on synthetic data and on a pair that has been
    perfectly collinear over a short window. Clamped just inside rather than special-cased, so the
    interval degenerates to a point instead of raising on a legitimate input.
    """
    if n <= 3:
        raise EffectivenessError("a Fisher interval needs more than three observations")
    clamped = min(1 - 1e-12, max(-1 + 1e-12, r))
    centre = math.atanh(clamped)
    spread = z / math.sqrt(n - 3)
    return math.tanh(centre - spread), math.tanh(centre + spread)


def measure(
    spot_prices: Sequence[float], hedge_prices: Sequence[float], *, spot: str, hedge: str,
    phase: str = "all", window_days: int = 0, now: datetime | None = None,
) -> HedgeEffectiveness:
    """Ederington effectiveness, the minimum-variance ratio, and a Fisher bound on the correlation.

    The two series must be already aligned on timestamp. Aligning them here would hide the join, and
    the join is the step where a stale index quietly becomes a same-instant observation — which is
    why `market/history.py:fetch_basis` does an inner join and says so.
    """
    if len(spot_prices) != len(hedge_prices):
        raise EffectivenessError(
            f"series differ in length: {len(spot_prices)} vs {len(hedge_prices)}"
        )
    spot_changes = _changes([float(v) for v in spot_prices])
    hedge_changes = _changes([float(v) for v in hedge_prices])
    n = len(spot_changes)
    if n < MIN_PAIRS:
        raise EffectivenessError(
            f"{n} change(s) is below the {MIN_PAIRS} this module will measure from"
        )

    spot_mean = sum(spot_changes) / n
    hedge_mean = sum(hedge_changes) / n
    spot_var = sum((x - spot_mean) ** 2 for x in spot_changes) / n
    hedge_var = sum((y - hedge_mean) ** 2 for y in hedge_changes) / n
    covariance = sum(
        (x - spot_mean) * (y - hedge_mean) for x, y in zip(spot_changes, hedge_changes, strict=True)
    ) / n
    if spot_var <= 0:
        raise EffectivenessError(
            "the position's own changes have no variance; there is no risk to hedge and no "
            "effectiveness to measure"
        )
    if hedge_var <= 0:
        raise EffectivenessError("the hedge instrument does not move; it cannot remove variance")

    correlation = covariance / math.sqrt(spot_var * hedge_var)
    low, _high = fisher_interval(correlation, n)
    ratio = covariance / hedge_var
    # Ederington: the variance of the optimally hedged position is Var(s) * (1 - rho^2), so the
    # share removed is rho^2. Computed from the correlation rather than by refitting, because the
    # two are the same number and one of them cannot drift from the other.
    r_squared = correlation ** 2

    residual = [x - y for x, y in zip(spot_changes, hedge_changes, strict=True)]
    residual_mean = sum(residual) / n
    residual_var = sum((v - residual_mean) ** 2 for v in residual) / n
    unit_ratio = 1.0 - residual_var / spot_var

    return HedgeEffectiveness(
        spot=spot, hedge=hedge, phase=phase, observations=n,
        correlation=correlation, correlation_low=low, hedge_ratio=ratio,
        r_squared=r_squared, unit_ratio_effectiveness=unit_ratio,
        measured_at=now or datetime.now(UTC), window_days=window_days,
    )


# --- the stored measurement the live path reads --------------------------------------------------

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "hedge_effectiveness.json"


def load(path: Path | None = None) -> dict[tuple[str, str], HedgeEffectiveness]:
    """Measurements by ``(symbol, phase)``. An unreadable file is an empty map, never a default.

    The risk layer's behaviour when this is empty is to say the factors were not measured — which
    is the state the whole module exists to make visible, so it must not be papered over by a
    fallback here.
    """
    target = path or REPORT_PATH
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[tuple[str, str], HedgeEffectiveness] = {}
    for row in blob.get("measurements", []):
        try:
            got = HedgeEffectiveness.from_dict(row)
        except (KeyError, TypeError, ValueError):
            continue
        out[(got.spot, got.phase)] = got
    return out


def lookup(
    symbol: str, phase: str, *, table: dict[tuple[str, str], HedgeEffectiveness] | None = None,
    now: datetime | None = None,
) -> HedgeEffectiveness | None:
    """The fresh measurement for this symbol and phase, or ``None``.

    Staleness is treated as absence rather than as a slightly worse number. A correlation measured
    three weeks ago is not a weaker version of today's — it is a statement about a different market.
    """
    found = (table if table is not None else load()).get((symbol, phase))
    if found is None or not found.is_fresh(now=now):
        return None
    return found


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import fetch_basis
    from argus.truth.clocks import DualClock

    parser = argparse.ArgumentParser(description="measure how well the index hedges the token")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--symbols", default="")
    args = parser.parse_args()

    wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or list(
        RTOKEN_SYMBOLS
    )
    clock = DualClock()
    rows: list[HedgeEffectiveness] = []
    for symbol in wanted:
        try:
            points = fetch_basis(symbol, days=args.days)
        except Exception as exc:
            print(f"  {symbol}: no basis history ({type(exc).__name__})")
            continue
        # Split by the session the anchor was in, never pooled: the same correlation measured while
        # the anchor trades and while it is shut is two different facts.
        buckets: dict[str, list[tuple[float, float]]] = {}
        for point in points:
            phase = clock.phase(point.ts).value
            buckets.setdefault(phase, []).append((float(point.market), float(point.index)))
            buckets.setdefault("all", []).append((float(point.market), float(point.index)))
        for phase, pairs in sorted(buckets.items()):
            if len(pairs) <= MIN_PAIRS:
                continue
            try:
                got = measure(
                    [m for m, _ in pairs], [i for _, i in pairs],
                    spot=symbol, hedge=f"{symbol.removesuffix('USDT')} index",
                    phase=phase, window_days=args.days,
                )
            except EffectivenessError as exc:
                print(f"  {symbol} {phase}: {exc}")
                continue
            rows.append(got)
            print("  " + got.render())

    if not rows:
        print("nothing measurable; no file written")
        return 1
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "window_days": args.days,
                "measurements": [r.as_dict() for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(rows)} measurement(s) written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CONFIDENCE_Z",
    "MIN_PAIRS",
    "REPORT_PATH",
    "STALE_AFTER_HOURS",
    "EffectivenessError",
    "HedgeEffectiveness",
    "Provenance",
    "fisher_interval",
    "load",
    "lookup",
    "measure",
]
