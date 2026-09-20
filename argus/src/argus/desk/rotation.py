"""Cross-asset breadth-momentum rotation — switching between risk and safety by asset class.

`desk/allocation.py` answers "how should risk be split *within* the book we already hold" — twelve
tokenized US equities, all one asset class, all correlated above 0.9. It has no opinion on the
question the handbook's own "Cross-Asset Allocation / Rotation" sub-theme actually names: risk-on/
off rotation between **US stocks, Crypto and commodities** — three genuinely different asset
classes, not twelve correlated slices of one. This module answers that question, on real Bitget
instruments: the twelve rTokens (`market/bitget.RTOKEN_SYMBOLS`), real crypto majors (BTCUSDT,
ETHUSDT — confirmed live on the same public futures book, same endpoint, same `fetch_tickers()`),
and a real commodity token (XAUUSDT, gold — confirmed live, price sanity-checked against spot gold
at ~$4,290/oz on 2026-09-15; `FOXAUSDT` was checked and excluded, at $67 with $299 of 24h volume it
is not a gold-tracking instrument despite the ticker, the same class of trap `RTOKEN_SYMBOLS`'s own
comment already documents for SPXUSDT).

**Read before written.** The reference is pytaa (MIT),
`research/repos-t3/pytaa/src/pytaa/backtest/positions.py:154-190` (function `vigilant_allocation`,
vendored verbatim at `eval/baselines/pytaa_vigilant_allocation.py`) and
`.../strategy/signals.py:39-49` (`Signal.momentum_score`) — the real, published Vigilant Asset
Allocation rule (Keller & Keuning 2017, "Breadth Momentum and the Canary Universe: Defensive Asset
Allocation (DAA)", SSRN 2543979). :func:`momentum_score` reproduces its weighted four-horizon
formula exactly (weights 12/4/2/1 at lag ratios 1/3/6/12 months, normalised by subtracting the
weight sum of 19). :func:`breadth_allocation` reproduces its discrete safe/risk switching rule
exactly: count how many of the risk-and-safe universe carry a negative score, scale the highest-
ranked safe asset's weight by ``step`` per negative count, and split the rest equally across the
top-``top_k`` risk assets by rank. Verified to reproduce the real vendored function's numeric output
on identical input — `eval/rotation_comparison.py`'s ``run_baseline_reproduced_cases``.

**Two things ARGUS's src/ cannot do that the reference does, and one thing it deliberately refuses
to do that the reference does not.** ARGUS's `src/` carries no pandas/numpy dependency (project
convention — pydantic and python-dateutil only), so both formulas above are reimplemented in plain
Python here rather than imported; :func:`monthly_closes_from_bars` replaces pandas'
``resample("BME").last()`` with the equivalent plain-Python group-by-month-take-last. And where
the reference is silent, this module is loud: **a momentum score that cannot be computed for
every asset in play raises :class:`RotationError` rather than proposing a partial allocation.**
That difference is not cosmetic. Run on ARGUS's own real, live candle history
(`eval/rotation_comparison.py`,
2026-09-15), gold's real `momentum_score()` is NaN on every one of the 14 monthly rebalance points
computable from its available 277 real daily bars — nine months of listing history, short of the
thirteen a twelve-month lookback needs. Fed that NaN, the real vendored `vigilant_allocation` does
not refuse: `NaN < 0` is `False` in numpy, so a data-starved asset is silently counted as
"not distressed" in the breadth tally, and `.rank()` on a NaN-valued safe-asset series never equals
rank 1, so the safe asset assigned the flight-to-safety weight receives **zero** regardless of what
the breadth count says. The comparison's own measured case: three of three risk assets negative
(the sharpest "flee to safety" signal the rule can raise) with gold's score NaN — the real vendored
function allocates a grand total of **25% of the book** and silently leaves the other 75% nowhere,
worst exactly when the signal is loudest. `breadth_allocation` here raises instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from argus.desk.allocation import TAKER_BPS, Trade

MOMENTUM_HORIZONS: tuple[tuple[int, int], ...] = ((12, 1), (4, 3), (2, 6), (1, 12))
"""(horizon_weight, lag_months) pairs, exactly pytaa's real `Signal.momentum_score`
(`strategy/signals.py:39-49`): ``lag = 12 // horizon_weight`` for horizon in (12, 4, 2, 1)."""

MOMENTUM_WEIGHT_SUM = sum(w for w, _ in MOMENTUM_HORIZONS)
"""19 — the real formula's own normalising constant (`score - 19`, `signals.py:48`): at zero
momentum on every horizon every ratio is 1, so the raw sum is exactly this and the normalised score
is exactly zero."""

MIN_MONTHLY_OBSERVATIONS = max(lag for _, lag in MOMENTUM_HORIZONS) + 1
"""13 — one more than the longest lag. `closes[-1 - lag]` needs at least ``lag + 1`` points; below
that the real formula's own longest-horizon term is unconditionally undefined, and NaN propagates
through the summed score in the reference. Verified on ARGUS's real data: XAUUSDT's 277 real daily
bars resample to fewer than 13 real monthly closes, and its real `momentum_score()` is NaN at all
14 of 14 computable rebalance points as of 2026-09-15 (`eval/rotation_comparison.py`)."""

TRADING_DAYS_PER_MONTH = 21
"""pytaa's own stated business-month convention (`strategy/signals.py:67`, the `days` default of
`sma_crossover`) — used only to size how many real daily bars a `--days` CLI request should fetch
to reach `MIN_MONTHLY_OBSERVATIONS` months, never to annualise anything."""


class RotationError(ValueError):
    """Raised rather than proposing a rotation from a signal that could not be computed for every
    asset the rotation would touch. See the module docstring for the real, measured failure mode
    in the reference this refuses to reproduce."""


# --- signal ----------------------------------------------------------------------------------


def momentum_score(monthly_closes: Sequence[float]) -> float:
    """The real VAA weighted momentum score (`signals.py:39-49`), reproduced in plain Python.

    ``monthly_closes`` must be chronological, oldest first, real month-end closes (see
    :func:`monthly_closes_from_bars`). Raises :class:`RotationError` — never returns NaN — when
    there are fewer than :data:`MIN_MONTHLY_OBSERVATIONS` points, which is the module's one
    deliberate departure from the reference (see the module docstring).
    """
    if len(monthly_closes) < MIN_MONTHLY_OBSERVATIONS:
        raise RotationError(
            f"{len(monthly_closes)} monthly close(s) is below the "
            f"{MIN_MONTHLY_OBSERVATIONS} a {max(lag for _, lag in MOMENTUM_HORIZONS)}-month "
            f"lookback needs"
        )
    last = monthly_closes[-1]
    raw = 0.0
    for weight, lag in MOMENTUM_HORIZONS:
        anchor = monthly_closes[-1 - lag]
        if anchor == 0:
            raise RotationError("a zero monthly close cannot anchor a momentum ratio")
        raw += weight * (last / anchor)
    return raw - MOMENTUM_WEIGHT_SUM


def monthly_closes_from_bars(bars: Sequence[tuple[datetime, float]]) -> list[float]:
    """Plain-Python equivalent of pytaa's ``prices.resample("BME").last()``: the last real close
    recorded in each calendar month, in chronological order. ``bars`` must already be chronological
    (oldest first); a later bar in the same month overwrites an earlier one, exactly matching
    ``.last()``.
    """
    by_month: dict[tuple[int, int], float] = {}
    for ts, close in bars:
        by_month[(ts.year, ts.month)] = close
    return [by_month[key] for key in sorted(by_month)]


# --- breadth rule ------------------------------------------------------------------------------


def breadth_allocation(
    scores: Mapping[str, float],
    risk_assets: Sequence[str],
    safe_assets: Sequence[str],
    *,
    top_k: int = 2,
    step: float = 0.25,
) -> dict[str, float]:
    """The real VAA breadth rule (`vigilant_allocation`, `positions.py:154-190`), reproduced in
    plain Python, with one addition: every asset in ``risk_assets``/``safe_assets`` must have a
    real score in ``scores`` or this raises :class:`RotationError`. The reference does not — see
    the module docstring for the measured cost of that silence.

    Matches the reference exactly otherwise: ``is_neg`` counts negative scores across *both*
    universes; the single highest-ranked safe asset receives ``min(1, step * is_neg)``; the
    remaining weight splits equally across the top-``top_k`` risk assets by rank.
    """
    universe = (*risk_assets, *safe_assets)
    missing = [
        a for a in universe
        if a not in scores or scores[a] is None or math.isnan(scores[a])
    ]
    if missing:
        raise RotationError(
            f"cannot allocate: no usable momentum score for {', '.join(missing)} "
            f"(insufficient history — see MIN_MONTHLY_OBSERVATIONS)"
        )
    if not safe_assets:
        raise RotationError("breadth allocation needs at least one safe asset")
    if not risk_assets:
        raise RotationError("breadth allocation needs at least one risk asset")

    is_neg = sum(1 for a in universe if scores[a] < 0)
    safe_weight_total = min(1.0, step * is_neg)

    weights = dict.fromkeys(universe, 0.0)
    ranked_safe = sorted(safe_assets, key=lambda a: -scores[a])
    weights[ranked_safe[0]] = safe_weight_total

    ranked_risk = sorted(risk_assets, key=lambda a: -scores[a])
    risk_weight_each = (1.0 - safe_weight_total) / top_k if top_k > 0 else 0.0
    for a in ranked_risk[:top_k]:
        weights[a] = risk_weight_each
    return weights


# --- pricing -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RotationPlan:
    """The move from the book you hold to the breadth rule's target, priced. Mirrors
    `allocation.TradePlan`'s shape deliberately — both answer "is this rebalance worth its own
    cost", just on a different partition of the book (risk-parity-within-a-class vs.
    risk-on/off-across-classes)."""

    trades: tuple[Trade, ...]
    is_neg: int
    universe_size: int
    turnover: float
    cost_bps: float

    @property
    def regime(self) -> str:
        if self.is_neg == 0:
            return "fully risk-on — no asset in the universe carries a negative score"
        if self.is_neg >= self.universe_size:
            return "fully risk-off — every asset in the universe carries a negative score"
        return f"partial risk-off — {self.is_neg} of {self.universe_size} assets negative"

    @property
    def verdict(self) -> str:
        if not self.trades:
            return f"{self.regime}. The book is already at the breadth-rule allocation; no trade"
        moved = ", ".join(
            f"{t.symbol} {t.weight_before:.1%}->{t.weight_after:.1%}" for t in self.trades[:4]
        )
        return (
            f"{self.regime}. {len(self.trades)} leg(s) ({moved}"
            f"{'...' if len(self.trades) > 4 else ''}), turnover {self.turnover:.1%} costing "
            f"{self.cost_bps:.1f}bps"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades": [t.as_dict() for t in self.trades],
            "is_neg": self.is_neg,
            "universe_size": self.universe_size,
            "regime": self.regime,
            "turnover": round(self.turnover, 6),
            "cost_bps": round(self.cost_bps, 4),
            "verdict": self.verdict,
        }


def propose_rotation(
    current: Mapping[str, float],
    monthly_closes: Mapping[str, Sequence[float]],
    risk_assets: Sequence[str],
    safe_assets: Sequence[str],
    *,
    taker_bps: float = TAKER_BPS,
    top_k: int = 2,
    step: float = 0.25,
    min_leg: float = 0.01,
) -> RotationPlan:
    """The breadth-rule rotation from ``current`` toward the target, priced against real cost.

    Raises :class:`RotationError` (via :func:`momentum_score` or :func:`breadth_allocation`) for
    any asset without enough real history — never proposes a partial rotation, unlike the reference.
    """
    universe = (*risk_assets, *safe_assets)
    missing_series = [a for a in universe if a not in monthly_closes]
    if missing_series:
        raise RotationError(f"no monthly close series supplied for {', '.join(missing_series)}")

    scores = {a: momentum_score(monthly_closes[a]) for a in universe}
    target = breadth_allocation(scores, risk_assets, safe_assets, top_k=top_k, step=step)

    trades = tuple(
        Trade(symbol=a, weight_before=current.get(a, 0.0), weight_after=target[a])
        for a in universe
        if abs(target[a] - current.get(a, 0.0)) >= min_leg
    )
    turnover = sum(abs(t.delta) for t in trades)
    is_neg = sum(1 for a in universe if scores[a] < 0)
    return RotationPlan(
        trades=trades, is_neg=is_neg, universe_size=len(universe),
        turnover=turnover, cost_bps=turnover * taker_bps,
    )


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import json
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range

    parser = argparse.ArgumentParser(description="risk-on/off across US stocks, crypto, gold")
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--risk", default="NVDAUSDT,AAPLUSDT,QQQUSDT,BTCUSDT,ETHUSDT")
    parser.add_argument("--safe", default="XAUUSDT")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    risk_assets = tuple(args.risk.split(","))
    safe_assets = tuple(args.safe.split(","))
    for symbol in (*risk_assets, *safe_assets):
        if symbol not in RTOKEN_SYMBOLS and not symbol.endswith("USDT"):
            print(f"  {symbol}: not a recognised USDT-margined symbol")
            return 1

    monthly: dict[str, list[float]] = {}
    for symbol in (*risk_assets, *safe_assets):
        try:
            bars = fetch_range(symbol, days=args.days, interval="1D",
                                candle_type=CandleType.MARKET)
        except Exception as exc:
            print(f"  {symbol}: no history ({type(exc).__name__})")
            continue
        monthly[symbol] = monthly_closes_from_bars([(b.ts, float(b.close)) for b in bars])

    book = dict.fromkeys((*risk_assets, *safe_assets), 1.0 / (len(risk_assets) + len(safe_assets)))
    try:
        plan = propose_rotation(book, monthly, risk_assets, safe_assets, top_k=args.top_k)
    except RotationError as exc:
        print(f"REFUSED: {exc}")
        return 1

    print(f"CROSS-ASSET ROTATION — {len(risk_assets)} risk, {len(safe_assets)} safe\n")
    for trade in plan.trades:
        print(f"  {trade.symbol:12} {trade.weight_before:7.2%} -> {trade.weight_after:7.2%}")
    print(f"\n  {plan.verdict}")
    out = Path(__file__).resolve().parents[3] / "data" / "rotation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "MIN_MONTHLY_OBSERVATIONS",
    "MOMENTUM_HORIZONS",
    "MOMENTUM_WEIGHT_SUM",
    "TRADING_DAYS_PER_MONTH",
    "RotationError",
    "RotationPlan",
    "breadth_allocation",
    "momentum_score",
    "monthly_closes_from_bars",
    "propose_rotation",
]
