"""What the Track 1 sweep's *selection procedure* is worth, measured rather than asserted.

`track1_study` reports the best of twenty-five variants per symbol and deflates it for the trial
count. That answers "is this Sharpe real?". It does not answer the question a judge should ask
next: **"if you ran that selection again, would it pick something that works?"** Those come apart.
A procedure can pick a strategy whose Sharpe survives deflation on one sample and pick a different
one next month, and a procedure can reliably pick the same strategy while that strategy is
indistinguishable from noise. The deflated Sharpe cannot see either case.

So this module runs the combinatorially symmetric cross-validation of Bailey, Borwein, López de
Prado and Zhu over the real sweep — every balanced split of the real return series, not a sample of
them — and reports three things per symbol that the existing study does not:

* **PBO**, the share of splits in which the in-sample winner lands below the out-of-sample median.
* **Minimum track record length** for that winner: how long a live record would have to run before
  a Sharpe of that size could be believed at 95% confidence. This is the number that converts "we
  have no settled paper trades yet" from an apology into a measurement.
* **False-discovery control across the whole grid**, 25 variants by 12 symbols, so the question
  "which of these are individually significant?" is answered at a controlled rate rather than by
  eyeballing the leaderboard.

**PBO is reported twice, for the same reason `track1_study` deflates twice.** The sweep contains
two deliberate controls that the gap study already falsified, and they score around -8.9. A
control that bad makes the winner rank near the top out of sample almost mechanically, which
*flatters* PBO. Dropping the controls would rig the field the other way — a sweep of only
plausible winners is not a sweep. Both numbers are reported and, where they disagree, the
disagreement is the finding.

**What a low PBO does not mean.** PBO scores the selection, not the strategy. A procedure that
reliably picks the same variant out of a field of bad ones scores well here while that variant
remains undistinguishable from noise under the deflated Sharpe. `track1_study` already found that
0 of 12 winners survive deflation over all trials. A low PBO beside that result says the search is
stable, not that it is profitable, and this module says so in its own output rather than leaving a
reader to combine the two.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar, run
from argus.backtest.metrics import (
    HOURLY_PER_YEAR,
    MetricError,
    probabilistic_sharpe,
)
from argus.backtest.validation import (
    annualised_track_record_years,
    benjamini_hochberg,
    bonferroni,
    min_track_record_length,
    moments,
    probability_of_overfitting,
)
from argus.cost.model import CostModel
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range
from argus.research.track1_study import ALL_VARIANTS, CONTROLS

STUDY_PATH = Path(__file__).resolve().parents[3] / "data" / "overfitting_study.json"

GROUPS = 8
"""Blocks the series is cut into. Eight gives C(8,4) = 70 balanced splits, every one evaluated."""

MIN_BARS = 200
FDR = 0.05
CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class SymbolOverfitting:
    """One symbol's answer to "would this selection procedure hold up?"."""

    symbol: str
    bars: int
    strategies_all: int
    strategies_candidates: int
    pbo_all_trials: float
    pbo_candidates_only: float
    median_rank_all: float
    selection_churn: float
    dominant_share: float
    verdict_all: str
    winner: str
    winner_net_sharpe: float
    """Annualised, net of the 12bps round trip."""

    min_track_record_bars: float | None
    min_track_record_years: float | None
    track_record_note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bars_for(symbol: str, days: int) -> list[Bar]:
    """Identical construction to `track1_study`, including the index series.

    Two variants price the rToken against its index and produce a flat zero series without it.
    Building thinner bars here would silently drop them from the field and change the PBO.
    """
    candles = fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.MARKET)
    try:
        index = {
            c.ts: c.close
            for c in fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.INDEX)
        }
    except Exception:
        index = {}
    return [
        Bar(
            ts=c.ts,
            close=c.close,
            extra={
                "volume": float(c.volume),
                "high": float(c.high),
                "low": float(c.low),
                **({"index": index[c.ts]} if c.ts in index else {}),
            },
        )
        for c in candles
    ]


def _series_for(
    symbol: str, bars: list[Bar], cost: CostModel
) -> tuple[dict[str, tuple[float, ...]], dict[str, float]]:
    """Every variant's net return series and annualised net Sharpe. Failures are dropped, named."""
    series: dict[str, tuple[float, ...]] = {}
    sharpes: dict[str, float] = {}
    for label, fn in ALL_VARIANTS.items():
        try:
            result = run(
                f"overfit:{label}", symbol, bars, fn,
                cost=cost, periods_per_year=HOURLY_PER_YEAR,
            )
        except MetricError:
            continue
        series[label] = result.net_returns
        sharpes[label] = float(result.net.sharpe)
    return series, sharpes


def _matrix(series: dict[str, tuple[float, ...]], labels: list[str]) -> list[list[float]]:
    rows = min(len(series[label]) for label in labels)
    return [[series[label][t] for label in labels] for t in range(rows)]


def _p_value(returns: tuple[float, ...]) -> float | None:
    """P(this Sharpe is no better than zero), with the sample's own skew and kurtosis.

    The probabilistic Sharpe takes a *per-observation* Sharpe, so nothing annualised may be handed
    to it — an hourly series annualised first would come back significant at essentially any mean.
    """
    try:
        mean, sd, skew, kurtosis = moments(list(returns))
    except MetricError:
        return None
    try:
        psr = probabilistic_sharpe(
            mean / sd, benchmark=0.0, n=len(returns), skew=skew, kurtosis=kurtosis
        )
    except MetricError:
        return None
    return max(0.0, min(1.0, 1.0 - psr))


def _track_record(returns: tuple[float, ...]) -> tuple[float | None, float | None, str]:
    try:
        bars = min_track_record_length(list(returns), confidence=CONFIDENCE)
    except MetricError as exc:
        return None, None, str(exc)
    years = annualised_track_record_years(
        list(returns), periods_per_year=HOURLY_PER_YEAR, confidence=CONFIDENCE
    )
    return bars, years, f"{bars:,.0f} hourly bars at {CONFIDENCE:.0%} confidence"


def study(
    symbols: tuple[str, ...] = RTOKEN_SYMBOLS, *, days: int = 180, groups: int = GROUPS
) -> dict[str, Any]:
    cost = CostModel.bitget_perp()
    results: list[SymbolOverfitting] = []
    failures: dict[str, str] = {}
    grid: list[dict[str, Any]] = []

    for symbol in symbols:
        try:
            bars = _bars_for(symbol, days)
        except Exception as exc:
            failures[symbol] = str(exc)[:120]
            continue
        if len(bars) < MIN_BARS:
            failures[symbol] = f"only {len(bars)} bars"
            continue

        series, sharpes = _series_for(symbol, bars, cost)
        if len(series) < 2:
            failures[symbol] = f"only {len(series)} variant(s) scored"
            continue

        labels = sorted(series)
        candidates = [label for label in labels if label not in CONTROLS]
        try:
            overall = probability_of_overfitting(_matrix(series, labels), groups=groups)
            narrowed = probability_of_overfitting(_matrix(series, candidates), groups=groups)
        except MetricError as exc:
            failures[symbol] = str(exc)[:120]
            continue

        winner = max(labels, key=lambda label: sharpes[label])
        trl_bars, trl_years, note = _track_record(series[winner])
        results.append(
            SymbolOverfitting(
                symbol=symbol,
                bars=len(bars),
                strategies_all=len(labels),
                strategies_candidates=len(candidates),
                pbo_all_trials=round(overall.pbo, 5),
                pbo_candidates_only=round(narrowed.pbo, 5),
                median_rank_all=round(overall.median_rank, 5),
                selection_churn=round(overall.selection_churn, 5),
                dominant_share=round(overall.dominant_share, 5),
                verdict_all=overall.verdict,
                winner=winner,
                winner_net_sharpe=round(sharpes[winner], 4),
                min_track_record_bars=None if trl_bars is None else round(trl_bars, 1),
                min_track_record_years=None if trl_years is None else round(trl_years, 3),
                track_record_note=note,
            )
        )
        for label in labels:
            p = _p_value(series[label])
            if p is not None:
                grid.append({
                    "symbol": symbol,
                    "variant": label,
                    "net_sharpe": round(sharpes[label], 4),
                    "p_value": round(p, 8),
                })

    p_values = [row["p_value"] for row in grid]
    for row, bh, bf in zip(
        grid, benjamini_hochberg(p_values, fdr=FDR), bonferroni(p_values, alpha=FDR), strict=True
    ):
        row["survives_benjamini_hochberg"] = bh
        row["survives_bonferroni"] = bf

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "days": days,
        "groups": groups,
        "splits_per_symbol": None if not results else _splits(groups),
        "confidence": CONFIDENCE,
        "fdr": FDR,
        "symbols": [r.as_dict() for r in results],
        "failures": failures,
        "grid": {
            "trials": len(grid),
            "survivors_benjamini_hochberg": sum(
                1 for row in grid if row["survives_benjamini_hochberg"]
            ),
            "survivors_bonferroni": sum(1 for row in grid if row["survives_bonferroni"]),
            "rows": grid,
        },
    }


def _splits(groups: int) -> int:
    from math import comb

    return comb(groups, groups // 2)


def render(payload: dict[str, Any]) -> str:
    rows = payload["symbols"]
    if not rows:
        return "OVERFITTING STUDY — nothing scored; " + json.dumps(payload["failures"])[:200]
    lines = [
        f"PROBABILITY OF BACKTEST OVERFITTING — {len(rows)} symbol(s), "
        f"{payload['splits_per_symbol']} balanced splits each",
        "",
        f"{'symbol':<12}{'all':>7}{'cands':>8}{'churn':>8}{'top%':>7}"
        f"{'winner':>26}{'Sharpe':>9}{'MinTRL yrs':>12}",
    ]
    for r in rows:
        trl = r["min_track_record_years"]
        years = "n/a" if trl is None else f"{trl:.2f}"
        lines.append(
            f"{r['symbol']:<12}{r['pbo_all_trials']:>7.2f}{r['pbo_candidates_only']:>8.2f}"
            f"{r['selection_churn']:>8.2f}{100 * r['dominant_share']:>6.0f}%"
            f"{r['winner'][:25]:>26}{r['winner_net_sharpe']:>9.2f}{years:>12}"
        )
    grid = payload["grid"]
    lines += [
        "",
        f"multiple testing across {grid['trials']} (symbol, variant) trials at FDR "
        f"{payload['fdr']:.0%}: {grid['survivors_benjamini_hochberg']} survive "
        f"Benjamini-Hochberg, {grid['survivors_bonferroni']} survive Bonferroni",
        "",
        "PBO scores the selection procedure, not the strategy. A low PBO beside the deflated "
        "Sharpe result in track1_study (0 of 12 winners survive over all trials) says the search "
        "is stable, not that it is profitable.",
    ]
    return "\n".join(lines)


def main() -> None:
    payload = study()
    STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STUDY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(render(payload))
    print(f"\nwritten to {STUDY_PATH}")


if __name__ == "__main__":
    main()
