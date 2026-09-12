"""Track 1 study — every session strategy, every rToken, net of fees, with the gate applied.

Produces the material Track 1 requires: runnable strategy code, a backtest over >= 60 days with a
chronological out-of-sample slice >= 30 days, and the metrics the track is scored on.

**A methodological point this module handles explicitly.** The Deflated Sharpe Ratio deflates by
the dispersion of trial Sharpes. Our variant set deliberately includes `closure_momentum`, a rule
the gap study already falsified, as a control. It scores about -8.9, which inflates the trial
variance enough to make the gate unpassable for *any* winner.

That is not a reason to drop the control — a sweep containing only plausible winners is a rigged
sweep. It is a reason to report **two deflations**:

* ``dsr_all_trials`` — every variant, controls included. The strictest reading.
* ``dsr_candidate_trials`` — only variants that were genuine deployment candidates, which is the
  population the DSR was designed for.

Both are reported. Neither is hidden, and if they disagree the disagreement is the finding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar, buy_and_hold, run, sweep
from argus.backtest.metrics import HOURLY_PER_YEAR, MetricError, deflated_sharpe
from argus.cost.model import CostModel
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range
from argus.strategies.session_alpha import VARIANTS
from argus.strategies.track1_suite import SUBTHEME_OF, TRACK1_VARIANTS

# Variants that were genuine deployment candidates. `closure_momentum` and `closure_reversion`
# are controls: the gap study measured 50.6% overnight continuation and a weekend effect that
# failed its train/test split, so neither was ever a candidate.
# Genuine deployment candidates. The two closure-direction rules are controls: the gap study
# measured 50.6% overnight continuation and a weekend effect that failed its train/test split,
# so neither was ever a candidate. Keeping them in the sweep but out of the candidate set is
# what lets both deflations be reported honestly.
CONTROLS = ("closure_momentum", "closure_reversion")
CANDIDATES = tuple(
    k for k in {**VARIANTS, **TRACK1_VARIANTS} if k not in CONTROLS
)

# Every Track-1 sub-theme, in one sweep, scored under one gate.
ALL_VARIANTS = {**VARIANTS, **TRACK1_VARIANTS}


@dataclass(frozen=True, slots=True)
class SymbolResult:
    symbol: str
    bars: int
    days: float
    baseline_sharpe: float
    best_variant: str
    best_net_sharpe: float
    best_net_return_pct: float
    beats_baseline: bool
    dsr_all: float
    dsr_candidates: float
    oos: dict[str, Any] | None


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - 1)


def study(symbols: tuple[str, ...] = RTOKEN_SYMBOLS, *, days: int = 90) -> dict[str, Any]:
    cost = CostModel.bitget_perp()
    results: list[SymbolResult] = []
    failures: dict[str, str] = {}

    for symbol in symbols:
        try:
            candles = fetch_range(
                symbol, days=days, interval="1H", candle_type=CandleType.MARKET
            )
        except Exception as exc:
            failures[symbol] = str(exc)[:120]
            continue

        try:
            index = {
                c.ts: c.close
                for c in fetch_range(
                    symbol, days=days, interval="1H", candle_type=CandleType.INDEX
                )
            }
        except Exception:
            index = {}
        bars = [
            Bar(ts=c.ts, close=c.close,
                extra={"index": index[c.ts]} if c.ts in index else None)
            for c in candles
        ]
        if len(bars) < 200:
            failures[symbol] = f"only {len(bars)} bars"
            continue

        try:
            baseline = run(
                "buy_and_hold", symbol, bars, buy_and_hold,
                cost=cost, periods_per_year=HOURLY_PER_YEAR,
            )
            swept = sweep(
                "track1", symbol, bars, ALL_VARIANTS,
                cost=cost, periods_per_year=HOURLY_PER_YEAR,
            )
        except MetricError as exc:
            failures[symbol] = str(exc)[:120]
            continue

        best_label = swept["best"]
        if not best_label:
            failures[symbol] = "no variant scored"
            continue

        all_sharpes = list(swept["all_net_sharpes"].values())
        cand_sharpes = [
            v for k, v in swept["all_net_sharpes"].items() if k in CANDIDATES
        ]
        best = swept["best_result"]
        best_sharpe = best["net"]["sharpe"]

        def _dsr(
            sharpes: list[float], *, observed: float = best_sharpe, n: int = len(bars)
        ) -> float:
            # Loop variables bound as defaults: a closure over them would silently use the
            # last symbol's values for every earlier symbol.
            try:
                return round(deflated_sharpe(
                    observed, n=n, trials=max(1, len(sharpes)),
                    variance_of_trials=_variance(sharpes),
                ), 4)
            except MetricError:
                return 0.0

        results.append(SymbolResult(
            symbol=symbol,
            bars=len(bars),
            days=round((bars[-1].ts - bars[0].ts).total_seconds() / 86400, 1),
            baseline_sharpe=round(baseline.net.sharpe, 3),
            best_variant=best_label.split(":")[-1],
            best_net_sharpe=round(best_sharpe, 3),
            best_net_return_pct=best["net"]["total_return_pct"],
            beats_baseline=best_sharpe > baseline.net.sharpe,
            dsr_all=_dsr(all_sharpes),
            dsr_candidates=_dsr(cand_sharpes),
            oos=best.get("decay"),
        ))

    beat = [r for r in results if r.beats_baseline]
    survived_all = [r for r in results if r.dsr_all > 0.95]
    survived_cand = [r for r in results if r.dsr_candidates > 0.95]
    variant_wins: dict[str, int] = {}
    subtheme_wins: dict[str, int] = {}
    for r in results:
        variant_wins[r.best_variant] = variant_wins.get(r.best_variant, 0) + 1
        theme = SUBTHEME_OF.get(r.best_variant, "t1-afopen (session_alpha)")
        subtheme_wins[theme] = subtheme_wins.get(theme, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": days,
        "round_trip_fee_bps": str(cost.round_trip_bps()),
        "symbols_tested": len(results),
        "symbols_failed": failures,
        "headline": {
            "beat_buy_and_hold_sharpe": f"{len(beat)}/{len(results)}",
            "survived_dsr_all_trials": f"{len(survived_all)}/{len(results)}",
            "survived_dsr_candidates_only": f"{len(survived_cand)}/{len(results)}",
            "most_frequent_winner": max(variant_wins, key=lambda k: variant_wins[k])
            if variant_wins else None,
            "variant_win_counts": variant_wins,
            "subtheme_win_counts": subtheme_wins,
        },
        "per_symbol": [
            {
                "symbol": r.symbol, "bars": r.bars, "days": r.days,
                "baseline_sharpe": r.baseline_sharpe,
                "best_variant": r.best_variant,
                "best_net_sharpe": r.best_net_sharpe,
                "best_net_return_pct": r.best_net_return_pct,
                "beats_baseline": r.beats_baseline,
                "dsr_all_trials": r.dsr_all,
                "dsr_candidates_only": r.dsr_candidates,
                "out_of_sample_decay": r.oos,
            }
            for r in results
        ],
    }


def main() -> int:
    result = study()
    out = Path(__file__).resolve().parents[3] / "data" / "track1_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    h = result["headline"]
    print(json.dumps({"headline": h, "symbols": result["symbols_tested"]}, indent=2))
    print(f"\n{'symbol':<12}{'base':>7}{'best':>8}{'sharpe':>9}{'ret%':>8}"
          f"{'DSR-all':>9}{'DSR-cand':>10}")
    for r in result["per_symbol"]:
        print(f"{r['symbol']:<12}{r['baseline_sharpe']:>7}{r['best_variant'][:7]:>8}"
              f"{r['best_net_sharpe']:>9}{r['best_net_return_pct']:>8}"
              f"{r['dsr_all_trials']:>9}{r['dsr_candidates_only']:>10}")
    print(f"\nfull report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
