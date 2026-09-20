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

**A second, sharper multiple-comparisons problem was found in this module on 2026-09-15, after it
had been shipping for a session without it.** `_dsr` corrects for the variants tried on ONE symbol
(``trials=len(sharpes)``) and nothing else — it does not correct for having run this same recipe
across **12 symbols** and reporting whichever one happened to clear 0.95, a second, uncorrected
layer of search. On the live data this was not theoretical: TQQQUSDT/``weekend_only`` cleared
``dsr_candidates_only`` at exactly 1.0, making the headline read *"1/12 survive."*
`research/overfitting_study.py` tests the identical claim by the standard, correction-complete
method — full combinatorial cross-validation (PBO) per symbol, plus Benjamini-Hochberg/Bonferroni
across the whole 25-variant x 12-symbol grid — and found **that exact pair the single most overfit
result in the study** (PBO 0.771, verdict *"the selection procedure is picking noise... what
choosing at random would do"*), with **0 of 300 grid trials surviving FDR or Bonferroni
correction.**

Neither module's own trial count was wrong for what it corrects for. The DSR here was simply being
read as a complete answer to a question only the PBO study actually answers. So every DSR survivor
this module reports now carries the sibling study's verdict attached (:func:`_pbo_cross_check`),
rather than leaving a reader to open both files and notice the disagreement themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar, buy_and_hold, run, sweep
from argus.backtest.metrics import HOURLY_PER_YEAR, MetricError, deflated_sharpe
from argus.cost.model import CostModel
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range
from argus.research.grammar import EXPANDED, as_signal_fn
from argus.strategies.session_alpha import VARIANTS
from argus.strategies.track1_suite import SUBTHEME_OF, TRACK1_VARIANTS

OVERFIT_STUDY_PATH = Path(__file__).resolve().parents[3] / "data" / "overfitting_study.json"
"""Where `overfitting_study.py` writes. Read here, never re-derived — see `_pbo_cross_check`."""

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
# The nine factors the expanded grammar made expressible, swept under the same gates as everything
# else. Including them raises the trial count, which *deflates* every Sharpe in the study — that is
# the honest direction. A search that widens its space and does not pay for the extra trials is
# reporting the maximum of a larger noise distribution and calling it discovery.
_EXPANDED_FNS = {name: as_signal_fn(sig) for name, sig in EXPANDED.items()}

ALL_VARIANTS = {**VARIANTS, **TRACK1_VARIANTS, **_EXPANDED_FNS}


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

    net: dict[str, Any] = field(default_factory=dict)
    """The winner's full net Performance — Sortino, max drawdown, turnover and win rate.

    Carried because the handbook scores Track 1 on six numbers and this study reported two. The
    others were being computed on every run and thrown away at the last step."""

    stability: dict[str, Any] | None = None
    """Rolling-window Sharpe stability, the handbook's sixth criterion. ``None`` when the series
    is too short for even one window, which is printed rather than defaulted."""

    cost_sweep: dict[str, float] = field(default_factory=dict)
    """The winner's net Sharpe at each fee level in :data:`COST_SWEEP_BPS`.

    A strategy tested at one fee is a strategy tested at one assumption. This desk's fee is 12bps
    round trip and that number is not a law of nature: it changes with tier, with venue, and with
    whether the fill is maker or taker. A result that is positive at 12 and negative at 20 is not
    an edge, it is a rebate.

    Added after the competitive sweep found `tianzeteam/stillwater-alpha` reporting exactly this —
    an out-of-sample Sharpe swept across 10, 20 and 30 basis points — and reporting it as the
    reason to believe their result. They are right that it is the better test."""

    survives_cost_sweep: bool = False
    """True only when the winner's net Sharpe stays positive at EVERY fee level tested.

    The conjunction, not the average. A mean across fee levels would let a large positive at the
    cheapest tier carry a negative at the realistic one, which is the arithmetic that makes a
    fee-sensitive strategy look robust."""


def _pbo_cross_check(symbol: str, path: Path = OVERFIT_STUDY_PATH) -> dict[str, Any] | None:
    """The sibling PBO/FDR study's verdict for one symbol — read, not re-derived.

    ``None`` means the cross-check is unavailable (the sibling study has not been run, or does not
    cover this symbol), and is reported as exactly that rather than folded into a false agreement.
    A DSR survivor with no PBO cross-check is unverified against this specific risk, and the
    headline says so instead of staying silent about the gap.
    """
    if not path.exists():
        return None
    try:
        study = json.loads(path.read_text(encoding="utf-8"))
        row = next(r for r in study["symbols"] if r["symbol"] == symbol)
    except (json.JSONDecodeError, KeyError, StopIteration):
        return None
    return {
        "pbo_all_trials": row["pbo_all_trials"],
        "pbo_verdict": row["verdict_all"],
        "picking_noise": "picking noise" in row["verdict_all"],
        "grid_survivors_benjamini_hochberg": study["grid"]["survivors_benjamini_hochberg"],
        "grid_survivors_bonferroni": study["grid"]["survivors_bonferroni"],
        "grid_trials": study["grid"]["trials"],
    }


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - 1)


COST_SWEEP_BPS: tuple[int, ...] = (6, 12, 20, 30, 40)
"""Round-trip fee levels every winner is re-scored at, in basis points.

Six is the best case anyone reaches on this venue (a maker fill both ways at the top tier), twelve
is what this desk actually pays and models everywhere else, and twenty through forty cover a worse
tier, a wider spread, or slippage the cost model does not see. A strategy that only works at six is
a strategy that works for somebody else.
"""


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
        # Volume, high and low are carried into `extra` so the grammar's newer fields are not
        # dead weight. They were dropped here until 2026-09-13, which meant an expression could
        # reference volume and always read 0.0 — a field that silently returns nothing is worse
        # than one that does not exist, because the search wastes trials on it.
        bars = [
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

        # Re-score the winner at every fee level. The variant is fixed — this is not a second
        # search and must not be counted as more trials — so only the cost assumption changes.
        sweep_result: dict[str, float] = {}
        for fee in COST_SWEEP_BPS:
            try:
                at_fee = run(
                    best_label, symbol, bars, ALL_VARIANTS[best_label.split(":")[-1]],
                    cost=CostModel(taker_bps=Decimal(fee) / 2, maker_bps=Decimal("2")),
                    periods_per_year=HOURLY_PER_YEAR,
                )
                sweep_result[f"{fee}bps"] = round(at_fee.net.sharpe, 3)
            except (MetricError, KeyError):
                continue

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
            net=best.get("net", {}),
            stability=best.get("stability"),
            cost_sweep=sweep_result,
            survives_cost_sweep=bool(sweep_result) and all(v > 0 for v in sweep_result.values()),
        ))

    beat = [r for r in results if r.beats_baseline]
    survived_all = [r for r in results if r.dsr_all > 0.95]
    survived_cand = [r for r in results if r.dsr_candidates > 0.95]

    # Every DSR survivor cross-checked against the sibling PBO/FDR study before this module gets
    # to call anything a winner. A survivor whose cross-check is unavailable, or whose PBO verdict
    # says "picking noise", is named explicitly rather than left inside a bare fraction.
    survivor_cross_checks = {
        r.symbol: _pbo_cross_check(r.symbol) for r in survived_cand
    }
    confirmed_survivors = [
        r.symbol for r in survived_cand
        if (check := survivor_cross_checks[r.symbol]) is not None
        and not check["picking_noise"]
        # Not "picking noise" is the per-symbol PBO verdict, and it is not enough on its own —
        # TSLAUSDT reads "carries real information but degrades substantially out of sample" on
        # 2026-09-15, softer than "noise", yet the grid-wide FDR/Bonferroni count for the SAME
        # 300-trial grid that pair sits inside is zero. A survivor only counts as confirmed when
        # the properly-corrected grid found at least one significant result anywhere in it; a
        # per-symbol verdict that merely avoids the word "noise" is not the same claim.
        and check["grid_survivors_benjamini_hochberg"] > 0
    ]

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
            # The number that actually answers "does Track 1 have a strategy": a DSR survivor
            # only counts once the sibling PBO/FDR study, which corrects for the 12-symbol search
            # this module's own DSR does not, has been asked and has not said "picking noise".
            "survived_dsr_and_pbo_cross_check": f"{len(confirmed_survivors)}/{len(results)}",
            "dsr_survivor_pbo_cross_checks": survivor_cross_checks,
            "most_frequent_winner": max(variant_wins, key=lambda k: variant_wins[k])
            if variant_wins else None,
            "variant_win_counts": variant_wins,
            "subtheme_win_counts": subtheme_wins,
            # The harder test, added because a competing entry reports exactly this and is right
            # to: a winner that only clears at our own fee is a winner at one assumption.
            "survived_cost_sweep": f"{sum(1 for r in results if r.survives_cost_sweep)}"
                                   f"/{len(results)}",
            "cost_levels_bps": list(COST_SWEEP_BPS),
        },
        "per_symbol": [
            {
                "symbol": r.symbol, "bars": r.bars, "days": r.days,
                "cost_sweep": r.cost_sweep,
                "survives_cost_sweep": r.survives_cost_sweep,
                "baseline_sharpe": r.baseline_sharpe,
                "best_variant": r.best_variant,
                "best_net_sharpe": r.best_net_sharpe,
                "best_net_return_pct": r.best_net_return_pct,
                "beats_baseline": r.beats_baseline,
                "dsr_all_trials": r.dsr_all,
                "dsr_candidates_only": r.dsr_candidates,
                "out_of_sample_decay": r.oos,
                "sortino": r.net.get("sortino"),
                "max_drawdown_pct": r.net.get("max_drawdown_pct"),
                "turnover": r.net.get("turnover"),
                "win_rate_pct": r.net.get("win_rate_pct"),
                "rolling_sharpe_stability": r.stability,
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
