"""Which forecast of a weekend gap is best: unconditional, regime-matched, or volatility-scaled?

The weekend answer (`lui/research/`, `market/equity_history.py`) states how a stock has opened
after its past weekends. The audit asked for the band to be conditioned on the regime the stock is
in now and the comparison with baserate re-run (finding 27): baserate (Season 2) matches the
current trend and momentum to past weekends and filters on them. Whether any conditioning helps is
an empirical question, so four forecasts of the Friday-close-to-Monday-open move are scored on
weekends none of them saw:

* **unconditional** — the 10th, 50th and 90th percentiles of every earlier weekend's move;
* **regime-matched** — the same, over earlier weekends that began in the same trend (20-day
  average against the 50-day, baserate's 2% band) and five-day momentum tercile as this one:
  baserate's filter, as `equity_history.matched_record` reads it;
* **volatility-scaled** — filtered historical simulation (Barone-Adesi, Giannopoulos and Vosper,
  1999; the ``bootstrap`` forecast of `arch`'s ARCHModel): each earlier move divided by the
  stock's trailing 20-day volatility on its Friday, the quantiles of those standardised moves
  multiplied by this Friday's volatility;
* **volatility-matched** — earlier weekends whose Friday volatility sat in the same tercile as
  this one's, terciles cut on earlier Fridays only.

Walk-forward and point-in-time: every quantile, tercile cut and volatility uses data up to that
Friday's close. Each forecast is scored by the pinball (quantile) loss at 0.1, 0.5 and 0.9, the
proper score for a quantile, and by how often the move landed inside its 10-90 band (80% is
calibrated). The first :data:`WARMUP` weekends of each stock only train.

    python -m argus.eval.weekend_quantiles       # needs the network (Yahoo daily history)
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from argus.market.equity_history import (
    Day,
    Friday,
    Gap,
    closure_gaps,
    daily,
    fridays,
    quantile,
)
from argus.truth.paths import DATA_DIR

REPORT_PATH = DATA_DIR / "weekend_quantiles.json"

TICKERS: tuple[str, ...] = ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "COIN",
                            "MSTR", "JPM", "XOM", "SPY", "QQQ")
"""Fixed 2026-09-27 before the first run: the rToken stocks a weekend question is most often
about, two from outside technology, and the two index funds."""

QUANTILES = (0.1, 0.5, 0.9)
WARMUP = 260
"""Five years of weekends before a forecast is scored."""
MIN_MATCHED = 30
"""A matched forecast needs this many earlier weekends; below it, it falls back to the
unconditional one and the fallback is counted."""

METHODS = ("unconditional", "regime_matched", "vol_scaled", "vol_scaled_ewma",
           "vol_scaled_ewma_conformal", "vol_scaled_ewma_adaptive", "vol_matched")

ACI_GAMMA = 0.005
"""Adaptive conformal inference's step (Gibbs and Candès, "Adaptive Conformal Inference Under
Distribution Shift", NeurIPS 2021, their suggested scale): after a miss the band's target level
rises by ``gamma * 0.8``, after a hit it falls by ``gamma * 0.2``, so the long-run miss rate is
held at 20% whatever the drift."""

CONFORMAL_MIN = 50
"""Earlier out-of-sample scores a conformal correction needs before it widens anything."""



def pinball(value: float, forecast: float, q: float) -> float:
    diff = value - forecast
    return max(q * diff, (q - 1) * diff)


def _cuts(earlier: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(earlier)
    return ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]


def _tercile(value: float, cuts: tuple[float, float]) -> int:
    return 0 if value < cuts[0] else 2 if value > cuts[1] else 1


Forecast = tuple[float, float, float]


def forecasts(history: Sequence[tuple[Gap, Friday]], now: Friday) -> dict[str, tuple[Forecast,
                                                                                     bool]]:
    """Each method's 10/50/90 forecast for the next weekend from ``history`` alone, and whether
    it fell back to the unconditional forecast for want of matched weekends."""
    moves = sorted(g.move for g, _ in history)
    base: Forecast = tuple(quantile(moves, q) for q in QUANTILES)  # type: ignore[assignment]

    def matched(keep: Callable[[Friday], bool]) -> tuple[Forecast, bool]:
        chosen = sorted(g.move for g, f in history if keep(f))
        if len(chosen) < MIN_MATCHED:
            return base, True
        return tuple(quantile(chosen, q) for q in QUANTILES), False  # type: ignore[return-value]

    def rescaled(standardised: list[float], vol: float) -> tuple[Forecast, bool]:
        # filtered historical simulation; with no volatility to scale by, the plain record
        if not standardised or vol <= 0:
            return base, True
        return tuple(quantile(standardised, q) * vol for q in QUANTILES), False  # type: ignore[return-value]

    # Tercile cuts from the earlier weekends only, cut once per forecast: this weekend's state
    # and every earlier one's are binned against the same two numbers.
    five_cuts = _cuts([f.five_day for _, f in history])
    momentum = _tercile(now.five_day, five_cuts)
    vol_cuts = _cuts([f.vol for _, f in history])
    vol_bin = _tercile(now.vol, vol_cuts)
    scaled = sorted(g.move / f.vol for g, f in history if f.vol > 0)
    scaled_ewma = sorted(g.move / f.ewma_vol for g, f in history if f.ewma_vol > 0)
    return {
        "unconditional": (base, False),
        "regime_matched": matched(lambda f: f.trend == now.trend
                                  and _tercile(f.five_day, five_cuts) == momentum),
        "vol_scaled": rescaled(scaled, now.vol),
        "vol_scaled_ewma": rescaled(scaled_ewma, now.ewma_vol),
        "vol_matched": matched(lambda f: _tercile(f.vol, vol_cuts) == vol_bin),
    }


def conformal(made: tuple[Forecast, bool], now: Friday,
              scores: Sequence[float], level: float = 0.8) -> tuple[Forecast, bool]:
    """Split-conformal widening of a 10-90 band (Romano, Patterson and Candès, "Conformalized
    Quantile Regression", NeurIPS 2019): each earlier weekend's conformity score is how far its
    move fell outside the band forecast for it, in units of that Friday's volatility; the band is
    widened at both ends by the 80th percentile of those scores, which makes its coverage 80% if
    the scores are exchangeable. Online, so only weekends already past are used."""
    band, fell_back = made
    if len(scores) < CONFORMAL_MIN or now.ewma_vol <= 0:
        return band, True
    widen = quantile(sorted(scores), min(max(level, 0.0), 1.0)) * now.ewma_vol
    return (band[0] - widen, band[1], band[2] + widen), fell_back


def evaluate(days: Sequence[Day], *, warmup: int = WARMUP) -> dict[str, Any] | None:
    """Walk forward through one stock's weekends; the scores of each method."""
    state = fridays(days)
    pairs = [(g, state[g.closed]) for g in closure_gaps(list(days)) if g.closed in state]
    if len(pairs) <= warmup:
        return None
    loss = {m: 0.0 for m in METHODS}
    inside = {m: 0 for m in METHODS}
    fallbacks = {m: 0 for m in METHODS}
    halves: dict[str, list[float]] = {m: [0.0, 0.0] for m in METHODS}
    scored = len(pairs) - warmup
    conformity: list[float] = []
    aci_level = 0.8
    for i in range(warmup, len(pairs)):
        gap, now = pairs[i]
        made = forecasts(pairs[:i], now)
        made["vol_scaled_ewma_conformal"] = conformal(made["vol_scaled_ewma"], now, conformity)
        made["vol_scaled_ewma_adaptive"] = conformal(made["vol_scaled_ewma"], now, conformity,
                                                     aci_level)
        ewma_band = made["vol_scaled_ewma"][0]
        if now.ewma_vol > 0:
            # scored after the forecast is made, so each correction uses only earlier weekends
            conformity.append(max(ewma_band[0] - gap.move, gap.move - ewma_band[2])
                              / now.ewma_vol)
        adaptive = made["vol_scaled_ewma_adaptive"][0]
        if len(conformity) > CONFORMAL_MIN:
            missed = not adaptive[0] <= gap.move <= adaptive[2]
            aci_level += ACI_GAMMA * ((1.0 if missed else 0.0) - 0.2)
        for method, (band, fell_back) in made.items():
            score = sum(pinball(gap.move, f, q) for f, q in zip(band, QUANTILES, strict=True))
            loss[method] += score
            halves[method][0 if i - warmup < scored // 2 else 1] += score
            inside[method] += band[0] <= gap.move <= band[2]
            fallbacks[method] += fell_back
    first, second = scored // 2, scored - scored // 2
    return {
        "weekends_scored": scored, "first": pairs[warmup][0].closed.isoformat(),
        "last": pairs[-1][0].closed.isoformat(),
        "mean_pinball_bps": {m: round(1e4 * loss[m] / (3 * scored), 3) for m in METHODS},
        "mean_pinball_bps_by_half": {m: [round(1e4 * halves[m][0] / (3 * first), 3),
                                         round(1e4 * halves[m][1] / (3 * second), 3)]
                                     for m in METHODS},
        "band_coverage": {m: round(inside[m] / scored, 4) for m in METHODS},
        "fallbacks": fallbacks,
    }


def summarise(results: Mapping[str, Any]) -> dict[str, Any]:
    rows = {t: r for t, r in results.items() if isinstance(r, dict) and "mean_pinball_bps" in r}
    total = sum(r["weekends_scored"] for r in rows.values())
    pooled = {m: round(sum(r["mean_pinball_bps"][m] * r["weekends_scored"]
                           for r in rows.values()) / total, 3) for m in METHODS} if total else {}
    best = {t: min(METHODS, key=lambda m: r["mean_pinball_bps"][m]) for t, r in rows.items()}
    both_halves = {m: sum(1 for r in rows.values()
                          if all(r["mean_pinball_bps_by_half"][m][h]
                                 < r["mean_pinball_bps_by_half"]["unconditional"][h]
                                 for h in (0, 1)))
                   for m in METHODS if m != "unconditional"}
    return {"stocks": len(rows), "weekends_scored": total, "pooled_mean_pinball_bps": pooled,
            "best_by_stock": best,
            "beats_unconditional_in_both_halves": both_halves,
            "pooled_band_coverage": {m: round(sum(r["band_coverage"][m] * r["weekends_scored"]
                                                  for r in rows.values()) / total, 4)
                                     for m in METHODS} if total else {}}


def run(tickers: Sequence[str] = TICKERS,
        load: Callable[[str], list[Day]] = daily) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for ticker in tickers:
        try:
            days = load(ticker)
        except Exception as exc:
            results[ticker] = {"error": f"history unreadable ({type(exc).__name__})"}
            continue
        results[ticker] = evaluate(days) or {"error": "too few weekends"}
    return {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "method": __doc__.split("\n\n")[2].strip() if __doc__ else "",
            "summary": summarise(results), "results": results}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI, live network
    report = run()
    REPORT_PATH.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["METHODS", "QUANTILES", "REPORT_PATH", "TICKERS", "WARMUP", "Friday", "evaluate",
           "forecasts", "fridays", "main", "pinball", "quantile", "run", "summarise"]
