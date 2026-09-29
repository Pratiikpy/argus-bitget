"""Where to put the stop: Rook's invalidation price against ARGUS's measured stop distance.

Rook (iamsuperfly, a Season 2 desk, MIT) ends every thesis with an ``invalidation_price``: a model
proposes it, and code overrides a proposal on the wrong side of the market with the 24-hour low
(a long), the SMA20, or 1% away (`lib/desk/invalidation.ts:34-60`). ARGUS's directional answers
state instead how far against a position 90% of past windows of the same length went at their
worst point (`desk/odds.py`), i.e. the distance inside which a stop is taken out by ordinary
movement more than one time in ten.

The two are different kinds of line — Rook's says where the thesis is wrong, ARGUS's where the noise
ends — and a trader uses either as a stop. So each is scored on the property a stop is chosen for:
**how often ordinary movement alone would have hit it.** For every past 24-hour window on Bitget's
daily bars, a long stop at distance ``d`` is hit when that day's low is ``d`` or more under the
prior close. Rook's price is read as the distance it implies from its own snapshot's last price.

ARGUS's distance is fitted on the first 60% of the history and scored on the last 40%, so its hit
rate is out of sample; Rook's is fixed by its run and scored on the same last 40%. Neither says
whether the thesis was right — that needs the next 24 hours, and the prospective check at the end
records both stops for grading then.

**ARGUS's distance is the console's own (changed 2026-09-26).** Until then this module computed it
with a nearest-rank copy of the rule written here (``_p90_adverse``) while `eval/standing.py`
credited `desk/odds.py`; the harness-validity canary (`data/harness_validity.json`) found
``directional_odds`` never ran, and the harness also fetched its bars live, so it could not be
re-run offline. Now the fit slice is handed to ``argus.desk.odds.directional_odds`` exactly as
`lui/research/analogue.py::_odds_lines` hands it the console's daily bars for a 24-hour long: closes
stamped with their close time, each bar's (low, high) as ``extremes``, one bar per window, and the
stop distance is its ``adverse_p90_bps`` (a linearly interpolated 10th percentile of the worst point
on the lows). The hurdle passed is the 12bps round trip without funding, which does not enter the
adverse measure. The bars are saved once by :func:`collect` (``data/h2h_rook/bars/``) and scored
from the files; the copy of the old rule is kept only to publish how far it differed
(``argus_distance_former_copy``). The prospective block records the stops as they stood when Rook
ran; a re-score keeps it as recorded rather than re-stamping it, because `eval/stopquality_
prospective.py` grades the 24 hours after that moment.
"""

from __future__ import annotations

import itertools
import json
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from argus.truth import artefact
from argus.truth.paths import DATA_DIR

DATA = DATA_DIR / "h2h_rook"
BARS = DATA / "bars"
REPORT = DATA_DIR / "stopquality_comparison.json"
FIT_SHARE = 0.6
DAYS = 500
"""The console's own look-back for a daily-horizon odds answer
(`lui/research/analogue.py::_odds_lines`)."""

Bar = tuple[datetime, float, float, float]
"""(bar open, close, low, high), oldest first."""


def _hit_rate(bars: list[tuple[float, float]], distance: float) -> float | None:
    """Share of 24h windows (prior close to the day's low) whose low reached ``distance`` down."""
    hits = [low <= prev_close * (1 - distance) for (prev_close, _), (_, low)
            in itertools.pairwise(bars)]
    return sum(hits) / len(hits) if hits else None


def _p90_adverse(bars: list[tuple[float, float]]) -> float | None:
    """The rule this harness scored before it called the console (nearest rank, 20-window floor).
    Kept only to publish how far it differs from ``directional_odds``; nothing is scored on it."""
    drops = sorted(1 - low / prev_close for (prev_close, _), (_, low)
                   in itertools.pairwise(bars) if prev_close > 0)
    if len(drops) < 20:
        return None
    return drops[int(0.9 * (len(drops) - 1))]


def console_stop_distance(bars: list[Bar]) -> float | None:
    """The console's stop distance for a 24-hour long over ``bars``, as a fraction.

    ``argus.desk.odds.directional_odds`` receives what `lui/research/analogue.py::_odds_lines`
    builds from the console's daily bars: each close stamped with its bar's close (open + 1 day),
    each bar's (low, high) as ``extremes``, one bar per window, side long. Its ``adverse_p90_bps``
    is the move against a long that 90% of windows stayed inside; the distance is its magnitude. An
    exception
    from the console propagates rather than becoming a missing row."""
    from argus.cost.model import CostModel
    from argus.desk.odds import directional_odds

    closes = [(ts + timedelta(days=1), close) for ts, close, _, _ in bars]
    extremes = [(low, high) for _, _, low, high in bars]
    odds = directional_odds(closes, 1, cost_bps=float(CostModel.bitget_perp().round_trip_bps()),
                            side="long", extremes=extremes)
    if odds is None or odds.adverse_p90_bps is None:
        return None
    return abs(odds.adverse_p90_bps) / 10_000


# ---------------------------------------------------------------------------------------------
# The general tool, and what a stop costs (Activity/28_CAPABILITY_CLOSE_PLAN_2.md §43).
#
# Hit rate alone rewards width: any wider stop is "hit less often", so ARGUS's lead over Rook's
# 0.1-1.7% stops is true by construction. The general-purpose answer to "where does the noise
# end" is a volatility-scaled stop, the ATR stop and LeBeau's chandelier (22-day high minus three
# ATR), formulas as TA-Lib computes them (`research/repos-themed/TA-Lib~ta-lib`, Wilder's ATR).
# Every arm trades the same long on every held-out window: bought at the prior close, sold at the
# stop if the day's low reaches it, otherwise at the close, 12bps round trip taker. The saved bars
# carry no open, so a gap through the stop is filled at the stop, which flatters every stop arm
# equally; that is stated in the report rather than hidden.

ATR_PERIOD = 14
CHANDELIER_LOOKBACK = 22
QUANTILE_GRID = (0.80, 0.90, 0.95)
ARMS = ("argus", "atr_1x", "atr_2x", "chandelier", "rook", "no_stop")


def atr(bars: list[Bar], period: int = ATR_PERIOD) -> list[float | None]:
    """Wilder's average true range at each bar, using only that bar and earlier ones; ``None``
    until ``period`` true ranges exist. True range: max(high - low, |high - prior close|,
    |low - prior close|)."""
    out: list[float | None] = [None]
    trs: list[float] = []
    value: float | None = None
    for (_, prev_close, _, _), (_, _, low, high) in itertools.pairwise(bars):
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        if value is None:
            value = sum(trs) / period if len(trs) == period else None
        else:
            value = (value * (period - 1) + tr) / period
        out.append(value)
    return out


def _quantile_distance(bars: list[Bar], q: float) -> float | None:
    """The ``q`` quantile of the prior-close-to-low drop on ``bars``, linearly interpolated, the
    same shape of figure ``directional_odds`` reports at 0.9."""
    drops = sorted(1 - low / prev_close for (_, prev_close, _, _), (_, _, low, _)
                   in itertools.pairwise(bars) if prev_close > 0)
    if len(drops) < 20:
        return None
    pos = q * (len(drops) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(drops) - 1)
    return drops[lo] + (drops[hi] - drops[lo]) * (pos - lo)


def window_rows(symbol: str, bars: list[Bar], cut: int, argus_d: float | None,
                rook_d: float | None, *, grid: dict[float, float | None] | None = None,
                cost_bps: float = 12.0) -> list[dict[str, Any]]:
    """One row per held-out 24h window: every arm's stop distance, whether ordinary movement hit
    it, and the long's net return in bps. Nothing after the window's prior close is used to set a
    stop: ATR and the chandelier high are read at the prior bar."""
    ranges = atr(bars)
    rows: list[dict[str, Any]] = []
    for i in range(max(cut, CHANDELIER_LOOKBACK), len(bars)):
        _, prev_close, _, _ = bars[i - 1]
        stamp, close, low, _ = bars[i]
        prior_atr = ranges[i - 1]
        if prev_close <= 0 or prior_atr is None:
            continue
        top = max(high for _, _, _, high in bars[i - CHANDELIER_LOOKBACK:i])
        chandelier_stop = top - 3 * prior_atr
        distances: dict[str, float | None] = {
            "argus": argus_d, "rook": rook_d,
            "atr_1x": prior_atr / prev_close, "atr_2x": 2 * prior_atr / prev_close,
            "chandelier": (1 - chandelier_stop / prev_close)
            if chandelier_stop < prev_close else None,
            "no_stop": None,
        }
        for q, d in (grid or {}).items():
            distances[f"argus_q{int(q * 100)}"] = d
        row: dict[str, Any] = {"symbol": symbol, "date": stamp.date().isoformat(),
                               "prior_close": prev_close, "low": low, "close": close,
                               "atr_pct": round(prior_atr / prev_close, 6)}
        for arm, d in distances.items():
            hit = d is not None and low <= prev_close * (1 - d)
            exit_price = prev_close * (1 - d) if hit and d is not None else close
            row[f"{arm}_distance"] = None if d is None else round(d, 6)
            row[f"{arm}_hit"] = hit
            row[f"{arm}_net_bps"] = round((exit_price / prev_close - 1) * 1e4 - cost_bps, 3)
        rows.append(row)
    return rows


def kupiec(hits: int, n: int, target: float) -> dict[str, Any]:
    """Kupiec's proportion-of-failures likelihood-ratio test: is ``hits`` in ``n`` consistent with
    a stop meant to be hit a ``target`` share of the time? Chi-square, one degree of freedom."""
    import math

    if n == 0:
        return {"hits": 0, "n": 0, "rate": None, "lr": None, "p_value": None}
    rate = hits / n

    def loglik(p: float) -> float:
        a = hits * math.log(p) if hits else 0.0
        b = (n - hits) * math.log(1 - p) if n - hits else 0.0
        return a + b

    lr = -2 * (loglik(target) - loglik(rate)) if 0 < rate < 1 else 0.0
    p_value = math.erfc(math.sqrt(max(lr, 0.0) / 2))
    return {"hits": hits, "n": n, "rate": round(rate, 4), "target": target,
            "lr": round(lr, 4), "p_value": round(p_value, 4)}


def collect(*, fetch: Any = None) -> dict[str, Any]:
    """Save each name's daily bars once (``data/h2h_rook/bars/<symbol>.json``), the console's own
    fetch: Bitget market candles, 1D, the last :data:`DAYS` days."""
    from argus.market.history import CandleType, fetch_window

    loader = fetch or (lambda s: fetch_window(s, start=datetime.now(UTC) - timedelta(days=DAYS),
                                              interval="1D", candle_type=CandleType.MARKET,
                                              pause=0.05))
    BARS.mkdir(parents=True, exist_ok=True)
    saved: dict[str, int] = {}
    for path in sorted(DATA.glob("*USDT.json")):
        rows = [[b.ts.isoformat(), str(b.close), str(b.low), str(b.high)]
                for b in loader(path.stem) if float(b.close) > 0]
        artefact.write(BARS / path.name, {"symbol": path.stem, "interval": "1D",
                                          "fetched": datetime.now(UTC).isoformat(
                                              timespec="seconds"),
                                          "bars": rows})
        saved[path.stem] = len(rows)
    return saved


def _saved_bars(symbol: str) -> list[Bar]:
    blob = json.loads((BARS / f"{symbol}.json").read_text("utf-8"))
    return [(datetime.fromisoformat(ts), float(close), float(low), float(high))
            for ts, close, low, high in blob["bars"]]


def score(*, fetch: Any = None) -> dict[str, Any]:
    """Score every saved Rook run against the saved bars (``fetch``, given, replaces the files:
    it returns objects with ``ts``, ``close``, ``low`` and ``high``)."""
    rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for path in sorted(DATA.glob("*USDT.json")):
        # Rook logs its fetches to stdout ("[bitget] ticker v2 ok ...") before its result line;
        # the capture keeps both, the log as lines and the result as parsed JSON.
        run = json.loads(path.read_text("utf-8")).get("result") or {"error": "no result line"}
        symbol = path.stem
        report, snapshot = run.get("report") or {}, run.get("snapshot") or {}
        last, stop = snapshot.get("last"), report.get("invalidation_price")
        bars: list[Bar] = ([(b.ts, float(b.close), float(b.low), float(b.high))
                            for b in fetch(symbol) if float(b.close) > 0] if fetch
                           else _saved_bars(symbol))
        daily = [(close, low) for _, close, low, _ in bars]
        cut = int(len(bars) * FIT_SHARE)
        fit, test = bars[:cut], daily[cut - 1:]
        argus_d = console_stop_distance(fit)
        rook_d = (1 - float(stop) / float(last)) if stop and last and float(stop) < float(last) \
            else None
        grid = {q: _quantile_distance(fit, q) for q in QUANTILE_GRID}
        windows.extend(window_rows(symbol, bars, cut, argus_d, rook_d, grid=grid))
        rows.append({
            "symbol": symbol, "rook_bias": report.get("bias"), "rook_last": last,
            "rook_invalidation": stop, "rook_distance": rook_d,
            "rook_noise_hit_rate_oos": _hit_rate(test, rook_d) if rook_d else None,
            "argus_distance_fit_on_first_60pct": argus_d,
            "argus_noise_hit_rate_oos": _hit_rate(test, argus_d) if argus_d else None,
            "argus_distance_former_copy": _p90_adverse(daily[:cut]),
            "bars": len(bars), "first_bar": bars[0][0].isoformat() if bars else None,
            "last_bar": bars[-1][0].isoformat() if bars else None,
            "test_windows": len(test) - 1, "error": run.get("error")})
    scored = [r for r in rows if r["rook_noise_hit_rate_oos"] is not None
              and r["argus_noise_hit_rate_oos"] is not None]
    try:
        # Recorded when Rook ran and graded on the 24 hours after it: never re-stamped.
        prospective = json.loads(REPORT.read_text("utf-8"))["prospective"]
    except (OSError, ValueError, KeyError):
        prospective = {"recorded_at": datetime.now(UTC).isoformat(),
                       "stops": {r["symbol"]: {"rook": r["rook_invalidation"],
                                               "argus_distance": r[
                                                   "argus_distance_fit_on_first_60pct"],
                                               "last": r["rook_last"]} for r in rows}}
    report_out = {
        "comparison": "stop placement: Rook's invalidation price vs ARGUS's measured stop distance",
        "baseline": "iamsuperfly/Rook runDebate (bull/bear + judge) on the hackathon Qwen, long "
                    "side, 24h horizon, from its clone",
        "metric": "share of held-out 24h windows in which ordinary movement alone reached the "
                  "stop (a long's stop: that day's low at or under prior close x (1 - distance))",
        "argus_stop_is_out_of_sample": True,
        "argus_code": "argus.desk.odds.directional_odds(adverse_p90_bps) on the fit slice, the "
                      "call lui/research/analogue.py::_odds_lines makes for a 24-hour long",
        "inputs": "data/h2h_rook/*.json (Rook's runs) and data/h2h_rook/bars/ (Bitget 1D market "
                  "candles saved by collect())",
        "rows": rows,
        "argus_mean_noise_hit_rate": (sum(r["argus_noise_hit_rate_oos"] for r in scored)
                                      / len(scored)) if scored else None,
        "rook_mean_noise_hit_rate": (sum(r["rook_noise_hit_rate_oos"] for r in scored)
                                     / len(scored)) if scored else None,
        "failure_cases": [r for r in rows if r["error"] or r["rook_distance"] is None],
        "not_covered": "whether either stop leaves the right trades on: the thesis outcome needs "
                       "the next 24 hours; six names only",
        "prospective": prospective,
        "arms": _arms(windows),
        "windows": windows,
    }
    artefact.write(REPORT, report_out, indent=1)
    return report_out


def _arms(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every arm on the same held-out windows: hit rate and mean net return, the ablation grid on
    ARGUS's quantile, ARGUS's coverage against its own 10% target, and the most volatile fifth of
    windows on their own."""
    from statistics import fmean

    def block(rows: list[dict[str, Any]], arms: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {"windows": len(rows)}
        for arm in arms:
            live = [r for r in rows if r.get(f"{arm}_distance") is not None or arm == "no_stop"]
            out[arm] = {"windows": len(live),
                        "hit_rate": round(fmean(r[f"{arm}_hit"] for r in live), 4)
                        if live else None,
                        "mean_net_bps": round(fmean(r[f"{arm}_net_bps"] for r in live), 3)
                        if live else None}
        return out

    if not windows:
        return {}
    grid = tuple(f"argus_q{int(q * 100)}" for q in QUANTILE_GRID)
    vols = sorted(r["atr_pct"] for r in windows)
    cut = vols[int(len(vols) * 0.8)]
    argus_rows = [r for r in windows if r["argus_distance"] is not None]
    return {
        "method": "a long bought at the prior close and sold at the stop if the day's low reaches "
                  "it, else at the close; 12bps round trip; ATR is Wilder's 14-day, the "
                  "chandelier is the 22-day high minus 3 ATR, both read at the prior bar; the "
                  "bars carry no open, so a gap through a stop is filled at the stop",
        "all": block(windows, ARMS),
        "ablation_quantile": block(windows, grid),
        "argus_coverage_vs_10pct": kupiec(sum(r["argus_hit"] for r in argus_rows),
                                          len(argus_rows), 0.10),
        "adversarial": {"most_volatile_fifth": {
            "min_atr_pct": round(cut, 5),
            **block([r for r in windows if r["atr_pct"] >= cut], ARMS)}},
    }


def main() -> int:  # pragma: no cover - CLI
    if "--collect" in sys.argv:
        print(collect())
    out = score()
    for r in out["rows"]:
        print(r["symbol"], r["rook_bias"], r["rook_distance"], r["rook_noise_hit_rate_oos"],
              r["argus_distance_fit_on_first_60pct"], r["argus_noise_hit_rate_oos"], r["error"])
    print("mean noise-hit: ARGUS", out["argus_mean_noise_hit_rate"], "Rook",
          out["rook_mean_noise_hit_rate"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
