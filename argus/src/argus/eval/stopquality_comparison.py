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
"""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data" / "h2h_rook"
REPORT = Path(__file__).resolve().parents[3] / "data" / "stopquality_comparison.json"
FIT_SHARE = 0.6


def _hit_rate(bars: list[tuple[float, float]], distance: float) -> float | None:
    """Share of 24h windows (prior close to the day's low) whose low reached ``distance`` down."""
    hits = [low <= prev_close * (1 - distance) for (prev_close, _), (_, low)
            in itertools.pairwise(bars)]
    return sum(hits) / len(hits) if hits else None


def _p90_adverse(bars: list[tuple[float, float]]) -> float | None:
    drops = sorted(1 - low / prev_close for (prev_close, _), (_, low)
                   in itertools.pairwise(bars) if prev_close > 0)
    if len(drops) < 20:
        return None
    return drops[int(0.9 * (len(drops) - 1))]


def score(*, fetch: Any = None) -> dict[str, Any]:
    from argus.market.history import CandleType, fetch_window

    loader = fetch or (lambda s: fetch_window(s, start=datetime.now(UTC) - timedelta(days=500),
                                              interval="1D", candle_type=CandleType.MARKET,
                                              pause=0.05))
    rows: list[dict[str, Any]] = []
    for path in sorted(DATA.glob("*USDT.json")):
        # Rook logs its fetches to stdout ("[bitget] ticker v2 ok ...") before its result line;
        # the capture keeps both, the log as lines and the result as parsed JSON.
        run = json.loads(path.read_text("utf-8")).get("result") or {"error": "no result line"}
        symbol = path.stem
        report, snapshot = run.get("report") or {}, run.get("snapshot") or {}
        last, stop = snapshot.get("last"), report.get("invalidation_price")
        daily = [(float(b.close), float(b.low)) for b in loader(symbol) if float(b.close) > 0]
        cut = int(len(daily) * FIT_SHARE)
        fit, test = daily[:cut], daily[cut - 1:]
        argus_d = _p90_adverse(fit)
        rook_d = (1 - float(stop) / float(last)) if stop and last and float(stop) < float(last) \
            else None
        rows.append({
            "symbol": symbol, "rook_bias": report.get("bias"), "rook_last": last,
            "rook_invalidation": stop, "rook_distance": rook_d,
            "rook_noise_hit_rate_oos": _hit_rate(test, rook_d) if rook_d else None,
            "argus_distance_fit_on_first_60pct": argus_d,
            "argus_noise_hit_rate_oos": _hit_rate(test, argus_d) if argus_d else None,
            "test_windows": len(test) - 1, "error": run.get("error")})
    scored = [r for r in rows if r["rook_noise_hit_rate_oos"] is not None
              and r["argus_noise_hit_rate_oos"] is not None]
    report_out = {
        "comparison": "stop placement: Rook's invalidation price vs ARGUS's measured stop distance",
        "baseline": "iamsuperfly/Rook runDebate (bull/bear + judge) on the hackathon Qwen, long "
                    "side, 24h horizon, from its clone",
        "metric": "share of held-out 24h windows in which ordinary movement alone reached the "
                  "stop (a long's stop: that day's low at or under prior close x (1 - distance))",
        "argus_stop_is_out_of_sample": True,
        "rows": rows,
        "argus_mean_noise_hit_rate": (sum(r["argus_noise_hit_rate_oos"] for r in scored)
                                      / len(scored)) if scored else None,
        "rook_mean_noise_hit_rate": (sum(r["rook_noise_hit_rate_oos"] for r in scored)
                                     / len(scored)) if scored else None,
        "failure_cases": [r for r in rows if r["error"] or r["rook_distance"] is None],
        "not_covered": "whether either stop leaves the right trades on: the thesis outcome needs "
                       "the next 24 hours; six names only",
        "prospective": {"recorded_at": datetime.now(UTC).isoformat(),
                        "stops": {r["symbol"]: {"rook": r["rook_invalidation"],
                                                "argus_distance": r[
                                                    "argus_distance_fit_on_first_60pct"],
                                                "last": r["rook_last"]} for r in rows}},
    }
    REPORT.write_text(json.dumps(report_out, indent=1), encoding="utf-8")
    return report_out


def main() -> int:  # pragma: no cover - CLI
    out = score()
    for r in out["rows"]:
        print(r["symbol"], r["rook_bias"], r["rook_distance"], r["rook_noise_hit_rate_oos"],
              r["argus_distance_fit_on_first_60pct"], r["argus_noise_hit_rate_oos"], r["error"])
    print("mean noise-hit: ARGUS", out["argus_mean_noise_hit_rate"], "Rook",
          out["rook_mean_noise_hit_rate"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
