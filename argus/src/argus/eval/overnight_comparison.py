"""What is the stock worth before it opens: gloaming's overnight fair value against the perpetual.

gloaming (angelraph, a Season 2 desk, MIT) estimates an rToken's fair value while NYSE is shut
from three proxies that trade overnight: an index-futures return (NQ=F, or ES=F for SPY), the
equal-weighted BTC/ETH return, and the inverted DXY return, blended 0.5 / 0.3 / 0.2
(`engine/fairvalue/config.py:39-43`, `engine/fairvalue/model.py:19-39`), or with weights
OLS-calibrated per symbol (`engine/fairvalue/model.py:65-98`). ARGUS reads the Bitget stock
perpetual itself, which trades through the night: the console shows the perpetual against the
stock's last close.

Both are estimates of one thing a trader can check the next morning: **the gap, from the last
regular close to the next regular open.** For every pair of consecutive NYSE sessions each
estimator is given prices up to 09:00 New York time (the last full hourly bar before the 09:30
open) and scored against the open Yahoo records.

Candidates, all fed the same hourly bars:

* ``gloaming_prior`` — gloaming's own blend function with its shipped weights, on the overnight
  window (close to 09:00). Its live agent uses 24-hour changes and a one-bar DXY change
  (`gloaming_agent/agent_loop.py:80-113`; its futures "24h" change is in fact first-to-last over
  a five-day download); the overnight window is the fairer input for this target, so the shipped
  inputs are reported separately as ``gloaming_shipped_24h``.
* ``gloaming_ols`` — its ``calibrate_weights`` refit before every night on that symbol's earlier
  nights only (walk-forward; the shipped prior until 10 nights exist, as in its own code).
* ``argus_perp`` — the perpetual's move since the close: perp(09:00) / perp(16:00) - 1.
* ``argus_perp_vs_close`` — what the console literally prints: perp(09:00) / stock close - 1,
  which carries the perpetual's standing basis.
* ``argus_perp_fitted`` — ``argus_perp`` scaled by a walk-forward, through-the-origin slope, the
  same calibration allowance gloaming's OLS gets.
* ``zero`` — no gap; the floor any estimator must beat.

gloaming's model functions are imported from its clone and run unmodified. Their BTC/ETH input is
Bitget spot in the original; here it is Bitget's USDT perpetuals, whose hourly closes track spot
within a few basis points — disclosed, not believed to matter.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import urllib.request
from datetime import UTC, date, datetime, time
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "h2h_gloaming"
REPORT = ROOT / "data" / "overnight_comparison.json"
GLOAMING_ENGINE = ROOT.parent / "research" / "repos-rivals" / "gloaming" / "engine"
NEW_YORK = ZoneInfo("America/New_York")

# gloaming's universe (`engine/fairvalue/config.py:28-38`) on the names Bitget lists as perpetuals.
UNIVERSE: dict[str, str] = {
    "AAPL": "NQ=F", "AMZN": "NQ=F", "META": "NQ=F", "TSLA": "NQ=F", "GOOGL": "NQ=F",
    "NVDA": "NQ=F", "MSFT": "NQ=F", "QQQ": "NQ=F",
}
PROXIES = ("NQ=F", "ES=F", "DX-Y.NYB")
CRYPTO = ("BTCUSDT", "ETHUSDT")
PRIOR = {"futures_proxy_return": 0.5, "crypto_beta_return": 0.3,
         "fx_risk_sentiment_return": 0.2}
MIN_FIT = 10
DAYS = 180
MOVE_FLOOR = 0.001
"""Direction is scored only on gaps of at least 10bps; smaller ones are noise around zero."""

CANDIDATES = ("zero", "gloaming_prior", "gloaming_shipped_24h", "gloaming_ols", "argus_perp",
              "argus_perp_vs_close", "argus_perp_fitted", "argus_perp_plus_futures")

Series = list[tuple[float, float]]
"""(bar end as a UTC timestamp, close), oldest first."""


# ---------------------------------------------------------------------------------------------
# Inputs: fetched once and saved, so the score is reproducible from the files alone.

def _yahoo(ticker: str, *, interval: str, span: str) -> dict[str, Any]:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval={interval}&range={span}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 argus-research"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload: dict[str, Any] = json.load(response)
    return payload


def _hourly_yahoo(ticker: str) -> Series:
    result = _yahoo(ticker, interval="60m", span=f"{DAYS}d")["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    return [(float(stamp) + 3600, float(close)) for stamp, close in
            zip(result["timestamp"], closes, strict=False) if close]


def _hourly_bitget(symbol: str) -> Series:
    from argus.market.history import fetch_range

    return [(candle.ts.timestamp() + 3600, float(candle.close))
            for candle in fetch_range(symbol, days=DAYS, interval="1H")]


def _sessions(ticker: str) -> list[dict[str, Any]]:
    """Unadjusted regular-session opens and closes: a gap is read between two adjacent days, and a
    split between them would show in both prices of any adjusted pair only if applied to both."""
    result = _yahoo(ticker, interval="1d", span="1y")["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    days = []
    for stamp, opened, closed in zip(result["timestamp"], quote["open"], quote["close"],
                                     strict=False):
        if opened and closed:
            day = datetime.fromtimestamp(stamp, tz=NEW_YORK).date()
            days.append({"day": day.isoformat(), "open": float(opened), "close": float(closed)})
    return days


def collect() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, Any] = {"fetched": datetime.now(UTC).isoformat(timespec="seconds"),
                              "days": DAYS, "hourly": {}, "sessions": {}, "failures": {}}
    for ticker in PROXIES:
        inputs["hourly"][ticker] = _hourly_yahoo(ticker)
    for symbol in CRYPTO:
        inputs["hourly"][symbol] = _hourly_bitget(symbol)
    for stock in UNIVERSE:
        try:
            inputs["hourly"][f"{stock}USDT"] = _hourly_bitget(f"{stock}USDT")
            inputs["sessions"][stock] = _sessions(stock)
        except Exception as exc:  # one name's failure is recorded, not fatal
            inputs["failures"][stock] = f"{type(exc).__name__}: {exc}"[:160]
    (DATA / "inputs.json").write_text(json.dumps(inputs), "utf-8")
    return inputs


# ---------------------------------------------------------------------------------------------
# Scoring.

def _at(series: Series, moment: float, *, stale: float = 3 * 3600) -> float | None:
    """The last close whose bar ended at or before ``moment``, if it is recent enough to be the
    price then (a proxy that stopped printing hours earlier is not a reading of the moment)."""
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= moment:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return None
    end, close = series[lo - 1]
    return close if moment - end <= stale else None


def _first_from(series: Series, moment: float) -> float | None:
    """The first close whose bar ended at or after ``moment``."""
    return next((close for end, close in series if end >= moment), None)


def _move(series: Series, start: float, end: float) -> float | None:
    a, b = _at(series, start), _at(series, end)
    return b / a - 1 if a and b else None


def _ny(day: date, clock: time) -> float:
    return datetime.combine(day, clock, tzinfo=NEW_YORK).timestamp()


def _gloaming() -> Any:
    """gloaming's own model module, vendored byte-for-byte but for one import line
    (`eval/baselines/gloaming_fairvalue.py`; the clone at ``GLOAMING_ENGINE`` is where it came
    from and what `tests/test_overnight_comparison.py` checks it against)."""
    from argus.eval.baselines import gloaming_fairvalue

    return gloaming_fairvalue


def _blend(model: Any, futures: float, crypto: float, fx: float,
           weights: dict[str, float]) -> float:
    import pandas as pd

    one = pd.Series([0.0])
    out = model.blended_fair_value_return(one + futures, one + crypto, one + fx, weights=weights)
    return float(out.iloc[0])


def nights(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (stock, adjacent session pair) with every input read at its moment."""
    hourly = {k: [(float(t), float(c)) for t, c in v] for k, v in inputs["hourly"].items()}
    rows = []
    for stock, proxy in UNIVERSE.items():
        sessions = inputs["sessions"].get(stock) or []
        perp = hourly.get(f"{stock}USDT") or []
        for before, after in itertools.pairwise(sessions):
            d0, d1 = date.fromisoformat(before["day"]), date.fromisoformat(after["day"])
            if (d1 - d0).days > 5:
                continue
            close_at, predict_at = _ny(d0, time(16)), _ny(d1, time(9))
            day_before = predict_at - 24 * 3600
            # gloaming's live futures input is the first-to-last change over a five-day hourly
            # download (`agent_loop.py:80-95`), whatever its docstring calls it.
            first = _first_from(hourly[proxy], predict_at - 5 * 24 * 3600)
            last = _at(hourly[proxy], predict_at)
            reads = {
                "perp_close": _at(perp, close_at), "perp_now": _at(perp, predict_at),
                "futures": _move(hourly[proxy], close_at, predict_at),
                "futures_24h": last / first - 1 if first and last else None,
                "dxy": _move(hourly["DX-Y.NYB"], close_at, predict_at),
                "dxy_last_bar": _move(hourly["DX-Y.NYB"], predict_at - 3600, predict_at),
            }
            crypto = [_move(hourly[c], close_at, predict_at) for c in CRYPTO]
            crypto_24h = [_move(hourly[c], day_before, predict_at) for c in CRYPTO]
            if any(v is None for v in (*reads.values(), *crypto, *crypto_24h)):
                continue
            rows.append({
                "stock": stock, "closed": d0.isoformat(), "opened": d1.isoformat(),
                "gap": after["open"] / before["close"] - 1,
                "close": before["close"], **reads,
                "crypto": mean(c for c in crypto if c is not None),
                "crypto_24h": mean(c for c in crypto_24h if c is not None),
            })
    return rows


def predict(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every candidate's gap estimate per row, calibrated candidates walking forward per stock."""
    import pandas as pd

    model = _gloaming()
    by_stock: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: (r["stock"], r["opened"])):
        by_stock.setdefault(row["stock"], []).append(row)
    out = []
    for stock_rows in by_stock.values():
        for i, row in enumerate(stock_rows):
            past = stock_rows[:i]
            fx = -row["dxy"]
            perp_move = row["perp_now"] / row["perp_close"] - 1
            if len(past) >= MIN_FIT:
                weights = model.calibrate_weights(
                    pd.Series([p["gap"] for p in past]),
                    pd.Series([p["futures"] for p in past]),
                    pd.Series([p["crypto"] for p in past]),
                    pd.Series([-p["dxy"] for p in past]))
                xs = [p["perp_now"] / p["perp_close"] - 1 for p in past]
                denominator = sum(x * x for x in xs)
                slope = (sum(x * p["gap"] for x, p in zip(xs, past, strict=True)) / denominator
                         if denominator else 1.0)
            else:
                weights, slope = dict(PRIOR), 1.0
            combined = perp_move
            if len(past) >= MIN_FIT:
                import numpy as np

                design = np.array([[p["perp_now"] / p["perp_close"] - 1,
                                    p["futures"] - (p["perp_now"] / p["perp_close"] - 1)]
                                   for p in past])
                coef = np.linalg.lstsq(design, np.array([p["gap"] for p in past]),
                                       rcond=None)[0]
                combined = float(coef[0] * perp_move + coef[1] * (row["futures"] - perp_move))
            out.append({**row, "estimates": {
                "zero": 0.0,
                "gloaming_prior": _blend(model, row["futures"], row["crypto"], fx, PRIOR),
                "gloaming_shipped_24h": _blend(model, row["futures_24h"], row["crypto_24h"],
                                               -row["dxy_last_bar"], PRIOR),
                "gloaming_ols": _blend(model, row["futures"], row["crypto"], fx, weights),
                "argus_perp": perp_move,
                "argus_perp_vs_close": row["perp_now"] / row["close"] - 1,
                "argus_perp_fitted": slope * perp_move,
                "argus_perp_plus_futures": combined,
            }, "fitted": len(past) >= MIN_FIT})
    return out


def _summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    errors = [abs(r["estimates"][name] - r["gap"]) * 1e4 for r in rows]
    moved = [r for r in rows if abs(r["gap"]) >= MOVE_FLOOR]
    hits = [(r["estimates"][name] > 0) == (r["gap"] > 0) for r in moved
            if r["estimates"][name] != 0]
    ordered = sorted(errors)
    return {"mae_bps": round(mean(errors), 2),
            "median_abs_error_bps": round(ordered[len(ordered) // 2], 2),
            "rmse_bps": round(mean(e * e for e in errors) ** 0.5, 2),
            "direction_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
            "direction_scored": len(hits)}


def _paired(rows: list[dict[str, Any]], a: str, b: str, *, draws: int = 4000,
            seed: int = 7) -> dict[str, Any]:
    """Mean of |err_a| - |err_b| in bps with a bootstrap over nights: every stock shares a night's
    news, so the night, not the row, is the independent unit."""
    by_night: dict[str, list[float]] = {}
    for r in rows:
        diff = (abs(r["estimates"][a] - r["gap"]) - abs(r["estimates"][b] - r["gap"])) * 1e4
        by_night.setdefault(r["opened"], []).append(diff)
    keys = sorted(by_night)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [d for k in (rng.choice(keys) for _ in keys) for d in by_night[k]]
        means.append(mean(sample))
    means.sort()
    point = mean(d for k in keys for d in by_night[k])
    low, high = means[int(0.025 * draws)], means[int(0.975 * draws) - 1]
    return {"a": a, "b": b, "mean_diff_bps": round(point, 2),
            "ci95_bps": [round(low, 2), round(high, 2)], "nights": len(keys),
            "verdict": ("a better" if high < 0 else "b better" if low > 0 else "not separable")}


def sunday_evening(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """nocturne's claim on the stock itself, at nocturne's own moment.

    nocturne (an S2 desk, MIT) finds that an rToken's weekend move reverses and publishes, before
    each Monday, that "Monday lands closer to the last regular close than to the weekend price"
    (`README.md`, claim 1). Read at Sunday 20:00 New York — the close of the bar nocturne reads
    at 19:00 — the perpetual's weekend move is scored here against the stock's actual Monday open,
    beside the last regular close that claim names. The slope says whether the perpetual's weekend
    move reverses (negative) or carries through (positive)."""
    hourly = {k: [(float(t), float(c)) for t, c in v] for k, v in inputs["hourly"].items()}
    rows: list[dict[str, Any]] = []
    for stock in UNIVERSE:
        perp = hourly.get(f"{stock}USDT") or []
        sessions = inputs["sessions"].get(stock) or []
        for before, after in itertools.pairwise(sessions):
            d0, d1 = date.fromisoformat(before["day"]), date.fromisoformat(after["day"])
            if d0.weekday() != 4 or d1.weekday() != 0:
                continue
            closed = _at(perp, _ny(d0, time(16)))
            sunday = _at(perp, _ny(date.fromordinal(d1.toordinal() - 1), time(20)))
            if not closed or not sunday:
                continue
            rows.append({"stock": stock, "closed": d0.isoformat(), "opened": d1.isoformat(),
                         "gap": after["open"] / before["close"] - 1,
                         "estimates": {"last_close": 0.0, "perp_weekend": sunday / closed - 1}})
    if len(rows) < 20:
        return None
    xs = [r["estimates"]["perp_weekend"] for r in rows]
    ys = [r["gap"] for r in rows]
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx if sxx else None
    return {"rows": len(rows), "weekends": len({r["opened"] for r in rows}),
            "summary": {n: _summary(rows, n) for n in ("last_close", "perp_weekend")},
            "slope_gap_on_perp_weekend_move": round(slope, 3) if slope is not None else None,
            "paired": _paired(rows, "perp_weekend", "last_close")}


def score(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    inputs = inputs or json.loads((DATA / "inputs.json").read_text("utf-8"))
    rows = predict(nights(inputs))
    scored = [r for r in rows if r["fitted"]]
    if not scored:
        return {"error": "no night had enough history to fit", "rows": len(rows)}
    summary = {name: _summary(scored, name) for name in CANDIDATES}
    best_gloaming = min(("gloaming_prior", "gloaming_shipped_24h", "gloaming_ols"),
                        key=lambda n: summary[n]["mae_bps"])
    per_stock = {}
    for stock in UNIVERSE:
        subset = [r for r in scored if r["stock"] == stock]
        if subset:
            per_stock[stock] = {"nights": len(subset), **{
                n: _summary(subset, n)["mae_bps"]
                for n in ("zero", best_gloaming, "argus_perp", "argus_perp_vs_close")},
                "argus_perp_direction_hit_rate": _summary(subset, "argus_perp")[
                    "direction_hit_rate"],
                "argus_perp_vs_best_gloaming": _paired(subset, "argus_perp", best_gloaming)}
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs_fetched": inputs.get("fetched"),
        "target": "gap from the last regular close to the next regular open, per stock",
        "predicted_at": "09:00 New York time on the opening day",
        "scored_rows": len(scored), "nights": len({r["opened"] for r in scored}),
        "first_night": min(r["opened"] for r in scored),
        "last_night": max(r["opened"] for r in scored),
        "summary": summary,
        "best_gloaming": best_gloaming,
        "paired": [_paired(scored, "argus_perp", best_gloaming),
                   _paired(scored, "argus_perp", "zero"),
                   _paired(scored, best_gloaming, "zero"),
                   _paired(scored, "argus_perp", "argus_perp_vs_close"),
                   _paired(scored, "argus_perp_fitted", "argus_perp")],
        "per_stock": per_stock,
        "sunday_evening_vs_nocturne_claim": sunday_evening(inputs),
        "out_of_sample": {
            "method": "walk-forward per stock: every fitted candidate is refit before each night "
                      "on that stock's earlier nights only",
            "warmup_nights_excluded_per_stock": MIN_FIT,
            "rows_excluded": len(rows) - len(scored), "rows_scored": len(scored),
        },
        "ablation": {
            "shipped": "argus_perp",
            "variants_mae_bps": {n: summary[n]["mae_bps"] for n in
                                 ("argus_perp", "argus_perp_vs_close", "argus_perp_fitted",
                                  "argus_perp_plus_futures")},
            "without_the_basis_correction": _paired(scored, "argus_perp", "argus_perp_vs_close"),
            "with_gloamings_futures_added": _paired(scored, "argus_perp_plus_futures",
                                                    "argus_perp"),
        },
        "failures": inputs.get("failures") or {},
        "scope": ("Rows before a stock's 11th night are excluded from every candidate so the "
                  "walk-forward fits are out of sample on the same rows as the fixed ones. "
                  "gloaming's crypto input is Bitget USDT perpetuals, not spot."),
    }


def main() -> int:  # pragma: no cover - CLI
    if "--collect" in sys.argv:
        collect()
    report = score()
    REPORT.write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({k: report.get(k) for k in ("scored_rows", "nights", "summary",
                                                 "paired")}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
