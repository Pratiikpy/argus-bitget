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
* ``argus_perp`` — **the console's own implied open**: `lui/research._implied_open_line` called
  at 09:00 New York with the perpetual's last price then, its gap read back from the sentence it
  prints (the implied price over the last regular close). See "What is scored" below.
* ``argus_perp_vs_close`` — what the console literally prints: perp(09:00) / stock close - 1,
  which carries the perpetual's standing basis.
* ``argus_perp_fitted`` — ``argus_perp`` scaled by a walk-forward, through-the-origin slope, the
  same calibration allowance gloaming's OLS gets.
* ``zero`` — no gap; the floor any estimator must beat.

gloaming's model functions are imported from its clone and run unmodified. Their BTC/ETH input is
Bitget spot in the original; here it is Bitget's USDT perpetuals, whose hourly closes track spot
within a few basis points — disclosed, not believed to matter.

**What is scored for ARGUS (changed 2026-09-26).** Until then this module scored its own copy of
the formula, ``perp(09:00) / perp(16:00) - 1``, and `eval/standing.py` credited the console for
the result; the harness-validity canary (`eval/harness_validity.py`, `data/harness_validity.json`)
found the console code never ran at scoring time. Now every row calls the console's
``_implied_open_line`` itself, with its two readers pointed at the saved inputs
(:func:`console_feed`): `market.history.fetch`, through which ``_perp_at_close`` reads the
perpetual's close bar, and `market.equity_history.daily`, through which ``_yahoo_close`` reads the
stock's regular close. Everything between them is the console's: the holiday-aware session clock
that picks the last regular close (``_last_regular_close``), the exact-bar selection, the
arithmetic and the printed sentence. The implied price is printed to the cent, so the scored gap
carries up to half a cent of rounding (under 0.5bps on every name here); the former formula is
kept beside it as ``argus_perp_formula`` so the difference is on the record. A night on which the
console prints no line (its close bar missing, say) is scored for no candidate, and is counted in
``console_replay``.
"""

from __future__ import annotations

import bisect
import contextlib
import itertools
import json
import re
import sys
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from argus.eval import artefact
from argus.eval.compare import ComparisonReport, finalise, legacy_outcome, paired_bootstrap

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
              "argus_perp_vs_close", "argus_perp_fitted", "argus_perp_plus_futures",
              "argus_perp_formula")

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
# The console, replayed on the saved inputs.

@contextlib.contextmanager
def console_feed(hourly: Mapping[str, Series],
                 closes: Mapping[str, Sequence[tuple[date, float]]]) -> Iterator[None]:
    """Point the console's two market readers at saved inputs while the block runs.

    ``hourly`` maps a Bitget symbol to ``(bar end as a UTC timestamp, close)`` (this module's
    :data:`Series`); it is served through `argus.market.history.fetch` as hourly
    :class:`~argus.market.history.Candle` rows stamped with their open, the shape the live
    endpoint returns. ``closes`` maps a stock ticker to ``(session day, regular close)``, served
    through `argus.market.equity_history.daily`. Both are module attributes the console looks up
    at call time (``_perp_at_close`` and ``_yahoo_close`` import them inside the function), so
    nothing in `lui/research.py` is replaced: only where its data comes from. The console's
    instrument registry (`market.universe.contracts`, which ``_implied_open_line`` asks whether a
    name is an equity) is served from its frozen snapshot, so a replay never depends on what
    Bitget lists today. Anything but a bounded hourly window is refused rather than invented, and
    the originals are restored however the block exits."""
    from argus.market import equity_history, history, universe

    ends = {symbol: [end for end, _ in series] for symbol, series in hourly.items()}

    def fetch(symbol: str, *, interval: str = "1H", start: datetime | None = None,
              end: datetime | None = None, **_: Any) -> list[history.Candle]:
        if interval != "1H" or start is None or end is None:
            raise history.HistoryError(f"the replay serves bounded 1H windows, not {interval}")
        stamps = ends.get(symbol) or []
        lo = bisect.bisect_left(stamps, start.timestamp() + 3600)
        hi = bisect.bisect_right(stamps, end.timestamp() + 3600)
        out = []
        for bar_end, close in (hourly.get(symbol) or [])[lo:hi]:
            price = Decimal(repr(close))
            out.append(history.Candle(ts=datetime.fromtimestamp(bar_end - 3600, UTC),
                                      open=price, high=price, low=price, close=price,
                                      volume=Decimal(0)))
        return out

    def daily(ticker: str, **_: Any) -> list[equity_history.Day]:
        return [equity_history.Day(day=day, open=close, close=close)
                for day, close in closes.get(ticker) or []]

    registry, _ = universe._from_snapshot()

    def contracts() -> dict[str, universe.Contract]:
        return registry

    real_fetch, real_daily, real_contracts = history.fetch, equity_history.daily, universe.contracts
    history.fetch = fetch
    equity_history.daily = daily
    universe.contracts = contracts
    try:
        yield
    finally:
        history.fetch = real_fetch
        equity_history.daily = real_daily
        universe.contracts = real_contracts


_IMPLIED = re.compile(r"puts the stock near (?P<implied>[\d,]+\.\d+) at the next open "
                      r"\(last close (?P<close>[^)]+)\)")


def console_implied_open(symbol: str, perp_last: float,
                         now: datetime) -> tuple[float, float, str] | None:
    """``(implied open, the last close it anchored on, the sentence)`` as the console prints them
    at ``now``, or None when it prints no implied-open line. Must run inside
    :func:`console_feed`. An exception from the console propagates: a harness that swallowed one
    would score nothing and still print a verdict. A sentence this cannot read is an error too,
    never a silently skipped row."""
    from argus.lui import research

    out = research._implied_open_line(symbol, Decimal(repr(perp_last)), now=now)
    if out is None:
        return None
    text = out[0]
    found = _IMPLIED.search(text)
    if found is None:
        raise ValueError(f"the console's implied-open sentence changed shape: {text[:200]}")
    return float(found["implied"].replace(",", "")), float(found["close"]), text


def _session_closes(inputs: Mapping[str, Any]) -> dict[str, list[tuple[date, float]]]:
    return {stock: [(date.fromisoformat(d["day"]), float(d["close"])) for d in days]
            for stock, days in (inputs.get("sessions") or {}).items()}


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
    """One row per (stock, adjacent session pair) with every input read at its moment, and the
    console's implied open at that moment (``console_gap``; None where it printed no line)."""
    hourly = {k: [(float(t), float(c)) for t, c in v] for k, v in inputs["hourly"].items()}
    with console_feed(hourly, _session_closes(inputs)):
        return _nights(inputs, hourly)


def _nights(inputs: dict[str, Any], hourly: dict[str, Series]) -> list[dict[str, Any]]:
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
            console = console_implied_open(f"{stock}USDT", float(reads["perp_now"] or 0.0),
                                           datetime.fromtimestamp(predict_at, UTC))
            rows.append({
                "stock": stock, "closed": d0.isoformat(), "opened": d1.isoformat(),
                "gap": after["open"] / before["close"] - 1,
                "close": before["close"], **reads,
                "crypto": mean(c for c in crypto if c is not None),
                "crypto_24h": mean(c for c in crypto_24h if c is not None),
                "console_gap": console[0] / before["close"] - 1 if console else None,
                "console_anchor": console[1] if console else None,
                "console_line": console[2] if console else None,
            })
    return rows


def predict(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every candidate's gap estimate per row, calibrated candidates walking forward per stock."""
    import pandas as pd

    model = _gloaming()
    by_stock: dict[str, list[dict[str, Any]]] = {}
    # Only the nights the console answered, so every candidate is scored on the same rows.
    for row in sorted((r for r in rows if r.get("console_gap") is not None),
                      key=lambda r: (r["stock"], r["opened"])):
        by_stock.setdefault(row["stock"], []).append(row)
    out = []
    for stock_rows in by_stock.values():
        for i, row in enumerate(stock_rows):
            past = stock_rows[:i]
            fx = -row["dxy"]
            perp_move = row["console_gap"]
            if len(past) >= MIN_FIT:
                weights = model.calibrate_weights(
                    pd.Series([p["gap"] for p in past]),
                    pd.Series([p["futures"] for p in past]),
                    pd.Series([p["crypto"] for p in past]),
                    pd.Series([-p["dxy"] for p in past]))
                xs = [p["console_gap"] for p in past]
                denominator = sum(x * x for x in xs)
                slope = (sum(x * p["gap"] for x, p in zip(xs, past, strict=True)) / denominator
                         if denominator else 1.0)
            else:
                weights, slope = dict(PRIOR), 1.0
            combined = perp_move
            if len(past) >= MIN_FIT:
                import numpy as np

                design = np.array([[p["console_gap"], p["futures"] - p["console_gap"]]
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
                # The formula this harness scored before it called the console, kept so the
                # difference between that copy and the console is on the record.
                "argus_perp_formula": row["perp_now"] / row["perp_close"] - 1,
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
    news, so the night, not the row, is the independent unit.

    The resampling itself is the spine's (`eval/compare.py::paired_bootstrap`) since 2026-09-25;
    it is this function's former loop moved there unchanged, and `tests/test_eval_spine.py` pins
    every interval in the pre-migration artefact."""
    by_night: dict[str, list[float]] = {}
    for r in rows:
        diff = (abs(r["estimates"][a] - r["gap"]) - abs(r["estimates"][b] - r["gap"])) * 1e4
        by_night.setdefault(r["opened"], []).append(diff)
    boot = paired_bootstrap(by_night, draws=draws, seed=seed)
    return {"a": a, "b": b, "mean_diff_bps": round(boot.mean_diff, 2),
            "ci95_bps": [round(boot.low, 2), round(boot.high, 2)], "nights": boot.units,
            "verdict": boot.legacy_verdict}


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
    with console_feed(hourly, _session_closes(inputs)):
        for stock in UNIVERSE:
            perp = hourly.get(f"{stock}USDT") or []
            sessions = inputs["sessions"].get(stock) or []
            for before, after in itertools.pairwise(sessions):
                d0, d1 = date.fromisoformat(before["day"]), date.fromisoformat(after["day"])
                if d0.weekday() != 4 or d1.weekday() != 0:
                    continue
                closed = _at(perp, _ny(d0, time(16)))
                evening = _ny(date.fromordinal(d1.toordinal() - 1), time(20))
                sunday = _at(perp, evening)
                if not closed or not sunday:
                    continue
                # The console's implied open at that moment, not this module's arithmetic.
                console = console_implied_open(f"{stock}USDT", sunday,
                                               datetime.fromtimestamp(evening, UTC))
                if console is None:
                    continue
                rows.append({"stock": stock, "closed": d0.isoformat(),
                             "opened": d1.isoformat(),
                             "gap": after["open"] / before["close"] - 1,
                             "estimates": {"last_close": 0.0,
                                           "perp_weekend": console[0] / before["close"] - 1}})
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


def _console_replay(all_rows: list[dict[str, Any]],
                    scored: list[dict[str, Any]]) -> dict[str, Any]:
    """How the console's printed implied open compares with the formula this harness used to
    score in its place, and which nights the console declined."""
    declined = [r for r in all_rows if r.get("console_gap") is None]
    diffs = [abs(r["estimates"]["argus_perp"] - r["estimates"]["argus_perp_formula"]) * 1e4
             for r in scored]
    mismatched = [r for r in all_rows if r.get("console_anchor") is not None
                  and abs(r["console_anchor"] / r["close"] - 1) > 1e-5]
    return {
        "method": "lui/research._implied_open_line called per night at 09:00 New York, its "
                  "market readers pointed at data/h2h_gloaming/inputs.json (console_feed); the "
                  "gap is the printed implied price over the session's regular close",
        "nights_offered": len(all_rows),
        "nights_the_console_answered": len(all_rows) - len(declined),
        "nights_the_console_declined": [
            {"stock": r["stock"], "closed": r["closed"], "opened": r["opened"]}
            for r in declined],
        "anchor_differs_from_the_session_close": [
            {"stock": r["stock"], "closed": r["closed"], "console_anchor": r["console_anchor"],
             "session_close": r["close"]} for r in mismatched],
        "max_abs_diff_vs_former_formula_bps": round(max(diffs), 3) if diffs else None,
        "mean_abs_diff_vs_former_formula_bps": round(mean(diffs), 4) if diffs else None,
        "sample_line": next((r["console_line"] for r in all_rows if r.get("console_line")),
                            None),
    }


def score(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    inputs = inputs or json.loads((DATA / "inputs.json").read_text("utf-8"))
    offered = nights(inputs)
    rows = predict(offered)
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
    report: dict[str, Any] = {
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
        "console_replay": _console_replay(offered, scored),
        "scope": ("Rows before a stock's 11th night are excluded from every candidate so the "
                  "walk-forward fits are out of sample on the same rows as the fixed ones. "
                  "gloaming's crypto input is Bitget USDT perpetuals, not spot."),
    }
    report["comparison_reports"] = [r.to_dict() for r in comparison_reports(report)]
    return report


def comparison_reports(report: dict[str, Any]) -> list[ComparisonReport]:
    """This harness's verdicts in the spine's shape (`eval/compare.py`), read from its own report.

    A pure function of the report dict, so it reads a pre-migration artefact exactly as it reads a
    fresh one — which is how `tests/test_eval_spine.py` proves the migration changed no verdict.
    The outcome is taken from the bootstrap's own verdict string (computed on unrounded bounds),
    never re-derived from the rounded interval printed beside it.
    """
    summary, best = report["summary"], report["best_gloaming"]
    paired = {(p["a"], p["b"]): p for p in report["paired"]}
    total = report["out_of_sample"]["rows_scored"] + report["out_of_sample"]["rows_excluded"]
    groups = {stock: legacy_outcome(row["argus_perp_vs_best_gloaming"]["verdict"],
                                    argus_is_a=True).value
              for stock, row in report["per_stock"].items()}
    out: list[ComparisonReport] = []
    for rival, rival_label, question, per_group in (
            (best, f"gloaming ({best}), its best variant", "the stock's overnight gap", groups),
            ("zero", "no gap (the last close)", "the stock's overnight gap: the floor", {})):
        block = paired[("argus_perp", rival)]
        out.append(finalise(ComparisonReport(
            comparison="overnight", question=question, argus="argus_perp", rival=rival_label,
            metric="mean absolute error of the predicted close-to-open gap, bps",
            lower_is_better=True, argus_score=summary["argus_perp"]["mae_bps"],
            rival_score=summary[rival]["mae_bps"], n=block["nights"], unit="night",
            outcome=legacy_outcome(block["verdict"], argus_is_a=True),
            basis="paired bootstrap over nights, 4000 draws, seed 7; ARGUS is the console's "
                  "_implied_open_line replayed on the saved inputs; ARGUS better if the whole "
                  "95% interval of (ARGUS - rival) absolute error is below zero",
            ci95=(block["ci95_bps"][0], block["ci95_bps"][1]),
            scored=report["scored_rows"], total=total, groups=per_group,
            artefact="data/overnight_comparison.json", created_at=report["generated"])))
    weekend = report.get("sunday_evening_vs_nocturne_claim")
    if weekend:
        block = weekend["paired"]
        out.append(finalise(ComparisonReport(
            comparison="overnight", question="the Monday open from Sunday 20:00 New York",
            argus="perp_weekend", rival="nocturne's claim: the last regular close",
            metric="mean absolute error of the predicted Friday-close-to-Monday-open gap, bps",
            lower_is_better=True,
            argus_score=weekend["summary"]["perp_weekend"]["mae_bps"],
            rival_score=weekend["summary"]["last_close"]["mae_bps"], n=block["nights"],
            unit="weekend", outcome=legacy_outcome(block["verdict"], argus_is_a=True),
            basis="paired bootstrap over weekends, 4000 draws, seed 7",
            ci95=(block["ci95_bps"][0], block["ci95_bps"][1]), scored=weekend["rows"],
            total=weekend["rows"], artefact="data/overnight_comparison.json",
            created_at=report["generated"])))
    return out


def main() -> int:  # pragma: no cover - CLI
    if "--collect" in sys.argv:
        collect()
    report = score()
    artefact.write(REPORT, report)
    print(json.dumps({k: report.get(k) for k in ("scored_rows", "nights", "summary",
                                                 "paired")}, indent=1))
    for r in report.get("comparison_reports", []):
        print(f"{r['rival']}: {r['outcome']} (valid {r['valid']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
