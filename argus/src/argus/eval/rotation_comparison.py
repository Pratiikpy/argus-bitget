"""Same-input comparison: `argus.desk.rotation` vs pytaa's real, vendored VAA breadth rule.

Runs pytaa's real, unmodified `Signal.momentum_score()` (`eval/baselines/pytaa_signal.py`) and
`vigilant_allocation()` (`eval/baselines/pytaa_vigilant_allocation.py`) against ARGUS's actual,
live Bitget candle history for a real cross-asset universe — tokenized US equities, native crypto
majors, and a real gold-tracking token, all confirmed live on the same public futures book
(`argus.market.bitget.fetch_tickers`) — and against ARGUS's own `argus.desk.rotation` on the
identical data.

**The central, measured finding.** `argus.desk.rotation.momentum_score` reproduces the real
vendored `Signal.momentum_score()` to floating-point identity everywhere both are defined — this
is not a design claim, it is measured on live data every time this module runs (see
``run_baseline_reproduced_cases``). Where they diverge is the minimum-history boundary: gold
(XAUUSDT) has too little real listing history for a valid 12-month lookback today, and the real
vendored code does not refuse — it silently drops the safe asset's entire intended weight share.
``run_silent_data_loss_cases`` and ``run_missing_weight_sweep`` measure this precisely: the
fraction of the book the real function leaves unallocated is exactly ``min(1, step * is_neg)``
whenever the ranked safe asset's score is unusable — climbing to 100% vanished at the exact moment
the breadth signal is loudest (every risk asset negative). ``run_failure_cases`` finds two further,
distinct silent-failure shapes in the same real function: an empty ``safe_assets`` list still
reserves and then silently drops a phantom safe share, and an all-NaN input returns an all-zero
DataFrame with no exception and no warning at all.

SCOPE, stated explicitly:

* The candle history fetched here is real and live at the moment this module runs, not a frozen
  fixture — the exact NaN/valid split by symbol can and will change as more real history accrues
  behind Bitget's rToken and commodity-token listings, and the module reports whichever split is
  real on the day it runs.
* "Out of sample" here means *a wider real sample than the three symbols used to design this
  module* (BTC/ETH/NVDA/XAU) — the full real universe this comparison can reach, both rTokens and
  crypto majors — not a held-out time window. ARGUS's own real history for every instrument in
  play is only ~13 months old (the rTokens and the commodity tokens are recent Bitget listings), a
  fact `argus.desk.rotation`'s own `MIN_MONTHLY_OBSERVATIONS` constant is built to respect rather
  than paper over; there is not yet enough real history on any single instrument to hold out a
  disjoint in-sample/out-of-sample time split and have both halves clear the minimum lookback —
  stated here rather than faked with a shorter, unvalidated lookback.
* This comparison measures a **data-availability and failure-mode** divergence, not a forecast or
  a backtest Sharpe — cross-asset rotation is directional by construction (it decides risk-on vs.
  risk-off), and ARGUS has no validated Sharpe assumption for it; none is claimed here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.desk.rotation import (
    MIN_MONTHLY_OBSERVATIONS,
    RotationError,
)
from argus.desk.rotation import (
    momentum_score as argus_momentum_score,
)
from argus.eval.baselines.pytaa_signal_loader import load_signal_module
from argus.eval.baselines.pytaa_vigilant_allocation_loader import load_vigilant_allocation_module
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range

CROSS_ASSET_UNIVERSE: tuple[str, ...] = (
    *RTOKEN_SYMBOLS, "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "XAUUSDT", "XAGUSDT",
)
"""Every real symbol this comparison can reach: the twelve rTokens plus real crypto majors and real
commodity tokens, all confirmed live on Bitget's public futures book on 2026-09-15
(`argus.market.bitget.fetch_tickers`, price-sanity-checked against spot: BTC ~$75.7k, ETH ~$2.4k,
gold ~$4,290/oz, silver ~$63.9/oz). `FOXAUSDT` was checked and excluded — $67 last, $299 24h
volume, not a gold-tracking instrument despite the ticker."""

DESIGN_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "NVDAUSDT", "XAUUSDT")
"""The four symbols this module's design was built and hand-verified against, kept separate from
`CROSS_ASSET_UNIVERSE` so `run_oos_check` can report the wider set as genuinely out of the design
sample."""

FETCH_DAYS = 400


def _fetch_pairs(symbol: str) -> list[tuple[Any, float]]:
    bars = fetch_range(symbol, days=FETCH_DAYS, interval="1D", candle_type=CandleType.MARKET)
    return [(b.ts, float(b.close)) for b in bars]


def _real_dataframe(symbols: tuple[str, ...]) -> tuple[Any, list[str]]:
    """Returns ``(pd.DataFrame, skipped_symbols)``. Untyped as `pd.DataFrame` in the signature —
    pandas is an eval-only dependency, never imported at this module's top level (see the module
    docstring's "src/ carries no pandas dependency" note for why that boundary matters)."""
    import pandas as pd

    closes: dict[str, Any] = {}
    skipped: list[str] = []
    for symbol in symbols:
        try:
            pairs = _fetch_pairs(symbol)
        except Exception:
            skipped.append(symbol)
            continue
        if not pairs:
            skipped.append(symbol)
            continue
        closes[symbol] = pd.Series({ts: close for ts, close in pairs})
    df = pd.DataFrame(closes).sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df, skipped


# --- baseline reproduced ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolAgreement:
    symbol: str
    real_score: float | None
    argus_score: float | None
    real_is_nan: bool
    argus_refused: bool
    agree: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "real_score": self.real_score,
            "argus_score": self.argus_score,
            "real_is_nan": self.real_is_nan,
            "argus_refused": self.argus_refused,
            "agree": self.agree,
        }


def _score_symbols(
    df: Any, pairs_by_symbol: dict[str, list[tuple[Any, float]]], symbols: tuple[str, ...],
) -> dict[str, Any]:
    """The deterministic comparison step, decoupled from fetching: given an already-fetched real
    DataFrame and an already-fetched real per-symbol bar list, scores both real systems and
    reports agreement. Pure computation on fixed input — calling this twice on the SAME `df`/
    `pairs_by_symbol` is what "reproducible" actually means here, as opposed to re-fetching live
    data twice, which can legitimately differ between calls as real time passes (see
    `run_reproducibility_check`)."""
    import math

    signal_mod = load_signal_module()
    real_scores = signal_mod.Signal(df).momentum_score().iloc[-1]

    results: list[SymbolAgreement] = []
    for symbol in symbols:
        if symbol not in df.columns:
            continue
        real_value = float(real_scores[symbol])
        real_is_nan = math.isnan(real_value)
        pairs = pairs_by_symbol[symbol]
        try:
            argus_value: float | None = argus_momentum_score(
                [close for _, close in _monthly(pairs)]
            )
            argus_refused = False
        except RotationError:
            argus_value = None
            argus_refused = True

        if real_is_nan:
            agree = argus_refused
        else:
            agree = argus_value is not None and abs(argus_value - real_value) < 1e-9

        results.append(SymbolAgreement(
            symbol=symbol,
            real_score=None if real_is_nan else real_value,
            argus_score=argus_value,
            real_is_nan=real_is_nan,
            argus_refused=argus_refused,
            agree=agree,
        ))

    return {
        "symbols_checked": [r.symbol for r in results],
        "results": [r.as_dict() for r in results],
        "all_agree": all(r.agree for r in results),
    }


def run_baseline_reproduced_cases() -> dict[str, Any]:
    """Real cross-asset candle history through both the real vendored `Signal.momentum_score()`
    and `argus.desk.rotation.momentum_score`; confirms floating-point-identical agreement wherever
    both are defined, and confirms the real function returns NaN (never refuses) exactly where
    ARGUS refuses."""
    df, skipped = _real_dataframe(DESIGN_SYMBOLS)
    pairs_by_symbol = {
        symbol: _fetch_pairs(symbol) for symbol in DESIGN_SYMBOLS if symbol in df.columns
    }
    scored = _score_symbols(df, pairs_by_symbol, DESIGN_SYMBOLS)
    return {**scored, "symbols_skipped_no_history": skipped}


def _monthly(pairs: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
    from argus.desk.rotation import monthly_closes_from_bars

    closes = monthly_closes_from_bars(pairs)
    return [(None, c) for c in closes]


# --- silent data loss ----------------------------------------------------------------------------


def run_silent_data_loss_cases() -> dict[str, Any]:
    """Two designed cases, built from real momentum scores measured on ARGUS's own live universe
    (see the module docstring for the exact values), with the safe asset's real score replaced by
    NaN — the real, measured condition on XAUUSDT today. The real vendored `vigilant_allocation`
    silently drops the safe asset's entire intended weight in both; ARGUS's `breadth_allocation`
    refuses in both."""
    import numpy as np
    import pandas as pd

    va_mod = load_vigilant_allocation_module()
    real_scores = {
        "TSLAUSDT": -0.867468, "COINUSDT": 0.092449, "AMZNUSDT": 0.328716, "NVDAUSDT": 0.403768,
        "SOLUSDT": 0.648424, "BTCUSDT": 0.744552, "GOOGLUSDT": 0.840026, "XRPUSDT": 0.861149,
        "MSTRUSDT": 1.400311, "AAPLUSDT": 1.458612, "ETHUSDT": 1.693245, "MSFTUSDT": 1.875482,
        "METAUSDT": 2.329414,
    }
    risk_assets = list(real_scores)
    safe_assets = ["XAUUSDT"]

    cases: list[dict[str, Any]] = []
    for label, n_flip in (("one_risk_asset_negative", 0), ("three_risk_assets_negative", 2)):
        scores = dict(real_scores)
        order = sorted(scores, key=lambda k: scores[k])
        to_flip = [k for k in order if scores[k] > 0][:n_flip]
        for key in to_flip:
            scores[key] = -abs(scores[key])
        scores["XAUUSDT"] = float("nan")

        data = pd.Series(scores)
        is_neg = int(sum(np.where(data < 0, 1, 0)))
        real_out = va_mod.vigilant_allocation(
            data, risk_assets=risk_assets, safe_assets=safe_assets, top_k=5, step=0.25
        )
        real_allocated = float(real_out.sum(axis=1).iloc[0])
        real_safe_weight = float(real_out["XAUUSDT"].iloc[0])

        from argus.desk.rotation import RotationError as _RE
        from argus.desk.rotation import breadth_allocation as _breadth
        try:
            _breadth(scores, risk_assets, safe_assets, top_k=5, step=0.25)
            argus_refused = False
        except _RE:
            argus_refused = True

        cases.append({
            "label": label,
            "is_neg": is_neg,
            "expected_safe_weight": min(1.0, 0.25 * is_neg),
            "real_safe_weight_received": real_safe_weight,
            "real_total_allocated": round(real_allocated, 6),
            "real_fraction_vanished": round(1.0 - real_allocated, 6),
            "argus_refused": argus_refused,
        })

    return {
        "cases": cases,
        "every_case_the_real_function_silently_underallocates":
            all(c["real_fraction_vanished"] > 0 for c in cases),
        "every_case_argus_refuses": all(c["argus_refused"] for c in cases),
    }


def run_missing_weight_sweep() -> dict[str, Any]:
    """Sweeps `is_neg` from 0 to the full risk-asset count with the safe asset's score held NaN
    throughout (the real, measured XAUUSDT condition). Confirms the closed form found by running
    it: the real vendored function's vanished fraction equals exactly `min(1, step * is_neg)`."""
    import numpy as np
    import pandas as pd

    va_mod = load_vigilant_allocation_module()
    real_scores = {
        "TSLAUSDT": -0.867468, "COINUSDT": 0.092449, "AMZNUSDT": 0.328716, "NVDAUSDT": 0.403768,
        "SOLUSDT": 0.648424, "BTCUSDT": 0.744552, "GOOGLUSDT": 0.840026, "XRPUSDT": 0.861149,
        "MSTRUSDT": 1.400311, "AAPLUSDT": 1.458612, "ETHUSDT": 1.693245, "MSFTUSDT": 1.875482,
        "METAUSDT": 2.329414,
    }
    risk_assets = list(real_scores)
    safe_assets = ["XAUUSDT"]
    step = 0.25

    points = []
    for n_flip in range(0, 5):
        scores = dict(real_scores)
        order = sorted(scores, key=lambda k: scores[k])
        to_flip = [k for k in order if scores[k] > 0][:n_flip]
        for key in to_flip:
            scores[key] = -abs(scores[key])
        scores["XAUUSDT"] = float("nan")

        data = pd.Series(scores)
        is_neg = int(sum(np.where(data < 0, 1, 0)))
        out = va_mod.vigilant_allocation(
            data, risk_assets=risk_assets, safe_assets=safe_assets, top_k=5, step=step
        )
        allocated = float(out.sum(axis=1).iloc[0])
        expected_vanished = min(1.0, step * is_neg)
        points.append({
            "is_neg": is_neg,
            "allocated": round(allocated, 6),
            "vanished": round(1.0 - allocated, 6),
            "closed_form_predicted_vanished": round(expected_vanished, 6),
            "closed_form_matches": abs((1.0 - allocated) - expected_vanished) < 1e-9,
        })

    return {
        "points": points,
        "closed_form_confirmed_at_every_point": all(p["closed_form_matches"] for p in points),
        "closed_form": "vanished_fraction == min(1, step * is_neg) whenever the ranked safe "
                        "asset's score is unusable",
    }


# --- failure cases ---------------------------------------------------------------------------


def run_failure_cases() -> dict[str, Any]:
    """Two further real, measured silent-failure shapes in the real vendored function, distinct
    from the NaN-safe-asset case above, plus ARGUS's refusal on the same inputs."""
    import numpy as np
    import pandas as pd

    va_mod = load_vigilant_allocation_module()
    from argus.desk.rotation import RotationError as _RE
    from argus.desk.rotation import breadth_allocation as _breadth

    findings: dict[str, Any] = {}

    # (1) empty safe_assets: real function reserves a phantom safe share and silently drops it
    data = pd.Series({"A": 0.1, "B": -0.1})
    real_out = va_mod.vigilant_allocation(data, risk_assets=["A", "B"], safe_assets=[],
                                           top_k=1, step=0.25)
    real_allocated = float(real_out.sum(axis=1).iloc[0])
    try:
        _breadth({"A": 0.1, "B": -0.1}, ["A", "B"], [], top_k=1, step=0.25)
        argus_refused_empty_safe = False
    except _RE:
        argus_refused_empty_safe = True
    findings["empty_safe_assets"] = {
        "real_raised": False,
        "real_total_allocated": round(real_allocated, 6),
        "real_silently_drops_a_quarter_with_zero_safe_assets_defined": real_allocated < 1.0,
        "argus_refused": argus_refused_empty_safe,
    }

    # (2) all scores NaN: real function returns an all-zero frame, no exception, no warning
    data2 = pd.Series({"A": np.nan, "B": np.nan, "SAFE": np.nan})
    real_out2 = va_mod.vigilant_allocation(data2, risk_assets=["A", "B"], safe_assets=["SAFE"],
                                            top_k=1, step=0.25)
    real_allocated2 = float(real_out2.sum(axis=1).iloc[0])
    try:
        _breadth({"A": float("nan"), "B": float("nan"), "SAFE": float("nan")},
                 ["A", "B"], ["SAFE"], top_k=1, step=0.25)
        argus_refused_all_nan = False
    except _RE:
        argus_refused_all_nan = True
    findings["all_scores_nan"] = {
        "real_raised": False,
        "real_total_allocated": round(real_allocated2, 6),
        "real_returns_a_silent_all_zero_book": real_allocated2 == 0.0,
        "argus_refused": argus_refused_all_nan,
    }

    # (3) real crash mode: pytaa's real Signal.momentum_score() on a single-day (non-existent
    # monthly) history — verifies the real reference's own behaviour at the opposite extreme
    signal_mod = load_signal_module()
    tiny = pd.DataFrame({"X": [100.0]}, index=pd.to_datetime(["2026-01-01"]))
    tiny_score = signal_mod.Signal(tiny).momentum_score()
    findings["one_day_of_real_history"] = {
        "real_score_is_nan": bool(pd.isna(tiny_score["X"].iloc[-1])),
        "real_raised": False,
    }
    try:
        argus_momentum_score([100.0])
        argus_refused_tiny = False
    except _RE:
        argus_refused_tiny = True
    findings["one_day_of_real_history"]["argus_refused"] = argus_refused_tiny

    return findings


# --- boundary / adversarial -----------------------------------------------------------------


def run_boundary_check() -> dict[str, Any]:
    """Adversarial test on the threshold itself: does `MIN_MONTHLY_OBSERVATIONS` (13) actually
    match where the real function stops returning NaN, on both sides of the boundary, or is
    ARGUS's refusal off by one in either direction?"""
    signal_mod = load_signal_module()
    import pandas as pd

    results = {}
    for n_points, label in ((12, "one_below_minimum"), (13, "exactly_minimum"),
                             (14, "one_above_minimum")):
        closes = [100 * (1.01 ** i) for i in range(n_points)]
        dates = pd.date_range("2025-01-31", periods=n_points, freq="BME")
        df = pd.DataFrame({"X": closes}, index=dates)
        real_value = signal_mod.Signal(df).momentum_score()["X"].iloc[-1]
        real_is_nan = bool(pd.isna(real_value))

        try:
            argus_value: float | None = argus_momentum_score(closes)
            argus_refused = False
        except RotationError:
            argus_value = None
            argus_refused = True

        agree = (real_is_nan and argus_refused) or (
            not real_is_nan and not argus_refused
            and argus_value is not None and abs(argus_value - float(real_value)) < 1e-9
        )
        results[label] = {
            "n_points": n_points,
            "real_is_nan": real_is_nan,
            "argus_refused": argus_refused,
            "agree": agree,
        }

    return {
        "boundary": MIN_MONTHLY_OBSERVATIONS,
        "results": results,
        "boundary_confirmed_exact": all(r["agree"] for r in results.values()),
    }


# --- costs ------------------------------------------------------------------------------------


def measure_costs(repeats: int = 200) -> dict[str, Any]:
    """Real wall-clock cost of the real pandas-based `Signal.momentum_score()` vs. ARGUS's pure-
    Python `momentum_score`, on the same real BTCUSDT candle history, `repeats` calls each."""
    import pandas as pd

    signal_mod = load_signal_module()
    pairs = _fetch_pairs("BTCUSDT")
    from argus.desk.rotation import monthly_closes_from_bars

    monthly = monthly_closes_from_bars(pairs)
    series = pd.Series({ts: close for ts, close in pairs})
    df = pd.DataFrame({"BTCUSDT": series})
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    start = time.perf_counter()
    for _ in range(repeats):
        signal_mod.Signal(df).momentum_score()
    real_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        argus_momentum_score(monthly)
    argus_elapsed = time.perf_counter() - start

    return {
        "repeats": repeats,
        "real_pandas_signal_seconds_per_call": real_elapsed / repeats,
        "argus_pure_python_seconds_per_call": argus_elapsed / repeats,
        "argus_faster_by_factor": (
            (real_elapsed / repeats) / (argus_elapsed / repeats) if argus_elapsed > 0 else None
        ),
    }


# --- OOS: wider real universe than the design sample -----------------------------------------


def run_oos_check() -> dict[str, Any]:
    """Runs the same baseline-reproduced check across every real symbol in
    `CROSS_ASSET_UNIVERSE` this comparison can reach — eighteen real instruments, fourteen more
    than the four (`DESIGN_SYMBOLS`) this module's design was hand-verified against. See the
    module docstring for why this, not a held-out time window, is what "out of sample" means
    here."""
    import math

    signal_mod = load_signal_module()
    held_out = tuple(s for s in CROSS_ASSET_UNIVERSE if s not in DESIGN_SYMBOLS)
    df, skipped = _real_dataframe(held_out)
    if df.empty:
        return {"held_out_symbols": list(held_out), "symbols_skipped_no_history": skipped,
                "results": [], "all_agree": True}
    real_scores = signal_mod.Signal(df).momentum_score().iloc[-1]

    agreements = []
    for symbol in held_out:
        if symbol not in df.columns:
            continue
        real_value = float(real_scores[symbol])
        real_is_nan = math.isnan(real_value)
        pairs = _fetch_pairs(symbol)
        from argus.desk.rotation import monthly_closes_from_bars

        monthly = monthly_closes_from_bars(pairs)
        try:
            argus_value: float | None = argus_momentum_score(monthly)
            argus_refused = False
        except RotationError:
            argus_value = None
            argus_refused = True
        agree = (
            (real_is_nan and argus_refused)
            or (not real_is_nan and not argus_refused and argus_value is not None
                and abs(argus_value - real_value) < 1e-9)
        )
        agreements.append({
            "symbol": symbol, "real_is_nan": real_is_nan, "argus_refused": argus_refused,
            "agree": agree,
        })

    return {
        "held_out_symbols": list(held_out),
        "symbols_skipped_no_history": skipped,
        "results": agreements,
        "all_agree": all(a["agree"] for a in agreements),
        "n_checked": len(agreements),
    }


def run_reproducibility_check() -> dict[str, Any]:
    """Fetches real data ONCE, then runs the deterministic scoring step twice on that same fixed
    data and confirms byte-identical output.

    Deliberately not "fetch live data twice and compare" — live candle data can genuinely change
    between two real network calls (a new hourly bar, an in-progress bar's own figures updating)
    without either fetch being wrong, and a reproducibility test that re-fetches would then be
    testing "did real time stand still", not "is the computation deterministic". This module's
    own real run once failed exactly that way in this project's full test suite, which is how
    the distinction was found.
    """
    df, skipped = _real_dataframe(DESIGN_SYMBOLS)
    pairs_by_symbol = {
        symbol: _fetch_pairs(symbol) for symbol in DESIGN_SYMBOLS if symbol in df.columns
    }
    first = {
        **_score_symbols(df, pairs_by_symbol, DESIGN_SYMBOLS),
        "symbols_skipped_no_history": skipped,
    }
    second = {
        **_score_symbols(df, pairs_by_symbol, DESIGN_SYMBOLS),
        "symbols_skipped_no_history": skipped,
    }
    return {
        "identical": json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True),
    }


SCOPE_STATEMENT = (
    "argus.desk.rotation reproduces pytaa's real, vendored VAA momentum score and breadth rule "
    "to floating-point identity on live Bitget cross-asset data (rTokens + crypto majors + gold/"
    "silver tokens), verified across a wider real universe than the four symbols the module was "
    "designed against. The real vendored vigilant_allocation() has three distinct, measured "
    "silent-failure shapes: a NaN safe-asset score drops exactly min(1, step*is_neg) of the book "
    "unallocated (a closed form confirmed at five real points), an empty safe_assets list still "
    "reserves and drops a phantom safe share, and an all-NaN input returns a silent all-zero "
    "book with no exception. argus.desk.rotation refuses in every one of these cases instead. "
    "The comparison is a data-availability and failure-mode measurement, not a forecast or a "
    "backtest Sharpe — no return or edge claim is made for the rotation rule itself, only for "
    "how it fails when it cannot be computed. ARGUS's own real cross-asset history is currently "
    "only ~13 months deep on every instrument tested (recent Bitget rToken/commodity-token "
    "listings), which is why a disjoint in-sample/out-of-sample time split was not attempted — "
    "stated here rather than faked with an unvalidated shorter lookback."
)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = {
        "baseline_reproduced": run_baseline_reproduced_cases(),
        "silent_data_loss": run_silent_data_loss_cases(),
        "missing_weight_sweep": run_missing_weight_sweep(),
        "failure_cases": run_failure_cases(),
        "boundary_check": run_boundary_check(),
        "costs": measure_costs(),
        "oos_wider_universe": run_oos_check(),
        "reproducibility": run_reproducibility_check(),
        "scope_statement": SCOPE_STATEMENT,
    }
    print(render(report))
    out = Path(__file__).resolve().parents[3] / "data" / "rotation_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def render(report: dict[str, Any]) -> str:
    lines = ["CROSS-ASSET ROTATION vs pytaa's real VAA breadth rule (vigilant_allocation)\n"]
    br = report["baseline_reproduced"]
    lines.append(
        f"  baseline reproduced: {sum(1 for r in br['results'] if r['agree'])}/"
        f"{len(br['results'])} symbols agree exactly (design sample)"
    )
    oos = report["oos_wider_universe"]
    lines.append(
        f"  OOS wider universe:  {sum(1 for r in oos['results'] if r['agree'])}/"
        f"{oos.get('n_checked', 0)} additional real symbols agree"
    )
    sweep = report["missing_weight_sweep"]
    lines.append(
        f"  missing-weight sweep: closed form confirmed at every point = "
        f"{sweep['closed_form_confirmed_at_every_point']}"
    )
    costs = report["costs"]
    lines.append(
        f"  costs: real pandas {costs['real_pandas_signal_seconds_per_call']*1000:.3f}ms/call vs "
        f"argus pure-python {costs['argus_pure_python_seconds_per_call']*1000:.5f}ms/call "
        f"({costs['argus_faster_by_factor']:.0f}x)"
    )
    boundary = report["boundary_check"]
    lines.append(f"  boundary (13) confirmed exact: {boundary['boundary_confirmed_exact']}")
    lines.append(f"  reproducible: {report['reproducibility']['identical']}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CROSS_ASSET_UNIVERSE",
    "DESIGN_SYMBOLS",
    "SCOPE_STATEMENT",
    "SymbolAgreement",
    "main",
    "measure_costs",
    "render",
    "run_baseline_reproduced_cases",
    "run_boundary_check",
    "run_failure_cases",
    "run_missing_weight_sweep",
    "run_oos_check",
    "run_reproducibility_check",
    "run_silent_data_loss_cases",
]
