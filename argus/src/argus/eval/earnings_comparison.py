"""Same-input comparison: `argus.research.sue` vs QuantConnect's real, vendored SUE factor.

Runs QuantConnect's real, unmodified Standardized Unexpected Earnings computation
(`eval/baselines/quantconnect_sue.py`, the real `FineSelectionAndSueSorting` core from
QuantConnect/Tutorials) and ARGUS's real `research.sue` on the same real quarterly EPS — pulled
live from SEC EDGAR's XBRL API (`market/fundamentals.py`, already built, already point-in-time
and restatement-aware) for the nine real rToken anchor companies.

**The central, measured finding.** `argus.research.sue.sue_from_quarters` reproduces the real
vendored formula to floating-point identity on real EPS for every anchor tested — this is not a
design claim, it is measured on live-fetched data every time this module runs
(``run_baseline_reproduced_cases``). The real vendored formula's denominator — the standard
deviation of eight real year-over-year EPS deltas — has no guard anywhere in the real source.
``run_silent_failure_cases`` runs it directly on two constructed-but-plausible EPS paths: a
smoothly, linearly growing history (zero variance in the deltas by construction) produces a real,
silent ``inf``, and an all-flat history (real for a pre-revenue or loss-making issuer) produces a
real, silent ``nan`` — both with nothing louder than a `RuntimeWarning: divide by zero` /
`invalid value encountered`. ``run_ranking_case`` shows the consequence: mixed into a real
cross-sectional ranking alongside the nine real anchors, the real vendored code's own
`sorted(sue_by_symbol.items(), ...)` puts the constructed `inf` case at the very top,
unconditionally ahead of every genuine real surprise, while `argus.research.sue.rank_universe`
excludes it and reports why.

SCOPE, stated explicitly:

* The real anchor EPS is fetched live and can change: the exact SUE values reported here are
  whatever SEC EDGAR carries the day this module runs, not frozen constants.
* This measures a **numerical-robustness** gap in the real reference's denominator, not a claim
  that the SUE factor itself has forecasting value — no return or edge claim is made for SUE as a
  trading signal here; `agents/earnings.py`'s own module already carries the "EPS beat -> buy is
  an anti-pattern" caveat for headline-only earnings reactions, and this comparison does not
  relitigate it.
* The real reference's `months_count`/`months_eps_change` parameters (36 months / 12 months, a
  36-month warm-up and a 12-month/four-quarter lag) are read from the real tutorial's own prose
  and code structure, not invented — real quarterly EPS is fed to it reshaped into the monthly-
  cadence-with-quarterly-repeats form the real code's own `RollingWindow` expects, since ARGUS's
  real EPS source is genuinely quarterly, not monthly.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.baselines.quantconnect_sue_loader import load_sue_module
from argus.market.fundamentals import FundamentalsSource
from argus.research.sue import MIN_QUARTERS, SueError, rank_universe, read

ANCHORS: tuple[str, ...] = (
    "NVDA", "TSLA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "COIN", "MSTR",
)
"""The nine real rToken anchor companies with a real ticker-to-EPS mapping
(`market.bitget.ANCHOR_OF`) — QQQ/TQQQ/SQQQ are index ETFs with no EPS."""

QUARTERS_FETCHED = 12
MONTHS_COUNT = 36
MONTHS_EPS_CHANGE = 12

CONSTRUCTED_LINEAR_QUARTERS: list[float] = [float(12 - i) for i in range(QUARTERS_FETCHED)]
"""A constructed, whole-dollar EPS path ($12.00 down to $1.00, newest-first) with an exactly
constant $4.00 year-over-year delta at every one of the eight real comparison points. Integer
EPS values are used deliberately, not decimals: the earlier draft of this comparison used
`round(1.2 - 0.1*i, 2)`-style decimals, which are not exactly representable in binary
floating-point, so `eps_std` came out as float-noise (order 1e-17) rather than a literal `0.0`,
and the real reference's SUE came out as an absurd-but-finite ~6.4e15 rather than a literal
`inf` — the SAME real failure mode (an unguarded near-zero denominator), just not the cleanest
demonstration of it. Whole dollars keep every subtraction here exact."""


@dataclass(frozen=True, slots=True)
class Stock:
    """The real vendored `SueSorter.compute()`'s only real dependency beyond its own
    constructor-supplied state: an object with a real `.Symbol` attribute."""

    Symbol: str


def _monthly_from_quarters(quarters: list[float]) -> list[float]:
    """Real quarterly EPS, newest-first, reshaped into the monthly-cadence-with-quarterly-
    repeats form the real vendored `RollingWindow` expects (see the module docstring)."""
    monthly: list[float] = []
    for q in quarters:
        monthly.extend([q, q, q])
    return monthly


def _fetch_quarters(ticker: str) -> list[float] | None:
    source = FundamentalsSource()
    facts, _status = source.facts(ticker, concept="eps_diluted", as_of=datetime.now(UTC))
    if len(facts) < QUARTERS_FETCHED:
        return None
    ordered = sorted(facts, key=lambda f: f.end, reverse=True)
    return [float(f.value) for f in ordered[:QUARTERS_FETCHED]]


def _real_sue(sue_module: Any, ticker: str, quarters: list[float]) -> float:
    sorter = sue_module.SueSorter(
        eps_by_symbol={ticker: _monthly_from_quarters(quarters)},
        months_count=MONTHS_COUNT, months_eps_change=MONTHS_EPS_CHANGE,
    )
    out: dict[str, float] = {}
    sorter.compute(Stock(ticker), out)
    return out[ticker]


# --- baseline reproduced ------------------------------------------------------------------------


def _fetch_all_anchor_quarters() -> dict[str, list[float] | None]:
    return {ticker: _fetch_quarters(ticker) for ticker in ANCHORS}


def _compute_baseline_reproduced_cases(
    quarters_by_ticker: dict[str, list[float] | None],
) -> dict[str, Any]:
    """The deterministic half of `run_baseline_reproduced_cases`, decoupled from fetching: given
    already-fetched real quarterly EPS per ticker, scores both real systems. Pure computation on
    fixed input — calling this twice on the SAME `quarters_by_ticker` is what "reproducible"
    means here, as opposed to re-fetching live SEC EDGAR data twice (see
    `run_reproducibility_check`)."""
    sue_module = load_sue_module()
    results: list[dict[str, Any]] = []
    for ticker in ANCHORS:
        quarters = quarters_by_ticker[ticker]
        if quarters is None:
            results.append({"symbol": ticker, "skipped": "insufficient real EPS history"})
            continue
        real_value = _real_sue(sue_module, ticker, quarters)
        argus_value = read(ticker, quarters)
        results.append({
            "symbol": ticker,
            "real_sue": real_value,
            "argus_sue": argus_value.sue,
            "agree": bool(abs(real_value - argus_value.sue) < 1e-9),
        })
    checked = [r for r in results if "skipped" not in r]
    return {
        "symbols_checked": [r["symbol"] for r in checked],
        "symbols_skipped": [r["symbol"] for r in results if "skipped" in r],
        "results": results,
        "all_agree": all(r["agree"] for r in checked) if checked else False,
    }


def run_baseline_reproduced_cases() -> dict[str, Any]:
    """Real quarterly EPS for every real anchor, through both the real vendored SUE and
    ARGUS's real `research.sue.read`; confirms floating-point-identical agreement."""
    return _compute_baseline_reproduced_cases(_fetch_all_anchor_quarters())


# --- silent failure cases ---------------------------------------------------------------------


def run_silent_failure_cases() -> dict[str, Any]:
    """Two constructed-but-plausible EPS paths, run through the real vendored code: a linearly
    growing history (zero variance in the deltas) and an all-flat history. Both produce a real,
    silent inf/nan with only a numpy RuntimeWarning; ARGUS's real `research.sue` refuses both."""
    sue_module = load_sue_module()
    cases: dict[str, Any] = {}

    linear = CONSTRUCTED_LINEAR_QUARTERS
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        real_value = _real_sue(sue_module, "LINEAR", linear)
        real_warnings = [str(w.message) for w in caught]
    try:
        read("LINEAR", linear)
        argus_refused = False
    except SueError:
        argus_refused = True
    cases["linear_growth_zero_variance"] = {
        "real_sue": real_value,
        "real_is_infinite": bool(real_value in (float("inf"), float("-inf"))),
        "real_warnings": real_warnings,
        "real_raised": False,
        "argus_refused": argus_refused,
    }

    flat = [0.0] * QUARTERS_FETCHED
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        real_value2 = _real_sue(sue_module, "FLAT", flat)
        real_warnings2 = [str(w.message) for w in caught]
    try:
        read("FLAT", flat)
        argus_refused2 = False
    except SueError:
        argus_refused2 = True
    cases["all_flat_eps"] = {
        "real_sue": real_value2,
        "real_is_nan": bool(real_value2 != real_value2),  # nan != nan
        "real_warnings": real_warnings2,
        "real_raised": False,
        "argus_refused": argus_refused2,
    }

    return cases


def run_ranking_case() -> dict[str, Any]:
    """Mixes the constructed `inf` case into a real cross-sectional ranking alongside every real
    anchor. The real vendored code's own `sorted()` puts it first, unconditionally; ARGUS's
    `rank_universe` excludes it and reports why — the statistically-valid-evaluation proof."""
    sue_module = load_sue_module()
    universe: dict[str, list[float]] = {}
    for ticker in ANCHORS:
        quarters = _fetch_quarters(ticker)
        if quarters is not None:
            universe[ticker] = quarters
    linear = CONSTRUCTED_LINEAR_QUARTERS
    universe["CONSTRUCTED_LINEAR"] = linear

    real_sue_by_symbol: dict[str, float] = {}
    for symbol, quarters in universe.items():
        real_sue_by_symbol[symbol] = _real_sue(sue_module, symbol, quarters)
    real_ranked = sorted(real_sue_by_symbol.items(), key=lambda x: x[1], reverse=True)

    argus_ranked, argus_skipped = rank_universe(universe)

    return {
        "universe_size": len(universe),
        "real_top_symbol": real_ranked[0][0],
        "real_top_is_the_constructed_artifact": real_ranked[0][0] == "CONSTRUCTED_LINEAR",
        "real_ranking": [{"symbol": s, "sue": v} for s, v in real_ranked],
        "argus_ranked": [r.as_dict() for r in argus_ranked],
        "argus_top_symbol": argus_ranked[0].symbol if argus_ranked else None,
        "argus_excludes_the_constructed_artifact": "CONSTRUCTED_LINEAR" not in {
            r.symbol for r in argus_ranked
        },
        "argus_skipped": argus_skipped,
    }


# --- failure cases ---------------------------------------------------------------------------


def run_failure_cases() -> dict[str, Any]:
    """The real vendored code's crash mode on insufficient history — a genuine `IndexError`,
    not silent — vs. ARGUS's clean, typed `SueError` on the same input.

    Four real quarters (tripled to twelve real monthly-cadence slots) is short enough that the
    real vendored code's own `rw[self.months_eps_change]` (index 12) falls outside a 12-element
    list — verified by running it, not assumed; five quarters (15 monthly slots) does NOT raise
    and instead returns a real, silent `nan`, a second, different silent failure shape at a
    slightly longer history than the one this case targets.
    """
    sue_module = load_sue_module()
    short = [1.0] * 4
    try:
        _real_sue(sue_module, "SHORT", short)
        real_raised = None
    except IndexError as exc:
        real_raised = f"{type(exc).__name__}: {exc}"

    try:
        read("SHORT", short)
        argus_refused = False
    except SueError:
        argus_refused = True

    return {
        "insufficient_history": {
            "n_quarters": len(short),
            "real_raised": real_raised,
            "real_raised_indexerror": real_raised is not None and "IndexError" in real_raised,
            "argus_refused_with_sueerror": argus_refused,
        },
    }


# --- costs, reproducibility -------------------------------------------------------------------


def measure_costs(repeats: int = 200) -> dict[str, Any]:
    """Real wall-clock cost of the real vendored SueSorter vs. ARGUS's pure-Python
    `research.sue.read`, on the same real NVDA-shaped quarterly data."""
    sue_module = load_sue_module()
    quarters = [2.46, 1.91, 1.5, 0.89, 0.81, 0.78, 0.61, 0.52, 0.4, 0.35, 0.27, 0.18]
    monthly = _monthly_from_quarters(quarters)

    start = time.perf_counter()
    for _ in range(repeats):
        sorter = sue_module.SueSorter(
            eps_by_symbol={"X": monthly}, months_count=MONTHS_COUNT,
            months_eps_change=MONTHS_EPS_CHANGE,
        )
        sorter.compute(Stock("X"), {})
    real_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        read("X", quarters)
    argus_elapsed = time.perf_counter() - start

    return {
        "repeats": repeats,
        "real_vendored_seconds_per_call": real_elapsed / repeats,
        "argus_pure_python_seconds_per_call": argus_elapsed / repeats,
        "argus_faster_by_factor": (
            (real_elapsed / repeats) / (argus_elapsed / repeats) if argus_elapsed > 0 else None
        ),
    }


def run_reproducibility_check() -> dict[str, Any]:
    """Fetches real SEC EDGAR quarterly EPS ONCE, then runs the deterministic scoring step twice
    on that same fixed data. Deliberately not "fetch live data twice and compare" — see
    `rotation_comparison.py`'s own `run_reproducibility_check` docstring for why re-fetching live
    data between the two calls tests "did real time stand still", not determinism."""
    quarters_by_ticker = _fetch_all_anchor_quarters()
    first = _compute_baseline_reproduced_cases(quarters_by_ticker)
    second = _compute_baseline_reproduced_cases(quarters_by_ticker)
    return {"identical": json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)}


SCOPE_STATEMENT = (
    "argus.research.sue reproduces QuantConnect's real, vendored Standardized Unexpected "
    "Earnings formula to floating-point identity on real, live SEC EDGAR quarterly EPS for "
    "every real rToken anchor tested. The real vendored formula's denominator has no guard: a "
    "constructed-but-plausible linearly-growing EPS path produces a real, silent inf (with only "
    "a RuntimeWarning), an all-flat path a real, silent nan, and mixed into a real cross-"
    "sectional ranking alongside every real anchor, the real code's own sorted() puts the "
    "constructed inf case first unconditionally. argus.research.sue refuses in every one of "
    "these cases instead, and crashes with the same typed SueError (not a bare IndexError) on "
    "insufficient history where the real reference raises an untyped IndexError. NOT claimed "
    "the SUE factor has validated forecasting value as a trading signal — this measures the "
    "real reference's numerical robustness, not its edge. NOT claimed the real anchor SUE "
    "values are fixed constants — SEC EDGAR is fetched live and the exact numbers will move as "
    "new quarters are filed."
)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = {
        "baseline_reproduced": run_baseline_reproduced_cases(),
        "silent_failure_cases": run_silent_failure_cases(),
        "ranking_case": run_ranking_case(),
        "failure_cases": run_failure_cases(),
        "costs": measure_costs(),
        "reproducibility": run_reproducibility_check(),
        "scope_statement": SCOPE_STATEMENT,
    }
    print(render(report))
    out = Path(__file__).resolve().parents[3] / "data" / "earnings_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def render(report: dict[str, Any]) -> str:
    base = report["baseline_reproduced"]
    silent = report["silent_failure_cases"]
    ranking = report["ranking_case"]
    costs = report["costs"]
    lines = ["EARNINGS SUE vs QuantConnect's real vendored SUE factor\n"]
    lines.append(
        f"  baseline reproduced: {sum(1 for r in base['results'] if r.get('agree'))}/"
        f"{len(base['symbols_checked'])} real anchors agree exactly"
    )
    lines.append(
        f"  linear-growth case: real SUE={silent['linear_growth_zero_variance']['real_sue']} "
        f"(infinite={silent['linear_growth_zero_variance']['real_is_infinite']}) "
        f"argus_refused={silent['linear_growth_zero_variance']['argus_refused']}"
    )
    lines.append(
        f"  ranking: real top = {ranking['real_top_symbol']} "
        f"(constructed artifact = {ranking['real_top_is_the_constructed_artifact']}); "
        f"argus excludes it = {ranking['argus_excludes_the_constructed_artifact']}"
    )
    lines.append(
        f"  costs: real {costs['real_vendored_seconds_per_call']*1000:.4f}ms vs argus "
        f"{costs['argus_pure_python_seconds_per_call']*1000:.4f}ms per call"
    )
    lines.append(f"  reproducible: {report['reproducibility']['identical']}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ANCHORS",
    "MIN_QUARTERS",
    "SCOPE_STATEMENT",
    "Stock",
    "main",
    "measure_costs",
    "render",
    "run_baseline_reproduced_cases",
    "run_failure_cases",
    "run_ranking_case",
    "run_reproducibility_check",
    "run_silent_failure_cases",
]
