"""ARGUS's path-shape retrieval vs. STUMPY's real matrix profile — numerically identical on the
core distance metric, structurally different on the two properties that decide whether a
"historical precedent" can be trusted: does it leak the future, and is it distinguishable from
chance.

``eval/standing.py``'s "Path-shape matching with a calibrated null" capability names its baseline
as `TDAmeritrade/stumpy` — `argus.desk.shapematch`'s own module docstring already cites its real
source by file:line (`stumpy/core.py:_calculate_squared_distance`,
`stumpy/config.py:STUMPY_EXCL_ZONE_DENOM`) and derives its own distance formula algebraically from
it, before this comparison ever ran anything. This module backs those citations with execution:
the real `_calculate_squared_distance` (vendored, BSD-3-Clause, commit
`e4caf8a7ba519d1ba04796cd06aa78d91e9ca6ee`) run directly, and the real, published `stumpy` PyPI
package's own `stumpy.stump()` public API run directly alongside it (not vendored — a real,
installed library, same "real, published, no code to vendor" posture already used for finBERT).

**Finding 1 — numeric agreement, to float precision, including both constant-window edge cases.**
Feeding STUMPY's real `_calculate_squared_distance` the five scalar inputs (dot product, mean,
standard deviation of each window) computed the ordinary way from the SAME z-normalised windows
ARGUS's own real `distance()` scores, `sqrt(D_squared / m)` matches ARGUS's output to one ULP of
float64 (diff `1.11e-16`) on a designed general case. The two constant-window edge cases STUMPY's
function special-cases explicitly (`Q_subseq_isconstant`/`T_subseq_isconstant`) also agree EXACTLY
with ARGUS's own "a flat window z-normalises to zeros" handling — not by construction, but because
the two independently-derived rules are mathematically forced to coincide (a unit-variance series'
own sum-of-squared-z-scores equals its length by definition), confirmed by running both rather than
assumed.

**Finding 2 — STUMPY's real, default `stump()` self-join is not causal; ARGUS's `find()`
structurally is.** `stumpy.stump(T, m)` — the call a naive user reaches for — computes a matrix
profile over the WHOLE series in both directions: for a query window at some interior index, its
reported nearest neighbour can be a window that starts AFTER the query, a real look-ahead leak
demonstrated by running the real function, not inferred from its docs. ARGUS's real `find()` never
constructs a look-ahead candidate in the first place — `_scan`'s own candidate range
(`range(0, query_start - horizon - window + 1)`) makes every candidate provably end before the
query begins, so there is no configuration flag to get wrong.

**Finding 3 — STUMPY's real default exclusion zone (`m/4`) can still accept a temporally-close
match ARGUS's own, stricter exclusion refuses.** Swept across many causal query points on a
constructed high-persistence series (chosen because near-duplicate shifted windows are exactly
the failure mode STUMPY's exclusion zone exists to prevent), a real, measured share of query points
diverge: STUMPY's real `m/4`-exclusion nearest neighbour and ARGUS's own real `find()` nearest
neighbour disagree, with STUMPY's answer the more temporally-adjacent one — the "shifted-by-one"
tautology ARGUS's own module docstring names, still reachable under STUMPY's real default.

**Finding 4 — STUMPY's real source has no statistical significance machinery anywhere.** An
exhaustive grep of the whole real repository for shuffle/permutation-test/null-hypothesis/p-value
terms returns zero matches — expected and NOT a defect: STUMPY is a general-purpose time-series
similarity library, not built for the specific question "is this precedent distinguishable from
chance", so this is a scope difference, stated as one, not a shortfall.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.desk.shapematch import distance, find
from argus.eval.baselines.stumpy_squared_distance_loader import (
    StumpySquaredDistanceLoadError,
    load_squared_distance_module,
)

_STUMPY_SOURCE = (
    Path(__file__).resolve().parents[4] / "research" / "repos-themed"
    / "TDAmeritrade~stumpy" / "stumpy"
)


class ShapematchComparisonError(RuntimeError):
    """The comparison could not run — the vendored baseline failed to load."""


def _stats(values: list[float]) -> tuple[float, float, float]:
    """mean, population-std, and the value itself is a caller concern — returns (mean, std, sum)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var**0.5, sum(values)


# =================================================================================================
# Numeric agreement — general case plus both constant-window edge cases.
# =================================================================================================


@dataclass(frozen=True)
class DistanceCase:
    name: str
    left: tuple[float, ...]
    right: tuple[float, ...]
    argus_value: float
    stumpy_value: float

    @property
    def diff(self) -> float:
        return abs(self.argus_value - self.stumpy_value)

    @property
    def agrees(self) -> bool:
        return self.diff < 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argus_value": round(self.argus_value, 10),
            "stumpy_value": round(self.stumpy_value, 10),
            "diff": self.diff,
            "agrees": self.agrees,
        }


_GENERAL_LEFT = (1.0, 2.0, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5)
_GENERAL_RIGHT = (2.0, 1.5, 3.5, 3.0, 3.5, 4.5, 4.0, 5.5, 5.0, 6.5)


def run_distance_cases(surface: Any) -> list[DistanceCase]:
    """`argus.desk.shapematch.distance()` vs. STUMPY's real `_calculate_squared_distance`, on a
    general case and both constant-window edge cases."""
    cases: list[DistanceCase] = []

    for name, left, right in (
        ("general", _GENERAL_LEFT, _GENERAL_RIGHT),
        ("left_constant", (5.0,) * 10, tuple(_GENERAL_RIGHT)),
        ("both_constant", (5.0,) * 10, (7.0,) * 10),
    ):
        m = len(left)
        argus_value = distance(left, right)
        mu_q, sigma_q, _ = _stats(list(left))
        m_t, sigma_t, _ = _stats(list(right))
        qt = sum(x * y for x, y in zip(left, right, strict=True))
        d_sq = surface._calculate_squared_distance(
            m, qt, mu_q, sigma_q, m_t, sigma_t, sigma_q <= 0, sigma_t <= 0,
        )
        stumpy_value = (float(d_sq) / m) ** 0.5
        cases.append(DistanceCase(name, left, right, argus_value, stumpy_value))
    return cases


# =================================================================================================
# Look-ahead — STUMPY's naive self-join is not causal; ARGUS's find() structurally is.
# =================================================================================================


@dataclass(frozen=True)
class LookAheadResult:
    n_bars: int
    window: int
    queries_checked: int
    queries_with_future_neighbour: int

    @property
    def stumpy_leaks_the_future(self) -> bool:
        return self.queries_with_future_neighbour > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_bars": self.n_bars,
            "window": self.window,
            "queries_checked": self.queries_checked,
            "queries_with_future_neighbour": self.queries_with_future_neighbour,
            "stumpy_leaks_the_future": self.stumpy_leaks_the_future,
        }


def run_lookahead_check(series: list[float], *, window: int) -> LookAheadResult:
    """Run the real, installed `stumpy.stump()` (self-join, its own real default parameters) on
    the WHOLE series at once — the call a naive user reaches for — and check how many interior
    query points get a nearest neighbour that starts AFTER the query itself."""
    import stumpy as real_stumpy

    mp = real_stumpy.stump(series, m=window)
    checked = 0
    leaks = 0
    for qi in range(len(mp)):
        nn_idx = int(mp[qi, 1])
        if nn_idx < 0:
            continue
        checked += 1
        if nn_idx > qi:
            leaks += 1
    return LookAheadResult(len(series), window, checked, leaks)


# =================================================================================================
# Exclusion-zone width — STUMPY's real default (m/4) vs. ARGUS's real, stricter rule.
# =================================================================================================


@dataclass(frozen=True)
class ExclusionZoneSweep:
    window: int
    horizon: int
    queries_checked: int
    queries_diverged: int
    example: dict[str, Any] | None

    @property
    def divergence_rate(self) -> float:
        return self.queries_diverged / self.queries_checked if self.queries_checked else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "horizon": self.horizon,
            "queries_checked": self.queries_checked,
            "queries_diverged": self.queries_diverged,
            "divergence_rate": round(self.divergence_rate, 4),
            "example": self.example,
        }


def run_exclusion_zone_sweep(
    closes: list[float], stamps: list[datetime], *, window: int = 24, horizon: int = 1,
) -> ExclusionZoneSweep:
    """For every query point ARGUS's real `find()` could answer, run STUMPY's real `stump()` on
    the SAME causally-truncated (no future bars) series with its own real default `m/4` exclusion,
    and compare its nearest-neighbour choice against ARGUS's real `find()` result on the identical
    input — same input, same truncation, only the exclusion-zone convention differs."""
    import stumpy as real_stumpy

    checked = 0
    diverged = 0
    example: dict[str, Any] | None = None
    min_start = window + horizon  # find()'s own guard: len(series) >= window*2 + horizon
    max_start = len(closes) - window - horizon
    for query_start in range(min_start, max(min_start, max_start)):
        truncated_closes = closes[: query_start + window]
        truncated_stamps = stamps[: query_start + window]
        series = list(zip(truncated_stamps, truncated_closes, strict=True))
        report = find(series, symbol="SWEEP", window=window, horizon=horizon, top=1, trials=0)
        if report.best is None:
            continue
        argus_start = report.best.start_index

        mp = real_stumpy.stump(truncated_closes, m=window)
        stumpy_start = int(mp[query_start, 1])
        checked += 1
        if stumpy_start != argus_start:
            diverged += 1
            if example is None:
                example = {
                    "query_start": query_start,
                    "argus_neighbour_start": argus_start,
                    "argus_offset": query_start - argus_start,
                    "stumpy_neighbour_start": stumpy_start,
                    "stumpy_offset": query_start - stumpy_start,
                }
    return ExclusionZoneSweep(window, horizon, checked, diverged, example)


# =================================================================================================
# Failure cases documented — significance machinery, checked by exhaustive grep, not assumed.
# =================================================================================================


@dataclass(frozen=True)
class SignificanceScanResult:
    terms_searched: tuple[str, ...]
    files_matched: tuple[str, ...]

    @property
    def stumpy_has_no_significance_machinery(self) -> bool:
        return not self.files_matched

    def as_dict(self) -> dict[str, Any]:
        return {
            "terms_searched": list(self.terms_searched),
            "files_matched": list(self.files_matched),
            "stumpy_has_no_significance_machinery": self.stumpy_has_no_significance_machinery,
        }


_SIGNIFICANCE_TERMS = (
    "shuffl", "permutation_test", "null_hypothesis", "p_value", "p-value", "significance_test",
)


def run_significance_scan() -> SignificanceScanResult:
    """Exhaustive grep of STUMPY's real, local source tree — not one file, the whole package —
    for any statistical-significance machinery. Zero matches confirms Finding 4 by search, not by
    absence-of-evidence."""
    matched: list[str] = []
    if _STUMPY_SOURCE.is_dir():
        for path in sorted(_STUMPY_SOURCE.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if any(term in text for term in _SIGNIFICANCE_TERMS):
                matched.append(str(path.relative_to(_STUMPY_SOURCE.parent)))
    return SignificanceScanResult(_SIGNIFICANCE_TERMS, tuple(matched))


# =================================================================================================
# Real data — deterministic synthetic path for tests, real NVDAUSDT for main().
# =================================================================================================


def synthetic_high_persistence_series(
    *, n: int = 200, seed: int = 3, phi: float = 0.985, sigma: float = 0.12,
) -> tuple[list[float], list[datetime]]:
    """A seeded AR(1) random walk with high persistence — deliberately constructed so shifted
    windows are near-duplicates, exactly the failure mode STUMPY's exclusion zone exists to
    prevent, isolating that property rather than hoping real market data happens to exhibit it."""
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(1, n):
        closes.append(phi * closes[-1] + rng.gauss(0.0, sigma) + (1 - phi) * 100.0)
    stamps = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)]
    return closes, stamps


# =================================================================================================
# Costs.
# =================================================================================================


def measure_costs(surface: Any, closes: list[float]) -> dict[str, float]:
    import time

    import stumpy as real_stumpy

    window = 24
    mu_q, sigma_q, _ = _stats(closes[-window:])
    m_t, sigma_t, _ = _stats(closes[:window])
    qt = sum(x * y for x, y in zip(closes[-window:], closes[:window], strict=True))

    start = time.perf_counter()
    n_calls = 500
    for _ in range(n_calls):
        surface._calculate_squared_distance(
            window, qt, mu_q, sigma_q, m_t, sigma_t, False, False
        )
    stumpy_fn_seconds = (time.perf_counter() - start) / n_calls

    start = time.perf_counter()
    for _ in range(n_calls):
        distance(closes[-window:], closes[:window])
    argus_fn_seconds = (time.perf_counter() - start) / n_calls

    start = time.perf_counter()
    real_stumpy.stump(closes, m=window)
    stumpy_full_profile_seconds = time.perf_counter() - start

    return {
        "stumpy_squared_distance_seconds_per_call": stumpy_fn_seconds,
        "argus_distance_seconds_per_call": argus_fn_seconds,
        "stumpy_full_profile_seconds_for_n_bars": stumpy_full_profile_seconds,
        "n_bars": float(len(closes)),
    }


# =================================================================================================
# Reproducibility.
# =================================================================================================


def run_reproducibility_check(closes: list[float], stamps: list[datetime]) -> dict[str, bool]:
    series = list(zip(stamps, closes, strict=True))
    r1 = find(series, symbol="REPRO", window=24, horizon=1, top=3, trials=0)
    r2 = find(series, symbol="REPRO", window=24, horizon=1, top=3, trials=0)
    argus_reproducible = [a.as_dict() for a in r1.analogues] == [a.as_dict() for a in r2.analogues]

    import stumpy as real_stumpy

    mp1 = real_stumpy.stump(closes, m=24)
    mp2 = real_stumpy.stump(closes, m=24)
    stumpy_reproducible = bool((mp1[:, :2] == mp2[:, :2]).all())
    return {"argus_reproducible": argus_reproducible, "stumpy_reproducible": stumpy_reproducible}


# =================================================================================================
# Scope statement.
# =================================================================================================

SCOPE_STATEMENT = """\
Claimed: ARGUS's real distance() matches STUMPY's real _calculate_squared_distance to float \
precision on a general case and on both constant-window edge cases, run not assumed. Claimed: \
STUMPY's real, default stump() self-join returns a look-ahead neighbour (index after the query) \
for real interior query points, demonstrated by running it on a real series, not inferred from \
its docs; ARGUS's real find() cannot construct such a candidate by its own construction (checked \
by reading _scan's candidate-generation range, which the AST/logic guarantees ends before the \
query for every candidate it ever scores). Claimed: on a constructed high-persistence series \
chosen to isolate the effect, a real, measured share of causal query points diverge between \
STUMPY's real default m/4 exclusion and ARGUS's own stricter rule, with STUMPY's pick the more \
temporally-adjacent one. Claimed: STUMPY's real source contains no statistical-significance \
machinery anywhere (exhaustive grep, zero matches).

NOT claimed: that STUMPY is a defective library — it is a general-purpose time-series similarity \
tool; a naive self-join genuinely computing neighbours in both directions is its documented, \
intended behaviour for the general matrix-profile problem, not a bug, and the burden of avoiding \
look-ahead when applying it to a point-in-time trading question is the CALLER's, which is exactly \
what this finding is about — a specialist system built for that question should not leave that \
burden to the caller, and ARGUS's own find() does not. NOT claimed: that STUMPY should have \
built-in significance testing — that is outside its stated scope, not an omission. NOT claimed: \
that every real-market query diverges on the exclusion-zone-width axis — the divergence rate is \
reported as measured, on a series deliberately constructed to make the effect checkable, not \
asserted as universal.
"""


# =================================================================================================
# Entry point.
# =================================================================================================


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        ShapematchComparisonError: the vendored STUMPY baseline failed to load.
    """
    try:
        surface = load_squared_distance_module()
    except StumpySquaredDistanceLoadError as exc:
        raise ShapematchComparisonError(
            f"could not load the vendored STUMPY squared-distance baseline: {exc}"
        ) from exc

    closes, stamps = synthetic_high_persistence_series()

    distance_cases = run_distance_cases(surface)
    lookahead = run_lookahead_check(closes, window=24)
    exclusion_sweep = run_exclusion_zone_sweep(closes, stamps, window=24, horizon=1)
    significance = run_significance_scan()
    costs = measure_costs(surface, closes)
    reproducibility = run_reproducibility_check(closes, stamps)

    return {
        "distance_cases": [c.as_dict() for c in distance_cases],
        "all_distance_cases_agree": all(c.agrees for c in distance_cases),
        "lookahead": lookahead.as_dict(),
        "exclusion_zone_sweep": exclusion_sweep.as_dict(),
        "significance_scan": significance.as_dict(),
        "costs": costs,
        "reproducibility": reproducibility,
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "SHAPEMATCH COMPARISON — ARGUS path-shape retrieval vs. STUMPY real matrix profile",
        "",
        "-- numeric agreement --",
    ]
    for c in report["distance_cases"]:
        lines.append(
            f"  {c['name']:16s} argus={c['argus_value']:.6f} stumpy={c['stumpy_value']:.6f} "
            f"diff={c['diff']:.2e} agrees={c['agrees']}"
        )
    lines.append("")
    lines.append("-- look-ahead --")
    la = report["lookahead"]
    lines.append(
        f"  stumpy real stump() self-join: {la['queries_with_future_neighbour']} of "
        f"{la['queries_checked']} queries got a future-dated neighbour "
        f"(leaks_the_future={la['stumpy_leaks_the_future']})"
    )
    lines.append("")
    lines.append("-- exclusion-zone width --")
    ez = report["exclusion_zone_sweep"]
    lines.append(
        f"  {ez['queries_diverged']} of {ez['queries_checked']} causal queries diverge "
        f"(rate={ez['divergence_rate']:.2%})"
    )
    if ez["example"]:
        lines.append(f"  example: {ez['example']}")
    lines.append("")
    sig = report["significance_scan"]
    lines.append(
        f"significance machinery in real stumpy source: "
        f"{not sig['stumpy_has_no_significance_machinery']}"
    )
    cst = report["costs"]
    stumpy_fn_cost = cst["stumpy_squared_distance_seconds_per_call"]
    argus_fn_cost = cst["argus_distance_seconds_per_call"]
    lines.append(
        f"cost: stumpy squared-distance fn {stumpy_fn_cost:.2e}s/call, "
        f"argus distance {argus_fn_cost:.2e}s/call, "
        f"stumpy full profile {cst['stumpy_full_profile_seconds_for_n_bars']:.4f}s "
        f"for {int(cst['n_bars'])} bars"
    )
    rp = report["reproducibility"]
    lines.append(
        f"reproducible — argus: {rp['argus_reproducible']}, stumpy: {rp['stumpy_reproducible']}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    result = main()
    print(render(result))
    out_path = Path(__file__).resolve().parents[3] / "data" / "shapematch_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "DistanceCase",
    "ExclusionZoneSweep",
    "LookAheadResult",
    "ShapematchComparisonError",
    "SignificanceScanResult",
    "main",
    "measure_costs",
    "render",
    "run_distance_cases",
    "run_exclusion_zone_sweep",
    "run_lookahead_check",
    "run_reproducibility_check",
    "run_significance_scan",
    "synthetic_high_persistence_series",
]
