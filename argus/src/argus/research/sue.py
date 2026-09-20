"""Standardized Unexpected Earnings (SUE) — cross-sectional earnings surprise, refusing rather
than silently going infinite.

`agents/earnings.py`'s seven-way decomposition scores one print's surprise qualitatively, in
[-1, 1], from an Earnings Analyst's judgment — it has no statistically-standardized number
anywhere, and no cross-sectional ranking across a universe. SUE is the field's own answer to
"how surprising, compared to what": a quarter's EPS change from a year ago, standardized by the
volatility of that same company's own historical year-over-year EPS changes. It is the textbook
post-earnings-announcement-drift (PEAD) proxy this project's real, published reference implements.

**Read before written.** The reference is QuantConnect's own tutorial library (Apache-2.0),
`research/repos-t3/Tutorials/04 Strategy Library/355 Standardized Unexpected Earnings/03 Method.
html` (the real `FineSelectionAndSueSorting` method, vendored verbatim at
`eval/baselines/quantconnect_sue.py`): ``SUE = (EPS_q - EPS_q-4) / stdev(eight historical
EPS_q - EPS_q-4 deltas)``. :func:`sue_from_quarters` reproduces that exact formula in plain Python
(matching `population` standard deviation — `numpy.std`'s default `ddof=0` — via
`statistics.pstdev`, not the sample `stdev`, which would silently disagree in the last few
digits). Verified to reproduce the real vendored function's real output to floating-point
identity on real quarterly EPS pulled live from SEC EDGAR (`market/fundamentals.py`, already
built, already point-in-time and restatement aware) for all nine real rToken anchor companies —
`eval/earnings_comparison.py`.

**What this module adds that the reference does not.** The real vendored formula's denominator —
the standard deviation of eight real deltas — has no guard anywhere in the real source. Run
directly (`eval/earnings_comparison.py`), a constructed but entirely plausible smoothly-growing
EPS path (a constant year-over-year delta every quarter — zero variance by construction) produces
a real, silent ``inf`` from the real code, with nothing louder than a `RuntimeWarning:
divide by zero` that a real `sorted()` cross-sectional ranking would never see — an `inf` SUE
sorts first, unconditionally, ahead of every genuine surprise in the universe. An all-flat EPS
history (real for a pre-revenue or loss-making issuer reporting $0.00 every quarter) produces a
real, silent ``nan`` instead, which corrupts `sorted()`'s ordering guarantee outright (Python's
sort treats NaN comparisons as always-false). :func:`sue_from_quarters` raises
:class:`SueError` in both cases instead of returning a number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

LAG_QUARTERS = 4
"""SUE compares a quarter's EPS to the same quarter one year (four quarters) earlier."""

N_DELTAS = 8
"""The real reference's own choice: eight historical year-over-year deltas set the volatility
denominator (`quantconnect_sue.py`'s real `new_eps_list`/`old_eps_list` slicing, both length 8)."""

MIN_QUARTERS = LAG_QUARTERS + N_DELTAS
"""12 — the earliest delta needs a value four quarters before the eighth-oldest delta point."""

ZERO_VARIANCE_TOLERANCE = 1e-9
"""Below this, `eps_std` is treated as zero rather than compared to it exactly.

A naive ``eps_std == 0`` check is not enough: real EPS is reported to 2-4 decimal places (dollars
per share), and a constructed-but-plausible linearly-growing EPS path was found, by running this
exact check, to produce a real `eps_std` of order 1e-17 — pure binary floating-point rounding
noise from arithmetic that is mathematically exactly zero — which a bare equality check misses
entirely and which still explodes the ratio to an absurd, wrong-looking-but-finite value (order
1e15) instead of either the real reference's own silent `inf` or a clean refusal. 1e-9 is many
orders of magnitude below any real EPS figure's own reporting precision, so it can only ever
trigger on genuine (near-)zero variance, never on a real, meaningfully-varying earnings history.
"""


class SueError(ValueError):
    """Raised rather than returning a SUE computed from insufficient or degenerate history."""


@dataclass(frozen=True, slots=True)
class SueRead:
    """One symbol's SUE, with the inputs that produced it — never just the scalar."""

    symbol: str
    sue: float
    eps_change: float
    eps_std: float
    quarters_used: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sue": round(self.sue, 6),
            "eps_change": round(self.eps_change, 6),
            "eps_std": round(self.eps_std, 6),
            "quarters_used": self.quarters_used,
        }


def _compute(quarters: Sequence[float]) -> tuple[float, float, float]:
    """Returns ``(sue, eps_change, eps_std)``. Raises :class:`SueError` — never returns
    ``inf``/``nan`` — below :data:`MIN_QUARTERS` or when the eight real year-over-year deltas
    have zero variance within :data:`ZERO_VARIANCE_TOLERANCE`, which is the module docstring's
    own measured, real divergence from the reference."""
    if len(quarters) < MIN_QUARTERS:
        raise SueError(
            f"{len(quarters)} quarter(s) is below the {MIN_QUARTERS} SUE needs "
            f"({LAG_QUARTERS} lag quarters plus {N_DELTAS} historical deltas)"
        )
    deltas = [quarters[i] - quarters[i + LAG_QUARTERS] for i in range(N_DELTAS)]
    eps_change = deltas[0]
    eps_std = pstdev(deltas)
    if eps_std <= ZERO_VARIANCE_TOLERANCE:
        raise SueError(
            "the eight real year-over-year EPS deltas have zero (or float-noise) variance; SUE "
            "is undefined, not infinite — the real reference silently returns inf/nan here "
            "instead of refusing"
        )
    return eps_change / eps_std, eps_change, eps_std


def sue_from_quarters(quarters: Sequence[float]) -> float:
    """The real SUE formula (`eval/baselines/quantconnect_sue.py`), reproduced in plain Python.

    ``quarters`` must be real quarterly EPS, newest first — ``quarters[0]`` is the most recent
    quarter.
    """
    sue, _eps_change, _eps_std = _compute(quarters)
    return sue


def read(symbol: str, quarters: Sequence[float]) -> SueRead:
    """Full :class:`SueRead`, with the intermediate values a caller would otherwise have to
    recompute to audit the scalar."""
    sue, eps_change, eps_std = _compute(quarters)
    return SueRead(
        symbol=symbol, sue=sue, eps_change=eps_change, eps_std=eps_std,
        quarters_used=len(quarters),
    )


def rank_universe(
    quarters_by_symbol: Mapping[str, Sequence[float]],
) -> tuple[list[SueRead], dict[str, str]]:
    """Every symbol with enough real history, ranked by SUE descending — the cross-sectional
    ranking `agents/earnings.py` has no equivalent of. Returns ``(ranked, skipped)`` rather than
    silently dropping symbols that cannot be scored: ``skipped`` names every excluded symbol and
    why, the same discipline `eval/rotation_comparison.py`'s own `_real_dataframe` applies to a
    real fetch that cannot cover every requested symbol."""
    reads: list[SueRead] = []
    skipped: dict[str, str] = {}
    for symbol, quarters in quarters_by_symbol.items():
        try:
            reads.append(read(symbol, quarters))
        except SueError as exc:
            skipped[symbol] = str(exc)
    return sorted(reads, key=lambda r: r.sue, reverse=True), skipped


__all__ = [
    "LAG_QUARTERS",
    "MIN_QUARTERS",
    "N_DELTAS",
    "SueError",
    "SueRead",
    "rank_universe",
    "read",
    "sue_from_quarters",
]
