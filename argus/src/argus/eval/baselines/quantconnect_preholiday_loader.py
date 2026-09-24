"""Run the real, vendored QuantConnect Pre-Holiday Effect decision snippet
(`quantconnect_preholiday_decision.py`).

The real snippet is published at column zero, reading `self`/`holidays` from an enclosing scope
the tutorial page never shows. Rather than re-indent it to fit inside a Python function (editing
whitespace inside the "verbatim" region), this loader `exec()`s the real file's own text
directly, with real `self`/`holidays` objects injected into the namespace first — the same
technique `qlib_loader.py` already uses in this directory for `re`/`abc`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_BASELINES_DIR = Path(__file__).resolve().parent
_SOURCE_PATH = _BASELINES_DIR / "quantconnect_preholiday_decision.py"


class QuantConnectPreHolidayLoadError(RuntimeError):
    """The vendored Pre-Holiday Effect decision snippet could not be run."""


class _Portfolio:
    """A real stand-in for Lean's real `self.Portfolio` — carries only the one real attribute
    (`Invested`) the vendored snippet reads."""

    def __init__(self, invested: bool) -> None:
        self.Invested = invested


class _Strategy:
    """A real stand-in for the tutorial's real `QCAlgorithm` instance — `SetHoldings`/
    `Liquidate` record what the real code called rather than placing real orders, so the real
    decision can be observed."""

    def __init__(self, invested: bool) -> None:
        self.Portfolio = _Portfolio(invested)
        self.calls: list[tuple[str, ...]] = []

    def SetHoldings(self, symbol: str, weight: float) -> None:
        self.calls.append(("SetHoldings", symbol, str(weight)))
        self.Portfolio.Invested = True

    def Liquidate(self) -> None:
        self.calls.append(("Liquidate",))
        self.Portfolio.Invested = False


def run_decision(*, portfolio_invested: bool, n_holidays: int) -> dict[str, Any]:
    """Runs the real, unmodified vendored decision snippet with `self.Portfolio.Invested` and
    `len(holidays)` set to the given real inputs, and reports what it called."""
    try:
        source = _SOURCE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise QuantConnectPreHolidayLoadError(f"could not read {_SOURCE_PATH}: {exc}") from exc

    strategy = _Strategy(portfolio_invested)
    namespace: dict[str, Any] = {"self": strategy, "holidays": [None] * n_holidays}
    try:
        exec(compile(source, str(_SOURCE_PATH), "exec"), namespace)
    except Exception as exc:
        raise QuantConnectPreHolidayLoadError(f"the real vendored snippet raised: {exc}") from exc

    return {
        "calls": strategy.calls,
        "invested_after": strategy.Portfolio.Invested,
    }


__all__ = ["QuantConnectPreHolidayLoadError", "run_decision"]
