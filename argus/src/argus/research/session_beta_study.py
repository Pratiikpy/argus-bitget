"""Beta to the index, split by whether the anchor market was open — measured, then written down.

**Why this module exists.** `data/session_beta.json` is cited across the docs, and
`desk/portfolio.py`'s own docstring describes it as *"reproducible"* — while **nothing in the
codebase could produce it**. The arithmetic was there (`portfolio.session_betas`), the clock was
there (`truth.clocks.DualClock`), and the step that joins them and saves the answer was not. A file
called reproducible that no command regenerates is the most expensive kind of claim, because it
reads as settled.

**The finding it carries.** Open-session beta exceeds shut-session beta on almost every rToken, and
roughly 82% of hourly bars fall while the anchor market is shut. So a single blended beta is
dominated by the quiet session and **understates open-session risk** — which is exactly when a
position is most likely to be hurt. That is the reason this project refuses to publish one number
where the market has two.

**What it deliberately does not do.** It does not fill a session with the other session's beta when
a symbol has too few open-hours bars to estimate one. `portfolio.beta` returns ``None`` there and
that ``None`` is carried through to the artefact as an absence, not as a zero and not as the
blended figure wearing a different label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.desk.portfolio import Session, leverage_consistency, session_betas
from argus.market.instruments import LeverageCheck

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "session_beta.json"

BENCHMARK = "QQQUSDT"
"""The index leg. Every rToken in the universe is a US equity or an index on one, so QQQ is the
common factor — and it is itself in the universe, which is why it is excluded from its own study."""

LOOKBACK_DAYS = 30
INTERVAL = "1H"

MIN_SESSION_BARS = 20
"""Below this a session's beta is not estimated at all.

A beta from a handful of bars is a number with an error bar wider than the number. The artefact
records the count beside every estimate so a reader can apply their own threshold, but this module
will not publish one it does not believe.
"""


class SessionBetaError(RuntimeError):
    """Raised rather than publishing a beta computed from a series that does not line up."""


@dataclass(frozen=True, slots=True)
class SymbolBeta:
    """One rToken's exposure to the index, split by session."""

    symbol: str
    observations: int
    beta_blended: float | None
    beta_open: float | None
    n_open: int
    beta_shut: float | None
    n_shut: int
    leverage: LeverageCheck | None = None
    """Declared-vs-measured, for a leveraged instrument. ``None`` for anything not declared
    leveraged in `market.instruments.REGISTRY`, or when no blended beta was estimated."""

    @property
    def open_exceeds_shut(self) -> bool:
        """True only when both were estimated. An absence is not a comparison."""
        if self.beta_open is None or self.beta_shut is None:
            return False
        return self.beta_open > self.beta_shut

    def as_dict(self) -> dict[str, Any]:
        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 4)

        return {
            "symbol": self.symbol,
            "observations": self.observations,
            "beta_blended": rounded(self.beta_blended),
            "beta_open": rounded(self.beta_open),
            "n_open": self.n_open,
            "beta_shut": rounded(self.beta_shut),
            "n_shut": self.n_shut,
            "leverage": None if self.leverage is None else self.leverage.as_dict(),
        }


def _returns(closes: list[tuple[datetime, float]]) -> dict[datetime, float]:
    """Bar-to-bar returns, keyed by the *closing* stamp of the bar that produced them."""
    out: dict[datetime, float] = {}
    for i in range(1, len(closes)):
        stamp, close = closes[i]
        prior = closes[i - 1][1]
        if prior <= 0:
            raise SessionBetaError(
                f"a close of {prior} at {closes[i - 1][0]} is not a price; a zero return read from "
                f"it would pull every beta estimated over it toward the benchmark"
            )
        out[stamp] = close / prior - 1.0
    return out


def measure(
    symbol: str,
    asset_closes: list[tuple[datetime, float]],
    benchmark_closes: list[tuple[datetime, float]],
    *,
    is_open: Any,
) -> SymbolBeta:
    """One symbol's session-split beta against the benchmark."""
    asset = _returns(asset_closes)
    benchmark = _returns(benchmark_closes)
    shared = set(asset) & set(benchmark)
    if not shared:
        raise SessionBetaError(
            f"{symbol}: the two series share no timestamps, so no beta can be estimated between "
            f"them. Comparing unaligned bars would produce a number with no meaning"
        )

    exposures = session_betas(asset, benchmark, symbol=symbol, is_open=is_open)

    def value(session: Session) -> tuple[float | None, int]:
        got = exposures.get(session)
        if got is None:
            return None, 0
        count = int(getattr(got, "observations", 0) or 0)
        beta = getattr(got, "beta", None)
        if beta is None or count < MIN_SESSION_BARS:
            return None, count
        return float(beta), count

    blended, n_blended = value(Session.BLENDED)
    open_beta, n_open = value(Session.OPEN)
    shut_beta, n_shut = value(Session.SHUT)

    return SymbolBeta(
        symbol=symbol,
        observations=n_blended or len(shared),
        beta_blended=blended,
        beta_open=open_beta,
        n_open=n_open,
        beta_shut=shut_beta,
        n_shut=n_shut,
        leverage=leverage_consistency(exposures),
    )


def report_of(rows: list[SymbolBeta]) -> dict[str, Any]:
    """The artefact, in the shape the docs already cite."""
    comparable = [r for r in rows if r.beta_open is not None and r.beta_shut is not None]
    higher = [r for r in comparable if r.open_exceeds_shut]
    total_open = sum(r.n_open for r in rows)
    total_bars = sum(r.n_open + r.n_shut for r in rows)
    shut_share = 100.0 * (total_bars - total_open) / total_bars if total_bars else 0.0

    finding = (
        f"open-session beta exceeds shut-session beta on {len(higher)} of {len(comparable)} "
        f"rTokens; the blended figure is dominated by the ~{shut_share:.0f}% of bars that fall "
        f"while the anchor market is shut, so a single beta understates open-session risk"
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": BENCHMARK,
        "interval": INTERVAL,
        "lookback_days": LOOKBACK_DAYS,
        "min_session_bars": MIN_SESSION_BARS,
        "symbols_compared": len(comparable),
        "symbols_higher_open": len(higher),
        "shut_bar_share_pct": round(shut_share, 1),
        "finding": finding,
        "rows": [r.as_dict() for r in rows],
    }


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from argus.eval.clearance import UNIVERSE
    from argus.market.history import CandleType, fetch_range
    from argus.truth.clocks import DualClock

    parser = argparse.ArgumentParser(description="session-split beta to the index, per rToken")
    parser.add_argument("--benchmark", default=BENCHMARK)
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--save", action="store_true", help=f"write {REPORT_PATH.name}")
    args = parser.parse_args()

    def closes(symbol: str) -> list[tuple[datetime, float]]:
        candles = fetch_range(symbol, days=args.days, interval=INTERVAL,
                              candle_type=CandleType.MARKET)
        return [(c.ts, float(c.close)) for c in candles]

    clock = DualClock()

    def is_open(stamp: datetime) -> bool:
        return bool(clock.phase(stamp).has_price_discovery)

    bench = closes(args.benchmark)
    print(f"benchmark {args.benchmark}: {len(bench)} bars over {args.days}d\n")

    rows: list[SymbolBeta] = []
    for symbol in UNIVERSE:
        if symbol == args.benchmark:
            continue  # a benchmark's beta to itself is 1.0 and says nothing
        got = measure(symbol, closes(symbol), bench, is_open=is_open)
        rows.append(got)

        def show(value: float | None) -> str:
            return "  n/a " if value is None else f"{value:+6.3f}"

        print(
            f"  {symbol:<10} blended {show(got.beta_blended)}   "
            f"open {show(got.beta_open)} (n={got.n_open:>4})   "
            f"shut {show(got.beta_shut)} (n={got.n_shut:>4})"
        )

    blob = report_of(rows)
    print(f"\n  {blob['finding']}")

    if args.save:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"\n  written to {REPORT_PATH}")
    else:
        print("\n  (not saved — pass --save)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BENCHMARK",
    "INTERVAL",
    "LOOKBACK_DAYS",
    "MIN_SESSION_BARS",
    "REPORT_PATH",
    "SessionBetaError",
    "SymbolBeta",
    "main",
    "measure",
    "report_of",
]
