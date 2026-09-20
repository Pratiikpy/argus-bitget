"""rToken factor divergence — does a real, published factor behave differently computed on the
rToken's own traded price than on its native-stock reference?

Bitget's index series (`market.history.CandleType.INDEX`) tracks the underlying US equity; the
rToken's own `MARKET` series is what actually trades, 24/7, against a low-liquidity, retail-
dominated order book. Track 1's "rToken Factor Strategies" sub-theme names the resulting question
directly: "traditional factors behave differently under rToken's low-liquidity, retail-dominated
market structure" (example approaches: "rToken momentum vs native stock; mean reversion in
closed-market windows; factor divergence arbitrage").

**Why this, and not a re-derived formula.** Per this project's standing rule to read the best real
implementation before writing one: `research/architecture/alpha101-port.md` already catalogues
WorldQuant's real, published Alpha#101 formula #23 as directly expressible, today, in ARGUS's own
grammar (`research/grammar.py`) with no extension needed — a real, named, already-vendor-verified
factor, not one invented to make a point. :data:`ALPHA_23` reproduces it exactly:
``delta(high, 2) if mean(high, 20) < high else 0`` (`Alpha101.py:259-265`), a reversal signal that
fires only when the current bar's high breaks above its own 20-bar mean.

**Why the raw Expr, not `Signal`.** `grammar.Signal.evaluate` clamps to ``[-1, 1]`` — a real,
deliberate constraint for treating an expression as a position size (`Signal`'s own docstring: "An
unbounded number is not a position size"). That clamp is exactly wrong for this use: Alpha 23's
raw value is a price delta, commonly several dollars on these instruments, so clamping would
collapse almost every non-zero reading to exactly ``-1.0`` or ``+1.0`` and destroy the rank
information an Information Coefficient needs to say anything. `ALPHA_23` is evaluated as the bare
`IfElse` node, deliberately never wrapped in `Signal` here.

Cross-referenced against Froot & Dabora (1999), "How are stock prices affected by the location of
trade?", *Journal of Financial Economics* 53(2) — the canonical study of "twin shares" (Royal
Dutch/Shell, Unilever NV/PLC): identical cash-flow claims, two listings, and a real, measured
finding that the relative price between them is NOT a random walk and co-moves with the market
where relative trading activity concentrates. No open-source implementation of that paper's own
test exists (checked directly, not assumed — see `eval/factor_divergence_comparison.py`'s own
docstring for where this was searched); it is cited here as the academic grounding for *why* a
divergence would be expected between a 24/7, crypto-venue-traded rToken and its native-market
reference, not as a vendored code baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from argus.backtest.engine import Bar
from argus.market.history import Candle, CandleType, fetch_range
from argus.research.grammar import BinOp, Const, Delay, Field, IfElse, Ref, Window

ALPHA_23 = IfElse(
    BinOp("lt", Window("mean", 20, Ref(Field.HIGH)), Ref(Field.HIGH)),
    Delay(2, Ref(Field.HIGH)),
    Const(0.0),
)
"""WorldQuant Alpha#101 formula #23, real and published, reproduced exactly."""

MIN_WARMUP = 25
"""`mean(high, 20)` needs 20 real bars of history; `Delay(2)` needs 2 more. A small margin past
the theoretical minimum (22) so no reading sits exactly on the boundary."""


def candles_to_bars(candles: list[Candle]) -> list[Bar]:
    """Real OHLC candles, in the shape `research.grammar`'s evaluator reads them: `high`/`low`
    via `Bar.extra`, exactly the keys `Field.HIGH`/`Field.LOW` look up."""
    return [
        Bar(ts=c.ts, close=c.close, extra={"high": float(c.high), "low": float(c.low)})
        for c in candles
    ]


def fetch_bars(symbol: str, *, days: int, candle_type: CandleType) -> list[Bar]:
    """Real candles for `symbol`, fetched fresh from Bitget, converted to `Bar`s. `candle_type`
    selects MARKET (the rToken's own traded price) or INDEX (Bitget's published native-stock
    reference) — the same real series `market.history.fetch_basis` already draws from, fetched
    here as full OHLC rather than the single `market.history.BasisPoint` price."""
    candles = fetch_range(symbol, days=days, interval="1H", candle_type=candle_type)
    return candles_to_bars(candles)


def alpha23_series(bars: list[Bar]) -> list[float]:
    """Alpha 23's real value at every bar. Bars before `MIN_WARMUP` are included (the grammar's
    `Window`/`Delay` nodes degrade gracefully rather than raising) but a caller building a panel
    should slice from `MIN_WARMUP` onward — see `eval/factor_divergence_comparison.py`."""
    return [ALPHA_23.evaluate(bars, i) for i in range(len(bars))]


def forward_returns(bars: list[Bar], *, horizon: int = 1) -> list[float | None]:
    """The real, realised close-to-close return `horizon` bars ahead of each bar; `None` where
    the horizon runs past the end of the series (the last `horizon` bars) rather than a value
    that would silently claim knowledge of a future bar that was never fetched."""
    out: list[float | None] = []
    n = len(bars)
    for i in range(n):
        j = i + horizon
        if j >= n:
            out.append(None)
            continue
        c0 = float(bars[i].close)
        if c0 <= 0:
            out.append(None)
            continue
        out.append((float(bars[j].close) - c0) / c0)
    return out


@dataclass(frozen=True, slots=True)
class SymbolSeries:
    """One symbol's real, aligned factor and forward-return readings, for one candle type."""

    symbol: str
    timestamps: list[datetime]
    factor: list[float]
    forward_return_1h: list[float | None]


def symbol_series(
    symbol: str, *, days: int, candle_type: CandleType, horizon: int = 1
) -> SymbolSeries:
    """Fetch real candles for `symbol`/`candle_type` once, and compute Alpha 23 plus its
    forward return from that single real fetch — never re-fetching to compute the second."""
    bars = fetch_bars(symbol, days=days, candle_type=candle_type)
    factor = alpha23_series(bars)
    fwd = forward_returns(bars, horizon=horizon)
    return SymbolSeries(
        symbol=symbol,
        timestamps=[b.ts for b in bars],
        factor=factor,
        forward_return_1h=fwd,
    )


__all__ = [
    "ALPHA_23",
    "MIN_WARMUP",
    "SymbolSeries",
    "alpha23_series",
    "candles_to_bars",
    "fetch_bars",
    "forward_returns",
    "symbol_series",
]
