"""Assembling the real record into instants a policy can be replayed over.

`eval.incremental` is pure: instants in, comparison out, no network and no ledger. This is the part
that reads the two ledgers and the price history and turns them into those instants, kept separate
so the comparison can be tested against constructed cases and this can be tested against the files.

**Two rules govern what goes in.**

*Trailing returns stop strictly before the instant.* A baseline that could see the bar at the
decision time would be reading the answer, and momentum is exactly the rule that would benefit. The
slice is taken with a strict inequality on the timestamp, and the test suite asserts that extending
the price history into the future leaves every instant's trailing window unchanged.

*The desk's direction comes from the ledger, not from re-running it.* Re-deciding today and calling
it last week's decision is the worst kind of backtest, and this module would be the natural place to
do it by accident. Every recorded verdict is mapped to a direction only when the desk actually
opened exposure; every abstention maps to flat, which is what it was.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from datetime import datetime
from itertools import pairwise

from argus.eval.hurdle import default_hurdle_bps, load_instants
from argus.eval.incremental import DIRECTION_LONG, DIRECTION_NONE, DIRECTION_SHORT, Instant

TRAILING_BARS = 72
"""Hours of trailing history handed to each baseline.

Three days. Long enough for a 24-bar momentum rule to have a full window and a volatility estimate
with some sample behind it, short enough that a rule cannot quietly become a long-horizon trend
follower fitted to the window this record happens to cover.
"""

MIN_TRAILING = 24
"""Below this an instant is dropped rather than handed to a rule with a stub of history.

Dropping is the honest option: a momentum rule given four bars still returns a direction, and that
direction is noise wearing the same label as a real signal.
"""

OPENING_VERDICTS = frozenset({"trade", "reduce", "hedge"})
"""Verdicts that put exposure on. Everything else — no_trade, delay, data_insufficient,
human_review — is flat, because that is what the desk actually did."""


def direction_of(verdict: str, side: str, quantity: str | float | None) -> int:
    """The direction a recorded decision actually took. Flat unless it opened with size."""
    if verdict.strip().lower() not in OPENING_VERDICTS:
        return DIRECTION_NONE
    try:
        size = float(quantity if quantity is not None else 0)
    except (TypeError, ValueError):
        return DIRECTION_NONE
    if size <= 0:
        return DIRECTION_NONE
    return DIRECTION_SHORT if side.strip().lower() == "sell" else DIRECTION_LONG


def trailing_returns(
    stamps: Sequence[datetime], returns: Sequence[float], at: datetime, *, bars: int = TRAILING_BARS
) -> tuple[float, ...]:
    """Returns strictly before ``at``, most recent last.

    ``stamps[i]`` is the timestamp at which ``returns[i]`` was realised, so a return stamped exactly
    at the instant is already the instant's own bar and is excluded. That off-by-one is the whole
    difference between a momentum baseline and a look-ahead.
    """
    cut = bisect_left(list(stamps), at)
    window = returns[max(0, cut - bars):cut]
    return tuple(window)


def build_instants(
    rows: Sequence[tuple[str, datetime, float, str, str, str | float | None]],
    history: dict[str, tuple[Sequence[datetime], Sequence[float]]],
    *,
    hurdle_bps: float | None = None,
    bars: int = TRAILING_BARS,
    minimum_trailing: int = MIN_TRAILING,
) -> tuple[list[Instant], list[int]]:
    """Turn recorded decisions plus price history into instants and the desk's own directions.

    ``rows`` are ``(symbol, at, realised_bps, verdict, side, quantity)``. An instant whose symbol
    has no history, or whose trailing window is too short, is dropped from **both** lists together —
    a decision kept without its instant would silently misalign the pairing.
    """
    hurdle = default_hurdle_bps() if hurdle_bps is None else hurdle_bps
    instants: list[Instant] = []
    directions: list[int] = []
    for symbol, at, realised_bps, verdict, side, quantity in rows:
        series = history.get(symbol)
        if series is None:
            continue
        stamps, returns = series
        trailing = trailing_returns(stamps, returns, at, bars=bars)
        if len(trailing) < minimum_trailing:
            continue
        instants.append(
            Instant(
                symbol=symbol,
                at=at,
                trailing=trailing,
                realised_bps=realised_bps,
                hurdle_bps=hurdle,
            )
        )
        directions.append(direction_of(verdict, side, quantity))
    return instants, directions


def _ledger_rows() -> list[tuple[str, datetime, float, str, str, str | float | None]]:
    """Every recorded decision that has a realised move, from both ledgers.

    The verdict, side and quantity are read back from the same row the move came from, so a
    decision can never be paired with a different decision's outcome.
    """
    import json
    from pathlib import Path

    from argus.eval.hurdle import LIVE_PATH, REPLAY_PATH

    out: list[tuple[str, datetime, float, str, str, str | float | None]] = []
    for path, move_field in ((REPLAY_PATH, "realised_bps"), (LIVE_PATH, "counterfactual_move_bps")):
        file = Path(path)
        if not file.exists():
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row.get(move_field)
            stamp = row.get("at") or row.get("decided_at")
            if raw is None or not stamp:
                continue
            try:
                move = float(raw)
            except (TypeError, ValueError):
                continue
            out.append((
                str(row.get("symbol", "?")),
                datetime.fromisoformat(str(stamp)),
                move,
                str(row.get("verdict", "")),
                str(row.get("side", "")),
                row.get("quantity"),
            ))
    return out


def collect_instants(
    *, days: int = 180, bars: int = TRAILING_BARS
) -> tuple[list[Instant], list[int]]:  # pragma: no cover - network
    """The live path: read both ledgers, fetch the history each instant needs, assemble."""
    from argus.backtest.engine import Bar
    from argus.market.history import CandleType, fetch_range

    rows = _ledger_rows()
    if not rows:
        return [], []
    history: dict[str, tuple[Sequence[datetime], Sequence[float]]] = {}
    for symbol in sorted({r[0] for r in rows}):
        try:
            candles = fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.MARKET)
        except Exception:
            continue
        series = [Bar(ts=c.ts, close=c.close, extra={}) for c in candles]
        stamps = [b.ts for b in series[1:]]
        returns = [
            0.0 if a.close == 0 else float(b.close) / float(a.close) - 1.0
            for a, b in pairwise(series)
        ]
        history[symbol] = (stamps, returns)
    return build_instants(rows, history, bars=bars)


def instant_count() -> int:  # pragma: no cover - convenience
    return len(load_instants())


__all__ = [
    "MIN_TRAILING",
    "OPENING_VERDICTS",
    "TRAILING_BARS",
    "build_instants",
    "collect_instants",
    "direction_of",
    "instant_count",
    "trailing_returns",
]
