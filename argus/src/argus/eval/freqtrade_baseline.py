"""Faithful ports of `freqtrade`'s risk protections — the named-specialist baseline for
Foundation 5's risk-control dimension, reproduced rather than asserted (the goal's own OWNED bar:
"a run comparison against the named best specialist"). **All four of freqtrade's protections are
ported**: ``MaxDrawdown``, ``StoplossGuard``, ``LowProfitPairs``, ``CooldownPeriod``.

**Read from source, not reimplemented from memory** — `freqtrade` cloned locally at
`okx/trading/best-of-the-best/repos/freqtrade`, read directly on 2026-09-15 and cited by file:line
throughout this module:

* ``freqtrade/plugins/protections/max_drawdown_protection.py`` — a lookback window, a
  minimum-trade-count gate before it can fire at all (``_trade_limit``, default 1), and the
  "ratios" calculation mode (the module's own default, ``calculation_mode="ratios"``), which sums
  each closed trade's **return ratio** (``close_profit``), not its dollar P&L.
* ``freqtrade/data/metrics.py:_calc_drawdown_series`` (``:130-147``) — the exact arithmetic:
  ``cumulative = cumsum(value)``; ``high_value = max(0, cummax(cumulative))``;
  ``drawdown = cumulative - high_value``; the reported figure is ``abs(min(drawdown))``, the worst
  point in the window.
* ``freqtrade/plugins/protections/stoploss_guard.py:44-78`` — N losing trades (below a profit
  threshold, optionally filtered by direction) within a lookback window → lock, default
  ``trade_limit=10``.
* ``freqtrade/plugins/protections/low_profit_pairs.py:41-73`` — sums one **symbol's** closed-trade
  return ratios within a lookback window and locks **only that symbol** below a profit floor,
  default ``trade_limit=1``. The genuinely per-symbol capability ARGUS's own circuit breaker
  entirely lacks (found reading this file earlier the same day, logged, not silently closed by a
  comparison instrument alone).
* ``freqtrade/plugins/protections/cooldown_period.py:29-46`` plus the shared
  ``IProtection.calculate_lock_end``/``__init__`` defaults (``iprotection.py:31-53, 124-142``) —
  any closed trade on a symbol locks it for a fixed duration (default 60 minutes), no profit/loss
  threshold at all.

**Operates on `backtest.engine.SyntheticTrade`, not freqtrade's SQLAlchemy `Trade` ORM.** Same
discrete-trade shape (entry/exit, a realised return), different storage layer — freqtrade queries
a live database of real fills; this reads whatever trade sequence the caller supplies, real or
reconstructed via `backtest.engine.extract_trades`. Nothing here queries a database, matching this
project's own `ConstitutionPolicy` discipline: a deterministic check receives its inputs, it does
not fetch them.

**What this deliberately does not port, and the direction of the bias where it matters.**
``MaxDrawdown``'s "equity" calculation mode (``max_drawdown_protection.py``'s
``if self._calculation_mode == "equity":`` branch) needs an accumulated account-equity curve
`desk.book.VenueMarginSnapshot` does not yet keep over time — not ported; the "ratios" mode, the
protection's own default, is complete. ``StoplossGuard``'s real filter also requires the closing
exit to have been an *actual stop-loss order fill*
(``exit_reason in {TRAILING_STOP_LOSS, STOP_LOSS, STOPLOSS_ON_EXCHANGE, LIQUIDATION}``) — a real
order event `SyntheticTrade` structurally cannot carry (it closes on a weight change, not a
triggering order type), so this port counts every losing trade under the threshold instead. Stated
precisely: this makes the port **strictly more inclusive** than freqtrade's real count, so it
locks at least as often as the genuine protection would on equivalent data, never less — a caller
must read that bias in the stated direction, not treat the numbers as identical to freqtrade's own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from argus.backtest.engine import SyntheticTrade


@dataclass(frozen=True, slots=True)
class DrawdownVerdict:
    """What the ported `MaxDrawdown` protection decided, and the evidence behind it."""

    locked: bool
    drawdown: float
    """``abs`` of the worst cumulative-return-ratio drawdown within the window. ``0.0`` when too
    few trades fall in the window to evaluate at all — not the same claim as "no drawdown"."""

    trades_in_window: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "locked": self.locked, "drawdown": round(self.drawdown, 6),
            "trades_in_window": self.trades_in_window, "reason": self.reason,
        }


def freqtrade_max_drawdown(
    trades: Sequence[SyntheticTrade],
    *,
    now: datetime,
    lookback_minutes: int,
    max_allowed_drawdown: float,
    trade_limit: int = 1,
) -> DrawdownVerdict:
    """Faithful port of ``MaxDrawdown._max_drawdown`` (``max_drawdown_protection.py:47-99``),
    "ratios" mode.

    ``trade_limit`` (default 1, matching the protection's own default,
    ``max_drawdown_protection.py:24``) is the minimum-sample gate the source code itself enforces
    before evaluating at all — reproduced here, not loosened: a call with fewer trades in the
    window than this returns ``locked=False`` with the trade count named, the same "not enough
    evidence yet" verdict the real protection gives, never read as a genuine zero drawdown.
    """
    window_start = now - timedelta(minutes=lookback_minutes)
    in_window = sorted(
        (t for t in trades if window_start <= t.exit_ts <= now), key=lambda t: t.exit_ts,
    )
    if len(in_window) < trade_limit:
        return DrawdownVerdict(
            locked=False, drawdown=0.0, trades_in_window=len(in_window),
            reason=f"{len(in_window)} trade(s) in the last {lookback_minutes}min, "
            f"{trade_limit} needed before evaluating",
        )

    cumulative = 0.0
    peak = 0.0
    worst = 0.0
    for trade in in_window:
        cumulative += trade.return_pct
        peak = max(0.0, peak, cumulative)
        worst = min(worst, cumulative - peak)
    drawdown = abs(worst)

    if drawdown > max_allowed_drawdown:
        return DrawdownVerdict(
            locked=True, drawdown=drawdown, trades_in_window=len(in_window),
            reason=f"{drawdown:.4f} passed {max_allowed_drawdown} within {lookback_minutes}min",
        )
    return DrawdownVerdict(
        locked=False, drawdown=drawdown, trades_in_window=len(in_window),
        reason=f"{drawdown:.4f} within the {max_allowed_drawdown} limit",
    )


@dataclass(frozen=True, slots=True)
class StoplossGuardVerdict:
    losing_trades: int
    trade_limit: int
    locked: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "losing_trades": self.losing_trades, "trade_limit": self.trade_limit,
            "locked": self.locked, "reason": self.reason,
        }


def freqtrade_stoploss_guard(
    trades: Sequence[SyntheticTrade],
    *,
    now: datetime,
    lookback_minutes: int,
    trade_limit: int = 10,
    required_profit: float = 0.0,
    only_per_side: str | None = None,
) -> StoplossGuardVerdict:
    """Faithful port of ``StoplossGuard._stoploss_guard``
    (``freqtrade/plugins/protections/stoploss_guard.py:44-78``): N losing trades within a lookback
    window, filtered by minimum profit (default 0.0, i.e. any loss) and optionally by direction
    (``only_per_side``), lock once the count reaches ``trade_limit`` (freqtrade's own default: 10).

    **One stated, one-directional simplification — read this before trusting the count.**
    freqtrade's real filter also requires the closing exit to have been an *actual stop-loss
    fill* (``exit_reason in {TRAILING_STOP_LOSS, STOP_LOSS, STOPLOSS_ON_EXCHANGE, LIQUIDATION}``,
    ``stoploss_guard.py:56-64``) — a real order event freqtrade's own `Trade` ORM records and a
    continuous-weight-reconstructed :class:`~argus.backtest.engine.SyntheticTrade` structurally
    cannot carry (it closes on a weight change, not a triggering order type). This port therefore
    counts **every** losing trade under the profit threshold, not only stop-loss-triggered ones —
    strictly more inclusive than freqtrade's own count, so this verdict locks **at least as often**
    as the real protection would on genuinely equivalent data, never less. A caller relying on this
    for an OWNED-status claim must read this bias in the stated direction, not ignore it.
    """
    window_start = now - timedelta(minutes=lookback_minutes)
    in_window = [t for t in trades if window_start <= t.exit_ts <= now]
    if only_per_side is not None:
        in_window = [t for t in in_window if t.direction == only_per_side]
    losing = [t for t in in_window if t.return_pct < required_profit]

    if len(losing) < trade_limit:
        return StoplossGuardVerdict(
            losing_trades=len(losing), trade_limit=trade_limit, locked=False,
            reason=f"{len(losing)} losing trade(s) in the last {lookback_minutes}min, "
            f"{trade_limit} needed",
        )
    return StoplossGuardVerdict(
        losing_trades=len(losing), trade_limit=trade_limit, locked=True,
        reason=f"{trade_limit} losing trades within {lookback_minutes} minutes",
    )


@dataclass(frozen=True, slots=True)
class LowProfitVerdict:
    """The mechanism that closes a real, verified capability gap: ARGUS's own circuit breaker
    (`risk/circuit.py`) is whole-book only — nothing anywhere locks out one specific
    underperforming symbol while leaving the rest of the book tradeable (found while reading
    freqtrade's Protections, logged the same day, not assumed away). This port is the first
    concrete implementation of that missing capability, not just a comparison instrument."""

    symbol: str
    summed_profit: float
    trades_in_window: int
    locked: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "summed_profit": round(self.summed_profit, 6),
            "trades_in_window": self.trades_in_window, "locked": self.locked,
            "reason": self.reason,
        }


def freqtrade_low_profit_pairs(
    trades: Sequence[SyntheticTrade],
    *,
    symbol: str,
    now: datetime,
    lookback_minutes: int,
    required_profit: float = 0.0,
    trade_limit: int = 1,
    only_per_side: str | None = None,
) -> LowProfitVerdict:
    """Faithful port of ``LowProfitPairs._low_profit``
    (``freqtrade/plugins/protections/low_profit_pairs.py:41-73``): sum a **single symbol's**
    closed-trade return ratios within a lookback window; lock **that symbol only** if the sum
    falls below ``required_profit`` (default 0.0). ``trades`` is filtered to ``symbol`` here —
    the caller passes the whole book's trades, exactly as freqtrade's own
    ``Trade.get_trades_proxy(pair=pair, ...)`` scopes to one pair inside the query rather than
    asking the caller to pre-filter.

    This is genuinely **per-symbol**, unlike every other gate Foundation 5 built earlier today:
    a book with five symbols, four profitable and one badly losing, locks only the losing one —
    the four keep trading. Nothing in `agents.desk.ConstitutionPolicy` can do that yet; wiring
    this into the live Constitution as a ninth dimension is the next step, not done here.
    """
    for_symbol = [t for t in trades if t.symbol == symbol]
    window_start = now - timedelta(minutes=lookback_minutes)
    in_window = [t for t in for_symbol if window_start <= t.exit_ts <= now]
    if only_per_side is not None:
        in_window = [t for t in in_window if t.direction == only_per_side]

    if len(in_window) < trade_limit:
        return LowProfitVerdict(
            symbol=symbol, summed_profit=0.0, trades_in_window=len(in_window), locked=False,
            reason=f"{len(in_window)} trade(s) in the last {lookback_minutes}min, "
            f"{trade_limit} needed",
        )

    summed = sum(t.return_pct for t in in_window)
    if summed < required_profit:
        return LowProfitVerdict(
            symbol=symbol, summed_profit=summed, trades_in_window=len(in_window), locked=True,
            reason=f"{summed:.4f} < {required_profit} within {lookback_minutes}min",
        )
    return LowProfitVerdict(
        symbol=symbol, summed_profit=summed, trades_in_window=len(in_window), locked=False,
        reason=f"{summed:.4f} at or above the {required_profit} floor",
    )


@dataclass(frozen=True, slots=True)
class CooldownVerdict:
    symbol: str
    locked: bool
    locked_until: datetime | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "locked": self.locked,
            "locked_until": None if self.locked_until is None else self.locked_until.isoformat(),
            "reason": self.reason,
        }


def freqtrade_cooldown_period(
    trades: Sequence[SyntheticTrade], *, symbol: str, now: datetime, lookback_minutes: int,
    stop_duration_minutes: int = 60,
) -> CooldownVerdict:
    """Faithful port of ``CooldownPeriod._cooldown_period``
    (``freqtrade/plugins/protections/cooldown_period.py:29-46``) plus the shared
    ``IProtection.calculate_lock_end`` (``iprotection.py:124-142``) it calls: **any** closed trade
    on ``symbol`` within the lookback window (default 60 minutes, ``IProtection.__init__``'s own
    default, ``iprotection.py:46``) locks that symbol until ``stop_duration_minutes`` (default 60,
    same file, same default) after the *most recent* such trade's close — no profit/loss
    threshold at all, unlike the other three protections. Trading a symbol, winning or losing,
    earns a mandatory pause before trading it again.

    **Not ported**: the ``unlock_at`` fixed-hour-of-day variant
    (``iprotection.py:134-140``, a config option this port's caller-supplied-``now`` interface has
    no analogue for) — a stated omission, not a silent one. The duration-based path, this
    protection's and freqtrade's own default, is complete.

    **Consolidates freqtrade's two-layer design into one self-contained query, on purpose.**
    freqtrade's real ``_cooldown_period`` always *requests* a lock (returns ``lock=True`` whenever
    a recent trade exists) and a separate registry (``PairLocks``) tracks whether ``until`` has
    actually passed. This function answers the single question every other port in this module
    answers — "would the symbol read as locked as of ``now``" — matching how
    ``agents.desk.ConstitutionPolicy.rule`` itself works: one pure query per decision, no external
    registry the caller must also maintain between calls.
    """
    window_start = now - timedelta(minutes=lookback_minutes)
    in_window = [
        t for t in trades if t.symbol == symbol and window_start <= t.exit_ts <= now
    ]
    if not in_window:
        return CooldownVerdict(
            symbol=symbol, locked=False, locked_until=None,
            reason=f"no trade on {symbol} in the last {lookback_minutes}min",
        )
    latest_close = max(t.exit_ts for t in in_window)
    locked_until = latest_close + timedelta(minutes=stop_duration_minutes)
    if now < locked_until:
        return CooldownVerdict(
            symbol=symbol, locked=True, locked_until=locked_until,
            reason=f"cooldown for {symbol} until {locked_until.isoformat()}",
        )
    return CooldownVerdict(
        symbol=symbol, locked=False, locked_until=locked_until,
        reason=f"cooldown for {symbol} expired at {locked_until.isoformat()}",
    )


__all__ = [
    "CooldownVerdict",
    "DrawdownVerdict",
    "LowProfitVerdict",
    "StoplossGuardVerdict",
    "freqtrade_cooldown_period",
    "freqtrade_low_profit_pairs",
    "freqtrade_max_drawdown",
    "freqtrade_stoploss_guard",
]
