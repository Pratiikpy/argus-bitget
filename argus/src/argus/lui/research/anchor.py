"""The US stock behind a perpetual: its last regular close, the perpetual at that close, and the
premium between them."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.lui.answer import Source
from argus.lui.question import (
    TRADED_SYMBOLS,
)
from argus.lui.research.kinds import (
    _t,
)
from argus.lui.research.session import (
    dual_clock,
)
from argus.lui.trace import trace_module


def _rtoken_tracking(spot: str, perp: str, days: int) -> tuple[str, Source] | None:
    """How far the spot rToken has sat from its perpetual, hour by hour, over ``days``."""
    from argus.market.rtoken_spot import hourly_bars

    try:
        spot_bars = hourly_bars(spot, spot=True, days=days)
        perp_bars = hourly_bars(perp, spot=False, days=days)
    except Exception:
        return None
    gaps = sorted((spot_bars[t][1] / perp_bars[t][1] - 1) * 10_000
                  for t in spot_bars if t in perp_bars and perp_bars[t][1] > 0)
    if len(gaps) < 12:
        return None
    mean = sum(gaps) / len(gaps)
    median = gaps[len(gaps) // 2]
    base = "r" + spot.removeprefix("R").removesuffix("USDT")
    return (f"Over the last {days} day(s), hour by hour ({len(gaps)} hours), {base} sat "
            f"{median:+.1f}bps from the {_t(perp)} perpetual at the median and {mean:+.1f}bps on "
            f"average, ranging {gaps[0]:+.1f} to {gaps[-1]:+.1f}bps.",
            Source(kind="computation", ref="argus.market.rtoken_spot.hourly_bars",
                   detail=f"{spot} vs {perp}, hourly closes, {days} days"))


PREMIUM_SANITY_BPS = 500.0


def _premium_line(symbol: str, rtoken_last: Decimal,
                  anchor_open: bool) -> tuple[str, Source] | None:
    """The rToken's premium or discount to the stock it tracks, from `bitget-mcp-server`'s quote
    of the underlying. The number that makes an rToken different from its stock, stated with the
    clock: while the anchor is shut the stock's price is its last close, so the gap is partly the
    overnight move the rToken has priced and the stock has not."""
    from argus.market import universe
    from argus.market.bitget_mcp import shared_service

    if symbol not in TRADED_SYMBOLS and not universe.is_equity(symbol):
        return None
    last = _shut_close(symbol) if not anchor_open else None
    if last is None:
        last = _stock_last(symbol, shared_service)
    if last is None:
        return None
    premium = (float(rtoken_last) / last - 1.0) * 10_000
    if not anchor_open:
        # While the stock is shut its price is its last close, so perpetual-now over that close
        # is mostly the overnight move. The premium is the gap at the close itself; the move
        # since is stated as a move (on 2026-09-25 a 6bps premium read as 60bps this way).
        close = _last_regular_close(datetime.now(UTC))
        at_close = _perp_at_close(symbol, close) if close is not None else None
        if at_close is not None:
            gap = (at_close / last - 1.0) * 10_000
            since = (float(rtoken_last) / at_close - 1.0) * 100
            if abs(gap) <= PREMIUM_SANITY_BPS:
                return (
                    f"Versus the stock: {_t(symbol)} closed at {last:g} → the perpetual traded at "
                    f"a {abs(gap):.1f}bps {'premium' if gap >= 0 else 'discount'} at that close; "
                    f"it has moved {since:+.2f}% since, which is the overnight move, not a "
                    f"premium.",
                    Source(kind="venue", ref="yahoo daily close",
                           detail=f"{_t(symbol)} regular close {last:g}; perpetual at the close "
                                  f"{at_close:g}"),
                )
    if abs(premium) > PREMIUM_SANITY_BPS:
        # A tokenised share trades within basis points of its stock. A gap this wide means the
        # two tickers are not the same company, or one price is bad — not a premium to report.
        return None
    clock = ("both live" if anchor_open else
             "the stock's price is its last close, so this includes the move since")
    return (
        f"Versus the stock: {_t(symbol)} {last:g} → the perpetual trades at a "
        f"{abs(premium):.1f}bps {'premium' if premium >= 0 else 'discount'} ({clock}).",
        Source(kind="venue", ref="bitget-mcp-server quote", detail=f"{_t(symbol)} "
               f"last {last:g}"),
    )


_STOCK_LAST: dict[str, tuple[float, float]] = {}


def _stock_last(symbol: str, service: Any) -> float | None:
    """The stock's own last price from `bitget-mcp-server`, held for a minute so the premium and
    implied-open lines of one answer read the same quote."""
    hit = _STOCK_LAST.get(symbol)
    if hit and time.monotonic() - hit[0] < 60:
        return hit[1]
    try:
        last = float(service().quote(_t(symbol)).get("last_price") or 0)
    except Exception:
        return None
    if last <= 0:
        return None
    _STOCK_LAST[symbol] = (time.monotonic(), last)
    return last


def _last_regular_close(now: datetime) -> datetime | None:
    """16:00 New York on the most recent session day at or before ``now``, from the session clock
    with its holiday calendar. Early-close days (13:00) are not in that calendar and read as 16:00:
    the perpetual's price three hours after such a close is then the reference, stated here
    rather than hidden."""
    from datetime import time as clock_time
    from zoneinfo import ZoneInfo

    from argus.truth.clocks import SessionPhase

    new_york = ZoneInfo("America/New_York")
    clock, _ = dual_clock()
    local = now.astimezone(new_york)
    for back in range(10):
        close = datetime.combine(local.date() - timedelta(days=back), clock_time(16),
                                 tzinfo=new_york)
        if close <= now and clock.phase(close - timedelta(minutes=1)) is SessionPhase.RTH:
            return close
    return None


def _shut_close(symbol: str) -> float | None:
    """The stock's last regular-session close, while its market is shut.

    Not `bitget-mcp-server`'s ``last_price``: outside regular hours that quote follows an
    extended session (09:28 UTC on 2026-09-25 it gave NVDA 226.36 with its own prev_close 223.82,
    against a regular close of 224.58), so "closed at" and the implied open were both read off a
    pre-market print. Yahoo's daily bar for the session is the regular close."""
    close = _last_regular_close(datetime.now(UTC))
    return _yahoo_close(symbol, close) if close is not None else None


def _yahoo_close(symbol: str, close: datetime) -> float | None:
    """The session's close from Yahoo's daily bars when `bitget-mcp-server` does not answer (it
    returned 503 on 2026-09-25); only a bar dated that session is used, never an older one."""
    from zoneinfo import ZoneInfo

    from argus.market import equity_history

    try:
        days = equity_history.daily(_t(symbol))
    except Exception:
        return None
    session = close.astimezone(ZoneInfo("America/New_York")).date()
    same = [d for d in days if d.day == session]
    return float(same[-1].close) if same else None


def _perp_at_close(symbol: str, close: datetime) -> float | None:
    """The perpetual's price at a regular close: the close of the hourly bar ending then."""
    from argus.market.history import fetch

    try:
        bars = fetch(symbol, interval="1H", start=close - timedelta(hours=3), end=close)
    except Exception:
        return None
    at_close = [b for b in bars if b.ts + timedelta(hours=1) == close]
    return float(at_close[0].close) if at_close and at_close[0].close > 0 else None


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
