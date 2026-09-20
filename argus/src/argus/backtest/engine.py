"""Backtest engine where a zero-fee run is structurally impossible.

Track 1 requires "runnable strategy code + verifiable backtest, total period >= 60 days,
out-of-sample >= 30 days". This engine produces exactly that, with the honesty properties the
teardowns showed are routinely missing:

* **Costs cannot be omitted.** The engine takes a :class:`~argus.cost.model.CostModel` and calls
  ``assert_gateable()`` before it will run. NautilusTrader backtests at zero fees unless configured;
  here a frictionless model raises.
* **Signals are lagged by construction.** A signal computed from bar *t* can only trade at bar
  *t+1*. This is enforced in the loop rather than left to the strategy author, because "I used
  today's close to trade today's close" is the most common backtest bug in existence and it is
  invisible in the output.
* **Out-of-sample is split chronologically and reported separately**, with the handbook's own
  decay alert (OS < 0.5x IS) computed automatically.
* **Turnover is net delta**, so holding a position costs nothing.

The engine is deliberately small. Its value is not features — Qlib and vectorbt have more — it is
that every number it produces has already had the fee subtracted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from argus.backtest.metrics import (
    MetricError,
    Performance,
    evaluate,
    out_of_sample_decay,
    sharpe,
)
from argus.backtest.metrics import stability as rolling_stability
from argus.cost.model import CostModel
from argus.execution.passive import PassiveExecution, PassiveExecutionError, book_from_bar

# A signal function sees only bars up to and including `i`, and returns a target weight in
# [-1, 1]. The engine applies it at i+1. The signature makes look-ahead awkward to express.
#
# Sequence is typed contravariantly via Bar rather than Any so a strategy written against Bar
# type-checks at the call site; Any here would silently accept a signal expecting a different bar.
SignalFn = Callable[[Sequence["Bar"], int], float]


@dataclass(frozen=True, slots=True)
class Bar:
    ts: datetime
    close: Decimal
    extra: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PassiveReport:
    """What passive execution actually delivered, when a run asked for it.

    Reported alongside the returns rather than folded into them, because the shortfall is the
    finding. A strategy whose ``fill_rate`` is 0.3 did not run at a maker fee — it ran a third of
    the time, and the other two thirds are either missing or were chased at the taker rate.
    """

    attempts: int
    intended: float
    filled_passive: float
    chased_taker: float
    unexecuted: float
    realised_fee_bps: float
    chase: bool

    @property
    def fill_rate(self) -> float:
        return self.filled_passive / self.intended if self.intended > 0 else 0.0

    @property
    def maker_claim_holds(self) -> bool:
        """Did passive execution actually deliver most of the intended size?

        Below 0.5 the strategy's economics are a taker's, whatever fee it was quoted.
        """
        return self.fill_rate >= 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "intended": round(self.intended, 4),
            "filled_passive": round(self.filled_passive, 4),
            "chased_taker": round(self.chased_taker, 4),
            "unexecuted": round(self.unexecuted, 4),
            "fill_rate": round(self.fill_rate, 4),
            "realised_fee_bps": round(self.realised_fee_bps, 3),
            "chase": self.chase,
            "maker_claim_holds": self.maker_claim_holds,
        }


@dataclass(frozen=True, slots=True)
class BacktestResult:
    name: str
    symbol: str
    bars: int
    periods_per_year: int
    gross: Performance
    net: Performance
    in_sample: Performance | None
    out_of_sample: Performance | None
    decay: dict[str, float | bool] | None
    total_cost_bps: float
    trades: int
    passive: PassiveReport | None = None

    net_returns: tuple[float, ...] = ()
    """The per-period net return series, kept so rolling statistics can be formed downstream.

    Track 1 is scored on rolling 30-day Sharpe stability as its own criterion, and a stability
    statistic cannot be recovered from a summary — the summary is what it is a statistic *of*.
    Carrying the series is the only way the study can report it without re-running the backtest."""

    weights: tuple[float, ...] = ()
    """The per-bar target weight the signal actually achieved, ``weights[i]`` held from bar ``i``
    to ``i+1`` — computed every call (``run``'s own loop, ``achieved`` appended each bar) but
    dropped before this session, so nothing downstream could ever answer "when did a position
    open and close" from a completed backtest, only the aggregate cost/return summary.

    This is a **continuous-weight rebalancing engine, not a discrete-trade one** — ``trades``
    above is a bare count of bars where the weight changed materially, never an entry/exit pair
    with its own realised P&L. Exposing the weight series is what lets a caller reconstruct
    discrete trade boundaries after the fact (a weight crossing zero, or reversing sign, is an
    open/close event) without re-running the backtest — needed to compare against any
    discrete-trade risk system (a circuit breaker, `freqtrade`'s Protections) on the same
    underlying data, which nothing in this module could do before this field existed.
    """

    @property
    def stability(self) -> dict[str, float | int] | None:
        """Rolling-window Sharpe stability, or ``None`` when the series is too short.

        ``None`` rather than a fabricated figure: a rolling statistic over fewer periods than one
        window is not a rolling statistic, and the study prints the absence.
        """
        window = min(30, max(2, len(self.net_returns) // 4))
        try:
            return rolling_stability(
                list(self.net_returns), window=window,
                periods_per_year=self.periods_per_year,
            ).as_dict()
        except MetricError:
            return None

    @property
    def cost_destroyed_the_edge(self) -> bool:
        """Did the fee flip a winning gross result into a losing net one?

        Reported explicitly because it is the single most common outcome on this venue, and a
        backtest that hides it is lying by omission.
        """
        return self.gross.total_return > 0 >= self.net.total_return

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "symbol": self.symbol,
            "bars": self.bars,
            "gross": self.gross.as_dict(),
            "net": self.net.as_dict(),
            "total_cost_bps": round(self.total_cost_bps, 2),
            "cost_destroyed_the_edge": self.cost_destroyed_the_edge,
            "in_sample": self.in_sample.as_dict() if self.in_sample else None,
            "out_of_sample": self.out_of_sample.as_dict() if self.out_of_sample else None,
            "decay": self.decay,
            "stability": self.stability,
            "passive": self.passive.as_dict() if self.passive else None,
        }


def run(
    name: str,
    symbol: str,
    bars: Sequence[Bar],
    signal: SignalFn,
    *,
    cost: CostModel,
    periods_per_year: int,
    oos_fraction: float = 0.35,
    max_weight: float = 1.0,
    passive: PassiveExecution | None = None,
    passive_size: Decimal = Decimal("1"),
) -> BacktestResult:
    """Run one strategy over one instrument.

    The loop is the important part:

    1. At bar ``i`` the signal sees bars ``0..i`` and produces a target weight.
    2. That weight is applied to the return from ``i`` to ``i+1``.
    3. The cost of changing weight is charged at ``i``, before the return is earned.

    A strategy therefore cannot trade on information it did not have, and cannot avoid paying to
    change its mind.

    ``passive`` switches execution from "assume the trade happened at the taker fee" to "simulate
    the queue and find out". It is not a discount. Every bar on which a passive strategy wants to
    trade must carry ``level_qty`` and ``traded_qty`` in :attr:`Bar.extra`, or the run raises — a
    maker claim with no book behind it is refused rather than quietly granted. ``passive_size`` is
    the order size, in the book's own units, that a full-weight position represents.
    """
    cost.assert_gateable()
    if len(bars) < 10:
        raise MetricError("a backtest needs at least 10 bars")
    if passive is not None and passive_size <= 0:
        raise PassiveExecutionError(f"passive_size={passive_size} must be positive")

    round_trip = float(cost.round_trip_bps()) / 10_000.0
    one_way = round_trip / 2

    gross_returns: list[float] = []
    net_returns: list[float] = []
    weights: list[float] = []
    total_cost = 0.0
    prev_weight = 0.0
    trades = 0

    p_attempts = 0
    p_intended = p_filled = p_chased = p_fee_weighted = 0.0

    for i in range(len(bars) - 1):
        target = max(-max_weight, min(max_weight, signal(bars, i)))
        delta = abs(target - prev_weight)
        if delta > 1e-9:
            trades += 1

        if passive is None or delta <= 1e-9:
            trade_cost = delta * one_way
            achieved = target
        else:
            book = book_from_bar(bars[i].extra)
            if book is None:
                raise PassiveExecutionError(
                    f"bar {i} ({bars[i].ts.isoformat()}) asks for passive execution but carries no "
                    "level_qty/traded_qty. A maker fee without a book behind it is the defect this "
                    "engine exists to prevent; supply the book or run at taker."
                )
            fill = passive.execute(Decimal(str(delta)) * passive_size, book)
            p_attempts += 1
            p_intended += float(fill.intended)
            p_filled += float(fill.filled_passive)
            p_chased += float(fill.chased_taker)
            p_fee_weighted += float(fill.achieved) * float(fill.fee_bps)

            # The fee is charged on what executed, at the rate it actually earned.
            executed_weight = float(fill.achieved / passive_size)
            trade_cost = executed_weight * float(fill.fee_bps) / 10_000.0
            # The position only moves as far as the fills carried it.
            direction = 1.0 if target >= prev_weight else -1.0
            achieved = prev_weight + direction * executed_weight

        total_cost += trade_cost

        px_now = float(bars[i].close)
        px_next = float(bars[i + 1].close)
        bar_return = (px_next - px_now) / px_now if px_now > 0 else 0.0

        gross_returns.append(achieved * bar_return)
        net_returns.append(achieved * bar_return - trade_cost)
        weights.append(achieved)
        prev_weight = achieved

    if passive is not None and p_attempts > 0 and (p_filled + p_chased) <= 0:
        # The most important passive outcome there is, and the one a metrics error would bury
        # behind "Sharpe is undefined for a zero-variance series". The strategy never traded.
        raise PassiveExecutionError(
            f"passive execution filled nothing across {p_attempts} attempts "
            f"({p_intended:.4g} units intended). The strategy never held a position, so there is "
            "no return stream to score. This is the result, not an error in the data: on this "
            "book, at this size, the queue never cleared."
        )

    gross = evaluate(gross_returns, weights, periods_per_year=periods_per_year)
    net = evaluate(net_returns, weights, periods_per_year=periods_per_year)

    # Chronological split. Never random — a random split leaks the future into the past.
    split = int(len(net_returns) * (1 - oos_fraction))
    in_sample = out_sample = None
    decay = None
    if split >= 10 and len(net_returns) - split >= 10:
        try:
            in_sample = evaluate(
                net_returns[:split], weights[:split], periods_per_year=periods_per_year
            )
            out_sample = evaluate(
                net_returns[split:], weights[split:], periods_per_year=periods_per_year
            )
            decay = out_of_sample_decay(in_sample.sharpe, out_sample.sharpe)
        except MetricError:
            # A flat half cannot be scored. Reported as absent rather than faked.
            in_sample = out_sample = None

    return BacktestResult(
        name=name,
        symbol=symbol,
        bars=len(bars),
        periods_per_year=periods_per_year,
        gross=gross,
        net=net,
        in_sample=in_sample,
        out_of_sample=out_sample,
        decay=decay,
        total_cost_bps=total_cost * 10_000,
        trades=trades,
        net_returns=tuple(net_returns),
        weights=tuple(weights),
        passive=(
            PassiveReport(
                attempts=p_attempts,
                intended=p_intended,
                filled_passive=p_filled,
                chased_taker=p_chased,
                unexecuted=p_intended - p_filled - p_chased,
                realised_fee_bps=(
                    p_fee_weighted / (p_filled + p_chased) if (p_filled + p_chased) > 0 else 0.0
                ),
                chase=passive.chase,
            )
            if passive is not None
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class SyntheticTrade:
    """One discrete position, reconstructed from a continuous-weight backtest's weight series —
    the unit a discrete-trade risk system (a circuit breaker, `freqtrade`'s Protections) reasons
    about, which :func:`run`'s native output never produces (see :attr:`BacktestResult.weights`'s
    docstring). "Synthetic" because it is reconstructed after the fact from a weight trajectory,
    not observed as a real fill — a fact this name states rather than lets a reader assume away.
    """

    symbol: str
    entry_bar: int
    exit_bar: int
    entry_ts: datetime
    exit_ts: datetime
    direction: str
    """``"long"`` or ``"short"`` — the sign of the weight held for this trade's whole span."""

    weight: float
    """The weight held. Simplification, stated plainly: a strategy that varies its weight while
    staying the same sign (0.5 -> 0.8, say) is reconstructed as ONE trade at its *entry* weight,
    not as a sequence of size adjustments — the entry/exit price pair is what a discrete-trade
    protection system needs, and modelling every intermediate resize would answer a question
    (position-sizing behaviour) this extractor does not claim to."""

    entry_price: Decimal
    exit_price: Decimal

    @property
    def return_pct(self) -> float:
        """Direction-adjusted price return over the holding period, entry to exit price only —
        **not** cost-adjusted and **not** weighted by the intermediate weight path within the
        trade (see :attr:`weight`'s docstring). A caller wanting a fee-aware figure should apply
        `cost.model.CostModel` to this trade's own entry/exit rather than trust this field alone
        for anything but direction and rough magnitude.
        """
        raw = float((self.exit_price - self.entry_price) / self.entry_price)
        return raw if self.direction == "long" else -raw


def extract_trades(
    bars: Sequence[Bar], weights: Sequence[float], *, symbol: str = "",
) -> list[SyntheticTrade]:
    """Reconstruct discrete trades from a completed backtest's ``(bars, weights)`` pair.

    ``symbol`` is stamped onto every trade verbatim (default ``""`` for a caller that does not
    need it) — one `run()` call is always over one instrument, so it is a single value here, not
    per-trade data reconstructed from anything; carried so multi-symbol callers (a per-pair
    protection like freqtrade's ``LowProfitPairs``) can combine trades from several `run()` calls
    and still tell which came from where.

    ``weights[i]`` is the weight held from bar ``i`` to ``i+1`` (:attr:`BacktestResult.weights`'s
    own docstring) — a trade **opens** at the first bar where the weight becomes nonzero after
    being flat or opposite-signed, and **closes** at the first bar it returns to flat or reverses
    sign. A reversal is simultaneously a close and a re-open at the same bar, mirroring
    `desk.book.Position.apply_fill`'s own reversal handling (verified against
    ``nautilus_trader``'s ``position.rs:545-551`` when that module was built) rather than
    inventing a second convention for the same event.

    Read ``bars[:-1]`` against ``weights`` (index ``i`` prices a trade opened at bar ``i``) and
    ``bars[1:]`` for the bar the position was actually marked to close *at* — the same one-bar
    lag :func:`run`'s own loop uses (a weight set at bar ``i`` earns the return from ``i`` to
    ``i+1``), so a reconstructed trade's exit price is the price the weight change actually paid
    or received against, not the price at the bar the *signal* changed.
    """
    if len(weights) != len(bars) - 1:
        raise MetricError(
            f"{len(weights)} weight(s) do not match {len(bars)} bar(s) — extract_trades needs "
            f"the exact (bars, weights) pair a single `run()` call produced, not a mismatched one"
        )

    def _direction(w: float) -> str | None:
        if w > 0:
            return "long"
        if w < 0:
            return "short"
        return None

    trades: list[SyntheticTrade] = []
    open_bar: int | None = None
    open_direction: str | None = None
    open_weight = 0.0

    def _close(exit_bar: int) -> None:
        nonlocal open_bar, open_direction
        if open_bar is None or open_direction is None:
            return
        trades.append(SyntheticTrade(
            symbol=symbol, entry_bar=open_bar, exit_bar=exit_bar,
            entry_ts=bars[open_bar].ts, exit_ts=bars[exit_bar].ts,
            direction=open_direction, weight=open_weight,
            entry_price=bars[open_bar].close, exit_price=bars[exit_bar].close,
        ))
        open_bar = None
        open_direction = None

    for i, w in enumerate(weights):
        direction = _direction(w)
        if open_direction is None:
            if direction is not None:
                open_bar, open_direction, open_weight = i, direction, w
            continue
        if direction == open_direction:
            continue
        # Flat or reversed: close what was open, then re-open if the new weight is nonzero.
        _close(i)
        if direction is not None:
            open_bar, open_direction, open_weight = i, direction, w

    if open_bar is not None:
        # Still open when the series ends: close it at the last available bar rather than
        # dropping it — an unrealised position is a real position, not an absent one.
        _close(len(bars) - 1)

    return trades


def sweep(
    name: str,
    symbol: str,
    bars: Sequence[Bar],
    variants: Mapping[str, SignalFn],
    *,
    cost: CostModel,
    periods_per_year: int,
) -> dict[str, Any]:
    """Run many variants and **record the trial count**.

    This is the function that makes the Deflated Sharpe gate usable. Reporting the best of N
    variants without N is the defect found in three separate factor-discovery systems; here the
    count and the dispersion of trial Sharpes come out alongside the winner, so the deflation can
    actually be computed.
    """
    results: dict[str, BacktestResult | str] = {}
    for label, fn in variants.items():
        try:
            results[label] = run(
                f"{name}:{label}", symbol, bars, fn,
                cost=cost, periods_per_year=periods_per_year,
            )
        except MetricError as exc:
            results[label] = str(exc)

    scored = {k: v for k, v in results.items() if isinstance(v, BacktestResult)}
    sharpes = [r.net.sharpe for r in scored.values()]
    best_label = max(scored, key=lambda k: scored[k].net.sharpe) if scored else None

    variance = 0.0
    if len(sharpes) > 1:
        mu = sum(sharpes) / len(sharpes)
        variance = sum((s - mu) ** 2 for s in sharpes) / (len(sharpes) - 1)

    return {
        "trials": len(variants),
        "scored": len(scored),
        "variance_of_trial_sharpes": round(variance, 4),
        "best": best_label,
        "best_result": scored[best_label].as_dict() if best_label else None,
        "all_net_sharpes": {k: round(v.net.sharpe, 3) for k, v in scored.items()},
    }


def buy_and_hold(bars: Sequence[Bar], i: int) -> float:
    """Baseline A. Every claim is measured against this before anything else."""
    return 1.0


def flat(bars: Sequence[Bar], i: int) -> float:
    """Baseline: do nothing. Costs nothing, earns nothing — and beats most strategies
    once fees are charged."""
    return 0.0


def rolling_sharpe(
    returns: list[float], *, window: int, periods_per_year: int
) -> list[float]:
    """Rolling Sharpe — Track 1 is scored on 30-day stability, not a single number.

    A strategy with a good full-sample Sharpe built from one profitable fortnight is not stable,
    and the rolling series is what exposes that.
    """
    out: list[float] = []
    for end in range(window, len(returns) + 1):
        try:
            out.append(sharpe(returns[end - window:end], periods_per_year=periods_per_year))
        except MetricError:
            out.append(0.0)
    return out
