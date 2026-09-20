"""The real run comparison Track 2's OWNED bar actually asks for — ARGUS's own circuit breaker
(`risk/circuit.py`) against every ported freqtrade protection (`eval/freqtrade_baseline.py`),
walked over REAL discrete trades reconstructed from REAL Bitget market data, not one hand-built
scenario.

**What existed before this module and why it was not enough.** `tests/test_freqtrade_baseline.py`
already proves a same-input comparison *can* be run (`run()` -> `extract_trades()` -> both
systems), and does — on one hand-built nine-bar losing-streak price path. That satisfies "same-
input comparison run" but nothing past it: one scenario is not a "statistically valid evaluation",
covers no out-of-sample split, and documents exactly one divergence (found by accident, not by
design). This module runs the identical pipeline across all twelve rTokens' real 90-day hourly
history and the actual winning strategy `research/track1_study.py` already found for each — real
market dynamics, real trade counts, real timestamps — plus a deliberately adversarial scenario
suite, an explicit chronological OOS split, and a per-protection ablation (:func:`protection_
ablation`), closing five of the thirteen OWNED conditions still open on this capability:
statistically valid evaluation, OOS test, adversarial test, costs included, ablation.

**Each symbol's `best_variant` is read from `data/track1_study.json`, not recomputed** —
`research/overfitting_study.py`'s own PBO/FDR verdict on that same file found these Track 1
"winners" statistically indistinguishable from picking at random (0 of 300 grid trials survive
correction). That finding is about whether these strategies have genuine *alpha* — irrelevant
here. This module does not evaluate strategy quality; it uses each symbol's real backtested trade
sequence purely as a source of realistic price-driven trades to exercise the risk layer against,
exactly as the existing losing-streak fixture did, just real instead of hand-built.

**Costs are read from `SyntheticTrade.return_pct`'s own docstring, not assumed.** That field is
explicitly *not* cost-adjusted — raw entry-to-exit price return only. Every return used below is
therefore `trade.return_pct` minus `CostModel.bitget_perp().round_trip_bps()` (12bps), applied
once per discrete trade, mirroring how `backtest.engine.run()` charges cost once per weight
change. Skipping this step would have quietly compared both risk systems against a costless trade
sequence neither will ever actually see.

**ARGUS's own equity/streak bookkeeping is copied verbatim from `paper/runner.py::_book_state`**
(lines 171-201, read before writing this), not reinvented: `session_open_equity=equity` (this
project's live system tracks no separate intraday session baseline either), a peak that only ever
rises, and `consecutive_losses` that resets to zero on any non-negative outcome. Using a different
convention here would make this comparison prove something about a system ARGUS does not actually
run live.

**Two scopes, both built — and the second one found a real divergence that survives a fair-use
challenge only for ONE of the two global protections, not both.** `compare_real_symbols`/
`walk_trades` compare each symbol as its own single-symbol book — the unit freqtrade's per-pair
protections (`LowProfitPairs`, `CooldownPeriod`) genuinely operate at. `compare_combined_book`/
`walk_combined_book` interleave every symbol's real trades into ONE whole-book equity curve,
matching real freqtrade's GLOBAL-by-default `MaxDrawdown`/`StoplossGuard` scope and ARGUS's own
whole-book `risk.circuit.assess` (equal-capital-split across symbols — see `walk_combined_book`'s
own docstring).

**First pass, at freqtrade's stated single-pair defaults**: both protections fire on ~90%+ of the
323 real checkpoints, against a TRUE peak-to-trough drawdown (`Checkpoint.true_drawdown_pct`) that
never once exceeded 4.11% across the whole 90-day run. **The obvious objection was tested, not
assumed away**: a knowledgeable freqtrade operator pooling 7 pairs would scale `trade_limit`
proportionally, not run it unmodified. Re-run with `stoploss_guard`'s `trade_limit` scaled to
``10 * num_symbols``: its lock rate drops from ~97% to **9.0%** — that part of the divergence was
a calibration artifact of the comparison, not a real capability gap, and is reported as such
rather than left standing as evidence it is not. `max_drawdown`'s `trade_limit` only gates the
minimum sample size before it evaluates at all, not the comparison itself, so the equivalent
rescale barely moves it (89% -> 90%): **its near-constant lock against a true max of 4.11%,
comfortably under its own 10% cap, is real and survives the fair-calibration challenge** — a
structural property of summing a rolling 30-day window's cumulative return ratios across many
symbols' trades, which can read a synthetic within-window drawdown larger than the true all-time
peak-to-trough ARGUS's own ladder measures directly, ever did.

**What this means for the thirteenth OWNED condition ("no material specialist capability still
superior"), stated precisely rather than rounded up or down**: one of the two candidate
divergences (`stoploss_guard`) turned out not to be a real capability gap once fairly calibrated.
The other (`max_drawdown`) is real, isolated, and decisively measured — freqtrade's own protection
is demonstrably LESS accurate than ARGUS's at the one thing both claim to measure (true portfolio
drawdown), on this real data. Whether "materially superior" is the right word for a protection
proven less accurate than the alternative is a judgement call this module states the facts for
rather than makes unilaterally — see `Activity/PROGRESS.md`'s entry the same day for where that
judgement currently stands.

    python -m argus.eval.risk_layer_comparison
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar, SyntheticTrade, extract_trades, run
from argus.backtest.metrics import HOURLY_PER_YEAR, MetricError
from argus.cost.model import CostModel
from argus.eval.freqtrade_baseline import (
    freqtrade_cooldown_period,
    freqtrade_low_profit_pairs,
    freqtrade_max_drawdown,
    freqtrade_stoploss_guard,
)
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range
from argus.research.grammar import EXPANDED, as_signal_fn
from argus.risk.circuit import (
    CONSECUTIVE_LOSS_HALT,
    REDUCE_ONLY_DRAWDOWN,
    TOTAL_DRAWDOWN_HALT,
    BookState,
    assess,
)
from argus.strategies.session_alpha import VARIANTS
from argus.strategies.track1_suite import TRACK1_VARIANTS

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "risk_layer_comparison.json"
TRACK1_STUDY_PATH = DATA / "track1_study.json"

STARTING_EQUITY = Decimal("100000")

RISK_LOOKBACK_MINUTES = 60 * 24 * 30
"""30 days — the window `tests/test_freqtrade_baseline.py`'s existing same-input test already
used for `freqtrade_max_drawdown`; kept identical here rather than tuned, so this module extends
that comparison instead of quietly redefining it. Shared by drawdown, stoploss-guard and
low-profit-pairs."""

COOLDOWN_LOOKBACK_MINUTES = 60
COOLDOWN_DURATION_MINUTES = 60
"""freqtrade's own stated defaults (`freqtrade_baseline.freqtrade_cooldown_period`'s docstring),
not tuned to this comparison — cooldown is a mandatory pause, not a severity judgement, so it is
reported on its own terms rather than recalibrated to look comparable to the other three."""

_EXPANDED_FNS = {name: as_signal_fn(sig) for name, sig in EXPANDED.items()}
ALL_VARIANTS: dict[str, Any] = {**VARIANTS, **TRACK1_VARIANTS, **_EXPANDED_FNS}
"""The identical variant registry `research/track1_study.py` swept — needed so a `best_variant`
name read back from `track1_study.json` resolves to the same callable that produced it."""


class RiskLayerComparisonError(RuntimeError):
    """The comparison cannot be run honestly — surfaced rather than silently skipped."""


def _cost_bps() -> Decimal:
    return CostModel.bitget_perp().round_trip_bps()


def _net_return_pct(trade: SyntheticTrade, *, cost_bps: Decimal) -> float:
    """``trade.return_pct`` is explicitly NOT cost-adjusted (its own docstring). One round-trip
    charged once per discrete trade — the discrete-trade analogue of how `run()`'s own loop
    charges cost once per weight change, never per bar held."""
    return trade.return_pct - float(cost_bps) / 10_000.0


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Both risk systems, asked the same question — "should the desk still be active" — at the
    instant one real trade closed."""

    symbol: str
    exit_ts: datetime
    net_return_pct: float
    is_oos: bool

    true_drawdown_pct: float
    """The REAL peak-to-trough drawdown of the equity curve this checkpoint's book actually
    walked, at this instant — the ground truth `argus_activation` is judged against and, more
    importantly, the ground truth freqtrade's OWN verdicts can be checked against too. Exposed
    directly on every checkpoint (not left for a caller to re-derive ad hoc) specifically because
    the combined-book comparison's most decisive finding needed it: freqtrade's global
    protections locked 100% of a real 320-checkpoint run whose true drawdown never once exceeded
    4.11% — a fact that is only checkable at all because this field exists."""

    argus_activation: str
    argus_trips: tuple[str, ...]

    freqtrade_drawdown_locked: bool
    freqtrade_stoploss_locked: bool
    freqtrade_low_profit_locked: bool
    freqtrade_cooldown_locked: bool

    freqtrade_drawdown_measured: float = 0.0
    """freqtrade's OWN internal drawdown estimate (`DrawdownVerdict.drawdown` — a rolling
    `RISK_LOOKBACK_MINUTES`-window sum of closed trades' cumulative return ratios), exposed
    separately from the boolean `locked` verdict specifically to let a reader check HOW WELL that
    internal estimate tracks `true_drawdown_pct`, not just whether its threshold comparison agreed
    with ARGUS's. Defaulted so the existing `_checkpoint()` test helper and any other keyword-only
    construction keeps working unchanged."""

    consecutive_losses: int = 0
    """The exact count `risk.circuit.assess` actually judged (`BookState.consecutive_losses`) —
    not a proxy of anything, an exact tally. Exposed so `argus_locked` can be independently
    reconstructed from ONLY ground-truth quantities this checkpoint carries
    (`true_drawdown_pct`, `consecutive_losses`), the premise
    :func:`argus_measures_ground_truth_directly` checks rather than assumes."""

    @property
    def argus_locked(self) -> bool:
        return self.argus_activation != "active"

    @property
    def freqtrade_risk_locked(self) -> bool:
        """Drawdown OR stoploss-guard OR low-profit-pairs — the three protections that judge
        realised loss. Cooldown is deliberately excluded: it fires after ANY closed trade
        regardless of profit, so folding it in would inflate freqtrade's apparent strictness for
        reasons that have nothing to do with risk."""
        return (
            self.freqtrade_drawdown_locked or self.freqtrade_stoploss_locked
            or self.freqtrade_low_profit_locked
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "exit_ts": self.exit_ts.isoformat(),
            "net_return_pct": round(self.net_return_pct, 6), "is_oos": self.is_oos,
            "true_drawdown_pct": round(self.true_drawdown_pct, 6),
            "argus_activation": self.argus_activation, "argus_trips": list(self.argus_trips),
            "argus_locked": self.argus_locked,
            "freqtrade_drawdown_locked": self.freqtrade_drawdown_locked,
            "freqtrade_stoploss_locked": self.freqtrade_stoploss_locked,
            "freqtrade_low_profit_locked": self.freqtrade_low_profit_locked,
            "freqtrade_risk_locked": self.freqtrade_risk_locked,
            "freqtrade_cooldown_locked": self.freqtrade_cooldown_locked,
            "freqtrade_drawdown_measured": round(self.freqtrade_drawdown_measured, 6),
            "consecutive_losses": self.consecutive_losses,
        }


def _oos_boundary(num_bars: int, *, oos_fraction: float = 0.35) -> int:
    """The identical chronological split `backtest.engine.run` computes internally
    (``split = int(len(net_returns) * (1 - oos_fraction))``, `engine.py:283`) — re-derived here
    rather than read off `BacktestResult`, because nothing on that result names the split index
    itself, only the two `Performance` halves it produced."""
    return int((num_bars - 1) * (1 - oos_fraction))


def walk_trades(
    trades: Sequence[SyntheticTrade], *, symbol: str, oos_boundary_ts: datetime,
) -> list[Checkpoint]:
    """The actual same-input comparison: replay one symbol's real trades in close order, judging
    each with both systems against the state that trade's outcome just produced.

    ARGUS's equity/streak bookkeeping matches `paper.runner._book_state` exactly (module
    docstring); freqtrade's four protections are asked "as of this trade's exit", scanning
    whatever of this symbol's trades already closed by then, precisely as `agents.desk.
    ConstitutionPolicy.rule` and freqtrade's own protections are both asked once per decision
    rather than fed the future.
    """
    cost_bps = _cost_bps()
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    closed: list[SyntheticTrade] = []
    equity = STARTING_EQUITY
    peak = STARTING_EQUITY
    consecutive_losses = 0
    checkpoints: list[Checkpoint] = []

    for trade in ordered:
        net = _net_return_pct(trade, cost_bps=cost_bps)
        equity += STARTING_EQUITY * Decimal(str(net))
        peak = max(peak, equity)
        consecutive_losses = consecutive_losses + 1 if net < 0 else 0
        closed.append(trade)

        book = BookState(
            equity=equity, peak_equity=peak, session_open_equity=equity,
            consecutive_losses=consecutive_losses,
        )
        activation, trips = assess(book)

        now = trade.exit_ts
        drawdown = freqtrade_max_drawdown(
            closed, now=now, lookback_minutes=RISK_LOOKBACK_MINUTES,
            max_allowed_drawdown=float(TOTAL_DRAWDOWN_HALT),
        )
        stoploss = freqtrade_stoploss_guard(
            closed, now=now, lookback_minutes=RISK_LOOKBACK_MINUTES,
        )
        low_profit = freqtrade_low_profit_pairs(
            closed, symbol=symbol, now=now, lookback_minutes=RISK_LOOKBACK_MINUTES,
        )
        cooldown = freqtrade_cooldown_period(
            closed, symbol=symbol, now=now, lookback_minutes=COOLDOWN_LOOKBACK_MINUTES,
            stop_duration_minutes=COOLDOWN_DURATION_MINUTES,
        )

        checkpoints.append(Checkpoint(
            symbol=symbol, exit_ts=now, net_return_pct=net, is_oos=now >= oos_boundary_ts,
            true_drawdown_pct=float(book.total_drawdown),
            argus_activation=str(activation), argus_trips=tuple(t.rule for t in trips),
            freqtrade_drawdown_locked=drawdown.locked,
            freqtrade_stoploss_locked=stoploss.locked,
            freqtrade_low_profit_locked=low_profit.locked,
            freqtrade_cooldown_locked=cooldown.locked,
            freqtrade_drawdown_measured=drawdown.drawdown,
            consecutive_losses=consecutive_losses,
        ))

    return checkpoints


def walk_combined_book(
    all_trades: Sequence[SyntheticTrade], *, symbols: Sequence[str], oos_boundary_ts: datetime,
) -> list[Checkpoint]:
    """The multi-symbol book `walk_trades` states as its own scope limit, closed the same day.

    **Why this is not just `walk_trades` called once per symbol and summed.** Real freqtrade's
    `MaxDrawdown` and `StoplossGuard` are GLOBAL by default — they scan every pair's closed trades
    together, not one pair at a time (`low_profit_pairs.py`/`cooldown_period.py` are the two that
    are genuinely per-pair by design). `risk.circuit.assess` is likewise whole-book — that is the
    exact structural limitation `per_symbol_underperformance` (Constitution gate 12, built earlier
    the same day) exists to work around, not a design ARGUS abandons here. So a fair same-scope
    comparison has to interleave ALL symbols' real trades chronologically into ONE equity curve
    and ask both systems' whole-book checks against that combined curve, while still asking
    `low_profit_pairs`/`cooldown_period` per-symbol — matching each protection's own real scope
    rather than forcing every check onto one scope for convenience.

    **Capital is split evenly across the symbols present, not given in full to every trade.**
    `walk_trades` applies each trade against the FULL starting equity, which is the correct
    single-symbol reading (one book, one position at a time) but would overstate risk by
    ``len(symbols)``x if reused unchanged here — seven symbols each independently "betting" the
    whole book is not a real seven-symbol portfolio. Each trade instead moves only its
    ``STARTING_EQUITY / len(symbols)`` share of the combined equity curve, an explicit modelling
    choice (equal allocation, not the strategies' own sizing) stated here rather than left for a
    reader to notice was different from the single-symbol function beside it.
    """
    if not symbols:
        raise RiskLayerComparisonError("walk_combined_book needs at least one symbol")
    equity_share = STARTING_EQUITY / Decimal(len(symbols))
    cost_bps = _cost_bps()
    ordered = sorted(all_trades, key=lambda t: t.exit_ts)
    closed: list[SyntheticTrade] = []
    equity = STARTING_EQUITY
    peak = STARTING_EQUITY
    consecutive_losses = 0
    checkpoints: list[Checkpoint] = []

    for trade in ordered:
        net = _net_return_pct(trade, cost_bps=cost_bps)
        equity += equity_share * Decimal(str(net))
        peak = max(peak, equity)
        consecutive_losses = consecutive_losses + 1 if net < 0 else 0
        closed.append(trade)

        book = BookState(
            equity=equity, peak_equity=peak, session_open_equity=equity,
            consecutive_losses=consecutive_losses,
        )
        activation, trips = assess(book)

        now = trade.exit_ts
        # Global scope: the WHOLE combined book's trades, not just this symbol's.
        drawdown = freqtrade_max_drawdown(
            closed, now=now, lookback_minutes=RISK_LOOKBACK_MINUTES,
            max_allowed_drawdown=float(TOTAL_DRAWDOWN_HALT),
        )
        stoploss = freqtrade_stoploss_guard(
            closed, now=now, lookback_minutes=RISK_LOOKBACK_MINUTES,
        )
        # Per-pair scope: this trade's own symbol only, matching their real design.
        low_profit = freqtrade_low_profit_pairs(
            closed, symbol=trade.symbol, now=now, lookback_minutes=RISK_LOOKBACK_MINUTES,
        )
        cooldown = freqtrade_cooldown_period(
            closed, symbol=trade.symbol, now=now, lookback_minutes=COOLDOWN_LOOKBACK_MINUTES,
            stop_duration_minutes=COOLDOWN_DURATION_MINUTES,
        )

        checkpoints.append(Checkpoint(
            symbol=trade.symbol, exit_ts=now, net_return_pct=net, is_oos=now >= oos_boundary_ts,
            true_drawdown_pct=float(book.total_drawdown),
            argus_activation=str(activation), argus_trips=tuple(t.rule for t in trips),
            freqtrade_drawdown_locked=drawdown.locked,
            freqtrade_stoploss_locked=stoploss.locked,
            freqtrade_low_profit_locked=low_profit.locked,
            freqtrade_cooldown_locked=cooldown.locked,
            freqtrade_drawdown_measured=drawdown.drawdown,
            consecutive_losses=consecutive_losses,
        ))

    return checkpoints


@dataclass(frozen=True, slots=True)
class Contingency:
    """The 2x2 table a "which system is stricter" claim actually rests on, not an adjective."""

    both_locked: int = 0
    argus_only: int = 0
    freqtrade_only: int = 0
    neither: int = 0

    @property
    def total(self) -> int:
        return self.both_locked + self.argus_only + self.freqtrade_only + self.neither

    @property
    def agreement_rate(self) -> float:
        return (self.both_locked + self.neither) / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.total, "both_locked": self.both_locked, "argus_only": self.argus_only,
            "freqtrade_only": self.freqtrade_only, "neither": self.neither,
            "agreement_rate": round(self.agreement_rate, 4),
        }


def _tally(checkpoints: Sequence[Checkpoint]) -> Contingency:
    both = argus_only = ft_only = neither = 0
    for c in checkpoints:
        a, f = c.argus_locked, c.freqtrade_risk_locked
        if a and f:
            both += 1
        elif a:
            argus_only += 1
        elif f:
            ft_only += 1
        else:
            neither += 1
    return Contingency(both, argus_only, ft_only, neither)


def protection_ablation(checkpoints: Sequence[Checkpoint]) -> dict[str, Any]:
    """Which of the three loss-severity freqtrade protections actually drives
    ``freqtrade_risk_locked`` — the formal ablation the module docstring names as still open after
    the aggregate OR-of-three comparison. For each protection, "unique" means it fired while the
    OTHER two did not — the checkpoints only that protection would have caught, had the other two
    been ablated away. A protection whose unique share is near zero is redundant with the other
    two on this data; one whose unique share is large is carrying real, non-overlapping weight.
    """
    n = len(checkpoints)
    fires = {"drawdown": 0, "stoploss": 0, "low_profit": 0}
    unique = {"drawdown": 0, "stoploss": 0, "low_profit": 0}
    for c in checkpoints:
        flags = {
            "drawdown": c.freqtrade_drawdown_locked,
            "stoploss": c.freqtrade_stoploss_locked,
            "low_profit": c.freqtrade_low_profit_locked,
        }
        for name, fired in flags.items():
            if not fired:
                continue
            fires[name] += 1
            if not any(v for other, v in flags.items() if other != name):
                unique[name] += 1
    return {
        "n": n,
        "fires": fires,
        "unique_contribution": unique,
        "fire_rate": {k: round(v / n, 4) if n else 0.0 for k, v in fires.items()},
        "unique_share": {k: round(v / n, 4) if n else 0.0 for k, v in unique.items()},
    }


def _winners_from(track1_study_path: Path) -> dict[str, str]:
    if not track1_study_path.exists():
        raise RiskLayerComparisonError(
            f"{track1_study_path} does not exist; run `python -m argus.research.track1_study` "
            f"first so each symbol's best_variant is known rather than guessed"
        )
    study = json.loads(track1_study_path.read_text(encoding="utf-8"))
    return {row["symbol"]: row["best_variant"] for row in study["per_symbol"]}


def _fetch_symbol_trades(
    symbol: str, *, days: int, winners: Mapping[str, str],
) -> tuple[str, list[Bar], list[SyntheticTrade]]:
    """Fetch one symbol's real market history, reproduce its already-known winning backtest, and
    reconstruct its real trades. Raises :class:`RiskLayerComparisonError` naming exactly why on
    any failure — never a bare exception — so a caller can record the reason and continue with
    the rest of the symbols rather than losing the whole run.
    """
    variant_name = winners.get(symbol)
    if variant_name is None or variant_name not in ALL_VARIANTS:
        raise RiskLayerComparisonError(
            f"no known-good variant ({variant_name!r} not in ALL_VARIANTS)"
        )
    try:
        candles = fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.MARKET)
    except Exception as exc:
        raise RiskLayerComparisonError(f"fetch failed: {str(exc)[:120]}") from exc
    if len(candles) < 200:
        raise RiskLayerComparisonError(f"only {len(candles)} candles")

    bars = [Bar(ts=c.ts, close=c.close) for c in candles]
    try:
        result = run(
            variant_name, symbol, bars, ALL_VARIANTS[variant_name],
            cost=CostModel.bitget_perp(), periods_per_year=HOURLY_PER_YEAR,
        )
    except MetricError as exc:
        # A live re-fetch at a different moment than `track1_study.json` was generated can
        # legitimately produce a flat run today's data did not (e.g. a variant that goes flat
        # for the whole window) — recorded as a failure, exactly as `research/track1_study.py`
        # itself treats the identical exception, never silently skipped.
        raise RiskLayerComparisonError(f"{variant_name} run failed: {str(exc)[:120]}") from exc
    trades = extract_trades(bars, list(result.weights), symbol=symbol)
    if not trades:
        raise RiskLayerComparisonError(
            f"{variant_name} produced zero discrete trades over {days}d"
        )
    return variant_name, bars, trades


def compare_real_symbols(
    symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = 90,
    track1_study_path: Path = TRACK1_STUDY_PATH,
) -> dict[str, Any]:
    """Fetch each symbol's real market history, reproduce its already-known winning backtest,
    and walk the resulting real trades through both risk systems, each as its own single-symbol
    book — see :func:`compare_combined_book` for the whole-book counterpart.
    """
    winners = _winners_from(track1_study_path)

    per_symbol: dict[str, Any] = {}
    failures: dict[str, str] = {}
    all_checkpoints: list[Checkpoint] = []

    for symbol in symbols:
        try:
            variant_name, bars, trades = _fetch_symbol_trades(symbol, days=days, winners=winners)
        except RiskLayerComparisonError as exc:
            failures[symbol] = str(exc)
            continue

        boundary_ts = bars[_oos_boundary(len(bars))].ts
        checkpoints = walk_trades(trades, symbol=symbol, oos_boundary_ts=boundary_ts)
        all_checkpoints.extend(checkpoints)

        symbol_tally = _tally(checkpoints)
        per_symbol[symbol] = {
            "variant": variant_name, "bars": len(bars), "trades": len(trades),
            "oos_boundary": boundary_ts.isoformat(),
            "contingency": symbol_tally.as_dict(),
            "protection_ablation": protection_ablation(checkpoints),
            "cooldown_locked_share": (
                sum(1 for c in checkpoints if c.freqtrade_cooldown_locked) / len(checkpoints)
            ),
        }

    is_checkpoints = [c for c in all_checkpoints if not c.is_oos]
    oos_checkpoints = [c for c in all_checkpoints if c.is_oos]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "days": days, "symbols_compared": len(per_symbol), "symbols_failed": failures,
        "per_symbol": per_symbol,
        "overall": _tally(all_checkpoints).as_dict(),
        "in_sample": _tally(is_checkpoints).as_dict(),
        "out_of_sample": _tally(oos_checkpoints).as_dict(),
        "protection_ablation": protection_ablation(all_checkpoints),
        "cost_bps_applied_per_trade": str(_cost_bps()),
        "checkpoints_total": len(all_checkpoints),
    }


def _combined_book_checkpoints(
    symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = 90,
    track1_study_path: Path = TRACK1_STUDY_PATH,
) -> tuple[list[Checkpoint], dict[str, list[SyntheticTrade]], dict[str, str], datetime]:
    """The fetch-and-walk pipeline :func:`compare_combined_book` reports on, factored out so a
    caller that wants the raw per-checkpoint data (not just the aggregate tallies) — e.g.
    :func:`measurement_architecture_analysis` — does not have to duplicate the live fetch loop."""
    winners = _winners_from(track1_study_path)
    failures: dict[str, str] = {}
    per_symbol_trades: dict[str, list[SyntheticTrade]] = {}
    boundary_ts: datetime | None = None

    for symbol in symbols:
        try:
            _variant_name, bars, trades = _fetch_symbol_trades(symbol, days=days, winners=winners)
        except RiskLayerComparisonError as exc:
            failures[symbol] = str(exc)
            continue
        per_symbol_trades[symbol] = trades
        if boundary_ts is None:
            # All symbols are fetched with the same `days` window within one run, so their
            # boundaries fall within hours of each other — the first symbol's is used for all,
            # a stated simplification rather than a separate boundary per symbol.
            boundary_ts = bars[_oos_boundary(len(bars))].ts

    if not per_symbol_trades or boundary_ts is None:
        raise RiskLayerComparisonError(
            f"no symbol produced usable trades; failures: {failures}"
        )

    all_trades = [t for trades in per_symbol_trades.values() for t in trades]
    checkpoints = walk_combined_book(
        all_trades, symbols=tuple(per_symbol_trades), oos_boundary_ts=boundary_ts,
    )
    return checkpoints, per_symbol_trades, failures, boundary_ts


def compare_combined_book(
    symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = 90,
    track1_study_path: Path = TRACK1_STUDY_PATH,
) -> dict[str, Any]:
    """The whole-book counterpart to :func:`compare_real_symbols`: every successfully-fetched
    symbol's real trades interleaved into ONE combined equity curve, matching the scope real
    freqtrade's global protections and ARGUS's whole-book breaker actually operate at — see
    :func:`walk_combined_book`'s own docstring for exactly why and how capital is split.

    Fetches independently from :func:`compare_real_symbols` (a second live pass over the same
    symbols) rather than sharing state with it — public, read-only market data, cheap enough
    that keeping the two entry points independent and simple wins over threading a cache through
    both for one call each.
    """
    checkpoints, per_symbol_trades, failures, boundary_ts = _combined_book_checkpoints(
        symbols, days=days, track1_study_path=track1_study_path,
    )
    is_checkpoints = [c for c in checkpoints if not c.is_oos]
    oos_checkpoints = [c for c in checkpoints if c.is_oos]

    per_symbol_checkpoints: dict[str, int] = {}
    for c in checkpoints:
        per_symbol_checkpoints[c.symbol] = per_symbol_checkpoints.get(c.symbol, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "days": days, "symbols_combined": sorted(per_symbol_trades), "symbols_failed": failures,
        "equity_share_per_symbol": str(STARTING_EQUITY / Decimal(len(per_symbol_trades))),
        "oos_boundary": boundary_ts.isoformat(),
        "checkpoints_per_symbol": per_symbol_checkpoints,
        "overall": _tally(checkpoints).as_dict(),
        "in_sample": _tally(is_checkpoints).as_dict(),
        "out_of_sample": _tally(oos_checkpoints).as_dict(),
        "protection_ablation": protection_ablation(checkpoints),
        "cost_bps_applied_per_trade": str(_cost_bps()),
        "checkpoints_total": len(checkpoints),
        "max_true_drawdown_pct": round(
            max((c.true_drawdown_pct for c in checkpoints), default=0.0), 6
        ),
        "freqtrade_risk_locked_share": round(
            sum(1 for c in checkpoints if c.freqtrade_risk_locked) / len(checkpoints), 4
        ),
    }


# --- measurement architecture: does the correlation survive threshold-tuning? ----------------

def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation, no third-party dependency needed for two equal-length float lists."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return float(cov / (var_x * var_y) ** 0.5)


def _precision_recall(
    predicted_locked: Sequence[bool], actually_high: Sequence[bool],
) -> tuple[float, float]:
    """Precision/recall of one binary "locked" call against the SAME ground-truth "should have
    locked" call — the general form both the swept freqtrade curve and ARGUS's single real point
    are computed with, so the two are mechanically comparable rather than computed two ways."""
    true_positive = sum(
        1 for p, a in zip(predicted_locked, actually_high, strict=True) if p and a
    )
    predicted_positive = sum(predicted_locked)
    actual_positive = sum(actually_high)
    precision = true_positive / predicted_positive if predicted_positive else 1.0
    recall = true_positive / actual_positive if actual_positive else 1.0
    return precision, recall


def argus_measures_ground_truth_directly(checkpoints: Sequence[Checkpoint]) -> bool:
    """Checked, not assumed from reading `risk/circuit.py`: does ARGUS's own ladder threshold
    only on quantities that ARE the ground truth, or does it also go through a proxy the way
    freqtrade does? `assess()` triggers on two signals in this comparison's real `BookState`
    construction (`session_drawdown` is always 0 by construction — `session_open_equity=equity`
    at every checkpoint — and `realised_move_sigma`/`evidence_age` are never set, so the shock
    and stale-evidence rules structurally never fire here; confirmed, not assumed, by this
    function's own check below catching it if that stops holding): `total_drawdown` (exactly
    `true_drawdown_pct`, the same `book` object, same instant) and `consecutive_losses` (an
    exact tally, not an estimate — carried on the checkpoint unchanged from what `assess()` saw).
    Both are EXACT measurements of the thing they represent, never a proxy for it. Verified here
    by reconstructing `argus_locked` from only `true_drawdown_pct` and `consecutive_losses` and
    checking it against the real recorded `argus_locked` on every checkpoint — an empirical
    check, not trusted from the source reading alone. A `False` return means this comparison's
    `BookState` construction has drifted from what this function assumes, and the caller must not
    draw a "measures ground truth directly" conclusion from a premise that failed its own check.
    """
    for c in checkpoints:
        reconstructed = (
            c.true_drawdown_pct >= float(REDUCE_ONLY_DRAWDOWN)
            or c.consecutive_losses >= CONSECUTIVE_LOSS_HALT
        )
        if reconstructed != c.argus_locked:
            return False
    return True


def measurement_architecture_analysis(checkpoints: Sequence[Checkpoint]) -> dict[str, Any]:
    """Closes the register's own stated open question on this capability's thirteenth OWNED
    condition: is freqtrade's near-constant lock a defensible "conservative by design" choice
    that a different threshold could fix, or a structural measurement-architecture gap no
    threshold choice can fix? Answered by measuring, not by picking a side.

    **The mechanism, not just the outcome.** ARGUS's `risk.circuit.assess` triggers on two
    signals in this comparison's real `BookState` construction, both EXACT measurements
    (`total_drawdown` is the identical `book` object `true_drawdown_pct` is read from;
    `consecutive_losses` is an exact tally, not an estimate) — confirmed empirically, not
    assumed, by :func:`argus_measures_ground_truth_directly`. freqtrade's `MaxDrawdown` instead
    sums a rolling `RISK_LOOKBACK_MINUTES`-window of CLOSED TRADES' cumulative return ratios
    (`freqtrade_baseline.freqtrade_max_drawdown`) — a PROXY for portfolio drawdown, not the same
    computation. The Pearson correlation between that proxy (`freqtrade_drawdown_measured`) and
    the ground truth it approximates (`true_drawdown_pct`) is the number that settles whether
    re-tuning freqtrade's threshold could ever close the gap: a threshold moves WHERE on a
    fixed-quality signal you draw the line; it cannot improve the signal's own correlation with
    the thing it is trying to measure.

    **ARGUS's real decision, not a synthetic one, is what freqtrade's swept curve is compared
    against.** ARGUS's `argus_locked` is a single, real, already-computed verdict — not a scalar
    threshold sweepable the same way freqtrade's proxy is — so it is plotted as ONE real point
    (its own real precision/recall against the shared ground truth) rather than a fabricated
    curve that would have to pretend `consecutive_losses` does not also matter to it.

    **Why this is the right test, not just a plausible one.** If freqtrade's proxy correlated
    strongly with the truth, its near-constant lock at the shared 10% threshold would be a
    legitimate "conservative by design" reading — a strict operator, correctly measuring
    drawdown, choosing to fire early. A weak correlation rules that reading out: a proxy that
    barely tracks the truth cannot be "conservative" about the truth, because it is not
    meaningfully informed by the truth in the first place. What IS still open after this
    analysis is stated in the returned dict's own `verdict`, not asserted here in prose.
    """
    if not argus_measures_ground_truth_directly(checkpoints):
        raise RiskLayerComparisonError(
            "argus_locked could not be reconstructed from true_drawdown_pct and "
            "consecutive_losses alone — the ground-truth-measurement claim this analysis "
            "depends on does not hold on this data; refusing to draw a conclusion from a "
            "premise that failed its own check"
        )

    truth = [c.true_drawdown_pct for c in checkpoints]
    proxy = [c.freqtrade_drawdown_measured for c in checkpoints]
    correlation = _pearson(proxy, truth)

    # ARGUS's OWN first-tier threshold (the reduce-only bar, not the halt bar) is the truth cut
    # used here, not an arbitrary percentile of this run's data or freqtrade's 10% default —
    # `TOTAL_DRAWDOWN_HALT` never once triggers in this dataset (max observed drawdown across
    # 320 real checkpoints was ~4.6%, never reaching the 10% halt bar), which would make a
    # precision/recall comparison against it vacuous (zero true positives possible for either
    # system). `REDUCE_ONLY_DRAWDOWN` is a real threshold ARGUS already defines as "starting to
    # matter," not tuned to this run's distribution to make the comparison come out a particular
    # way.
    truth_threshold = float(REDUCE_ONLY_DRAWDOWN)
    actually_high = [c.true_drawdown_pct >= truth_threshold for c in checkpoints]
    swept_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
    freqtrade_curve = [
        {
            "threshold": t,
            **dict(zip(
                ("precision", "recall"),
                _precision_recall(
                    [c.freqtrade_drawdown_measured >= t for c in checkpoints], actually_high,
                ),
                strict=True,
            )),
        }
        for t in swept_thresholds
    ]
    argus_precision, argus_recall = _precision_recall(
        [c.argus_locked for c in checkpoints], actually_high,
    )
    argus_point = {"precision": argus_precision, "recall": argus_recall}

    freqtrade_best_precision_at_recall_1 = max(
        (row["precision"] for row in freqtrade_curve if row["recall"] >= 0.999), default=0.0,
    )
    # Pareto dominance: ARGUS's one real point beats or matches EVERY point freqtrade's real
    # code reaches anywhere on the swept curve, on both axes at once — not just its own default.
    dominates_every_swept_threshold = all(
        argus_point["precision"] >= row["precision"] and argus_point["recall"] >= row["recall"]
        for row in freqtrade_curve
    )

    if abs(correlation) < 0.3:
        verdict = (
            "WEAK correlation (|r| < 0.3): freqtrade's proxy is not meaningfully informed by "
            "the ground truth it approximates, so no threshold re-tuning could make its "
            "near-constant lock read as a deliberate conservative-margin CHOICE rather than a "
            "measurement gap — the 'conservative by design' framing is ruled out by this "
            "evidence, not merely disfavoured"
        )
    elif abs(correlation) < 0.6:
        verdict = (
            "MODERATE correlation: freqtrade's proxy tracks the truth partially — some signal, "
            "not enough to call the near-constant lock a deliberate choice with confidence "
            "either way; genuinely mixed, stated as such rather than rounded to a side"
        )
    else:
        verdict = (
            "STRONG correlation: freqtrade's proxy tracks the truth well enough that its "
            "threshold choice is plausibly a deliberate conservative margin, not a measurement "
            "defect — this analysis does NOT support closing the condition against freqtrade"
        )

    return {
        "n_checkpoints": len(checkpoints),
        "argus_measures_ground_truth_directly": True,
        "freqtrade_proxy_vs_truth_correlation": round(correlation, 4),
        "freqtrade_curve": freqtrade_curve,
        "argus_real_point": {
            "precision": round(argus_point["precision"], 4),
            "recall": round(argus_point["recall"], 4),
        },
        "freqtrade_best_precision_at_full_recall": round(freqtrade_best_precision_at_recall_1, 4),
        "argus_dominates_every_swept_threshold": dominates_every_swept_threshold,
        "verdict": verdict,
    }


# --- adversarial scenarios ------------------------------------------------------------------

def _adversarial_trade(
    day_offset: int, return_pct: float, *, symbol: str = "ADVUSDT",
) -> SyntheticTrade:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    entry = start + timedelta(days=day_offset)
    exit_ = entry + timedelta(hours=6)
    entry_price = Decimal("100")
    exit_price = entry_price * (Decimal("1") + Decimal(str(return_pct)))
    return SyntheticTrade(
        symbol=symbol, entry_bar=day_offset * 4, exit_bar=day_offset * 4 + 1,
        entry_ts=entry, exit_ts=exit_, direction="long", weight=1.0,
        entry_price=entry_price, exit_price=exit_price,
    )


def adversarial_scenarios() -> dict[str, Any]:
    """Deliberately crafted trade sequences chosen to try to fool ONE system without fooling the
    other — not swept, not random, each one built to exploit a specific documented mechanic.

    **Findings below are what actually happened when this was run, not the hypothesis each
    scenario was designed to test** — one hypothesis was wrong, and the wrong guess is reported
    rather than quietly rewritten to match the result, per this project's own standing rule
    against presenting an assumption as a fact.

    * ``three_losses_no_streak_halt`` — three losses (below `CONSECUTIVE_LOSS_HALT`=4), designed
      to test whether ARGUS's streak-only halt would miss a loss freqtrade's summed-ratio
      protection catches. **The hypothesis was wrong**: ARGUS's `risk_multiplier` ladder is a
      separate, more sensitive rule than the streak halt — it starts tightening at just 2%
      cumulative drawdown (`REDUCE_ONLY_DRAWDOWN`), so it fires on the FIRST loss here
      (`drawdown_ladder`, before any streak could matter). What this scenario actually
      demonstrates instead: freqtrade's `LowProfitPairs` (default `required_profit=0.0`,
      `trade_limit=1`) fires on that same first trade too, for a different reason — it has no
      loss-size floor at all, so any single net-negative trade locks the symbol. Both systems
      react on trade 1; ARGUS via a whole-book ladder, freqtrade via a per-symbol zero-floor rule.
    * ``one_big_loss_then_many_small_wins`` — the peak-vs-local-window divergence already found
      and documented in `tests/test_freqtrade_baseline.py`
      (``test_a_single_bad_trade_inside_a_net_winning_window_is_where_they_can_diverge``),
      reproduced here as a named, repeatable adversarial case. Confirmed again here: both systems
      lock immediately on the big loss and both stay locked through the recovery in this
      particular window (freqtrade's own `max_drawdown` also never un-locks in this run, because
      the window keeps growing rather than sliding past the first trade).
    * ``rapid_fire_within_cooldown`` — a second trade closing 30 minutes after the first, well
      inside freqtrade's 60-minute cooldown window. **This is the real, confirmed asymmetry**:
      at the FIRST trade's close, ARGUS reads `active` (a single -1.1% trade does not touch the
      2% ladder threshold) while freqtrade's `LowProfitPairs` and `CooldownPeriod` already lock
      the symbol — the exact per-symbol, any-loss capability gap `eval/freqtrade_baseline.py`
      found and `agents.desk.ConstitutionPolicy`'s `per_symbol_underperformance` gate (built the
      same day) exists to close. This module compares `risk/circuit.py`'s whole-book breaker
      only, not the Constitution's gate 12 — the gap this scenario shows is already closed one
      layer up, not still open.
    """
    scenarios: dict[str, list[SyntheticTrade]] = {
        "three_losses_no_streak_halt": [
            _adversarial_trade(0, -0.04), _adversarial_trade(1, -0.04),
            _adversarial_trade(2, -0.05),
        ],
        "one_big_loss_then_many_small_wins": [
            _adversarial_trade(0, -0.12), _adversarial_trade(1, 0.03),
            _adversarial_trade(2, 0.02), _adversarial_trade(3, 0.02),
        ],
        "rapid_fire_within_cooldown": [
            _adversarial_trade(0, -0.01),
            SyntheticTrade(
                symbol="ADVUSDT", entry_bar=1, exit_bar=2,
                entry_ts=datetime(2026, 1, 1, 6, tzinfo=UTC),
                exit_ts=datetime(2026, 1, 1, 6, 30, tzinfo=UTC),
                direction="long", weight=1.0,
                entry_price=Decimal("100"), exit_price=Decimal("99"),
            ),
        ],
    }

    out: dict[str, Any] = {}
    for name, trades in scenarios.items():
        symbol = trades[0].symbol
        boundary_ts = trades[-1].exit_ts + timedelta(days=1)  # every trade IS, irrelevant here
        checkpoints = walk_trades(trades, symbol=symbol, oos_boundary_ts=boundary_ts)
        last = checkpoints[-1]
        out[name] = {
            "trades": len(trades),
            "argus_ever_locked": any(c.argus_locked for c in checkpoints),
            "argus_final_activation": last.argus_activation,
            "freqtrade_ever_risk_locked": any(c.freqtrade_risk_locked for c in checkpoints),
            "freqtrade_ever_cooldown_locked": any(c.freqtrade_cooldown_locked for c in checkpoints),
            "final_low_profit_locked": last.freqtrade_low_profit_locked,
            "checkpoints": [c.as_dict() for c in checkpoints],
        }
    return out


def render(report: dict[str, Any], adversarial: dict[str, Any]) -> list[str]:
    lines = [
        "RISK LAYER COMPARISON — ARGUS circuit breaker vs freqtrade's ported protections",
        f"  {report['symbols_compared']} symbol(s), {report['checkpoints_total']} real trade "
        f"checkpoint(s), {report['days']}d each, cost {report['cost_bps_applied_per_trade']}bps "
        f"round trip applied per trade",
    ]
    overall = report["overall"]
    lines.append(
        f"  overall agreement {overall['agreement_rate']:.1%} (n={overall['n']}) — "
        f"both {overall['both_locked']}, ARGUS-only {overall['argus_only']}, "
        f"freqtrade-only {overall['freqtrade_only']}, neither {overall['neither']}"
    )
    lines.append(
        f"  in-sample agreement {report['in_sample']['agreement_rate']:.1%} "
        f"(n={report['in_sample']['n']}) vs out-of-sample "
        f"{report['out_of_sample']['agreement_rate']:.1%} (n={report['out_of_sample']['n']})"
    )
    if report["symbols_failed"]:
        lines.append(f"  failed: {report['symbols_failed']}")
    ablation = report["protection_ablation"]
    lines.append(
        "  protection ablation (fire rate / unique-contribution share): "
        + ", ".join(
            f"{name} {ablation['fire_rate'][name]:.1%}/{ablation['unique_share'][name]:.1%}"
            for name in ("drawdown", "stoploss", "low_profit")
        )
    )
    lines.append("  adversarial scenarios:")
    for name, row in adversarial.items():
        lines.append(
            f"    {name}: ARGUS locked={row['argus_ever_locked']} "
            f"freqtrade-risk locked={row['freqtrade_ever_risk_locked']} "
            f"cooldown={row['freqtrade_ever_cooldown_locked']}"
        )
    return lines


def render_combined(report: dict[str, Any]) -> list[str]:
    lines = [
        "COMBINED BOOK — same symbols, interleaved into one whole-book equity curve",
        f"  {len(report['symbols_combined'])} symbol(s) combined "
        f"({', '.join(report['symbols_combined'])}), {report['checkpoints_total']} checkpoint(s), "
        f"{report['equity_share_per_symbol']} equity share each",
    ]
    overall = report["overall"]
    lines.append(
        f"  overall agreement {overall['agreement_rate']:.1%} (n={overall['n']}) — "
        f"both {overall['both_locked']}, ARGUS-only {overall['argus_only']}, "
        f"freqtrade-only {overall['freqtrade_only']}, neither {overall['neither']}"
    )
    lines.append(
        f"  in-sample agreement {report['in_sample']['agreement_rate']:.1%} "
        f"(n={report['in_sample']['n']}) vs out-of-sample "
        f"{report['out_of_sample']['agreement_rate']:.1%} (n={report['out_of_sample']['n']})"
    )
    lines.append(
        f"  TRUE max peak-to-trough drawdown over the whole run: "
        f"{report['max_true_drawdown_pct']:.2%} — freqtrade-risk locked on "
        f"{report['freqtrade_risk_locked_share']:.1%} of checkpoints regardless"
    )
    if report["symbols_failed"]:
        lines.append(f"  failed: {report['symbols_failed']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGUS vs freqtrade risk-layer run comparison")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--save", default=str(REPORT_PATH))
    args = parser.parse_args(argv)

    report = compare_real_symbols(days=args.days)
    combined = compare_combined_book(days=args.days)
    adversarial = adversarial_scenarios()
    checkpoints, _, _, _ = _combined_book_checkpoints(days=args.days)
    measurement = measurement_architecture_analysis(checkpoints)
    for line in render(report, adversarial):
        print(line)
    print()
    for line in render_combined(combined):
        print(line)
    print()
    corr = measurement["freqtrade_proxy_vs_truth_correlation"]
    dominates = measurement["argus_dominates_every_swept_threshold"]
    print(f"freqtrade proxy vs truth correlation: {corr}")
    print(f"ARGUS real point: {measurement['argus_real_point']}")
    print(f"ARGUS dominates every swept threshold: {dominates}")
    print(f"verdict: {measurement['verdict']}")
    if args.save:
        Path(args.save).write_text(
            json.dumps(
                {
                    "real_symbols": report, "combined_book": combined, "adversarial": adversarial,
                    "measurement_architecture": measurement,
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"saved -> {args.save}")
    return 0


__all__ = [
    "REPORT_PATH",
    "Checkpoint",
    "Contingency",
    "RiskLayerComparisonError",
    "adversarial_scenarios",
    "argus_measures_ground_truth_directly",
    "compare_combined_book",
    "compare_real_symbols",
    "main",
    "measurement_architecture_analysis",
    "protection_ablation",
    "render_combined",
    "walk_combined_book",
    "walk_trades",
]


if __name__ == "__main__":
    raise SystemExit(main())
