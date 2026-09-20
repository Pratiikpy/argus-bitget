"""Cross-sectional study — the first factors this project could not previously express.

:mod:`argus.research.panel` made cross-sectional factors evaluable. Evaluable is not the same as
worth anything, and the capability register says so: until this module ran, the honest standing was
"proven to evaluate correctly, not yet proven to be worth a basis point". This is the experiment
that settles it.

**What a cross-sectional factor is, and why it is a different bet.** Every Track-1 variant so far
asks "should I be long this instrument". A cross-sectional factor asks "which of these twelve
should I be long, and which short, right now". The difference matters on this venue specifically:
the twelve rTokens share one dominant factor — the US market — so a long-only signal is mostly a
bet on that factor, while a rank-neutral book cancels it and bets only on the dispersion. The panel
run on live bars makes this concrete: a 24-hour momentum rank at the last shared bar went long SQQQ
and short TQQQ, which are the inverse and triple-levered versions of the same index. That trade has
almost no market exposure by construction.

**The construction, and the choices that could flatter it.**

* **Dollar-neutral by construction.** Weights are demeaned across the universe each bar, so the
  book carries no net exposure. This is not a way to improve the number — it is what makes the
  result a statement about the factor rather than about the market's direction over ninety days.
* **Gross exposure normalised to one.** Without it, a factor whose raw values happen to be large
  would look better purely for being loud.
* **Costs are charged on weight changes, at the venue's own per-side rate.** ``CostModel`` carries
  ``taker_bps = 6`` per side, read from :meth:`argus.cost.model.CostModel.bitget_perp` rather than
  assumed, and the charge is ``sum(|change in w|) * 6bps``: every unit of weight moved pays one
  side. Rebalancing twelve names hourly is expensive and the arithmetic says so, which is the
  point.
* **The return used at bar ``i`` is the return realised into bar ``i+1``.** The weight is set from
  information available at ``i`` and earns what happens next. Multiplying weights by contemporaneous
  returns is the most common way a cross-sectional backtest reports an edge it does not have, and
  the index arithmetic here is written to make that impossible rather than to be checked for.

**Both Deflated Sharpe gates are applied**, on the same terms as the single-symbol study: once
against every trial run, and once against the candidates only. The all-trials gate is the honest
one and it is the one quoted first.

**Every rule is also run at every phase of its rebalance cycle**, and scored on the mean rather
than on the phase that happened to be tried. That check found and killed this module's own best
result — see :data:`TRADING_RULES`. On ninety days of live bars, across 80 trials and 1,336
backtests, not one rule produced net Sharpes that agree on sign across its own phases, and the
widest spread is 8.7 Sharpe units.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar
from argus.backtest.metrics import (
    HOURLY_PER_YEAR,
    MetricError,
    deflated_sharpe,
    max_drawdown,
    out_of_sample_decay,
    sharpe,
    sortino,
    stability,
)
from argus.cost.model import CostModel
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range
from argus.research.grammar import (
    BinOp,
    Const,
    CrossRank,
    Expr,
    Field,
    Ref,
    Signal,
    UnOp,
    Window,
)
from argus.research.panel import Panel, build_panel, evaluate_panel

OUT_PATH = Path(__file__).resolve().parents[3] / "data" / "crosssection_study.json"

MIN_BARS = 200
"""Below this a Sharpe is a rumour. Same floor as the single-symbol study, for the same reason."""


def _centred_rank(inner: Expr) -> Expr:
    """rank in 0..1 mapped to a weight in -1..+1, so the top name is long and the bottom short."""
    return BinOp(
        op="sub",
        left=BinOp(op="mul", left=Const(value=2.0), right=CrossRank(operand=inner)),
        right=Const(value=1.0),
    )


def _momentum(hours: int) -> Expr:
    return Window(op="sum", lookback=hours, operand=Ref(field=Field.RETURN_1))


CANDIDATES: dict[str, Signal] = {
    # Cross-sectional momentum at three horizons. The canonical equity factor, and the one the
    # 101 Formulaic Alphas spend most of their vocabulary on.
    "xs_momentum_6h": Signal(operand=_centred_rank(_momentum(6))),
    "xs_momentum_24h": Signal(operand=_centred_rank(_momentum(24))),
    "xs_momentum_72h": Signal(operand=_centred_rank(_momentum(72))),
    # The same ranks inverted. Short-horizon cross-sectional *reversal* is the better-documented
    # effect in equities, and running both directions is how the sweep avoids assuming which way
    # the sign goes — assuming it is how a search quietly becomes a confirmation.
    "xs_reversal_6h": Signal(operand=UnOp(op="neg", operand=_centred_rank(_momentum(6)))),
    "xs_reversal_24h": Signal(operand=UnOp(op="neg", operand=_centred_rank(_momentum(24)))),
    # Rank on realised range: long the calm names, short the turbulent ones. Expressible only
    # since RANGE_BPS was added, and cross-sectional only since the panel existed.
    "xs_low_range": Signal(
        operand=UnOp(
            op="neg",
            operand=_centred_rank(Window(op="mean", lookback=24,
                                         operand=Ref(field=Field.RANGE_BPS))),
        )
    ),
    # Volume shock, ranked. `zscore` of volume against its own recent history, compared across
    # names: which instrument is unusually busy *for itself* relative to the others.
    "xs_volume_shock": Signal(
        operand=_centred_rank(Window(op="zscore", lookback=48, operand=Ref(field=Field.VOLUME)))
    ),
    # Trend slope, ranked. `slope` is one of the operators added in the same expansion.
    "xs_trend_slope": Signal(
        operand=_centred_rank(Window(op="slope", lookback=24, operand=Ref(field=Field.CLOSE)))
    ),
}
"""Eight cross-sectional candidates, every one inexpressible before the grammar expansion.

Deliberately small. Each entry consumes a trial and makes the Deflated Sharpe gate harder for all
of them, so a candidate is here only if there is a published reason to expect it to carry
information — not because the grammar can now write it.
"""


def _or_absent(compute: Callable[[], float]) -> float | str:
    """Run a metric, or name why it could not be computed. Never substitutes a value.

    Returning 0.0 for an undefined Sortino would put a real-looking number in a report, and the row
    that gets it is the row with no losing bars — the best-looking one, where a fabricated figure
    does the most damage.
    """
    try:
        return compute()
    except MetricError as exc:
        return f"undefined: {exc}"


def _stability_or_absent(returns: list[float]) -> dict[str, Any] | str:
    """Rolling 30-day Sharpe stability, or the reason it could not be measured."""
    try:
        return stability(
            returns, window=30 * 24, periods_per_year=HOURLY_PER_YEAR
        ).as_dict()
    except MetricError as exc:
        return f"undefined: {exc}"


@dataclass(frozen=True, slots=True)
class Book:
    """One bar's weights, after neutralisation and normalisation."""

    weights: dict[str, float]

    @property
    def gross(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    @property
    def net(self) -> float:
        return sum(self.weights.values())


def neutralise(raw: dict[str, float]) -> Book:
    """Demean, then scale gross exposure to one. A dead universe stays flat rather than levered.

    Both steps matter for what the result *means*: demeaning removes the shared market factor so
    the number is about dispersion, and normalising removes the effect of a factor simply producing
    larger numbers than another.
    """
    if not raw:
        return Book(weights={})
    mean = sum(raw.values()) / len(raw)
    centred = {s: v - mean for s, v in raw.items()}
    gross = sum(abs(v) for v in centred.values())
    if gross <= 1e-12:
        return Book(weights=dict.fromkeys(centred, 0.0))
    return Book(weights={s: v / gross for s, v in centred.items()})


@dataclass(frozen=True)
class CrossResult:
    """What one cross-sectional factor did, net of cost."""

    name: str
    canonical: str
    periods: int
    net_returns: tuple[float, ...]
    gross_returns: tuple[float, ...]
    turnover_per_bar: tuple[float, ...]
    cost_bps_per_bar: float
    rebalance_every: int = 1
    band: float = 0.0
    phase: int = 0

    @property
    def net_sharpe(self) -> float:
        return sharpe(list(self.net_returns), periods_per_year=HOURLY_PER_YEAR)

    @property
    def gross_sharpe(self) -> float:
        """Before costs. Reported only beside the net figure, never alone.

        It is here because the gap between the two is the most useful single number in this study:
        a factor with a strong gross Sharpe and a negative net one has found something real and
        cannot afford to trade it, which is a different finding from having found nothing.
        """
        return sharpe(list(self.gross_returns), periods_per_year=HOURLY_PER_YEAR)

    @property
    def equity(self) -> list[float]:
        out = [1.0]
        for r in self.net_returns:
            out.append(out[-1] * (1 + r))
        return out

    @property
    def mean_turnover(self) -> float:
        return (
            sum(self.turnover_per_bar) / len(self.turnover_per_bar)
            if self.turnover_per_bar else 0.0
        )

    @property
    def cost_drag_bps_per_bar(self) -> float:
        """What the rebalancing actually cost, per bar, in basis points."""
        return self.mean_turnover * self.cost_bps_per_bar

    @property
    def scorable(self) -> bool:
        """Can this result be scored at all?

        A book that never moved has no variance and therefore no Sharpe. That is a real outcome —
        a factor whose signal never differs across the universe produces exactly it — and the study
        must be able to report it rather than stopping. It still consumed a trial.
        """
        try:
            _ = self.net_sharpe
        except MetricError:
            return False
        return True

    @property
    def gross_is_positive(self) -> bool:
        """Did it make money before costs? False when the gross Sharpe cannot be computed.

        A separate question from :attr:`scorable`, because the interesting population in this
        study is "found something, could not afford to trade it" and that count needs the gross
        side specifically.
        """
        try:
            return self.gross_sharpe > 0
        except MetricError:
            return False

    def as_dict(self) -> dict[str, Any]:
        # Every metric below can legitimately refuse. `argus.backtest.metrics` raises instead of
        # returning NaN, which is the behaviour that makes its gates trustworthy, and a report is
        # the one place that refusal must be caught rather than propagated: a study that cannot
        # print a row because that row has no losing bars is a study that hides its best result.
        # Each absence is named. None of them is replaced by a number.
        half = len(self.net_returns) // 2
        decay: dict[str, float | bool] | str = {}
        if half >= 2:
            try:
                decay = out_of_sample_decay(
                    sharpe(list(self.net_returns[:half]), periods_per_year=HOURLY_PER_YEAR),
                    sharpe(list(self.net_returns[half:]), periods_per_year=HOURLY_PER_YEAR),
                )
            except MetricError as exc:
                decay = f"undefined: {exc}"
        return {
            "name": self.name,
            "canonical": self.canonical,
            "periods": self.periods,
            "rebalance_every": self.rebalance_every,
            "band": self.band,
            "net_sharpe": _or_absent(lambda: round(self.net_sharpe, 3)),
            "gross_sharpe": _or_absent(lambda: round(self.gross_sharpe, 3)),
            "sortino": _or_absent(
                lambda: round(sortino(list(self.net_returns),
                                      periods_per_year=HOURLY_PER_YEAR), 3)
            ),
            "max_drawdown_pct": round(100 * max_drawdown(self.equity), 3),
            "total_return_pct": round(100 * (self.equity[-1] - 1.0), 3),
            "mean_turnover_per_bar": round(self.mean_turnover, 4),
            "cost_drag_bps_per_bar": round(self.cost_drag_bps_per_bar, 3),
            "win_rate_pct": round(
                100 * sum(1 for r in self.net_returns if r > 0)
                / max(1, sum(1 for r in self.net_returns if r != 0)), 2
            ),
            "out_of_sample_decay": decay,
            "rolling_sharpe_stability": _stability_or_absent(list(self.net_returns)),
        }


def backtest(
    factor: Signal,
    panel: Panel,
    *,
    name: str,
    cost: CostModel,
    rebalance_every: int = 1,
    band: float = 0.0,
    phase: int = 0,
    precomputed: Mapping[str, Sequence[float]] | None = None,
) -> CrossResult:
    """Run one cross-sectional factor over the panel and net its costs.

    The index arithmetic is the load-bearing part. ``weights[i]`` is built from bars up to and
    including ``i``; the return it earns is the move from ``i`` to ``i + 1``. The loop therefore
    stops one short of the end, and there is no path through this function in which a weight meets
    a return it could have seen.

    **Between rebalances the book drifts rather than being held constant.** A position whose price
    rose is a larger share of the book the next hour, and pretending otherwise would credit the
    strategy with a free rebalance every bar — which is precisely the cost this parameter exists to
    avoid paying. Weights are *not* renormalised on drift bars, because renormalising is itself a
    trade.

    ``band`` is a no-trade threshold in weight units: a symbol whose target has moved less than
    this is left alone. It is the standard way to keep a high-turnover signal affordable, and it is
    applied per symbol rather than to the book as a whole, because a book-level test would trade
    every name whenever any one of them moved.
    """
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1 bar")
    if not 0 <= phase < rebalance_every:
        raise ValueError(
            f"phase {phase} must be in [0, {rebalance_every}); it selects which bar of the "
            f"rebalance cycle the book trades on"
        )
    if band < 0:
        raise ValueError("a no-trade band cannot be negative")
    per_symbol = precomputed if precomputed is not None else evaluate_panel(factor, panel)
    symbols = panel.symbols
    side_bps = float(cost.taker_bps)

    held: dict[str, float] = dict.fromkeys(symbols, 0.0)
    net: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []

    for i in range(panel.length - 1):
        traded = 0.0
        if (i - phase) % rebalance_every == 0:
            target = neutralise({s: per_symbol[s][i] for s in symbols}).weights
            for s in symbols:
                move = target[s] - held[s]
                if abs(move) > band:
                    held[s] = target[s]
                    traded += abs(move)
        cost_fraction = traded * side_bps / 10_000.0

        period_return = 0.0
        drifted: dict[str, float] = {}
        for s in symbols:
            bars = panel.bars[s]
            before, after = float(bars[i].close), float(bars[i + 1].close)
            r = (after - before) / before if before > 0 else 0.0
            period_return += held[s] * r
            drifted[s] = held[s] * (1 + r)

        gross_returns.append(period_return)
        net.append(period_return - cost_fraction)
        turnovers.append(traded)
        held = drifted

    return CrossResult(
        name=name,
        canonical=factor.canonical(),
        periods=len(net),
        net_returns=tuple(net),
        gross_returns=tuple(gross_returns),
        turnover_per_bar=tuple(turnovers),
        cost_bps_per_bar=side_bps,
        rebalance_every=rebalance_every,
        band=band,
        phase=phase,
    )


@dataclass(frozen=True)
class RuleResult:
    """One (factor, trading rule) pair, run at **every** phase of its rebalance cycle.

    This exists because reporting a single phase is not reporting the rule. A 72-hour rebalance has
    seventy-two possible starting bars, and on live data the choice moved the net Sharpe of
    ``xs_reversal_6h`` from -0.05 to -2.41 — a spread of more than two whole Sharpe units produced
    by nothing but which hour of the cycle the book happened to trade on. An earlier version of
    this module ran phase 0 only and reported +2.28 for that rule, which was the best phase of a
    noise distribution wearing the clothes of a result.

    The Deflated Sharpe gate does not catch this, and that is the important part: the phases are
    not separate trials in the search, so the trial count never rises to account for them. The only
    defence is to stop picking one.

    So the headline figure is the **mean across phases**, and the spread is published beside it. A
    rule whose phases disagree is a rule that has not found anything, however good its best phase
    looks.
    """

    name: str
    canonical: str
    rebalance_every: int
    band: float
    phases: tuple[CrossResult, ...]

    @property
    def scorable_phases(self) -> tuple[CrossResult, ...]:
        return tuple(r for r in self.phases if r.scorable)

    @property
    def net_sharpes(self) -> tuple[float, ...]:
        return tuple(r.net_sharpe for r in self.scorable_phases)

    @property
    def scorable(self) -> bool:
        return bool(self.scorable_phases)

    @property
    def mean_net_sharpe(self) -> float:
        values = self.net_sharpes
        return sum(values) / len(values) if values else 0.0

    @property
    def spread(self) -> float:
        """Best phase minus worst. The number that says whether the mean means anything."""
        values = self.net_sharpes
        return max(values) - min(values) if values else 0.0

    @property
    def phases_agree(self) -> bool:
        """Do all scorable phases at least share a sign?

        A weaker test than statistical significance and a much harder one to fake. A rule whose
        phases disagree on whether the strategy makes or loses money has not established anything.
        """
        values = self.net_sharpes
        return bool(values) and (all(v > 0 for v in values) or all(v < 0 for v in values))

    @property
    def mean_gross_sharpe(self) -> float:
        """Before cost, averaged the same way. Reported only beside the net figure."""
        values = [r.gross_sharpe for r in self.scorable_phases]
        return sum(values) / len(values) if values else 0.0

    @property
    def best_phase(self) -> CrossResult | None:
        scorable = self.scorable_phases
        return max(scorable, key=lambda r: r.net_sharpe) if scorable else None

    def as_dict(self) -> dict[str, Any]:
        values = self.net_sharpes
        representative = self.phases[0].as_dict()
        return {
            "name": self.name,
            "canonical": self.canonical,
            "rebalance_every": self.rebalance_every,
            "band": self.band,
            "phases_run": len(self.phases),
            "phases_scorable": len(self.scorable_phases),
            "mean_net_sharpe": round(self.mean_net_sharpe, 3),
            "best_phase_net_sharpe": round(max(values), 3) if values else None,
            "worst_phase_net_sharpe": round(min(values), 3) if values else None,
            "phase_spread": round(self.spread, 3),
            "phases_agree_on_sign": self.phases_agree,
            "mean_gross_sharpe": round(self.mean_gross_sharpe, 3) if self.scorable else None,
            "mean_turnover_per_bar": round(
                sum(r.mean_turnover for r in self.phases) / len(self.phases), 4
            ),
            "cost_drag_bps_per_bar": round(
                sum(r.cost_drag_bps_per_bar for r in self.phases) / len(self.phases), 3
            ),
            "phase_0": {
                k: representative[k]
                for k in ("net_sharpe", "gross_sharpe", "sortino", "max_drawdown_pct",
                          "total_return_pct", "win_rate_pct", "out_of_sample_decay",
                          "rolling_sharpe_stability")
            },
        }


def across_phases(
    factor: Signal,
    panel: Panel,
    *,
    name: str,
    cost: CostModel,
    rebalance_every: int,
    band: float,
    precomputed: Mapping[str, Sequence[float]] | None = None,
) -> RuleResult:
    """Run one rule at every phase of its rebalance cycle.

    Every phase, not a sample of them: a sampled subset would reintroduce the same arbitrary choice
    one level up. A 72-hour rule therefore costs 72 backtests, which is the price of the number
    meaning what it says.
    """
    values = precomputed if precomputed is not None else evaluate_panel(factor, panel)
    return RuleResult(
        name=name,
        canonical=factor.canonical(),
        rebalance_every=rebalance_every,
        band=band,
        phases=tuple(
            backtest(
                factor, panel, name=name, cost=cost, rebalance_every=rebalance_every,
                band=band, phase=phase, precomputed=values,
            )
            for phase in range(rebalance_every)
        ),
    )


def load_panel(*, days: int = 90, symbols: tuple[str, ...] = RTOKEN_SYMBOLS) -> Panel:
    """Hourly market candles for the universe, aligned. Volume, high and low carried through."""
    series: dict[str, list[Bar]] = {}
    for symbol in symbols:
        try:
            candles = fetch_range(
                symbol, days=days, interval="1H", candle_type=CandleType.MARKET
            )
        except Exception:
            # One dead symbol must not lose the universe; the panel reports the width it got.
            continue
        if len(candles) < MIN_BARS:
            continue
        series[symbol] = [
            Bar(
                ts=c.ts,
                close=c.close,
                extra={
                    "volume": float(c.volume),
                    "high": float(c.high),
                    "low": float(c.low),
                },
            )
            for c in candles
        ]
    return build_panel(series)


TRADING_RULES: tuple[tuple[int, float], ...] = (
    (1, 0.0), (4, 0.0), (12, 0.0), (24, 0.0), (72, 0.0),
    (1, 0.02), (1, 0.05), (4, 0.05), (24, 0.05), (24, 0.10),
)
"""(rebalance_every_bars, no_trade_band) pairs swept for every candidate.

**Every pair is a trial and is counted as one.** Eight factors across ten rules is eighty trials,
not eight, and the Deflated Sharpe gate is charged for all eighty.

The phases are **not** trials, and the distinction is the one that matters here. A rule is run at
every phase of its rebalance cycle by :func:`across_phases` and scored on the mean, because the
first version of this module ran phase 0 only and reported a net Sharpe of +2.28 for a 72-hour
cross-sectional reversal — the first net-positive cross-sectional result this project had produced.
It was not a result. Running the same rule on the same data at every starting hour gives a worst
phase of -3.84 and a best of +4.36, and a mean of +0.14. The number that looked like a discovery
was the top of an eight-Sharpe-wide noise distribution, selected by an arbitrary alignment nobody
would have thought to report.

The Deflated Sharpe gate cannot catch that, which is why it is worth writing down: the phases are
not separate searches, so the trial count never rises to account for them and the gate never sees
the selection. The only defence is to stop picking one, and the artefact publishes the spread
beside every mean so a reader can see how wide the distribution was.

The grid is small and hand-chosen for the same reason. Rebalance horizons span hourly to three days
because the cost arithmetic says that is where the answer changes; bands span nothing to ten
percent of the book. A denser grid would find a better-looking maximum and deserve less belief.
"""


def study(*, days: int = 90, out: Path = OUT_PATH) -> dict[str, Any]:
    """Sweep every candidate over every trading rule, at every phase, and write it down.

    Two counts matter and they are different numbers. **Trials** is factors times rules, because
    that is what the search chose between and it is what the Deflated Sharpe gate must be charged
    for. **Backtests** is much larger, because each rule is run at every phase of its rebalance
    cycle — those are not trials, they are one trial measured properly, and inflating the trial
    count with them would make the gate harsher for the wrong reason.
    """
    panel = load_panel(days=days)
    cost = CostModel.bitget_perp()
    values = {name: evaluate_panel(f, panel) for name, f in CANDIDATES.items()}

    rules: list[RuleResult] = []
    for name, factor in CANDIDATES.items():
        for rebalance, band in TRADING_RULES:
            rules.append(
                across_phases(
                    factor, panel, name=name, cost=cost,
                    rebalance_every=rebalance, band=band, precomputed=values[name],
                )
            )

    # The rule's score is the mean over its phases, never its best phase.
    scorable = [r for r in rules if r.scorable]
    sharpes = [r.mean_net_sharpe for r in scorable]
    trials = len(rules)
    backtests = sum(len(r.phases) for r in rules)
    if not sharpes:
        raise MetricError("no rule produced a scorable Sharpe; there is nothing to deflate")
    mean = sum(sharpes) / len(sharpes)
    variance = sum((s - mean) ** 2 for s in sharpes) / max(1, len(sharpes) - 1)
    positives = [s for s in sharpes if s > 0]

    rows: list[dict[str, Any]] = []
    for rule in rules:
        row = rule.as_dict()
        if not rule.scorable:
            row["dsr_all_trials"] = "undefined: this rule has no Sharpe to deflate"
            row["dsr_candidates_only"] = row["dsr_all_trials"]
            rows.append(row)
            continue
        periods = rule.phases[0].periods
        for label, n_trials in (
            ("dsr_all_trials", trials),
            # Candidates-only: the softer gate, counting just the rules with a positive mean,
            # which is what a study that quietly drops its failures is really reporting.
            ("dsr_candidates_only", max(1, len(positives))),
        ):
            try:
                row[label] = round(
                    deflated_sharpe(
                        rule.mean_net_sharpe, n=periods,
                        trials=n_trials, variance_of_trials=variance,
                    ), 4
                )
            except MetricError as exc:
                row[label] = f"error: {exc}"
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["mean_net_sharpe"] if isinstance(r["mean_net_sharpe"], float) else float("-inf")
        ),
        reverse=True,
    )

    def _survived(key: str) -> int:
        return sum(1 for r in rows if isinstance(r[key], float) and r[key] >= 0.95)

    agreeing = [r for r in scorable if r.phases_agree and r.mean_net_sharpe > 0]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": days,
        "per_side_fee_bps": float(cost.taker_bps),
        "panel": panel.as_dict(),
        "factors": len(CANDIDATES),
        "trading_rules": len(TRADING_RULES),
        "trials": trials,
        "backtests_run": backtests,
        "headline": {
            "survived_dsr_all_trials": _survived("dsr_all_trials"),
            "survived_dsr_candidates_only": _survived("dsr_candidates_only"),
            "positive_mean_net_sharpe": len(positives),
            "positive_mean_gross_sharpe": sum(
                1 for r in scorable if r.mean_gross_sharpe > 0
            ),
            "profitable_and_phase_consistent": len(agreeing),
            "unscorable_rules": trials - len(scorable),
            "widest_phase_spread": round(
                max((r.spread for r in scorable), default=0.0), 3
            ),
            "best": rows[0]["name"] if rows else "",
            "best_rule": (
                f"{rows[0]['rebalance_every']}h rebalance, band {rows[0]['band']}"
                if rows else ""
            ),
            "best_mean_net_sharpe": rows[0]["mean_net_sharpe"] if rows else 0.0,
            "best_phase_spread": rows[0]["phase_spread"] if rows else 0.0,
        },
        "results": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:  # pragma: no cover - CLI
    import contextlib
    import sys

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    report = study()
    head = report["headline"]
    print(f"panel: {report['panel']['symbols']}")
    print(f"{report['factors']} factor(s) x {report['trading_rules']} rule(s) = "
          f"{report['trials']} trials, run at every phase = {report['backtests_run']} backtests")
    print(f"{report['window_days']} days, {report['per_side_fee_bps']}bps per side")
    print(f"{head['positive_mean_gross_sharpe']} positive before cost, "
          f"{head['positive_mean_net_sharpe']} after, "
          f"{head['profitable_and_phase_consistent']} of those agree across every phase")
    print(f"DSR all trials: {head['survived_dsr_all_trials']}/{report['trials']}   "
          f"candidates only: {head['survived_dsr_candidates_only']}/{report['trials']}")
    print(f"widest phase spread: {head['widest_phase_spread']} Sharpe units")
    print()
    print(f"{'factor':18} {'rule':11} {'gross':>7} {'net(mean)':>10} {'worst':>8} "
          f"{'best':>8} {'agree':>6} {'dsr':>6}")
    for row in report["results"][:12]:
        rule = f"{row['rebalance_every']}h/b{row['band']:.2f}"
        dsr = row["dsr_all_trials"] if isinstance(row["dsr_all_trials"], float) else 0.0
        print(f"{row['name']:18} {rule:11} {row['mean_gross_sharpe'] or 0:7.3f} "
              f"{row['mean_net_sharpe']:10.3f} {row['worst_phase_net_sharpe'] or 0:8.3f} "
              f"{row['best_phase_net_sharpe'] or 0:8.3f} "
              f"{row['phases_agree_on_sign']!s:>6} {dsr:6.3f}")
    return 0


__all__ = [
    "CANDIDATES",
    "MIN_BARS",
    "OUT_PATH",
    "Book",
    "CrossResult",
    "backtest",
    "load_panel",
    "neutralise",
    "study",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
