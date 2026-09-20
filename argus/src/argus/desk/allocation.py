"""Hierarchical risk parity, and the trade that moves a book toward it.

`desk/portfolio.py` can grade a trade it is handed — beta before and after, risk share, effective
positions, the holding it is most correlated with. What it could never do is **propose** one. The
desk could ask "is this allocation sensible?" and never "what allocation would be better?", which
leaves the most consequential decision in the system unexamined: the model picks the instrument, and
nothing checks the shape of the book that results.

**Why HRP and not mean-variance.** Markowitz needs the covariance matrix inverted, and these twelve
instruments are tokenized US equities whose hourly returns correlate at 0.9 and above. A near-
singular
matrix inverts into enormous offsetting long/short weights that are an artefact of estimation noise,
which is the failure López de Prado's 2016 paper was written about. Hierarchical risk parity never
inverts anything: it clusters, orders, and splits risk down the tree.

**Read before written**, as the standing rule requires. The reference is PyPortfolioOpt (MIT),
`repos/PyPortfolioOpt/pypfopt/hierarchical_portfolio.py`:

* `hierarchical_portfolio.py:188` — the distance matrix is ``sqrt((1 - corr) / 2)``, clipped to
  [0, 1] before the square root "to avoid some nasty floating point issues". We clip too, for the
  same reason: a correlation of 1.0000000002 makes a negative radicand.
* `hierarchical_portfolio.py:191-194` — single linkage by default, then quasi-diagonalisation by
  pre-order traversal of the tree.
* `hierarchical_portfolio.py:143-160` — the recursive bisection: split each cluster in half, compute
  each half's variance under **inverse-variance weights** (`:103-105`), and allocate
  ``alpha = 1 - V1 / (V1 + V2)`` to the first.

One deliberate difference from the paper it implements. De Prado's Chapter 16 code clusters on the
Euclidean distance *between the columns* of the correlation-distance matrix — a distance of
distances. PyPortfolioOpt clusters on the correlation-distance directly. We follow PyPortfolioOpt,
because that is the implementation we read and can reproduce; the choice is noted here rather than
left as an undocumented divergence from the paper.

**What this module adds that neither has.** Both stop at a weight vector. A weight vector is not a
decision — the decision is whether moving from the book you hold to the book you want is worth what
the move costs, and on this venue the move costs 6bps per side of everything you touch. So
:func:`optimize_trade` prices the turnover, computes the variance reduction it buys, and reports the
**break-even holding period**: how long the improved book must be held before the rebalance pays for
itself, at a stated Sharpe assumption. Where that period exceeds the horizon, the honest answer is
*do not rebalance*, and it is the answer this module gives most of the time on these instruments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MIN_ASSETS = 3
"""Below this HRP degenerates to inverse-variance weighting, which is a different method.

Two assets have exactly one split and no hierarchy to exploit; reporting that as "hierarchical risk
parity" would dress up a two-line calculation in a paper's name.
"""

MIN_OBSERVATIONS = 60
"""Fewest aligned return observations before a covariance is worth clustering on."""

TAKER_BPS = 6.0
"""Bitget taker fee per side, measured. Every unit of turnover crosses it once."""

TRADING_HOURS_PER_YEAR = 24 * 365
"""These tokens trade continuously, unlike their anchors. Used only to annualise a vol for display,
never to annualise a Sharpe estimated on too little data."""


class AllocationError(ValueError):
    """Raised rather than returning weights computed from a degenerate covariance."""


# --- clustering ----------------------------------------------------------------------------------


def correlation_distance(cov: Sequence[Sequence[float]]) -> list[list[float]]:
    """``sqrt((1 - rho) / 2)``, the correlation distance.

    It maps rho = 1 to 0, rho = 0 to sqrt(0.5) = 0.707 and rho = -1 to 1, so it is a proper distance
    bounded in [0, 1]. Clipping before the root is not defensive decoration — a sample
    correlation of 1 + 1e-12 produces a negative radicand and a domain error
    (`hierarchical_portfolio.py:188` clips for exactly this).
    """
    size = len(cov)
    out: list[list[float]] = []
    for i in range(size):
        row: list[float] = []
        for j in range(size):
            denominator = math.sqrt(cov[i][i] * cov[j][j])
            if denominator <= 0:
                raise AllocationError(
                    "an instrument with zero variance cannot be clustered; it has no risk to "
                    "allocate and no correlation to measure"
                )
            rho = cov[i][j] / denominator
            row.append(math.sqrt(min(1.0, max(0.0, (1.0 - rho) / 2.0))))
        out.append(row)
    return out


@dataclass(frozen=True, slots=True)
class Merge:
    """One row of a linkage matrix, in scipy's format: two cluster ids, their distance, the size."""

    left: int
    right: int
    distance: float
    size: int


def single_linkage(distance: Sequence[Sequence[float]]) -> list[Merge]:
    """Agglomerative single-linkage clustering, returning scipy-shaped merges.

    Naive O(n^3) by design. scipy uses SLINK, which is O(n^2) and matters at ten thousand leaves;
    twelve instruments make 66 pairs and the whole clustering costs less than one HTTP round trip.
    Choosing the simple algorithm here is a readability decision, not a performance oversight — and
    single linkage is exact under this procedure, so the tree is identical to scipy's rather than an
    approximation of it.
    """
    n = len(distance)
    if n < 2:
        raise AllocationError("clustering needs at least two items")
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    merges: list[Merge] = []
    next_id = n
    while len(members) > 1:
        best: tuple[float, int, int] | None = None
        ids = sorted(members)
        for a_index, a in enumerate(ids):
            for b in ids[a_index + 1:]:
                # Single linkage: the distance between clusters is the closest pair across them.
                gap = min(distance[i][j] for i in members[a] for j in members[b])
                if best is None or gap < best[0]:
                    best = (gap, a, b)
        assert best is not None
        gap, a, b = best
        merged = members.pop(a) + members.pop(b)
        members[next_id] = merged
        merges.append(Merge(left=a, right=b, distance=gap, size=len(merged)))
        next_id += 1
    return merges


def quasi_diagonal(merges: Sequence[Merge], leaves: int) -> list[int]:
    """Leaf order from a pre-order traversal of the tree (`hierarchical_portfolio.py:122`).

    The ordering is what makes the bisection meaningful: adjacent leaves are similar, so splitting
    the list in half splits the tree at its widest gap rather than at an arbitrary point.
    """
    if not merges:
        return list(range(leaves))
    children: dict[int, tuple[int, int]] = {
        leaves + i: (m.left, m.right) for i, m in enumerate(merges)
    }
    root = leaves + len(merges) - 1
    order: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node < leaves:
            order.append(node)
            continue
        left, right = children[node]
        stack.extend((right, left))  # pop order puts `left` first
    return order


def _cluster_variance(cov: Sequence[Sequence[float]], items: Sequence[int]) -> float:
    """Variance of the inverse-variance portfolio of a cluster (`hierarchical_portfolio.py:103`)."""
    inverse = [1.0 / cov[i][i] for i in items]
    total = sum(inverse)
    weights = [w / total for w in inverse]
    return sum(
        weights[a] * cov[items[a]][items[b]] * weights[b]
        for a in range(len(items))
        for b in range(len(items))
    )


def hrp_weights(names: Sequence[str], cov: Sequence[Sequence[float]]) -> dict[str, float]:
    """Hierarchical risk parity weights, summing to one, all non-negative.

    Long-only falls out of the construction rather than being imposed: every step multiplies a
    positive weight by a split in [0, 1], so no short position can appear. That is one of HRP's real
    advantages over an unconstrained mean-variance solution on correlated assets, where the
    optimiser
    expresses estimation noise as large offsetting longs and shorts.
    """
    if len(names) != len(cov):
        raise AllocationError("the covariance matrix does not match the instrument list")
    if len(names) < MIN_ASSETS:
        raise AllocationError(
            f"{len(names)} instrument(s) is below the {MIN_ASSETS} that make a hierarchy; "
            f"use inverse-variance weights and call them that"
        )
    distance = correlation_distance(cov)
    order = quasi_diagonal(single_linkage(distance), len(names))

    weights = dict.fromkeys(order, 1.0)
    clusters: list[list[int]] = [list(order)]
    while clusters:
        split: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) > 1:
                middle = len(cluster) // 2
                split.append(cluster[:middle])
                split.append(cluster[middle:])
        clusters = split
        for index in range(0, len(clusters), 2):
            first, second = clusters[index], clusters[index + 1]
            first_var = _cluster_variance(cov, first)
            second_var = _cluster_variance(cov, second)
            if first_var + second_var <= 0:
                continue
            alpha = 1.0 - first_var / (first_var + second_var)
            for i in first:
                weights[i] *= alpha
            for i in second:
                weights[i] *= 1.0 - alpha
    return {names[i]: weights[i] for i in sorted(weights)}


# --- what the weights are worth ------------------------------------------------------------------


def portfolio_variance(weights: Mapping[str, float], names: Sequence[str],
                       cov: Sequence[Sequence[float]]) -> float:
    index = {name: i for i, name in enumerate(names)}
    return sum(
        weights.get(a, 0.0) * cov[index[a]][index[b]] * weights.get(b, 0.0)
        for a in index for b in index
    )


def diversification_ratio(weights: Mapping[str, float], names: Sequence[str],
                          cov: Sequence[Sequence[float]]) -> float | None:
    """Weighted average volatility divided by portfolio volatility (Choueifaty-Coignard 2008).

    One at perfect correlation, larger when the holdings genuinely offset. Reported because on these
    instruments it is the number that shows how little diversification is available: a book of
    twelve
    tokenized US equities is closer to one position than to twelve, and a method that cannot say so
    would be flattering the allocation it just produced.
    """
    index = {name: i for i, name in enumerate(names)}
    weighted = sum(
        abs(weights.get(name, 0.0)) * math.sqrt(cov[i][i]) for name, i in index.items()
    )
    total = portfolio_variance(weights, names, cov)
    if total <= 0 or weighted <= 0:
        return None
    return weighted / math.sqrt(total)


@dataclass(frozen=True, slots=True)
class Trade:
    """One leg of a rebalance, as a change in weight."""

    symbol: str
    weight_before: float
    weight_after: float

    @property
    def delta(self) -> float:
        return self.weight_after - self.weight_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "weight_before": round(self.weight_before, 6),
            "weight_after": round(self.weight_after, 6),
            "delta": round(self.delta, 6),
        }


@dataclass(frozen=True, slots=True)
class TradePlan:
    """The move from the book you hold to the book HRP wants, priced.

    Every field is a number the caller can check. The verdict is assembled from them by rule, never
    narrated: a rebalance recommendation is exactly the kind of statement that should not depend on
    who is describing it.
    """

    trades: tuple[Trade, ...]
    vol_before: float
    vol_after: float
    turnover: float
    """Sum of absolute weight changes. One unit of turnover crosses the fee once."""

    cost_bps: float
    assumed_sharpe: float
    horizon_bars: int

    @property
    def annual_sharpe(self) -> float:
        """The assumption, annualised for display. Never a measurement; see
        :func:`optimize_trade`."""
        return self.assumed_sharpe * math.sqrt(TRADING_HOURS_PER_YEAR)

    @property
    def variance_reduction(self) -> float:
        """Share of portfolio variance the rebalance removes. Negative if it adds risk."""
        if self.vol_before <= 0:
            return 0.0
        return 1.0 - (self.vol_after ** 2) / (self.vol_before ** 2)

    @property
    def break_even_bars(self) -> float | None:
        """How long the new book must be held before the turnover is repaid.

        **This number requires an assumption and the assumption is stated rather than hidden.**
        Reducing volatility does not by itself earn anything — at a constant Sharpe, a less volatile
        book earns proportionally less. The benefit appears only when the book is scaled to a
        volatility target, at which point the lower-volatility allocation can be held larger by
        ``vol_before / vol_after`` and earns that multiple more.

        So with a per-bar Sharpe ``s`` and a target volatility equal to the current one, the gain
        per
        bar is ``s * vol_before * (vol_before / vol_after - 1)`` and the break-even is the one-off
        cost divided by it. ``None`` when the rebalance buys no volatility improvement, because then
        there is no horizon at which it pays.
        """
        if self.vol_after <= 0 or self.vol_before <= 0:
            return None
        multiple = self.vol_before / self.vol_after
        if multiple <= 1.0:
            return None
        gain_per_bar_bps = self.assumed_sharpe * self.vol_before * (multiple - 1.0) * 10_000
        if gain_per_bar_bps <= 0:
            return None
        return self.cost_bps / gain_per_bar_bps

    @property
    def worth_doing(self) -> bool:
        """Pays for itself inside the stated horizon. Nothing else counts as an improvement."""
        payback = self.break_even_bars
        return payback is not None and payback <= self.horizon_bars

    @property
    def verdict(self) -> str:
        if not self.trades:
            return "The book is already at the hierarchical-risk-parity allocation; no trade"
        moved = ", ".join(
            f"{t.symbol} {t.weight_before:.1%}->{t.weight_after:.1%}" for t in self.trades[:4]
        )
        head = (
            f"{len(self.trades)} leg(s) ({moved}{'...' if len(self.trades) > 4 else ''}), "
            f"turnover {self.turnover:.1%} costing {self.cost_bps:.1f}bps. Portfolio volatility "
            f"{self.vol_before * 10_000:.0f} -> {self.vol_after * 10_000:.0f}bps per bar, "
            f"{self.variance_reduction:+.1%} of variance."
        )
        payback = self.break_even_bars
        if payback is None:
            return head + (
                " The rebalance does not lower volatility, so there is no horizon at which the "
                "turnover repays itself. Do not trade"
            )
        if not self.worth_doing:
            return head + (
                f" At an assumed annualised Sharpe of {self.annual_sharpe:.1f} it repays in "
                f"{payback:.0f} bars, beyond the {self.horizon_bars}-bar "
                f"horizon. Do not trade — the improvement is real and smaller than its cost"
            )
        return head + (
            f" It repays in {payback:.0f} bars, inside the {self.horizon_bars}-bar horizon, at an "
            f"assumed annualised Sharpe of {self.annual_sharpe:.1f}. Worth doing"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades": [t.as_dict() for t in self.trades],
            "vol_before_bps": round(self.vol_before * 10_000, 4),
            "vol_after_bps": round(self.vol_after * 10_000, 4),
            "variance_reduction": round(self.variance_reduction, 6),
            "turnover": round(self.turnover, 6),
            "cost_bps": round(self.cost_bps, 4),
            "break_even_bars": (
                None if self.break_even_bars is None else round(self.break_even_bars, 2)
            ),
            "assumed_sharpe_per_bar": self.assumed_sharpe,
            "horizon_bars": self.horizon_bars,
            "worth_doing": self.worth_doing,
            "verdict": self.verdict,
        }


def optimize_trade(
    current: Mapping[str, float],
    columns: Mapping[str, Sequence[float]],
    *,
    taker_bps: float = TAKER_BPS,
    min_leg: float = 0.01,
    horizon_bars: int = 168,
    assumed_sharpe_annual: float = 1.0,
) -> TradePlan:
    """The trade from ``current`` toward hierarchical risk parity, priced against its benefit.

    ``min_leg`` drops legs smaller than 1% of the book: below that the fee is a larger share of the
    leg than any risk it moves, and a plan of twelve dust trades is how a rebalance quietly
    becomes a
    fee-generating machine. The dropped weight is not reallocated — the plan is a partial move
    toward
    the target and says so by its own arithmetic, rather than pretending to reach it.

    ``horizon_bars`` defaults to a week of hourly bars, matching the desk's own holding behaviour.
    ``assumed_sharpe_annual`` defaults to 1.0 and is **an assumption, not a measurement** — ARGUS
    has
    no live Sharpe (zero executed trades), so the break-even is stated conditionally. It appears in
    the verdict every time for that reason.
    """
    names = sorted(columns)
    if len(names) < MIN_ASSETS:
        raise AllocationError(f"{len(names)} instrument(s) cannot be allocated hierarchically")
    size = len(columns[names[0]])
    if size < MIN_OBSERVATIONS:
        raise AllocationError(
            f"{size} observation(s) is below the {MIN_OBSERVATIONS} needed for a covariance"
        )
    from argus.desk.portfolio import covariance_matrix

    built = covariance_matrix(columns)
    if built is None:
        raise AllocationError("the covariance matrix could not be built from these columns")
    matrix_names, cov = built

    target = hrp_weights(matrix_names, cov)
    trades = tuple(
        Trade(symbol=name, weight_before=current.get(name, 0.0), weight_after=target[name])
        for name in matrix_names
        if abs(target[name] - current.get(name, 0.0)) >= min_leg
    )
    applied = dict(current)
    for trade in trades:
        applied[trade.symbol] = trade.weight_after

    turnover = sum(abs(t.delta) for t in trades)
    vol_before = math.sqrt(max(0.0, portfolio_variance(current, matrix_names, cov)))
    vol_after = math.sqrt(max(0.0, portfolio_variance(applied, matrix_names, cov)))
    per_bar_sharpe = assumed_sharpe_annual / math.sqrt(TRADING_HOURS_PER_YEAR)
    return TradePlan(
        trades=trades, vol_before=vol_before, vol_after=vol_after, turnover=turnover,
        cost_bps=turnover * taker_bps, assumed_sharpe=per_bar_sharpe, horizon_bars=horizon_bars,
    )


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import json
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.desk.portfolio import returns
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range

    parser = argparse.ArgumentParser(description="what allocation would this book be better at?")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--book", default="", help="SYMBOL=weight,... (default: equal weight)")
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--sharpe", type=float, default=1.0)
    args = parser.parse_args()

    series: dict[str, dict[Any, float]] = {}
    for symbol in RTOKEN_SYMBOLS:
        try:
            bars = fetch_range(symbol, days=args.days, interval="1H",
                               candle_type=CandleType.MARKET)
        except Exception as exc:
            print(f"  {symbol}: no history ({type(exc).__name__})")
            continue
        series[symbol] = returns([(c.ts, float(c.close)) for c in bars])

    stamps = sorted(set.intersection(*(set(v) for v in series.values()))) if series else []
    columns = {name: [series[name][t] for t in stamps] for name in series}
    if len(columns) < MIN_ASSETS or len(stamps) < MIN_OBSERVATIONS:
        print("not enough aligned history to allocate")
        return 1

    if args.book:
        book = {}
        for part in args.book.split(","):
            name, _, weight = part.partition("=")
            book[name.strip().upper()] = float(weight)
    else:
        book = dict.fromkeys(columns, 1.0 / len(columns))

    plan = optimize_trade(
        book, columns, horizon_bars=args.horizon, assumed_sharpe_annual=args.sharpe,
    )
    names = sorted(columns)
    from argus.desk.portfolio import covariance_matrix

    built = covariance_matrix(columns)
    print(f"HIERARCHICAL RISK PARITY — {len(names)} instruments, {len(stamps)} hourly bars\n")
    for trade in plan.trades:
        print(f"  {trade.symbol:12} {trade.weight_before:7.2%} -> {trade.weight_after:7.2%}")
    if built is not None:
        matrix_names, cov = built
        before = diversification_ratio(book, matrix_names, cov)
        after = diversification_ratio(
            {**book, **{t.symbol: t.weight_after for t in plan.trades}}, matrix_names, cov,
        )
        if before is not None and after is not None:
            print(f"\n  diversification ratio {before:.3f} -> {after:.3f}")
    print(f"\n  {plan.verdict}")
    out = Path(__file__).resolve().parents[3] / "data" / "allocation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "MIN_ASSETS",
    "MIN_OBSERVATIONS",
    "TAKER_BPS",
    "AllocationError",
    "Merge",
    "Trade",
    "TradePlan",
    "correlation_distance",
    "diversification_ratio",
    "hrp_weights",
    "optimize_trade",
    "portfolio_variance",
    "quasi_diagonal",
    "single_linkage",
]
