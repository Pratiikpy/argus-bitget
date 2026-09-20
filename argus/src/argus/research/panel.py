"""Panel evaluation — the universe at one instant, so a factor can say "compared to what".

Every evaluator above this one reads a single instrument's own history. That is the shape almost
all of our factors have, and it is also the shape that made forty-six of the hundred and one
Formulaic Alphas inexpressible: `rank(x)` in that vocabulary asks where a name sits **among all
names right now**, which no amount of looking at one name's past can answer. The port in
`research/architecture/alpha101-port.md` counted the cost — 46 blocked on cross-sectional rank
alone, roughly eight more on `scale()`.

**The alignment problem is the whole problem.** A cross-sectional comparison is only meaningful
between values observed at the same instant, and twelve rToken series do not arrive with the same
bars: a venue gap, a late candle or a symbol that listed later each shift one series against the
others. Ranking index 400 of NVDA against index 400 of COIN when their index 400 are different
hours is not a factor, it is a bug that produces plausible numbers. So :class:`Panel` is built from
timestamps, not from positions, and **only timestamps present in every symbol are kept**. The
count that was dropped is carried on the panel and printed, because a universe that silently shrank
to three names is a different experiment from the one that was requested.

**Evaluation is innermost-first.** ``crossrank(sub(crossrank(close), 1))`` is legal and has to
resolve bottom-up: the inner rank is computed for the whole universe, then the outer expression is
evaluated per symbol with the inner result already available. :func:`evaluate_panel` walks the tree
by depth so each cross-sectional node is computed only after everything beneath it.

**Point-in-time is preserved.** The universe used at index ``i`` is the universe's values at index
``i``, never a later one. Cross-sectional operators are the easiest place in a factor library to
introduce look-ahead — a rank computed over the full sample and then indexed back into is a
standard and invisible way to leak the future — so the ranking happens per index, over that index's
values only, and :func:`evaluate_panel` never sees a value from beyond the index it is filling.

Read against ``microsoft/qlib``, which computes the same thing by grouping a (datetime, instrument)
frame with ``groupby(level='datetime').rank(pct=True)``. The grouping key there is the datetime,
not the row number, for exactly the alignment reason above; ours is the aligned timestamp for the
same reason, without the frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from argus.research.grammar import (
    CrossContext,
    Expr,
    GrammarError,
    _CrossSectional,
)

MIN_UNIVERSE = 2
"""Fewest symbols before a cross-sectional expression means anything.

One name ranks 0.5 against itself at every instant — a constant, which is not a factor. Rather than
let that constant flow into a sweep and consume a trial, :func:`evaluate_panel` refuses. The refusal
is the point: a cross-sectional result computed on a one-symbol universe looks like a number and is
not one.
"""


class PanelError(ValueError):
    """The universe cannot support the evaluation that was asked for."""


def _timestamp(bar: Any) -> datetime:
    """The bar's instant.

    Read by name, never with ``getattr``. ``tests/test_grammar.py`` parses the grammar and fails on
    any dynamic attribute access, and the same rule is worth keeping here: the panel decides what a
    factor is allowed to see, so it must not be able to name a field dynamically either.
    """
    try:
        return bar.ts  # type: ignore[no-any-return]
    except AttributeError as exc:
        raise PanelError(
            f"a bar of type {type(bar).__name__} has no `ts`; the panel aligns on timestamps and "
            f"cannot align on position"
        ) from exc


@dataclass(frozen=True)
class Panel:
    """Several instruments' bars, aligned to the timestamps they all share.

    ``bars`` is symbol -> the kept bars, every series the same length, index ``i`` meaning the same
    instant in all of them. That invariant is what makes a cross-sectional comparison legitimate,
    and it is established once here rather than assumed at every call site.
    """

    bars: Mapping[str, tuple[Any, ...]]
    timestamps: tuple[datetime, ...]
    dropped: Mapping[str, int]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.bars))

    @property
    def width(self) -> int:
        return len(self.bars)

    @property
    def length(self) -> int:
        return len(self.timestamps)

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())

    def render(self) -> str:
        lines = [
            f"PANEL — {self.width} symbol(s) aligned on {self.length} shared timestamp(s)",
        ]
        if self.total_dropped:
            worst = sorted(self.dropped.items(), key=lambda kv: -kv[1])[:5]
            lines.append(
                f"  {self.total_dropped} bar(s) dropped as unshared: "
                + ", ".join(f"{s} -{n}" for s, n in worst if n)
            )
            lines.append(
                "  A bar present in one symbol and not another cannot take part in a "
                "cross-sectional comparison, so it is removed rather than compared against a "
                "neighbouring hour."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "length": self.length,
            "dropped": dict(self.dropped),
            "first": self.timestamps[0].isoformat() if self.timestamps else None,
            "last": self.timestamps[-1].isoformat() if self.timestamps else None,
        }


def build_panel(series: Mapping[str, Sequence[Any]]) -> Panel:
    """Align several bar series on the timestamps every one of them has.

    An intersection rather than a union, and rather than a forward fill. Filling a missing bar with
    the previous one would put a stale price into a comparison of simultaneous prices, which is the
    same error as ranking mismatched indices and is harder to see afterwards.

    Duplicate timestamps within one symbol keep the **last** bar for that instant, matching the way
    a venue restates a candle: the later message is the corrected one.
    """
    if not series:
        raise PanelError("a panel needs at least one symbol")
    by_symbol: dict[str, dict[datetime, Any]] = {}
    for symbol, bars in series.items():
        indexed: dict[datetime, Any] = {}
        for bar in bars:
            indexed[_timestamp(bar)] = bar
        if not indexed:
            raise PanelError(f"{symbol} carries no bars, so it cannot be aligned")
        by_symbol[symbol] = indexed

    shared: set[datetime] | None = None
    for indexed in by_symbol.values():
        shared = set(indexed) if shared is None else (shared & set(indexed))
    stamps = tuple(sorted(shared or ()))
    if not stamps:
        raise PanelError(
            "no timestamp is present in every symbol; these series do not overlap and cannot be "
            "compared instant by instant"
        )
    return Panel(
        bars={s: tuple(idx[t] for t in stamps) for s, idx in by_symbol.items()},
        timestamps=stamps,
        dropped={s: len(idx) - len(stamps) for s, idx in by_symbol.items()},
    )


def cross_sectional_nodes(expr: Expr) -> tuple[Expr, ...]:
    """Every cross-sectional node in the tree, innermost first.

    Ordered by depth ascending so a node is always returned after everything it contains, which is
    the order :func:`evaluate_panel` must compute them in. Duplicates by canonical form are
    collapsed: the same sub-expression appearing twice is one computation and, just as importantly,
    one trial.
    """
    found: dict[str, Expr] = {}

    def walk(node: Expr) -> None:
        for child in node.children:
            walk(child)
        if node.is_cross_sectional:
            found.setdefault(node.canonical(), node)

    walk(expr)
    return tuple(sorted(found.values(), key=lambda n: n.depth))


def evaluate_panel(
    expr: Expr, panel: Panel, *, minimum_universe: int = MIN_UNIVERSE
) -> dict[str, tuple[float, ...]]:
    """Evaluate one expression across the whole universe. Returns symbol -> values by index.

    Works for expressions with no cross-sectional node too, in which case it is simply a loop —
    that is deliberate, so a sweep can send every candidate down one path and never has to branch
    on whether a factor happens to be cross-sectional.
    """
    nodes = cross_sectional_nodes(expr)
    if nodes and panel.width < minimum_universe:
        raise PanelError(
            f"{expr.canonical()} is cross-sectional and the universe has {panel.width} symbol(s); "
            f"at least {minimum_universe} are needed. A rank against oneself is the constant 0.5, "
            f"not a factor, and running it would spend a trial on a number that cannot vary."
        )

    symbols = panel.symbols
    # Per symbol, the values of every cross-sectional node resolved so far. Grown innermost-first
    # so an outer node's operand can itself be cross-sectional.
    resolved: dict[str, dict[str, list[float]]] = {s: {} for s in symbols}

    for node in nodes:
        if not isinstance(node, _CrossSectional):  # pragma: no cover - defensive
            raise GrammarError(
                f"{node.canonical()} reports itself cross-sectional but does not implement the "
                f"cross-sectional interface"
            )
        # The operand, per symbol, over every index. Computed with the context built so far, which
        # already holds anything nested inside it.
        raw: dict[str, list[float]] = {}
        for symbol in symbols:
            ctx = CrossContext(symbol=symbol, values=resolved[symbol])
            bars = panel.bars[symbol]
            raw[symbol] = [node.operand.evaluate(bars, i, ctx) for i in range(panel.length)]

        # Combine across the universe, one index at a time. Index i sees only index i — the single
        # place look-ahead could enter, and the loop is written so it cannot.
        combined: dict[str, list[float]] = {s: [] for s in symbols}
        canonical = node.canonical()
        for i in range(panel.length):
            at_i = [raw[s][i] for s in symbols]
            for symbol, value in zip(symbols, node.combine(at_i), strict=True):
                combined[symbol].append(value)
        for symbol in symbols:
            resolved[symbol][canonical] = combined[symbol]

    out: dict[str, tuple[float, ...]] = {}
    for symbol in symbols:
        ctx = CrossContext(symbol=symbol, values=resolved[symbol])
        bars = panel.bars[symbol]
        out[symbol] = tuple(expr.evaluate(bars, i, ctx) for i in range(panel.length))
    return out


__all__ = [
    "MIN_UNIVERSE",
    "Panel",
    "PanelError",
    "build_panel",
    "cross_sectional_nodes",
    "evaluate_panel",
]
