"""The most likely way a position breaks a loss limit, searched over moves and books that happened.

`desk/stress.py` answers "how bad is a move of this size, and how often has it happened" and, in
reverse, "at what depth does this position stop being exitable". Each holds the other dimension
fixed. A position fails through their combination: a fall that is survivable into a normal book is
not survivable into the book of a thin hour, and a thin book is harmless if the price holds.

Adaptive stress testing (Lee et al., "Adaptive Stress Testing", JAIR 2020; `sisl/
AdaptiveStressTestingToolbox`, MIT, `ast_toolbox/rewards/example_av_reward.py:40-117`) frames
this as a search for the **most likely failure**: among the disturbances that end in failure, the
one the disturbance model gives the highest probability. Its solvers (MCTS, genetic search) exist
because a driving simulation's disturbance space is too large to enumerate. This one is not:

* the move is one of the position's own observed horizon moves (`stress.horizon_moves`), several
  thousand windows of real price history;
* the book is one of the order books actually recorded for that symbol (`data/book_tape.jsonl`).

Every pair is evaluated, so the most likely failure reported is the exact one, not the best a
search happened to reach — which is the study's own kill criterion (research/harvest/
16-adaptive-stress-testing.md: "if a cheap non-adaptive baseline finds equally bad scenarios at the
same budget, the adaptive solver isn't earning its complexity"): enumeration is cheaper than
either, and :func:`random_search` is kept so the comparison is measured rather than asserted.

**Probability, and the assumption under it.** A move's probability is its share of observed
windows at least as adverse; a book's is its share of recorded books at least as costly to exit
this size into. Their product treats the move and the book as independent. That is an assumption,
stated on every result; the price path and the book tape come from different samples and their
dependence has not been measured.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.desk.stress import Move, StressError
from argus.market.depth import DepthError, Level, OrderBook
from argus.truth.paths import DATA_DIR

TAPE_PATH = DATA_DIR / "book_tape.jsonl"
EXIT_FEE_BPS = Decimal("6")
"""Bitget's taker fee on the exit (`cost/model.CostModel.bitget_perp`)."""
INDEPENDENCE = ("the move and the book are treated as independent; their dependence has not been "
                "measured")


def tape_books(symbol: str, path: Path = TAPE_PATH) -> list[OrderBook]:
    """Every recorded book for ``symbol``; a row that is one-sided or crossed is skipped."""
    books: list[OrderBook] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return books
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("symbol") != symbol:
            continue
        try:
            books.append(OrderBook(
                symbol=symbol, fetched_at=datetime.fromisoformat(row["taken_at"]),
                bids=tuple(Level(Decimal(str(p)), Decimal(str(q))) for p, q in row["bids"]),
                asks=tuple(Level(Decimal(str(p)), Decimal(str(q))) for p, q in row["asks"])))
        except (DepthError, KeyError, ValueError, TypeError):
            continue
    return books


@dataclass(frozen=True, slots=True)
class ExitCost:
    """What leaving a position into one recorded book costs, for this size."""

    taken_at: datetime
    slippage_bps: Decimal
    complete: bool


def exit_costs(books: Sequence[OrderBook], quantity: Decimal, *, long: bool) -> list[ExitCost]:
    """Slippage to exit ``quantity`` units into each book (a long sells into the bids). Walked in
    units at the book's own prices, so the cost in basis points carries to a shocked price with
    the book's shape unchanged — the assumption every depth scaling here makes."""
    out: list[ExitCost] = []
    for book in books:
        sweep = book.sweep(quantity * book.mid, direction="SELL" if long else "BUY")
        out.append(ExitCost(book.fetched_at, sweep.slippage_bps, sweep.complete))
    return out


@dataclass(frozen=True, slots=True)
class Scenario:
    move_pct: Decimal
    move_at: datetime
    book_at: datetime
    slippage_bps: Decimal
    exitable: bool
    loss_pct: Decimal
    """Loss on the entry notional: the move against the position, the exit slippage and fee."""
    probability: Decimal
    """Share of observed windows at least this adverse, times share of books at least this
    costly — under the independence assumption."""

    def as_dict(self) -> dict[str, Any]:
        return {"move_pct": f"{self.move_pct:.4f}", "move_at": self.move_at.isoformat(),
                "book_at": self.book_at.isoformat(), "slippage_bps": f"{self.slippage_bps:.2f}",
                "exitable": self.exitable, "loss_pct": f"{self.loss_pct:.4f}",
                "probability": f"{self.probability:.6f}"}


@dataclass(frozen=True, slots=True)
class SearchResult:
    symbol: str
    quantity: Decimal
    long: bool
    tolerance_pct: Decimal
    moves: int
    books: int
    evaluated: int
    failure_probability: Decimal
    """Share of (move, book) pairs that fail: the probability of breaking the limit over the
    horizon, under independence."""
    most_likely: Scenario | None
    worst: Scenario | None
    assumption: str = INDEPENDENCE

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "quantity": str(self.quantity), "long": self.long,
                "tolerance_pct": str(self.tolerance_pct), "moves": self.moves,
                "books": self.books, "evaluated": self.evaluated,
                "failure_probability": f"{self.failure_probability:.6f}",
                "most_likely": None if self.most_likely is None else self.most_likely.as_dict(),
                "worst": None if self.worst is None else self.worst.as_dict(),
                "assumption": self.assumption}

    def sentence(self) -> str:
        side = "long" if self.long else "short"
        if self.most_likely is None:
            return (f"No combination of the {self.moves} observed moves and {self.books} recorded "
                    f"books takes this {side} past a {self.tolerance_pct}% loss.")
        m = self.most_likely
        how = ("the book cannot absorb the exit" if not m.exitable
               else f"a {m.loss_pct:.2f}% loss including {m.slippage_bps:.1f}bps of slippage")
        return (f"Chance of losing more than {self.tolerance_pct}% over the horizon: "
                f"{self.failure_probability:.1%}. Most likely way: a {m.move_pct:+.2f}% move "
                f"into a book like {m.book_at:%Y-%m-%d %H:%M} UTC — {how} ({INDEPENDENCE}).")


def _loss_pct(move_pct: Decimal, slippage_bps: Decimal, *, long: bool) -> Decimal:
    adverse = -move_pct if long else move_pct
    remaining = Decimal("1") + (move_pct if long else -move_pct) / 100
    return adverse + remaining * (slippage_bps + EXIT_FEE_BPS) / 100


def _grid(moves: Sequence[Move], costs: Sequence[ExitCost], *, long: bool,
          ) -> tuple[list[tuple[Move, Decimal]], list[tuple[ExitCost, Decimal]]]:
    """Each move and each book with its tail probability (share at least as adverse / costly)."""
    n, k = len(moves), len(costs)
    ranked_moves = sorted(moves, key=lambda m: m.pct if long else -m.pct)
    move_p = [(m, Decimal(i + 1) / n) for i, m in enumerate(ranked_moves)]
    ranked_books = sorted(costs, key=lambda c: (c.complete, -c.slippage_bps))
    book_p = [(c, Decimal(i + 1) / k) for i, c in enumerate(ranked_books)]
    return move_p, book_p


def _scenario(move: Move, p_move: Decimal, cost: ExitCost, p_book: Decimal, *,
              long: bool) -> Scenario:
    return Scenario(move_pct=move.pct, move_at=move.at, book_at=cost.taken_at,
                    slippage_bps=cost.slippage_bps, exitable=cost.complete,
                    loss_pct=_loss_pct(move.pct, cost.slippage_bps, long=long),
                    probability=p_move * p_book)


def search(symbol: str, moves: Sequence[Move], books: Sequence[OrderBook], *,
           quantity: Decimal, tolerance_pct: Decimal, long: bool = True) -> SearchResult:
    """Every (move, book) pair: the share that fails, the most likely failure, the worst one."""
    if not moves:
        raise StressError(f"no observed moves for {symbol}")
    if not books:
        raise StressError(f"no recorded order books for {symbol} in {TAPE_PATH.name}")
    if quantity <= 0 or tolerance_pct <= 0:
        raise StressError("quantity and the loss tolerance must be positive")
    move_p, book_p = _grid(moves, exit_costs(books, quantity, long=long), long=long)
    failing = 0
    most: Scenario | None = None
    worst: Scenario | None = None
    for move, pm in move_p:
        for cost, pb in book_p:
            loss = _loss_pct(move.pct, cost.slippage_bps, long=long)
            if cost.complete and loss < tolerance_pct:
                continue
            failing += 1
            if most is None or pm * pb > most.probability:
                most = _scenario(move, pm, cost, pb, long=long)
            if worst is None or (not cost.complete, loss) > (not worst.exitable, worst.loss_pct):
                worst = _scenario(move, pm, cost, pb, long=long)
    total = len(move_p) * len(book_p)
    return SearchResult(symbol=symbol, quantity=quantity, long=long, tolerance_pct=tolerance_pct,
                        moves=len(move_p), books=len(book_p), evaluated=total,
                        failure_probability=Decimal(failing) / total, most_likely=most,
                        worst=worst)


def random_search(symbol: str, moves: Sequence[Move], books: Sequence[OrderBook], *,
                  quantity: Decimal, tolerance_pct: Decimal, budget: int, seed: int,
                  long: bool = True) -> Scenario | None:
    """The non-adaptive baseline the study requires: ``budget`` pairs drawn uniformly, the most
    likely failure among them. Kept to measure what enumeration buys, not to be used."""
    move_p, book_p = _grid(moves, exit_costs(books, quantity, long=long), long=long)
    rng = random.Random(seed)
    best: Scenario | None = None
    for _ in range(budget):
        move, pm = move_p[rng.randrange(len(move_p))]
        cost, pb = book_p[rng.randrange(len(book_p))]
        loss = _loss_pct(move.pct, cost.slippage_bps, long=long)
        if cost.complete and loss < tolerance_pct:
            continue
        if best is None or pm * pb > best.probability:
            best = _scenario(move, pm, cost, pb, long=long)
    return best


__all__ = ["EXIT_FEE_BPS", "INDEPENDENCE", "ExitCost", "Scenario", "SearchResult",
           "exit_costs", "random_search", "search", "tape_books"]
