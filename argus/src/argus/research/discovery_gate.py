"""The trial-corrected gate for a `searchoff` pool: which searched factors beat flat, trials paid.

`research/searchoff.py` scores every candidate it tries and keeps two numbers per candidate — the
in-sample and out-of-sample per-bar Sharpe. That is enough to rank candidates and not enough to
correct for having tried them: a data-snooping test needs each candidate's *return series*,
because the correction depends on how the candidates co-move, and 400 grammar trees built from the
same six fields co-move a great deal. This module recovers those series without changing how the
search runs:

* :class:`RecordingArena` is `searchoff.Arena` with one addition — it remembers the expression
  behind every canonical form it scores. The budget, the canonical-form de-duplication, the cost
  refusal and the in/out-of-sample split are all the parent's, untouched.
* :func:`net_returns` recomputes one candidate's per-bar net returns with the same arithmetic as
  `searchoff._score_half`, and :func:`pool_returns` **refuses** to return a series whose Sharpe
  differs from the one the arena recorded, by even one bit. A mirror that drifted from the scorer
  would gate something other than what was searched, and this check is what makes that loud.
* :func:`gate` runs Romano-Wolf's StepM (`backtest/snooping.py`) over the in-sample series against
  a flat benchmark, and reports the deflated Sharpe beside it — not as the decision, but so the
  two can be read side by side: `eval/general_factorsafety_comparison.py` measured the deflated
  Sharpe gate as nearly powerless on exactly these pools.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from argus.backtest.metrics import MetricError, deflated_sharpe
from argus.backtest.snooping import DEFAULT_SIZE, SnoopingBootstrap, StepMResult
from argus.research.grammar import Expr
from argus.research.searchoff import Arena, Candidate


class GateError(ValueError):
    """The pool cannot be gated honestly — empty, or its series do not match what was scored."""


class RecordingArena(Arena):
    """`searchoff.Arena`, plus the expression behind every canonical form it scored."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.expressions: dict[str, Expr] = {}

    def evaluate(self, expr: Expr) -> Candidate | None:
        found = super().evaluate(expr)
        if found is not None and found.canonical not in self.expressions:
            self.expressions[found.canonical] = expr
        return found


def net_returns(
    expr: Expr, bars: Sequence[Any], start: int, stop: int, cost_bps: float
) -> list[float]:
    """Per-bar net returns in basis points, by `searchoff._score_half`'s arithmetic, step for step.

    Sign-of-signal position, next-bar close-to-close move, half the round-trip cost charged on
    each unit of position change. Kept operation-for-operation identical so the Sharpe of this
    series equals the arena's recorded score exactly — which :func:`pool_returns` checks.
    """
    returns: list[float] = []
    previous = 0.0
    for i in range(start, stop - 1):
        raw = expr.evaluate(bars, i)
        position = 1.0 if raw > 0 else (-1.0 if raw < 0 else 0.0)
        move = 0.0
        entry, nxt = float(bars[i].close), float(bars[i + 1].close)
        if entry:
            move = (nxt / entry - 1.0) * 10_000
        turnover = abs(position - previous)
        returns.append(position * move - turnover * cost_bps / 2.0)
        previous = position
    return returns


def _sharpe(series: Sequence[float]) -> float:
    mean = sum(series) / len(series)
    variance = sum((r - mean) ** 2 for r in series) / (len(series) - 1)
    return mean / math.sqrt(variance)


@dataclass(frozen=True, slots=True)
class PoolReturns:
    """Every candidate the arena scored, with its in- and out-of-sample net return series."""

    canonicals: tuple[str, ...]
    in_sample: tuple[tuple[float, ...], ...]
    out_of_sample: tuple[tuple[float, ...], ...]
    in_sample_sharpes: tuple[float, ...]
    out_of_sample_sharpes: tuple[float, ...]
    trials_spent: int


def pool_returns(arena: RecordingArena) -> PoolReturns:
    """Recover each scored candidate's series and prove they are the ones the arena scored."""
    if not arena.seen:
        raise GateError("the arena scored no candidate; there is nothing to gate")
    cut = arena.split
    canonicals: list[str] = []
    ins: list[tuple[float, ...]] = []
    outs: list[tuple[float, ...]] = []
    for key, candidate in arena.seen.items():
        expr = arena.expressions.get(key)
        if expr is None:
            raise GateError(f"no expression was recorded for {key[:80]!r}")
        inside = net_returns(expr, arena.bars, 0, cut, arena.cost_bps)
        outside = net_returns(expr, arena.bars, cut, len(arena.bars), arena.cost_bps)
        if _sharpe(inside) != candidate.in_sample or _sharpe(outside) != candidate.out_of_sample:
            raise GateError(
                f"recomputed series for {key[:80]!r} does not reproduce the arena's own score; "
                f"net_returns has drifted from searchoff._score_half"
            )
        canonicals.append(key)
        ins.append(tuple(inside))
        outs.append(tuple(outside))
    return PoolReturns(
        canonicals=tuple(canonicals), in_sample=tuple(ins), out_of_sample=tuple(outs),
        in_sample_sharpes=tuple(c.in_sample for c in arena.seen.values()),
        out_of_sample_sharpes=tuple(c.out_of_sample for c in arena.seen.values()),
        trials_spent=arena.spent,
    )


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """StepM's answer for the pool, with the deflated Sharpe beside it for reference."""

    stepm: StepMResult
    survivors: tuple[str, ...]
    survivors_out_of_sample: tuple[float, ...]
    deflated_sharpe_of_best: float | None
    deflated_sharpe_error: str | None
    trials_spent: int
    candidates: int

    @property
    def verdict(self) -> str:
        if not self.survivors:
            return (
                f"No searched factor beats flat once all {self.trials_spent} trials are paid for "
                f"(StepM, family-wise error {self.stepm.size:.0%}, consistent SPA "
                f"p={self.stepm.spa.p_consistent:.4f})."
            )
        paired = ", ".join(
            f"{c[:60]} (out-of-sample {o:+.4f})"
            for c, o in zip(self.survivors, self.survivors_out_of_sample, strict=True)
        )
        return (
            f"{len(self.survivors)} searched factor(s) beat flat in sample with the family-wise "
            f"error held at {self.stepm.size:.0%} across {self.trials_spent} trials: {paired}. "
            f"In-sample survival is a candidacy for the out-of-sample check, not a pass of it."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stepm": self.stepm.as_dict(),
            "survivors": list(self.survivors),
            "survivors_out_of_sample": [round(v, 6) for v in self.survivors_out_of_sample],
            "deflated_sharpe_of_best": self.deflated_sharpe_of_best,
            "deflated_sharpe_error": self.deflated_sharpe_error,
            "trials_spent": self.trials_spent,
            "candidates": self.candidates,
            "verdict": self.verdict,
        }


def gate(
    pool: PoolReturns,
    *,
    size: float = DEFAULT_SIZE,
    resamples: int = 1000,
    block_length: float | None = None,
    seed: int = 20260925,
) -> GateVerdict:
    """StepM over the pool's in-sample series against flat (a zero-return benchmark)."""
    n = len(pool.in_sample[0])
    boot = SnoopingBootstrap(
        pool.in_sample, [0.0] * n, resamples=resamples, block_length=block_length, seed=seed,
    )
    result = boot.stepm(size=size, studentize=True)
    survivors = tuple(pool.canonicals[k] for k in result.superior)
    oos = tuple(pool.out_of_sample_sharpes[k] for k in result.superior)

    sharpes = pool.in_sample_sharpes
    mean = sum(sharpes) / len(sharpes)
    variance = sum((s - mean) ** 2 for s in sharpes) / len(sharpes)
    try:
        dsr: float | None = deflated_sharpe(
            max(sharpes), n=max(n, 2), trials=pool.trials_spent, variance_of_trials=variance,
        )
        error = None
    except MetricError as exc:
        dsr, error = None, str(exc)
    return GateVerdict(
        stepm=result, survivors=survivors, survivors_out_of_sample=oos,
        deflated_sharpe_of_best=dsr, deflated_sharpe_error=error,
        trials_spent=pool.trials_spent, candidates=len(pool.canonicals),
    )


__all__ = [
    "GateError",
    "GateVerdict",
    "PoolReturns",
    "RecordingArena",
    "gate",
    "net_returns",
    "pool_returns",
]
