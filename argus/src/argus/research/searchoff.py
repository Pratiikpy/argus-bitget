"""Which search method actually finds robust alpha? Same grammar, same data, same budget.

The Track 1 result this project publishes is that **0 of 12 symbols survive the deflated Sharpe
over all trials**, and `research/overfitting_study.py` adds that the selection procedure itself is
unstable — TQQQ's headline winner is the in-sample best in only 30% of balanced splits. Both are
statements about *what the search found*. Neither says anything about *how it searched*, because
until now there was one search: enumerate a fixed variant list.

A competing entry's claim to have "discovered" a factor is a claim about a search method, and there
is no way to read it without knowing what a different method would have found on the same data
under the same budget. This module makes that comparison, and it makes it against ourselves first.

**The budget is the experiment.** Every strategy gets exactly the same number of evaluations, and
that is the only way the comparison means anything: a method that tries ten times as many
candidates will find a better maximum of the noise distribution, and reporting that as a better
method is the error the whole study exists to avoid. Trials are counted per *canonical* expression,
so two spellings of one factor consume one trial — the grammar already provides that identity and
a search that rediscovers the same tree is not exploring.

**Five strategies, chosen so at least one should embarrass the others:**

* ``random`` — uniform sampling from the grammar. The null. A method that cannot beat this is not
  a search method, and on a space this noisy it is a much harder baseline than it sounds.
* ``beam`` — keep the best ``k`` and expand only those. Exploitation with no exploration.
* ``evolutionary`` — mutate the survivors. Exploitation with local exploration.
* ``novelty`` — select for canonical forms unlike anything tried, ignoring score entirely. Pure
  exploration, and it is in the field precisely because it should lose on in-sample score and may
  not lose on out-of-sample.
* ``anneal`` — accept worse candidates with a decaying probability. The middle of that spectrum.

**Every candidate is scored out of sample, and the winner is chosen in sample.** That separation is
the point: picking on the training half and reporting the test half is what a real deployment does,
and a study that picks on the full sample measures nothing. The gap between the two is reported per
strategy, because a method that finds a high in-sample score and loses it out of sample is worse
than one that found less and kept it.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from argus.research.grammar import (
    BinOp,
    Const,
    Delay,
    Expr,
    Field,
    GrammarError,
    Ref,
    Return,
    UnOp,
    Window,
)

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "search_bakeoff.json"
SWEEP_PATH = DATA / "search_sweep.json"

BUDGET = 400
"""Evaluations each strategy is allowed. Identical across strategies, which is the experiment."""

MAX_DEPTH = 4

MAX_NODE_COST = 5_000
"""Nested-window operations per bar above which a candidate is refused as untradeable.

**Depth alone does not bound the cost, and the first run of this module hung because of it.**
Window and Delay nodes nest multiplicatively: ``delay(24, min(6, slope(48, std(48, volume))))``
is depth 5 and costs 24 x 6 x 48 x 48 = **331,776 operations per bar**, which over 1,500 bars is
half a billion operations for one candidate. Measured across 3,000 random trees the median cost is
1 and the worst is that one, so the pathology is rare and fatal — exactly the shape that survives
a quick test and kills a long run.

The limit is not only about wall-clock. A rule needing five thousand operations per hourly bar is
not a rule anyone would trade, so refusing it is a statement about the search space rather than a
concession to the machine. A refused candidate still **costs a trial**, because a strategy that
generates unevaluable trees must not explore for free.
"""
LOOKBACKS = (1, 2, 3, 6, 12, 24, 48)
OOS_FRACTION = 0.35
MIN_BARS = 400

NUMERIC_FIELDS = (
    Field.CLOSE, Field.RETURN_1, Field.HOURS_TO_DISCOVERY,
    Field.BASIS_BPS, Field.VOLUME, Field.RANGE_BPS,
)


class SearchError(ValueError):
    """Raised rather than reporting a bake-off that did not run fairly."""


# --- generating candidates --------------------------------------------------------------------


def random_expr(rng: random.Random, *, depth: int = 0) -> Expr:
    """One expression from the grammar, uniformly over node types at each level.

    Returns a **signal**-kind tree: the search space is trading rules, not arbitrary arithmetic.
    Depth is capped because an unbounded generator produces trees that are expensive to evaluate
    and impossible to read, and an alpha nobody can read is one nobody can refuse.
    """
    if depth >= MAX_DEPTH:
        return Ref(rng.choice(NUMERIC_FIELDS))
    choice = rng.random()
    if choice < 0.22:
        return Ref(rng.choice(NUMERIC_FIELDS))
    if choice < 0.32:
        return Const(round(rng.uniform(-2.0, 2.0), 3))
    if choice < 0.42:
        return Return(rng.choice(LOOKBACKS))
    if choice < 0.60:
        return Window(
            rng.choice(Window.OPS), rng.choice(LOOKBACKS), random_expr(rng, depth=depth + 1)
        )
    if choice < 0.70:
        return Delay(rng.choice(LOOKBACKS), random_expr(rng, depth=depth + 1))
    if choice < 0.80:
        return UnOp(rng.choice(("neg", "abs", "sign")), random_expr(rng, depth=depth + 1))
    return BinOp(
        rng.choice(("add", "sub", "mul", "div")),
        random_expr(rng, depth=depth + 1),
        random_expr(rng, depth=depth + 1),
    )


def node_cost(expr: Expr) -> int:
    """Nested-window operations per bar. Multiplicative, because the nodes nest.

    ``Window`` and ``Delay`` each re-read ``lookback`` bars, and a window inside a window multiplies
    rather than adds. Anything else costs one. This is the estimate `MAX_NODE_COST` is applied to.
    """
    own = expr.lookback if isinstance(expr, Window | Delay) else 1
    children = [node_cost(c) for c in expr.children]
    return own * (max(children) if children else 1)


def mutate(expr: Expr, rng: random.Random) -> Expr:
    """Replace one randomly chosen subtree with a fresh one.

    Subtree replacement rather than point mutation: changing a constant explores a neighbourhood
    the score is usually flat over, while replacing a branch is the move that can actually find a
    different rule.
    """
    if not expr.children or rng.random() < 0.3:
        return random_expr(rng, depth=1)
    if isinstance(expr, Window):
        return Window(expr.op, expr.lookback, mutate(expr.operand, rng))
    if isinstance(expr, Delay):
        return Delay(expr.lookback, mutate(expr.operand, rng))
    if isinstance(expr, UnOp):
        return UnOp(expr.op, mutate(expr.operand, rng))
    if isinstance(expr, BinOp):
        if rng.random() < 0.5:
            return BinOp(expr.op, mutate(expr.left, rng), expr.right)
        return BinOp(expr.op, expr.left, mutate(expr.right, rng))
    return random_expr(rng, depth=1)


# --- scoring ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One expression and what it scored on each half of the data."""

    canonical: str
    in_sample: float
    out_of_sample: float

    @property
    def decay(self) -> float:
        """In-sample score minus out-of-sample. Positive means it did not hold up."""
        return self.in_sample - self.out_of_sample

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical[:200],
            "in_sample": round(self.in_sample, 5),
            "out_of_sample": round(self.out_of_sample, 5),
            "decay": round(self.decay, 5),
        }


def _score_half(expr: Expr, bars: Sequence[Any], start: int, stop: int, cost_bps: float) -> float:
    """Per-observation Sharpe of a sign-of-signal rule, net of cost on every position change.

    Charged on the change rather than every bar: a rule that holds does not pay again, and charging
    it would make every slow rule look worse than every fast one for no reason in the tape.
    """
    returns: list[float] = []
    previous = 0.0
    for i in range(start, stop - 1):
        try:
            raw = expr.evaluate(bars, i)
        except (GrammarError, ZeroDivisionError, ValueError, OverflowError):
            return float("nan")
        if not math.isfinite(raw):
            return float("nan")
        position = 1.0 if raw > 0 else (-1.0 if raw < 0 else 0.0)
        move = 0.0
        entry, nxt = float(bars[i].close), float(bars[i + 1].close)
        if entry:
            move = (nxt / entry - 1.0) * 10_000
        turnover = abs(position - previous)
        returns.append(position * move - turnover * cost_bps / 2.0)
        previous = position
    if len(returns) < 20:
        return float("nan")
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if variance <= 0:
        return float("nan")
    return mean / math.sqrt(variance)


@dataclass
class Arena:
    """The shared, budgeted evaluator. Every strategy gets the same one and the same limit."""

    bars: Sequence[Any]
    cost_bps: float = 12.0
    budget: int = BUDGET
    seen: dict[str, Candidate] = field(default_factory=dict)
    spent: int = 0
    refused: int = 0
    """Candidates refused as too expensive to evaluate. Reported, never hidden: a strategy that
    spends its budget generating untradeable trees has a real defect and this is where it shows."""

    @property
    def split(self) -> int:
        return int(len(self.bars) * (1 - OOS_FRACTION))

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.budget

    def is_exhausted(self) -> bool:
        """The same question as :attr:`exhausted`, callable inside a loop that already tested it.

        A property read is narrowed by the type checker: inside ``while not arena.exhausted`` it
        believes the value stays False and marks a later ``break`` unreachable. The budget does
        change, so the check is real and the narrowing is wrong. A method call is not narrowed.
        """
        return self.spent >= self.budget

    def evaluate(self, expr: Expr) -> Candidate | None:
        """Score one candidate, or return None if the budget is gone or it scored nothing.

        A canonical form already evaluated costs **no** trial and returns its stored result: the
        grammar's canonical spelling is an identity, and charging twice for one factor would let a
        strategy that rediscovers the same tree look busier than one that explores.
        """
        key = expr.canonical()
        if key in self.seen:
            return self.seen[key]
        if self.exhausted:
            return None
        self.spent += 1
        if node_cost(expr) > MAX_NODE_COST:
            # Charged a trial and refused. See MAX_NODE_COST: this is a property of the candidate,
            # not of the hardware, and letting it through is how a search hangs on one draw.
            self.refused += 1
            return None
        cut = self.split
        in_sample = _score_half(expr, self.bars, 0, cut, self.cost_bps)
        if not math.isfinite(in_sample):
            return None
        out_sample = _score_half(expr, self.bars, cut, len(self.bars), self.cost_bps)
        if not math.isfinite(out_sample):
            return None
        found = Candidate(canonical=key, in_sample=in_sample, out_of_sample=out_sample)
        self.seen[key] = found
        return found


Strategy = Callable[[Arena, random.Random], list[Candidate]]


def search_random(arena: Arena, rng: random.Random) -> list[Candidate]:
    found: list[Candidate] = []
    while not arena.exhausted:
        got = arena.evaluate(random_expr(rng))
        if got is not None:
            found.append(got)
    return found


def search_beam(arena: Arena, rng: random.Random, *, width: int = 8) -> list[Candidate]:
    """Keep the best ``width`` by in-sample score and expand only those. No exploration."""
    population: list[tuple[float, Expr]] = []
    found: list[Candidate] = []
    for _ in range(width * 2):
        if arena.exhausted:
            break
        expr = random_expr(rng)
        got = arena.evaluate(expr)
        if got is not None:
            found.append(got)
            population.append((got.in_sample, expr))
    while not arena.exhausted:
        population.sort(key=lambda p: -p[0])
        population = population[:width]
        if not population:
            break
        for _, parent in list(population):
            if arena.is_exhausted():
                break
            child = mutate(parent, rng)
            got = arena.evaluate(child)
            if got is not None:
                found.append(got)
                population.append((got.in_sample, child))
    return found


def search_evolutionary(
    arena: Arena, rng: random.Random, *, size: int = 20, keep: int = 6
) -> list[Candidate]:
    """Mutate the survivors. Exploitation with local exploration."""
    population: list[tuple[float, Expr]] = []
    found: list[Candidate] = []
    for _ in range(size):
        if arena.exhausted:
            break
        expr = random_expr(rng)
        got = arena.evaluate(expr)
        if got is not None:
            found.append(got)
            population.append((got.in_sample, expr))
    while not arena.exhausted and population:
        population.sort(key=lambda p: -p[0])
        parents = population[:keep]
        population = list(parents)
        for _ in range(size - keep):
            if arena.is_exhausted():
                break
            _, parent = rng.choice(parents)
            child = mutate(parent, rng)
            got = arena.evaluate(child)
            if got is not None:
                found.append(got)
                population.append((got.in_sample, child))
    return found


def search_novelty(arena: Arena, rng: random.Random, *, pool: int = 12) -> list[Candidate]:
    """Select for canonical forms unlike anything tried, ignoring score entirely.

    In the field because it *should* lose on in-sample score, and may not lose out of sample. A
    search that never looks at the objective cannot overfit to it, which is either a weakness or
    the whole point depending on how noisy the space is.
    """
    archive: list[str] = []
    found: list[Candidate] = []

    def distance(key: str) -> float:
        if not archive:
            return 1.0
        return min(
            1.0 - len(set(key) & set(other)) / max(1, len(set(key) | set(other)))
            for other in archive[-40:]
        )

    while not arena.exhausted:
        batch = [random_expr(rng) for _ in range(pool)]
        pick = max(batch, key=lambda e: distance(e.canonical()))
        got = arena.evaluate(pick)
        archive.append(pick.canonical())
        if got is not None:
            found.append(got)
    return found


def search_anneal(arena: Arena, rng: random.Random) -> list[Candidate]:
    """Accept a worse candidate with a decaying probability. The middle of the spectrum."""
    found: list[Candidate] = []
    current = random_expr(rng)
    scored = arena.evaluate(current)
    while scored is None and not arena.exhausted:
        current = random_expr(rng)
        scored = arena.evaluate(current)
    if scored is None:
        return found
    found.append(scored)
    best = scored.in_sample
    while not arena.exhausted:
        temperature = max(0.01, 1.0 - arena.spent / max(1, arena.budget))
        candidate = mutate(current, rng)
        got = arena.evaluate(candidate)
        if got is None:
            continue
        found.append(got)
        if got.in_sample > best or rng.random() < math.exp(
            (got.in_sample - best) / max(1e-9, temperature)
        ):
            current, best = candidate, got.in_sample
    return found


STRATEGIES: dict[str, Strategy] = {
    "random": search_random,
    "beam": search_beam,
    "evolutionary": search_evolutionary,
    "novelty": search_novelty,
    "anneal": search_anneal,
}


# --- the comparison ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """One strategy's run, picked in sample and reported out of sample."""

    name: str
    evaluated: int
    distinct: int
    refused: int
    picked: Candidate | None

    @property
    def in_sample(self) -> float | None:
        return None if self.picked is None else self.picked.in_sample

    @property
    def out_of_sample(self) -> float | None:
        return None if self.picked is None else self.picked.out_of_sample

    @property
    def decay(self) -> float | None:
        return None if self.picked is None else self.picked.decay

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evaluated": self.evaluated,
            "distinct": self.distinct,
            "refused_as_too_costly": self.refused,
            "picked": None if self.picked is None else self.picked.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class BakeOff:
    """Every strategy on one dataset under one budget."""

    symbol: str
    bars: int
    budget: int
    results: tuple[StrategyResult, ...]

    @property
    def by_out_of_sample(self) -> tuple[StrategyResult, ...]:
        scored = [r for r in self.results if r.out_of_sample is not None]
        return tuple(sorted(scored, key=lambda r: -(r.out_of_sample or 0.0)))

    @property
    def budget_was_equal(self) -> bool:
        """The experiment's own precondition, checked rather than assumed."""
        return len({r.evaluated for r in self.results}) == 1

    @property
    def verdict(self) -> str:
        ranked = self.by_out_of_sample
        if not ranked:
            return "no strategy produced a scorable candidate; the bake-off is undefined"
        best = ranked[0]
        baseline = next((r for r in self.results if r.name == "random"), None)
        head = (
            f"{best.name} picked the best out-of-sample candidate at "
            f"{best.out_of_sample:.4f} per-observation Sharpe on {self.symbol}, "
            f"from {best.evaluated} evaluations."
        )
        if baseline is None or baseline.out_of_sample is None:
            return head
        if best.name == "random":
            return (
                f"{head} **No search strategy beat uniform random sampling on the same budget.** "
                f"On a space this noisy that is the expected result and the one worth publishing: "
                f"a method that cannot beat random is not a search method, and three of these are "
                f"in the literature."
            )
        margin = (best.out_of_sample or 0.0) - baseline.out_of_sample
        return (
            f"{head} It beats uniform random by {margin:+.4f}. One dataset and one seed, so this "
            f"ranks the strategies on this run and does not establish a general ordering."
        )

    def render(self) -> str:
        lines = [
            f"SEARCH BAKE-OFF — {self.symbol}, {self.bars:,} bars, {self.budget} evaluations each",
            "",
            f"{'strategy':>14}{'evaluated':>11}{'distinct':>10}{'refused':>9}"
            f"{'in-sample':>12}{'out-of-sample':>15}{'decay':>9}",
        ]
        for result in sorted(self.results, key=lambda r: -(r.out_of_sample or -99)):
            ins = "n/a" if result.in_sample is None else f"{result.in_sample:.4f}"
            oos = "n/a" if result.out_of_sample is None else f"{result.out_of_sample:.4f}"
            dec = "n/a" if result.decay is None else f"{result.decay:+.4f}"
            lines.append(
                f"{result.name:>14}{result.evaluated:>11}{result.distinct:>10}"
                f"{result.refused:>9}{ins:>12}{oos:>15}{dec:>9}"
            )
        lines += [
            "",
            f"  equal budget honoured: {self.budget_was_equal}",
            "",
            f"  {self.verdict}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "symbol": self.symbol,
            "bars": self.bars,
            "budget": self.budget,
            "budget_was_equal": self.budget_was_equal,
            "results": [r.as_dict() for r in self.results],
            "verdict": self.verdict,
        }


def run(
    bars: Sequence[Any],
    *,
    symbol: str,
    budget: int = BUDGET,
    seed: int = 20260913,
    cost_bps: float = 12.0,
    strategies: dict[str, Strategy] | None = None,
) -> BakeOff:
    """Run every strategy on the same bars, with the same budget and the same starting seed.

    Each strategy gets its **own** generator seeded identically, so differences come from the
    search and not from the draw. A shared generator would make the comparison depend on the order
    the strategies happened to run in.
    """
    if len(bars) < MIN_BARS:
        raise SearchError(
            f"{len(bars)} bar(s) is below the {MIN_BARS} this needs; the out-of-sample half would "
            f"be too short to score a candidate"
        )
    chosen = STRATEGIES if strategies is None else strategies
    results: list[StrategyResult] = []
    for name, strategy in chosen.items():
        arena = Arena(bars=bars, cost_bps=cost_bps, budget=budget)
        found = strategy(arena, random.Random(seed))
        # Picked on the in-sample half, reported on the out-of-sample half. Picking on the full
        # sample is the error that makes every search method look like it works.
        picked = max(found, key=lambda c: c.in_sample) if found else None
        results.append(StrategyResult(
            name=name, evaluated=arena.spent, distinct=len(arena.seen),
            refused=arena.refused, picked=picked,
        ))
    return BakeOff(symbol=symbol, bars=len(bars), budget=budget, results=tuple(results))


SWEEP_SEEDS: tuple[int, ...] = (20260913, 20260914, 20260915)
"""Seeds for the multi-seed sweep. Three, fixed, and published so the run is reproducible.

**One seed ranks nothing, and this was learned the hard way.** The first published bake-off table
put `anneal` first on decay at +0.0123 and `beam` last at +0.0619. A rerun on a different seed put
`anneal` first again but moved `evolutionary` to last at +0.0439 with `beam` third. Across three
seeds `anneal` is **last**. Every one of those runs spent an identical, verified budget; the
ordering still moved, because a single run reports the best maximum of one draw of noise.
"""


@dataclass(frozen=True, slots=True)
class StrategySummary:
    """One strategy across every seed, with the per-seed spread kept rather than averaged away."""

    name: str
    decays: tuple[float | None, ...]
    in_samples: tuple[float | None, ...]
    out_of_samples: tuple[float | None, ...]
    ranks: tuple[int, ...]
    """Rank by decay within each seed, 1 = least decay. A strategy that found nothing ranks last."""

    @staticmethod
    def _mean(values: tuple[float | None, ...]) -> float | None:
        got = [v for v in values if v is not None]
        return fmean(got) if got else None

    @property
    def mean_decay(self) -> float | None:
        return self._mean(self.decays)

    @property
    def mean_in_sample(self) -> float | None:
        return self._mean(self.in_samples)

    @property
    def mean_out_of_sample(self) -> float | None:
        return self._mean(self.out_of_samples)

    @property
    def rank_spread(self) -> int:
        """Best rank to worst across the seeds. Zero means the seed did not change its place."""
        return max(self.ranks) - min(self.ranks) if self.ranks else 0

    def as_dict(self) -> dict[str, Any]:
        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 6)

        return {
            "name": self.name,
            "mean_in_sample": rounded(self.mean_in_sample),
            "mean_out_of_sample": rounded(self.mean_out_of_sample),
            "mean_decay": rounded(self.mean_decay),
            "per_seed_decay": [rounded(v) for v in self.decays],
            "per_seed_rank": list(self.ranks),
            "rank_spread": self.rank_spread,
        }


@dataclass(frozen=True, slots=True)
class Sweep:
    """The bake-off repeated across seeds, which is the only form in which it ranks anything."""

    symbol: str
    bars: int
    budget: int
    seeds: tuple[int, ...]
    runs: tuple[BakeOff, ...]

    @property
    def summaries(self) -> tuple[StrategySummary, ...]:
        names = [r.name for r in self.runs[0].results] if self.runs else []
        per_seed_rank: dict[str, list[int]] = {n: [] for n in names}
        for one in self.runs:
            # A strategy that found nothing has no decay to rank; it goes last rather than being
            # dropped, because dropping it would flatter a method that produced nothing at all.
            ordered = sorted(
                one.results,
                key=lambda r: (r.decay is None, r.decay if r.decay is not None else 0.0),
            )
            for position, result in enumerate(ordered, start=1):
                per_seed_rank[result.name].append(position)

        out = []
        for name in names:
            picked = [next(r for r in one.results if r.name == name) for one in self.runs]
            out.append(StrategySummary(
                name=name,
                decays=tuple(r.decay for r in picked),
                in_samples=tuple(r.in_sample for r in picked),
                out_of_samples=tuple(r.out_of_sample for r in picked),
                ranks=tuple(per_seed_rank[name]),
            ))
        return tuple(out)

    @property
    def by_mean_decay(self) -> tuple[StrategySummary, ...]:
        return tuple(sorted(
            self.summaries,
            key=lambda s: (s.mean_decay is None, s.mean_decay if s.mean_decay is not None else 0.0),
        ))

    @property
    def stable_winner(self) -> str | None:
        """The strategy ranking first on decay in **every** seed, or None if the seed decided it.

        This is the only ordering claim a sweep of this size supports. Anything else is a mean of a
        few numbers whose spread between seeds is wider than the differences between them.
        """
        ranked = self.by_mean_decay
        if not ranked:
            return None
        best = ranked[0]
        return best.name if best.ranks and set(best.ranks) == {1} else None

    @property
    def budget_was_equal(self) -> bool:
        return all(r.budget_was_equal for r in self.runs)

    @property
    def moved_two_or_more(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.by_mean_decay if s.rank_spread >= 2)

    @property
    def verdict(self) -> str:
        if len(self.seeds) < 2:
            return "UNDEFINED: one seed does not rank search strategies. Rerun with at least two"
        if not self.budget_was_equal:
            return (
                "INVALID: the strategies did not spend the same budget, so this ranks spend rather "
                "than method. No ordering below may be read"
            )
        ranked = self.by_mean_decay
        best, worst = ranked[0], ranked[-1]
        moved = ", ".join(self.moved_two_or_more) or "none"
        head = (
            f"Across {len(self.seeds)} seeds at an equal, verified budget, {best.name} decays "
            f"least (mean {best.mean_decay:+.4f}) and {worst.name} most "
            f"(mean {worst.mean_decay:+.4f})."
        )
        if self.stable_winner:
            return (
                f"{head} {self.stable_winner} holds first place in every seed, which is the one "
                f"ordering claim this sweep supports. Strategies moving two or more places between "
                f"seeds: {moved} — for those, a table published from a single run ranks the draw "
                f"rather than the method"
            )
        return (
            f"{head} No strategy holds first place in every seed, so the mean ordering describes "
            f"these draws rather than the methods. Moving two or more places: {moved}"
        )

    def render(self) -> str:
        lines = [
            f"SEARCH SWEEP - {self.symbol}, {self.bars:,} bars, {self.budget} evaluations each, "
            f"seeds {', '.join(str(s) for s in self.seeds)}",
            "",
            f"{'strategy':>14}{'mean IS':>11}{'mean OOS':>11}{'mean decay':>12}"
            f"{'ranks':>12}{'spread':>8}",
        ]

        def shown(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        for s in self.by_mean_decay:
            lines.append(
                f"{s.name:>14}{shown(s.mean_in_sample):>11}{shown(s.mean_out_of_sample):>11}"
                f"{shown(s.mean_decay):>12}{'/'.join(str(r) for r in s.ranks):>12}"
                f"{s.rank_spread:>8}"
            )
        lines += [
            "",
            f"  equal budget honoured across every seed: {self.budget_was_equal}",
            "",
            f"  {self.verdict}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bars": self.bars,
            "budget": self.budget,
            "seeds": list(self.seeds),
            "budget_was_equal": self.budget_was_equal,
            "stable_winner": self.stable_winner,
            "moved_two_or_more": list(self.moved_two_or_more),
            "strategies": [s.as_dict() for s in self.by_mean_decay],
            "verdict": self.verdict,
        }


def sweep(
    bars: Sequence[Any],
    *,
    symbol: str,
    budget: int = BUDGET,
    seeds: Sequence[int] = SWEEP_SEEDS,
    cost_bps: float = 12.0,
    strategies: dict[str, Strategy] | None = None,
) -> Sweep:
    """Run the whole bake-off once per seed and keep the disagreement.

    The single run remains the unit of work; this repeats it and refuses to average the spread
    away. Every seed sees the same bars, the same budget and the same strategy set, so the only
    thing that changes between runs is the draw — which is exactly the quantity a one-seed table
    silently treats as zero.
    """
    if len(seeds) < 1:
        raise SearchError("a sweep needs at least one seed")
    runs = tuple(
        run(bars, symbol=symbol, budget=budget, seed=s, cost_bps=cost_bps, strategies=strategies)
        for s in seeds
    )
    return Sweep(symbol=symbol, bars=len(bars), budget=budget, seeds=tuple(seeds), runs=runs)


def main() -> int:  # pragma: no cover - CLI
    """``--sweep`` repeats the bake-off across :data:`SWEEP_SEEDS`; without it, one seed.

    The single-seed form is kept because it is what the sweep is made of and it finishes fast, but
    its own verdict already refuses to generalise from one run. Read the sweep for any ordering.
    """
    import sys

    from argus.backtest.engine import Bar
    from argus.market.history import CandleType, fetch_range

    symbol = "NVDAUSDT"
    # 62 days of hourly bars: above Track 1's 60-day floor with its 35% out-of-sample slice, and
    # small enough that five strategies at 400 evaluations each finish in minutes rather than
    # hours. The window is a stated scope, not a convenience — a bake-off nobody can rerun is a
    # claim rather than an experiment.
    candles = fetch_range(symbol, days=62, interval="1H", candle_type=CandleType.MARKET)
    bars = [
        Bar(ts=c.ts, close=c.close,
            extra={"volume": float(c.volume), "high": float(c.high), "low": float(c.low)})
        for c in candles
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "--sweep" in sys.argv[1:]:
        swept = sweep(bars, symbol=symbol)
        SWEEP_PATH.write_text(json.dumps(swept.as_dict(), indent=2), encoding="utf-8")
        print(swept.render())
        print(f"\nwritten to {SWEEP_PATH}")
        return 0
    result = run(bars, symbol=symbol)
    REPORT_PATH.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(result.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BUDGET",
    "MAX_DEPTH",
    "MAX_NODE_COST",
    "MIN_BARS",
    "OOS_FRACTION",
    "STRATEGIES",
    "SWEEP_SEEDS",
    "Arena",
    "BakeOff",
    "Candidate",
    "SearchError",
    "StrategyResult",
    "StrategySummary",
    "Sweep",
    "mutate",
    "node_cost",
    "random_expr",
    "run",
    "sweep",
]
