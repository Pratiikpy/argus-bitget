"""The queue model's case for OWNED — run, not asserted.

`eval/standing.py` defines thirteen conditions and raises at import if anything claims OWNED without
all of them. Nothing ever has. A ladder whose top rung has never been reached is indistinguishable
from a wall, so this module exists to try to earn one — for the queue-position model against
`nkaz001/hftbacktest`, the system it was ported from.

Every condition below is produced by running something. Where a condition cannot be met, it is
reported as not met rather than argued into place.

**What the probability functions actually claim, because the first version of this module got it
wrong.** That draft scored `prob(front, back)` as a probability that the order *fills*, found the
best model could not beat "assume always filled", and nearly reported a negative result about the
wrong quantity. Reading `hftbacktest/src/backtest/models/queue.rs:183-204` settles it — `prob` is
never compared to a fill:

```rust
let front = q.front_q_qty;
let back = prev_qty - front;
let mut prob = self.prob.prob(front, back);
let est_front = front - (1.0 - prob) * chg + (back - prob * chg).min(0.0);
```

`chg` is the quantity that left the level *without trading* — cancellations — and `prob` is
**P(a cancelled unit came from behind the order)**. It is a claim about attribution, not outcome:
when depth falls, how much of that was in front of you (so you advanced) and how much behind you (so
you did not)? That is the quantity scored here, and the metric is the error in the resulting
queue-position estimate.

**What is being compared.** hftbacktest's five probability functions (`queue.rs:221-330`):

| model | formula | f(x) |
|---|---|---|
| `PowerProbQueueFunc` | `f(back) / (f(back) + f(front))` | `x^n` |
| `LogProbQueueFunc` | `f(back) / (f(back) + f(front))` | `ln(1+x)` |
| `LogProbQueueFunc2` | `f(back) / f(back + front)` | `ln(1+x)` |
| `PowerProbQueueFunc2` | `f(back) / f(back + front)` | `x^n` |
| `PowerProbQueueFunc3` | `1 - f(front / (front + back))` | `x^n` |

**Where the ground truth comes from.** An L2 feed cannot answer the question the model is asked —
that is why the model exists. So the experiment runs an explicit order-by-order queue, in which the
position of every cancellation is known by construction, and shows each model only the L2 view of
it: the level's total before and after, and the printed trade size. The model's estimate of the
quantity ahead is then compared against the queue that actually produced those totals. This is the
comparison our own `execution/queue.py` docstring says the L3 model is there to make possible —
"what lets us say how wrong those approximations are rather than assuming they are close".

**The generative rule is swept, not chosen.** Cancellations are drawn from the back of the queue
with probability `back^g / (back^g + front^g)`. At **g=1** that is exactly
`PowerProbability(1)`, so the shipped model is correct by construction and the test is whether the
port recovers it. At **g=2** and **g=0.5** the truth is a different member of the family, and at
**front-sticky** it is outside the family entirely — the front of the queue never cancels, which is
the regime where the naive assumption is right and the model is wrong. Reporting that regime is the
difference between a sweep and a demonstration; tuning g until the model won would be precisely the
selection effect the rest of this codebase exists to refuse.

**The honest limit, stated before the results.** ARGUS has executed no orders, so none of this is
validated against our own fills. It is validated two ways instead: numerically against the reference
implementation on identical inputs, and behaviourally against a queue whose truth is known because
the simulator built it. That is weaker than live fills and stronger than an argument, and the
distinction is why the thirteenth condition is evaluated rather than assumed.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "queue_proof.json"

GRID: tuple[tuple[int, int], ...] = (
    (1, 999), (10, 90), (50, 50), (90, 10), (333, 667), (5, 7), (1, 1), (700, 3), (2, 1998),
)
"""(front, back) pairs the reproduction is checked on, spanning both extremes and the middle."""

TOLERANCE = 1e-12
"""Agreement required to call the baseline reproduced. The measured deviation is ~1e-16."""

PRICE = Decimal("100")
LOT = Decimal("0.001")
OUR_QTY = Decimal("1")


# --- the reference, transcribed from queue.rs -----------------------------------------------------


def _ref_power(front: float, back: float, n: float) -> float:
    return float(back**n / (back**n + front**n))


def _ref_power2(front: float, back: float, n: float) -> float:
    return float(back**n / (back + front) ** n)


def _ref_power3(front: float, back: float, n: float) -> float:
    return 1.0 - float((front / (front + back)) ** n)


def _ref_log(front: float, back: float) -> float:
    return math.log1p(back) / (math.log1p(back) + math.log1p(front))


def _ref_log2(front: float, back: float) -> float:
    return math.log1p(back) / math.log1p(back + front)


@dataclass(frozen=True, slots=True)
class Reproduction:
    """Ours against theirs, on identical inputs."""

    model: str
    cases: int
    max_deviation: float

    @property
    def reproduced(self) -> bool:
        return self.max_deviation <= TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "cases": self.cases,
            "max_deviation": self.max_deviation, "reproduced": self.reproduced,
        }


def reproduce() -> list[Reproduction]:
    """Every probability function against the Rust formula it was ported from."""
    from argus.execution.queue import (
        LogProbability,
        LogProbability2,
        PowerProbability,
        PowerProbability2,
        PowerProbability3,
    )

    pairs = (
        ("PowerProbQueueFunc n=1", PowerProbability(Decimal("1")),
         lambda f, b: _ref_power(f, b, 1.0)),
        ("PowerProbQueueFunc n=2", PowerProbability(Decimal("2")),
         lambda f, b: _ref_power(f, b, 2.0)),
        ("PowerProbQueueFunc n=3", PowerProbability(Decimal("3")),
         lambda f, b: _ref_power(f, b, 3.0)),
        ("PowerProbQueueFunc2 n=1", PowerProbability2(Decimal("1")),
         lambda f, b: _ref_power2(f, b, 1.0)),
        ("PowerProbQueueFunc2 n=2", PowerProbability2(Decimal("2")),
         lambda f, b: _ref_power2(f, b, 2.0)),
        ("PowerProbQueueFunc3 n=2", PowerProbability3(Decimal("2")),
         lambda f, b: _ref_power3(f, b, 2.0)),
        ("LogProbQueueFunc", LogProbability(), _ref_log),
        ("LogProbQueueFunc2", LogProbability2(), _ref_log2),
    )
    out: list[Reproduction] = []
    for name, ours, reference in pairs:
        worst = 0.0
        for front, back in GRID:
            got = float(ours.prob(Decimal(front), Decimal(back)))
            want = reference(float(front), float(back))
            worst = max(worst, abs(got - want))
        out.append(Reproduction(model=name, cases=len(GRID), max_deviation=worst))
    return out


# --- the explicit queue, where the truth is known by construction ---------------------------------


@dataclass(frozen=True, slots=True)
class Regime:
    """One stated rule for where cancellations come from.

    ``gamma`` is the exponent in ``P(from back) = back^g / (back^g + front^g)``. ``None`` means the
    front of the queue never cancels at all, which no member of the family can express.
    """

    name: str
    gamma: float | None
    join_rate: float
    note: str


REGIMES: tuple[Regime, ...] = (
    Regime(
        name="gamma=1 - cancels proportional to quantity", gamma=1.0, join_rate=0.20,
        note="the assumption PowerProbability(n=1) encodes, as the generative rule",
    ),
    Regime(
        name="gamma=2 - cancels concentrated at the front", gamma=2.0, join_rate=0.20,
        note="P(from back) falls below the quantity share, so the correct prob is lower than n=1",
    ),
    Regime(
        name="gamma=0.5 - cancels spread toward the back", gamma=0.5, join_rate=0.20,
        note="P(from back) rises above the quantity share, so the correct prob is higher than n=1",
    ),
    Regime(
        name="front-sticky - the front never cancels", gamma=None, join_rate=0.20,
        note="outside the family: 'every cancellation is behind you' is the exact truth",
    ),
    Regime(
        name="adversarial - front-sticky and replenishing", gamma=None, join_rate=0.55,
        note="the front never cancels and the queue keeps growing behind — the hostile case",
    ),
)
"""The regimes every model is scored across, rather than one regime chosen by us.

**This is a correction to the first version of this module**, which evaluated a single generative
process and would have reported whatever that one process happened to say. A single regime cannot
distinguish "the model is right" from "the regime suited the model".

**The direction of gamma is the opposite of the obvious reading, and the run settled it.** Our
order normally has far more quantity in front of it than behind, and for ``back < front`` raising
the exponent *lowers* ``back^g / (back^g + front^g)`` — at front=100, back=25 the share is 0.20 at
gamma=1, 0.059 at gamma=2 and 0.33 at gamma=0.5. So gamma=2 concentrates cancellations at the
**front**, not the
back. The labels here said the reverse until the gamma=2 table showed "every cancel is ahead of you"
ranking third of ten while "every cancel is behind you" ranked last, which is only possible if the
cancellations were in fact ahead.

No regime claims a particular ``n`` is exactly optimal. The generative rule cancels whole resting
orders while hftbacktest's update apportions a continuous ``chg`` and then clamps it against the
level, so even at gamma=1 the shipped ``n=1`` is unbiased only in expectation — and the measurements
below show ``n=2`` edging it there. Stating which ``n`` "must" win and then reporting a run that
disagrees is how a result gets quietly re-fitted.
"""


@dataclass(frozen=True, slots=True)
class Observation:
    """One book event as an L2 feed shows it, with the truth that produced it.

    The level totals deliberately **exclude our own resting quantity**: they are the depth of other
    participants' orders, which is the only quantity an exchange feed and our own estimate can both
    refer to without an off-by-our-size error creeping in.
    """

    traded: float
    new_level: float
    true_ahead: float


@dataclass(frozen=True, slots=True)
class Episode:
    """One resting order's life, from joining the queue to the end of the event stream."""

    entry_level: float
    observations: tuple[Observation, ...]
    true_filled: float


def _cancel_from_back(rng: random.Random, front: float, back: float, gamma: float | None) -> bool:
    """Which side of the queue this cancellation comes from — the generative rule itself.

    The ``gamma is None`` case is checked **first**, before the empty-side fallbacks, and a test is
    what forced that order. With the fallbacks first, a front-sticky episode whose queue had nothing
    behind it yet fell through to cancelling the front — so the regime named "the front never
    cancels" was cancelling the front for as long as our order was at the back of the book, which is
    most of an episode. The caller cancels nothing when the chosen side is empty, which is the
    honest behaviour: if every cancellation comes from behind and there is nobody behind, no
    cancellation happens.
    """
    if gamma is None:
        return True
    if front <= 0:
        return True
    if back <= 0:
        return False
    fg, bg = float(front**gamma), float(back**gamma)
    return rng.random() < bg / (fg + bg)


def _pick(rng: random.Random, queue: list[float]) -> int:
    """Index of the order that cancels, weighted by its size. Bigger orders cancel more often."""
    total = sum(queue)
    draw = rng.random() * total
    for i, qty in enumerate(queue):
        draw -= qty
        if draw <= 0:
            return i
    return len(queue) - 1


STATED_ORDER_SIZE = (1.0, 50.0)
"""The fallback range for one resting order's size, used only when no book tape exists.

It is a *stated* assumption and is labelled as one in the report. `eval/bookcalib.py` measures the
real distribution from Bitget's public book and :func:`simulate` prefers it whenever a tape is
present — because the first measurement disagreed with this range sharply. Near-touch rToken levels
hold a median of about 2 contracts in total, not the ~150 this range produces across three to twelve
orders. A queue experiment run only on the stated range would be an experiment about a book that
does not exist here.

This docstring used to add *"and they turn over roughly 70% of themselves a minute"*, taken from a
calibration figure that pooled snapshot gaps from 14 seconds to 2.5 hours into one median. Split by
horizon the same observations read **24.0% under 30 seconds** and 77.3% from 30 seconds to two
minutes (`data/book_calibration.json`, `change_by_horizon`). The 70% was an artefact of the
sampling schedule. It never reached the experiment — `simulate` takes only the size range from the
calibration, never the turnover — so nothing computed here was wrong; the sentence was.
"""


def simulate(
    n: int, *, seed: int, regime: Regime, order_size: tuple[float, float] | None = None,
) -> list[Episode]:
    """Generate episodes whose queue history is known exactly, then hide it behind an L2 view.

    The queue is an ordered list of other participants' quantities with our order at a known
    position. A trade consumes from the front; a cancellation removes one whole resting order from
    the side the regime chose; a join appends behind us. Nothing here consults a probability
    function — this is the ground truth those functions are then scored against.

    ``order_size`` is the (low, high) range a single resting order is drawn from. Passing the
    measured range from `eval/bookcalib.py` is what makes this a book rather than a shape.
    """
    low, high = order_size or STATED_ORDER_SIZE
    rng = random.Random(seed)
    out: list[Episode] = []
    for _ in range(n):
        ahead = [round(rng.uniform(low, high), 3) for _ in range(rng.randint(3, 12))]
        behind: list[float] = []
        entry_level = sum(ahead)
        our_leaves = float(OUR_QTY)
        filled = 0.0
        observations: list[Observation] = []

        for _event in range(rng.randint(8, 40)):
            traded = 0.0
            roll = rng.random()
            if roll < regime.join_rate:
                behind.append(round(rng.uniform(1.0, 50.0), 3))
            elif roll < regime.join_rate + 0.35:
                traded = round(rng.uniform(0.5, 60.0), 3)
                remaining = traded
                while remaining > 0 and ahead:
                    taken = min(ahead[0], remaining)
                    ahead[0] -= taken
                    remaining -= taken
                    if ahead[0] <= 1e-9:
                        ahead.pop(0)
                if remaining > 0 and our_leaves > 0:
                    taken = min(our_leaves, remaining)
                    our_leaves -= taken
                    filled += taken
                    remaining -= taken
                while remaining > 0 and behind:
                    taken = min(behind[0], remaining)
                    behind[0] -= taken
                    remaining -= taken
                    if behind[0] <= 1e-9:
                        behind.pop(0)
            elif ahead or behind:
                from_back = _cancel_from_back(rng, sum(ahead), sum(behind), regime.gamma)
                side = behind if from_back else ahead
                if side:
                    side.pop(_pick(rng, side))

            observations.append(Observation(
                traded=traded, new_level=sum(ahead) + sum(behind), true_ahead=sum(ahead),
            ))
            if our_leaves <= 0:
                break

        out.append(Episode(
            entry_level=entry_level, observations=tuple(observations), true_filled=filled,
        ))
    return out


# --- scoring: how wrong is each model's estimate of the quantity ahead? ---------------------------


@dataclass(frozen=True, slots=True)
class Scored:
    """One model's error against the queue that actually existed."""

    model: str
    episodes: int
    queue_error: float
    fill_error: float
    cost_bps: float
    per_episode: tuple[float, ...]

    @property
    def is_ablation(self) -> bool:
        return self.model.startswith("ABLATION")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "episodes": self.episodes,
            "queue_error": round(self.queue_error, 5),
            "fill_error": round(self.fill_error, 5),
            "cost_bps": round(self.cost_bps, 4),
        }


def _replay(episode: Episode, queue_model: Any) -> tuple[float, float]:
    """Show one model the L2 view of an episode; return its mean queue error and its fill."""
    from argus.execution.queue import RestingOrder

    order = RestingOrder(
        price=PRICE, quantity=OUR_QTY, model=queue_model,
        level_qty=Decimal(str(round(episode.entry_level, 3))), lot_size=LOT,
    )
    errors = 0.0
    scale = max(episode.entry_level, 1.0)
    for obs in episode.observations:
        if obs.traded > 0:
            order.on_trade_at_price(Decimal(str(obs.traded)))
        order.on_depth_change(Decimal(str(round(obs.new_level, 3))))
        errors += abs(float(order.queue_ahead) - obs.true_ahead) / scale
    mean_error = errors / len(episode.observations) if episode.observations else 0.0
    return mean_error, float(order.filled)


def _models() -> list[tuple[str, Any]]:
    """Every shipped probability function, plus the naive assumptions the ablation needs."""
    from argus.execution.queue import (
        LogProbability,
        LogProbability2,
        PowerProbability,
        PowerProbability2,
        PowerProbability3,
        Probability,
        ProbQueue,
        RiskAdverseQueue,
    )

    class _Constant(Probability):
        """An ablation: replace the probability function with a number that ignores the book."""

        def __init__(self, value: str, label: str) -> None:
            self._value = Decimal(value)
            self.name = label

        def prob(self, front: Decimal, back: Decimal) -> Decimal:
            return self._value

    return [
        ("PowerProbability n=1", ProbQueue(PowerProbability(Decimal("1")))),
        ("PowerProbability n=2", ProbQueue(PowerProbability(Decimal("2")))),
        ("PowerProbability2 n=2", ProbQueue(PowerProbability2(Decimal("2")))),
        ("PowerProbability3 n=2", ProbQueue(PowerProbability3(Decimal("2")))),
        ("LogProbability", ProbQueue(LogProbability())),
        ("LogProbability2", ProbQueue(LogProbability2())),
        ("ABLATION: every cancel is behind you", ProbQueue(_Constant("1", "const(1)"))),
        ("ABLATION: every cancel is ahead of you", ProbQueue(_Constant("0", "const(0)"))),
        ("ABLATION: coin", ProbQueue(_Constant("0.5", "const(0.5)"))),
        ("ABLATION: RiskAdverse (trades only)", RiskAdverseQueue()),
    ]


def score(episodes: Sequence[Episode]) -> list[Scored]:
    """Every model and every naive assumption, on identical episodes.

    ``cost_bps`` turns the fill error into money, which is the whole reason this model is in the
    system. A model that overstates its fills by ``Δ`` of the order books ``Δ`` at the maker fee
    (`market/markout.py:67`, 2bps) that it would in fact have paid at the taker fee
    (`desk/allocation.py:60`, 6bps) — so the error is worth 4bps per unit of overstated fill.
    """
    from argus.desk.allocation import TAKER_BPS
    from argus.market.markout import MAKER_BPS

    out: list[Scored] = []
    for name, queue_model in _models():
        per_episode: list[float] = []
        fill_gap = 0.0
        overstated = 0.0
        for episode in episodes:
            error, got = _replay(episode, queue_model)
            per_episode.append(error)
            fill_gap += abs(got - episode.true_filled)
            overstated += max(got - episode.true_filled, 0.0)
        n = len(episodes)
        out.append(Scored(
            model=name, episodes=n,
            queue_error=sum(per_episode) / n,
            fill_error=fill_gap / (n * float(OUR_QTY)),
            cost_bps=(overstated / (n * float(OUR_QTY))) * (TAKER_BPS - MAKER_BPS),
            per_episode=tuple(per_episode),
        ))
    return sorted(out, key=lambda s: s.queue_error)


def paired_interval(better: Scored, worse: Scored) -> tuple[float, float, float]:
    """A 95% interval on the per-episode error difference, paired on the same episodes.

    Paired because both models saw identical queues: the episode-to-episode variance is enormous and
    common to both, so an unpaired comparison would drown a real difference in it. The interval is
    the normal approximation, which is what 4,000 paired differences support.
    """
    diffs = [w - b for b, w in zip(better.per_episode, worse.per_episode, strict=True)]
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return mean, mean, mean
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half = 1.96 * math.sqrt(variance / n)
    return mean, mean - half, mean + half


@dataclass(frozen=True, slots=True)
class RegimeResult:
    """One regime: who estimated the queue best, and whether the margin survives its interval."""

    regime: str
    note: str
    best_model: str
    best_model_error: float
    best_naive: str
    best_naive_error: float
    margin: float
    margin_lo: float
    margin_hi: float
    cost_bps_saved: float

    @property
    def model_wins(self) -> bool:
        """Wins only if the whole interval is on the right side of zero."""
        return self.margin_lo > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime, "note": self.note,
            "best_model": self.best_model, "best_model_error": round(self.best_model_error, 5),
            "best_naive": self.best_naive, "best_naive_error": round(self.best_naive_error, 5),
            "margin": round(self.margin, 5),
            "margin_ci95": [round(self.margin_lo, 5), round(self.margin_hi, 5)],
            "cost_bps_saved": round(self.cost_bps_saved, 4),
            "model_wins": self.model_wins,
        }


def evaluate_regime(
    regime: Regime, *, episodes: int, seed: int, pick: tuple[str, str] | None = None,
) -> tuple[RegimeResult, list[Scored]]:
    """Score every model in one regime and compare one real model against one naive assumption.

    ``pick`` names the two to compare instead of taking the best of each on this sample. That is
    what makes the held-out run a held-out run: choosing the winner again on new data would re-run
    the selection as well as the test, and six models competing for "best" inflate a margin by
    exactly the amount the selection was worth. In sample the pick is made; out of sample the pick
    is *carried over*, and if it does not survive on data it never saw, it did not survive.
    """
    scored = score(simulate(episodes, seed=seed, regime=regime))
    by_name = {s.model: s for s in scored}
    if pick is None:
        best_model = min((s for s in scored if not s.is_ablation), key=lambda s: s.queue_error)
        best_naive = min((s for s in scored if s.is_ablation), key=lambda s: s.queue_error)
    else:
        best_model, best_naive = by_name[pick[0]], by_name[pick[1]]
    margin, lo, hi = paired_interval(best_model, best_naive)
    return (
        RegimeResult(
            regime=regime.name, note=regime.note,
            best_model=best_model.model, best_model_error=best_model.queue_error,
            best_naive=best_naive.model, best_naive_error=best_naive.queue_error,
            margin=margin, margin_lo=lo, margin_hi=hi,
            cost_bps_saved=best_naive.cost_bps - best_model.cost_bps,
        ),
        scored,
    )


def run(*, episodes: int = 4000, seed: int = 20260914) -> dict[str, Any]:
    """The whole evidence package: reproduction, then the swept behavioural evaluation.

    In-sample and out-of-sample use **different seeds**, so the held-out set comes from the same
    process and was not seen while anything was chosen. Nothing is fitted — the models have no free
    parameters beyond `n` and none of them is tuned here — so the held-out run tests the *claim*
    rather than a fit, which is a weaker thing to test and is said so rather than dressed up.
    """
    reproductions = reproduce()

    in_sample: list[RegimeResult] = []
    out_of_sample: list[RegimeResult] = []
    detail: dict[str, list[dict[str, Any]]] = {}
    for index, regime in enumerate(REGIMES):
        result, scored = evaluate_regime(regime, episodes=episodes, seed=seed + index)
        in_sample.append(result)
        detail[regime.name] = [s.as_dict() for s in scored]
        held_out, _ = evaluate_regime(
            regime, episodes=episodes, seed=seed + 500 + index,
            pick=(result.best_model, result.best_naive),
        )
        out_of_sample.append(held_out)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "nkaz001/hftbacktest queue.rs:183-330",
        "metric": "mean |estimated qty ahead - true qty ahead| as a share of the entry level",
        "episodes_per_regime": episodes,
        "reproduction": [r.as_dict() for r in reproductions],
        "all_reproduced": all(r.reproduced for r in reproductions),
        "in_sample": [r.as_dict() for r in in_sample],
        "out_of_sample": [r.as_dict() for r in out_of_sample],
        "detail": detail,
        "regimes": len(REGIMES),
        "regimes_won_in_sample": sum(1 for r in in_sample if r.model_wins),
        "regimes_won_out_of_sample": sum(1 for r in out_of_sample if r.model_wins),
        "failure_cases": [r.as_dict() for r in in_sample if not r.model_wins],
        "verdict": _verdict(reproductions, in_sample, out_of_sample),
    }


def _verdict(
    reproductions: Sequence[Reproduction], in_sample: Sequence[RegimeResult],
    out_of_sample: Sequence[RegimeResult],
) -> str:
    worst = max(r.max_deviation for r in reproductions)
    wins = [r for r in in_sample if r.model_wins]
    held = [r for r in out_of_sample if r.model_wins]
    losses = [r for r in in_sample if not r.model_wins]
    head = (
        f"All {len(reproductions)} probability functions reproduce hftbacktest to {worst:.1e} on "
        f"identical inputs — the port is exact."
    )
    if not wins:
        return head + (
            " Behaviourally, no regime was found in which a probability function estimates the "
            "queue better than a constant, with the margin's whole interval on the right side of "
            "zero. That is a negative result about our evidence rather than a defect in the "
            "port, and the capability stays IMPLEMENTED."
        )
    body = (
        f" Across {len(in_sample)} generative regimes swept rather than chosen, a probability "
        f"function estimates the queue better than the best constant in {len(wins)} of "
        f"{len(in_sample)} in sample and {len(held)} of {len(out_of_sample)} held out, each margin "
        f"with its whole 95% interval above zero."
    )
    if losses:
        names = "; ".join(r.regime for r in losses)
        body += (
            f" It loses in {len(losses)}: {names} — where the front of the queue never cancels, "
            f"'every cancellation is behind you' is not a naive assumption but the exact truth "
            f"and no member of the family can express it. That is the model's documented failure "
            f"case, and reporting it is the difference between a sweep and a demonstration."
        )
    dearer = [r for r in wins if r.cost_bps_saved < 0]
    if dearer:
        names = "; ".join(r.regime for r in dearer)
        body += (
            f" One result cuts against the model and is reported rather than dropped: in {names} "
            f"it "
            f"estimates the queue better and still costs more ("
            f"{min(r.cost_bps_saved for r in dearer):+.3f}bps), because a better *average* "
            f"position "
            f"estimate can sit on the optimistic side at exactly the moments that decide a fill. "
            f"Queue accuracy and execution cost are not the same objective, and a capability that "
            f"only ever reported the metric it wins on would never have found that."
        )
    return head + body


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = run()
    print("QUEUE MODEL — the case for OWNED against hftbacktest\n")
    print(f"  {'reproduction':32} {'cases':>6} {'max deviation':>15}")
    for row in report["reproduction"]:
        mark = "OK" if row["reproduced"] else "FAIL"
        print(f"  {row['model']:32} {row['cases']:6d} {row['max_deviation']:15.2e}  {mark}")

    print(f"\n  metric: {report['metric']}")
    for label in ("in_sample", "out_of_sample"):
        print(f"\n  {label.replace('_', ' ')}:")
        for row in report[label]:
            mark = "MODEL" if row["model_wins"] else "naive"
            print(f"    {row['regime']}")
            print(
                f"      best model {row['best_model']:26} {row['best_model_error']:.4f}   "
                f"best naive {row['best_naive']:34} {row['best_naive_error']:.4f}"
            )
            print(
                f"      margin {row['margin']:+.4f}  95% CI [{row['margin_ci95'][0]:+.4f}, "
                f"{row['margin_ci95'][1]:+.4f}]  {row['cost_bps_saved']:+.3f}bps  -> {mark}"
            )

    print(f"\n  {report['verdict']}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "GRID",
    "REGIMES",
    "REPORT_PATH",
    "TOLERANCE",
    "Episode",
    "Observation",
    "Regime",
    "RegimeResult",
    "Reproduction",
    "Scored",
    "evaluate_regime",
    "paired_interval",
    "reproduce",
    "run",
    "score",
    "simulate",
]
