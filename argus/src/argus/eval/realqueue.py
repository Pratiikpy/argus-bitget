"""Scoring queue models against real market-by-order truth, and the rival they are measured against.

The rival is ``nkaz001/hftbacktest`` — the queue models ARGUS ported — configured the way that
project actually ships and uses them, not the way a comparison would find convenient:

* its Python default, ``BacktestAsset()`` with no queue call, is **LogProbQueueModel2**
  (``py-hftbacktest/src/lib.rs:142``). The first real-data artefact here called
  ``PowerProbability n=1`` "hftbacktest's own shipped default"; that was wrong, and the model it
  reported beating the "default" *was* the default;
* its examples use ``power_prob_queue_model(2.0)`` and ``(3.0)`` (market-making notebooks),
  ``power_prob_queue_model3(3.0)`` (the Level-3 study's L2 arm) and ``risk_adverse_queue_model()``;
* the remaining shipped functions (``PowerProbQueueFunc`` n=1, ``PowerProbQueueFunc2``, ``LogProb``)
  are included so that no configuration of theirs is left out.

Every rival model is ARGUS's own port (``execution/queue.py``), which reproduces the Rust formulas
to 1e-16 (`eval/queueproof.py::reproduce`). Whether the *engine* around them also agrees is a
separate question, answered by running hftbacktest itself (:mod:`argus.eval.hftbacktest_run`).

**How a model is driven** follows hftbacktest's L2 no-partial-fill exchange
(``backtest/proc/nopartialfillexchange.rs``): a print at our price calls ``trade`` and then checks
for execution; a print or quote through our price fills; a depth event calls ``depth``. The first
execution is the model's fill time. Truth and model are compared only at the end of an exchange
event, so no model is penalised for the instant between a print and the book update that follows
it.

**Metrics.** ``queue_error``: mean over the episode's event ends of
``|estimated ahead - true ahead| / max(entry, 1 lot)``, averaged over episodes — the metric
`eval/queueproof.py` has always used, kept so the synthetic and real numbers are comparable.
``fill``: the model's fill decision within the horizon against the truth's, as false fills
(the model claims a maker fill the queue never gave — the expensive error) and missed fills.
``cost_bps``: false fills priced at the maker/taker gap, as in `eval/queueproof.py::score`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from argus.eval.l3queue import STEP_BATCH, STEP_DEPTH, STEP_THROUGH, STEP_TRADE, RealEpisode
from argus.execution.queue import (
    LogProbability,
    LogProbability2,
    PowerProbability,
    PowerProbability2,
    PowerProbability3,
    Probability,
    ProbQueue,
    QueueModel,
    QueuePosition,
    RiskAdverseQueue,
)

ModelFactory = Callable[[], QueueModel]

RIVAL_DEFAULT = "hftbacktest default: LogProbQueueFunc2"


def rival_models() -> list[tuple[str, ModelFactory]]:
    """hftbacktest's queue models as that project ships and uses them (see module docstring)."""
    d = Decimal
    return [
        (RIVAL_DEFAULT, lambda: ProbQueue(LogProbability2())),
        ("hftbacktest PowerProbQueueFunc n=1", lambda: ProbQueue(PowerProbability(d("1")))),
        ("hftbacktest PowerProbQueueFunc n=2", lambda: ProbQueue(PowerProbability(d("2")))),
        ("hftbacktest PowerProbQueueFunc n=3", lambda: ProbQueue(PowerProbability(d("3")))),
        ("hftbacktest PowerProbQueueFunc2 n=2", lambda: ProbQueue(PowerProbability2(d("2")))),
        ("hftbacktest PowerProbQueueFunc3 n=3", lambda: ProbQueue(PowerProbability3(d("3")))),
        ("hftbacktest LogProbQueueFunc", lambda: ProbQueue(LogProbability())),
        ("hftbacktest RiskAdverseQueueModel", RiskAdverseQueue),
    ]


class _Constant(Probability):
    def __init__(self, value: str) -> None:
        self._v = Decimal(value)
        self.name = f"const({value})"

    def prob(self, front: Decimal, back: Decimal) -> Decimal:
        return self._v


def ablation_models() -> list[tuple[str, ModelFactory]]:
    """The probability function replaced by a constant that ignores the book."""
    return [
        ("ABLATION: every cancel is ahead of you", lambda: ProbQueue(_Constant("0"))),
        ("ABLATION: coin", lambda: ProbQueue(_Constant("0.5"))),
    ]


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    error: float          # mean normalised |est - true| over the episode's samples
    samples: int
    model_fill: int       # batch index of the model's first execution, -1 if none
    true_fill: int

    def model_fill_ts(self, ep: RealEpisode) -> int:
        """Timestamp of the exchange event in which the model first executed; -1 if never."""
        return int(ep.batch_ts[self.model_fill]) if self.model_fill >= 0 else -1


def drive(
    ep: RealEpisode, model: QueueModel, *, lot: Decimal, view: str = "record",
) -> EpisodeScore:
    """Replay one episode's L2 view through one model and score it against the truth.

    ``view="record"`` feeds every depth update as its own event (hftbacktest's
    ``convert_l3_to_l2``); ``view="event"`` feeds only the last depth update of each exchange event
    (a market-by-price feed). Trades are fed identically in both.
    """
    pos: QueuePosition = model.new_order(Decimal(repr(ep.entry)))
    level = Decimal(repr(ep.entry))
    leaves = lot
    filled_at = -1
    batch = 0
    err = 0.0
    scale = max(ep.entry, float(lot))
    pending_depth: float | None = None
    on_batch_end = getattr(model, "on_batch_end", None)
    truth = ep.truth
    for kind, value in zip(ep.kinds, ep.vals, strict=True):
        if kind == STEP_BATCH:
            if filled_at < 0 and pending_depth is not None:
                new = Decimal(repr(pending_depth))
                model.on_depth(pos, level, new)
                level = new
            pending_depth = None
            if on_batch_end is not None and filled_at < 0:
                on_batch_end(pos)
            est = 0.0 if filled_at >= 0 else max(float(pos.front_qty), 0.0)
            err += abs(est - truth[batch]) / scale
            batch += 1
            continue
        if filled_at >= 0:
            continue
        if kind == STEP_TRADE:
            model.on_trade(pos, Decimal(repr(value)))
            if model.executable(pos, leaves, lot_size=lot) > 0:
                filled_at = batch
        elif kind == STEP_DEPTH:
            if view == "event":
                pending_depth = value
            else:
                new = Decimal(repr(value))
                model.on_depth(pos, level, new)
                level = new
        elif kind == STEP_THROUGH:
            filled_at = batch
    samples = len(truth)
    return EpisodeScore(
        error=err / samples if samples else 0.0, samples=samples,
        model_fill=filled_at, true_fill=ep.true_fill_batch,
    )


@dataclass(frozen=True, slots=True)
class ModelResult:
    model: str
    episodes: int
    queue_error: float
    false_fills: int
    missed_fills: int
    agree: int
    fill_batch_mae: float
    """Mean |model fill batch - true fill batch| over episodes both filled."""
    cost_bps: float
    per_episode: tuple[float, ...]
    per_episode_false: tuple[int, ...]

    @property
    def is_ablation(self) -> bool:
        return self.model.startswith("ABLATION")

    def as_dict(self) -> dict[str, Any]:
        n = max(self.episodes, 1)
        return {
            "model": self.model, "episodes": self.episodes,
            "queue_error": round(self.queue_error, 5),
            "false_fill_rate": round(self.false_fills / n, 5),
            "missed_fill_rate": round(self.missed_fills / n, 5),
            "fill_agreement": round(self.agree / n, 5),
            "fill_batch_mae": round(self.fill_batch_mae, 3),
            "cost_bps": round(self.cost_bps, 4),
        }


def score_models(
    episodes: Sequence[RealEpisode], models: Sequence[tuple[str, ModelFactory]], *,
    lot: Decimal, view: str = "record",
) -> list[ModelResult]:
    """Every model on identical episodes. Only episodes with at least one sample are scored."""
    from argus.desk.allocation import TAKER_BPS
    from argus.market.markout import MAKER_BPS

    usable = [e for e in episodes if e.batches > 0]
    out: list[ModelResult] = []
    for name, factory in models:
        per: list[float] = []
        per_false: list[int] = []
        false = missed = agree = 0
        gaps: list[int] = []
        for ep in usable:
            s = drive(ep, factory(), lot=lot, view=view)
            per.append(s.error)
            mf, tf = s.model_fill >= 0, s.true_fill >= 0
            is_false = int(mf and not tf)
            per_false.append(is_false)
            false += is_false
            missed += int(tf and not mf)
            agree += int(mf == tf)
            if mf and tf:
                gaps.append(abs(s.model_fill - s.true_fill))
        n = max(len(usable), 1)
        out.append(ModelResult(
            model=name, episodes=len(usable), queue_error=sum(per) / n,
            false_fills=false, missed_fills=missed, agree=agree,
            fill_batch_mae=(sum(gaps) / len(gaps)) if gaps else 0.0,
            cost_bps=(false / n) * float(TAKER_BPS - MAKER_BPS),
            per_episode=tuple(per), per_episode_false=tuple(per_false),
        ))
    return sorted(out, key=lambda r: r.queue_error)


def cluster_bootstrap(
    a: Sequence[float], b: Sequence[float], clusters: Sequence[Any], *,
    resamples: int = 2000, seed: int = 20260925,
) -> tuple[float, float, float]:
    """Mean of ``b - a`` with a 95% percentile interval, resampling whole clusters.

    Episodes that overlap in time share the same book events, so they are not independent; an
    i.i.d. interval over them would be too narrow. Resampling contiguous time blocks keeps that
    dependence inside each draw. ``seed`` is fixed so the interval is reproducible exactly.
    """
    groups: dict[Any, list[float]] = {}
    for x, y, c in zip(a, b, clusters, strict=True):
        groups.setdefault(c, []).append(y - x)
    keys = sorted(groups, key=str)
    sums = [sum(groups[k]) for k in keys]
    counts = [len(groups[k]) for k in keys]
    total_n = sum(counts)
    mean = sum(sums) / total_n if total_n else 0.0
    if len(keys) < 2:
        return mean, mean, mean
    rng = random.Random(seed)
    k = len(keys)
    draws: list[float] = []
    for _ in range(resamples):
        s = n = 0.0
        for _j in range(k):
            i = rng.randrange(k)
            s += sums[i]
            n += counts[i]
        draws.append(s / n if n else 0.0)
    draws.sort()
    lo = draws[int(0.025 * resamples)]
    hi = draws[min(math.ceil(0.975 * resamples) - 1, resamples - 1)]
    return mean, lo, hi


def time_blocks(episodes: Sequence[RealEpisode], block_ns: int = 300_000_000_000) -> list[str]:
    """The cluster key of each scored episode: its dataset and 5-minute join block."""
    return [f"{e.dataset}|{e.t_join // block_ns}" for e in episodes if e.batches > 0]


__all__ = [
    "RIVAL_DEFAULT",
    "EpisodeScore",
    "ModelFactory",
    "ModelResult",
    "ablation_models",
    "cluster_bootstrap",
    "drive",
    "rival_models",
    "score_models",
    "time_blocks",
]
