"""The factor laboratory — a searcher that cannot see its own scores.

Track 2's Factor Discovery sub-theme asks how an agent "proposes hypotheses, discovers alpha
factors, and translates into tradable decisions". Almost every system answers by looping
propose → score → feed the score back. Three we tore down do exactly that, and it is the defect
that makes their results meaningless:

* **mcts-llm-alpha** computes a genuine IS/OOS overfitting score at ``qlib_evaluator.py:143``, then
  **unconditionally overwrites it** with the generating model's self-judgment
  (``comprehensive.py:90-98``).
* **RD-Agent** — Microsoft's, the strongest factor machinery in existence — uses a single
  ``APIBackend()`` to both propose factors and judge them. No independent evaluator, **no purged
  CV, no embargo, no deflated Sharpe, no PBO, and no trial counter.**
* **FactorForge** feeds the generator the IC of the top three factors every round and asks for
  variations (``evolution_engine.py:95-99``).

**FactorMiner is the closest thing to a counter-example, and it does not hold either.** This
docstring previously credited its generator with seeing "only syntax errors, never scores", and the
source does not support that. ``factorminer/agent/factor_generator.py:113-118`` declares
``generate_batch(memory_signal=..., library_state=...)`` — "guided by memory priors" — and the
signal is rendered straight into the user prompt at ``:165-171``. What it carries is not neutral:
``factorminer/memory/retrieval.py:742-750`` writes ``=== RECOMMENDED DIRECTIONS (P_succ) ===``
followed by each pattern's ``success_rate`` grade, and ``library_state`` carries
``recent_admissions``, the names of the factors that scored well enough to be admitted. Coarse
grades are still outcomes. Its *evaluator* is genuinely independent of its generator, which is more
than RD-Agent manages; its *memory* is the side door.

That finding is what `argus.research.memory` is built around: the lab's own memory records
everything and publishes only identity and structural facts, with an import-time guard that fails
if an outcome-bearing field is ever added to the signal.

This lab enforces the separation structurally rather than by convention:

1. The proposer is handed :class:`ProposerContext`, which physically cannot carry a score — it has
   no field for one. Adding one would be a visible change to a frozen dataclass.
2. Evaluation is deterministic Python over price data. No model grades a factor.
3. Every proposal increments a **trial counter**, and the counter is what the Deflated Sharpe gate
   consumes. A search that does not record how many times it looked has not produced a result.
4. A factor holds an explicit lifecycle state and cannot skip one. ``RETIRE`` exists and fires on
   measured decay, because a library that only ever grows is lying about decay.
5. **A factor must reproduce itself (added 2026-09-25).** Certification also requires the
   split-half reliability gate (:func:`split_half`): the factor's payoff on one half of every block
   must predict its payoff on the other half, against an exact sign-flip null. It is GoEmotions'
   split-half PPCA (google-research, Apache-2.0, `goemotions/ppca.py:87-99,136-217`) with one
   stated departure — uncentred, because a factor's mean payoff is what is under test — and it asks
   a question none of the other gates asks: not "is this better than shuffled" or "does it survive
   costs", but "would an independent half of the same evidence have found the same thing". Planted
   noise fails it and a planted real edge passes it (`tests/test_factor_split_half.py`); what it
   does to today's primitives on real data is in ``data/factor_split_half.json``
   (:mod:`argus.eval.factor_split_half`). :func:`reliable_components` applies the full
   multi-dimensional PPCA to the library and counts how many independent payoff dimensions it
   actually reproduces.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from argus.backtest.engine import Bar, run
from argus.backtest.metrics import (
    HOURLY_PER_YEAR,
    MetricError,
    deflated_sharpe,
)
from argus.cost.model import CostModel
from argus.research.overfit import Observation, Outcome, OverfitReport, run_all
from argus.truth.clocks import DualClock, SessionPhase


class Lifecycle(StrEnum):
    """A factor's state. It cannot skip one, and RETIRE is reachable."""

    PROPOSED = "proposed"
    FORMALIZED = "formalized"
    BACKTESTED = "backtested"
    COST_CHECKED = "cost_checked"
    OOS_TESTED = "oos_tested"
    DSR_GATED = "dsr_gated"
    CERTIFIED = "certified"
    DEPLOYED = "deployed"
    DECAYED = "decayed"
    RETIRED = "retired"
    REJECTED = "rejected"


_ORDER = (
    Lifecycle.PROPOSED, Lifecycle.FORMALIZED, Lifecycle.BACKTESTED, Lifecycle.COST_CHECKED,
    Lifecycle.OOS_TESTED, Lifecycle.DSR_GATED, Lifecycle.CERTIFIED, Lifecycle.DEPLOYED,
    Lifecycle.DECAYED, Lifecycle.RETIRED,
)


class LifecycleViolation(RuntimeError):
    """A factor tried to skip a gate. Certification is the whole product; skipping is a bug."""


@dataclass(frozen=True, slots=True)
class ProposerContext:
    """Everything the proposer is allowed to see.

    **There is deliberately no field for performance.** Not a filtered one, not an aggregate — the
    shape itself cannot carry a score, so contaminating the loop requires editing this class, which
    is visible in a diff. That is the difference between a convention and a guarantee.
    """

    market_structure: str
    already_proposed: tuple[str, ...]
    trials_so_far: int
    """The proposer may know *how many* times it has been asked, which is not a signal about
    quality — and it prevents the pathological loop of re-proposing the same thing forever."""

    memory: tuple[str, ...] = ()
    """Lines from :meth:`argus.research.memory.MemorySignal.render`, when a memory is attached.

    Every one of them is identity or structure — what has already been evaluated, and what could not
    be measured at all. They are built from a whitelist in `research/memory.py` and a guard there
    fails at import if an outcome-bearing field is ever added, because a memory is the one way to
    contaminate this class without editing it."""


@dataclass(frozen=True, slots=True)
class Factor:
    """A proposed factor. ``expression`` is a name in :data:`PRIMITIVES`."""

    name: str
    expression: str
    rationale: str
    horizon_bars: int = 24

    def __post_init__(self) -> None:
        if self.expression not in PRIMITIVES:
            raise ValueError(
                f"unknown expression {self.expression!r}; the evaluator only runs vetted "
                f"primitives, never model-authored code"
            )


@dataclass
class FactorRecord:
    """A factor and everything measured about it. Append-only history."""

    factor: Factor
    trial_number: int
    state: Lifecycle = Lifecycle.PROPOSED
    history: list[tuple[str, str]] = field(default_factory=list)
    gross_sharpe: float | None = None
    net_sharpe: float | None = None
    oos_sharpe: float | None = None
    dsr: float | None = None
    overfit: dict[str, Any] | None = None
    """The anti-overfit report, when one could be computed. ``None`` means the factor never
    reached the gate, which is different from reaching it and being found wanting."""

    split_half: dict[str, Any] | None = None
    """The split-half reliability verdict (:class:`SplitHalf`), with the same ``None`` meaning."""

    rejection_reason: str = ""

    def advance(self, to: Lifecycle, *, note: str = "") -> None:
        if to in (Lifecycle.REJECTED, Lifecycle.RETIRED):
            self.history.append((str(self.state), str(to)))
            self.state = to
            return
        try:
            here, there = _ORDER.index(self.state), _ORDER.index(to)
        except ValueError:
            raise LifecycleViolation(f"{self.state} -> {to} is not on the lifecycle") from None
        if there != here + 1:
            raise LifecycleViolation(
                f"{self.factor.name}: {self.state} -> {to} skips a gate. Certification is the "
                f"product; a factor that skips a gate has not been certified."
            )
        self.history.append((str(self.state), str(to)))
        self.state = to

    def reject(self, reason: str) -> None:
        self.rejection_reason = reason
        self.advance(Lifecycle.REJECTED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.factor.name,
            "expression": self.factor.expression,
            "trial": self.trial_number,
            "state": str(self.state),
            "gross_sharpe": self.gross_sharpe,
            "net_sharpe": self.net_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "dsr": self.dsr,
            "overfit": self.overfit,
            "split_half": self.split_half,
            "rejection_reason": self.rejection_reason,
            "path": [f"{a}->{b}" for a, b in self.history],
        }


# --- the primitive library the evaluator will run --------------------------------------------
# Vetted signal functions, referenced by name. The proposer picks from these rather than emitting
# code: model-authored code in an evaluator is an arbitrary-execution surface, and sandboxing it
# is a larger problem than this sub-theme needs.

_CLOCK = DualClock()


def _ret(bars: Sequence[Bar], i: int, n: int) -> float:
    j = i - n
    if j < 0:
        return 0.0
    a, b = float(bars[j].close), float(bars[i].close)
    return (b - a) / a if a > 0 else 0.0


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _closed(bars: Sequence[Bar], i: int) -> bool:
    return not _CLOCK.phase(bars[i].ts).has_price_discovery


PRIMITIVES: dict[str, Any] = {
    "long_while_closed": lambda b, i: 1.0 if _closed(b, i) else 0.0,
    "long_while_open": lambda b, i: 1.0 if not _closed(b, i) else 0.0,
    "weekend_only": lambda b, i: 1.0 if _CLOCK.phase(b[i].ts) is SessionPhase.WEEKEND else 0.0,
    # Flat when the lookback has no history. These two previously fell through to a full -1.0
    # (and +1.0) position on six bars of missing data, because `_ret` returns 0.0 and `0.0 > 0` is
    # False. A momentum factor taking a maximum short on data it does not have is a bug; it was
    # found by reconstructing these primitives in `argus.research.grammar` and diffing the two.
    "closure_momentum": lambda b, i: (
        _sign(_ret(b, i, 6)) if _closed(b, i) else 0.0
    ),
    "closure_reversion": lambda b, i: (
        -_sign(_ret(b, i, 6)) if _closed(b, i) else 0.0
    ),
    "near_reopen": lambda b, i: (
        1.0 if _closed(b, i) and _CLOCK.state(b[i].ts).hours_to_next_discovery <= 4 else 0.0
    ),
    "slow_trend": lambda b, i: 1.0 if _ret(b, i, 48) > 0 else 0.0,
    "slow_fade": lambda b, i: -1.0 if _ret(b, i, 48) > 0 else 1.0,
}


# --- split-half reliability ------------------------------------------------------------------
# Adapted from GoEmotions' split-half Probabilistic PCA (google-research/goemotions, Apache-2.0,
# `ppca.py:87-99` for PPCA and `ppca.py:136-217` for the held-out split). Copyright 2021 The
# Google Research Authors; notice in `licenses/google-research-goemotions-APACHE-2.0.txt`.

SPLIT_HALF_ALPHA = 0.05
"""Largest sign-flip p-value at which a factor's payoff counts as reproduced across halves."""

SPLIT_HALF_FLIPS = 2000
"""Sign-flip draws per null. 2,000 puts the resolution of the p-value at 0.0005, twenty times finer
than the threshold it is compared against."""

SPLIT_HALF_MIN_BLOCKS = 12
"""Fewer paired blocks than this and the gate returns INCONCLUSIVE rather than a verdict: a
correlation over a handful of pairs is decided by one of them."""

SPLIT_HALF_BLOCK_BARS = 120
"""Bars per item: five days of hourly bars, so 90 days give 18 paired blocks. **Measured, not
chosen for looks.** The statistic's z-score is roughly ``sqrt(n) * rho / sqrt(1 + 2 * rho)`` with
``n`` blocks and ``rho = snr² * B / 2`` the squared signal-to-noise of one half-block's mean payoff,
so short blocks bury a real edge in per-half noise. Written first with one block per day (the lab's
``horizon_bars``), a planted factor calling the next hour's sign right 58% of the time — an enormous
edge — passed only 14 times in 40 on fat-tailed synthetic returns. Swept on 2026-09-25 over
2,160-bar planted series: at 24 bars the 58% factor passed 37% of 60; at 96, 120 and 144 bars the
58% and 60% factors passed 60/93, 74/96 and 73/95 of 100; noise passed 5-8% throughout against a
nominal 5%. 120 is the shortest block at which a 60%-accurate factor passes at least 95% of the
time, and it keeps the block count above :data:`SPLIT_HALF_MIN_BLOCKS` on a 90-day history. The
real-return version of this sweep is published in ``data/factor_split_half.json`` and is weaker:
on the twelve rTokens' real hourly returns (fatter tails than the synthetic series) a 60%-accurate
factor passed 65-69% of the time at 120 bars and a 62% one 80%, with noise passing 3.3-5.2%. The
block was not re-tuned on those numbers, since they are the check on the choice. The ceiling is
structural: requiring each half to reproduce the payoff costs about half the z-score of a plain
mean test on the whole sample, which is the price of asking for replication rather than
significance."""

SPLIT_HALF_SEED = 20260925
"""Fixed for the same reason as :data:`argus.research.overfit.PERMUTATION_SEED`: a null drawn
afresh each run is a different test each run."""


@dataclass(frozen=True, slots=True)
class SplitHalf:
    """Is a factor's payoff reproduced by an independent half of the same evidence?

    The design is GoEmotions' (`ppca.py:136-217`): the same items rated by two disjoint halves of
    the raters, and a dimension kept only when one half's scores predict the other's. Here an item
    is a block of ``block`` bars (:data:`SPLIT_HALF_BLOCK_BARS` in the lab), the two "raters" are
    its even- and odd-offset bars, and the rating is the factor's payoff — factor value times the
    next bar's return — averaged over that half.
    Two halves of one block share the block's regime and nothing else, because a one-bar forward
    return on one bar does not overlap the next bar's.

    **Uncentred, deliberately, and this is the departure from the source.** GoEmotions demeans each
    half (`ppca.py:89-90`) because it looks for dimensions along which *examples differ*; the grand
    mean of a rating is a scale artefact there. A factor's mean payoff is the very thing under test:
    demeaned, a factor with a perfectly constant edge scores exactly zero reliability, and one whose
    payoff merely swings with the market scores high. So the statistic is ``Σ x_b y_b`` — the
    one-dimensional uncentred symmetrised cross-covariance, half of what `PPCA` would eigendecompose
    — and its scale-free form, Tucker's congruence, is reported beside it.

    **The null is exact, not asymptotic.** Under "no reproducible payoff" each block's product
    ``x_b y_b`` is as likely to be negative as positive, so flipping their signs at random draws
    from the statistic's own null distribution. It is one-sided: agreement is the claim.

    **What passing does not mean.** A factor that is always long reproduces the market's drift in
    both halves and passes. Reliability is not validity, as it is not in GoEmotions; the placebo and
    the cost gate are the validity tests, and this gate runs beside them, not instead of them.
    """

    outcome: Outcome
    blocks: int
    statistic: float
    congruence: float | None
    p_value: float | None
    reason: str

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.outcome),
            "blocks": self.blocks,
            "statistic": self.statistic,
            "congruence": None if self.congruence is None else round(self.congruence, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 4),
            "reason": self.reason,
        }


def payoff_halves(
    values: Sequence[float], returns: Sequence[float], *, block: int,
) -> tuple[list[float], list[float]]:
    """Per block of ``block`` bars, the mean payoff on even-offset bars and on odd-offset bars.

    ``values[i]`` is the factor reading at bar ``i`` and ``returns[i]`` the return that followed it.
    A trailing partial block is dropped rather than paired unevenly. Blocks where either half is
    empty are skipped.
    """
    if block < 2:
        raise ValueError("a block needs at least two bars to have two halves")
    if len(values) != len(returns):
        raise ValueError(f"{len(values)} readings against {len(returns)} returns")
    xs: list[float] = []
    ys: list[float] = []
    for start in range(0, len(values) - block + 1, block):
        even = [values[i] * returns[i] for i in range(start, start + block, 2)]
        odd = [values[i] * returns[i] for i in range(start + 1, start + block, 2)]
        if even and odd:
            xs.append(sum(even) / len(even))
            ys.append(sum(odd) / len(odd))
    return xs, ys


def split_half(
    values: Sequence[float], returns: Sequence[float], *, block: int,
    flips: int = SPLIT_HALF_FLIPS, alpha: float = SPLIT_HALF_ALPHA, seed: int = SPLIT_HALF_SEED,
) -> SplitHalf:
    """The split-half reliability gate for one factor. See :class:`SplitHalf`."""
    xs, ys = payoff_halves(values, returns, block=block)
    products = [x * y for x, y in zip(xs, ys, strict=True)]
    statistic = sum(products)
    if len(products) < SPLIT_HALF_MIN_BLOCKS:
        return SplitHalf(Outcome.INCONCLUSIVE, len(products), statistic, None, None,
                         f"{len(products)} paired block(s); {SPLIT_HALF_MIN_BLOCKS} are needed")
    norm = math.sqrt(sum(x * x for x in xs) * sum(y * y for y in ys))
    if norm == 0.0:
        return SplitHalf(Outcome.FAIL, len(products), statistic, None, 1.0,
                         "the factor never paid or lost anything: no payoff to reproduce")
    congruence = statistic / norm
    rng = random.Random(seed)
    at_least = sum(
        1 for _ in range(flips)
        if sum(p if rng.random() < 0.5 else -p for p in products) >= statistic
    )
    p_value = (1 + at_least) / (1 + flips)
    if statistic > 0 and p_value <= alpha:
        return SplitHalf(Outcome.PASS, len(products), statistic, congruence, p_value,
                         f"payoff reproduced across halves (congruence {congruence:.3f}, "
                         f"p={p_value:.4f})")
    return SplitHalf(Outcome.FAIL, len(products), statistic, congruence, p_value,
                     f"payoff not reproduced across independent halves (congruence "
                     f"{congruence:.3f}, p={p_value:.4f} > {alpha})")


def symmetric_eigh(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues (descending) and eigenvectors (as columns) of a real symmetric matrix.

    Cyclic Jacobi rotations, pure Python, because ARGUS's shipped code does not depend on numpy
    (see `research/overfit.py`). GoEmotions calls ``numpy.linalg.eigh`` and flips the result into
    descending order (`ppca.py:95-98`); this returns the same ordering, and the test suite pins it
    against numpy. The matrices here are one row per factor, so O(n³) per sweep is nothing.
    """
    n = len(matrix)
    a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > 1e-9 * (1.0 + abs(a[i][j])):
                raise ValueError("matrix is not symmetric")
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(100):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        scale = math.sqrt(sum(a[i][i] ** 2 for i in range(n))) or 1.0
        if off <= 1e-14 * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if a[p][q] == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: a[i][i], reverse=True)
    values = [a[i][i] for i in order]
    vectors = [[v[r][i] for i in order] for r in range(n)]
    return values, vectors


def ppca(
    x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], *, demean: bool = True,
) -> tuple[list[float], list[list[float]]]:
    """GoEmotions' ``PPCA(x, y)`` (`ppca.py:87-99`): eigen-decompose ``xᵀy + yᵀx``.

    Rows are items, columns are dimensions, and ``x`` and ``y`` are the two independent halves.
    ``demean=True`` is the source exactly; ``demean=False`` keeps the mean, which is the form the
    factor gate needs (:class:`SplitHalf` says why). Returns eigenvalues in descending order — the
    component covariances — and the eigenvectors as columns, as the source does after its flips.
    """
    if not x or len(x) != len(y):
        raise ValueError("the two halves must rate the same, non-empty set of items")
    dims = len(x[0])

    def centred(m: Sequence[Sequence[float]]) -> list[list[float]]:
        rows = [[float(v) for v in row] for row in m]
        if any(len(row) != dims for row in rows):
            raise ValueError("every item must be rated on every dimension")
        if not demean:
            return rows
        means = [sum(row[j] for row in rows) / len(rows) for j in range(dims)]
        return [[row[j] - means[j] for j in range(dims)] for row in rows]

    xc, yc = centred(x), centred(y)
    cross = [[sum(xc[r][i] * yc[r][j] + yc[r][i] * xc[r][j] for r in range(len(xc)))
              for j in range(dims)] for i in range(dims)]
    return symmetric_eigh(cross)


def reliable_components(
    x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], *,
    flips: int = 500, alpha: float = SPLIT_HALF_ALPHA, seed: int = SPLIT_HALF_SEED,
) -> dict[str, Any]:
    """How many independent payoff dimensions a factor library reproduces across halves.

    GoEmotions keeps the principal preserved components whose held-out correlation is significant
    (`ppca.py:136-217`, via a leave-one-rater-out Spearman). Here the null flips the sign of whole
    block rows of one half, which destroys agreement between halves while keeping each half's own
    structure, and component ``i`` counts as reliable when its eigenvalue exceeds the
    ``1 - alpha`` quantile of the null's ``i``-th eigenvalue. Uncentred, for the reason
    :class:`SplitHalf` gives. The count, not the loadings, is what the lab reports: eight vetted
    primitives that reproduce two dimensions are two ideas, not eight.
    """
    observed, _ = ppca(x, y, demean=False)
    rng = random.Random(seed)
    nulls: list[list[float]] = []
    for _ in range(flips):
        signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in y]
        flipped = [[s * v for v in row] for s, row in zip(signs, y, strict=True)]
        nulls.append(ppca(x, flipped, demean=False)[0])
    thresholds: list[float] = []
    for i in range(len(observed)):
        ranked = sorted(null[i] for null in nulls)
        thresholds.append(ranked[min(len(ranked) - 1, math.ceil((1 - alpha) * len(ranked)) - 1)])
    reliable = 0
    for value, threshold in zip(observed, thresholds, strict=True):
        if value <= threshold:
            break
        reliable += 1
    return {
        "dimensions": len(observed),
        "reliable_components": reliable,
        "eigenvalues": [round(v, 10) for v in observed],
        "null_thresholds": [round(t, 10) for t in thresholds],
        "flips": flips,
    }


class Evaluator:
    """Deterministic scoring. No model touches this.

    Kept as a separate object from the proposer so the boundary is a real one: the evaluator
    receives a :class:`Factor` and returns numbers, and has no channel back to whatever produced it.
    """

    def __init__(self, bars: Sequence[Bar], *, cost: CostModel | None = None) -> None:
        self._bars = bars
        self._cost = cost or CostModel.bitget_perp()
        self._cost.assert_gateable()

    def observations(self, record: FactorRecord) -> list[Observation]:
        """Turn one factor into the (factor value, next-bar return) pairs the gates score.

        ARGUS's lab evaluates a factor on a **single** instrument's bar series, so the natural
        "cross-section at a period" does not exist. Rather than fabricate one, each bar becomes its
        own period with one observation, and the rank correlation is taken across a rolling window
        of bars instead of across names at an instant. That is a real difference from the
        cross-sectional IC in the literature and it is stated rather than glossed: the gates are
        measuring time-series predictive power here, not cross-sectional ranking power.
        """
        bars = self._bars
        signal = PRIMITIVES[record.factor.expression]
        out: list[Observation] = []
        window = max(2, record.factor.horizon_bars)
        for i in range(len(bars) - 1):
            try:
                value = float(signal(bars, i))
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                continue
            nxt = _ret(bars, i + 1, 1)
            # One period per window keeps enough readings inside a period for a rank correlation
            # to be defined at all; a period holding a single pair has no ranks to correlate.
            out.append(Observation(period=i // window, name=f"bar{i % window}",
                                   factor=value, forward_return=nxt))
        return out

    def overfit_report(self, record: FactorRecord) -> OverfitReport | None:
        """Run the four anti-overfit gates over this factor's own readings."""
        rows = self.observations(record)
        if not rows:
            return None
        return run_all(rows)

    def payoff_series(self, factor: Factor) -> tuple[list[float], list[float]]:
        """Factor readings and the next bar's return — exactly the pairs :meth:`observations`
        scores, so the split-half gate and the anti-overfit gates see the same evidence."""
        signal = PRIMITIVES[factor.expression]
        values: list[float] = []
        returns: list[float] = []
        for i in range(len(self._bars) - 1):
            try:
                value = float(signal(self._bars, i))
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                continue
            values.append(value)
            returns.append(_ret(self._bars, i + 1, 1))
        return values, returns

    def split_half(self, factor: Factor) -> SplitHalf:
        """The split-half reliability gate, one item per :data:`SPLIT_HALF_BLOCK_BARS` bars."""
        values, returns = self.payoff_series(factor)
        return split_half(values, returns, block=SPLIT_HALF_BLOCK_BARS)

    def score(self, record: FactorRecord) -> FactorRecord:
        """Walk one factor through every gate, in order, stopping at the first failure."""
        record.advance(Lifecycle.FORMALIZED)
        signal = PRIMITIVES[record.factor.expression]

        try:
            result = run(
                record.factor.name, "lab", self._bars, signal,
                cost=self._cost, periods_per_year=HOURLY_PER_YEAR,
            )
        except MetricError as exc:
            record.reject(f"unscoreable: {exc}")
            return record

        record.gross_sharpe = round(result.gross.sharpe, 3)
        record.net_sharpe = round(result.net.sharpe, 3)
        record.advance(Lifecycle.BACKTESTED)

        # Cost gate. On this venue the fee is larger than most effects, so this is where most
        # factors die — which is the honest outcome, not a tuning problem.
        if result.net.sharpe <= 0:
            record.advance(Lifecycle.COST_CHECKED)
            record.reject(
                f"net Sharpe {result.net.sharpe:.3f} after the 12bps round trip "
                f"(gross was {result.gross.sharpe:.3f})"
            )
            return record
        record.advance(Lifecycle.COST_CHECKED)

        if result.out_of_sample is None:
            record.reject("no scoreable out-of-sample slice")
            return record
        record.oos_sharpe = round(result.out_of_sample.sharpe, 3)
        record.advance(Lifecycle.OOS_TESTED)

        if result.out_of_sample.sharpe <= 0:
            record.reject(f"out-of-sample Sharpe {result.out_of_sample.sharpe:.3f}")
            return record

        return record


@dataclass
class FactorLab:
    """The loop. Proposals in, certified factors out, everything else in the cemetery."""

    evaluator: Evaluator
    records: list[FactorRecord] = field(default_factory=list)
    memory: Any | None = None
    """An optional :class:`argus.research.memory.FactorMemory`. Attached, the lab stops re-scoring
    hypotheses it has a record of, and the proposer is told what has been tried — never how any of
    it did. Typed loosely to keep the memory module's dependency one-directional."""

    @property
    def trials(self) -> int:
        """Every proposal ever scored. This is what the DSR gate consumes.

        A search that reports its best result without this number has not produced a result; it has
        reported the maximum of a noise distribution.
        """
        return len(self.records)

    def context(self) -> ProposerContext:
        """What the proposer gets. Carries no performance information by construction."""
        return ProposerContext(
            market_structure=(
                "Tokenized US equity perpetual. The token trades continuously; the underlying "
                "equity market closes. Measured: price discovery attenuates ~7x during weekends "
                "(32.17 -> 4.53 bps/hour). Round trip costs 12bps."
            ),
            already_proposed=tuple(r.factor.name for r in self.records),
            trials_so_far=self.trials,
            memory=() if self.memory is None else tuple(self.memory.signal().render()),
        )

    def submit(self, factor: Factor) -> FactorRecord | None:
        """Score one proposal. Returns ``None`` when memory has seen this hypothesis before.

        A repeat is suppressed rather than re-scored, and deliberately does **not** increment
        :attr:`trials`. The Deflated Sharpe gate consumes that count as the number of distinct looks
        taken; re-testing one hypothesis produces no new maximum, so counting it again would deflate
        against a search that never happened. The suppression is counted in the memory and surfaced
        by :meth:`funnel`, so a proposer going in circles is visible rather than merely cheap.
        """
        if self.memory is not None and self.memory.seen(factor):
            self.memory.note_duplicate()
            return None
        record = FactorRecord(factor=factor, trial_number=self.trials + 1)
        self.records.append(record)
        scored = self.evaluator.score(record)
        if self.memory is not None:
            self.memory.remember(scored)
        return scored

    def gate(self) -> dict[str, Any]:
        """Apply the Deflated Sharpe gate to survivors, using the real trial count."""
        survivors = [r for r in self.records if r.state is Lifecycle.OOS_TESTED]
        if not survivors:
            return {"certified": [], "note": "nothing reached the DSR gate"}

        sharpes = [r.net_sharpe or 0.0 for r in self.records if r.net_sharpe is not None]
        variance = 0.0
        if len(sharpes) > 1:
            mu = sum(sharpes) / len(sharpes)
            variance = sum((s - mu) ** 2 for s in sharpes) / (len(sharpes) - 1)

        certified = []
        for record in survivors:
            record.advance(Lifecycle.DSR_GATED)
            try:
                record.dsr = round(deflated_sharpe(
                    record.net_sharpe or 0.0, n=len(self.evaluator._bars),
                    trials=self.trials, variance_of_trials=variance,
                ), 4)
            except MetricError as exc:
                record.reject(f"DSR uncomputable: {exc}")
                continue

            if record.dsr <= 0.95:
                record.reject(
                    f"DSR {record.dsr} over {self.trials} trials — not distinguishable from "
                    f"the best of a random search"
                )
                continue

            # The Deflated Sharpe asks whether this result beats the best of a random *search*.
            # The anti-overfit gates ask a different question — whether the factor beats its own
            # shuffled self, holds its sign across sub-periods and regimes, and decays like a real
            # signal. A factor can clear the first and fail the second, so both are required before
            # anything is called certified. An INCONCLUSIVE verdict does not certify either:
            # "we could not tell" is not "it passed".
            overfit = self.evaluator.overfit_report(record)
            if overfit is not None:
                record.overfit = overfit.as_dict()
                if overfit.failed or overfit.inconclusive:
                    record.reject(f"anti-overfit: {overfit.verdict}")
                    continue

            # Reliability, a third question: would an independent half of the same evidence have
            # found this? INCONCLUSIVE does not certify, for the reason given above.
            reliability = self.evaluator.split_half(record.factor)
            record.split_half = reliability.as_dict()
            if not reliability.passed:
                record.reject(f"split-half: {reliability.reason}")
                continue

            record.advance(Lifecycle.CERTIFIED)
            certified.append(record)
        if self.memory is not None:
            # The gate is where a factor reaches its terminal state. Memory written at score time
            # records every survivor as oos_tested, so it is refreshed here or it is wrong.
            for record in self.records:
                self.memory.update(record)
        return {
            "trials": self.trials,
            "variance_of_trial_sharpes": round(variance, 4),
            "certified": [r.name for r in (x.factor for x in certified)],
        }

    def cemetery(self) -> list[dict[str, str]]:
        """Everything that died, and what killed it.

        Retained deliberately. A library that only ever grows is publication bias inside our own
        system, and the rejection reasons are the most informative output this lab produces.
        """
        return [
            {"name": r.factor.name, "died_at": str(r.state), "reason": r.rejection_reason}
            for r in self.records if r.state is Lifecycle.REJECTED
        ]

    def funnel(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[str(r.state)] = counts.get(str(r.state), 0) + 1
        return {
            "proposed": self.trials,
            "by_final_state": counts,
            "certified": counts.get(str(Lifecycle.CERTIFIED), 0),
            "rejected": counts.get(str(Lifecycle.REJECTED), 0),
            # Repeats are suppressed rather than scored, so they do not appear above. Reporting the
            # number keeps a proposer that is going in circles visible instead of merely cheap.
            "duplicates_suppressed": (
                0 if self.memory is None else self.memory.duplicates_suppressed
            ),
        }


def report(lab: FactorLab) -> dict[str, Any]:
    """Run the gate, then describe the lab.

    **The gate is called first, deliberately.** It was second here, and since a dict literal is
    evaluated top to bottom the funnel was computed before any factor reached its terminal state —
    the published `data/factor_lab.json` recorded ``by_final_state: {oos_tested: 1}`` and
    ``certified: 0`` for a factor the gate had not yet judged. Nothing certified at the time, so the
    two numbers happened to agree and the defect stayed invisible; the first certification would
    have produced a report whose funnel contradicted its own gate.
    """
    gate = lab.gate()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "separation": (
            "the proposer receives ProposerContext, which has no field for performance; "
            "evaluation is deterministic Python and no model grades a factor"
        ),
        "funnel": lab.funnel(),
        "gate": gate,
        "factors": [r.as_dict() for r in lab.records],
        "cemetery": lab.cemetery(),
    }


def main() -> int:
    from argus.market.history import CandleType, fetch_range
    from argus.research.memory import FactorMemory

    candles = fetch_range("NVDAUSDT", days=90, interval="1H", candle_type=CandleType.MARKET)
    bars = [
        Bar(
            ts=c.ts,
            close=c.close,
            # Same reason as `track1_study`: a grammar field with no data behind it reads 0.0 for
            # every bar, and the search cannot tell that apart from a real constant.
            extra={"volume": float(c.volume), "high": float(c.high), "low": float(c.low)},
        )
        for c in candles
    ]

    root = Path(__file__).resolve().parents[3] / "data"
    memory_path = root / "factor_memory.json"
    memory = FactorMemory.load(memory_path)
    lab = FactorLab(evaluator=Evaluator(bars), memory=memory)

    # Stand-ins for model proposals: every vetted primitive, submitted blind. The lab behaves
    # identically whether these come from a model or a list — which is the point of the boundary.
    # On the second run every one of these is already in memory, so the lab suppresses them all
    # rather than re-scoring them; that is the loop working, not a failure.
    for name in PRIMITIVES:
        lab.submit(Factor(name=name, expression=name, rationale="session-structure hypothesis"))

    out = root / "factor_lab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report(lab)
    memory.save(memory_path)

    f = payload["funnel"]
    print(f"proposed {f['proposed']} · certified {f['certified']} · rejected {f['rejected']}")
    print(f"suppressed as already evaluated: {f['duplicates_suppressed']}")
    print(f"trials fed to the DSR gate: {payload['gate'].get('trials')}")

    # A run in which every proposal was already known has produced no new evidence, and writing its
    # empty funnel over the previous report would destroy a real result to record that nothing
    # happened. Found by running this twice: the second pass overwrote eight scored factors with
    # zeroes. The memory is still saved — it is the thing that legitimately changed.
    if not lab.records:
        print(f"\nevery proposal was already evaluated; {out.name} left as it was")
        return 0

    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\ncemetery:")
    for row in payload["cemetery"]:
        print(f"  {row['name']:<22} died at {row['died_at']:<14} {row['reason'][:70]}")
    print(f"\nfull report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PRIMITIVES", "SPLIT_HALF_ALPHA", "SPLIT_HALF_BLOCK_BARS", "SPLIT_HALF_FLIPS",
    "SPLIT_HALF_MIN_BLOCKS", "Evaluator", "Factor", "FactorLab", "FactorRecord", "Lifecycle",
    "LifecycleViolation", "ProposerContext", "SplitHalf", "payoff_halves", "ppca",
    "reliable_components", "report", "split_half", "symmetric_eigh",
]
