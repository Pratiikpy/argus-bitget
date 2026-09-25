"""Does the desk abstain on the right calls? Risk-coverage scoring of abstention as a decision.

`eval/observatory.py:abstention_quality` answers one question about a record of refusals: summed
over them, did standing aside avoid more loss than it missed gain? That is a necessary number and
not a sufficient one, because it never reads the desk's confidence. A gate that refused every call
at random on a tape where most would-be trades lose after fees earns exactly the same
`abstention_quality` as a gate that refused precisely the losers — the function is blind to which
refusals were made, only to how they summed.

**The general function this is.** Outside trading, "score a system that may decline to act" is
*selective prediction* (Geifman & El-Yaniv, NeurIPS 2017; Traub et al., "Overcoming Common Flaws in
the Evaluation of Selective Classification Systems", NeurIPS 2024). Every case carries a prediction
and a confidence, the system acts on the most confident and abstains on the rest, and the whole
abstention *policy* is scored by the risk-coverage (RC) curve: the loss rate of what was acted on,
at every coverage the confidence threshold could have produced. Its area (AURC), its
coverage-weighted area (AUGRC), and the distance of each from a random ranking and from a perfect
one say whether the confidence that drives the abstention carries information at all. That is the
question a trading desk that has refused every trade most needs answered, and it is the question
`abstention_quality` cannot ask.

For this desk the mapping is exact. Every decision — refusals included — records a **lean**, the
direction it would take if forced, and a ``lean_confidence``; every settled refusal records the move
that followed (`paper/ledger.py:Entry.counterfactual_move_bps`). So each refusal is a withheld
prediction whose outcome is known: acting on the lean would have netted
``direction * move - round_trip`` basis points. A refusal whose lean is ``none`` made no prediction
and is counted, not scored — the same rule `eval/shadow.py` applies.

**What was taken, from where** (both Apache-2.0; the licence permits adaptation with attribution):

* **fd-shifts** (`IML-DKFZ/fd-shifts` @ c4467aec, the reference code of Traub et al. 2024).
  - Tie-aware RC points: a point is emitted only where the confidence changes, so every case tied
    at one confidence is accepted or refused together
    (`fd_shifts/analysis/rc_stats_utils.py:35-50` selective risk, `:91-104` generalized risk).
  - Linear-trapezoid AUC over those points (`fd_shifts/analysis/rc_stats.py:434-438`).
  - Closed-form optimal AURC and AUGRC for binary residuals (`rc_stats.py:256-261`, `:287-288`),
    which give the excess-over-optimal (e-AURC) reference.
  - Working-point selection — the most coverage whose risk stays under a target
    (`rc_stats.py:576-621`), except that where no point meets the target this returns ``None``;
    theirs raises ``ValueError`` from ``np.argmax`` on an empty mask (`rc_stats.py:610-612`).
  Parity with their real, unmodified code on the same input is pinned in
  `tests/test_general_abstention_comparison.py` (to 1e-12 on every quantity above).
* **torch-uncertainty** (`torch-uncertainty/torch-uncertainty` @ 3f82fe5d) was run on the same input
  and its ranking was **not** taken: `_aurc_rejection_rate_compute`
  (`src/torch_uncertainty/metrics/classification/risk_coverage.py:190-203`) orders cases with a
  plain ``argsort``, so cases tied at one confidence are split by row order, and the reported AURC
  moves when the rows are merely reordered. This desk's confidences are coarse — four leans in
  five sit at 0.55, 0.58 or 0.62 — so on this record that is not a corner case; the measured
  spread is in `eval/general_abstention_comparison.py`.

**What ARGUS adds, and why each departs from the reference.**

* **A value curve in basis points, beside the risk curve.** Selective-classification residuals are
  a loss in ``[0, 1]`` (`rc_stats.py:636-642`); a +500bps trade and a +1bps trade are the same
  zero. A desk is paid in basis points, so each RC point also carries the mean and total **net**
  bps of the calls acted on — fee included — and the abstention verdict is read off that, not off
  the error rate alone.
* **A cluster bootstrap, not an iid one.** fd-shifts resamples cases independently
  (`rc_stats.py:538-574`). Calls decided on one day share one market move — the counterfactual
  horizon is about a day, so the overlap is worse than the 2h cycle spacing suggests — and an iid
  bootstrap would report precision the record does not have. Whole days are resampled, the same
  choice `eval/refusal.py:cluster_interval` makes for cycles.
* **An out-of-sample check on the threshold.** Choosing the best threshold from an RC curve and
  quoting its value is selection on the outcome. The chronological first half picks the threshold;
  the second half says what it would have earned.

Pure Python, like the rest of this package: no numpy at runtime.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isinf, log
from typing import Any

ROUND_TRIP_BPS = 12.0
"""Bitget taker fee on both legs — the same hurdle `eval/scorecard.py` and `eval/refusal.py` use."""

MIN_CALLS = 30
"""Fewest scored leans before a curve is summarised; below this the ranking is noise."""

MIN_ACTED = 10
"""Fewest calls a threshold must act on before it may be named the best one.

Without this floor the "best threshold in hindsight" on the live record is a cut-off at 0.78 that
acts on exactly one call, which happened to win +84bps — a coin flip reported as a policy. fd-shifts
has no such floor because a selective classifier's working point is chosen by a target risk, not by
the maximum of a noisy value curve; here the value curve is maximised, so the floor is needed.
"""

BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20260925
"""Fixed so the published interval reproduces to the digit from the same record."""


class CoverageError(ValueError):
    """Raised rather than scoring a curve with nothing behind it."""


# =============================================================================================
# The scored case.
# =============================================================================================


@dataclass(frozen=True, slots=True)
class LeanCall:
    """One refusal that made a prediction: the lean it withheld, and what it would have made."""

    seq: int
    decided_at: str
    symbol: str
    lean: str
    confidence: float
    move_bps: float
    stated_confidence: float = 0.0
    round_trip_bps: float = ROUND_TRIP_BPS

    @property
    def direction(self) -> int:
        return 1 if self.lean == "up" else -1

    @property
    def net_bps(self) -> float:
        """What acting on the lean would have netted, after the round trip it never paid."""
        return self.direction * self.move_bps - self.round_trip_bps

    @property
    def error(self) -> int:
        """1 if acting on the lean would have lost (or only broken even) after fees."""
        return 1 if self.net_bps <= 0 else 0

    @property
    def cluster(self) -> str:
        """The UTC day it was decided — calls on one day share one market move."""
        return self.decided_at[:10]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "decided_at": self.decided_at, "symbol": self.symbol,
            "lean": self.lean, "confidence": self.confidence,
            "stated_confidence": self.stated_confidence, "move_bps": self.move_bps,
        }


@dataclass(frozen=True, slots=True)
class Extraction:
    calls: tuple[LeanCall, ...]
    no_lean: int
    """Settled refusals whose lean was ``none``: no prediction was made, so none is scored."""
    unsettled: int


def calls_from_entries(
    entries: Iterable[Any], *, upto_seq: int | None = None, settled_by: str | None = None,
    round_trip_bps: float = ROUND_TRIP_BPS, counterfactuals: Mapping[int, Any] | None = None,
) -> Extraction:
    """Every settled refusal on a ledger, as scoreable :class:`LeanCall` rows.

    ``upto_seq`` and ``settled_by`` (ISO-8601) freeze the input: a settlement is written once and
    sealed (`paper/ledger.py:settle_abstention`), so the same pair selects the same rows forever.
    ``counterfactuals`` overrides the recorded move by sequence number, exactly as
    `eval/scorecard.py:score_ledger` does, so the two agree on which rows are scored.
    """
    cutoff = datetime.fromisoformat(settled_by) if settled_by else None
    overrides = counterfactuals or {}
    calls: list[LeanCall] = []
    no_lean = 0
    unsettled = 0
    for entry in entries:
        if not entry.is_abstention:
            continue
        if upto_seq is not None and entry.seq > upto_seq:
            continue
        if entry.seq in overrides:
            move = overrides[entry.seq]
        else:
            move = entry.counterfactual_move_bps
            if move is None or entry.settled_at is None:
                unsettled += 1
                continue
            if cutoff is not None and datetime.fromisoformat(entry.settled_at) > cutoff:
                unsettled += 1
                continue
        lean = str(getattr(entry, "lean", "none") or "none").lower()
        if lean not in ("up", "down"):
            no_lean += 1
            continue
        calls.append(LeanCall(
            seq=int(entry.seq), decided_at=str(entry.decided_at), symbol=str(entry.symbol),
            lean=lean, confidence=float(getattr(entry, "lean_confidence", 0.0)),
            move_bps=float(move), stated_confidence=float(entry.stated_confidence),
            round_trip_bps=round_trip_bps,
        ))
    return Extraction(calls=tuple(calls), no_lean=no_lean, unsettled=unsettled)


# =============================================================================================
# Risk-coverage points — tie-aware, after fd-shifts.
# =============================================================================================


@dataclass(frozen=True, slots=True)
class RCPoint:
    """Act on every call with confidence >= ``threshold``; refuse the rest."""

    threshold: float
    coverage: float
    acted: int
    selective_risk: float
    """Share of acted calls that would have lost after fees."""
    generalized_risk: float
    """Losing acted calls over *all* calls — Traub et al.'s risk, the AUGRC integrand."""
    mean_net_bps: float
    total_net_bps: float

    def as_dict(self) -> dict[str, Any]:
        """``threshold`` is ``None`` on the coverage-0 point: every call refused, no cut-off."""
        return {
            "threshold": None if isinf(self.threshold) else self.threshold,
            "coverage": self.coverage, "acted": self.acted,
            "selective_risk": self.selective_risk, "generalized_risk": self.generalized_risk,
            "mean_net_bps": self.mean_net_bps, "total_net_bps": self.total_net_bps,
        }


def rc_points(confidences: Sequence[float], errors: Sequence[float],
              nets: Sequence[float] | None = None) -> list[RCPoint]:
    """The RC curve from full coverage down to the last distinct confidence, then coverage 0.

    Same traversal as fd-shifts' ``selective_risk_stats``/``generalized_risk_stats``
    (`rc_stats_utils.py:18-61`, `:75-115`): sort ascending, peel off the least confident case, and
    emit a point only where the next confidence differs — so ties are always accepted or refused
    together and row order cannot change the curve. The final coverage-0 point repeats the last
    selective risk and carries zero generalized risk and zero value, exactly as theirs does.
    """
    n = len(confidences)
    if n == 0 or len(errors) != n or (nets is not None and len(nets) != n):
        raise CoverageError("rc_points needs equal-length, non-empty inputs")
    values = list(nets) if nets is not None else [0.0] * n
    order = sorted(range(n), key=lambda i: confidences[i])
    conf = [confidences[i] for i in order]
    err = [float(errors[i]) for i in order]
    val = [float(values[i]) for i in order]

    acted = n
    err_sum = sum(err)
    val_sum = sum(val)
    points = [RCPoint(threshold=conf[0], coverage=1.0, acted=n, selective_risk=err_sum / n,
                      generalized_risk=err_sum / n, mean_net_bps=val_sum / n,
                      total_net_bps=val_sum)]
    for i in range(n - 1):
        acted -= 1
        err_sum -= err[i]
        val_sum -= val[i]
        if conf[i] != conf[i + 1]:
            points.append(RCPoint(
                threshold=conf[i + 1], coverage=acted / n, acted=acted,
                selective_risk=err_sum / acted, generalized_risk=err_sum / n,
                mean_net_bps=val_sum / acted, total_net_bps=val_sum,
            ))
    points.append(RCPoint(threshold=float("inf"), coverage=0.0, acted=0,
                          selective_risk=points[-1].selective_risk, generalized_risk=0.0,
                          mean_net_bps=0.0, total_net_bps=0.0))
    return points


def _trapezoid_desc(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Area under ``ys`` over descending ``xs`` — ``-np.trapz(ys, xs)`` (`rc_stats.py:438`)."""
    return sum((xs[i] - xs[i + 1]) * (ys[i] + ys[i + 1]) / 2.0 for i in range(len(xs) - 1))


def aurc(points: Sequence[RCPoint]) -> float:
    return _trapezoid_desc([p.coverage for p in points], [p.selective_risk for p in points])


def augrc(points: Sequence[RCPoint]) -> float:
    return _trapezoid_desc([p.coverage for p in points], [p.generalized_risk for p in points])


def aurc_optimal(error_rate: float) -> float:
    """AURC of a perfect ranking for binary errors (`rc_stats.py:256-261`, without their x1000).

    fd-shifts adds machine epsilon inside the log so an error rate of exactly 1 does not take
    ``log(0)``; the same guard is kept so the two agree to the last digit.
    """
    eps = 2.220446049250313e-16
    return error_rate + (1.0 - error_rate) * log(1.0 - error_rate + eps)


def augrc_optimal(error_rate: float) -> float:
    """AUGRC of a perfect ranking for binary errors (`rc_stats.py:287-288`, without x1000)."""
    return 0.5 * error_rate ** 2


def working_point(points: Sequence[RCPoint], *, target_risk: float) -> RCPoint | None:
    """The most coverage whose selective risk stays at or under ``target_risk``.

    fd-shifts' ``get_working_point`` (`rc_stats.py:576-621`) restricts to points where the risk
    reaches a new minimum; the same mask is applied, so the chosen point is one no lower-coverage
    point dominates. ``None`` when no threshold meets the target.
    """
    best: RCPoint | None = None
    running_min = float("inf")
    for p in points[:-1]:  # the coverage-0 point is not a working point (`rc_stats_utils.py:54`)
        if p.selective_risk < running_min:
            running_min = p.selective_risk
            if p.selective_risk <= target_risk and (best is None or p.coverage > best.coverage):
                best = p
    return best


# =============================================================================================
# The abstention report.
# =============================================================================================


def _curve_stats(calls: Sequence[LeanCall], confidences: Sequence[float]) -> dict[str, Any]:
    errors = [c.error for c in calls]
    nets = [c.net_bps for c in calls]
    points = rc_points(confidences, errors, nets)
    err_rate = sum(errors) / len(errors)
    a, g = aurc(points), augrc(points)
    eligible = [p for p in points if p.acted == 0 or p.acted >= MIN_ACTED]
    best = max(eligible, key=lambda p: (p.total_net_bps, -p.coverage))
    return {
        "points": points,
        "aurc": a,
        "augrc": g,
        "aurc_random": err_rate,
        "augrc_random": err_rate / 2.0,
        "aurc_optimal": aurc_optimal(err_rate),
        "augrc_optimal": augrc_optimal(err_rate),
        "error_rate": err_rate,
        "gate_skill": err_rate - a,
        "best": best,
    }


def _cluster_bootstrap(
    calls: Sequence[LeanCall], confidence_of: Any, *, resamples: int, seed: int,
) -> dict[str, tuple[float, float] | None]:
    """95% percentile intervals, resampling whole days of calls."""
    by_day: dict[str, list[LeanCall]] = defaultdict(list)
    for c in calls:
        by_day[c.cluster].append(c)
    days = sorted(by_day)
    if len(days) < 2:
        return {"gate_skill": None, "value_of_abstaining_bps": None, "error_rate": None,
                "aurc": None}
    rng = random.Random(seed)
    skills: list[float] = []
    values: list[float] = []
    rates: list[float] = []
    aurcs: list[float] = []
    for _ in range(resamples):
        draw = [c for _ in days for c in by_day[days[rng.randrange(len(days))]]]
        stats = _curve_stats(draw, [confidence_of(c) for c in draw])
        skills.append(stats["gate_skill"])
        values.append(-sum(c.net_bps for c in draw))
        rates.append(stats["error_rate"])
        aurcs.append(stats["aurc"])

    def interval(xs: list[float]) -> tuple[float, float]:
        xs.sort()
        return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]

    return {"gate_skill": interval(skills), "value_of_abstaining_bps": interval(values),
            "error_rate": interval(rates), "aurc": interval(aurcs)}


def _out_of_sample(calls: Sequence[LeanCall], confidence_of: Any) -> dict[str, Any]:
    """Pick the threshold on the earlier half; score it on the later half."""
    ordered = sorted(calls, key=lambda c: (c.decided_at, c.seq))
    half = len(ordered) // 2
    early, late = ordered[:half], ordered[half:]
    if len(early) < MIN_CALLS // 2 or len(late) < MIN_CALLS // 2:
        return {"gradeable": False, "reason": "too few calls in one half to pick and test"}
    chosen = _curve_stats(early, [confidence_of(c) for c in early])["best"]
    acted = [c for c in late if confidence_of(c) >= chosen.threshold]
    realised = sum(c.net_bps for c in acted)
    return {
        "gradeable": True,
        "split_at": late[0].decided_at,
        "chosen_threshold": None if chosen.coverage == 0.0 else chosen.threshold,
        "chosen_coverage_in_sample": chosen.coverage,
        "in_sample_total_net_bps": chosen.total_net_bps,
        "held_out_acted": len(acted),
        "held_out_total_net_bps": realised,
        "held_out_beats_abstaining": realised > 0,
    }


def score(
    extraction: Extraction, *, ranking: str = "lean_confidence",
    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The risk-coverage report on a record of refusals.

    ``ranking`` picks the confidence that orders the calls: ``"lean_confidence"`` (conviction in
    the direction it withheld — the selective predictor's own confidence, and the primary one) or
    ``"abstention_doubt"`` (``1 - stated_confidence``: the refusals the desk was least sure of are
    the ones it came closest to acting on).
    """
    calls = extraction.calls
    if len(calls) < MIN_CALLS:
        raise CoverageError(
            f"{len(calls)} scored lean(s) — below the floor of {MIN_CALLS}; a ranking over fewer "
            f"is noise"
        )
    if ranking == "lean_confidence":
        def confidence_of(c: LeanCall) -> float:
            return c.confidence
    elif ranking == "abstention_doubt":
        def confidence_of(c: LeanCall) -> float:
            return 1.0 - c.stated_confidence
    else:
        raise CoverageError(f"unknown ranking {ranking!r}")

    stats = _curve_stats(calls, [confidence_of(c) for c in calls])
    points: list[RCPoint] = stats["points"]
    best: RCPoint = stats["best"]
    full = points[0]
    ci = _cluster_bootstrap(calls, confidence_of, resamples=resamples, seed=seed)
    oos = _out_of_sample(calls, confidence_of)
    skill_ci = ci["gate_skill"]
    wp = working_point(points, target_risk=0.5)

    if best.coverage == 0.0:
        verdict = (
            "abstaining on every call was the best threshold in hindsight: no confidence cut-off "
            "would have netted a positive total after fees"
        )
    else:
        verdict = (
            f"acting on leans at confidence >= {best.threshold:.2f} ({best.acted} call(s)) would "
            f"have netted {best.total_net_bps:+.0f}bps in hindsight — an in-sample pick; see "
            f"out_of_sample for whether the threshold holds"
        )
    if skill_ci is None:
        skill = "undefined: fewer than two days of calls"
    elif skill_ci[0] > 0:
        skill = "the confidence ranks losing calls below winning ones (interval excludes zero)"
    elif skill_ci[1] < 0:
        skill = "the confidence ranks losing calls ABOVE winning ones (interval excludes zero)"
    else:
        skill = "no detectable ranking skill: the interval on AURC_random - AURC contains zero"

    return {
        "ranking": ranking,
        "scored_calls": len(calls),
        "no_lean": extraction.no_lean,
        "unsettled": extraction.unsettled,
        "days": len({c.cluster for c in calls}),
        "error_rate": stats["error_rate"],
        "aurc": stats["aurc"],
        "aurc_ci95_by_day": ci["aurc"],
        "augrc": stats["augrc"],
        "aurc_random": stats["aurc_random"],
        "augrc_random": stats["augrc_random"],
        "aurc_optimal": stats["aurc_optimal"],
        "augrc_optimal": stats["augrc_optimal"],
        "e_aurc": stats["aurc"] - stats["aurc_optimal"],
        "gate_skill": stats["gate_skill"],
        "gate_skill_ci95_by_day": skill_ci,
        "gate_skill_verdict": skill,
        "value_of_abstaining_bps": -full.total_net_bps,
        "value_of_abstaining_ci95_by_day": ci["value_of_abstaining_bps"],
        "error_rate_ci95_by_day": ci["error_rate"],
        "best_threshold_in_sample": best.as_dict(),
        "abstention_verdict": verdict,
        "working_point_at_50pct_risk": None if wp is None else wp.as_dict(),
        "out_of_sample": oos,
        "curve": [p.as_dict() for p in points],
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "MIN_ACTED",
    "MIN_CALLS",
    "ROUND_TRIP_BPS",
    "CoverageError",
    "Extraction",
    "LeanCall",
    "RCPoint",
    "augrc",
    "augrc_optimal",
    "aurc",
    "aurc_optimal",
    "calls_from_entries",
    "rc_points",
    "score",
    "working_point",
]
