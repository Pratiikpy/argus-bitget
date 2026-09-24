"""Our forecast record, restated in the metrics public forecasting benchmarks actually use.

`eval.forecasts` produces the record: 49,140 hourly probability forecasts over twelve rTokens,
scored against whether the move cleared the cost hurdle, with the clustering correction that turns
the raw count into an effective one. It reports a Brier score and an expected calibration error.
Those are the right two numbers and they are not the ones a reader can put beside anything else.

This module restates the same record in the metric set the public benchmarks report, so the
question "is this any good?" has an answer that does not depend on trusting us:

* **Brier Index**, ``100 * (1 - sqrt(BS))``, the headline ForecastBench moved to. Verified against
  forecastbench.org, which describes it as "an interpretable 0-100% scale where higher is better
  (100% = perfect accuracy, 50% = maximally uninformed, 0% = maximally wrong)" — the three
  endpoints that pin the formula, since BS 0.25 maps to 50, BS 0 to 100 and BS 1 to 0.
* **Brier Skill Score** against three reference forecasts rather than one, because which baseline
  is used changes the answer and quoting the flattering one is the oldest trick in forecasting.
* **The Murphy (1973) decomposition** into reliability, resolution and uncertainty. This is the
  one that earns its place: a Brier score says how wrong we were, and the decomposition says
  whether that came from miscalibration or from failing to discriminate, which are different
  defects with different fixes.
* **Expected and maximum calibration error** on the standard ten-bin convention.

**The decomposition is also a self-test that cannot be faked.** Murphy's identity is
``BS = REL - RES + UNC`` exactly, so an implementation that gets any of the three wrong fails to
reconstruct the Brier score it was computed from. :attr:`Restatement.identity_residual` reports
that residual, and the test suite asserts it is at floating-point zero. A calibration module that
cannot rebuild its own input is not measuring calibration.

**Where we are ahead of the benchmarks, and it is worth saying plainly.** None of ForecastBench,
FinBench or Foresight Arena documents a correction for non-independent forecasts. Ours are
profoundly non-independent — twelve symbols scored on the same hour share one market — and
`eval.forecasts` already measures that with an intraclass correlation and a Kish design effect.
Every figure here is therefore reported twice, at the raw count and at the effective count, and the
effective one is the one to believe.

**What is NOT established here.** No score from another system is quoted in this module, and none
should be until it is read from the source rather than from a summary. The ForecastBench leaderboard
table did not render when fetched on 2026-09-13, so the comparison to named models is **NOT
VERIFIED** and is deliberately absent. Restating our own record in a comparable metric is a
precondition for that comparison, not the comparison itself. Separately, the benchmarks score
open-ended world questions and this record scores one narrow mechanical question — whether a move
clears a cost hurdle — so even a verified leaderboard number would not be a like-for-like contest,
and :attr:`Restatement.comparability` says so in the output rather than in a footnote.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "forecastbench_restatement.json"

BINS = 10
"""The ten equal-width bins that ECE, MCE and the Murphy decomposition conventionally use.

Not a free parameter to tune. Calibration error falls monotonically as bins are widened — one bin
makes every forecaster perfectly calibrated — so the count has to be fixed by convention before the
number is computed, and stated whenever it is quoted.
"""

COIN_FLIP = 0.5


class RestatementError(ValueError):
    """Raised instead of returning a metric computed from nothing."""


@dataclass(frozen=True, slots=True)
class Bin:
    """One calibration bin: what was promised, what was delivered, how often."""

    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_rate: float
    within_term: float = 0.0
    """``sum_i (p_i - p_bar)(p_i - p_bar - 2*o_i)`` over the bin's own forecasts.

    The term the textbook three-way decomposition drops, and it is only zero when every forecast in
    the bin carries the identical probability. Murphy's original statement is over a forecaster who
    issues finitely many distinct values, one bin per value; bin a continuum into ten equal-width
    buckets and the deviations inside each bucket do not vanish. Carried here so the identity
    closes on any input rather than only on the discrete case — see :func:`murphy`.
    """

    @property
    def gap(self) -> float:
        return abs(self.observed_rate - self.mean_probability)

    def as_dict(self) -> dict[str, Any]:
        return {
            "range": [round(self.lower, 3), round(self.upper, 3)],
            "count": self.count,
            "mean_probability": round(self.mean_probability, 5),
            "observed_rate": round(self.observed_rate, 5),
            "gap": round(self.gap, 5),
            "within_term": round(self.within_term, 8),
        }


def bin_forecasts(
    probabilities: Sequence[float], outcomes: Sequence[bool], *, bins: int = BINS
) -> list[Bin]:
    """Equal-width bins over [0, 1], with 1.0 falling in the top bin rather than off the end.

    Empty bins are dropped. Keeping them would contribute nothing to any of the three statistics
    (each is weighted by bin count) while inviting a reader to divide by a bin count that includes
    bins nothing landed in.
    """
    if len(probabilities) != len(outcomes):
        raise RestatementError("every probability needs exactly one outcome")
    if bins < 1:
        raise RestatementError("at least one bin is needed")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, o in zip(probabilities, outcomes, strict=True):
        if not isfinite(p) or p < 0.0 or p > 1.0:
            raise RestatementError(f"probability {p} is not in [0, 1]")
        index = min(bins - 1, int(p * bins))
        buckets[index].append((p, o))
    out: list[Bin] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        out.append(
            Bin(
                lower=index / bins,
                upper=(index + 1) / bins,
                count=len(bucket),
                mean_probability=mean_p,
                observed_rate=sum(1 for _, o in bucket if o) / len(bucket),
                within_term=sum(
                    (p - mean_p) * (p - mean_p - 2.0 * (1.0 if o else 0.0)) for p, o in bucket
                ),
            )
        )
    return out


def brier(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error of the probability against the outcome.

    Lower is better, and 0.25 is what a coin scores.
    """
    if not probabilities:
        raise RestatementError("the Brier score of an empty record is not zero, it is undefined")
    if len(probabilities) != len(outcomes):
        raise RestatementError("every probability needs exactly one outcome")
    return sum(
        (p - (1.0 if o else 0.0)) ** 2
        for p, o in zip(probabilities, outcomes, strict=True)
    ) / len(probabilities)


def brier_index(brier_score: float) -> float:
    """``100 * (1 - sqrt(BS))`` — ForecastBench's headline, on a 0-100 scale where higher is better.

    Verified against forecastbench.org's own description of the scale on 2026-09-13: 100% perfect,
    50% maximally uninformed, 0% maximally wrong. Those three endpoints determine the formula, and
    a Brier score of 0.25 mapping to exactly 50 is the check that it is this transform and not
    another monotone one.
    """
    if brier_score < 0.0 or brier_score > 1.0:
        raise RestatementError(f"a Brier score of {brier_score} is outside [0, 1]")
    return 100.0 * (1.0 - sqrt(brier_score))


def brier_skill_score(brier_score: float, reference: float) -> float:
    """``1 - BS/BS_ref``. Positive beats the reference, zero ties it, negative loses to it."""
    if reference <= 0:
        raise RestatementError(
            "a reference forecast with a Brier score of zero is already perfect; a skill score "
            "against it is a division by zero, not an infinite skill"
        )
    return 1.0 - brier_score / reference


def murphy(bins: Sequence[Bin], base_rate: float) -> tuple[float, float, float, float]:
    """Reliability, resolution, uncertainty and the within-bin term.

    **The textbook identity is ``BS = REL - RES + UNC``, and as usually implemented it is false.**
    Murphy (1973) states it for a forecaster issuing finitely many distinct probabilities, with one
    bin per distinct value. Bin a continuum of probabilities into ten equal-width buckets — which
    is what every implementation of ECE does, and what these benchmarks do — and the deviations of
    each forecast from its own bin mean no longer vanish. The residual is
    ``(1/N) * sum_k sum_i (p_i - p_bar_k)(p_i - p_bar_k - 2*o_i)``, and it is returned as a fourth
    component so that ``BS = REL - RES + UNC + WITHIN`` closes exactly on **any** input.

    This was not a design choice made up front. The three-term version was written first and the
    self-test caught it: the residual was zero on a record of six discrete probabilities and
    -0.0044 once continuous noise was added, and -0.0186 at two bins. An implementation whose
    identity holds only on the inputs it was tested against is the failure mode this module's
    self-test exists to prevent.

    * **Reliability** — the count-weighted mean squared gap between what a bin promised and what it
      delivered. Zero when the forecaster is perfectly calibrated. Lower is better.
    * **Resolution** — how far the bins' observed rates spread away from the base rate. This is
      discrimination: a forecaster who says the base rate every time is perfectly calibrated and
      completely useless, and resolution is the term that catches it. Higher is better.
    * **Uncertainty** — ``o(1-o)``, a property of the questions and not of the forecaster. It is
      the Brier score a climatological forecast achieves, and it is the ceiling on what skill can
      be demonstrated on this question set.
    * **Within-bin** — the binning artefact above. It is not a property of the forecaster at all,
      and it goes to zero as the bins narrow around distinct forecast values. Reported so that it
      can be seen to be small rather than assumed to be zero.
    """
    total = sum(b.count for b in bins)
    if not total:
        raise RestatementError("no forecasts to decompose")
    reliability = sum(b.count * (b.mean_probability - b.observed_rate) ** 2 for b in bins) / total
    resolution = sum(b.count * (b.observed_rate - base_rate) ** 2 for b in bins) / total
    uncertainty = base_rate * (1.0 - base_rate)
    within = sum(b.within_term for b in bins) / total
    return reliability, resolution, uncertainty, within


def expected_calibration_error(bins: Sequence[Bin]) -> float:
    total = sum(b.count for b in bins)
    if not total:
        raise RestatementError("no forecasts to calibrate")
    return sum(b.count * b.gap for b in bins) / total


def maximum_calibration_error(bins: Sequence[Bin]) -> float:
    """The worst bin, not the average. Always at least the expected calibration error."""
    if not bins:
        raise RestatementError("no forecasts to calibrate")
    return max(b.gap for b in bins)


@dataclass(frozen=True, slots=True)
class Restatement:
    """One forecast record, in the metrics the public benchmarks report."""

    count: int
    effective_count: int
    distinct_instants: int
    icc: float
    base_rate: float
    brier: float
    bins: tuple[Bin, ...]
    persistence_brier: float | None
    """Brier of the momentum reference: the previous instant's outcome, carried forward.

    FinBench scores against a momentum heuristic as well as a coin flip and a base rate, and it is
    the reference that most often embarrasses a model — beating a coin is easy, beating "assume it
    keeps doing what it just did" is the real bar. None when the record is too short to form one.
    """

    @property
    def brier_index(self) -> float:
        return brier_index(self.brier)

    @property
    def climatology_brier(self) -> float:
        """The Brier a forecaster achieves by stating the base rate every single time."""
        return self.base_rate * (1.0 - self.base_rate)

    @property
    def skill_vs_coin_flip(self) -> float:
        return brier_skill_score(self.brier, COIN_FLIP * (1.0 - COIN_FLIP))

    @property
    def skill_vs_climatology(self) -> float:
        """The one that matters. A forecaster can beat a coin purely by knowing the base rate."""
        return brier_skill_score(self.brier, self.climatology_brier)

    @property
    def skill_vs_persistence(self) -> float | None:
        if self.persistence_brier is None:
            return None
        return brier_skill_score(self.brier, self.persistence_brier)

    @property
    def decomposition(self) -> tuple[float, float, float, float]:
        """Reliability, resolution, uncertainty, within-bin."""
        return murphy(self.bins, self.base_rate)

    @property
    def identity_residual(self) -> float:
        """``BS - (REL - RES + UNC + WITHIN)``. Floating-point zero, or the decomposition is wrong.

        Asserted at 1e-12 by the test suite across five bin counts and both a calibrated and a
        miscalibrated record. A decomposition that cannot rebuild the score it came from is not a
        decomposition of that score.
        """
        reliability, resolution, uncertainty, within = self.decomposition
        return self.brier - (reliability - resolution + uncertainty + within)

    @property
    def ece(self) -> float:
        return expected_calibration_error(self.bins)

    @property
    def mce(self) -> float:
        return maximum_calibration_error(self.bins)

    @property
    def comparability(self) -> str:
        """What this restatement licenses a reader to conclude, and what it does not."""
        return (
            "These metrics are computed the way ForecastBench, FinBench and Foresight Arena "
            "compute them, over a record of one narrow mechanical question — whether a move "
            "clears a cost hurdle — not over the open-ended world questions those benchmarks "
            "ask. A Brier score on an easier or harder question set is not a rival score, so no "
            "leaderboard figure is quoted here. What the restatement does establish is that this "
            "record can be read by anyone who reads those benchmarks, and that its skill is "
            "stated against three references rather than the most flattering one."
        )

    @property
    def verdict(self) -> str:
        reliability, resolution, _, _ = self.decomposition
        if self.skill_vs_climatology <= 0:
            return (
                "NO SKILL over climatology: stating the base rate every time would have scored at "
                "least as well, so the forecasts carry no information about which instants differ"
            )
        if resolution <= reliability:
            return (
                f"CALIBRATED BUT WEAKLY DISCRIMINATING: resolution {resolution:.4f} does not "
                f"exceed reliability {reliability:.4f}, so most of the score comes from knowing "
                f"the base rate rather than from telling instants apart"
            )
        return (
            f"SKILL ESTABLISHED over climatology ({self.skill_vs_climatology:+.1%}), with "
            f"resolution {resolution:.4f} against reliability {reliability:.4f} — the forecasts "
            f"discriminate, and the discrimination is larger than the miscalibration"
        )

    def render(self) -> str:
        reliability, resolution, uncertainty, within = self.decomposition
        persistence = (
            "n/a" if self.skill_vs_persistence is None else f"{self.skill_vs_persistence:+.1%}"
        )
        return "\n".join([
            f"FORECAST RECORD RESTATED — {self.count:,} forecasts, "
            f"{self.effective_count:,} effective ({self.distinct_instants:,} distinct instants, "
            f"ICC {self.icc:.3f})",
            "",
            f"  Brier score            {self.brier:.4f}   (0.25 is a coin)",
            f"  Brier Index            {self.brier_index:.1f}      (ForecastBench scale, "
            f"50 is a coin, higher is better)",
            f"  base rate              {self.base_rate:.4f}",
            "",
            f"  skill vs coin flip     {self.skill_vs_coin_flip:+.1%}",
            f"  skill vs climatology   {self.skill_vs_climatology:+.1%}   <- the one that counts",
            f"  skill vs persistence   {persistence}",
            "",
            f"  reliability (lower better)  {reliability:.5f}",
            f"  resolution  (higher better) {resolution:.5f}",
            f"  uncertainty (the question)  {uncertainty:.5f}",
            f"  within-bin  (binning only)  {within:.5f}",
            f"  Murphy identity residual    {self.identity_residual:.2e}",
            "",
            f"  expected calibration error  {self.ece:.5f}  over {len(self.bins)} occupied bins",
            f"  maximum calibration error   {self.mce:.5f}",
            "",
            f"  {self.verdict}",
            "",
            f"  {self.comparability}",
        ])

    def as_dict(self) -> dict[str, Any]:
        reliability, resolution, uncertainty, within = self.decomposition
        return {
            "count": self.count,
            "effective_count": self.effective_count,
            "distinct_instants": self.distinct_instants,
            "icc": round(self.icc, 5),
            "bins": BINS,
            "base_rate": round(self.base_rate, 5),
            "brier": round(self.brier, 5),
            "brier_index": round(self.brier_index, 3),
            "climatology_brier": round(self.climatology_brier, 5),
            "persistence_brier": (
                None if self.persistence_brier is None else round(self.persistence_brier, 5)
            ),
            "skill_vs_coin_flip": round(self.skill_vs_coin_flip, 5),
            "skill_vs_climatology": round(self.skill_vs_climatology, 5),
            "skill_vs_persistence": (
                None if self.skill_vs_persistence is None
                else round(self.skill_vs_persistence, 5)
            ),
            "reliability": round(reliability, 6),
            "resolution": round(resolution, 6),
            "uncertainty": round(uncertainty, 6),
            "within_bin": round(within, 8),
            "murphy_identity_residual": self.identity_residual,
            "ece": round(self.ece, 6),
            "mce": round(self.mce, 6),
            "occupied_bins": [b.as_dict() for b in self.bins],
            "verdict": self.verdict,
            "comparability": self.comparability,
        }


def _persistence_brier(outcomes: Sequence[bool]) -> float | None:
    """Brier of "it does again what it just did", scored from the second observation onward.

    The first observation has nothing before it, so it is dropped rather than given a default —
    a default would be a free guess the reference did not earn.
    """
    if len(outcomes) < 2:
        return None
    predictions = [1.0 if prev else 0.0 for prev in outcomes[:-1]]
    return brier(predictions, list(outcomes[1:]))


def restate(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    effective_count: int | None = None,
    distinct_instants: int | None = None,
    icc: float = 0.0,
    bins: int = BINS,
) -> Restatement:
    """Restate one record. ``effective_count`` should come from the clustering correction."""
    if not probabilities:
        raise RestatementError("nothing to restate")
    if len(probabilities) != len(outcomes):
        raise RestatementError("every probability needs exactly one outcome")
    score = brier(probabilities, outcomes)
    base_rate = sum(1 for o in outcomes if o) / len(outcomes)
    if base_rate in (0.0, 1.0):
        raise RestatementError(
            f"every outcome is the same ({'all cleared' if base_rate else 'none cleared'}); "
            f"uncertainty is zero, so skill against climatology is a division by zero and the "
            f"record cannot demonstrate discrimination either way"
        )
    return Restatement(
        count=len(probabilities),
        effective_count=len(probabilities) if effective_count is None else effective_count,
        distinct_instants=(
            len(probabilities) if distinct_instants is None else distinct_instants
        ),
        icc=icc,
        base_rate=base_rate,
        brier=score,
        bins=tuple(bin_forecasts(probabilities, outcomes, bins=bins)),
        persistence_brier=_persistence_brier(outcomes),
    )


def from_scorecard(scorecard: Any) -> Restatement:
    """Restate an `eval.forecasts.Scorecard` without recomputing anything it already knows.

    Takes the clustering correction from the scorecard rather than re-deriving it, so the effective
    count reported here is the same number that module publishes and the two can never disagree.
    """
    forecasts = list(scorecard.forecasts)
    if not forecasts:
        raise RestatementError("the scorecard holds no forecasts")
    return restate(
        [f.probability for f in forecasts],
        [f.cleared for f in forecasts],
        effective_count=scorecard.effective_count,
        distinct_instants=scorecard.distinct_instants,
        icc=scorecard.icc,
    )


def main() -> int:  # pragma: no cover - CLI
    from argus.eval.forecasts import build_scorecard

    restatement = from_scorecard(build_scorecard())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(restatement.as_dict(), indent=2), encoding="utf-8")
    print(restatement.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BINS",
    "COIN_FLIP",
    "Bin",
    "Restatement",
    "RestatementError",
    "bin_forecasts",
    "brier",
    "brier_index",
    "brier_skill_score",
    "expected_calibration_error",
    "from_scorecard",
    "maximum_calibration_error",
    "murphy",
    "restate",
]
