"""The overfitting tests the deflated Sharpe does not do — and the one that prices our own gap.

`argus.backtest.metrics` already carries the probabilistic and deflated Sharpe from Bailey and
López de Prado. A grep of this tree for the rest of that literature returned nothing: no
probability of backtest overfitting, no combinatorially symmetric cross-validation, no purged or
embargoed splits, no false-discovery control, and no minimum track record length. Each is a
different question, and one of them answers ours.

**Minimum track record length is the number this project needs most.** The paper ledger holds 126
decisions and no position has settled, so there is no Sharpe to report and Track 2 scores half on
exactly that. The usual response is to apologise for it. The better one is to compute how long a
track record would have to be before a Sharpe of a given size *could* be believed — which turns
"we have no Sharpe yet" from an excuse into a measurement. :func:`min_track_record_length` does
that, and on plausible numbers the answer is large enough to be worth saying out loud.

**Probability of backtest overfitting answers what the deflated Sharpe cannot.** The deflated
Sharpe asks whether *one* observed Sharpe survives a search of known size. PBO asks a sharper
question about the *selection procedure itself*: if you pick the best strategy in-sample, how often
does it land below median out-of-sample? A process with PBO near 0.5 is picking noise, and it looks
exactly like a process that works right up until it is used. :func:`probability_of_overfitting`
implements the combinatorially symmetric cross-validation of Bailey, Borwein, López de Prado and
Zhu (2014) — every balanced split, not a sample of them, so the answer does not depend on which
splits were drawn.

**Purging and embargoing are about leakage, not overfitting.** A label computed over the next
twenty-four hours overlaps the features of the next twenty-four bars, so a naive split trains on
rows whose outcomes are already in the test set. :func:`purged_splits` removes the overlap and then
embargoes a further gap, following López de Prado's *Advances in Financial Machine Learning*.

**False-discovery control is for the sweep, not the winner.** Deflating one Sharpe for the trial
count and asking which of twenty-five variants are individually significant are different
questions. :func:`benjamini_hochberg` answers the second at a controlled false-discovery rate, and
:func:`bonferroni` gives the conservative bound beside it, because a reader should see both.

Every function here raises rather than returning a comforting default. A validation layer that
returns 0.0 for "could not compute" is a validation layer that passes everything, which is the
exact defect `metrics.deflated_sharpe` was written to avoid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb, isfinite, sqrt
from statistics import NormalDist
from typing import Any

from argus.backtest.metrics import MetricError, _is_negligible

MIN_OBSERVATIONS = 10
"""Fewest return observations before any of this is meaningful."""

DEFAULT_CONFIDENCE = 0.95


def moments(returns: Sequence[float]) -> tuple[float, float, float, float]:
    """Mean, standard deviation, skew and kurtosis. Raises on a degenerate series.

    Public because the probabilistic and deflated Sharpe both take skew and kurtosis and both
    default them to the normal values. Anything computing a p-value from a return series needs
    the real moments, and a second implementation of them is a second thing to get wrong.
    """
    n = len(returns)
    if n < MIN_OBSERVATIONS:
        raise MetricError(
            f"{n} observation(s) is below the {MIN_OBSERVATIONS} these statistics need; "
            f"reporting one anyway would be a number with no sample behind it"
        )
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = sqrt(variance)
    # Not `variance <= 0`. A constant series of 0.01 returns has sd 1.8e-18 rather than 0.0, so an
    # exact test never fires and the Sharpe comes back near 1e15 — which then produces a
    # higher-moment adjustment that raises for the wrong reason and reports the wrong cause.
    # `metrics._is_negligible` already characterises this exact failure; reused rather than
    # restated so the two tolerances cannot drift apart.
    if variance <= 0 or _is_negligible(sd, mean):
        raise MetricError("a zero-variance series has no Sharpe and therefore no track record")
    skew = sum(((r - mean) / sd) ** 3 for r in returns) / n
    kurtosis = sum(((r - mean) / sd) ** 4 for r in returns) / n
    return mean, sd, skew, kurtosis


def min_track_record_length(
    returns: Sequence[float],
    *,
    benchmark_sharpe: float = 0.0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """How many observations are needed before this Sharpe beats the benchmark at ``confidence``.

    Bailey and López de Prado (2012). The per-observation Sharpe is estimated from the sample, and
    the result is the sample size at which the probabilistic Sharpe ratio would first exceed the
    confidence level — adjusted for the skew and fat tails that make a short record flatter than a
    normal one would suggest.

    Raises when the observed Sharpe does not exceed the benchmark at all: there is no length of
    record that makes a losing strategy significant, and returning a large number would imply
    there is.
    """
    mean, sd, skew, kurtosis = moments(returns)
    sharpe = mean / sd
    if sharpe <= benchmark_sharpe:
        raise MetricError(
            f"observed Sharpe {sharpe:.4f} does not exceed the benchmark {benchmark_sharpe:.4f}; "
            f"no length of track record makes it significant"
        )
    z = NormalDist().inv_cdf(confidence)
    adjustment = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if adjustment <= 0:
        raise MetricError(
            "the higher-moment adjustment is non-positive, so the track record length is "
            "undefined for this return distribution"
        )
    return 1.0 + adjustment * (z / (sharpe - benchmark_sharpe)) ** 2


def annualised_track_record_years(
    returns: Sequence[float], *, periods_per_year: int, **kwargs: Any
) -> float:
    """:func:`min_track_record_length` expressed in years, which is what a reader can act on."""
    return min_track_record_length(returns, **kwargs) / periods_per_year


@dataclass(frozen=True)
class OverfittingResult:
    """The outcome of a combinatorially symmetric cross-validation."""

    pbo: float
    """Share of splits where the in-sample best landed below the out-of-sample median."""

    splits: int
    strategies: int
    observations: int
    out_of_sample_ranks: tuple[float, ...]
    """The chosen strategy's out-of-sample rank in each split, as a fraction in [0, 1]."""

    chosen: tuple[int, ...] = ()
    """Which strategy the in-sample half selected, per split.

    Kept because PBO answers *whether* the selection degrades and not *why*, and the two commonest
    mechanisms leave opposite fingerprints here. A procedure that picks the same strategy in every
    split and still ranks below the out-of-sample median has found something stable and useless. A
    procedure that picks a different strategy in most splits is not selecting at all — it is
    sampling, and its headline number is the maximum of a noise distribution whichever variant
    happens to carry it. :attr:`selection_churn` is the statistic that tells them apart.
    """

    @property
    def selection_churn(self) -> float:
        """Distinct winners over splits. 1/n means one strategy always won; 1.0 means never twice.

        On our own Track 1 sweep this is the diagnostic that explains the headline: the symbol with
        the worst PBO is the one whose winner changes most.
        """
        if not self.chosen:
            return 0.0
        return len(set(self.chosen)) / len(self.chosen)

    @property
    def dominant_share(self) -> float:
        """How often the most-chosen strategy was chosen. The complement of churn, and the one a
        reader intuits: 0.9 means one variant won nine splits in ten."""
        if not self.chosen:
            return 0.0
        counts: dict[int, int] = {}
        for index in self.chosen:
            counts[index] = counts.get(index, 0) + 1
        return max(counts.values()) / len(self.chosen)

    @property
    def median_rank(self) -> float:
        ordered = sorted(self.out_of_sample_ranks)
        if not ordered:
            return 0.0
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    SINGLE_ESTIMATE_SD = 0.25
    """Measured dispersion of PBO across independent datasets drawn from the same null.

    Thirty realisations of 12 strategies over 1,200 observations of pure noise gave a mean PBO of
    0.475 — correct, the null is 0.5 — with a **standard deviation of 0.252**. Three single draws
    from that same null gave 0.614, 0.643 and 0.343.

    So one PBO number is nearly uninformative on its own, and this constant exists so the result
    says that itself rather than leaving a reader to infer a precision it does not have. It is the
    same lesson the cross-sectional phase sweep taught: a statistic computed once, from one
    alignment of one dataset, is a draw and not a measurement.
    """

    @property
    def reliability(self) -> str:
        """What this single estimate can and cannot support."""
        return (
            f"A single PBO estimate has a standard deviation of about "
            f"{self.SINGLE_ESTIMATE_SD:.2f} across independent datasets from the same null "
            f"(measured: 30 realisations, mean 0.475 under a true 0.5). Read {self.pbo:.2f} as "
            f"roughly {max(0.0, self.pbo - self.SINGLE_ESTIMATE_SD):.2f} to "
            f"{min(1.0, self.pbo + self.SINGLE_ESTIMATE_SD):.2f}, and do not separate two "
            f"strategies on a PBO difference smaller than that."
        )

    @property
    def verdict(self) -> str:
        if self.pbo >= 0.5:
            return (
                "the selection procedure is picking noise: the in-sample best lands below the "
                "out-of-sample median at least half the time, which is what choosing at random "
                "would do"
            )
        if self.pbo >= 0.25:
            return "the selection carries real information but degrades substantially out of sample"
        return "the selection survives out of sample on this evidence"

    def render(self) -> str:
        return "\n".join([
            f"PROBABILITY OF BACKTEST OVERFITTING — {self.pbo:.1%}",
            f"  {self.splits} balanced split(s) of {self.observations} observation(s) across "
            f"{self.strategies} strategies; every split, not a sample of them",
            f"  median out-of-sample rank of the in-sample winner: {self.median_rank:.2f} "
            f"(0.5 is a coin)",
            f"  selection churn {self.selection_churn:.2f}; the most-chosen strategy won "
            f"{self.dominant_share:.0%} of splits",
            f"  {self.verdict}",
            f"  {self.reliability}",
        ])

    def as_dict(self) -> dict[str, Any]:
        return {
            "pbo": round(self.pbo, 5),
            "splits": self.splits,
            "strategies": self.strategies,
            "observations": self.observations,
            "median_out_of_sample_rank": round(self.median_rank, 5),
            "selection_churn": round(self.selection_churn, 5),
            "dominant_share": round(self.dominant_share, 5),
            "verdict": self.verdict,
            "single_estimate_sd": self.SINGLE_ESTIMATE_SD,
            "reliability": self.reliability,
        }


def probability_of_overfitting(
    matrix: Sequence[Sequence[float]], *, groups: int = 8
) -> OverfittingResult:
    """CSCV: pick the in-sample best, see where it ranks out of sample, over every balanced split.

    ``matrix`` is observations by strategies — row *t* holds each strategy's return at time *t*.
    The series is cut into ``groups`` contiguous blocks and every way of assigning half of them to
    training is evaluated, which is ``C(groups, groups/2)`` splits: 70 at eight groups, 924 at
    twelve. Contiguous blocks, not shuffled rows, because shuffling a time series puts a test
    observation between two training observations that bracket it.

    The statistic is the share of splits in which the training winner ranks **below the median**
    out of sample. At 0.5 the procedure is indistinguishable from choosing at random.

    Ties take the **midrank**, and that is not a detail. Counting only strictly-worse strategies
    puts every member of a tie at rank 0, so a matrix containing duplicate columns — which is what
    a sweep produces whenever two variants never trade and both return a flat zero series — scores
    a PBO of 1.0 and reads as catastrophic overfitting when the truth is that the strategies cannot
    be told apart. With the midrank, a field of identical strategies sits exactly at the median and
    the procedure is scored as the coin toss it is. Distinct columns are unaffected.
    """
    if groups < 2 or groups % 2:
        raise MetricError(f"groups must be an even number of at least 2, got {groups}")
    rows = [list(r) for r in matrix]
    if not rows:
        raise MetricError("no observations to cross-validate")
    strategies = len(rows[0])
    if strategies < 2:
        raise MetricError("CSCV compares strategies against each other; it needs at least two")
    if any(len(r) != strategies for r in rows):
        raise MetricError("every observation must carry a return for every strategy")
    if len(rows) < groups * MIN_OBSERVATIONS:
        raise MetricError(
            f"{len(rows)} observation(s) cannot be split into {groups} blocks of at least "
            f"{MIN_OBSERVATIONS}; a block too small to compute a Sharpe makes the split a coin toss"
        )

    size = len(rows) // groups
    blocks = [rows[i * size:(i + 1) * size] for i in range(groups)]

    def sharpe_of(block_set: Sequence[Sequence[Sequence[float]]], index: int) -> float:
        series = [row[index] for block in block_set for row in block]
        n = len(series)
        mean = sum(series) / n
        variance = sum((v - mean) ** 2 for v in series) / (n - 1) if n > 1 else 0.0
        return mean / sqrt(variance) if variance > 0 else 0.0

    ranks: list[float] = []
    picks: list[int] = []
    half = groups // 2
    for train_idx in combinations(range(groups), half):
        test_idx = [i for i in range(groups) if i not in train_idx]
        train = [blocks[i] for i in train_idx]
        test = [blocks[i] for i in test_idx]
        in_sample = [sharpe_of(train, s) for s in range(strategies)]
        best = max(range(strategies), key=lambda s: in_sample[s])
        out_sample = [sharpe_of(test, s) for s in range(strategies)]
        chosen = out_sample[best]
        below = sum(1 for v in out_sample if v < chosen)
        tied = sum(1 for v in out_sample if v == chosen) - 1
        picks.append(best)
        ranks.append((below + tied / 2) / (strategies - 1))

    return OverfittingResult(
        pbo=sum(1 for r in ranks if r < 0.5) / len(ranks),
        splits=comb(groups, half),
        strategies=strategies,
        observations=len(rows),
        out_of_sample_ranks=tuple(ranks),
        chosen=tuple(picks),
    )


def purged_splits(
    n: int, *, folds: int = 5, label_horizon: int = 1, embargo: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Time-series folds with the overlapping observations removed and a gap embargoed.

    A label computed over the next ``label_horizon`` bars overlaps the features of the following
    ``label_horizon`` bars, so a naive split trains on rows whose outcomes already sit in the test
    set. Purging drops the training rows that overlap the test window; the embargo then drops a
    further ``embargo`` rows after it, because serial correlation leaks past the exact overlap.

    Returns ``(train, test)`` index lists per fold. López de Prado, *Advances in Financial Machine
    Learning*, chapter 7.
    """
    if folds < 2:
        raise MetricError("purged cross-validation needs at least two folds")
    if n < folds:
        raise MetricError(f"{n} observation(s) cannot be split into {folds} folds")
    if label_horizon < 0 or embargo < 0:
        raise MetricError("the label horizon and embargo cannot be negative")

    size = n // folds
    out: list[tuple[list[int], list[int]]] = []
    for fold in range(folds):
        start = fold * size
        stop = n if fold == folds - 1 else (fold + 1) * size
        test = list(range(start, stop))
        blocked_from = start - label_horizon
        blocked_to = stop + label_horizon + embargo
        train = [i for i in range(n) if i < blocked_from or i >= blocked_to]
        out.append((train, test))
    return out


def benjamini_hochberg(p_values: Sequence[float], *, fdr: float = 0.05) -> list[bool]:
    """Which p-values survive at a controlled false-discovery rate. Order is preserved.

    Bonferroni controls the chance of *any* false positive and is brutal across twenty-five
    variants; Benjamini-Hochberg controls the expected *share* of false positives among the
    rejections, which is the right question for a sweep whose purpose is to shortlist.
    """
    if not p_values:
        return []
    if any(not isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise MetricError("every p-value must be a finite number in [0, 1]")
    m = len(p_values)
    ordered = sorted(range(m), key=lambda i: p_values[i])
    cutoff = -1
    for rank, index in enumerate(ordered, start=1):
        if p_values[index] <= rank / m * fdr:
            cutoff = rank
    keep = {ordered[i] for i in range(cutoff)} if cutoff > 0 else set()
    return [i in keep for i in range(m)]


def bonferroni(p_values: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """The conservative bound, reported beside the false-discovery result rather than instead."""
    if any(not isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise MetricError("every p-value must be a finite number in [0, 1]")
    threshold = alpha / len(p_values) if p_values else alpha
    return [p <= threshold for p in p_values]


__all__ = [
    "DEFAULT_CONFIDENCE",
    "MIN_OBSERVATIONS",
    "OverfittingResult",
    "annualised_track_record_years",
    "benjamini_hochberg",
    "bonferroni",
    "min_track_record_length",
    "moments",
    "probability_of_overfitting",
    "purged_splits",
]
