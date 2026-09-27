"""Probabilistic calibration: does a stated confidence mean what it says?

Moved from `eval/observatory.py` (its first capability) on 2026-09-27, because the risk
layer sizes on it (`risk/sizing.py`) and the paper runner records it: both imported it
upward from the evaluation layer. The observatory still reports it, from here.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    """A stated confidence and what actually happened."""

    confidence: float
    correct: bool


def brier_score(predictions: list[Prediction]) -> float:
    """Mean squared error of stated probabilities. Lower is better; 0.25 is a coin flip.

    Exhaustively grepped across the four evaluation repos in the corpus: **neither ECE nor Brier
    appears anywhere.** Abstention calibration exists in places; probability calibration does not.
    """
    if not predictions:
        raise ValueError("Brier score needs at least one prediction")
    return sum((p.confidence - (1.0 if p.correct else 0.0)) ** 2 for p in predictions) / len(
        predictions
    )


def expected_calibration_error(predictions: list[Prediction], *, bins: int = 10) -> float:
    """ECE — the gap between stated confidence and realised frequency, weighted by bin size.

    The number that makes "90% confident" mean something. If a model's 90%-confidence calls land
    60% of the time, sizing on that confidence is a mistake, and this is what surfaces it.
    """
    if not predictions:
        raise ValueError("ECE needs at least one prediction")

    buckets: dict[int, list[Prediction]] = {}
    for p in predictions:
        idx = min(bins - 1, int(p.confidence * bins))
        buckets.setdefault(idx, []).append(p)

    total = len(predictions)
    error = 0.0
    for group in buckets.values():
        avg_conf = sum(p.confidence for p in group) / len(group)
        accuracy = sum(1 for p in group if p.correct) / len(group)
        error += (len(group) / total) * abs(avg_conf - accuracy)
    return error


def reliability_curve(
    predictions: list[Prediction], *, bins: int = 10
) -> list[dict[str, float | int]]:
    """Per-bin stated confidence versus realised accuracy. The plot behind the ECE number."""
    buckets: dict[int, list[Prediction]] = {}
    for p in predictions:
        idx = min(bins - 1, int(p.confidence * bins))
        buckets.setdefault(idx, []).append(p)

    return [
        {
            "bin": i,
            "range_low": round(i / bins, 2),
            "range_high": round((i + 1) / bins, 2),
            "n": len(buckets[i]),
            "stated": round(sum(p.confidence for p in buckets[i]) / len(buckets[i]), 3),
            "realised": round(sum(1 for p in buckets[i] if p.correct) / len(buckets[i]), 3),
        }
        for i in sorted(buckets)
    ]


@dataclass(frozen=True, slots=True)
class Recalibration:
    """A monotone map from stated confidence to the frequency the record actually realised.

    Built by :func:`isotonic`. Evaluated the way scikit-learn's ``IsotonicRegression.predict``
    evaluates its fit (the estimator both uncertainty-toolbox's ``recalibration.py:iso_recal`` and
    netcal's ``binning/IsotonicRegression.py`` wrap): linear interpolation between the fitted
    points, clipped to the first and last value outside them.
    """

    points: tuple[tuple[float, float], ...]
    """``(stated, realised)`` pairs, stated non-decreasing, realised non-decreasing."""

    def __call__(self, confidence: float) -> float:
        first, last = self.points[0], self.points[-1]
        if confidence <= first[0]:
            return first[1]
        if confidence >= last[0]:
            return last[1]
        for (x0, y0), (x1, y1) in itertools.pairwise(self.points):
            if x0 <= confidence <= x1:
                if x1 == x0:
                    return y1
                return y0 + (confidence - x0) / (x1 - x0) * (y1 - y0)
        return last[1]  # pragma: no cover - the points cover the interval

    def as_dict(self) -> dict[str, list[list[float]]]:
        return {"points": [[round(x, 4), round(y, 4)] for x, y in self.points]}


def isotonic(predictions: list[Prediction]) -> Recalibration:
    """Pool-adjacent-violators: the non-decreasing step fit of hit rate on stated confidence.

    Rebuilt rather than imported (research/harvest/23-uncertainty-toolbox.md): both sources reach
    it through scikit-learn and numpy, and this package keeps the risk path pure Python. Equal
    confidences are pooled first; then any block whose hit rate is below the block before it is
    merged with it, weighted by count, until the rates never fall as confidence rises. Each block
    contributes its lowest and highest stated confidence, both at the block's rate — the thresholds
    scikit-learn keeps.
    """
    if not predictions:
        raise ValueError("isotonic recalibration needs at least one prediction")
    pooled: dict[float, list[int]] = {}
    for p in predictions:
        cell = pooled.setdefault(p.confidence, [0, 0])
        cell[0] += 1 if p.correct else 0
        cell[1] += 1
    # Each block: [low confidence, high confidence, hits, count].
    blocks: list[list[float]] = []
    for confidence in sorted(pooled):
        hits, count = pooled[confidence]
        blocks.append([confidence, confidence, hits, count])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            merged = blocks.pop()
            blocks[-1][1] = merged[1]
            blocks[-1][2] += merged[2]
            blocks[-1][3] += merged[3]
    points: list[tuple[float, float]] = []
    for low, high, block_hits, block_count in blocks:
        rate = block_hits / block_count
        points.append((low, rate))
        if high != low:
            points.append((high, rate))
    return Recalibration(points=tuple(points))


def held_out_ece(predictions: list[Prediction], *, folds: int = 5) -> float:
    """ECE of the recalibrated confidences, each scored by a map fitted without it.

    A map fitted and scored on the same record is calibrated by construction, so its in-sample ECE
    says nothing. The record is cut into ``folds`` contiguous blocks, in the order it was written;
    each block is mapped by an :func:`isotonic` fit on the others and the pooled result scored.
    """
    if len(predictions) < folds:
        raise ValueError(f"{len(predictions)} predictions cannot fill {folds} folds")
    bounds = [round(i * len(predictions) / folds) for i in range(folds + 1)]
    scored: list[Prediction] = []
    for start, stop in itertools.pairwise(bounds):
        fitted = isotonic(predictions[:start] + predictions[stop:])
        scored.extend(Prediction(confidence=fitted(p.confidence), correct=p.correct)
                      for p in predictions[start:stop])
    return expected_calibration_error(scored)


WILSON_Z = 1.96


def wilson(successes: float, n: float, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a proportion; accepts a fractional effective count."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


__all__ = ["WILSON_Z", "Prediction", "Recalibration", "brier_score",
           "expected_calibration_error", "held_out_ece", "isotonic", "reliability_curve",
           "wilson"]
