"""Probabilistic calibration: does a stated confidence mean what it says?

Moved from `eval/observatory.py` (its first capability) on 2026-09-27, because the risk
layer sizes on it (`risk/sizing.py`) and the paper runner records it: both imported it
upward from the evaluation layer. The observatory still reports it, from here.
"""

from __future__ import annotations

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


__all__ = ["WILSON_Z", "Prediction", "brier_score", "expected_calibration_error",
           "reliability_curve", "wilson"]
