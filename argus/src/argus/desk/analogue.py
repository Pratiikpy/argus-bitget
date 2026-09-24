"""Historical analogue retrieval — "has this setup happened before, and what happened next?"

Track 3's Decision Stress Testing sub-theme asks, verbatim: *"Before opening a position, how does AI
retrieve historically similar scenarios?"* ARGUS already answers the easier half —
:func:`argus.desk.workbench.stress_position` shocks a position through preset scenarios with a
realistic exit cost. That is scenario stress, and it is a different question. Scenario stress asks
*what if the market fell 5%*. Analogue retrieval asks *when conditions looked like this before, what
actually happened* — and answers with a distribution rather than an anecdote.

The sweep found nobody implementing it properly. Systems either shock a portfolio through canned
scenarios or retrieve a handful of superficially similar dates and narrate them. The gap between
those and a usable answer is entirely in the methodology, so that is where the work is here.

**Four things that separate a distribution from an anecdote**, each of which is a way the naive
version lies:

1. **Match on state, not on outcome.** Selecting past days by what happened next is the purest form
   of look-ahead, and it is easy to do by accident — "find me days like this that rallied" is a
   question that answers itself. :func:`find` can only see features stamped at or before the match
   date.
2. **Report the whole distribution.** A mean alone hides whether five analogues agreed or two
   extremes cancelled. Quartiles and the hit rate come back with it.
3. **Count the effective sample, not the raw one.** Twenty analogues drawn from three clustered
   episodes are not twenty independent observations. Overlapping windows are collapsed, and both
   counts are reported so the shrinkage is visible.
4. **Refuse below a floor.** A distribution over four analogues is an anecdote with error bars. The
   honest output is a refusal naming the count, which is the same rule
   :mod:`argus.eval.performance` applies to Sharpe.

**Similarity is explicit and inspectable.** Each feature is z-scored across the corpus so that a
feature with a wide natural range does not dominate one with a narrow one, then distance is a
weighted Euclidean norm over those z-scores. The per-feature contribution comes back with every
match, so "why is this an analogue" is answerable rather than asserted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

MIN_ANALOGUES = 8
"""Below this, the honest answer is a refusal that names the count."""

DEFAULT_MAX_DISTANCE = 2.0
"""Beyond two standard deviations of combined feature distance, it is not the same setup."""

OVERLAP_WINDOW = timedelta(days=3)
"""Analogues closer together than this describe one episode, not two observations."""


class AnalogueError(RuntimeError):
    """The corpus cannot support the question asked of it."""


@dataclass(frozen=True)
class Observation:
    """One historical state and what followed it.

    ``features`` are stamped **at** ``as_of``. ``forward_return_bps`` is what happened after, and is
    never available to the matcher — it is the answer, not part of the question.
    """

    as_of: datetime
    symbol: str
    features: Mapping[str, float]
    forward_return_bps: float

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise AnalogueError("observation timestamps must be timezone-aware")


@dataclass(frozen=True)
class Match:
    """One retrieved analogue, and why it matched."""

    observation: Observation
    distance: float
    contributions: Mapping[str, float]
    """Per-feature share of the distance. Answers "why is this an analogue" with numbers."""

    @property
    def dominant_feature(self) -> str:
        if not self.contributions:
            return "none"
        return max(self.contributions, key=lambda k: self.contributions[k])

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.observation.as_of.isoformat(),
            "symbol": self.observation.symbol,
            "distance": round(self.distance, 4),
            "forward_return_bps": round(self.observation.forward_return_bps, 2),
            "dominant_feature": self.dominant_feature,
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
        }


@dataclass(frozen=True)
class Distribution:
    """What actually happened across the analogues. Never a single number."""

    returns_bps: tuple[float, ...]
    effective_n: int

    @property
    def count(self) -> int:
        return len(self.returns_bps)

    @property
    def median(self) -> float:
        ordered = sorted(self.returns_bps)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    @property
    def mean(self) -> float:
        return sum(self.returns_bps) / len(self.returns_bps)

    @property
    def hit_rate(self) -> float:
        """Fraction that moved up. Direction, separate from magnitude."""
        return sum(1 for r in self.returns_bps if r > 0) / len(self.returns_bps)

    def quantile(self, pct: float) -> float:
        ordered = sorted(self.returns_bps)
        if len(ordered) == 1:
            return ordered[0]
        pos = (pct / 100) * (len(ordered) - 1)
        low, high = math.floor(pos), math.ceil(pos)
        if low == high:
            return ordered[int(pos)]
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

    @property
    def dispersion(self) -> float:
        """Interquartile range. A wide one means the analogues disagreed."""
        return self.quantile(75) - self.quantile(25)

    def as_dict(self) -> dict[str, Any]:
        return {
            "analogues": self.count,
            "effective_n": self.effective_n,
            "median_bps": round(self.median, 2),
            "mean_bps": round(self.mean, 2),
            "p25_bps": round(self.quantile(25), 2),
            "p75_bps": round(self.quantile(75), 2),
            "worst_bps": round(min(self.returns_bps), 2),
            "best_bps": round(max(self.returns_bps), 2),
            "iqr_bps": round(self.dispersion, 2),
            "hit_rate": round(self.hit_rate, 3),
        }


@dataclass(frozen=True)
class AnalogueReport:
    """The answer, or a named refusal."""

    matches: tuple[Match, ...]
    distribution: Distribution | None
    refused: str = ""

    @property
    def usable(self) -> bool:
        return self.distribution is not None

    def render(self) -> list[str]:
        if not self.usable:
            return [f"[analogue] no distribution: {self.refused}"]
        assert self.distribution is not None
        d = self.distribution
        lines = [
            f"[analogue] {d.count} historical matches ({d.effective_n} independent episodes): "
            f"median {d.median:+.0f}bps, IQR {d.quantile(25):+.0f} to {d.quantile(75):+.0f}bps, "
            f"up {d.hit_rate:.0%} of the time",
            f"[analogue] worst {min(d.returns_bps):+.0f}bps, best {max(d.returns_bps):+.0f}bps "
            f"— the spread, not the average, is what a position has to survive",
        ]
        if d.effective_n < d.count:
            lines.append(
                f"[analogue] {d.count - d.effective_n} match(es) overlapped in time and were "
                f"collapsed; clustered analogues are one observation wearing several dates"
            )
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "refused": self.refused,
            "distribution": None if self.distribution is None else self.distribution.as_dict(),
            "matches": [m.as_dict() for m in self.matches],
        }


def _standardise(
    corpus: Sequence[Observation], keys: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation per feature, so no feature dominates by unit choice alone."""
    stats: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = [float(o.features[key]) for o in corpus if key in o.features]
        if len(values) < 2:
            stats[key] = (0.0, 1.0)
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        stats[key] = (mean, math.sqrt(var) or 1.0)
    return stats


def _collapse_overlapping(matches: Sequence[Match]) -> int:
    """Count independent episodes: matches within :data:`OVERLAP_WINDOW` are one observation."""
    if not matches:
        return 0
    by_time = sorted(matches, key=lambda m: (m.observation.symbol, m.observation.as_of))
    episodes = 1
    previous = by_time[0]
    for match in by_time[1:]:
        same_symbol = match.observation.symbol == previous.observation.symbol
        gap = match.observation.as_of - previous.observation.as_of
        if not (same_symbol and gap <= OVERLAP_WINDOW):
            episodes += 1
        previous = match
    return episodes


def find(
    *,
    query: Mapping[str, float],
    corpus: Sequence[Observation],
    as_of: datetime,
    weights: Mapping[str, float] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    limit: int = 50,
) -> AnalogueReport:
    """Retrieve historical states resembling ``query`` and report what followed them.

    ``as_of`` is the decision instant: no observation stamped at or after it can be returned, so a
    retrieval run during a backtest cannot see its own future. This is the same gate
    :func:`argus.market.evidence.gather` applies to evidence, and for the same reason.
    """
    if as_of.tzinfo is None:
        raise AnalogueError("as_of must be timezone-aware; a naive clock cannot bound a search")
    keys = [k for k in query if any(k in o.features for o in corpus)]
    if not keys:
        return AnalogueReport((), None, refused="no query feature appears in the corpus")

    eligible = [o for o in corpus if o.as_of < as_of]
    if len(eligible) < MIN_ANALOGUES:
        return AnalogueReport(
            (), None,
            refused=(
                f"{len(eligible)} observation(s) predate the decision instant; "
                f"{MIN_ANALOGUES} are needed for a distribution"
            ),
        )

    stats = _standardise(eligible, keys)
    weight = dict(weights or {})

    scored: list[Match] = []
    for observation in eligible:
        contributions: dict[str, float] = {}
        total = 0.0
        usable = True
        for key in keys:
            if key not in observation.features:
                usable = False
                break
            mean, sd = stats[key]
            delta = (float(observation.features[key]) - mean) / sd - (
                (float(query[key]) - mean) / sd
            )
            w = float(weight.get(key, 1.0))
            term = w * delta * delta
            contributions[key] = term
            total += term
        if not usable:
            continue
        distance = math.sqrt(total)
        if distance <= max_distance:
            share = {k: (v / total if total > 0 else 0.0) for k, v in contributions.items()}
            scored.append(Match(observation, distance, share))

    scored.sort(key=lambda m: m.distance)
    matches = tuple(scored[:limit])

    if len(matches) < MIN_ANALOGUES:
        return AnalogueReport(
            matches, None,
            refused=(
                f"{len(matches)} analogue(s) within distance {max_distance}; "
                f"{MIN_ANALOGUES} are needed. A distribution over fewer is an anecdote with "
                f"error bars"
            ),
        )

    distribution = Distribution(
        returns_bps=tuple(m.observation.forward_return_bps for m in matches),
        effective_n=_collapse_overlapping(matches),
    )
    return AnalogueReport(matches, distribution)


def corpus_from_closes(
    closes: Sequence[tuple[datetime, float]], *, symbol: str, window: int = 24, horizon: int = 24,
) -> list[Observation]:
    """Every bar of an hourly history as a state and the move that followed it.

    The state is the trailing ``window``-bar return and realised volatility, both in bps; the
    forward ``horizon``-bar return is the answer and is never a feature. Factored out of
    `desk/research.py`'s research chain on 2026-09-23 so the console's analogue answers and the
    chain's are built by one function rather than two copies that could drift apart.
    """
    corpus: list[Observation] = []
    for i in range(window, len(closes) - horizon):
        past = [closes[j][1] for j in range(i - window, i + 1)]
        rets = [b / a - 1.0 for a, b in pairwise(past) if a > 0]
        if not rets:
            continue
        mean = sum(rets) / len(rets)
        vol = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
        close_now = closes[i][1]
        if close_now <= 0 or past[0] <= 0:
            continue
        corpus.append(Observation(
            as_of=closes[i][0], symbol=symbol,
            features={
                "trailing_return": (close_now / past[0] - 1.0) * 10_000,
                "volatility_bps": vol * 10_000,
            },
            forward_return_bps=(closes[i + horizon][1] / close_now - 1.0) * 10_000,
        ))
    return corpus


def current_state(
    closes: Sequence[tuple[datetime, float]], *, window: int = 24,
) -> dict[str, float] | None:
    """The state right now, described exactly as :func:`corpus_from_closes` describes the past —
    the query an analogue search needs. None when there is not a full window yet."""
    if len(closes) <= window:
        return None
    past = [c for _, c in closes[-(window + 1):]]
    rets = [b / a - 1.0 for a, b in pairwise(past) if a > 0]
    if not rets or past[0] <= 0:
        return None
    mean = sum(rets) / len(rets)
    vol = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    return {
        "trailing_return": (past[-1] / past[0] - 1.0) * 10_000,
        "volatility_bps": vol * 10_000,
    }


__all__ = [
    "DEFAULT_MAX_DISTANCE",
    "MIN_ANALOGUES",
    "OVERLAP_WINDOW",
    "AnalogueError",
    "AnalogueReport",
    "Distribution",
    "Match",
    "Observation",
    "corpus_from_closes",
    "current_state",
    "find",
]
