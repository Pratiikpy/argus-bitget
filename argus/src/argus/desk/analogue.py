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

**Diversity-aware retrieval, opt-in (added 2026-09-25).** The nearest fifty states in an hourly
corpus are mostly the same few episodes seen an hour apart; the effective-sample count above makes
that visible but does not change which states are retrieved. ``find(diversity=λ)`` re-ranks with
Maximal Marginal Relevance, taken from paper-qa (Apache-2.0,
`src/paperqa/llms.py:111-170`, ``VectorStore.max_marginal_relevance_search``): fetch
``fetch_k = 2 * limit`` candidates by relevance (the ``2 * k`` is paper-qa's own call site,
`docs.py:483`), seed with the most relevant, then greedily add the candidate maximising
``λ * relevance - (1 - λ) * max_similarity_to_already_selected`` (`llms.py:158-161`). What is ours:
paper-qa's relevance and similarity are embedding cosines; here both are one Gaussian kernel,
``exp(-d² / (2 h²))``, over the same weighted z-score distance the matcher already uses, with
the bandwidth ``h`` set to the pool's own radius (:func:`relevance` records the run that forced
that), so ``λ = 1`` reproduces the plain nearest-first order exactly and the penalty is measured in
the space the match was made in. paper-qa's per-group ``partitioning_fn`` is not taken: a
single-symbol corpus has one group.

``find(explain=True)`` attaches a :class:`ScoreDetails` to each match in the shape of mem0's
``score_and_rank(explain=True)`` (Apache-2.0, `mem0/utils/scoring.py:60-140`): every raw component,
the threshold, and the final score, so the reason an analogue was chosen — and the diversity
penalty it paid — is a set of numbers rather than a sentence. mem0's rule that the threshold gates
the relevance term *before* anything is combined (`scoring.py:108-110`) is the rule here too: a
state beyond ``max_distance`` never enters the MMR pool however novel it is. mem0's BM25 and
entity-boost terms are not taken, because a numeric state vector has no tokens and no entities; a
keyword term with nothing to match would be a zero pretending to be a signal.

**Measured, and the default is unchanged.** Diversity is off unless asked for (paper-qa's own
default is off as well: ``texts_index_mmr_lambda = 1.0``, `settings.py:804`). The re-ranking was
re-proved on the comparison this capability stands on — AnalogDesk's pre-registered test, in
:mod:`argus.eval.retrieval_diversity`, ``data/retrieval_diversity.json`` (2026-09-25). The existing
comparison reproduced exactly first (Winkler difference 0.0, verdict unchanged). The calibration era
chose λ = 0.3; on the 2,690 scored test queries it moved retrieval — analogues sharing an episode
56.3% → 51.9%, recall cap at 5 analogues 60.2% → 65.8% — and did not move the answer: Winkler
15.522% → 15.491%, Diebold-Mariano p = 0.51, coverage 84.8% → 84.7%. **A tie**, so it stays opt-in:
a diversity that changes which states are shown without changing the band is an explanation
feature, not an accuracy one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
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

FETCH_MULTIPLIER = 2
"""Candidates fetched per analogue returned when re-ranking for diversity — paper-qa's ``fetch_k =
2 * k`` (`src/paperqa/docs.py:483`). A pool no larger than the answer leaves MMR nothing to choose
between; a pool much larger admits states too far from the query to be analogues at all."""


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
    realised_at: datetime | None = None
    """When ``forward_return_bps`` became known — the end of its forward window. ``None`` means
    the caller did not say, and :func:`find` then treats the outcome as known at ``as_of``, the
    behaviour before this field existed.

    Added 2026-09-24 after a rival review found the look-ahead this closes: :func:`find` gated only
    ``as_of < decision``, so an observation stamped fewer than ``horizon`` bars before the decision
    carried a forward return that had not happened yet. The live console was safe —
    :func:`corpus_from_closes` stops ``horizon`` bars before the series ends — but a backtest
    passing a full-history corpus and an ``as_of`` cut leaked up to ``horizon`` bars of future.
    AnalogDesk's retrieval embargoes the same way (``jHi = q - H``, ``src/engine/analog.mjs``)."""

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise AnalogueError("observation timestamps must be timezone-aware")
        if self.realised_at is not None and self.realised_at < self.as_of:
            raise AnalogueError("an outcome cannot be realised before its state was observed")


@dataclass(frozen=True)
class ScoreDetails:
    """Why one analogue was retrieved, component by component.

    The shape of mem0's ``score_details`` (`mem0/utils/scoring.py:127-137`, Apache-2.0): the raw
    terms, the gate, and the final score, so a reader can recompute the ranking by hand.
    """

    relevance: float
    """``exp(-distance² / (2 * bandwidth²))``: 1.0 for an identical state, 0.61 at the edge of
    the pool."""
    distance: float
    bandwidth: float
    """The kernel's scale: the distance of the farthest candidate in the pool."""
    max_distance: float
    """The threshold that gated relevance before anything was combined."""
    diversity_penalty: float
    """Highest similarity to an analogue already selected, in the relevance kernel. Zero for the
    first pick, and zero throughout when diversity was not requested."""
    mmr_lambda: float
    """1.0 means relevance alone — the plain nearest-first order."""
    final_score: float
    """``mmr_lambda * relevance - (1 - mmr_lambda) * diversity_penalty``."""
    selection_rank: int
    """1-based order in which this analogue was chosen."""
    pool: int
    """Candidates it was chosen from."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "relevance": round(self.relevance, 4),
            "distance": round(self.distance, 4),
            "bandwidth": round(self.bandwidth, 4),
            "max_distance": self.max_distance,
            "diversity_penalty": round(self.diversity_penalty, 4),
            "mmr_lambda": self.mmr_lambda,
            "final_score": round(self.final_score, 4),
            "selection_rank": self.selection_rank,
            "pool": self.pool,
        }


@dataclass(frozen=True)
class Match:
    """One retrieved analogue, and why it matched."""

    observation: Observation
    distance: float
    contributions: Mapping[str, float]
    """Per-feature share of the distance. Answers "why is this an analogue" with numbers."""
    details: ScoreDetails | None = None
    """The full scoring breakdown, when :func:`find` was asked to ``explain``."""

    @property
    def dominant_feature(self) -> str:
        if not self.contributions:
            return "none"
        return max(self.contributions, key=lambda k: self.contributions[k])

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "as_of": self.observation.as_of.isoformat(),
            "symbol": self.observation.symbol,
            "distance": round(self.distance, 4),
            "forward_return_bps": round(self.observation.forward_return_bps, 2),
            "dominant_feature": self.dominant_feature,
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
        }
        if self.details is not None:
            out["why"] = self.details.as_dict()
        return out


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
    diversity: float | None = None
    """The MMR lambda the matches were re-ranked with; ``None`` is plain nearest-first."""

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
        if self.diversity is not None:
            lines.append(
                f"[analogue] re-ranked for diversity (MMR lambda {self.diversity:g}): each pick "
                f"traded relevance against its similarity to the analogues already chosen"
            )
        return lines

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "usable": self.usable,
            "refused": self.refused,
            "distribution": None if self.distribution is None else self.distribution.as_dict(),
            "matches": [m.as_dict() for m in self.matches],
        }
        if self.diversity is not None:
            out["diversity_lambda"] = self.diversity
        return out


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


def relevance(distance: float, bandwidth: float = 1.0) -> float:
    """The kernel shared by relevance and pairwise similarity: ``exp(-d² / (2 h²))``.

    Monotone in distance, so ranking by it is ranking by distance; bounded in (0, 1], so a
    relevance and a similarity can be subtracted as MMR requires.

    **The bandwidth is the pool's own radius, and that was found, not assumed.** Written first with
    ``h = 1`` (one z-unit), MMR at λ = 0.7 and 0.9 returned *exactly* the nearest-first set on a
    real 2,500-session history (2026-09-25): the fifty nearest states sat within 0.18 z of the
    query, where the kernel is flat — relevance 0.98-1.00, every pairwise similarity above 0.98 —
    so the relevance and penalty terms differed only in the third decimal and nothing re-ranked.
    Scaling by the distance of the farthest candidate in the pool makes the kernel span the pool
    whatever the corpus's density, which is what cosine gives paper-qa for free.
    """
    h = bandwidth if bandwidth > 0 else 1.0
    return math.exp(-0.5 * (distance / h) ** 2)


def mmr_order(
    relevances: Sequence[float],
    similarity: Callable[[int, int], float],
    *,
    k: int,
    mmr_lambda: float,
) -> list[tuple[int, float, float]]:
    """Greedy Maximal Marginal Relevance over candidates already sorted by relevance, descending.

    Returns ``(index, penalty, score)`` per pick, in pick order. A transcription of paper-qa's
    ``max_marginal_relevance_search`` loop (`src/paperqa/llms.py:151-166`, Apache-2.0): the first
    pick is the most relevant, every later pick maximises ``λ * relevance - (1 - λ) *
    max_similarity_to_selected``, already-selected items are excluded. Two departures, both stated:
    the similarity is a callable rather than a precomputed matrix, so a caller whose items are
    price windows need not build one; and ties are broken by the lower index (numpy's ``argmax``
    does the same), which keeps the order reproducible. paper-qa's early return when ``λ >= 1`` or
    the pool is no larger than ``k`` (`llms.py:141-142`) is kept, as the plain relevance order.
    """
    n = len(relevances)
    if n == 0 or k <= 0:
        return []
    if not 0.0 <= mmr_lambda <= 1.0:
        raise AnalogueError(f"MMR lambda must lie in [0, 1], got {mmr_lambda}")
    if n <= k or mmr_lambda >= 1.0:
        return [(i, 0.0, relevances[i]) for i in range(min(n, k))]
    selected = [0]
    picks = [(0, 0.0, mmr_lambda * relevances[0])]
    # max similarity of every candidate to the selected set, updated incrementally per pick
    nearest = [similarity(i, 0) for i in range(n)]
    chosen = [False] * n
    chosen[0] = True
    while len(selected) < k:
        best, best_score = -1, -math.inf
        for i in range(n):
            if chosen[i]:
                continue
            score = mmr_lambda * relevances[i] - (1.0 - mmr_lambda) * nearest[i]
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        chosen[best] = True
        picks.append((best, nearest[best], best_score))
        for i in range(n):
            if not chosen[i]:
                nearest[i] = max(nearest[i], similarity(i, best))
    return picks


def find(
    *,
    query: Mapping[str, float],
    corpus: Sequence[Observation],
    as_of: datetime,
    weights: Mapping[str, float] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    limit: int = 50,
    diversity: float | None = None,
    explain: bool = False,
) -> AnalogueReport:
    """Retrieve historical states resembling ``query`` and report what followed them.

    ``as_of`` is the decision instant: no observation stamped at or after it can be returned, so a
    retrieval run during a backtest cannot see its own future. This is the same gate
    :func:`argus.market.evidence.gather` applies to evidence, and for the same reason.

    ``diversity`` is an MMR lambda in [0, 1]: ``None`` (the default) or 1.0 keeps the plain
    nearest-first order; lower values trade relevance for spread across distinct states, drawing
    from the ``FETCH_MULTIPLIER * limit`` nearest candidates inside ``max_distance``. ``explain``
    attaches a :class:`ScoreDetails` to every match.
    """
    if as_of.tzinfo is None:
        raise AnalogueError("as_of must be timezone-aware; a naive clock cannot bound a search")
    if diversity is not None and not 0.0 <= diversity <= 1.0:
        raise AnalogueError(f"diversity is an MMR lambda in [0, 1], got {diversity}")
    keys = [k for k in query if any(k in o.features for o in corpus)]
    if not keys:
        return AnalogueReport((), None, refused="no query feature appears in the corpus")

    # A state before the decision is not enough: its outcome must also be known by then.
    eligible = [o for o in corpus
                if o.as_of < as_of and (o.realised_at or o.as_of) <= as_of]
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
    # Each scored observation's weighted z-coordinates, keyed by ``id(match)``, so pairwise
    # similarity for MMR is measured in exactly the space the query distance was.
    position: dict[int, list[float]] = {}
    for observation in eligible:
        contributions: dict[str, float] = {}
        total = 0.0
        usable = True
        coords: list[float] = []
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
            coords.append(math.sqrt(max(w, 0.0)) * (float(observation.features[key]) - mean) / sd)
        if not usable:
            continue
        distance = math.sqrt(total)
        if distance <= max_distance:
            share = {k: (v / total if total > 0 else 0.0) for k, v in contributions.items()}
            match = Match(observation, distance, share)
            scored.append(match)
            position[id(match)] = coords

    scored.sort(key=lambda m: m.distance)
    lam = 1.0 if diversity is None else diversity
    pool = scored[:limit] if lam >= 1.0 else scored[:FETCH_MULTIPLIER * limit]
    bandwidth = pool[-1].distance if pool and pool[-1].distance > 0 else 1.0
    rels = [relevance(m.distance, bandwidth) for m in pool]

    def similarity(i: int, j: int) -> float:
        a, b = position[id(pool[i])], position[id(pool[j])]
        gap = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
        return relevance(gap, bandwidth)

    picks = mmr_order(rels, similarity, k=limit, mmr_lambda=lam)
    chosen: list[Match] = []
    for rank, (index, penalty, final) in enumerate(picks, start=1):
        base = pool[index]
        if explain:
            base = Match(base.observation, base.distance, base.contributions, ScoreDetails(
                relevance=rels[index], distance=base.distance, bandwidth=bandwidth,
                max_distance=max_distance,
                diversity_penalty=penalty, mmr_lambda=lam, final_score=final,
                selection_rank=rank, pool=len(pool),
            ))
        chosen.append(base)
    matches = tuple(chosen)

    if len(matches) < MIN_ANALOGUES:
        return AnalogueReport(
            matches, None,
            refused=(
                f"{len(matches)} analogue(s) within distance {max_distance}; "
                f"{MIN_ANALOGUES} are needed. A distribution over fewer is an anecdote with "
                f"error bars"
            ),
            diversity=diversity,
        )

    distribution = Distribution(
        returns_bps=tuple(m.observation.forward_return_bps for m in matches),
        effective_n=_collapse_overlapping(matches),
    )
    return AnalogueReport(matches, distribution, diversity=diversity)


def explain_lines(report: AnalogueReport, *, top: int = 3) -> list[str]:
    """Console lines saying why the leading analogues were chosen, from their :class:`ScoreDetails`.

    Empty unless :func:`find` ran with ``explain=True``, so a caller cannot print an explanation
    that was never computed. Each line names the date, the distance and its relevance, the feature
    that dominated the distance, and — when diversity re-ranked the set — the penalty paid.
    """
    out: list[str] = []
    for match in report.matches[:top]:
        d = match.details
        if d is None:
            continue
        share = match.contributions.get(match.dominant_feature, 0.0)
        line = (
            f"Why {match.observation.as_of:%Y-%m-%d %H:%M}: pick {d.selection_rank} of {d.pool}, "
            f"distance {d.distance:.2f} (relevance {d.relevance:.2f}, cut-off {d.max_distance:g}), "
            f"{share:.0%} of the gap from {match.dominant_feature}"
        )
        if d.mmr_lambda < 1.0:
            line += (f"; similarity to analogues already chosen {d.diversity_penalty:.2f}, "
                     f"MMR score {d.final_score:.2f} at lambda {d.mmr_lambda:g}")
        out.append(line)
    return out


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
            realised_at=closes[i + horizon][0],
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
    "FETCH_MULTIPLIER",
    "MIN_ANALOGUES",
    "OVERLAP_WINDOW",
    "AnalogueError",
    "AnalogueReport",
    "Distribution",
    "Match",
    "Observation",
    "ScoreDetails",
    "corpus_from_closes",
    "current_state",
    "explain_lines",
    "find",
    "mmr_order",
    "relevance",
]
