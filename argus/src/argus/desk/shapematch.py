""""When has this happened before?" — matched on the path, not on a summary of it.

Track 3's sub-theme asks, in its own words: *before opening a position, how does AI **retrieve
historically similar scenarios**?* ARGUS answers it twice, deliberately, and the two answers are
not interchangeable:

* :mod:`argus.desk.analogue` matches a **state vector** — trailing return, realised volatility —
  z-scored across the corpus, and returns the distribution of what followed. That is the right
  retrieval when the question is "what kind of market is this".
* This module matches the **path itself**. Two windows can carry an identical trailing return and
  an identical realised volatility while one is a steady grind and the other a crash that fully
  retraced. A state vector cannot tell those apart, because it has already thrown the ordering
  away; a trader who says *similar* usually means the ordering.

`desk/stress.py` answers a third, neighbouring question very well — it takes the realised
distribution of an instrument's own moves and can say "a 5.1% two-hour fall is the 1st percentile of
718 observed windows, which happened 7 times in 90 days, most recently on 2026-08-29". That is a
**magnitude** match, and it is the right answer to "how bad can it get". It is not a shape.

**The method is the matrix profile's distance, computed directly.** For every past window of the
same length, z-normalise both it and the query and take the Euclidean distance. Z-normalising is
what makes it a shape comparison rather than a level one: a 3% drift from $118 and the same drift
from $440 are the same pattern, and an un-normalised distance would call them unrelated. This is the
metric `stumpy` builds its matrix profile from (`stumpy/core.py:_calculate_squared_distance`, which
computes it from the sliding means and standard deviations rather than element-wise); we compute it
directly rather than take the dependency, because ARGUS is pure Python and the series here are a
thousand bars — a brute-force scan is a million operations, which is nothing, while numpy and numba
are not nothing.

**The exclusion zone is not an optimisation, it is the whole correctness of the thing.** Without it
the nearest neighbour of any window is the window shifted by one bar, which overlaps it almost
entirely and is a tautology dressed as a finding. `stumpy` excludes ``m/4`` either side by default
(`stumpy/config.py:STUMPY_EXCL_ZONE_DENOM = 4`); we exclude a **full window length**, which is
stricter and means no returned match shares a single bar with the query or with any other match. A
"most similar historical moment" that overlaps the present moment is the analogue of look-ahead, and
this module refuses to produce one. :mod:`argus.desk.analogue` enforces the same property on its own
corpus by collapsing overlapping matches into episodes before it counts them; the mechanism differs
because the unit differs, the rule does not.

**What happened next is the point.** A match with no forward outcome is trivia. Each analogue
carries the move over the same horizon that *followed* it, so the report is a distribution of
outcomes conditioned on the pattern rather than on the calendar — and the forward window is taken
strictly after the match ends, so nothing in the outcome is inside the thing being matched.

**It refuses rather than reassures.** Fewer than :data:`MIN_ANALOGUES` non-overlapping matches, or a
best distance worse than :data:`WEAK_MATCH_DISTANCE`, and the report says the present has no
precedent in the window examined instead of returning the least-bad row. "Nothing like this has
happened before" is a finding; the fifth-nearest of five poor matches presented as a precedent is a
fabrication.

**Diversity among the analogues, opt-in (added 2026-09-25).** The exclusion zone guarantees no two
analogues share a bar; it does not stop five of them being the same *shape*, one sharp V repeated
across five months, which is one precedent counted five times in a different way.
``find(diversity=λ)`` re-ranks the non-overlapping candidates with Maximal Marginal Relevance
through :func:`argus.desk.analogue.mmr_order`, a transcription of paper-qa's loop (Apache-2.0,
`src/paperqa/llms.py:151-166`). Relevance is the Pearson correlation ``rho = 1 - d² / 2`` of a
candidate with the query — the same quantity :func:`_distances` already computes — and similarity
between two candidates is their own correlation, so the penalty is paid in the metric the match was
made in and ``λ = 1`` is exactly the old order. The pool is the ``2 * top`` nearest
non-overlapping windows (paper-qa's ``fetch_k = 2 * k``, `docs.py:483`), so the exclusion zone is
applied first and never traded away. The null calibration is untouched: it scans for the single
best match, and MMR's first pick is always the nearest. ``explain=True`` records each pick's
relevance, penalty and score in the shape of mem0's ``score_details`` (Apache-2.0,
`mem0/utils/scoring.py:127-137`). **Off by default, because on the test it lost.** On AnalogDesk's
pre-registered grid (:mod:`argus.eval.retrieval_diversity`, ``data/retrieval_diversity.json``,
2026-09-25) the calibration era chose λ = 0.9, and on 2,698 test queries it was worse: Winkler
16.396% → 16.417%, +0.021 points, Diebold-Mariano p = 0.028 over 38 clustered dates — small but
significant. Lower lambdas cut shape near-duplicates sharply (40.1% of analogues correlated ≥ 0.9
with an earlier pick → 14.2% at λ = 0.3) without improving the band either (16.610% at λ = 0.3).
The exclusion zone already removes the redundancy that matters to the outcome distribution.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

from argus.desk.analogue import mmr_order

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "shape_matches.json"

MIN_ANALOGUES = 3
"""Fewest non-overlapping matches before a distribution of outcomes is reported.

Two analogues give a range and no shape. Below this the report names the matches it found and
declines to summarise what followed them.
"""

WEAK_MATCH_DISTANCE = 1.0
"""Per-bar z-normalised distance beyond which a "match" is not one. **Derived, then measured.**

The distance used here is a pure function of the correlation between the two z-normalised windows:
``stumpy/core.py:1118`` computes ``D_squared = 2 * m * (1 - rho)``, and dividing by ``m`` as this
module does gives ``d = sqrt(2 * (1 - rho))``. So the scale is fixed and knowable rather than a
matter of taste: identical shapes are 0, **uncorrelated shapes are sqrt(2) = 1.414**, and perfectly
inverted ones are exactly 2.

This constant was first written as 2.0, with a docstring claiming uncorrelated shapes "land above
2.0". That was a guess and it was wrong by the width of the entire scale: 2.0 is the *maximum
possible* distance, so the gate could never fire — every match, including a perfectly inverted one,
passed it. Reading the reference implementation is what found it. 1.0 is ``rho = 0.5``: below it the
match explains more than half the variance of the query's shape.

Measured on our own hourly rTokens (60 days, 24-bar windows) the nearest match lands at 0.29 to
0.42 — comfortably inside this — and 2,000 independent random pairs averaged 1.405 against the
sqrt(2) the algebra predicts. But see :data:`NULL_TRIALS`: passing this threshold is necessary and
nowhere near sufficient.
"""

NULL_TRIALS = 50
"""Shuffled-return scans used to calibrate what "close" means **on this series**.

The finding that forced this into existence: on real 60-day hourly rToken series the best 24-bar
shape match sits at distance 0.35-0.42, and when the same series' returns are randomly reordered —
same bars, same volatility, shape structure destroyed — the best match of the shuffled path lands at
a median of 0.40 (NVDAUSDT) and 0.44 (TSLAUSDT), reaching 0.20 at its closest. A shape at least as
close as the real one arises from noise **47% and 37% of the time**.

So a fixed distance threshold, however carefully derived, cannot separate a precedent from a
coincidence: scanning ~1,400 candidate windows for a minimum guarantees a small number whether or
not the series has any repeating structure at all. Every retrieval system in our corpus stops at the
fixed threshold, which is exactly why they can always show you five analogues.

The null reorders the series' own returns, preserving the marginal distribution and the volatility
while destroying the ordering — the ordering being the only thing a shape match looks at.
:data:`NULL_ALPHA` is the share of those scans that may beat the observed match before the match is
reported as indistinguishable from chance.
"""

NULL_ALPHA = 0.05
"""How often noise may produce a match this close before the precedent is refused."""


class AnalogueError(ValueError):
    """Raised rather than returning a precedent that does not exist."""


def znormalise(values: Sequence[float]) -> list[float]:
    """Zero mean, unit standard deviation. A flat window has no shape and returns zeros.

    Population standard deviation, not sample: the window is the entire thing being described, not
    a draw from something larger, and using the sample form would make short windows look more
    variable than they are.
    """
    if not values:
        return []
    mean = fmean(values)
    spread = pstdev(values)
    if spread <= 0:
        # A perfectly flat window. Its shape is "no shape"; returning zeros makes it maximally
        # distant from anything with structure, which is the honest comparison.
        return [0.0] * len(values)
    return [(v - mean) / spread for v in values]


def distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Z-normalised Euclidean distance per bar, so windows of different lengths compare.

    Divided by ``sqrt(len)`` rather than reported raw: without it a 48-bar window scores worse than
    a 12-bar one for the same quality of match, purely because it has more terms, and any threshold
    would then have to be re-derived per window length.
    """
    if len(left) != len(right):
        raise AnalogueError(f"windows differ in length: {len(left)} vs {len(right)}")
    if not left:
        raise AnalogueError("cannot compare empty windows")
    a, b = znormalise(left), znormalise(right)
    total = sum((x - y) ** 2 for x, y in zip(a, b, strict=True))
    return math.sqrt(total) / math.sqrt(len(left))


@dataclass(frozen=True, slots=True)
class Analogue:
    """One past window that resembled the present, and what happened after it."""

    ends_at: datetime
    distance: float
    forward_pct: float | None
    """The move over the forward horizon that followed this window. ``None`` when the series ends
    before the horizon does — a match at the very end of history has no outcome, and inventing a
    truncated one would bias the distribution toward whatever the last bars did."""

    start_index: int

    why: Mapping[str, float] | None = None
    """Relevance, diversity penalty, MMR score and pick order, when :func:`find` was asked to
    ``explain``. Absent otherwise, so an unexplained report serialises exactly as it always has."""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ends_at": self.ends_at.isoformat(),
            "distance": round(self.distance, 4),
            "forward_pct": None if self.forward_pct is None else round(self.forward_pct, 4),
        }
        if self.why is not None:
            out["why"] = {k: round(v, 4) for k, v in self.why.items()}
        return out


@dataclass(frozen=True, slots=True)
class AnalogueReport:
    """What the present most resembles, and how that resolved."""

    symbol: str
    window: int
    horizon: int
    searched: int
    analogues: tuple[Analogue, ...]

    null_trials: int = 0
    """Shuffled-return scans run to calibrate the observed distance. Zero means uncalibrated."""

    null_better: int = 0
    """How many of those scans produced a best match at least as close as the observed one."""

    null_median: float | None = None
    """Median best distance under the null. The number the observed distance has to beat."""

    @property
    def outcomes(self) -> tuple[float, ...]:
        return tuple(a.forward_pct for a in self.analogues if a.forward_pct is not None)

    @property
    def best(self) -> Analogue | None:
        return self.analogues[0] if self.analogues else None

    @property
    def has_precedent(self) -> bool:
        """Close, non-overlapping, **and closer than this series' own noise produces.**

        The third clause is the one that matters and the one every system we read omits. An
        uncalibrated report never claims a precedent: a search whose null was not run has not been
        shown to have found anything, and defaulting that to "yes" is how five coincidences become
        a distribution.
        """
        p = self.null_p
        return (
            len(self.analogues) >= MIN_ANALOGUES
            and self.best is not None
            and self.best.distance <= WEAK_MATCH_DISTANCE
            and p is not None
            and p <= NULL_ALPHA
        )

    @property
    def null_p(self) -> float | None:
        """Share of shuffled-return scans that matched the query at least as closely.

        ``None`` when the calibration was not run — which is not the same as passing it, and
        :attr:`has_precedent` treats it as a refusal rather than a pass.
        """
        if self.null_trials <= 0:
            return None
        return self.null_better / self.null_trials

    @property
    def median_outcome(self) -> float | None:
        return median(self.outcomes) if self.outcomes else None

    @property
    def upside_share(self) -> float | None:
        """Share of analogues that rose afterwards. Not a forecast — a base rate."""
        rows = self.outcomes
        return (sum(1 for x in rows if x > 0) / len(rows)) if rows else None

    @property
    def verdict(self) -> str:
        if not self.analogues:
            return (
                f"No non-overlapping precedent for the last {self.window} bars of {self.symbol} in "
                f"{self.searched} candidate window(s). The present has no analogue here"
            )
        best = self.best
        assert best is not None
        if not self.has_precedent:
            p = self.null_p
            if p is None:
                return (
                    f"Uncalibrated: {len(self.analogues)} match(es), nearest at distance "
                    f"{best.distance:.2f}, and no null was run. A distance alone cannot say "
                    f"whether that is a precedent or the arithmetic consequence of scanning "
                    f"{self.searched} windows for a minimum"
                )
            if best.distance <= WEAK_MATCH_DISTANCE and p > NULL_ALPHA:
                assert self.null_median is not None
                return (
                    f"No precedent: the nearest match is at distance {best.distance:.2f}, but "
                    f"reordering this series' own returns produces a match at least that close in "
                    f"{p:.0%} of {self.null_trials} scans (median {self.null_median:.2f}). The "
                    f"shape is not distinguishable from chance, so what followed these "
                    f"{len(self.analogues)} window(s) is not evidence about what follows now"
                )
            return (
                f"Weak precedent only: {len(self.analogues)} match(es), nearest at distance "
                f"{best.distance:.2f} against a {WEAK_MATCH_DISTANCE:.1f} threshold. Reported as "
                f"insufficient rather than summarised — the nearest of several poor matches is not "
                f"a precedent"
            )
        outcomes = self.outcomes
        if not outcomes:
            return (
                f"{len(self.analogues)} close match(es) found, none with a complete forward "
                f"window. What followed them is unknown rather than neutral"
            )
        return (
            f"The last {self.window} bars of {self.symbol} most resemble the window ending "
            f"{best.ends_at:%Y-%m-%d %H:%M} (distance {best.distance:.2f}). Across "
            f"{len(outcomes)} close analogue(s), the next {self.horizon} bar(s) moved a median of "
            f"{self.median_outcome:+.2f}%, up {self.upside_share:.0%} of the time — a base rate "
            f"from {len(outcomes)} observations, not a forecast. Shuffled returns from this same "
            f"series beat that match in only {self.null_p:.0%} of {self.null_trials} scans"
        )

    def render(self) -> str:
        lines = [
            f"HISTORICAL ANALOGUES — {self.symbol}, {self.window}-bar shape, "
            f"{self.horizon}-bar outcome",
            "",
        ]
        for item in self.analogues:
            forward = "no outcome" if item.forward_pct is None else f"{item.forward_pct:+.2f}%"
            lines.append(
                f"  {item.ends_at:%Y-%m-%d %H:%M}  distance {item.distance:.3f}  then {forward}"
            )
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "symbol": self.symbol,
            "window": self.window,
            "horizon": self.horizon,
            "searched": self.searched,
            "has_precedent": self.has_precedent,
            "null_trials": self.null_trials,
            "null_p": self.null_p,
            "null_median_distance": (
                None if self.null_median is None else round(self.null_median, 4)
            ),
            "median_outcome_pct": (
                None if self.median_outcome is None else round(self.median_outcome, 4)
            ),
            "upside_share": None if self.upside_share is None else round(self.upside_share, 4),
            "analogues": [a.as_dict() for a in self.analogues],
            "verdict": self.verdict,
        }


def _scan(
    closes: Sequence[float],
    stamps: Sequence[datetime] | None,
    *,
    window: int,
    horizon: int,
    top: int,
    diversity: float | None = None,
    explain: bool = False,
) -> tuple[list[Analogue], int]:
    """One brute-force pass: every candidate window scored against the final one.

    Separated from :func:`find` because the null calibration runs exactly this scan over reordered
    returns, and a null computed by a second, subtly different code path would be measuring the
    difference between the two implementations rather than the difference between signal and noise.
    """
    query_start = len(closes) - window
    # A candidate must END before the query begins, and leave room for its own forward window.
    last_start = query_start - horizon - window
    starts = range(0, max(0, last_start) + 1) if last_start >= 0 else range(0)
    distances = _distances(closes, window, starts)

    def analogue(start: int, dist: float) -> Analogue:
        forward_end = start + window + horizon - 1
        forward = None
        if forward_end < query_start:
            base = closes[start + window - 1]
            if base > 0:
                forward = (closes[forward_end] - base) / base * 100.0
        return Analogue(
            ends_at=(
                stamps[start + window - 1] if stamps is not None
                else datetime.fromtimestamp(0, UTC)
            ),
            distance=dist,
            forward_pct=forward,
            start_index=start,
        )

    ranked = sorted(zip(distances, starts, strict=True))

    # Exclusion zone: a full window either side, so no two reported analogues share a bar. Without
    # this the top-5 is one event reported five times, and the outcome distribution is one outcome
    # counted five times.
    lam = 1.0 if diversity is None else diversity
    wanted = top if lam >= 1.0 else 2 * top
    chosen: list[Analogue] = []
    for dist, start in ranked:
        if all(abs(start - kept.start_index) >= window for kept in chosen):
            chosen.append(analogue(start, dist))
        if len(chosen) == wanted:
            break
    if lam >= 1.0 and not explain:
        return chosen, len(ranked)

    # MMR over the non-overlapping pool. rho = 1 - d^2 / 2 is the correlation with the query, and
    # the pairwise similarity is the correlation between two candidate windows.
    rels = [1.0 - 0.5 * a.distance * a.distance for a in chosen]
    pool = [closes[a.start_index:a.start_index + window] for a in chosen]
    cache: dict[tuple[int, int], float] = {}

    def similarity(i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in cache:
            cache[key] = _correlation(pool[i], pool[j])
        return cache[key]

    picks = mmr_order(rels, similarity, k=top, mmr_lambda=lam)
    out: list[Analogue] = []
    for rank, (index, penalty, score) in enumerate(picks, start=1):
        base = chosen[index]
        why = None
        if explain:
            why = {"relevance_rho": rels[index], "diversity_penalty": penalty,
                   "mmr_lambda": lam, "final_score": score, "selection_rank": float(rank),
                   "pool": float(len(chosen))}
        out.append(Analogue(ends_at=base.ends_at, distance=base.distance,
                            forward_pct=base.forward_pct, start_index=base.start_index, why=why))
    return out, len(ranked)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Pearson correlation of two equal-length windows, with :func:`znormalise`'s flat-window rule:
    a flat window has correlation 0 with anything shaped, and 1 with another flat window."""
    n = len(left)
    lm, rm = math.fsum(left) / n, math.fsum(right) / n
    ld = [v - lm for v in left]
    rd = [v - rm for v in right]
    lss, rss = math.fsum(d * d for d in ld), math.fsum(d * d for d in rd)
    if lss <= 0.0 or rss <= 0.0:
        return 1.0 if lss <= 0.0 and rss <= 0.0 else 0.0
    return math.fsum(a * b for a, b in zip(ld, rd, strict=True)) / math.sqrt(lss * rss)


def _distances(closes: Sequence[float], window: int, starts: Sequence[int]) -> list[float]:
    """:func:`distance` from the final ``window`` bars to the window at each start, all at once.

    The same quantity by a cheaper route. For z-normalised windows the per-bar Euclidean distance
    is ``sqrt(2 * (1 - rho))``, where ``rho`` is the Pearson correlation of the raw windows — the
    identity `stumpy/core.py:_calculate_squared_distance` computes the matrix profile from, with
    its sliding means and standard deviations. Scoring each candidate through :func:`distance`
    re-normalised the query every time with the exact-arithmetic `statistics` functions, and 51
    scans of ninety days took 17s on 2026-09-24 — too slow for a question someone is waiting on.
    A flat window keeps :func:`znormalise`'s convention (all zeros): 1.0 against any shaped window,
    0.0 against another flat one. `tests/test_shapematch.py` pins agreement with :func:`distance`
    on real series to 1e-9.
    """
    query = closes[-window:]
    q_mean = math.fsum(query) / window
    q_dev = [v - q_mean for v in query]
    q_ss = math.fsum(d * d for d in q_dev)
    q_flat = q_ss <= 0.0
    out: list[float] = []
    for start in starts:
        candidate = closes[start:start + window]
        c_mean = math.fsum(candidate) / window
        c_dev = [v - c_mean for v in candidate]
        c_ss = math.fsum(d * d for d in c_dev)
        if q_flat or c_ss <= 0.0:
            out.append(0.0 if q_flat and c_ss <= 0.0 else 1.0)
            continue
        rho = math.fsum(a * b for a, b in zip(q_dev, c_dev, strict=True)) / math.sqrt(q_ss * c_ss)
        out.append(math.sqrt(max(0.0, 2.0 * (1.0 - rho))))
    return out


def shuffled_closes(closes: Sequence[float], rng: random.Random) -> list[float]:
    """The same series' returns in a random order, rebuilt into a price path.

    A permutation null rather than a Gaussian one on purpose: it keeps every return the series
    actually printed — the fat tails, the overnight gaps, the flat weekend hours — and destroys only
    their order. What survives is exactly the thing a shape match does not look at, so any distance
    advantage the real path holds over these paths is attributable to its ordering.
    """
    returns = [closes[i + 1] / closes[i] for i in range(len(closes) - 1) if closes[i] > 0]
    rng.shuffle(returns)
    out = [closes[0]]
    for ratio in returns:
        out.append(out[-1] * ratio)
    return out


def find(
    series: Sequence[tuple[datetime, float]],
    *,
    symbol: str,
    window: int = 24,
    horizon: int = 24,
    top: int = 5,
    trials: int = NULL_TRIALS,
    seed: int = 20260914,
    diversity: float | None = None,
    explain: bool = False,
) -> AnalogueReport:
    """The ``top`` past windows most resembling the final ``window`` bars, and what followed each.

    The query is always the **most recent** window, because the question is about now. Candidates
    are every earlier window that ends at least ``horizon`` bars before the query begins, so a match
    can never overlap the present and its forward window can never reach into it.

    ``trials`` shuffled-return scans then say whether the best distance found means anything on this
    series; ``trials=0`` skips that and the report says so rather than claiming a precedent. The
    seed is fixed and stated so the p-value is reproducible — a calibration that moves between runs
    is a number a reader cannot check.

    ``diversity`` is an MMR lambda in [0, 1] (``None``, the default, keeps nearest-first);
    ``explain`` records why each analogue was picked. Neither touches the null calibration.
    """
    if diversity is not None and not 0.0 <= diversity <= 1.0:
        raise AnalogueError(f"diversity is an MMR lambda in [0, 1], got {diversity}")
    if window < 3:
        raise AnalogueError(f"a {window}-bar window has no shape to match")
    if horizon < 1:
        raise AnalogueError("the forward horizon must be at least one bar")
    if len(series) < window * 2 + horizon:
        raise AnalogueError(
            f"{len(series)} bar(s) cannot support a {window}-bar query with a {horizon}-bar "
            f"outcome and a non-overlapping candidate; {window * 2 + horizon} is the minimum"
        )

    closes = [c for _, c in series]
    stamps = [t for t, _ in series]
    chosen, searched = _scan(closes, stamps, window=window, horizon=horizon, top=top,
                             diversity=diversity, explain=explain)

    null_better = 0
    null_median: float | None = None
    if trials > 0 and chosen:
        rng = random.Random(seed)
        observed = chosen[0].distance
        bests: list[float] = []
        for _ in range(trials):
            shuffled, _count = _scan(
                shuffled_closes(closes, rng), None, window=window, horizon=horizon, top=1,
            )
            if shuffled:
                bests.append(shuffled[0].distance)
        null_better = sum(1 for d in bests if d <= observed)
        null_median = median(bests) if bests else None
        trials = len(bests)

    return AnalogueReport(
        symbol=symbol, window=window, horizon=horizon, searched=searched,
        analogues=tuple(chosen),
        null_trials=trials if chosen else 0,
        null_better=null_better,
        null_median=null_median,
    )


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.market.history import CandleType, fetch_range

    parser = argparse.ArgumentParser(description="when has this happened before?")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()

    candles = fetch_range(
        args.symbol, days=args.days, interval="1H", candle_type=CandleType.MARKET
    )
    series = [(c.ts, float(c.close)) for c in candles]
    report = find(
        series, symbol=args.symbol, window=args.window, horizon=args.horizon,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print("\nwritten to " + str(REPORT_PATH))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "MIN_ANALOGUES",
    "NULL_ALPHA",
    "NULL_TRIALS",
    "WEAK_MATCH_DISTANCE",
    "Analogue",
    "AnalogueError",
    "AnalogueReport",
    "distance",
    "find",
    "znormalise",
]
