"""Six headlines, or one headline six times? The question an evidence count cannot answer.

`agents/analysts.SourceIndependenceGraph` discounts *analysts* who agree after reading the same
article — five opinions from one source is one piece of evidence. It does not look at the evidence
itself, so a panel handed six near-identical wire copies of the same story sees six items and a
research report prints "6 item(s) across 1 channel(s)". That count is the input to a sentiment
reading, a novelty score and a confidence, and all three are wrong in the same direction.

**This is the named fix for the weakest Track 2 sub-theme we have.** The sentiment analyst is
DEMOTED in our own capability register, and its docstring says the reason is the feed rather than
the reasoning. That is true and it is not the whole story: even with a live feed, a sentiment
signal built on an uncounted pile of syndicated copies measures republication volume. The question
worth asking is the one this module answers — *is this new information, or four thousand copies of
the same story?*

**Near-duplicate detection by shingling and Jaccard similarity**, exactly, not by MinHash. MinHash
exists to approximate Jaccard when the corpus is too large to compare pairwise; a decision cycle
sees tens of items, where the exact computation is instant and the approximation would add a false
negative rate for no gain. The choice is stated because it inverts at scale and a reader should
know which regime this is built for.

**The one prior implementation in the corpus solves a different problem, and the differences are
deliberate.** `research/repos/FinNLP/finnlp/data_engineering/data_cleaning.py:52-70` uses MinHash
with 128 permutations and LSH at a 0.8 threshold over **character** 13-grams, to strip train/test
contamination from a training corpus. Three of its four choices are right for that job and wrong
for this one:

* MinHash and LSH are for a corpus too large to compare pairwise. Tens of headlines are not.
* Character n-grams suit code and mixed text; **word** shingles are the standard for news, where a
  rewrite changes words rather than characters.
* A 0.8 threshold finds near-exact copies, which is what contamination is. Syndication rewrites
  drift much further, hence 0.55 here.
* And their purpose is to **delete** duplicates from a dataset. Ours is to **count** them: nothing
  is dropped, because a reader needs to know that six items were six copies rather than silently
  receiving one.

Disclosed rather than implied: this module was written before that grep returned, so the prior art
was read afterwards and confirmed the design rather than informing it.

**Three signals, not one.** Duplication alone is innocent — wire services republish by design:

* **Distinct stories.** The count that should have been used wherever the raw item count is used.
* **Coordination.** Many *independent* sources carrying near-identical text inside a short window
  is the shape of a promoted narrative rather than of independent reporting. A single outlet
  repeating itself is not coordination, and the measure separates them.
* **Velocity.** How fast the copies arrive. A story that lands everywhere within minutes is being
  pushed; one that spreads over a day is being picked up.

None of these is a verdict. A promoted narrative can be true and a slow one can be wrong. They are
reported as properties of the evidence so that a confidence built on volume can be corrected, and
`Cluster.verdict` says what each one licenses rather than implying a trade.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

SHINGLE = 4
"""Words per shingle. Four is the conventional choice for news-length text.

Shorter shingles match on common phrasing and call unrelated stories duplicates; longer ones miss a
rewrite that changed a few words, which is exactly what syndication does. Stated here because the
threshold below is only meaningful with respect to it.
"""

DUPLICATE_AT = 0.55
"""Jaccard similarity at or above which two items are treated as the same story.

Calibrated against the failure that matters: a wire copy rewritten with a new lead still shares
most of its four-word shingles, while two genuinely different stories about one company share the
company name and little else. Set too low, independent reporting collapses into one cluster and the
evidence base looks thinner than it is; too high and syndication passes as novelty. Both errors are
possible and only one of them flatters, which is why this leans toward calling copies copies.
"""

COORDINATION_WINDOW = timedelta(hours=2)
MIN_COORDINATED_SOURCES = 3
"""Distinct sources carrying one story inside the window before it reads as coordinated.

Two outlets agreeing is ordinary. The threshold is on *distinct sources*, not items, because one
outlet publishing the same story three times is a busy newsroom rather than a campaign.
"""

_WORD = re.compile(r"[a-z0-9]+")


class NoveltyError(ValueError):
    """Raised rather than reporting a novelty figure computed from nothing."""


def shingles(text: str, *, size: int = SHINGLE) -> frozenset[str]:
    """Overlapping word n-grams, lowercased and stripped of punctuation.

    A text shorter than one shingle returns its whole word set, so a one-line headline is still
    comparable rather than silently unmatched against everything.
    """
    words = _WORD.findall(text.lower())
    if not words:
        return frozenset()
    if len(words) < size:
        return frozenset([" ".join(words)])
    return frozenset(
        " ".join(words[i:i + size]) for i in range(len(words) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Exact Jaccard similarity. Two empty texts are not similar, they are uncomparable."""
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


@dataclass(frozen=True, slots=True)
class Cluster:
    """One story, and every item carrying it."""

    representative: str
    """The earliest item's claim. Earliest rather than longest: the first to publish is the one
    the others may have copied, and picking the longest would often pick a later aggregation."""

    item_ids: tuple[str, ...]
    sources: tuple[str, ...]
    first_seen: datetime
    last_seen: datetime

    @property
    def size(self) -> int:
        return len(self.item_ids)

    @property
    def distinct_sources(self) -> int:
        return len(set(self.sources))

    @property
    def span(self) -> timedelta:
        return self.last_seen - self.first_seen

    @property
    def coordinated(self) -> bool:
        """Several *distinct* sources carrying one story inside a short window."""
        return (
            self.distinct_sources >= MIN_COORDINATED_SOURCES
            and self.span <= COORDINATION_WINDOW
        )

    @property
    def velocity(self) -> float | None:
        """Copies per hour. None for a single item, which has no rate."""
        if self.size < 2:
            return None
        hours = self.span.total_seconds() / 3600
        return float(self.size) if hours <= 0 else self.size / hours

    @property
    def verdict(self) -> str:
        if self.size == 1:
            return f"one item, no duplicate found: {self.representative[:70]}"
        if self.coordinated:
            return (
                f"{self.size} near-identical item(s) from {self.distinct_sources} distinct "
                f"source(s) within {self.span}. That is the shape of a promoted narrative rather "
                f"than of independent reporting — which does not make it false, only not "
                f"{self.size} pieces of evidence"
            )
        return (
            f"{self.size} near-identical item(s) from {self.distinct_sources} source(s) over "
            f"{self.span}: syndication. One story, not {self.size}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "representative": self.representative[:200],
            "size": self.size,
            "items": list(self.item_ids),
            "sources": sorted(set(self.sources)),
            "distinct_sources": self.distinct_sources,
            "first_seen": self.first_seen.isoformat(),
            "span_hours": round(self.span.total_seconds() / 3600, 3),
            "coordinated": self.coordinated,
            "velocity_per_hour": None if self.velocity is None else round(self.velocity, 3),
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class NoveltyReport:
    """What the evidence base actually contains, once copies are counted once."""

    items: int
    clusters: tuple[Cluster, ...]

    @property
    def distinct_stories(self) -> int:
        return len(self.clusters)

    @property
    def duplication_ratio(self) -> float:
        """Items per distinct story. 1.0 means nothing was a copy."""
        return self.items / self.distinct_stories if self.distinct_stories else 0.0

    @property
    def coordinated(self) -> tuple[Cluster, ...]:
        return tuple(c for c in self.clusters if c.coordinated)

    @property
    def effective_items(self) -> int:
        """The count that should be used wherever the raw item count is used today."""
        return self.distinct_stories

    @property
    def verdict(self) -> str:
        if self.items == 0:
            return "no evidence to examine"
        if self.distinct_stories == self.items:
            return (
                f"{self.items} item(s), {self.items} distinct story(ies): nothing here is a copy "
                f"of anything else, so the item count is the evidence count"
            )
        head = (
            f"{self.items} item(s) carry {self.distinct_stories} distinct story(ies) "
            f"({self.duplication_ratio:.1f} items per story). Any confidence built on the raw "
            f"count is overstated by that factor"
        )
        if self.coordinated:
            return (
                f"{head}. {len(self.coordinated)} of them arrived from several distinct sources "
                f"inside {COORDINATION_WINDOW}, which is a promoted narrative rather than "
                f"corroboration"
            )
        return head

    def render(self) -> str:
        lines = [
            f"NOVELTY — {self.items} item(s), {self.distinct_stories} distinct story(ies)",
            "",
        ]
        lines.extend(f"  {c.verdict}" for c in self.clusters)
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "distinct_stories": self.distinct_stories,
            "effective_items": self.effective_items,
            "duplication_ratio": round(self.duplication_ratio, 4),
            "coordinated_clusters": len(self.coordinated),
            "clusters": [c.as_dict() for c in self.clusters],
            "verdict": self.verdict,
        }


def cluster(evidence: Sequence[Any], *, threshold: float = DUPLICATE_AT) -> NoveltyReport:
    """Group evidence into distinct stories by near-duplicate text.

    ``evidence`` items need ``id``, ``claim``, ``source`` and ``available_at`` — the fields
    `agents.analysts.Evidence` already carries, so nothing upstream has to change to be measured.

    Single-link clustering: an item joins a cluster if it is similar enough to **any** member, not
    to the cluster's representative. Rewrites chain — a copy of a copy can drift past the threshold
    from the original while matching its immediate parent — and comparing only to a representative
    would split one story into several, which is the error that would understate duplication.
    """
    if not 0.0 < threshold <= 1.0:
        raise NoveltyError(f"a similarity threshold of {threshold} is not in (0, 1]")
    items = list(evidence)
    if not items:
        return NoveltyReport(items=0, clusters=())

    ordered = sorted(items, key=lambda e: e.available_at)
    prints = [shingles(str(e.claim)) for e in ordered]
    groups: list[list[int]] = []
    for index in range(len(ordered)):
        for group in groups:
            if any(jaccard(prints[index], prints[member]) >= threshold for member in group):
                group.append(index)
                break
        else:
            groups.append([index])

    clusters = []
    for group in groups:
        members = [ordered[i] for i in group]
        stamps = [m.available_at for m in members]
        clusters.append(Cluster(
            representative=str(members[0].claim),
            item_ids=tuple(str(m.id) for m in members),
            sources=tuple(str(m.source) for m in members),
            first_seen=min(stamps),
            last_seen=max(stamps),
        ))
    clusters.sort(key=lambda c: (-c.size, c.first_seen))
    return NoveltyReport(items=len(items), clusters=tuple(clusters))


__all__ = [
    "COORDINATION_WINDOW",
    "DUPLICATE_AT",
    "MIN_COORDINATED_SOURCES",
    "SHINGLE",
    "Cluster",
    "NoveltyError",
    "NoveltyReport",
    "cluster",
    "jaccard",
    "shingles",
]
