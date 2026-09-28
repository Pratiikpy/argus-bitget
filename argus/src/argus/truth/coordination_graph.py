"""Accounts that keep posting the same stories together: a persistent graph across reports.

`truth/novelty.cluster` flags one story as coordinated when several distinct sources carry it inside
a short window. It is stateless: each report is judged alone, so two accounts that appear together
on story after story — the signature of a network rather than of one busy news hour — are never
connected. CooRTweet (Righetti and Balluff, *Computational Communication Research* 7(1), 2025; MIT)
builds that connection: every pair of accounts sharing an object inside a window adds to a
weighted edge (`detect_groups.R:53-120`, `generate_coordinated_network.R:95-177`), each edge
carries how evenly the two contributed (`generate_coordinated_network.R:178-186`), and the edges
kept are those above a percentile of the graph's own weight distribution
(`generate_coordinated_network.R:391-433`), so the threshold follows a symbol's baseline chatter
rather than a fixed count (research/harvest/11-coortweet.md).

The object shared is a story: the cluster id `novelty.cluster` gives near-identical items. CooRTweet
keys on a retweeted tweet's id; retweets are excluded from ARGUS's collection by design (three
accounts retweeting one post is virality, not three sources), so the story cluster stands in.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_PERCENTILE = 0.5
"""CooRTweet's default ``edge_weight`` quantile (`generate_coordinated_network.R:391`)."""


@dataclass(frozen=True, slots=True)
class Share:
    """One account carrying one story, ``count`` times."""

    account: str
    story: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class Edge:
    left: str
    right: str
    weight: int
    """Distinct stories both accounts carried."""
    symmetry: float
    """``min(a, b) / max(a, b)`` over the two accounts' share counts on those stories: 1 when they
    contributed equally, near 0 when one account did nearly all the posting."""
    stories: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"left": self.left, "right": self.right, "weight": self.weight,
                "symmetry": round(self.symmetry, 4), "stories": list(self.stories)}


def edges(shares: Iterable[Share], *, max_story_accounts: int = 60) -> list[Edge]:
    """Every account pair that carried at least one story together.

    A story carried by more than ``max_story_accounts`` accounts adds no pairs: at that size it is
    the day's news, and its tens of thousands of pairs would swamp the stories that small groups
    push together. CooRTweet's ``min_participation`` bounds the other end (accounts, not stories);
    this bound is ours and is stated in every report that uses it."""
    by_story: dict[str, Counter[str]] = defaultdict(Counter)
    for share in shares:
        by_story[share.story][share.account] += share.count
    pair_stories: dict[tuple[str, str], list[str]] = defaultdict(list)
    pair_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for story, counts in by_story.items():
        accounts = sorted(counts)
        if len(accounts) < 2 or len(accounts) > max_story_accounts:
            continue
        for a, b in itertools.combinations(accounts, 2):
            pair_stories[(a, b)].append(story)
            pair_counts[(a, b)][0] += counts[a]
            pair_counts[(a, b)][1] += counts[b]
    out = []
    for (a, b), stories in pair_stories.items():
        n_a, n_b = pair_counts[(a, b)]
        out.append(Edge(a, b, len(stories), min(n_a, n_b) / max(n_a, n_b), tuple(sorted(stories))))
    return sorted(out, key=lambda e: (-e.weight, -e.symmetry, e.left, e.right))


def percentile_cut(graph: Sequence[Edge], percentile: float = DEFAULT_PERCENTILE) -> int:
    """The edge weight at ``percentile`` of the graph's own weights (type-7 quantile, R's default,
    the one CooRTweet's ``quantile`` call uses), rounded up to a whole count."""
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be in [0, 1]")
    weights = sorted(e.weight for e in graph)
    if not weights:
        return 0
    h = (len(weights) - 1) * percentile
    lo = int(h)
    hi = min(lo + 1, len(weights) - 1)
    value = weights[lo] + (h - lo) * (weights[hi] - weights[lo])
    return int(-(-value // 1))


def strong(graph: Sequence[Edge], *, percentile: float = DEFAULT_PERCENTILE, min_weight: int = 2,
           min_symmetry: float = 0.0) -> list[Edge]:
    """Edges above the graph's own percentile and at least ``min_weight`` stories: a pair that met
    once is a coincidence whatever the distribution says."""
    cut = max(percentile_cut(graph, percentile), min_weight)
    return [e for e in graph if e.weight >= cut and e.symmetry >= min_symmetry]


def components(graph: Sequence[Edge]) -> list[set[str]]:
    """Connected groups of accounts, largest first."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in graph:
        parent[find(e.left)] = find(e.right)
    groups: dict[str, set[str]] = defaultdict(set)
    for node in list(parent):
        groups[find(node)].add(node)
    return sorted(groups.values(), key=lambda g: (-len(g), sorted(g)))


__all__ = ["DEFAULT_PERCENTILE", "Edge", "Share", "components", "edges", "percentile_cut",
           "strong"]
