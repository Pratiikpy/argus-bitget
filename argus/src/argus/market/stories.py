"""Headlines grouped into stories, so repetition is counted once.

The desk's sentiment analyst works to one rule (`agents/analysts.py`, `SentimentAnalyst.role`):
*five accounts repeating one article is one source.* Its measured defence against coordinated
posting (`eval/sentiment_comparison.py`) is the reason the Market Sentiment capability is OWNED.
The console never applied that rule to what it showed: it printed "12 headlines name NVDA", and a
syndicated article carried by four outlets read as four pieces of evidence.

**Neither news-sentiment system in the corpus separates them.** `KVignesh122/
AssetNewsSentimentAnalyzer` and `6551Team/opennews-mcp` (both under `research/repos-themed/`) were
searched for any duplicate, similarity or clustering step before this was written; each scores
every headline it fetches as its own item.

**The method is Broder's (1997), on word sets rather than shingles.** "On the resemblance and
containment of documents" defines two measures over token sets: resemblance, the intersection over
the union (Jaccard), and containment, the intersection over one set. A headline is a dozen
words, so exact sets are cheap and MinHash would only add error. Two headlines are one story when
their resemblance reaches ``SAME_STORY`` or when the shorter one is contained in the longer at
``CONTAINED`` — the second catches a rewrite that adds a clause ("..., Analysts Say"). Stories
are the connected components of that relation, so the grouping does not depend on the order
headlines arrive in.

**What it deliberately does not do.** It does not score tone: finBERT is the classification
baseline, the console host cannot load it, and this capability's claim was never better
classification. It groups; it does not judge.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

SAME_STORY = 0.5
"""Resemblance at which two headlines are the same story. Measured on 2026-09-24, not assumed:

* the coordinated posts of `eval/sentiment_comparison.coordinated_scenario` (five rewrites of one
  claim, for NVDA and COIN) pair at resemblance 0.68 or more and containment 0.81 or more — all 20
  pairs group into one story each;
* every headline of the previous 48 hours across the 13 outlet feeds and ten Yahoo ticker feeds
  (298 headlines, 44,253 pairs) grouped 5 pairs. Four were one event reported by two outlets
  (a CME listing, a Binance stake in Circle, a BlackRock note, one obituary); one was a templated
  headline about two different companies ("Alphabet (GOOGL) / Strategy (MSTR) Registers a Bigger
  Fall Than the Market"), a merge that cannot happen inside one name's headlines, which is the
  only way the console groups them. 99.9% of pairs sat below 0.3."""
CONTAINED = 0.8
"""Containment of the shorter headline in the longer at which it is a rewrite of the same story."""

_STOP = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "how",
    "in", "into", "is", "it", "its", "of", "on", "or", "over", "than", "that", "the", "their",
    "this", "to", "up", "was", "were", "what", "when", "which", "who", "why", "will", "with",
    "after", "before", "about", "more", "most", "new", "now", "just", "says", "said", "say",
    "stock", "stocks", "shares", "share", "today", "week", "year",
))
_WORD = re.compile(r"[a-z0-9$%.]+")


class Headline(Protocol):
    title: str
    feed: str
    published: datetime


@dataclass(frozen=True, slots=True)
class Story:
    """One story, however many times it was carried."""

    title: str
    outlets: tuple[str, ...]
    copies: int
    latest: datetime


def words(title: str, names: Iterable[str] = ()) -> frozenset[str]:
    """The words that make a headline this story rather than another: stop words and the
    instrument's own names are dropped, since every headline about NVDA shares "nvidia"."""
    drop = _STOP | {n.lower() for n in names}
    return frozenset(w.strip(".") for w in _WORD.findall(title.lower())
                     if w.strip(".") and w.strip(".") not in drop)


def same_story(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return a == b
    shared = len(a & b)
    return (shared / len(a | b) >= SAME_STORY
            or shared / min(len(a), len(b)) >= CONTAINED)


def outlet(feed: str) -> str:
    return "Yahoo Finance" if feed.startswith("yahoo") else feed


def group(headlines: Sequence[Headline], names: Iterable[str] = ()) -> list[Story]:
    """Stories, most-carried first; ties broken by the most recent."""
    named = tuple(names)
    sets = [words(h.title, named) for h in headlines]
    parent = list(range(len(headlines)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(headlines)):
        for j in range(i + 1, len(headlines)):
            if same_story(sets[i], sets[j]):
                parent[root(j)] = root(i)
    members: dict[int, list[int]] = {}
    for i in range(len(headlines)):
        members.setdefault(root(i), []).append(i)
    stories = []
    for idx in members.values():
        first = min(idx, key=lambda k: headlines[k].published)
        stories.append(Story(
            title=headlines[first].title,
            outlets=tuple(dict.fromkeys(outlet(headlines[k].feed) for k in idx)),
            copies=len(idx),
            latest=max(headlines[k].published for k in idx)))
    return sorted(stories, key=lambda s: (-s.copies, -s.latest.timestamp()))


__all__ = ["CONTAINED", "SAME_STORY", "Story", "group", "outlet", "same_story", "words"]
