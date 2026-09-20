"""Episode grouping — one bet counted once, not once per time it was restated.

ARGUS decides roughly every two hours. On a quiet weekend it will say ``no_trade`` on NVDA six
times for the same reason, and the scorecard currently treats that as six decisions. That inflates
every count it feeds and flatters any rate computed over them: a desk that repeats one correct
refusal twelve times scores as though it made twelve good calls.

The fix, from ``clawock`` (``src/clawock/decision/ledger.py:1-14``, MIT): group decisions into
**episodes** and score one representative per episode, carrying the episode's mean outcome rather
than an elected member's. clawock's docstring records why the mean and not a member — the choice of
which row represents the episode moves their headline across the 50% line on its own, which is a
result about the metric rather than about the trading.

**What counts as one episode here.** Consecutive decisions on the same symbol, with the same
verdict, in the same session phase, inside a bounded window:

* *same symbol* — obvious;
* *same verdict* — a switch from ``no_trade`` to ``open_long`` is a new bet whatever else holds;
* *same session phase* — a view carried from the weekend into the open is being taken about a
  different market, whatever the words say;
* *bounded window* — two identical refusals a week apart are two judgements about two different
  markets, not one standing view.

Every one of those is a recorded fact. An earlier version of this rule also required the stated
*invalidation* to be unchanged, on the reasoning that revising what would disprove your thesis is a
change of mind. That reasoning still holds; the implementation did not survive contact with the
data, and :func:`invalidation_similarity` records what happened and why the condition became
reported information rather than a blocking test.

**Measured on the live ledger:** 27 decisions collapse to 8 episodes, 3.375 decisions per episode.
Any per-decision rate computed over that log was overstating its denominator by that factor.

``clawock`` settles this with an ``episode_id`` written at decision time and an invariant that
settlement must never move it (``settlement.py:40-50``). Ours is derived at scoring time instead,
because the live ledger already exists without episode ids and rewriting it to add them would edit
history — which the chain exists to prevent. The cost of deriving is that the grouping rule is not
frozen in the record; the benefit is that no historical row is touched. That trade is stated here
rather than buried.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.paper.ledger import Entry

EPISODE_WINDOW = timedelta(hours=36)
"""Two identical refusals further apart than this are two judgements, not one standing view."""


@dataclass(frozen=True)
class Episode:
    """One bet, however many times it was restated."""

    symbol: str
    verdict: str
    entries: tuple[Entry, ...]

    @property
    def seqs(self) -> tuple[int, ...]:
        return tuple(e.seq for e in self.entries)

    @property
    def restatements(self) -> int:
        """How many times the same view was recorded after the first."""
        return len(self.entries) - 1

    @property
    def representative(self) -> Entry:
        """The first decision. The episode is dated from when the view was *taken*."""
        return self.entries[0]

    @property
    def settled(self) -> tuple[Entry, ...]:
        return tuple(e for e in self.entries if e.is_settled)

    @property
    def mean_confidence(self) -> float:
        return sum(e.stated_confidence for e in self.entries) / len(self.entries)

    @property
    def mean_net_pnl(self) -> Decimal | None:
        """The episode's mean outcome, not a chosen member's.

        Picking a member is the failure clawock documents: which one you pick moves the answer.
        """
        settled = [Decimal(e.net_pnl or "0") for e in self.settled if not e.is_abstention]
        if not settled:
            return None
        return sum(settled, Decimal("0")) / len(settled)

    @property
    def mean_counterfactual_bps(self) -> Decimal | None:
        values = [
            Decimal(e.counterfactual_move_bps)
            for e in self.settled
            if e.is_abstention and e.counterfactual_move_bps is not None
        ]
        if not values:
            return None
        return sum(values, Decimal("0")) / len(values)

    @property
    def is_abstention(self) -> bool:
        return self.representative.is_abstention

    @property
    def invalidation_agreement(self) -> float | None:
        """Mean content-word overlap between consecutive restatements, or ``None`` if there are
        none. Low values mean the desk reworded its reasoning heavily while holding the same
        verdict, which is worth seeing but is not treated as a change of mind."""
        if len(self.entries) < 2:
            return None
        scores = [
            invalidation_similarity(tuple(a.invalidation), tuple(b.invalidation))
            for a, b in zip(self.entries, self.entries[1:], strict=False)
        ]
        return sum(scores) / len(scores)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "verdict": self.verdict,
            "seqs": list(self.seqs),
            "decisions": len(self.entries),
            "restatements": self.restatements,
            "mean_confidence": round(self.mean_confidence, 3),
            "mean_net_pnl": None if self.mean_net_pnl is None else str(self.mean_net_pnl),
            "mean_counterfactual_bps": (
                None if self.mean_counterfactual_bps is None
                else str(self.mean_counterfactual_bps)
            ),
            "invalidation_agreement": (
                None if self.invalidation_agreement is None
                else round(self.invalidation_agreement, 3)
            ),
        }


# Written as prose and split at import so the list stays readable and diffable. `noqa` because
# the linter's preferred list-of-strings form is materially harder to scan at this length.
_STOPWORD_TEXT = (
    "a an the and or of to in on at before after that this if is are was were be been being "
    "for with without from by as it its into over under more less than then no not any some "
    "emerges arrives occurs creating materially significant credible verifiable"
)
_STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())

INVALIDATION_OVERLAP = 0.45
"""Reported, never used to split an episode. See :func:`invalidation_similarity`."""


def _content_words(phrases: tuple[str, ...]) -> frozenset[str]:
    words = {
        word.strip(".,;:()'\"").lower()
        for phrase in phrases
        for word in phrase.split()
    }
    return frozenset(w for w in words if len(w) > 2 and w not in _STOPWORDS)


def invalidation_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Jaccard overlap of content words. 1.0 is identical, 0.0 shares nothing.

    **Reported, not used to split episodes, and the reason is a measurement.** This rule first
    tested exact string equality, which never matched because the model rewords its invalidation
    every cycle. Content-word overlap was the obvious repair and it does not work either: on the
    real ledger, three consecutive NVDA refusals that mean the same thing — "anchor market
    reopens", "price-discovery event", "genuine price discovery" — score 0.06 to 0.14, because the
    vocabulary changes even when the condition does not.

    The available responses were to lower the threshold until the grouping fired, or to stop using
    an unvalidated text heuristic as a *blocking* condition. The first is tuning a measure until it
    returns the wanted answer. So the similarity is attached to each episode as information, where
    a reader can weigh it, and the grouping itself rests only on facts that are not free text.
    """
    a, b = _content_words(left), _content_words(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _same_episode(previous: Entry, current: Entry) -> bool:
    """Same symbol, same verdict, same session, close in time.

    Every condition is a recorded fact rather than an interpretation of free text. A verdict change
    is a new bet; a session change means the view is being taken about a different market; and two
    identical refusals far apart are two judgements rather than one standing view.
    """
    if previous.symbol != current.symbol or previous.verdict != current.verdict:
        return False
    if previous.session_phase != current.session_phase:
        return False
    gap = datetime.fromisoformat(current.decided_at) - datetime.fromisoformat(previous.decided_at)
    return gap <= EPISODE_WINDOW


def group(entries: Sequence[Entry]) -> tuple[Episode, ...]:
    """Collapse a ledger into episodes, preserving order.

    Grouping is per symbol and chronological: decisions on different symbols interleave in the log
    and must not break each other's episodes.
    """
    by_symbol: dict[str, list[Entry]] = {}
    for entry in sorted(entries, key=lambda e: (e.symbol, e.decided_at, e.seq)):
        by_symbol.setdefault(entry.symbol, []).append(entry)

    episodes: list[Episode] = []
    for symbol, rows in by_symbol.items():
        current: list[Entry] = []
        for entry in rows:
            if current and _same_episode(current[-1], entry):
                current.append(entry)
                continue
            if current:
                episodes.append(Episode(symbol, current[0].verdict, tuple(current)))
            current = [entry]
        if current:
            episodes.append(Episode(symbol, current[0].verdict, tuple(current)))

    return tuple(sorted(episodes, key=lambda e: e.representative.seq))


@dataclass(frozen=True)
class EpisodeSummary:
    """Both numbers, side by side. The gap between them is the inflation being corrected."""

    decisions: int
    episodes: tuple[Episode, ...]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def restatements(self) -> int:
        return sum(e.restatements for e in self.episodes)

    @property
    def inflation(self) -> float:
        """Decisions per episode. 1.0 means every decision was a distinct bet."""
        if not self.episodes:
            return 0.0
        return self.decisions / len(self.episodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "episodes": self.episode_count,
            "restatements": self.restatements,
            "decisions_per_episode": round(self.inflation, 3),
            "note": (
                "a scorecard counted per decision overstates a repeated view by this ratio; "
                "both numbers are reported so the correction is visible rather than applied "
                "silently"
            ),
            "detail": [e.as_dict() for e in self.episodes],
        }


def summarise(entries: Sequence[Entry]) -> EpisodeSummary:
    return EpisodeSummary(decisions=len(entries), episodes=group(entries))


__all__ = ["EPISODE_WINDOW", "Episode", "EpisodeSummary", "group", "summarise"]
