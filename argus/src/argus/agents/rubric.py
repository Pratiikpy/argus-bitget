"""A named-rubric self-score, logged beside the desk's own measurement of the same axes.

**The problem with one confidence number.** The Meta-PM states ``confidence`` as a single figure in
[0, 1], and nothing says what it is confident *about*. Measured on the record on 2026-09-26
(`eval/decision_primitives.py`, section ``rubric``, 641 decisions), that one figure says nothing
about whether the thesis's figures trace to what it was given (AUC 0.50 against a fully grounded
thesis) and runs *against* the other two axes the desk measures for every decision: it is higher
when the panel read fewer than six distinct sources (AUC 0.32) and when the panel's independence was
under 2.5 (AUC 0.33), both p < 0.001. A confidence that rises as the evidence narrows cannot be
audited against the evidence; a score per axis, set beside the desk's own figure for that axis, can.

**Taken from Self-Refine** (``madaan/self-refine``, Apache-2.0; licence text and the "Used in"
record at ``argus/licenses/self-refine-APACHE-2.0.txt``). Its feedback prompts score several named,
independently meaningful qualities before a total — five for acronyms
(``src/acronym/feedback.py:20-32``: pronunciation, spelling, relation to the title, connotation,
well-known), ten for dialogue responses (``src/responsegen/feedback.py:20-38``). A reader sees
*which* quality was weak rather than one opaque number. Adapted, changed in three ways:

1. **Every axis names the desk's own measurement of it**, so the model's score and the checker's
   figure sit side by side in the record (:func:`pairs`, :func:`pair_note`) — the "checkable pair"
   the harvest asked for (`Activity/17_HARVEST_CHECKLIST.md`, Part 2). Self-Refine's rubric has no
   external counterpart for any of its dimensions; its authors grade final quality with a separate
   judge or metric (``src/sentiment_reversal/gpt4_eval.py``, ``src/pie/pie_eval.py``), which is a
   tacit admission that a self-score is not a measurement. Here the pairing makes that admission
   testable per decision.
2. **Structured, not parsed out of prose.** Self-Refine elicits the scores as free text matching a
   few-shot template and splits on ``"Scores:"`` (``src/acronym/feedback.py:70-71``); its
   instruction naming the five qualities is assigned and then immediately overwritten, so the model
   only ever saw the examples (``src/acronym/feedback.py:50-56``, and the same two lines again at
   ``src/responsegen/feedback.py:65-66``). Here the axes and their anchors are in the
   schema, and a missing or out-of-range score is returned to the model with the specific fault.
3. **Beside the stated confidence, not instead of it.** ``confidence`` is a field of the hashed,
   ledgered :class:`~argus.decision.verdicts.Intent` and is graded by the calibration and refusal
   evaluations; removing it would break every consumer of the ledger for a signal not yet measured.
   The rubric is recorded next to it, so the two can be compared on the same decisions.

**Rejected:** a total score. Summing the axes would reintroduce the single number this replaces,
and weighting them would need a fit nobody has run.

The flag that asks for the rubric (`MetaPM(rubric=True)`) is **off by default**: this item was
built with no paid model calls permitted, so how the model actually scores itself is not yet
measured, and a default that changes every decision prompt would also invalidate the recorded
replays other evaluations answer from.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from argus.desk.review import LOW_INDEPENDENCE, NARROW_SOURCES


@dataclass(frozen=True, slots=True)
class Dimension:
    """One named axis: what the model is asked, and what the desk measures for it independently."""

    name: str
    anchor: str
    """The question and its scale, as the model reads it."""

    measured_by: str
    """The independent measurement it is paired with, as a reader of the record sees it."""


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        name="groundedness",
        anchor=(
            "how much of your thesis rests on figures you were given, quoted as given — 1 if every "
            "figure appears in the frame exactly, 0 if the thesis rests on numbers you estimated"
        ),
        measured_by="the grounding checker's support score (agents/grounding.py)",
    ),
    Dimension(
        name="evidence_breadth",
        anchor=(
            "how many distinct sources the decision rests on — 1 if many independent sources "
            "bear on it, 0 if one source or none"
        ),
        measured_by=f"distinct sources the panel read (narrow below {NARROW_SOURCES:.0f})",
    ),
    Dimension(
        name="independence",
        anchor=(
            "how independent the views you were shown are of one another — 1 if the analysts "
            "read different evidence, 0 if they are one view repeated"
        ),
        measured_by=f"the panel's independence ratio (low below {LOW_INDEPENDENCE})",
    ),
    Dimension(
        name="edge_over_hurdle",
        anchor=(
            "how far the move you expect exceeds the TOTAL HURDLE — 1 if clearly beyond it, 0.5 "
            "if about equal, 0 if well short of it"
        ),
        measured_by="the settled move against the hurdle, once the decision settles",
    ),
)

NAMES: tuple[str, ...] = tuple(d.name for d in DIMENSIONS)

RUBRIC_PROMPT = (
    "Your JSON object must ALSO carry \"self_rubric\": an object scoring your own answer on each "
    "named axis, every value a number from 0 to 1, each judged on its own:\n"
    + "\n".join(f"  \"{d.name}\": {d.anchor}" for d in DIMENSIONS)
    + "\nThese are not a second confidence. Each is set beside what the desk measures "
    "independently for the same axis, decision by decision, and the pair is kept in the record."
)
"""Appended to the Meta-PM's system prompt when the rubric flag is on; the default prompt is
byte-for-byte unchanged, so recorded replays keyed by it still match."""


def parse(response: Mapping[str, Any]) -> dict[str, float] | None:
    """The rubric as numbers, or ``None`` when any axis is missing or not a number in [0, 1].

    All or nothing: a rubric with an axis missing cannot be paired axis by axis, and filling the gap
    would record a score the model never gave.
    """
    raw = response.get("self_rubric")
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, float] = {}
    for name in NAMES:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            return None
        try:
            number = float(value)
        except ValueError:
            return None
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return None
        out[name] = number
    return out


def complaint(response: Mapping[str, Any]) -> str:
    """What is wrong with the rubric, fed back to the model by `complete_json`, or ``""``."""
    raw = response.get("self_rubric")
    if not isinstance(raw, Mapping):
        return "\"self_rubric\" is missing or is not an object with the named axes"
    missing = [name for name in NAMES if name not in raw]
    if missing:
        return f"\"self_rubric\" is missing {missing}"
    if parse(response) is None:
        return "every \"self_rubric\" value must be a number from 0 to 1"
    return ""


@dataclass(frozen=True, slots=True)
class RubricPair:
    """The model's score on one axis, beside the desk's independent measure of it."""

    dimension: str
    self_score: float
    measured: float | None
    """The desk's figure. ``None`` when it does not exist yet (edge settles later) or was not
    supplied — never zero, which would read as a measurement."""

    meets_bar: bool | None
    """Whether the measure clears the bar the desk already uses for this axis: fully grounded,
    at least :data:`NARROW_SOURCES` sources, independence at least :data:`LOW_INDEPENDENCE`."""

    def as_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "self": self.self_score,
                "measured": self.measured, "meets_bar": self.meets_bar}


def meets_bar(dimension: str, measured: float | None) -> bool | None:
    """Whether the desk's measure of one axis clears the bar the desk already applies to it.

    Fully grounded (support score 1), at least :data:`NARROW_SOURCES` distinct sources, independence
    at least :data:`LOW_INDEPENDENCE`, and a settled move beyond the hurdle (a ratio above 1). One
    function, so the pair written at decision time and the agreement computed afterwards
    (`eval/decision_primitives.py`) can never apply two different bars. ``None`` without a measure.
    """
    if measured is None:
        return None
    if dimension == "groundedness":
        return measured >= 1.0
    if dimension == "evidence_breadth":
        return measured >= NARROW_SOURCES
    if dimension == "independence":
        return measured >= LOW_INDEPENDENCE
    if dimension == "edge_over_hurdle":
        return measured > 1.0
    raise ValueError(f"{dimension!r} is not a rubric axis")


def pairs(
    rubric: Mapping[str, float], *, grounding_support: float | None = None,
    distinct_sources: float | None = None, independence: float | None = None,
    settled_move_bps: float | None = None, hurdle_bps: float | None = None,
) -> tuple[RubricPair, ...]:
    """Pair each self-score with the desk's measure. Any measure may be absent; it is then ``None``.

    ``edge_over_hurdle`` is measured as the settled absolute move over the hurdle, and only once
    both exist: before settlement the desk has no measure of edge, and saying so beats inventing
    one.
    """
    edge: float | None = None
    if settled_move_bps is not None and hurdle_bps:
        edge = abs(settled_move_bps) / hurdle_bps
    measured: dict[str, float | None] = {
        "groundedness": grounding_support,
        "evidence_breadth": distinct_sources,
        "independence": independence,
        "edge_over_hurdle": edge,
    }
    return tuple(
        RubricPair(dimension=name, self_score=float(rubric[name]), measured=measured[name],
                   meets_bar=meets_bar(name, measured[name]))
        for name in NAMES if name in rubric
    )


def desk_note(
    scores: Mapping[str, float] | None, *, grounding_support: float | None,
    distinct_sources: float | None, independence: float | None,
) -> str | None:
    """The pair line for one decision's notes, or ``None`` when the answer carried no rubric.

    The desk's integration point, written here so the desk needs one call: ``scores`` is
    ``pm.last_deliberation.committed_attempt.rubric`` and the three measures are the ones the desk
    already holds right after its grounding check — ``grounding.support_score``,
    ``len(panel.distinct_sources)`` and ``panel.independence_ratio``. The line lands in the same
    note row `desk/review.py` reads its own measures from, so the self-score and the measurement of
    the same axis can be compared decision by decision (:func:`parse_pair_note`). Edge is left to
    settlement and written as ``n/a``.
    """
    if scores is None:
        return None
    return pair_note(pairs(scores, grounding_support=grounding_support,
                           distinct_sources=distinct_sources, independence=independence))


def pair_note(checked: Sequence[RubricPair]) -> str:
    """One line for the decision record. Parsed back by :func:`parse_pair_note`."""
    parts = []
    for p in checked:
        shown = "n/a" if p.measured is None else f"{p.measured:.2f}"
        parts.append(f"{p.dimension} self {p.self_score:.2f} measured {shown}")
    return "[rubric] " + "; ".join(parts)


_PAIR = re.compile(r"(?P<name>[a-z_]+) self (?P<self>[\d.]+) measured (?P<measured>[\d.]+|n/a)")


def parse_pair_note(line: str) -> dict[str, tuple[float, float | None]]:
    """Axis → (self score, measure) from a line :func:`pair_note` wrote; empty for other lines."""
    if not line.startswith("[rubric] "):
        return {}
    out: dict[str, tuple[float, float | None]] = {}
    for match in _PAIR.finditer(line):
        measured = match.group("measured")
        out[match.group("name")] = (
            float(match.group("self")), None if measured == "n/a" else float(measured)
        )
    return out


# --- agreement between a self-score and its measure -----------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, 1-based, ties sharing the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability a decision that meets the bar scored higher than one that does not (ties half).

    The Mann-Whitney U over the two groups, divided by the product of their sizes. 0.5 is a score
    that carries no information about the axis; ``None`` when either group is empty.
    """
    positives = sum(1 for flag in labels if flag)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = _ranks(scores)
    rank_sum = sum(r for r, flag in zip(ranks, labels, strict=True) if flag)
    u = rank_sum - positives * (positives + 1) / 2
    return u / (positives * negatives)


def auc_p_value(value: float, positives: int, negatives: int) -> float:
    """Two-sided normal-approximation p-value for an AUC against 0.5 (no tie correction, so it is
    slightly conservative when scores tie, as a confidence rounded to two decimals does)."""
    n1, n2 = positives, negatives
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma == 0:
        return 1.0
    z = abs(value * n1 * n2 - n1 * n2 / 2) / sigma
    return math.erfc(z / math.sqrt(2))


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation; ``None`` below three pairs or when either side is constant."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def agreement(self_scores: Sequence[float], measured: Sequence[float],
              meets_bar: Sequence[bool]) -> dict[str, Any]:
    """How well one axis's self-scores track the desk's measure of it, over many decisions."""
    positives = sum(1 for flag in meets_bar if flag)
    negatives = len(meets_bar) - positives
    value = auc(self_scores, meets_bar)
    rho = spearman(self_scores, measured)
    return {
        "n": len(self_scores),
        "meets_bar": positives,
        "misses_bar": negatives,
        "auc": None if value is None else round(value, 4),
        "auc_p_value": (None if value is None
                        else round(auc_p_value(value, positives, negatives), 4)),
        "spearman": None if rho is None else round(rho, 4),
    }


__all__ = [
    "DIMENSIONS",
    "NAMES",
    "RUBRIC_PROMPT",
    "Dimension",
    "RubricPair",
    "agreement",
    "auc",
    "auc_p_value",
    "complaint",
    "desk_note",
    "meets_bar",
    "pair_note",
    "pairs",
    "parse",
    "parse_pair_note",
    "spearman",
]
