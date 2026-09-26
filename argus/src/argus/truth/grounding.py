"""Numeric grounding — every figure in a thesis must resolve to something that produced it.

A thesis is prose written by a model, and prose containing numbers is the easiest place in the whole
system for an unsupported fact to enter. The desk's own record is full of them: *"the 24h change of
+4.4bps is below the 18.80bps total hurdle"* is a good sentence precisely because both numbers came
from code — and nothing, until now, checked that.

This module checks it. Given a thesis and the facts that were available when it was written, it
extracts every number, tries to resolve each one, and reports what it could not. The LUI already
promises that every answer resolves to a ledger row or an evidence id; this is what makes the same
promise true of the numbers *inside* a thesis, which is where it is easiest to quietly break.

**Why matching is tolerant, and bounded.** A model writes "4.4bps" where the computed value is
4.4012, "+2%" where the fact is 0.0203, and "roughly 19bps" where the hurdle is 18.80. Demanding
exact equality would flag every honest paraphrase and train a reader to ignore the report. So a
figure resolves when it is within :data:`TOLERANCE` of a known value, relative to that value — and
that tolerance is small enough that a materially different number cannot slip through as a rounding.

**What it deliberately does not do.** It does not judge whether the number is *right*, only whether
it came from somewhere. Correctness is the backtest's job and the settlement's job. This is the
weaker, checkable property: no figure in the record is unattributable.

**Ignored by design.** Years, ordinal counts, and the numbers inside identifiers are not claims
about the market and flagging them would bury the real findings. The exclusions are listed rather
than inferred, so a reader can disagree with a specific one.

**Three states, not two (added 2026-09-25, item S23).** The check used to be binary: a thesis was
grounded or it was not, and a thesis with one invented figure among nine quoted ones read exactly
like a thesis that quoted nothing it was given. Self-RAG grades support on three levels
(``AkariAsai/self-rag`` ``retrieval_lm/utils.py:48-49``, MIT: ``[Fully supported]``,
``[Partially supported]``, ``[No support / Contradictory]``) and defines the middle one as output
"supported by the evidence to some extent, but there is major information in the output that is
not discussed in the evidence" (``data_creation/critic/gpt4_reward/chatgpt_groundness.py:48``).
:class:`Support` is that vocabulary, applied twice:

* **per figure** — a figure that resolves is fully supported; one that misses every known value at
  :data:`TOLERANCE` but lands within :data:`PARTIAL_TOLERANCE` of a known value *declared in the
  same unit* is a *near miss*, the same quantity misquoted (a stale hurdle, a rounded-away digit),
  and partially supported; anything else has no support. The unit requirement is a measured
  correction, not caution: see :data:`NEAR_MISS_UNITS`. The relative distance also settles sign:
  two numbers of opposite sign are at least 100% apart, so a flipped sign is never a near miss.
* **per thesis** — every figure fully supported is full support, none supported at all is none, and
  anything between is partial, which is Self-RAG's own definition read at the level of a thesis.

:attr:`GroundingReport.support_score` is Self-RAG's ``ground_score`` with its weights as shipped —
fully supported 1, partially 0.5, none 0 (``retrieval_lm/run_long_form_static.py:77-78``) — read
from the checker's verdicts rather than from token probabilities a hosted model does not expose.

**What did not change, deliberately.** :attr:`Resolution.resolved`, :attr:`GroundingReport.grounded`
and :attr:`GroundingReport.coverage` mean exactly what they meant: a near miss is still unresolved.
The desk's escalation counts unresolved figures, `paper/runner.py` flags on the first render line,
and `desk/review.py` grades rules against those flags; a middle state that quietly turned half the
failures into passes would move every one of those numbers without anyone having decided to. The
partial state is added information, reported on its own line, and it is not a flag.

**Measured on the record, 2026-09-26** (`eval/decision_primitives.py`, written to
``data/decision_primitives.json``). Of 640 recorded decisions with a grounding note, 261 failed the
binary check, and **all 261 are partial**: every failing thesis traced at least one figure, so no
decision on record had no support at all. The middle state changes how the failures read, not what
they predict: partial theses' leans were right 52.6% of the time (110 of 209 graded) against 50.2%
(136 of 271) for fully supported ones, two-sided Fisher p = 0.65. It is more honest about degree
and, on this record, says nothing about being right. The per-figure near miss was redesigned after
its first measurement (:data:`NEAR_MISS_UNITS`); rebuilt from the record, the shipped unit-strict
rule finds none — the one unit-bearing value that can be rebuilt there is the 12bps round trip, not
the hurdle theses misquote — and none among the ten recorded thesis-quality answers, where every
known value is on disk. The desk itself, which holds every value, has printed the support line on
each failing decision since it shipped, and one live near miss so far (seq 696: "19bps" against an
``earnings_magnitude_bps`` of 18); the evaluation counts those lines as they accrue.

**Rejected from Self-RAG:** the trained reflection tokens and the logprob read-out that scores them
(``run_long_form_static.py:58-78``). They need a fine-tuned checkpoint and full-vocabulary logprobs;
the hosted Qwen endpoint offers neither (`_SYNTHESIS.md` §4), and a model grading its own support is
the anti-pattern this module exists to avoid.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

TOLERANCE = 0.02
"""A figure resolves within 2% of a known value. Tight enough that 19 cannot pass for 25."""

PARTIAL_TOLERANCE = 0.10
"""A figure that misses at :data:`TOLERANCE` but lands within 10% of a known value is a near miss.

Stated before any measurement, as the band inside which a number is still recognisably the same
quantity quoted wrongly rather than a different quantity. Distances are on :func:`_close`'s scale,
relative to the larger of the two: a hurdle quoted as "20.80bps" against ``total_hurdle_bps``
18.80 is 9.6% off and partial; "25bps" against it is 24.8% off and unsupported.
`eval/decision_primitives.py` measures how often a figure lands in the band *by chance*, by pairing
each thesis with another decision's known values, so the band's false-positive rate is a published
number rather than an assumption."""

MIN_MAGNITUDE = 1e-9
"""Below this, relative tolerance is meaningless and an absolute comparison is used."""


class Support(StrEnum):
    """Self-RAG's groundedness vocabulary (``retrieval_lm/utils.py:48-49``) as a checker verdict."""

    FULL = "fully_supported"
    PARTIAL = "partially_supported"
    NONE = "no_support"

    @property
    def score(self) -> float:
        """Self-RAG's weights: ``ground_score = P(full) + 0.5 * P(partial)``
        (``run_long_form_static.py:77-78``). Here the probabilities are 0 or 1."""
        return {Support.FULL: 1.0, Support.PARTIAL: 0.5, Support.NONE: 0.0}[self]


def combine(levels: Sequence[Support]) -> Support:
    """Thesis-level support from per-item support: all full is full, all none is none, else partial.

    Self-RAG's definition of the middle state — supported "to some extent", with material the
    evidence does not carry — is exactly a mixture, so a mixture is what earns it. An empty sequence
    is full support vacuously, matching :attr:`GroundingReport.grounded` on a figure-free thesis.
    """
    if not levels or all(level is Support.FULL for level in levels):
        return Support.FULL
    if all(level is Support.NONE for level in levels):
        return Support.NONE
    return Support.PARTIAL


# Numbers that are not claims about the market. Listed rather than inferred so each can be argued
# with individually.
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ORDINAL = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I)

# A time-window label names the period a quantity was measured over; it is not itself a claim about
# the market. Found on the live ledger: "the 24h change of +4.4bps" contains two numbers, and only
# one of them is an assertion. Flagging the window would bury the real findings under noise, which
# is how a report trains its reader to ignore it.
_TIME_WINDOW = re.compile(
    r"\d+\s*(?:h|hr|hrs|hour|hours|d|day|days|m|min|mins|minute|minutes|w|wk|week|weeks|"
    r"mo|month|months|y|yr|year|years)\b",
    re.I,
)

_NUMBER = re.compile(
    r"(?<![\w.])"          # not mid-identifier
    r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(%|bps|bp|basis points)?",
    re.I,
)


@dataclass(frozen=True)
class Figure:
    """One number lifted out of a thesis, with the text around it."""

    raw: str
    value: float
    unit: str
    context: str

    def as_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "value": self.value, "unit": self.unit, "context": self.context}


@dataclass(frozen=True)
class NearMiss:
    """The known value an unresolved figure came closest to, inside :data:`PARTIAL_TOLERANCE`."""

    source: str
    known_value: float
    distance: float
    """Relative distance of the figure's closest written form from ``known_value``."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "known_value": self.known_value,
            "distance": round(self.distance, 4),
        }


@dataclass(frozen=True)
class Resolution:
    """Whether a figure could be traced, and to what."""

    figure: Figure
    source: str | None
    known_value: float | None
    near: NearMiss | None = None
    """Set only on an unresolved figure that lands within :data:`PARTIAL_TOLERANCE` of a value."""

    @property
    def resolved(self) -> bool:
        return self.source is not None

    @property
    def support(self) -> Support:
        if self.resolved:
            return Support.FULL
        return Support.PARTIAL if self.near is not None else Support.NONE

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.figure.as_dict(),
            "resolved": self.resolved,
            "source": self.source,
            "known_value": self.known_value,
            "support": str(self.support),
            "near": None if self.near is None else self.near.as_dict(),
        }


@dataclass(frozen=True)
class GroundingReport:
    resolutions: tuple[Resolution, ...]

    @property
    def unresolved(self) -> tuple[Resolution, ...]:
        return tuple(r for r in self.resolutions if not r.resolved)

    @property
    def near_misses(self) -> tuple[Resolution, ...]:
        """Unresolved figures that misquote a value the desk was given, by less than 10%."""
        return tuple(r for r in self.resolutions if r.support is Support.PARTIAL)

    @property
    def unsupported(self) -> tuple[Resolution, ...]:
        """Unresolved figures that are not near any value the desk was given."""
        return tuple(r for r in self.resolutions if r.support is Support.NONE)

    @property
    def grounded(self) -> bool:
        return not self.unresolved

    @property
    def coverage(self) -> float:
        if not self.resolutions:
            return 1.0
        return 1 - len(self.unresolved) / len(self.resolutions)

    @property
    def support(self) -> Support:
        """Thesis-level support; see :func:`combine`. ``FULL`` exactly when :attr:`grounded`."""
        return combine([r.support for r in self.resolutions])

    @property
    def support_score(self) -> float:
        """Self-RAG's ``ground_score`` averaged over figures: 1 full, 0.5 near miss, 0 none.

        Differs from :attr:`coverage` only by the half credit a near miss earns. 1.0 on a thesis
        with no figures, as coverage is.
        """
        if not self.resolutions:
            return 1.0
        return sum(r.support.score for r in self.resolutions) / len(self.resolutions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "figures": len(self.resolutions),
            "unresolved": len(self.unresolved),
            "coverage": round(self.coverage, 3),
            "grounded": self.grounded,
            "support": str(self.support),
            "support_score": round(self.support_score, 3),
            "near_misses": len(self.near_misses),
            "detail": [r.as_dict() for r in self.resolutions],
        }

    def render(self) -> list[str]:
        if not self.resolutions:
            return ["[grounding] the thesis states no figures"]
        if self.grounded:
            return [
                f"[grounding] all {len(self.resolutions)} figure(s) resolve to a computed value "
                f"or a cited fact"
            ]
        names = ", ".join(sorted({r.figure.raw for r in self.unresolved})[:6])
        return [
            f"[grounding] {len(self.unresolved)} of {len(self.resolutions)} figure(s) do not "
            f"resolve to anything the desk was given: {names}",
            "[grounding] an unattributable number in a thesis is the easiest place for an "
            "unsupported fact to enter the record",
            self._support_line(),
        ]

    def _support_line(self) -> str:
        """The three-state reading of a failed check, on a line of its own.

        Worded so it contains none of the phrases `paper/runner.py` and `desk/review.py` treat as a
        finding: it qualifies the failure the two lines above already report, and it must not be
        counted as a second one.
        """
        resolved = len(self.resolutions) - len(self.unresolved)
        verb = "misquotes" if len(self.near_misses) == 1 else "misquote"
        near = "; ".join(
            f"{r.figure.raw} is {r.near.distance:.0%} off {r.near.source} = {r.near.known_value:g}"
            for r in self.near_misses[:4] if r.near is not None
        )
        if self.support is Support.NONE:
            return (
                f"[grounding] support: none — not one of the {len(self.resolutions)} figure(s) "
                f"traces to a given value, even within {PARTIAL_TOLERANCE:.0%}"
            )
        return (
            f"[grounding] support: partial ({self.support_score:.2f}) — {resolved} of "
            f"{len(self.resolutions)} figure(s) trace to a given value"
            + (f"; {len(self.near_misses)} more {verb} one by under "
               f"{PARTIAL_TOLERANCE:.0%}: {near}" if self.near_misses else "")
        )


def _is_excluded(raw: str, trailing: str) -> bool:
    """``trailing`` is the text immediately after the number, which is what names its unit.

    A number embedded in an identifier — ``Q3``, ``NVDA2026`` — needs no rule here: the lookbehind
    in :data:`_NUMBER` already refuses a digit preceded by a word character. A rule for it was
    written anyway and excluded ``25bps`` along with it, because "digits followed by letters"
    describes a unit just as well as it describes an identifier.
    """
    if _YEAR.fullmatch(raw.strip("+-")):
        return True
    # Both the ordinal suffix and the time unit live in the *trailing* text, not in the captured
    # digits, so both are tested against the join rather than against `raw` alone.
    joined = f"{raw}{trailing}"
    return bool(_ORDINAL.match(joined) or _TIME_WINDOW.match(joined))


def extract(text: str, *, window: int = 32) -> tuple[Figure, ...]:
    """Pull every numeric claim out of a thesis, with surrounding text for the reader."""
    out: list[Figure] = []
    for match in _NUMBER.finditer(text):
        raw, unit = match.group(1), (match.group(2) or "").lower()
        if _is_excluded(raw, text[match.end(1): match.end(1) + 12]):
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:  # pragma: no cover — the pattern cannot produce this
            continue
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        out.append(
            Figure(
                raw=match.group(0).strip(),
                value=value,
                unit={"bp": "bps", "basis points": "bps"}.get(unit, unit),
                context=text[start:end].replace("\n", " ").strip(),
            )
        )
    return tuple(out)


def _close(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b))
    if scale < MIN_MAGNITUDE:
        return abs(a - b) < MIN_MAGNITUDE
    return abs(a - b) / scale <= TOLERANCE


def _distance(a: float, b: float) -> float:
    """Relative distance on :func:`_close`'s own scale. Opposite signs are always 1.0 or more."""
    scale = max(abs(a), abs(b))
    if scale < MIN_MAGNITUDE:
        return 0.0 if abs(a - b) < MIN_MAGNITUDE else math.inf
    return abs(a - b) / scale


NEAR_MISS_UNITS = frozenset({"bps", "%"})
"""Only a figure written with one of these units can be a near miss.

**Why, measured.** The first version compared an unresolved figure with every known value through
every unit conversion, as resolution does. Replayed over the 255 failing theses on the record
(`eval/decision_primitives.py`, 2026-09-25) it "found" near misses in 18.0% of named figures — and
every example it printed was a coincidence across quantities: a hurdle written "20.80bps" landing
near 20 *hours* to discovery, "320bps" landing near 3 *analysts* through the /100 conversion. The
same figures paired with another decision's values scored 10.9%, so most of the band was chance. A
near miss claims "the same quantity, misquoted", and that claim needs the quantity: a figure and a
known value are compared only when both were written in the same unit, and only at face value."""


def fact_unit(name: str) -> str | None:
    """The unit a computed fact's name declares — ``*_bps`` or ``*_pct`` — or ``None``.

    A fact named without a unit (a price, a count of hours or analysts) is not eligible for a near
    miss: its unit is not stated anywhere a checker could read it, and guessing it is exactly what
    produced the coincidences described at :data:`NEAR_MISS_UNITS`.
    """
    if name.endswith("_bps"):
        return "bps"
    if name.endswith(("_pct", "_percent")):
        return "%"
    return None


def nearest(
    figure: Figure,
    known: Sequence[tuple[str, float, str | None]],
    *,
    band: float = PARTIAL_TOLERANCE,
) -> NearMiss | None:
    """The same-unit known value ``figure`` comes closest to, if within ``band``; ties go first.

    ``known`` carries each value's unit, ``None`` when undeclared. Only a figure written in a unit
    of :data:`NEAR_MISS_UNITS` is eligible, only against values declared in the same unit, and the
    two are compared as written — no conversion, because a conversion is how 320bps met 3.
    """
    if figure.unit not in NEAR_MISS_UNITS:
        return None
    best: NearMiss | None = None
    for name, value, unit in known:
        if unit != figure.unit:
            continue
        distance = _distance(figure.value, value)
        if distance <= band and (best is None or distance < best.distance):
            best = NearMiss(source=name, known_value=value, distance=distance)
    return best


def _candidates(value: float, unit: str) -> tuple[float, ...]:
    """The same quantity written the ways a model actually writes it.

    A fact of 0.0203 is quoted as "2%" and as "203bps"; refusing those would flag correct prose.
    """
    if unit == "%":
        return (value, value / 100, value * 100)
    if unit == "bps":
        return (value, value / 10_000, value / 100)
    return (value, value * 100, value / 100, value * 10_000, value / 10_000)


def check(
    thesis: str,
    *,
    facts: Mapping[str, float],
    evidence_values: Sequence[tuple[str, float]] = (),
    evidence_units: Sequence[str] | None = None,
) -> GroundingReport:
    """Resolve every figure in ``thesis`` against the values the desk actually had.

    ``facts`` are computed quantities keyed by name — the hurdle, the observed move, the position
    size. ``evidence_values`` are numbers carried by cited evidence, paired with the evidence id
    so a resolution can name it. ``evidence_units``, when given, is the unit each evidence value
    was written in (``"bps"``, ``"%"`` or ``""``), in the same order; it only decides which values
    a near miss may be measured against, never whether a figure resolves.

    Resolution is unchanged by the three-state reading: the first known value within
    :data:`TOLERANCE` wins, as before. Only a figure that resolves to nothing is then offered to
    :func:`nearest`, with the facts' units read from their names (:func:`fact_unit`).
    """
    if evidence_units is not None and len(evidence_units) != len(evidence_values):
        raise ValueError("evidence_units must give one unit per evidence value, in order")
    known: list[tuple[str, float]] = [*facts.items(), *evidence_values]
    typed: list[tuple[str, float, str | None]] = [
        *((name, value, fact_unit(name)) for name, value in facts.items()),
        *((name, value, None if evidence_units is None else evidence_units[i])
          for i, (name, value) in enumerate(evidence_values)),
    ]
    resolutions: list[Resolution] = []

    for figure in extract(thesis):
        matched: tuple[str, float] | None = None
        for name, value in known:
            written = _candidates(figure.value, figure.unit)
            if any(_close(candidate, value) for candidate in written):
                matched = (name, value)
                break
        resolutions.append(
            Resolution(
                figure=figure,
                source=None if matched is None else matched[0],
                known_value=None if matched is None else matched[1],
                near=nearest(figure, typed) if matched is None else None,
            )
        )

    return GroundingReport(resolutions=tuple(resolutions))


__all__ = [
    "MIN_MAGNITUDE",
    "NEAR_MISS_UNITS",
    "PARTIAL_TOLERANCE",
    "TOLERANCE",
    "Figure",
    "GroundingReport",
    "NearMiss",
    "Resolution",
    "Support",
    "check",
    "combine",
    "extract",
    "fact_unit",
    "nearest",
]
