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
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TOLERANCE = 0.02
"""A figure resolves within 2% of a known value. Tight enough that 19 cannot pass for 25."""

MIN_MAGNITUDE = 1e-9
"""Below this, relative tolerance is meaningless and an absolute comparison is used."""

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
class Resolution:
    """Whether a figure could be traced, and to what."""

    figure: Figure
    source: str | None
    known_value: float | None

    @property
    def resolved(self) -> bool:
        return self.source is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.figure.as_dict(),
            "resolved": self.resolved,
            "source": self.source,
            "known_value": self.known_value,
        }


@dataclass(frozen=True)
class GroundingReport:
    resolutions: tuple[Resolution, ...]

    @property
    def unresolved(self) -> tuple[Resolution, ...]:
        return tuple(r for r in self.resolutions if not r.resolved)

    @property
    def grounded(self) -> bool:
        return not self.unresolved

    @property
    def coverage(self) -> float:
        if not self.resolutions:
            return 1.0
        return 1 - len(self.unresolved) / len(self.resolutions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "figures": len(self.resolutions),
            "unresolved": len(self.unresolved),
            "coverage": round(self.coverage, 3),
            "grounded": self.grounded,
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
        ]


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
) -> GroundingReport:
    """Resolve every figure in ``thesis`` against the values the desk actually had.

    ``facts`` are computed quantities keyed by name — the hurdle, the observed move, the position
    size. ``evidence_values`` are numbers carried by cited evidence, paired with the evidence id
    so a resolution can name it.
    """
    known: list[tuple[str, float]] = [*facts.items(), *evidence_values]
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
            )
        )

    return GroundingReport(resolutions=tuple(resolutions))


__all__ = [
    "MIN_MAGNITUDE",
    "TOLERANCE",
    "Figure",
    "GroundingReport",
    "Resolution",
    "check",
    "extract",
]
