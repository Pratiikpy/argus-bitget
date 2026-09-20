"""Claim grounding — catching a thesis that contradicts its own evidence.

:mod:`argus.agents.grounding` checks the *figures* in a thesis. This checks the *claims*, and it
exists because of one real decision.

**Ledger seq 41, 2026-09-12.** The desk was shown two insider sales and wrote:

    "insider sales are pre-arranged 10b5-1 plans already reflected in price"

Both filings carry ``aff10b5One = 0``. They are not pre-arranged, and
:mod:`argus.market.insider` forwards a sale *only* when it is not — a pre-arranged trade is
filtered out before the desk ever sees it. So the thesis asserted, about the two pieces of evidence
it was looking at, the exact property that would have excluded them. The refusal was still the
right call for other reasons; the reasoning contained a fact that was false.

That is invisible to numeric grounding, because no number was wrong.

**What this module can and cannot do.** It does not detect hallucination in general — that needs a
judge model, and a model grading another model's reasoning is the methodological hole this project
has documented in FinanceBenchmark. It detects **contradictions against structured attributes the
evidence already carries**. That is a narrow, checkable subset, and narrow-and-sound beats
broad-and-suggestive in a record a judge will read.

Each rule pairs a phrase the desk actually writes with a field on the evidence that settles it. A
claim with no such field is not checked and is not reported as passing — :class:`ClaimReport`
counts what it examined, so coverage is visible rather than implied.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

NEGATION_WINDOW = 40
"""Characters before a phrase searched for a negation. "not pre-arranged" is the opposite claim."""

_NEGATORS = re.compile(r"\b(?:not|no|never|isn'?t|aren'?t|wasn'?t|weren'?t|without)\b", re.I)


class HasAttributes(Protocol):
    """Any evidence-like object exposing structured attributes to check a claim against."""

    def as_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Rule:
    """A phrase the desk writes, and the evidence field that settles whether it is true."""

    name: str
    pattern: str
    field: str
    """Attribute on the evidence record that decides the claim."""

    asserts: bool
    """The value the field must hold for the claim to be true."""

    explain: str

    @property
    def regex(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.I)


# Only claims that a structured field can actually settle. Each one is a sentence the desk has
# written or plausibly will, paired with the field that makes it checkable.
RULES: tuple[Rule, ...] = (
    Rule(
        name="pre_arranged",
        pattern=r"(?:pre-?arranged|10b5-?1|scheduled\s+sale|planned\s+sale)",
        field="pre_arranged",
        asserts=True,
        explain=(
            "the thesis calls the transaction pre-arranged; the filing's aff10b5One field says "
            "otherwise, and an insider sale only reaches the desk when it is not pre-arranged"
        ),
    ),
    Rule(
        name="open_market_purchase",
        pattern=r"insider\s+(?:buying|purchase|bought)",
        field="acquired",
        asserts=True,
        explain="the thesis describes insider buying; the evidence records a disposal",
    ),
    Rule(
        name="fundamental_growth",
        pattern=(
            r"(?:revenue|sales|earnings|top[- ]line|bottom[- ]line)\s+"
            r"(?:growth|grew|growing|expansion|expanding|rose|rising|increase[ds]?|up\b)"
        ),
        field="growing",
        asserts=True,
        explain=(
            "the thesis describes the reported line as growing; the filed figures show a "
            "sequential decline"
        ),
    ),
    Rule(
        name="fundamental_decline",
        pattern=(
            r"(?:revenue|sales|earnings|top[- ]line|bottom[- ]line)\s+"
            r"(?:decline|declining|declined|contraction|contracting|fell|falling|shrank|"
            r"shrinking|deteriorat\w+)"
        ),
        field="growing",
        asserts=False,
        explain=(
            "the thesis describes the reported line as shrinking; the filed figures show "
            "sequential growth"
        ),
    ),
    Rule(
        name="conviction",
        pattern=r"(?:high[- ]conviction|conviction\s+(?:buy|purchase)|strong\s+insider\s+signal)",
        field="conviction",
        asserts=True,
        explain=(
            "the thesis treats the transaction as a conviction signal; only an unplanned "
            "open-market purchase qualifies, and this is not one"
        ),
    ),
)


@dataclass(frozen=True)
class Contradiction:
    """A claim the thesis made that its own evidence refutes."""

    rule: str
    claim_text: str
    evidence_id: str
    field: str
    asserted: bool
    actual: Any
    explain: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "claim": self.claim_text,
            "evidence": self.evidence_id,
            "field": self.field,
            "asserted": self.asserted,
            "actual": self.actual,
            "explain": self.explain,
        }

    def render(self) -> str:
        return (
            f"[claim] {self.rule}: \"{self.claim_text}\" is contradicted by {self.evidence_id} "
            f"({self.field}={self.actual!r}) — {self.explain}"
        )


@dataclass(frozen=True)
class ClaimReport:
    contradictions: tuple[Contradiction, ...]
    claims_examined: int
    records_checked: int

    @property
    def sound(self) -> bool:
        return not self.contradictions

    def as_dict(self) -> dict[str, Any]:
        return {
            "sound": self.sound,
            "claims_examined": self.claims_examined,
            "records_checked": self.records_checked,
            "contradictions": [c.as_dict() for c in self.contradictions],
        }

    def render(self) -> list[str]:
        if not self.records_checked:
            return ["[claim] no structured evidence to check the thesis against"]
        if not self.claims_examined:
            return [
                f"[claim] no checkable claim found in the thesis "
                f"({self.records_checked} structured record(s) available)"
            ]
        if self.sound:
            return [
                f"[claim] {self.claims_examined} checkable claim(s) agree with the "
                f"{self.records_checked} structured record(s) behind them"
            ]
        return [c.render() for c in self.contradictions] + [
            "[claim] a thesis that contradicts its own evidence is a defect in the reasoning, "
            "not a difference of opinion"
        ]


def _negated(text: str, start: int) -> bool:
    """Is the phrase at ``start`` preceded by a negation inside the window?"""
    window = text[max(0, start - NEGATION_WINDOW): start]
    return bool(_NEGATORS.search(window))


def check(
    thesis: str,
    *,
    records: Sequence[tuple[str, dict[str, Any]]],
    rules: Sequence[Rule] = RULES,
) -> ClaimReport:
    """Check a thesis against the structured attributes of the evidence behind it.

    ``records`` pairs an evidence id with its attribute dictionary — for insider evidence that is
    :meth:`argus.market.insider.InsiderDecision.as_dict`. A claim is a contradiction when the
    thesis asserts a property and **every** record carrying that field disagrees: one matching
    record is enough to make the claim true of something, and a thesis is allowed to be about a
    subset of its evidence.
    """
    contradictions: list[Contradiction] = []
    examined = 0

    for rule in rules:
        match = rule.regex.search(thesis)
        if match is None:
            continue
        if _negated(thesis, match.start()):
            # "not pre-arranged" asserts the opposite; this rule cannot settle it.
            continue

        relevant = [(rid, attrs) for rid, attrs in records if rule.field in attrs]
        if not relevant:
            continue
        examined += 1

        agreeing = [rid for rid, attrs in relevant if bool(attrs[rule.field]) is rule.asserts]
        if agreeing:
            continue

        rid, attrs = relevant[0]
        start = max(0, match.start() - 30)
        contradictions.append(
            Contradiction(
                rule=rule.name,
                claim_text=thesis[start: match.end() + 30].strip(),
                evidence_id=rid,
                field=rule.field,
                asserted=rule.asserts,
                actual=attrs[rule.field],
                explain=rule.explain,
            )
        )

    return ClaimReport(
        contradictions=tuple(contradictions),
        claims_examined=examined,
        records_checked=len(records),
    )


__all__ = [
    "NEGATION_WINDOW",
    "RULES",
    "ClaimReport",
    "Contradiction",
    "Rule",
    "check",
]
