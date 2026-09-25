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

**Partially supported, as a state of its own (added 2026-09-25, item S23).** A claim used to pass
the moment one relevant record agreed, which is right about the question "is the thesis
contradicted" and silent about a second one: a thesis that says "the insider sales are
pre-arranged" over two filings, one planned and one not, is true of half of what it describes. That
is Self-RAG's ``[Partially supported]`` exactly — "supported by the evidence to some extent, but
there is major information in the output that is not discussed in the evidence"
(``AkariAsai/self-rag``, MIT, ``data_creation/critic/gpt4_reward/chatgpt_groundness.py:48``;
vocabulary at ``retrieval_lm/utils.py:48-49``). Such a claim is now a :class:`PartialClaim`:
reported, serialised, counted in :attr:`ClaimReport.support`, and deliberately **not** a
contradiction — :attr:`ClaimReport.sound` keeps its meaning, because a thesis may be about a subset
of its evidence and the desk's flags, escalation and review all key on contradictions. Adapted, not
copied: Self-RAG reads the label from a fine-tuned model's token probabilities; here it is counted
from the records, and nothing is asked of a model.

**The conviction rule no longer fires on the word alone (fixed 2026-09-25).** It matched
``high-conviction`` anywhere, and the desk writes that about its own hurdle: "the 20.80bps total
hurdle requires a high-conviction directional call". Replayed over the record, six of the seven
contradictions the desk ever logged (seqs 175, 237, 461, 485, 520, 575) were this rule reading a
sentence about the hurdle as a claim about an insider trade, and every one of the eight theses on
the ledger that used the phrase used it that way. A contradiction that is not one is the most
expensive kind of false alarm in a record a judge reads. The rule now needs the claim to be about an
insider or a purchase, which is the claim its field can actually settle.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from argus.agents.grounding import Support, combine

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
        # About an insider or a purchase, within one clause — never "high-conviction" on its own,
        # which the desk writes about its hurdle (see the module docstring, and the regression
        # fixtures in tests/test_claims.py taken verbatim from the ledger).
        pattern=(
            r"(?:\binsider\b[^.;:]{0,40}?\bconviction\b|\bconviction\b[^.;:]{0,40}?\binsider\b"
            r"|\bconviction\s+(?:buy|buying|purchase)\b|\bstrong\s+insider\s+signal\b)"
        ),
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
class PartialClaim:
    """A claim some of its relevant records support and others refute.

    Not a contradiction — the thesis is true of part of what it describes — and not full support
    either: it generalises past its evidence. Both sides are named so a reader can see which.
    """

    rule: str
    claim_text: str
    field: str
    asserted: bool
    agreeing: tuple[str, ...]
    disagreeing: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "claim": self.claim_text,
            "field": self.field,
            "asserted": self.asserted,
            "agreeing": list(self.agreeing),
            "disagreeing": list(self.disagreeing),
        }

    def render(self) -> str:
        """Worded to contain none of the finding phrases `paper/runner.py` flags on: this is a
        qualification of a claim that passed, not a contradiction."""
        shown = ", ".join(self.disagreeing[:3]) + (
            f" and {len(self.disagreeing) - 3} more" if len(self.disagreeing) > 3 else ""
        )
        return (
            f"[claim] {self.rule}: partially supported — \"{self.claim_text}\" holds for "
            f"{len(self.agreeing)} of {len(self.agreeing) + len(self.disagreeing)} record(s) "
            f"carrying {self.field}, not for {shown}; the thesis generalises past its evidence"
        )


@dataclass(frozen=True)
class ClaimReport:
    contradictions: tuple[Contradiction, ...]
    claims_examined: int
    records_checked: int
    partial: tuple[PartialClaim, ...] = ()
    """Examined claims that some relevant records support and others refute."""

    @property
    def sound(self) -> bool:
        return not self.contradictions

    @property
    def fully_supported(self) -> int:
        """Examined claims every relevant record agrees with."""
        return self.claims_examined - len(self.contradictions) - len(self.partial)

    @property
    def support(self) -> Support | None:
        """Self-RAG's three levels over the examined claims; ``None`` when none was examined.

        ``None`` rather than full support on purpose: this module's founding rule is that a claim
        it could not settle is never reported as passing, and a thesis-level "fully supported" over
        zero examined claims would be exactly that.
        """
        if not self.claims_examined:
            return None
        return combine(
            [Support.FULL] * self.fully_supported
            + [Support.PARTIAL] * len(self.partial)
            + [Support.NONE] * len(self.contradictions)
        )

    def as_dict(self) -> dict[str, Any]:
        support = self.support
        return {
            "sound": self.sound,
            "claims_examined": self.claims_examined,
            "records_checked": self.records_checked,
            "support": None if support is None else str(support),
            "contradictions": [c.as_dict() for c in self.contradictions],
            "partial": [p.as_dict() for p in self.partial],
        }

    def render(self) -> list[str]:
        if not self.records_checked:
            return ["[claim] no structured evidence to check the thesis against"]
        if not self.claims_examined:
            return [
                f"[claim] no checkable claim found in the thesis "
                f"({self.records_checked} structured record(s) available)"
            ]
        if self.sound and not self.partial:
            return [
                f"[claim] {self.claims_examined} checkable claim(s) agree with the "
                f"{self.records_checked} structured record(s) behind them"
            ]
        if self.sound:
            return [
                f"[claim] {self.claims_examined} checkable claim(s) examined against the "
                f"{self.records_checked} structured record(s) behind them: "
                f"{self.fully_supported} fully supported, {len(self.partial)} partially"
            ] + [p.render() for p in self.partial]
        return [c.render() for c in self.contradictions] + [p.render() for p in self.partial] + [
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
    subset of its evidence. When some records agree and some do not, the claim is a
    :class:`PartialClaim` — passing, and reported as reaching past its evidence.
    """
    contradictions: list[Contradiction] = []
    partial: list[PartialClaim] = []
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
        start = max(0, match.start() - 30)
        if agreeing:
            disagreeing = [rid for rid, attrs in relevant
                           if bool(attrs[rule.field]) is not rule.asserts]
            if disagreeing:
                partial.append(
                    PartialClaim(
                        rule=rule.name,
                        claim_text=thesis[start: match.end() + 30].strip(),
                        field=rule.field,
                        asserted=rule.asserts,
                        agreeing=tuple(agreeing),
                        disagreeing=tuple(disagreeing),
                    )
                )
            continue

        rid, attrs = relevant[0]
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
        partial=tuple(partial),
    )


__all__ = [
    "NEGATION_WINDOW",
    "RULES",
    "ClaimReport",
    "Contradiction",
    "PartialClaim",
    "Rule",
    "check",
]
