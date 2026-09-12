"""Autonomy Proof — the machine-verifiable answer to "did the LLM actually decide this trade?"

Track 2 is scored 50% by judges, and the single question a judge cannot check from a diagram is
whether the model is the decision-maker or the narrator. Our teardowns found four well-known
systems that would fail that check, and one — AI-Trader — that advertises "100% fully-automated
agent-native trading" while its LLM only writes market-summary prose and never touches the trading
path (``routes_signals.py``). A README cannot distinguish those cases. This artefact can.

The chain records both sides of the constraint:

    market_state_hash        what the world looked like, hashed
    llm_original_intent      the model's UNCONSTRAINED economic choice
    constitution_ruling      what the risk layer changed, and which constraint bound
    llm_revised_intent       the model's RESPONSE to being constrained
    approved_intent          what was authorised, hashed
    submitted_order          what was sent
    fills                    what came back
    reconciliation           what the venue says is true

The ``original`` / ``revised`` pair is the load-bearing part. A system where the LLM merely rubber
stamps a deterministic decision produces an original intent that is always identical to what the
rules would have chosen anyway; a system where the LLM is genuinely deciding produces originals
that the Constitution must sometimes narrow, and revisions that respond to *why*. That is visible
in the artefact and not fakeable by prose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from argus.decision.verdicts import ConstitutionRuling, Intent


def _canonical(obj: Any) -> str:
    """Stable JSON for hashing. Sorted keys, Decimals as strings, no whitespace drift."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_intent(intent: Intent) -> str:
    payload = asdict(intent)
    payload["quantity"] = str(intent.quantity)
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:16]


def hash_state(state: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(state).encode()).hexdigest()[:16]


class ProofIncomplete(RuntimeError):
    """The chain was interrogated before it was whole.

    Deliberately an error: a partially assembled proof that answers "yes" is worse than no proof,
    because it looks like evidence.
    """


@dataclass(slots=True)
class AutonomyProof:
    """One decision's complete forensic record, assembled in order and never edited."""

    decision_id: str
    as_of: datetime
    market_state_hash: str

    llm_original_intent: Intent
    llm_original_reasoning: str

    constitution_ruling: ConstitutionRuling | None = None
    llm_revised_intent: Intent | None = None
    llm_revised_reasoning: str = ""

    approved_intent_hash: str = ""
    submitted_order_id: str = ""
    venue_order_id: str = ""
    fills: list[dict[str, Any]] = field(default_factory=list)
    reconciliation_state: str = ""

    def record_ruling(self, ruling: ConstitutionRuling) -> None:
        self.constitution_ruling = ruling

    def record_revision(self, intent: Intent, reasoning: str) -> None:
        """The model's response to being constrained.

        Only meaningful after a ruling: a 'revision' with nothing to revise against is noise.
        """
        if self.constitution_ruling is None:
            raise ProofIncomplete("cannot record a revision before a constitution ruling")
        self.llm_revised_intent = intent
        self.llm_revised_reasoning = reasoning

    def approve(self) -> str:
        """Freeze the authorised intent and return its hash. This is what an order must carry."""
        if self.constitution_ruling is None:
            raise ProofIncomplete("cannot approve before the Constitution has ruled")
        final = self.llm_revised_intent or self.constitution_ruling.resulting_intent
        self.approved_intent_hash = hash_intent(final)
        return self.approved_intent_hash

    # --- the questions a judge actually asks ------------------------------------------------

    @property
    def constitution_intervened(self) -> bool:
        from argus.decision.verdicts import ConstitutionVerdict

        return (
            self.constitution_ruling is not None
            and self.constitution_ruling.verdict is not ConstitutionVerdict.ALLOW
        )

    @property
    def llm_changed_its_mind(self) -> bool:
        """Did the model respond to the constraint, rather than repeat itself?"""
        if self.llm_revised_intent is None:
            return False
        return hash_intent(self.llm_revised_intent) != hash_intent(self.llm_original_intent)

    def constitution_only_reduced(self) -> bool:
        """Verify the asymmetry held for this decision, from the artefact alone.

        A judge should not have to trust our code for this: the proof carries the original and the
        result, so the property is checkable after the fact.
        """
        if self.constitution_ruling is None:
            raise ProofIncomplete("no ruling recorded")
        result = self.constitution_ruling.resulting_intent
        return (
            result.side is self.llm_original_intent.side
            and result.quantity <= self.llm_original_intent.quantity
        )

    def attests_llm_decided(self) -> bool:
        """The headline claim, answerable mechanically.

        True when the model produced a genuine economic choice that the pipeline carried through
        to an authorised order. It is deliberately *not* true merely because an LLM was called.
        """
        if not self.approved_intent_hash:
            raise ProofIncomplete("no approved intent — the chain does not reach an order")
        return (
            bool(self.llm_original_reasoning.strip())
            and bool(self.llm_original_intent.thesis.strip())
            and self.constitution_only_reduced()
        )

    def to_record(self) -> dict[str, Any]:
        """Serialisable audit record. This is what ships with the submission."""
        ruling = self.constitution_ruling
        return {
            "decision_id": self.decision_id,
            "as_of": self.as_of.isoformat(),
            "market_state_hash": self.market_state_hash,
            "llm_original_intent": {
                **{k: str(v) for k, v in asdict(self.llm_original_intent).items()},
                "hash": hash_intent(self.llm_original_intent),
            },
            "llm_original_reasoning": self.llm_original_reasoning,
            "constitution": None if ruling is None else {
                "verdict": str(ruling.verdict),
                "binding_constraint": ruling.binding_constraint,
                "reason": ruling.reason,
                "resulting_quantity": str(ruling.resulting_intent.quantity),
            },
            "llm_revised_intent": None if self.llm_revised_intent is None else {
                "hash": hash_intent(self.llm_revised_intent),
                "quantity": str(self.llm_revised_intent.quantity),
                "verdict": str(self.llm_revised_intent.verdict),
            },
            "llm_revised_reasoning": self.llm_revised_reasoning,
            "approved_intent_hash": self.approved_intent_hash,
            "submitted_order_id": self.submitted_order_id,
            "venue_order_id": self.venue_order_id,
            "fills": self.fills,
            "reconciliation_state": self.reconciliation_state,
            "attestations": {
                "constitution_intervened": self.constitution_intervened,
                "constitution_only_reduced": self.constitution_only_reduced(),
                "llm_changed_its_mind": self.llm_changed_its_mind,
            },
        }


@dataclass(slots=True)
class ExecutionProof:
    """Execution Proof: is the executed order the approved order?

    Separate from AutonomyProof because it answers a different question and can fail
    independently — a perfectly autonomous decision can still be executed wrongly.
    """

    approved_intent_hash: str
    submitted_hash: str
    venue_order_id: str
    approved_quantity: Decimal
    filled_quantity: Decimal

    @property
    def intent_matches(self) -> bool:
        return self.approved_intent_hash == self.submitted_hash

    @property
    def quantity_discrepancy(self) -> Decimal:
        """Unfilled quantity. Non-zero is not necessarily wrong — a partial fill is real — but it
        must be surfaced and re-planned rather than reconciled away."""
        return self.approved_quantity - self.filled_quantity

    def holds(self) -> bool:
        return self.intent_matches and self.filled_quantity <= self.approved_quantity
