"""Meta-PM — the LLM in the decision seat.

This module is where ARGUS either satisfies Bitget's Track 2 positioning rule or fails it. The rule
requires the LLM to be the *primary trading decision-maker*, so three properties are enforced here:

1. **The model chooses the direction and the size.** Nothing in this module computes a side or a
   quantity from a rule and asks the model to agree. If the deterministic layer were choosing, the
   Autonomy Proof would show it — the original intent would never need narrowing.

2. **The model is shown what it cannot do, and why, and gets to respond.** After the Constitution
   rules, the model is re-prompted with the binding constraint and produces a revised intent. That
   round trip is what distinguishes a decision-maker from a proposal generator.

3. **Numbers come from code.** Every figure in the prompt — session state, hours to discovery, cost
   in bps, hedge availability — is computed and injected. The model interprets them; it never
   produces them. This is FinRobot's shipped discipline (``financial_data_processor.py`` computes,
   agents narrate, prompts carry an explicit "do not estimate" instruction), adopted verbatim as
   policy because LLM arithmetic on financial values is an anti-pattern we named and banned.

The prompt deliberately offers ``DATA_INSUFFICIENT`` and ``NO_TRADE`` as first-class answers.
A model that can only pick a direction will always pick one, and our own measurements say the
honest answer on a 0.12% round-trip fee against a ~0.00% intraday edge is usually "no".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from argus.decision.verdicts import (
    ConstitutionRuling,
    Intent,
    Side,
    Verdict,
)
from argus.execution.latency import thinking_budget_cost_bps
from argus.llm.base import ChatModel
from argus.llm.qwen import Thinking
from argus.proof.autonomy import AutonomyProof, hash_state
from argus.truth.clocks import SessionPhase, SessionState

# Measured Qwen wall-clock at each thinking budget, from our own bake-off. These are the numbers
# `deliberation_cost_bps` charges against; they are stated here rather than buried so a change in
# model or endpoint is a visible edit.
THINKING_MS: dict[Thinking, int] = {
    Thinking.OFF: 3_000,
    Thinking.LOW: 8_000,
    Thinking.FULL: 40_000,
}

# Off-hours book depth runs near a third of RTH, so a given delay crosses three times as much book.
# Measured in §3.4; the same number the passive fill model uses.
DEPTH_MULTIPLIER: dict[SessionPhase, Decimal] = {
    SessionPhase.RTH: Decimal("1"),
    SessionPhase.EXTENDED: Decimal("2"),
    SessionPhase.OVERNIGHT: Decimal("3"),
    SessionPhase.WEEKEND: Decimal("3"),
    SessionPhase.HOLIDAY: Decimal("3"),
}


def deliberation_cost_bps(
    session: SessionState, *, thinking: Thinking = Thinking.FULL, annualised_vol: Decimal
) -> Decimal:
    """What it costs, in basis points, to think this hard in this session.

    This is the term the field omits entirely. Every finance-agent harness we tore down treats model
    latency as free, which is true at the millisecond scale they inherited from HFT tooling and
    false at the ten-second scale a reasoning model actually runs at: off-hours, a full budget costs
    more than the round-trip fee.
    """
    return thinking_budget_cost_bps(
        think_ms=THINKING_MS[thinking],
        annualised_vol=annualised_vol,
        depth_multiplier=DEPTH_MULTIPLIER.get(session.phase, Decimal("3")),
    )

SYSTEM_PROMPT = """You are the portfolio manager for ARGUS, trading tokenized US equities.

You are the decision-maker. Nothing downstream will invent a trade for you; a risk layer may only
reduce what you choose, never create or reverse it. If you do not decide, nothing happens.

Hard rules:
- Every number you are given was computed by code. Do NOT estimate, recompute, or adjust any
  figure. Interpret them.
- You must clear the transaction cost. A round trip costs the stated bps. An edge smaller than
  that is a loss, not a small gain.
- NO_TRADE and DATA_INSUFFICIENT are correct answers and are scored as decisions, not failures.
  NO_TRADE means the evidence was adequate and the answer is no. DATA_INSUFFICIENT means the
  evidence was not adequate to decide. They are different; choose precisely.
- If you open exposure you must state at least one invalidation condition: a specific, checkable
  observation that would prove your thesis wrong.

Return ONLY a JSON object:
{
  "verdict": "TRADE" | "REDUCE" | "HEDGE" | "DELAY" | "NO_TRADE" | "HUMAN_REVIEW"
             | "DATA_INSUFFICIENT",
  "side": "BUY" | "SELL",
  "quantity": <number, units of the symbol; 0 if not opening exposure>,
  "confidence": <number between 0 and 1>,
  "thesis": "<why, in one or two sentences>",
  "invalidation": ["<what would prove this wrong>", ...],
  "counter_case": "<the strongest argument against your own decision>",
  "lean": "UP" | "DOWN" | "NONE",
  "lean_confidence": <number between 0 and 1>
}

About "lean". When you decline to trade, you are saying the edge does not clear the hurdle — you
are NOT saying you have no view about direction. State the view anyway: if you were forced to take
a position at this instant, which way would it be? "NONE" is a legitimate answer and means you
genuinely cannot call the direction, not that the trade is unattractive.

This costs you nothing and it is the only way your judgement can be scored while you are correctly
standing aside. It is recorded, settled against the move that actually followed, and graded. Answer
it as carefully as you answer the verdict."""


@dataclass(frozen=True, slots=True)
class MarketFrame:
    """Everything the model is allowed to see, all of it computed by code.

    Constructed from a ``DecisionContext`` at a single ``as_of``. Any field added here must be
    derivable from evidence bounded by that instant — this is the boundary the two-clock PIT model
    protects, and a field that slips in from a later timestamp is look-ahead wearing a hat.
    """

    symbol: str
    as_of: datetime
    session: SessionState
    token_price: Decimal
    position_quantity: Decimal
    round_trip_bps: Decimal
    hedge_menu: tuple[dict[str, Any], ...] = ()
    evidence: tuple[str, ...] = ()
    debate_block: str = ""
    """The bull case, the bear case, and whether they were resolved.

    Placed after the evidence and before the decision, because a debate is a reading OF the
    evidence and must not be mistaken for more evidence. An unresolved debate reaches the PM as an
    unresolved disagreement rather than as an average of two views that neither side holds — see
    :mod:`argus.agents.debate`."""

    memory_block: str = ""
    """What this desk did on this symbol before, and what it cost.

    Arithmetic over the hash-chained ledger, never a model's recollection — see
    :mod:`argus.agents.recall`. It sits *after* the mandate and *before* the market state so the
    reasoning reads it as history rather than as instruction, and every line in it is a number a
    reader can recompute from ``data/paper_ledger.jsonl``.

    Empty when the desk has not seen this symbol before, and the frame then says so explicitly
    rather than omitting the section. An absent heading reads as "no memory exists"; a heading
    saying "first look" reads as "memory exists and is empty", and only the second is true."""

    mandate_block: str = ""
    """Whose money this is, stated to the decision-maker **before** it reasons.

    The audit against Vibe-Trading (`research/architecture/personalisation-audit.md`) found the
    difference that matters: Vibe-Trading injects the user's mandate into the model's context so the
    reasoning is shaped by it, while ARGUS applied the profile *after* the model had already decided
    and merely narrowed the result. Both produce a compliant position; only one produces a
    *personalised thesis*, which is what this track scores. A constraint applied afterwards is
    invisible in the sentence a judge reads.

    It stays a constraint, not a suggestion: `agents/mandate.py` still enforces every limit in code
    after the fact, so a model that ignores this block is still bound. Telling it first means the
    reasoning is about the right question; checking it after means the answer is still correct."""

    deliberation_bps: Decimal = Decimal("0")
    """Expected adverse move over the time this decision takes to make.

    Measured, not assumed: see §3.4. A full reasoning budget off-hours costs **15.2bps** against a
    12bps round trip, so the model's own thinking is a larger cost than the trade. Telling the model
    its hurdle *without* this term would understate it by more than half in exactly the sessions
    where the desk runs unattended.
    """

    @property
    def total_hurdle_bps(self) -> Decimal:
        """What the edge actually has to beat: the fee plus the cost of deciding."""
        return self.round_trip_bps + self.deliberation_bps

    def to_prompt_block(self) -> str:
        hedges = (
            "\n".join(
                f"  - {h['instrument']}: risk reduction {h['risk_reduction']}, "
                f"cost {h['cost_bps']}bps, execution probability {h['execution_probability']}"
                for h in self.hedge_menu
            )
            if self.hedge_menu
            else "  (none — no hedge is placeable in this session)"
        )
        evidence = (
            "\n".join(f"  - {e}" for e in self.evidence) if self.evidence else "  (none)"
        )
        mandate = self.mandate_block or (
            "  (none stated — decide for a general mandate, and say so in the thesis)"
        )
        memory = self.memory_block or (
            "  (no prior decision on this symbol — this is the first look)"
        )
        debate = self.debate_block or "  (no debate was held for this decision)"
        return f"""SYMBOL: {self.symbol}
AS OF: {self.as_of.isoformat()}

WHOSE MONEY THIS IS
{mandate}
  Your thesis must say how this mandate shaped the decision. A recommendation that would read
  identically for any trader has not been personalised, it has merely been filtered afterwards.

WHAT THIS DESK ALREADY LEARNED HERE
{memory}

THE ARGUMENT ON BOTH SIDES
{debate}
  This is a reading of the evidence above, not additional evidence. Where the two sides did not
  converge, say which one you are siding with and why — an answer that splits the difference has
  adopted a view neither analyst holds.

SESSION STATE (anchor market)
  phase: {self.session.phase}
  anchor asleep: {self.session.is_anchor_asleep}
  hours to next genuine price discovery: {self.session.hours_to_next_discovery:.1f}
  NAV stale: {self.session.nav_is_stale()}

POSITION
  current quantity: {self.position_quantity}
  token price: {self.token_price}

COSTS
  round trip: {self.round_trip_bps} bps
  cost of deciding: {self.deliberation_bps} bps — the price moves while you reason, and this
    session's book depth is already priced in
  TOTAL HURDLE: {self.total_hurdle_bps} bps — your edge must exceed this, not the fee alone

HEDGE MENU (ranked, computed)
{hedges}

EVIDENCE (each item was available at or before AS OF)
{evidence}"""

    def state_hash(self) -> str:
        return hash_state({
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat(),
            "phase": str(self.session.phase),
            "hours_to_discovery": round(self.session.hours_to_next_discovery, 4),
            "token_price": str(self.token_price),
            "position": str(self.position_quantity),
            "round_trip_bps": str(self.round_trip_bps),
            "deliberation_bps": str(self.deliberation_bps),
            "hedges": [dict(sorted(h.items())) for h in self.hedge_menu],
            "evidence": list(self.evidence),
        })


class MetaPM:
    """The decision-maker. Produces an intent, then responds to being constrained."""

    def __init__(
        self, client: ChatModel, *, max_tokens: int = 1024, thinking: Thinking = Thinking.FULL
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        # The budget is a *cost*, priced by `deliberation_cost_bps`: off-hours a FULL budget
        # costs 15.2bps against a 12bps round trip, and a desk quoted that hurdle correctly
        # abstains every cycle. Routine paper cycles run LOW so the record is scoreable; FULL is
        # reserved for decisions where the extra reasoning has a chance of paying for itself.
        self.thinking = thinking

    def decide(self, frame: MarketFrame, *, decision_id: str) -> AutonomyProof:
        """First pass: the model's unconstrained economic choice."""
        response = self._client.complete_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": frame.to_prompt_block()},
            ],
            required_keys=("verdict", "side", "quantity", "confidence", "thesis"),
            validate=_complain_about,
            max_tokens=self._max_tokens,
            # The decision that matters gets the full reasoning budget, and streams so it stays
            # under the endpoint's 120s gateway timeout.
            thinking=self.thinking,
        )
        intent = _to_intent(response, frame.symbol)
        return AutonomyProof(
            decision_id=decision_id,
            as_of=frame.as_of,
            market_state_hash=frame.state_hash(),
            llm_original_intent=intent,
            llm_original_reasoning=_reasoning_of(response),
        )

    def revise(
        self, proof: AutonomyProof, frame: MarketFrame, ruling: ConstitutionRuling
    ) -> AutonomyProof:
        """Second pass: the model is told what bound, and decides what to do about it.

        This is the round trip that separates a decision-maker from a proposal generator. The model
        may accept the narrowed size, abstain entirely, or restructure — but it cannot widen, and
        the Constitution will narrow again if it tries.
        """
        proof.record_ruling(ruling)

        constraint_block = f"""Your proposed action was constrained by the risk layer.

YOUR PROPOSAL
  verdict: {proof.llm_original_intent.verdict}
  side: {proof.llm_original_intent.side}
  quantity: {proof.llm_original_intent.quantity}

RISK LAYER RULING
  verdict: {ruling.verdict}
  binding constraint: {ruling.binding_constraint}
  reason: {ruling.reason}
  permitted quantity: {ruling.resulting_intent.quantity}

The risk layer can only reduce. It cannot increase your size or reverse your side.
Given this constraint, decide again. Accepting the reduced size is a valid answer; so is
abstaining entirely because the reduced size no longer justifies the cost."""

        response = self._client.complete_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": frame.to_prompt_block()},
                {"role": "assistant", "content": json.dumps({
                    "verdict": str(proof.llm_original_intent.verdict).upper(),
                    "quantity": str(proof.llm_original_intent.quantity),
                })},
                {"role": "user", "content": constraint_block},
            ],
            required_keys=("verdict", "side", "quantity", "confidence", "thesis"),
            max_tokens=self._max_tokens,
            thinking=self.thinking,
        )
        revised = _to_intent(response, frame.symbol)

        # The model cannot widen past what the Constitution permitted. Clamping here rather than
        # re-running the Kernel keeps the asymmetry a single enforced invariant.
        if revised.quantity > ruling.resulting_intent.quantity:
            revised = Intent(
                symbol=revised.symbol,
                side=revised.side,
                quantity=ruling.resulting_intent.quantity,
                verdict=revised.verdict,
                stated_confidence=revised.stated_confidence,
                thesis=revised.thesis,
                invalidation=revised.invalidation,
                required_hedge=ruling.resulting_intent.required_hedge,
            )

        proof.record_revision(revised, _reasoning_of(response))
        return proof


def _reasoning_of(response: dict[str, Any]) -> str:
    """The counter-case is the reasoning we keep: a decision that cannot argue against itself is
    not a judgement, and debate quality is one of the things Track 2 is scored on."""
    parts = [str(response.get("thesis", ""))]
    counter = str(response.get("counter_case", "")).strip()
    if counter:
        parts.append(f"Counter-case: {counter}")
    return " ".join(p for p in parts if p)


def _complain_about(response: dict[str, Any]) -> str:
    """What is wrong with this decision object, or "" if nothing is.

    Fed to `complete_json`, which retries with the complaint attached. The model puts a *side* in
    the verdict field — ``"verdict": "sell"`` — on roughly one call in five of the decision prompt,
    measured across five replays of one state (`data/consistency.json`). Presence of the key is not
    validity of its value, and `required_keys` only ever checked presence.

    Retrying with the specific failure is the mechanism `complete_json` was built around; routing a
    malformed verdict to HUMAN_REVIEW stays as the last resort after the retries are spent, not the
    first answer to a typo.
    """
    raw = str(response.get("verdict", "")).strip().lower()
    allowed = [v.value for v in Verdict]
    if raw not in allowed:
        side_like = raw in {"buy", "sell", "long", "short"}
        hint = (
            " That is a SIDE, not a verdict — put it in the \"side\" field and choose a verdict."
            if side_like else ""
        )
        return f"verdict {raw!r} is not one of {allowed}.{hint}"
    return ""


def _to_intent(response: dict[str, Any], symbol: str) -> Intent:
    # A verdict outside the enum is not a decision, and it must not be a crash either. The live
    # model returned ``"verdict": "sell"`` — a *side* in the verdict field — and the raw
    # ``Verdict(...)`` call raised ValueError straight out of the decision path, taking the whole
    # cycle with it. Nothing downstream can repair that: a malformed answer is unactionable however
    # confident it reads.
    #
    # Routed to HUMAN_REVIEW rather than guessed at, which is the same choice this function already
    # makes for exposure opened without a falsifier. Mapping "sell" onto TRADE would be inventing a
    # decision the model did not make, and mapping it to NO_TRADE would record a refusal it did not
    # make either. The raw value is carried into the thesis so the trail says what arrived.
    raw_verdict = str(response.get("verdict", "")).strip().lower()
    unparseable = ""
    try:
        verdict = Verdict(raw_verdict)
    except ValueError:
        verdict = Verdict.HUMAN_REVIEW
        unparseable = raw_verdict or "(empty)"
    side = Side(str(response.get("side", "buy")).strip().lower())
    quantity = Decimal(str(response.get("quantity", 0)))
    invalidation = tuple(
        str(x) for x in response.get("invalidation", []) if str(x).strip()
    )

    # A model that opens exposure without naming a falsifier has not finished thinking. Rather
    # than fabricate one, downgrade to HUMAN_REVIEW: the decision is real but not actionable.
    if verdict.opens_exposure and not invalidation:
        verdict = Verdict.HUMAN_REVIEW
        quantity = Decimal("0")

    if unparseable:
        quantity = Decimal("0")

    # A verdict that acts with size but names none is not actionable either. REDUCE 0 is a
    # sentiment, not an instruction.
    if verdict.carries_quantity and quantity <= 0:
        verdict = Verdict.HUMAN_REVIEW
        quantity = Decimal("0")

    # The lean is normalised here rather than trusted: an unknown value becomes "none", which is
    # the honest reading of an answer nobody can grade.
    raw_lean = str(response.get("lean", "none")).strip().lower()
    lean = raw_lean if raw_lean in {"up", "down", "none"} else "none"
    try:
        lean_confidence = float(response.get("lean_confidence", 0.0))
    except (TypeError, ValueError):
        lean_confidence = 0.0

    return Intent(
        symbol=symbol,
        side=side,
        quantity=quantity if verdict.carries_quantity else Decimal("0"),
        verdict=verdict,
        stated_confidence=float(response.get("confidence", 0.0)),
        thesis=(
            f"[unparseable verdict {unparseable!r}; routed to human review] "
            + (str(response.get("thesis", "")).strip() or "(no thesis given)")
            if unparseable
            else str(response.get("thesis", "")).strip() or "(no thesis given)"
        ),
        invalidation=invalidation,
        lean=lean,
        lean_confidence=max(0.0, min(1.0, lean_confidence)),
    )
