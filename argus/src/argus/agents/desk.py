"""The Track-2 desk: five analysts, one decision, one proof.

This is the complete Agentic Trading loop and the artefact the track is judged on:

    evidence -> five sub-theme analysts -> source-independence discount
             -> Meta-PM decides -> Constitution narrows -> Meta-PM responds
             -> signed order -> Autonomy Proof

Every named Track-2 sub-theme is an entry point into this one loop rather than a separate product:
event, sentiment, earnings and cross-asset execution are analysts inside it; factor discovery feeds
it; the Open Theme is the evaluation harness around it.

**The property that makes it pass the positioning rule.** The analysts never decide — they return
intelligence. The Constitution never creates — it may only reduce. So the economic choice is the
model's, and the risk layer is still binding. Four of the best-known systems in this field fail one
half of that: AI-Trader's LLM only writes summaries, Vibe-Trading's prompt forbids recommendations,
inalpha keeps the model off the order path by design, FinRobot has no broker at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from argus.agents.analysts import (
    AnalystView,
    CrossAssetAnalyst,
    EarningsAnalyst,
    EventAnalyst,
    Evidence,
    SentimentAnalyst,
    SourceIndependenceGraph,
)
from argus.agents.causality import CausalChain
from argus.agents.earnings import EarningsRead
from argus.agents.meta_pm import MarketFrame, MetaPM, deliberation_cost_bps
from argus.cost.model import CostModel
from argus.decision.verdicts import (
    ConstitutionRuling,
    ConstitutionVerdict,
    Intent,
    apply_constraint,
)
from argus.execution.orders import Order, OrderBook
from argus.llm.base import ChatModel
from argus.llm.qwen import Thinking
from argus.proof.autonomy import AutonomyProof
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionState


@dataclass
class DeskRun:
    """One complete pass. The serialisable record is the Track-2 submission artefact."""

    symbol: str
    as_of: datetime
    panel: SourceIndependenceGraph
    proof: AutonomyProof
    ruling: ConstitutionRuling | None = None
    order: Order | None = None
    notes: list[str] = field(default_factory=list)
    causal_chain: CausalChain | None = None
    """The event transmission chain, ready to be graded at the next price discovery. A chain that
    reaches the right direction through broken links is a failure, and only this carries it."""

    earnings_read: EarningsRead | None = None
    """The seven surprises, kept apart. A beat with cut guidance is a different asset."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat(),
            "panel": self.panel.as_dict(),
            "decision": self.proof.to_record(),
            "order": None if self.order is None else {
                "client_order_id": self.order.client_order_id,
                "state": str(self.order.state),
                "quantity": str(self.order.quantity),
                "approved_intent_hash": self.order.approved_intent_hash,
                "history": [
                    {"from": str(t.frm), "to": str(t.to), "reason": t.reason}
                    for t in self.order.history
                ],
            },
            "notes": self.notes,
            "causal_chain": None if self.causal_chain is None else self.causal_chain.as_dict(),
            "earnings": None if self.earnings_read is None else self.earnings_read.as_dict(),
        }


class TradingDesk:
    """Assembles the analysts, the decision-maker, the Constitution and the order book."""

    def __init__(
        self,
        client: ChatModel,
        *,
        cost: CostModel | None = None,
        analyst_thinking: Thinking = Thinking.LOW,
        annualised_vol: Decimal = Decimal("0.45"),
    ) -> None:
        self._client = client
        self._cost = cost or CostModel.bitget_perp()
        # Used to price the desk's own deliberation time. 45% is a plausible single-name rToken
        # vol and is stated here so a caller can pass the measured one instead of inheriting a
        # constant it never saw.
        self.annualised_vol = annualised_vol
        self.event = EventAnalyst(client, thinking=analyst_thinking)
        self.sentiment = SentimentAnalyst(client, thinking=analyst_thinking)
        self.earnings = EarningsAnalyst(client, thinking=analyst_thinking)
        self.cross_asset = CrossAssetAnalyst(client, thinking=analyst_thinking)
        # The Meta-PM gets the full reasoning budget. It is the only call that decides anything,
        # and it streams so it stays under the endpoint's 120s gateway timeout.
        self.pm = MetaPM(client, max_tokens=900)
        self.book = OrderBook()

    def run(
        self,
        *,
        symbol: str,
        session: SessionState,
        token_price: Decimal,
        position: Decimal,
        evidence: list[Evidence],
        hedges: HedgeabilitySurface,
        decision_id: str,
        constitution: ConstitutionPolicy | None = None,
    ) -> DeskRun:
        panel = SourceIndependenceGraph()
        notes: list[str] = []

        # --- 1. the analysts. Each sees only evidence bounded by the decision instant. ---
        event_evidence = [e for e in evidence if e.source in ("sec-edgar", "news", "macro")]
        social_evidence = [e for e in evidence if e.source == "social"]
        earnings_evidence = [e for e in evidence if e.source in ("filing", "transcript")]

        causal_chain: CausalChain | None = None
        earnings_read: EarningsRead | None = None

        if event_evidence:
            view, causal_chain = self.event.analyse_with_chain(symbol, session, event_evidence)
            panel.add(view)
            if causal_chain.links:
                notes.append(
                    f"causal chain: {len(causal_chain.links)} links stated, gradable at the "
                    f"next price discovery"
                )
        if social_evidence:
            panel.add(self.sentiment.analyse(symbol, social_evidence))
        if earnings_evidence:
            view, earnings_read = self.earnings.analyse_decomposed(symbol, earnings_evidence)
            panel.add(view)
            if earnings_read.surprises.is_contradictory:
                notes.append(
                    f"earnings print contradicts itself: headline "
                    f"{earnings_read.surprises.headline:+.2f}, forward "
                    f"{earnings_read.surprises.forward:+.2f} — {earnings_read.dominant} dominates"
                )

        menu = [
            {
                "instrument": c.instrument,
                "risk_reduction": str(round(c.effective_risk_reduction, 4)),
                "cost_bps": str(c.all_in_cost_bps),
                "execution_probability": str(c.execution_probability),
                "efficiency": str(round(c.risk_neutralisation_efficiency, 4)),
            }
            for c in hedges.menu
        ]
        panel.add(self.cross_asset.analyse(
            symbol, session, menu, position_notional=position * token_price
        ))

        if hedges.is_empty:
            notes.append(
                f"hedge menu empty: nothing placeable for {session.hours_to_next_discovery:.1f}h; "
                f"100% of risk carried as priced residual"
            )

        deliberation = round(
            deliberation_cost_bps(
                session, thinking=Thinking.FULL, annualised_vol=self.annualised_vol
            ),
            2,
        )
        if deliberation > self._cost.round_trip_bps():
            notes.append(
                f"deliberation costs {deliberation}bps against a "
                f"{self._cost.round_trip_bps()}bps round trip: in this session, thinking about "
                f"the trade costs more than the trade. Hurdle is "
                f"{self._cost.round_trip_bps() + deliberation}bps, not the fee alone"
            )

        consensus_signal, consensus_confidence = panel.consensus()
        notes.append(
            f"panel: {len(panel.views)} analysts, {len(panel.distinct_sources)} distinct sources, "
            f"independence {panel.independence_ratio:.2f} -> {consensus_signal} "
            f"at {consensus_confidence:.2f} after provenance discount"
        )

        # --- 2. the decision. The model sees the panel as evidence, and decides. ---
        frame = MarketFrame(
            symbol=symbol,
            as_of=session.as_of,
            session=session,
            token_price=token_price,
            position_quantity=position,
            round_trip_bps=self._cost.round_trip_bps(),
            # The model is told what its own reasoning costs. Off-hours a full budget is 15.2bps
            # against a 12bps fee (§3.4), so omitting this would understate the hurdle by more than
            # half in exactly the sessions the desk runs unattended.
            deliberation_bps=deliberation,
            hedge_menu=tuple(menu),
            evidence=tuple(
                [e.render() for e in evidence]
                + [
                    f"[panel] {v.analyst}: {v.signal} {v.magnitude_bps}bps "
                    f"(confidence {v.confidence:.2f}) — {v.reasoning}"
                    for v in panel.views
                ]
                + [
                    f"[panel] consensus {consensus_signal} at {consensus_confidence:.2f}, "
                    f"discounted for {len(panel.distinct_sources)} distinct sources across "
                    f"{len(panel.views)} analysts"
                ]
            ),
        )
        proof = self.pm.decide(frame, decision_id=decision_id)

        # --- 3. the Constitution. It may only reduce. ---
        policy = constitution or ConstitutionPolicy()
        ruling = policy.rule(proof.llm_original_intent, session=session, hedges=hedges)
        self.pm.revise(proof, frame, ruling)
        approved_hash = proof.approve()

        # --- 4. the order, bound to the approved intent. ---
        final = proof.llm_revised_intent or ruling.resulting_intent
        order = None
        if final.quantity > 0 and final.verdict.carries_quantity:
            order = Order(
                client_order_id=f"{decision_id}-1",
                symbol=symbol,
                side=str(final.side).upper(),
                quantity=final.quantity,
                approved_intent_hash=approved_hash,
            )
            if ruling.verdict is ConstitutionVerdict.REJECT:
                self.book.deny(order, at=session.as_of, reason=ruling.reason)
            else:
                self.book.submit(order, at=session.as_of)
        else:
            notes.append(f"no order: final verdict {final.verdict} with quantity {final.quantity}")

        return DeskRun(
            symbol=symbol, as_of=session.as_of, panel=panel,
            proof=proof, ruling=ruling, order=order, notes=notes,
            causal_chain=causal_chain, earnings_read=earnings_read,
        )


@dataclass(frozen=True, slots=True)
class ConstitutionPolicy:
    """Deterministic risk rules, in versioned config rather than in a prompt.

    TradingAgents' own teardown records that its risk management is "via prompt guidance, not code
    enforcement". A rule a model can be talked out of is not a rule, so every limit here is applied
    in Python and the model is told the outcome rather than asked to respect it.
    """

    max_position_notional: Decimal = Decimal("50000")
    max_unhedged_notional: Decimal = Decimal("20000")
    """Ceiling on exposure carried with an empty hedge menu — the Sleeping-Anchor constraint."""

    min_confidence_to_trade: float = 0.55

    def rule(
        self, intent: Intent, *, session: SessionState, hedges: HedgeabilitySurface
    ) -> ConstitutionRuling:
        if not intent.verdict.carries_quantity or intent.quantity <= 0:
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.ALLOW,
                binding_constraint="none", reason="no exposure proposed; nothing to narrow",
            )

        if intent.stated_confidence < self.min_confidence_to_trade:
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.REJECT,
                binding_constraint="min_confidence",
                reason=(
                    f"stated confidence {intent.stated_confidence:.2f} is below the "
                    f"{self.min_confidence_to_trade} floor"
                ),
            )

        if session.nav_is_stale() and session.is_anchor_asleep:
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.DELAY,
                binding_constraint="oracle_stale",
                reason="NAV is stale while the anchor is shut; no reliable reference price",
            )

        if hedges.is_empty and intent.quantity > self.max_unhedged_notional:
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.RESIZE,
                binding_constraint="unhedgeable_gap",
                reason=(
                    f"no hedge placeable for {session.hours_to_next_discovery:.1f}h; "
                    f"unhedged exposure capped at {self.max_unhedged_notional}"
                ),
                resized_quantity=self.max_unhedged_notional,
            )

        if intent.quantity > self.max_position_notional:
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.RESIZE,
                binding_constraint="max_position",
                reason=f"position capped at {self.max_position_notional}",
                resized_quantity=self.max_position_notional,
            )

        return apply_constraint(
            intent, verdict=ConstitutionVerdict.ALLOW,
            binding_constraint="none", reason="within every configured limit",
        )


__all__ = ["AnalystView", "ConstitutionPolicy", "DeskRun", "Evidence", "TradingDesk"]
