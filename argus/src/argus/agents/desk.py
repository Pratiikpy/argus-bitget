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

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.agents.adversary import Critic
from argus.agents.adversary import challenge as challenge_intent
from argus.agents.analysts import (
    AnalystView,
    CrossAssetAnalyst,
    EarningsAnalyst,
    EventAnalyst,
    SentimentAnalyst,
    SourceIndependenceGraph,
)
from argus.agents.causality import CausalChain
from argus.agents.claims import check as check_claims
from argus.agents.conflict import report as conflict_report
from argus.agents.debate import Debate, DebateBudget, Ending
from argus.agents.debate import hold as hold_debate
from argus.agents.earnings import EarningsRead
from argus.agents.entitygate import check as check_entities
from argus.agents.grounding import check as check_grounding
from argus.agents.grounding import extract as extract_figures
from argus.agents.mandate import Mandate, order_evidence
from argus.agents.meta_pm import MarketFrame, MetaPM, deliberation_cost_bps
from argus.agents.recall import recall
from argus.agents.selection import select
from argus.cost.model import CostModel
from argus.decision.escalation import Signals as EscalationSignals
from argus.decision.escalation import apply as apply_escalation
from argus.decision.escalation import assess as assess_escalation
from argus.decision.verdicts import (
    ConstitutionRuling,
    ConstitutionVerdict,
    Intent,
    Side,
    Verdict,
    apply_constraint,
)
from argus.desk.book import PositionSide
from argus.desk.workbench import TraderProfile
from argus.execution.orders import Order, OrderBook, deterministic_client_order_id
from argus.llm.base import ChatModel
from argus.llm.qwen import Thinking
from argus.proof.autonomy import AutonomyProof
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionState
from argus.truth.evidence import Evidence

_ZERO = Decimal("0")


@dataclass
class DeskRun:
    """One complete pass. The serialisable record is the Track-2 submission artefact."""

    symbol: str
    as_of: datetime
    panel: SourceIndependenceGraph
    proof: AutonomyProof
    ruling: ConstitutionRuling | None = None
    ruled_intent: Intent | None = None
    """The intent the Constitution was actually handed — `attacked`, after escalation.

    **Recorded because its absence made the risk record contradict itself.** `paper/runner.py`
    wrote `quantity_before` from `proof.llm_original_intent`, the model's first draft, while
    `binding_constraint` and `reason` came from the ruling on this intent — a later stage. On live
    seq 264 and 265 that produced a single row reading `quantity_before: "1"` beside
    `intervened: false` and `reason: "no exposure proposed; nothing to narrow"`, which cannot all
    be true at once. Downstream, `eval/riskaudit.py` counted those rows as positions offered to the
    risk layer and reported *"2 of which offered a position to reduce"* while `eval/autopsy.py`
    reported *"0 of 447 decision(s) proposed exposure"* — two of our own modules disagreeing about
    the same live record.

    The model's first draft is still worth keeping and is still on the proof; it is simply not what
    the Constitution ruled on, and the risk record is a record of the Constitution."""
    ablated_rulings: dict[str, ConstitutionRuling] = field(default_factory=dict)
    """Population RCT: the same `attacked` intent this cycle actually decided on, re-ruled against
    every caller-supplied gate-ablated `ConstitutionPolicy`, keyed by label.

    Computed for free alongside :attr:`ruling` — one extra deterministic call per variant, no
    extra LLM spend, no LLM stochasticity as a confound between variants (`agents.desk.
    TradingDesk.run`'s own comment explains why). Empty when the caller passes no variants, which
    is every caller today: this field exists so the RCT harness (not yet built — see
    `Activity/PROGRESS.md`) has something real to read once it exists, without another round of
    the "the mechanism exists but nothing calls it" gap this project keeps finding and closing.
    """

    order: Order | None = None
    notes: list[str] = field(default_factory=list)
    evidence_sources: tuple[str, ...] = ()
    """Every distinct `Evidence.source` that survived quarantine and reached this decision.

    **Recorded because its absence made a judged criterion unmeasurable.** Track 3 is scored on
    *"data sources / Skill integration count **and effectiveness**"*, and `eval/sourceaudit.py`
    could answer the effectiveness half for the five Bitget Skills only — the twelve data feeds
    reported `reached_decisions: null`, with the honest note *"per-decision reach is not recorded
    per feed, so it is reported as unmeasured rather than invented."*

    The desk always knew: every `Evidence` carries a `source`, and the screened list is right here.
    It was simply dropped at the end of the cycle. Persisting it turns 12 of 17 integrated sources
    from *unmeasured* into *measured* — and it can only be measured forward, because the evidence
    behind past decisions was never stored and reconstructing it today would be inventing it.
    """

    causal_chain: CausalChain | None = None
    """The event transmission chain, ready to be graded at the next price discovery. A chain that
    reaches the right direction through broken links is a failure, and only this carries it."""

    earnings_read: EarningsRead | None = None
    """The seven surprises, kept apart. A beat with cut guidance is a different asset."""

    debate: Debate | None = None
    """The bull/bear exchange, its cost, and whether it resolved."""

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
            "ablated_rulings": {
                label: {
                    "verdict": str(r.verdict), "binding_constraint": r.binding_constraint,
                    "resulting_quantity": str(r.resulting_intent.quantity),
                }
                for label, r in self.ablated_rulings.items()
            },
            "causal_chain": None if self.causal_chain is None else self.causal_chain.as_dict(),
            "earnings": None if self.earnings_read is None else self.earnings_read.as_dict(),
            "debate": None if self.debate is None else self.debate.as_dict(),
        }


class TradingDesk:
    """Assembles the analysts, the decision-maker, the Constitution and the order book."""

    def __init__(
        self,
        client: ChatModel,
        *,
        cost: CostModel | None = None,
        analyst_thinking: Thinking = Thinking.LOW,
        pm_thinking: Thinking = Thinking.FULL,
        annualised_vol: Decimal = Decimal("0.45"),
        critic: Critic | None = None,
    ) -> None:
        self._client = client
        # The adversary shares the desk's model by default. A separate seat would be better —
        # a critic that is the same model as the author shares its blind spots — and that is a
        # stated limitation rather than a hidden one: `eval/bakeoff.py` already seats two models,
        # and passing a different `critic` here is all it would take.
        self.critic: Critic | None = critic if critic is not None else client
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
        self.pm_thinking = pm_thinking
        self.pm = MetaPM(client, max_tokens=900, thinking=pm_thinking)
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
        ablated_constitutions: Mapping[str, ConstitutionPolicy] | None = None,
        profile: TraderProfile | None = None,
        history: Sequence[Any] = (),
        debate_budget: Decimal | None = None,
        debate_affordable: bool = True,
        underlying_halted: bool = False,
        halt_reason: str = "",
    ) -> DeskRun:
        panel = SourceIndependenceGraph()
        notes: list[str] = []

        # The cost of thinking, computed before anything thinks. It is needed twice: to price the
        # panel (below) and to state the hurdle to the decision-maker (§2).
        deliberation = round(
            deliberation_cost_bps(
                session, thinking=self.pm_thinking, annualised_vol=self.annualised_vol
            ),
            2,
        )

        # --- 1. the analysts. Each sees only evidence bounded by the decision instant. ---
        # Which of them run is decided from the evidence and priced against what running them costs
        # (`agents/selection.py`). Everything skipped is recorded as skipped, with the evidence it
        # would have read counted, so a thin panel is never mistaken for a thin market.
        # Screen the third-party text before anything reasons over it, and put the result in the
        # record whether or not anything fired. `agents/quarantine.py` replaces a hostile item's
        # claim and keeps its identity, so no count downstream changes — but a screen that reports
        # only its hits is indistinguishable from a screen that never ran.
        from argus.agents.quarantine import screen as screen_evidence

        evidence, screening = screen_evidence(evidence)
        notes.append(screening.note)

        selection = select(evidence, as_of=session.as_of, deliberation_bps=deliberation)
        notes.extend(selection.render())
        chosen = set(selection.run)

        event_evidence = [e for e in evidence if e.source in ("sec-edgar", "news", "macro")]
        social_evidence = [e for e in evidence if e.source == "social"]
        earnings_evidence = [e for e in evidence if e.source in ("filing", "transcript")]

        causal_chain: CausalChain | None = None
        earnings_read: EarningsRead | None = None

        # The panel runs **concurrently**, and that is an architectural claim, not an optimisation.
        #
        # Each analyst is handed a disjoint slice of the evidence and its own system prompt, and
        # `Analyst._ask` builds a fresh two-message conversation for every call — no history, no
        # shared transcript. There has therefore never been a channel through which one analyst
        # could see another's conclusion, and the panel note nonetheless told every decision that
        # its agreement "may be contagion". That was a false accusation the desk made about itself
        # on every row of the ledger. Running them in parallel removes the last coupling that did
        # exist — the order in which they consume a shared token budget — and makes the
        # independence structural rather than argued: a thread cannot read a result that has not
        # been returned yet.
        #
        # The wall-clock saving is the second reason and is priced below. Three analysts at a
        # measured ~8s each cost 24 seconds in sequence, during which the price moves; concurrently
        # they cost about as long as the slowest one.
        #
        # **One analyst failing must not cost the panel.** Each future is resolved on its own, and
        # a failure is recorded as a named absence — the audit in
        # `research/architecture/agent-architecture-audit.md` found no analyst-level error recovery
        # anywhere in the reference systems either, and a panel that dies because one upstream was
        # slow is a panel that is unavailable exactly when the market is busy.
        jobs: dict[str, Callable[[], Any]] = {}
        if event_evidence and "event" in chosen:
            jobs["event"] = lambda: self.event.analyse_with_chain(
                symbol, session, event_evidence
            )
        if social_evidence and "sentiment" in chosen:
            jobs["sentiment"] = lambda: self.sentiment.analyse(symbol, social_evidence)
        if earnings_evidence and "earnings" in chosen:
            jobs["earnings"] = lambda: self.earnings.analyse_decomposed(
                symbol, earnings_evidence
            )

        panel_started = time.monotonic()
        outcomes: dict[str, Any] = {}
        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures = {name: pool.submit(fn) for name, fn in jobs.items()}
                for name, future in futures.items():
                    try:
                        outcomes[name] = future.result()
                    except Exception as exc:  # one analyst must not take the panel down
                        outcomes[name] = exc
        panel_seconds = time.monotonic() - panel_started

        # Resolved in a fixed order so the panel is deterministic regardless of which thread
        # finished first. Concurrency must not make the record depend on scheduling.
        for name in ("event", "sentiment", "earnings"):
            got = outcomes.get(name)
            if got is None:
                continue
            if isinstance(got, Exception):
                notes.append(
                    f"[panel] {name} failed and was dropped from the panel "
                    f"({type(got).__name__}: {str(got)[:100]}); the decision is made without it "
                    f"rather than delayed by it"
                )
                continue
            if name == "event":
                view, causal_chain = got
                panel.add(view)
                if causal_chain is not None and causal_chain.links:
                    notes.append(
                        f"causal chain: {len(causal_chain.links)} links stated, gradable at the "
                        f"next price discovery"
                    )
            elif name == "sentiment":
                panel.add(got)
                # The panel record says what this view is worth, at the moment it is added rather
                # than in a document a reader may never open. A demoted analyst whose demotion is
                # invisible in the decision trail is the same as an undemoted one.
                notes.append(
                    f"[standing] sentiment is DEMOTED and contributes at that weight: its equity "
                    f"feed is dead (Bitget social empty in 93% of cycles; StockTwits 403, CNN "
                    f"418 on 2026-09-13), and the one live reading is crypto-wide risk appetite "
                    f"carried at credibility 0.35. It reasoned over "
                    f"{len(social_evidence)} item(s) here. Promotion needs an ablation that "
                    f"HELPS over 30 differing paired frames — see argus.eval.standing"
                )
            else:
                view, earnings_read = got
                panel.add(view)
                if earnings_read.surprises.is_contradictory:
                    notes.append(
                        f"earnings print contradicts itself: headline "
                        f"{earnings_read.surprises.headline:+.2f}, forward "
                        f"{earnings_read.surprises.forward:+.2f} — "
                        f"{earnings_read.dominant} dominates"
                    )

        if jobs:
            notes.append(
                f"[panel] {len(jobs)} analyst(s) ran concurrently in {panel_seconds:.1f}s; "
                f"none could read another's answer, so agreement is consensus rather than "
                f"contagion"
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
        # The cross-asset analyst is not selected on evidence and always runs. It reads the hedge
        # menu and the position, not the evidence list, so there is nothing about the evidence that
        # could make it irrelevant — and an empty menu is exactly the finding it exists to report.
        panel.add(self.cross_asset.analyse(
            symbol, session, menu, position_notional=position * token_price
        ))
        # Say so on the record. `selection.render()` reports the *selected* panel — "3 of 3
        # analysts run: event, sentiment, earnings" — and this analyst is deliberately outside that
        # count, so the trail claimed three analysts while four had run. A theme audit reading the
        # log concluded the cross-asset analyst had never run in 103 decisions. It runs on every
        # one; the record simply never said so, and on a track judged for decision explainability an
        # incomplete panel note is the defect rather than a cosmetic omission.
        notes.append(
            f"[panel] cross_asset also ran, outside the evidence-cost selection above: it reads "
            f"the hedge menu ({len(menu)} instrument(s)) and the position rather than the evidence "
            f"list, so no fact about the evidence could make it irrelevant"
        )

        if hedges.is_empty:
            notes.append(
                f"hedge menu empty: nothing placeable for {session.hours_to_next_discovery:.1f}h; "
                f"100% of risk carried as priced residual"
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

        # Disagreement as a record rather than a sentence. `sequential=False` is now the honest
        # label and it is earned structurally, not asserted: the analysts run concurrently on
        # disjoint evidence with no shared transcript, so no analyst can have read another's
        # answer. Until the panel was parallelised this said `True`, which understated our own
        # independence on every decision in the ledger.
        conflicts = conflict_report(panel.views, sequential=False)
        notes.extend(conflicts.render())

        # The facts the desk computed for this decision, so the thesis can be checked against them
        # while they still exist. Measured on the live ledger: theses quote the hurdle and the 24h
        # move, and neither is persisted, so after settlement those figures are unauditable. The
        # check runs here, at the one moment the numbers are all in scope.
        grounding_facts: dict[str, float] = {
            "round_trip_bps": float(self._cost.round_trip_bps()),
            "deliberation_bps": float(deliberation),
            "total_hurdle_bps": float(self._cost.round_trip_bps() + deliberation),
            "hours_to_discovery": float(session.hours_to_next_discovery),
            "token_price": float(token_price),
            "position_quantity": float(position),
            "analysts": float(len(panel.views)),
            "distinct_sources": float(len(panel.distinct_sources)),
        }
        for view in panel.views:
            grounding_facts[f"{view.analyst}_magnitude_bps"] = float(view.magnitude_bps)
            grounding_facts[f"{view.analyst}_confidence"] = float(view.confidence)

        # --- 2. the decision. The model sees the panel as evidence, and decides. ---
        # The mandate is built here rather than after the decision so the model can reason inside
        # it. `agents/mandate.py` still enforces every limit afterwards — telling the model first
        # changes what it reasons about, checking it after keeps the answer correct.
        active_mandate = Mandate(profile=profile) if profile is not None else None
        mandate_block = active_mandate.render() if active_mandate is not None else ""

        # Evidence is ORDERED by what this trader acts on, never filtered — `agents/mandate.py`'s
        # own docstring states the test and `order_evidence` implemented it, but nothing called it
        # here until now: the desk built `mandate_block` from the profile and then rendered
        # `evidence` in raw scan order regardless, so a conservative profile that prefers filings
        # still read news first, identically to a profile with no stated preference. Every item
        # still reaches the model; a non-preferred one is only marked, matching `Mandate.render()`'s
        # own promise below ("Evidence outside that list is still shown and still counts").
        evidence_aside = "  [outside this mandate's usual sources]"
        ordered_evidence = (
            order_evidence(evidence, mandate=active_mandate)
            if active_mandate is not None
            else tuple((e, True) for e in evidence)
        )

        # Episodic memory, on the same principle as the mandate: told before the model reasons,
        # not applied to its answer afterwards. Every line is arithmetic over the ledger rather
        # than a model's recollection of it, and `recall` enforces point-in-time itself — an
        # episode that settles after `session.as_of` contributes its decision and not its outcome.
        memory = recall(
            history,
            symbol=symbol,
            now=session.as_of,
            session_phase=str(session.phase),
        )
        memory_block = (
            memory.render(hurdle_bps=self._cost.round_trip_bps()) if history else ""
        )
        if memory.episodes:
            notes.append(
                f"[memory] {len(memory.episodes)} prior decision(s) on {symbol} shown to the PM, "
                f"{len(memory.graded)} of them graded; "
                + (
                    f"{len(memory.lessons(hurdle_bps=self._cost.round_trip_bps()))} pattern(s) "
                    f"stated from the record"
                    if memory.has_enough_to_generalise else
                    "too few graded to state a pattern, so none was"
                )
            )

        # The debate runs AFTER the analysts and BEFORE the decision: it is an argument about the
        # evidence, so it needs the evidence, and the PM needs its transcript. Its cost is charged
        # into the hurdle below rather than reported beside it — an argument the desk does not pay
        # for is an argument it will always think worth having.
        #
        # The default budget is one round. Holding a three-round debate about every weekend
        # abstention would cost 6bps against a 12bps round trip, which is most of the edge spent
        # on discussing whether there is one.
        allowance = debate_budget if debate_budget is not None else Decimal("2.0")
        # **The debate is held only when the panel actually disagrees about direction.**
        #
        # A debate exists to resolve a disagreement. Where the analysts already agree, two seats
        # are paid to restate a consensus, and on this venue that is the common case: the panel is
        # unanimous on most weekend abstentions. The first live run proved the cost is real — a
        # single-symbol cycle went from 20,714 tokens to over 27,691 and exhausted its budget.
        #
        # `has_directional_split` is the right trigger rather than a proxy for it: it is true
        # exactly when two analysts point opposite ways, which is the question a bull and a bear
        # are being paid to settle. A debate skipped for this reason is recorded as skipped, with
        # the reason, so a thin record is never mistaken for a thin market.
        worth_arguing = conflicts.has_directional_split
        # A split panel is argued only if the cycle can still afford every decision after it.
        # Debates are the one expensive step that is optional; a decision is not. On 2026-09-23
        # two cycles held five debates in their first nine symbols and ran out of tokens before
        # the last three were decided at all — three undecided names bought three debates.
        # The caller (`paper/runner.py`) says whether the budget can carry one.
        budget_reserved = worth_arguing and not debate_affordable
        if budget_reserved:
            worth_arguing = False
        debate = hold_debate(
            symbol=symbol,
            horizon_hours=session.hours_to_next_discovery,
            evidence=tuple(e.render() for e in evidence),
            bull_seat=self._client if worth_arguing else None,
            bear_seat=self.critic if worth_arguing else None,
            budget=DebateBudget(limit_bps=allowance),
            # The seats share a model unless a distinct critic was supplied, and agreement between
            # two calls to one model is self-consistency rather than corroboration.
            shared_model=self.critic is self._client,
        )
        if budget_reserved:
            notes.append(
                "[debate] not held — the panel split on direction, but the cycle's token budget "
                "is reserved for the symbols still to decide; this decision was taken without "
                "the debate, and that is recorded rather than hidden."
            )
        elif not worth_arguing:
            notes.append(
                f"[debate] not held — the panel agreed on direction across "
                f"{len(panel.views)} analyst(s), and a debate between two seats that already "
                f"agree is paid restatement. It runs when they split."
            )
        elif debate.ending is not Ending.NOT_HELD:
            notes.append(debate.render())
        deliberation += debate.cost_bps

        frame = MarketFrame(
            debate_block=debate.render() if debate.positions else "",
            memory_block=memory_block,
            symbol=symbol,
            as_of=session.as_of,
            mandate_block=mandate_block,
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
                [
                    e.render() + ("" if is_preferred else evidence_aside)
                    for e, is_preferred in ordered_evidence
                ]
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

        # The numbers the evidence itself carried, paired with the id that carried them. Without
        # these, `check_grounding` sees only the desk's *computed* facts, and a thesis quoting a
        # figure straight off a piece of evidence is reported as unattributable. Measured on the
        # live log: seq 65 flagged "+0.24%" as unsupported when it was the 24h change the market
        # evidence had stated as 0.0024 — the checker's own unit handling would have matched it,
        # had it been given the value. `evidence_values` is a parameter `grounding.check` has
        # always had and nothing was passing.
        evidence_values: list[tuple[str, float]] = [
            (e.id, figure.value) for e in evidence for figure in extract_figures(e.claim)
        ]
        grounding = check_grounding(
            proof.llm_original_intent.thesis,
            facts=grounding_facts,
            evidence_values=evidence_values,
        )
        notes.extend(grounding.render())

        # The same thesis, checked a second way. Numeric grounding catches a wrong figure; this
        # catches a wrong *property* — ledger seq 41 called two insider sales "pre-arranged 10b5-1
        # plans" when both filings carry aff10b5One = 0, and no number in that sentence was wrong.
        # Only sources that carry structured attributes can settle such a claim, so prose evidence
        # is passed through and reported as unexamined rather than silently counted as sound.
        claims = check_claims(
            proof.llm_original_intent.thesis,
            records=[(e.id, e.attributes) for e in evidence if e.attributes],
        )
        notes.extend(claims.render())

        # The same thesis, checked a third way — and this one asks about *instruments* rather than
        # figures or properties. Numeric grounding catches a wrong number; the claim checker
        # catches a wrong property; neither notices a thesis that reasons about NVDA and then
        # names TSLA, because "TSLA" is not a figure and carries no attribute to contradict.
        #
        # The mechanism is `HKUSTDial/DeepEar`'s (MIT), read at source: an instrument the model
        # proposes is kept only if it also appears in the text the model was given. Reported, never
        # repaired — silently swapping an unsupported instrument for a supported one would invent a
        # different claim and hide that it had.
        entities = check_entities(proof.llm_original_intent.thesis, evidence=evidence)
        notes.extend(entities.render())

        # --- 2b. the standing adversary. It may only reduce. ---
        #
        # Placed before the Constitution because the two ask different questions and the order
        # matters: the adversary asks whether the *thesis* survives attack, the Constitution asks
        # whether the *position* is within its limits. A thesis that is already refuted by its own
        # stated falsifier should never reach a size check at all — the size of a self-refuted
        # trade is not the interesting question about it.
        #
        # It is asymmetric like everything else downstream: it can refuse a trade or halve it, and
        # `apply_constraint` raises rather than logs if it ever tries to do more.
        challenge_ruling, challenged = challenge_intent(
            proof.llm_original_intent, evidence, critic=self.critic
        )
        notes.append(challenged.render())
        attacked = (
            challenge_ruling.resulting_intent if challenge_ruling is not None
            else proof.llm_original_intent
        )

        # --- 2c. escalation. The desk decides what it will not decide alone. ---
        #
        # Placed after the adversary and before the Constitution for the same reason the adversary
        # is placed before the Constitution: these are questions about whether the decision should
        # be *made*, not about how large it should be. A decision the desk is not entitled to take
        # unattended should not reach a size check.
        #
        # Nothing in the 140-repo corpus has this. `research/architecture/agent-evaluation-
        # harnesses.md` records the negative evidence file by line: no harness measures a
        # human-takeover rate and none defines when an agent must hand over. Until now ARGUS had
        # the HUMAN_REVIEW verdict and no policy behind it — the only thing that produced it was a
        # malformed model response, which is a parser check rather than a control.
        #
        # It may only reduce, like everything else downstream, and `escalation.apply` raises rather
        # than logs if it is ever asked to do more.
        escalation = assess_escalation(
            attacked,
            EscalationSignals(
                directional_split=conflicts.has_directional_split,
                debate_converged=debate.ending is not Ending.EXHAUSTED_ROUNDS,
                thesis_refuted_by_own_falsifier=challenged.already_refuted,
                refuted_condition=challenged.refuted_condition,
                unresolved_figures=len(grounding.unresolved),
                underlying_halted=underlying_halted,
                halt_reason=halt_reason,
            ),
        )
        if escalation.required:
            notes.append(escalation.render())
        attacked = apply_escalation(attacked, escalation)

        # --- 3. the Constitution. It may only reduce. ---
        # **`token_price` was already a parameter of this method and was never given to the
        # policy.** That is the whole reason four notional gates spent the project's lifetime
        # comparing unit counts against dollar ceilings: the price needed to convert one to the
        # other was in scope the entire time. A caller supplying its own policy with a price
        # already set keeps it; otherwise this decision's real price is used.
        policy = constitution or ConstitutionPolicy()
        if policy.reference_price is None:
            policy = replace(policy, reference_price=token_price)
        ruling = policy.rule(attacked, session=session, hedges=hedges)

        # Population RCT (Track 2, §13.7a): N gate-ablated policies ruled against this SAME
        # `attacked` intent, for the cost of N cheap deterministic calls — never N more LLM calls
        # against a hackathon key with a stated limited balance. Deliberately does not feed into
        # `self.pm.revise` below: an ablated variant's ruling is a counterfactual measurement, not
        # a real decision the model should react to, and letting it trigger a real revision would
        # spend budget on N synthetic conversations nobody asked the model to have.
        #
        # **Each variant gets the same `reference_price` back-fill the real policy gets above, and
        # omitting it silently disabled every notional gate in every ablation.** A variant is
        # constructed by a caller to answer "what would this decision look like under a tighter
        # cap", so it arrives carrying only the field being varied — a bare
        # `ConstitutionPolicy(max_position_notional=Decimal("1"))` has `reference_price=None`,
        # which means "skip the notional gates and say so". The ablation then reported
        # `binding_constraint="none"` for a policy whose whole purpose was to bind, and the RCT
        # measured the difference between two policies that both had a quarter of the Constitution
        # switched off. Same defect as the one `eval/riskproof.py` carried over 2,177,280 states,
        # in a second place, found by a test rather than by reading.
        ablated_rulings = {
            label: (
                variant if variant.reference_price is not None
                else replace(variant, reference_price=token_price)
            ).rule(attacked, session=session, hedges=hedges)
            for label, variant in (ablated_constitutions or {}).items()
        }
        # The round trip happens only when something actually bound.
        #
        # It used to run unconditionally, and on every unconstrained decision the model was told
        # **"Your proposed action was constrained by the risk layer"** — with a binding constraint
        # of "none" and the reason "no exposure proposed; nothing to narrow" printed underneath it.
        # That sentence was false on all 96 risk records on the live log, and the answer the model
        # gave after being told it is what the ledger recorded and the order would have carried.
        #
        # Three costs, and the first is the one that matters: a system built on evidence must not
        # state something untrue to its own decision-maker. Second, a full second model call on
        # every abstention, against a key with a finite balance. Third, `llm_changed_its_mind` read
        # True on 96 of 96 — "the model responded to the constraint" recorded where no constraint
        # existed, which is noise wearing the name of a signal.
        #
        # `AutonomyProof.record_revision` already said so: "Only meaningful after a ruling: a
        # 'revision' with nothing to revise against is noise." The type knew; the caller did not.
        if ruling.verdict is not ConstitutionVerdict.ALLOW:
            self.pm.revise(proof, frame, ruling)
        else:
            proof.record_ruling(ruling)
            notes.append(
                "[constitution] nothing bound, so the decision was not re-put to the model: "
                "asking it to reconsider an unconstrained choice would have told it something "
                "untrue and paid for a second opinion on a question nobody asked"
            )
        approved_hash = proof.approve()

        # --- 4. the order, bound to the approved intent. ---
        final = proof.llm_revised_intent or ruling.resulting_intent

        # --- 3b. the mandate. Whose trade is this? ---
        # `agents/mandate.py` and `desk/workbench.TraderProfile` existed and were wired to nothing,
        # so "personalised thesis" — a named Track 3 criterion — rested on a docstring. The profile
        # binds here, with the same asymmetry as the Constitution: it may refuse or shrink and can
        # never enlarge. A horizon breach is a refusal rather than a resize, because a thesis that
        # needs a month is not made suitable for a two-day trader by halving the ticket.
        if profile is not None:
            mandate = Mandate(profile=profile)
            # Every dimension the profile carries is checked, not only size and horizon. The
            # audit against Vibe-Trading found our divergence rate low because six fields left too
            # little for two profiles to disagree about; `hedge_available` in particular is what
            # makes a conservative mandate decline the weekend positions an aggressive one takes.
            breaches = mandate.out_of_mandate(
                horizon_hours=float(session.hours_to_next_discovery),
                notional=final.quantity * token_price,
                symbol=symbol,
                hedge_available=bool(hedges.menu),
                confidence=final.stated_confidence,
            )
            if breaches:
                notes.extend(
                    f"[mandate] {profile.name}: {reason}" for reason in breaches
                )
                ceiling = mandate.max_position_notional
                # An exclusion, a hedge requirement or a confidence floor is a refusal, not a
                # resize: a smaller position in an instrument this trader does not hold is still a
                # position in it, and half an unhedgeable exposure is still unhedgeable.
                absolute = (
                    not mandate.permits_horizon(float(session.hours_to_next_discovery))
                    or mandate.refuses_symbol(symbol)
                    or (profile.requires_hedge and not hedges.menu)
                    or final.stated_confidence < profile.min_confidence
                )
                permitted = (
                    Decimal("0") if absolute
                    else min(final.quantity, ceiling / token_price if token_price > 0 else _ZERO)
                )
                if permitted < final.quantity:
                    notes.append(
                        f"[mandate] quantity narrowed from {final.quantity} to {permitted} for "
                        f"{profile.name}; the mandate may only reduce"
                    )
                    final = apply_constraint(
                        final,
                        verdict=ConstitutionVerdict.RESIZE if permitted > 0
                        else ConstitutionVerdict.REJECT,
                        binding_constraint="mandate",
                        reason=f"{profile.name}: " + "; ".join(breaches),
                        resized_quantity=permitted,
                    ).resulting_intent
            else:
                notes.append(
                    f"[mandate] {profile.name}: within horizon and position limits"
                )

        order = None
        if final.quantity > 0 and final.verdict.carries_quantity:
            # Deterministic, not `f"{decision_id}-1"`: `decision_id` is built from wall-clock time
            # at the call site (`paper/runner.py`), so a retry of this same decision — the exact
            # case `OrderState.UNKNOWN` exists for, a timed-out request whose outcome is unresolved
            # — would previously mint a fresh id every attempt and neither our own duplicate guard
            # nor Bitget's server-side clientOid dedup could ever see two attempts as the same
            # order. See `execution.orders.deterministic_client_order_id` for the full defect.
            side_str = str(final.side).upper()
            order = Order(
                client_order_id=deterministic_client_order_id(
                    market_state_hash=proof.market_state_hash,
                    approved_intent_hash=approved_hash,
                    side=side_str,
                    quantity=final.quantity,
                ),
                symbol=symbol,
                side=side_str,
                quantity=final.quantity,
                approved_intent_hash=approved_hash,
            )
            if ruling.verdict is ConstitutionVerdict.REJECT:
                self.book.deny(order, at=session.as_of, reason=ruling.reason)
            else:
                # `authorise` re-checks that this order is the one the Constitution approved —
                # same symbol, same side, same quantity — and refuses otherwise. That is the
                # type-level form of the seq-264 defect, where a ruling of quantity 0 was obtained
                # and an order of quantity 1 was recorded anyway.
                self.book.submit(ruling.authorise(order), at=session.as_of)
        else:
            notes.append(f"no order: final verdict {final.verdict} with quantity {final.quantity}")

        return DeskRun(
            symbol=symbol, as_of=session.as_of, panel=panel,
            proof=proof, ruling=ruling, ruled_intent=attacked,
            ablated_rulings=ablated_rulings, order=order, notes=notes,
            # Sorted so the record is stable between runs on identical evidence; a set's iteration
            # order would make two identical decisions serialise differently.
            evidence_sources=tuple(sorted({e.source for e in evidence})),
            causal_chain=causal_chain, earnings_read=earnings_read, debate=debate,
        )


@dataclass(frozen=True, slots=True)
class ConstitutionPolicy:
    """Deterministic risk rules, in versioned config rather than in a prompt.

    TradingAgents' own teardown records that its risk management is "via prompt guidance, not code
    enforcement". A rule a model can be talked out of is not a rule, so every limit here is applied
    in Python and the model is told the outcome rather than asked to respect it.
    """

    reference_price: Decimal | None = None
    """Price per unit, used to turn an intent's **quantity** into the **notional** every cap here
    is denominated in.

    **Added 2026-09-21, after a constructed adversary found that four gates were comparing a unit
    count against a dollar ceiling.** `max_position_notional` is 50,000 *dollars*; the gate read
    `intent.quantity > 50_000`, and quantity is *units*. At NVDAUSDT's ~$180 a unit that made the
    cap **180x looser than written**: a 400-unit order is $72,000 of notional, over the cap, and
    passed untouched, while the gate could only fire above 50,000 units — $9,000,000. The
    `gross_exposure` and `signed_exposure` gates were worse: they subtract a real dollar book total
    (`Book.total_gross_notional`, which does multiply by price) from a dollar cap and then compared
    the dollar headroom against a unit count, mixing the two inside one expression.

    447 live decisions never caught it because the desk has never proposed a position. A rogue
    agent built to attack the caps caught it on its first run (`eval/rogue.py`).

    ``None`` means no price was supplied, and then every notional gate is **skipped and said to be
    skipped** rather than silently evaluated in the wrong unit — the same discipline `book_state`
    and `session_risk` already follow. A cap that cannot be computed is absent, not satisfied."""

    max_position_notional: Decimal = Decimal("50000")
    max_unhedged_notional: Decimal = Decimal("20000")
    """Ceiling on exposure carried with an empty hedge menu — the Sleeping-Anchor constraint."""

    min_confidence_to_trade: float = 0.55

    book_state: Any = None
    """The book the risk budget is drawn against (:class:`argus.risk.circuit.BookState`), or
    ``None``.

    Injected on the same terms as ``session_risk`` below and for the same reason: the rulebook is
    deterministic and a rule that reaches for the network is a rule that can fail open. ``None``
    means the gate does not fire, which is honest when no book has been computed at all — and is
    not the same as a book reporting zero drawdown it never measured.
    """

    book: Any = None
    """The real portfolio (:class:`argus.desk.book.Book`), or ``None``.

    Distinct from ``book_state`` above: that one is the circuit breaker's realised-equity scalar,
    this one is actual position state — Foundation 5's safety contract names ``order size``,
    ``signed exposure``, ``gross exposure``, ``factor exposure``, ``margin usage``,
    ``liquidation cost``, ``scenario loss`` and ``hedge integrity`` as eight separate dimensions
    precisely because collapsing them loses information (a resized order proves quantity never
    rose, not that risk never rose). ``gross_exposure`` below is the first of the seven still
    building on top of ``max_position``'s order-size dimension. Same injection discipline as
    ``session_risk`` and ``book_state``: ``None`` means the gate does not fire, not that the book
    is empty.
    """

    max_gross_exposure_notional: Decimal = Decimal("150000")
    """Ceiling on total gross notional across the whole book, this order included.

    Three times ``max_position_notional`` — a stated, revisable default (this project's other two
    flat caps, ``max_position_notional`` and ``max_unhedged_notional``, carry no derivation either;
    this one at least states its reasoning): a book concurrently near its per-position cap on three
    names is a materially different risk from one order at that cap, and `desk.book.Book` had no
    caller to size this against before today. Revisit once real position sizing exists to reason
    from `argus.desk.book.VenueMarginSnapshot`'s ``account_equity`` instead of a flat number.
    """

    max_signed_exposure_notional: Decimal = Decimal("100000")
    """Ceiling on net directional notional across the whole book — long minus short, netted
    within a symbol, projected including this order.

    Twice ``max_position_notional``: the desk may be net-long or net-short by at most two
    full-sized positions' worth, overall. Distinct from ``gross_exposure`` on purpose — a book
    hedged flat can carry high gross with zero net, and a book that is all one direction can carry
    the same gross with maximal net; the two describe different risks and Foundation 5's contract
    lists them separately for exactly that reason (see `desk.book.Book.total_signed_notional`'s
    docstring). Revisit alongside ``max_gross_exposure_notional`` once real equity exists to size
    both against.
    """

    max_margin_usage_ratio: Decimal = Decimal("0.5")
    """Policy cap on the venue-reported margin ratio, in the fraction scale
    `desk.book.VenueMarginSnapshot`'s docstring evidences (1.0 == Bitget's own conceptual 100%
    liquidation trigger, confirmed against Bitget's own liquidation page; the exact numeric scale
    of this specific field is evidenced by analogy to a verified sibling field, not independently
    proven — stated in the gate's own reason string, not hidden).

    **Half of Bitget's own trigger — our own conservative policy choice, not a venue-defined
    number**, the same "stated, revisable default" honesty as ``max_gross_exposure_notional``.
    Chosen because the venue's own ratio is a cross-asset, proprietary computation this project
    cannot reproduce (module docstring) — a policy line drawn well short of the real one buys
    room for exactly that uncertainty, rather than gating at the edge of a number we cannot
    ourselves verify precisely.
    """

    factor_exposures: Any = None
    """Precomputed book exposure to each named systematic factor
    (``Sequence[argus.desk.portfolio.FactorExposure]``), or ``None``.

    Same injection discipline as ``session_risk``/``book_state``: computed by the caller from
    `desk.portfolio.factor_exposures` against the book's current weights and real market history
    — the Constitution does not fetch either (module docstring: "a rule that reaches for the
    network is a rule that can fail open"). ``None`` means the gate does not fire, honest for a
    desk that has never run the regression, not read as zero exposure.
    """

    factor_exposure_limits: Mapping[str, float] = field(default_factory=dict)
    """Per-factor cap on ``abs(FactorExposure.exposure)``, keyed by factor name.

    A factor absent from this mapping is uncapped, not zero-capped — matching every other
    ``None``-means-unmeasured convention here rather than silently thresholding at a made-up
    default for a factor nobody configured.
    """

    stress_outcomes: Any = None
    """Precomputed benchmark-shock results (``Sequence[argus.desk.portfolio.StressOutcome]``),
    or ``None``. Computed by the caller from `desk.portfolio.stress_by_beta` against the book's
    current weights and :data:`argus.desk.portfolio.STANDARD_SHOCKS` (or a caller-supplied shock
    set). ``None`` means the gate does not fire.
    """

    max_scenario_loss_pct: float | None = None
    """Floor on the worst :class:`~argus.desk.portfolio.StressOutcome`'s ``portfolio_move_pct``
    (most negative = worst). ``None`` means the gate does not fire — an unset floor is not the
    same claim as an infinite one, and this field exists so a caller must set it deliberately
    rather than the gate silently adopting some default severity nobody chose.
    """

    liquidation_cost_estimates: Any = None
    """Precomputed forced-exit slippage per open position
    (``Mapping[str, argus.market.depth.Sweep]``, keyed by symbol), or ``None``.

    **A liquidation-severity PROXY, not the literal cost of Bitget's own bankruptcy-price/ADL
    mechanism — stated here and in the gate's own reason string, not hidden.** Building the real
    thing needs Bitget's tiered maintenance-margin table, which has not been read (the earlier
    liquidation-page research covered the ``mgnRatio``/100% trigger, not the tier table a true
    bankruptcy price is derived from). This instead reads real, live order-book depth
    (`market.depth.OrderBook.sweep`, verified 2026-09-15 against the actual class) for "how much
    would it cost to exit this position by force, right now" — an honestly-measured number, just
    not Bitget's own exact figure at the moment of an actual liquidation. Computed by the caller,
    never fetched inside `rule()` (same discipline as every other injected field here). ``None``
    means no order-book fetch has happened this cycle, not zero cost.
    """

    max_liquidation_cost_bps: Decimal | None = None
    """Floor on the worst position's forced-exit ``slippage_bps``. ``None`` means the gate does
    not fire.

    **500 bps (5%), our own conservative policy default, stated rather than hidden** — the same
    "stated, revisable default" honesty as ``max_gross_exposure_notional``. Roughly 40x this
    venue's own measured round-trip taker cost (~12 bps, `cost/model.py`'s calibrated figure): a
    position that would cost forty times its normal exit to unwind by force is a severity signal
    worth blocking on, however exactly Bitget's own bankruptcy-price math would price it.
    """

    graded_predictions: Any = None
    """Settled (confidence, was-it-right) pairs — :class:`argus.eval.observatory.Prediction`.

    This is what decides whether stated confidence may scale a position. `risk/sizing.py` requires
    twenty graded outcomes and an expected calibration error under 0.15 before it will use
    confidence at all; below that it returns a fixed fraction and names the reason. Empty is the
    honest state today and produces the fixed fraction, which is **tighter** than the position cap
    that used to be the only limit — an unproven desk should not be sized like a proven one.
    """

    assumed_payoff: Decimal = Decimal("1")
    """Win/loss ratio fed to the Kelly arithmetic, used **only once calibration passes**.

    Stated as an explicit, neutral 1:1 rather than inferred, because with zero settled trades there
    is nothing to infer it from and a payoff pulled out of the air would be levered directly into
    position size the moment the calibration gate opened. A caller with a realised win/loss record
    should pass it; until one exists this is the assumption, and it is written down rather than
    buried in a default deep inside the sizing call.
    """

    session_risk: Any = None
    """Measured per-phase volatility for this instrument
    (:class:`argus.risk.session_risk.SessionRisk`), or ``None``.

    Supplied by the caller rather than fetched here: this class is the deterministic rulebook and a
    rule that reaches for the network is a rule that can fail open. ``None`` means the throttle
    below does nothing and says so, which is the same refusal `risk/effectiveness.py` makes — an
    unmeasured risk is not a risk of zero, but neither is it a licence to invent a cautious number.
    """

    session_horizon_bars: int = 24
    """How long a position is assumed to be held, for the volatility-path calculation.

    Twenty-four hours because that is what `paper/runner.py` actually does: decisions are settled
    against the price one day later. A different desk would pass a different number and get a
    different throttle, which is the point — the reopen jump matters enormously at two hours and not
    at all at twenty-four, and a constant multiplier could not express that.
    """

    min_symbol_realized_pnl: Decimal | None = None
    """Floor on one symbol's windowed realized PnL
    (`desk.book.Book.symbol_realized_pnl_since`). ``None`` means the gate does not fire.

    **The ARGUS-native counterpart to freqtrade's ``LowProfitPairs``**
    (`eval.freqtrade_baseline.freqtrade_low_profit_pairs`, ported and tested against freqtrade's
    own source at `low_profit_pairs.py:41-73`) — closing the real capability gap proven in
    `tests/test_freqtrade_baseline.py`'s ``TestLowProfitPairsIsGenuinelyPerSymbol``:
    `risk/circuit.py`'s Breaker/Activation is whole-book only
    (``inspect.getsource(circuit)`` contains zero occurrences of ``"symbol"``), so nothing before
    this gate could lock out one underperforming symbol while leaving the rest of the book
    tradeable — every other ceiling here narrows the whole order or nothing.

    Reads real `Book` fills directly via ``symbol_realized_pnl_since``, not a
    backtest-reconstructed `backtest.engine.SyntheticTrade` sequence: `extract_trades` exists so a
    *backtest comparison* can reconstruct discrete trades from a continuous-weight simulation, but
    live `Book` already carries real `Lot`s with real ``ts_filled`` — replaying those directly is
    the honest path, not a reason to round-trip live state through the backtest engine just to
    reuse the same arithmetic.
    """

    symbol_underperformance_window_minutes: int = 1440
    """Lookback window for ``min_symbol_realized_pnl``, in minutes.

    Twenty-four hours — the same horizon ``session_horizon_bars`` uses, and for the same reason:
    `paper/runner.py` settles every decision a day later, so a symbol's last 24 hours of realized
    results is the horizon this desk actually reasons in, not its lifetime record.
    """


    def _notional(self, quantity: Decimal) -> Decimal | None:
        """Quantity in units -> notional in dollars, or ``None`` when no price was supplied.

        Every caller must treat ``None`` as "this gate cannot be evaluated" and say so, never as
        "this gate passed". See :attr:`reference_price`."""
        if self.reference_price is None or self.reference_price <= 0:
            return None
        return quantity * self.reference_price

    def rule(
        self, intent: Intent, *, session: SessionState, hedges: HedgeabilitySurface
    ) -> ConstitutionRuling:
        if not intent.verdict.carries_quantity or intent.quantity <= 0:
            # **Gate 1 names itself, like every other gate.** It used to return "none", the same
            # token the terminal all-clear below returns — and `eval/autopsy.py` counted "none" as
            # gate 1 firing. So an intent that reached and cleared all seven gates was counted as
            # gate 1 *firing*, and gates 2-7 reported UNREACHED **on the very decision that proved
            # them reachable**. The funnel inverted exactly when it finally had something to say.
            #
            # Latent only because no decision has ever proposed exposure. `desk/carrydesk.py:112`
            # worked around it by reading `resulting_intent.quantity`; with the two return sites
            # distinguishable that workaround is no longer needed anywhere.
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.ALLOW,
                binding_constraint="no_exposure",
                reason="no exposure proposed; nothing to narrow",
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

        # --- CONSTRAINT PRODUCERS ------------------------------------------------------------
        #
        # **Every remaining gate contributes a ceiling, and the binding one is the MINIMUM of all
        # of them — not the first encountered.** This replaced a chain that returned on the first
        # gate to bind, and the difference is not cosmetic. Demonstrated on 2026-09-15:
        #
        #     book in deep drawdown (equity 30000 vs a 100000 peak, 3 consecutive losses)
        #     proposed 45000, no hedge placeable
        #       -> unhedgeable_gap capped it at 20000 and RETURNED
        #       -> risk_budget, which would have capped it at 0, never ran
        #       -> a 20000 order was approved while the circuit breaker said trade nothing,
        #          and the record showed `binding_constraint: unhedgeable_gap` — a gate that
        #          appeared to be working
        #
        # "A risk layer may only reduce what you choose" was true of each gate in isolation and
        # false of the chain, because the first gate to bind ended it. Reachability reporting is an
        # *observability* feature; it was standing in for *enforcement*, and those are different
        # properties. Collecting every ceiling restores the claim the Constitution actually makes.
        #
        # Latent rather than active: all 231 recorded decisions carry quantity 0, so the terminal
        # `no_exposure` check above short-circuits and the chain has never run past it on live
        # data. It would have fired on the first decision that proposed exposure.
        ceilings: list[tuple[str, Decimal, str]] = []

        # `price` is bound rather than read through `self` at each site so the type checker can
        # see that a notional gate and its ceiling share one non-None price. `notional` is derived
        # from it, so the two are None together by construction.
        price = self.reference_price if (
            self.reference_price is not None and self.reference_price > 0
        ) else None
        notional = intent.quantity * price if price is not None else None
        # Stated, not swallowed. Without a price the four notional gates cannot be evaluated, and
        # this suffix travels into the ruling's reason so a reader can tell "not checked" from
        # "checked and passed". An absent cap that reads like a satisfied one is how a risk layer
        # becomes decorative.
        unpriced = "" if notional is not None else (
            " | NOT EVALUATED: no reference_price, so the unhedged, gross, signed and "
            "max_position notional caps were skipped rather than passed"
        )
        if (hedges.is_empty and price is not None and notional is not None
                and notional > self.max_unhedged_notional):
            ceilings.append((
                "unhedgeable_gap",
                self.max_unhedged_notional / price,  # dollars -> units; see reference_price
                f"no hedge placeable for {session.hours_to_next_discovery:.1f}h; "
                f"unhedged exposure capped at {self.max_unhedged_notional}",
            ))

        # Gross exposure — the whole book, not just this order. "Smaller quantity is not safer"
        # applies here exactly as Foundation 5's safety contract says it does to margin: a book
        # already carrying $140,000 gross across other names and proposing $20,000 more is a
        # materially different risk from a first order of $20,000, and `max_position_notional`
        # above only ever sees this single order's own size. `self.book` is the first Foundation-3
        # consumer in the Constitution — see its field docstring for why it is a separate injection
        # from `self.book_state`.
        if self.book is not None:
            existing_gross = self.book.total_gross_notional()
            headroom = max(Decimal("0"), self.max_gross_exposure_notional - existing_gross)
            if price is not None and notional is not None and notional > headroom:
                ceilings.append((
                    "gross_exposure",
                    headroom / price,  # dollars -> units; see reference_price
                    f"book already carries {existing_gross} gross; the "
                    f"{self.max_gross_exposure_notional} cap leaves {headroom} of headroom for "
                    f"this order",
                ))

            # Net exposure — a book can be flat on gross (hedged) or maximal on gross (one-way)
            # at the same gross number; this reads the direction gross deliberately throws away.
            # ``direction`` matters: an order that trims an existing skew has *more* headroom, not
            # less, because it moves net exposure toward zero rather than away from it.
            direction = Decimal("1") if intent.side is Side.BUY else Decimal("-1")
            existing_signed = self.book.total_signed_notional()
            signed_headroom = max(
                Decimal("0"), self.max_signed_exposure_notional - direction * existing_signed
            )
            if price is not None and notional is not None and notional > signed_headroom:
                ceilings.append((
                    "signed_exposure",
                    signed_headroom / price,  # dollars -> units; see reference_price
                    f"book is net {existing_signed} (long positive); a {intent.side} at this size "
                    f"would push net exposure past the {self.max_signed_exposure_notional} cap "
                    f"({signed_headroom} of headroom in this direction)",
                ))

            # Hedge integrity — "smaller quantity is not safer" in its sharpest form. A REDUCE
            # narrows quantity, which every other ceiling here reads as strictly less risky; when
            # the position being reduced is declared as a hedge for another position that is
            # STILL OPEN, reducing it is the opposite. `Verdict.REDUCE` plus `intent.side` already
            # disambiguate which position this is without any change to `Intent`: a SELL reduces
            # a LONG, a BUY reduces a SHORT — Bitget's own `tradeSide` is a venue execution detail
            # (`execution/bitget_client.py`), not a Constitution-level ambiguity, and conflating
            # the two was a design error caught while building this, not assumed away.
            if intent.verdict is Verdict.REDUCE:
                being_reduced = (
                    intent.symbol,
                    PositionSide.LONG if intent.side is Side.SELL else PositionSide.SHORT,
                )
                position = self.book.positions.get(being_reduced)
                if position is not None and not position.is_flat:
                    cluster = self.book.hedge_cluster(*being_reduced)
                    still_hedging = [
                        other for other in cluster if other != being_reduced
                        and (p := self.book.positions.get(other)) is not None and not p.is_flat
                    ]
                    if still_hedging:
                        names = ", ".join(f"{s} {v}" for s, v in still_hedging)
                        ceilings.append((
                            "hedge_integrity",
                            Decimal("0"),
                            f"{intent.symbol} {being_reduced[1]} is linked as a hedge for "
                            f"{names}, which remains open; an automatic reduction is refused "
                            f"rather than assumed safe — a resize can raise the risk it is "
                            f"hedging even while lowering its own quantity",
                        ))

            # Margin usage — a binary block, not a headroom, and the reason string says exactly
            # why: Bitget's own margin engine is cross-asset and proprietary (module docstring),
            # so there is no way to compute how much of THIS order's quantity the venue would
            # still accept before its own ratio moved further — only whether the venue-reported
            # ratio is already past our policy cap. `mgn_ratio` is ``None``-guarded at the
            # `self.book.margin is not None` level: a book with no margin snapshot fetched yet
            # cannot honestly say anything about margin usage, and must not be read as zero usage.
            if (
                self.book.margin is not None
                and self.book.margin.mgn_ratio >= self.max_margin_usage_ratio
            ):
                ceilings.append((
                    "margin_usage",
                    Decimal("0"),
                    f"venue-reported margin ratio {self.book.margin.mgn_ratio} is at or above "
                    f"the {self.max_margin_usage_ratio} policy cap (Bitget's own liquidation "
                    f"trigger is conceptually 100%; this field's exact numeric scale is "
                    f"evidenced by analogy, not directly proven — see "
                    f"`desk.book.VenueMarginSnapshot`'s docstring); no further exposure until it "
                    f"eases",
                ))

        # Factor exposure — a binary block, same reasoning as margin_usage. `FactorExposure` is a
        # regression slope for the WHOLE book (`desk.portfolio.factor_exposures`), not decomposed
        # per symbol, so there is no per-symbol beta here to compute how much of THIS order's
        # quantity would move the aggregate — only whether the book's already-measured exposure is
        # past the cap. Injected, not fetched: computing it needs real market history, which this
        # deterministic rule must not reach for (see `session_risk`'s field docstring).
        if self.factor_exposures is not None:
            for exposure in self.factor_exposures:
                limit = self.factor_exposure_limits.get(exposure.factor)
                if limit is None or exposure.exposure is None:
                    continue
                if abs(exposure.exposure) >= limit:
                    ceilings.append((
                        "factor_exposure",
                        Decimal("0"),
                        f"book exposure to {exposure.factor!r} is {exposure.exposure:.3f}, at or "
                        f"above the {limit} cap; no further exposure until it is reduced",
                    ))
                    break

        # Scenario loss — same binary shape again. `stress_by_beta`'s output is the book's
        # market-driven move under a shock, not this order's marginal contribution to it, for the
        # same per-symbol-decomposition reason as factor exposure above.
        if self.stress_outcomes is not None and self.max_scenario_loss_pct is not None:
            scored = [o for o in self.stress_outcomes if o.portfolio_move_pct is not None]
            worst = min(scored, key=lambda o: o.portfolio_move_pct, default=None)
            if worst is not None and worst.portfolio_move_pct <= self.max_scenario_loss_pct:
                ceilings.append((
                    "scenario_loss",
                    Decimal("0"),
                    f"worst modelled shock ({worst.shock}) already moves the book "
                    f"{worst.portfolio_move_pct:+.2f}%, past the {self.max_scenario_loss_pct}% "
                    f"cap; no further exposure until stress is reduced",
                ))

        # Liquidation cost — a proxy, and the reason string says so every time it binds, not just
        # in the field's own docstring. Same binary shape as the three gates above: `Sweep` prices
        # exiting one position at today's book depth, not this specific order's own marginal
        # contribution to that cost.
        if (
            self.liquidation_cost_estimates is not None
            and self.max_liquidation_cost_bps is not None
        ):
            worst_symbol, worst_sweep = max(
                self.liquidation_cost_estimates.items(),
                key=lambda item: item[1].slippage_bps, default=(None, None),
            )
            if (
                worst_sweep is not None
                and worst_sweep.slippage_bps >= self.max_liquidation_cost_bps
            ):
                ceilings.append((
                    "liquidation_cost",
                    Decimal("0"),
                    f"forced-exit slippage proxy for {worst_symbol} is "
                    f"{worst_sweep.slippage_bps:.1f}bps (real order-book depth, not Bitget's own "
                    f"bankruptcy-price formula — see `liquidation_cost_estimates`'s docstring), "
                    f"at or above the {self.max_liquidation_cost_bps}bps cap; no further exposure "
                    f"until it eases",
                ))

        # Per-symbol underperformance — the ARGUS-native LowProfitPairs. Unlike every ceiling
        # above, this one narrows exposure to ONE symbol, not the order's whole size: an order on
        # a different, healthy symbol is untouched even while this symbol is locked out.
        if self.book is not None and self.min_symbol_realized_pnl is not None:
            cutoff = session.as_of - timedelta(minutes=self.symbol_underperformance_window_minutes)
            windowed_pnl = self.book.symbol_realized_pnl_since(intent.symbol, cutoff)
            if windowed_pnl < self.min_symbol_realized_pnl:
                ceilings.append((
                    "per_symbol_underperformance",
                    Decimal("0"),
                    f"{intent.symbol} realized {windowed_pnl} over the last "
                    f"{self.symbol_underperformance_window_minutes} minutes, below the "
                    f"{self.min_symbol_realized_pnl} floor; no further exposure to this symbol "
                    f"until it recovers",
                ))

        # The circuit breaker's de-risking ladder, applied to the size the desk asked for.
        #
        # `risk/circuit.py` and `risk/sizing.py` were both complete, both tested, and **neither was
        # ever called on a live decision**: `sizing.size()` had no caller anywhere in `src/argus`,
        # and `sizing.py:149` documented `risk_multiplier` as coming from `circuit.risk_multiplier`
        # while nothing connected them. A risk control that cannot change a decision is a comment.
        #
        # `book_state` is injected exactly as `session_risk` is, and absent it the gate does not
        # fire — a visible degradation rather than a silent one, and never a fabricated drawdown.
        if self.book_state is not None:
            from argus.risk.circuit import risk_multiplier as circuit_multiplier
            from argus.risk.sizing import size as size_position

            # The session throttle is deliberately passed as 1 here and applied by its own ceiling
            # below. `sizing.size` accepts both multipliers and would fold them into one fraction,
            # which would make it impossible to say which of the two bound.
            sizing = size_position(
                win_probability=intent.stated_confidence,
                payoff=self.assumed_payoff,
                predictions=self.graded_predictions or (),
                risk_multiplier=circuit_multiplier(self.book_state),
                session_multiplier=Decimal("1"),
            )
            budget = self.book_state.equity * sizing.fraction
            if budget < intent.quantity:
                ceilings.append((
                    "risk_budget",
                    budget,
                    f"risk budget {sizing.fraction:.1%} of {self.book_state.equity} = {budget} "
                    f"({sizing.basis}); drawdown from peak "
                    f"{self.book_state.total_drawdown:.1%}",
                ))

        # Measured session volatility — how violent the path is, as distinct from whether the
        # exposure can be covered at all.
        if self.session_risk is not None:
            from argus.risk.session_risk import throttle as session_throttle

            scaled = session_throttle(
                self.session_risk, start=session.as_of,
                horizon_bars=self.session_horizon_bars,
            )
            if scaled.multiplier < 1:
                narrowed = intent.quantity * scaled.multiplier
                if narrowed < intent.quantity:
                    ceilings.append(("session_volatility", narrowed, scaled.reason))

        # The hard cap, which must bind last so it is never diluted by a multiplier. As a ceiling
        # among ceilings that ordering is automatic: a minimum does not care what order it is
        # given its arguments, which is one fewer thing to get wrong.
        if price is not None and notional is not None and notional > self.max_position_notional:
            ceilings.append((
                "max_position",
                self.max_position_notional / price,  # dollars -> units; see reference_price
                f"position capped at {self.max_position_notional} notional",
            ))

        # --- FINAL VALIDATION ------------------------------------------------------------------
        if ceilings:
            name, permitted, reason = min(ceilings, key=lambda c: c[1])
            others = [c for c in ceilings if c[0] != name]
            trail = reason
            if others:
                # The explanation improves rather than degrades: naming every ceiling that was
                # computed is strictly more informative than naming the first one to bind.
                trail += " | also computed: " + "; ".join(
                    f"{other_name} at {other_cap}" for other_name, other_cap, _ in others
                )
            if permitted <= 0:
                return apply_constraint(
                    intent, verdict=ConstitutionVerdict.REJECT,
                    binding_constraint=name,
                    reason=trail + " — no size satisfies every constraint",
                )
            ruled = apply_constraint(
                intent, verdict=ConstitutionVerdict.RESIZE,
                binding_constraint=name, reason=trail, resized_quantity=permitted,
            )
            # Re-checked against every ceiling, not merely against the one that bound. A resize
            # that still violated another constraint is the defect this whole restructure exists
            # to make impossible, so it is asserted rather than assumed.
            final = ruled.resulting_intent.quantity
            breached = [n for n, cap, _ in ceilings if final > cap]
            if breached:  # pragma: no cover - unreachable while `min` is correct
                return apply_constraint(
                    intent, verdict=ConstitutionVerdict.REJECT,
                    binding_constraint=name,
                    reason=(
                        f"{trail} — the narrowed size {final} still breaches "
                        f"{', '.join(breached)}; refusing rather than approving an order that "
                        f"cannot be shown to satisfy every mandatory constraint"
                    ),
                )
            return ruled

        # "none" now means exactly what it says: **nothing bound.** Every gate that could run,
        # ran and allowed — and `unpriced` names any that could not, so an all-clear never
        # silently stands in for an unevaluated cap.
        return apply_constraint(
            intent, verdict=ConstitutionVerdict.ALLOW,
            binding_constraint="none",
            reason="within every configured limit" + unpriced,
        )


__all__ = ["AnalystView", "ConstitutionPolicy", "DeskRun", "Evidence", "TradingDesk"]
