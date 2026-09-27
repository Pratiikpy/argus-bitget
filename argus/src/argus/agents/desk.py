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

import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
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
from argus.agents.circuit import (
    DEFAULT_THRESHOLDS,
    PM_POLICY,
    CircuitRecord,
    RetryPolicy,
    Step,
    Thresholds,
    call_with_retry,
    nudge_for,
)
from argus.agents.circuit import assess as assess_trajectory
from argus.agents.claims import check as check_claims
from argus.agents.conflict import report as conflict_report
from argus.agents.debate import Debate, DebateBudget, Ending
from argus.agents.debate import hold as hold_debate
from argus.agents.earnings import EarningsRead
from argus.agents.entitygate import check as check_entities
from argus.agents.meta_pm import MarketFrame, MetaPM, deliberation_cost_bps
from argus.agents.recall import recall
from argus.agents.selection import select
from argus.cost.model import CostModel
from argus.decision.escalation import Escalation
from argus.decision.escalation import Signals as EscalationSignals
from argus.decision.escalation import apply as apply_escalation
from argus.decision.escalation import assess as assess_escalation
from argus.decision.pause import (
    DEFAULT_ANSWER_WINDOW,
    AlreadyResolved,
    HumanResponse,
    PauseError,
    PauseExpired,
    PauseRequest,
    PauseStore,
    check_response,
    resolve_intent,
)
from argus.decision.verdicts import (
    ConstitutionRuling,
    ConstitutionVerdict,
    Intent,
    Side,
    Verdict,
    apply_constraint,
)
from argus.desk.feed_sanity import market_move, screen_views
from argus.desk.feed_sanity import screen_evidence as sanity_screen
from argus.desk.mandate import Mandate, order_evidence
from argus.desk.workbench import TraderProfile
from argus.execution.orders import Order, OrderBook, deterministic_client_order_id
from argus.llm.base import ChatModel
from argus.llm.qwen import BudgetExhausted, Thinking
from argus.proof.autonomy import AutonomyProof, hash_intent
from argus.risk.constitution import ConstitutionPolicy
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionState
from argus.truth.evidence import Evidence
from argus.truth.grounding import check as check_grounding
from argus.truth.grounding import extract as extract_figures

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
    the working log, which is not published) has something real to read once it exists, without
    another round of the "the mechanism exists but nothing calls it" gap this project keeps finding
    and closing.
    """

    order: Order | None = None
    notes: list[str] = field(default_factory=list)
    evidence: tuple[str, ...] = ()
    """The evidence lines the decision-maker was shown, exactly as its frame carried them (id,
    source, credibility, when it became available, the claim). The ledger commits to them only
    through ``market_state_hash``; kept here so the runner can write them beside the entry and a
    reader can see what the decision rested on (readiness backlog L43)."""
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

    pause: PauseRequest | None = None
    """Set when this decision was held for a human and its continuation written to disk.

    A held run is still a complete record — the Constitution ruled on the held intent and no order
    was placed — so a paused decision and a finished one read the same way to everything
    downstream. What the pause adds is the way back: `decision/pause.py`."""

    human: HumanResponse | None = None
    """Set on the run a human's answer resumed. The owner of the takeover, and what they said."""

    continuation: DeskContinuation | None = field(default=None, repr=False)
    """The paused-point state, kept in memory beside the one on disk.

    Lets a caller in the same process — the console answering while the desk is still up — resume
    with :meth:`TradingDesk.resume_with` without a disk round trip. Not serialised: the durable copy
    is the one in the pause store, and a record that carried both would carry the state twice."""

    deliberation: dict[str, Any] | None = None
    """`MetaPM.last_deliberation.as_dict()` for this decision — every attempt, which one was
    committed and how many model calls it took — persisted beside the proof it produced.

    Set by :meth:`TradingDesk.run` rather than carried on :class:`DeskContinuation` on purpose:
    adding a field there changes :func:`resume_signature` and would refuse every continuation
    already paused on disk. A decision resumed from disk therefore has ``None`` here; its notes
    still carry the rendered deliberation, which is written before the pause point."""

    feed_sanity: dict[str, Any] | None = None
    """What the deterministic feed-sanity gate (`desk/feed_sanity.py`) checked and withheld, for the
    evidence and for the analyst panel, before the Meta-PM read either. Same lifetime as above."""

    circuit: dict[str, Any] | None = None
    """The trajectory breaker's record (`agents/circuit.py` :class:`CircuitRecord`): trips that
    changed this decision, nudges the model was shown, and every failed call attempt."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat(),
            "panel": self.panel.as_dict(),
            "decision": self.proof.to_record(),
            "deliberation": self.deliberation,
            "feed_sanity": self.feed_sanity,
            "circuit": self.circuit,
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
            "human_loop": self._human_loop(),
        }

    def _human_loop(self) -> dict[str, Any] | None:
        if self.pause is None:
            return None
        return {
            "state": "paused" if self.human is None else "resumed",
            "request_id": self.pause.request_id,
            "state_hash": self.pause.state_hash,
            "triggers": [t.value for t in self.pause.escalation.triggers],
            "expires_at": self.pause.expires_at.isoformat(),
            "response": None if self.human is None else self.human.as_dict(),
        }


RESUME_PIPELINE: tuple[str, ...] = (
    "constitution", "ablated_rulings", "revise_if_bound", "approve", "mandate", "order",
)
"""The stages :meth:`TradingDesk._conclude` runs, in order — what a resume executes.

Part of :func:`resume_signature`. A change to this tuple, or to the fields of
:class:`DeskContinuation`, changes the signature, and every continuation written before the change
is refused on resume rather than replayed through a pipeline it never saw (MAF
`_workflows/_runner.py:316-320`)."""


def resume_signature() -> str:
    """The fingerprint of the resumable half of the desk. MAF `_workflow.py:372-375,1296-1361`."""
    shape = {
        "stages": list(RESUME_PIPELINE),
        "continuation": [f.name for f in fields(DeskContinuation)],
    }
    return hashlib.sha256(
        json.dumps(shape, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class DeskContinuation:
    """Everything the desk holds at the escalation point that the rest of the decision needs.

    Written to disk when the desk pauses (`decision/pause.py`) and restored on resume. The split is
    at stage 2c because that is where the question changes: before it, the desk works out *what* it
    would do (analysts, debate, the model's decision, the adversary); after it, the Constitution,
    the model's revision, the mandate and the order decide *how much of it* reaches the venue. A
    human answers the first question's result, so a resume runs only the second half — the model's
    first half is not re-run, which is what makes "continues from the paused point" true rather
    than "starts again and hopes the model says the same thing".

    The model client is deliberately absent. It is the resuming desk's, not the paused one's: a
    continuation carries state, never a live connection or a key.
    """

    decision_id: str
    symbol: str
    session: SessionState
    token_price: Decimal
    hedges: HedgeabilitySurface
    constitution: ConstitutionPolicy | None
    ablated_constitutions: dict[str, ConstitutionPolicy]
    profile: TraderProfile | None
    proof: AutonomyProof
    frame: MarketFrame
    panel: SourceIndependenceGraph
    notes: list[str]
    evidence_sources: tuple[str, ...]
    causal_chain: CausalChain | None
    earnings_read: EarningsRead | None
    debate: Debate
    proposed: Intent
    """The intent after the adversary and before escalation — what a human is asked about."""

    escalation: Escalation

    def signature_state(self) -> dict[str, Any]:
        """The canonical view the state hash is taken over. Every field, no reprs.

        ``proof`` goes through :func:`dataclasses.asdict` rather than `AutonomyProof.to_record`,
        because `to_record` asserts the Constitution has ruled and at the pause point it has not.
        """
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "session": asdict(self.session),
            "token_price": self.token_price,
            "hedges": asdict(self.hedges),
            "constitution": None if self.constitution is None else asdict(self.constitution),
            "ablated_constitutions": {
                k: asdict(v) for k, v in sorted(self.ablated_constitutions.items())
            },
            "profile": None if self.profile is None else asdict(self.profile),
            "proof": asdict(self.proof),
            "frame": asdict(self.frame),
            "panel": self.panel.as_dict(),
            "notes": list(self.notes),
            "evidence_sources": list(self.evidence_sources),
            "causal_chain": None if self.causal_chain is None else self.causal_chain.as_dict(),
            "earnings_read": (
                None if self.earnings_read is None else self.earnings_read.as_dict()
            ),
            "debate": self.debate.as_dict(),
            "proposed": asdict(self.proposed),
            "proposed_hash": hash_intent(self.proposed),
            "escalation": self.escalation.as_dict(),
        }


def _no_decision(frame: MarketFrame, *, decision_id: str, reason: str) -> AutonomyProof:
    """The safe no-op the desk records when the Meta-PM call gave up (`agents/circuit.py`).

    The intent is NO_TRADE with zero size, and its thesis says in its first words that no model
    decided — ``[circuit] no model decision`` is one of :data:`circuit.INVALID_MARKERS`, so the
    breaker counts it as a not-a-decision on the next pass and a reader of the ledger can never
    mistake it for an abstention the model chose.
    """
    return AutonomyProof(
        decision_id=decision_id,
        as_of=frame.as_of,
        market_state_hash=frame.state_hash(),
        llm_original_intent=Intent(
            symbol=frame.symbol, side=Side.BUY, quantity=_ZERO, verdict=Verdict.NO_TRADE,
            stated_confidence=0.0,
            thesis=f"[circuit] no model decision — {reason}; degraded to a safe no-op",
        ),
        llm_original_reasoning=reason,
    )


def _stopped(intent: Intent) -> Intent:
    """An exposure-opening intent reduced to a no-op by a breaker trip. Only ever reduces."""
    stopped = replace(intent, verdict=Verdict.NO_TRADE, quantity=_ZERO)
    if stopped.quantity != _ZERO or stopped.verdict.opens_exposure:  # the asymmetry, checked
        raise AssertionError("a circuit-breaker trip produced exposure")
    return stopped


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
        pm_policy: RetryPolicy = PM_POLICY,
        circuit_thresholds: Thresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self._client = client
        # The trajectory breaker (`agents/circuit.py`): how the Meta-PM call is guarded, and when a
        # run of bad answers or repeated orders stops a decision. Both default to the measured
        # settings; a caller passes its own only to test the breaker, never to disarm it.
        self.pm_policy = pm_policy
        self.circuit_thresholds = circuit_thresholds
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
        pause_store: PauseStore | None = None,
        answer_window: timedelta = DEFAULT_ANSWER_WINDOW,
    ) -> DeskRun:
        """One decision, end to end. With ``pause_store``, an escalation holds it for a human.

        Without a store the behaviour is exactly what it was before pauses existed: an escalated
        decision concludes as ``HUMAN_REVIEW`` and nothing more happens to it. With one, the same
        held record is returned *and* the continuation is written so a human's answer can resume
        it — see :meth:`resume` and `decision/pause.py`.
        """
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

        # The feed-sanity gate (`desk/feed_sanity.py`), deterministic and before anything reasons.
        # Measured on 2026-09-25 (`data/feedbugged.json`): the Meta-PM's detection F1 on planted
        # wrong values was 0.0 — it read a +999% 24h change as an ordinary move and a NaN price
        # without comment. Quarantine screens for hostile *text*; this screens for impossible or
        # self-contradicting *figures*. A failing figure is withheld with its item's identity kept,
        # so it never reaches an analyst or the model as a fact, and every count downstream holds.
        evidence, sanity = sanity_screen(evidence, token_price=token_price)
        notes.extend(sanity.render())

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

        # The same gate, on the panel. An analyst is an upstream reasoner, and the injected-step
        # eval showed one wrong step is enough: a quant line claiming a 6895bps edge turned the NVDA
        # abstention into TRADE BUY 50. A view claiming an absurd edge, or quoting the 24h move at
        # 100x the feed's own figure, is dropped before consensus is taken and before the frame is
        # built — and the decision records which view and why.
        kept_views, panel_sanity = screen_views(
            panel.views, move_24h=market_move([e.claim for e in evidence])
        )
        panel.views[:] = kept_views
        notes.extend(panel_sanity.render())

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
        # The panel's agreement at every threshold, the stated one marked (`decision/verdicts.py`).
        # Recorded for transparency only: m-of-K voting lost to the desk's plurality on the settled
        # record (45.1% against 49.0%), so it never binds a decision.
        from argus.decision.verdicts import agreement_note

        notes.append(agreement_note([v.signal for v in panel.views]))

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
        # it. `desk/mandate.py` still enforces every limit afterwards — telling the model first
        # changes what it reasons about, checking it after keeps the answer correct.
        active_mandate = Mandate(profile=profile) if profile is not None else None
        mandate_block = active_mandate.render() if active_mandate is not None else ""

        # Evidence is ORDERED by what this trader acts on, never filtered — `desk/mandate.py`'s
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

        # The trajectory the breaker reads (`agents/circuit.py`): every prior decision in the
        # ledger history, reduced to instrument, side, size and whether it was a decision at all.
        # Read from the same history the memory block uses, so a replay of the ledger through
        # `circuit.assess` is the computation the desk ran live, not an approximation of it.
        circuit = CircuitRecord()
        trajectory = [s for s in (Step.from_record(row) for row in history) if s is not None]
        # browser_use's escalating nudge, told to the model before it decides: one equivalent
        # order short of the trip, the model is warned rather than silently overruled next time.
        nudge = nudge_for(
            trajectory, symbol=symbol, now=session.as_of, thresholds=self.circuit_thresholds
        )
        if nudge is not None:
            circuit.nudges.append(nudge)
            notes.append(nudge)

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
                + ([nudge] if nudge is not None else [])
            ),
        )
        # The one call that decides, guarded: a deadline, a named give-up, and a safe no-op in
        # place of an exception. Until now a hung or failing Meta-PM call raised straight out of
        # `run`, and the cycle lost every symbol still to be decided after it. The deliberation is
        # returned *with* the proof from inside the guarded call rather than read off `self.pm`
        # afterwards: a call abandoned at the deadline may still finish on its daemon thread and
        # would otherwise overwrite `last_deliberation` under a later decision.
        pm = self.pm
        outcome = call_with_retry(
            lambda: (pm.decide(frame, decision_id=decision_id), pm.last_deliberation),
            name="meta_pm.decide", policy=self.pm_policy,
            # A spent budget is the runner ending the cycle (it is raised before any request),
            # not a fault of this decision: it propagates exactly as it always has.
            propagate=(BudgetExhausted,),
        )
        circuit.faults.extend(a.as_dict() for a in outcome.attempts if a.fault is not None)
        notes.extend(outcome.render())
        deliberation_record: dict[str, Any] | None = None
        if outcome.value is not None:
            proof, last = outcome.value
            if last is not None:
                # How the committed answer was reached, in the notes and beside the proof — and
                # priced. `Deliberation.calls` counts every model call `decide` made; the first is
                # the single-shot baseline already in the hurdle the model was quoted, and every
                # call past it is extra thinking time charged at the same per-call rate.
                notes.extend(last.render())
                per_call = deliberation_cost_bps(
                    session, thinking=self.pm_thinking, annualised_vol=self.annualised_vol
                )
                extra_bps = round((last.calls - 1) * per_call, 2)
                deliberation_record = {
                    **last.as_dict(),
                    "extra_calls": last.calls - 1,
                    "extra_cost_bps": str(extra_bps),
                    "total_deliberation_bps": str(round(deliberation + extra_bps, 2)),
                }
                if extra_bps > 0:
                    notes.append(
                        f"[deliberation] {last.calls - 1} call(s) beyond the first cost "
                        f"{extra_bps}bps more; this decision's deliberation is "
                        f"{round(deliberation + extra_bps, 2)}bps, not the {deliberation}bps "
                        f"quoted before it ran"
                    )
                deliberation += extra_bps
        else:
            reason = outcome.gave_up or "the Meta-PM call gave up"
            proof = _no_decision(frame, decision_id=decision_id, reason=reason)
            circuit.trips.append({"trip": "call_gave_up", "reason": reason})
            notes.append(
                "[circuit] TRIP call_gave_up — no model decision was obtained, so the desk "
                "degraded to a safe no-op: no order, nothing reaches the venue, and the cycle "
                "continues with the next symbol"
            )

        # The numbers the evidence itself carried, paired with the id that carried them. Without
        # these, `check_grounding` sees only the desk's *computed* facts, and a thesis quoting a
        # figure straight off a piece of evidence is reported as unattributable. Measured on the
        # live log: seq 65 flagged "+0.24%" as unsupported when it was the 24h change the market
        # evidence had stated as 0.0024 — the checker's own unit handling would have matched it,
        # had it been given the value. `evidence_values` is a parameter `grounding.check` has
        # always had and nothing was passing.
        figures = [(e.id, figure) for e in evidence for figure in extract_figures(e.claim)]
        evidence_values: list[tuple[str, float]] = [(i, f.value) for i, f in figures]
        # the unit each evidence value was written in, so a same-unit near miss is found and
        # reported as partial support rather than as an unattributable figure
        grounding = check_grounding(
            proof.llm_original_intent.thesis,
            facts=grounding_facts,
            evidence_values=evidence_values,
            evidence_units=[f.unit for _, f in figures],
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
        #
        # With a pause store (2026-09-25) this is also where the desk *stops*: the state up to this
        # line is written to disk as a `DeskContinuation`, and a human's answer resumes the
        # decision from here rather than re-running the analysts — see `decision/pause.py`.
        escalation = assess_escalation(
            attacked,
            EscalationSignals(
                directional_split=conflicts.has_directional_split,
                # A STALLED debate (`agents/debate.py`, added 2026-09-25) ended because neither side
                # was making progress, with the sides still apart: its own docstring calls the
                # disagreement live. Reading it as converged would tell escalation a split had
                # been settled when it had only stopped being paid for.
                debate_converged=debate.ending not in (Ending.EXHAUSTED_ROUNDS, Ending.STALLED),
                thesis_refuted_by_own_falsifier=challenged.already_refuted,
                refuted_condition=challenged.refuted_condition,
                unresolved_figures=len(grounding.unresolved),
                underlying_halted=underlying_halted,
                halt_reason=halt_reason,
            ),
        )
        if escalation.required:
            notes.append(escalation.render())
        proposed = attacked
        attacked = apply_escalation(attacked, escalation)

        # --- 2d. the trajectory breaker. It may only reduce. ---
        #
        # WebArena's `early_stop` checks, over the desk's own decision history: a step budget, a
        # run of answers that were not decisions, and the k-th functionally-equivalent order —
        # same instrument, same side, size within tolerance, inside the window. Placed after
        # escalation because it asks a different question: not whether this decision is one the
        # desk may take alone, but whether the desk is repeating itself. A trip turns an
        # exposure-opening intent into a no-op and says why; it never adds size or flips a side.
        current = Step(
            symbol=symbol, at=session.as_of, verdict=str(attacked.verdict),
            side=str(attacked.side), quantity=attacked.quantity,
            valid=outcome.ok and not proof.llm_original_intent.thesis.startswith(
                "[unparseable verdict"
            ),
        )
        reading = assess_trajectory(trajectory, current, thresholds=self.circuit_thresholds)
        if reading.tripped and not any(t.get("trip") == "call_gave_up" for t in circuit.trips):
            circuit.trips.append(reading.as_dict())
            if attacked.verdict.opens_exposure:
                attacked = _stopped(attacked)
                notes.append(
                    f"[circuit] TRIP {reading.trip} — {reading.reason}. The intent was reduced to "
                    f"a no-op before the Constitution; the model's own proposal stays on the proof"
                )
            else:
                notes.append(
                    f"[circuit] TRIP {reading.trip} — {reading.reason}. Nothing opened exposure, "
                    f"so nothing was reduced; the stop is recorded for the takeover count"
                )

        continuation = DeskContinuation(
            decision_id=decision_id, symbol=symbol, session=session, token_price=token_price,
            hedges=hedges, constitution=constitution,
            ablated_constitutions=dict(ablated_constitutions or {}), profile=profile,
            proof=proof, frame=frame, panel=panel, notes=notes,
            # Sorted so the record is stable between runs on identical evidence; a set's iteration
            # order would make two identical decisions serialise differently.
            evidence_sources=tuple(sorted({e.source for e in evidence})),
            causal_chain=causal_chain, earnings_read=earnings_read, debate=debate,
            proposed=proposed, escalation=escalation,
        )
        if pause_store is None:
            run = self._conclude(continuation, attacked)
        else:
            run = self._hold_or_replay(
                continuation, attacked, store=pause_store, answer_window=answer_window
            )
        run.deliberation = deliberation_record
        run.feed_sanity = {"evidence": sanity.as_dict(), "analysts": panel_sanity.as_dict()}
        run.circuit = circuit.as_dict()
        return run

    # --- the human loop ---------------------------------------------------------------------------

    def _hold_or_replay(
        self,
        continuation: DeskContinuation,
        held: Intent,
        *,
        store: PauseStore,
        answer_window: timedelta,
    ) -> DeskRun:
        """Record the decision; pause it if it escalated; replay an answer it already has.

        The replay is LangGraph's (`types.py:1004-1011`) keyed by proposal, per `decision/pause.py`:
        the same decision re-run with the same proposal consumes the human's recorded answer rather
        than asking again, and one already acted on is never acted on twice.
        """
        c = continuation
        store.record_decision(
            decision_id=c.decision_id, symbol=c.symbol, at=c.session.as_of,
            reachable=c.proposed.verdict.opens_exposure, escalation=c.escalation,
            model_requested_review=c.proof.llm_original_intent.verdict is Verdict.HUMAN_REVIEW,
        )
        if not c.escalation.required:
            return self._conclude(c, held)

        # Snapshot BEFORE the held run concludes: `_conclude` records a ruling on the proof and
        # appends to the notes, and the continuation must be the state at the paused point, not
        # the state after the held intent was ruled on.
        snapshot = copy.deepcopy(c)
        prior = store.prior(c.decision_id, hash_intent(c.proposed))
        request: PauseRequest | None = None
        if prior is not None and prior.same_proposal:
            if prior.status == "answered" and prior.response is not None:
                if c.session.as_of > prior.request.expires_at:
                    store.expire(prior.request, now=c.session.as_of, stage="replay")
                else:
                    return self.resume_with(
                        snapshot, prior.request, prior.response, store=store,
                        now=c.session.as_of, replayed=True,
                    )
            elif prior.status == "resumed":
                c.notes.append(
                    f"[pause] {prior.request.request_id} was already answered and acted on for "
                    f"this proposal; the re-run is held rather than executed a second time"
                )
                run = self._conclude(c, held)
                run.pause = prior.request
                return run
            elif prior.status == "paused":
                request = prior.request
        elif prior is not None and prior.status in ("paused", "answered"):
            # An answer to a proposal this run no longer makes is withdrawn, not left live: left
            # "answered", a later `resume` would act on the old proposal after the new one was
            # asked about — the same decision acted on twice.
            store.supersede(prior.request, at=c.session.as_of, by_proposal=hash_intent(c.proposed))

        created = request is None
        if request is None:
            request = store.pause(
                snapshot, decision_id=c.decision_id, symbol=c.symbol, as_of=c.session.as_of,
                proposal=c.proposed, escalation=c.escalation,
                pipeline_signature=resume_signature(), answer_window=answer_window,
            )
        c.notes.append(
            f"[pause] held for a human as {request.request_id} until "
            f"{request.expires_at.isoformat()}; the state at this point is written to disk "
            f"(state {request.state_hash[:16]}) and an answer resumes the decision from here — "
            f"python -m argus.decision.pause answer {request.request_id} approve|reject|modify"
        )
        run = self._conclude(c, held)
        run.pause = request
        # Only when this run wrote the request: a re-run that found the same proposal still pending
        # holds a snapshot of *its* state, and the request's durable continuation is the earlier
        # one. Handing out the newer snapshot would pair a request with a state it does not hash to.
        run.continuation = snapshot if created else None
        return run

    def resume(self, store: PauseStore, request_id: str, *, now: datetime) -> DeskRun:
        """Continue a paused decision from its persisted state, on the human's recorded answer.

        This is the durable path: nothing from the process that paused is needed. The continuation
        is refused unless its bytes, its state hash and the pipeline signature all match (see
        `PauseStore.load_continuation`), and the answer is refused unless it fits the request.
        """
        request = store.load_request(request_id)
        status = store.status(request_id)
        if status == "paused":
            raise PauseError(f"{request_id} has not been answered yet")
        if status != "answered":
            raise AlreadyResolved(f"{request_id} is {status or 'unknown'}; nothing to resume")
        response = store.load_response(request_id)
        if response is None:
            raise PauseError(f"{request_id} is marked answered but its response file is missing")
        if now > request.expires_at:
            store.expire(request, now=now, stage="resume")
            raise PauseExpired(
                f"{request_id} closed at {request.expires_at.isoformat()}; an approval of a "
                f"proposal priced at {request.as_of.isoformat()} is not acted on after that"
            )
        continuation = store.load_continuation(request_id, pipeline_signature=resume_signature())
        if not isinstance(continuation, DeskContinuation):
            raise PauseError(f"{request_id} did not restore to a desk continuation")
        return self.resume_with(continuation, request, response, store=store, now=now)

    def resume_with(
        self,
        continuation: DeskContinuation,
        request: PauseRequest,
        response: HumanResponse,
        *,
        store: PauseStore | None = None,
        now: datetime,
        replayed: bool = False,
    ) -> DeskRun:
        """Run the second half of the decision on the intent a human released.

        The in-memory form of :meth:`resume`, and the one it delegates to. The released intent goes
        through the same Constitution, revision, mandate and order stages as any other: a human can
        release a hold, not the risk layer.
        """
        check_response(request, response)
        if hash_intent(continuation.proposed) != request.proposal_hash:
            raise PauseError(
                f"{request.request_id}: the continuation's proposal is not the one the human saw"
            )
        c = copy.deepcopy(continuation)
        released = resolve_intent(c.proposed, response)
        quantity = "" if response.quantity is None else f" to {response.quantity}"
        c.notes.append(
            f"[human] {response.reviewer} answered {response.action.value}{quantity} on "
            f"{request.request_id} (proposed {c.proposed.side.value} {c.proposed.quantity}); "
            f"resumed from the paused point"
            + (" by replaying the recorded answer" if replayed else "")
            + (f" — {response.note}" if response.note else "")
        )
        run = self._conclude(c, released)
        run.pause = request
        run.human = response
        if store is not None:
            store.mark_resumed(
                request, at=now, replayed=replayed,
                outcome={
                    "released_verdict": released.verdict.value,
                    "released_quantity": str(released.quantity),
                    "constitution": None if run.ruling is None else run.ruling.verdict.value,
                    "approved_intent_hash": run.proof.approved_intent_hash,
                    "order": None if run.order is None else run.order.client_order_id,
                    "order_quantity": None if run.order is None else str(run.order.quantity),
                },
            )
        return run

    # --- the second half: from the Constitution to the order -------------------------------------

    def _conclude(self, continuation: DeskContinuation, attacked: Intent) -> DeskRun:
        """Stages 3 onward, on ``attacked`` — the escalated intent, or one a human released."""
        c = continuation
        symbol, session, token_price = c.symbol, c.session, c.token_price
        hedges, constitution, profile = c.hedges, c.constitution, c.profile
        ablated_constitutions = c.ablated_constitutions
        proof, frame, notes = c.proof, c.frame, c.notes

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
        # `desk/mandate.py` and `desk/workbench.TraderProfile` existed and were wired to nothing,
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
            symbol=symbol, as_of=session.as_of, panel=c.panel,
            proof=proof, ruling=ruling, ruled_intent=attacked,
            ablated_rulings=ablated_rulings, order=order, notes=notes,
            evidence=tuple(c.frame.evidence), evidence_sources=c.evidence_sources,
            causal_chain=c.causal_chain, earnings_read=c.earnings_read, debate=c.debate,
        )


__all__ = ["AnalystView", "ConstitutionPolicy", "DeskRun", "Evidence", "TradingDesk"]
