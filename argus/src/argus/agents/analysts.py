"""The five Track-2 sub-theme analysts.

Each answers one of Bitget's named Agentic Trading sub-themes. They produce *intelligence*, never
decisions — the Meta-PM decides and the Constitution binds. That separation is what lets one engine
serve five sub-themes without becoming five products.

**What we take from TradingAgents, and what we fix.** Its bull/bear debate feeding a structured
Research Manager verdict is a good shape and is adopted. Its own teardown records two weaknesses we
correct here (``research/architecture/tradingagents.md``):

* *"Risk management is via prompt guidance, not code enforcement."* Here every number in a prompt is
  computed in Python and injected; the model interprets, never calculates. Enforcement is the
  Constitution, in code.
* Costs never reach the trader prompt. Here the 12bps round trip is stated in every analyst prompt,
  because on this venue the fee is larger than nearly every effect an analyst can find.

**The Source Independence Graph is ours and has no equivalent in the corpus.** Five agents agreeing
after reading one Reuters article is one piece of evidence, not five. Every analyst returns the
source ids behind its view so agreement can be discounted by shared provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from argus.agents.causality import CausalChain, chain_from_response
from argus.agents.earnings import EarningsRead
from argus.agents.earnings import from_response as earnings_from_response
from argus.agents.quarantine import STANDING_INSTRUCTION, render_for_prompt
from argus.llm.base import ChatModel
from argus.llm.qwen import Thinking
from argus.truth.clocks import SessionState
from argus.truth.evidence import Evidence

COST_PREAMBLE = """You are measuring, not deciding. Report the move you actually see, with its
direction and its size in basis points, however small. A 4bps move is a 4bps move: say "bullish,
4bps", not "neutral".

Do NOT withhold a signal because it looks too small to trade. The desk prices transaction cost and
the cost of its own deliberation separately, once, at the decision — a round trip is 12bps and the
full hurdle is higher. If you also suppress small moves, that cost is charged twice and the
decision-maker never learns the move existed.

Use "neutral" when the evidence genuinely points no way, and "insufficient_evidence" when it does
not let you tell. Neither is a comment on whether the move is worth trading.

Every number below was computed by code. Do NOT recompute, estimate or adjust any figure."""

SCHEMA_NOTE = """Return ONLY a JSON object with exactly these keys:
  "signal": one of "bullish" | "bearish" | "neutral" | "insufficient_evidence"
  "magnitude_bps": integer, your estimate of the move in basis points (0 if neutral)
  "confidence": number 0..1
  "reasoning": one or two sentences
  "counter_case": the strongest argument against your own signal
  "source_ids": array of the evidence ids you actually relied on"""


ACTIONABLE_CONFIDENCE = 0.5
"""The line between a view the desk may act on and one it may only note.

Named because `argus.agents.conflict` has to reason about the same boundary: two analysts on
opposite sides of it disagree about whether to trade at all, however close their numbers look.
"""


@dataclass(frozen=True, slots=True)
class AnalystView:
    """One analyst's output. Intelligence, not a decision."""

    analyst: str
    signal: str
    magnitude_bps: int
    confidence: float
    reasoning: str
    counter_case: str
    source_ids: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)
    """The model's full response. Kept because the structured detail an analyst returns — a
    transmission chain, seven surprise scores — is the input to the graders, and discarding it
    here would leave those modules with nothing to grade."""

    @property
    def clears_round_trip(self) -> bool:
        """Is the claimed move even larger than the fee to capture it?"""
        return abs(self.magnitude_bps) > 12

    @property
    def is_actionable(self) -> bool:
        return (
            self.signal in ("bullish", "bearish")
            and self.clears_round_trip
            and self.confidence > ACTIONABLE_CONFIDENCE
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "analyst": self.analyst,
            "signal": self.signal,
            "magnitude_bps": self.magnitude_bps,
            "confidence": round(self.confidence, 3),
            "clears_round_trip": self.clears_round_trip,
            "actionable": self.is_actionable,
            "reasoning": self.reasoning,
            "counter_case": self.counter_case,
            "source_ids": list(self.source_ids),
        }


def _parse_view(analyst: str, raw: dict[str, Any]) -> AnalystView:
    """Coerce a model response into a view, without inventing anything it did not say."""
    signal = str(raw.get("signal", "insufficient_evidence")).strip().lower()
    if signal not in ("bullish", "bearish", "neutral", "insufficient_evidence"):
        signal = "insufficient_evidence"
    try:
        magnitude = int(float(raw.get("magnitude_bps", 0)))
    except (TypeError, ValueError):
        magnitude = 0
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    # A directional call with no magnitude is not a signal. Downgraded rather than guessed at.
    #
    # This stays, but note what it must NOT be used for: it exists because "bullish, 0bps" is
    # self-contradictory, not to filter small moves. Every analyst signal in the first 105 live
    # decisions came back `neutral`, and the cause was the preamble telling analysts that an
    # effect under 12bps "is a loss" — so they reported a real move as no move, and the desk
    # charged its own cost twice: once by never hearing the signal, once by applying the hurdle.
    if signal in ("bullish", "bearish") and magnitude == 0:
        signal = "neutral"

    return AnalystView(
        analyst=analyst,
        signal=signal,
        magnitude_bps=magnitude,
        confidence=confidence,
        reasoning=str(raw.get("reasoning", "")).strip(),
        counter_case=str(raw.get("counter_case", "")).strip(),
        source_ids=tuple(str(s) for s in raw.get("source_ids", []) if str(s).strip()),
        raw=dict(raw),
    )


class Analyst:
    """Base class. Subclasses supply a role prompt and the context they need."""

    name = "analyst"
    role = ""

    def __init__(self, client: ChatModel, *, thinking: Thinking = Thinking.LOW) -> None:
        self._client = client
        self._thinking = thinking

    def _ask(self, body: str, *, max_tokens: int = 700) -> AnalystView:
        raw = self._client.complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        f"{self.role}\n\n{COST_PREAMBLE}\n\n{SCHEMA_NOTE}\n\n"
                        f"{STANDING_INSTRUCTION}"
                    ),
                },
                {"role": "user", "content": body},
            ],
            required_keys=("signal", "confidence", "reasoning"),
            max_tokens=max_tokens,
            thinking=self._thinking,
        )
        return _parse_view(self.name, raw)


class EventAnalyst(Analyst):
    """Sub-theme: Event-Driven Agent.

    Beyond classification. The model must state the **transmission chain** — event to mechanism to
    affected variable to asset — so each link can be graded after the fact. A correct direction
    reached through three wrong links is a failure our scorer catches and a P&L scorer records as
    a success.
    """

    name = "event"
    role = """You are the Event Analyst for a tokenized-equity trading desk.

For the event given, work through the transmission chain explicitly:
  what changed -> what was expected -> how surprising -> which variable moves ->
  which asset prices it -> which assets CANNOT price it right now -> expected magnitude.

State plainly whether the move is already priced. An event everyone has read is not an edge.
Add a "chain" key to your JSON: an array of short strings, one per link, in order."""

    def analyse(
        self, symbol: str, session: SessionState, evidence: list[Evidence]
    ) -> AnalystView:
        body = f"""SYMBOL: {symbol}
ANCHOR SESSION: {session.phase} (asleep: {session.is_anchor_asleep},
  {session.hours_to_next_discovery:.1f}h to genuine price discovery)

EVENTS AND EVIDENCE (each was available at or before the decision instant):
{render_for_prompt(evidence)}"""
        return self._ask(body)

    def analyse_with_chain(
        self, symbol: str, session: SessionState, evidence: list[Evidence]
    ) -> tuple[AnalystView, CausalChain]:
        """The view plus its transmission chain, ready to be graded link by link once the
        outcome is known. A chain that reaches the right direction through broken links is a
        failure, and only this path can record that."""
        view = self.analyse(symbol, session, evidence)
        chain = chain_from_response(
            event=evidence[0].claim if evidence else symbol,
            as_of=session.as_of,
            response=view.raw,
        )
        return view, chain


class SentimentAnalyst(Analyst):
    """Sub-theme: Market Sentiment Agent.

    We do not try to own the best sentiment *model* — FinBERT is the baseline and beating it on
    classification is not where the edge is. We own the **integrity** layer: separating "people are
    bullish" from "new independent information credibly changed expectations".

    The manipulation attack is real and quantified at roughly 50% profit uplift against a naive
    consumer, and no defence exists anywhere in the 922-source corpus.

    **The feed problem this docstring described is fixed; the caveat about the underlying
    literature is not.** Bitget's social Skill returned nothing in 93% of live cycles, and the
    free replacements the sentiment audit recommended were probed directly on 2026-09-13:
    StockTwits answers 403 to four different User-Agents, CNN's Fear & Greed endpoint answers
    418. Only ``alternative.me`` answered, with a crypto-wide risk-appetite index carried at
    credibility 0.35 and labelled for what it measures — that source is still real and still
    used. **Re-checked 2026-09-16, not carried forward**: ``agent-reach doctor --json`` now
    reports Twitter/X reachable (``twitter-cli``), a real change in this machine's own network
    access since the note above was written.
    ``market.evidence.TwitterSource`` is wired into the real live desk cycle
    (``paper/runner.py``, opt-in, LIVE ONLY) and a real sweep of the full twelve-symbol rToken
    universe found 0 of 12 empty — a real reversal of the 93%-empty figure. One real end-to-end
    call (real Twitter evidence for NVDAUSDT, this analyst, the real Qwen client) produced a
    sensible, source-aware read that explicitly discounted the evidence for being
    "low-credibility social posts." ``market.evidence.RedditSource`` followed the same day, same
    pattern, also 0 of 12 empty after fixing two real Windows-encoding crashes inside the real
    ``rdt-cli`` dependency itself. The published literature caveat is unrelated to either feed
    and still holds: BloombergGPT trails an always-neutral predictor on two of its own five
    sentiment tasks and FinMA sits at chance.

    It stays in the code at its honest standing rather than being deleted, because the promotion
    gate is a real experiment that can still be run: 30 differing paired frames through
    :func:`argus.eval.ablation.paired`, scored on realised basis points. HELPS promotes it;
    NO_EFFECT removes it from the panel. Deleting it now would hide the finding, and leaving it
    unmarked would let a dead feed count as a data source. Recorded in
    :data:`argus.eval.standing.REGISTER`.
    """

    name = "sentiment"
    role = """You are the Sentiment Integrity Analyst.

Your job is NOT to measure mood. It is to decide whether a narrative carries new, independent,
credible information. Weigh each item by: source credibility, likelihood of automation or
coordination, novelty, how many INDEPENDENT sources carry it, whether price has already confirmed
it, and whether it has persisted.

Loud and unanimous is a warning sign, not a confirmation. Five accounts repeating one article is
one source. Say "insufficient_evidence" when a narrative is unsourced, however strong it looks."""

    def analyse(
        self, symbol: str, evidence: list[Evidence], *, social_volume_z: float = 0.0
    ) -> AnalystView:
        body = f"""SYMBOL: {symbol}
SOCIAL VOLUME (z-score vs 30d): {social_volume_z:+.2f}

NARRATIVES:
{render_for_prompt(evidence)}"""
        return self._ask(body)


class EarningsAnalyst(Analyst):
    """Sub-theme: Earnings-Driven Trading Agent.

    Seven independent surprises, not one. They routinely disagree, and the disagreement is the
    signal: a beat with cut guidance and evasive Q&A is a different asset from a beat with raised
    guidance. "EPS beat -> buy" is the field default and it is what we refuse to ship.
    """

    name = "earnings"
    role = """You are the Earnings Analyst.

Decompose the print into SEVEN independent surprises and put them in a "surprises" object in your
JSON, each scored -1..+1:
  reported, consensus, guidance, narrative, valuation, management_credibility, qa

They will often disagree. When they do, say which dominates and why. A headline beat with
deteriorating guidance and evasive Q&A is a bearish print, and calling it bullish because EPS beat
is the single most common error in this field."""

    def analyse(self, symbol: str, evidence: list[Evidence]) -> AnalystView:
        body = f"""SYMBOL: {symbol}

EARNINGS MATERIAL:
{render_for_prompt(evidence)}"""
        return self._ask(body, max_tokens=900)

    def analyse_decomposed(
        self, symbol: str, evidence: list[Evidence]
    ) -> tuple[AnalystView, EarningsRead]:
        """The view plus the seven surprises scored apart.

        The decomposition is deterministic arithmetic over the model's scores, not a second
        opinion: the model reports what it saw on each dimension, and the weighting — guidance
        above the reported number — is applied in code where it can be argued with.
        """
        view = self.analyse(symbol, evidence)
        return view, earnings_from_response(symbol, view.raw)


class CrossAssetAnalyst(Analyst):
    """Sub-theme: Cross-Asset Execution Agent.

    The hedge question, asked properly: given what is *placeable right now*, what is the cheapest
    instrument that removes the most relevant risk? Nothing in the corpus hedges a live token
    against a shut anchor, and for 65.5 hours a week the honest answer is "nothing is placeable".
    """

    name = "cross_asset"
    role = """You are the Cross-Asset Execution Analyst.

You are given a ranked hedge menu, already costed. Choose from it or decline it. An empty menu is a
normal state, not an error — for roughly 65 hours a week the anchor market is shut and no hedge in
the underlying can be placed.

Declining to hedge when nothing worthwhile is placeable is a correct answer. Recommending a hedge
whose cost exceeds the risk it removes is not."""

    def analyse(
        self,
        symbol: str,
        session: SessionState,
        hedge_menu: list[dict[str, Any]],
        *,
        position_notional: Decimal = Decimal("0"),
    ) -> AnalystView:
        menu = (
            chr(10).join(
                f"  - {h['instrument']}: removes {h['risk_reduction']} of risk, "
                f"all-in cost {h['cost_bps']}bps, fill probability {h['execution_probability']}, "
                f"efficiency {h.get('efficiency', 'n/a')}"
                for h in hedge_menu
            )
            if hedge_menu
            else "  (EMPTY — nothing is placeable in this session)"
        )
        body = f"""SYMBOL: {symbol}
POSITION NOTIONAL: {position_notional}
ANCHOR SESSION: {session.phase}, {session.hours_to_next_discovery:.1f}h to discovery

RANKED HEDGE MENU (computed, cost-inclusive):
{menu}"""
        return self._ask(body)


class FactorAnalyst(Analyst):
    """Sub-theme: Factor Discovery Agent — the *proposer* half only.

    Deliberately split from evaluation. Three independent systems were found contaminating this
    exact loop: mcts-llm-alpha overwrites a real overfitting score with the generating model's
    self-judgment, RD-Agent shares one ``APIBackend()`` between proposer and judge, and FactorForge
    feeds the generator the IC of the top factors and asks for variations.

    This class therefore **never sees a score**. It proposes; deterministic code evaluates; the
    trial count is recorded and consumed by the Deflated Sharpe gate.
    """

    name = "factor"
    role = """You are the Factor Proposer.

Propose ONE testable factor for a tokenized equity whose underlying market closes while the token
keeps trading. Put it in a "factor" object: {"name", "formula", "economic_rationale", "horizon"}.

The formula must be computable from: token price, index price, premium, volume, session phase,
hours to next price discovery, and time since last discovery.

You will NOT be told how your previous proposals scored. That is deliberate — a proposer that sees
its own scores optimises against the evaluator instead of against the market."""

    def propose(self, *, context: str, already_proposed: list[str]) -> AnalystView:
        body = f"""MARKET STRUCTURE:
{context}

ALREADY PROPOSED (do not repeat; no performance information is given for these):
{chr(10).join('  - ' + p for p in already_proposed) or '  (none yet)'}"""
        return self._ask(body, max_tokens=900)


@dataclass
class SourceIndependenceGraph:
    """Discount agreement that comes from shared provenance.

    Five analysts agreeing after reading the same article is one piece of evidence, not five.
    Nothing in the corpus does this, and it is the difference between a debate and an echo.
    """

    views: list[AnalystView] = field(default_factory=list)

    def add(self, view: AnalystView) -> None:
        self.views.append(view)

    @property
    def distinct_sources(self) -> set[str]:
        return {s for v in self.views for s in v.source_ids}

    @property
    def agreement_count(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.views:
            out[v.signal] = out.get(v.signal, 0) + 1
        return out

    @property
    def independence_ratio(self) -> float:
        """Distinct sources per analyst. 1.0 means every view rests on its own evidence;
        values near 0 mean the panel is one source wearing several hats."""
        if not self.views:
            return 0.0
        return len(self.distinct_sources) / len(self.views)

    def consensus(self) -> tuple[str, float]:
        """The majority signal and a confidence *discounted by shared provenance*.

        This is the mechanism: a unanimous panel reading one article scores lower than a split
        panel reading five independent sources.
        """
        if not self.views:
            return ("insufficient_evidence", 0.0)
        counts = self.agreement_count
        signal = max(counts, key=lambda k: counts[k])
        backing = [v for v in self.views if v.signal == signal]
        raw = sum(v.confidence for v in backing) / len(backing)
        return (signal, raw * min(1.0, self.independence_ratio))

    def as_dict(self) -> dict[str, Any]:
        signal, confidence = self.consensus()
        return {
            "views": [v.as_dict() for v in self.views],
            "agreement": self.agreement_count,
            "distinct_sources": len(self.distinct_sources),
            "analysts": len(self.views),
            "independence_ratio": round(self.independence_ratio, 3),
            "consensus_signal": signal,
            "provenance_discounted_confidence": round(confidence, 3),
        }
