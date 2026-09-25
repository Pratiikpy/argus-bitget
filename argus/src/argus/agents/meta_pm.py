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

**Rank-before-commit and checker-triggered repair (added 2026-09-25, items S5 and S6 of
``research/mypr-teardowns/_SYNTHESIS.md``).** The first pass used to be the only pass: one call, one
thesis, committed as written, and the desk's grounding check ran *after* commit and could only
report. On the live record that check flagged an unattributable figure in 252 of 610 checked
theses (``data/desk_notes.jsonl``, counted 2026-09-25), and nothing repaired a single one. Two
mechanisms now sit between the model's answer and the proof, each behind a constructor flag:

* **Candidates ranked before commit** (``candidates``). Tree of Thoughts generates N samples and
  keeps the best under a value function (``princeton-nlp/tree-of-thought-llm``
  ``src/tot/methods/bfs.py:16-88``, MIT: ``get_samples`` → ``get_values`` → greedy select).
  ToolBench asks for each further sample to differ from the ones already produced
  (``OpenBMB/ToolBench`` ``toolbench/inference/Algorithms/DFS.py:151-181`` and
  ``Prompts/Tree_search_prompts.py:1-4``, Apache-2.0; licence text at
  ``argus/licenses/toolbench-APACHE-2.0.txt``). The value function is Self-RAG's weighted critique
  sum ``w_rel·relevance + w_sup·support + w_use·utility`` with its own default weights 1.0 / 1.0 /
  0.5 (``AkariAsai/self-rag`` ``retrieval_lm/run_long_form_static.py:11-104``, MIT; fully
  supported = 1, partially = 0.5) — but every term is computed by code from the thesis and the
  frame, never read from a model's opinion of itself.
* **Bounded repair on a named defect** (``repair_rounds``). Reflexion re-runs an attempt with the
  failing test's feedback attached (``noahshinn/reflexion`` ``programming_runs/reflexion.py:48-80``,
  MIT); Self-Refine loops a fixed number of rounds and means to keep the best
  (``madaan/self-refine`` ``src/acronym/run.py:42-75``, Apache-2.0; licence text at
  ``argus/licenses/self-refine-APACHE-2.0.txt``). Here the "failing test" is this module's own
  checker and the feedback names the exact defect — the figure that resolves to nothing, the
  missing falsifier — so the model is told what is wrong rather than asked to find it.

**What was rejected, and why.** ToolBench ranks candidates with pairwise LLM comparisons in both
orders (``LLM_rank/rank_candidate.py:10-72``) and ToT asks the model to value its own samples. Both
put the model in the grader's seat, which is the anti-pattern ``_SYNTHESIS.md`` §4 names for
Self-Refine ("the model grades its own homework"), and both cost calls on a metered key. Reflexion
keeps the *last* attempt; a later round can be worse, so the best-scoring round is kept instead.
Self-Refine's own best-tracking is a no-op as shipped — its gate is ``if total_score >= 0`` on a
non-negative score (``run.py:62``), so every attempt is admitted — and the ranking here is a
lexicographic order on (named defects, score), so an attempt that adds a defect cannot win on
style. Reflexion's separate self-reflection call is fused into the repair call: the checker's
message is already exact, and a model's paraphrase of it would be a second call that adds nothing.
Round counts are capped at two per the synthesis (Reflexion's ``max_iters=10`` is out of scale for
this key).

**Defaults follow the measurement, not the mechanism.** ``eval/thesis_quality.py`` replays the
desk over recorded frames and publishes ``data/thesis_quality.json``; the constructor defaults
:data:`DEFAULT_CANDIDATES` and :data:`DEFAULT_REPAIR_ROUNDS` below state what that run showed and
are switched on only where it helped.

**Forced reflection and a named-rubric self-score (added 2026-09-25, item S23), both off by
default.** Two optional additions to the one JSON object the model returns, each behind its own
flag:

* **Reflection** (``reflection=True``). page-agent makes the model call one macro tool per step
  whose schema carries ``evaluation_previous_goal``, ``memory`` and ``next_goal`` beside the action,
  with the tool choice pinned so the call cannot be skipped (``alibaba/page-agent``, MIT,
  ``packages/core/src/PageAgentCore.ts:386-401`` and ``:285-288``; field wording from
  ``packages/core/src/prompts/system_prompt.md:143-147``). Adapted as three required fields of the
  decision object — :data:`REFLECTION_FIELDS` — so the model evaluates its own last decision on
  this symbol, states what the next look must carry, and names the observation that would change it,
  in the same answer that decides. Changed: page-agent declares all three ``.optional()``
  (``PageAgentCore.ts:397-399``), so its "forced" reflection is forced only in the prompt; here each
  is required and non-empty, and the evaluation must name a prior decision *by a seq the memory
  block actually lists* — checked against the frame with :class:`~argus.llm.idmap.IdMap`, so a
  reflection on a decision that never happened goes back to the model with the list it was shown.
* **Rubric** (``rubric=True``). Four named self-scores beside the single ``confidence`` —
  :mod:`argus.agents.rubric`, after Self-Refine's rubric prompts — each one paired in the record
  with the desk's own measure of the same axis.

Both are off because both change the decision prompt: this item was built with no paid calls
permitted, so neither has been measured on the model, and the default prompt must stay byte-for-byte
what the recorded replays in ``data/thesis_quality_recording.json`` were keyed on
(`eval/decision_primitives.py` replays them offline: a default prompt that drifted would miss the
recording on the first frame and the report would show zero frames replayed; on 2026-09-26 all five
recorded frames replayed, and `tests/test_decision_primitives.py` pins the prompt byte for byte).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from argus.agents import rubric as rubric_mod
from argus.agents.grounding import GroundingReport
from argus.agents.grounding import check as check_grounding
from argus.agents.grounding import extract as extract_figures
from argus.decision.verdicts import (
    ConstitutionRuling,
    Intent,
    Side,
    Verdict,
)
from argus.execution.latency import thinking_budget_cost_bps
from argus.llm.base import ChatModel
from argus.llm.idmap import IdMap
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

REFLECTION_FIELDS = ("evaluation_previous_decision", "memory", "next_goal")
"""page-agent's ``evaluation_previous_goal`` / ``memory`` / ``next_goal``
(``PageAgentCore.ts:395-401``), with the first renamed for what it evaluates here: the desk's last
decision on this symbol, which is the only "previous step" a decision desk has."""

REFLECTION_PROMPT = """Your JSON object must ALSO carry three reflection fields, written before \
you settle the verdict:
  "evaluation_previous_decision": one sentence on the most recent prior decision listed under \
WHAT THIS DESK ALREADY LEARNED HERE. Name it as "seq N" and say whether it proved right, proved \
wrong, or is not yet graded. If that section lists no prior decision, write FIRST LOOK and name no \
seq.
  "memory": one to three sentences — what the next look at this symbol must know from this one.
  "next_goal": the single observation that would change your next decision on this symbol, stated \
so that code or a person can check it.
Name only a seq that section lists. An answer that names any other is returned to you."""
"""Appended to :data:`SYSTEM_PROMPT` only when the reflection flag is on. Wording adapted from
page-agent's output contract (``system_prompt.md:143-147``: "Clearly state success, failure, or
uncertain"; "State the next immediate goal ... in one clear sentence")."""

_MEMORY_SEQ = re.compile(r"^\s*seq (\d+) \(", re.M)
"""An episode line as `agents/recall.py` renders it: ``seq 12 (2026-09-13, weekend): ...``."""

_NAMED_SEQ = re.compile(r"\bseq\s*#?\s*(\d+)", re.I)


def shown_seqs(frame: MarketFrame) -> tuple[str, ...]:
    """The prior decisions the memory block listed, as the model saw them. Empty on a first look."""
    return tuple(_MEMORY_SEQ.findall(frame.memory_block))


def reflection_complaint(response: dict[str, Any], frame: MarketFrame) -> str:
    """What is wrong with the reflection fields, or ``""``. Fed back by `complete_json`.

    The evaluation is checked against the frame: it must name, by seq, a decision the memory block
    listed — :class:`IdMap` in identity mode refuses any other — or say FIRST LOOK when the block
    listed none. A model evaluating a decision it was never shown is reflecting on a hallucination.
    """
    faults: list[str] = []
    for name in REFLECTION_FIELDS:
        value = response.get(name)
        if not isinstance(value, str) or not value.strip():
            faults.append(f"\"{name}\" is missing or empty")
    evaluation = response.get("evaluation_previous_decision")
    if isinstance(evaluation, str) and evaluation.strip():
        shown = shown_seqs(frame)
        named = _NAMED_SEQ.findall(evaluation)
        if shown:
            resolved = IdMap.identities(shown).resolve_all(named)
            if not named:
                faults.append(
                    "\"evaluation_previous_decision\" names no prior decision; name the most "
                    f"recent one as \"seq N\" (listed: {', '.join('seq ' + s for s in shown)})"
                )
            elif resolved.rejected:
                faults.append(
                    f"\"evaluation_previous_decision\" names "
                    f"{', '.join('seq ' + s for s in resolved.rejected)}, which your memory does "
                    f"not list (listed: {', '.join('seq ' + s for s in shown)})"
                )
        elif named:
            faults.append(
                "\"evaluation_previous_decision\" names a seq, but your memory lists no prior "
                "decision on this symbol; write FIRST LOOK"
            )
        elif "first look" not in evaluation.lower():
            faults.append(
                "your memory lists no prior decision on this symbol, so "
                "\"evaluation_previous_decision\" must say FIRST LOOK"
            )
    return "; ".join(faults)


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


DEFAULT_CANDIDATES = 1
"""How many first-pass answers are drawn and ranked before one is committed.

One — rank-before-commit is **off** by default, and that is a measured decision rather than a
cautious one. See ``data/thesis_quality.json`` (``s5``) for the replay it rests on; the flag exists
so the comparison can be re-run and the default changed the day a measurement says it pays."""

DEFAULT_REPAIR_ROUNDS = 0
"""How many repair calls a thesis with a named defect may receive before it is committed.

One, from the replay in ``data/thesis_quality.json`` (``s6``). Capped at :data:`MAX_REPAIR_ROUNDS`
whatever a caller asks for."""

MAX_REPAIR_ROUNDS = 2
"""Hard ceiling on repair calls per decision. `_SYNTHESIS.md` §4: Reflexion's ``max_iters=10`` is
out of scale on a metered key, and a defect that two exact complaints did not fix is a prompt
problem, not a sampling one."""

MAX_CANDIDATES = 3
"""Hard ceiling on ranked candidates. The synthesis sizes S5 at "x3 per decision"; more than three
samples of one frame at temperature 0 mostly reproduce each other."""

W_RELEVANCE = 1.0
W_SUPPORT = 1.0
W_UTILITY = 0.5
"""Self-RAG's own default critique weights (``run_long_form_static.py:13``: ``w_rel=1.0,
w_sup=1.0, w_use=0.5``), taken as shipped rather than tuned here — a weight fitted to the replay
set would be measured on the set it was fitted to."""

DIVERSITY_PROMPT = """This is not the first answer to this decision. Here is an earlier one:
{previous_candidate}
Produce an independent answer. Re-read the evidence and re-derive the decision from it before
looking at the earlier answer's conclusion. You may reach the same verdict, but only for reasons
you checked against the evidence yourself; do not reuse its sentences, and where you disagree with
it, say which piece of evidence decides the disagreement. Return ONLY the same JSON object."""
"""ToolBench's diversity instruction (``Prompts/Tree_search_prompts.py:1-4``, Apache-2.0), adapted.

**Deliberate departure.** ToolBench tells the model its previous trials *failed* and demands actions
"different from all of them". Nothing here failed, and forcing a different *verdict* would buy
diversity by manufacturing a disagreement the evidence may not support — a candidate that trades
only because it was told not to repeat an abstention. So the demand is for an independently
derived answer, with the verdict left free."""

_EVIDENCE_LINE = re.compile(r"^\[(?P<id>[^\]]+)\]\s*(?:\([^)]*\)\s*)?(?P<claim>.*)$", re.S)


def given_values(
    frame: MarketFrame, *, response: dict[str, Any] | None = None
) -> tuple[dict[str, float], list[tuple[str, float]]]:
    """Every number the model was given for this decision, as ``grounding.check`` expects them.

    The same two sets `agents/desk.py` builds before its own check — the computed facts, and the
    figures each piece of evidence carried under that evidence's id — derived here from the frame
    alone, so the Meta-PM can run the check *before* it commits instead of the desk reporting it
    afterwards. Beyond the desk's set it adds the numbers printed in the memory, debate, mandate and
    hedge blocks, because those were printed in the prompt too: a thesis quoting its own memory line
    is quoting something it was given.

    The evidence prefix — ``[id] (source, credibility 0.70, available 2026-...)`` — is stripped
    before extraction. Its credibility and timestamp digits are provenance, not claims, and leaving
    them in would let an invented figure resolve against a timestamp.

    ``response`` adds the model's *own* size and confidence: a thesis that says "buy 5 units at
    0.6 confidence" is stating its decision, not quoting an unattributed fact.
    """
    facts, values, _units = given_values_with_units(frame, response=response)
    return facts, values


def given_values_with_units(
    frame: MarketFrame, *, response: dict[str, Any] | None = None
) -> tuple[dict[str, float], list[tuple[str, float]], list[str]]:
    """:func:`given_values`, plus the unit each evidence value was written in, in the same order.

    The units only decide which values a near miss may be measured against
    (`agents/grounding.py`, ``NEAR_MISS_UNITS``); resolution never reads them.
    """
    facts: dict[str, float] = {
        "round_trip_bps": float(frame.round_trip_bps),
        "deliberation_bps": float(frame.deliberation_bps),
        "total_hurdle_bps": float(frame.total_hurdle_bps),
        "hours_to_discovery": float(frame.session.hours_to_next_discovery),
        "token_price": float(frame.token_price),
        "position_quantity": float(frame.position_quantity),
    }
    if response is not None:
        for key in ("quantity", "confidence"):
            try:
                facts[f"own_{key}"] = float(response.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
    values: list[tuple[str, float]] = []
    units: list[str] = []
    for index, line in enumerate(frame.evidence):
        match = _EVIDENCE_LINE.match(line.strip())
        name, claim = (
            (match.group("id"), match.group("claim")) if match else (f"evidence[{index}]", line)
        )
        for figure in extract_figures(claim):
            values.append((name, figure.value))
            units.append(figure.unit)
    blocks = {
        "memory": frame.memory_block,
        "debate": frame.debate_block,
        "mandate": frame.mandate_block,
        "hedges": json.dumps([dict(h) for h in frame.hedge_menu], default=str),
    }
    for name, text in blocks.items():
        for figure in extract_figures(text):
            values.append((name, figure.value))
            units.append(figure.unit)
    return facts, values, units


@dataclass(frozen=True, slots=True)
class ThesisCheck:
    """What the code-side checker found in one answer, and the score it ranks answers by.

    ``defects`` are *named*, because a repair prompt has to say exactly what is wrong. Each is a
    property the rest of the pipeline already treats as a failure: `_to_intent` downgrades the first
    three to HUMAN_REVIEW, and the desk's grounding note reports the fourth.
    """

    defects: tuple[str, ...]
    unresolved_figures: tuple[str, ...]
    figures: int
    grounding_coverage: float
    engages_evidence: bool
    score: float
    """Self-RAG's weighted sum over code-computed terms; see :func:`check_response`."""

    grounding_support: float = 1.0
    """The grounding checker's three-state support score (fully 1, near miss 0.5, none 0), recorded
    beside ``grounding_coverage``. **Not** used in :attr:`score` — the ranking still uses coverage,
    exactly as it was measured in ``data/thesis_quality.json``; `eval/decision_primitives.py`
    reports how often the two would rank a recorded pair differently before anyone switches."""

    near_miss_figures: tuple[str, ...] = ()
    """Unresolved figures that misquote a given value by under 10% — partially supported."""

    @property
    def clean(self) -> bool:
        return not self.defects

    def as_dict(self) -> dict[str, Any]:
        return {
            "defects": list(self.defects),
            "unresolved_figures": list(self.unresolved_figures),
            "figures": self.figures,
            "grounding_coverage": round(self.grounding_coverage, 4),
            "grounding_support": round(self.grounding_support, 4),
            "near_miss_figures": list(self.near_miss_figures),
            "engages_evidence": self.engages_evidence,
            "score": round(self.score, 4),
        }


STRUCTURAL_DEFECTS = ("unparseable_verdict", "no_falsifier", "no_size", "side_contradicts_lean")
"""Defects of the decision object itself, as distinct from its prose."""


def check_response(response: dict[str, Any], frame: MarketFrame) -> ThesisCheck:
    """Check one raw answer against the frame it was written for. No model call.

    Named defects:

    * ``unparseable_verdict`` — the verdict is not in the enum (`_complain_about`'s case, surviving
      its retries).
    * ``no_falsifier`` — the verdict opens exposure and names no invalidation.
    * ``no_size`` — the verdict acts with a size and states none.
    * ``side_contradicts_lean`` — the answer opens exposure on one side while its own lean points
      the other way: two fields of one answer asserting opposite directions.
    * ``ungrounded_figure`` — a number in the thesis resolves to nothing the model was given
      (:func:`given_values`, checked by `agents/grounding.py` at its own 2% tolerance).
    * ``no_counter_case`` — the prompt requires the strongest argument against the decision, and
      this answer gave none.

    The score is Self-RAG's ``w_rel·relevance + w_sup·support + w_use·utility``
    (``run_long_form_static.py:81-87``) with every term computed rather than generated:
    *support* is the grounding coverage (Self-RAG scores fully supported 1, partially 0.5; a
    coverage fraction is the continuous form of the same thing); *relevance* is whether the thesis
    rests on the evidence at all — a figure resolving to an evidence line or a cited evidence id —
    and *utility* runs from 1 (no structural defect) down to -1, Self-RAG's own range.
    """
    raw_verdict = str(response.get("verdict", "")).strip().lower()
    defects: list[str] = []
    try:
        verdict: Verdict | None = Verdict(raw_verdict)
    except ValueError:
        verdict = None
        defects.append("unparseable_verdict")
    invalidation = [str(x) for x in response.get("invalidation", []) or [] if str(x).strip()]
    try:
        quantity = Decimal(str(response.get("quantity", 0)))
    except (ArithmeticError, ValueError):
        quantity = Decimal("0")
    if verdict is not None and verdict.opens_exposure and not invalidation:
        defects.append("no_falsifier")
    if verdict is not None and verdict.carries_quantity and quantity <= 0:
        defects.append("no_size")
    side = str(response.get("side", "")).strip().lower()
    lean = str(response.get("lean", "none")).strip().lower()
    if verdict is not None and verdict.opens_exposure and (
        (side == "buy" and lean == "down") or (side == "sell" and lean == "up")
    ):
        defects.append("side_contradicts_lean")

    thesis = str(response.get("thesis", ""))
    facts, evidence_values, evidence_units = given_values_with_units(frame, response=response)
    report: GroundingReport = check_grounding(
        thesis, facts=facts, evidence_values=evidence_values, evidence_units=evidence_units
    )
    unresolved = tuple(r.figure.raw for r in report.unresolved)
    if unresolved:
        defects.append("ungrounded_figure")
    if not str(response.get("counter_case", "")).strip():
        defects.append("no_counter_case")

    evidence_names = {name for name, _ in evidence_values} - {
        "memory", "debate", "mandate", "hedges"
    }
    cited_ids = {
        m.group("id") for line in frame.evidence
        if (m := _EVIDENCE_LINE.match(line.strip())) is not None and m.group("id") in thesis
    }
    engages = bool(cited_ids) or any(
        r.resolved and r.source in evidence_names for r in report.resolutions
    )
    structural = sum(1 for d in defects if d in STRUCTURAL_DEFECTS)
    utility = max(-1.0, 1.0 - float(structural))
    score = (
        W_RELEVANCE * (1.0 if engages else 0.0)
        + W_SUPPORT * report.coverage
        + W_UTILITY * utility
    )
    return ThesisCheck(
        defects=tuple(defects),
        unresolved_figures=unresolved,
        figures=len(report.resolutions),
        grounding_coverage=report.coverage,
        engages_evidence=engages,
        score=score,
        grounding_support=report.support_score,
        near_miss_figures=tuple(r.figure.raw for r in report.near_misses),
    )


def rank_key(check: ThesisCheck, index: int) -> tuple[int, float, int]:
    """Fewer named defects first, then the higher score, then the earlier answer.

    Lexicographic on purpose: an answer that adds a defect cannot outrank one without it on score,
    and on a full tie the first answer — the one the single-shot path would have committed —
    stands, so ranking changes a decision only when the checker has a reason to."""
    return (len(check.defects), -check.score, index)


def _repair_instruction(check: ThesisCheck) -> str:
    """The checker's findings, stated so the model can act on them. One line per defect."""
    lines: list[str] = []
    for defect in check.defects:
        if defect == "ungrounded_figure":
            figures = ", ".join(check.unresolved_figures[:8])
            lines.append(
                f"- These figures in your thesis match no number you were given: {figures}. "
                f"Quote a figure exactly as it was given to you, or remove it. Do not derive, "
                f"round-trip or estimate a new number."
            )
        elif defect == "no_falsifier":
            lines.append(
                "- You open exposure without an invalidation condition. Name a specific, "
                "checkable observation that would prove the thesis wrong, or do not open exposure."
            )
        elif defect == "no_size":
            lines.append("- Your verdict acts with a size but states quantity 0. State the size.")
        elif defect == "unparseable_verdict":
            lines.append("- Your verdict is not one of the allowed values.")
        elif defect == "side_contradicts_lean":
            lines.append(
                "- Your side and your lean point in opposite directions. Decide which direction "
                "you hold and make both fields agree, or do not open exposure."
            )
        elif defect == "no_counter_case":
            lines.append("- You gave no counter_case. State the strongest argument against you.")
    return (
        "An automatic checker found these defects in your answer:\n"
        + "\n".join(lines)
        + "\nFix exactly these. Change the decision only if fixing them changes what the evidence "
        "supports. Return ONLY the same JSON object."
    )


@dataclass(frozen=True, slots=True)
class Attempt:
    """One answer the model gave while this decision was being made. Every one is kept."""

    stage: str
    """``candidate`` (a first-pass sample) or ``repair`` (an answer to a named defect)."""

    index: int
    response: dict[str, Any] | None
    check: ThesisCheck | None
    note: str = ""
    """Why an attempt carries no response — the call failed — or which defects prompted it."""

    @property
    def reflection(self) -> dict[str, str] | None:
        """The three reflection fields, when the answer carries all of them."""
        if self.response is None:
            return None
        fields = {name: self.response.get(name) for name in REFLECTION_FIELDS}
        if not all(isinstance(v, str) and v.strip() for v in fields.values()):
            return None
        return {name: str(value).strip() for name, value in fields.items()}

    @property
    def rubric(self) -> dict[str, float] | None:
        """The named self-scores, when the answer carries a complete, valid rubric."""
        return None if self.response is None else rubric_mod.parse(self.response)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "stage": self.stage,
            "index": self.index,
            "verdict": None if self.response is None else str(self.response.get("verdict", "")),
            "thesis": None if self.response is None else str(self.response.get("thesis", "")),
            "check": None if self.check is None else self.check.as_dict(),
            "note": self.note,
        }
        # Present only when asked for, so a record written without the flags is unchanged.
        if self.reflection is not None:
            out["reflection"] = self.reflection
        if self.rubric is not None:
            out["rubric"] = self.rubric
        return out


@dataclass(frozen=True)
class Deliberation:
    """How the committed answer was reached: every attempt, which one won, and what it cost.

    The desk persists this beside the proof, so a reader can see that a thesis was repaired, what
    the repair fixed, and that the pre-repair text existed — a repair that overwrote its input
    would turn the checker into an eraser.
    """

    attempts: tuple[Attempt, ...]
    committed: int
    """Index into :attr:`attempts` of the answer that became the intent."""

    calls: int
    """Model calls made by :meth:`MetaPM.decide`. One is the single-shot baseline; every call past
    the first is extra deliberation the caller should charge at ``deliberation_cost_bps``."""

    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def committed_attempt(self) -> Attempt:
        return self.attempts[self.committed]

    @property
    def repaired(self) -> bool:
        return self.committed_attempt.stage == "repair"

    def as_dict(self) -> dict[str, Any]:
        return {
            "committed": self.committed,
            "calls": self.calls,
            "repaired": self.repaired,
            "attempts": [a.as_dict() for a in self.attempts],
            "notes": list(self.notes),
        }

    def render(self) -> list[str]:
        """One line per fact a reader needs, in the desk-notes style."""
        committed = self.committed_attempt
        lines = [
            f"[deliberation] {self.calls} model call(s), {len(self.attempts)} attempt(s); "
            f"committed {committed.stage} #{committed.index}"
        ]
        first = self.attempts[0]
        if first.check is not None and first.check.defects:
            after = committed.check.defects if committed.check is not None else ()
            lines.append(
                f"[deliberation] first answer carried {', '.join(first.check.defects)}; "
                f"committed answer carries {', '.join(after) if after else 'none'}"
            )
        lines.extend(f"[deliberation] {n}" for n in self.notes)
        reflection = committed.reflection
        if reflection is not None:
            lines.append(
                f"[reflection] evaluated: {reflection['evaluation_previous_decision']} | carries "
                f"forward: {reflection['memory']} | next: {reflection['next_goal']}"
            )
        scores = committed.rubric
        if scores is not None:
            lines.append("[rubric] self-scored: " + ", ".join(
                f"{name} {value:.2f}" for name, value in scores.items()
            ))
        return lines


class MetaPM:
    """The decision-maker. Produces an intent, then responds to being constrained."""

    def __init__(
        self,
        client: ChatModel,
        *,
        max_tokens: int = 1024,
        thinking: Thinking = Thinking.FULL,
        candidates: int = DEFAULT_CANDIDATES,
        repair_rounds: int = DEFAULT_REPAIR_ROUNDS,
        reflection: bool = False,
        rubric: bool = False,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        # The budget is a *cost*, priced by `deliberation_cost_bps`: off-hours a FULL budget
        # costs 15.2bps against a 12bps round trip, and a desk quoted that hurdle correctly
        # abstains every cycle. Routine paper cycles run LOW so the record is scoreable; FULL is
        # reserved for decisions where the extra reasoning has a chance of paying for itself.
        self.thinking = thinking
        self.candidates = max(1, min(candidates, MAX_CANDIDATES))
        self.repair_rounds = max(0, min(repair_rounds, MAX_REPAIR_ROUNDS))
        self.reflection = reflection
        """Require page-agent's three reflection fields in the decision object. Off by default;
        see the module docstring for why."""
        self.rubric = rubric
        """Require the named self-rubric (:mod:`argus.agents.rubric`). Off by default."""
        self.last_deliberation: Deliberation | None = None
        """How the most recent :meth:`decide` reached its answer. ``None`` before the first."""

    def system_prompt(self) -> str:
        """:data:`SYSTEM_PROMPT`, plus a section per enabled flag. With no flag on it is the
        constant itself, so every recorded replay keyed by the default prompt still matches."""
        sections = [SYSTEM_PROMPT]
        if self.reflection:
            sections.append(REFLECTION_PROMPT)
        if self.rubric:
            sections.append(rubric_mod.RUBRIC_PROMPT)
        return "\n\n".join(sections)

    def _validator(self, frame: MarketFrame | None) -> Callable[[dict[str, Any]], str]:
        """`_complain_about`, plus the reflection and rubric checks when their flags are on."""
        if not ((self.reflection and frame is not None) or self.rubric):
            return _complain_about

        def validate(response: dict[str, Any]) -> str:
            faults = [_complain_about(response)]
            if self.reflection and frame is not None:
                faults.append(reflection_complaint(response, frame))
            if self.rubric:
                faults.append(rubric_mod.complaint(response))
            return "; ".join(f for f in faults if f)

        return validate

    def _ask(
        self, messages: list[dict[str, Any]], frame: MarketFrame | None = None
    ) -> dict[str, Any]:
        required: tuple[str, ...] = ("verdict", "side", "quantity", "confidence", "thesis")
        if self.reflection and frame is not None:
            required = (*required, *REFLECTION_FIELDS)
        if self.rubric:
            required = (*required, "self_rubric")
        return self._client.complete_json(
            messages,
            required_keys=required,
            validate=self._validator(frame),
            max_tokens=self._max_tokens,
            # The decision that matters gets the full reasoning budget, and streams so it stays
            # under the endpoint's 120s gateway timeout.
            thinking=self.thinking,
        )

    def deliberate(self, frame: MarketFrame) -> Deliberation:
        """Draw, rank, and if needed repair — then say which answer is committed. No proof yet.

        The first call is exactly the single-shot call: same messages, same parameters. Extra
        candidates and repairs are *optional* calls, so a failure in one is recorded and the best
        answer already in hand stands; only the first call's failure propagates, as it always has.
        """
        base = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": frame.to_prompt_block()},
        ]
        first = self._ask(base, frame)
        attempts: list[Attempt] = [
            Attempt(stage="candidate", index=0, response=first,
                    check=check_response(first, frame))
        ]
        calls = 1
        notes: list[str] = []

        for k in range(1, self.candidates):
            previous = "\n".join(
                json.dumps(a.response, default=str) for a in attempts if a.response is not None
            )
            try:
                calls += 1
                answer = self._ask(
                    [*base, {"role": "user", "content": DIVERSITY_PROMPT.format(
                        previous_candidate=previous
                    )}],
                    frame,
                )
            except Exception as exc:  # an optional sample must not take the decision down
                attempts.append(Attempt(
                    stage="candidate", index=k, response=None, check=None,
                    note=f"call failed ({type(exc).__name__}: {str(exc)[:120]})",
                ))
                notes.append(f"candidate #{k} unavailable; ranked the answers in hand")
                break
            attempts.append(Attempt(
                stage="candidate", index=k, response=answer, check=check_response(answer, frame)
            ))

        def best_index() -> int:
            scored = [
                (rank_key(a.check, i), i) for i, a in enumerate(attempts) if a.check is not None
            ]
            return min(scored)[1]

        committed = best_index()
        if committed != 0:
            notes.append(
                f"rank-before-commit chose candidate #{attempts[committed].index} over the first "
                f"answer"
            )

        for r in range(self.repair_rounds):
            current = attempts[committed]
            assert current.check is not None and current.response is not None
            if current.check.clean:
                break
            instruction = _repair_instruction(current.check)
            try:
                calls += 1
                answer = self._ask([
                    *base,
                    {"role": "assistant", "content": json.dumps(current.response, default=str)},
                    {"role": "user", "content": instruction},
                ], frame)
            except Exception as exc:  # an optional repair must not take the decision down
                attempts.append(Attempt(
                    stage="repair", index=r, response=None, check=None,
                    note=f"call failed ({type(exc).__name__}: {str(exc)[:120]})",
                ))
                notes.append("repair unavailable; committed the best answer in hand")
                break
            attempts.append(Attempt(
                stage="repair", index=r, response=answer, check=check_response(answer, frame),
                note="prompted by: " + ", ".join(current.check.defects),
            ))
            committed = best_index()

        return Deliberation(
            attempts=tuple(attempts), committed=committed, calls=calls, notes=tuple(notes)
        )

    def decide(self, frame: MarketFrame, *, decision_id: str) -> AutonomyProof:
        """First pass: the model's unconstrained economic choice.

        With the defaults of one candidate and no clean-answer repair this is one call, exactly as
        before; see :meth:`deliberate` for what the flags add. The committed answer is the model's
        own in every case — ranking and repair choose among, and ask for, answers; they never write
        one — so the proof's ``llm_original_intent`` still attests that the model decided.
        """
        deliberation = self.deliberate(frame)
        self.last_deliberation = deliberation
        response = deliberation.committed_attempt.response
        assert response is not None  # best_index only ever commits an attempt that answered
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
                {"role": "system", "content": self.system_prompt()},
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
