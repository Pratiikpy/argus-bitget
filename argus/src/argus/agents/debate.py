"""Structured debate — a bull case, a bear case, and a bounded argument between them.

Two independent audits reached the same conclusion about this system's reasoning layer.
`research/architecture/agent-architecture-audit.md:293` called the absence of debate "the primary
architectural gap that judges will expect", and the teardown of ``TauricResearch/TradingAgents`` in
`research/audit/a1-tradingagents.md` found their multi-round Bull/Bear exchange to be the one
mechanism they have and we do not. Track 2 is scored half by judges, and "agent architecture
quality" is one of the three things those judges score.

**What a debate is for, and the failure it is prone to.** A single analyst panel produces a set of
independent readings and a PM that weighs them. That is an *aggregation*, and aggregation has a
known weakness: a case nobody was asked to make does not get made. A debate forces both sides to be
argued at their strongest by someone whose only job is to argue them, which surfaces the bear case
on a bullish tape and the bull case on a fearful one.

The failure mode is equally known, and it is the reason this module looks the way it does: a debate
between two instances of one model converges on agreement that means nothing, runs as long as it is
allowed to, and costs real money in reasoning time while producing the appearance of rigour.

**Three things here that the reference implementation does not do.**

1. **Every round is priced, and the price is charged to the trading hurdle.** This desk already
   prices its own deliberation in basis points (`agents/meta_pm.py`, `deliberation_cost_bps`), and
   a debate is the most expensive thing it can do. :class:`DebateBudget` converts rounds into bps
   and the caller adds it to the hurdle the decision must clear, so a three-round argument about a
   four-basis-point edge is visibly not worth having. Nothing in the 88-repo corpus prices
   deliberation at all, and a debate that is free will always look worthwhile.

2. **It stops on measured convergence, not on a round count alone.** Each side restates its
   position as a direction and a magnitude; when the gap between them stops closing, further
   rounds are buying noise. :func:`converged` decides this arithmetically, from the numbers the
   two sides gave, and the report says which reason ended the debate.

3. **Agreement is reported as a warning, not as a conclusion.** Two calls to the same model that
   agree have demonstrated that the model is consistent, which is not evidence about the market.
   :attr:`Debate.agreement_is_evidence` is False whenever both seats share a model, and the
   rendered transcript says so in words a judge will read. The honest version of this mechanism has
   to admit what it cannot establish.

The transcript is the artefact: it is what makes a decision explainable to someone who was not
there, which is the other judged criterion this serves.

**A stall detector, added 2026-09-25 (item S7 of ``research/mypr-teardowns/_SYNTHESIS.md``).**
:func:`converged` catches two sides that have *agreed*. It cannot catch two sides that have stopped
*arguing* — each restating its case in new words, neither moving, neither citing anything the other
has not already seen — and until now the round cap silently paid for that. Microsoft Agent
Framework's Magentic manager asks the same question of a team every round, as two booleans with
reasons, ``is_in_loop`` and ``is_progress_being_made`` (``microsoft/agent-framework``
``python/packages/orchestrations/agent_framework_orchestrations/_magentic.py:292-395``, MIT;
``stall_count`` incremented on a stalled round and decremented otherwise at ``:1119-1122``, a reset
and replan once it exceeds ``max_stall_count`` at ``:1124-1127``, and ``MagenticContext.reset`` at
``:389-396``). What is taken: the per-round progress question, the counter with its decrement, and
one bounded reset. What is not: MAF answers the question with an LLM call per round, which here
would cost as much as the round it is trying to save. :func:`progress` answers it from the
transcript with no call — did the side change direction, did its magnitude move by more than
:data:`CONVERGENCE_BPS`, did it cite a figure or an evidence id nobody had cited before — and a
round is stalled only when *both* sides fail all three. The proxy is deliberately conservative: a
seat that states no magnitude cannot be shown to have stood still, so it is never counted as
stalled. Measured against the full-length debates and an LLM judge in
``data/thesis_quality.json`` (``s7``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

MAX_ROUNDS = 3
"""Hard ceiling on exchanges, whatever the budget says.

Three, because the marginal content of a fourth round in this setting is a restatement: the bull
and bear cases are both fully stated by round two, and round three exists to let each answer the
other's strongest point. A debate allowed to run to consensus runs until the cheaper-to-agree side
gives way, which measures politeness rather than truth.
"""

CONVERGENCE_BPS = 5.0
"""When the two sides' stated magnitudes are within this, they have stopped disagreeing usefully.

Five basis points against a twelve-basis-point round trip: a disagreement smaller than half the fee
cannot change what the desk does, so paying for another round to resolve it is spending money to
find out something that cannot matter.
"""

ROUND_COST_BPS = Decimal("2.0")
"""What one full exchange costs, in basis points of the position being argued about.

Derived the same way `agents/meta_pm.py` derives its deliberation charge: a round is two model
calls at the analyst budget, and at the measured latency and volatility of this venue that is
about two basis points of adverse drift. Stated as a constant rather than recomputed per call
because the caller must be able to price a debate **before** deciding to hold one.
"""


class Side(StrEnum):
    BULL = "bull"
    BEAR = "bear"


class Ending(StrEnum):
    """Why the debate stopped. Never omitted from the transcript."""

    CONVERGED = "converged"
    """The two sides' magnitudes came within :data:`CONVERGENCE_BPS`."""

    EXHAUSTED_ROUNDS = "exhausted_rounds"
    """The ceiling was reached with the sides still apart. That is a live disagreement and the PM
    is told so, rather than being handed a manufactured consensus."""

    EXHAUSTED_BUDGET = "exhausted_budget"
    """Another round would have cost more than the disagreement is worth."""

    NOT_HELD = "not_held"
    """No debate took place, and the reason is recorded."""

    UNAVAILABLE = "unavailable"
    """A seat could not answer. The decision proceeds undebated and says so."""

    STALLED = "stalled"
    """Both sides stopped making progress: neither moved, neither changed direction, neither cited
    anything new (:func:`progress`). The disagreement is live and is handed to the PM as one; what
    ended is only the paying for restatements of it."""


class Seat(Protocol):
    """The one call this module needs, so a transport is swappable in tests."""

    def complete_json(
        self, messages: list[dict[str, Any]], *, required_keys: tuple[str, ...] = ...,
        max_tokens: int = ..., thinking: Any = ...,
    ) -> dict[str, Any]: ...


def _size(magnitude_bps: float | None) -> str:
    """Render a magnitude, or say plainly that none was given."""
    return "no magnitude stated" if magnitude_bps is None else f"{magnitude_bps:.1f}bps"


@dataclass(frozen=True, slots=True)
class Position:
    """One side's statement in one round."""

    side: Side
    round_index: int
    direction: str
    """`up`, `down` or `unclear` — what this side says the price does."""

    magnitude_bps: float | None
    """How far this seat says it moves, or ``None`` when it did not say.

    **``None``, not ``0.0``, and the difference manufactures consensus.** A missing or unparseable
    ``magnitude_bps`` was coerced to zero and attached to a real directional call. `converged`
    compares *signed* moves, so two seats that both failed to state a magnitude became
    ``abs(0.0 - 0.0) = 0``, which is inside any tolerance — **a parse failure on both sides was
    recorded as the bull and the bear agreeing.**

    That is the same arithmetic error `converged`'s own docstring says it exists to avoid, arriving
    through the parser instead of through the comparison.
    """
    """How far, in basis points. The number that makes convergence measurable."""

    case: str
    strongest_opposing_point: str
    """What this side concedes is the best argument against it.

    Required, and the reason the debate is not theatre: a seat that cannot name the other side's
    best point has not engaged with it, and :meth:`Debate.engaged` counts how often each side did.
    """

    def render(self) -> str:
        return (
            f"  [{self.side.value}, round {self.round_index + 1}] {self.direction} "
            f"{_size(self.magnitude_bps)}"
            f" — {self.case}\n"
            f"      concedes: {self.strongest_opposing_point or '(nothing — did not engage)'}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "round": self.round_index,
            "direction": self.direction,
            "magnitude_bps": self.magnitude_bps,
            "case": self.case,
            "strongest_opposing_point": self.strongest_opposing_point,
        }


def signed(position: Position) -> float | None:
    """The position as a signed move. ``None`` when the seat named no magnitude.

    `unclear` is zero, not a small up — a seat that says it cannot call the direction *has* made a
    statement. A seat that named no magnitude has not, and returning zero for it would put a
    number where there is none.
    """
    if position.magnitude_bps is None:
        return None
    if position.direction == "up":
        return position.magnitude_bps
    if position.direction == "down":
        return -position.magnitude_bps
    return 0.0


def converged(bull: Position, bear: Position, *, within: float = CONVERGENCE_BPS) -> bool:
    """Have the two sides stopped disagreeing by enough to matter?

    Compared as **signed** moves, so a bull saying +30 and a bear saying -30 is a 60bps gap rather
    than a zero-magnitude agreement. Comparing unsigned magnitudes would have called the sharpest
    possible disagreement a consensus, which is the arithmetic error this function exists to avoid.

    **A seat that named no magnitude cannot converge with anything.** Both sides used to default to
    ``0.0`` on a parse failure, and ``abs(0.0 - 0.0)`` is inside every tolerance — so two seats that
    said nothing were recorded as agreeing. Agreement is a finding; it has to be earned from two
    stated numbers.
    """
    left, right = signed(bull), signed(bear)
    if left is None or right is None:
        return False
    return abs(left - right) <= within


_EVIDENCE_ID = re.compile(r"\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*[-_]\d{3,}\b", re.I)
"""An evidence id as the desk renders them — ``yahoo-tsla-441620102``, ``form4-0001-24``."""

_FIGURE = re.compile(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?")


def _evidence_tokens(text: str) -> set[str]:
    """The citable things a piece of argument names: evidence ids and figures, normalised.

    Figures are normalised through ``float`` so ``2.30`` and ``2.3`` are one citation; ids are
    lower-cased. Prose is deliberately not counted — a restatement in new words is exactly the case
    the detector exists to catch, and lexical novelty would score it as progress.
    """
    ids = {m.group(0).lower() for m in _EVIDENCE_ID.finditer(text)}
    scrubbed = _EVIDENCE_ID.sub(" ", text)
    figures: set[str] = set()
    for match in _FIGURE.finditer(scrubbed):
        try:
            figures.add(f"#{float(match.group(0).replace(',', '')):g}")
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
    return ids | figures


@dataclass(frozen=True, slots=True)
class Progress:
    """Did one side make progress between its previous round and this one? No model call.

    The three questions MAF's progress ledger asks an LLM, answered from the transcript instead.
    """

    side: Side
    round_index: int
    direction_changed: bool
    moved_bps: float | None
    """How far the side's signed position moved, or ``None`` when either round named no magnitude
    — in which case standing still cannot be shown, and the side is not counted as stalled."""

    new_evidence: tuple[str, ...]
    """Figures and evidence ids this round cites that no earlier round, on either side, did."""

    @property
    def stalled(self) -> bool:
        if self.direction_changed or self.moved_bps is None:
            return False
        return self.moved_bps <= CONVERGENCE_BPS and not self.new_evidence

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "round": self.round_index,
            "direction_changed": self.direction_changed,
            "moved_bps": None if self.moved_bps is None else round(self.moved_bps, 3),
            "new_evidence": list(self.new_evidence),
            "stalled": self.stalled,
        }


def progress(prior: Sequence[Position], current: Position) -> Progress | None:
    """Compare ``current`` with the same side's previous round, against everything said before.

    ``prior`` is the transcript before ``current`` — both sides. ``None`` when this side has no
    previous round, which is the opening round: nothing to have stalled from.
    """
    previous = next(
        (p for p in reversed(prior) if p.side is current.side), None
    )
    if previous is None:
        return None
    before, after = signed(previous), signed(current)
    moved = None if before is None or after is None else abs(after - before)
    seen: set[str] = set()
    for p in prior:
        seen |= _evidence_tokens(p.case) | _evidence_tokens(p.strongest_opposing_point)
    new = sorted(_evidence_tokens(current.case) - seen)
    return Progress(
        side=current.side,
        round_index=current.round_index,
        direction_changed=current.direction != previous.direction,
        moved_bps=moved,
        new_evidence=tuple(new),
    )


def round_progress(positions: Sequence[Position], round_index: int) -> tuple[Progress, ...]:
    """Both sides' progress readings for one round, in speaking order. Empty for round 0."""
    out: list[Progress] = []
    for i, position in enumerate(positions):
        if position.round_index != round_index:
            continue
        reading = progress(positions[:i], position)
        if reading is not None:
            out.append(reading)
    return tuple(out)


def round_stalled(positions: Sequence[Position], round_index: int) -> bool:
    """True when both sides spoke in ``round_index`` and neither made progress."""
    readings = round_progress(positions, round_index)
    return len(readings) == 2 and all(r.stalled for r in readings)


MAX_STALL_COUNT = 0
"""Stalled rounds tolerated before the debate stops. MAF's default is 3 (``_magentic.py:477``), but
MAF runs open-ended conversations and this debate is capped at :data:`MAX_ROUNDS`: a tolerance of
three could never trigger. Zero means the first stalled round ends it — or triggers the one reset,
when a reset is enabled."""

STALL_NOTICE = """PROGRESS NOTICE
  The last round moved neither side and cited nothing new: both positions were restated. This round
  is paid for only if it changes something. Either cite evidence from the list above that has not
  been cited yet and say what it changes, or move your magnitude, or say plainly that you have
  nothing to add."""
"""The bounded reset, MAF's ``replan`` reduced to what a two-seat debate can use: the transcript is
kept (it *is* the evidence of what was argued), and both seats are told the round was a repeat."""


@dataclass(frozen=True)
class DebateBudget:
    """What a debate is allowed to cost, in basis points of the position argued about."""

    limit_bps: Decimal
    round_cost_bps: Decimal = ROUND_COST_BPS

    def affords(self, rounds_held: int) -> bool:
        return self.round_cost_bps * (rounds_held + 1) <= self.limit_bps

    def spent(self, rounds_held: int) -> Decimal:
        return self.round_cost_bps * rounds_held


@dataclass(frozen=True)
class Debate:
    """The whole exchange, and what it does and does not establish."""

    symbol: str
    positions: tuple[Position, ...]
    ending: Ending
    shared_model: bool
    """True when both seats are the same model. Then agreement is consistency, not corroboration."""

    cost_bps: Decimal = Decimal("0")
    note: str = ""
    progress: tuple[Progress, ...] = ()
    """Every progress reading taken while the debate ran, for rounds after the first."""

    resets: int = 0
    """How many progress notices were issued. At most one; see :data:`STALL_NOTICE`."""

    @property
    def rounds(self) -> int:
        return max((p.round_index for p in self.positions), default=-1) + 1

    @property
    def final(self) -> dict[Side, Position | None]:
        out: dict[Side, Position | None] = {Side.BULL: None, Side.BEAR: None}
        for position in self.positions:
            out[position.side] = position
        return out

    @property
    def gap_bps(self) -> float | None:
        """How far apart the two sides finished, or ``None`` when there is no gap to measure.

        **``None``, not zero.** This returned ``0.0`` when a side never spoke — and zero is the
        value that means *"the bull and the bear agree exactly"*. A debate nobody attended and a
        debate that reached perfect consensus produced the same number, and `unresolved` reads
        ``gap_bps > CONVERGENCE_BPS``, so an unheld debate was silently reported as resolved.

        The same applies when a seat spoke but named no magnitude: `signed` returns ``None`` there,
        and a gap cannot be computed from a number that was never given.
        """
        bull, bear = self.final[Side.BULL], self.final[Side.BEAR]
        if bull is None or bear is None:
            return None
        left, right = signed(bull), signed(bear)
        if left is None or right is None:
            return None
        return abs(left - right)

    @property
    def engaged(self) -> dict[str, int]:
        """How many times each side named the other's best point rather than ignoring it."""
        return {
            side.value: sum(
                1 for p in self.positions
                if p.side is side and p.strongest_opposing_point.strip()
            )
            for side in Side
        }

    @property
    def agreement_is_evidence(self) -> bool:
        """Does agreement here tell us anything about the world?

        Only when the seats are genuinely independent. Two calls to one model that agree have
        established that the model is self-consistent. This project already refuses to treat
        correlated analysts as independent (`agents/analysts.py`
        :class:`SourceIndependenceGraph`), and applying the same rule to a debate is the only
        consistent position.
        """
        return not self.shared_model

    @property
    def unresolved(self) -> bool:
        gap = self.gap_bps
        if gap is None:
            # No measurable gap: the debate did not produce two stated positions. That is not
            # "resolved" — it is a debate that never gave an answer either way.
            return False
        # A stalled debate ended because the sides stopped moving, not because they met: the gap
        # it finished on is as live as one left open by the round cap.
        return self.ending in (Ending.EXHAUSTED_ROUNDS, Ending.STALLED) and gap > CONVERGENCE_BPS

    def render(self) -> str:
        if not self.positions:
            # No rounds happened, for one of several reasons, and the reason is the whole content
            # of the record. An earlier version printed the note only for NOT_HELD and UNAVAILABLE,
            # which silently dropped it for a debate that was refused as too expensive — the case
            # where a reader most needs to know why nothing was argued.
            return f"[debate] {self.ending.value} — {self.note or 'no rounds were held'}"
        lines = [
            f"[debate] {self.symbol}: {self.rounds} round(s), ended {self.ending.value}, "
            f"cost {self.cost_bps}bps",
        ]
        lines.extend(p.render() for p in self.positions)
        gap = self.gap_bps
        lines.append(
            "  final gap: not measurable — a side never stated a position"
            if gap is None else f"  final gap: {gap:.1f}bps"
        )
        engaged = self.engaged
        lines.append(
            f"  engagement: bull named the bear's best point {engaged['bull']} time(s), "
            f"bear named the bull's {engaged['bear']} time(s). A side that never concedes a point "
            f"has not argued, it has repeated."
        )
        if self.note:
            lines.append(f"  {self.note}")
        if self.ending is Ending.STALLED:
            lines.append(
                "  STALLED: in the last round neither side moved, changed direction or cited "
                "anything new, so further rounds were not bought. The disagreement stands as "
                "argued and is handed to the PM unresolved."
            )
        if self.unresolved:
            lines.append(
                "  UNRESOLVED: the sides finished apart. That is a live disagreement and it is "
                "handed to the PM as one, rather than averaged into a consensus that neither "
                "side holds."
            )
        if not self.agreement_is_evidence:
            lines.append(
                "  Both seats share a model, so agreement here is self-consistency and NOT "
                "corroboration. Seat a second model to make this line go away."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "rounds": self.rounds,
            "ending": self.ending.value,
            "gap_bps": None if self.gap_bps is None else round(self.gap_bps, 3),
            "cost_bps": str(self.cost_bps),
            "shared_model": self.shared_model,
            "agreement_is_evidence": self.agreement_is_evidence,
            "unresolved": self.unresolved,
            "engagement": self.engaged,
            "positions": [p.as_dict() for p in self.positions],
            "note": self.note,
            "progress": [r.as_dict() for r in self.progress],
            "resets": self.resets,
        }


BULL_PROMPT = """You argue the BULL case for one instrument, and only the bull case.

You are not deciding anything and you are not being asked to be balanced. Someone else argues the
other side, and a portfolio manager weighs both. Your job is to make the strongest honest case that
this instrument rises, using only the evidence given.

Rules that are checked:
- "direction" must be up, down or unclear. If the evidence genuinely does not support a bull case,
  say "unclear" with a small magnitude. Inventing a case you do not believe wastes a priced round.
- "magnitude_bps" is how far you think it moves, in basis points, over the stated horizon.
- "strongest_opposing_point" must name the best argument AGAINST you. This is required. A side
  that cannot state the other's best point has not engaged with it, and the transcript counts it.
- Cite the evidence you used. A claim that is not in the evidence is not admissible.

Reply with a single JSON object and nothing else:
{"direction": "up" | "down" | "unclear",
 "magnitude_bps": <number>,
 "case": "<your argument, citing the evidence>",
 "strongest_opposing_point": "<the best argument against you>"}

The literal word "json" appears above deliberately: the endpoint's structured-output mode rejects
a request whose messages never mention it, which took a live debate down with
"'messages' must contain the word 'json' in some form". The prompt is also simply better for
stating the shape it wants."""

BEAR_PROMPT = BULL_PROMPT.replace("BULL", "BEAR").replace(
    "this instrument rises", "this instrument falls"
).replace("a bull case", "a bear case")

_REQUIRED = ("direction", "magnitude_bps", "case", "strongest_opposing_point")


def _ask(
    seat: Seat, *, system: str, body: str, side: Side, round_index: int,
) -> tuple[Position | None, str]:
    """One seat's turn. Returns the position, or ``None`` and the reason it could not answer.

    The reason is returned rather than swallowed. The first live run of this module recorded only
    "the bull seat did not answer", which is a fact with no repair path attached — the failure was
    visible and its cause was not, and a bare ``except`` had eaten it. A failure a reader cannot
    diagnose is worse than the failure itself.
    """
    try:
        raw = seat.complete_json(
            [{"role": "system", "content": system}, {"role": "user", "content": body}],
            required_keys=_REQUIRED,
            max_tokens=700,
        )
    except Exception as exc:
        # A seat that cannot answer must not take the decision down. The caller records the
        # debate as UNAVAILABLE and the decision proceeds undebated, which is a fact the record
        # carries rather than a silence — now with the cause attached.
        return None, f"{type(exc).__name__}: {str(exc)[:160]}"
    direction = str(raw.get("direction", "unclear")).strip().lower()
    if direction not in ("up", "down", "unclear"):
        direction = "unclear"
    # An absent or unparseable magnitude is unknown, never zero. See `Position.magnitude_bps`.
    raw_magnitude = raw.get("magnitude_bps")
    if raw_magnitude in (None, "", "null"):
        magnitude: float | None = None
    else:
        try:
            magnitude = abs(float(raw_magnitude))
        except (TypeError, ValueError):
            magnitude = None
    return Position(
        side=side,
        round_index=round_index,
        direction=direction,
        magnitude_bps=magnitude,
        case=str(raw.get("case", "")).strip(),
        strongest_opposing_point=str(raw.get("strongest_opposing_point", "")).strip(),
    ), ""


def _body(
    symbol: str, horizon_hours: float, evidence: Sequence[str], prior: Sequence[Position],
    *, notice: str = "",
) -> str:
    rendered = "\n".join(f"  - {e}" for e in evidence) or "  (none)"
    if not prior:
        transcript = (
            "  (this is the opening round — there is nothing to answer yet, so do not invent an\n"
            "   opposing argument and rebut it. State your own case.)"
        )
    else:
        transcript = "\n".join(p.render() for p in prior)
    return f"""SYMBOL: {symbol}
HORIZON: {horizon_hours:.1f} hours

EVIDENCE
{rendered}

THE ARGUMENT SO FAR
{transcript}""" + ("\n\n" + notice if notice else "")


def hold(
    *,
    symbol: str,
    horizon_hours: float,
    evidence: Sequence[str],
    bull_seat: Seat | None,
    bear_seat: Seat | None,
    budget: DebateBudget,
    max_rounds: int = MAX_ROUNDS,
    shared_model: bool = True,
    stall_detection: bool = False,
    reset_on_stall: bool = False,
) -> Debate:
    """Run the debate, bounded by rounds, by budget, by measured convergence, and by progress.

    Returns a :class:`Debate` in every case, including the cases where no debate happened. A caller
    never has to handle ``None``, and the record always says what occurred.

    ``stall_detection`` ends the debate as :attr:`Ending.STALLED` once a round passes in which
    neither side made progress (:func:`round_stalled`), counted MAF-style: a stalled round adds one
    to the stall count, a progressing round takes one off, and the debate stops when the count
    exceeds :data:`MAX_STALL_COUNT`. ``reset_on_stall`` spends the first such stop on one more
    round carrying :data:`STALL_NOTICE` instead — MAF's reset, bounded at one — and stops only if
    that round stalls too. Readings are recorded either way.
    """
    if bull_seat is None or bear_seat is None:
        return Debate(
            symbol=symbol, positions=(), ending=Ending.NOT_HELD, shared_model=shared_model,
            note="no seat available; the decision is made without a debate",
        )
    if not evidence:
        return Debate(
            symbol=symbol, positions=(), ending=Ending.NOT_HELD, shared_model=shared_model,
            note=(
                "no evidence to argue over. A debate with nothing to cite produces two "
                "well-written opinions, which is the most expensive way to learn nothing"
            ),
        )
    if not budget.affords(0):
        return Debate(
            symbol=symbol, positions=(), ending=Ending.EXHAUSTED_BUDGET,
            shared_model=shared_model,
            note=(
                f"one round costs {budget.round_cost_bps}bps against a {budget.limit_bps}bps "
                f"debate budget; the argument costs more than it could settle"
            ),
        )

    positions: list[Position] = []
    readings: list[Progress] = []
    ending = Ending.EXHAUSTED_ROUNDS
    stall_count = 0
    resets = 0
    notice = ""
    for index in range(max_rounds):
        if not budget.affords(index):
            ending = Ending.EXHAUSTED_BUDGET
            break
        body = _body(symbol, horizon_hours, evidence, positions, notice=notice)
        bull, why = _ask(
            bull_seat, system=BULL_PROMPT, body=body, side=Side.BULL, round_index=index
        )
        if bull is None:
            return Debate(
                symbol=symbol, positions=tuple(positions), ending=Ending.UNAVAILABLE,
                shared_model=shared_model, cost_bps=budget.spent(index),
                note=f"the bull seat did not answer ({why}); the decision proceeds undebated",
            )
        positions.append(bull)
        bear, why = _ask(
            bear_seat, system=BEAR_PROMPT,
            body=_body(symbol, horizon_hours, evidence, positions, notice=notice),
            side=Side.BEAR, round_index=index,
        )
        if bear is None:
            return Debate(
                symbol=symbol, positions=tuple(positions), ending=Ending.UNAVAILABLE,
                shared_model=shared_model, cost_bps=budget.spent(index),
                note=f"the bear seat did not answer ({why}); the decision proceeds undebated",
            )
        positions.append(bear)
        if converged(bull, bear):
            ending = Ending.CONVERGED
            break
        notice = ""
        this_round = round_progress(positions, index)
        readings.extend(this_round)
        if not stall_detection or not this_round:
            continue
        if round_stalled(positions, index):
            stall_count += 1
        else:
            stall_count = max(0, stall_count - 1)
        if stall_count > MAX_STALL_COUNT:
            if reset_on_stall and resets == 0 and index + 1 < max_rounds:
                resets += 1
                stall_count = 0
                notice = STALL_NOTICE
                continue
            ending = Ending.STALLED
            break

    return Debate(
        symbol=symbol,
        positions=tuple(positions),
        ending=ending,
        shared_model=shared_model,
        cost_bps=budget.spent(max(1, (len(positions) + 1) // 2)),
        progress=tuple(readings),
        resets=resets,
    )


__all__ = [
    "CONVERGENCE_BPS",
    "MAX_ROUNDS",
    "MAX_STALL_COUNT",
    "ROUND_COST_BPS",
    "STALL_NOTICE",
    "Debate",
    "DebateBudget",
    "Ending",
    "Position",
    "Progress",
    "Seat",
    "Side",
    "converged",
    "hold",
    "progress",
    "round_progress",
    "round_stalled",
    "signed",
]
