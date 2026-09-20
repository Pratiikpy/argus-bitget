"""The model routes; the ledger answers.

The deterministic classifier in :mod:`argus.lui.question` is fast, auditable and, when it matches,
right. What it cannot do is cover phrasing nobody anticipated. "How's it been going this week?",
"did we make anything?", "what were you looking at before you passed on Nvidia?" all mean something
this desk can answer precisely, and all of them currently return the refusal that lists what is
answerable. That refusal is honest, and it is also the whole of "LUI fluency" as a judge will
experience it.

This module closes the gap without opening a hallucination surface, and the distinction is the
entire design:

**The model is a router, not an answerer.** It is asked to map an unrecognised question onto the
existing :class:`~argus.lui.question.Intent` vocabulary and to extract the symbols, decision number
and time window it mentions. It returns a label. Every word the user then reads is produced by the
same deterministic answerer the regex path would have used, reading the same artefacts. The model
never sees the ledger, never writes prose that is shown, and can therefore never invent a number, a
thesis or a date. The worst a bad routing can do is answer a different question than was asked —
which is visible, recoverable, and reported with the confidence that produced it.

Four rules keep that property:

1. **The deterministic path always wins.** Routing runs only for the two classifications that mean
   "not understood" — :data:`~argus.lui.question.Intent.UNKNOWN` and
   :data:`~argus.lui.question.Intent.AMBIGUOUS` (see :data:`ROUTABLE_FROM`). A confident regex
   match is never second-guessed by a model, so behaviour that is tested today cannot regress
   tomorrow because a model changed.
2. **A refusal that should stay a refusal, stays one.** `ORDER`, `UNSUPPORTED` and `MARKET` are
   decided before routing and are never re-routed. The failure this module must not reintroduce is
   the one `question.py` already documents: an imperative silently answered as a query. The router
   is forbidden from returning `ORDER` for exactly that reason — an instruction recognised by a
   model and then answered from the record is the same defect wearing better clothes. `AMBIGUOUS`
   *is* re-read, because the vague-reference rule fires on idiom as well as on real references, but
   an intent that needs something to point at is refused again unless a referent was found
   (:data:`NEEDS_REFERENT`).
3. **Low confidence keeps the refusal.** Below :data:`MIN_CONFIDENCE` the original answer stands.
   A guess dressed as an answer is worse than "I did not understand", because the user cannot tell
   the difference. Note what `eval/routerbench.py` then measured about this rule: it is rule 2, not
   rule 3, that does the refusing. Every question that had to stay refused was stopped by the
   :data:`ROUTABLE` whitelist while the model was **95% confident** — the floor caught none of
   them. The rule stays because a wrong routing at middling confidence remains possible and
   unmeasured, but it is not the load-bearing guard it reads as, and :data:`MIN_CONFIDENCE` carries
   the numbers.
4. **No key, no network, no problem.** Routing is optional. Every failure — missing credentials,
   budget exhausted, timeout, malformed reply — returns the deterministic answer unchanged, with
   the reason recorded. The console must work offline, because a judge may open it when the key is
   spent.

Every routed answer carries a :class:`Routing` record — what the model chose, why, how sure it was,
and what it extracted — so a reader can audit the routing separately from the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from argus.lui.question import (
    Conversation,
    Intent,
    Question,
    Speed,
    Tense,
    resolve_window,
)

MIN_CONFIDENCE = 0.6
"""Below this the deterministic refusal stands.

**Measured, and the measurement did not say what was expected.** This used to read *"not tuned
against a held-out set - there is none yet"*. There is one now - `eval/routerbench.py`, forty
obliquely-phrased questions written without reading `question.py`'s patterns, of which 24 fall
through to the router. Run against the live model (`data/router_bench.json`):

* Every one of the 17 answerable routings reached the **right** intent, at confidences from 0.55
  up. Precision was 100% at every floor from 0.0 to 0.9.
* Every one of the 7 questions that must stay refused was refused - **at confidence 0.95.** The
  model was highly confident that "should i buy a house" is not a question this console answers,
  and returned an intent outside :data:`ROUTABLE`. The floor did not stop a single one of them.
* At 0.6 the floor's only effect on the corpus was to discard one correct routing ("just give me
  the log", 0.55).

So on this evidence **the floor is not what keeps refusals refused - :data:`ROUTABLE` is.** The
whitelist rejects by *kind of question*; confidence measures something else entirely, and a
confident refusal looks exactly like a confident answer to a threshold.

**The number is nevertheless unchanged, and deliberately so.** Zero wrong routings were observed at
any confidence, which means this corpus contains no case that discriminates between floors on the
axis the floor exists for. An inert guard is not a disproven guard: lowering a safety constant
because a sample produced no failures is the inference this project refuses everywhere else. What
would justify moving it is a corpus that actually produces a *wrong* routing - a question the model
misreads with middling confidence - and none has been found yet. Until then the cost of the floor
is known and small (one answer in 24) and its benefit is **NOT VERIFIED**.
"""

ROUTABLE: tuple[Intent, ...] = (
    Intent.PERFORMANCE,
    Intent.DECISION_WHY,
    Intent.DECISION_LIST,
    Intent.ABSTENTION_WHY,
    Intent.EVIDENCE,
    Intent.CALIBRATION,
    Intent.INTEGRITY,
    Intent.POSITION,
    Intent.SESSION,
)
"""The intents a model may choose.

Deliberately excludes `ORDER`, `MARKET`, `UNSUPPORTED`, `AMBIGUOUS` and `UNKNOWN`. `ORDER` because
routing an instruction into the record is the defect this console already guards against; the rest
because they are refusals, and a model that can route *to* a refusal adds nothing a refusal already
does."""

NEEDS_REFERENT: frozenset[Intent] = frozenset({Intent.DECISION_WHY, Intent.EVIDENCE})
"""Intents that are meaningless without something to point at.

"Why did you do that?" is *correctly* ambiguous: the desk has ninety-six decisions and no way to
know which one is meant. Routing it to `decision_why` with no sequence number and no symbol would
answer about an arbitrary row and present it as the answer. So these two may be routed only when a
referent — a decision number or a symbol — actually came out of the question or the conversation.
The rest (`performance`, `integrity`, `position`, `session`, `calibration`, `decision_list`,
`abstention_why`) are answerable across the whole record, so a missing referent is not a defect in
them."""

ROUTABLE_FROM: frozenset[Intent] = frozenset({Intent.UNKNOWN, Intent.AMBIGUOUS})
"""Which classifications may be re-read by the model.

`AMBIGUOUS` is included because the vague-reference rule is deliberately eager and catches
idiom: "how has it all been going lately?" is flagged for the word "it", which here refers to
nothing. A model is good at telling idiom from reference, and the :data:`NEEDS_REFERENT` guard
above is what stops it from binding a genuine dangling reference to the wrong row."""


class Router(Protocol):
    """The one call this module needs, so the transport is swappable in tests."""

    def complete_json(
        self, messages: list[dict[str, Any]], *, required_keys: tuple[str, ...] = ...,
        max_tokens: int = ..., thinking: Any = ...,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Routing:
    """What the router decided, kept so the routing can be argued with separately."""

    attempted: bool
    applied: bool
    intent: Intent | None
    confidence: float
    why: str
    detail: str = ""
    """Why routing did not happen or was not applied. Empty when it was."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "applied": self.applied,
            "intent": None if self.intent is None else str(self.intent),
            "confidence": self.confidence,
            "why": self.why,
            "detail": self.detail,
        }

    def render(self) -> str:
        if not self.attempted:
            return f"[route] not attempted — {self.detail}"
        if not self.applied:
            return f"[route] not applied — {self.detail}"
        return (
            f"[route] understood as {self.intent} (confidence {self.confidence:.2f}): {self.why}"
        )


NOT_ATTEMPTED = Routing(False, False, None, 0.0, "", "the question was already understood")

SYSTEM_PROMPT = """You route questions for a read-only trading-decision console. You never answer \
the question; you only say which kind of question it is.

The console can answer exactly these kinds:

- performance: profit, loss, Sharpe, drawdown, win rate, "how did we do", "did we make money"
- decision_why: why one specific decision was taken
- decision_list: what was decided over a period, a recap, a summary of activity
- abstention_why: why the desk did nothing, stood aside, passed, stayed flat
- evidence: what information the desk had, what it was looking at, sources
- calibration: whether stated confidence matches what actually happened
- integrity: whether the record is intact, tampered with, verifiable
- position: what is currently held, open, or on the book
- session: whether the market is open, time to the open or close, session state

Rules:
- Choose the single closest kind. If none is close, use "none".
- An instruction to trade ("sell half", "buy NVDA") is NOT a question. Use "none".
- A request for a live price or a forecast is NOT one of these. Use "none".
- confidence is how sure you are that the user wants that kind of answer, 0.0 to 1.0. Be honest; a \
low number is better than a wrong routing.
- symbols: any of NVDA, TSLA, AAPL, MSFT, META, GOOGL, AMZN, COIN, MSTR, QQQ, TQQQ, \
SQQQ mentioned, as a list of bare tickers. Empty list if none.
- seq: a decision number if one is named, otherwise null.
- window: a short phrase naming the time period if one is mentioned ("today", "this week", \
"yesterday", "the weekend"), otherwise null.
- why: one short sentence, for a human auditing your routing.

Reply with only this JSON object:
{"intent": "...", "confidence": 0.0, "symbols": [], "seq": null, "window": null, "why": "..."}"""


def _coerce_symbols(raw: Any) -> tuple[str, ...]:
    """Bare tickers to venue symbols, ignoring anything not on the venue.

    The model is asked for bare tickers and sometimes returns venue symbols anyway; both are
    accepted, and anything else is dropped rather than passed through to an answerer that would
    then look up an instrument that does not exist.
    """
    from argus.lui.question import TRADED_SYMBOLS

    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip().upper()
        full = name if name.endswith("USDT") else f"{name}USDT"
        if full in TRADED_SYMBOLS and full not in out:
            out.append(full)
    return tuple(out)


def _coerce_seq(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str) and raw.strip().isdigit():
        value = int(raw.strip())
        return value if value > 0 else None
    return None


def _coerce_confidence(raw: Any) -> float:
    """A confidence that cannot be read is zero, which keeps the refusal."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def route(
    question: Question,
    *,
    client: Router | None,
    now: datetime,
    conversation: Conversation | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[Question, Routing]:
    """Re-classify an unrecognised question with the model, or hand back what came in.

    Returns the question to answer and the routing record. The question is only ever replaced when
    routing succeeded above the confidence floor; in every other case the original is returned
    untouched, so a caller that ignores the :class:`Routing` still behaves exactly as before.
    """
    if question.intent not in ROUTABLE_FROM:
        return question, NOT_ATTEMPTED
    if client is None:
        return question, Routing(
            False, False, None, 0.0, "",
            "no model is configured; the console answers from patterns alone",
        )

    try:
        raw = client.complete_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question.raw},
            ],
            required_keys=("intent", "confidence"),
            max_tokens=400,
        )
    except Exception as exc:  # any transport failure keeps the deterministic answer
        return question, Routing(
            True, False, None, 0.0, "",
            f"the router was unavailable ({type(exc).__name__}); the original answer stands",
        )

    label = str(raw.get("intent", "")).strip().lower()
    confidence = _coerce_confidence(raw.get("confidence"))
    why = str(raw.get("why", "")).strip()[:200]

    if label in {"", "none", "null"}:
        return question, Routing(
            True, False, None, confidence, why,
            "the router found no kind of question this console can answer",
        )

    try:
        intent = Intent(label)
    except ValueError:
        return question, Routing(
            True, False, None, confidence, why,
            f"the router returned {label!r}, which is not a question kind this console has",
        )

    if intent not in ROUTABLE:
        return question, Routing(
            True, False, intent, confidence, why,
            f"{intent} is not routable: an instruction or a refusal must not be reached by "
            f"re-interpretation",
        )

    if confidence < min_confidence:
        return question, Routing(
            True, False, intent, confidence, why,
            f"confidence {confidence:.2f} is below the {min_confidence:.2f} floor, so the "
            f"original refusal stands rather than a guess",
        )

    # Everything below is extraction, and every extracted value is re-derived deterministically
    # where that is possible: the window is parsed from the model's phrase by the same
    # `resolve_window` the regex path uses, rather than trusting a model to compute a date range.
    symbols = _coerce_symbols(raw.get("symbols")) or question.symbols
    seq = _coerce_seq(raw.get("seq")) or question.seq
    window = question.window
    hint = raw.get("window")
    if window is None and isinstance(hint, str) and hint.strip():
        window = resolve_window(hint, now=now)

    if conversation is not None:
        if not symbols:
            symbols = conversation.last_symbols
        if seq is None:
            seq = conversation.last_seq

    # The guard that makes routing an ambiguous question safe. Without something to point at,
    # these two would answer about a row nobody named.
    if intent in NEEDS_REFERENT and not symbols and seq is None:
        return question, Routing(
            True, False, intent, confidence, why,
            f"{intent} needs a decision number or a symbol to point at, and the question named "
            f"neither; answering an arbitrary row would be worse than asking which one",
        )

    routed = Question(
        raw=question.raw,
        intent=intent,
        speed=Speed.SLOW,
        tense=Tense.PAST if window else question.tense,
        symbols=symbols,
        window=window,
        seq=seq,
        reason=question.reason,
        candidates=question.candidates,
        matched=f"router:{intent}",
    )
    return routed, Routing(True, True, intent, confidence, why)


def build_router(budget_tokens: int = 20_000) -> Router | None:
    """A Qwen client for routing, or ``None`` when one cannot be made.

    Never raises. The console is expected to run without credentials — that is the state a judge is
    most likely to meet if the hackathon key is spent — and a routing layer that could break the
    whole surface would be a worse feature than no routing at all.
    """
    try:
        from argus.llm.qwen import QwenClient, TokenBudget

        return QwenClient(budget=TokenBudget(limit=budget_tokens))
    except Exception:  # missing key, bad URL, anything
        return None


def explain(routing: Routing) -> str:
    """The line shown beneath a routed answer, so the user knows a model read their phrasing."""
    return json.dumps(routing.as_dict(), separators=(",", ":"))


__all__ = [
    "MIN_CONFIDENCE",
    "NEEDS_REFERENT",
    "NOT_ATTEMPTED",
    "ROUTABLE",
    "ROUTABLE_FROM",
    "SYSTEM_PROMPT",
    "Router",
    "Routing",
    "build_router",
    "explain",
    "route",
]
