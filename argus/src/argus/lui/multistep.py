"""A question with several parts, answered part by part, each by the engine that owns it.

"Is NVDA overbought, and what would adding 20% of it do to my book? Then how do I split a $50k
buy?" is three questions. The console answered the first engine the question matched and dropped
the rest: a trader got a technicals answer and nothing about their book or the order.

**What was taken, from where.**

* ReAct (Yao et al. 2023; `react/hotpotqa.ipynb` cell 5, MIT): the trace is the explanation. Each
  step here is a named engine call with its question, and the answer shows every step in order, so
  a reader sees what was done to reach the conclusion. Also taken: a bounded step budget and a
  count of steps that could not be read (``n_badcalls`` there, ``unread`` here).
* open_deep_research (`deep_researcher.py:60-175`, MIT): decompose before researching, run the
  sub-questions in parallel, and ask for clarification rather than guess when a part names nothing
  it can be run on.
* WebArena / VisualWebArena (`run.py:198-244`, Apache-2.0 / MIT): the loop breaker. A part that
  repeats an earlier one is not run twice.

**What was not taken.** ReAct's model-chosen next action. Here the decomposition is read from the
question by the same patterns that route single questions, so no model call is spent and the same
question always decomposes the same way; the model is still the reader of each part where the
console already uses one. A model-driven planner would decide *which* engines run — that choice is
the one this console keeps deterministic everywhere else, and it is kept so here.

A part that names no instrument inherits the one named before it ("…and what would adding 20% of
it do to my book?" is about NVDA), through the same follow-up reader the conversation uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from argus.truth.coverage import ContextPool

MAX_PARTS = 4
"""The step budget. A fifth part is named in the answer as not run, never silently dropped."""

_SPLIT = re.compile(
    r"(?<=[?.!;])\s+(?=\S)|\s*;\s*|\s+(?:and\s+)?then\s+(?=(?:what|how|is|are|should|can|tell|"
    r"show|give|split|hedge|compare|stress|check|run)\b)|\s+(?:and\s+)?also\s+(?=(?:what|how|is|"
    r"are|should|can|tell|show|give)\b)|,?\s+and\s+(?=(?:what|how|is|are|should|can|where|when|"
    r"which|will|does|do)\b)", re.I)


_HOLDINGS_ONLY = re.compile(
    r"^\s*(?:i\s+(?:hold|own|have)|i'?m\s+(?:long|short)|my\s+(?:book|portfolio|holdings)\s+"
    r"(?:is|are))\b(?!.*\b(?:what|how|should|which|when|where|why|is\s+it|can)\b)", re.I)


_ELABORATES = re.compile(
    r"^\s*(?:and\s+|but\s+|so\s+)?(?:is|was|isn'?t|are)\s+(?:that|it|this)\s+(?:normal|typical|"
    r"unusual|good|bad|high|low|expensive|cheap|a\s+lot|much|weighing|worrying|concerning|"
    r"bullish|bearish|a\s+(?:good|bad)\s+sign)\b|"
    r"^\s*(?:and\s+|so\s+)?(?:how|does|did|will)\s+(?:does\s+|did\s+|will\s+)?(?:that|it|this)\s+"
    r"(?:affect|hit|move|impact|weigh|matter|change)\w*|^\s*(?:is|are)\s+(?:that|it|they)\s+"
    r"(?:normal|high|low|weighing|a\s+lot|good|bad|much)", re.I)
"""A clause that leans on the one before with a pronoun — "is that normal", "is it weighing on
crypto", "how does that affect the market" — elaborates it: one question, answered whole. Splitting
them cut 3 of 680 single held-out questions in two on the first run (`eval/multistep_eval.py`)."""


@dataclass(frozen=True)
class Part:
    text: str
    request: Any
    inherited: str = ""
    """The instrument carried from an earlier part, when this part named none."""


def parts(question: str, book: str = "") -> list[Part] | None:
    """The question's parts, each with its research request, or None when it is one question.

    A split counts only when at least two pieces are each a research question on their own (or
    through the name an earlier piece gave them), and they are not the same question twice."""
    from dataclasses import replace

    from argus.lui.research import ResearchKind, detect, follow_up, research_symbols, with_book

    pieces = [p.strip(" ,.;") for p in _SPLIT.split(question.strip()) if p and p.strip(" ,.;")]
    if len(pieces) < 2 and not re.search(r"\b(?:compare|vs\.?|versus|between|riskier|safer|"
                                         r"better|worse)\b", question, re.I):
        # "BTC funding rate and ETH open interest": two noun phrases joined by "and", each
        # naming its own instrument — split only when both halves are questions on their own
        halves = re.split(r"\s+and\s+(?=[A-Z]{2,6}\b\s+\w)", question.strip(), maxsplit=1)
        if len(halves) == 2 and all(detect(h) is not None for h in halves):
            pieces = [h.strip(" ,.;?") for h in halves]
    # A piece that only states holdings ("I hold 60% NVDA 40% AAPL") is context for the question
    # beside it, not a question: joined back, "what if the nasdaq drops 10%? I hold …" is one.
    joined: list[str] = []
    for piece in pieces:
        if joined and (_HOLDINGS_ONLY.match(piece) or _HOLDINGS_ONLY.match(joined[-1])
                       or _ELABORATES.match(piece)):
            joined[-1] = f"{joined[-1]} {piece}"
        else:
            joined.append(piece)
    pieces = joined
    if len(pieces) < 2:
        return None
    found: list[Part] = []
    seen: set[tuple[Any, ...]] = set()
    earlier: list[str] = []
    for piece in pieces:
        request = with_book(detect(piece), book, piece)
        inherited = ""
        if request is None and earlier:
            request = follow_up(piece, earlier, book)
            if request is not None:
                named = research_symbols(" ".join(earlier))[0]
                inherited = named[-1] if named else ""
        if request is None and earlier and not research_symbols(piece)[0]:
            named = research_symbols(" ".join(earlier))[0]
            if named:
                name = named[-1].removesuffix("USDT")
                spelled = re.sub(r"\bit'?s\b", f"{name} is", piece, count=1, flags=re.I)
                spelled = re.sub(r"\b(?:it|its|that|this|them)\b", name, spelled, count=1,
                                 flags=re.I)
                if spelled == piece:
                    spelled = f"{piece} {name}"
                request = with_book(detect(spelled), book, piece)
                inherited = named[-1] if request is not None else ""
        if request is None and earlier and found and found[-1].request.kind is ResearchKind.MACRO:
            # "…; how does that affect BTC": the macro read, for the name this part gives
            named = research_symbols(piece)[0]
            if named:
                request = replace(found[-1].request, symbols=named[:1])
        earlier.append(piece)
        if request is None:
            continue
        held = next((p.request for p in reversed(found) if p.request.book), None)
        if (held is not None and found[-1].request.kind is ResearchKind.BOOK
                and re.search(r"\b(?:its|it|the\s+book'?s?)\b", piece, re.I)
                and re.search(r"\bbeta|\bcorrelat|\bvolatil|\bsharpe|\bdrawdown", piece, re.I)):
            # "…also what is its beta to the S&P?" after a book: the same book, asked again
            request = found[-1].request
        if (held is not None and not request.book
                and request.kind in (ResearchKind.HEDGE, ResearchKind.STRESS, ResearchKind.BOOK)):
            # "…and how do I hedge it?" is the book the earlier part stated
            request = replace(request, book=dict(held.book), symbols=tuple(held.book),
                              cash=held.cash)
            inherited = "the book stated earlier"
        key = (str(request.kind), tuple(request.symbols), request.horizon_hours,
               request.weekend)
        if key in seen:
            # the loop breaker: the same engine on the same names runs once, reading both
            # clauses ("how risky is 60/40? also what is its beta to the S&P?")
            at = next(i for i, p in enumerate(found) if (
                str(p.request.kind), tuple(p.request.symbols), p.request.horizon_hours,
                p.request.weekend) == key)
            found[at] = Part(text=f"{found[at].text}; {piece}", request=found[at].request,
                             inherited=found[at].inherited)
            continue
        seen.add(key)
        found.append(Part(text=piece, request=request, inherited=inherited))
    return found if len(found) >= 2 else None


def answer(question: str, found: list[Part], run: Any) -> tuple[list[str], list[Any], int]:
    """Run each part (in parallel, up to :data:`MAX_PARTS`) with ``run(text, request)`` and
    compose one answer: a lead naming the parts, each part's own lead, then each part in full.
    Returns (lines, sources, parts that could not be answered)."""
    from argus.lui.research import _t

    budget = found[:MAX_PARTS]
    # A context-carrying pool: a plain ThreadPoolExecutor starts each part with an empty context,
    # so the parts' source counts (`truth/coverage.py`) and their step trace (`truth/trace.py`)
    # were silently dropped from multi-part answers.
    with ContextPool(max_workers=len(budget)) as pool:
        results = list(pool.map(lambda p: run(p.text, p.request), budget))
    unread = sum(1 for r in results if getattr(r, "refused", False))
    leads: list[str] = []
    body: list[str] = []
    sources: list[Any] = []
    for number, (part, result) in enumerate(zip(budget, results, strict=True), start=1):
        lines = [str(line) for line in result.lines]
        lead = next((line for line in lines if line.startswith("Actionable")), lines[0] if lines
                    else "no answer")
        lead = re.sub(r"^Actionable(?: \(\w+\))?:\s*", "", lead)
        leads.append(f"{number}. {lead[:1].upper()}{lead[1:]}")
        carried = (f" (read as {_t(part.inherited)})" if part.inherited.endswith("USDT") else
                   f" (read with {part.inherited})" if part.inherited else "")
        body.append(f"Part {number} — “{part.text}”{carried}:")
        # one bold lead per answer: a part's own lead is already listed above, so it is plain here
        body.extend(re.sub(r"^Actionable(?: \(\w+\))?:\s*", "", line) for line in lines
                    if not line.startswith("Data:"))
        sources.extend(result.sources)
    head = (f"Actionable: your question has {len(budget)} parts, each answered by its own "
            f"engine below" + (f"; {unread} could not be answered and say why" if unread else "")
            + ":")
    tail = []
    if len(found) > MAX_PARTS:
        tail.append(f"Assumed: only the first {MAX_PARTS} parts were run; ask the rest on its own: "
                    + "; ".join(f"“{p.text}”" for p in found[MAX_PARTS:]) + ".")
    tail.append("Data: each part's sources are listed with its engine above. This is analysis, "
                "not advice — you make the call.")
    return [head, *leads, *body, *tail], sources, unread


__all__ = ["MAX_PARTS", "Part", "answer", "parts"]
