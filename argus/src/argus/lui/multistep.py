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

from argus.lui.answer import LEAD
from argus.lui.research.kinds import bare_symbol
from argus.truth.coverage import ContextPool

MAX_PARTS = 8
"""The step budget. A ninth part is named in the answer as not run, never silently dropped. Raised
from 4 for a numbered checklist of eight questions (a hostile review, round 30); past four parts
each is shown by its own lead and two lines, the rest one ask away."""
FULL_PARTS = 4
"""Up to this many parts, each is shown in full below the leads."""
_NUMBERED = re.compile(r"(?:^|[\s,;:])(?:and\s+)?(?:\(?\d{1,2}[).]\s+|\([a-h]\)\s*)(?=\S)")
"""An item of a numbered or lettered list: "1) ", "(2) ", "3. ", "(a) " — "(a) what is
BTC's price (b) is it a good time to buy (c) what leverage" answered (b) alone (a hostile
review, round 31)."""

_SPLIT = re.compile(
    r"(?<=[?.!;])\s+(?=\S)|\s*;\s*|\s+(?:and\s+)?then\s+(?=(?:what|how|is|are|should|can|tell|"
    r"show|give|split|hedge|compare|stress|check|run)\b)|\s+(?:and\s+)?also\s+(?=(?:what|how|is|"
    r"are|should|can|tell|show|give)\b)|,?\s+and\s+(?=(?:what|how|is|are|should|can|where|when|"
    r"which|will|does|do)\b)", re.I)


_HOLDINGS_ONLY = re.compile(
    # "I am short $50,000 of NVDA." split off and the move lost its dollars (round 22)
    r"^\s*(?:i\s+(?:hold|own|have)|i(?:'?m|\u2019m|\s+am)\s+(?:long|short)|my\s+(?:book|portfolio|"
    r"holdings)\s+"
    r"(?:is|are)|"
    # "I want to short TSLA." states the trade the next part asks about; split off, its side was
    # lost and a short got a long's liquidation price (a hostile review, round 19, row 646)
    r"i\s+(?:want|plan|intend|am\s+going|'?m\s+going|'?d\s+like|would\s+like)\s+to\s+"
    r"(?:go\s+)?(?:short|long|buy|sell))\b"
    r"(?!.*\b(?:what|how|should|which|when|where|why|is\s+it|can)\b)",
    re.I)

_WORST_CASE = re.compile(r"^\s*(?:and\s+)?(?:what(?:'s|\s+is|\s+are)\s+(?:my|the)\s+worst[\s-]case|"
                         r"how\s+much\s+(?:can|could|would)\s+i\s+lose)\b", re.I)
"""A worst case asked beside a leveraged trade is that trade's worst case, which the leverage
reader answers (worst 24 hours against the side, liquidation odds); read alone it went to the book
stress engine and asked for holdings the question never had (row 646)."""


_ELABORATES = re.compile(
    r"^\s*(?:and\s+|but\s+|so\s+)?(?:is|was|isn'?t|are)\s+(?:that|it|this)\s+(?:normal|typical|"
    r"unusual|good|bad|high|low|expensive|cheap|a\s+lot|much|weighing|worrying|concerning|"
    r"bullish|bearish|a\s+(?:good|bad)\s+sign)\b|"
    r"^\s*(?:and\s+|so\s+)?(?:how|does|did|will)\s+(?:does\s+|did\s+|will\s+)?(?:that|it|this)\s+"
    r"(?:affect|hit|move|impact|weigh|matter|change)\w*|^\s*(?:is|are)\s+(?:that|it|they)\s+"
    r"(?:normal|high|low|weighing|a\s+lot|good|bad|much)|"
    # "…get liquidated, and does that price sit above or below my entry?" lost "that price"
    # from the liquidation it named (a hostile review, round 30)
    r"^\s*(?:and\s+)?(?:does|is|would|will)\s+(?:that|this|the)\s+(?:price|level|number|"
    r"line)\b", re.I)
"""A clause that leans on the one before with a pronoun — "is that normal", "is it weighing on
crypto", "how does that affect the market" — elaborates it: one question, answered whole. Splitting
them cut 3 of 680 single held-out questions in two on the first run (`eval/multistep_eval.py`)."""


_TYPED: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.I), plain) for pattern, plain in (
        (r"^\W*(?:hey|hi|yo|ok(?:ay)?|so|um+|uh+|bro|lol)\b[\s,]*", ""),
        (r"\bwh?ats\b|\bwats?\b", "what is"), (r"\bwat\b", "what"), (r"\bhows\b", "how is"),
        (r"\brn\b", "right now"), (r"\bur\b", "your"), (r"\bu\b", "you"),
        (r"\b(?:shud|shld)\b", "should"), (r"\bwanna\b", "want to"), (r"\bgonna\b", "going to")))
"""Chat spelling, read as written English for a part's reading only: "hey whats btc doing today
and also should i buy eth rn" lost its first half because "whats btc doing today" read as nothing
(a first-time user, round 30). The part keeps its own words in the answer."""


def _formal(piece: str) -> str:
    for typed, plain in _TYPED:
        piece = typed.sub(plain, piece)
    return piece.strip()


_CONSOLE_ONLY = re.compile(r"\b(?:max(?:imum)?|highest|most)\s+(?:allowed\s+)?leverage\b|"
                           r"\bleverage\s+(?:cap|limit)\b|\b(?:good|right|bad)\s+(?:time\s+to\s+"
                           r"(?:buy|sell|get\s+in)|entry)\b|\bwhat\s+leverage\s+should\b", re.I)
"""Items a research engine would misread: "what's the max leverage Bitget allows on TSLA" went to
the liquidation reader at 10x; the console's own tier-list reader answers it."""


@dataclass(frozen=True)
class Part:
    text: str
    request: Any
    inherited: str = ""
    """The instrument carried from an earlier part, when this part named none."""


def _numbered(question: str) -> list[str] | None:
    """The items of a question laid out as a numbered list, or None when it is not one."""
    marks = list(_NUMBERED.finditer(question))
    if len(marks) < 2:
        return None
    items = [question[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(question))]
             for i, m in enumerate(marks)]
    items = [re.sub(r"[\s,;]+(?:and)?\s*$", "", item).strip(" ,.;") for item in items]
    return [item for item in items if item] if len(items) >= 2 else None


def parts(question: str, book: str = "") -> list[Part] | None:
    """The question's parts, each with its research request, or None when it is one question.

    A split counts only when at least two pieces are each a research question on their own (or
    through the name an earlier piece gave them), and they are not the same question twice."""
    from dataclasses import replace

    from argus.lui.research import ResearchKind, detect, follow_up, research_symbols, with_book

    listed = _numbered(question)
    pieces = listed or [p.strip(" ,.;") for p in _SPLIT.split(question.strip())
                        if p and p.strip(" ,.;")]
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
                       or _ELABORATES.match(piece)
                       or (_WORST_CASE.match(piece)
                           and re.search(r"\bliquidat|\b\d+(?:\.\d+)?\s*x\b|\bleverage",
                                         joined[-1], re.I))):
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
        request = with_book(detect(piece) or detect(_formal(piece)), book, piece)
        console_only = bool(listed and _CONSOLE_ONLY.search(piece))
        if console_only:
            request = None  # a venue fact the console's own reader answers (see _CONSOLE_ONLY)
        inherited = ""
        if request is None and earlier and not console_only:
            request = follow_up(piece, earlier, book)
            if request is not None:
                named = research_symbols(" ".join(earlier))[0]
                inherited = named[-1] if named else ""
        if (request is None and earlier and not console_only and not research_symbols(piece)[0]
                and re.search(r"\bwhich\b|\bbetter\b|\bstronger\b", piece, re.I)):
            # "…which is cheaper on valuation and which has better momentum?" asks both parts of
            # both names; the second ran as one name's technicals (a judge, round 19, row 673)
            named = research_symbols(" ".join(earlier))[0]
            if len(named) >= 2:
                both = " or ".join(n.removesuffix("USDT") for n in named[:2])
                request = with_book(detect(f"{piece}, {both}"), book, piece)
                inherited = (" and ".join(n.removesuffix("USDT") for n in named[:2])
                             if request is not None else "")
        if request is None and earlier and not console_only and not research_symbols(piece)[0]:
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
        if (request is None and earlier and not console_only and found
                and found[-1].request is not None
                and found[-1].request.kind is ResearchKind.MACRO):
            # "…; how does that affect BTC": the macro read, for the name this part gives
            named = research_symbols(piece)[0]
            if named:
                request = replace(found[-1].request, symbols=named[:1])
        earlier.append(piece)
        if request is None:
            if listed:
                # an item of a numbered list is a question the asker numbered: it is answered by
                # the console as a whole (``request`` None), or said to be unanswered, never
                # dropped (a hostile review, round 30)
                named = research_symbols(" ".join(earlier))[0]
                found.append(Part(text=piece, request=None,
                                  inherited=named[-1] if named and not research_symbols(piece)[0]
                                  else ""))
            continue
        held = next((p.request for p in reversed(found) if p.request is not None
                     and p.request.book), None)
        if (held is not None and found[-1].request is not None
                and found[-1].request.kind is ResearchKind.BOOK
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


_LEVERED = re.compile(r"\b(?P<x>\d+(?:\.\d+)?)\s*x\s+(?:leveraged\s+)?(?P<side>long|short)\b|"
                      r"\b(?P<side2>long|short)\b[^.?]{0,30}?\bat\s+(?P<x2>\d+(?:\.\d+)?)\s*x\b",
                      re.I)
_MOVE_ASKED = re.compile(r"\b(?P<dir>fall|falls|fell|drop|drops|crash\w*|tank\w*|sink\w*|rise|"
                         r"rises|rall\w*|jump\w*|gain\w*)\s+(?:by\s+)?(?P<pct>\d+(?:\.\d+)?)\s*%",
                         re.I)


def _levered_move(question: str) -> str | None:
    """A move asked of a leveraged position stated in the same message, as one figure: the move
    times the leverage, on the margin and in dollars when the size is given."""
    lev, moved = _LEVERED.search(question), _MOVE_ASKED.search(question)
    if lev is None or moved is None:
        return None
    times = float(lev.group("x") or lev.group("x2"))
    side = (lev.group("side") or lev.group("side2")).lower()
    pct = float(moved.group("pct")) / 100
    down = moved.group("dir").lower().startswith(("fall", "fell", "drop", "crash", "tank", "sink"))
    against = down == (side == "long")
    on_margin = pct * times * (-1 if against else 1)
    size = re.search(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?", question, re.I)
    money = ""
    if size is not None:
        position = float(size.group(1).replace(",", "")) * (1000 if size.group(2) else 1)
        margin = position / times
        money = (f" — {'a loss' if on_margin < 0 else 'a gain'} of about "
                 f"${abs(position * pct):,.0f} on the ${position:,.0f} position, against "
                 f"${margin:,.0f} of margin")
    wiped = on_margin <= -1
    return (f"Bottom line: at {times:g}x {side}, that {pct:.0%} move is {on_margin:+.0%} on the "
            f"margin{money}"
            + (" — more than the margin, so the position is liquidated before it gets there."
               if wiped else "; the liquidation line is in part 1.") )


def answer(question: str, found: list[Part], run: Any) -> tuple[list[str], list[Any], int]:
    """Run each part (in parallel, up to :data:`MAX_PARTS`) with ``run(text, request)`` and
    compose one answer: a lead naming the parts, each part's own lead, then each part in full.
    Returns (lines, sources, parts that could not be answered)."""

    budget = found[:MAX_PARTS]
    # A context-carrying pool: a plain ThreadPoolExecutor starts each part with an empty context,
    # so the parts' source counts (`truth/coverage.py`) and their step trace (`lui/trace.py`)
    # were silently dropped from multi-part answers.
    with ContextPool(max_workers=len(budget)) as pool:
        # an item the console answers whole carries the name an earlier item gave it ("(b) is it
        # a good time to buy" after "(a) what is BTC's price")
        results = list(pool.map(lambda p: run(
            f"{p.text} ({bare_symbol(p.inherited)})" if p.request is None and p.inherited
            else p.text, p.request), budget))
    unread = sum(1 for r in results if getattr(r, "refused", False))
    leads: list[str] = []
    body: list[str] = []
    sources: list[Any] = []
    for number, (part, result) in enumerate(zip(budget, results, strict=True), start=1):
        lines = [str(line) for line in result.lines]
        lead = next((line for line in lines if bool(LEAD.match(line))), lines[0] if lines
                    else "no answer")
        lead = LEAD.sub("", lead, count=1)
        leads.append(f"{number}. {lead[:1].upper()}{lead[1:]}")
        carried = (f" (read as {bare_symbol(part.inherited)})"
                   if part.inherited.endswith("USDT") else
                   f" (read with {part.inherited})" if part.inherited else "")
        body.append(f"Part {number} — “{part.text}”{carried}:")
        # one bold lead per answer: a part's own lead is already listed above, so it is plain here
        shown = [LEAD.sub("", line, count=1) for line in lines if not line.startswith("Data:")]
        if len(budget) > FULL_PARTS:
            # past four parts the answer would run to a hundred lines: each part keeps its lead
            # and the next two lines, and the full answer is one ask away
            shown = shown[1:3]
        body.extend(shown)
        sources.extend(result.sources)
    head = (f"Bottom line: your question has {len(budget)} parts, each answered by its own "
            f"engine below" + (f"; {unread} could not be answered and say why" if unread else "")
            + ":")
    levered = _levered_move(question)
    if levered:
        # "I'm 3x leveraged long $30k of TSLA perp. What happens if TSLA falls 25%?" answered the
        # fall at -25% with the leverage dropped (a hostile review, round 21): the two parts
        # together are one figure, and it leads
        head = levered + " " + head.removeprefix("Bottom line: ").capitalize()
    tail = []
    if len(found) > MAX_PARTS:
        tail.append(f"Assumed: only the first {MAX_PARTS} parts were run; ask the rest on its own: "
                    + "; ".join(f"“{p.text}”" for p in found[MAX_PARTS:]) + ".")
    tail.append("Data: each part's sources are listed with its engine above. This is analysis, "
                "not advice — you make the call.")
    return [head, *leads, *body, *tail], sources, unread


__all__ = ["FULL_PARTS", "MAX_PARTS", "Part", "answer", "parts"]
