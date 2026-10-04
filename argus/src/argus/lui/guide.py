"""The guided research task: five steps from a book to an actionable view (build-list 4.3).

Track 3's demo requirement is "one complete research task, the full flow from question to
actionable insight". The console answers any one question; a first-time visitor still has to know
which five to ask, in which order. Coinbase's AiFi launch (2026-10-03, read in full,
`research/s2-field/ai_trading_notes/32_coinbase-aifi-launch.md`) is the clearest published shape of
that path — portfolio briefing, form a view, research it with the data cost shown, compare ways to
act, stress-test — and it is a closed product, so the arc is rebuilt here and nothing is copied.

What each step runs is an engine the console already has, so the guide adds no second answer path:

1. **Briefing** — :func:`briefing`: each holding's 24-hour move split into the market's part and its
   own (the news engine's beta split), the latest headline naming it with its link, and where the
   book's risk sits (the book engine). It ends by naming the thread to pull: the holding whose own
   move was largest, the part the market does not explain.
2. **Form a view** — a claim about that name, put in the box for the trader to edit, then tested
   by the thesis engine (`lui/thesis.py`).
3. **Research it** — the full research case (`lui/task.py`), with a line saying every source the
   answer read, which did not answer, and what it cost: the data is public and keyless, so the cost
   is model calls, counted at the transport (`truth/coverage.py`).
4. **Choose how to act** — spot, the perpetual at 1x and 3x, and a 30-day call side by side for the
   trader's size (`lui/research/expressions.py`). The decision stays with the trader.
5. **Stress-test it** — the book under a two-standard-deviation month in that name, the move taken
   from its own last 90 days.

The page shows the step, puts the next question in the box (editable, never sent on its own), and
after step five lists each step's bottom line: the insight, in five sentences a trader can act on.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Final

from argus.truth.coverage import ContextPool, Record


@dataclass(frozen=True, slots=True)
class Step:
    title: str
    what: str


STEPS: Final = (
    Step("Briefing", "what moved each holding, the news on it, and where the risk sits"),
    Step("Form a view", "one claim about the name, tested against the record"),
    Step("Research it", "the full case on the name, with every source read and what it cost"),
    Step("Choose how to act", "spot, perpetual or a call, side by side for your size"),
    Step("Stress-test it", "the book under a two-standard-deviation month in the name"),
)
BRIEF_ASKED: Final = re.compile(
    r"\bbrief\s+me\s+on\s+(?:my\s+)?(?:book|portfolio|holdings|positions)\b|"
    r"\b(?:portfolio|book)\s+brief(?:ing)?\b|\bbriefing\s+on\s+my\s+(?:book|portfolio)\b|"
    r"\bwhat(?:'s|\s+is)\s+(?:driving|moving)\s+my\s+(?:book|portfolio|holdings)\b|"
    r"\bwhat\s+moved\s+my\s+(?:book|portfolio|holdings)\b", re.I)
MAX_NAMES: Final = 6
DEFAULT_SIZE: Final = 10_000
OPEN, CLOSE, MORE = "“", "”", ", \u2026"


def _t(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def _own_move(news: dict[str, Any], lines: list[str]) -> tuple[float | None, float | None]:
    """The 24-hour move and the part of it the market does not explain, read from the news
    engine's split line ("\u2026; the other +0.17% is BTC's own")."""
    move = news.get("change_24h_pct")
    own = None
    for line in lines:
        found = re.search(r"the other ([+-]\d+(?:\.\d+)?)% is \w+'s own", line)
        if found:
            own = float(found.group(1))
            break
    return (float(move) if isinstance(move, (int, float)) else None), own


def briefing(book: dict[str, float]) -> tuple[list[str], list[Any], str | None] | None:
    """Step one: the lines, their sources, and the holding to dig into; None without a book."""
    from argus.lui.research import ResearchKind, ResearchRequest, run

    names = [s for s, w in sorted(book.items(), key=lambda kv: -kv[1]) if w > 0][:MAX_NAMES]
    if not names:
        return None
    pool = ContextPool(max_workers=len(names) + 1)
    try:
        news: dict[str, Future[Any]] = {
            s: pool.submit(run, f"news on {_t(s)}", ResearchRequest(kind=ResearchKind.NEWS,
                                                                     symbols=(s,)))
            for s in names}
        risk = pool.submit(run, "where does my book's risk sit",
                           ResearchRequest(kind=ResearchKind.BOOK, symbols=tuple(names),
                                           book=dict(book)))
        rows: list[str] = []
        sources: list[Any] = []
        moves: dict[str, tuple[float | None, float | None]] = {}
        for symbol in names:
            try:
                answer = news[symbol].result(timeout=40)
            except Exception:
                rows.append(f"{_t(symbol)}: the news engine did not answer in time.")
                continue
            sources.extend(answer.sources)
            data = (answer.data or {}).get("news") or {}
            move, own = _own_move(data, answer.lines)
            moves[symbol] = (move, own)
            heads = data.get("headlines") or []
            latest = (f"latest: {OPEN}{heads[0]['title']}{CLOSE} ({heads[0]['feed']}) "
                      f"{heads[0]['link']}" if heads else "no headline named it in 48 hours")
            filed = data.get("filings") or []
            said = (f"{_t(symbol)}: {move:+.2f}% in 24 hours" if move is not None
                    else f"{_t(symbol)}: no 24-hour move read")
            if own is not None and move is not None:
                said += f" ({own:+.2f}% of it its own, the rest the market's)"
            said += (f"; {len(heads)} headline{'s' if len(heads) != 1 else ''} in 48 hours, "
                     f"{latest}" if heads else f"; {latest}")
            if filed:
                said += f"; {len(filed)} SEC filing{'s' if len(filed) != 1 else ''} this week"
            rows.append(said + ".")
        sits = None
        try:
            book_answer = risk.result(timeout=40)
            sources.extend(book_answer.sources)
            sits = next((x for x in book_answer.lines if x.startswith("Where the risk sits:")),
                        None)
        except Exception:
            sits = None
    finally:
        pool.shutdown(wait=False)
    scored = {s: abs(own) for s, (_m, own) in moves.items() if own is not None}
    focus = (max(scored, key=lambda s: scored[s]) if scored
             else max(moves, key=lambda s: abs(moves[s][0] or 0.0)) if moves else names[0])
    move, own = moves.get(focus, (None, None))
    why = (f"its own move, {own:+.2f}% beyond what the market explains, is the largest in the book"
           if own is not None else
           f"it moved most ({move:+.2f}%)" if move is not None else "it is the largest holding")
    lead = (f"Bottom line: the thread to pull in your book today is {_t(focus)} — {why}. Each "
            f"holding's move, the news on it and where the risk sits are below.")
    out = [lead, *rows]
    if sits:
        out.append(sits)
    return out, sources, focus


def two_sigma_month(symbol: str) -> float | None:
    """A two-standard-deviation 30-day fall, in percent, from the last 90 days' daily closes."""
    from argus.lui.research.quick_stats import realised_volatility

    sigma = realised_volatility(symbol, 90)
    if sigma is None:
        return None
    return (1 - math.exp(-2 * sigma * math.sqrt(30 / 365))) * 100


def next_question(step: int, focus: str, *, size: float | None = None) -> str | None:
    """The question step ``step`` asks about ``focus``; None past the last step."""
    name = _t(focus)
    if step == 1:
        return "Brief me on my book: what moved each holding, and what's in the news?"
    if step == 2:
        return f"I think {name} keeps rising over the next month — test that view"
    if step == 3:
        return f"Give me the full research case on {name}"
    if step == 4:
        return (f"Compare spot, perpetual and options for ${size or DEFAULT_SIZE:,.0f} of {name}")
    if step == 5:
        fall = two_sigma_month(focus)
        return (f"Stress test my book if {name} falls {fall:.0f}%" if fall is not None
                else f"Stress test my book if {name} falls 20%")
    return None


def cost_line(record: Record) -> str:
    """What one guided step read and what it cost, from the transport's own count."""
    reached = sorted(s for s, ok in record.answered.items() if ok)
    missing = record.missing
    if not reached and not missing:
        # a step answered wholly from reads this server made minutes ago (the stress step after
        # the research) reached nothing new, and says so rather than "0 sources answered ()"
        calls = record.model_calls
        return ("What this step read: nothing new — every figure came from data this server read "
                "in the last few minutes. Cost: $0 in data; "
                + (f"{calls} call{'s' if calls != 1 else ''} to the language model (Qwen) on the "
                   f"hackathon key." if calls else "no language-model call."))
    said = (f"What this step read: {len(reached)} source{'s' if len(reached) != 1 else ''} "
            f"answered ({', '.join(reached[:12])}{MORE if len(reached) > 12 else ''})")
    if missing:
        said += f"; {len(missing)} did not ({', '.join(missing[:6])})"
    said += (". Cost: $0 in data — every source is public and keyless"
             + (f"; {record.model_calls} call{'s' if record.model_calls != 1 else ''} to the "
                f"language model (Qwen) on the hackathon key" if record.model_calls else
                "; no language-model call") + ".")
    return said


def envelope(step: int, focus: str | None, *, size: float | None = None,
             lines: list[str] | None = None) -> dict[str, Any]:
    """What the page needs to show the step, offer the next one, and close the task."""
    current = STEPS[step - 1]
    following = STEPS[step] if step < len(STEPS) else None
    ask = next_question(step + 1, focus, size=size) if (following and focus) else None
    return {"step": step, "of": len(STEPS), "title": current.title, "what": current.what,
            "focus": focus or "", "takeaway": takeaway(step, lines or []),
            "next": ({"step": step + 1, "title": following.title, "what": following.what,
                      "ask": ask} if following and ask else None)}


def _sentence(line: str) -> str:
    """The first sentence of a line, its "Bottom line:" label dropped."""
    body = re.sub(r"^(?:Bottom line|Actionable)(?: \([^)]*\))?:\s*", "", line.strip())
    cut = re.search(r"(?<=[a-z0-9%)])\.\s+(?=[A-Z])", body)
    return body[:cut.start() + 1] if cut else body


def takeaway(step: int, lines: list[str]) -> str:
    """What step ``step`` concluded, in one sentence, for the task's closing list: the lead's first
    sentence, or the line that carries the step's verdict when the lead only frames it ("your
    thesis on BTC, reason by reason" says nothing until its verdict line)."""
    if not lines:
        return ""
    pick = {
        2: r"^(?:Supported|Not supported|Contradicted|Mixed|Not measurable|Unclear)\b",
        3: r"^Sizing:",
        5: r"^If \w+ moves [+-]?\d",
    }.get(step)
    if pick is not None:
        found = next((x for x in lines if re.match(pick, x.strip())), None)
        if found is not None:
            if step == 3:
                after = lines[lines.index(found) + 1] if lines.index(found) + 1 < len(lines) else ""
                return f"{_sentence(lines[0])} {found.strip()} {_sentence(after)}".strip()
            return _sentence(found) if step != 2 else found.strip()[:400]
    return _sentence(lines[0])
