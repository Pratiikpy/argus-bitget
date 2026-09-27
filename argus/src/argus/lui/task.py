"""One complete research task, run live: question to actionable insight, every step by its engine.

Track 3 requires "an accessible demo showing one complete research task — the full flow from
question to actionable insight". The page this replaces rendered a research chain recorded on
2026-09-14: ten days old, a fixed question, raw data structures on the page, and a verdict
("add smaller") contradicted by its own allocation step. A judge reading it saw an artefact, not a
workbench.

This module runs the task now. A trader types the question the way they would ask a colleague
("I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?"); :func:`read_question` reads the
name, the size and the book out of it with the console's own pattern reader and the trader's saved
book, without a model call, and the page says what it read so a misreading is visible and editable.
Eight engines then answer in parallel, each the same engine the console uses for that kind of
question:

1. the live quote and what a round trip costs;
2. the technical picture (Bitget's Skill, checked against a recomputation);
3. the news and filings that name the company, and how much of today's move is the market;
4. earnings, analyst targets and the SEC-filed surprise (equities only);
5. what followed the name's similar past states — base rates, not a forecast;
6. what the proposed trade does to the book held: risk share, beta, stress, worst day, hedge;
7. the book's sector and factor exposure before and after the trade;
8. how to execute the size on today's order book.

The page ends in one verdict — add, add smaller, or do not add — composed by :func:`verdict` from
the figures the engines computed (the IMPACT engine's size ceiling and risk shares, the
execution engine's slices and cost), never from their prose and never by a model. Below it, each
engine's own bottom line, in the order a trader would act on them.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from argus.lui.answer import LEAD
from argus.lui.research import (
    ANALOGUE_DAYS,
    LOOKBACK_DAYS,
    ResearchKind,
    ResearchRequest,
    detect,
    parse_book,
    research_symbols,
    run,
    with_book,
)
from argus.market.universe import contracts, is_equity
from argus.truth.paths import DATA_DIR

DEFAULT_NAME = "TSLA"
DEFAULT_SIZE_PCT = 15.0
DEFAULT_BOOK = "40% NVDA, 30% MSFT, 30% AAPL"
DEFAULT_BOOK_VALUE = Decimal("100000")
"""The account size the execution step is sized against when none is stated, said on the page."""

TASK_RECORD_PATH = DATA_DIR / "research_task_example.json"
"""The kept worked research task: `eval/research_task_record.py` writes it, /brand and the
documents read it."""

_DISCLAIMER = " This is analysis, not advice — you make the call."

IMPACT_TITLE = "What the trade does to your book"
EXPOSURE_TITLE = "Sector and factor exposure, before and after"
EXECUTION_TITLE = "How to execute the size"

STEP_DEADLINE_S = 60.0
"""How long the task waits for its slowest engine. The engines run side by side and usually
finish in a few seconds; one that hangs on an upstream used to hold the whole page with it
(audit, 2026-09-26). A step still running at the deadline is shown as not answering, and the
rest of the task is served."""

# Each label names where the step's figures come from, in words a trader reads; the cards used to
# show module paths ("lui.research", "desk.portfolio.copilot") and an "89 days" that no longer
# matched the analogue engine's window (audit, 2026-09-26).
STEPS: tuple[tuple[str, ResearchKind | None, str], ...] = (
    ("Where it trades and what a trade costs", ResearchKind.QUOTE,
     "Bitget live ticker, funding and the stock's close"),
    ("The technical picture", ResearchKind.TECHNICALS, "bitget-signal, checked against Bitget 4h"),
    ("News, filings and today's move", ResearchKind.NEWS,
     "news feeds and SEC EDGAR; the move split by beta"),
    ("Earnings, analysts and the surprise", ResearchKind.FUNDAMENTALS,
     "bitget-mcp-server and SEC XBRL filings"),
    ("Has it been here before?", ResearchKind.ANALOGUE,
     f"similar past hours in up to {ANALOGUE_DAYS} days of Bitget candles"),
    (IMPACT_TITLE, ResearchKind.IMPACT,
     f"your book's risk on {LOOKBACK_DAYS} days of hourly returns"),
    (EXPOSURE_TITLE, None, "Yahoo sector classification and a four-factor regression"),
    (EXECUTION_TITLE, ResearchKind.EXECUTION, "Bitget live order book"),
)
"""The steps in the order they are shown. ``None`` is the exposures engine, which answers a book
rather than a research request (`lui/exposures.exposures_answer`)."""

CONCLUSION_ORDER = (IMPACT_TITLE, EXPOSURE_TITLE, EXECUTION_TITLE,
                    "Earnings, analysts and the surprise", "News, filings and today's move",
                    "Has it been here before?", "The technical picture",
                    "Where it trades and what a trade costs")
"""The order a trader acts on the steps: what the book can carry first."""


UNSTATED_SIZE_PCT = 20.0
"""The size a typed question that names no size is assessed at: the console's own default
(``research.DEFAULT_SIZE``), said on the page."""

_NOT_A_TASK: dict[ResearchKind, str] = {
    ResearchKind.STRESS: "That is a what-if about the market, not about adding a name, so it has "
    "no single name to research. The console answers it directly",
    ResearchKind.BOOK: "That asks about the book as it stands, not about adding a name. The "
    "console answers it directly",
    ResearchKind.MACRO: "That is a question about the economy, not about one name. The console "
    "answers it directly",
    ResearchKind.CONSTRUCT: "That asks for a whole book to be built, not for one name to be "
    "researched. The console answers it directly",
}


@dataclass(frozen=True)
class Reading:
    """What a typed question was read as: the name, the size and the book the task runs on."""

    name: str
    size_pct: float
    book: dict[str, float]
    cash: float = 0.0
    notes: tuple[str, ...] = ()
    notional: Decimal | None = None
    """A dollar size the question stated ("buy 50k of NVDA"), which the execution step is sized
    on instead of the default book value."""

    @property
    def summary(self) -> str:
        held = " / ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in self.book.items())
        if self.cash:
            held = f"{held} / {self.cash:.0%} cash" if held else f"{self.cash:.0%} cash"
        into = f"to {held}" if held else "to an empty book"
        more = " more" if self.name in self.book else ""
        dollars = f" (${self.notional:,.0f} to execute)" if self.notional else ""
        return f"add {self.size_pct:g}%{more} {self.name.removesuffix('USDT')} {into}{dollars}"


def read_question(text: str, saved_book: str = "") -> Reading | str:
    """The name, size and book a trader's own question asks about, or why it cannot be read.

    The same reader the console uses (``research.detect``), with the trader's saved book applied
    exactly as the console applies it (``research.with_book``), and no model call: the task's page
    promises nothing on it is written by a language model, and that includes reading the question.
    A question the reader does not recognise but that names one listed contract ("do full research
    on TSLA for my book") is run on that name, the saved book and the console's default size, and
    every one of those choices is a note on the page.
    """
    text = text.strip()
    if not text:
        return ("Type a question, for example: I hold 40% NVDA, 30% MSFT, 30% AAPL — should I "
                "add 15% TSLA?")
    request = with_book(detect(text), saved_book, text)
    if request is not None and request.kind in _NOT_A_TASK:
        return f"{_NOT_A_TASK[request.kind]}: open the console and ask it there."
    notes: list[str] = list(request.notes) if request is not None else []
    if request is None or not request.symbols:
        named, _ = research_symbols(text)
        if not named:
            return ("I could not find a name Bitget lists in that question. Name one, and if you "
                    "like a size and what you hold — for example: I hold 40% NVDA, 30% MSFT, 30% "
                    "AAPL — should I add 15% TSLA?")
        book, cash = _saved(saved_book)
        notes.append(f"read {named[0].removesuffix('USDT')} as the name to research")
        if book or cash:
            notes.append("used your saved book")
        notes.append(f"no size was given, so the task assesses a {UNSTATED_SIZE_PCT:g}% position "
                     "— say the size you have in mind to change it")
        notes.extend(_held_note(named[0], book))
        return Reading(name=named[0], size_pct=UNSTATED_SIZE_PCT, book=book, cash=cash,
                       notes=tuple(notes))
    name = request.symbols[0]
    if request.kind is ResearchKind.COMPARE and len(request.symbols) > 1:
        notes.append(f"a comparison names several contracts; the task researches the first, "
                     f"{name.removesuffix('USDT')}")
    book = dict(request.book)
    cash = request.cash
    if not book and not cash:
        book, cash = _saved(saved_book)
        if book or cash:
            notes.append("used your saved book")
    size_pct = request.size * 100 if request.size_stated else UNSTATED_SIZE_PCT
    if not request.size_stated and not any("no size was given" in n for n in notes):
        notes.append(f"no size was given, so the task assesses a {UNSTATED_SIZE_PCT:g}% position "
                     "— say the size you have in mind to change it")
    if request.notional is not None:
        notes.append(f"the execution step is sized on the ${request.notional:,.0f} you named")
    notes.extend(_held_note(name, book))
    return Reading(name=name, size_pct=size_pct, book=book, cash=cash, notes=tuple(notes),
                   notional=request.notional)


def _held_note(name: str, book: dict[str, float]) -> list[str]:
    """Said when the name is already held: the task assesses buying more of it, which is how the
    console's book arithmetic reads an add (``desk.portfolio.rebalance``)."""
    if name not in book:
        return []
    return [f"you already hold {book[name]:.0%} {name.removesuffix('USDT')}, so the task assesses "
            f"buying more of it on top, the rest of the book scaled down"]


def _saved(book_text: str) -> tuple[dict[str, float], float]:
    from argus.lui.research import split_cash

    if not book_text.strip():
        return {}, 0.0
    book, cash = split_cash(book_text, parse_book(book_text))
    return dict(book), cash


@dataclass
class Step:
    title: str
    engine: str
    lines: list[str] = field(default_factory=list)
    refused: bool = False
    applicable: bool = True
    seconds: float = 0.0
    path: tuple[tuple[Any, float], ...] = ()
    """The price path the quote step's chart is drawn from (`lui/task_page.py`), or empty."""
    data: dict[str, Any] = field(default_factory=dict)
    """The engine's figures as data (``Answer.data``), which the verdict is composed from."""

    @property
    def actionable(self) -> str | None:
        for line in self.lines:
            if bool(LEAD.match(line)):
                return line.split(":", 1)[1].strip()
        return None


@dataclass
class Task:
    question: str
    name: str
    size_pct: float
    book: dict[str, float]
    steps: list[Step]
    seconds: float
    asked: str = ""
    """The trader's own words, when the task was run from a typed question."""
    reading: Reading | None = None
    """What those words were read as, shown above the answer so a misreading is visible."""

    @property
    def verdict(self) -> Verdict | None:
        return verdict(self)

    @property
    def conclusion(self) -> list[tuple[str, str]]:
        """Each engine's bottom line, the book's answer first."""
        by_title = {s.title: s for s in self.steps}
        out = []
        for title in CONCLUSION_ORDER:
            step = by_title.get(title)
            if step is not None and step.applicable and step.actionable:
                out.append((step.title, step.actionable or ""))
        return out


def research_task(name: str = DEFAULT_NAME, size_pct: float = DEFAULT_SIZE_PCT,
                  book_text: str = DEFAULT_BOOK, *, ledger: Any = None,
                  reading: Reading | None = None, asked: str = "", memory: str = "") -> Task:
    """Run every step for ``name`` at ``size_pct`` of a book described by ``book_text``, or for
    what a typed question was read as (``reading``, from :func:`read_question`).

    ``memory`` is what the trader told the console before (`lui/memory.py`, kept in their
    browser): a risk budget, a loss limit, a style, an account size. Each step applies it exactly
    as the console does and says so on a ``Remembered:`` line, so the verdict is sized on the
    trader's own budget and checked against their own mandate (audit finding 55)."""
    from argus.lui import memory as mem

    facts = mem.parse(memory)
    started = time.perf_counter()
    cash = 0.0
    if reading is not None:
        symbol, book, cash = reading.name, dict(reading.book), reading.cash
        size = max(0.01, min(reading.size_pct, 100.0)) / 100.0
        book_text = ", ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in book.items())
    else:
        symbols, _ = research_symbols(name)
        symbol = symbols[0] if symbols else f"{name.strip().upper()}USDT"
        book = dict(parse_book(book_text)) if book_text.strip() else {}
        size = max(0.01, min(size_pct, 100.0)) / 100.0
    # A held name stays in the book: the IMPACT engine reads the add as buying more of it
    # (desk.portfolio.rebalance). Dropping it here once reported a 40% NVDA holder's "add NVDA" as
    # a first position in a book without NVDA.
    question = asked.strip() or (f"I hold {book_text.strip() or 'nothing yet'} — should I add "
                                 f"{size:.0%} {symbol.removesuffix('USDT')}?")
    notional = (reading.notional if reading is not None and reading.notional
                else (DEFAULT_BOOK_VALUE * Decimal(str(size))).quantize(Decimal("1")))
    listed = contracts()
    if listed and symbol not in listed:
        # Seven engines run against a name Bitget does not list produced four refusals and one
        # false sentence (a judge's probe, 2026-09-24). One honest line is the whole answer.
        note = Step(title="Is it on Bitget?", engine="Bitget contract list",
                    lines=[f"{symbol.removesuffix('USDT')} is not listed on Bitget — none of its "
                           f"{len(listed)} contracts tracks it — so there is no price, order book "
                           f"or filing feed to research. Try a listed name: NVDA, TSLA, gold "
                           f"(XAU), BTC."], refused=True)
        return Task(question=question, name=symbol.removesuffix("USDT"), size_pct=size * 100,
                    book=book, steps=[note], seconds=time.perf_counter() - started, asked=asked,
                    reading=reading)

    def one(step: tuple[str, ResearchKind | None, str]) -> Step:
        title, kind, engine = step
        began = time.perf_counter()
        if kind is None:
            return exposure_step(title, engine, symbol, book, size, began)
        if kind is ResearchKind.IMPACT:
            request = ResearchRequest(kind=kind,
                                      symbols=(symbol, *[s for s in book if s != symbol]),
                                      book=book, size=size, size_stated=True, cash=cash)
        elif kind is ResearchKind.EXECUTION:
            request = ResearchRequest(kind=kind, symbols=(symbol,), notional=notional)
        else:
            request = ResearchRequest(kind=kind, symbols=(symbol,))
        if kind is ResearchKind.FUNDAMENTALS and not is_equity(symbol):
            # A crypto or commodity contract has no earnings, analysts or 13F filings. The console
            # answers that with a suggestion to ask something else, which is right for a question
            # and wrong as a step in a task: here the step is simply not applicable.
            return Step(title=title, engine=engine, applicable=False,
                        lines=[f"{symbol.removesuffix('USDT')} is not a company's shares, so there "
                               f"is no earnings calendar, analyst target or 13F filing to read."])
        data: dict[str, Any] = {}
        used: list[str] = []
        try:
            if facts:
                request, used = mem.apply(request, facts, question)
            answer = run(question, request, ledger=ledger if kind is ResearchKind.IMPACT else None)
            data = dict(answer.data or {})
            # Every console answer ends with the analysis-not-advice line; the page says it once.
            lines = [line.replace(_DISCLAIMER, "").rstrip() for line in answer.lines]
            extra = [*used, *mem.after(lines, request, facts)] if facts else []
            if extra:
                at = next((i for i, line in enumerate(lines) if line.startswith("Data:")),
                          len(lines))
                lines[at:at] = extra
            refused = answer.refused
        except Exception as exc:  # one engine failing must not take the task down
            lines, refused = [f"This step could not run just now ({type(exc).__name__})."], True
        return Step(title=title, engine=engine, lines=lines, refused=refused,
                    seconds=time.perf_counter() - began,
                    data=data if not refused else {})

    # Not a `with` block: leaving one waits for every thread, which is exactly the hang the
    # deadline exists to cut. A step still running is abandoned; its thread ends on its own.
    pool = ThreadPoolExecutor(max_workers=len(STEPS) + 1)
    try:
        path = pool.submit(_price_path, symbol)
        futures = [pool.submit(one, step) for step in STEPS]
        pending: list[Future[Any]] = [*futures, path]
        done, _ = wait(pending, timeout=STEP_DEADLINE_S)
        steps = [future.result() if future in done else Step(
                     title=title, engine=engine, refused=True, seconds=STEP_DEADLINE_S,
                     lines=[f"This step did not answer within {STEP_DEADLINE_S:.0f} seconds, so "
                            f"the task went on without it."])
                 for (title, _, engine), future in zip(STEPS, futures, strict=True)]
        try:
            steps[0].path = tuple(path.result()) if path in done else ()
        except Exception:
            steps[0].path = ()  # the quote step stands without its chart
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return Task(question=question, name=symbol.removesuffix("USDT"), size_pct=size * 100,
                book=book, steps=steps, seconds=time.perf_counter() - started, asked=asked,
                reading=reading)


@dataclass(frozen=True)
class Verdict:
    """One call and the reasons for it, every figure taken from an engine's data."""

    call: str
    lines: tuple[str, ...]


def _t(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def verdict(task: Task) -> Verdict | None:
    """Add, add smaller, or do not add — from the IMPACT engine's sizing figures
    (``research.impact_sizing``) and the execution engine's plan. None when the IMPACT step did
    not answer: a verdict without its sizing would be a sentence with nothing under it."""
    impact = next((s for s in task.steps if s.title == IMPACT_TITLE), None)
    sizing = (impact.data.get("sizing") if impact is not None else None) or None
    if not sizing:
        return None
    name, budget = _t(str(sizing["symbol"])), float(sizing["budget"])
    proposed = float(sizing["proposed"])
    share, ceiling = sizing.get("share_after"), sizing.get("ceiling")
    lines: list[str] = []
    if sizing.get("standalone"):
        worst = sizing.get("worst_24h_pct")
        call = "Size it by its worst day"
        lines.append(f"No book was given, so there is no share of risk to size {name} against"
                     + (f": size it so a {worst:+.1f}% day, its worst 24 hours in the "
                        f"hourly data the engine read, is a loss you accept."
                        if worst is not None else "."))
    elif sizing.get("alone"):
        call = "Add"
        lines.append(f"With the rest in cash, {name} is all of this book's market risk at any "
                     f"size; what the {proposed:.0%} changes is how much of the book is exposed.")
    elif ceiling is not None and proposed <= float(ceiling) + 1e-9:
        call = f"Add, at {proposed:.0%}"
        lines.append(f"At {proposed:.0%}, {name} would carry {share:.0%} of the book's risk, "
                     f"inside the {budget:.0%} budget; {float(ceiling):.0%} is the most that stays "
                     f"inside.")
    elif ceiling is not None:
        call = f"Add smaller: at most {float(ceiling):.0%}"
        lines.append(f"At the {proposed:.0%} asked, {name} would carry {share:.0%} of the book's "
                     f"risk, over the {budget:.0%} budget; {float(ceiling):.0%} is the most that "
                     f"stays inside.")
    else:
        call = "Do not add"
        trim = sizing.get("trim_to")
        if sizing.get("held"):
            before = sizing.get("share_before")
            lines.append(f"{name} already carries "
                         + (f"{before:.0%} of the book's risk" if before is not None
                            else "the book's risk")
                         + f" at {float(sizing['held']):.0%}, over the {budget:.0%} budget, so "
                           f"any add takes it further over"
                         + (f"; trimming it to {float(trim):.0%} brings it inside." if trim
                            else "."))
        else:
            lines.append(f"Even a 1% position in {name} would carry more than {budget:.0%} of "
                         f"this book's risk.")
    crowded = sizing.get("crowded") or {}
    if (crowded.get("trim_to") is not None and crowded.get("symbol") != sizing["symbol"]
            and float(crowded.get("share") or 0) > budget):
        held = _t(str(crowded["symbol"]))
        lines.append(f"The book's concentration is {held}, not {name}: after the trade {held} "
                     f"carries {float(crowded['share']):.0%} of the risk at "
                     f"{float(crowded['weight']):.0%} of the money. If that is the worry, trimming "
                     f"{held} to {float(crowded['trim_to']):.0%} brings it inside the "
                     f"{budget:.0%} budget.")
    execution = next((s for s in task.steps if s.title == EXECUTION_TITLE), None)
    plan = (execution.data.get("execution") if execution is not None else None) or {}
    slices = plan.get("slices") or []
    notional = ((execution.data.get("request") or {}).get("notional")
                if execution is not None else None)
    if slices and call != "Do not add" and not sizing.get("standalone"):
        parts = [f"{float(s['fraction']):.0%} {s['style']}" for s in slices]
        # The live book's figure, the one the execution step states; the plan's own estimate
        # prices impact at a default volatility and read 12.8 bps beside a book that said 8.4
        # (2026-09-27, `eval/research_task_record.py`). The estimate stays the fallback.
        sliced, single = plan.get("sliced_bps"), plan.get("single_order_bps")
        cost = sliced if sliced is not None else plan.get("expected_cost_bps")
        where = " on the live book" if sliced is not None else ""
        versus = (f" (one market order: {float(single):.1f} bps)"
                  if sliced is not None and single is not None and len(parts) > 1 else "")
        lines.append("Fill it as " + (" then ".join(parts) if len(parts) > 1 else parts[0])
                     + (f" for ${float(notional):,.0f}" if notional else "")
                     + (f", about {float(cost):.1f} bps all in{where}{versus}"
                        if cost is not None else "")
                     + (f"; at the {float(ceiling):.0%} ceiling that order is "
                        f"${float(notional) * float(ceiling) / proposed:,.0f}."
                        if notional and ceiling is not None and proposed > float(ceiling) + 1e-9
                        else "."))
    if call.startswith("Add, at") and ceiling is not None and proposed < float(ceiling) - 1e-9:
        lines.append(_invalidation(name, proposed, float(ceiling)))
    return Verdict(call=call, lines=tuple(lines))


def _invalidation(name: str, weight: float, ceiling: float) -> str:
    """When the call stops holding. A weight w drifts to ``w(1+r) / (w(1+r) + 1 - w)`` when the
    name outruns the rest of the book by r, so it reaches the ceiling c at
    ``1 + r = c(1 - w) / (w(1 - c))``. Exact for the drift, and only for today's correlations:
    the ceiling itself moves when they do, which the line says."""
    outrun = ceiling * (1 - weight) / (weight * (1 - ceiling)) - 1
    return (f"What would change the call: if {name} outruns the rest of the book by "
            f"{outrun:.0%}, it drifts past {ceiling:.0%} of the book and back over the budget; "
            f"trim it to {weight:.0%} then. A jump in its correlation with the book lowers that "
            f"ceiling, so re-ask after a large move.")


def unread_task(asked: str, reason: str) -> Task:
    """The page for a question that could not be read: the reason, and the form to try again."""
    return Task(question=asked, name="", size_pct=0.0, book={}, seconds=0.0, asked=asked,
                steps=[Step(title="What I could not read", engine="the question reader",
                            lines=[reason], refused=True)])


def exposure_step(title: str, engine: str, symbol: str, book: dict[str, float], size: float,
                  began: float) -> Step:
    """The book's sectors and factor loadings before and after the add, by the engine the console
    answers "what are my exposures" with. The add is read as the IMPACT engine reads it
    (``desk.portfolio.rebalance``): a held name ends at its scaled weight plus the add."""
    if not book:
        return Step(title=title, engine=engine, applicable=False, seconds=0.0,
                    lines=["No book was given, so there is no sector or factor mix to move; say "
                           "what you hold to see one."])
    from argus.lui.exposures import exposures_answer

    final = book.get(symbol, 0.0) * (1.0 - size) + size
    try:
        lines, _, data = exposures_answer(book, {symbol: final})
        refused = False
    except Exception as exc:  # one engine failing must not take the task down
        lines = [f"This step could not run just now ({type(exc).__name__})."]
        data, refused = {}, True
    return Step(title=title, engine=engine, lines=lines, refused=refused,
                seconds=time.perf_counter() - began, data=data if not refused else {})


def _price_path(symbol: str) -> list[tuple[Any, float]]:
    """Thirty days of 4-hour closes from Bitget, oldest first."""
    from argus.market.history import CandleType, fetch

    bars = fetch(symbol, interval="4H", candle_type=CandleType.MARKET, recent=True, limit=180)
    return [(b.ts, float(b.close)) for b in bars]


def headline(blob: dict[str, Any]) -> dict[str, float]:
    """The figures the verdict rests on, rounded as the page rounds them: sizing in whole
    percentages, fill costs in basis points to one decimal, the run time in seconds."""
    sizing = blob["data"][IMPACT_TITLE]["sizing"]
    execution = blob["data"][EXECUTION_TITLE]["execution"]
    crowded = sizing.get("crowded") or {}

    def pct(value: Any) -> float:
        return float(round(float(value) * 100))

    return {
        "proposed": pct(sizing["proposed"]),
        "ceiling": pct(sizing["ceiling"]),
        "budget": pct(sizing["budget"]),
        "share_after": pct(sizing["share_after"]),
        "crowded_share": pct(crowded["share"]),
        "crowded_weight": pct(crowded["weight"]),
        "crowded_trim_to": pct(crowded["trim_to"]),
        "seconds": round(float(blob["task"]["seconds"]), 1),
        "sliced_bps": round(float(execution["sliced_bps"]), 1),
        "single_order_bps": round(float(execution["single_order_bps"]), 1),
    }


def as_dict(task: Task) -> dict[str, Any]:
    return {
        "question": task.question, "name": task.name, "size_pct": task.size_pct,
        "book": task.book, "seconds": round(task.seconds, 2),
        "read_as": None if task.reading is None else {
            "summary": task.reading.summary, "notes": list(task.reading.notes),
            "cash": task.reading.cash},
        "verdict": None if task.verdict is None else {
            "call": task.verdict.call, "lines": list(task.verdict.lines)},
        "conclusion": [{"step": t, "actionable": a} for t, a in task.conclusion],
        "steps": [{"title": s.title, "engine": s.engine, "lines": s.lines,
                   "refused": s.refused, "seconds": round(s.seconds, 2)} for s in task.steps],
    }


__all__ = ["DEFAULT_BOOK", "DEFAULT_NAME", "DEFAULT_SIZE_PCT", "UNSTATED_SIZE_PCT", "Reading",
           "Step", "Task", "as_dict", "read_question", "research_task",
           "unread_task"]
