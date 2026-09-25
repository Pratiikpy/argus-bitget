"""One research question, read by several independent researchers and merged with their citations.

**What a single pass leaves out.** The console answers a research question with the one engine its
kind names (`lui/research.py`): "what's the news on NVDA" is read by the news engine, and that is
the answer. The same trader's honest next reads — where NVDA stands technically, what its company
reports next and what the analysts expect, how it reacts to a CPI print, what its own history did
from a state like this — are each one engine away and never run. `lui/multistep.py` splits a
question that *says* several things; this module is for a question that says one thing whose
honest answer needs several independent reads, each on its own sources.

**Taken, from open_deep_research** (langchain-ai/open_deep_research, MIT,
`src/open_deep_research/deep_researcher.py`):

* **Supervisor and isolated researcher subgraphs.** The supervisor delegates research units
  (``ConductResearch``, ``:178-223``) and runs them in parallel, each against its own compiled
  researcher subgraph with a fresh message list (``:294-305``; the subgraph is built at
  ``:589-605`` over ``ResearcherState``, `state.py:79-89`). Only the compressed result crosses back
  (``ResearcherOutputState``, `state.py:92-96`). Here each unit is its own call to the engine that
  owns its kind, on its own topic text — never the parent question, so no engine reads another
  engine's cue — with its own sources and its own coverage record (`truth/coverage.py`, opened per
  call by ``research.run``). Nothing crosses back but the unit's compressed finding and its sources.
* **A cap on concurrent units, with the overflow named.** ``max_concurrent_research_units``
  (default 5, `configuration.py:64-69`) and an explicit message for every unit past it
  (``:291-321``). Here :data:`MAX_UNITS`, and a unit past it is listed in the answer's data as not
  run, never dropped silently.
* **Per-tool failure as a result, not a crash** (``execute_tool_safely``, ``:427-432``).
* **Compress each researcher before merging** (``compress_research``, ``:511-585``), under the
  rule its prompt states twice: repeat the findings *verbatim* and number the citations
  sequentially (`prompts.py:186-222`).

**Changed, each deliberately.**

* **The supervisor is a plan table by default, not a model loop.** Upstream's supervisor is a
  model that reflects (``think_tool``) and delegates until it calls ``ResearchComplete`` or runs
  out of iterations (``:247-262``). Here the units for each kind are a fixed, auditable table
  (:data:`PLAN`), filtered by what each engine can answer for the name (a crypto contract has no
  earnings, and only the twelve traded names have an event study). A model may choose instead
  (:func:`plan_units_with_model`), from a closed menu of unit ids: an id outside the menu is
  rejected and counted, and the model chooses *which* engines run, never a figure — the boundary
  `research.plan_with_model` already keeps. Whether its choice beats the table is measured, not
  assumed (`eval/research_depth.py`).
* **Compression is verbatim selection, not a model rewrite.** Upstream asks a cheaper model to
  "repeat key information verbatim" (`prompts.py:197,221`). The property it wants is verbatim, and
  selecting whole lines from the researcher's own answer delivers it with no model and no chance of
  altering a figure: a finding is the researcher's lead line and, space allowing, its next line
  that carries a figure. The evaluation checks that every number in every finding appears in that
  researcher's own answer.
* **A failed unit costs only itself.** Upstream wraps the whole ``asyncio.gather`` in one ``try``
  and ends the research phase on *any* exception (``is_token_limit_exceeded(...) or True``,
  ``:331-341``), discarding every other unit's result in the batch. Here each unit's failure,
  refusal or timeout becomes that unit's own "not answered" line with its reason, and the rest
  stand.
* **One round, no re-planning loop.** Upstream's supervisor loops up to
  ``max_researcher_iterations`` (default 6), re-planning from what came back. Re-planning needs a
  judgement about what is missing from the findings, and the only thing here that could make it is
  a model reading figures — which is the step this console does not let a model take.
* **Citations stay with their finding.** Each finding ends with the numbers of its own sources in
  the merged list, so a reader can tell which researcher's evidence supports which line.

**Measured, 2026-09-26** (`eval/research_depth.py`, ``data/research_depth_fanout.json``; 46 live
questions from the held-out routing corpora and the research bench, twelve kinds, every one read
correctly by the console's no-token path). Distinct cited sources, median: 2 for the single pass,
13 fanned out, more on 46 of 46. Researchers planned 131, answered 131. Verbatim violations 0 and
citations resolving to another researcher's source 0. The losses, stated as losses: the answer is
slower on 45 of 46 questions — median 2.1s alone, 6.2s fanned out, +4.3s added (p90 +10.3s) — and
twice as long (median 1,163 characters against 2,387). The model supervisor, run on 28 of those
questions for 28 Qwen calls, **lost to the table**: it cited fewer distinct sources on 14 and more
on 7 (median 11 against 13), chose an identical plan on 7, returned an empty plan once (the table
answered), and waited a median 4.3s for its planning call — 9.8s end to end against the table's
6.0s. The table stays the default; the model path is kept, measured, and off unless a client is
passed.

    python -m argus.lui.fanout "what's the news on NVDA today"
"""

from __future__ import annotations

import contextvars
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from typing import Any

from argus.lui.research import ResearchKind, ResearchRequest

MAX_UNITS = 4
"""Researchers per question, the primary included. Upstream's default is five; four keeps a
fan-out inside the console's thirty-second budget for a slow question (`lui/cli.BUDGET_MS`) when
the Skills are slow, and is the number the evaluation measured."""

DEFAULT_CONCURRENCY = MAX_UNITS
"""Units in flight at once: every unit, so none waits in a queue behind a slow one. The burst
Bitget's candle endpoint refuses with HTTP 429 is already throttled where the calls are made
(`lui/research._FETCH_SLOTS`, three in flight across the whole process), so a second cap here
would only serialise units that do not touch that endpoint. The first live run used three and one
unit of every four waited for a slot; the figures in ``data/research_depth_fanout.json`` are from
the run with this setting."""

DEADLINE_S = 25.0
"""Wall-clock allowed for every unit together. A unit still running at the deadline is reported as
not finished rather than holding the whole answer, inside the console's 30s slow-question budget."""

FINDING_CHARS = 420
"""A finding's length allowance. The lead line is always kept whole, however long — a cut lead is
a changed statement — and the supporting line is added only if both fit."""

Runner = Callable[[str, ResearchRequest], Any]
"""``research.run``'s shape: a topic text and a request in, an ``Answer`` out."""

_IN_UNIT: contextvars.ContextVar[bool] = contextvars.ContextVar("argus_fanout_unit",
                                                               default=False)
"""Set inside a unit's worker, so a runner that itself calls :func:`fan_out` cannot recurse."""

_LEAD = re.compile(r"^Actionable(?: \([^)]*\))?:\s*")
_SKIP = ("Data:", "Sources reached:", "Assumed:")
_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


# =============================================================================================
# The plan: which researchers a question of each kind gets
# =============================================================================================

K = ResearchKind

PLAN: dict[ResearchKind, tuple[ResearchKind, ...]] = {
    K.QUOTE: (K.TECHNICALS, K.NEWS, K.FUNDAMENTALS, K.SENTIMENT),
    K.TECHNICALS: (K.NEWS, K.ANALOGUE, K.FUNDAMENTALS, K.SENTIMENT),
    K.NEWS: (K.TECHNICALS, K.FUNDAMENTALS, K.SENTIMENT, K.ANALOGUE),
    K.FUNDAMENTALS: (K.NEWS, K.TECHNICALS, K.EVENT),
    K.SENTIMENT: (K.NEWS, K.TECHNICALS, K.MACRO),
    K.EVENT: (K.FUNDAMENTALS, K.NEWS, K.TECHNICALS),
    K.ANALOGUE: (K.TECHNICALS, K.NEWS, K.SENTIMENT, K.FUNDAMENTALS),
    K.MACRO: (K.SENTIMENT, K.NEWS, K.TECHNICALS),
    K.LEVERAGE: (K.TECHNICALS, K.SENTIMENT, K.NEWS),
    K.EXECUTION: (K.QUOTE, K.TECHNICALS, K.NEWS),
    K.IMPACT: (K.TECHNICALS, K.NEWS, K.FUNDAMENTALS, K.SENTIMENT),
    K.COMPARE: (K.TECHNICALS, K.NEWS),
    K.STRESS: (K.HEDGE, K.MACRO),
    K.HEDGE: (K.STRESS, K.MACRO),
    K.BOOK: (K.STRESS, K.HEDGE, K.MACRO),
}
"""For each kind, the other researchers in the order they are added, each a different engine on
different sources: the name's price structure, what is being said about it, the company behind it,
the crowd's positioning, its own history. The lists run past :data:`MAX_UNITS` on purpose — a unit
the name cannot use is skipped and the next one takes its place. Kinds absent here (``venue``,
``construct``) are answered alone."""

BOOK_KINDS = frozenset({K.STRESS, K.HEDGE, K.BOOK})
"""Researchers that read the trader's book rather than one name."""

LABELS: dict[ResearchKind, str] = {
    K.TECHNICALS: "Technicals", K.NEWS: "News", K.FUNDAMENTALS: "Company",
    K.SENTIMENT: "Positioning", K.EVENT: "Event reactions", K.ANALOGUE: "Own history",
    K.MACRO: "Macro backdrop", K.QUOTE: "Market", K.STRESS: "Stress", K.HEDGE: "Hedge",
    K.BOOK: "Book risk", K.IMPACT: "Impact", K.COMPARE: "Comparison",
    K.EXECUTION: "Execution", K.LEVERAGE: "Leverage", K.VENUE: "Venue",
    K.CONSTRUCT: "Construction",
}

TOPICS: dict[ResearchKind, str] = {
    K.TECHNICALS: "{name} technicals: RSI, MACD, support and resistance",
    K.NEWS: "what is the news on {name} today",
    K.FUNDAMENTALS: "{name} fundamentals: next earnings, analyst consensus and holders",
    K.SENTIMENT: "sentiment and positioning on {name}",
    K.EVENT: "how does {name} react to CPI releases and Fed decisions",
    K.ANALOGUE: "has {name} been here before, and what followed",
    K.MACRO: "how do rates and the dollar affect {name}",
    K.QUOTE: "{name} price and round-trip cost",
    K.STRESS: "what if the Nasdaq drops 10%, for my book",
    K.HEDGE: "how should I hedge my book",
    K.BOOK: "how risky is my book",
}
"""Each researcher's own topic — upstream's ``research_topic``. A unit never reads the parent
question: an engine that keys a detail on the wording (the event types, a fundamentals focus)
must read its own topic, or one researcher's cue leaks into another's answer."""


def asset_class(symbol: str) -> str:
    """``equity``, ``fund``, ``commodity``, ``fx``, ``index`` or ``crypto``: which engines can
    answer for a name.

    A fund is read from the Instrument Master (`market/instruments.py`: QQQ as an index ETF, TQQQ
    and SQQQ as leveraged ones, each verified against its issuer's fund page), and has no earnings,
    analyst consensus or 13F holders of its own — the first live run of `eval/research_depth.py`
    (2026-09-25) showed QQQ's company researcher returning only "the data source holds no
    institutional holders figure for QQQ", a finding that says nothing. Otherwise the twelve traded
    names are equities by definition and the rest are read from the venue's own registry
    (`market/universe.py`), which falls back to a dated snapshot when Bitget does not answer."""
    from argus.lui.question import TRADED_SYMBOLS
    from argus.market import universe
    from argus.market.instruments import REGISTRY, InstrumentKind

    identity = REGISTRY.get(symbol)
    if identity is not None and identity.kind in (InstrumentKind.INDEX_ETF,
                                                  InstrumentKind.LEVERAGED_ETF):
        return "fund"
    if symbol in TRADED_SYMBOLS:
        return "equity"
    other = universe.NOT_EQUITY.get(symbol)
    if other is not None:
        return other
    return "equity" if universe.is_equity(symbol) else "crypto"


def _eligible(kind: ResearchKind, symbol: str, has_book: bool,
              classify: Callable[[str], str]) -> bool:
    """Whether ``kind``'s engine can answer for ``symbol`` at all, read from the engines' own
    refusals in `lui/research.py`: no earnings, consensus or 13F for a non-equity
    (``_fundamentals``); an event study only for the twelve names the desk trades
    (``_event_reaction``); the crowd read is crypto's fear and greed index and funding
    (``_sentiment``); a book researcher needs a book."""
    from argus.lui.question import TRADED_SYMBOLS

    if kind in BOOK_KINDS:
        return has_book
    if kind is K.FUNDAMENTALS:
        return classify(symbol) == "equity"
    if kind is K.EVENT:
        return symbol in TRADED_SYMBOLS
    if kind is K.SENTIMENT:
        return classify(symbol) == "crypto"
    return kind in TOPICS


def _ticker(symbol: str) -> str:
    base = symbol.removesuffix("USDT")
    return base.removesuffix("STOCK") if base.endswith("STOCK") and len(base) > 5 else base


@dataclass(frozen=True)
class Unit:
    """One researcher: the engine it runs, the name it reads, and its own topic and request."""

    kind: ResearchKind
    subject: str
    """The symbol the unit reads, or ``book`` for a researcher of the whole book."""

    topic: str
    request: ResearchRequest
    primary: bool = False

    @property
    def id(self) -> str:
        return f"{self.kind.value}:{self.subject}"

    @property
    def label(self) -> str:
        name = "your book" if self.subject == "book" else _ticker(self.subject)
        return f"{LABELS.get(self.kind, self.kind.value)} ({name})"


def _unit(kind: ResearchKind, symbol: str, request: ResearchRequest) -> Unit:
    if kind in BOOK_KINDS or (kind is K.MACRO and symbol == "book"):
        book = dict(request.book)
        sub = ResearchRequest(kind=kind, symbols=tuple(book), book=book, cash=request.cash,
                              parsed_by="fanout")
        topic = TOPICS[kind] if kind in BOOK_KINDS else "how do rates and the dollar affect my book"
        return Unit(kind=kind, subject="book", topic=topic, request=sub)
    sub = ResearchRequest(kind=kind, symbols=(symbol,), parsed_by="fanout")
    return Unit(kind=kind, subject=symbol, topic=TOPICS[kind].format(name=_ticker(symbol)),
                request=sub)


def _subjects(request: ResearchRequest) -> list[str]:
    """The names researchers are pointed at: the candidate for an add, both names of a
    comparison, the name asked about otherwise."""
    if request.kind is K.COMPARE:
        return list(request.symbols[:2])
    return list(request.symbols[:1])


def menu(request: ResearchRequest, *,
         classify: Callable[[str], str] = asset_class) -> list[Unit]:
    """Every researcher that could answer something about this question, besides its own engine:
    the closed set a model supervisor chooses from."""
    has_book = bool(request.book)
    out: list[Unit] = []
    seen: set[str] = set()
    for symbol in _subjects(request):
        for kind in TOPICS:
            if kind in BOOK_KINDS or kind is request.kind:
                continue
            if _eligible(kind, symbol, has_book, classify):
                unit = _unit(kind, symbol, request)
                if unit.id not in seen:
                    seen.add(unit.id)
                    out.append(unit)
    if has_book:
        for kind in (K.STRESS, K.HEDGE, K.BOOK, K.MACRO):
            if kind is request.kind:
                continue
            unit = _unit(kind, "book", request)
            if unit.id not in seen:
                seen.add(unit.id)
                out.append(unit)
    return out


def plan_units(request: ResearchRequest, *, max_units: int = MAX_UNITS,
               classify: Callable[[str], str] = asset_class) -> list[Unit]:
    """The table's plan: up to ``max_units - 1`` researchers beside the question's own engine.

    Empty when the kind is not in :data:`PLAN` or nothing in its row can answer for the name — the
    question is then answered alone, as it always was."""
    row = PLAN.get(request.kind)
    if row is None or max_units < 2:
        return []
    subjects = _subjects(request)
    has_book = bool(request.book)
    if not subjects and not has_book:
        return []
    out: list[Unit] = []
    seen: set[str] = set()
    for kind in row:
        targets = (["book"] if kind in BOOK_KINDS or (kind is K.MACRO and has_book
                                                       and request.kind in BOOK_KINDS)
                   else subjects)
        for symbol in targets:
            if symbol != "book" and not _eligible(kind, symbol, has_book, classify):
                continue
            if symbol == "book" and not has_book:
                continue
            unit = _unit(kind, symbol, request)
            if unit.id in seen:
                continue
            seen.add(unit.id)
            out.append(unit)
            if len(out) >= max_units - 1:
                return out
    return out


def plan_units_with_model(
    text: str, request: ResearchRequest, client: Any, *, max_units: int = MAX_UNITS,
    classify: Callable[[str], str] = asset_class,
) -> tuple[list[Unit], dict[str, Any]]:
    """A model supervisor's plan, from the closed menu, validated; the table's plan on any failure.

    The model sees the question and the menu of unit ids and returns up to ``max_units - 1`` of
    them. It is never shown a figure and cannot put one anywhere: an id outside the menu is
    rejected and counted, a repeat is dropped, and ids past the cap are recorded as overflow — the
    message upstream returns for units past its limit (`deep_researcher.py:316-321`). A call that
    fails, or a reply with no valid id, falls back to :func:`plan_units` and says so in the audit.
    """
    options = menu(request, classify=classify)
    audit: dict[str, Any] = {"planner": "model", "menu": [u.id for u in options]}
    if not options:
        return [], {**audit, "detail": "nothing on the menu for this question"}
    by_id = {u.id: u for u in options}
    listing = "\n".join(f"- {u.id}: {u.label} — {u.topic}" for u in options)
    messages = [
        {"role": "system", "content": (
            "You are the supervisor of a trading research desk. Choose which independent "
            "researchers to run for the trader's question, from the menu only. Each researcher "
            "reads its own data and returns a finding; you never see or state any figure. Pick "
            "the ones whose findings would most change what the trader does. Return JSON only: "
            '{"units": [{"id": "<menu id>", "why": "<one short clause>"}]}')},
        {"role": "user", "content": (
            f"Question: {text}\nIts own engine already answers it as: {request.kind.value}.\n"
            f"Choose at most {max_units - 1} researchers from this menu:\n{listing}")},
    ]
    try:
        reply = client.complete_json(messages, required_keys=("units",), max_tokens=400,
                                     attempts=1)
    except Exception as exc:
        fallback = plan_units(request, max_units=max_units, classify=classify)
        return fallback, {**audit, "planner": "table (model failed)",
                          "detail": f"{type(exc).__name__}: {exc}"}
    raw = reply.get("units") if isinstance(reply, dict) else None
    chosen: list[Unit] = []
    rejected: list[str] = []
    overflow: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        uid = str(item.get("id", "")) if isinstance(item, dict) else str(item)
        unit = by_id.get(uid)
        if unit is None:
            rejected.append(uid)
        elif unit in chosen:
            continue
        elif len(chosen) >= max_units - 1:
            overflow.append(uid)
        else:
            chosen.append(unit)
    audit.update(chosen=[u.id for u in chosen], rejected=rejected, overflow=overflow,
                 why={str(i.get("id")): str(i.get("why", ""))[:160]
                      for i in (raw if isinstance(raw, list) else []) if isinstance(i, dict)})
    if not chosen:
        fallback = plan_units(request, max_units=max_units, classify=classify)
        return fallback, {**audit, "planner": "table (model chose nothing valid)"}
    return chosen, audit


# =============================================================================================
# Running, compressing, merging
# =============================================================================================


@dataclass
class Finding:
    """One researcher's compressed result, with the sources that support it."""

    unit: Unit
    answered: bool
    lines: list[str] = field(default_factory=list)
    """Verbatim lines from the researcher's own answer: its lead, and its next figure line."""

    sources: list[Any] = field(default_factory=list)
    reason: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    full: list[str] = field(default_factory=list)
    """Every line the researcher returned, kept so the verbatim property can be checked."""


def compress(answer: Any) -> tuple[list[str], str]:
    """The lead line and, if both fit :data:`FINDING_CHARS`, the next line carrying a figure.

    Lines are selected whole and never edited except for dropping the ``Actionable:`` prefix, which
    marks the lead of an answer and would claim the lead of the merged one. Bookkeeping lines
    (``Data:``, ``Sources reached:``, ``Assumed:``) are left to the merged answer's own.
    Returns the lines and a reason when there is nothing to keep."""
    body = [str(line) for line in getattr(answer, "lines", []) or []
            if str(line).strip() and not str(line).startswith(_SKIP)]
    if not body:
        return [], "the researcher returned no finding"
    lead_at = next((i for i, line in enumerate(body) if _LEAD.match(line)), 0)
    lead = _LEAD.sub("", body[lead_at])
    lead = lead[:1].upper() + lead[1:]  # "momentum turning down" once its prefix is gone
    kept = [lead]
    for line in body[lead_at + 1:] + body[:lead_at]:
        if line != body[lead_at] and re.search(r"\d", line):
            if len(lead) + len(line) <= FINDING_CHARS:
                kept.append(_LEAD.sub("", line))
            break
    return kept, ""


def _source_key(source: Any) -> tuple[str, str, str]:
    return (str(getattr(source, "kind", "")), str(getattr(source, "ref", source)),
            str(getattr(source, "detail", "")))


def _worker(run: Runner, unit: Unit) -> tuple[Any, float]:
    token = _IN_UNIT.set(True)
    started = time.perf_counter()
    try:
        return run(unit.topic, unit.request), (time.perf_counter() - started) * 1000
    finally:
        _IN_UNIT.reset(token)


@dataclass
class FanOut:
    """What :func:`fan_out` ran and found, before it is rendered into one answer."""

    primary: Finding
    findings: list[Finding]
    planner: str
    audit: dict[str, Any]
    elapsed_ms: float
    primary_answer: Any = None


def run_units(text: str, request: ResearchRequest, units: Sequence[Unit], run: Runner, *,
              concurrency: int = DEFAULT_CONCURRENCY, deadline_s: float = DEADLINE_S,
              planner: str = "table", audit: Mapping[str, Any] | None = None) -> FanOut:
    """Run the question's own engine and every unit side by side, each in isolation.

    A unit that raises, refuses or is still running at ``deadline_s`` becomes a finding that says
    so; its siblings are unaffected. The pool is left without waiting for a unit past the deadline
    (the same shutdown the console's candle fetch uses), so one slow Skill cannot hold the answer.
    """
    from argus.truth.coverage import ContextPool

    started = time.perf_counter()
    primary = Unit(kind=request.kind, subject=(request.symbols[0] if request.symbols else "book"),
                   topic=text, request=request, primary=True)
    everyone = [primary, *units]
    pool = ContextPool(max_workers=max(1, min(concurrency, len(everyone))))
    futures: dict[Future[tuple[Any, float]], int] = {}
    try:
        for i, unit in enumerate(everyone):
            futures[pool.submit(_worker, run, unit)] = i
        wait(futures, timeout=deadline_s)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    results: list[Finding] = []
    primary_answer: Any = None
    for future, i in sorted(futures.items(), key=lambda item: item[1]):
        unit = everyone[i]
        finding = Finding(unit=unit, answered=False)
        if not future.done() or future.cancelled():
            # still running, or never started because the deadline cancelled it in the queue
            finding.reason = f"did not finish within {deadline_s:g}s"
        elif future.exception() is not None:
            exc = future.exception()
            finding.reason = f"{type(exc).__name__}: {exc}"
        else:
            answer, elapsed = future.result()
            finding.elapsed_ms = elapsed
            finding.full = [str(line) for line in getattr(answer, "lines", []) or []]
            finding.coverage = dict((getattr(answer, "data", {}) or {}).get("coverage") or {})
            if unit.primary:
                primary_answer = answer
            if getattr(answer, "refused", False):
                finding.reason = str(getattr(answer, "reason", "")) or "the engine refused"
            else:
                finding.lines, why = compress(answer)
                finding.answered = bool(finding.lines)
                finding.reason = why
                finding.sources = list(getattr(answer, "sources", []) or [])
        results.append(finding)
    return FanOut(primary=results[0], findings=results[1:], planner=planner,
                  audit=dict(audit or {}), elapsed_ms=(time.perf_counter() - started) * 1000,
                  primary_answer=primary_answer)


def merge(result: FanOut, *, overflow: Sequence[str] = ()) -> Any:
    """One answer: the question's own answer in full, then each researcher's finding with the
    numbers of its own sources, then what did not answer, then one merged coverage line.

    The question's own engine keeps the lead — it answers what was asked — and its ``Data:`` line
    stays last, as every answer here ends. Returns the primary's answer unchanged (with the fan-out
    recorded in its data) when the question itself was refused: extra findings beside "I need your
    holdings" would answer questions the trader has not been able to ask yet.
    """
    from argus.lui.answer import Answer

    base = result.primary_answer
    data: dict[str, Any] = {"planner": result.planner, "elapsed_ms": round(result.elapsed_ms, 1),
                            "audit": result.audit, "overflow": list(overflow)}
    if base is None:
        return None
    if getattr(base, "refused", False) or not result.findings:
        base.data["fanout"] = {**data, "merged": False,
                               "why": "the question itself was not answered" if getattr(
                                   base, "refused", False) else "no researcher was planned"}
        return base
    lines = [str(line) for line in base.lines]
    data_lines = [line for line in lines if line.startswith("Data:")]
    body = [line for line in lines if not line.startswith("Data:")]
    sources = list(base.sources)
    index = {_source_key(s): i + 1 for i, s in enumerate(sources)}
    answered = [f for f in result.findings if f.answered]
    missed = [f for f in result.findings if not f.answered]
    block = [f"Wider read: {len(result.findings)} independent researcher"
             f"{'s' if len(result.findings) != 1 else ''}, each on its own sources — "
             f"{len(answered)} answered" + (f", {len(missed)} did not" if missed else "") + ":"]
    records: list[dict[str, Any]] = []
    for finding in result.findings:
        cited: list[int] = []
        for source in finding.sources:
            key = _source_key(source)
            if key not in index:
                sources.append(source)
                index[key] = len(sources)
            if index[key] not in cited:
                cited.append(index[key])
        marks = "".join(f"[{n}]" for n in cited)
        if finding.answered:
            block.append(f"{finding.unit.label}: {' '.join(finding.lines)}"
                         + (f" {marks}" if marks else ""))
        records.append({"unit": finding.unit.id, "answered": finding.answered,
                        "lines": finding.lines, "sources": cited, "reason": finding.reason,
                        "coverage": finding.coverage, "elapsed_ms": round(finding.elapsed_ms, 1)})
    block.extend(f"Not answered — {f.unit.label}: {f.reason}." for f in missed)
    reached: set[str] = set(result.primary.coverage.get("reached", []))
    silent: set[str] = set(result.primary.coverage.get("did_not_answer", {}))
    for finding in result.findings:
        reached |= set(finding.coverage.get("reached", []))
        silent |= set(finding.coverage.get("did_not_answer", {}))
    silent -= reached
    if reached or silent:
        # The merged line supersedes the question's own: two "Sources reached" lines with two
        # different totals read as a contradiction (found reading the first saved answers,
        # 2026-09-26, where "11 of 11" sat above "13 of 22").
        body = [line for line in body if not line.startswith("Sources reached:")]
        total = len(reached) + len(silent)
        block.append(f"Sources reached across all researchers: {len(reached)} of {total} "
                     f"answered" + (f"; did not answer: {', '.join(sorted(silent))}"
                                    if silent else "") + ".")
    data.update(merged=True, findings=records, reached=sorted(reached),
                did_not_answer=sorted(silent))
    merged = Answer(question=base.question, lines=[*body, *block, *data_lines],
                    sources=sources, refused=False, reason="", data={**base.data, "fanout": data})
    return merged


def fan_out(
    text: str,
    request: ResearchRequest,
    *,
    run: Runner,
    client: Any = None,
    max_units: int = MAX_UNITS,
    concurrency: int = DEFAULT_CONCURRENCY,
    deadline_s: float = DEADLINE_S,
    classify: Callable[[str], str] = asset_class,
) -> Any:
    """The console's call: the question's answer widened by independent researchers, or ``None``.

    ``run`` answers one topic with one request — ``research.run`` bound to the ledger. ``client``,
    when given, is a model supervisor that picks the researchers from the menu
    (:func:`plan_units_with_model`); without one the table picks them (:func:`plan_units`).
    Returns ``None`` when nothing would be added — a kind with no plan, a name no other engine can
    answer for, or a call made from inside a unit — so the caller answers as it always did.
    """
    if _IN_UNIT.get():
        return None
    if client is not None:
        units, audit = plan_units_with_model(text, request, client, max_units=max_units,
                                             classify=classify)
        planner = str(audit.get("planner", "model"))
    else:
        units = plan_units(request, max_units=max_units, classify=classify)
        audit, planner = {"planner": "table"}, "table"
    if not units:
        return None
    result = run_units(text, request, units, run, concurrency=concurrency,
                       deadline_s=deadline_s, planner=planner, audit=audit)
    return merge(result, overflow=audit.get("overflow", []))


def figures(text: str) -> list[str]:
    """Every number written in ``text``, as written — the unit of the verbatim check."""
    return [m.group(0).lstrip("+").lstrip("$").replace(",", "") for m in _NUMBER.finditer(text)]


def verbatim_violations(finding: Finding) -> list[str]:
    """Numbers in a finding that its researcher's own answer does not contain. Always empty unless
    compression altered something; the evaluation counts it on every live finding."""
    own = set()
    for line in finding.full:
        own.update(figures(line))
    return [n for line in finding.lines for n in figures(line) if n not in own]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI over live data
    import argparse

    from argus.lui import research
    from argus.lui.kindmodel import LocalPlanner, kind_model

    parser = argparse.ArgumentParser(description="ARGUS research fan-out")
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    local = kind_model()
    planned = (research.plan_with_model(args.question, LocalPlanner(local))[0]
               if local is not None else None)
    patterned = research.detect(args.question)
    request = (patterned if patterned is not None and (
        planned is None or research.pattern_reading_wins(patterned, args.question)) else planned)
    if request is None:
        print("not a research question the console reads")
        return 1
    answer = fan_out(args.question, request, run=research.run)
    if answer is None:
        answer = research.run(args.question, request)
        print("(answered alone: no researcher to add)")
    for line in answer.lines:
        print(line)
    for n, source in enumerate(answer.sources, start=1):
        print(f"[{n}] {source}")
    if args.json:
        print(json.dumps(answer.data.get("fanout", {}), indent=2, default=str))
    return 0


__all__ = [
    "DEADLINE_S",
    "DEFAULT_CONCURRENCY",
    "MAX_UNITS",
    "PLAN",
    "FanOut",
    "Finding",
    "Unit",
    "asset_class",
    "compress",
    "fan_out",
    "figures",
    "menu",
    "merge",
    "plan_units",
    "plan_units_with_model",
    "run_units",
    "verbatim_violations",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
