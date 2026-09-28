"""The console in a terminal — ask ARGUS a research question or about its own record, in words.

Run it with ``python -m argus.lui`` for an interactive session, or pass a question as arguments to
ask one and exit: ``python -m argus.lui why did you do nothing all weekend``.

**What this surface is for.** It is the answer to a judge's most dangerous question, which is never
"what does it do" but "show me". Every reply carries the sources it was built from and refuses by
name when nothing can support it. The transcript of a session *is* the decision-explainability
evidence; there is nothing to prepare beforehand and nothing that can be staged.

**One console, four doors.** Until 2026-09-28 this terminal answered from the decision record
alone: "what is the NVDA price?" was told to ask the research console, and "is Deutsche Bank a good
buy" got a generic decline where the web console, MCP and Telegram name the company as unlisted
(QA surfaces pass, Activity/26_QA_SURFACES_RESULTS.md). Its own help promised research questions.
It now asks the same function they do, ``server.handle_ask``, so a question gets the same answer
whichever door it comes through; the session keeps its turns and the trader's memory the way the
browser does.

Rendering rules, from ``research/subthemes/_CENSUS-lui.md``:

- a refusal is shown as a refusal, with the reason and what to ask instead;
- sources are printed under every answer, never folded into the prose;
- the latency budget for the question class is shown next to the time actually taken, so a slow
  answer is visibly slow rather than quietly slow.

**For programs as well as people** (Command Line Interface Guidelines, clig.dev,
``content/_index.md:555`` and ``:568``; research/harvest/51-cli-guidelines.md). ``--json`` prints
one JSON object per question — the answer's ``as_dict()`` with the time taken and its budget — so
an agent reads fields rather than scraping prose; Bitget's own ``bgc`` makes the same choice,
printing JSON by default (``agent-cli/src/index.ts:236-239``). ``-q`` prints the answer's lines and
nothing else, for scripts that only want the words. Exit status is the same in every mode: 0
answered, 1 refused, 2 a usage error, so a caller can branch on it without parsing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from argus.lui.answer import Answer
from argus.lui.plural import resolve_plurals
from argus.lui.question import Speed
from argus.paper.ledger import PaperLedger

MAX_TURNS = 12
"""Turns replayed to the console, as the browser keeps them (`server.handle_ask`)."""

BUDGET_MS: dict[Speed, int] = {Speed.FAST: 500, Speed.MEDIUM: 5_000, Speed.SLOW: 30_000}

BANNER = """ARGUS research console - ask a research question or about the record.

Every figure is computed from live data or the hash-chained decision ledger and cites its source.
When nothing can support an answer, you get a refusal and the reason, not a guess.

Try:  where is NVDA trading right now      is TSLA overbought
      I hold 50% NVDA, 50% AAPL - what does adding 20% TSLA do to my risk?
      why did you do nothing all weekend   show me decision 25
      is the log tamper-evident            are you well calibrated

Ctrl-D or "quit" to leave.
"""


def render(result: Answer | ConsoleAnswer, *, elapsed_ms: float) -> str:
    """Format one answer for a terminal."""
    q = result.question
    budget = BUDGET_MS[q.speed]
    over = " OVER BUDGET" if elapsed_ms > budget else ""
    head = f"[{q.intent}]  {elapsed_ms:.0f}ms / {budget}ms{over}"

    body = []
    if result.refused:
        body.append("REFUSED — " + (result.reason or "no reason recorded"))
        body.extend(f"  {line}" for line in result.lines[1:])
    else:
        body.extend(f"  {line}" for line in result.lines)

    if result.sources:
        body.append("  sources:")
        body.extend(f"    - {src}" for src in result.sources)
    elif not result.refused:
        # Reaching here is a defect: an assertion with nothing behind it.
        body.append("  WARNING: answer carries no source — this is a bug, not a style choice")

    return resolve_plurals(head + "\n" + "\n".join(body))


@dataclass(frozen=True)
class ConsoleQuestion:
    intent: str
    speed: Speed


@dataclass
class ConsoleAnswer:
    """The console's answer (a ``server.handle_ask`` payload) in the shape this terminal renders:
    a question with its intent and speed, the lines, the sources, a refusal and its reason."""

    question: ConsoleQuestion
    refused: bool
    reason: str
    lines: list[str]
    sources: list[str]
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ConsoleAnswer:
        try:
            speed = Speed(str(payload.get("speed") or "slow"))
        except ValueError:
            speed = Speed.SLOW
        sources = [" · ".join(str(s.get(k)) for k in ("kind", "ref", "detail") if s.get(k))
                   if isinstance(s, dict) else str(s) for s in payload.get("sources") or []]
        return cls(question=ConsoleQuestion(intent=str(payload.get("intent") or "unknown"),
                                            speed=speed),
                   refused=bool(payload.get("refused")),
                   reason=str(payload.get("reason") or ""),
                   lines=[str(line) for line in payload.get("lines") or []],
                   sources=sources, payload=payload)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.payload.items() if k not in ("elapsed_ms", "budget_ms")}


@dataclass
class Session:
    """What the browser keeps between questions: the turns, the saved book and the memory."""

    turns: list[str] = field(default_factory=list)
    book: str = ""
    memory: str = ""


def ask_console(text: str, session: Session, *,
                now: datetime | None = None) -> tuple[ConsoleAnswer, float]:
    """One question through the console every other door uses."""
    from argus.lui import server

    started = time.perf_counter()
    payload = server.handle_ask(text, session.turns[-MAX_TURNS:], now=now, visitor="cli",
                                book=session.book, memory=session.memory)
    session.turns = [str(t) for t in payload.get("turns") or [*session.turns, text]][-MAX_TURNS:]
    session.memory = str(payload.get("memory") or session.memory)
    return ConsoleAnswer.from_payload(payload), (time.perf_counter() - started) * 1000


def as_json(result: Answer | ConsoleAnswer, *, elapsed_ms: float) -> str:
    """One answer as a single line of JSON: everything :meth:`Answer.as_dict` carries, plus the
    timing a person sees in the header."""
    budget = BUDGET_MS[result.question.speed]
    return json.dumps({**result.as_dict(), "elapsed_ms": round(elapsed_ms, 1),
                       "budget_ms": budget, "over_budget": elapsed_ms > budget},
                      ensure_ascii=False, default=str)


def _emit(result: Answer | ConsoleAnswer, elapsed: float, *, mode: str) -> None:
    if mode == "json":
        print(as_json(result, elapsed_ms=elapsed))
    elif mode == "quiet":
        print(resolve_plurals("\n".join(result.lines)))
    else:
        print(render(result, elapsed_ms=elapsed))


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="python -m argus.lui",
        description="Ask ARGUS a research question or about its own record. With a question, "
                    "answer it and exit; without one, start an interactive session.",
        epilog="exit status: 0 answered, 1 refused, 2 usage error")
    out = cli.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true",
                     help="print each answer as one line of JSON, for programs")
    out.add_argument("-q", "--quiet", action="store_true",
                     help="print only the answer's lines: no header, no sources")
    cli.add_argument("question", nargs="*", help="the question, in words")
    return cli


def main(argv: list[str] | None = None) -> int:
    from argus.paper.runner import LEDGER_PATH

    options = parser().parse_args(argv if argv is not None else sys.argv[1:])
    mode = "json" if options.json else "quiet" if options.quiet else "human"
    ledger = PaperLedger(path=LEDGER_PATH)
    session = Session()

    if options.question:
        result, elapsed = ask_console(" ".join(options.question), session)
        _emit(result, elapsed, mode=mode)
        return 0 if not result.refused else 1

    if mode != "human":
        # A session reads questions line by line and answers each in the chosen form; the
        # banner and the prompt are for a person and are left out.
        for line in sys.stdin:
            text = line.strip()
            if text:
                result, elapsed = ask_console(text, session)
                _emit(result, elapsed, mode=mode)
        return 0

    print(BANNER)
    print(f"ledger: {len(ledger.entries)} decision(s), chain "
          f"{'intact' if ledger.verify()['chain_intact'] else 'BROKEN'}\n")

    while True:
        try:
            text = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"quit", "exit", ":q"}:
            return 0
        result, elapsed = ask_console(text, session)
        print(render(result, elapsed_ms=elapsed))
        print()


__all__ = ["BUDGET_MS", "ConsoleAnswer", "Session", "as_json", "ask_console", "main", "parser",
           "render"]


if __name__ == "__main__":
    raise SystemExit(main())
