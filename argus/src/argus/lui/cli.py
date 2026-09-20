"""The desk console — ask ARGUS about its own record, in words.

Run it with ``python -m argus.lui`` for an interactive session, or pass a question as arguments to
ask one and exit: ``python -m argus.lui why did you do nothing all weekend``.

**What this surface is for.** It is the answer to a judge's most dangerous question, which is never
"what does it do" but "show me". Every reply here is reconstructed from the hash-chained ledger,
carries the sources it was built from, and refuses by name when the record cannot support it. The
transcript of a session *is* the decision-explainability evidence; there is nothing to prepare
beforehand and nothing that can be staged.

Rendering rules, from ``research/subthemes/_CENSUS-lui.md``:

- a refusal is shown as a refusal, with the reason and what to ask instead;
- sources are printed under every answer, never folded into the prose;
- the latency budget for the question class is shown next to the time actually taken, so a slow
  answer is visibly slow rather than quietly slow.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from datetime import UTC, datetime

from argus.lui.answer import Answer, Source, answer
from argus.lui.question import Conversation, Speed, classify
from argus.lui.router import Router, build_router, route
from argus.paper.ledger import PaperLedger

BUDGET_MS: dict[Speed, int] = {Speed.FAST: 500, Speed.MEDIUM: 5_000, Speed.SLOW: 30_000}

BANNER = """ARGUS desk console - ask about the record.

Every answer is reconstructed from the hash-chained decision ledger and cites its sources.
When the record cannot support an answer, you get a refusal and the reason, not a guess.

Try:  why did you do nothing all weekend
      show me decision 25          then:  what evidence backed that
      what is the sharpe           is the log tamper-evident
      are you well calibrated      what is my position

Ctrl-D or "quit" to leave.
"""


def render(result: Answer, *, elapsed_ms: float) -> str:
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

    return head + "\n" + "\n".join(body)


def ask(ledger: PaperLedger, text: str, *, conversation: Conversation,
        now: datetime | None = None, router: Router | None = None) -> tuple[Answer, float]:
    """Classify, answer and time one question.

    When the patterns do not understand the question, ``router`` gets one chance to say which kind
    of question it is — and nothing more. The answer is produced by the same answerer either way,
    from the same ledger, so routing changes which grounded answer is given and never what it says.
    """
    started = time.perf_counter()
    clock = now or datetime.now(UTC)
    question = classify(text, now=clock, conversation=conversation)
    question, routing = route(question, client=router, now=clock, conversation=conversation)
    conversation.remember(question)
    result = answer(ledger, question)
    if routing.applied:
        result = replace(result, sources=[*result.sources, Source("router", routing.render())])
    return result, (time.perf_counter() - started) * 1000


def main(argv: list[str] | None = None) -> int:
    from argus.paper.runner import LEDGER_PATH

    args = list(argv if argv is not None else sys.argv[1:])
    ledger = PaperLedger(path=LEDGER_PATH)
    conversation = Conversation()
    # Built once for the session. Returns None without credentials, and the console then answers
    # from patterns alone rather than failing to start.
    router = build_router()

    if args:
        result, elapsed = ask(ledger, " ".join(args), conversation=conversation, router=router)
        print(render(result, elapsed_ms=elapsed))
        return 0 if not result.refused else 1

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
        result, elapsed = ask(ledger, text, conversation=conversation, router=router)
        print(render(result, elapsed_ms=elapsed))
        print()


__all__ = ["BUDGET_MS", "ask", "main", "render"]


if __name__ == "__main__":
    raise SystemExit(main())
