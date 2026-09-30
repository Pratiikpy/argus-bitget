"""A thesis stated in the console, tested reason by reason where it was asked.

"I think NVDA runs on AI capex through 2027 — test my thesis" is the Track 3 "personalized thesis"
line asked in so many words, and the console answered it with the next day's base rates or with the
desk's own decision on NVDA (judge audit, 2026-09-30). The tester already existed, on the research
task's page (`lui/task.py`, `lui/thesis.py`), where nothing in the console led. This runs that same
task for the named contract — its eight engines side by side, so every reason finds the measurement
that bears on it — and answers with the verdicts, each with the figures behind it and where they
came from. The engines' own pages stay one click away, under the answer.
"""

from __future__ import annotations

import re
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import emit, trace_module

THESIS_ASK = re.compile(
    r"\b(?:test|check|challenge|stress[\s-]*test|pressure[\s-]*test|poke\s+holes\s+in|kill|"
    r"critique|validate|evaluate|assess)\s+(?:my|this|the|that)\s+(?:thesis|idea|view|theory|"
    r"take|call)\b|\bmy\s+thesis\s*(?:is\b|:)|\bhere'?s\s+my\s+(?:thesis|view)\b|"
    r"\bis\s+my\s+(?:thesis|view|idea)\s+(?:right|wrong|sound|valid|any\s+good)\b|"
    r"\bam\s+i\s+(?:right|wrong)\s+(?:that|to\s+think)\b", re.I)
"""A trader stating a view and asking for it to be tested."""

_ORDER = {"contradicted": 0, "supported": 1, "not measurable": 2, "not tested": 3}


def asks(text: str) -> bool:
    return bool(THESIS_ASK.search(text))


def answer(text: str, *, book: str = "", memory: str = ""
           ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The thesis's reasons, each tested (`lui/thesis.check` through `lui/task.research_task`)."""
    from urllib.parse import quote

    from argus.lui import thesis
    from argus.lui.research import research_symbols
    from argus.lui.task import read_question, research_task

    named = research_symbols(text)[0]
    if not named:
        lines = ["Bottom line: name the stock or coin the thesis is about, and why you hold it — "
                 "for example \"I think NVDA runs on AI capex through 2027, test my thesis\" or "
                 "\"long SOL because on-chain activity is growing and this dip is temporary\"."]
        return lines, [], {"thesis": None}
    symbol = named[0]
    name = symbol.removesuffix("USDT")
    stated = thesis.reasons(text)
    link = f"/research?q={quote(text)}"
    if not stated:
        lines = [f"Bottom line: I could not find a reason to test in that — the thesis names "
                 f"{name} but not why. Say it with the reason, for example \"{name} keeps rising "
                 f"because demand is growing\", and each reason is tested against its own data.",
                 f"The full research on {name} runs at {link}."]
        return lines, [], {"thesis": {"name": name, "tested": []}}
    reading = read_question(text, book)
    task = research_task(reading=None if isinstance(reading, str) else reading, asked=text,
                         name=symbol, memory=memory)
    tested = sorted(task.tested, key=lambda t: (t.implied, _ORDER.get(t.result.value, 9)))
    counts = {r: sum(1 for t in tested if not t.implied and t.result.value == r) for r in _ORDER}
    tally = ", ".join(f"{n} {r}" for r, n in counts.items() if n)
    lines = [f"Bottom line: your thesis on {name}, reason by reason — {tally}. Each verdict is "
             f"the one test named beside it; the figures under it inform, they do not vote."]
    sources: list[Source] = []
    seen: set[str] = set()
    for t in tested:
        head = "Implied, and tested" if t.implied else t.result.value.capitalize()
        lines.append(f"{head} — \"{t.reason}\": {t.line}")
        for finding in t.evidence[:4]:
            lines.append(f"  · {finding.text}")
            if finding.url and finding.url not in seen:
                seen.add(finding.url)
                sources.append(Source(kind="venue", ref=finding.url, detail=finding.source))
    lines.append(f"The full research on {name} — price and cost, technicals, news and filings, "
                 f"earnings, history, what it does to your book, exposure and execution — ran "
                 f"beside these tests in {task.seconds:.0f}s; its page is {link}.")
    emit(lines[1:-1], "computed")
    return lines, sources, {"thesis": {"name": name,
                                       "tested": [t.as_dict() for t in task.tested]}}


trace_module(globals())
