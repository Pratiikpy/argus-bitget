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
    r"\bis\s+(?:my|that|this|the)\s+(?:thesis|view|idea|take|theory|reasoning)\s+(?:right|correct|"
    r"wrong|sound|valid|any\s+good|true)\b|\bam\s+i\s+(?:right|wrong)\s+(?:that|to\s+think)\b|"
    # "I think NVDA is undervalued because its P/E is low relative to growth. Is that thesis
    # right?" reached a data dump (a judge, round 11): a view with its reason is a thesis
    r"\bi\s+(?:think|believe|reckon|expect|feel)\s+(?:that\s+)?[^?.]{0,160}\bbecause\b", re.I)
"""A trader stating a view and asking for it to be tested."""

_NOT_NAMES = frozenset({
    "test", "check", "thesis", "think", "believe", "because", "fed", "the", "this", "that", "my",
    "is", "it", "and", "ai", "ceo", "etf", "etfs", "gdp", "cpi", "fomc", "sec", "usa", "us", "eu",
    "china", "chinese", "america", "american", "europe", "japan", "india", "q1", "q2", "q3", "q4",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "poke", "kill", "here", "am", "right", "wrong"})
"""Capitalised words that are not a company a thesis could be about."""

_MARKETS: tuple[tuple[str, str, str], ...] = (
    (r"\b(?:crypto|coins|altcoins|the\s+crypto\s+market)\b", "BTCUSDT",
     "no single name was given, so the crypto market is read as bitcoin (BTCUSDT)"),
    (r"\b(?:stocks|equities|the\s+(?:stock\s+)?market|us\s+stocks|wall\s+street)\b", "SP500USDT",
     "no single name was given, so the stock market is read as Bitget's S&P 500 contract "
     "(SP500USDT)"),
)
"""A thesis about a whole market, and the one contract that stands for it."""

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
    market_note = ""
    if not named:
        # "I think stocks will go up" is a thesis about the market: tested on the broad index
        # contract Bitget lists, said on the answer (round 11: it was asked to name a stock).
        broad = next(((sym, label) for pattern, sym, label in _MARKETS
                      if re.search(pattern, text, re.I)), None)
        if broad is not None:
            named, market_note = (broad[0],), broad[1]
    unlisted = [w for w in re.findall(r"(?<!^)(?<![.?!]\s)\b([A-Z][A-Za-z&.-]{2,})\b", text)
                if w.lower() not in _NOT_NAMES]
    if not named and unlisted:
        # "I think Tencent will outperform because of gaming approvals" was told to name a stock
        # (a hostile review, round 11): it named one Bitget does not list.
        lines = [f"Bottom line: {unlisted[0]} is not a contract Bitget lists, so there is no "
                 f"price, filing or positioning here to test the thesis against. Ask it of a "
                 f"listed name — any US stock or ETF Bitget carries, gold, oil, an index or a "
                 f"coin."]
        return lines, [], {"thesis": None, "unlisted": unlisted[0]}
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
    said: dict[str, str] = {}
    for t in tested:
        head = "Implied, and tested" if t.implied else t.result.value.capitalize()
        if t.line in said:
            # Two reasons read by one test ("NVDA is undervalued", "its P/E is low") say it once.
            lines.append(f"{head} — \"{t.reason}\": the same reading as \"{said[t.line]}\".")
            continue
        said[t.line] = t.reason
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
    if market_note:
        lines.append(f"Assumed: {market_note}.")
    return lines, sources, {"thesis": {"name": name,
                                       "tested": [t.as_dict() for t in task.tested]}}


trace_module(globals())
