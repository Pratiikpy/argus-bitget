"""Small text helpers the answers share: the question as asked, a cleaned line, a cut sentence."""

from __future__ import annotations

import re

from argus.lui.question import (
    Intent,
    Question,
    Speed,
    Tense,
)
from argus.lui.research.kinds import ResearchRequest
from argus.lui.trace import trace_module


def _question(raw: str, request: ResearchRequest) -> Question:
    return Question(raw=raw, intent=Intent.RESEARCH, speed=Speed.SLOW, tense=Tense.FUTURE,
                    symbols=request.symbols, matched=f"research:{request.kind}:{request.parsed_by}")


def clean_line(line: str) -> str:
    for tag in ("[portfolio] ", "[stress] "):
        line = line.replace(tag, "")
    return re.sub(r"\b([A-Z][A-Z0-9]*?)(?:STOCK)?USDT\b", r"\1", line)


def sentence_cut(text: str, limit: int = 240) -> str:
    """Shorten at a sentence end rather than mid-word; a thesis cut at "The memory rec" reads as
    a bug, because it is one."""
    ends = [m.end() - 1 for m in re.finditer(r"[.;](?=\s)", text)]
    if text.rstrip().endswith(("...", "…")):
        # Already clipped upstream (the ledger stores a bounded thesis): drop the broken tail.
        whole = [e for e in ends if e > 20]
        if whole:
            return text[: whole[-1] + 1].strip()
    if len(text) <= limit:
        return text
    # A full stop beats a semicolon, and neither counts inside an open parenthesis: "(VIX 14.81,
    # normal regime; F&G 71;" is a clause cut in half, not a sentence.
    def closed(end: int) -> bool:
        return text[:end].count("(") <= text[:end].count(")")

    for mark in (".", ";"):
        inside = [e for e in ends if 60 < e < limit and text[e] == mark and closed(e)]
        if inside:
            return text[: inside[-1] + 1].strip()
    # No sentence ends in the window: finish the sentence that is running if it ends soon, since
    # one whole long sentence reads better than a clipped one; clip at a word only as a last resort.
    after = [e for e in ends if limit <= e < limit * 2]
    if after:
        return text[: after[0] + 1].strip()
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:(") + "…"


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
