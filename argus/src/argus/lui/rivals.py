"""How ARGUS stands against a named system, answered from the capability register.

"How does ARGUS's risk layer compare to Nautilus Trader?" was answered with the risk layer's own
intervention count and never mentioned Nautilus (fresh-eyes audit, 2026-09-29). Every comparison
ARGUS has run is already a register row (`eval/standing.py`) that names its rival in ``baseline``
and says in its first blocker what was measured, so the answer is those rows, read at the moment
the question is asked: the state (OWNED, TIED, IMPLEMENTED, LOST) and the measurement behind it.
A rival no row names is said plainly, rather than answered with something adjacent.
"""

from __future__ import annotations

import re
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import trace_module

_COMPARE = re.compile(r"\b(?:compar\w*|vs\.?|versus|against|better\s+than|worse\s+than|beat\w*|"
                      r"stack\s+up|measure[sd]?\s+up)\b", re.I)

_GENERIC = frozenset({
    "argus", "argus's", "the", "risk", "layer", "trader", "trading", "desk", "compare", "compared",
    "against", "versus", "better", "than", "does", "how", "what", "with", "your", "you", "this",
    "that", "model", "agent", "agents", "system", "engine", "stack", "measure", "worse", "beat",
    "data", "real", "research", "market", "stock", "stocks", "crypto", "bitget", "open", "source",
    "bitcoin", "ether", "ethereum", "gold", "silver", "nasdaq", "index", "funds", "fund",
})
"""Words a question and a baseline share without naming a rival."""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def rival_tokens(text: str) -> list[str]:
    """The words in the question that could name a system: four letters or more, not generic."""
    from argus.lui.question import resolve_symbol

    return [w for w in _norm(text).split()
            if len(w) >= 4 and w not in _GENERIC and resolve_symbol(w) is None]


_SELF = re.compile(r"\bargus\b|\byou(?:r|rs)?\b|\bthis\s+(?:desk|tool|console|product|system)\b",
                   re.I)
"""The question has to be about ARGUS: "compare gold and bitcoin" is a comparison of two assets."""


def asks_about_a_rival(text: str) -> bool:
    return bool(_COMPARE.search(text)) and bool(_SELF.search(text)) and bool(rows_naming(text))


def rows_naming(text: str) -> list[Any]:
    """Register rows whose named rival contains a word of the question."""
    from argus.eval.standing import REGISTER

    words = rival_tokens(text)
    return [cap for cap in REGISTER
            if any(re.search(rf"\b{re.escape(w)}", _norm(cap.baseline)) for w in words)]


def _first_sentence(blocker: str) -> str:
    """The row's measurement: its first sentence, after any "RE-GRADED ... from X to Y" line, which
    is the register's bookkeeping rather than what was measured, and without internal file refs."""
    flat = re.sub(r"\s*\((?:Activity|tracker)[^)]*\)", "", " ".join(blocker.split()))
    sentences = re.split(r"(?<=[a-z0-9)\]%])\.\s+(?=[A-Z])", flat)
    keep = [x for x in sentences if not re.match(r"(?:RE-)?GRADED\b", x)] or sentences
    first = keep[0].rstrip(".")
    return (first[:297] + "...") if len(first) > 300 else first + "."


def answer(text: str) -> tuple[list[str], list[Source], dict[str, Any]]:
    rows = rows_naming(text)
    hits = {w for w in rival_tokens(text)
            if any(re.search(rf"\b{re.escape(w)}", _norm(c.baseline)) for c in rows)}
    # The rival as the question spelled it ("Nautilus", not "nautilus").
    named = ", ".join(sorted({m.group(0) for w in hits
                              for m in [re.search(re.escape(w), text, re.I)] if m}))
    order = {"owned": 0, "tied": 1, "implemented": 2, "lost": 3}
    rows = sorted(rows, key=lambda c: (order.get(c.state.value, 9), c.name))
    counts = {s: sum(1 for c in rows if c.state.value == s) for s in order}
    tally = ", ".join(f"{n} {s.upper()}" for s, n in counts.items() if n)
    lines = [f"Bottom line: {len(rows)} capability row(s) in ARGUS's register were measured "
             f"against {named}: {tally}. Each state below is the register's own, re-derived "
             f"from its artefacts, not a claim written for this answer."]
    for cap in rows[:6]:
        why = _first_sentence(cap.blockers[0]) if cap.blockers else ""
        lines.append(f"{cap.state.value.upper()} — {cap.name}"
                     + (f": {why}" if why else "."))
    if len(rows) > 6:
        lines.append(f"{len(rows) - 6} more on the proof page (/proof).")
    return lines, [Source(kind="computation", ref="argus.eval.standing register",
                          detail=f"rows whose named rival matches {named}")], {
        "rival_rows": [{"name": c.name, "state": c.state.value} for c in rows]}


trace_module(globals())
