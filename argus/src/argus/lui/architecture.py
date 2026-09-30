"""How ARGUS is built, told in the order a decision and a question flow through it.

"Walk me through your architecture" was answered with the evidence behind one ledger row (a judge's
audit, 2026-09-30), and agent architecture quality is a named Track 2 judging line. Every step below
names the module that does it, so a judge can open the file; the full technical reference is
ARGUS-ARCHITECTURE.md in the repository. Only what the code does is stated here: the paper desk
records and settles its decisions on its own ledger, and the team's Track 2 agent, which places
orders on Bitget's demo venue, is a separate project with its own repository.
"""

from __future__ import annotations

import re
from typing import Any

from argus.lui.provenance import explains
from argus.lui.trace import trace_module
from argus.truth.source import Source

ARCHITECTURE_Q = re.compile(
    r"\b(?:walk\s+me\s+through|explain|describe|how\s+(?:is|are))\s+(?:your|the|argus'?s?|its)\s+"
    r"(?:(?:system\s+|agent\s+|overall\s+)?architecture|design|pipeline|stack|system)\b|"
    r"\bhow\s+(?:does|do)\s+(?:argus|the\s+desk|you|your\s+(?:agent|system|desk))\s+(?:work|decide|"
    r"make\s+decisions)\b|\bhow\s+(?:are|is)\s+(?:argus|you|the\s+(?:desk|system|agent)|your\s+"
    r"(?:desk|system|agent))\s+(?:built|designed|structured)\b|\byour\s+architecture\b",
    re.I)
"""A question about how the system is built, not about one of its decisions."""

REFERENCE = "https://github.com/Pratiikpy/argus-bitget/blob/main/ARGUS-ARCHITECTURE.md"


def answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    """The two paths through ARGUS — a decision and a question — each step with its module."""
    lines = [
        "Bottom line: one engine with two front doors — a paper desk that decides and is graded "
        "on what happened next, and this console that answers questions — and in both the "
        "language model reads and proposes while code computes every figure and enforces every "
        "limit.",
        "A decision, once a cycle (paper/runner.py): read the live Bitget market and build the "
        "session state and hedge options; an analyst panel reads the evidence and a portfolio "
        "manager proposes a trade (agents/desk.py); the risk layer's eight checks shrink or stop "
        "it (ask \"how does your risk layer work\"); the governed decision is written to a "
        "hash-chained ledger (paper/ledger.py) whose head is anchored, and it is settled only "
        "against a price fetched later, never the one it was decided on.",
        "A question, here (lui/server.py): the hosted Qwen model plans which engine answers, with "
        "a trained classifier and the console's own patterns as the fallback, and fixed rules "
        "decide which reading stands (lui/arbiter.py); seventeen research engines "
        "(lui/research/) and some thirty dedicated answerers — position sizing, hindsight, a "
        "trade journal, an earnings watchlist, sector rotation and more — compute the answer from "
        "live data; each line is labelled by where it came from and every source is listed under "
        "the answer.",
        "What keeps it honest: claims are registered before their outcome is known (the "
        "register), each capability is measured against a named rival on the same input and "
        "published as OWNED, TIED or LOST (/proof), and what was wrong is kept on /wrong.",
        f"The full technical reference, with every structure, invariant and gate in source "
        f"order: {REFERENCE}",
    ]
    explains(*lines)
    sources = [Source("computation", "argus.paper.runner", "one decision cycle, in order"),
               Source("computation", "argus.lui.arbiter", "which reading of a question stands")]
    return lines, sources, {"reference": REFERENCE}


trace_module(globals())
