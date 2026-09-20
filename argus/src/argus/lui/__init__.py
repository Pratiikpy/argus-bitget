"""ARGUS language user interface — ask the desk about its own record.

Three pieces: :mod:`argus.lui.question` classifies a question deterministically on the fast path,
:mod:`argus.lui.answer` reconstructs a reply from the hash-chained ledger with its sources, and
:mod:`argus.lui.cli` is the console a person actually uses.

The property that holds across all three: every assertion resolves to a ledger row, a named
computation, or an explicit refusal. There is no path by which a sentence reaches the user without
something behind it.
"""

from argus.lui.answer import Answer, Source, answer
from argus.lui.question import Conversation, Intent, Question, Speed, classify

__all__ = [
    "Answer",
    "Conversation",
    "Intent",
    "Question",
    "Source",
    "Speed",
    "answer",
    "classify",
]
