"""ARGUS's console — a research workbench a trader questions in plain words, in any language.

It began as three modules that explained the desk's own record; it is now the product, in four
kinds of module:

* **Reading a question** — :mod:`.question` (the deterministic fast path), :mod:`.normalise`,
  :mod:`.phrasebook`, :mod:`.ngram`, :mod:`.kindmodel`, :mod:`.router` (the language model, when
  one is configured), and :mod:`.arbiter`, which decides whose reading stands.
* **Answering it** — :mod:`.research` (the engines: book impact, quote, technicals, macro,
  sentiment, news, fundamentals, execution, analogues), :mod:`.answer` (the desk's own record),
  :mod:`.exposures`, :mod:`.journal`, :mod:`.watchlist`, :mod:`.trending`, :mod:`.onchain`,
  :mod:`.honesty` (the questions no console can answer, said with their reason), :mod:`.memory`,
  :mod:`.multistep`, :mod:`.task` (a whole research task), :mod:`.fanout`, :mod:`.skillroute`,
  :mod:`.agenthub`, :mod:`.translate`, :mod:`.provenance` and :mod:`.trace` (what each line came
  from).
* **Reaching the trader** — :mod:`.server` (the web console), :mod:`.cli`, :mod:`.mcp_server`
  (for agents), :mod:`.telegram_bot` (the hosted bot), :mod:`.selfhost` (``argus-bot``: a trader's
  own bot) and :mod:`.watch` (alerts).
* **Pages** — :mod:`.design` and the status, proof, corrections, agent, brand and materials pages.

The property that holds across all of it: every figure a trader reads is computed from data and
names its source, or the answer says plainly that it cannot be given and why. The language model
reads the question; it never writes a number.
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
