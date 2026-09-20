"""What the desk knows, and when it became knowable. The type both halves of the system share.

**This lived in `agents/analysts.py` and that was a real coupling defect, found by measuring the
import graph rather than by reading the code.** The market layer *produces* evidence and the agent
layer *consumes* it, so a producer importing its consumer's module inverted the dependency — and
because `agents/analysts.py` imports `argus.llm.base` and `argus.llm.qwen`, the consequence was
concrete and worse than untidy:

    >>> import argus.market.evidence      # a pure data fetcher
    # ... also loads argus.llm.qwen

Fetching an SEC filing pulled in the model client. ARGUS's central architectural claim is that the
deterministic layers never touch the model — verified for `truth`, `cost`, `risk`, `decision` and
`backtest`, and quietly false for `market`, which is the layer that *gathers the facts the claim is
about*.

`truth` is the right home rather than a new shared package: this module already owns
`clocks.py`, and the two types answer the same question. `SessionState` says whether a price can be
discovered at an instant; `Evidence` says what was knowable at one. Point-in-time discipline is a
`truth` concern, and `available_at` is the field the whole discipline turns on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Evidence:
    """One piece of evidence with the provenance the independence graph needs."""

    id: str
    claim: str
    source: str
    available_at: datetime
    """When this became knowable — never when it was fetched.

    The difference is the whole point-in-time model: stamping ingestion time would make every
    decision look prescient by exactly the fetch latency.
    """

    credibility: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)
    """Structured fields behind ``claim``, when the source has them.

    ``claim`` is prose, and prose cannot refute a thesis: a desk that writes "these sales are
    pre-arranged" about a filing whose ``aff10b5One`` is 0 has contradicted its own evidence, and
    nothing in the rendered sentence makes that checkable. Sources that carry structured attributes
    (Form 4, XBRL) put them here so :mod:`argus.agents.claims` can settle such a claim against the
    record rather than against another model's opinion. Empty for prose-only sources, and a claim
    with no field behind it is reported as unexamined rather than as passing."""

    def render(self) -> str:
        return (
            f"[{self.id}] ({self.source}, credibility {self.credibility:.2f}, "
            f"available {self.available_at.isoformat()}) {self.claim}"
        )


__all__ = ["Evidence"]
