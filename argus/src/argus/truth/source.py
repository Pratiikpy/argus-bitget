"""Where a fact came from, recorded so a reader can go and check it.

Moved from `lui/answer.py` on 2026-09-27. Every layer that states a fact cites one: the research
studies and the stress tree did, and imported this type upward from the console to do it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """Where a fact came from. Enough for a reader to go and check it."""

    kind: str
    """``ledger`` | ``computation`` | ``evidence`` | ``venue``."""

    ref: str
    """A ledger seq, an evidence id, a module path, or a URL."""

    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.kind}:{self.ref}" + (f" ({self.detail})" if self.detail else "")
