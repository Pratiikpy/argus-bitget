"""Crossfire (CryptoCT01/Crossfire @7a3bdfa) in the XA arena — its own keyless decision path.

Run as ``python crossfire_runner.py`` with the Crossfire clone as working directory. Calls its
unmodified ``crossfire.agent_engine.policy_decide(books)`` (`agent_engine.py:330-436`) every hour on
books built from the tape in its own shape (``crossfire/bitget_public.py:101-123``: rows carrying
``change24h_pct`` in percent). That function is the path Crossfire runs without an LLM key, and it
suppresses every open by design (`agent_engine.py:420-428`, "policy does not open risk"). The runner
therefore places no orders; it counts how often the rule would have fired, which is what the
LLM would have been asked to act on.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from _proto import Tape, serve  # type: ignore[import-not-found, unused-ignore]


def _load(name: str) -> Any:
    """One of the rival's own modules, from its clone on ``sys.path``."""
    return importlib.import_module(name)

class Crossfire:
    def __init__(self, init: dict[str, Any], tape: Tape) -> None:
        self.engine: Any = _load("crossfire.agent_engine")
        self.config: Any = _load("crossfire.config")
        self.policy = self.config.active_policy()
        self.fired: dict[str, int] = {}
        self.decisions = 0
        self.first_thesis: list[str] = []

    def _row(self, tape: Tape, sym: str) -> dict[str, Any]:
        ch = tape.change(f"perp:{sym}", 24)
        last = tape.close(f"perp:{sym}")
        return {"symbol": sym, "available": last is not None, "last": last,
                "change24h_pct": ch * 100.0 if ch is not None else None}

    def step(self, msg: dict[str, Any], tape: Tape) -> dict[str, Any]:
        books = {"us": [self._row(tape, s) for s in self.policy["mag7_symbols"]],
                 "crypto": [self._row(tape, s) for s in ("BTCUSDT", "ETHUSDT")]}
        dec = self.engine.policy_decide(books)
        self.decisions += 1
        sup = dec.get("suppressed_action")
        if sup:
            self.fired[sup] = self.fired.get(sup, 0) + 1
            if len(self.first_thesis) < 3:
                self.first_thesis.append(str(dec.get("thesis", ""))[:300])
        return {"orders": [], "action": str(dec.get("action", "HOLD")),
                "note": str(dec.get("thesis", ""))[:240] if sup else ""}

    def finish(self) -> dict[str, Any]:
        return {"decisions": self.decisions, "would_have_fired": self.fired,
                "threshold_pct": self.policy.get("divergence_threshold_pct"),
                "mode": self.config.get_agent_mode(), "sample_suppressed": self.first_thesis}


if __name__ == "__main__":
    serve(Crossfire)
