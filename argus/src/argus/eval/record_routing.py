"""How well the console routes questions about the desk's own record, on blind question sets.

The three held-out corpora in ``data/lui_*_2026-09-25.jsonl`` score which *research* engine a
question reaches and lump every record question into one class, so a question about the desk that
reached the wrong record answer ("list all decisions" answered with the open positions) counted as
right. This scores the record kinds themselves: performance, decision_why, decision_list,
abstention_why, evidence, calibration, integrity, position, session, and ``none`` for a question
that must not reach the record at all.

**The sets and how they were used** (2026-09-26). Each was written by an agent that was told not to
open this repository and worked only from the kinds' definitions:

* ``data/lui_record_intents_2026-09-26.jsonl`` — 240 questions, split by kind with seed 20260926
  into ``_tune`` and ``_test`` halves. Round 1 changes were made from the tuning half only; the test
  half was scored once afterwards and then used to tune round 2, so it is no longer held out.
* ``data/lui_record_intents_2026-09-26_round2.jsonl`` — 200 questions by five personas, never used
  for tuning. Its "after" figure is the one to quote; the four regressions it exposed (the economy's
  "we" in "are we heading into a recession") were fixed afterwards, so later scores on it are not
  held out.

Every question is routed exactly as ``/ask`` routes it (`lui/server.handle_ask`), offline: market
fetches raise, so a question the research layer claims is recorded as the research kind it reached
without being answered. No model is called.

    python -m argus.eval.record_routing data/lui_record_intents_2026-09-26_round2.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
REPORT_PATH = DATA / "lui_record_routing_2026-09-26.json"

RECORD_KINDS = frozenset({
    "performance", "decision_why", "decision_list", "abstention_why", "evidence", "calibration",
    "integrity", "position", "session",
})


class _Routed(Exception):
    """Raised by the research stand-in: the question reached a research engine."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


@contextmanager
def _offline() -> Iterator[None]:
    """Route without answering: every market read fails and a research answer is not built."""
    from argus.lui import research, server
    from argus.market import history, universe
    from argus.paper.ledger import PaperLedger

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("offline: no market reads while scoring routing")

    def reached(text: str, prior: list[str], request: Any, ledger: PaperLedger, started: float,
                classified_by: str, audit: dict[str, Any]) -> dict[str, Any]:
        del text, prior, ledger, started, classified_by, audit
        raise _Routed("research:" + str(request.kind))

    saved = (research._fetch_live, history.fetch, universe._fetch_live, server._research_payload)
    research._fetch_live = refuse
    history.fetch = refuse
    universe._fetch_live = refuse
    server._research_payload = reached
    try:
        yield
    finally:
        (research._fetch_live, history.fetch, universe._fetch_live,
         server._research_payload) = saved


def route(text: str) -> tuple[str, str]:
    """The kind ``/ask`` gives ``text`` and the layer that decided it."""
    from argus.lui import server

    try:
        payload = server.handle_ask(text, [], visitor="record-routing", book="")
    except _Routed as routed:
        return routed.kind, "research"
    return str(payload.get("intent")), str(payload.get("classified_by") or "")


def score(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Each row routed and marked right or wrong, with totals by language and by kind."""
    results = []
    with _offline():
        for row in rows:
            got, by = route(row["text"])
            right = got not in RECORD_KINDS if row["expected"] == "none" else got == row["expected"]
            results.append({**row, "got": got, "by": by, "right": right})
    kinds = sorted({r["expected"] for r in results})
    return {
        "correct": sum(r["right"] for r in results),
        "rows": len(results),
        "by_language": {
            lang: [sum(r["right"] for r in results if r["lang"] == lang),
                   sum(1 for r in results if r["lang"] == lang)]
            for lang in sorted({r["lang"] for r in results})
        },
        "by_kind": {k: [sum(r["right"] for r in results if r["expected"] == k),
                        sum(1 for r in results if r["expected"] == k)] for k in kinds},
        "wrong_by_layer": dict(Counter(r["by"] for r in results if not r["right"])),
        "results": results,
    }


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m argus.eval.record_routing <questions.jsonl>")
        return 2
    report = score(load(Path(args[0])))
    print(f"{report['correct']} / {report['rows']}")
    for lang, (right, total) in report["by_language"].items():
        print(f"  {lang}: {right} / {total}")
    for kind, (right, total) in report["by_kind"].items():
        print(f"  {kind:15s} {right:3d} / {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
