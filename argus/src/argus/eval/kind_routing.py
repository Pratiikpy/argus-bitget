"""Which engine the console sends a free-form question to, scored on three blind question sets.

The submission quotes these figures, and until 2026-09-27 no committed code produced them: they
came from a scoring script run by hand on 2026-09-25/26 (audit finding 99). This is that scoring,
written down, so the figure is recomputed rather than remembered and the documentation gate
(`eval/docclaims.py`, claim ``kind_routing``) fails the moment the draft and the console disagree.

**What is scored.** Each row carries the kind a writer expected: one of the thirteen research
kinds of `eval/kindtrain.LABELS`, ``record`` for a question about the desk's own history, or
``refuse`` for one the console must not answer (an unlisted ticker, a trade instruction, an
exact-price prophecy, small talk). A row is routed exactly as ``/ask`` routes it
(`lui/server.handle_ask`), offline, through `eval/record_routing._offline`: market reads raise and
a question that reaches the research layer is recorded as the kind it reached, without being
answered. No model is called, so this is the console a visitor gets with no Qwen key.

**How a reply is read.** The console's own reply decides, never a label mapping chosen to flatter
it: a research reply is its ``kind``; a reply the console marks ``refused`` — or an ``unknown`` /
``ambiguous`` intent, which it answers by saying it did not understand — is ``refuse``; any record
intent, the risk-control summary included, is ``record``. Anything else is its intent name and
matches no expected label, so it counts as wrong.

**The sets, in the order the draft quotes them, and how each was used.** All three were written by
agents forbidden to read this repository, from the kinds' definitions alone:

* ``lui_final_heldout_2026-09-25.jsonl`` — 240 rows, the third writer. Scored once, then its
  misses were read on 2026-09-27 and four general defects fixed (below), so it is a development
  set now; its 2026-09-26 figure, 210, is the last held-out one.
* ``lui_heldout_corpus_2026-09-25.jsonl`` — 200 rows. Held out once (patterns 55.0% before that
  day's fixes), then used as development and as kind-model training data.
* ``lui_blind_corpus_2026-09-25.jsonl`` — 240 rows. The set the 2026-09-25 pattern fixes were made
  against, and kind-model training data: a fit figure, never a test.

The 2026-09-27 fixes, from reading the sets' misses: "execute a buy order for ..." was worked as an
execution plan instead of refused as an instruction; "hedge my book against a crash" was refused as
an order instead of answered by the hedge engine; "why is SOL pumping right now", asked by all three
writers, got the positioning read instead of the news; and "the price of QWXZ coin" was quoted
Coinbase stock, "coin" read as the ticker. Each is tested on phrasings that are in none of these
sets (`tests/test_kind_routing.py`). The misses left are published per row in the artefact; most
are non-English, or name a stock the venue does not list, which the console refuses by design.

    python -m argus.eval.kind_routing          # score all three, write the artefact
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from argus.eval.record_routing import RECORD_KINDS, _offline, _Routed, load

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
REPORT_PATH = DATA / "lui_kind_routing.json"

CORPORA = (
    ("final", DATA / "lui_final_heldout_2026-09-25.jsonl"),
    ("heldout", DATA / "lui_heldout_corpus_2026-09-25.jsonl"),
    ("blind", DATA / "lui_blind_corpus_2026-09-25.jsonl"),
)
"""In the order the submission draft quotes them."""

REFUSING_INTENTS = frozenset({"unknown", "ambiguous"})
"""Intents answered with "I did not understand that question" — a refusal whatever the flag."""


def route(text: str) -> tuple[str, str]:
    """The kind the console gives ``text``, read from its own reply, and the layer that decided."""
    from argus.lui import server

    try:
        payload = server.handle_ask(text, [], visitor="kind-routing", book="")
    except _Routed as routed:
        return routed.kind.removeprefix("research:"), "research"
    intent = str(payload.get("intent"))
    by = str(payload.get("classified_by") or "")
    if payload.get("refused") or intent in REFUSING_INTENTS:
        return "refuse", by
    if intent in RECORD_KINDS or intent == "risk_control":
        return "record", by
    return intent, by


def score(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Each row routed and marked right or wrong, with totals by language and by expected kind."""
    results = []
    with _offline():
        for row in rows:
            got, by = route(str(row["text"]))
            results.append({"id": row.get("id"), "lang": row.get("lang"), "text": row["text"],
                            "expected": row["expected"], "got": got, "by": by,
                            "right": got == row["expected"]})

    def tally(key: str) -> dict[str, list[int]]:
        return {value: [sum(r["right"] for r in results if r[key] == value),
                        sum(1 for r in results if r[key] == value)]
                for value in sorted({str(r[key]) for r in results})}

    return {
        "correct": sum(r["right"] for r in results),
        "rows": len(results),
        "by_language": tally("lang"),
        "by_kind": tally("expected"),
        "wrong_by_layer": dict(Counter(r["by"] for r in results if not r["right"])),
        "results": results,
    }


def run() -> dict[str, Any]:
    return {name: {"corpus": path.name, **score(load(path))} for name, path in CORPORA}


def main(argv: Sequence[str] | None = None) -> int:
    import json

    del argv
    report = run()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    for name, blob in report.items():
        print(f"{name:8s} {blob['correct']} / {blob['rows']}  {blob['corpus']}")
    print(f"wrote {REPORT_PATH.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
