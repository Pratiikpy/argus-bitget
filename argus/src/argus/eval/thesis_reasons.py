"""Does each reason in a thesis rest on a typed piece of evidence? (build-list 5.2).

A thesis is the desk's reasons in prose. Each sentence is one reason; a reason that states a figure
can be traced to the item that carried the figure, and that item has a kind
(`truth/evidence.Kind`), a channel and a time it became available. This takes every decision whose
evidence the runner kept (``data/desk_notes.jsonl``, joined to ``data/paper_ledger.jsonl`` by
``seq``), splits its thesis into sentences, and resolves every figure with the desk's own grounding
check (`truth/grounding.check`, the one `agents/meta_pm.py` runs at decision time) against the
decision's frame facts and its kept evidence lines.

Each sentence ends in one of four states: **traced** (every figure resolved, to items whose kinds
are listed), **partly** (some did), **untraced** (none did) and **no figure** — a qualitative reason
("the anchor is asleep", "no hedge is available") that no deterministic check can tie to one item,
counted rather than guessed at. The frame facts a thesis may quote are the ones the ledger row
carries (entry price, hours to discovery, entry cost, the stated and lean confidences); a fact the
frame held but the ledger did not keep cannot be resolved here, so ``untraced`` is an upper bound.

**What the kinds mean, and do not.** A figure resolves to the first known value within tolerance,
the grounding check's own rule, so the kind credited is that first item's; a small whole number can
match a line by coincidence (the coverage note's "0 filings" is an example). The states are the
measurement; ``traced_to_kind`` says what the reasons lean on, approximately.

**Finding, 2026-10-05.** 385 decisions, 1,367 reasons: 1,100 state a figure, and 978 of those
(88.9%) trace every figure to the frame or a kept evidence item; 103 trace some, 19 none; 267
reasons (19.5%) state no figure. So not every reason carries a typed item: an untraced figure is
marked where the console shows the thesis (build-list 4.2), and a qualitative reason is not
checked by anything deterministic.

    python -m argus.eval.thesis_reasons      # writes data/thesis_reasons.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Final

from argus.eval.evidence_routing import _LINE, _lines, kind_from_line
from argus.truth import artefact
from argus.truth.grounding import check, extract
from argus.truth.paths import DATA_DIR

NOTES: Final = DATA_DIR / "desk_notes.jsonl"
LEDGER: Final = DATA_DIR / "paper_ledger.jsonl"
OUT: Final = DATA_DIR / "thesis_reasons.json"
FRAME_FACTS: Final = ("entry_price", "hours_to_discovery", "entry_cost_bps", "stated_confidence",
                      "lean_confidence")
_SENTENCE: Final = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def sentences(thesis: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(thesis.strip()) if s.strip()]


def _facts(row: dict[str, Any]) -> dict[str, float]:
    out = {}
    for name in FRAME_FACTS:
        try:
            out[name] = float(row[name])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def reason_states(thesis: str, facts: dict[str, float],
                  evidence: list[str]) -> list[dict[str, Any]]:
    """Each sentence's state and the kinds of the items its figures resolved to."""
    values: list[tuple[str, float]] = []
    kinds: dict[str, str] = {}
    for line in evidence:
        match = _LINE.match(line)
        if match is None:
            continue
        item_id = match.group("id")
        kind = kind_from_line(item_id, match.group("source"))
        kinds[item_id] = kind.value if kind else "unknown"
        claim = line.split(") ", 1)[1] if ") " in line else line
        values.extend((item_id, f.value) for f in extract(claim))
    out = []
    for sentence in sentences(thesis):
        report = check(sentence, facts=facts, evidence_values=values)
        resolved = [r for r in report.resolutions if r.resolved]
        total = len(report.resolutions)
        state = ("no figure" if total == 0 else "traced" if len(resolved) == total
                 else "untraced" if not resolved else "partly")
        out.append({"state": state, "figures": total, "resolved": len(resolved),
                    "kinds": sorted({kinds.get(str(r.source), "frame") for r in resolved})})
    return out


def count(notes: Path = NOTES, ledger: Path = LEDGER) -> dict[str, Any]:
    theses: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("thesis") and row.get("seq") is not None:
                theses[str(row["seq"])] = row
    states: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    decisions = 0
    for line in notes.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        note = json.loads(line)
        row = theses.get(str(note.get("seq")))
        lines = _lines(note.get("evidence"))
        if row is None or not lines:
            continue
        decisions += 1
        for reason in reason_states(str(row["thesis"]), _facts(row), lines):
            states[reason["state"]] += 1
            for kind in reason["kinds"]:
                by_kind[kind] += 1
    with_figures = states["traced"] + states["partly"] + states["untraced"]
    return {
        "decisions": decisions, "sentences": sum(states.values()),
        "states": dict(states.most_common()),
        "with_a_figure": with_figures,
        "traced_share_of_figure_reasons": (round(states["traced"] / with_figures, 4)
                                           if with_figures else None),
        "traced_to_kind": dict(by_kind.most_common()),
        "frame_facts_available": list(FRAME_FACTS),
    }


def run(*, out: Path = OUT) -> dict[str, Any]:
    blob = count()
    artefact.write(out, blob)
    return blob


def main() -> int:  # pragma: no cover - CLI
    print(json.dumps(run(), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
