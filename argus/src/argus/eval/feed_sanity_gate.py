"""The feed-sanity gate, measured on the S3 corrupted-feed cases and on every clean snapshot.

`eval/feedbugged.py` bought twelve corrupted decisions from the live Meta-PM (six planted wrong
values, six injected wrong reasoning steps) and the model's detection F1 was **0.0**. This module
runs the deterministic gate (`desk/feed_sanity.py`) over **the same corrupted inputs** — rebuilt
byte for byte from the recorded bug kind, feature and seed, and checked against the wrong value the
artefact recorded — and over every clean recorded snapshot in ``data/robustness_snapshots.json``
plus the six correct-reasoning lines, which are the negatives. No model is called.

**The acceptance set and the coverage matrix are reported separately.** The acceptance set is the
twelve corruptions the desk actually failed on. The coverage matrix plants every bug family on every
figure of every snapshot, and exists to say honestly what the gate *cannot* see: a sign-flipped 24h
change or a small VIX offset with a matching implied move contradicts nothing in the frame, and
those cells are expected to read as misses. A gate reported only on the cases it was built against
would overstate itself.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from argus.desk.feed_sanity import SanityReport, screen_claims
from argus.eval.artefact import write
from argus.eval.feedbugged import (
    BUG_KINDS,
    REPORT_PATH,
    features,
    plant,
    precision_recall_fscore,
)
from argus.eval.perturbations import DATA, Snapshot, load_snapshots, split_evidence

OUT_PATH = DATA / "feed_sanity_gate.json"

_ID = re.compile(r"^\[([^\]]+)\]")


def items(lines: Sequence[str]) -> list[tuple[str, str]]:
    """``(item_id, claim)`` for rendered evidence lines, as the desk's `Evidence` would carry them.

    A header-less ``[panel] ...`` line — how upstream reasoning reaches the Meta-PM — is kept whole,
    because the panel rules read its leading ``[panel] <analyst>: <signal> <N>bps``.
    """
    out: list[tuple[str, str]] = []
    for line in lines:
        offset, claim = split_evidence(line)
        if offset == 0:
            m = _ID.match(line)
            out.append((m.group(1) if m else "line", line))
        else:
            m = _ID.match(line)
            out.append((m.group(1) if m else "line", claim))
    return out


def gate(snapshot: Snapshot, lines: Sequence[str]) -> SanityReport:
    return screen_claims(items(lines), token_price=snapshot.token_price)


def acceptance(snaps: dict[str, Snapshot], recorded: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    pairs: list[tuple[bool, bool]] = []
    for row in recorded["cases"]:
        snap = snaps[row["snapshot_id"]]
        detail = row["detail"]
        if row["condition"] == "bugged":
            bug = plant(snap, kind=detail["kind"], feature=detail["feature"], seed=0)
            if bug is None or bug.wrong != detail["wrong"]:
                raise AssertionError(
                    f"{snap.id}: rebuilt corruption does not match the recorded one "
                    f"({None if bug is None else bug.wrong!r} vs {detail['wrong']!r})"
                )
            lines, corrupted = list(bug.snapshot.evidence), True
            what = f"{detail['feature']} {detail['kind']}: {detail['true']} -> {detail['wrong']}"
        else:
            lines = [*snap.evidence, detail["injected"]]
            corrupted = row["condition"] == "incorrect_reasoning"
            what = row["condition"]
        report = gate(snap, lines)
        caught = not report.clean
        pairs.append((caught, corrupted))
        cases.append({
            "snapshot_id": snap.id, "condition": row["condition"], "corruption": what,
            "corrupted": corrupted, "caught": caught, "rules": list(report.rules_fired),
            "model_decision_recorded": row["decision"].get("action"),
        })
    tp, fp, tn, fn, accuracy, precision, recall, f1 = precision_recall_fscore(pairs)
    return {
        "cases": cases,
        "detection": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": round(accuracy, 4),
                      "precision": round(precision, 4), "recall": round(recall, 4),
                      "f1": round(f1, 4)},
        "model_detection_f1_on_the_same_cases": recorded["detection"]["f1"],
    }


def clean_negatives(snapshots: Sequence[Snapshot]) -> dict[str, Any]:
    rows = []
    for snap in snapshots:
        report = gate(snap, snap.evidence)
        rows.append({"snapshot_id": snap.id, "items_checked": report.checked,
                     "false_flags": len(report.findings), "rules": list(report.rules_fired)})
    return {
        "snapshots": len(rows),
        "items_checked": sum(r["items_checked"] for r in rows),
        "false_flags": sum(r["false_flags"] for r in rows),
        "rows": rows,
    }


def coverage(snapshots: Sequence[Snapshot]) -> dict[str, Any]:
    """Every bug family on every figure of every snapshot: what the gate can and cannot see."""
    cells: dict[str, dict[str, int]] = {}
    for snap in snapshots:
        for feat in {f.name for f in features(snap)}:
            for kind in BUG_KINDS:
                bug = plant(snap, kind=kind, feature=feat, seed=0)
                if bug is None or bug.feature.name != feat:
                    continue
                cell = cells.setdefault(f"{feat}/{kind}", {"planted": 0, "caught": 0})
                cell["planted"] += 1
                cell["caught"] += int(not gate(snap, bug.snapshot.evidence).clean)
    planted = sum(c["planted"] for c in cells.values())
    caught = sum(c["caught"] for c in cells.values())
    blind = sorted(k for k, c in cells.items() if c["caught"] == 0)
    return {
        "planted": planted, "caught": caught,
        "recall": None if not planted else round(caught / planted, 4),
        "cells": dict(sorted(cells.items())),
        "never_caught": blind,
        "note": (
            "cells that are never caught are corruptions that contradict nothing in the frame "
            "(a plausible value with no second statement of it); catching them needs a second "
            "source for the same figure, which the desk does not have"
        ),
    }


def run() -> dict[str, Any]:
    snapshots = load_snapshots()
    snaps = {s.id: s for s in snapshots}
    recorded = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    blob = {
        "generated_at": datetime.now(UTC).isoformat(),
        "item": "S17 feed-sanity gate (desk/feed_sanity.py) on the S3 corrupted-feed cases",
        "method": (
            "deterministic; corrupted inputs rebuilt from data/feedbugged.json's recorded kind, "
            "feature and seed and checked against its recorded wrong value; negatives are every "
            "clean snapshot in data/robustness_snapshots.json plus the six correct-reasoning lines"
        ),
        "acceptance": acceptance(snaps, recorded),
        "clean": clean_negatives(snapshots),
        "coverage_matrix": coverage(snapshots),
        "qwen_requests": 0,
    }
    write(OUT_PATH, blob)
    return blob


def main() -> int:  # pragma: no cover - CLI
    blob = run()
    print(json.dumps({"acceptance": blob["acceptance"]["detection"],
                      "clean_false_flags": blob["clean"]["false_flags"],
                      "coverage_recall": blob["coverage_matrix"]["recall"],
                      "never_caught": blob["coverage_matrix"]["never_caught"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["OUT_PATH", "acceptance", "clean_negatives", "coverage", "gate", "items", "run"]
