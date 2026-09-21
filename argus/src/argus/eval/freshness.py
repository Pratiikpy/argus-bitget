"""Is this artefact older than the record it was computed from?

**The defect this closes was found by accident.** On 2026-09-21 the paper ledger stood at 495
decisions while the documents quoted studies computed at 447. Re-running one of them —
`eval/refusal.py`, the study the README calls *"the whole thesis measured on our own record"* —
moved two of its four headline figures: the 2h directional accuracy rose from 58.2% to 59.6% and
the overnight horizon **fell from 51.2% to 40.4%, below a coin flip**. Neither number was wrong
when it was written. Both were stale, and nothing said so.

`eval/docclaims.py` cannot catch this. It compares a quoted figure against the artefact, and a
stale artefact agrees with a stale document perfectly — the pair is consistent and both are
describing a record that has moved on. `eval/artefactkind.py` cannot catch it either: that sorts
artefacts by whether re-running them *should* reproduce, which is a different question from
whether anybody has re-run them.

So this asks the third question: **for every artefact derived from the decision ledger, was it
computed after the last decision it claims to describe?**

**Attribution is by ownership, not by mention.** An artefact belongs to the module that declares
it as its own ``REPORT_PATH``. A first attempt matched any module whose source mentioned the
filename, which attributed nineteen artefacts to `eval/docclaims.py` and `demo/cockpit.py` —
modules that *read* almost everything and write almost none of it. A reader chasing one of those
would have found no code to re-run.

**Staleness here is a prompt, not a defect.** An artefact older than the ledger is not wrong; it
is a measurement of an earlier record, and for some studies that is exactly right — a dated
finding is a legitimate thing to publish. What is not legitimate is *not knowing*. This prints
which ones are behind and by how long, so re-running is a decision rather than an oversight.

    python -m argus.eval.freshness
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

ARGUS = Path(__file__).resolve().parents[3]
SRC = ARGUS / "src" / "argus"
DATA = ARGUS / "data"
REPORT_PATH = DATA / "artefact_freshness.json"
LEDGER = DATA / "paper_ledger.jsonl"

_OWNS = re.compile(
    r"^[A-Z_]*(?:REPORT_PATH|OUTPUT_PATH)\s*(?::[^=\n]+)?=\s*[^\n]*?[\"']"
    r"([A-Za-z0-9_]+\.(?:json|jsonl))[\"']",
    re.MULTILINE,
)
"""A module declaring an artefact as its own output.

Ownership, not mention. `eval/docclaims.py` names almost every artefact in the repository because
its job is to read them; attributing them to it would send a reader to the wrong module."""

_READS_LEDGER = re.compile(r"PaperLedger\s*\(|evaluate_ledger\s*\(|LEDGER_PATH")

_STAMPS = ("generated_at", "as_of", "measured_at", "synced_at")


class FreshnessError(RuntimeError):
    """The comparison cannot be made. Raised rather than reported as fresh."""


@dataclass(frozen=True, slots=True)
class Artefact:
    """One ledger-derived artefact and how far behind the record it is."""

    name: str
    owner: str
    stamped_at: str | None
    behind_hours: float | None

    @property
    def dated(self) -> bool:
        return self.stamped_at is not None

    @property
    def stale(self) -> bool:
        """Computed before the ledger's most recent decision."""
        return self.behind_hours is not None and self.behind_hours > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "artefact": self.name,
            "owner": self.owner,
            "stamped_at": self.stamped_at,
            "behind_hours": (
                round(self.behind_hours, 1) if self.behind_hours is not None else None
            ),
            "stale": self.stale,
            "rerun": f"python -m {self.owner}" if self.owner else None,
        }


def _last_decision() -> datetime:
    if not LEDGER.is_file():
        raise FreshnessError(f"no ledger at {LEDGER}; nothing to be stale against")
    stamps = [
        str(json.loads(line)["decided_at"])
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not stamps:
        raise FreshnessError("the ledger is empty; an empty record makes everything look current")
    return datetime.fromisoformat(max(stamps))


def _owners() -> dict[str, str]:
    """``artefact -> dotted module path``, for modules that declare it as their own output."""
    out: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not _READS_LEDGER.search(text):
            continue
        dotted = "argus." + path.relative_to(SRC).with_suffix("").as_posix().replace("/", ".")
        for name in _OWNS.findall(text):
            out.setdefault(name, dotted)
    return out


def survey() -> list[Artefact]:
    """Every ledger-derived artefact, with how far behind the record it was computed."""
    last = _last_decision()
    found: list[Artefact] = []
    for name, owner in sorted(_owners().items()):
        path = DATA / name
        if not path.is_file():
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(blob, dict):
            continue
        stamp = next(
            (blob[k] for k in _STAMPS if isinstance(blob.get(k), str)), None
        )
        behind: float | None = None
        if stamp:
            try:
                behind = (last - datetime.fromisoformat(stamp)).total_seconds() / 3600.0
            except ValueError:
                stamp = None
        found.append(Artefact(name=name, owner=owner, stamped_at=stamp, behind_hours=behind))
    return found


def run() -> dict[str, Any]:
    last = _last_decision()
    artefacts = survey()
    stale = [a for a in artefacts if a.stale]
    undated = [a for a in artefacts if not a.dated]
    return {
        "last_decision": last.isoformat(),
        "ledger_derived_artefacts": len(artefacts),
        "stale": len(stale),
        "undated": len(undated),
        "stale_detail": [a.as_dict() for a in sorted(
            stale, key=lambda a: -(a.behind_hours or 0)
        )],
        "undated_names": [a.name for a in undated],
        "detail": [a.as_dict() for a in artefacts],
        "why_this_matters": (
            "eval/docclaims.py compares a quoted figure against its artefact, and a stale artefact "
            "agrees with a stale document perfectly — the pair is consistent and both describe a "
            "record that has moved on. Re-running eval/refusal.py after the ledger grew from 447 "
            "to 495 moved its 2h accuracy from 58.2% to 59.6% and dropped its overnight horizon "
            "from 51.2% to 40.4%, below a coin flip. Neither figure was wrong when written."
        ),
        "scope_statement": (
            "An artefact is attributed to the module that declares it as its own REPORT_PATH, not "
            "to every module that mentions the filename. NOT CLAIMED: that a stale artefact is "
            "wrong — it measures an earlier record, and a dated finding is a legitimate thing to "
            "publish. What it must not be is unnoticed. NOT CLAIMED: that re-running is always "
            "right; some studies are deliberately pinned to a window, and this reports the lag so "
            "that keeping it becomes a decision."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    lines = [
        f"ARTEFACT FRESHNESS — against the last decision "
        f"{report['last_decision'][:19]}",
        f"  {report['ledger_derived_artefacts']} ledger-derived artefact(s): "
        f"{report['stale']} stale, {report['undated']} undated",
    ]
    if report["stale_detail"]:
        lines.append("")
        for row in report["stale_detail"]:
            lines.append(
                f"  {row['behind_hours']:8.1f}h behind  {row['artefact']:30} "
                f"{row['rerun'] or ''}"
            )
    if report["undated_names"]:
        lines += [
            "",
            "  carry no timestamp, so staleness cannot be judged: "
            + ", ".join(report["undated_names"]),
        ]
    lines.append(
        "  Stale is a prompt, not a defect: an earlier measurement of an earlier record is a "
        "legitimate thing to publish, and not knowing it is behind is not."
    )
    return lines


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = run()
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["REPORT_PATH", "Artefact", "FreshnessError", "main", "render", "run", "survey"]
