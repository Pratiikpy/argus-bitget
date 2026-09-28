"""Every component ablation ARGUS has run, in one table: what it removed, what changed, and how much
evidence there is.

**Why this exists.** Three separate studies ask "does this component earn its place": the pipeline
ablations (`eval/ablations.py`: analyst selection, the as-of evidence gate), the risk-gate
population test (`eval/gate_ablation.py`: each Constitution dimension removed in turn over the
recorded risk records), and the analyst panel's promotion gate (`agents/analysts.py`), specified and
never run. Each wrote its own artefact in its own shape, so no page said, in one place, which
components change decisions, which never have, and which have not been tested enough to say
(research/harvest/19-shap.md).

**What was taken, and from where.** The shape of FactorMiner's ``AblationStudy``
(``factorminer/benchmark/runners.py:542-704``, MIT): one runner over a registry of components, one
table out, and a LaTeX export for a written appendix. Not its statistics: every row here is the
exact count its own study recorded (``eval/ablation.DeterministicResult`` — no sampling, so no
p-value), and nothing is re-run or re-estimated. SHAP's Shapley averaging over removal orders was
read and rejected: it answers a per-decision question at thousands of full pipeline runs each,
against a metered model budget.

**How much evidence a row carries** is part of the row, not a footnote: a component removed on
fewer than :data:`MIN_FRAMES` frames is ``TOO FEW FRAMES`` whatever it showed, because an inert gate
on one calm frame is untested, not useless (`eval/ablation.DeterministicResult.inert`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.truth.paths import DATA_DIR

REPORT_PATH = DATA_DIR / "ablation_report.json"
SOURCES = {
    "pipeline": DATA_DIR / "ablations.json",
    "risk gates": DATA_DIR / "population_rct.json",
}
MIN_FRAMES = 10
"""Fewest frames before a component's row is read as a finding rather than a sample of one."""

NOT_RUN = (
    ("analyst panel promotion gate (agents/analysts.py)",
     "specified as a paired comparison and never executed on recorded chains"),
)


@dataclass(frozen=True)
class Row:
    study: str
    component: str
    frames: int
    changed: int
    risk_violations: int | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {"study": self.study, "component": self.component, "frames": self.frames,
                "changed": self.changed, "risk_violations": self.risk_violations,
                "status": self.status}


def _status(frames: int, changed: int, recorded: str | None) -> str:
    if frames == 0 or recorded == "UNTESTED":
        return "UNTESTED"
    if frames < MIN_FRAMES:
        return "TOO FEW FRAMES"
    return "CHANGES DECISIONS" if changed else "INERT"


def rows(sources: dict[str, Path] = SOURCES) -> tuple[list[Row], list[str]]:
    """Every recorded ablation, and the studies that could not be read."""
    out: list[Row] = []
    missing: list[str] = []
    for study, path in sources.items():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            missing.append(f"{study}: {path.name} unreadable")
            continue
        results = blob.get("results")
        if results is None:
            results = list((blob.get("dimensions") or {}).values())
        for result in results:
            frames = int(result.get("frames", 0))
            changed = int(result.get("changed", 0))
            violations = result.get("risk_violations")
            out.append(Row(study=study, component=str(result.get("component")), frames=frames,
                           changed=changed,
                           risk_violations=None if violations is None else int(violations),
                           status=_status(frames, changed, result.get("status"))))
    return out, missing


def report(sources: dict[str, Path] = SOURCES) -> dict[str, Any]:
    found, missing = rows(sources)
    counts: dict[str, int] = {}
    for row in found:
        counts[row.status] = counts.get(row.status, 0) + 1
    return {
        "rows": [r.as_dict() for r in found],
        "not_run": [{"component": c, "why": why} for c, why in NOT_RUN],
        "unreadable": missing,
        "by_status": counts,
        "min_frames": MIN_FRAMES,
        "reading": (
            f"{counts.get('CHANGES DECISIONS', 0)} component(s) change decisions on at least "
            f"{MIN_FRAMES} frames; {counts.get('INERT', 0)} are inert on that many; "
            f"{counts.get('TOO FEW FRAMES', 0) + counts.get('UNTESTED', 0)} have too little "
            f"evidence to say; {len(NOT_RUN)} specified and never run."),
    }


def _tex(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(a, b)
    return text


def to_latex(blob: dict[str, Any]) -> str:
    """The table as a LaTeX ``tabular``, for a written appendix (FactorMiner's ``to_latex_table``
    idea; the columns are ours)."""
    lines = [r"\begin{tabular}{llrrrl}", r"\hline",
             r"Study & Component & Frames & Changed & Wider risk & Status \\", r"\hline"]
    for r in blob["rows"]:
        wider = "--" if r["risk_violations"] is None else str(r["risk_violations"])
        lines.append(f"{_tex(r['study'])} & {_tex(r['component'])} & {r['frames']} & "
                     f"{r['changed']} & {wider} & {_tex(r['status'])} \\\\")
    for r in blob["not_run"]:
        lines.append(f"-- & {_tex(r['component'])} & -- & -- & -- & NOT RUN \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write(path: Path = REPORT_PATH) -> dict[str, Any]:
    blob = report()
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8", newline="\n")
    path.with_suffix(".tex").write_text(to_latex(blob), encoding="utf-8", newline="\n")
    return blob


def main() -> int:  # pragma: no cover - CLI
    blob = write()
    print(blob["reading"])
    print(f"written to {REPORT_PATH} and {REPORT_PATH.with_suffix('.tex')}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["MIN_FRAMES", "NOT_RUN", "REPORT_PATH", "Row", "report", "rows", "to_latex", "write"]
