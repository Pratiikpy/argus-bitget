"""How much of an answer the per-line provenance labels cover, and whether they still say what a
person checked them to say.

`lui/provenance.py` labels a line from its wording and leaves a line no rule recognises
unlabelled. Two figures follow from that, and both are published rather than assumed:

* **coverage** — the share of content lines (meta lines such as "Data:" excluded) that carry a
  label. An unlabelled content line is not wrong, but it is a line the reader gets no help with.
* **agreement** — against ``data/provenance_sample.json``: 229 lines from 27 answers across every
  kind of question, collected 2026-09-25 with the model router off, each label read and corrected
  by hand. **Those labels were reviewed after the rules were written**, so agreement is a
  regression check on a reviewed sample, not an independent estimate of accuracy on new wording;
  coverage on answers collected later is the figure that tests generalisation. Two later samples
  are kept: ``provenance_fresh.json`` (69% on first read, then used to add rules) and
  ``provenance_unseen.json`` (94.2% on its one unseen read, 156 content lines from 18 questions).

    python -m argus.eval.provenance_audit
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from argus.lui.provenance import _META, label

SAMPLE = Path(__file__).resolve().parents[3] / "data" / "provenance_sample.json"
REPORT = Path(__file__).resolve().parents[3] / "data" / "provenance_audit.json"


def audit(sample: Path = SAMPLE) -> dict[str, Any]:
    rows = json.loads(sample.read_text("utf-8"))["lines"]
    content = [r for r in rows if not _META.search(str(r["text"]).strip())]
    got = [(r, label(r["text"])) for r in content]
    agree = [r for r, lab in got if lab == r["gold"]]
    confusion = Counter(f"{r['gold']} -> {lab}" for r, lab in got if lab != r["gold"])
    return {
        "sample": sample.name, "lines": len(rows), "content_lines": len(content),
        "coverage": round(sum(lab is not None for _, lab in got) / len(content), 3),
        "agreement": round(len(agree) / len(content), 3),
        "by_label": dict(Counter(str(lab) for _, lab in got)),
        "disagreements": dict(confusion),
        "limitations": "labels hand-reviewed after the rules were written: a regression check, "
                       "not an independent accuracy estimate",
    }


def coverage(path: Path) -> dict[str, Any]:
    """Coverage alone, for a sample with no hand labels."""
    blob = json.loads(path.read_text("utf-8"))
    content = [r for r in blob["lines"] if not _META.search(str(r["text"]).strip())]
    labelled = sum(label(r["text"]) is not None for r in content)
    return {"sample": path.name, "content_lines": len(content),
            "coverage_now": round(labelled / len(content), 3) if content else None,
            "first_read": blob.get("first_read"), "note": blob.get("note")}


def main() -> int:  # pragma: no cover - CLI
    report = audit()
    report["later_samples"] = [coverage(SAMPLE.parent / name) for name in
                               ("provenance_fresh.json", "provenance_unseen.json")]
    REPORT.write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
