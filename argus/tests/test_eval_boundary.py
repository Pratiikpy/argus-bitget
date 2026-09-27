"""The evaluation harness sits at the top of the layer map, beside the console (audit 163).

`eval/architecture.py` puts `eval`, `lui`, `demo` and `status` on one layer: the console answers
"how good is the desk" by running an evaluation, and that is a same-layer call. Everything below
must never reach up into `eval`, including from inside a function, which the layer check reads
past: `research/session_beta_study.py` took the cycle's universe from `eval/clearance.py` until
2026-09-27, and the band's quantile, the default hurdle, the holiday calendar and the kept task's
headline were imported the same way by product code. They now live in `desk/analogue.py`,
`cost/model.py`, `truth/clocks.py` and `lui/task.py`, and this test keeps it so.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "argus"
TOP = ("eval", "lui", "demo", "status.py")


def test_nothing_below_the_top_layer_imports_the_evaluation_harness() -> None:
    upward = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith(TOP) or rel.startswith("vendor/"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("argus.eval"):
                upward.append(f"{rel}:{node.lineno}")
            if isinstance(node, ast.Import) and any(a.name.startswith("argus.eval")
                                                     for a in node.names):
                upward.append(f"{rel}:{node.lineno}")
    assert upward == []
