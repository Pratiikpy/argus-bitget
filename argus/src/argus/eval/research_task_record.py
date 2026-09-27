"""The worked research task the submission quotes, run live and kept, so its figures have a source.

The draft's example ("I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?") was a run of
`/research` on 2026-09-26 copied into the form by hand, and nothing kept it (audit finding 99):
every figure in the paragraph had to be taken on trust, and the next run, on different market
data, could not be compared with it. This runs the same task through the same code the page uses
(`lui/task.read_question` then `lui/task.research_task`, no model call), and writes the whole of
it — the reading, the verdict, every engine's lines and the data each computed them from — to
``data/research_task_example.json``.

The figures are live data and move with the market, so a re-run is expected to change them; what
must never happen is the draft quoting figures this file does not hold. `eval/docclaims.py` checks
the verdict's sizing and fill-cost figures in the draft against :func:`headline` (claims
``task_sizing``, ``task_crowding`` and ``task_fill_cost``).

    python -m argus.eval.research_task_record     # needs the network: Bitget, SEC, Yahoo, news
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parents[3]
REPORT_PATH = PACKAGE / "data" / "research_task_example.json"
QUESTION = "I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?"


def record(question: str = QUESTION) -> dict[str, Any]:
    """Run the task live and return everything it produced, JSON-ready."""
    from argus.lui.task import as_dict, read_question, research_task

    reading = read_question(question)
    if isinstance(reading, str):
        raise ValueError(f"the question was not read: {reading}")
    task = research_task(reading=reading, asked=question)
    return {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "written_by": "python -m argus.eval.research_task_record",
        "task": as_dict(task),
        "data": {step.title: json.loads(json.dumps(step.data, default=str))
                 for step in task.steps},
    }


def headline(blob: dict[str, Any]) -> dict[str, float]:
    """The figures the verdict rests on, rounded as the page rounds them: sizing in whole
    percentages, fill costs in basis points to one decimal, the run time in seconds."""
    from argus.lui.task import EXECUTION_TITLE, IMPACT_TITLE

    sizing = blob["data"][IMPACT_TITLE]["sizing"]
    execution = blob["data"][EXECUTION_TITLE]["execution"]
    crowded = sizing.get("crowded") or {}

    def pct(value: Any) -> float:
        return float(round(float(value) * 100))

    return {
        "proposed": pct(sizing["proposed"]),
        "ceiling": pct(sizing["ceiling"]),
        "budget": pct(sizing["budget"]),
        "share_after": pct(sizing["share_after"]),
        "crowded_share": pct(crowded["share"]),
        "crowded_weight": pct(crowded["weight"]),
        "crowded_trim_to": pct(crowded["trim_to"]),
        "seconds": round(float(blob["task"]["seconds"]), 1),
        "sliced_bps": round(float(execution["sliced_bps"]), 1),
        "single_order_bps": round(float(execution["single_order_bps"]), 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    blob = record()
    REPORT_PATH.write_text(json.dumps(blob, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    verdict = blob["task"]["verdict"] or {}
    print(f"{verdict.get('call')} — {blob['task']['seconds']} s")
    for line in verdict.get("lines", []):
        print(f"  {line}")
    print(f"wrote {REPORT_PATH.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
