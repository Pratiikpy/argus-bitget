"""HONESTY-EVAL — `lui/honesty.py` graded by the infeasibility bench's own grader, and its false
positives.

**What it measures.** (a) Every infeasible case of ``data/infeasibility_bench.json`` is re-asked of
:func:`argus.lui.honesty.honest_answer` (and :func:`~argus.lui.honesty.order_prefix` for orders);
where it answers, its lines replace the console's recorded reply, where it does not, the recorded
reply stands, and each is graded by `eval/infeasibilitybench.grade_infeasible` — the same
vocabulary that produced the "before" counts, so the two columns are comparable. (b) Detection is
run over every question of the three 2026-09-25 LUI corpora (the rows marked answerable; the
``refuse`` rows are counted separately as coverage) and over the bench's 72 controls: any hit on
an answerable question is a false positive and is listed verbatim.

Not measured here: the reply as served through `lui/server._answer` (this module is not wired in
yet), the Qwen path, and whether the figures the two data builders print are correct beyond the
arithmetic tested in ``tests/test_honesty.py``.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval import artefact
from argus.eval import infeasibilitybench as ib
from argus.lui import honesty

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "honesty_eval.json"
CORPORA = tuple(ROOT / "data" / f"lui_{name}_2026-09-25.jsonl"
                for name in ("blind_corpus", "final_heldout", "heldout_corpus"))
BENCH_CAUSE = {
    honesty.PRIVATE_ACCOUNT: "private_data", honesty.OTHERS_POSITIONS: "private_data",
    honesty.FUTURE_PRICE: "future_price", honesty.BEFORE_DATA: "before_data",
    honesty.PAST_PRICE: "before_data", honesty.HORIZON: "horizon_exceeds_data",
    honesty.LONG_HORIZON: "horizon_exceeds_data", honesty.UNLISTED: "unlisted",
    honesty.AMBIGUOUS: "ambiguous_name", honesty.ORDER_INSTRUCTION: "order",
}


def regrade(*, data: bool = True) -> dict[str, Any]:
    """Before (the artefact) and after (this module's reply where it gives one) per cause."""
    report = json.loads(ib.REPORT_PATH.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in report["rows"] if r["group"] == "infeasible"}
    cases = {c.id: c for c in ib.INFEASIBLE}
    before: Counter[tuple[str, str]] = Counter()
    after: Counter[tuple[str, str]] = Counter()
    graded: list[dict[str, Any]] = []
    for cid, row in rows.items():
        case = cases[cid]
        if not row["truth"]["valid"]:
            continue
        before[(row["cause"], row["outcome"])] += 1
        started = time.perf_counter()
        answer: tuple[str, list[str]] | None
        if data:
            answer = honesty.honest_answer(case.ask, prior=list(case.prior), book="")
        else:
            found = honesty.detect(case.ask, prior=list(case.prior), book="")
            answer = (found.cause, []) if found else None
        prefix = honesty.order_prefix(case.ask)
        if answer is not None and answer[1]:
            payload: dict[str, Any] = {"refused": True, "reason": "", "lines": answer[1]}
            by = "honesty"
        elif answer is not None and not data:
            payload = {"refused": True, "reason": "", "lines": [f"detected: {answer[0]}"]}
            by = "honesty (detection only)"
        elif prefix is not None:
            payload = {"refused": row["refused"], "reason": row["reason"],
                       "lines": [prefix, *row["lines"]]}
            by = "order_prefix+console"
        else:
            payload = {"refused": row["refused"], "reason": row["reason"], "lines": row["lines"]}
            by = "console (unchanged)"
        grade = ib.grade_infeasible(case, payload, probe=lambda: True)
        outcome = str(grade.outcome)
        if not data and by == "honesty (detection only)":
            outcome = ("detected_right_cause" if BENCH_CAUSE.get(answer[0] if answer else "")
                       == row["cause"] else "detected_other_cause")
        after[(row["cause"], outcome)] += 1
        graded.append({"id": cid, "ask": case.ask, "cause": row["cause"],
                       "before": row["outcome"], "after": outcome, "answered_by": by,
                       "honesty_cause": answer[0] if answer else "",
                       "cause_agrees": (BENCH_CAUSE.get(answer[0]) == row["cause"]
                                        if answer else None),
                       "matched": grade.matched, "lines": payload["lines"],
                       "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
    return {"before": _by_cause(before), "after": _by_cause(after), "rows": graded}


def _by_cause(counts: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for (cause, outcome), n in sorted(counts.items()):
        out.setdefault(cause, {})[outcome] = n
    total: dict[str, int] = {}
    for per in list(out.values()):
        for outcome, n in per.items():
            total[outcome] = total.get(outcome, 0) + n
    out["ALL"] = total
    return out


def false_positives() -> dict[str, Any]:
    """Detection over the answerable held-out questions and the bench's controls."""
    hits: list[dict[str, str]] = []
    refuse: list[dict[str, str]] = []
    scanned = refuse_rows = 0
    started = time.perf_counter()
    for path in CORPORA:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            found = honesty.detect(row["text"], prior=[], book="")
            prefix = honesty.order_prefix(row["text"])
            what = found.cause if found else ("order_prefix" if prefix else "")
            if row["expected"] == "refuse":
                refuse_rows += 1
                if what:
                    refuse.append({"corpus": path.name, "text": row["text"], "hit": what})
                continue
            scanned += 1
            if what:
                hits.append({"corpus": path.name, "kind": row["expected"], "text": row["text"],
                             "hit": what})
    for control in ib.controls():
        scanned += 1
        found = honesty.detect(control.ask, prior=list(control.prior), book=control.book)
        prefix = honesty.order_prefix(control.ask)
        what = found.cause if found else ("order_prefix" if prefix else "")
        if what:
            hits.append({"corpus": "infeasibility_bench controls", "kind": control.kind,
                         "text": control.ask, "hit": what})
    return {"answerable_scanned": scanned, "false_positives": len(hits), "hits": hits,
            "refuse_rows": refuse_rows, "refuse_rows_detected": len(refuse),
            "refuse_detected_by_cause": dict(Counter(r["hit"] for r in refuse)),
            "detection_ms_total": round((time.perf_counter() - started) * 1000, 1)}


CALL_SITE = (
    "argus.lui.server._answer, directly after the multistep block (the `if split is not None:` "
    "return) and before `followed = follow_up(text, prior, book)`: `honest = "
    "honesty.honest_answer(text, prior=prior, book=book)`; when not None, `return "
    "engine_payload(honest[1], [], {'cause': honest[0]}, by='honesty')`. order_prefix: in "
    "_answer, compute `prefix = honesty.order_prefix(text)` once at the top; wherever a payload "
    "is returned (the research payload included), insert it at lines[0] when not None.")


def run(out: Path = OUT) -> dict[str, Any]:
    """Grade, scan, and write the artefact."""
    blob: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "item": "lui/honesty.py — true reasons for questions no console can answer as asked",
        "method": __doc__,
        "detection_offline": regrade(data=False)["after"],
        "regrade": regrade(data=True),
        "false_positive_scan": false_positives(),
        "call_site": CALL_SITE,
    }
    artefact.write(out, blob)
    return blob


if __name__ == "__main__":  # pragma: no cover
    result = run()
    print(json.dumps({"before": result["regrade"]["before"]["ALL"],
                      "after": result["regrade"]["after"]["ALL"],
                      "fp": result["false_positive_scan"]["false_positives"]}, indent=1))
