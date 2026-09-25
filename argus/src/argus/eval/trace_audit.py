"""How much of an answer's per-line provenance the live trace decides, against the wording rules.

`lui/provenance.py` labels a line from its wording after the answer is written. `truth/trace.py`
records, while the answer is being built, which step made each line, what it read and when — and
takes the label from that step where the step declares one. This module measures the difference
on the same answers, rather than asserting that one is better:

* **trace-determined share** — content lines (meta lines such as "Data:" excluded, exactly as
  `eval/provenance_audit.py` excludes them) whose label the trace decides: a declared producer, a
  request default, or a refusal after a failed read.
* **wording share** — content lines the wording rules label today.
* **disagreements** — every line where the two differ, including every line the trace labels and
  the wording rules leave unlabelled. Listed in full, each with the step that made it and why the
  trace says what it says, so a reader can judge each one instead of trusting a rate.
* **producer coverage** — content lines whose producing step the trace found at all, labelled or
  not; the rest were composed inline by a function too big to declare.
* **the worklist** — engines that made lines and declared nothing, by line count: where an
  ``emit(lines, label)`` at the line site in `lui/research.py` would move lines from inferred to
  recorded.
* **overhead** — seconds per answer with the recorder wired in and without it, on a fixed subset.

**How the console runs here.** Offline: every outbound socket is refused (an ``OSError``, so the
console's own degrade paths run, as they would with the venue down), no model key is present, and
the console's trained kind model reads each question as it does on a deployment with no key. The
recorder is wired through a monkeypatched wrapper (:func:`argus.truth.trace.instrumented`), and
`lui/multistep.py`'s plain thread pool is swapped for `truth/coverage.ContextPool` for the run —
the one-line change the integration asks of that module. Nothing here edits the console.

**What offline cannot show.** Every live source fails, so ``live`` lines barely occur: this run
measures the computed, record, desk, assumed and missing paths, and the refusal rule on real
transport failures. A live-network run of the same audit is the missing half.

The corpora are the three held-out routing sets (`data/lui_*_2026-09-25.jsonl`, 680 questions
written by writers who never read this repository).

    python -m argus.eval.trace_audit
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import socket
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "trace_audit.json"
CORPORA = ("lui_final_heldout_2026-09-25.jsonl", "lui_heldout_corpus_2026-09-25.jsonl",
           "lui_blind_corpus_2026-09-25.jsonl")

OVERHEAD_STRIDE = 11
"""Every 11th question is answered twice more, once traced and once not, to time the recorder."""


class NetworkRefused(ConnectionRefusedError):
    """Every outbound connection during an offline run. An ``OSError``, so the console's own
    offline handling runs exactly as it does when a venue is down."""


@dataclass
class Blocked:
    attempts: int = 0
    writes: list[str] = field(default_factory=list)
    """Files under ``data/`` this process opened for writing during the run."""


@contextmanager
def offline() -> Iterator[Blocked]:
    """Refuse every outbound connection and hide every model key for the duration, and note every
    file under ``data/`` this process opens for writing (written, not blocked: other processes in
    the workspace write there too, and only this process's writes are this run's).

    Local connections are allowed (nothing in the console makes one). Restored on exit."""
    blocked = Blocked()
    real_open = io.open

    def noting_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, str | os.PathLike) and any(c in mode for c in "wax+"):
            target = Path(os.fspath(file)).resolve()
            if target.is_relative_to(DATA.resolve()):
                blocked.writes.append(target.name)
        return real_open(file, mode, *args, **kwargs)

    saved_env = {k: v for k, v in os.environ.items() if "QWEN" in k}
    for key in saved_env:
        del os.environ[key]
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection,
                 socket.getaddrinfo)

    def refuse(*_a: Any, **_k: Any) -> Any:
        blocked.attempts += 1
        raise NetworkRefused("outbound connections are refused in an offline run")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
    io.open = noting_open
    builtins.open = noting_open
    try:
        yield blocked
    finally:
        io.open = real_open
        builtins.open = real_open
        (socket.socket.connect, socket.socket.connect_ex,  # type: ignore[method-assign]
         socket.create_connection, socket.getaddrinfo) = originals
        os.environ.update(saved_env)


def _corpus() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in CORPORA:
        path = DATA / name
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append({"corpus": name, "id": row["id"], "text": str(row["text"]),
                             "expected": row.get("expected")})
    return rows


@contextmanager
def _console_wired() -> Iterator[None]:
    """The recorder wired into the console for the duration: every engine a step. The multi-part
    answer's pool carries the trace into its workers itself (`lui/multistep.py` uses
    `truth.coverage.ContextPool` since 2026-09-26; this used to patch it in)."""
    from argus.truth import trace

    with trace.instrumented():
        yield


def _data_state() -> dict[str, tuple[int, int]]:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in DATA.iterdir()
            if p.is_file()}


def audit(limit: int | None = None) -> dict[str, Any]:
    """Answer every corpus question offline, traced, and compare the two labellings."""
    from argus.lui.provenance import label as by_wording
    from argus.lui.server import handle_ask
    from argus.truth import trace

    rows = _corpus()[:limit] if limit else _corpus()
    before = _data_state()
    content = meta = determined = worded = combined = producer_known = 0
    by_engine: Counter[str] = Counter()
    worklist: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    agree = both = trace_only = 0
    disagreements: list[dict[str, Any]] = []
    crashed: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    traced_seconds = 0.0
    started = time.perf_counter()
    with offline() as blocked, _console_wired():
        for row in rows:
            t0 = time.perf_counter()
            with trace.recording(row["text"]) as recorded:
                try:
                    payload = handle_ask(row["text"], [])
                except Exception as exc:  # a console crash offline is a finding, not a stop
                    crashed.append({"corpus": row["corpus"], "id": row["id"],
                                    "question": row["text"],
                                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                                    "last_steps": [s.engine for s in recorded.steps[-4:]]})
                    continue
            traced_seconds += time.perf_counter() - t0
            trace.attach(payload, recorded)
            lines = [str(x) for x in payload.get("lines") or []]
            labelled = trace.line_labels(recorded, lines)
            if example is None and len({x.engine for x in labelled if x.engine}) >= 3:
                example = {"question": row["text"], "lines": lines,
                           "labels": [x.as_dict() for x in labelled],
                           "trace": recorded.render()}
            for item in labelled:
                if item.origin == "meta":
                    meta += 1
                    continue
                content += 1
                wording = by_wording(item.line)
                worded += wording is not None
                combined += item.label is not None
                producer_known += item.engine is not None
                if item.origin == "trace":
                    determined += 1
                    by_engine[str(item.engine)] += 1
                    by_label[str(item.label)] += 1
                    if wording is None:
                        trace_only += 1
                    else:
                        both += 1
                        agree += wording == item.label
                    if wording != item.label:
                        disagreements.append({
                            "corpus": row["corpus"], "id": row["id"], "question": row["text"],
                            "line": item.line, "wording": wording, "trace": item.label,
                            "engine": item.engine, "why": item.reason})
                elif item.engine is not None:
                    worklist[item.engine] += 1
    elapsed = time.perf_counter() - started
    # A crash under the recorder is the console's only if the console, unwired, crashes on the
    # same question too; otherwise it is the recorder's, and that is the finding.
    with offline():
        for crash in crashed:
            try:
                handle_ask(crash["question"], [])
            except Exception as exc:
                crash["without_recorder"] = f"also raises {type(exc).__name__}"
            else:
                crash["without_recorder"] = "answers: the crash is the recorder's"
    recorder_crashes = sum(1 for c in crashed if c["without_recorder"].startswith("answers"))
    changed = sorted(name for name, state in _data_state().items()
                     if before.get(name) != state and name != REPORT_PATH.name)
    written = sorted(set(blocked.writes))

    def share(n: int) -> float | None:
        return round(n / content, 4) if content else None

    answered = len(rows) - len(crashed)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpora": list(CORPORA),
        "questions": len(rows),
        "answered": answered,
        "crashed": crashed,
        "crashes_caused_by_the_recorder": recorder_crashes,
        "lines": {"meta": meta, "content": content},
        "trace_determined": {"lines": determined, "share": share(determined),
                             "by_label": dict(by_label.most_common()),
                             "by_engine": dict(by_engine.most_common())},
        "wording_labelled_today": {"lines": worded, "share": share(worded)},
        "labelled_with_trace_first": {"lines": combined, "share": share(combined)},
        "producer_found": {"lines": producer_known, "share": share(producer_known)},
        "where_both_label": {"lines": both, "agree": agree,
                             "agreement": round(agree / both, 4) if both else None},
        "trace_labels_where_wording_has_none": trace_only,
        "disagreements": disagreements,
        "worklist_undeclared_producers": dict(worklist.most_common()),
        "offline": {"outbound_connections_refused": blocked.attempts,
                    "seconds": round(elapsed, 1),
                    "traced_seconds_per_answer": round(traced_seconds / answered, 3)
                    if answered else None,
                    "data_files_this_run_wrote": written,
                    "data_files_changed_by_any_process": changed},
        "example": example,
        "limitations": [
            "offline: every live source fails, so live lines are barely exercised; a live-network "
            "run of this audit is not done",
            "a declaration is written by reading the engine's code; the disagreements list is "
            "where it should be checked, and it is published in full",
            "lines composed inline by lui/research.py's large functions have no producing step "
            "of their own and keep the wording label; the worklist counts them by engine",
        ],
    }


def overhead(stride: int = OVERHEAD_STRIDE) -> dict[str, Any]:
    """Seconds per answer, the same questions answered with the recorder wired in and without."""
    from argus.lui.server import handle_ask
    from argus.truth import trace

    subset = _corpus()[::stride]
    timings: dict[str, float] = {}
    with offline():
        # a warm-up pass, so neither arm pays for first imports and caches
        for row in subset[:3]:
            handle_ask(row["text"], [])
        t0 = time.perf_counter()
        for row in subset:
            try:
                handle_ask(row["text"], [])
            except Exception:  # the crash list is the audit's; timing counts what ran
                continue
        timings["plain"] = (time.perf_counter() - t0) / len(subset)
        with _console_wired():
            t0 = time.perf_counter()
            for row in subset:
                try:
                    with trace.recording(row["text"]):
                        handle_ask(row["text"], [])
                except Exception:
                    continue
            timings["traced"] = (time.perf_counter() - t0) / len(subset)
    return {"questions": len(subset), "seconds_per_answer": {k: round(v, 3)
                                                             for k, v in timings.items()},
            "ratio": round(timings["traced"] / timings["plain"], 2) if timings["plain"] else None}


def run(limit: int | None = None) -> dict[str, Any]:
    report = audit(limit)
    report["overhead"] = overhead()
    write(REPORT_PATH, report)
    return report


def main() -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="first N questions only")
    args = parser.parse_args()
    report = run(args.limit or None)
    t, w = report["trace_determined"], report["wording_labelled_today"]
    print(f"{report['answered']}/{report['questions']} answered offline, "
          f"{len(report['crashed'])} crashed ({report['crashes_caused_by_the_recorder']} of them "
          f"only with the recorder wired); {report['lines']['content']} content lines")
    print(f"trace decides {t['lines']} ({t['share']:.1%}); wording rules label {w['lines']} "
          f"({w['share']:.1%}); trace first, wording as fallback: "
          f"{report['labelled_with_trace_first']['share']:.1%}")
    both = report["where_both_label"]
    print(f"where both label: {both['agree']}/{both['lines']} agree; trace labels "
          f"{report['trace_labels_where_wording_has_none']} lines the wording rules leave bare; "
          f"{len(report['disagreements'])} disagreements listed")
    print(f"overhead: {report['overhead']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
