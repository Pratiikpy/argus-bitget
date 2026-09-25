"""Who used the hosted console, what they asked it to do, and whether it did it.

Pulls the anonymous events `lui/usage.py` writes to the hosted console's function log, keeps every
one it has ever seen in ``data/usage_events.jsonl`` (the host keeps its logs for a limited time,
so the store is what makes the record outlive them), and writes ``data/usage_report.json``:

* **Real use** is a ``browser`` event without the ``internal`` flag. Scripted clients and our own
  marked browsers are counted separately and never in the headline.
* **Visitor-days** — distinct daily visitor hashes, summed over days. The hash rotates daily by
  design, so the same person on two days is two visitor-days and there is no all-time unique count.
* **Answered** — questions not refused. **Task completion** — "Yes" clicks over all clicks on
  "Did this answer your question?", with the number of clicks beside it; with few clicks the rate
  is stated and flagged, not rounded into a claim.

Every figure is *observed*: counted from events, nothing estimated.

    python -m argus.eval.usage_report            # pull from the host, then report
    python -m argus.eval.usage_report --offline  # report from the stored events only
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.artefact import write

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
STORE = DATA / "usage_events.jsonl"
REPORT_PATH = DATA / "usage_report.json"
VERCEL_DIR = Path(os.environ.get("ARGUS_VERCEL_DIR") or ROOT.parent / "deploy")
FEW_CLICKS = 30
"""Below this many feedback clicks the completion rate is reported with ``thin: true``."""


def events_in(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The usage events inside one line of ``vercel logs --json``."""
    messages = [str(record.get("message") or "")]
    messages += [str(x.get("message") or "") for x in record.get("logs") or []
                 if isinstance(x, dict)]
    found = []
    for message in messages:
        for line in message.splitlines():
            line = line.strip()
            if not line.startswith('{"argus_usage"'):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("kind") in ("ask", "feedback"):
                found.append(event)
    return found


def pull(since: str = "72h", limit: int = 5000) -> list[dict[str, Any]]:
    """Every usage event the host still holds. Raises when the CLI is missing or fails — a pull
    that silently returns nothing would read as "nobody used it"."""
    exe = shutil.which("vercel")
    if exe is None:
        raise RuntimeError("the vercel CLI is not on PATH; run with --offline to report the store")
    done = subprocess.run(
        [exe, "logs", "--json", "--since", since, "-n", str(limit), "--no-branch", "--expand",
         "--environment", "production"],
        cwd=VERCEL_DIR, capture_output=True, text=True, encoding="utf-8", timeout=300,
        check=False)
    if done.returncode != 0:
        raise RuntimeError(f"vercel logs failed: {done.stderr.strip()[:300]}")
    found: list[dict[str, Any]] = []
    for line in done.stdout.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            found.extend(events_in(record))
    return found


def _key(event: dict[str, Any]) -> tuple[str, str]:
    return str(event.get("kind")), str(event.get("id"))


def load() -> list[dict[str, Any]]:
    if not STORE.exists():
        return []
    rows = []
    for line in STORE.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def store(new: list[dict[str, Any]]) -> int:
    """Append the events not already stored; returns how many were new. One answer is asked once
    and clicked at most once per kind, so (kind, id) identifies an event."""
    seen = {_key(e) for e in load()}
    fresh = []
    for event in new:
        if _key(event) not in seen:
            seen.add(_key(event))
            fresh.append(event)
    if fresh:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        with STORE.open("a", encoding="utf-8") as handle:
            for event in sorted(fresh, key=lambda e: str(e.get("at"))):
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    return len(fresh)


def _real(event: dict[str, Any]) -> bool:
    return event.get("client") == "browser" and not event.get("internal")


def summarise(events: list[dict[str, Any]]) -> dict[str, Any]:
    asks = [e for e in events if e.get("kind") == "ask"]
    clicks = [e for e in events if e.get("kind") == "feedback"]
    real_asks = [e for e in asks if _real(e)]
    real_ids = {e["id"] for e in real_asks}
    # a click counts as real when its answer was a real question (the click itself carries the
    # same flags, but the answer is the better witness: one person, one browser)
    real_clicks = [e for e in clicks if e.get("id") in real_ids and _real(e)]
    days: dict[str, set[str]] = defaultdict(set)
    for event in real_asks:
        days[str(event.get("at", ""))[:10]].add(str(event.get("v")))
    answered = [e for e in real_asks if not e.get("refused")]
    yes = sum(1 for e in real_clicks if e.get("useful"))
    latency = [float(e.get("ms") or 0) for e in real_asks]
    return {
        "label": "observed",
        "first_event": min((str(e.get("at")) for e in real_asks), default=None),
        "last_event": max((str(e.get("at")) for e in real_asks), default=None),
        "days_with_use": len(days),
        "visitor_days": sum(len(v) for v in days.values()),
        "by_day": {d: {"visitors": len(v),
                       "questions": sum(1 for e in real_asks if str(e.get("at", ""))[:10] == d)}
                   for d, v in sorted(days.items())},
        "questions": len(real_asks),
        "answered": len(answered),
        "answered_rate": round(len(answered) / len(real_asks), 3) if real_asks else None,
        "answers_with_a_missing_line": sum(1 for e in answered if e.get("missing")),
        "median_ms": round(statistics.median(latency)) if latency else None,
        "feedback_clicks": len(real_clicks),
        "feedback_yes": yes,
        "task_completion_rate": round(yes / len(real_clicks), 3) if real_clicks else None,
        "thin": len(real_clicks) < FEW_CLICKS,
        "by_engine": dict(Counter(str(e.get("route")) for e in real_asks).most_common()),
        "excluded": {"automated": sum(1 for e in asks if e.get("client") != "browser"),
                     "internal": sum(1 for e in asks if e.get("client") == "browser"
                                     and e.get("internal"))},
    }


def run(offline: bool = False) -> dict[str, Any]:
    pulled: int | None = None
    error = ""
    if not offline:
        try:
            pulled = store(pull())
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
    report = {"generated_at": datetime.now(UTC).isoformat(), "new_events_pulled": pulled,
              "pull_error": error, "stored_events": len(load()), **summarise(load())}
    write(REPORT_PATH, report)
    return report


def main() -> int:  # pragma: no cover - CLI
    report = run(offline="--offline" in sys.argv)
    if report["pull_error"]:
        print("pull failed:", report["pull_error"])
    print(f"stored {report['stored_events']} events ({report['new_events_pulled']} new)")
    print(f"real use: {report['questions']} questions, {report['visitor_days']} visitor-days "
          f"over {report['days_with_use']} days; answered {report['answered']}")
    print(f"feedback: {report['feedback_yes']}/{report['feedback_clicks']} said answered"
          + (" (thin)" if report["thin"] else ""))
    print("excluded:", report["excluded"])
    return 1 if report["pull_error"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
