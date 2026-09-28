"""How often each data surface answered, over time — not only whether it answers right now.

**Why this exists.** `/status` probes Bitget's market API, the Bitget MCP server and a
`bitget-signal` Skill on every load, and forgot each answer the moment the page was drawn: an outage
last night and a clean record were indistinguishable by morning (research/harvest/54-upptime.md).
Upptime keeps each probe as a committed row and publishes a rolling percentage per endpoint
(``api/<name>/uptime-{day,week}.json``, the shape read from its generated artefacts); this keeps the
same shape, and the rows the way ARGUS keeps every record it publishes: each carries the hash of the
one before, so a failure cannot be quietly deleted from the history (``verify``).

**Where it runs.** The console is served from a read-only serverless host, so the page cannot write
the history as it probes. The scheduled local cycle records a probe round (``python -m
argus.lui.status_page --record``), the history is published with the rest of `data/`, and `/status`
reads it: the live table is as of this page load, the uptime as of the last recorded round.

**Not ported.** Upptime's percentage is computed by its ``uptime-monitor`` action, whose source was
not read (NOT VERIFIED); the arithmetic here is ours and is the plain one: answered probes over all
probes in the window, per surface, with the probe count shown beside it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from argus.truth.paths import DATA_DIR

HISTORY_PATH = DATA_DIR / "status_history.jsonl"
GENESIS = "0" * 64
WINDOWS = (("24 h", timedelta(hours=24)), ("7 d", timedelta(days=7)))


def _hash(row: dict[str, Any]) -> str:
    body = {k: v for k, v in row.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                          ).hexdigest()


def read(path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def record(checks: Iterable[Any], *, at: datetime, path: Path = HISTORY_PATH) -> int:
    """Append one row per probe, each chained to the row before. Returns how many were written."""
    rows = read(path)
    prev = rows[-1]["hash"] if rows else GENESIS
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as out:
        for check in checks:
            row = {"at": at.isoformat(timespec="seconds"), "surface": check.surface,
                   "what": check.what, "ok": bool(check.ok), "ms": round(float(check.ms), 1),
                   "prev": prev}
            row["hash"] = _hash(row)
            out.write(json.dumps(row) + "\n")
            prev = row["hash"]
            written += 1
    return written


def verify(rows: Sequence[dict[str, Any]]) -> str | None:
    """``None`` when the chain holds, else the first row that breaks it and how."""
    prev = GENESIS
    for number, row in enumerate(rows, 1):
        if row.get("prev") != prev:
            return f"row {number}: does not follow the row before it"
        if _hash(row) != row.get("hash"):
            return f"row {number}: its contents were changed after it was written"
        prev = row["hash"]
    return None


def uptime(rows: Sequence[dict[str, Any]], *, now: datetime) -> list[dict[str, Any]]:
    """Per surface and check: answered probes over all probes, in each window."""
    keys: dict[tuple[str, str], None] = {}
    for row in rows:
        keys[(str(row.get("surface")), str(row.get("what")))] = None
    out = []
    for surface, what in keys:
        mine = [r for r in rows if r.get("surface") == surface and r.get("what") == what]
        entry: dict[str, Any] = {"surface": surface, "what": what}
        for label, span in WINDOWS:
            recent = [r for r in mine if now - datetime.fromisoformat(str(r["at"])) <= span]
            ok = sum(1 for r in recent if r.get("ok"))
            entry[label] = {"probes": len(recent),
                            "uptime": round(100 * ok / len(recent), 1) if recent else None}
        out.append(entry)
    return out


__all__ = ["HISTORY_PATH", "WINDOWS", "read", "record", "uptime", "verify"]
