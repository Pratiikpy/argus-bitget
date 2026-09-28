"""A kill switch that does not depend on the thing it has to stop.

**Why this exists.** Every control ARGUS had over its own orders ran inside the decision pipeline:
the Constitution, the circuit breaker, the pause store. A failure of that pipeline — the case a
kill switch exists for — would take them all with it (research/harvest/27-resilience4j-breakers.md,
decision 2). FIA's automated-trading guidance asks for order-cancel capability as "risk mitigation
functionality in the event of an electronic trading system failure"; vigil, a Bitget S2 entrant,
keeps a file it checks before every send and that only a human clears; resilience4j offers
``transitionToForcedOpenState()``, which is in-process and so rejected here for the reason FIA
gives.

**How it works.** A file, ``data/KILL`` (or ``ARGUS_KILL_FILE``). While it exists, the two doors to
a venue — ``BitgetTradingClient.place_order`` and ``OrderBook.submit`` — refuse every order before
building a payload, and the paper runner stops a cycle before deciding anything. Nothing in the
pipeline can clear it: engaging and releasing are this module's command, run by a person, each
recorded with who and why in ``data/kill_events.jsonl``. A kill file that cannot be read is treated
as engaged: an unreadable stop is still a stop.

    python -m argus.execution.kill engage --by alice --why "venue returned fills we did not send"
    python -m argus.execution.kill status
    python -m argus.execution.kill release --by alice --why "reconciled; fills were a replay"
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.truth.paths import DATA_DIR

ENV = "ARGUS_KILL_FILE"
DEFAULT_PATH = DATA_DIR / "KILL"
EVENTS = DATA_DIR / "kill_events.jsonl"


class KillSwitchEngaged(RuntimeError):
    """An order was refused because a person engaged the kill switch."""


def path() -> Path:
    return Path(os.environ.get(ENV) or DEFAULT_PATH)


def engaged(at: Path | None = None) -> dict[str, Any] | None:
    """The engagement record while the switch is on, else ``None``."""
    target = at or path()
    if not target.exists():
        return None
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else {"why": "unreadable kill file"}
    except (OSError, ValueError):
        return {"why": "unreadable kill file", "by": "unknown"}


def check(at: Path | None = None) -> None:
    """Raise :class:`KillSwitchEngaged` while the switch is on; the order doors call this first."""
    record = engaged(at)
    if record is not None:
        raise KillSwitchEngaged(
            f"kill switch engaged by {record.get('by', 'unknown')} at {record.get('at', '?')}: "
            f"{record.get('why', '')} (release with python -m argus.execution.kill release)")


def _log(event: str, by: str, why: str, when: datetime, events: Path) -> None:
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps({"event": event, "by": by, "why": why,
                              "at": when.isoformat(timespec="seconds")}) + "\n")


def engage(by: str, why: str, *, at: Path | None = None, events: Path = EVENTS,
           now: datetime | None = None) -> dict[str, Any]:
    if not by.strip() or not why.strip():
        raise ValueError("say who is engaging the kill switch and why")
    when = now or datetime.now(UTC)
    record = {"by": by.strip(), "why": why.strip(), "at": when.isoformat(timespec="seconds")}
    target = at or path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record), encoding="utf-8", newline="\n")
    _log("engaged", record["by"], record["why"], when, events)
    return record


def release(by: str, why: str, *, at: Path | None = None, events: Path = EVENTS,
            now: datetime | None = None) -> None:
    if not by.strip() or not why.strip():
        raise ValueError("say who is releasing the kill switch and why")
    target = at or path()
    if not target.exists():
        raise ValueError("the kill switch is not engaged")
    target.unlink()
    _log("released", by.strip(), why.strip(), now or datetime.now(UTC), events)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    cli = argparse.ArgumentParser(description="ARGUS's out-of-band kill switch")
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("engage", "release"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--by", required=True)
        cmd.add_argument("--why", required=True)
    sub.add_parser("status")
    args = cli.parse_args(argv)
    if args.command == "status":
        record = engaged()
        print("engaged: " + json.dumps(record) if record else "not engaged")
        return 0
    if args.command == "engage":
        print("engaged: " + json.dumps(engage(args.by, args.why)))
        return 0
    release(args.by, args.why)
    print("released")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["ENV", "KillSwitchEngaged", "check", "engage", "engaged", "path", "release"]
