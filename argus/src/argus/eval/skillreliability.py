"""Is a data source *reliable*, or did it merely answer the one time we asked?

**One probe cannot tell a dead service from a flaky one, and this project has now been caught by
that twice in one day.** `market/skills.py` swept Bitget's five official research Skills on
2026-09-20 and recorded 6 of 19 tools `ok`, 10 `empty`, 3 `tool_error`. Re-run on 2026-09-21 the
same sweep recorded 6 `ok` and **13 `timeout`** — the same six answering, and the other thirteen
failing for an entirely different stated reason. Both artefacts were written by the same code
against the same endpoint. Only one of them can be describing the service.

Neither was wrong. The service is **intermittent**, and a single snapshot is structurally unable
to say so: it reports whichever failure mode happened to occur, with the authority of a
measurement. Track 3 scores *"data sources / Skill integration count **and effectiveness**"*, and
"effectiveness" measured once is a coin toss written down.

**So this calls every tool N times and reports the distribution.** A tool that answers 5 of 5 is a
dependency. One that answers 2 of 5 is a liability you can plan around. One that answers 0 of 5
across separated attempts is genuinely down. Those are three different facts and a single sweep
collapses them into one word.

**What was established directly before building this**, because the obvious conclusion was wrong:

* Both hosts answer a handshake in under a second — `agent.bitget.com` in 0.71s,
  `datahub.noxiaohao.com` (which is where `bitget-signal` actually points, and it is not a Bitget
  domain) in 0.32s. The transport is not the problem.
* All **19 tools are present** on a direct `tools/list`.
* Calling the failing ones by hand: `macro_indicators` answers in 20.5s with ``{"error": ""}``,
  `rates_yields` in 20.4s with every leg carrying an empty error, `news_feed` in 20.7s with empty
  item arrays, `sentiment_index` in 30.3s with ``{"alt_me_error": ""}``, and `crypto_market`
  returns ``ConnectTimeout`` explicitly.

That last one is the finding: **Bitget's service is timing out against its own upstream data
providers**, and returning a well-formed envelope containing an empty error. Our client is correct;
the 45-second timeout is correct and matches the measured 20-31s replies. The unreliability is
upstream, and the honest response is to measure it rather than to route around it or to quote
whichever snapshot flatters the integration count.

**This changes nothing about what the desk does.** `market/skills.py` already refuses to treat
silence as information, and `agents/analysts.py` already records which sources reached a decision.
This measures how often the door is open.

    python -m argus.eval.skillreliability --attempts 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.artefact import write
from argus.market.skills import DEFAULT_TIMEOUT, PROBES, Health, Probe, _classify

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "skill_reliability.json"

ATTEMPTS = 3
"""Calls per tool. Three is the minimum that can distinguish always / sometimes / never.

Not more by default, because the slow tools take 20-31 seconds each and nineteen tools times five
attempts is a twenty-minute sweep against somebody else's service. `--attempts` raises it when a
stronger claim is wanted."""

SPACING = 2.0
"""Seconds between attempts on one tool.

Separated deliberately: back-to-back calls share whatever transient condition is causing the
failure, so three failures in three seconds is closer to one observation than to three."""


@dataclass(slots=True)
class ToolReliability:
    """One tool, called repeatedly."""

    skill: str
    tool: str
    action: str
    attempts: int = 0
    answered: int = 0
    latencies: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.answered / self.attempts if self.attempts else 0.0

    @property
    def verdict(self) -> str:
        """Three states, and the middle one is the reason this module exists.

        A tool that answers sometimes is not a tool that works, and it is not a tool that is down.
        Calling it either would be a claim the evidence does not support."""
        if self.attempts == 0:
            return "not_probed"
        if self.answered == self.attempts:
            return "reliable"
        if self.answered == 0:
            return "down"
        return "intermittent"

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill, "tool": self.tool, "action": self.action,
            "attempts": self.attempts, "answered": self.answered,
            "rate": round(self.rate, 4), "verdict": self.verdict,
            "median_latency_s": (
                round(statistics.median(self.latencies), 1) if self.latencies else None
            ),
            "failures": self.failures[:3],
        }


def measure(
    client: Any, *, symbol: str = "NVDAUSDT", attempts: int = ATTEMPTS,
    probes: Sequence[Probe] = PROBES, timeout: int = DEFAULT_TIMEOUT,
) -> list[ToolReliability]:
    """Call every probe ``attempts`` times, spacing the attempts apart."""
    results: dict[tuple[str, str], ToolReliability] = {}
    for probe in probes:
        key = (probe.tool, probe.action)
        results.setdefault(
            key, ToolReliability(skill=probe.skill, tool=probe.tool, action=probe.action)
        )
    for attempt in range(attempts):
        for probe in probes:
            row = results[(probe.tool, probe.action)]
            started = time.time()
            try:
                payload, status = client.call(
                    probe.tool, probe.for_symbol(symbol), timeout=timeout
                )
            except Exception as exc:
                row.attempts += 1
                row.failures.append(f"{type(exc).__name__}: {str(exc)[:80]}")
                continue
            row.attempts += 1
            row.latencies.append(time.time() - started)
            # Classified by `market/skills.py`'s own rule, on the payload as well as the status.
            # Reading the status alone counted a call that returned only an upstream error
            # envelope ("...: ok" with {"error": ""}) as answered: on 2026-09-26 that made three
            # tools "reliable" that the health sweep, minutes earlier, correctly found empty.
            health, detail = _classify(payload, status)
            if health is Health.OK:
                row.answered += 1
            else:
                row.failures.append(detail[:100])
        if attempt + 1 < attempts:
            time.sleep(SPACING)
    return list(results.values())


def run(symbol: str = "NVDAUSDT", *, attempts: int = ATTEMPTS) -> dict[str, Any]:
    from argus.market.evidence import BitgetSkillSource

    client = BitgetSkillSource()
    rows = measure(client, symbol=symbol, attempts=attempts)

    by_verdict: dict[str, int] = {}
    for row in rows:
        by_verdict[row.verdict] = by_verdict.get(row.verdict, 0) + 1
    skills = {r.skill for r in rows}
    reliable_skills = {
        skill for skill in skills
        if any(r.verdict == "reliable" for r in rows if r.skill == skill)
    }
    return {
        # When the sweep ran (added 2026-09-26): /status set this beside another sweep taken hours
        # apart, and without a date the two read as one contradictory measurement.
        "checked_at": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "attempts_per_tool": attempts,
        "tools": len(rows),
        "by_verdict": by_verdict,
        "skills_total": len(skills),
        "skills_with_a_reliable_tool": len(reliable_skills),
        "reliable_skills": sorted(reliable_skills),
        "results": sorted(
            (r.as_dict() for r in rows), key=lambda d: (-d["rate"], d["tool"])
        ),
        "scope_statement": (
            f"Each of Bitget's {len(rows)} official research-Skill tools is called {attempts} "
            f"times, {SPACING}s apart, through the same client the desk uses, and classified "
            f"reliable / intermittent / down. NOT CLAIMED: that {attempts} attempts characterise "
            f"a service — they distinguish always from sometimes from never and nothing finer. "
            f"NOT CLAIMED: that a failure is Bitget's fault; the failures observed carry the "
            f"service's own error envelopes, including an explicit ConnectTimeout from its "
            f"upstream, so the most this says is that the door was shut when we knocked."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    verdicts = report["by_verdict"]
    lines = [
        f"SKILL RELIABILITY — {report['tools']} tools, "
        f"{report['attempts_per_tool']} attempts each",
        "  " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())),
        f"  skills with at least one reliable tool: {report['skills_with_a_reliable_tool']}"
        f"/{report['skills_total']} — {', '.join(report['reliable_skills']) or 'none'}",
    ]
    intermittent = [r for r in report["results"] if r["verdict"] == "intermittent"]
    if intermittent:
        lines.append("  intermittent (the state a single sweep cannot see):")
        lines += [
            f"    {r['tool']}.{r['action']:20} {r['answered']}/{r['attempts']}"
            for r in intermittent
        ]
    lines.append(
        "  A single sweep reports whichever failure happened to occur. This one separates a tool "
        "that is down from one that is merely unreliable."
    )
    return lines


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="how often Bitget's Skills actually answer")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    args = parser.parse_args()

    report = run(args.symbol, attempts=args.attempts)
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ATTEMPTS", "REPORT_PATH", "ToolReliability", "main", "measure", "render", "run"]
