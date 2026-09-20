"""Did the cycle do what it was supposed to? Checked after every run, not assumed.

The paper cycle fires four times a day and nothing has ever inspected the result. Its exit code is
0 whether it decided twelve symbols or stopped after two, whether the chain still verifies or not,
and whether the decisions carry a lean or quietly stopped carrying one. A log read only when
something looks wrong is a log that reports success by default.

This matters most on **the first regular-hours cycle**. All 183 decisions to date were taken in
weekend sessions with the anchor market shut, and `eval/autopsy.py` concluded that the desk's
unbroken run of abstentions is explained by that sampling rather than by the desk — naming the
falsifier explicitly: *one cycle in a session with price discovery in which the desk still proposes
nothing*. That cycle is the most important one this project will run, and it will run unattended.

**Every check is a property of the recorded artefacts, never of the exit code.** Each returns PASS,
FAIL or a stated UNKNOWN, and a check that cannot be evaluated says so rather than passing. The
`session` check is the one the falsifier turns on: it reports the phase the cycle actually saw, and
an RTH cycle that produced no exposure is reported as **the falsifier firing** rather than as a
quiet fifth day of abstentions.

**What it deliberately does not do: judge whether abstaining was right.** A cycle where the desk
looked at twelve RTH symbols and declined all twelve is a PASS here, because the desk is allowed to
say no and the quality bar is about whether the machinery did its job. Whether the no was correct is
the Observatory's arithmetic against the counterfactual, and conflating the two would turn a
verification harness into a second opinion nobody asked for.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
RUNS_DIR = DATA / "paper_runs"
REPORT_PATH = DATA / "cycle_check.json"

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
"""The check could not be evaluated. Never folded into PASS — the four honest states, not three."""

EXPECTED_SYMBOLS = 12
"""The scheduled cycle names twelve symbols (`run_paper_cycle.ps1`). Fewer means it stopped."""


class CycleCheckError(RuntimeError):
    """Raised rather than reporting a clean cycle from a log that could not be read."""


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class CycleReport:
    """One cycle, checked."""

    log: str
    checks: tuple[Check, ...]
    saw_price_discovery: bool = False
    proposed_exposure: int = 0

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == FAIL)

    @property
    def unknowns(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == UNKNOWN)

    @property
    def sound(self) -> bool:
        return not self.failures

    @property
    def falsifier_fired(self) -> bool:
        """An RTH cycle in which the desk still proposed nothing.

        The observation `eval/autopsy.py` named as the one that would move the cause of ARGUS's
        unbroken abstentions from the sampling back to the desk. Reported the moment it happens,
        because the alternative is a fifth quiet day that looks like the four before it.
        """
        return self.saw_price_discovery and self.proposed_exposure == 0

    @property
    def verdict(self) -> str:
        head = f"{len(self.checks) - len(self.failures)}/{len(self.checks)} checks passed"
        if self.failures:
            named = "; ".join(f"{c.name}: {c.detail}" for c in self.failures)
            head += f". FAILED — {named}"
        if self.unknowns:
            head += f". {len(self.unknowns)} could not be evaluated: " + ", ".join(
                c.name for c in self.unknowns
            )
        if not self.saw_price_discovery:
            return (
                f"{head}. The cycle ran with the anchor market shut, so it says nothing about "
                f"whether this desk trades"
            )
        if self.falsifier_fired:
            return (
                f"{head}. **FALSIFIER FIRED**: this cycle had price discovery and the desk still "
                f"proposed no exposure on any symbol. eval/autopsy.py named exactly this as the "
                f"observation that moves the cause of the abstentions from the sampling to the "
                f"desk. Its conclusion is now refuted and must be rewritten, not defended"
            )
        return (
            f"{head}. The desk proposed exposure on {self.proposed_exposure} symbol(s) in a "
            f"session with price discovery — the first time it has done so on the live record"
        )

    def render(self) -> str:
        lines = [f"CYCLE CHECK — {self.log}", ""]
        for check in self.checks:
            lines.append(f"  {check.status:<8} {check.name}")
            lines.append(f"           {check.detail}")
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "log": self.log,
            "sound": self.sound,
            "saw_price_discovery": self.saw_price_discovery,
            "proposed_exposure": self.proposed_exposure,
            "falsifier_fired": self.falsifier_fired,
            "checks": [c.as_dict() for c in self.checks],
            "verdict": self.verdict,
        }


def latest_log(directory: Path = RUNS_DIR) -> Path:
    logs = sorted(directory.glob("cycle_*.log"))
    if not logs:
        raise CycleCheckError(f"no cycle logs under {directory}")
    return logs[-1]


def _decisions_in(log_text: str) -> list[dict[str, Any]]:
    """The decision rows the cycle reported, read from the log's own JSON summary."""
    match = re.search(r'"decisions_written":\s*(\[.*?\])\s*,\s*"settled_this_cycle"', log_text,
                      re.DOTALL)
    if not match:
        return []
    try:
        rows: list[dict[str, Any]] = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return rows


LEGAL_LEANS = ("up", "down", "none")
"""What `agents/meta_pm.py:435` normalises the model's answer into. Anything else is drift."""


def _lean_checks(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]] | None,
) -> list[Check]:
    """The two separate properties of the lean, which the first version of this conflated into one.

    **The bug this replaces.** The old single check read `getattr(e, "lean", "none") in {up, down}`
    off the loaded entry and called anything else a failure. That is wrong twice over:

    * `decision/verdicts.py:128` documents ``none`` as **a legitimate answer** meaning the direction
      genuinely cannot be called, and `agents/meta_pm.py:107` offers the model ``"NONE"`` as one of
      three choices. One honest ``none`` among eleven directional calls failed the cycle — a harness
      marking a correct answer wrong. It fired on 2026-09-14 for TQQQUSDT.
    * It could never have caught the failure its own comment named. ``Entry.lean`` defaults to
      ``"none"``, so a row that lost the key on disk and a row that honestly answered ``none`` are
      the same object after loading. The check that existed to detect a dropped field was blind to
      a dropped field.

    So the two properties are separated and each is checked against the artefact that carries it:

    ``leans_recorded`` — **machinery.** Every persisted row carries both keys with a legal value.
    Read from the raw JSON line, where absence is still absence.

    ``leans_gradeable`` — **information.** How many are directional. A cycle where *every* symbol
    answers ``none`` produces nothing the shadow record can score, which is the exact state the lean
    was introduced to end (all 156 decisions before seq 157 read ``none``). The artefact cannot
    distinguish a genuinely uncallable session from a lean path that has stopped producing output,
    so it is surfaced rather than passed. That is not this module judging whether abstaining was
    right — it is reporting that the cycle graded nothing.
    """
    if not rows:
        return [
            Check("leans_recorded", UNKNOWN, "no decisions in this cycle to check"),
            Check("leans_gradeable", UNKNOWN, "no decisions in this cycle to check"),
        ]
    if records is None:
        return [
            Check("leans_recorded", UNKNOWN,
                  "the persisted ledger rows were not supplied, and a loaded entry cannot show a "
                  "missing key"),
            Check("leans_gradeable", UNKNOWN, "the persisted ledger rows were not supplied"),
        ]

    seqs = {r["seq"] for r in rows if "seq" in r}
    written = [r for r in records if r.get("seq") in seqs]
    if not written:
        return [
            Check("leans_recorded", UNKNOWN, "no ledger rows matched this cycle"),
            Check("leans_gradeable", UNKNOWN, "no ledger rows matched this cycle"),
        ]

    broken = [
        r for r in written
        if "lean" not in r or "lean_confidence" not in r or r.get("lean") not in LEGAL_LEANS
    ]
    directional = [r for r in written if r.get("lean") in {"up", "down"}]
    out = [Check(
        "leans_recorded",
        FAIL if broken else PASS,
        f"{len(written) - len(broken)} of {len(written)} row(s) carry a lean and a lean_confidence"
        + (
            "; BROKEN: " + ", ".join(
                f"seq {r.get('seq')} -> {r.get('lean', '<key absent>')!r}" for r in broken[:4]
            )
            if broken else ""
        ),
    )]

    counts = ", ".join(
        f"{sum(1 for r in written if r.get('lean') == v)} {v}" for v in LEGAL_LEANS
    )
    out.append(Check(
        "leans_gradeable",
        PASS if directional else FAIL,
        f"{len(directional)} of {len(written)} decision(s) state a direction ({counts})"
        + (
            "" if directional else
            "; the cycle graded nothing, and the record cannot say whether the session was "
            "genuinely uncallable or the lean path stopped producing output"
        ),
    ))
    return out


def cumulative_tokens(directory: Path = RUNS_DIR) -> tuple[int, int]:
    """Total tokens spent across every cycle log on disk, and how many logs that covers.

    **Nothing tracked this until 2026-09-15.** `token_budget` and `budget_not_exhausted` are both
    *per-cycle* — they answer "did this run stay inside its cap", which says nothing about how much
    of a finite hackathon key is left. A key that exhausts mid-judging stops the scheduled cycle,
    freezes the ledger, and silently degrades the console to its deterministic layer, on the one
    surface judges are watching.

    Returns ``(tokens, cycles)``. The **runway is deliberately not computed**: the key's remaining
    balance is not observable from here, and dividing by a number we do not have would be exactly
    the kind of invented figure this project refuses. The burn rate is reported; the judgement is
    the reader's.
    """
    total = 0
    seen = 0
    for path in sorted(directory.glob("cycle_*.log")):
        match = re.search(r'"tokens_spent":\s*(\d+)', path.read_text(
            encoding="utf-8", errors="replace"))
        if match:
            total += int(match.group(1))
            seen += 1
    return total, seen


def check(
    log: Path,
    *,
    entries: Sequence[Any] | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> CycleReport:
    """Verify one cycle from its log and the ledger it wrote into.

    ``entries`` are the loaded ledger objects; ``records`` are the **raw persisted rows**. Both are
    needed and they are not interchangeable: ``PaperLedger._load`` constructs ``Entry(**row)``, and
    every optional field carries a default, so a row that lost a key on disk loads as a row that
    answered the default. The lean checks read ``records`` for exactly that reason.
    """
    if not log.exists():
        raise CycleCheckError(f"{log} does not exist")
    text = log.read_text(encoding="utf-8", errors="replace")
    checks: list[Check] = []

    # **Anchored to the start of a line, and it must be.** The unanchored `exit=(-?\d+)` matched
    # the *stage* markers the cycle script emits — `bookcalib_exit=`, `markout_exit=`,
    # `effectiveness_exit=`, `check_exit=` — and `re.search` takes the first. A complete cycle log
    # contains ten strings matching it and only one is the process exit, so this check was reading
    # `markout_exit=0` and reporting "process exited 0".
    #
    # Worse on an in-flight cycle, which has stage markers but no process exit yet: the first and
    # most load-bearing check in this harness reported PASS for a cycle that had not exited at all.
    # Found 2026-09-14 on `cycle_2026-09-14_2300.log`.
    exit_match = re.search(r"(?m)^exit=(-?\d+)", text)
    if exit_match is None:
        checks.append(Check(
            "exit_code", UNKNOWN,
            "the log records no process exit — the cycle is still running, or it was killed "
            "before it could write one. Neither is a pass",
        ))
    else:
        code = int(exit_match.group(1))
        checks.append(Check(
            "exit_code", PASS if code == 0 else FAIL, f"process exited {code}",
        ))

    rows = _decisions_in(text)
    checks.append(Check(
        "symbols_decided",
        PASS if len(rows) >= EXPECTED_SYMBOLS else FAIL,
        f"{len(rows)} of {EXPECTED_SYMBOLS} symbols produced a decision",
    ))

    stopped = re.search(r'"stopped_early":\s*"([^"]*)"', text)
    if stopped is None:
        checks.append(Check("ran_to_completion", UNKNOWN, "no stopped_early field in the log"))
    else:
        reason = stopped.group(1)
        checks.append(Check(
            "ran_to_completion", PASS if not reason else FAIL,
            reason or "the cycle reached every symbol it was given",
        ))

    spent = re.search(r'"tokens_spent":\s*(\d+)', text)
    budget = re.search(r'"token_budget":\s*(\d+)', text)
    if spent is None or budget is None:
        checks.append(Check("token_budget", UNKNOWN, "the log records no token figures"))
    else:
        used, cap = int(spent.group(1)), int(budget.group(1))
        checks.append(Check(
            "token_budget", PASS if used < cap else FAIL,
            f"{used:,} of {cap:,} spent ({used / cap:.0%})",
        ))

    # **Cumulative drain.** The two checks above are per-cycle; this one is the one that would
    # actually see a hackathon key running out. Reported as a fact rather than graded, because the
    # remaining balance is not observable from here — a PASS/FAIL against a threshold we invented
    # would be worse than the number alone.
    burned, cycles = cumulative_tokens()
    unbilled = sum(
        int(m.group(1))
        for path in sorted(RUNS_DIR.glob("cycle_*.log"))
        for m in re.finditer(
            r'"unreported_calls":\s*(\d+)', path.read_text(encoding="utf-8", errors="replace")
        )
    ) if RUNS_DIR.exists() else 0
    if cycles:
        per_cycle = burned / cycles
        checks.append(Check(
            "cumulative_tokens", PASS,
            f"{burned:,} token(s) across {cycles} recorded cycle(s), "
            f"{per_cycle:,.0f} per cycle — roughly {per_cycle * 4:,.0f}/day at four cycles. "
            f"The key's remaining balance is not observable from here; this is the burn rate, not "
            f"a runway"
            + (
                f". ⚠ {unbilled} call(s) returned no usage block and are NOT in that total — "
                f"the real burn is higher by an unknown amount"
                if unbilled else ""
            ),
        ))
    else:
        checks.append(Check(
            "cumulative_tokens", UNKNOWN,
            "no cycle log records a token figure, so the drain on the key cannot be measured",
        ))

    if "token budget exhausted" in text:
        checks.append(Check("budget_not_exhausted", FAIL, "the guard fired during the cycle"))
    else:
        checks.append(Check("budget_not_exhausted", PASS, "the guard did not fire"))

    intact = re.search(r'"chain_intact":\s*(true|false)', text)
    if intact is None:
        checks.append(Check("chain_intact", UNKNOWN, "the log records no chain verification"))
    else:
        ok = intact.group(1) == "true"
        checks.append(Check(
            "chain_intact", PASS if ok else FAIL,
            "the hash chain verified after the write" if ok else "THE CHAIN DID NOT VERIFY",
        ))

    checks.extend(_lean_checks(rows, records))

    # The malformed-verdict path. A cycle full of HUMAN_REVIEW means the retry stopped working.
    malformed = text.count("unparseable verdict")
    checks.append(Check(
        "verdicts_parsed",
        PASS if malformed == 0 else FAIL,
        "every verdict parsed"
        if malformed == 0
        else f"{malformed} decision(s) reached HUMAN_REVIEW through an unparseable verdict; the "
             f"validate-and-retry path in complete_json is not holding",
    ))

    # The session comes from the **ledger**, not the log. A `decisions_written` row carries seq,
    # symbol, verdict, quantity, confidence and flags — and no session phase. The first version of
    # this check read it from the row, found nothing, and reported "none recorded" while passing:
    # the falsifier this module exists to catch could never have fired. Caught by running it.
    seqs = {r.get("seq") for r in rows}
    written = [e for e in (entries or ()) if getattr(e, "seq", None) in seqs]
    phases = {str(getattr(e, "session_phase", "")).lower() for e in written}
    saw_discovery = bool(phases & {"regular", "rth", "open"})
    if not written:
        checks.append(Check(
            "session", UNKNOWN,
            "no ledger rows matched this cycle, so the session it saw is unknown — and an unknown "
            "session cannot be read as a weekend one",
        ))
    else:
        checks.append(Check(
            "session", PASS,
            f"session phase(s) seen: {', '.join(sorted(phases))}"
            + ("  <-- price discovery" if saw_discovery else "  (anchor market shut)"),
        ))

    proposed = sum(1 for r in rows if str(r.get("verdict", "")).lower() not in {
        "no_trade", "data_insufficient", "human_review",
    })
    return CycleReport(
        log=log.name, checks=tuple(checks),
        saw_price_discovery=saw_discovery, proposed_exposure=proposed,
    )


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from argus.paper.ledger import PaperLedger
    from argus.paper.runner import LEDGER_PATH

    entries = PaperLedger(path=LEDGER_PATH).entries if LEDGER_PATH.exists() else None
    records = (
        [
            json.loads(line)
            for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if LEDGER_PATH.exists()
        else None
    )
    report = check(latest_log(), entries=entries, records=records)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print("\nwritten to " + str(REPORT_PATH))
    return 0 if report.sound else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "EXPECTED_SYMBOLS",
    "FAIL",
    "LEGAL_LEANS",
    "PASS",
    "UNKNOWN",
    "Check",
    "CycleCheckError",
    "CycleReport",
    "check",
    "cumulative_tokens",
    "latest_log",
]
