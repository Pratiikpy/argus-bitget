"""Per-step Qwen cost ledger: every model call, what it cost, and which part of ARGUS asked.

**Why this exists.** The hackathon form has a required field, *"Role of the LLM in Your Project"*,
and the only honest answer to it is a measured one: which modules called the model, how often, and
what share of the spend each took. Before this module the spend was known only per client
(:class:`~argus.llm.qwen.TokenBudget`, which dies with the process) and per paper cycle
(`eval/cyclecheck.cumulative_tokens`, which sees one caller of many). A Track 3 console answer that
consulted Qwen left no trace anywhere.

**Taken from gpt-researcher** (assafelovic/gpt-researcher, Apache-2.0, notice in
``licenses/gpt_researcher-APACHE-2.0.txt``): ``gpt_researcher/agent.py:166-167`` keeps
``step_costs: dict[str, float]`` beside the running total and a ``_current_step`` the orchestrator
sets as it moves through its phases (``agent.py:352,356,379,472``); ``add_costs`` at
``agent.py:773-794`` adds each call's cost to the total *and* to the current step's bucket, and
``get_step_costs`` (``agent.py:757-763``) returns the breakdown. Taken: the two-level attribution
(total plus per-step) and the idea that the step is ambient state the caller sets, not an argument
threaded through every call.

**What is ours, and why it differs.**

* **Ambient step as a ``ContextVar``, not an instance attribute.** gpt-researcher has one
  researcher object per run; ARGUS's console answers concurrent requests on a
  ``ThreadingHTTPServer``, where a shared ``_current_step`` would attribute one thread's call to
  another thread's step. :func:`cost_step` sets it per context.
* **The module is recorded even when nobody set a step.** gpt-researcher defaults to
  ``"general"`` (``agent.py:167``), which in a codebase whose callers were written before the
  ledger existed would put every call in one bucket. Here the calling module is read from the
  stack — the first frame outside ``argus.llm`` — so every existing caller is attributed on day
  one without being edited, and an explicit :func:`cost_step` refines it.
* **Tokens, not dollars.** gpt-researcher prices each call from a model table. The hackathon key
  is billed against a balance whose unit price is not published anywhere we have read, so a
  dollar figure here would be invented. The ledger records what the endpoint reported — prompt,
  completion, reasoning and cached tokens — and a call with no usage block is recorded as
  *unmeasured*, never as free (the same rule as :class:`~argus.llm.qwen.Usage`).
* **Durable, append-only JSONL.** One line per call in ``data/qwen_cost_ledger.jsonl``, written
  under a lock, so the ledger survives the process and concurrent threads cannot interleave half
  lines. A torn last line from a killed process is skipped and counted on read, not fatal.
* **A test run never writes the real ledger.** Under pytest (``PYTEST_CURRENT_TEST``) the default
  ledger is in-memory, so a scripted fake model cannot pollute the record the form is built from.
  ``ARGUS_QWEN_LEDGER`` overrides the path; ``ARGUS_QWEN_LEDGER=off`` keeps it in memory.

Rejected: gpt-researcher's ``ValueError`` on a non-numeric cost (``agent.py:784-785``). A ledger
that raises inside the model call would turn an accounting fault into a failed trading decision;
here a failed write is swallowed after the in-memory entry is kept.

    python -m argus.llm.ledger            # the "Role of the LLM" summary, from the ledger on disk
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import Any

LEDGER_PATH = Path(__file__).resolve().parents[3] / "data" / "qwen_cost_ledger.jsonl"

_STEP: ContextVar[str] = ContextVar("argus_llm_step", default="")
_WRITE_LOCK = threading.Lock()

_SKIP_PREFIXES = ("argus.llm", "contextlib", "functools", "threading", "concurrent", "asyncio")


class Outcome(StrEnum):
    """What happened to one request to the model."""

    OK = "ok"
    """The endpoint answered; its usage block (if any) is the cost."""

    CACHE_HIT = "cache_hit"
    """Served from the client's cache. Nothing was sent, nothing was spent."""

    ERROR = "error"
    """The request failed. Tokens are unknown: a stream cut off mid-answer may still be billed."""

    BUDGET_REFUSED = "budget_refused"
    """The session budget refused the call before it was sent."""


@contextmanager
def cost_step(name: str) -> Iterator[None]:
    """Attribute every model call inside this block to ``name``.

    The ``_current_step`` of gpt-researcher (``agent.py:352-472``), made safe for threads.
    Nested blocks override and restore.
    """
    token = _STEP.set(name)
    try:
        yield
    finally:
        _STEP.reset(token)


def current_step() -> str:
    return _STEP.get()


def calling_module(depth_limit: int = 40) -> str:
    """The first module on the stack outside the model client, preferring an ``argus.`` one.

    Reads frame globals only (``__name__``); never locals, so nothing a caller holds — a prompt, a
    key — can reach the ledger this way.
    """
    frame: FrameType | None = sys._getframe(1)
    fallback = ""
    for _ in range(depth_limit):
        if frame is None:
            break
        name = str(frame.f_globals.get("__name__", ""))
        if name == "__main__":
            # `python -m argus.eval.x` runs the module as `__main__`; its spec keeps the real
            # name. Found on the first day of real traffic: every call from an eval run with
            # `-m` was attributed to "__main__", which says nothing on the form.
            spec = frame.f_globals.get("__spec__")
            name = str(getattr(spec, "name", "") or name)
        if name and not name.startswith(_SKIP_PREFIXES):
            if name.startswith("argus."):
                return name
            fallback = fallback or name
        frame = frame.f_back
    return fallback or "unknown"


@dataclass(frozen=True, slots=True)
class Entry:
    """One ledger line. No field can carry the key: headers are never passed in."""

    ts: str
    module: str
    step: str
    outcome: str
    model: str
    host: str
    thinking: str
    stream: bool
    json_mode: bool
    tools: bool
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cached_tokens: int
    usage_reported: bool
    latency_ms: int
    finish_reason: str
    error: str

    @property
    def billed_tokens(self) -> int:
        """Tokens the endpoint said it charged. Zero for anything that was not a real answer."""
        return self.total_tokens if self.outcome == Outcome.OK and self.usage_reported else 0

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Entry:
        return cls(**{spec.name: row[spec.name] for spec in fields(cls)})


@dataclass(frozen=True, slots=True)
class CallShape:
    """What was asked for, independent of how it went: the same for every outcome of one call."""

    model: str
    host: str
    thinking: str
    stream: bool
    json_mode: bool
    tools: bool


def new_entry(
    *, outcome: Outcome, shape: CallShape, started: float | None = None, prompt_tokens: int = 0,
    completion_tokens: int = 0, reasoning_tokens: int = 0, total_tokens: int = 0,
    cached_tokens: int = 0, usage_reported: bool = False, finish_reason: str = "",
    error: str = "", module: str | None = None,
) -> Entry:
    return Entry(
        ts=datetime.now(UTC).isoformat(timespec="milliseconds"),
        module=module or calling_module(),
        step=current_step(),
        outcome=str(outcome), model=shape.model, host=shape.host, thinking=shape.thinking,
        stream=shape.stream, json_mode=shape.json_mode, tools=shape.tools,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens, cached_tokens=cached_tokens, usage_reported=usage_reported,
        latency_ms=0 if started is None else int((time.monotonic() - started) * 1000),
        finish_reason=finish_reason, error=error[:200],
    )


class CostLedger:
    """Every call a client made, in memory and — when ``path`` is set — appended to disk."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.entries: list[Entry] = []
        self.write_failures = 0

    @classmethod
    def default(cls) -> CostLedger:
        """The shared on-disk ledger, except under a test run or when switched off."""
        override = os.environ.get("ARGUS_QWEN_LEDGER", "").strip()
        if override.lower() in ("off", "memory", "0"):
            return cls(None)
        if override:
            return cls(Path(override))
        if "PYTEST_CURRENT_TEST" in os.environ:
            return cls(None)
        return cls(LEDGER_PATH)

    def record(self, entry: Entry) -> None:
        """Keep the entry; append it to disk if persistent. Never raises into the model call."""
        self.entries.append(entry)
        if self.path is None:
            return
        line = json.dumps(asdict(entry), ensure_ascii=False, allow_nan=False) + "\n"
        try:
            with _WRITE_LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except OSError:
            self.write_failures += 1

    @property
    def billed_tokens(self) -> int:
        return sum(entry.billed_tokens for entry in self.entries)

    def by_module(self) -> dict[str, int]:
        """gpt-researcher's ``get_step_costs`` (``agent.py:757-763``), keyed by module."""
        out: dict[str, int] = {}
        for entry in self.entries:
            out[entry.module] = out.get(entry.module, 0) + entry.billed_tokens
        return out


def read(path: Path = LEDGER_PATH) -> tuple[list[Entry], int]:
    """Every intact line, and how many were skipped as torn or foreign."""
    if not path.exists():
        return [], 0
    entries: list[Entry] = []
    skipped = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            entries.append(Entry.from_dict(json.loads(raw)))
        except (ValueError, KeyError, TypeError):
            skipped += 1
    return entries, skipped


def summarise(entries: Iterable[Entry]) -> dict[str, Any]:
    """Totals, and the per-module and per-step breakdown, for the form and for the console."""
    rows = list(entries)
    answered = [e for e in rows if e.outcome == Outcome.OK]
    billed = sum(e.billed_tokens for e in rows)

    def bucket(key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for entry in answered:
            name = getattr(entry, key) or "(no step set)"
            slot = out.setdefault(name, {"calls": 0, "tokens": 0, "reasoning_tokens": 0,
                                         "unmeasured_calls": 0})
            slot["calls"] += 1
            slot["tokens"] += entry.billed_tokens
            slot["reasoning_tokens"] += entry.reasoning_tokens if entry.usage_reported else 0
            slot["unmeasured_calls"] += not entry.usage_reported
        for slot in out.values():
            slot["share"] = round(slot["tokens"] / billed, 4) if billed else 0.0
        return dict(sorted(out.items(), key=lambda kv: (-kv[1]["tokens"], kv[0])))

    completion = sum(e.completion_tokens for e in answered if e.usage_reported)
    reasoning = sum(e.reasoning_tokens for e in answered if e.usage_reported)
    return {
        "calls_answered": len(answered),
        "cache_hits": sum(e.outcome == Outcome.CACHE_HIT for e in rows),
        "errors": sum(e.outcome == Outcome.ERROR for e in rows),
        "budget_refusals": sum(e.outcome == Outcome.BUDGET_REFUSED for e in rows),
        "unmeasured_calls": sum(not e.usage_reported for e in answered),
        "billed_tokens": billed,
        "prompt_tokens": sum(e.prompt_tokens for e in answered if e.usage_reported),
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "reasoning_share_of_completion": round(reasoning / completion, 4) if completion else 0.0,
        "models": sorted({e.model for e in answered}),
        "first": min((e.ts for e in rows), default=""),
        "last": max((e.ts for e in rows), default=""),
        "by_module": bucket("module"),
        "by_step": bucket("step"),
    }


def role_of_the_llm(path: Path = LEDGER_PATH) -> str:
    """The measured half of the form's "Role of the LLM" answer, from the ledger on disk.

    Only what the ledger shows. What each module *does* with the model is the author's to write;
    this states which modules called it, how often, and what share of the spend each took.
    """
    entries, skipped = read(path)
    report = summarise(entries)
    if not report["calls_answered"]:
        return ("No Qwen call has been recorded in the cost ledger yet "
                f"({path.name}: {len(entries)} entries, {skipped} unreadable).")
    lines = [
        f"Qwen ({', '.join(report['models'])}) answered {report['calls_answered']} calls between "
        f"{report['first'][:10]} and {report['last'][:10]}, billing {report['billed_tokens']:,} "
        f"tokens, {report['reasoning_share_of_completion']:.0%} of completion tokens spent "
        f"reasoning. {report['cache_hits']} repeat requests were served from cache at no cost; "
        f"{report['errors']} failed; {report['unmeasured_calls']} answered without a usage "
        f"block and are counted as unmeasured, not free.",
        "By module (share of billed tokens):",
    ]
    lines += [f"  {name}: {row['calls']} calls, {row['tokens']:,} tokens ({row['share']:.0%})"
              for name, row in report["by_module"].items()]
    if skipped:
        lines.append(f"{skipped} ledger line(s) were unreadable and are not counted.")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - CLI
    print(role_of_the_llm())
    return 0


__all__ = [
    "LEDGER_PATH",
    "CallShape",
    "CostLedger",
    "Entry",
    "Outcome",
    "calling_module",
    "cost_step",
    "current_step",
    "new_entry",
    "read",
    "role_of_the_llm",
    "summarise",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
