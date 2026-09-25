"""The live Thought / Action / Observation trace — the explanation of record for an answer.

**Why this exists.** `lui/provenance.py` labels each line of a finished answer by reading its
wording, and says so: "the label is read from the wording those places share", with the share left
unlabelled measured rather than assumed. That is provenance *inferred after the fact*. A line can be
labelled from the wrong rule, a quoted thesis that starts "RSI(14)" reads as a live quote, and a
refusal worded outside the rule list carries no label at all. The rivals attach a label where the
number is made; this module records where each line is made, while the answer is being built, so
the label can come from the step that produced the line instead of from a pattern over its text.

**What was taken, from where.**

* ReAct (Yao et al. 2023, ``ysymyth/ReAct``, MIT), ``hotpotqa.ipynb`` cell 4 (0-indexed),
  ``webthink()``: each step is written as ``Thought i / Action i / Observation i`` as it happens
  (``step_str``), the trace *is* the record, and a malformed step is counted rather than hidden
  (``n_badcalls``). Taken: the three-part step and :meth:`Trace.render`, which prints it. The
  "Thought" here is the router's recorded reading of the question (the request kind, the reader
  that chose it); ARGUS's engines are deterministic, so there is no free-text reasoning to record
  and none is invented.
* openai/evals (MIT), ``evals/record.py:34-36, 83-99`` (a ``ContextVar`` default recorder, set for
  the duration of one sample) and ``:157-183`` (``record_event``: every event numbered and
  timestamped under a lock). Taken: :func:`recording` and the thread-safe step list. Also
  ``evals/solvers/solver.py:75-93``: every postprocessor records its input *and* output, so a bad
  answer can be traced to the step that mangled it. Taken as the pass-through rule below — a step
  that returns a line it was handed did not produce it.
* OSWorld (``xlang-ai/OSWorld``, Apache-2.0), ``lib_run_single.py:44-59``: one record per step
  with the action, the response and a timestamp, so a reviewer can replay an episode. Taken as a
  pattern only (no code): each step carries when it started and finished and which sources it read
  at what time.

**What was rejected, and why.** evals' buffered flush (``MIN_FLUSH_EVENTS``) and remote recorders:
one answer's trace is small and goes out with the answer. ReAct's model-chosen actions: the console
keeps routing deterministic (`lui/multistep.py` says why), so the trace records the choice the
router made rather than asking a model to narrate one.

**How a line gets its label from the trace.** Three structural facts, none read from wording:

1. *The producer.* The step that first returned the line (earliest to finish) produced it; a step
   that returns a line it was given as an argument is passing it through (the evals postprocessor
   rule). Lines are compared after the console's own display transforms — the "Actionable:" lead
   prefix, a multi-part answer's "1." numbering, and `research._clean`'s contract-suffix stripping —
   because the console applies those after the line is made.
2. *The producer's declaration.* An engine declares what its lines are (:data:`DECLARED`, or the
   :func:`traced` decorator at its definition). A declaration is written by reading the engine's
   code, and each one below says what was read. An engine whose lines are of several kinds declares
   nothing, and its lines keep the wording-based label, marked as such (``origin: "regex"``).
3. *Refusal after a failed read.* A step that returned a refused answer while a source it tried did
   not answer (recorded at the transport) has said what it could not read: its first line is
   ``missing``. A refusal with no failed read — an order refused as out of scope, a question not
   recognised — is not labelled missing by this rule, because it is not about a missing input.

Plus the request reader's defaults: the notes a reader applied because the question did not say
("no size was given, so NVDA is assessed at 20%") are recorded as a step the moment an engine is
handed the request, and the "Assumed:" line that states each one is attributed to that step.

**How the console is wired to it without editing the engines.** :func:`instrument` wraps the
module-level functions of the console's modules in place — the same technique `truth/coverage.py`
uses for ``urlopen`` — so every engine call becomes a step. :func:`install` wraps ``urlopen`` once
more, above `truth/coverage.py`'s wrapper, so each network read is recorded against the step that
made it with its time and outcome. Worker threads: `truth/coverage.ContextPool` already runs tasks
in the submitter's context, which carries the trace; a plain ``ThreadPoolExecutor`` (as in
`lui/multistep.py`) does not, and :func:`carry` is the wrapper for a function handed to one.

**Measured, not assumed.** `eval/trace_audit.py` runs the console offline over the three held-out
routing corpora with the recorder wired through, and publishes the share of content lines whose
label the trace determines, the share still labelled only by the wording rules, and every line on
which the two disagree.

*Finding, 2026-09-26 (data/trace_audit.json).* 680 questions, 647 answered offline; the 33 that
crash also crash with the recorder unwired (a hedge answer whose instrument is not in the frozen
history), so none is the recorder's. Of 1,807 content lines the trace decides 742 (41.1%) and finds
the producing step for 99.0%; the wording rules label 1,369 (75.8%); trace first with wording as
the fallback labels 1,510 (83.6%). Where both label a line they agree on all 601; the 141 lines
the trace labels and the wording rules leave bare are listed one by one — 118 refusals after a
failed read, 21 ledger lines (18 of them in Chinese, which no wording rule reads), one computed
ranking and one dated flow reading. Recording costs 1.02x the unrecorded answer time. What stays
inferred is named by producer: `research._macro` (405 lines) and `answer.answer` (190) lead a
worklist of engines whose lines are of several kinds and need ``emit`` at the line site.
"""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from types import ModuleType
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

LABEL_ORIGINS = ("trace", "regex", "none", "meta")
"""Where a line's label came from. ``trace``: the step that produced it declared it, or the
structural refusal/default rules decided it. ``regex``: no step declared it, and the wording rules
of `lui/provenance.py` did. ``none``: neither says anything. ``meta``: a line that describes the
answer's provenance itself ("Data:", "Sources reached"), unlabelled on purpose, as in the audit."""

MAX_INPUT_CHARS = 120
"""An input is summarised, never copied whole: a candle series is thousands of numbers, and the
trace records what was asked of an engine, not the data it was handed."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# --- the record ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceRead:
    """One network read, recorded at the transport against the step that made it."""

    name: str
    """The reader's name for the source (`truth/coverage.source_name`)."""

    at: str
    """When the read began, UTC."""

    ok: bool
    why: str = ""
    """Why it failed, in `market/rpc.py`'s taxonomy ("unreachable", "timed out", "HTTP 503")."""

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.name, "at": self.at, "ok": self.ok, "why": self.why}


Declaration = Callable[["Step", int], "str | None"]
"""What a producer's line at a position in its output is. ``None`` means the producer does not
say, and the line keeps its wording-based label."""


@dataclass(eq=False)
class Step:
    """One engine call: the thought that led to it, the action, and what was observed."""

    engine: str
    parent: Step | None
    started_at: str
    thought: str = ""
    """The routing decision this step carries out, when it is one (the request kind and the reader
    that chose it). Empty for an engine called by another engine."""

    inputs: dict[str, str] = field(default_factory=dict)
    sources: list[SourceRead] = field(default_factory=list)
    output: tuple[str, ...] = ()
    """Every line the call returned, in order."""

    given: frozenset[str] = frozenset()
    """The display forms of the lines it was handed; returning one of them is passing it through."""

    facts: dict[str, str] = field(default_factory=dict)
    """Structural facts the declarations read: the request ``kind``, ``refused``."""

    declared: Declaration | None = None
    declared_by: str = ""
    """Where the declaration was written: ``DECLARED`` or the decorator at the definition."""

    refused_head: str = ""
    """The display form of the first line of a refused answer this call returned."""

    finished_at: str = ""
    finish_order: int = -1
    error: str = ""
    order: int = -1
    """Start order within the trace."""

    @property
    def produced(self) -> tuple[str, ...]:
        """The returned lines this step made, not merely passed on."""
        return tuple(line for line in self.output if display(line) not in self.given)


@dataclass
class Trace:
    """Every step of one answer, in the order they started, with a finishing order kept beside it.

    A step with no line, no source read, no error and no thought, and no such step beneath it, is
    dropped from :meth:`as_dict`: parsing helpers run hundreds of times per answer and a reader
    needs the engines, not the arithmetic of a ticker lookup."""

    question: str
    started_at: str = field(default_factory=_now)
    steps: list[Step] = field(default_factory=list)
    finished_at: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _finished: int = 0
    _defaults_seen: set[str] = field(default_factory=set, repr=False)

    def begin(self, engine: str, parent: Step | None, **fields: Any) -> Step:
        step = Step(engine=engine, parent=parent, started_at=_now(), **fields)
        with self._lock:
            step.order = len(self.steps)
            self.steps.append(step)
        return step

    def end(self, step: Step) -> None:
        with self._lock:
            step.finished_at = _now()
            step.finish_order = self._finished
            self._finished += 1

    def note_defaults(self, notes: Sequence[str], parent: Step | None) -> None:
        """The request reader's defaults, recorded once each, as a step of their own."""
        fresh = [str(n) for n in notes if str(n) not in self._defaults_seen]
        if not fresh:
            return
        self._defaults_seen.update(fresh)
        step = self.begin("request.defaults", parent, declared=every("assumed"),
                          declared_by="the request reader's notes: defaults applied because the "
                                      "question did not say")
        step.output = tuple(f"Assumed: {note}." for note in fresh)
        self.end(step)

    # --- reading it back ------------------------------------------------------------------------

    def children(self) -> dict[int, list[Step]]:
        out: dict[int, list[Step]] = {}
        for step in self.steps:
            if step.parent is not None:
                out.setdefault(step.parent.order, []).append(step)
        return out

    def subtree_failures(self, step: Step) -> list[SourceRead]:
        """Every failed read made by ``step`` or anything it called."""
        kids = self.children()
        failed: list[SourceRead] = []
        stack = [step]
        while stack:
            current = stack.pop()
            failed.extend(read for read in current.sources if not read.ok)
            stack.extend(kids.get(current.order, []))
        return failed

    def kept(self) -> list[Step]:
        """The steps a reader is shown, in start order."""
        kids = self.children()
        keep: dict[int, bool] = {}

        def visit(step: Step) -> bool:
            if step.order in keep:
                return keep[step.order]
            below = [visit(child) for child in kids.get(step.order, [])]
            keep[step.order] = bool(step.produced or step.sources or step.error or step.thought
                                    or step.refused_head or any(below))
            return keep[step.order]

        return [step for step in self.steps if visit(step)]

    def producers(self) -> dict[str, Step]:
        """Display form of a line → the step that produced it (the earliest to finish)."""
        out: dict[str, Step] = {}
        for step in sorted((s for s in self.steps if s.finish_order >= 0),
                           key=lambda s: s.finish_order):
            for line in step.produced:
                out.setdefault(display(line), step)
        return out

    def reconcile(self, coverage: Mapping[str, Any]) -> None:
        """Take the transport judge's final verdict (`truth/coverage.Record.as_dict`) for sources
        whose reply arrived and was later read as a failure — an MCP ``isError`` inside HTTP 200 is
        judged when the body is closed, after this module saw the response come back."""
        failed = coverage.get("did_not_answer") or {}
        if not isinstance(failed, Mapping):
            return
        for step in self.steps:
            step.sources = [
                replace(read, ok=False, why=str(failed.get(read.name) or read.why or "failed"))
                if read.ok and read.name in failed else read
                for read in step.sources
            ]

    def as_dict(self) -> dict[str, Any]:
        kept = self.kept()
        index = {step.order: i for i, step in enumerate(kept, start=1)}
        origin = self.producers()

        def parent_index(step: Step) -> int | None:
            parent = step.parent
            while parent is not None and parent.order not in index:
                parent = parent.parent
            return None if parent is None else index[parent.order]

        rows = []
        for step in kept:
            made = [line for line in step.produced if origin.get(display(line)) is step]
            rows.append({
                "index": index[step.order],
                "parent": parent_index(step),
                "thought": step.thought,
                "action": {"engine": step.engine, "inputs": step.inputs},
                "observation": {
                    "sources": [read.as_dict() for read in step.sources],
                    "lines_made": made,
                    "lines_returned": len(step.output),
                    **({"refused": True} if step.facts.get("refused") == "yes" else {}),
                },
                "declared_by": step.declared_by,
                "started_at": step.started_at,
                "finished_at": step.finished_at,
                **({"error": step.error} if step.error else {}),
            })
        return {"question": self.question, "started_at": self.started_at,
                "finished_at": self.finished_at, "steps": rows}

    def render(self) -> list[str]:
        """The trace as ReAct prints it: Thought, Action, Observation, one block per step."""
        blob = self.as_dict()
        out: list[str] = []
        for row in blob["steps"]:
            i = row["index"]
            if row["thought"]:
                out.append(f"Thought {i}: {row['thought']}")
            inputs = ", ".join(f"{k}={v}" for k, v in row["action"]["inputs"].items())
            out.append(f"Action {i}: {row['action']['engine']}({inputs})"
                       + (f" [inside step {row['parent']}]" if row["parent"] else ""))
            seen = row["observation"]["sources"]
            reads = "; ".join(f"{s['source']} {'answered' if s['ok'] else 'did not answer'}"
                              + (f" ({s['why']})" if s["why"] else "") + f" at {s['at'][11:23]}"
                              for s in seen)
            made = len(row["observation"]["lines_made"])
            out.append(f"Observation {i}: " + (reads + "; " if reads else "")
                       + f"{made} line(s) made"
                       + (", refused" if row["observation"].get("refused") else "")
                       + (f", raised {row['error']}" if row.get("error") else ""))
        return out


# --- declarations -------------------------------------------------------------------------------


def every(label: str) -> Declaration:
    """Every line the producer makes is ``label``."""
    def declare(step: Step, position: int) -> str | None:
        return label
    return declare


def last(label: str, rest: str | None) -> Declaration:
    """The producer's final line is ``label``; the others are ``rest``."""
    def declare(step: Step, position: int) -> str | None:
        return label if position == len(step.output) - 1 else rest
    return declare


def only(positions: Mapping[int, str]) -> Declaration:
    """Declared at named positions only; the rest are left to the wording rules."""
    def declare(step: Step, position: int) -> str | None:
        return positions.get(position)
    return declare


def all_but(label: str, positions: Sequence[int]) -> Declaration:
    """``label`` everywhere except the named positions, which say nothing."""
    skipped = frozenset(positions)

    def declare(step: Step, position: int) -> str | None:
        return None if position in skipped else label
    return declare


def by_kind(kinds: Mapping[str, str]) -> Declaration:
    """A composing engine whose own lines depend on the branch the request's kind took. A refused
    answer is never covered: its lines are the refusal, decided by the refusal rule."""
    def declare(step: Step, position: int) -> str | None:
        if step.facts.get("refused") == "yes":
            return None
        return kinds.get(step.facts.get("kind", ""))
    return declare


DECLARED: dict[str, tuple[Declaration, str]] = {
    "argus.lui.research._impact_lines": (
        last("record", "computed"),
        "research.py:5814-5942. Every line is arithmetic on the copilot's report for this answer "
        "(beta, risk share, the risk budget, beta stress, the hedge line); the final line, always "
        "appended last (`lines.append(worst)`), is the realised worst window from history, which "
        "the reviewed convention files as a record."),
    "argus.lui.research._distribution_line": (
        every("computed"),
        "research.py:3663. The return distribution's moments, computed from the candles now."),
    "argus.lui.research._hedge_line": (
        every("computed"), "research.py:5791. A hedge ratio from the book's beta."),
    "argus.lui.research._desk_view": (
        every("desk"),
        "research.py:3857-3875. The desk's last ledger row on the name, quoted, or the statement "
        "that the desk's record holds none (the desk does not trade it, or has not decided it)."),
    "argus.lui.research._beta_track_record": (
        every("record"),
        "research.py:4777. Read from data/copilot_rivals.json and copilot_stress.json, past runs."),
    "argus.lui.research._coordination_test": (
        every("record"),
        "research.py:4812. Read from data/sentiment_comparison.json, a dated past test."),
    "argus.lui.research._event_lines": (
        every("live"),
        "research.py:5605-5625. The next CPI and FOMC dates from the BLS and Federal Reserve "
        "schedules, read from the snapshot `market/calendar.upcoming` keeps and stated with the "
        "date it was read — the convention the wording rules follow for a dated reading. The "
        "step's own source list shows no network read at answer time; a reader sees that too."),
    "argus.lui.research._flow_lines": (
        every("live"),
        "research.py:4847 and market/etf_flows.py:154-184. Dated ETF-flow and treasury-purchase "
        "readings from SoSoValue, read from the latest data/etf_flows.json snapshot and each "
        "stated with its date (the wording rules' convention for a dated reading); the step's "
        "source list shows no network read at answer time."),
    "argus.lui.research._run": (
        by_kind({"compare": "computed"}),
        "research.py:9514-9573, the COMPARE branch: per-name realised vol, session beta, beta "
        "stress and worst window, the pair's correlation and the lead ranking them, all computed "
        "from the candles for this answer. Other kinds mix computed, record and missing lines in "
        "the same branch and declare nothing."),
}
"""Declarations for engines in files this module does not own, keyed by qualified name. Each
states what was read to justify it. `lui/answer.py` declares its own with :func:`traced`."""


# --- the context ---------------------------------------------------------------------------------

_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("argus_trace", default=None)
_step: contextvars.ContextVar[Step | None] = contextvars.ContextVar("argus_trace_step",
                                                                    default=None)


@contextmanager
def recording(question: str) -> Iterator[Trace]:
    """Record every traced step of one answer. Nested recordings are independent."""
    install()
    trace = Trace(question=question)
    token, step_token = _trace.set(trace), _step.set(None)
    try:
        yield trace
    finally:
        trace.finished_at = _now()
        _step.reset(step_token)
        _trace.reset(token)


def current() -> Trace | None:
    return _trace.get()


def carry(fn: F) -> F:
    """``fn`` running in another thread records into the trace and step that are current here.

    For a function handed to a plain ``ThreadPoolExecutor``, which starts its tasks with an empty
    context. The trace and step are captured when this is called, not when ``fn`` runs."""
    trace, step = _trace.get(), _step.get()

    @functools.wraps(fn)
    def carried(*args: Any, **kwargs: Any) -> Any:
        token, step_token = _trace.set(trace), _step.set(step)
        try:
            return fn(*args, **kwargs)
        finally:
            _step.reset(step_token)
            _trace.reset(token)

    return carried  # type: ignore[return-value]


@contextmanager
def step(engine: str, *, thought: str = "", label: str | None = None,
         inputs: Mapping[str, Any] | None = None) -> Iterator[Step | None]:
    """A step for a block of code rather than a function: ``with step("research.compare"):``.

    Yields ``None`` outside a recording, so a call site costs nothing when nobody is recording."""
    trace = _trace.get()
    if trace is None:
        yield None
        return
    parent = _step.get()
    current_step = trace.begin(engine, parent, thought=thought,
                               inputs={k: _summary(v) for k, v in (inputs or {}).items()},
                               declared=every(label) if label else None,
                               declared_by="the step's own label" if label else "")
    token = _step.set(current_step)
    try:
        yield current_step
    finally:
        _step.reset(token)
        trace.end(current_step)


def emit(lines: Sequence[str], label: str | None = None) -> None:
    """Record ``lines`` as made by the current step, with the label they carry, at the line site.

    The integration for an engine whose lines are of several kinds: each group is emitted where it
    is built (``emit([fred_line], "live" if read_live else "record")``), so the label is a
    property of generation. Outside a recording this does nothing."""
    trace = _trace.get()
    if trace is None or not lines:
        return
    made = trace.begin("emit", _step.get(), declared=every(label) if label else None,
                       declared_by="emitted at the line site" if label else "")
    made.output = tuple(str(line) for line in lines)
    trace.end(made)


# --- the wrapper ---------------------------------------------------------------------------------

_SIGNATURES: dict[Callable[..., Any], inspect.Signature | None] = {}

_NOT_INPUTS = frozenset({"self", "ledger", "data", "raw", "columns", "started"})
"""Arguments that are state or bookkeeping rather than what was asked: the ledger, the candle
data, a timer. Recording them would put noise, or thousands of numbers, into the action."""


def _signature(fn: Callable[..., Any]) -> inspect.Signature | None:
    if fn not in _SIGNATURES:
        try:
            _SIGNATURES[fn] = inspect.signature(fn)
        except (TypeError, ValueError):
            _SIGNATURES[fn] = None
    return _SIGNATURES[fn]


def _summary(value: Any) -> str:
    """A short, safe description of an argument: never the data itself."""
    if value is None or isinstance(value, bool | int | float | Decimal):
        text = str(value)
    elif isinstance(value, str):
        text = repr(value)
    elif hasattr(value, "kind") and hasattr(value, "symbols") and hasattr(value, "notes"):
        text = f"{getattr(value.kind, 'value', value.kind)} on {', '.join(value.symbols) or '—'}"
    elif hasattr(value, "intent") and hasattr(value, "raw"):
        text = f"record question, intent {getattr(value.intent, 'value', value.intent)}"
    elif isinstance(value, Mapping):
        text = f"{{{len(value)} keys}}"
    elif isinstance(value, list | tuple | set | frozenset):
        text = f"[{len(value)} items]"
    else:
        text = type(value).__name__
    return text if len(text) <= MAX_INPUT_CHARS else text[:MAX_INPUT_CHARS - 1] + "…"


def _lines_of(value: Any) -> tuple[str, ...]:
    """The answer lines inside a return value: an answer's lines, a list of lines, a line, or the
    first element of an engine's ``(lines, sources[, data])`` tuple."""
    if value is None:
        return ()
    lines = getattr(value, "lines", None)
    if isinstance(lines, list) and hasattr(value, "refused"):
        return tuple(str(line) for line in lines)
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, list | tuple) and value:
        if all(isinstance(x, str) for x in value):
            return tuple(value)
        head = value[0]
        if isinstance(head, list | tuple) and all(isinstance(x, str) for x in head):
            return tuple(head)
        if isinstance(head, str) and isinstance(value, tuple):
            return (head,)
    return ()


def _given(bound: Mapping[str, Any]) -> frozenset[str]:
    out: set[str] = set()
    for value in bound.values():
        if isinstance(value, str) and len(value) > 15:
            out.add(display(value))
        elif isinstance(value, list | tuple):
            out.update(display(x) for x in value if isinstance(x, str) and len(x) > 15)
    return frozenset(out)


def _thought(engine: str, bound: Mapping[str, Any]) -> str:
    """The routing decision a call carries out, read from its own arguments."""
    request = next((v for v in bound.values() if hasattr(v, "kind") and hasattr(v, "symbols")
                    and hasattr(v, "notes")), None)
    if engine.endswith("._research_payload") and request is not None:
        audit = bound.get("audit")
        detail = audit.get("detail", "") if isinstance(audit, Mapping) else ""
        return (f"the question was read as kind '{getattr(request.kind, 'value', request.kind)}' "
                f"on {', '.join(request.symbols) or 'no named contract'} by "
                f"{bound.get('classified_by', 'the router')}" + (f" ({detail})" if detail else ""))
    question = bound.get("question")
    if engine.endswith("answer.answer") and question is not None:
        intent = getattr(getattr(question, "intent", None), "value", "")
        matched = str(getattr(question, "matched", "") or "")
        return f"read as a question about the desk's record, intent {intent}" + (
            f" (matched {matched[:60]})" if matched else "")
    return ""


def traced(label: str | None = None, *, declaration: Declaration | None = None,
           name: str | None = None) -> Callable[[F], F]:
    """Declare an engine where it is defined: ``@traced("desk")`` over an answerer.

    Outside a recording the wrapped function runs as it always did; the only cost is one context
    variable read."""
    def wrap(fn: F) -> F:
        declared = declaration or (every(label) if label else None)
        return _wrap(fn, name or f"{fn.__module__}.{fn.__qualname__}", declared,
                     "declared at the definition" if declared else "")
    return wrap


def _wrap(fn: F, engine: str, declared: Declaration | None, declared_by: str) -> F:
    if getattr(fn, "__argus_traced__", False):
        return fn
    signature = _signature(fn)

    @functools.wraps(fn)
    def traced_call(*args: Any, **kwargs: Any) -> Any:
        trace = _trace.get()
        if trace is None:
            return fn(*args, **kwargs)
        bound: dict[str, Any] = {}
        if signature is not None:
            try:
                bound = dict(signature.bind_partial(*args, **kwargs).arguments)
            except TypeError:
                bound = {}
        parent = _step.get()
        request = next((v for v in bound.values() if hasattr(v, "kind")
                        and hasattr(v, "symbols") and hasattr(v, "notes")), None)
        notes: tuple[Any, ...] = () if request is None else tuple(request.notes or ())
        if notes:
            trace.note_defaults(tuple(str(n) for n in notes), parent)
        me = trace.begin(engine, parent, thought=_thought(engine, bound),
                         inputs={k: _summary(v) for k, v in bound.items()
                                 if k not in _NOT_INPUTS},
                         given=_given(bound), declared=declared, declared_by=declared_by)
        if request is not None:
            me.facts["kind"] = str(getattr(request.kind, "value", request.kind))
        token = _step.set(me)
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            me.error = f"{type(exc).__name__}: {str(exc)[:160]}"
            raise
        finally:
            _step.reset(token)
            trace.end(me)
        me.output = _lines_of(result)
        if getattr(result, "refused", False) is True:
            me.facts["refused"] = "yes"
            if me.output:
                me.refused_head = display(me.output[0])
        return result

    traced_call.__argus_traced__ = True  # type: ignore[attr-defined]
    return traced_call  # type: ignore[return-value]


# --- wiring the console --------------------------------------------------------------------------

CONSOLE_MODULES = ("argus.lui.research", "argus.lui.answer", "argus.lui.multistep",
                   "argus.lui.memory")
"""Every module-level function in these becomes a step, except :data:`NOT_ENGINES`."""

SERVER_NAMES = ("answer", "run_research", "_research_payload", "_language_note")
"""Names `lui/server.py` holds that answer a question; wrapped where server holds them, because
it imported ``answer`` and ``run_research`` by name."""

NOT_ENGINES = frozenset({
    # display transforms: they re-emit lines another step made
    "_t", "_clean", "_sentence_cut", "_lead_with", "_lead_with_what_was_asked",
    "_answer_the_state_asked", "_fundamentals_focus",
    # parsers of the question's words, called hundreds of times per answer; none reads a source
    # or returns an answer line
    "research_symbols", "_resolve", "_resolve_any", "_read", "_number", "_pairs", "_group_pairs",
    "_amount_pairs", "_is_equity", "_is_fx", "_horizon", "_theme", "_period_days",
    "_strip_budget", "_resize", "_shouting", "_normalise", "_spot_rtoken", "_is_an_order",
    "_is_equity_or_traded", "_horizon_words", "parse_budget", "_parse_notional", "_group_note",
    "shock_numbers", "split_cash", "unread_holdings", "hedge_instruments", "rtoken_named",
    "may_name_a_contract", "worth_asking_the_model", "price_forecast_asked", "about_the_record",
    "daily_technicals_asked", "hold_cost_question", "leveraged_fund_asked", "_vol_multiple",
    "_stated_move_size", "_states_holdings", "parse_book",
})


def _module(name: str) -> ModuleType:
    # `argus.lui.answer` is shadowed on the package by the function of the same name
    # (`lui/__init__.py` re-exports it), so modules are resolved through the import system.
    return importlib.import_module(name)


def instrument(modules: Sequence[str] = CONSOLE_MODULES, *,
               server: bool = True) -> Callable[[], None]:
    """Wrap the console's engines in place so each call is a step. Returns the undo.

    Idempotent: a function already wrapped (here, or by :func:`traced` at its definition) is left
    as it is. A declared engine takes its declaration from :data:`DECLARED`."""
    install()
    undo: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, attr: str, qualified: str) -> None:
        fn = getattr(owner, attr)
        if getattr(fn, "__argus_traced__", False):
            return
        declared, why = DECLARED.get(qualified, (None, ""))
        setattr(owner, attr, _wrap(fn, qualified, declared, why and f"DECLARED: {why}"))
        undo.append((owner, attr, fn))

    for name in modules:
        module = _module(name)
        for attr, fn in list(vars(module).items()):
            if (inspect.isfunction(fn) and fn.__module__ == module.__name__
                    and attr not in NOT_ENGINES):
                patch(module, attr, f"{module.__name__}.{attr}")
        answerers = getattr(module, "_ANSWERERS", None)
        if isinstance(answerers, dict):
            # the ledger answerers are reached through this table, which holds the functions
            # themselves, so the table is re-pointed at the wrapped ones
            for key, fn in list(answerers.items()):
                wrapped = getattr(module, fn.__name__, fn)
                if wrapped is not fn:
                    answerers[key] = wrapped
                    undo.append((answerers, key, fn))
    if server:
        srv = _module("argus.lui.server")
        for attr in SERVER_NAMES:
            fn = getattr(srv, attr, None)
            if fn is None:
                continue
            origin = f"{getattr(fn, '__module__', srv.__name__)}.{getattr(fn, '__name__', attr)}"
            patch(srv, attr, origin if attr != "_language_note" else f"{srv.__name__}.{attr}")

    def restore() -> None:
        for owner, attr, fn in reversed(undo):
            if isinstance(owner, dict):
                owner[attr] = fn
            else:
                setattr(owner, attr, fn)

    return restore


@contextmanager
def instrumented(modules: Sequence[str] = CONSOLE_MODULES, *,
                 server: bool = True) -> Iterator[None]:
    """:func:`instrument` for the duration of a block, undone afterwards."""
    restore = instrument(modules, server=server)
    try:
        yield
    finally:
        restore()


_installed = False
_install_lock = threading.Lock()


def install() -> None:
    """Record each network read against the step that made it. Wraps ``urlopen`` once, above
    `truth/coverage.py`'s wrapper, which keeps counting sources exactly as before."""
    global _installed
    with _install_lock:
        if _installed:
            return
        from argus.truth import coverage

        coverage.install()
        below = urllib.request.urlopen

        def observed(url: Any, *args: Any, **kwargs: Any) -> Any:
            step_now, trace = _step.get(), _trace.get()
            if step_now is None or trace is None:
                return below(url, *args, **kwargs)
            if isinstance(url, urllib.request.Request):
                target = url.full_url
                body = url.data if isinstance(url.data, bytes) else None
            else:
                target = str(url)
                raw = args[0] if args else kwargs.get("data")
                body = raw if isinstance(raw, bytes) else None
            name = coverage.source_name(target, body)
            if not name:
                return below(url, *args, **kwargs)  # protocol plumbing, not a source
            at = _now()
            try:
                response = below(url, *args, **kwargs)
            except Exception as exc:
                from argus.market.rpc import classify_exception

                why = (f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError)
                       else classify_exception(exc).label)
                with trace._lock:
                    step_now.sources.append(SourceRead(name, at, False, why))
                raise
            with trace._lock:
                step_now.sources.append(SourceRead(name, at, True))
            return response

        urllib.request.urlopen = observed
        _installed = True


# --- per-line labels ------------------------------------------------------------------------------

_LEAD = re.compile(r"^Actionable(?: \(\w+\))?:\s*")
_NUMBERED = re.compile(r"^\d{1,2}\.\s+")
_CLEAN: Callable[[str], str] | None = None
"""`research._clean`, read once. It is a display transform and is never wrapped (NOT_ENGINES)."""


def display(line: str) -> str:
    """A line as the console finally shows it, for matching a produced line to a shown one.

    The console applies three transforms after a line is made: a lead is prefixed "Actionable:"
    (or stripped when another lead takes the place), a multi-part answer numbers its leads, and
    `research._clean` strips contract suffixes ("NVDAUSDT" → "NVDA") and sentence-cases. The same
    transforms are applied to both sides, using the console's own ``_clean``."""
    global _CLEAN
    if _CLEAN is None:
        from argus.lui.research import _clean

        _CLEAN = _clean
    cleaned = _CLEAN(_NUMBERED.sub("", _LEAD.sub("", line.strip()))).strip()
    return cleaned[:1].upper() + cleaned[1:]


@dataclass(frozen=True, slots=True)
class LineLabel:
    """One shown line: its label, where the label came from, and the step that made the line."""

    line: str
    label: str | None
    origin: str
    step: int | None = None
    engine: str | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "origin": self.origin, "step": self.step,
                "engine": self.engine, "reason": self.reason}


def line_labels(trace: Trace, lines: Sequence[str]) -> list[LineLabel]:
    """Each shown line's label: from the trace where a step determines it, else from the wording
    rules, marked as such. Meta lines stay unlabelled, as `eval/provenance_audit.py` treats them."""
    from argus.lui.provenance import _META
    from argus.lui.provenance import label as by_wording

    made = trace.producers()
    kept = {step.order: i for i, step in enumerate(trace.kept(), start=1)}
    refusals: dict[str, tuple[Step, list[SourceRead]]] = {}
    for step in sorted((s for s in trace.steps if s.refused_head and s.finish_order >= 0),
                       key=lambda s: s.finish_order):
        if step.refused_head not in refusals:
            failed = trace.subtree_failures(step)
            if failed:
                refusals[step.refused_head] = (step, failed)

    out: list[LineLabel] = []
    for line in lines:
        text = str(line)
        if not text.strip() or _META.search(text.strip()):
            out.append(LineLabel(text, None, "meta"))
            continue
        shown = display(text)
        refused = refusals.get(shown)
        if refused is not None:
            step_made, failed = refused
            names = sorted({read.name for read in failed})
            out.append(LineLabel(text, "missing", "trace", kept.get(step_made.order),
                                 step_made.engine,
                                 f"a refusal after a failed read: {', '.join(names)}"))
            continue
        producer = made.get(shown)
        # A declaration describes what an engine makes when it answers; a refused answer's lines
        # are the refusal, and only the refusal rule above speaks for them.
        if (producer is not None and producer.declared is not None
                and producer.facts.get("refused") != "yes"):
            position = next((i for i, x in enumerate(producer.output) if display(x) == shown), 0)
            declared = producer.declared(producer, position)
            if declared is not None:
                out.append(LineLabel(text, declared, "trace", kept.get(producer.order),
                                     producer.engine, producer.declared_by))
                continue
        worded = by_wording(text)
        out.append(LineLabel(
            text, worded, "regex" if worded is not None else "none",
            None if producer is None else kept.get(producer.order),
            None if producer is None else producer.engine,
            "no step declared this line; the wording rules of lui/provenance.py labelled it"
            if worded is not None else "no step declared this line and no wording rule matched"))
    return out


def attach(payload: dict[str, Any], trace: Trace) -> None:
    """Put the trace and the per-line labels on an ``/ask`` payload.

    Replaces what `lui/server.py` sets today (``payload["line_labels"] =
    provenance_labels(lines)``): ``line_labels`` keeps its shape, one label or ``None`` per line,
    so the page, the MCP server and the Telegram bot read it unchanged; ``line_origins`` says for
    each line whether the label is the trace's or the wording rules'."""
    data = payload.get("data")
    coverage = data.get("coverage") if isinstance(data, Mapping) else None
    if isinstance(coverage, Mapping):
        trace.reconcile(coverage)
    labelled = line_labels(trace, [str(x) for x in payload.get("lines") or []])
    payload["line_labels"] = [x.label for x in labelled]
    payload["line_origins"] = [x.origin for x in labelled]
    payload["line_steps"] = [x.step for x in labelled]
    payload["trace"] = trace.as_dict()


def answered(question: str, ask: Callable[[], dict[str, Any]],
             finish: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """One answer, recorded, with its trace and per-line labels attached: the call every surface
    makes instead of calling the console bare.

    ``ask`` is the console call (``lambda: handle_ask(text, prior, ...)``); ``finish`` is anything
    the surface does to the payload after the answer and before the labels are read (the web
    console's ``offer_translation``). The engines must already be instrumented, once, at start-up
    (:func:`instrument`); an engine that is not wrapped makes no step, and its lines keep the
    wording label, marked ``regex``."""
    with recording(question) as recorded:
        payload = ask()
    if finish is not None:
        finish(payload)
    attach(payload, recorded)
    return payload


__all__ = [
    "DECLARED",
    "LABEL_ORIGINS",
    "LineLabel",
    "SourceRead",
    "Step",
    "Trace",
    "all_but",
    "answered",
    "attach",
    "by_kind",
    "carry",
    "current",
    "display",
    "emit",
    "every",
    "install",
    "instrument",
    "instrumented",
    "last",
    "line_labels",
    "only",
    "recording",
    "step",
    "traced",
]
