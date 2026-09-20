"""Factor memory across runs — remembering what was tried without remembering what worked.

`argus.research.factor_lab` starts cold. Every run re-proposes the same eight primitives,
re-scores them, and forgets. A search that cannot remember is not a search; it is the same
experiment repeated.

**The trap this module is built around, and it is a real one.** The obvious memory — store what
scored well and tell the proposer — destroys the property the lab exists for. `ProposerContext` has
no field for performance *by construction* (`factor_lab.py:79-93`), so a memory that hands the
proposer "these directions worked" reintroduces exactly the contamination the frozen dataclass was
shaped to prevent, and it does it through a side door that no diff of that class would show.

**FactorMiner walks into it, and it is the system this lab credits for avoiding it.** Its generator
is invoked as ``generate_batch(memory_signal=..., library_state=...)`` —
``factorminer/agent/factor_generator.py:113-118`` — and the memory signal is rendered into the user
prompt at ``:165-171``. What that signal contains is not neutral: ``memory/retrieval.py:742-750``
writes ``=== RECOMMENDED DIRECTIONS (P_succ) ===`` followed by each pattern's ``success_rate``
label, and ``library_state`` carries ``recent_admissions`` — the names of the factors that scored
well enough to be admitted. The grades are coarse (High/Medium/Low) rather than raw ICs, but they
are outcomes, and the generator is conditioned on them. Our own docstring in `factor_lab.py` credits
FactorMiner's generator with seeing "only syntax errors, never scores"; against this source that
claim is too strong, and `factor_lab.py` now says so.

**So the rule here is a filter, not a convention.** Memory records everything, because the audit
trail is worth having. What reaches the proposer is built from a whitelist and contains exactly two
kinds of fact:

* **what has already been evaluated** — identity, not quality;
* **what is structurally unscoreable** — no data, no out-of-sample slice, unparseable. A factor that
  cannot be measured at all is a property of the factor's form, and telling the proposer costs it no
  information about which surviving hypothesis is better.

A rejection on a *number* — net Sharpe, out-of-sample Sharpe, deflated Sharpe, an anti-overfit
verdict — is an outcome and never leaves this module. :func:`classify` is where that line is drawn,
and :meth:`FactorMemory.signal` is the only way out.

**The second thing FactorMiner does not have, and it matters more than the first.** Its memory
stores *patterns*, never formulas, so nothing prevents the same expression being generated,
evaluated and paid for again on a later iteration: there is no cross-iteration identity check
anywhere in ``core/ralph_loop.py``. Here identity is the canonical form of the hypothesis — the
expression and the horizon, *not* the name — so two proposals that differ only in what they are
called are one trial, and the second is suppressed rather than re-scored.

That distinction also protects the Deflated Sharpe gate. It consumes the trial count as the number
of distinct looks taken; re-testing one hypothesis produces no new maximum and must not inflate it.
Suppressed duplicates are counted and reported rather than dropped silently.

**Caps are enforced, not declared.** FactorMiner sets ``max_success_patterns=50``,
``max_failure_patterns=100`` and ``max_insights=30`` at ``architecture/memory_policy.py:78-85`` and
never applies them; its ``_quality_ledger`` (``:1046``) grows without bound for the life of the
process. Here the bound is applied on every write and there is a test that fills past it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from argus.research.factor_lab import Factor, FactorRecord, Lifecycle

SCHEMA_VERSION = 1
"""Written into every persisted file. A memory read back under a different shape is refused rather
than silently half-loaded — a partially-understood history is worse than a cold start, because it
looks like a warm one."""

MAX_TRIALS = 5_000
"""Hard bound on retained trials. Reached, the oldest are dropped and the drop is recorded in
:attr:`FactorMemory.dropped`, so a memory that has forgotten something says so."""

MAX_BARRED_IN_SIGNAL = 32
"""Structural bars shown to the proposer at once. The list is prompt text; an unbounded one crowds
out the market-structure description that is the proposer's actual brief."""


class Rejection(StrEnum):
    """Why a factor died, classified by *what kind of fact* the reason is.

    This enum is the whole safety property. ``STRUCTURAL`` may cross to the proposer;
    ``PERFORMANCE`` may not, ever, in any aggregated or quantized form.
    """

    NONE = "none"
    STRUCTURAL = "structural"
    """The factor could not be measured: no data, no out-of-sample slice, unscoreable. A fact about
    its form."""

    PERFORMANCE = "performance"
    """The factor was measured and found wanting. A fact about the market, and the thing the
    proposer must not be told."""


# Matched against the rejection strings `factor_lab.Evaluator` and `FactorLab.gate` actually write.
# Ordered: the first match wins, and anything unrecognised is treated as PERFORMANCE. That default
# is deliberate — a new rejection reason added to the lab should fail closed, keeping a number
# inside this module, rather than fail open and leak it the first time nobody updates this table.
_STRUCTURAL_MARKERS = (
    "unscoreable:",
    "no scoreable out-of-sample slice",
    "dsr uncomputable",
)


def classify(reason: str) -> Rejection:
    """Structural or performance? The one decision that keeps the proposer clean.

    An empty reason is :attr:`Rejection.NONE`. Everything else that is not a recognised structural
    marker is :attr:`Rejection.PERFORMANCE`, including reasons this table has never seen.
    """
    if not reason.strip():
        return Rejection.NONE
    low = reason.lower()
    if any(marker in low for marker in _STRUCTURAL_MARKERS):
        return Rejection.STRUCTURAL
    return Rejection.PERFORMANCE


def canonical_form(factor: Factor) -> str:
    """The identity of a hypothesis.

    The expression and the horizon, and deliberately **not** the name. Two proposals that differ
    only in what they are called are the same experiment, and a memory keyed on names would let a
    renamed factor be paid for twice — which is the state FactorMiner is in, where nothing is keyed
    on the formula at all.
    """
    return f"{factor.expression}@{factor.horizon_bars}"


@dataclass(frozen=True, slots=True)
class Trial:
    """One evaluated hypothesis, as remembered. The full record, performance included."""

    canonical: str
    name: str
    trial_number: int
    terminal_state: str
    rejection_reason: str
    rejection: Rejection
    at: str
    """ISO timestamp. Present so a memory can be read as a history rather than a set."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "name": self.name,
            "trial_number": self.trial_number,
            "terminal_state": self.terminal_state,
            "rejection_reason": self.rejection_reason,
            "rejection": str(self.rejection),
            "at": self.at,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> Trial:
        return Trial(
            canonical=str(payload["canonical"]),
            name=str(payload["name"]),
            trial_number=int(payload["trial_number"]),
            terminal_state=str(payload["terminal_state"]),
            rejection_reason=str(payload.get("rejection_reason", "")),
            rejection=Rejection(payload.get("rejection", "none")),
            at=str(payload.get("at", "")),
        )


@dataclass(frozen=True, slots=True)
class MemorySignal:
    """The only thing that crosses from memory to the proposer.

    Every field here is identity or structure. There is no field for a score, an aggregate of
    scores, a rank, a grade, or a count of successes — and :func:`assert_carries_no_outcome` checks
    at import time that none has been added, so the guarantee survives a future edit by someone who
    has not read this docstring.
    """

    already_evaluated: tuple[str, ...]
    """Canonical forms that have been through the lab. Identity, not quality."""

    structurally_barred: tuple[str, ...]
    """Forms that could not be measured at all. Capped at :data:`MAX_BARRED_IN_SIGNAL`."""

    trials_remembered: int
    """How many distinct hypotheses this memory holds. A count of looks taken, which the proposer is
    already trusted with (`factor_lab.py:90-92`), not a count of successes."""

    runs: int
    """How many sessions have contributed. Says the memory is warm; says nothing about outcomes."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "already_evaluated": list(self.already_evaluated),
            "structurally_barred": list(self.structurally_barred),
            "trials_remembered": self.trials_remembered,
            "runs": self.runs,
        }

    def render(self) -> list[str]:
        """Prompt-ready lines. Phrased so nothing reads as a recommendation."""
        out = [
            f"[memory] {self.trials_remembered} hypothesis form(s) already evaluated across "
            f"{self.runs} run(s); proposing one again will be suppressed rather than re-scored",
        ]
        if self.already_evaluated:
            out.append(f"[memory] already evaluated: {', '.join(self.already_evaluated)}")
        if self.structurally_barred:
            out.append(
                f"[memory] unmeasurable on this data (no scoreable slice, not a judgement of "
                f"merit): {', '.join(self.structurally_barred)}"
            )
        return out


# Field names on `FactorRecord` that carry, or summarise, a measured outcome. `MemorySignal` may
# never grow one of these, nor anything derived from one.
PERFORMANCE_FIELDS = frozenset({
    "gross_sharpe", "net_sharpe", "oos_sharpe", "dsr", "overfit",
    "success_rate", "ic", "rank", "score", "grade", "admitted", "certified",
    "recommended", "recent_admissions", "best", "top",
})


def assert_carries_no_outcome() -> None:
    """Fail loudly if :class:`MemorySignal` ever grows a field that could carry an outcome.

    Checked at import, not only in the test suite, because the failure this prevents is silent: a
    contaminated search still runs, still reports, and still looks exactly like a clean one.
    """
    offending = sorted(set(MemorySignal.__dataclass_fields__) & PERFORMANCE_FIELDS)
    if offending:
        raise RuntimeError(
            f"MemorySignal carries outcome field(s) {offending}; the factor lab's separation "
            f"between proposer and evaluator is broken by this field alone"
        )


assert_carries_no_outcome()


@dataclass
class FactorMemory:
    """Append-only memory of every hypothesis evaluated, across runs.

    Holds the whole record. Emits only :class:`MemorySignal`.
    """

    trials: list[Trial] = field(default_factory=list)
    runs: int = 0
    dropped: int = 0
    """Trials evicted at :data:`MAX_TRIALS`. Non-zero means this memory is no longer complete, and
    saying so is the difference between a bounded memory and a lying one."""

    duplicates_suppressed: int = 0
    """Proposals matching a remembered canonical form. Counted rather than hidden: the number is
    how often the proposer is going in circles, which is worth seeing."""

    # --- reading -------------------------------------------------------------------------------

    @property
    def known(self) -> set[str]:
        return {t.canonical for t in self.trials}

    def seen(self, factor: Factor) -> bool:
        return canonical_form(factor) in self.known

    def first_trial_of(self, factor: Factor) -> Trial | None:
        """The earliest remembered evaluation of this hypothesis, if any."""
        form = canonical_form(factor)
        for trial in self.trials:
            if trial.canonical == form:
                return trial
        return None

    def signal(self) -> MemorySignal:
        """Build the proposer's view from a whitelist.

        Constructed field by field rather than filtered from the full record: a filter has to be
        kept in step with every field the record gains, and a whitelist does not.
        """
        barred = [
            t.canonical for t in self.trials if t.rejection is Rejection.STRUCTURAL
        ]
        # Keep the most recent, and deduplicate while preserving order.
        seen: set[str] = set()
        unique_barred: list[str] = []
        for form in reversed(barred):
            if form not in seen:
                seen.add(form)
                unique_barred.append(form)
        return MemorySignal(
            already_evaluated=tuple(sorted(self.known)),
            structurally_barred=tuple(unique_barred[:MAX_BARRED_IN_SIGNAL]),
            trials_remembered=len(self.known),
            runs=self.runs,
        )

    # --- writing -------------------------------------------------------------------------------

    def remember(self, record: FactorRecord, *, at: datetime | None = None) -> Trial:
        """Record one evaluated factor. Returns the stored trial."""
        stamp = (at or datetime.now(UTC)).isoformat()
        trial = Trial(
            canonical=canonical_form(record.factor),
            name=record.factor.name,
            trial_number=record.trial_number,
            terminal_state=str(record.state),
            rejection_reason=record.rejection_reason,
            rejection=classify(record.rejection_reason),
            at=stamp,
        )
        self.trials.append(trial)
        self._enforce_cap()
        return trial

    def update(self, record: FactorRecord, *, at: datetime | None = None) -> Trial:
        """Refresh a remembered trial whose state moved after it was first written.

        A factor is remembered when it is scored, which is before the Deflated Sharpe and
        anti-overfit gates run — so the state written at that moment is not the one it ends in.
        Without this the memory would record every survivor as ``oos_tested`` and
        :func:`certified_forms` would always be empty, which is the quiet kind of wrong: the search
        would look like it had never certified anything.

        Matched on (canonical form, trial number) so two evaluations of one form in different runs
        stay separate rows.
        """
        fresh = self.remember(record, at=at)
        self.trials.pop()  # remember() appended it; place it over the row it supersedes instead
        for i, existing in enumerate(self.trials):
            if (existing.canonical, existing.trial_number) == (fresh.canonical, fresh.trial_number):
                self.trials[i] = fresh
                return fresh
        self.trials.append(fresh)
        self._enforce_cap()
        return fresh

    def note_duplicate(self) -> None:
        self.duplicates_suppressed += 1

    def _enforce_cap(self) -> None:
        excess = len(self.trials) - MAX_TRIALS
        if excess > 0:
            del self.trials[:excess]
            self.dropped += excess

    # --- persistence ---------------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "runs": self.runs,
            "dropped": self.dropped,
            "duplicates_suppressed": self.duplicates_suppressed,
            "trials": [t.as_dict() for t in self.trials],
        }

    def save(self, path: Path) -> Path:
        """Persist, counting this as one completed run."""
        self.runs += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path) -> FactorMemory:
        """Read a memory back. A missing file is a cold start; a wrong shape is an error.

        The distinction matters: "no memory yet" is a normal first run, and "a memory I cannot read"
        is a defect that must not be papered over with an empty one, because the search would then
        redo work it has a record of and nothing would say so.
        """
        if not path.exists():
            return FactorMemory()
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} is schema version {version!r}, this build reads {SCHEMA_VERSION}; "
                f"refusing to load a partially-understood history"
            )
        return FactorMemory(
            trials=[Trial.from_dict(t) for t in payload.get("trials", [])],
            runs=int(payload.get("runs", 0)),
            dropped=int(payload.get("dropped", 0)),
            duplicates_suppressed=int(payload.get("duplicates_suppressed", 0)),
        )


def certified_forms(memory: FactorMemory) -> tuple[str, ...]:
    """Canonical forms that reached CERTIFIED, for the *deployment* side of the system.

    Deliberately a module-level function and not a method of :class:`MemorySignal`: this is outcome
    information, it is legitimately needed by whatever decides what to trade, and keeping it out of
    the signal's reach is the point. Nothing on the proposer path may call this.
    """
    return tuple(
        t.canonical for t in memory.trials if t.terminal_state == str(Lifecycle.CERTIFIED)
    )


__all__ = [
    "MAX_BARRED_IN_SIGNAL",
    "MAX_TRIALS",
    "PERFORMANCE_FIELDS",
    "SCHEMA_VERSION",
    "FactorMemory",
    "MemorySignal",
    "Rejection",
    "Trial",
    "assert_carries_no_outcome",
    "canonical_form",
    "certified_forms",
    "classify",
]
