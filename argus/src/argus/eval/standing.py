"""Standing — what each capability has actually proven, checked against the artefacts.

The project's governing rule names four states and forbids conflating them: **LOST** (a competitor
has something materially better), **TIED** (comparable, no demonstrated advantage), **IMPLEMENTED**
(the mechanism exists, proof is insufficient) and **OWNED** (demonstrated superiority through a
valid experiment). Until this module, those states lived only in prose, which is exactly where a
standard goes to rot: a README says "best-in-class", nothing checks it, and by the time a judge
reads it the sentence has outlived the evidence.

So the register is **falsifiable rather than descriptive**. Every proof names an artefact on disk
or a test that runs, :func:`audit` checks that each one is really there, and a capability claiming
OWNED without all thirteen conditions is a build failure, not a footnote. The register can be wrong
about the world — a passing artefact can contain a wrong number — but it cannot claim evidence that
does not exist. That is the part that was failing before.

**Whatever this module currently claims as OWNED is checked here, not asserted in this docstring**
— a specific count or name written in prose above the code it describes is exactly the kind of
claim that outlives its evidence this module exists to prevent (see the very next paragraph's own
history: this sentence used to read "nothing here is OWNED, and that is the honest baseline," which
was true when written and became false the day the first capability genuinely cleared all thirteen
— and stayed wrong in this docstring for a time after that, caught only when someone read the
module top-to-bottom instead of trusting its opening claim). Call :func:`audit` and read
``.owned`` for the real, current answer; ``.render()`` names every owned capability explicitly
rather than by omission. Most of the register is still below OWNED, usually missing the ablation
(there was no harness until :mod:`argus.eval.ablation`) or a reproduced baseline (reading a
competitor's source is not running it) — but "most" is itself a claim worth re-deriving from
:func:`audit`, not read off this sentence.

Two specific demotions are recorded here rather than quietly deleted:

* **Sentiment.** Its feed is dead. The Bitget social Skill returned nothing in 93% of live cycles,
  and the free replacements the audit recommended were probed directly on 2026-09-13: StockTwits
  answers 403 to four different User-Agents, CNN's Fear & Greed endpoint answers 418, and only
  ``alternative.me`` answers at all — with a *crypto-wide* index, which is a real reading of venue
  risk appetite and is not a view on any single equity. The analyst stays in the code at its honest
  standing, with the ablation that would promote it named. Deleting it would hide the finding;
  leaving it unmarked would let a dead feed count as a data source.

* **Self-evolving review rules.** Replayed over 40 real decisions, none of the five standing rules
  earned its place. They are recorded as not-earned rather than presented as a learning loop.

Read against ``microsoft/qlib``'s benchmark tables, which pair every reported number with the
config that produced it, and against the four honest QA states — PASS, FAIL, PENDING, BLOCKED — in
the project's own testing standard. Both make the same move: the claim and its evidence travel
together, or the claim does not travel.

**Three mechanical checks added 2026-09-25 (S18 and the evaluator spine).**

* **No statistical or out-of-sample condition without its breakdown.** A proof of
  ``statistically_valid_evaluation`` or ``out_of_sample_test`` now also needs a groupwise check to
  have run on the capability's own artefact: :mod:`argus.eval.groupwise` (Mind2Web's per-group
  macro average and error histogram, ``src/action_prediction/metric.py:236-259``, MIT), run over
  every artefact this register names by :mod:`argus.eval.groupwise_audit` into
  ``data/groupwise_audit.json``. The proof passes only if some headline that is the capability's
  own claim was broken down by its groups (symbol, date, regime, claim kind) or split
  chronologically, none of those headlines is carried by one group, rests on a single group, has a
  macro average pointing the other way, or flips between halves, none favours the rival, and the
  audit was run on the artefact as it is now (its SHA-256 is compared). This is the project's
  twice-recorded failure — a single-name result, momentum good in full sample and negative in both
  halves — made into a rule rather than a reminder. A designed demonstration (cases an author chose,
  a parameter sweep) is not a measurement over a population and cannot pass it; the rows it demoted
  carry the reason, and the route back, as their first blocker.
* **State changes are legal moves or nothing.** Between one persisted register and the next, a
  capability may rise one rung at a time along LOST -> TIED -> IMPLEMENTED -> OWNED and fall any
  number; anything else raises in :func:`write_report`, and every change is appended to the
  persisted ``transition_log``. Adapted from the MCP specification repository's SEP lifecycle
  automation (modelcontextprotocol/modelcontextprotocol, ``tools/sep-automation/src/rules.ts:16-44``
  ``STATE_TRANSITIONS``, ``:156-161`` ``isValidTransition``, ``src/actions/transition.ts:91``
  ``validateTransition``; Apache-2.0 for new contributions, notice and "Used in" record at
  ``argus/licenses/mcp_specification-APACHE-2.0.txt``). Taken: the legal moves are data, the
  transition consults the table rather than scattered conditions, an illegal move is refused with
  the legal targets named, and an initial assignment is always valid (``rules.ts:157-158``; here
  the thirteen conditions already govern a capability that enters at OWNED). Changed: the table is
  keyed by this register's four states, downward moves are open from every rung because evidence can
  be lost at any rank (SEP allows only named backward edges), and nothing is dormant or terminal —
  OWNED can fall. Rejected: the GitHub-label and Discord machinery around it.
* **Verdicts are read, not written.** Where an artefact carries ``comparison_reports`` (the eval
  spine, :mod:`argus.eval.compare`), the register shows their outcomes as ``measured_outcomes``
  instead of restating them in prose, and an OWNED row whose own artefact records a valid
  ``rival_better`` outcome is not earned.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parents[3]       # .../bitget/argus
ROOT = PACKAGE.parent                                # the workspace directory above argus/
DATA = PACKAGE / "data"
SRC = PACKAGE / "src" / "argus"
TESTS = PACKAGE / "tests"
REPORT_PATH = DATA / "standing.json"
GROUPWISE_PATH = DATA / "groupwise_audit.json"


class State(StrEnum):
    """The four states. A fifth is never added, however convenient it would be."""

    LOST = "lost"
    TIED = "tied"
    IMPLEMENTED = "implemented"
    OWNED = "owned"


ORDER = (State.LOST, State.TIED, State.IMPLEMENTED, State.OWNED)
"""Ascending. Used only to name the next state a capability could reach, never to average states."""


OWNED_CONDITIONS = (
    "best_implementation_studied",
    "best_method_studied",
    "baseline_reproduced",
    "implementation_complete",
    "same_input_comparison",
    "statistically_valid_evaluation",
    "costs_included",
    "out_of_sample_test",
    "ablation",
    "adversarial_test",
    "failure_cases_documented",
    "reproducibility_proven",
    "no_specialist_capability_superior",
)
"""The thirteen, verbatim from the project's governing rule and in its order.

They are a conjunction, not a score. Twelve of thirteen is IMPLEMENTED, because the missing one is
always the one that would have found the problem — an ablation is skipped precisely when a
component is assumed to help, and an out-of-sample test is skipped precisely when the in-sample
result looks good.
"""


class StandingError(ValueError):
    """A register entry that claims more than its evidence. Raised at import, deliberately."""


TRANSITIONS: dict[State, frozenset[State]] = {
    # LOST: a published loss can only be re-measured to level first. It cannot reappear as
    # "mechanism exists" or as a win in one edit: the rebuild has to be run against the rival.
    State.LOST: frozenset({State.LOST, State.TIED}),
    # TIED: level with the rival. Up one rung to IMPLEMENTED when ahead but not yet proven; down to
    # LOST when a rerun loses.
    State.TIED: frozenset({State.TIED, State.IMPLEMENTED, State.LOST}),
    # IMPLEMENTED: the only rung OWNED is reached from.
    State.IMPLEMENTED: frozenset({State.IMPLEMENTED, State.OWNED, State.TIED, State.LOST}),
    # OWNED: nothing is terminal. Every rung below is reachable, because evidence can be lost at
    # any rank — an artefact regenerated, a rival that turns out to lead, a gate added.
    State.OWNED: frozenset({State.OWNED, State.IMPLEMENTED, State.TIED, State.LOST}),
}
"""The closed table of legal moves between two persisted registers, as data.

The MCP specification's SEP automation keeps its lifecycle as a ``Record<State, State[]>``
(``tools/sep-automation/src/rules.ts:16-44``) that the transition handler consults before it
touches a label; this is the same shape for this register. Up is one rung at a time along
:data:`ORDER`; down is any number of rungs; staying put is always legal.
"""


class IllegalTransition(StandingError):
    """A state change between two persisted registers that :data:`TRANSITIONS` does not allow."""


def _check_transition_table() -> None:
    """The table must cover every state, contain only states, and climb exactly one rung.

    Checked at import, like an OWNED entry short of thirteen conditions: a table that let a state
    skip a rung, or forgot a state, would be the register's rule quietly changed.
    """
    if set(TRANSITIONS) != set(State):
        raise IllegalTransition("TRANSITIONS must name every state exactly once")
    for state, targets in TRANSITIONS.items():
        if state not in targets:
            raise IllegalTransition(f"{state.value} must be allowed to stay {state.value}")
        rank = ORDER.index(state)
        up = {s for s in targets if ORDER.index(s) > rank}
        expected_up = {ORDER[rank + 1]} if rank + 1 < len(ORDER) else set()
        if up != expected_up:
            raise IllegalTransition(
                f"{state.value} must climb exactly one rung, to "
                f"{', '.join(s.value for s in expected_up) or 'nothing'}")
        if {s for s in State if ORDER.index(s) < rank} - targets:
            raise IllegalTransition(f"{state.value} must be able to fall to every lower rung")


_check_transition_table()


def check_transition(name: str, before: State | None, after: State) -> None:
    """Raise :class:`IllegalTransition` unless ``before -> after`` is a legal move.

    ``before`` is ``None`` for a capability the previous register did not contain: an initial
    assignment, always legal (``rules.ts:156-161``), because a new entry is already held to the
    thirteen conditions at construction and to the artefacts by :func:`audit`. The message names the
    legal targets, as ``transition.ts``'s ``validateTransition`` does.
    """
    if before is None or after in TRANSITIONS[before]:
        return
    legal = ", ".join(s.value for s in ORDER if s in TRANSITIONS[before] and s is not before)
    raise IllegalTransition(
        f"{name!r}: {before.value} -> {after.value} is not a legal move; from {before.value} a "
        f"capability may move only to {legal}. Promote one rung per persisted register, and "
        f"record each rung's evidence")


@dataclass(frozen=True, slots=True)
class Verification:
    """One condition, and whether a machine could actually confirm it.

    Three statuses, and the distinction between them is the whole value of this record:

    ``VERIFIED``  a predicate opened the artefact and found what the condition is about.
    ``ATTESTED``  a person wrote down where they looked. The three judgement conditions can only
                  ever be this, and so can a proof that names a test but no artefact.
    ``UNPROVEN``  the condition is claimed and the evidence is not in the file.

    Reported rather than collapsed into a single pass/fail, because "ten verified, three attested"
    and "thirteen verified" are different claims and a register that cannot tell them apart is the
    press release this module was accused of being.
    """

    capability: str
    condition: str
    status: str
    detail: str

    def render(self) -> str:
        return f"{self.status:9} {self.capability} · {self.condition} — {self.detail}"


@dataclass(frozen=True, slots=True)
class Proof:
    """One piece of evidence, with the thing a reader can go and open.

    At least one of ``artefact`` or ``test`` must be present. A proof with neither is prose, and
    prose is what this module exists to replace.
    """

    condition: str
    how: str
    artefact: str = ""
    test: str = ""

    def __post_init__(self) -> None:
        if self.condition not in OWNED_CONDITIONS:
            raise StandingError(
                f"{self.condition!r} is not one of the thirteen conditions; "
                f"inventing a fourteenth is how the bar gets lowered"
            )
        if not self.artefact and not self.test:
            raise StandingError(
                f"proof of {self.condition!r} names neither an artefact nor a test: "
                f"{self.how!r}"
            )

    def locate(self) -> tuple[Path | None, Path | None]:
        """Where this proof's artefact and test file should be. Existence is :func:`audit`'s job."""
        artefact = (PACKAGE / self.artefact) if self.artefact else None
        test = (TESTS / self.test.split("::")[0]) if self.test else None
        return artefact, test

    def render(self) -> str:
        where = " · ".join(x for x in (self.artefact, self.test) if x)
        return f"{self.condition}: {self.how}  [{where}]"


@dataclass(frozen=True, slots=True)
class Capability:
    """One thing ARGUS does, its honest state, and what would move it."""

    name: str
    subtheme: str
    module: str
    state: State
    baseline: str
    proofs: tuple[Proof, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        met = self.conditions_met
        if self.state is State.OWNED and len(met) != len(OWNED_CONDITIONS):
            raise StandingError(
                f"{self.name!r} claims OWNED with {len(met)}/{len(OWNED_CONDITIONS)} conditions "
                f"proven; missing {', '.join(self.conditions_missing)}"
            )
        if self.state in (State.TIED, State.OWNED) and not self.baseline:
            raise StandingError(
                f"{self.name!r} claims {self.state.value} without naming the system it is "
                f"measured against; a comparison with nothing is not a comparison"
            )

    @property
    def conditions_met(self) -> tuple[str, ...]:
        seen = {p.condition for p in self.proofs}
        return tuple(c for c in OWNED_CONDITIONS if c in seen)

    @property
    def conditions_missing(self) -> tuple[str, ...]:
        seen = {p.condition for p in self.proofs}
        return tuple(c for c in OWNED_CONDITIONS if c not in seen)

    @property
    def next_state(self) -> State | None:
        """The one rung above this state that :data:`TRANSITIONS` allows, or ``None`` at the top.

        Read from the table rather than from :data:`ORDER`'s index arithmetic, so the rung a row
        names as its next is the rung the persisted register will actually accept.
        """
        rank = ORDER.index(self.state)
        up = [s for s in ORDER[rank + 1:] if s in TRANSITIONS[self.state]]
        return up[0] if up else None

    def render(self) -> str:
        lines = [
            f"{self.name}  [{self.state.value.upper()}]  ({self.subtheme})",
            f"  lives in: {self.module}",
            f"  measured against: {self.baseline or '(no baseline named)'}",
            f"  conditions: {len(self.conditions_met)}/{len(OWNED_CONDITIONS)}",
        ]
        for proof in self.proofs:
            lines.append(f"    ✓ {proof.render()}")
        for missing in self.conditions_missing:
            lines.append(f"    · {missing} — not established")
        for blocker in self.blockers:
            lines.append(f"  BLOCKER: {blocker}")
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subtheme": self.subtheme,
            "module": self.module,
            "state": self.state.value,
            "baseline": self.baseline,
            "conditions_met": list(self.conditions_met),
            "conditions_missing": list(self.conditions_missing),
            "proofs": [
                {"condition": p.condition, "how": p.how, "artefact": p.artefact, "test": p.test}
                for p in self.proofs
            ],
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Finding:
    """Something the register claims that the tree does not support."""

    capability: str
    problem: str
    detail: str

    def render(self) -> str:
        return f"{self.capability}: {self.problem} — {self.detail}"


@dataclass(frozen=True)
class Report:
    """The register, checked."""

    capabilities: tuple[Capability, ...]
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    verifications: tuple[Verification, ...] = field(default_factory=tuple)
    outcomes: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    """Per capability, the ``comparison_reports`` its own artefacts carry — the harness's verdict,
    read rather than restated (:func:`comparison_outcomes`)."""

    @property
    def by_verification(self) -> dict[str, int]:
        """How many conditions a machine confirmed, versus how many rest on a person's word."""
        counts = {"VERIFIED": 0, "ATTESTED": 0, "UNPROVEN": 0}
        for v in self.verifications:
            counts[v.status] = counts.get(v.status, 0) + 1
        return counts

    @property
    def earned(self) -> tuple[Capability, ...]:
        """Capabilities whose every claimed condition actually checks out.

        **Not the same set as `owned`.** `owned` is what the register *declares*; this is what
        survives inspection. Where the two disagree, the register is overstating, and the honest
        headline is this number rather than that one.
        """
        unproven = {v.capability for v in self.verifications if v.status == "UNPROVEN"}
        return tuple(c for c in self.owned if c.name not in unproven)

    @property
    def by_state(self) -> dict[str, int]:
        counts = {s.value: 0 for s in ORDER}
        for cap in self.capabilities:
            counts[cap.state.value] += 1
        return counts

    @property
    def owned(self) -> tuple[Capability, ...]:
        return tuple(c for c in self.capabilities if c.state is State.OWNED)

    @property
    def blocked(self) -> tuple[Capability, ...]:
        return tuple(c for c in self.capabilities if c.blockers)

    @property
    def clean(self) -> bool:
        return not self.findings

    def render(self) -> str:
        counts = self.by_state
        lines = [
            "CAPABILITY STANDING — four states, and the evidence behind each",
            f"  {len(self.capabilities)} capability(ies): "
            + ", ".join(f"{counts[s.value]} {s.value}" for s in reversed(ORDER)),
            "",
        ]
        for cap in self.capabilities:
            lines.append(cap.render())
            for row in self.outcomes.get(cap.name, ()):
                lines.append(f"  measured: {render_outcome(row)}")
            lines.append("")
        if self.findings:
            lines.append("REGISTER DEFECTS — evidence named but not found:")
            lines.extend(f"  - {f.render()}" for f in self.findings)
        else:
            lines.append("Every artefact and test named above exists in the tree.")
        if not self.owned:
            lines.append(
                "\nNothing is OWNED. That is the honest baseline: no capability has cleared all "
                f"{len(OWNED_CONDITIONS)} conditions, and the register says which are missing "
                "rather than rounding up."
            )
        else:
            names = ", ".join(c.name for c in self.owned)
            lines.append(
                f"\n{len(self.owned)} capability(ies) OWNED — all {len(OWNED_CONDITIONS)} "
                f"conditions cleared and independently re-checked by this same audit, not "
                f"declared and trusted: {names}. Everything else stays at its own honest state "
                f"below, which is most of the register."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "capabilities": [
                {**c.as_dict(), "measured_outcomes": list(self.outcomes.get(c.name, ()))}
                for c in self.capabilities
            ],
            "by_state": self.by_state,
            "findings": [
                {"capability": f.capability, "problem": f.problem, "detail": f.detail}
                for f in self.findings
            ],
            "clean": self.clean,
            "owned_conditions": list(OWNED_CONDITIONS),
        }


# --- the register ------------------------------------------------------------------------------
#
# Each entry states the state we can actually defend. Where a capability is strong, the missing
# conditions are still listed, because "strong" and "proven" are different words.

CAPABILITIES = Path(__file__).with_name("capabilities")
"""One TOML file per capability, in register order by filename. Until 2026-09-27 the register was a
6,300-line Python literal inside this module (audit finding 167): data a reader had to scroll past
the rules to find, and could not diff one capability at a time. The files carry every field and
every comment the literal had; :func:`load_register` is the only reader, and it refuses a key it
does not know rather than dropping it."""

_FIELDS = frozenset({"name", "subtheme", "modules", "state", "baseline", "proofs", "blockers",
                     "note"})
_PROOF_FIELDS = frozenset({"condition", "how", "artefact", "test"})


def load_register(directory: Path = CAPABILITIES) -> tuple[Capability, ...]:
    """Every capability, built through :class:`Capability` so its invariants run at import."""
    out: list[Capability] = []
    for path in sorted(directory.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        unknown = set(data) - _FIELDS
        if unknown:
            raise StandingError(f"{path.name}: unknown field(s) {sorted(unknown)}")
        proofs = []
        for proof in data.get("proofs", []):
            if set(proof) - _PROOF_FIELDS:
                raise StandingError(f"{path.name}: unknown proof field(s) "
                                    f"{sorted(set(proof) - _PROOF_FIELDS)}")
            proofs.append(Proof(**proof))
        out.append(Capability(
            name=data["name"], subtheme=data["subtheme"], module=",".join(data["modules"]),
            state=State(data["state"]), baseline=data["baseline"], proofs=tuple(proofs),
            blockers=tuple(data.get("blockers", ())), note=data.get("note", "")))
    if not out:
        raise StandingError(f"no capabilities under {directory}")
    return tuple(out)


REGISTER: tuple[Capability, ...] = load_register()


# --- what a condition actually requires inside its artefact ------------------------------------

JUDGEMENT_CONDITIONS = frozenset({
    "best_implementation_studied",
    "best_method_studied",
    "no_specialist_capability_superior",
})
"""The three conditions a machine genuinely cannot check, named rather than faked.

Whether the *best* implementation was studied, whether the *best* method was studied, and whether
any specialist capability remains superior are judgements about a field, not properties of a file.
No predicate over an artefact can establish them — a repository can contain a flawless comparison
against the second-best system in the world and look identical to one against the best.

They are therefore reported as **ATTESTED**, not VERIFIED, and an attestation must name a source a
reader can open. Pretending these were machine-checked would be the same class of dishonesty this
register exists to prevent: it is better to say three of thirteen rest on a person's word, and say
whose word and where they looked, than to let a filename stand in for a judgement.
"""

SIGNATURES: dict[str, tuple[str, ...]] = {
    # Each tuple is the vocabulary that must appear as a KEY somewhere in the artefact, at any
    # depth. Keys rather than values, deliberately: a key is a thing the producing module chose to
    # record, while a value can be any string that happens to contain the word.
    "baseline_reproduced": (
        "baseline", "reference", "parity", "agreement", "expected", "divergence",
        "max_abs_diff", "matches", "reproduce",
    ),
    "same_input_comparison": (
        "argus", "ours", "comparison", "both", "arms", "side_by_side", "versus", "vs",
        # Added 2026-09-21. A comparison is often recorded by its *outcome* rather than by the
        # word "comparison": `schedule_comparison.json` carries `ac_wins_in_sample` against a
        # TWAP arm, `queue_proof.json` carries `best_naive` and `regimes_won_in_sample`,
        # `track1_study.json` carries `beats_baseline`. Each is two arms on one input, recorded
        # under a name the original vocabulary could not see.
        #
        # Each token was tested against every artefact before being added, and two candidates
        # were REJECTED for matching things that are not comparisons: "baseline" matches
        # `standing.json` itself, because every register entry names the baseline it is measured
        # against, and "counterfactual" matches `scorecard.json`'s `mean_counterfactual_bps`,
        # which is a P&L figure. A vocabulary wide enough to match anything is the filename check
        # this module was built to replace.
        "wins", "beats", "won", "head_to_head", "naive",
    ),
    "statistically_valid_evaluation": (
        "p_value", "pvalue", "ci", "ci95", "interval", "wilson", "stderr", "std_error",
        "significance", "confidence", "n_seeds", "n_events", "sample", "nobs", "n",
    ),
    "costs_included": (
        "cost", "costs", "bps", "fee", "fees", "turnover", "net", "round_trip", "slippage",
    ),
    "out_of_sample_test": (
        "out_of_sample", "oos", "holdout", "held_out", "heldout", "train", "test", "split",
        "in_sample", "forward",
    ),
    "ablation": ("ablation", "ablations", "ablated", "arms", "variant", "without", "null_ablation"),
    "adversarial_test": (
        "adversarial", "mutant", "mutants", "attack", "attacks", "red_team", "refuted",
        "falsifier", "broken", "challenge", "violations", "sound",
    ),
    "failure_cases_documented": (
        "failure", "failures", "failure_cases", "misses", "limitation", "limitations",
        "not_verified", "weakness", "weaknesses", "undefined", "unreachable", "errors",
    ),
    "reproducibility_proven": (
        "reproducibility", "reproducible", "deterministic", "identical", "seed", "seeds",
        "digest", "hash", "generated_at", "as_of",
    ),
    "implementation_complete": (),   # structural: the module must exist. Checked separately.
}
"""What must be recorded in an artefact before a condition counts as proved.

**Every one of these used to be satisfied by a filename.** ``audit`` checked that the artefact path
existed, that the test file existed, and that a named test function appeared as a substring of it —
so "same-input comparison run", "out-of-sample test", "ablation" and "adversarial test" were all
established by a file being on disk. A capability could therefore be graded OWNED without a single
one of its thirteen claims having been evaluated, which is how 23 of 24 entries came to carry the
top grade eight days after the register recorded zero.

This is not a proof that the comparison was correct. It is a proof that the producing module wrote
down the thing the condition is about — that an artefact claiming an ablation contains ablation
arms, and one claiming out-of-sample carries a split. That is a far weaker statement than "OWNED"
sounds, and it is exactly as strong as the evidence supports, which is the point.
"""


def _keys_in(blob: Any, into: set[str]) -> None:
    """Every key name in a JSON document, at any depth. Lists are walked, not indexed."""
    if isinstance(blob, dict):
        for key, value in blob.items():
            into.add(str(key).lower())
            _keys_in(value, into)
    elif isinstance(blob, list):
        for item in blob:
            _keys_in(item, into)


def _artefact_keys(path: Path) -> set[str] | None:
    """The key vocabulary of an artefact, or ``None`` if it cannot be read as one.

    ``.jsonl`` is read line by line: several artefacts here are append-only logs, and a log whose
    first line parses is a log this can read.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    keys: set[str] = set()
    if path.suffix == ".jsonl":
        for line in text.splitlines()[:200]:
            if line.strip():
                try:
                    _keys_in(json.loads(line), keys)
                except json.JSONDecodeError:
                    return None
        return keys
    try:
        _keys_in(json.loads(text), keys)
    except json.JSONDecodeError:
        return None
    return keys


_DATA_REF = re.compile(r"data/[A-Za-z0-9_./-]+\.jsonl?")
_EVAL_SOURCE = re.compile(r"(?:src/)?argus/eval/([a-z_0-9]+)\.py")


def _data_artefact(ref: str) -> str | None:
    """``ref`` itself when it names a data artefact (``data/....json`` or ``.jsonl``), else None."""
    return ref if ref and _DATA_REF.fullmatch(ref) else None


def capability_artefacts(capability: Capability) -> tuple[str, ...]:
    """Every data artefact a capability names: its proofs' own, and each cited harness's.

    Three forms occur in the register, the same three :mod:`argus.eval.harness_validity` reads:
    a proof names ``data/<harness>.json``; a proof names the harness's source
    (``src/argus/eval/<harness>.py``); or the capability's ``module`` field lists the harness. The
    last two resolve to ``data/<harness>.json`` when that file exists.
    """
    out: set[str] = set()
    for proof in capability.proofs:
        data = _data_artefact(proof.artefact)
        if data is not None:
            out.add(data)
        source = _EVAL_SOURCE.fullmatch(proof.artefact)
        if source and (DATA / f"{source.group(1)}.json").exists():
            out.add(f"data/{source.group(1)}.json")
    for stem in _EVAL_SOURCE.findall(capability.module):
        if (DATA / f"{stem}.json").exists():
            out.add(f"data/{stem}.json")
    return tuple(sorted(out))


def proof_scope(capability: Capability, proof: Proof) -> tuple[str, ...]:
    """The artefacts a proof's groupwise check is read from.

    A proof that names a data artefact is read from that artefact alone: it chose its evidence.
    A test-only proof (or one citing source code) chose no artefact, so it is read from every data
    artefact the capability names, plus the one written by the module its test file is named for
    (``test_cointegration.py`` -> ``data/cointegration.json``), because a test named for a module
    is evidence about that module's output.
    """
    data = _data_artefact(proof.artefact)
    if data is not None:
        return (data,)
    scope = set(capability_artefacts(capability))
    if proof.test:
        stem = Path(proof.test.split("::")[0]).stem.removeprefix("test_")
        if (DATA / f"{stem}.json").exists():
            scope.add(f"data/{stem}.json")
    return tuple(sorted(scope))


def verify(proof: Proof, module_present: bool) -> tuple[str, str]:
    """Is this proof's condition actually evidenced? Returns ``(status, detail)``.

    ``VERIFIED``  — a machine opened the artefact and found what the condition is about.
    ``ATTESTED``  — one of the three judgement conditions, resting on a named source.
    ``UNPROVEN``  — the proof is claimed and the evidence is not there.
    """
    if proof.condition in JUDGEMENT_CONDITIONS:
        if not proof.how.strip():
            return "UNPROVEN", "a judgement condition with nothing written down"
        return "ATTESTED", proof.how.strip()[:160]

    if proof.condition == "implementation_complete":
        return ("VERIFIED", "module on disk") if module_present else (
            "UNPROVEN", "the module this capability names does not exist")

    artefact, _test = proof.locate()
    if artefact is None:
        # A test-only proof. The test must exist and must name the function claimed; that is
        # checked in `audit`. It is real evidence, but it is not an artefact, so it cannot be
        # inspected for content and is reported as the weaker status on purpose.
        return "ATTESTED", f"test-only evidence: {proof.test}"

    if not artefact.exists():
        return "UNPROVEN", f"{proof.artefact} is not on disk"

    keys = _artefact_keys(artefact)
    if keys is None:
        # **A non-JSON artefact is evidence this cannot read, not evidence that is absent.** Many
        # proofs here name a vendored baseline *module* — `eval/baselines/qlib_cs_processor.py` and
        # its siblings — which is exactly the right thing to cite for `baseline_reproduced`: it is
        # the specialist's own code, extracted byte-exact and pinned by SHA256. A key-vocabulary
        # predicate cannot inspect Python source, so calling it UNPROVEN would be the verifier
        # reporting its own blind spot as the register's defect. It is ATTESTED: the file exists,
        # a reader can open it, and no machine here confirmed its contents.
        return "ATTESTED", f"non-inspectable artefact on disk: {proof.artefact}"
    wanted = SIGNATURES.get(proof.condition, ())
    hits = sorted(k for k in keys if any(w in k for w in wanted))
    if not hits:
        return "UNPROVEN", (
            f"{proof.artefact} records nothing about {proof.condition}: no key among "
            f"{', '.join(wanted[:6])}..."
        )
    return "VERIFIED", f"{proof.artefact} records {', '.join(hits[:4])}"


GROUPWISE_CONDITIONS = frozenset({"statistically_valid_evaluation", "out_of_sample_test"})
"""The two conditions that also need a groupwise check on the capability's own artefact."""

GATING_ROLES = frozenset({"argus_vs_rival", "argus_result"})
"""Headline roles in ``data/groupwise_audit.json`` that are a capability's own claim. A
``context`` headline (the rival's P&L, the strategies a gate judged) is listed with its flags but
never decides a capability's standing: its flags are about something else."""


def load_groupwise(path: Path = GROUPWISE_PATH) -> dict[str, Any] | None:
    """The groupwise audit as written, or ``None`` if it is missing or unreadable."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return blob if isinstance(blob, dict) else None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def groupwise_verdict(capability: Capability, proof: Proof,
                      groupwise: dict[str, Any] | None) -> tuple[bool, str]:
    """Has a groupwise check run on this proof's artefacts, and did it contradict the headline?

    Reads :func:`proof_scope`. Passes only when at least one headline in scope is the capability's
    own claim (:data:`GATING_ROLES`) and was checked on the artefact as it is now, and no such
    headline carries a flag or favours the rival. Every way of failing names the artefact.
    """
    if groupwise is None:
        return False, ("no groupwise check has run: data/groupwise_audit.json is missing or "
                       "unreadable (python -m argus.eval.groupwise_audit)")
    scope = proof_scope(capability, proof)
    if not scope:
        return False, "names no data artefact for a groupwise check to run on"
    entries: dict[str, Any] = groupwise.get("artefacts", {})
    checked: list[str] = []
    flagged: list[str] = []
    rival: list[str] = []
    stale: list[str] = []
    missing: list[str] = []
    for ref in scope:
        entry = entries.get(ref)
        if entry is None:
            missing.append(f"{ref}: not in the audit")
            continue
        if entry.get("status") != "checked":
            missing.append(f"{ref}: {entry.get('status')} - {entry.get('reason', '')}")
            continue
        claims = [h for h in entry.get("headlines", []) if h.get("role") in GATING_ROLES]
        if not claims:
            missing.append(f"{ref}: only context headlines were checked")
            continue
        recorded = entry.get("sha256")
        if recorded is not None and recorded != _sha256(PACKAGE / ref):
            stale.append(ref)
            continue
        for h in claims:
            label = f"{ref} :: {h.get('name')}"
            checked.append(label)
            if h.get("flags"):
                flagged.append(f"{label} [{', '.join(h['flags'])}]")
            if h.get("favours") == "rival":
                rival.append(label)
    if stale:
        return False, (f"the groupwise audit predates the current {', '.join(stale)}; re-run "
                       f"python -m argus.eval.groupwise_audit")
    if flagged:
        return False, "the groupwise audit flags " + "; ".join(flagged)
    if rival:
        return False, "the per-item headline favours the rival: " + "; ".join(rival)
    if not checked:
        return False, "no groupwise check has run on its artefacts: " + "; ".join(missing)
    more = f" (+{len(checked) - 1} more)" if len(checked) > 1 else ""
    return True, f"groupwise-checked on {checked[0]}{more}"


OUTCOME_FIELDS = ("question", "rival", "metric", "outcome", "argus_score", "rival_score", "n",
                  "unit", "ci95", "p_value", "every_group", "groups", "valid", "artefact")


def comparison_outcomes(capability: Capability) -> tuple[dict[str, Any], ...]:
    """Every ``comparison_reports`` entry in the capability's artefacts, as the harness wrote it.

    The verdict a row shows is this, not a sentence in its blockers: a harness on the eval spine
    (:mod:`argus.eval.compare`) states who won, on what basis, and whether the result was valid,
    and prose restating that is a second copy that can drift from the first.
    """
    out: list[dict[str, Any]] = []
    for ref in capability_artefacts(capability):
        path = PACKAGE / ref
        if path.suffix != ".json":
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reports = blob.get("comparison_reports") if isinstance(blob, dict) else None
        for report in reports or ():
            row = {k: report.get(k) for k in OUTCOME_FIELDS}
            row["artefact"] = row["artefact"] or ref
            out.append(row)
    return tuple(out)


def render_outcome(row: dict[str, Any]) -> str:
    """One measured outcome, in the words the harness used."""
    scores = f"{row.get('argus_score')} against {row.get('rival_score')}"
    basis = (f"95% interval {row['ci95']}" if row.get("ci95") is not None
             else f"p = {row['p_value']}" if row.get("p_value") is not None else "no test")
    groups = f"; {row['every_group']}" if row.get("every_group") else ""
    valid = "" if row.get("valid") else " (INVALID: not counted)"
    return (f"{row.get('outcome')}{valid}: {row.get('question')} vs {row.get('rival')} - "
            f"{row.get('metric')}, {scores} over {row.get('n')} {row.get('unit')}(s), "
            f"{basis}{groups}  [{row.get('artefact')}]")


def audit(register: tuple[Capability, ...] = REGISTER, *,
          groupwise_path: Path = GROUPWISE_PATH) -> Report:
    """Check every named artefact, test and module actually exists.

    A missing artefact is a finding rather than an exception: artefacts are generated, and a fresh
    clone that has not run the cycle yet should get a report saying which evidence is absent, not
    an import error. A *dishonest* entry — OWNED without the thirteen — is an exception, because
    that one is a property of the register itself and no amount of running fixes it.
    """
    findings: list[Finding] = []
    verifications: list[Verification] = []
    groupwise = load_groupwise(groupwise_path)
    outcomes: dict[str, tuple[dict[str, Any], ...]] = {}
    for cap in register:
        module_present = True
        for part in cap.module.split(","):
            path = SRC.parent / part.strip()
            if part.strip() and not path.exists():
                module_present = False
                findings.append(
                    Finding(cap.name, "module not found", part.strip())
                )
        for proof in cap.proofs:
            artefact, test = proof.locate()
            if artefact is not None and not artefact.exists():
                findings.append(
                    Finding(cap.name, f"artefact for {proof.condition} is missing",
                            proof.artefact)
                )
            if test is not None:
                if not test.exists():
                    findings.append(
                        Finding(cap.name, f"test file for {proof.condition} is missing",
                                proof.test)
                    )
                elif "::" in proof.test:
                    wanted = proof.test.split("::")[-1]
                    if wanted not in test.read_text(encoding="utf-8"):
                        findings.append(
                            Finding(cap.name, f"test for {proof.condition} is not in the file",
                                    proof.test)
                        )

            # **The check that used to be missing entirely.** Everything above establishes that a
            # file is on disk and that a function name appears inside it. That is presence, not
            # proof: "same-input comparison run", "out-of-sample test", "ablation" and "adversarial
            # test" were all satisfiable by a filename, which is how 23 of 24 capabilities came to
            # be graded OWNED eight days after this register recorded zero. `verify` opens the
            # artefact and looks for the thing the condition is about.
            status, detail = verify(proof, module_present=module_present)
            # **And, since 2026-09-25, the breakdown.** A statistical or out-of-sample claim whose
            # artefact was never broken down by its groups is the aggregate the project has twice
            # been misled by. Applied after `verify`, never instead of it: the vocabulary check
            # still has to pass on its own.
            if status != "UNPROVEN" and proof.condition in GROUPWISE_CONDITIONS:
                ok, why = groupwise_verdict(cap, proof, groupwise)
                status, detail = (status, f"{detail}; {why}") if ok else ("UNPROVEN", why)
            verifications.append(Verification(cap.name, proof.condition, status, detail))
            if status == "UNPROVEN":
                findings.append(
                    Finding(cap.name, f"{proof.condition} is claimed but not evidenced", detail)
                )
        outcomes[cap.name] = comparison_outcomes(cap)
        # A harness's own verdict outranks the row's state: an OWNED capability whose artefact
        # records a valid comparison the rival won is contradicted by its own evidence.
        against = [o for o in outcomes[cap.name]
                   if o.get("outcome") == "rival_better" and o.get("valid")]
        if cap.state is State.OWNED and against:
            detail = "; ".join(f"{o.get('question')} vs {o.get('rival')}" for o in against)
            verifications.append(Verification(
                cap.name, "no_specialist_capability_superior", "UNPROVEN",
                f"a comparison report in its own artefact records the rival ahead: {detail}"))
            findings.append(Finding(cap.name, "a comparison report records the rival ahead",
                                    detail))
    return Report(
        capabilities=register, findings=tuple(findings), verifications=tuple(verifications),
        outcomes=outcomes,
    )


def summary(report: Report | None = None) -> str:
    """One line, for argus.status — and it leads with the number that survives inspection.

    **It used to quote `owned` alone, which is what the register declares about itself.** That is
    the number an adversarial review called a rubber stamp, and it was right to: 23 of 24 entries
    carried the top grade while nothing had ever opened an artefact to check one. `earned` is the
    subset whose every claimed condition actually checks out. When the two differ, the difference
    is the overstatement, and it belongs in the headline rather than in a field nobody reads.
    """
    rep = report or audit()
    counts = rep.by_state
    checks = rep.by_verification
    earned, declared = len(rep.earned), counts["owned"]
    gap = "" if earned == declared else f" ({declared} declared, {declared - earned} unevidenced)"
    return (
        f"{len(rep.capabilities)} capability(ies): {earned} owned{gap}, "
        f"{counts['implemented']} implemented, {counts['tied']} tied, {counts['lost']} lost"
        f" · conditions {checks['VERIFIED']} verified, {checks['ATTESTED']} attested, "
        f"{checks['UNPROVEN']} unproven"
    )


def transitions_since(previous: dict[str, str],
                      register: tuple[Capability, ...]) -> list[dict[str, str | None]]:
    """Every state change from ``previous`` (name -> state) to ``register``, each checked.

    Raises :class:`IllegalTransition` on the first illegal move, before anything is written. A
    capability present before and absent now is recorded as ``to: None`` rather than refused:
    rows are renamed as their claims are narrowed, and refusing a rename would freeze a wrong name.
    The record keeps a disappearance visible, which is what matters — a loss removed from the
    register is how a bad result quietly goes away.
    """
    changes: list[dict[str, str | None]] = []
    names = {cap.name for cap in register}
    for cap in register:
        raw = previous.get(cap.name)
        before = State(raw) if raw is not None else None
        check_transition(cap.name, before, cap.state)
        if before is not cap.state:
            changes.append({"capability": cap.name,
                            "from": before.value if before is not None else None,
                            "to": cap.state.value})
    for name, state in sorted(previous.items()):
        if name not in names:
            changes.append({"capability": name, "from": state, "to": None})
    return changes


def write_report(path: Path = REPORT_PATH, *, report: Report | None = None) -> Report:
    """Persist the register so a document's claim about it can be checked against a file.

    Every state change since the file on disk is checked against :data:`TRANSITIONS` first, and an
    illegal one raises before anything is written. Legal changes are appended to the persisted
    ``transition_log`` with the time, the way the SEP automation posts a comment for each label it
    moves: the history of a capability's standing is part of its evidence.
    """
    report = report or audit()
    previous_blob: dict[str, Any] = {}
    if path.exists():
        try:
            previous_blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StandingError(
                f"{path} is not readable JSON ({exc}); the previous states are what a transition "
                f"is checked against, so it must be repaired or removed deliberately") from exc
    previous = {str(c["name"]): str(c["state"]) for c in previous_blob.get("capabilities", [])}
    changes = transitions_since(previous, report.capabilities)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    log = [*previous_blob.get("transition_log", []), *({**c, "at": stamp} for c in changes)]
    blob = {**report.as_dict(), "transitions_this_write": changes, "transition_log": log}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:  # pragma: no cover - CLI
    """Regenerate `data/standing.json`.

    `write_report` existed and **nothing called it**, so the artefact drifted from the code it
    describes: on 2026-09-15 the file said 13 capabilities / 10 implemented while `audit()` computed
    14 / 11. A function that can persist the truth but is never invoked leaves a stale file wearing
    a current filename — the same defect as an artefact with no writer at all, one step milder.
    """
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = write_report()
    print(summary(report))
    unproven = [v for v in report.verifications if v.status == "UNPROVEN"]
    if unproven:
        print("")
        print(f"  {len(unproven)} condition(s) claimed with no evidence in the artefact:")
        for v in unproven:
            print(f"    {v.capability} - {v.condition} - {v.detail}")
        print("")
        print(
            "  Either make the artefact record the thing the condition is about, or lower "
            "the capability's state. A claimed condition with nothing behind it is the "
            "defect this register exists to prevent, so it fails rather than prints."
        )
    print(f"written to {REPORT_PATH}")
    # **Fails on an overstatement, not on a gap.** An unevidenced condition under an OWNED
    # capability is the register claiming something it cannot support, which is the exact defect
    # this module exists to prevent. The same condition under an IMPLEMENTED one is a known gap
    # that is already honestly labelled — printing it is useful, failing on it would leave the
    # gate permanently red and teach a reader to ignore it, which costs more than it buys.
    # Before 2026-09-20 this returned 0 unconditionally and the only check was file existence.
    overstated = len(report.owned) - len(report.earned)
    return 1 if overstated else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DATA",
    "GROUPWISE_CONDITIONS",
    "GROUPWISE_PATH",
    "ORDER",
    "OWNED_CONDITIONS",
    "REGISTER",
    "REPORT_PATH",
    "TRANSITIONS",
    "Capability",
    "Finding",
    "IllegalTransition",
    "Proof",
    "Report",
    "StandingError",
    "State",
    "audit",
    "capability_artefacts",
    "check_transition",
    "comparison_outcomes",
    "groupwise_verdict",
    "load_groupwise",
    "main",
    "proof_scope",
    "render_outcome",
    "summary",
    "transitions_since",
    "write_report",
]
