"""Review & Self-Evolution against the specialists that lead it — their code, our decisions.

The standing register re-graded "Self-evolving review rules" from OWNED to IMPLEMENTED on
2026-09-24 because the only rival it had been run against, TradingAgents' reflection memory, is not
the specialist for the sub-theme. The ones that are — each has a mechanism that decides which
lessons a trading agent keeps — had never been run on ARGUS's input:

``mnemox-ai/tradememory-protocol`` (MIT)
    Promotes a pattern to semantic memory once it has 10 episodes (`owm/induction.py:7-56`),
    tracks drift with Bayesian online changepoint detection (`owm/changepoint.py:185-355`) and, on
    a drift flag, discounts the lesson's confidence by 30% (`mcp_server.py:532`).
``cholhwanjung/trading-agent`` (no licence file)
    A statistical lesson lifecycle: admission on n ≥ 5 with a sign test at p ≤ 0.05, probation on
    independent live samples, retirement on a sign flip or redundancy (`memory/admission.py`,
    `memory/retention.py`).
``OpenByteInc/QuantDinger`` (Apache-2.0)
    A post-trade review that flags a strategy by rules over its record — win rate, profit factor,
    net result, losing streaks (`app/services/strategy_review.py:859-1139`).
``Azedfx/TradePilot-AI`` (no licence file; an S2 entry naming this sub-theme)
    A self-evolution review that marks a pattern *recurring* when it appears in two or more of the
    last twenty reviews (`server/review/review.service.ts:176-208`).

**Every rival is run from its own source, unmodified.** tradememory's, cholhwanjung's and
QuantDinger's Python is imported from the local clones; TradePilot's TypeScript is transpiled and
run by `baselines/tradepilot_review_runner.mjs`. Only I/O is shimmed — QuantDinger's database, LLM
and strategy services (never called by the review method run), cholhwanjung's SQLite store opened
in memory, TradePilot's repository answering from the stream. Nothing of the two unlicensed rivals
is copied into ARGUS; the clones are read at run time, and a missing clone is reported as *not
run*, never replaced by a reimplementation. Each rival's commit and the SHA-256 of every file run
are written into the artefact.

**The same input, for everyone.** A *candidate rule* is a condition on decision-time features and
the defect kind it claims to predict. Every gate — ARGUS's and each rival's — sees the same
candidates and the same decisions, day by day, and may use only outcomes observable before that day
(a checker's defect the day after the decision; a lean's once it has settled). Each rival's own
unit is mapped onto that input as literally as its code allows, and the mapping is stated in
:data:`MAPPINGS`. What each gate would put in front of the trader on a decision is its *warnings*.

**How a gate is scored, fixed before the first run.** Prequentially, on decisions it had not seen
the outcome of: for each defect kind, the union of the decisions any of its warnings flagged is
compared with the decisions that carried the defect. Reported: precision, the realised base rate,
lift, a one-sided Fisher exact p-value, **excess catches** (defects caught beyond what the same
number of randomly placed warnings would catch — the checklist's contribution in units of defects),
and **attention** (every warning shown, counting a rule each time it fires). On the real record the
scored decisions are the embargoed held-out half `eval/regression_gate.py` already uses, and every
candidate was written from the other half. Because the real record has no ground truth about which
rules are real, three synthetic suites with planted rules supply it:

``S1`` stationary — null rules and real rules (lift 1.5 and 2) at base rates of 5%, 25%, 60% and
75%: how many null rules does each gate warn with (false admission), how many real ones (power)?
``S2`` decay — half the real rules stop working half-way: how long does each gate keep warning with
a dead rule, and how many live rules does it lose?
``S3`` base-rate shift — the defect's base rate falls from 50% to 20% while every real rule keeps
its lift: does the gate mistake a better desk for a decaying rule?

    python -m argus.eval.review_rivals --typescript <dir containing node_modules/typescript>
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import hashlib
import json
import math
import random
import subprocess
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Protocol

from argus.desk.review import (
    STANDING_RULES,
    DefectKind,
    Status,
    _corrected,
    _upper_tail,
    features_of,
    spec_from,
)
from argus.desk.rule_lifecycle import (
    Candidate,
    Lifecycle,
    SetEvidence,
    grade,
    parameters,
)
from argus.eval.artefact import write

PACKAGE = Path(__file__).resolve().parents[3]
ROOT = PACKAGE.parent
DATA = PACKAGE / "data"
REPORT_PATH = DATA / "review_rivals.json"
PROPOSALS_PATH = DATA / "rule_proposals.json"
RUNNER = Path(__file__).resolve().parent / "baselines" / "tradepilot_review_runner.mjs"

CLONES: dict[str, Path] = {
    "tradememory": ROOT / "research" / "repos-themed" / "mnemox-ai~tradememory-protocol",
    "cholhwanjung": ROOT / "research" / "repos-owned" / "cholhwanjung~trading-agent",
    "quantdinger": ROOT / "research" / "corpus" / "repos" / "QuantDinger",
    "tradepilot": ROOT / "research" / "repos-owned" / "Azedfx~TradePilot-AI",
}
FILES_RUN: dict[str, tuple[str, ...]] = {
    "tradememory": ("src/tradememory/owm/induction.py", "src/tradememory/owm/changepoint.py",
                    "src/tradememory/owm_helpers.py"),
    "cholhwanjung": ("memory/admission.py", "memory/retention.py", "memory/store.py",
                     "memory/journal.py", "memory/influence.py"),
    "quantdinger": ("backend_api_python/app/services/strategy_review.py",),
    "tradepilot": ("server/review/review.service.ts",),
}
LICENCES = {"tradememory": "MIT", "cholhwanjung": "none (no LICENSE file; run only, not copied)",
            "quantdinger": "Apache-2.0",
            "tradepilot": "none (no LICENSE file; run only, not copied)"}

MAPPINGS = {
    "tradingagents_unconditional": (
        "TradingAgents stores a reflection on every graded decision and re-injects it with no "
        "track-record test (proven by running its vendored TradingMemoryLog in "
        "test_review_comparison.py::TestUnconditionalReuse); mapped as every candidate warning "
        "from the first day."),
    "tradememory_induction": (
        "Each observed firing of a candidate is one episodic memory with pattern_name = the rule "
        "and pnl_r = +1 if the decision was clean, -1 if it carried the targeted defect; the real "
        "check_auto_induction(threshold=10) promotes a pattern to semantic memory, which is then "
        "recalled — so a promoted rule warns."),
    "tradememory_induction_caution": (
        "As tradememory_induction, but only promoted patterns whose win_rate (share clean) is "
        "below 0.5 warn — the reading most favourable to tradememory, since induction itself "
        "promotes regardless of outcome."),
    "cholhwanjung_active": (
        "Each observed firing is one episodic entry (pattern_key = the rule, day = the decision's "
        "day, outcome = +1 clean / -1 defect) in the real MemoryStore opened in memory; the real "
        "promote_candidates, review_probation and review_retention run every day in the order "
        "its scripts/run_paper_step.py:689-694 runs them. A rule warns while its procedural "
        "(forbidden) lesson is active — retrieval reads status='active' only "
        "(memory/retrieval.py:55)."),
    "cholhwanjung_with_probation": (
        "As cholhwanjung_active, also warning while the lesson is on probation."),
    "quantdinger_review": (
        "Each candidate is reviewed as a strategy whose trades are its observed firings, with unit "
        "payoff (+1 clean, -1 defect): the real StrategyReviewService._build_rule_review is called "
        "with the metrics its own _build_metrics defines (closed trades, win rate, profit factor, "
        "window net, longest losing streak); the rule warns when the review returns any warning "
        "or danger diagnostic."),
    "quantdinger_low_win_rate": (
        "As quantdinger_review, warning only on its most specific diagnostic, low_win_rate "
        "(n >= 5 and win rate < 35%)."),
    "tradepilot_recurring": (
        "Each decision is one review session whose flagged patterns are the candidates firing on "
        "it; TradePilot's real findRecurring marks the patterns seen in >= 2 of the last 20 "
        "reviews (top 2), and those are its self-evolution warnings. It never reads an outcome."),
    "argus_review": (
        "ARGUS before this work: desk/review.py's grading (lift over base rate, one-sided Fisher, "
        "Benjamini-Hochberg across all candidates) re-run on everything observed so far; ACTIVE "
        "rules warn. Stateless, so it has no probation and no retirement."),
    "argus_lifecycle": (
        "desk/rule_lifecycle.py: the same admission, then probation on decisions not used to "
        "admit, then CUSUM retirement relative to the current base rate; rules on probation and "
        "active rules warn."),
    "argus_lifecycle_active_only": (
        "As argus_lifecycle, warning only once confirmed (the reading symmetric with "
        "cholhwanjung_active)."),
}

TRADEMEMORY_THRESHOLD = 10
"""`check_auto_induction`'s own default (`owm/induction.py:9`)."""

TRADEPILOT_WINDOW = 20
"""`findRecentReviewFindings(sessionId, 20)` — the call `findRecurring` makes
(`review.service.ts:184`)."""

TRADEMEMORY_CHANGEPOINT = 0.8
"""`CHANGEPOINT_THRESHOLD` (`owm_helpers.py:19`)."""

SCORED_KINDS = (DefectKind.GROUNDING, DefectKind.CONFLICT, DefectKind.CONTRADICTION,
                DefectKind.LEAN)


# --- the stream every gate sees ------------------------------------------------------------------


@dataclass(frozen=True)
class Stream:
    """Decisions in order, what each carried, when each outcome became observable, the candidate
    rules, and which decisions are scored."""

    name: str
    seqs: tuple[int, ...]
    day_of: dict[int, int]
    seen_from: dict[DefectKind, dict[int, int]]
    """Per kind: decision → first day index at whose start its outcome for that kind is known."""
    observed_from: dict[int, int]
    """Decision → first day its checker outcome is known (the day after it was taken)."""
    eligible: dict[DefectKind, frozenset[int]]
    marked: dict[DefectKind, frozenset[int]]
    candidates: tuple[Candidate, ...]
    scored: frozenset[int]
    kinds: tuple[DefectKind, ...]

    @property
    def days(self) -> range:
        return range(max(self.day_of.values()) + 1)

    def evidence(self, day: int) -> SetEvidence:
        eligible = {k: frozenset(s for s in self.eligible.get(k, frozenset())
                                 if self.seen_from.get(k, {}).get(s, 1 << 30) <= day)
                    for k in self.kinds}
        marked = {k: self.marked.get(k, frozenset()) & eligible[k] for k in self.kinds}
        everything = frozenset(s for s in self.seqs if self.observed_from[s] <= day)
        return SetEvidence(eligible_by_kind=eligible, marked_by_kind=marked,
                           everything=everything)

    def by_day(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {d: [] for d in self.days}
        for s in self.seqs:
            out[self.day_of[s]].append(s)
        return out

    def fires_at(self) -> dict[int, frozenset[str]]:
        acc: dict[int, set[str]] = {s: set() for s in self.seqs}
        for c in self.candidates:
            for s in c.fires:
                if s in acc:
                    acc[s].add(c.name)
        return {s: frozenset(v) for s, v in acc.items()}

    def outcome(self, kind: DefectKind, seq: int) -> bool:
        return seq in self.marked.get(kind, frozenset())


class Gate(Protocol):
    name: str

    def begin_day(self, day: int, evidence: SetEvidence) -> None: ...

    def warnings(self, seq: int) -> frozenset[str]: ...


class _ActiveSet:
    """A gate that holds a set of warning rules for the day."""

    name = ""

    def __init__(self, stream: Stream) -> None:
        self.stream = stream
        self.active: frozenset[str] = frozenset()
        self._fires = stream.fires_at()
        self.by_name = {c.name: c for c in stream.candidates}

    def warnings(self, seq: int) -> frozenset[str]:
        return self._fires.get(seq, frozenset()) & self.active

    def observed_firings(self, evidence: SetEvidence) -> Iterator[tuple[Candidate, list[int]]]:
        """Each candidate's firings whose outcome is observed, in decision order."""
        for c in self.stream.candidates:
            eligible = evidence.eligible(c.targets)
            yield c, sorted(c.fires & eligible)


def _clean(stream: Stream, candidate: Candidate, seq: int) -> bool:
    return not any(stream.outcome(k, seq) for k in candidate.targets)


# --- ARGUS ---------------------------------------------------------------------------------------


class ArgusReview(_ActiveSet):
    name = "argus_review"

    def begin_day(self, day: int, evidence: SetEvidence) -> None:
        graded = _corrected([
            grade(c, evidence.eligible(c.targets), evidence.marked(c.targets))
            for c in self.stream.candidates
        ])
        self.active = frozenset(p.rule for p in graded if p.status is Status.ACTIVE)


class ArgusLifecycle(_ActiveSet):
    def __init__(self, stream: Stream, *, include_probation: bool = True) -> None:
        super().__init__(stream)
        self.lifecycle = Lifecycle(stream.candidates)
        self.include_probation = include_probation
        self.name = "argus_lifecycle" if include_probation else "argus_lifecycle_active_only"

    def begin_day(self, day: int, evidence: SetEvidence) -> None:
        self.lifecycle.step(evidence, at=f"day {day}")
        self.active = self.lifecycle.warning(include_probation=self.include_probation)


class Unconditional(_ActiveSet):
    name = "tradingagents_unconditional"

    def begin_day(self, day: int, evidence: SetEvidence) -> None:
        self.active = frozenset(self.by_name)


# --- loading the rivals' own code ----------------------------------------------------------------


class RivalUnavailable(RuntimeError):
    """A rival could not be run from its own source. Reported, never papered over."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(name: str) -> dict[str, Any]:
    clone = CLONES[name]
    if not clone.exists():
        # Relative to the workspace: this text lands in the artefact's "not_run" field.
        where = clone.relative_to(ROOT).as_posix()
        raise RivalUnavailable(f"{name}: clone not found at {where}")
    commit = ""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        commit = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=30, check=False,
        ).stdout.strip()
    return {
        "clone": str(clone.relative_to(ROOT)).replace("\\", "/"), "commit": commit,
        "licence": LICENCES[name],
        "files": {f: _sha256(clone / f) for f in FILES_RUN[name]},
    }


@contextlib.contextmanager
def _isolated_import(root: Path, owned: Sequence[str],
                     stubs: dict[str, dict[str, Any]] | None = None) -> Iterator[None]:
    """Put ``root`` on the path for one import, then take it and every module it added back out.

    The rivals' top-level package names (``memory``, ``adapters``, ``app``) are generic; leaving
    them in :data:`sys.modules` would let a later import anywhere in the process resolve to a
    rival's module. Refuses if any of those names is already taken by something else."""
    for top in owned:
        if top in sys.modules:
            raise RivalUnavailable(f"sys.modules already holds {top!r}; refusing to shadow it")
    before = set(sys.modules)
    for name, attrs in (stubs or {}).items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
    sys.path.insert(0, str(root))
    here = str(root.resolve()).lower()
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for name in set(sys.modules) - before:
            origin = str(getattr(sys.modules[name], "__file__", "") or "")
            from_clone = origin and str(Path(origin).resolve()).lower().startswith(here)
            if name.split(".")[0] in owned or from_clone:
                del sys.modules[name]


def load_tradememory() -> dict[str, Any]:
    with _isolated_import(CLONES["tradememory"] / "src", ["tradememory"]):
        from tradememory.owm.changepoint import (  # type: ignore[import-not-found]
            BayesianChangepoint,
        )
        from tradememory.owm.induction import (  # type: ignore[import-not-found]
            check_auto_induction,
        )
    return {"check_auto_induction": check_auto_induction,
            "BayesianChangepoint": BayesianChangepoint}


def load_cholhwanjung() -> dict[str, Any]:
    with _isolated_import(CLONES["cholhwanjung"], ["memory", "adapters"]):
        from memory.admission import (  # type: ignore[import-not-found]
            promote_candidates,
            review_probation,
        )
        from memory.retention import review_retention  # type: ignore[import-not-found]
        from memory.store import MemoryStore  # type: ignore[import-not-found]
    return {"promote_candidates": promote_candidates, "review_probation": review_probation,
            "review_retention": review_retention, "MemoryStore": MemoryStore}


def load_quantdinger() -> Callable[..., Any]:
    """The real `_build_rule_review`. Its module imports three services that open a database or a
    model client at call time; the review method never calls them, so they are stubbed by name."""
    stubs: dict[str, dict[str, Any]] = {
        "app.services.llm": {"LLMService": object},
        "app.services.strategy": {"StrategyService": object},
        "app.utils.db": {"get_db_connection": lambda: None},
    }
    root = CLONES["quantdinger"] / "backend_api_python"
    with _isolated_import(root, ["app"], stubs):
        from app.services.strategy_review import (  # type: ignore[import-not-found]
            StrategyReviewService,
        )
    service = object.__new__(StrategyReviewService)  # __init__ builds a StrategyService: I/O only
    method: Callable[..., Any] = service._build_rule_review
    return method


# --- the rival gates -----------------------------------------------------------------------------


class TradeMemoryInduction(_ActiveSet):
    def __init__(self, stream: Stream, induce: Callable[..., Any], *, caution: bool) -> None:
        super().__init__(stream)
        self.induce = induce
        self.caution = caution
        self.name = "tradememory_induction_caution" if caution else "tradememory_induction"

    def begin_day(self, day: int, evidence: SetEvidence) -> None:
        episodes = [
            {"pattern_name": c.name, "pnl_r": 1.0 if _clean(self.stream, c, s) else -1.0}
            for c, seqs in self.observed_firings(evidence) for s in seqs
        ]
        promoted = self.induce(episodes, threshold=TRADEMEMORY_THRESHOLD)
        self.active = frozenset(
            p["pattern_name"] for p in promoted if not self.caution or p["win_rate"] < 0.5
        )


class Cholhwanjung(_ActiveSet):
    MARKET = "argus"
    EPOCH = date(2026, 1, 1)

    def __init__(self, stream: Stream, code: dict[str, Any], *, include_probation: bool) -> None:
        super().__init__(stream)
        self.code = code
        self.store = code["MemoryStore"](":memory:")
        self.recorded: set[tuple[str, int]] = set()
        self.include_probation = include_probation
        self.name = "cholhwanjung_with_probation" if include_probation else "cholhwanjung_active"
        self.events: list[dict[str, Any]] = []

    def _day(self, index: int) -> date:
        return self.EPOCH + timedelta(days=index)

    def begin_day(self, day: int, evidence: SetEvidence) -> None:
        for c, seqs in self.observed_firings(evidence):
            for s in seqs:
                if (c.name, s) in self.recorded:
                    continue
                self.recorded.add((c.name, s))
                self.store.add(
                    self.MARKET, "episodic", self._day(self.stream.day_of[s]),
                    f"{c.name} fired on decision {s}", {"seq": s}, pattern_key=c.name,
                    outcome=1.0 if _clean(self.stream, c, s) else -1.0,
                )
        asof = self._day(day)
        for fn in ("promote_candidates", "review_probation", "review_retention"):
            self.events.extend(self.code[fn](self.store, self.MARKET, asof))
        statuses = {"active", "probation"} if self.include_probation else {"active"}
        self.active = frozenset(
            e.pattern_key for e in self.store.query(self.MARKET, store="procedural")
            if e.status in statuses and e.data.get("kind") == "forbidden" and e.pattern_key
        )

    def lessons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for store in ("semantic", "procedural"):
            for e in self.store.query(self.MARKET, store=store):
                key = f"{store}:{e.data.get('kind')}:{e.status}"
                out[key] = out.get(key, 0) + 1
        return out


class QuantDinger(_ActiveSet):
    HIT_RATE_CODES = frozenset({"negative_expectancy_window", "profit_factor_below_one",
                                "low_win_rate", "consecutive_losses"})

    def __init__(self, stream: Stream, review: Callable[..., Any], *, strict: bool) -> None:
        super().__init__(stream)
        self.review = review
        self.strict = strict
        self.name = "quantdinger_low_win_rate" if strict else "quantdinger_review"
        self.codes: dict[str, int] = {}

    def metrics(self, outcomes: Sequence[bool]) -> dict[str, Any]:
        """`_build_metrics`'s definitions (`strategy_review.py:398-534`) at unit payoff."""
        wins = sum(outcomes)
        losses = len(outcomes) - wins
        streak = worst = 0
        for clean in outcomes:
            streak = 0 if clean else streak + 1
            worst = max(worst, streak)
        profit_factor = wins / losses if losses else (float(wins) if wins else 0.0)
        return {
            "closed_trades_with_pnl": len(outcomes),
            "win_rate": 100.0 * wins / len(outcomes) if outcomes else 0.0,
            "profit_factor": profit_factor, "window_net_pnl": float(wins - losses),
            "total_net_pnl": float(wins - losses), "max_consecutive_losses": worst,
        }

    def begin_day(self, day: int, evidence: SetEvidence) -> None:
        active: set[str] = set()
        for c, seqs in self.observed_firings(evidence):
            if not seqs:
                continue
            outcomes = [_clean(self.stream, c, s) for s in seqs]
            diagnostics, _ = self.review(metrics=self.metrics(outcomes), strategy={},
                                         trading_config={}, bot_type="signal", language="en")
            codes = {d["code"] for d in diagnostics if d["severity"] in {"warning", "danger"}}
            for code in codes:
                self.codes[code] = self.codes.get(code, 0) + 1
            if ("low_win_rate" in codes) if self.strict else bool(codes):
                active.add(c.name)
        self.active = frozenset(active)


class TradePilot(_ActiveSet):
    name = "tradepilot_recurring"

    def __init__(self, stream: Stream, recurring: dict[int, frozenset[str]]) -> None:
        super().__init__(stream)
        self.recurring = recurring

    def begin_day(self, day: int, evidence: SetEvidence) -> None:
        return None

    def warnings(self, seq: int) -> frozenset[str]:
        return self.recurring.get(seq, frozenset())


def run_tradepilot(stream: Stream, typescript: Path) -> dict[int, frozenset[str]]:
    """TradePilot's findRecurring over the stream, in one node process."""
    fires = stream.fires_at()
    payload = {"decisions": [{"id": str(s), "patterns": sorted(fires[s])} for s in stream.seqs]}
    done = subprocess.run(
        ["node", str(RUNNER), str(CLONES["tradepilot"]), str(typescript)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=600, check=False,
        encoding="utf-8",
    )
    if done.returncode != 0:
        raise RivalUnavailable(f"tradepilot runner failed: {done.stderr.strip()[:400]}")
    out: dict[int, frozenset[str]] = {}
    for line in done.stdout.splitlines():
        row = json.loads(line)
        out[int(row["id"])] = frozenset(r["id"] for r in row["recurring"])
    return out


def find_typescript(explicit: Path | None) -> Path | None:
    """The TypeScript compiler module directory, or None. Never installed on the caller's behalf:
    `npm i typescript` in a directory with its own package.json, then pass --typescript."""
    candidates = [explicit] if explicit else []
    candidates.append(ROOT / "node_modules" / "typescript")
    for c in candidates:
        if c is not None and (c / "package.json").exists():
            return c
    return None


# --- scoring -------------------------------------------------------------------------------------


@dataclass
class Score:
    gate: str
    by_kind: dict[str, dict[str, Any]] = field(default_factory=dict)
    attention: int = 0
    scored_decisions: int = 0
    warned_rules: set[str] = field(default_factory=set)
    last_warned: dict[str, int] = field(default_factory=dict)
    warnings_by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def excess(self) -> float:
        return sum(float(v["excess"]) for v in self.by_kind.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate, "scored_decisions": self.scored_decisions,
            "attention": self.attention,
            "warnings_per_decision": (self.attention / self.scored_decisions
                                      if self.scored_decisions else None),
            "rules_that_warned": len(self.warned_rules),
            "excess_catches": round(self.excess, 3),
            "excess_per_100_warnings": (round(100 * self.excess / self.attention, 3)
                                        if self.attention else None),
            "by_kind": self.by_kind,
        }


def score(stream: Stream, gate: Gate) -> Score:
    """Run ``gate`` over every day of ``stream`` and score its warnings on the scored decisions."""
    result = Score(gate=gate.name)
    targets = {c.name: c.targets for c in stream.candidates}
    flagged: dict[DefectKind, set[int]] = {k: set() for k in stream.kinds}
    days = stream.by_day()
    for day in stream.days:
        gate.begin_day(day, stream.evidence(day))
        for seq in days[day]:
            if seq not in stream.scored:
                continue
            result.scored_decisions += 1
            shown = gate.warnings(seq)
            result.attention += len(shown)
            for rule in shown:
                result.warned_rules.add(rule)
                result.last_warned[rule] = seq
                result.warnings_by_rule[rule] = result.warnings_by_rule.get(rule, 0) + 1
                for kind in targets[rule]:
                    if kind in flagged and seq in stream.eligible.get(kind, frozenset()):
                        flagged[kind].add(seq)
    for kind in stream.kinds:
        pool = stream.scored & stream.eligible.get(kind, frozenset())
        hits = pool & stream.marked.get(kind, frozenset())
        fired = flagged[kind]
        caught = len(fired & hits)
        base = len(hits) / len(pool) if pool else 0.0
        result.by_kind[str(kind)] = {
            "decisions": len(pool), "defects": len(hits), "base_rate": round(base, 4),
            "flagged": len(fired), "caught": caught,
            "precision": round(caught / len(fired), 4) if fired else None,
            "recall": round(caught / len(hits), 4) if hits else None,
            "lift": round(caught / len(fired) / base, 4) if fired and base else None,
            "excess": round(caught - len(fired) * base, 3),
            "p_value": (_upper_tail(len(pool), len(hits), len(fired), caught)
                        if fired and hits else None),
        }
    return result


# --- the gates, built once per stream ------------------------------------------------------------


@dataclass
class Rivals:
    tradememory: dict[str, Any] | None
    cholhwanjung: dict[str, Any] | None
    quantdinger: Callable[..., Any] | None
    typescript: Path | None
    unavailable: dict[str, str]


def load_rivals(typescript: Path | None) -> Rivals:
    missing: dict[str, str] = {}
    tm: dict[str, Any] | None = None
    ch: dict[str, Any] | None = None
    qd: Callable[..., Any] | None = None
    try:
        tm = load_tradememory()
    except (RivalUnavailable, ImportError) as exc:
        missing["tradememory"] = str(exc)
    try:
        ch = load_cholhwanjung()
    except (RivalUnavailable, ImportError) as exc:
        missing["cholhwanjung"] = str(exc)
    try:
        qd = load_quantdinger()
    except (RivalUnavailable, ImportError) as exc:
        missing["quantdinger"] = str(exc)
    ts = find_typescript(typescript)
    if ts is None or not CLONES["tradepilot"].exists():
        missing["tradepilot"] = ("TypeScript compiler not found (npm i typescript in its own "
                                 "project, then pass --typescript)" if ts is None
                                 else "clone not found")
    return Rivals(tm, ch, qd, ts if "tradepilot" not in missing else None, missing)


def gates_for(stream: Stream, rivals: Rivals) -> list[Gate]:
    gates: list[Gate] = [
        Unconditional(stream), ArgusReview(stream), ArgusLifecycle(stream),
        ArgusLifecycle(stream, include_probation=False),
    ]
    if rivals.tradememory is not None:
        induce = rivals.tradememory["check_auto_induction"]
        gates += [TradeMemoryInduction(stream, induce, caution=False),
                  TradeMemoryInduction(stream, induce, caution=True)]
    if rivals.cholhwanjung is not None:
        gates += [Cholhwanjung(stream, rivals.cholhwanjung, include_probation=False),
                  Cholhwanjung(stream, rivals.cholhwanjung, include_probation=True)]
    if rivals.quantdinger is not None:
        gates += [QuantDinger(stream, rivals.quantdinger, strict=False),
                  QuantDinger(stream, rivals.quantdinger, strict=True)]
    if rivals.typescript is not None:
        gates.append(TradePilot(stream, run_tradepilot(stream, rivals.typescript)))
    return gates


# --- the real record -----------------------------------------------------------------------------


def _candidate_specs(proposals: dict[str, Any]) -> tuple[list[Any], int]:
    """Every distinct rule the proposers wrote (admitted by the regression gate or not), and the
    latest decision any of them was written from."""
    seen: dict[str, Any] = {}
    latest = -1
    for proposer in proposals.get("proposers", {}).values():
        for row in proposer.get("results", []):
            pair = row.get("pair") or {}
            latest = max(latest, int(pair.get("bad_seq", -1)), int(pair.get("good_seq", -1)))
            raw = row.get("rule")
            if not raw:
                continue
            spec = spec_from(raw)
            key = json.dumps([spec.as_dict()["conditions"], sorted(map(str, spec.targets))],
                             sort_keys=True)
            if key not in seen:
                seen[key] = spec
    names: dict[str, int] = {}
    out = []
    for spec in seen.values():
        names[spec.name] = names.get(spec.name, 0) + 1
        out.append((spec, names[spec.name]))
    return out, latest


def real_stream() -> tuple[Stream, dict[str, Any]]:
    """ARGUS's recorded decisions as a stream, and the facts about how it was cut."""
    from argus.eval.regression_gate import load_record, split_windows

    records, defects, graded, settled = load_record()
    fit, held, embargoed = split_windows(records, defects, lean_graded=graded, settled_at=settled)
    fit_last = max(int(r.get("seq", 0)) for r in fit.records)
    proposals = json.loads(PROPOSALS_PATH.read_text(encoding="utf-8"))
    specs, written_from = _candidate_specs(proposals)
    if written_from > fit_last:
        raise ValueError(f"a candidate was written from decision {written_from}, after the fit "
                         f"window ends at {fit_last}: the held-out scoring would leak")
    ordered = sorted(records, key=lambda r: int(r.get("seq", 0)))
    first = datetime.fromisoformat(str(ordered[0]["at"])).astimezone(UTC).date()

    def day(at: Any) -> int:
        return (datetime.fromisoformat(str(at)).astimezone(UTC).date() - first).days

    seqs = tuple(int(r.get("seq", 0)) for r in ordered)
    day_of = {int(r.get("seq", 0)): day(r["at"]) for r in ordered}
    observed = {s: d + 1 for s, d in day_of.items()}
    seen_from: dict[DefectKind, dict[int, int]] = {
        k: dict(observed) for k in SCORED_KINDS if k is not DefectKind.LEAN
    }
    seen_from[DefectKind.LEAN] = {s: day(at) + 1 for s, at in settled.items() if s in graded}
    everyone = frozenset(seqs)
    eligible = {k: everyone for k in SCORED_KINDS}
    eligible[DefectKind.LEAN] = frozenset(graded) & everyone
    marked = {k: frozenset(d.seq for d in defects if d.kind is k) & everyone
              for k in SCORED_KINDS}

    feats = {int(r.get("seq", 0)): features_of(r) for r in ordered}
    candidates: list[Candidate] = []
    for rule in STANDING_RULES:
        fires: set[int] = set()
        for r in ordered:
            try:
                if rule.predicate(r):
                    fires.add(int(r.get("seq", 0)))
            except Exception:  # a broken predicate must not take the replay down
                continue
        candidates.append(Candidate(rule.name, rule.targets, frozenset(fires)))
    for spec, n in specs:
        name = spec.name if n == 1 else f"{spec.name}~{n}"
        candidates.append(Candidate(
            name, spec.targets, frozenset(s for s in seqs if spec.matches(feats[s]))))
    stream = Stream(
        name="argus_record", seqs=seqs, day_of=day_of, seen_from=seen_from,
        observed_from=observed, eligible=eligible, marked=marked,
        candidates=tuple(candidates), scored=frozenset(int(r.get("seq", 0)) for r in held.records),
        kinds=SCORED_KINDS,
    )
    facts = {
        "decisions": len(seqs), "days": len(stream.days),
        "first_decision": str(ordered[0]["at"]), "last_decision": str(ordered[-1]["at"]),
        "fit_decisions": len(fit.records), "fit_last_seq": fit_last,
        "held_out_scored": len(stream.scored), "embargoed": embargoed,
        "candidates": len(candidates), "hand_written": len(STANDING_RULES),
        "machine_written": len(specs), "candidates_written_from_last_seq": written_from,
        "defects": {str(k): len(v) for k, v in marked.items()},
    }
    return stream, facts


# --- synthetic ground truth ----------------------------------------------------------------------


@dataclass(frozen=True)
class Planted:
    name: str
    lift: float
    """1.0 is a null rule."""
    fire_rate: float
    decays_at: int | None = None
    """Decision index from which the rule's lift becomes 1.0."""


def synthetic_stream(
    *, seed: int, planted: Sequence[Planted], decisions: int, base: float,
    base_after: float | None = None, change_at: int | None = None, per_day: int = 44,
    scored_from: int,
) -> Stream:
    """One defect kind, decisions ``per_day`` a day, outcomes observed the next day.

    Each decision fires each planted rule independently at its fire rate; the defect probability is
    the base rate (``base_after`` from ``change_at``) times the lift of every real, not-yet-decayed
    rule that fired, capped at 0.95. Ground truth is which rules are real."""
    rng = random.Random(seed)
    kind = DefectKind.CONFLICT
    fires: dict[str, set[int]] = {p.name: set() for p in planted}
    marked: set[int] = set()
    for i in range(decisions):
        b = base_after if base_after is not None and change_at is not None and i >= change_at \
            else base
        p = b
        for rule in planted:
            if rng.random() < rule.fire_rate:
                fires[rule.name].add(i)
                live = rule.decays_at is None or i < rule.decays_at
                if live:
                    p *= rule.lift
        if rng.random() < min(0.95, p):
            marked.add(i)
    seqs = tuple(range(decisions))
    day_of = {i: i // per_day for i in seqs}
    observed = {i: d + 1 for i, d in day_of.items()}
    return Stream(
        name=f"synthetic-{seed}", seqs=seqs, day_of=day_of, seen_from={kind: observed},
        observed_from=observed, eligible={kind: frozenset(seqs)}, marked={kind: frozenset(marked)},
        candidates=tuple(Candidate(p.name, frozenset({kind}), frozenset(fires[p.name]))
                         for p in planted),
        scored=frozenset(i for i in seqs if i >= scored_from), kinds=(kind,),
    )


def _wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959964
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": _wilson(k, n)}


FIRE_RATES = (0.05, 0.1, 0.2, 0.3)


def suite_stationary(rivals: Rivals, *, seeds: int, bases: Sequence[float]) -> dict[str, Any]:
    """S1: 30 null rules and 10 real ones (five at lift 1.5, five at 2.0), 600 decisions, scored on
    the second half."""
    planted = [Planted(f"null-{i}", 1.0, FIRE_RATES[i % 4]) for i in range(30)] + [
        Planted(f"real-{i}", 1.5 if i < 5 else 2.0, FIRE_RATES[i % 4]) for i in range(10)
    ]
    out: dict[str, Any] = {}
    for base in bases:
        tallies: dict[str, dict[str, float]] = {}
        for seed in range(seeds):
            stream = synthetic_stream(seed=1000 * int(base * 100) + seed, planted=planted,
                                      decisions=600, base=base, scored_from=300)
            for gate in gates_for(stream, rivals):
                s = score(stream, gate)
                t = tallies.setdefault(gate.name, {"null_warned": 0, "real_warned": 0,
                                                   "excess": 0.0, "attention": 0, "caught": 0,
                                                   "flagged": 0, "runs": 0})
                t["null_warned"] += sum(1 for r in s.warned_rules if r.startswith("null-"))
                t["real_warned"] += sum(1 for r in s.warned_rules if r.startswith("real-"))
                t["excess"] += s.excess
                t["attention"] += s.attention
                kind = s.by_kind[str(DefectKind.CONFLICT)]
                t["caught"] += kind["caught"]
                t["flagged"] += kind["flagged"]
                t["runs"] += 1
        out[f"base_{base:.2f}"] = {
            name: {
                "false_admission": _rate(int(t["null_warned"]), 30 * int(t["runs"])),
                "power": _rate(int(t["real_warned"]), 10 * int(t["runs"])),
                "mean_excess_catches": round(t["excess"] / t["runs"], 3),
                "mean_attention": round(t["attention"] / t["runs"], 1),
                "precision": round(t["caught"] / t["flagged"], 4) if t["flagged"] else None,
            }
            for name, t in sorted(tallies.items())
        }
    return out


def _changepoint_flags(bocpd: Any, outcomes: Sequence[bool], change_index: int) -> dict[str, Any]:
    """tradememory's real detector on one rule's firing outcomes, fed as its own helper feeds it
    (`owm_helpers.py:96-107`): won = clean, pnl_r = +1/-1, flag when P(changepoint) > 0.8."""
    detector = bocpd(hazard_lambda=50.0)
    before = 0
    first_after: int | None = None
    for i, clean in enumerate(outcomes):
        cp = detector.update({"won": clean, "pnl_r": 1.0 if clean else -1.0})
        if cp.changepoint_probability > TRADEMEMORY_CHANGEPOINT:
            if i < change_index:
                before += 1
            elif first_after is None:
                first_after = i - change_index
    return {"flags_before_change": before, "firings_to_first_flag_after": first_after}


def suite_decay(rivals: Rivals, *, seeds: int) -> dict[str, Any]:
    """S2: 10 real rules at lift 2.0 on a 30% base rate; five stop working at decision 600 of
    1200. Scored after the change."""
    change = 600
    planted = [Planted(f"null-{i}", 1.0, FIRE_RATES[i % 4]) for i in range(20)] + [
        Planted(f"decays-{i}", 2.0, FIRE_RATES[i % 4] if i % 4 else 0.1, decays_at=change)
        for i in range(5)
    ] + [Planted(f"stays-{i}", 2.0, FIRE_RATES[i % 4] if i % 4 else 0.1) for i in range(5)]
    tallies: dict[str, dict[str, Any]] = {}
    bocpd_rows: list[dict[str, Any]] = []
    for seed in range(seeds):
        stream = synthetic_stream(seed=77_000 + seed, planted=planted, decisions=1200, base=0.3,
                                  scored_from=change)
        tail = 1200 - 100
        for gate in gates_for(stream, rivals):
            s = score(stream, gate)
            t = tallies.setdefault(gate.name, {"dead_warnings": 0, "dead_still_warning": 0,
                                               "live_lost": 0, "delays": [], "excess": 0.0,
                                               "runs": 0, "live_warnings": 0})
            for i in range(5):
                dead, live = f"decays-{i}", f"stays-{i}"
                t["dead_warnings"] += s.warnings_by_rule.get(dead, 0)
                t["live_warnings"] += s.warnings_by_rule.get(live, 0)
                last = s.last_warned.get(dead)
                if last is not None and last >= tail:
                    t["dead_still_warning"] += 1
                elif last is not None:
                    t["delays"].append(last - change)
                if s.last_warned.get(live, -1) < tail:
                    t["live_lost"] += 1
            t["excess"] += s.excess
            t["runs"] += 1
        if rivals.tradememory is not None:
            kind = DefectKind.CONFLICT
            for c in stream.candidates:
                if c.name.startswith(("decays-", "stays-")):
                    ordered = sorted(c.fires)
                    outcomes = [not stream.outcome(kind, s) for s in ordered]
                    k = bisect.bisect_left(ordered, change)
                    bocpd_rows.append({"rule": c.name.split("-")[0], **_changepoint_flags(
                        rivals.tradememory["BayesianChangepoint"], outcomes, k)})
    table = {
        name: {
            "dead_rule_warnings_after_change": t["dead_warnings"],
            "dead_rules_still_warning_at_end": _rate(t["dead_still_warning"], 5 * t["runs"]),
            "median_decisions_until_dead_rule_stops": (median(t["delays"]) if t["delays"]
                                                      else None),
            "live_rule_warnings_after_change": t["live_warnings"],
            "live_rules_lost_by_end": _rate(t["live_lost"], 5 * t["runs"]),
            "mean_excess_catches": round(t["excess"] / t["runs"], 3),
        }
        for name, t in sorted(tallies.items())
    }
    if bocpd_rows:
        decays = [r for r in bocpd_rows if r["rule"] == "decays"]
        stays = [r for r in bocpd_rows if r["rule"] == "stays"]
        detected = [r["firings_to_first_flag_after"] for r in decays
                    if r["firings_to_first_flag_after"] is not None]
        table["tradememory_bocpd_detector"] = {
            "note": ("tradememory's real BayesianChangepoint on each rule's firing outcomes. "
                     "A flag only multiplies the lesson's confidence by 0.7 (mcp_server.py:532); "
                     "the lesson is never retired, so its warnings are those of "
                     "tradememory_induction."),
            "decayed_rules_flagged_after_change": _rate(len(detected), len(decays)),
            "median_firings_to_flag": median(detected) if detected else None,
            "stable_rules_with_a_false_flag_after_change": _rate(
                sum(1 for r in stays if r["firings_to_first_flag_after"] is not None),
                len(stays)),
            "rules_with_a_flag_before_change": _rate(
                sum(1 for r in bocpd_rows if r["flags_before_change"]), len(bocpd_rows)),
        }
    return table


def suite_base_shift(rivals: Rivals, *, seeds: int) -> dict[str, Any]:
    """S3: 10 real rules at lift 1.8 throughout; the base rate falls from 50% to 20% at decision
    600 of 1200. A rule that keeps its lift is still worth its place."""
    change = 600
    planted = [Planted(f"null-{i}", 1.0, FIRE_RATES[i % 4]) for i in range(20)] + [
        Planted(f"real-{i}", 1.8, FIRE_RATES[i % 4] if i % 4 else 0.1) for i in range(10)]
    tallies: dict[str, dict[str, Any]] = {}
    for seed in range(seeds):
        stream = synthetic_stream(seed=88_000 + seed, planted=planted, decisions=1200, base=0.5,
                                  base_after=0.2, change_at=change, scored_from=change)
        tail = 1200 - 100
        for gate in gates_for(stream, rivals):
            s = score(stream, gate)
            t = tallies.setdefault(gate.name, {"lost": 0, "null_warned": 0, "excess": 0.0,
                                               "runs": 0})
            t["lost"] += sum(1 for i in range(10)
                             if s.last_warned.get(f"real-{i}", -1) < tail)
            t["null_warned"] += sum(1 for r in s.warned_rules if r.startswith("null-"))
            t["excess"] += s.excess
            t["runs"] += 1
    return {
        name: {
            "real_rules_lost_by_end": _rate(t["lost"], 10 * t["runs"]),
            "null_rules_warning_after_shift": _rate(t["null_warned"], 20 * t["runs"]),
            "mean_excess_catches": round(t["excess"] / t["runs"], 3),
        }
        for name, t in sorted(tallies.items())
    }


# --- the run -------------------------------------------------------------------------------------


def run(*, typescript: Path | None, seeds: int = 40) -> dict[str, Any]:
    rivals = load_rivals(typescript)
    stream, facts = real_stream()
    real: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    for gate in gates_for(stream, rivals):
        real[gate.name] = score(stream, gate).as_dict()
        if isinstance(gate, ArgusLifecycle) and gate.include_probation:
            detail["argus_lifecycle"] = gate.lifecycle.as_dict()
        if isinstance(gate, Cholhwanjung) and not gate.include_probation:
            detail["cholhwanjung_lessons"] = gate.lessons()
        if isinstance(gate, QuantDinger) and not gate.strict:
            detail["quantdinger_diagnostic_counts"] = gate.codes
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "fixed_before_running": True,
            "scoring": ("prequential: a gate's warnings on a decision come from outcomes observed "
                        "before that decision's day; per defect kind the union of flagged "
                        "decisions is scored against the defects; excess = caught - flagged x "
                        "realised base rate"),
            "argus_lifecycle_parameters": parameters(),
            "synthetic_seeds_per_cell": seeds,
            "qwen_calls": 0,
        },
        "rivals": {name: (provenance(name) if name not in rivals.unavailable
                          else {"not_run": rivals.unavailable[name]}) for name in CLONES},
        "mappings": MAPPINGS,
        "real_record": {"facts": facts, "gates": real, "detail": detail},
        "synthetic": {
            "S1_stationary": suite_stationary(rivals, seeds=seeds,
                                              bases=(0.05, 0.25, 0.60, 0.75)),
            "S2_decay": suite_decay(rivals, seeds=seeds),
            "S3_base_rate_shift": suite_base_shift(rivals, seeds=seeds),
        },
    }
    report["verdict"] = verdict(report)
    return report


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Who led each measurement, stated as numbers, with ARGUS's primary gate named first."""
    real = report["real_record"]["gates"]
    ranked = sorted(real.items(), key=lambda kv: -kv[1]["excess_catches"])
    lines = ["real record, excess catches: " + ", ".join(
        f"{name} {row['excess_catches']:+.1f} ({row['attention']} warnings)"
        for name, row in ranked)]
    s1 = report["synthetic"]["S1_stationary"]
    for base, rows in s1.items():
        lines.append(f"S1 {base}: " + ", ".join(
            f"{name} FA {row['false_admission']['rate']} / power {row['power']['rate']}"
            for name, row in rows.items()))
    s2 = report["synthetic"]["S2_decay"]
    lines.append("S2: " + ", ".join(
        f"{name} dead-still-warning {row['dead_rules_still_warning_at_end']['rate']}, "
        f"live-lost {row['live_rules_lost_by_end']['rate']}"
        for name, row in s2.items() if "dead_rules_still_warning_at_end" in row))
    s3 = report["synthetic"]["S3_base_rate_shift"]
    lines.append("S3: " + ", ".join(
        f"{name} real-lost {row['real_rules_lost_by_end']['rate']}" for name, row in s3.items()))
    return {"leader_on_real_record_by_excess": ranked[0][0] if ranked else None,
            "summary": lines}


def render(report: dict[str, Any]) -> list[str]:
    return [f"[review-rivals] {line}" for line in report["verdict"]["summary"]]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="review lifecycle vs the specialists")
    parser.add_argument("--typescript", type=Path, default=None,
                        help="directory of the typescript npm package (for TradePilot)")
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args(argv)
    report = run(typescript=args.typescript, seeds=args.seeds)
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"written to {REPORT_PATH}")
    return 0


__all__ = [
    "CLONES",
    "MAPPINGS",
    "REPORT_PATH",
    "ArgusLifecycle",
    "ArgusReview",
    "Planted",
    "RivalUnavailable",
    "Stream",
    "Unconditional",
    "gates_for",
    "load_rivals",
    "real_stream",
    "run",
    "score",
    "synthetic_stream",
]


if __name__ == "__main__":
    raise SystemExit(main())
