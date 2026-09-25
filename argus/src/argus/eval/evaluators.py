"""Many small evaluators, combined with AND, cheapest first: the grading half of the eval spine.

Every ``eval/*_comparison.py`` module used to own its own idea of what "scored" means. Some wrote a
verdict string, some a pair of counts, some a mean over whichever rows survived; none of them could
say, in a form another module could read, *which checks the result had to pass before it counted*.
This module is that shared vocabulary. It is deliberately small: a check, a tier, a combinator and
an all-or-nothing aggregate. ``eval/compare.py`` builds the always-emitted comparison report on top
of it, and ``eval/harness_validity.py`` uses it to grade the harnesses themselves.

**What was taken, from where** (licences verified upstream 2026-09-25, see
``research/mypr-teardowns/_SYNTHESIS.md`` section 5; the local ``mypr`` LICENSE files were edited):

* **AND by multiplication** — WebArena's ``EvaluatorComb`` multiplies every evaluator's score, so a
  single zero fails the task however well the others did (``evaluation_harness/evaluators.py:
  336-352``, web-arena-x/webarena, **Apache-2.0**; the notice ships at
  ``argus/licenses/webarena-APACHE-2.0.txt``). :func:`combine` keeps that rule. Changed: WebArena
  routes evaluators from a task config (``evaluator_router``, ``:355-374``); here the caller passes
  them, because every ARGUS harness already knows what it measures.
* **Cheap rules before any model judge** — ToolBench's ``normalized_openai_completions`` settles a
  comparison by hash equality and empty-answer rules and only calls the model when those cannot
  decide (``toolbench/tooleval/evaluators/registered_cls/tooleval.py:110-196``, OpenBMB/ToolBench,
  **Apache-2.0**; notice at ``argus/licenses/toolbench-APACHE-2.0.txt``). :class:`Tier` makes the
  order explicit and :func:`combine` sorts by it, so a model judge is never paid for on a subject a
  free rule has already failed. Rejected: ToolBench's ``random.choice`` tie-breaks
  (``tooleval.py:117,145,190-194``) — a grade that changes between runs on the same input is not a
  grade.
* **An error is not a score** — mle-bench's ``Grader.__call__`` turns an invalid submission and any
  unexpected exception into ``None`` and logs the grading function's ``file:line``, rather than
  either crashing the run or letting a half-graded result through (``mlebench/grade_helpers.py:
  36-55``, openai/mle-bench, **MIT**). :func:`combine` records an evaluator that raised as
  ``passed=None`` with the evaluator's location, and ``None`` fails an AND: no evidence that a
  check passed is not a pass.
* **All or nothing** — TweetEval reports its aggregate only when every task produced a result
  (``is_all_good``, ``evaluation_script.py:100-101``). TweetEval publishes **no licence**, so no
  code was taken; :func:`all_or_nothing` is rebuilt from that one observable behaviour. The point
  is the one the TweetEval authors made: a mean over the components that happened to finish is a
  different number from the benchmark's mean, and must not be printed under the same name.

**Why evaluators here are functions of one subject rather than of a trajectory.** WebArena's
evaluators take a browser trajectory, a config file, a page and a CDP session. ARGUS's subjects are
already-computed reports and rows; binding an evaluator to one typed subject keeps each one a pure
function a test can call directly.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Tier(IntEnum):
    """How much an evaluator costs to run. :func:`combine` runs lower tiers first.

    The order is the whole point: a rule costs nothing, a statistic costs compute, a model judge
    costs money on a metered key and is the least trustworthy of the three (it has to be audited
    itself — see ``eval/metaeval.py``). Nothing more expensive runs on a subject a cheaper check
    has already failed.
    """

    RULE = 0
    """A deterministic predicate on the subject: a count, a presence check, an equality."""

    STATISTIC = 1
    """A computed test: a bootstrap, a sign test, a reproduction of a published figure."""

    MODEL = 2
    """An LLM judge. Always last, never run once a cheaper check has failed."""


@dataclass(frozen=True)
class Check:
    """One evaluator's result on one subject.

    ``passed`` is ``None`` when the evaluator did not produce a verdict — it raised, or it was
    skipped because an earlier check had already failed. Both are recorded, never dropped, so a
    reader can tell "failed" from "never looked".
    """

    name: str
    tier: Tier
    passed: bool | None
    score: float
    detail: str = ""
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "tier": self.tier.name.lower(), "passed": self.passed,
                "score": self.score, "detail": self.detail, "skipped": self.skipped}

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> Check:
        return cls(name=str(blob["name"]), tier=Tier[str(blob["tier"]).upper()],
                   passed=blob["passed"], score=float(blob["score"]),
                   detail=str(blob.get("detail", "")), skipped=bool(blob.get("skipped", False)))


@dataclass(frozen=True)
class Evaluator(Generic[T]):
    """A named, tiered check. ``fn`` returns a bool, or a ``(bool, detail)`` pair, or a float score
    in ``[0, 1]`` (``1.0`` passes, anything lower fails — WebArena's scores are the same shape)."""

    name: str
    tier: Tier
    fn: Callable[[T], bool | float | tuple[bool, str]]

    def __call__(self, subject: T) -> Check:
        outcome = self.fn(subject)
        detail = ""
        if isinstance(outcome, tuple):
            ok, detail = outcome
            return Check(self.name, self.tier, bool(ok), 1.0 if ok else 0.0, detail)
        if isinstance(outcome, bool):
            return Check(self.name, self.tier, outcome, 1.0 if outcome else 0.0, detail)
        score = float(outcome)
        return Check(self.name, self.tier, score >= 1.0, score, detail)


def rule(name: str, fn: Callable[[T], bool | float | tuple[bool, str]]) -> Evaluator[T]:
    """A zero-cost deterministic check."""
    return Evaluator(name, Tier.RULE, fn)


def statistic(name: str, fn: Callable[[T], bool | float | tuple[bool, str]]) -> Evaluator[T]:
    """A computed test (bootstrap, sign test, reproduction of a published figure)."""
    return Evaluator(name, Tier.STATISTIC, fn)


def model_judge(name: str, fn: Callable[[T], bool | float | tuple[bool, str]]) -> Evaluator[T]:
    """An LLM judge. :func:`combine` runs it last and only if every cheaper check passed."""
    return Evaluator(name, Tier.MODEL, fn)


@dataclass(frozen=True)
class Grade:
    """Every evaluator combined with AND.

    ``passed`` is True only if every evaluator ran and passed. ``score`` is the product of the
    scores, WebArena's rule, with a check that did not produce a verdict counting as zero.
    ``model_calls`` counts the model-tier evaluators that actually ran — the number a metered key
    is billed for — so a harness can report what grading cost.
    """

    passed: bool
    score: float
    checks: tuple[Check, ...]
    model_calls: int = 0

    @property
    def first_failure(self) -> Check | None:
        return next((c for c in self.checks if c.passed is not True and not c.skipped), None)

    def as_dict(self) -> dict[str, Any]:
        failure = self.first_failure
        return {"passed": self.passed, "score": self.score, "model_calls": self.model_calls,
                "first_failure": failure.name if failure else None,
                "checks": [c.as_dict() for c in self.checks]}

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> Grade:
        return cls(passed=bool(blob["passed"]), score=float(blob["score"]),
                   checks=tuple(Check.from_dict(c) for c in blob["checks"]),
                   model_calls=int(blob.get("model_calls", 0)))


def _where(fn: Callable[..., Any]) -> str:
    """``file:line`` of an evaluator's function, as mle-bench logs a grader that raised."""
    try:
        return f"{inspect.getsourcefile(fn)}:{inspect.getsourcelines(fn)[1]}"
    except (TypeError, OSError):
        return repr(fn)


def combine(evaluators: Sequence[Evaluator[T]], subject: T, *,
            stop_at_first_failure: bool = True) -> Grade:
    """Run ``evaluators`` on ``subject`` cheapest tier first and AND the results.

    The sort is stable, so within a tier the caller's order is kept. With
    ``stop_at_first_failure`` (the default) every evaluator after the first failure is recorded as
    skipped rather than run; a :data:`Tier.MODEL` evaluator is *always* skipped after a failure,
    even with it off, because paying a judge to grade a subject that has already failed a free rule
    is the waste ToolBench's ordering exists to avoid.

    An evaluator that raises is recorded as ``passed=None`` with its location and the exception,
    and fails the AND. It does not propagate: the grade is always produced.
    """
    if not evaluators:
        raise ValueError("combine() needs at least one evaluator: an empty AND passes everything")
    ordered = sorted(evaluators, key=lambda e: e.tier)
    checks: list[Check] = []
    failed = False
    model_calls = 0
    for evaluator in ordered:
        if failed and (stop_at_first_failure or evaluator.tier is Tier.MODEL):
            checks.append(Check(evaluator.name, evaluator.tier, None, 0.0,
                                "not run: an earlier check failed", skipped=True))
            continue
        if evaluator.tier is Tier.MODEL:
            model_calls += 1
        try:
            check = evaluator(subject)
        except Exception as exc:
            check = Check(evaluator.name, evaluator.tier, None, 0.0,
                          f"raised {type(exc).__name__}: {exc} (at {_where(evaluator.fn)})")
        checks.append(check)
        if check.passed is not True:
            failed = True
    score = 1.0
    for check in checks:
        score *= check.score if check.passed is not None else 0.0
    return Grade(passed=not failed, score=score, checks=tuple(checks), model_calls=model_calls)


@dataclass(frozen=True)
class Aggregate:
    """A mean that exists only when every component does.

    ``value`` is ``None`` whenever any component is missing, and ``missing`` names them. The mean
    over the present components is kept as ``partial`` under a different name, so it can be shown
    without being mistaken for the aggregate.
    """

    value: float | None
    components: int
    missing: tuple[str, ...] = field(default_factory=tuple)
    partial: float | None = None

    @property
    def complete(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "components": self.components,
                "missing": list(self.missing), "partial_mean_of_present": self.partial}


def all_or_nothing(components: Mapping[str, float | None]) -> Aggregate:
    """The mean of ``components``, or ``None`` if any of them is ``None``.

    Rebuilt from TweetEval's observable behaviour (no code taken; the repository has no licence):
    the aggregate is published only when every component produced a result. An empty mapping has
    no aggregate either — a mean of nothing is not zero.
    """
    missing = tuple(name for name, value in components.items() if value is None)
    present = [value for value in components.values() if value is not None]
    partial = sum(present) / len(present) if present else None
    if missing or not components:
        return Aggregate(None, len(components), missing, partial)
    return Aggregate(partial, len(components), (), partial)
