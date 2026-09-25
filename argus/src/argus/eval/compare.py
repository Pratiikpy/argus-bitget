"""The report every head-to-head emits, whatever happened: the reporting half of the eval spine.

Before this module, each ``eval/*_comparison.py`` wrote its own dict and its verdict lived in one of
four places: a ``"verdict"`` string inside a bootstrap block (``overnight``, ``void``), two counts a
reader had to subtract (``claimcheck``), two means with no test at all (``stopquality``), or only in
the prose of ``eval/standing.py``'s blockers. A reader — or ``standing.py`` itself — could not ask
every harness the same question: *who won, on what basis, and is the result valid?*
:class:`ComparisonReport` is that question's answer, in one shape, and :func:`emit` guarantees it
exists even when the scoring code raised.

**What was taken, from where** (licences verified upstream 2026-09-25; see
``research/mypr-teardowns/_SYNTHESIS.md`` section 5):

* **A report object that always exists** — mle-bench's ``CompetitionReport`` is written for every
  competition, including the ones with no submission or an invalid one, with ``submission_exists``
  and ``valid_submission`` as fields and the score ``None`` rather than absent
  (``mlebench/grade_helpers.py:160-224``, openai/mle-bench, **MIT**). :class:`ComparisonReport`
  keeps that: :data:`Outcome.NO_RESULT` is a verdict, not a missing file, and ``inputs_exist`` /
  ``valid`` are always present. Its ``to_dict``/``from_dict`` round trip is the same pattern.
* **Rank against a reference, with the thresholds in the report** — ``Grader.rank_score`` returns
  the thresholds beside the medal booleans (``grade_helpers.py:57-155``). Here the equivalent is
  that the confidence interval, the p-value and the rule used (``basis``) travel with the outcome,
  so a reader can re-derive the verdict from the report alone. Rejected: the Kaggle medal cut-offs
  themselves (``:84-117``) — hand-set constants for a different population
  (``_SYNTHESIS.md`` section 4).
* **Grading failures become ``None``, with the grader's location** — ``Grader.__call__``
  (``grade_helpers.py:36-55``). :func:`emit` does the same for a comparison whose scoring code
  raises: the report is still produced, as :data:`Outcome.NO_RESULT`, carrying the exception and
  the ``file:line`` it came from.

**What is ARGUS's own.** The paired bootstrap below is the one ``void_comparison`` and
``overnight_comparison`` each carried a private copy of (resampling the shared unit — a night or a
weekend — never the row, because every stock shares a night's news). It moved here unchanged, and
both modules now call it; their migrated verdicts are pinned against their pre-migration artefacts
in ``tests/test_eval_spine.py``. The exact sign test for paired binary outcomes is new: it gives
count-based harnesses (``claimcheck``) a test where they previously had only a subtraction.
"""

from __future__ import annotations

import inspect
import random
import statistics
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from math import comb
from typing import Any

from argus.eval.evaluators import Evaluator, Grade, all_or_nothing, combine, rule

SEPARATION = 0.05
"""Two-sided level for the exact sign test. The bootstrap's own level is its 95% interval."""


class Outcome(StrEnum):
    """Who won one comparison. Always one of these five; never a missing field."""

    ARGUS_BETTER = "argus_better"
    RIVAL_BETTER = "rival_better"
    LEVEL = "level"
    """Identical results on every scored unit — no difference to test."""

    NOT_SEPARABLE = "not_separable"
    """A difference, but the test cannot tell it from noise at the stated level."""

    NO_RESULT = "no_result"
    """Scoring failed, an input was missing, or the result failed a validity check."""


@dataclass(frozen=True)
class PairedBootstrap:
    """Mean paired difference ``a - b`` with a percentile interval, resampled over units."""

    mean_diff: float
    low: float
    high: float
    units: int
    draws: int
    seed: int
    all_zero: bool

    @property
    def legacy_verdict(self) -> str:
        """The string ``void``/``overnight`` wrote before migration, for a lower-is-better diff."""
        return "a better" if self.high < 0 else "b better" if self.low > 0 else "not separable"

    def outcome(self, *, argus_is_a: bool) -> Outcome:
        """Lower is better: a whole interval below zero means ``a`` had the smaller error."""
        if self.all_zero:
            return Outcome.LEVEL
        if self.high < 0:
            return Outcome.ARGUS_BETTER if argus_is_a else Outcome.RIVAL_BETTER
        if self.low > 0:
            return Outcome.RIVAL_BETTER if argus_is_a else Outcome.ARGUS_BETTER
        return Outcome.NOT_SEPARABLE


def paired_bootstrap(diffs_by_unit: Mapping[str, Sequence[float]], *, draws: int = 4000,
                     seed: int) -> PairedBootstrap:
    """Resample whole units (nights, weekends) with replacement; 95% percentile interval.

    This is, operation for operation, the loop ``void_comparison._paired`` and
    ``overnight_comparison._paired`` carried before the spine existed: keys sorted, one
    ``rng.choice`` per key per draw, the rows of each drawn unit concatenated, the draw's mean
    taken, the means sorted and read at ``int(0.025 * draws)`` and ``int(0.975 * draws) - 1``.
    Keeping the order of random draws identical is what lets the migrated modules reproduce their
    published intervals to the last digit.
    """
    keys = sorted(diffs_by_unit)
    if not keys or not any(diffs_by_unit[k] for k in keys):
        raise ValueError("a paired bootstrap needs at least one unit with at least one difference")
    rng = random.Random(seed)
    means = sorted(statistics.mean([d for k in (rng.choice(keys) for _ in keys)
                                    for d in diffs_by_unit[k]]) for _ in range(draws))
    point = statistics.mean(d for k in keys for d in diffs_by_unit[k])
    return PairedBootstrap(
        mean_diff=point, low=means[int(0.025 * draws)], high=means[int(0.975 * draws) - 1],
        units=len(keys), draws=draws, seed=seed,
        all_zero=all(d == 0 for k in keys for d in diffs_by_unit[k]))


LEGACY_VERDICTS = ("a better", "b better", "not separable")
"""The three strings the pre-spine paired blocks wrote. Anything else is not a legacy verdict."""


def legacy_outcome(verdict: str, *, argus_is_a: bool) -> Outcome:
    """Read a pre-spine ``"a better" / "b better" / "not separable"`` string as an Outcome.

    Raises on anything else, so a renamed or corrupted artefact cannot be read as a verdict.
    """
    if verdict not in LEGACY_VERDICTS:
        raise ValueError(f"not a legacy paired verdict: {verdict!r}")
    if verdict == "not separable":
        return Outcome.NOT_SEPARABLE
    a_won = verdict == "a better"
    return Outcome.ARGUS_BETTER if a_won == argus_is_a else Outcome.RIVAL_BETTER


def sign_test(a_only: int, b_only: int) -> float:
    """Exact two-sided sign test on the discordant pairs of a paired binary comparison.

    ``a_only`` counts units only ``a`` got right, ``b_only`` units only ``b`` got right; units both
    or neither got right carry no information about which is better and are not arguments. With no
    discordant pairs at all the p-value is 1.0 — there is nothing to test.
    """
    if a_only < 0 or b_only < 0:
        raise ValueError("counts cannot be negative")
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail: float = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def sign_test_outcome(a_only: int, b_only: int, *, argus_is_a: bool = True) -> Outcome:
    """Outcome of :func:`sign_test` at :data:`SEPARATION`, LEVEL when there is no discordance."""
    if a_only == b_only == 0:
        return Outcome.LEVEL
    if sign_test(a_only, b_only) >= SEPARATION:
        return Outcome.NOT_SEPARABLE
    a_wins = a_only > b_only
    return Outcome.ARGUS_BETTER if a_wins == argus_is_a else Outcome.RIVAL_BETTER


@dataclass(frozen=True)
class ComparisonReport:
    """One head-to-head, in the one shape every harness emits.

    ``outcome`` is the validated verdict. When a validity check fails it is
    :data:`Outcome.NO_RESULT` and the verdict the numbers would have given is kept in
    ``unvalidated_outcome`` — shown, never counted, as mle-bench keeps the thresholds of an
    invalid submission without awarding a medal.
    """

    comparison: str
    question: str
    argus: str
    rival: str
    metric: str
    lower_is_better: bool
    argus_score: float | None
    rival_score: float | None
    n: int
    unit: str
    outcome: Outcome
    basis: str
    ci95: tuple[float, float] | None = None
    p_value: float | None = None
    scored: int = 0
    total: int = 0
    groups: Mapping[str, str] = field(default_factory=dict)
    every_group: str | None = None
    inputs_exist: bool = True
    valid: bool = True
    unvalidated_outcome: Outcome | None = None
    validity: Grade | None = None
    errors: tuple[str, ...] = ()
    artefact: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison": self.comparison, "question": self.question, "argus": self.argus,
            "rival": self.rival, "metric": self.metric, "lower_is_better": self.lower_is_better,
            "argus_score": self.argus_score, "rival_score": self.rival_score, "n": self.n,
            "unit": self.unit, "outcome": self.outcome.value, "basis": self.basis,
            "ci95": list(self.ci95) if self.ci95 is not None else None,
            "p_value": self.p_value, "scored": self.scored, "total": self.total,
            "groups": dict(self.groups), "every_group": self.every_group,
            "inputs_exist": self.inputs_exist, "valid": self.valid,
            "unvalidated_outcome": (self.unvalidated_outcome.value
                                    if self.unvalidated_outcome is not None else None),
            "validity": self.validity.as_dict() if self.validity is not None else None,
            "errors": list(self.errors), "artefact": self.artefact,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> ComparisonReport:
        ci = blob.get("ci95")
        unvalidated = blob.get("unvalidated_outcome")
        validity = blob.get("validity")
        return cls(
            comparison=str(blob["comparison"]), question=str(blob["question"]),
            argus=str(blob["argus"]), rival=str(blob["rival"]), metric=str(blob["metric"]),
            lower_is_better=bool(blob["lower_is_better"]),
            argus_score=None if blob["argus_score"] is None else float(blob["argus_score"]),
            rival_score=None if blob["rival_score"] is None else float(blob["rival_score"]),
            n=int(blob["n"]), unit=str(blob["unit"]), outcome=Outcome(blob["outcome"]),
            basis=str(blob["basis"]),
            ci95=(float(ci[0]), float(ci[1])) if ci is not None else None,
            p_value=None if blob.get("p_value") is None else float(blob["p_value"]),
            scored=int(blob.get("scored", 0)), total=int(blob.get("total", 0)),
            groups={str(k): str(v) for k, v in (blob.get("groups") or {}).items()},
            every_group=blob.get("every_group"),
            inputs_exist=bool(blob.get("inputs_exist", True)), valid=bool(blob.get("valid", True)),
            unvalidated_outcome=Outcome(unvalidated) if unvalidated is not None else None,
            validity=Grade.from_dict(validity) if validity is not None else None,
            errors=tuple(str(e) for e in blob.get("errors") or ()),
            artefact=str(blob.get("artefact", "")), created_at=str(blob.get("created_at", "")))


def _outcome_agrees_with_evidence(r: ComparisonReport) -> tuple[bool, str]:
    """The verdict must be the one its own interval, p-value and scores give."""
    if r.ci95 is not None:
        low, high = r.ci95
        # Differences are always stated ARGUS minus rival in the report.
        lower_wins = high < 0
        upper_wins = low > 0
        if lower_wins or upper_wins:
            argus_ahead = lower_wins == r.lower_is_better
            want = Outcome.ARGUS_BETTER if argus_ahead else Outcome.RIVAL_BETTER
            return r.outcome is want, f"interval [{low}, {high}] gives {want.value}"
        ok = r.outcome in (Outcome.NOT_SEPARABLE, Outcome.LEVEL)
        return ok, f"interval [{low}, {high}] spans zero"
    if r.p_value is not None and r.p_value >= SEPARATION:
        ok = r.outcome in (Outcome.NOT_SEPARABLE, Outcome.LEVEL)
        return ok, f"p = {r.p_value} is not below {SEPARATION}"
    if r.outcome in (Outcome.ARGUS_BETTER, Outcome.RIVAL_BETTER):
        if r.argus_score is None or r.rival_score is None:
            return False, "a winner was named without both scores"
        argus_ahead = (r.argus_score < r.rival_score) == r.lower_is_better
        if r.argus_score == r.rival_score:
            return False, "a winner was named on equal scores"
        want = Outcome.ARGUS_BETTER if argus_ahead else Outcome.RIVAL_BETTER
        return r.outcome is want, f"scores {r.argus_score} vs {r.rival_score}"
    return True, "no winner named"


VALIDITY: tuple[Evaluator[ComparisonReport], ...] = (
    rule("inputs_exist", lambda r: (r.inputs_exist, "inputs were read")),
    rule("something_scored", lambda r: (r.n > 0 and r.scored > 0,
                                        f"n={r.n} {r.unit}, {r.scored} rows scored")),
    rule("coverage_declared", lambda r: (0 < r.scored <= r.total,
                                         f"{r.scored} of {r.total} rows scored")),
    rule("both_sides_scored", lambda r: (r.argus_score is not None and r.rival_score is not None,
                                         f"ARGUS {r.argus_score}, rival {r.rival_score}")),
    rule("outcome_agrees_with_its_evidence", _outcome_agrees_with_evidence),
)
"""The rules a comparison must pass before its verdict counts. All free; no model judge."""


def finalise(report: ComparisonReport, *,
             extra: Sequence[Evaluator[ComparisonReport]] = ()) -> ComparisonReport:
    """Run :data:`VALIDITY` (and ``extra``) with AND; demote the outcome to NO_RESULT on failure.

    ``every_group`` is set only if every group produced a result, and then says whether ARGUS won
    all of them — an all-or-nothing aggregate: a harness where two names failed to score does not
    get to say "ahead on every name".
    """
    grade = combine((*VALIDITY, *extra), report)
    groups = dict(report.groups)
    every: str | None = None
    if groups:
        # The share of groups ARGUS won, published only if every group produced a result.
        aggregate = all_or_nothing({
            name: None if verdict == Outcome.NO_RESULT.value
            else float(verdict == Outcome.ARGUS_BETTER.value)
            for name, verdict in groups.items()})
        if aggregate.value is not None:
            distinct = set(groups.values())
            every = f"{distinct.pop()}_on_every_group" if len(distinct) == 1 else "mixed"
    stamped = replace(report, validity=grade, every_group=every,
                      created_at=report.created_at or datetime.now(UTC).isoformat(
                          timespec="seconds"))
    if grade.passed:
        return replace(stamped, valid=True, unvalidated_outcome=None)
    return replace(stamped, valid=False, outcome=Outcome.NO_RESULT,
                   unvalidated_outcome=report.outcome)


def emit(build: Callable[[], ComparisonReport], *, comparison: str, question: str, argus: str,
         rival: str, metric: str, lower_is_better: bool, unit: str,
         artefact: str = "") -> ComparisonReport:
    """Build a report; if building raises, still return one, as NO_RESULT with the cause.

    mle-bench's grader returns ``None`` and logs ``file:line`` rather than letting one bad
    submission take the run down (``grade_helpers.py:36-55``); this is the same contract for a
    comparison. The identifying fields are passed separately from ``build`` precisely so they are
    available when ``build`` is the thing that failed.
    """
    try:
        return finalise(build())
    except Exception as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
        where = f"{frame.filename}:{frame.lineno}" if frame else _where(build)
        failed = ComparisonReport(
            comparison=comparison, question=question, argus=argus, rival=rival, metric=metric,
            lower_is_better=lower_is_better, argus_score=None, rival_score=None, n=0, unit=unit,
            outcome=Outcome.NO_RESULT, basis="scoring raised before a verdict existed",
            inputs_exist=not isinstance(exc, FileNotFoundError), valid=False,
            errors=(f"{type(exc).__name__}: {exc} (at {where})",), artefact=artefact,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"))
        return replace(failed, validity=combine(VALIDITY, failed))


def _where(fn: Callable[..., Any]) -> str:
    try:
        return f"{inspect.getsourcefile(fn)}:{inspect.getsourcelines(fn)[1]}"
    except (TypeError, OSError):
        return repr(fn)


def from_bootstrap(boot: PairedBootstrap, *, comparison: str, question: str, argus: str,
                   rival: str, metric: str, unit: str, argus_score: float | None,
                   rival_score: float | None, scored: int, total: int,
                   groups: Mapping[str, str] | None = None, artefact: str = "",
                   created_at: str = "") -> ComparisonReport:
    """A lower-is-better paired comparison, ARGUS as ``a``, as a report (not yet finalised)."""
    return ComparisonReport(
        comparison=comparison, question=question, argus=argus, rival=rival, metric=metric,
        lower_is_better=True, argus_score=argus_score, rival_score=rival_score, n=boot.units,
        unit=unit, outcome=boot.outcome(argus_is_a=True),
        basis=(f"paired bootstrap over {unit}s, {boot.draws} draws, seed {boot.seed}; "
               f"ARGUS better if the whole 95% interval of (ARGUS - rival) error is below zero"),
        ci95=(boot.low, boot.high), scored=scored, total=total, groups=dict(groups or {}),
        artefact=artefact, created_at=created_at)
