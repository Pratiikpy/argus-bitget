"""Root-cause diagnosis — from "you are overconfident" to "here is the rule that would have helped".

:class:`argus.desk.workbench.ErrorProfile` already detects *patterns*: a calibration gap, a
concentration of failure modes, a symbol that keeps costing money. That is one step further than
the field, and it is still one step short of useful, because a pattern is a description and a trader
needs a change.

This module closes that step. It takes the graded autopsies and produces **diagnoses**: a named
cause, the evidence in the record that supports it, and a checkable rule that would have changed
the outcome. The rule is the deliverable — "you are early on reversals" is an observation, "wait for
a second confirming close before entering a reversal" is something you can follow and later test.

The reference is ``levkila-trade``'s ``auditeur.py:40-50`` (proprietary, patterns only), whose Chief
Performance Officer audits the entry rationale of closed trades against the outcome and converts raw
metrics into causes: a fee-to-gross-profit ratio above 30% is diagnosed as *overtrading or a
timeframe too short*, not merely reported as a number. That conversion — metric to cause to remedy —
is the part worth taking. Its adversarial framing ("be cold, ruthless, data-driven") is a prompt
instruction and is not; the diagnoses here are deterministic arithmetic over the record, because a
model grading its own decisions is the failure this project has documented elsewhere.

**Three rules this module obeys.**

*A diagnosis needs a minimum sample and says so.* Under the floor it returns the count rather than a
confident story, matching :class:`ErrorProfile`'s own refusal to draw a pattern from four decisions.

*Every diagnosis carries its evidence.* A cause with no decision ids behind it cannot be checked,
and an unfalsifiable diagnosis is worse than none because it cannot be wrong.

*The remedy is checkable.* Each one names a condition that can be evaluated against a future
decision, so the next review can ask whether following it helped rather than whether it sounded
sensible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from argus.desk.workbench import Autopsy

MIN_FOR_DIAGNOSIS = 5
"""The same floor ErrorProfile uses. Below it, a pattern is a coincidence with a name."""

MIN_FOR_SUBGROUP = 3
"""A cause attributed to a subgroup needs at least this many cases inside it."""

OVERCONFIDENCE_GAP = 0.15
FEE_SHARE_OF_GROSS = Decimal("0.30")
"""levkila-trade's threshold: fees above this share of gross profit indicate overtrading rather
than a bad thesis."""

MAGNITUDE_SHARE = 0.4
"""When this share of failures are sizing errors rather than direction errors."""


@dataclass(frozen=True)
class Diagnosis:
    """A named cause, its evidence, and a rule that would have changed the outcome."""

    cause: str
    severity: str
    """``primary`` when it explains most of the damage, ``contributing`` otherwise."""

    evidence: tuple[str, ...]
    """Decision ids. A cause with nothing behind it cannot be checked."""

    detail: str
    remedy: str
    """A condition checkable against a future decision, not advice."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "severity": self.severity,
            "decisions": list(self.evidence),
            "detail": self.detail,
            "remedy": self.remedy,
        }

    def render(self) -> str:
        return (
            f"[{self.severity}] {self.cause}: {self.detail} "
            f"(decisions {', '.join(self.evidence[:5])}) -> {self.remedy}"
        )


CONFIDENT_LEVEL = 0.6
"""Above this a wrong call was made with conviction; below it the desk said it was unsure.

The distinction is not cosmetic and it is already in the codebase:
:attr:`argus.desk.workbench.Autopsy.failure_mode` separates ``overconfident`` (wrong and sure) from
``thesis`` (wrong and appropriately unsure), because they call for different responses. A first
version of this diagnosis ignored that and reported a desk stating 35% and losing every time as
"overconfident", which is the wrong word and points at the wrong fix — that desk's problem is that
its numbers do not mean anything, not that it is too bold.
"""


def _calibration(autopsies: Sequence[Autopsy]) -> Diagnosis | None:
    """A gap between stated confidence and realised hit rate, named by which gap it is."""
    stated = sum(a.stated_confidence for a in autopsies) / len(autopsies)
    realised = sum(1 for a in autopsies if a.direction_correct) / len(autopsies)
    gap = stated - realised
    if gap <= OVERCONFIDENCE_GAP:
        return None

    confident_losses = [a for a in autopsies if a.failure_mode == "overconfident"]
    shared = (
        f"stated confidence averages {stated:.0%} against a realised hit rate of "
        f"{realised:.0%}, a gap of {gap:.0%} across {len(autopsies)} graded decisions"
    )

    if stated >= CONFIDENT_LEVEL:
        return Diagnosis(
            cause="overconfidence",
            severity=(
                "primary" if len(confident_losses) >= len(autopsies) * 0.3 else "contributing"
            ),
            evidence=(
                tuple(a.decision_id for a in confident_losses)
                or tuple(a.decision_id for a in autopsies)
            ),
            detail=f"{shared}; the losing calls were made with conviction",
            remedy=(
                "size on the realised hit rate rather than the stated one until the gap closes "
                f"below {OVERCONFIDENCE_GAP:.0%}; the calibration gate in argus.risk.sizing "
                "already refuses to size on confidence this far out"
            ),
        )

    # Stated low and still worse than stated: the numbers are not wrong in the bold direction,
    # they are not carrying information at all.
    return Diagnosis(
        cause="miscalibration",
        severity="contributing",
        evidence=tuple(a.decision_id for a in autopsies),
        detail=(
            f"{shared}; the desk said it was unsure and was still worse than it said, so the "
            f"stated probability is not tracking outcomes in either direction"
        ),
        remedy=(
            "treat stated confidence as unusable for sizing until it separates winners from "
            "losers at all: check whether high-confidence calls beat low-confidence ones before "
            "trusting the level of either"
        ),
    )


def _sizing_not_thesis(autopsies: Sequence[Autopsy]) -> Diagnosis | None:
    """Right direction, wrong size is a different problem from being wrong about the world."""
    magnitude = [a for a in autopsies if a.failure_mode == "magnitude"]
    failures = [a for a in autopsies if a.failure_mode != "correct"]
    if len(magnitude) < MIN_FOR_SUBGROUP or not failures:
        return None
    share = len(magnitude) / len(failures)
    if share < MAGNITUDE_SHARE:
        return None
    mean_error = sum(a.magnitude_error_bps for a in magnitude) / len(magnitude)
    return Diagnosis(
        cause="sizing, not thesis",
        severity="primary" if share > 0.6 else "contributing",
        evidence=tuple(a.decision_id for a in magnitude),
        detail=(
            f"{len(magnitude)} of {len(failures)} failures ({share:.0%}) had the direction right "
            f"and the size wrong, by {mean_error:.0f}bps on average"
        ),
        remedy=(
            "the thesis process is working and the sizing is not: hold the entry rule and scale "
            "position size to the analogue distribution's interquartile range rather than to "
            "conviction"
        ),
    )


def _fee_drag(
    autopsies: Sequence[Autopsy], *, gross_pnl: Decimal | None, fees: Decimal | None
) -> Diagnosis | None:
    """levkila-trade's rule: fees eating a third of gross is overtrading, not bad picking."""
    if gross_pnl is None or fees is None or gross_pnl <= 0:
        return None
    share = fees / gross_pnl
    if share < FEE_SHARE_OF_GROSS:
        return None
    return Diagnosis(
        cause="fee drag",
        severity="primary" if share > Decimal("0.5") else "contributing",
        evidence=tuple(a.decision_id for a in autopsies),
        detail=(
            f"fees are {share:.0%} of gross profit; on this venue the round trip is 12bps against "
            f"a measured intraday edge near zero, so turnover is the cost, not the market"
        ),
        remedy=(
            "reduce the number of round trips rather than holding longer: a position held twice as "
            "long still pays the same round trip, so only trading less often lowers cost per unit "
            "of return"
        ),
    )


def _repeated_blind_spot(autopsies: Sequence[Autopsy]) -> Diagnosis | None:
    """The same evidence ignored across several losses is a process gap, not bad luck."""
    counts: dict[str, list[str]] = {}
    for autopsy in autopsies:
        if autopsy.failure_mode == "correct":
            continue
        for ignored in autopsy.evidence_ignored:
            counts.setdefault(ignored, []).append(autopsy.decision_id)
    repeated = {k: v for k, v in counts.items() if len(v) >= MIN_FOR_SUBGROUP}
    if not repeated:
        return None
    worst = max(repeated, key=lambda k: len(repeated[k]))
    return Diagnosis(
        cause="repeated blind spot",
        severity="primary" if len(repeated[worst]) >= 4 else "contributing",
        evidence=tuple(repeated[worst]),
        detail=(
            f"{worst!r} was available and unused in {len(repeated[worst])} losing decisions; "
            f"the same evidence being skipped repeatedly is a process gap rather than bad luck"
        ),
        remedy=(
            f"make {worst!r} a required field in the decision record: a thesis that does not "
            f"mention it is incomplete rather than merely brief"
        ),
    )


@dataclass
class DiagnosisReport:
    diagnoses: tuple[Diagnosis, ...] = ()
    sample: int = 0
    refused: str = ""

    @property
    def usable(self) -> bool:
        return not self.refused

    @property
    def primary(self) -> tuple[Diagnosis, ...]:
        return tuple(d for d in self.diagnoses if d.severity == "primary")

    def as_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "refused": self.refused,
            "sample": self.sample,
            "primary": len(self.primary),
            "diagnoses": [d.as_dict() for d in self.diagnoses],
        }

    def render(self) -> list[str]:
        if not self.usable:
            return [f"[root-cause] {self.refused}"]
        if not self.diagnoses:
            return [
                f"[root-cause] {self.sample} graded decisions and no dominant cause; the losses "
                f"look like variance rather than a repeatable mistake"
            ]
        return [f"[root-cause] {d.render()}" for d in self.diagnoses]


def diagnose(
    autopsies: Sequence[Autopsy],
    *,
    gross_pnl: Decimal | None = None,
    fees: Decimal | None = None,
) -> DiagnosisReport:
    """Turn graded decisions into named causes with checkable remedies.

    ``gross_pnl`` and ``fees`` are optional because the fee-drag diagnosis needs them and the rest
    do not; omitting them removes one diagnosis rather than degrading the others.
    """
    if len(autopsies) < MIN_FOR_DIAGNOSIS:
        return DiagnosisReport(
            sample=len(autopsies),
            refused=(
                f"{len(autopsies)} graded decision(s); {MIN_FOR_DIAGNOSIS} are needed before a "
                f"cause can be told apart from a coincidence"
            ),
        )

    found = [
        d for d in (
            _calibration(autopsies),
            _sizing_not_thesis(autopsies),
            _fee_drag(autopsies, gross_pnl=gross_pnl, fees=fees),
            _repeated_blind_spot(autopsies),
        )
        if d is not None
    ]
    found.sort(key=lambda d: (d.severity != "primary", d.cause))
    return DiagnosisReport(diagnoses=tuple(found), sample=len(autopsies))


__all__ = [
    "FEE_SHARE_OF_GROSS",
    "MIN_FOR_DIAGNOSIS",
    "MIN_FOR_SUBGROUP",
    "Diagnosis",
    "DiagnosisReport",
    "diagnose",
]
