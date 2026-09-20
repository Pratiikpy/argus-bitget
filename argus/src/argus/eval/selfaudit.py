"""Self-audit — the checks we run against our own results before publishing them.

Every other evaluation module here asks "how did the desk do". This one asks "is that number
trustworthy", and it is pointed inward on purpose. Catching ourselves is worth more than the
headline, because a judge who finds a flattered result has learned something about the whole
submission, whereas a judge who reads our own flag on it has learned something about our method.

Ported from ``agent-backtest-lab`` (Apache-2.0), whose ``abl/leakage/reward_hacking.py:57-172``
splits a return series in-sample / out-of-sample and looks for the signatures of a result that was
fitted rather than found:

* **Sharpe collapse** — strong in-sample, gone out-of-sample. Their thresholds (IS above 1.0, a drop
  beyond 1.5, OOS below 0.5) are kept, because they were chosen against a body of real backtests and
  inventing our own would be worse-founded, not better.
* **Drawdown widening** — the out-of-sample trough materially deeper than the in-sample one, which
  is what happens when risk was tuned to a window rather than controlled.
* **Calibration divergence** — stated confidence tracking outcomes in-sample and drifting out of
  sample. ARGUS measures expected calibration error already, so this is the cheapest of the three
  and the one most specific to a system that states its own confidence.

**Two departures.** Theirs returns flags only when there is enough data; below the minimum it
returns an empty list, which reads identically to "we looked and found nothing wrong". Here an
undersized sample produces an explicit :data:`Finding` of severity ``inconclusive``, for the same
reason :mod:`argus.research.overfit` reports INCONCLUSIVE rather than FAIL. And the leakage check
adds one rule that only applies to a live decision log rather than a backtest: a decision whose
evidence is stamped *after* the decision instant is not overfitting, it is time travel, and it
invalidates the row outright rather than casting doubt on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, TypeVar

from argus.backtest.metrics import MetricError, max_drawdown, sharpe
from argus.eval.observatory import Prediction, expected_calibration_error
from argus.eval.performance import PERIODS_PER_YEAR, daily_series
from argus.paper.ledger import Entry, PaperLedger

IN_SAMPLE_FRACTION = 0.7
MIN_PERIODS_PER_SIDE = 8
"""Below this a split produces two samples too small to compare. Reported, not silently skipped."""

SHARPE_IS_FLOOR = 1.0
SHARPE_DROP = 1.5
SHARPE_OOS_CEILING = 0.5
SHARPE_RETENTION = 0.5
"""Out-of-sample Sharpe below half of in-sample: the handbook's own overfitting alert,
and the same threshold `backtest.metrics.out_of_sample_decay` applies."""
DRAWDOWN_FLOOR = 0.10
DRAWDOWN_WIDENING = 1.5
ECE_DIVERGENCE = 0.15


class Severity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"
    CLEAN = "clean"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": str(self.severity),
            "message": self.message,
            **self.detail,
        }


_T = TypeVar("_T")


def _split(values: Sequence[_T]) -> tuple[list[_T], list[_T]]:
    """Chronological split. Generic because the same cut applies to returns and to predictions,
    and duplicating it once per element type is how the two drift apart."""
    cut = int(len(values) * IN_SAMPLE_FRACTION)
    return list(values[:cut]), list(values[cut:])


def _equity(returns: Sequence[float]) -> list[float]:
    curve = [1.0]
    for r in returns:
        curve.append(curve[-1] * (1 + r))
    return curve


def check_sharpe_collapse(returns: Sequence[float]) -> Finding:
    """Strong in-sample, gone out-of-sample: the signature of a fitted result."""
    in_sample, out_sample = _split(returns)
    if len(in_sample) < MIN_PERIODS_PER_SIDE or len(out_sample) < MIN_PERIODS_PER_SIDE:
        return Finding(
            "SHARPE_DROP", Severity.INCONCLUSIVE,
            f"{len(in_sample)}/{len(out_sample)} periods either side of the split; "
            f"{MIN_PERIODS_PER_SIDE} are needed on each to compare",
        )
    try:
        is_sharpe = sharpe(in_sample, periods_per_year=PERIODS_PER_YEAR)
        oos_sharpe = sharpe(out_sample, periods_per_year=PERIODS_PER_YEAR)
    except MetricError as exc:
        return Finding("SHARPE_DROP", Severity.INCONCLUSIVE, str(exc))

    drop = is_sharpe - oos_sharpe
    detail = {
        "in_sample_sharpe": round(is_sharpe, 3),
        "out_of_sample_sharpe": round(oos_sharpe, 3),
        "drop": round(drop, 3),
    }
    retention = oos_sharpe / is_sharpe if is_sharpe > 0 else 0.0
    detail["retention_ratio"] = round(retention, 3)

    if is_sharpe > SHARPE_IS_FLOOR and drop > SHARPE_DROP and oos_sharpe < SHARPE_OOS_CEILING:
        return Finding(
            "SHARPE_DROP", Severity.CRITICAL,
            f"Sharpe collapses from {is_sharpe:.2f} in-sample to {oos_sharpe:.2f} out-of-sample; "
            f"a drop of {drop:.2f} is consistent with a result fitted to the visible window",
            detail,
        )
    # The warning tier is *relative*, not absolute, and that is a correction rather than a
    # preference. An absolute drop of 1.5 is serious at a Sharpe of 2 and meaningless at a Sharpe
    # of 8, so the absolute form flagged a deliberately steady fixture as decaying. The threshold
    # used here is the handbook's own alert and the one `backtest.metrics.out_of_sample_decay`
    # already applies: out-of-sample below half of in-sample.
    if is_sharpe > SHARPE_IS_FLOOR and retention < SHARPE_RETENTION:
        return Finding(
            "SHARPE_DROP", Severity.WARNING,
            f"Sharpe retains only {retention:.0%} out-of-sample ({is_sharpe:.2f} to "
            f"{oos_sharpe:.2f}) without meeting every collapse condition",
            detail,
        )
    return Finding("SHARPE_DROP", Severity.CLEAN, "no material out-of-sample decay", detail)


def check_drawdown_widening(returns: Sequence[float]) -> Finding:
    """Risk tuned to a window shows up as a deeper trough once the window ends."""
    in_sample, out_sample = _split(returns)
    if len(in_sample) < MIN_PERIODS_PER_SIDE or len(out_sample) < MIN_PERIODS_PER_SIDE:
        return Finding(
            "DRAWDOWN_WIDENING", Severity.INCONCLUSIVE,
            "too few periods either side of the split to compare drawdowns",
        )
    is_dd = max_drawdown(_equity(in_sample))
    oos_dd = max_drawdown(_equity(out_sample))
    detail = {"in_sample_drawdown": round(is_dd, 4), "out_of_sample_drawdown": round(oos_dd, 4)}

    if oos_dd > DRAWDOWN_FLOOR and (is_dd <= 0 or oos_dd > is_dd * DRAWDOWN_WIDENING):
        return Finding(
            "DRAWDOWN_WIDENING", Severity.CRITICAL,
            f"out-of-sample drawdown {oos_dd:.2%} against {is_dd:.2%} in-sample; risk behaved "
            f"differently once the window it was chosen on ended",
            detail,
        )
    return Finding("DRAWDOWN_WIDENING", Severity.CLEAN, "drawdown is stable across the split",
                   detail)


def check_calibration_divergence(predictions: Sequence[Prediction]) -> Finding:
    """Stated confidence that tracks outcomes in-sample and drifts out of it.

    Specific to a system that states its own confidence, which most backtests do not.
    """
    in_sample, out_sample = _split(list(predictions))
    if len(in_sample) < MIN_PERIODS_PER_SIDE or len(out_sample) < MIN_PERIODS_PER_SIDE:
        return Finding(
            "CALIBRATION_DIVERGENCE", Severity.INCONCLUSIVE,
            f"{len(in_sample)}/{len(out_sample)} graded predictions either side of the split; "
            f"{MIN_PERIODS_PER_SIDE} are needed on each",
        )
    is_ece = expected_calibration_error(in_sample)
    oos_ece = expected_calibration_error(out_sample)
    detail = {"in_sample_ece": round(is_ece, 4), "out_of_sample_ece": round(oos_ece, 4)}
    if oos_ece - is_ece > ECE_DIVERGENCE:
        return Finding(
            "CALIBRATION_DIVERGENCE", Severity.CRITICAL,
            f"calibration error rises from {is_ece:.3f} to {oos_ece:.3f}; the desk's confidence "
            f"stopped meaning what it meant in-sample",
            detail,
        )
    return Finding("CALIBRATION_DIVERGENCE", Severity.CLEAN, "calibration holds across the split",
                   detail)


def check_lookahead(entries: Sequence[Entry]) -> Finding:
    """A decision settled before it was taken, or two decisions sharing a sequence number.

    Not an overfitting signal — an impossibility. A backtest harness has no equivalent because its
    clock is synthetic; a live decision log can genuinely record the wrong order, and that row is
    void rather than doubtful.
    """
    offences: list[str] = []
    seen: set[int] = set()
    for entry in entries:
        if entry.seq in seen:
            offences.append(f"seq {entry.seq} appears more than once")
        seen.add(entry.seq)
        if entry.settled_at is None:
            continue
        decided = datetime.fromisoformat(entry.decided_at)
        settled = datetime.fromisoformat(entry.settled_at)
        if settled < decided:
            offences.append(f"seq {entry.seq} settled {decided - settled} before it was decided")

    if offences:
        return Finding(
            "LOOKAHEAD", Severity.CRITICAL,
            "the log records an impossible ordering; affected rows are void, not merely suspect",
            {"offences": offences[:10], "total": len(offences)},
        )
    return Finding("LOOKAHEAD", Severity.CLEAN, "every settlement follows its decision",
                   {"checked": len(entries)})


@dataclass(frozen=True)
class AuditReport:
    findings: tuple[Finding, ...]

    @property
    def critical(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.CRITICAL)

    @property
    def inconclusive(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.INCONCLUSIVE)

    @property
    def publishable(self) -> bool:
        """Whether these results can be shown without a caveat attached.

        Inconclusive counts against publishing. A result nobody could check is not a clean result,
        and saying so is the whole point of running this against ourselves.
        """
        return not self.critical and not self.inconclusive

    @property
    def verdict(self) -> str:
        if self.critical:
            codes = ", ".join(f.code for f in self.critical)
            return f"do not publish without the caveat: {codes}"
        if self.inconclusive:
            return "not enough data to audit these results; publish with the sample size stated"
        return "clean on every check run"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "publishable": self.publishable,
            "critical": len(self.critical),
            "inconclusive": len(self.inconclusive),
            "findings": [f.as_dict() for f in self.findings],
        }


def audit(ledger: PaperLedger, *, capital: Decimal = Decimal("10000")) -> AuditReport:
    """Run every self-check against our own log."""
    returns = [d.ret for d in daily_series(ledger, capital=capital)]
    predictions = [
        Prediction(confidence=e.stated_confidence, correct=bool(e.direction_correct))
        for e in ledger.entries
        if e.is_settled and not e.is_abstention and e.direction_correct is not None
    ]
    return AuditReport(findings=(
        check_lookahead(ledger.entries),
        check_sharpe_collapse(returns),
        check_drawdown_widening(returns),
        check_calibration_divergence(predictions),
    ))


__all__ = [
    "AuditReport",
    "Finding",
    "Severity",
    "audit",
    "check_calibration_divergence",
    "check_drawdown_widening",
    "check_lookahead",
    "check_sharpe_collapse",
]
