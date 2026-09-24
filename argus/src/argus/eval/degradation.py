"""Does it still work later? The three signals that catch a result fitted to its own sample.

`desk/stress.py` runs a scenario library against a held position, and ARGUS-BENCH grades it while
saying in its own output that it *does not test the desk deciding under stress*. This is the other
half: rather than constructing a shock, it splits the record chronologically and asks whether
performance, risk and calibration all held up in the later half.

The design is taken from `agent-backtest-lab/abl/leakage/reward_hacking.py:57-172`, the only
rigorous treatment of this in the local corpus, read at source before being written here. Three
signals, each catching something the other two miss:

* **Sharpe collapse.** The obvious one. A strategy that scored well in-sample and poorly out of it
  was fitted, and the size of the drop is more informative than either level alone.
* **Drawdown widening.** A strategy can keep a respectable Sharpe while its losses get much deeper,
  because Sharpe divides by total volatility and does not care which side it came from. This is
  the signal that catches a result whose mean survived and whose tail did not.
* **Calibration drift.** The one this project had the ingredients for and was not reading as a
  degradation test. `eval/forecasts.py` already computes expected calibration error on both halves
  of the record; comparing them asks whether the system still knows how much it knows. A forecaster
  whose accuracy holds while its *confidence* drifts is on its way to a loss it will not see coming.

**The thresholds are theirs and are quoted rather than invented**, because a threshold chosen after
seeing our own numbers is not a threshold. Two deliberate departures are recorded where they occur:
the in-sample fraction is a parameter here rather than a fixed 0.7, since their default is
aggressive for a long backtest and can leave the out-of-sample window too short to detect the
overfitting it is looking for; and every flag carries the numbers that produced it, so a reader can
disagree with the threshold without having to rerun anything.

**A clean result here is weak evidence, and the verdict says so.** Not degrading is what a strategy
with no edge at all also does. These signals can refute; they cannot confirm.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import sqrt
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "degradation.json"

DEFAULT_IN_SAMPLE_FRACTION = 0.7
"""Their default, kept as the default and exposed as a parameter.

`reward_hacking.py:81` fixes this at 0.7. On a long record that leaves an out-of-sample window too
short to detect what the test is looking for, so it is a parameter here and the value used is
reported with every result.
"""

MIN_OBSERVATIONS = 40
"""Fewest observations before a split is worth making.

Twenty a side. Below that the two Sharpe ratios are both noise and their difference is noise about
noise, which would produce a degradation flag from nothing at all.
"""

# Thresholds transcribed from `reward_hacking.py:87-169`. Not tuned, not chosen after seeing our
# own numbers, and quoted in the flag text so a reader can disagree with them explicitly.
SHARPE_CRITICAL_IS = 1.0
SHARPE_CRITICAL_DROP = 1.5
SHARPE_CRITICAL_OOS = 0.5
SHARPE_WARN_IS = 0.5
SHARPE_WARN_DROP = 1.0
DRAWDOWN_WARN_FLOOR = -0.10
DRAWDOWN_WIDENING_FACTOR = 1.5
DRAWDOWN_WIDENING_OFFSET = 0.05
ECE_WARN_LEVEL = 0.15
ECE_WARN_RISE = 0.10


class Severity(StrEnum):
    CLEAN = "clean"
    WARN = "warn"
    CRITICAL = "critical"


class DegradationError(ValueError):
    """Raised rather than reporting a degradation computed from too little."""


@dataclass(frozen=True, slots=True)
class Signal:
    """One of the three, with the numbers that produced it."""

    name: str
    severity: Severity
    in_sample: float | None
    out_of_sample: float | None
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "in_sample": None if self.in_sample is None else round(self.in_sample, 5),
            "out_of_sample": None if self.out_of_sample is None else round(self.out_of_sample, 5),
            "message": self.message,
        }


def _sharpe(returns: Sequence[float], periods_per_year: int) -> float | None:
    """Annualised, and the frequency is required.

    **The thresholds below are annualised numbers and were nearly applied to per-observation
    ones.** The reference computes `_sharpe_ann(is_r, 252)` before comparing against 1.0, 1.5 and
    0.5; transcribing the thresholds without the annualisation makes the Sharpe signal almost never
    fire, because a per-observation Sharpe of 0.37 is an annualised 5.9 on daily data. Caught by
    running the module against a deliberately fitted series and noticing the signal stayed clean
    while the other two fired.
    """
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if variance <= 0:
        return None
    return mean / sqrt(variance) * sqrt(periods_per_year)


def max_drawdown(returns: Sequence[float]) -> float:
    """Peak-to-trough of the compounded curve, as a negative fraction. Zero for a curve that only
    rises, which is a real answer and not a missing one."""
    peak = 1.0
    equity = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, equity / peak - 1.0)
    return worst


def sharpe_signal(
    in_sample: Sequence[float], out_of_sample: Sequence[float], *, periods_per_year: int
) -> Signal:
    """Sharpe collapse, at the thresholds the reference publishes — which are annualised."""
    sr_is = _sharpe(in_sample, periods_per_year)
    sr_oos = _sharpe(out_of_sample, periods_per_year)
    if sr_is is None or sr_oos is None:
        return Signal(
            "sharpe_collapse", Severity.CLEAN, sr_is, sr_oos,
            "one half of the split has no computable Sharpe, so no comparison was made",
        )
    drop = sr_is - sr_oos
    if sr_is > SHARPE_CRITICAL_IS and drop > SHARPE_CRITICAL_DROP and sr_oos < SHARPE_CRITICAL_OOS:
        return Signal(
            "sharpe_collapse", Severity.CRITICAL, sr_is, sr_oos,
            f"Sharpe collapses from {sr_is:.2f} in sample to {sr_oos:.2f} out of it, a drop of "
            f"{drop:.2f}. That is the shape of a fitted result",
        )
    if sr_is > SHARPE_WARN_IS and drop > SHARPE_WARN_DROP:
        return Signal(
            "sharpe_collapse", Severity.WARN, sr_is, sr_oos,
            f"Sharpe drops from {sr_is:.2f} to {sr_oos:.2f}, a fall of {drop:.2f} against a "
            f"warning threshold of {SHARPE_WARN_DROP}",
        )
    return Signal(
        "sharpe_collapse", Severity.CLEAN, sr_is, sr_oos,
        f"Sharpe went from {sr_is:.2f} to {sr_oos:.2f}; the drop of {drop:.2f} does not reach the "
        f"warning threshold of {SHARPE_WARN_DROP}",
    )


def drawdown_signal(in_sample: Sequence[float], out_of_sample: Sequence[float]) -> Signal:
    """Drawdown widening — the signal that catches a surviving mean and a broken tail."""
    dd_is = max_drawdown(in_sample)
    dd_oos = max_drawdown(out_of_sample)
    widened = dd_oos < DRAWDOWN_WIDENING_FACTOR * dd_is - DRAWDOWN_WIDENING_OFFSET
    if dd_oos < DRAWDOWN_WARN_FLOOR and widened:
        return Signal(
            "drawdown_widening", Severity.WARN, dd_is, dd_oos,
            f"maximum drawdown widens from {dd_is:+.2%} to {dd_oos:+.2%}. Sharpe divides by total "
            f"volatility and does not care which side it came from, so this can fire while the "
            f"Sharpe signal does not",
        )
    return Signal(
        "drawdown_widening", Severity.CLEAN, dd_is, dd_oos,
        f"maximum drawdown went from {dd_is:+.2%} to {dd_oos:+.2%}, which does not clear the "
        f"widening threshold",
    )


def calibration_signal(ece_in: float | None, ece_out: float | None) -> Signal:
    """Calibration drift — does the system still know how much it knows?

    The signal this project had the ingredients for and was not reading. Accuracy holding while
    confidence drifts is the quiet failure: nothing looks wrong until a position is sized on a
    number that no longer means what it used to.
    """
    if ece_in is None or ece_out is None:
        return Signal(
            "calibration_drift", Severity.CLEAN, ece_in, ece_out,
            "one half of the split has no calibration error, so no comparison was made",
        )
    if ece_out > ECE_WARN_LEVEL and ece_out > ece_in + ECE_WARN_RISE:
        return Signal(
            "calibration_drift", Severity.WARN, ece_in, ece_out,
            f"expected calibration error rises from {ece_in:.4f} to {ece_out:.4f}, past both the "
            f"level threshold of {ECE_WARN_LEVEL} and a rise of {ECE_WARN_RISE}",
        )
    return Signal(
        "calibration_drift", Severity.CLEAN, ece_in, ece_out,
        f"expected calibration error went from {ece_in:.4f} to {ece_out:.4f}, which stays inside "
        f"both thresholds",
    )


@dataclass(frozen=True, slots=True)
class Degradation:
    """All three signals, and an honest statement of what a clean result is worth."""

    label: str
    observations: int
    in_sample_fraction: float
    signals: tuple[Signal, ...]

    @property
    def severity(self) -> Severity:
        if any(s.severity is Severity.CRITICAL for s in self.signals):
            return Severity.CRITICAL
        if any(s.severity is Severity.WARN for s in self.signals):
            return Severity.WARN
        return Severity.CLEAN

    @property
    def verdict(self) -> str:
        flagged = [s.name for s in self.signals if s.severity is not Severity.CLEAN]
        if flagged:
            return (
                f"DEGRADATION DETECTED in {', '.join(flagged)} over {self.observations} "
                f"observation(s) split {self.in_sample_fraction:.0%}/"
                f"{1 - self.in_sample_fraction:.0%}. These signals refute; they do not diagnose"
            )
        return (
            f"NO DEGRADATION DETECTED over {self.observations} observation(s). This is weak "
            f"evidence and is reported as such: a strategy with no edge at all also fails to "
            f"degrade, so a clean result here rules something out rather than establishing "
            f"anything"
        )

    def render(self) -> str:
        lines = [
            f"DEGRADATION — {self.label}: {self.observations} observation(s), "
            f"{self.in_sample_fraction:.0%} in sample",
            "",
        ]
        for signal in self.signals:
            lines.append(f"  [{signal.severity.value.upper():>8}] {signal.name}: {signal.message}")
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "observations": self.observations,
            "in_sample_fraction": self.in_sample_fraction,
            "severity": self.severity.value,
            "signals": [s.as_dict() for s in self.signals],
            "verdict": self.verdict,
        }


def assess(
    label: str,
    returns: Sequence[float],
    *,
    periods_per_year: int,
    in_sample_fraction: float = DEFAULT_IN_SAMPLE_FRACTION,
    ece_in: float | None = None,
    ece_out: float | None = None,
) -> Degradation:
    """Split chronologically and run all three signals. Never shuffles.

    A random split leaks: a shuffled test observation sits between two training observations that
    bracket it. The split here is by position in the series, which for an ordered return series is
    chronological.
    """
    n = len(returns)
    if n < MIN_OBSERVATIONS:
        raise DegradationError(
            f"{n} observation(s) is below the {MIN_OBSERVATIONS} a split needs; the difference "
            f"between two Sharpe ratios computed from ten points each is noise about noise"
        )
    if not 0.1 <= in_sample_fraction <= 0.9:
        raise DegradationError(
            f"an in-sample fraction of {in_sample_fraction} leaves one side of the split too "
            f"small to measure"
        )
    cut = int(n * in_sample_fraction)
    return Degradation(
        label=label,
        observations=n,
        in_sample_fraction=in_sample_fraction,
        signals=(
            sharpe_signal(returns[:cut], returns[cut:], periods_per_year=periods_per_year),
            drawdown_signal(returns[:cut], returns[cut:]),
            calibration_signal(ece_in, ece_out),
        ),
    )


def _track1_winner_returns() -> tuple[str, list[float]] | None:  # pragma: no cover - network
    """The best Track 1 variant's net returns, recomputed from cached history.

    Recomputed rather than stored because `track1_study.json` keeps the metrics and not the series,
    and a degradation test needs the path.
    """
    from argus.backtest.engine import Bar, run
    from argus.backtest.metrics import HOURLY_PER_YEAR, MetricError
    from argus.cost.model import CostModel
    from argus.market.history import CandleType, fetch_range
    from argus.research.track1_study import ALL_VARIANTS

    study = DATA / "track1_study.json"
    if not study.exists():
        return None
    payload = json.loads(study.read_text(encoding="utf-8"))
    rows = payload.get("per_symbol") or []
    if not rows:
        return None
    best = max(rows, key=lambda r: r.get("best_net_sharpe", float("-inf")))
    symbol, variant = best["symbol"], best["best_variant"]
    signal = ALL_VARIANTS.get(variant)
    if signal is None:
        return None
    try:
        candles = fetch_range(symbol, days=180, interval="1H", candle_type=CandleType.MARKET)
    except Exception:
        return None
    bars = [
        Bar(ts=c.ts, close=c.close,
            extra={"volume": float(c.volume), "high": float(c.high), "low": float(c.low)})
        for c in candles
    ]
    try:
        result = run(
            f"degradation:{variant}", symbol, bars, signal,
            cost=CostModel.bitget_perp(), periods_per_year=HOURLY_PER_YEAR,
        )
    except MetricError:
        return None
    return f"{symbol} {variant}", list(result.net_returns)


def main(*, rebuild: bool = False) -> int:  # pragma: no cover - CLI
    """Degradation over the Track 1 winner's own returns, with calibration drift beside it.

    **The two market signals need a strategy return series, not a market one.** An earlier version
    ran them on the realised move of every forecast instant, which is the underlying's path: a
    Sharpe collapse there is a statement about the tape, not about us, and reporting it under our
    own name would have been the flattering kind of confusion. They now run on the best Track 1
    variant's net returns, which is a P&L series the project actually produced.

    Calibration drift is read from the saved forecast artefact rather than recomputed, because
    rebuilding 49,142 forecasts to re-read two numbers that are already on disk costs a quarter of
    an hour of venue fetches and cannot change them. Pass ``rebuild=True`` to recompute anyway.
    """
    from argus.backtest.metrics import HOURLY_PER_YEAR

    # The artefact publishes both halves under these exact names. Guessing the key is how the
    # out-of-sample value silently fell back to the in-sample one on the first run, which made the
    # calibration signal compare a number against itself and report CLEAN by construction.
    artefact = json.loads((DATA / "policy_forecasts.json").read_text(encoding="utf-8"))
    ece_in = artefact.get("in_sample_ece")
    ece_out = artefact.get("out_of_sample_ece")
    if ece_in is None or ece_out is None:
        raise DegradationError(
            "the forecast artefact carries no in-sample/out-of-sample calibration split; "
            "rerun `python -m argus.eval.forecasts` before asking for a drift reading"
        )
    if rebuild:
        from argus.eval.forecasts import build_scorecard

        card = build_scorecard()
        in_half, out_half = card.split()
        ece_in, ece_out = in_half.ece, out_half.ece

    returns = _track1_winner_returns()
    if returns is None:
        print(
            "no Track 1 return series available; run `python -m argus.research.track1_study` "
            "first. Calibration drift alone is not a degradation report"
        )
        return 1
    label, series = returns
    result = assess(
        f"track 1 winner: {label}",
        series,
        periods_per_year=HOURLY_PER_YEAR,
        ece_in=ece_in,
        ece_out=ece_out,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(result.render())
    print(f"\ngenerated {datetime.now(UTC).isoformat()} -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DEFAULT_IN_SAMPLE_FRACTION",
    "MIN_OBSERVATIONS",
    "Degradation",
    "DegradationError",
    "Severity",
    "Signal",
    "assess",
    "calibration_signal",
    "drawdown_signal",
    "max_drawdown",
    "sharpe_signal",
]
