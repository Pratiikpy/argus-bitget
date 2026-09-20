"""A scored forecast record for the policy layer — one forecast per symbol per hour, graded.

**Why this exists.** Track 2 is scored half on quantitative evidence, and the hash-chained ledger
holds 126 model decisions of which two have settled. A competing entry found in the GitHub sweep
reports 2,304 scored forecasts, and 2,304 is exactly twelve symbols times a hundred and ninety-two
hours. Whatever they did, the unit they are counting is the **symbol-hour**, not the deliberated
decision — and the honest response is neither to dismiss that nor to inflate our own count by
renaming things, but to produce the same kind of record and label it for exactly what it is.

**What is being forecast, stated precisely.** Not the model's judgement. This is the *deterministic
policy layer* — the session clock, the cost hurdle, and the volatility regime — asked the one
question it can answer without a model at every bar:

> *Will the next hour move far enough, in absolute terms, to pay a round trip?*

That is a real, falsifiable, probabilistic forecast. It is the question the desk's abstentions turn
on, it is answerable from the tape alone, and it is the reason the desk stands aside so often. A
policy that abstains 126 times in a row is making a claim about the world, and this measures
whether the claim is true.

**Three things keep it honest, and each is the difference between a record and a number.**

1. **It is labelled as policy, never as the desk's judgement.** The ledger holds what the model
   decided; this holds what the deterministic layer predicted. Reporting them as one number would
   be the inflation this file exists not to commit, and
   :attr:`Scorecard.what_this_is_not` says so in the artefact itself.
2. **Point-in-time by construction.** The forecast at bar *i* is computed from bars up to *i* and
   graded against the move from *i* to *i+1*. The loop stops one bar short of the data, so there
   is no index at which a forecast can see its own outcome.
3. **The confidence is calibrated, not asserted.** :func:`probability_of_clearing` estimates the
   chance from the *realised* distribution of recent absolute moves, so it is an empirical
   frequency rather than a number chosen to look good — and it is then scored with the Brier score
   and expected calibration error the observatory already computes, which punishes exactly that.

The value to the reader is not the hit rate. It is the calibration: a desk that says 30% and is
right 30% of the time knows what it does not know, which on a venue where the fee exceeds most
effects is worth more than a lucky quarter.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.observatory import (
    Prediction,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)

OUT_PATH = Path(__file__).resolve().parents[3] / "data" / "policy_forecasts.json"

WINDOW = 168
"""Bars of realised history used to estimate the next bar's move distribution. One week of hours.

Long enough that the frequency is not noise, short enough that a regime change reaches it inside a
week. It is also the reason the first 168 bars of every symbol produce no forecast: an estimate
with nothing behind it is a guess, and a guess scored as a forecast is how a calibration curve gets
manufactured.
"""

MIN_OBSERVATIONS = 60
"""Fewest historical moves before a probability is stated at all."""


def probability_of_clearing(moves_bps: Sequence[float], hurdle_bps: float) -> float:
    """The empirical frequency with which a move of this size has been exceeded.

    Deliberately not a fitted distribution. A normal fitted to hourly equity-token returns
    understates the tail, which is the part that decides whether a hurdle is cleared, so the
    estimate is the raw share of the last :data:`WINDOW` bars that exceeded it. The estimator is
    then exactly as wrong as the sample, which is a property a reader can reason about.
    """
    if len(moves_bps) < MIN_OBSERVATIONS:
        return -1.0
    cleared = sum(1 for m in moves_bps if abs(m) > hurdle_bps)
    return cleared / len(moves_bps)


def intraclass_correlation(groups: Sequence[Sequence[float]]) -> float:
    """One-way ICC over grouped observations. Zero when groups carry no shared component.

    The measure that decides whether a large record is large. Observations that share a timestamp
    share a market, so twelve symbols scored on the same hour are not twelve independent facts
    about the world — they are closer to one fact observed twelve times. ICC quantifies how much.

    Computed by the standard one-way random-effects decomposition: between-group variance over
    total variance. Negative values, which arise when within-group spread exceeds between, are
    clamped to zero — a negative correlation between siblings is not meaningful here and would
    make the design effect below one, implying a clustered sample is *more* informative than an
    independent one.
    """
    sizes = [len(g) for g in groups if g]
    if len(sizes) < 2 or sum(sizes) < 3:
        return 0.0
    flat = [v for g in groups for v in g]
    grand = sum(flat) / len(flat)
    between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups if g)
    within = sum((v - sum(g) / len(g)) ** 2 for g in groups if g for v in g)
    k = sum(sizes) / len(sizes)
    df_between = len(sizes) - 1
    df_within = sum(sizes) - len(sizes)
    if df_between <= 0 or df_within <= 0 or k <= 1:
        return 0.0
    ms_between = between / df_between
    ms_within = within / df_within
    denominator = ms_between + (k - 1) * ms_within
    if denominator <= 0:
        return 0.0
    return max(0.0, (ms_between - ms_within) / denominator)


def design_effect(icc: float, average_cluster_size: float) -> float:
    """Kish's design effect: how many raw observations one independent one is worth.

    ``1 + (m - 1) * ICC``. At ICC zero it is one and the raw count stands; at ICC 0.14 with twelve
    symbols per hour it is about 2.5, meaning a record of twelve thousand rows carries the
    information of roughly five thousand.
    """
    return 1.0 + max(0.0, average_cluster_size - 1.0) * max(0.0, icc)


@dataclass(frozen=True, slots=True)
class Forecast:
    """One symbol-hour: what the policy said, and what happened next."""

    symbol: str
    at: datetime
    hurdle_bps: float
    probability: float
    """Stated chance the next bar clears the hurdle in absolute terms."""

    realised_bps: float
    """The move from this bar to the next. Never visible to the forecast."""

    @property
    def cleared(self) -> bool:
        return abs(self.realised_bps) > self.hurdle_bps

    @property
    def predicted_clear(self) -> bool:
        """The policy's binary call, at the natural threshold."""
        return self.probability > 0.5

    @property
    def correct(self) -> bool:
        return self.predicted_clear == self.cleared

    @property
    def as_prediction(self) -> Prediction:
        """The probability assigned to clearing, and whether clearing happened.

        **A bug lived here and produced a meaningless number.** The first version passed
        ``correct=True`` on every row — reasoning that a forecast of 30% which did not clear was
        "70% confident in what occurred" — which makes every prediction correct by construction.
        Brier then measured ``(confidence - 1)^2`` and expected calibration error came out at
        0.461, a figure that looked computed and measured nothing but the mean confidence.

        The right shape is the plain one that `argus.eval.observatory` documents: the stated
        probability of an event, paired with whether that event happened. Both statistics are
        defined against exactly that, and neither needs the forecast re-expressed in terms of the
        outcome.
        """
        return Prediction(confidence=self.probability, correct=self.cleared)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "at": self.at.isoformat(),
            "hurdle_bps": round(self.hurdle_bps, 3),
            "probability": round(self.probability, 4),
            "realised_bps": round(self.realised_bps, 3),
            "cleared": self.cleared, "correct": self.correct,
        }


def forecast_series(
    symbol: str,
    bars: Sequence[Any],
    *,
    hurdle_bps: float,
    window: int = WINDOW,
) -> list[Forecast]:
    """One forecast per bar, from bar ``window`` to the second-to-last.

    The loop bound is the whole point: ``range(window, len(bars) - 1)`` stops one short, so every
    forecast is graded against a bar the forecast could not have been built from. There is no index
    at which this function can see its own outcome.
    """
    out: list[Forecast] = []
    moves: list[float] = []
    for index in range(1, len(bars)):
        before, after = float(bars[index - 1].close), float(bars[index].close)
        moves.append((after - before) / before * 10_000.0 if before > 0 else 0.0)

    for index in range(window, len(bars) - 1):
        history = moves[max(0, index - window):index]
        probability = probability_of_clearing(history, hurdle_bps)
        if probability < 0:
            continue
        before, after = float(bars[index].close), float(bars[index + 1].close)
        if before <= 0:
            continue
        out.append(Forecast(
            symbol=symbol,
            at=bars[index].ts,
            hurdle_bps=hurdle_bps,
            probability=probability,
            realised_bps=(after - before) / before * 10_000.0,
        ))
    return out


@dataclass(frozen=True)
class Scorecard:
    """What the forecast record establishes, and — as loudly — what it does not."""

    forecasts: tuple[Forecast, ...]
    hurdle_bps: float

    what_this_is_not = (
        "These are forecasts by the DETERMINISTIC POLICY LAYER — the session clock, the cost "
        "hurdle and the realised move distribution — not decisions by the analyst panel. The "
        "panel's decisions live in data/paper_ledger.jsonl and are counted separately. Adding the "
        "two together would be inflating a judgement record with an arithmetic one, and the two "
        "measure different things: this measures whether the tape pays a round trip, the ledger "
        "measures whether the desk can call the direction."
    )

    @property
    def count(self) -> int:
        return len(self.forecasts)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted({f.symbol for f in self.forecasts}))

    @property
    def hit_rate(self) -> float | None:
        if not self.forecasts:
            return None
        return sum(1 for f in self.forecasts if f.correct) / len(self.forecasts)

    @property
    def base_rate(self) -> float | None:
        """How often the hurdle was actually cleared. The number a forecast must beat."""
        if not self.forecasts:
            return None
        return sum(1 for f in self.forecasts if f.cleared) / len(self.forecasts)

    @property
    def clusters(self) -> list[list[float]]:
        """Squared errors grouped by timestamp. Symbols scored on the same hour share a market."""
        by_time: dict[datetime, list[float]] = {}
        for f in self.forecasts:
            prediction = f.as_prediction
            realised = 1.0 if prediction.correct else 0.0
            by_time.setdefault(f.at, []).append((prediction.confidence - realised) ** 2)
        return list(by_time.values())

    @property
    def distinct_instants(self) -> int:
        """How many genuinely different moments this record covers.

        The number a headline count hides. A teardown of a competing entry found 2,304 reported
        forecasts resting on 119 distinct timestamps — roughly twenty copies of each night — with
        confidence intervals computed as though all 2,304 were independent. Publishing this number
        beside the raw count is the cheapest possible defence against making the same claim."""
        return len({f.at for f in self.forecasts})

    @property
    def icc(self) -> float:
        return intraclass_correlation(self.clusters)

    @property
    def design_effect(self) -> float:
        groups = [g for g in self.clusters if g]
        if not groups:
            return 1.0
        return design_effect(self.icc, sum(len(g) for g in groups) / len(groups))

    @property
    def effective_count(self) -> int:
        """The raw count divided by the design effect. The honest sample size.

        Every interval and every significance claim over this record should use this number, not
        :attr:`count`. Reporting the raw count as the sample size is how a clustered record starts
        producing intervals that are too narrow by the square root of the design effect."""
        return int(self.count / self.design_effect) if self.count else 0

    @property
    def brier(self) -> float | None:
        preds = [f.as_prediction for f in self.forecasts]
        return brier_score(preds) if preds else None

    @property
    def ece(self) -> float | None:
        preds = [f.as_prediction for f in self.forecasts]
        return expected_calibration_error(preds) if preds else None

    def split(self) -> tuple[Scorecard, Scorecard]:
        """Chronological halves. In-sample first, out-of-sample second, never shuffled.

        A random split on a time series leaks: a shuffled test bar sits between two training bars
        that bracket it. The handbook asks for out-of-sample validation and this is the only split
        that provides it.
        """
        ordered = sorted(self.forecasts, key=lambda f: f.at)
        half = len(ordered) // 2
        return (
            Scorecard(forecasts=tuple(ordered[:half]), hurdle_bps=self.hurdle_bps),
            Scorecard(forecasts=tuple(ordered[half:]), hurdle_bps=self.hurdle_bps),
        )

    def render(self) -> str:
        if not self.forecasts:
            return "POLICY FORECASTS — none produced."
        in_sample, out_sample = self.split()
        lines = [
            f"POLICY FORECASTS — {self.count:,} scored forecast(s) across "
            f"{len(self.symbols)} symbol(s), one per symbol-hour",
            f"  hurdle {self.hurdle_bps:.1f}bps; the tape actually cleared it "
            f"{self.base_rate:.1%} of the time",
            f"  binary hit rate {self.hit_rate:.1%} against a "
            f"{max(self.base_rate or 0.0, 1.0 - (self.base_rate or 0.0)):.1%} "
            f"always-say-the-majority baseline",
            f"  Brier {self.brier:.4f}, expected calibration error {self.ece:.4f} "
            f"(0.25 Brier is a coin)",
            f"  {self.distinct_instants:,} distinct instant(s); intra-cluster correlation "
            f"{self.icc:.3f}, design effect {self.design_effect:.2f}, so the EFFECTIVE sample is "
            f"{self.effective_count:,} not {self.count:,}. Every interval over this record uses "
            f"the effective number — a clustered record reported at its raw count produces "
            f"intervals too narrow by the root of the design effect.",
        ]
        if in_sample.ece is not None and out_sample.ece is not None:
            lines.append(
                f"  calibration in-sample {in_sample.ece:.4f} -> out-of-sample "
                f"{out_sample.ece:.4f} on a chronological split"
            )
        lines.append("  " + self.what_this_is_not)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        in_sample, out_sample = self.split()
        return {
            "forecasts": self.count,
            "symbols": list(self.symbols),
            "hurdle_bps": round(self.hurdle_bps, 3),
            "base_rate": None if self.base_rate is None else round(self.base_rate, 4),
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 4),
            "distinct_instants": self.distinct_instants,
            "icc": round(self.icc, 5),
            "design_effect": round(self.design_effect, 4),
            "effective_count": self.effective_count,
            "brier": None if self.brier is None else round(self.brier, 5),
            "ece": None if self.ece is None else round(self.ece, 5),
            "in_sample_ece": None if in_sample.ece is None else round(in_sample.ece, 5),
            "out_of_sample_ece": None if out_sample.ece is None else round(out_sample.ece, 5),
            "reliability": reliability_curve([f.as_prediction for f in self.forecasts]),
            "what_this_is_not": self.what_this_is_not,
        }


def run(
    *,
    days: int = 180,
    hurdle_bps: float | None = None,
    out: Path = OUT_PATH,
) -> dict[str, Any]:
    """Build the record across the whole universe and write it."""
    from argus.backtest.engine import Bar
    from argus.cost.model import CostModel
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range

    hurdle = hurdle_bps if hurdle_bps is not None else float(
        CostModel.bitget_perp().round_trip_bps()
    )
    everything: list[Forecast] = []
    per_symbol: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for symbol in RTOKEN_SYMBOLS:
        try:
            candles = fetch_range(
                symbol, days=days, interval="1H", candle_type=CandleType.MARKET
            )
        except Exception as exc:
            skipped[symbol] = str(exc)[:80]
            continue
        bars = [Bar(ts=c.ts, close=c.close, extra={}) for c in candles]
        series = forecast_series(symbol, bars, hurdle_bps=hurdle)
        everything.extend(series)
        per_symbol[symbol] = len(series)

    return card_and_payload(everything, hurdle, per_symbol, skipped, days, out)


def build_scorecard(*, days: int = 180, hurdle_bps: float | None = None) -> Scorecard:
    """The record itself, without writing anything.

    Split out of :func:`run` because `eval.forecastbench` needs the scorecard and `run` returns the
    serialised payload. Re-deriving the record there would mean two fetch loops that could drift.
    """
    from argus.backtest.engine import Bar
    from argus.cost.model import CostModel
    from argus.market.bitget import RTOKEN_SYMBOLS
    from argus.market.history import CandleType, fetch_range

    hurdle = hurdle_bps if hurdle_bps is not None else float(
        CostModel.bitget_perp().round_trip_bps()
    )
    everything: list[Forecast] = []
    for symbol in RTOKEN_SYMBOLS:
        try:
            candles = fetch_range(
                symbol, days=days, interval="1H", candle_type=CandleType.MARKET
            )
        except Exception:
            continue
        bars = [Bar(ts=c.ts, close=c.close, extra={}) for c in candles]
        everything.extend(forecast_series(symbol, bars, hurdle_bps=hurdle))
    return Scorecard(forecasts=tuple(everything), hurdle_bps=hurdle)


def card_and_payload(
    everything: list[Forecast],
    hurdle: float,
    per_symbol: dict[str, int],
    skipped: dict[str, str],
    days: int,
    out: Path,
) -> dict[str, Any]:
    card = Scorecard(forecasts=tuple(everything), hurdle_bps=hurdle)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": days,
        "history_window_bars": WINDOW,
        "per_symbol": per_symbol,
        "skipped": skipped,
        **card.as_dict(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:  # pragma: no cover - CLI
    import contextlib
    import sys

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    payload = run()
    print(Scorecard(
        forecasts=(), hurdle_bps=payload["hurdle_bps"]
    ).what_this_is_not)
    print()
    print(json.dumps({k: v for k, v in payload.items() if k != "reliability"}, indent=2)[:1800])
    return 0


__all__ = [
    "MIN_OBSERVATIONS",
    "OUT_PATH",
    "WINDOW",
    "Forecast",
    "Scorecard",
    "forecast_series",
    "probability_of_clearing",
    "run",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
