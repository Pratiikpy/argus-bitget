"""Closing the evaluation loop — the paper ledger feeds the Observatory.

The Observatory implements eight capabilities absent from every finance-agent harness we tore down.
Implementing them is half the job; until something feeds them they are arithmetic with no data, and
a benchmark nobody runs on real decisions is a spreadsheet.

This module is the join. It reads the hash-chained paper ledger, converts settled decisions into
:class:`~argus.eval.observatory.Prediction` records, and produces the scorecard — calibration,
abstention value, and the honest count of what is not yet gradeable.

**The rule that shapes it: only settled decisions score.** An open position has no outcome, and
counting it would let a system flatter itself by holding losers. Abstentions score separately,
through Abstention Value, because a NO_TRADE is a decision and standing aside is not automatically
right.

**Nothing here is graded by a model.** Every number is deterministic arithmetic over recorded
decisions — which is the methodological gap in Microsoft's FinanceBenchmark, where a fixed third
model judges every subject regardless of what is under test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval.episodes import summarise
from argus.eval.observatory import (
    AbstentionOutcome,
    ModelScorecard,
    Prediction,
    abstention_quality,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from argus.eval.performance import evaluate_ledger
from argus.paper.ledger import Entry, PaperLedger

ROUND_TRIP_BPS = Decimal("12")


@dataclass(frozen=True, slots=True)
class LedgerScore:
    """What the ledger says, once only the gradeable rows are counted."""

    entries: int
    settled: int
    abstentions: int
    ungradeable: int
    predictions: list[Prediction]
    abstention_outcomes: list[AbstentionOutcome]
    chain_intact: bool

    @property
    def has_enough_to_judge(self) -> bool:
        """Five graded outcomes is the floor.

        Below it, calibration is noise wearing a decimal point. The desk's own error profile uses
        the same threshold and refuses to draw a pattern under it.
        """
        return len(self.predictions) >= 5


def _prediction_from(entry: Entry) -> Prediction | None:
    """A settled, non-abstention decision becomes one prediction.

    ``direction_correct`` is written by the ledger at settlement from realised P&L, so the outcome
    is never inferred here — this module reads history, it does not decide it.
    """
    if not entry.is_settled or entry.is_abstention:
        return None
    if entry.direction_correct is None:
        return None
    return Prediction(confidence=entry.stated_confidence, correct=bool(entry.direction_correct))


def _abstention_from(
    entry: Entry, *, realised_move_bps: Decimal | None
) -> AbstentionOutcome | None:
    """An abstention scores only when we know what the move would have been.

    Without a counterfactual it is ungradeable, and counting it as correct is the exact flattery
    this metric exists to prevent.
    """
    if not entry.is_abstention:
        return None
    if realised_move_bps is None and entry.counterfactual_move_bps is not None:
        # Recorded by the runner at settlement. Preferred over anything reconstructed later from a
        # price series, which may since have been revised.
        realised_move_bps = Decimal(entry.counterfactual_move_bps)
    if realised_move_bps is None:
        return None
    return AbstentionOutcome(
        decision_id=str(entry.seq),
        counterfactual_move_bps=realised_move_bps,
        intended_side=entry.side,
        round_trip_bps=ROUND_TRIP_BPS,
    )


def score_ledger(
    ledger: PaperLedger, *, counterfactuals: dict[int, Decimal] | None = None
) -> LedgerScore:
    """Convert a ledger into gradeable records.

    ``counterfactuals`` maps an abstention's sequence number to the move that actually happened
    over its horizon, and **overrides** whatever the ledger recorded — for replaying a scorecard
    against a corrected price history. Omitted, each abstention uses the counterfactual the runner
    wrote at settlement. An abstention with neither is counted but ungraded, which remains the
    honest default.
    """
    cf = counterfactuals or {}
    predictions: list[Prediction] = []
    abstentions: list[AbstentionOutcome] = []
    ungradeable = 0

    for entry in ledger.entries:
        if entry.is_abstention:
            outcome = _abstention_from(entry, realised_move_bps=cf.get(entry.seq))
            if outcome is not None:
                abstentions.append(outcome)
            else:
                ungradeable += 1
            continue

        prediction = _prediction_from(entry)
        if prediction is not None:
            predictions.append(prediction)
        else:
            ungradeable += 1

    return LedgerScore(
        entries=len(ledger.entries),
        settled=sum(1 for e in ledger.entries if e.is_settled),
        abstentions=sum(1 for e in ledger.entries if e.is_abstention),
        ungradeable=ungradeable,
        predictions=predictions,
        abstention_outcomes=abstentions,
        chain_intact=bool(ledger.verify()["chain_intact"]),
    )


def scorecard(
    ledger: PaperLedger,
    *,
    model: str = "argus/qwen3.8-max",
    counterfactuals: dict[int, Decimal] | None = None,
    capital: Decimal = Decimal("10000"),
) -> dict[str, Any]:
    """The Observatory scorecard, computed from real recorded decisions."""
    score = score_ledger(ledger, counterfactuals=counterfactuals)

    base: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "ledger": {
            "entries": score.entries,
            "settled": score.settled,
            "abstentions": score.abstentions,
            "ungradeable": score.ungradeable,
            "chain_intact": score.chain_intact,
        },
        "graded_predictions": len(score.predictions),
    }

    if not score.chain_intact:
        # A broken chain invalidates everything downstream. Scoring a tampered ledger would
        # produce numbers that look exactly like honest ones.
        base["error"] = "hash chain is broken; refusing to score a ledger that may be edited"
        return base

    # The three numbers the handbook names for Track 2's quantitative half are reported whatever
    # the calibration floor says, because they answer a different question and have a different
    # minimum. Calibration needs a run of graded predictions; Sharpe needs a return series. Each
    # is withheld on its own terms, and `performance.undefined` says which and why.
    base["performance"] = evaluate_ledger(ledger, capital=capital).as_dict()

    # Both the per-decision and per-episode counts, side by side. A desk that restates one correct
    # refusal six times has made one good call, not six, and any rate computed per decision is
    # overstating its denominator by the ratio reported here.
    base["episodes"] = summarise(ledger.entries).as_dict()

    if not score.has_enough_to_judge:
        base["note"] = (
            f"{len(score.predictions)} graded outcomes — below the floor of 5. Calibration on "
            f"fewer is noise, so none is reported."
        )
        if score.abstention_outcomes:
            base["abstention"] = abstention_quality(score.abstention_outcomes)
        return base

    card = ModelScorecard(
        model=model,
        predictions=score.predictions,
        abstentions=score.abstention_outcomes,
        decisions=score.entries,
    )
    base.update(card.as_dict())
    base["brier"] = round(brier_score(score.predictions), 4)
    base["ece"] = round(expected_calibration_error(score.predictions), 4)
    base["reliability"] = reliability_curve(score.predictions)
    base["note"] = (
        "every figure is deterministic arithmetic over recorded decisions; no model grades "
        "another, and only settled non-abstention rows contribute to calibration"
    )
    return base


def main() -> int:
    from argus.paper.runner import LEDGER_PATH

    ledger = PaperLedger(path=LEDGER_PATH)
    result = scorecard(ledger)

    out = Path(__file__).resolve().parents[3] / "data" / "scorecard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    led = result["ledger"]
    print(f"entries {led['entries']} | settled {led['settled']} | "
          f"abstentions {led['abstentions']} · chain intact {led['chain_intact']}")
    perf = result.get("performance")
    if perf:
        def show(key: str, label: str, pct: bool = False) -> str:
            value = perf.get(key)
            if value is None:
                return f"{label} n/a"
            return f"{label} {value}{'%' if pct else ''}"
        print(f"{show('sharpe', 'Sharpe')} | {show('max_drawdown_pct', 'maxDD', pct=True)} | "
              f"{show('win_rate_pct', 'win rate', pct=True)} | "
              f"over {perf['trades']} trade(s) in a {perf['window_days']}-day window")
        for stat, why in (perf.get("undefined") or {}).items():
            print(f"  {stat}: {why}")
    eps = result.get("episodes")
    if eps and eps["episodes"]:
        print(f"episodes {eps['episodes']} from {eps['decisions']} decision(s) "
              f"({eps['decisions_per_episode']} per episode, "
              f"{eps['restatements']} restatement(s))")

    if "ece" in result:
        print(f"ECE {result['ece']} | Brier {result['brier']} | "
              f"accuracy {result.get('accuracy_pct')}%")
    else:
        print(result.get("note") or result.get("error"))
    print(f"\nfull report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LedgerScore", "score_ledger", "scorecard"]
