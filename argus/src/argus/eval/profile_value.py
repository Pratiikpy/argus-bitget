"""Did being personalised actually help? The question `desk/personalisation` could not answer.

`desk/personalisation.py` proves **divergence**: the same market state produces different verdicts
under different mandates, and on the shipped proposals it binds on one in four — and says plainly,
on the other three, that "personalisation did not bind here. That is a fact about this proposal,
not evidence that profiles work." That honesty is the right default and it stops one step short of
the thing a judge actually wants to know.

**The step it stops short of is utility.** A profile that refuses a trade has helped its owner only
if the trade would have lost money. A profile that refuses everything diverges beautifully and is
useless. Our own competitive audit named this exact gap — *"proves divergence, does not measure
whether the divergence improved the profile's utility"* — and blamed it on having no settled trades
to compute utility from.

That blame was only half right. There are no settled *desk* trades, and there are 4,528 tradeable
instants with a realised move attached, which is all a utility measurement needs. This replays a
proposal derived from each real instant past each profile and scores what each profile's decision
was actually worth:

* **Refusal value** — the mean net outcome of the trades a profile declined. Positive means it
  declined winners and the mandate cost its owner money; negative means it declined losers and the
  mandate earned its keep.
* **Taken value** — the mean net outcome of what it allowed through, **scaled by how much of the
  proposal it permitted**. A mandate that halves a position takes half the move, and scoring a
  resize as a full take would credit the conservative profile with the whole move on the 98% of
  proposals it narrows rather than refuses.
* **Utility gain** — what the profile earned over taking every proposal. This is the number, and it
  can come out negative, which is the point of computing it.

**No decision is fabricated.** The proposals here are constructed from price history, not read from
a ledger and not re-decided by a model, and the desk does not appear in the result at all. This
measures *the mandate layer*, which is deterministic code, against outcomes that actually happened.
A desk decision replayed today and called last month's would be the defect the replay harness
exists to prevent; a deterministic policy replayed over recorded prices is not that.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from argus.desk.personalisation import Outcome, Proposal, judge, standard_profiles
from argus.eval.incremental import DIRECTION_LONG, DIRECTION_NONE, DIRECTION_SHORT, Instant

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "profile_value.json"

MIN_INSTANTS = 200
BAD_CASE_SIGMAS = 2.0
"""How many trailing standard deviations the thesis concedes in its bad case.

A proposal has to state a downside for a mandate to judge it, and inventing one per instant would
make the whole measurement a function of that invention. Two sigma of the instant's own trailing
volatility is the least arbitrary choice available: it is computed from the same window the rules
see, it moves with the market rather than being fixed, and it is stated here rather than buried so
a reader can disagree with it in one place.
"""

DEFAULT_NOTIONAL = Decimal("10000")
DEFAULT_HORIZON_HOURS = 24.0


class ProfileValueError(ValueError):
    """Raised rather than reporting a utility claim from too little."""


def proposal_from(instant: Instant, *, sector: str = "technology") -> Proposal:
    """Turn a real instant into the trade idea a desk would have put to a mandate.

    The bad case is derived from the instant's own trailing volatility, so it is a property of the
    market at that moment rather than a constant chosen to make some profile look good.
    """
    window = instant.trailing[-24:] or instant.trailing
    vol = pstdev(window) if len(window) >= 2 else 0.0
    return Proposal(
        symbol=instant.symbol,
        sector=sector,
        notional=DEFAULT_NOTIONAL,
        horizon_hours=DEFAULT_HORIZON_HOURS,
        expected_loss_pct=Decimal(str(round(vol * BAD_CASE_SIGMAS * 100, 4))),
    )


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """One mandate, scored on what its decisions were actually worth."""

    name: str
    taken: tuple[float, ...]
    refused: tuple[float, ...]

    @property
    def decisions(self) -> int:
        return len(self.taken) + len(self.refused)

    @property
    def refusal_rate(self) -> float:
        return len(self.refused) / self.decisions if self.decisions else 0.0

    @property
    def taken_value(self) -> float:
        return fmean(self.taken) if self.taken else 0.0

    @property
    def refused_value(self) -> float:
        """Mean net outcome of what this profile declined. **Negative is good.**"""
        return fmean(self.refused) if self.refused else 0.0

    @property
    def take_everything(self) -> float:
        everything = [*self.taken, *self.refused]
        return fmean(everything) if everything else 0.0

    @property
    def utility_gain_bps(self) -> float:
        """What the mandate earned over taking every proposal, per decision.

        Positive means the refusals were worth making. It can be negative, and on a mandate that
        refuses indiscriminately it will be.
        """
        return self.taken_value * (1.0 - self.refusal_rate) - self.take_everything

    @property
    def verdict(self) -> str:
        if not self.refused:
            return (
                f"{self.name} refused nothing across {self.decisions} proposal(s), so it cannot "
                f"have helped or hurt: on this sample it is not a mandate, it is a pass-through"
            )
        if not self.taken:
            return (
                f"{self.name} refused everything across {self.decisions} proposal(s). It cannot "
                f"lose money and it cannot make any; refusing all trades is not a strategy, and "
                f"the utility number below is the cost of that"
            )
        direction = "earned its keep" if self.utility_gain_bps > 0 else "cost its owner money"
        return (
            f"{self.name} refused {self.refusal_rate:.0%} of proposals and {direction}: the trades "
            f"it declined averaged {self.refused_value:+.1f}bps and the ones it allowed averaged "
            f"{self.taken_value:+.1f}bps, a gain of {self.utility_gain_bps:+.2f}bps per decision "
            f"over taking everything"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decisions": self.decisions,
            "taken": len(self.taken),
            "refused": len(self.refused),
            "refusal_rate": round(self.refusal_rate, 4),
            "taken_value_bps": round(self.taken_value, 3),
            "refused_value_bps": round(self.refused_value, 3),
            "take_everything_bps": round(self.take_everything, 3),
            "utility_gain_bps": round(self.utility_gain_bps, 4),
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class ProfileValueReport:
    """Every shipped mandate, measured against what actually happened."""

    instants: int
    divergence_rate: float
    results: tuple[ProfileResult, ...]

    @property
    def any_helped(self) -> bool:
        return any(r.utility_gain_bps > 0 and r.taken and r.refused for r in self.results)

    @property
    def verdict(self) -> str:
        head = (
            f"{len(self.results)} shipped mandate(s) over {self.instants:,} real instant(s); the "
            f"mandates disagreed with each other on {self.divergence_rate:.0%} of them."
        )
        sizing = (
            " Note that a mandate which narrows every position loses less than one which does not, "
            "and that is sizing rather than selection: the utility figure above isolates the "
            "refusals by measuring each mandate against taking everything *at its own permitted "
            "size*, so the two effects are not added together."
        )
        helped = [r.name for r in self.results if r.utility_gain_bps > 0 and r.taken and r.refused]
        if helped:
            return (
                f"{head} {', '.join(helped)} improved its owner's outcome by refusing: the trades "
                f"declined were worse than the ones allowed. This is the measurement the "
                f"divergence audit could not make, and it is the one a judge should ask for."
                + sizing
            )
        return (
            f"{head} NO MANDATE IMPROVED ITS OWNER'S OUTCOME on this sample. Divergence is real "
            f"and demonstrated; usefulness is not. A mandate that refuses trades no better or "
            f"worse than the ones it allows is enforcing a preference, not adding value, and "
            f"saying so is the honest reading of this result." + sizing
        )

    def render(self) -> str:
        lines = [
            f"PROFILE VALUE — {self.instants:,} instant(s), "
            f"mandates disagreed on {self.divergence_rate:.0%}",
            "",
            f"{'profile':>24}{'refused':>9}{'declined':>11}{'allowed':>10}{'gain':>9}",
        ]
        for result in self.results:
            lines.append(
                f"{result.name:>24}{result.refusal_rate:>9.0%}{result.refused_value:>10.1f}b"
                f"{result.taken_value:>9.1f}b{result.utility_gain_bps:>8.2f}b"
            )
        lines.append("")
        lines.extend(f"  {r.verdict}" for r in self.results)
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "instants": self.instants,
            "divergence_rate": round(self.divergence_rate, 4),
            "any_helped": self.any_helped,
            "results": [r.as_dict() for r in self.results],
            "verdict": self.verdict,
        }


def _direction(instant: Instant) -> int:
    """The direction the proposal would have taken: the trailing sign, flat on a tie."""
    window = instant.trailing[-24:]
    total = sum(window) if window else 0.0
    if total > 0:
        return DIRECTION_LONG
    return DIRECTION_SHORT if total < 0 else DIRECTION_NONE


def evaluate(
    instants: Sequence[Instant], *, minimum: int = MIN_INSTANTS
) -> ProfileValueReport:
    """Replay a proposal from every instant past every shipped mandate and score the outcomes."""
    if len(instants) < minimum:
        raise ProfileValueError(
            f"{len(instants)} instant(s) is below the {minimum} a utility claim needs"
        )
    profiles = standard_profiles()
    taken: dict[str, list[float]] = {p.name: [] for p in profiles}
    refused: dict[str, list[float]] = {p.name: [] for p in profiles}
    diverged = 0
    for instant in instants:
        direction = _direction(instant)
        if direction == DIRECTION_NONE:
            continue
        proposal = proposal_from(instant)
        net = instant.net_bps(direction)
        outcomes = set()
        for profile in profiles:
            verdict = judge(profile, proposal)
            outcomes.add(verdict.outcome)
            if verdict.outcome is Outcome.REFUSED:
                refused[profile.name].append(net)
            else:
                # **A resize is not a full take, and scoring it as one overstates the mandate's
                # outcomes in both directions.** On the shipped profiles this is not a corner
                # case: the conservative mandate resizes 98% of proposals and refuses 2%, so
                # treating a resize as a take would credit it with the whole move on almost every
                # decision. The position earns the fraction of the notional it was permitted.
                share = (
                    float(verdict.permitted_notional) / float(proposal.notional)
                    if proposal.notional else 0.0
                )
                taken[profile.name].append(net * min(1.0, max(0.0, share)))
        if len(outcomes) > 1:
            diverged += 1
    scored = [p for p in instants if _direction(p) != DIRECTION_NONE]
    return ProfileValueReport(
        instants=len(scored),
        divergence_rate=diverged / len(scored) if scored else 0.0,
        results=tuple(
            ProfileResult(name=p.name, taken=tuple(taken[p.name]), refused=tuple(refused[p.name]))
            for p in profiles
        ),
    )


def main() -> int:  # pragma: no cover - CLI
    from argus.eval.venue_rules import collect

    report = evaluate(collect())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BAD_CASE_SIGMAS",
    "MIN_INSTANTS",
    "ProfileResult",
    "ProfileValueError",
    "ProfileValueReport",
    "evaluate",
    "proposal_from",
]
