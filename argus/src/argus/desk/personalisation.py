"""Personalisation, proved rather than claimed.

`argus.desk.workbench.TraderProfile` carries a docstring that states its own acceptance test:

    The proof that personalisation is real is that two profiles get **different verdicts on
    identical market state**.

Nothing demonstrated it. The profile was a dataclass with two presets, `argus.agents.mandate` built
limits from it, and neither was wired into the desk's decision path — so "personalised thesis", a
named Track 3 judging criterion, rested on an assertion in a comment.

This module is the demonstration, and it is deliberately shaped so it can **fail**.
:func:`diverge` runs one market state past every profile and reports what each would do. When they
all reach the same answer it says so in those words: **personalisation did not bind here**. A
harness that can only confirm is not evidence, and the same discipline is already applied to the
risk layer in `argus.eval.riskaudit`, where a guard that never fires is graded untested rather than
safe.

**Why the proof is deterministic and has no model in it.** The verdicts below come from arithmetic
over the profile's own limits — horizon, notional, position share, sector share, loss tolerance.
Asking a language model whether a trade suits a conservative investor would produce a different
answer on a second run, and a demonstration that does not reproduce demonstrates nothing. The model
still writes the thesis; what *binds* is code, which is the same asymmetry the Constitution uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from argus.agents.mandate import HORIZON_SLACK, Mandate
from argus.desk.workbench import TraderProfile

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


class Outcome(StrEnum):
    """What a profile would do with the same proposal."""

    TAKEN = "taken"
    RESIZED = "resized"
    """Permitted, but smaller. The position survives; the size does not."""

    REFUSED = "refused"
    """Not a smaller version of this trade — a different trader's trade."""

    @property
    def carries_risk(self) -> bool:
        return self is not Outcome.REFUSED


@dataclass(frozen=True, slots=True)
class Proposal:
    """One trade idea, before any profile has looked at it."""

    symbol: str
    sector: str
    notional: Decimal
    horizon_hours: float
    expected_loss_pct: Decimal
    """The loss the thesis itself concedes in its bad case, in per cent of the position."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "sector": self.sector,
            "notional": str(self.notional), "horizon_hours": self.horizon_hours,
            "expected_loss_pct": str(self.expected_loss_pct),
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """What one profile does with the proposal, and why."""

    profile: str
    outcome: Outcome
    permitted_notional: Decimal
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "outcome": str(self.outcome),
            "permitted_notional": str(self.permitted_notional),
            "reasons": list(self.reasons),
        }

    def render(self) -> str:
        head = f"[profile] {self.profile}: {self.outcome}"
        if self.outcome is Outcome.RESIZED:
            head += f" to {self.permitted_notional}"
        return head if not self.reasons else f"{head} — {'; '.join(self.reasons)}"


def judge(profile: TraderProfile, proposal: Proposal, *, book: Any = None) -> Verdict:
    """What this trader does with this trade. Arithmetic over the profile's own limits.

    The order of the checks is the order of severity, and it matters. A horizon breach is a
    **refusal** rather than a resize: a thesis that needs a month to play out does not become
    suitable for a two-day trader by halving the size, it becomes a smaller bet on the wrong
    horizon. `argus.agents.mandate` states the same thing — "this is a different trader's trade,
    not a smaller version of this one" — and this is where it binds.

    ``book`` (:class:`argus.desk.book.Book`, or ``None``) is new 2026-09-15, the day Foundation 3
    (a real portfolio) shipped. **`TraderProfile.max_concurrent_positions`'s own docstring has said
    "a cap the book must respect" since before any book existed to respect it** — and
    `agents.mandate.Mandate.out_of_mandate` has accepted an `open_positions` count from the start,
    correctly, but nothing in this proof-of-personalisation path ever supplied one, so the cap was
    declared, documented and never once enforced here. Same class of defect this module already
    found and fixed once for `excluded_symbols` (`eval/standing.py`'s register entry for this
    capability) — a field the profile carries that the judge silently never reads. ``None`` means
    no book was supplied and the check stays inert, honest for every caller that has not been
    updated yet, not read as "zero positions open".
    """
    mandate = Mandate(profile=profile)
    reasons: list[str] = []

    # 0. A named exclusion, which is absolute and outranks every other check.
    #
    # **This was missing entirely.** `Mandate.refuses_symbol` existed and its docstring said "a
    # named exclusion is absolute; no thesis outvotes it" — and `judge` never called it. The
    # conservative preset excludes TQQQUSDT and SQQQUSDT, and this function reported both as TAKEN
    # with the reason "within a 5% position limit", citing limits it had checked while silently
    # skipping the one that should have ended the decision.
    #
    # It surfaced from `eval/profilestudy.py`: over 195 real frames every symbol diverged at
    # exactly 83%, including the two excluded ones. A symbol the mandate forbids behaving
    # identically to one it permits is not a plausible result, and it was not one.
    if mandate.refuses_symbol(proposal.symbol):
        reasons.append(
            f"{proposal.symbol} is named in this mandate's exclusions; an excluded instrument is "
            f"refused outright, and no size or horizon makes it eligible"
        )
        return Verdict(profile.name, Outcome.REFUSED, _ZERO, tuple(reasons))

    # 1. Horizon — a refusal, never a resize.
    if not mandate.permits_horizon(proposal.horizon_hours):
        reasons.append(
            f"the thesis needs {proposal.horizon_hours:.0f}h against a "
            f"{profile.holding_horizon_hours}h mandate; that is a different trader's trade"
        )
        return Verdict(profile.name, Outcome.REFUSED, _ZERO, tuple(reasons))

    # 2. Loss tolerance — also a refusal. A position whose own bad case exceeds what this trader
    #    can absorb is not made acceptable by being smaller: the percentage loss is unchanged.
    if proposal.expected_loss_pct > profile.loss_tolerance_pct:
        reasons.append(
            f"the thesis concedes {proposal.expected_loss_pct}% in its bad case against a "
            f"{profile.loss_tolerance_pct}% tolerance; resizing does not change a percentage"
        )
        return Verdict(profile.name, Outcome.REFUSED, _ZERO, tuple(reasons))

    # 2.5. Concurrent-position cap — a refusal, not a resize. A book already at its position cap
    #      is not fixed by taking this one idea smaller; the trader needs to close something else
    #      first, which is a different decision from the one being judged here.
    if book is not None and profile.max_concurrent_positions:
        open_positions = book.total_open_positions()
        if open_positions >= profile.max_concurrent_positions:
            reasons.append(
                f"{open_positions} position(s) already open, at this mandate's cap of "
                f"{profile.max_concurrent_positions}; this proposal is declined rather than "
                f"squeezed in at a smaller size"
            )
            return Verdict(profile.name, Outcome.REFUSED, _ZERO, tuple(reasons))

    # 3. Size — a resize, because the idea survives at a smaller ticket.
    ceiling = mandate.max_position_notional
    if proposal.notional > ceiling:
        reasons.append(
            f"notional {proposal.notional} exceeds the {profile.max_position_pct}% position limit "
            f"({ceiling}) on {profile.capital} of capital"
        )
        return Verdict(profile.name, Outcome.RESIZED, ceiling, tuple(reasons))

    # The horizon actually applied is the profile's figure times `HORIZON_SLACK`, so quoting the
    # bare mandate hours here read as a contradiction: a 52h proposal was reported TAKEN "within a
    # 48h horizon". The ceiling that was tested is the one named.
    return Verdict(
        profile.name, Outcome.TAKEN, proposal.notional,
        (f"within a {profile.max_position_pct}% position limit and the "
         f"{mandate.horizon_ceiling_hours:.0f}h ceiling this mandate actually applies "
         f"({profile.holding_horizon_hours}h x {HORIZON_SLACK} slack)",),
    )


@dataclass(frozen=True, slots=True)
class Divergence:
    """What every profile did with one identical proposal."""

    proposal: Proposal
    verdicts: tuple[Verdict, ...]

    @property
    def outcomes(self) -> set[Outcome]:
        return {v.outcome for v in self.verdicts}

    @property
    def diverged(self) -> bool:
        """Did the profiles actually reach different answers?

        The whole acceptance test. ``False`` is a real result and is reported as one — it means the
        profiles were not binding on this proposal, not that personalisation works.
        """
        return len(self.outcomes) > 1

    @property
    def sizes_diverged(self) -> bool:
        """Even where the outcome matches, the permitted size may not."""
        return len({v.permitted_notional for v in self.verdicts}) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.as_dict(),
            "diverged": self.diverged,
            "sizes_diverged": self.sizes_diverged,
            "outcomes": sorted(str(o) for o in self.outcomes),
            "verdicts": [v.as_dict() for v in self.verdicts],
        }

    def render(self) -> list[str]:
        lines = [v.render() for v in self.verdicts]
        if self.diverged:
            lines.append(
                f"[profile] the same market state produced {len(self.outcomes)} different "
                f"outcomes across {len(self.verdicts)} profile(s) — personalisation bound here"
            )
        elif self.sizes_diverged:
            lines.append(
                "[profile] every profile reached the same outcome at a different size; "
                "personalisation bound on size but not on direction"
            )
        else:
            lines.append(
                "[profile] every profile reached the same answer — **personalisation did not bind "
                "here**. That is a fact about this proposal, not evidence that profiles work"
            )
        return lines


def diverge(
    proposal: Proposal, profiles: Sequence[TraderProfile], *, book: Any = None,
) -> Divergence:
    """Run one market state past every profile."""
    return Divergence(
        proposal=proposal,
        verdicts=tuple(judge(profile, proposal, book=book) for profile in profiles),
    )


@dataclass(frozen=True, slots=True)
class DivergenceReport:
    """How often personalisation actually changed the answer, across many proposals."""

    cases: tuple[Divergence, ...]

    @property
    def rate(self) -> float | None:
        if not self.cases:
            return None
        return sum(1 for c in self.cases if c.diverged) / len(self.cases)

    @property
    def is_evidence(self) -> bool:
        """Personalisation is demonstrated only if it changed the outcome at least once.

        Deliberately not a threshold. One proposal on which two profiles genuinely disagree proves
        the mechanism binds; a high rate over proposals chosen to diverge proves only that they were
        chosen to diverge.
        """
        return any(c.diverged for c in self.cases)

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposals": len(self.cases),
            "diverged": sum(1 for c in self.cases if c.diverged),
            "divergence_rate": None if self.rate is None else round(self.rate, 3),
            "is_evidence": self.is_evidence,
            "cases": [c.as_dict() for c in self.cases],
        }

    def render(self) -> list[str]:
        if not self.cases:
            return ["[profile] no proposals were judged; personalisation is untested"]
        diverged = sum(1 for c in self.cases if c.diverged)
        lines = [
            f"[profile] {diverged} of {len(self.cases)} proposal(s) produced different outcomes "
            f"across the profiles"
        ]
        if not self.is_evidence:
            lines.append(
                "[profile] no proposal separated the profiles, so personalisation is **not "
                "demonstrated** by this set — the limits may simply never have been reached"
            )
        return lines


def audit(
    proposals: Sequence[Proposal], profiles: Sequence[TraderProfile], *, book: Any = None,
) -> DivergenceReport:
    return DivergenceReport(tuple(diverge(p, profiles, book=book) for p in proposals))


def standard_profiles() -> tuple[TraderProfile, ...]:
    """The two shipped presets, which sit at opposite ends of every limit."""
    return (TraderProfile.conservative(), TraderProfile.aggressive())


def main() -> int:
    """Demonstrate personalisation on proposals a real desk would see."""
    import json

    profiles = standard_profiles()
    proposals = (
        Proposal("NVDAUSDT", "technology", Decimal("20000"), 24.0, Decimal("8")),
        Proposal("NVDAUSDT", "technology", Decimal("4000"), 24.0, Decimal("2")),
        Proposal("MSTRUSDT", "technology", Decimal("15000"), 2000.0, Decimal("12")),
        Proposal("COINUSDT", "technology", Decimal("3000"), 12.0, Decimal("20")),
    )
    report = audit(proposals, profiles)
    for case in report.cases:
        print(f"\n{case.proposal.symbol} {case.proposal.notional} over "
              f"{case.proposal.horizon_hours:.0f}h, bad case {case.proposal.expected_loss_pct}%")
        for line in case.render():
            print("  ", line)
    print()
    for line in report.render():
        print(line)
    print("\n" + json.dumps({"divergence_rate": report.as_dict()["divergence_rate"],
                             "is_evidence": report.is_evidence}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Divergence",
    "DivergenceReport",
    "Outcome",
    "Proposal",
    "Verdict",
    "audit",
    "diverge",
    "judge",
    "main",
    "standard_profiles",
]
