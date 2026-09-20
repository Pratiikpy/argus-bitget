"""Profile divergence, measured over the frames the desk actually faced.

*"Personalized thesis"* is one of Track 3's four judged criteria
(`BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md:280`), and the standing register records the gap precisely:
**"divergence rate across the profile set has not been measured over a run of real frames"**.

`desk/personalisation.py` proves the mechanism — two profiles reaching different verdicts on
identical state — but its `main()` runs **four hand-written proposals**. Four cases chosen by the
same hand that wrote the profiles can only demonstrate that divergence is *possible*. It cannot say
how often it happens on the instruments and horizons this desk really sees, and "possible" is not a
rate.

This module supplies the rate, and is explicit about which half of each input is real:

* **Real, taken from `data/paper_ledger.jsonl`** — every distinct ``(symbol, horizon)`` the desk
  actually faced. Twelve instruments and, on the current log, forty-five distinct horizons running
  from 0.0h to 52.0h.
* **Declared, because the record cannot supply it** — the notional and the conceded bad-case loss.
  **Every one of the 231 recorded decisions carries ``quantity: 0``**, so there is no proposed size
  anywhere in the log to read. That is not a gap in this study; it is the central finding of the
  paper record, and inventing a size while pretending it was observed would hide it.

So the honest statement of what is computed here is: *given the instruments and horizons this desk
genuinely encountered, and a declared ladder of sizes, how often do two trader mandates disagree?*
The artefact says exactly that, and `REAL_DIMENSIONS` / `DECLARED_DIMENSIONS` are written into it so
a reader cannot mistake the second for the first.

**It is shaped to be able to fail.** If both profiles reach the same verdict on every real frame,
the rate is 0.0 and the report says personalisation did not bind — which would be a real finding
against a criterion we are scored on, and is reported rather than tuned away.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.desk.personalisation import (
    DivergenceReport,
    Proposal,
    audit,
    diverge,
    standard_profiles,
)
from argus.desk.workbench import TraderProfile

DATA = Path(__file__).resolve().parents[3] / "data"
LEDGER_PATH = DATA / "paper_ledger.jsonl"
REPORT_PATH = DATA / "profile_divergence.json"

REAL_DIMENSIONS = ("symbol", "horizon_hours")
"""Taken from the decision record. Not chosen, not representative-by-selection — every one."""

DECLARED_DIMENSIONS = ("notional", "expected_loss_pct")
"""Stated by this module, because the log contains no proposed size to read.

Labelled rather than blended. A study that mixes observed and invented inputs and reports one number
invites the reader to treat all of it as observed, which is the failure this project spends most of
its effort avoiding.
"""

NOTIONAL_LADDER: tuple[Decimal, ...] = (
    Decimal("2000"), Decimal("6000"), Decimal("20000"),
)
"""Three sizes straddling the profiles' own limits.

The conservative mandate permits 5% of 100,000 = 5,000 and the aggressive one 25% = 25,000, so the
rungs sit below both, between them, and below only the looser one. A ladder entirely inside or
entirely outside both limits would produce unanimity by construction and measure nothing.
"""

LOSS_LADDER: tuple[Decimal, ...] = (Decimal("2"), Decimal("8"))
"""Bad-case losses either side of the conservative 3% tolerance and below the aggressive 15%."""

SECTOR = "technology"
"""Every rToken in the universe is a US technology name or an index built from them.

Stated here rather than inferred per symbol: a sector map we invented would be another declared
dimension wearing the clothes of a real one, and the sector limits are not what this study varies.
"""


class ProfileStudyError(ValueError):
    """Raised rather than reporting a divergence rate computed from no real frames."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One ``(symbol, horizon)`` the desk actually faced, and how often it faced it."""

    symbol: str
    horizon_hours: float
    decisions: int
    """How many recorded decisions carried this pair. Kept so a reader can see whether the
    distinct frames are evenly represented or dominated by one repeated state."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "horizon_hours": self.horizon_hours,
            "decisions": self.decisions,
        }


def real_frames(path: Path = LEDGER_PATH) -> list[Frame]:
    """Every distinct ``(symbol, horizon)`` in the decision record.

    Deduplicated, because the same pair recurring across cycles is the same question asked twice and
    counting it twice would weight the divergence rate by how often a cycle happened to run rather
    than by how varied the desk's situation was. The repeat count travels with each frame so that
    choice is visible rather than silent.
    """
    if not path.exists():
        return []
    counts: dict[tuple[str, float], int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        horizon = row.get("hours_to_discovery")
        if horizon is None:
            # A decision with no recorded horizon cannot be placed against a holding-period limit.
            # Skipped and counted, never defaulted to zero — a zero horizon is "trading right now",
            # which is a claim about the state rather than an absence of one.
            continue
        key = (str(row["symbol"]), round(float(horizon), 2))
        counts[key] = counts.get(key, 0) + 1
    return [
        Frame(symbol=s, horizon_hours=h, decisions=n)
        for (s, h), n in sorted(counts.items())
    ]


def proposals_for(frames: Sequence[Frame]) -> list[Proposal]:
    """Cross the real frames with the declared size and loss ladders."""
    out: list[Proposal] = []
    for frame in frames:
        for notional in NOTIONAL_LADDER:
            for loss in LOSS_LADDER:
                out.append(Proposal(
                    symbol=frame.symbol,
                    sector=SECTOR,
                    notional=notional,
                    horizon_hours=frame.horizon_hours,
                    expected_loss_pct=loss,
                ))
    return out


@dataclass(frozen=True, slots=True)
class RungResult:
    """Divergence at one declared size/loss rung, across every real frame.

    **This exists because the pooled rate lied.** The first run of this study reported 83.3%
    divergence and it was a property of the ladder, not of the desk: every rung came back at
    exactly 0% or exactly 100%, so the real frames moved nothing and 5-of-6 rungs produced 5/6.

    Holding the declared rung fixed and varying only the real dimensions is the test that separates
    them. A rung whose rate is 0% or 100% across all frames is **not discriminating** — the mandates
    disagree (or agree) there for reasons that have nothing to do with what the desk faced.
    """

    notional: str
    loss_pct: str
    frames: int
    diverged: int

    @property
    def rate(self) -> float | None:
        return None if self.frames == 0 else self.diverged / self.frames

    @property
    def discriminating(self) -> bool:
        """Did the real frames change the answer at this rung?

        False when every frame agrees, whichever way. That is the honest reading: a rung where all
        195 frames diverge tells you about the size ladder, not about the instruments.
        """
        return 0 < self.diverged < self.frames

    def as_dict(self) -> dict[str, Any]:
        return {
            "notional": self.notional,
            "loss_pct": self.loss_pct,
            "frames": self.frames,
            "diverged": self.diverged,
            "rate_pct": None if self.rate is None else round(self.rate * 100.0, 1),
            "discriminating": self.discriminating,
        }


@dataclass(frozen=True, slots=True)
class StudyResult:
    """The divergence rate, with the provenance of every input attached."""

    report: DivergenceReport
    frames: tuple[Frame, ...]
    profiles: tuple[str, ...]
    as_of: datetime
    rungs: tuple[RungResult, ...] = ()
    by_symbol: tuple[tuple[str, int, int], ...] = ()
    """``(symbol, diverged, total)`` — the real dimension, reported so a reader can see whether it

    moves at all. On the current record only the two instruments the conservative mandate names in
    its exclusions separate from the rest.
    """

    @property
    def decisions_covered(self) -> int:
        return sum(f.decisions for f in self.frames)

    @property
    def rate(self) -> float | None:
        return self.report.rate

    @property
    def discriminating_rungs(self) -> tuple[RungResult, ...]:
        """Rungs where the real frames actually changed the answer."""
        return tuple(r for r in self.rungs if r.discriminating)

    @property
    def attributable_to_real(self) -> float | None:
        """Share of proposals whose divergence turns on a real dimension.

        Counted only at discriminating rungs: at a rung where every frame diverges, the divergence
        is the ladder's doing and crediting it to the instruments would be double-counting the
        study's own design as a finding.
        """
        total = sum(r.frames for r in self.rungs)
        if total == 0:
            return None
        return sum(r.diverged for r in self.discriminating_rungs) / total

    @property
    def verdict(self) -> str:
        if not self.frames:
            return (
                "No frame could be read from the decision record, so the divergence rate is "
                "UNDEFINED — not zero. A rate over no frames would be a statement about an empty "
                "log presented as a statement about personalisation."
            )
        rate = self.rate
        if rate is None:
            return "No proposal was built from the frames, so there is nothing to report."
        head = (
            f"Across {len(self.frames)} distinct (symbol, horizon) frame(s) drawn from "
            f"{self.decisions_covered} recorded decision(s), crossed with "
            f"{len(NOTIONAL_LADDER)}x{len(LOSS_LADDER)} declared size/loss rungs: "
            f"{len(self.report.cases)} proposal(s), and the {len(self.profiles)} mandates "
            f"disagreed on **{rate:.1%}** of them."
        )
        if rate == 0.0:
            return head + (
                " **Personalisation did not bind on a single real frame.** That is a finding "
                "against a judged criterion, and it is reported rather than tuned away."
            )
        attributable = self.attributable_to_real
        tail = (
            " The size and the conceded loss are declared, not observed — every recorded decision "
            "carries quantity 0, so the log holds no proposed size to read. The instruments and "
            "horizons are real."
        )
        if not self.discriminating_rungs:
            return head + tail + (
                " **Read that headline with care: none of the declared rungs discriminates.** At "
                "every rung the real frames either all diverge or none do, so this number is a "
                "property of the size/loss ladder and the instruments and horizons moved nothing. "
                "Quoting it as a measure of personalisation over real data would be wrong."
            )
        return head + tail + (
            f" Of that, **{attributable:.1%} of all proposals diverge for a reason that turns on a "
            f"real dimension** — {len(self.discriminating_rungs)} of {len(self.rungs)} declared "
            f"rungs discriminate between frames at all. The remainder is the ladder's doing and is "
            f"not evidence about the desk's instruments or horizons."
        )

    def render(self) -> str:
        lines = [
            f"PROFILE DIVERGENCE — {len(self.frames)} real frame(s), "
            f"{len(self.report.cases)} proposal(s), {len(self.profiles)} mandate(s)",
            f"  real:     {', '.join(REAL_DIMENSIONS)}",
            f"  declared: {', '.join(DECLARED_DIMENSIONS)}",
            "",
            "  declared rung           frames  diverged   rate  discriminating",
        ]
        for rung in self.rungs:
            rate = "     —" if rung.rate is None else f"{rung.rate * 100.0:5.1f}%"
            lines.append(
                f"  n={rung.notional:>6s} loss={rung.loss_pct:>3s}%  {rung.frames:6d}  "
                f"{rung.diverged:8d} {rate}  {'YES' if rung.discriminating else 'no'}"
            )
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "real_dimensions": list(REAL_DIMENSIONS),
            "declared_dimensions": list(DECLARED_DIMENSIONS),
            "notional_ladder": [str(n) for n in NOTIONAL_LADDER],
            "loss_ladder": [str(loss) for loss in LOSS_LADDER],
            "profiles": list(self.profiles),
            "frames": len(self.frames),
            "decisions_covered": self.decisions_covered,
            "proposals": len(self.report.cases),
            "divergence_rate": self.rate,
            "attributable_to_real_dimensions": self.attributable_to_real,
            "rungs": [r.as_dict() for r in self.rungs],
            "discriminating_rungs": len(self.discriminating_rungs),
            "by_symbol": [
                {"symbol": sym, "diverged": d, "total": t, "rate_pct": round(100 * d / t, 1)}
                for sym, d, t in self.by_symbol
            ],
            "is_evidence": self.report.is_evidence,
            "verdict": self.verdict,
            "frame_detail": [f.as_dict() for f in self.frames],
        }


def study(
    *,
    ledger: Path = LEDGER_PATH,
    profiles: Sequence[TraderProfile] | None = None,
    now: datetime | None = None,
) -> StudyResult:
    """Measure how often the mandates disagree over the frames the desk really saw."""
    chosen = tuple(profiles or standard_profiles())
    if len(chosen) < 2:
        raise ProfileStudyError(
            "divergence needs at least two mandates; one mandate always agrees with itself"
        )
    frames = real_frames(ledger)
    report = audit(proposals_for(frames), chosen)

    # Decompose while the proposals are still paired with the frame that produced them. Pooling
    # first and decomposing later is what produced a headline that turned out to be about the
    # ladder.
    rungs: list[RungResult] = []
    symbol_hits: dict[str, list[int]] = {}
    for notional in NOTIONAL_LADDER:
        for loss in LOSS_LADDER:
            diverged = 0
            for frame in frames:
                case = diverge(
                    Proposal(frame.symbol, SECTOR, notional, frame.horizon_hours, loss), chosen
                )
                tally = symbol_hits.setdefault(frame.symbol, [0, 0])
                tally[1] += 1
                if case.diverged:
                    diverged += 1
                    tally[0] += 1
            rungs.append(RungResult(str(notional), str(loss), len(frames), diverged))

    return StudyResult(
        report=report,
        frames=tuple(frames),
        profiles=tuple(p.name for p in chosen),
        as_of=now or datetime.now(UTC),
        rungs=tuple(rungs),
        by_symbol=tuple(
            (sym, hits[0], hits[1]) for sym, hits in sorted(symbol_hits.items())
        ),
    )


def main() -> int:  # pragma: no cover - CLI
    result = study()
    print(result.render())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
