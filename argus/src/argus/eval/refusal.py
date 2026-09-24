"""Refusal Alpha — is standing aside a judgement, or is it paralysis?

A desk that never trades cannot be scored the usual way. There is no Sharpe, no drawdown and no win
rate, and on the current record there never has been: 231 decisions, every one ``no_trade``. Read
only through the usual metrics, such a desk is indistinguishable from one that is broken — and also
from one that is right.

The committed protocol says the record should settle that question: abstentions are settled *"so a
correct refusal is gradeable"*. This module is the arithmetic that grades it, and the first thing it
has to do is refuse to grade what cannot be graded.

**Why the settled ledger cannot answer this.** Every abstention settles once, at the hold period.
Measured across all 178 settled on 2026-09-15: minimum 24.02h, median 24.57h, maximum 36.39h.
Decisions are taken two hours apart, so consecutive 24h windows share ~92% of their span, and 178
rows spread over three calendar days carry on the order of **three independent observations**. A hit
rate over them would have three things behind it and a denominator of 178 printed next to it, which
is worse than no number at all.

So the input here is `paper/marks.py` — a second, shorter observation of the same decision, at a
horizon where consecutive windows do not overlap. Marks began under protocol v3 and **cannot be
back-filled**: decisions taken under v1 and v2 keep exactly the observations those versions
declared, and their accuracy stays UNDEFINED rather than being reconstructed from a price series
fetched after the outcomes were known.

**Four measurements, because "was the refusal right" is four different questions.**

* **coverage** — how often the desk stated a direction at all. A desk that answers "none" every
  time cannot be wrong and has told you nothing; one that always calls a direction may just be
  guessing. This is the denominator every other number here depends on, so it is reported first.
* **directional accuracy** — of the calls it did make, how many went the right way. The interval
  is a bootstrap over *cycles*, not over calls: the symbols decided in one cycle share one market
  move, so 289 calls from 26 cycles are nowhere near 289 independent observations, and the per-call
  Wilson interval once published here overstated the precision (52.7-64.0% against a cycle
  bootstrap's 50.3-66.4%; found by an independent audit on 2026-09-24). It is also compared with
  the naive call — the desk's own majority lean, stated every time — because a desk that leans up
  in a rising tape beats a coin without knowing anything.
* **hurdle clearance** — how often the move the desk passed on was even large enough to pay for the
  round trip. If it rarely cleared, abstaining was correct arithmetic, not timidity.
* **forgone edge** — the signed move in the direction the desk leaned, net of the hurdle it would
  have paid. This is the number that says whether refusing *cost* anything. It is the sharpest
  possible test of this desk's central claim and it is reported whether or not it flatters it.

**Horizons are never pooled.** Every mark carries the horizon actually measured, and a decision
taken in the last cycle before the overnight gap is next seen ~18h later rather than ~2h. Averaging
those together would produce a number about the recording schedule rather than about the desk —
the defect `eval/bookcalib.py` was found committing and which is not repeated here.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import comb, sqrt
from pathlib import Path
from typing import Any

from argus.paper.marks import Mark, read_marks

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "refusal_alpha.json"

HORIZONS: tuple[tuple[str, float, float], ...] = (
    ("about_2h", 1.5, 4.0),
    ("4h_to_12h", 4.0, 12.0),
    ("overnight_12h_plus", 12.0, 24.0),
)
"""Measured-horizon buckets, as ``(name, low, high)`` in hours.

``about_2h`` is the one the cycle schedule is designed to produce and the only one whose windows are
guaranteed not to overlap, since decisions are themselves 2h apart. The others exist because the
last cycle of the day is not followed by another for eighteen hours, and those marks are real
observations that must be reported at their own horizon rather than folded into the 2h figure.
"""

MIN_FOR_A_RATE = 30
"""Directional calls a horizon needs before an accuracy is quoted at all.

Not a tuned number and not presented as one. Below this the Wilson interval is so wide that every
plausible value survives, so quoting a point estimate would be publishing noise with a decimal
point. The raw counts are always reported, so a reader can disagree with this line without losing
the data.
"""

ROUND_TRIP_BPS = 12.0
"""Bitget taker fee both sides, from the committed protocol's own hurdle rule.

The deliberation term is deliberately **not** added here. It varies per cycle with session and
volatility, and this module scores a decision that was already taken rather than re-deriving the
hurdle that was applied to it — using a hurdle the desk did not face would be grading it against
the wrong bar. The effect is that `clears_hurdle` is a *lower* bound on how often abstaining was
correct, which is the safe direction for a claim about our own desk.
"""


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260924
"""Fixed, so the published interval is reproducible to the digit from the same marks."""


class RefusalError(ValueError):
    """Raised rather than reporting refusal quality computed from nothing."""


def _horizon_of(hours: float) -> str | None:
    """The bucket a measured horizon falls in, or ``None`` when it falls outside all of them."""
    for name, low, high in HORIZONS:
        if low <= hours < high:
            return name
    return None


def wilson(hits: int, n: int, *, z: float = 1.96) -> tuple[float, float] | None:
    """A 95% Wilson interval for a proportion, or ``None`` when there is nothing to bound.

    Wilson rather than the normal approximation because the normal one is badly wrong exactly where
    this module operates — small samples and proportions near 0 or 1, where it happily returns a
    bound below zero. Closed form, so no dependency: this package is pure Python by design.
    """
    if n <= 0:
        return None
    phat = hits / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def cluster_interval(cycles: list[tuple[int, int]], *, resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    """A 95% percentile-bootstrap interval for a hit rate, resampling whole cycles.

    ``cycles`` holds ``(hits, calls)`` per cycle. Resampling cycles rather than calls keeps the
    calls that shared one market move together, which is what makes the interval honest here."""
    usable = [c for c in cycles if c[1] > 0]
    if len(usable) < 2:
        return None
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(resamples):
        draw = [usable[rng.randrange(len(usable))] for _ in usable]
        rates.append(sum(h for h, _ in draw) / sum(n for _, n in draw))
    rates.sort()
    return rates[int(0.025 * resamples)], rates[int(0.975 * resamples) - 1]


def sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact binomial sign test p-value; ties are left out by the caller."""
    n = wins + losses
    if n == 0:
        return None
    tail: float = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True, slots=True)
class HorizonResult:
    """What the marks at one measured horizon say, and what they refuse to say."""

    horizon: str
    marks: int
    directional: int
    """Marks where the desk actually named a direction. ``lean: none`` is excluded, not counted
    as a miss — declining to call a direction is an answer, and the PM prompt invites it."""

    correct: int
    cleared_hurdle: int
    forgone_bps: tuple[float, ...]
    """Signed move in the leaned direction, net of the round trip, one per directional mark."""

    cycles: tuple[tuple[int, int, int], ...] = field(default=())
    """Per decision cycle: ``(lean hits, naive hits, directional calls)``. The naive call is the
    desk's majority lean, stated every time."""

    naive_lean: str = "up"

    @property
    def coverage(self) -> float | None:
        """Share of marks where a direction was stated. ``None`` when there are no marks."""
        return None if self.marks == 0 else self.directional / self.marks

    @property
    def accuracy(self) -> float | None:
        """Directional hit rate, or ``None`` when no direction was ever stated."""
        return None if self.directional == 0 else self.correct / self.directional

    @property
    def interval(self) -> tuple[float, float] | None:
        """The cycle-bootstrap interval; the per-call Wilson one only when no cycles are known."""
        if self.cycles:
            return cluster_interval([(hits, n) for hits, _, n in self.cycles])
        return wilson(self.correct, self.directional)

    @property
    def naive_correct(self) -> int:
        return sum(naive for _, naive, _ in self.cycles)

    @property
    def naive_accuracy(self) -> float | None:
        if not self.cycles or self.directional == 0:
            return None
        return self.naive_correct / self.directional

    @property
    def cycle_record(self) -> tuple[int, int, int]:
        """Cycles where the lean beat the naive call, lost to it, and tied."""
        wins = sum(1 for hits, naive, _ in self.cycles if hits > naive)
        losses = sum(1 for hits, naive, _ in self.cycles if hits < naive)
        return wins, losses, len(self.cycles) - wins - losses

    @property
    def beats_a_coin(self) -> bool | None:
        """Does the interval exclude 0.5? ``None`` when there is no interval to ask it of.

        Three states, never two: better than chance, not better than chance, and **not enough
        evidence to tell** — which a boolean would hide.
        """
        bounds = self.interval
        if bounds is None or self.directional < MIN_FOR_A_RATE:
            return None
        return bounds[0] > 0.5

    @property
    def beats_the_naive_call(self) -> bool | None:
        """Does the lean beat its own majority direction, cycle by cycle, at p < 0.05?"""
        if not self.cycles or self.directional < MIN_FOR_A_RATE:
            return None
        wins, losses, _ = self.cycle_record
        p = sign_test(wins, losses)
        return p is not None and p < 0.05 and wins > losses

    @property
    def clearance(self) -> float | None:
        """Share of passed-on moves that were large enough to pay the round trip."""
        return None if self.marks == 0 else self.cleared_hurdle / self.marks

    @property
    def median_forgone_bps(self) -> float | None:
        """Median net edge the desk gave up by standing aside, in the direction it leaned.

        Positive means refusing cost money on the typical decision. Negative means refusing saved
        it. This is the number with the most power to embarrass the desk, so it is computed on
        every run and never gated behind a sample-size rule — only its interpretation is.
        """
        if not self.forgone_bps:
            return None
        ordered = sorted(self.forgone_bps)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def as_dict(self) -> dict[str, Any]:
        def pct(value: float | None) -> float | None:
            return None if value is None else round(value * 100.0, 1)

        bounds = self.interval
        return {
            "horizon": self.horizon,
            "marks": self.marks,
            "directional": self.directional,
            "correct": self.correct,
            "coverage_pct": pct(self.coverage),
            "accuracy_pct": pct(self.accuracy),
            "accuracy_ci95": None if bounds is None else [pct(bounds[0]), pct(bounds[1])],
            "interval_method": ("bootstrap over decision cycles" if self.cycles
                                else "Wilson, per call"),
            "cycles": len(self.cycles),
            "beats_a_coin": self.beats_a_coin,
            "naive_lean": self.naive_lean,
            "naive_accuracy_pct": pct(self.naive_accuracy),
            "cycles_lean_beat_naive": self.cycle_record[0],
            "cycles_naive_beat_lean": self.cycle_record[1],
            "cycles_tied": self.cycle_record[2],
            "sign_test_p_vs_naive": (None if not self.cycles
                                     else sign_test(*self.cycle_record[:2])),
            "beats_the_naive_call": self.beats_the_naive_call,
            "hurdle_clearance_pct": pct(self.clearance),
            "median_forgone_bps": (
                None if self.median_forgone_bps is None else round(self.median_forgone_bps, 2)
            ),
            "enough_for_a_rate": self.directional >= MIN_FOR_A_RATE,
        }


@dataclass(frozen=True, slots=True)
class RefusalReport:
    """The marks, read at every horizon they were actually taken over."""

    horizons: tuple[HorizonResult, ...]
    total_marks: int
    outside_buckets: int
    as_of: datetime

    @property
    def usable(self) -> HorizonResult | None:
        """The shortest horizon carrying enough directional calls to quote a rate."""
        for row in self.horizons:
            if row.directional >= MIN_FOR_A_RATE:
                return row
        return None

    @property
    def verdict(self) -> str:
        if self.total_marks == 0:
            return (
                "No mark has been recorded yet, so refusal quality is UNDEFINED — not zero, and "
                "not poor. Marks began under protocol v3 and cannot be back-filled onto decisions "
                "taken before it, because choosing a new measurement after the outcomes are "
                "visible is the thing pre-registration exists to prevent. The number becomes "
                "computable as cycles run, and not before."
            )
        row = self.usable
        if row is None:
            best = max(self.horizons, key=lambda r: r.directional)
            return (
                f"{self.total_marks} mark(s) recorded, the largest horizon bucket carrying "
                f"{best.directional} directional call(s) against the {MIN_FOR_A_RATE} needed to "
                f"quote a rate. **Directional accuracy is UNDEFINED.** The counts are published "
                f"below so the gap is visible rather than implied; this is a function of elapsed "
                f"cycles and cannot be hurried."
            )
        verdict = (
            f"At the `{row.horizon}` horizon: {row.correct} of {row.directional} directional calls "
            f"went the right way."
        )
        bounds = row.interval
        if bounds is not None:
            method = (f"bootstrap over {len(row.cycles)} decision cycles" if row.cycles
                      else "Wilson")
            verdict += f" 95% interval ({method}) {bounds[0]:.1%} to {bounds[1]:.1%}."
        if row.beats_a_coin is True:
            verdict += " It excludes a coin flip."
        elif row.beats_a_coin is False:
            verdict += (" It includes 0.5, so this sample does not show the lean beating a coin "
                        "flip.")
        naive = row.naive_accuracy
        if naive is not None:
            wins, losses, ties = row.cycle_record
            verdict += (
                f" But calling `{row.naive_lean}` every time was right {naive:.1%} of the time, "
                f"and cycle by cycle the lean beat that naive call {wins} times and lost {losses} "
                f"({ties} tied)"
            )
            verdict += (" — it beats the naive call." if row.beats_the_naive_call
                        else " — so there is no evidence yet that the lean knows more than the "
                             "direction of the tape.")
        forgone = row.median_forgone_bps
        if forgone is not None:
            verdict += (
                f" Median net edge forgone by standing aside: {forgone:+.1f}bps after a "
                f"{ROUND_TRIP_BPS}bps round trip"
            )
            verdict += (
                " — refusing cost money on the typical decision."
                if forgone > 0
                else " — refusing saved money on the typical decision."
            )
        return verdict

    def render(self) -> str:
        lines = [
            f"REFUSAL ALPHA — {self.total_marks} mark(s)"
            + (f", {self.outside_buckets} outside every horizon bucket" if self.outside_buckets
               else ""),
            "",
            "  horizon              marks  directional  correct  coverage  accuracy  forgone",
        ]
        for row in self.horizons:
            def show(value: float | None) -> str:
                return "      —" if value is None else f"{value * 100.0:6.1f}%"

            forgone = (
                "      —" if row.median_forgone_bps is None
                else f"{row.median_forgone_bps:+7.1f}"
            )
            lines.append(
                f"  {row.horizon:20s} {row.marks:5d}  {row.directional:11d}  {row.correct:7d}"
                f" {show(row.coverage)} {show(row.accuracy)} {forgone}"
            )
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "round_trip_bps": ROUND_TRIP_BPS,
            "min_directional_for_a_rate": MIN_FOR_A_RATE,
            "total_marks": self.total_marks,
            "outside_buckets": self.outside_buckets,
            "horizons": [row.as_dict() for row in self.horizons],
            "verdict": self.verdict,
        }


def score(marks: list[Mark], *, now: datetime | None = None) -> RefusalReport:
    """Read a set of marks at every horizon they were taken over. Pools nothing."""
    buckets: dict[str, list[Mark]] = {name: [] for name, _, _ in HORIZONS}
    outside = 0
    for row in marks:
        name = _horizon_of(row.horizon_hours)
        if name is None:
            outside += 1
            continue
        buckets[name].append(row)

    results: list[HorizonResult] = []
    for name, _, _ in HORIZONS:
        rows = buckets[name]
        directional = [r for r in rows if r.is_directional]
        correct = sum(1 for r in directional if r.lean_was_right)
        ups = sum(1 for r in directional if r.lean == "up")
        naive = "up" if ups * 2 >= len(directional) else "down"
        per_cycle: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for r in directional:
            # Cycles are two hours apart and take minutes, so the decision hour names the cycle.
            cell = per_cycle[r.decided_at[:13]]
            cell[0] += 1 if r.lean_was_right else 0
            cell[1] += 1 if (r.move > 0 if naive == "up" else r.move < 0) else 0
            cell[2] += 1
        cleared = sum(1 for r in rows if abs(r.move) > ROUND_TRIP_BPS)
        forgone = tuple(
            (r.move if r.lean == "up" else -r.move) - ROUND_TRIP_BPS for r in directional
        )
        results.append(HorizonResult(
            horizon=name,
            marks=len(rows),
            directional=len(directional),
            correct=correct,
            cleared_hurdle=cleared,
            forgone_bps=forgone,
            cycles=tuple((c[0], c[1], c[2]) for _, c in sorted(per_cycle.items())),
            naive_lean=naive,
        ))
    return RefusalReport(
        horizons=tuple(results),
        total_marks=len(marks),
        outside_buckets=outside,
        as_of=now or datetime.now(UTC),
    )


def orphans(marks: list[Mark], *, ledger_path: Path | None = None) -> list[int]:
    """Marks whose sequence number does not correspond to a real decision.

    **This guard exists because the absence of it nearly cost something.** While the mark path was
    being verified by hand, a throwaway script wrote two synthetic rows — prices of 100 and 200,
    sequence numbers no decision had yet reached — straight into the live
    `data/refusal_marks.jsonl`. They were caught and removed within the minute, but nothing in the
    code would have objected, and a fabricated row in the file that feeds a published metric is the
    exact failure this project is built against.

    So a mark is now only scoreable if the ledger contains the decision it claims to observe. An
    orphan is named and refused, never silently dropped: a quiet drop would shrink the denominator
    and make the report look cleaner than the data.
    """
    path = ledger_path or (DATA / "paper_ledger.jsonl")
    if not path.exists():
        # No ledger to check against. Every mark is unverifiable, which is not the same as every
        # mark being valid, and is reported as the former.
        return sorted({m.seq for m in marks})
    known = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            known.add(int(json.loads(line)["seq"]))
    return sorted({m.seq for m in marks if m.seq not in known})


def run(*, out: Path = REPORT_PATH, now: datetime | None = None) -> dict[str, Any]:
    """Score every mark on disk and write the artefact.

    Refuses outright if any mark has no decision behind it, rather than scoring what it can.
    """
    loaded = read_marks()
    stray = orphans(loaded)
    if stray:
        raise RefusalError(
            f"{len(stray)} mark(s) name a decision the ledger does not contain "
            f"(seq {stray[:8]}{'...' if len(stray) > 8 else ''}). Refusing to score a refusal "
            f"record against observations that have no decision behind them."
        )
    report = score(loaded, now=now)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report.as_dict()


def main() -> int:  # pragma: no cover - CLI
    report = score(read_marks())
    print(report.render())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
