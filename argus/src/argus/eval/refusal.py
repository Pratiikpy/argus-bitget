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

**A fifth question, added 2026-09-25: was the stated reason true?** The four measurements above ask
whether standing aside was *economically* right. None of them asks whether the reason the desk
wrote down for standing aside was true — and a refusal that is right for a false reason is an
explainability defect even when it saves money, because the reason is what a reader acts on.
VisualWebArena grades exactly this for its unachievable tasks: ``llm_ua_match``
(``evaluation_harness/helper_functions.py:610-642``, MIT) sets the *reported* reason for giving up
beside the *actual* one and asks only "same or different". ARGUS does not need a model for most of
it, because the record already holds the facts the theses cite:

* **session** — a thesis that says the anchor is asleep, the market closed, or that it is the
  weekend, against the ``session_phase`` the ledger recorded at decision time (and the reverse);
* **hours to price discovery** — a number quoted in the thesis against ``hours_to_discovery``;
* **the hedge menu** — "no hedge available" against the desk's own note (`agents/desk.py:531`
  writes ``hedge menu empty`` exactly when the menu is empty);
* **the risk layer** — a thesis that blames the risk layer for the abstention against
  `data/risk_records.jsonl`, which says whether the Constitution actually intervened;
* **size against direction** — the two forward claims a thesis makes about magnitude ("the binding
  constraint was direction, not size" and "any move would be too small to clear the round trip"),
  graded against the next ~2h mark and set beside the base rate of clearance over every mark, so a
  claim that is borne out no more often than the base rate is visible as carrying no information.

What was taken from VWA: the question (reported reason against actual reason) and its binary
verdict. What was not: the model call — a claim the record can settle is settled by the record, so
this grading is deterministic and reproducible to the digit. Claims the record cannot settle (the
24h change a thesis quotes, a panel's edge in bps) are **not graded** rather than guessed at; the
report says how many refusals carried no gradeable claim at all. Voided rows (seqs 264 and 265,
`paper/corrections.py`) are not graded: their thesis argues for the trade the record voided. Results
are in ``data/refusal_reasons.json`` (:func:`run_reasons`).

**Finding, 2026-09-25.** 682 refusals; 325 state at least one reason the record can check. Of the
691 decision-time reasons graded — session 275, hours to price discovery 266, hedge menu 150 —
all 691 agree with the record and none contradicts it; no refusal blames the risk layer, and the
risk records agree it never intervened. The one forward claim the desk makes about magnitude, "the
binding constraint was direction, not size", was borne out by the next ~2h move 37 of 46 times:
80%, against a base rate of 75% over every ~2h refusal mark (one-sided exact binomial p = 0.26).
True, and no more informative than the tape. The first version of this grader reported two
contradictions (seqs 139 and 460); both were its own misreading of "**no** anchor market open", and
the negation guard in :func:`_asserted` exists because of them.

**Re-run 2026-09-26** on the grown record: 696 refusals, 333 with a checkable reason. The
decision-time tally is unchanged — 691 of 691 consistent — because none of the 14 newer theses
(seqs 685-698, all RTH) states a session, hours-to-discovery or hedge-menu claim in words the
patterns read, and six state "direction, not size" (five borne out, one not); that is a
coverage limit of this grader, not evidence about those theses. "Direction, not size" was borne
out 43 of 54 times (80%) against the same 75% base rate (p = 0.26): still true, still no more
informative than the tape.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import comb, sqrt
from pathlib import Path
from typing import Any

from argus.eval import artefact
from argus.paper.marks import Mark, read_marks

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "refusal_alpha.json"
REASONS_PATH = DATA / "refusal_reasons.json"
LEDGER_PATH = DATA / "paper_ledger.jsonl"
NOTES_PATH = DATA / "desk_notes.jsonl"
RISK_PATH = DATA / "risk_records.jsonl"

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


def binomial_upper(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact. The one-sided test a forward claim's hit rate is
    held to against the base rate."""
    return min(1.0, sum(comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k, n + 1)))


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


# --- the stated reason, against the record -------------------------------------------------------


class Claim(StrEnum):
    """A kind of reason a refusal thesis gives that the record can check."""

    SESSION_CLOSED = "session_closed"
    SESSION_OPEN = "session_open"
    HOURS_TO_DISCOVERY = "hours_to_discovery"
    NO_HEDGE = "no_hedge"
    RISK_LAYER = "risk_layer"
    MOVES_ENOUGH = "moves_enough"
    MOVE_TOO_SMALL = "move_too_small"


FORWARD_CLAIMS = frozenset({Claim.MOVES_ENOUGH, Claim.MOVE_TOO_SMALL})
"""Claims about a move that had not happened yet. Graded against the next ~2h mark, and reported
apart from the decision-time claims, because one later price is a noisy test of a claim about
magnitude in a way that a recorded session phase is not a noisy test of "the market is closed"."""

CONSISTENT = "consistent"
CONTRADICTED = "contradicted"
UNGRADEABLE = "ungradeable"

HOURS_TOLERANCE = 1.5
"""How far a quoted "N hours to price discovery" may sit from the recorded figure and still agree.

Not tuned. A thesis is written minutes after the session clock is read and rounds freely ("52
hours" for 51.99, "32+ hours" for 32.13); a cycle takes minutes, not hours. A gap wider than this
cannot be rounding: it is a number from a different session, or no session at all."""

_CLOSED_CLAIM = re.compile(
    r"\bweekend\s+(?:session|phase|anchor|pricing|liquidity|tape|gap)\b|"
    r"\b(?:it\s+is|it's|during\s+the|over\s+the|this)\s+weekend\b|"
    r"\banchor(?:\s+market)?\s+(?:is\s+)?(?:asleep|closed|shut|dormant)\b|"
    r"\b(?:us\s+|anchor\s+)?markets?\s+(?:is|are)\s+(?:now\s+)?(?:closed|shut)\b|"
    r"\b(?:without|no)\s+(?:a\s+)?live\s+anchor\b|"
    r"\b(?:after[- ]hours|overnight|extended[- ]hours)\s+session\b", re.I)
"""A thesis saying there is no price discovery **now**. "The weekend memory", "prior weekend
moves" and "8 prior weekend decisions" are references to the past and deliberately do not match:
six RTH theses use the word that way (seqs 196-213) and none of them claims the market is shut."""

_OPEN_CLAIM = re.compile(
    r"\b(?:us\s+|anchor\s+)?markets?\s+(?:is|are)\s+(?:now\s+)?open\b|"
    r"\bregular\s+(?:trading\s+)?(?:session|hours)\b|"
    r"\bduring\s+(?:us\s+)?(?:market|regular|trading)\s+hours\b|"
    r"\banchor(?:\s+market)?\s+(?:is\s+)?(?:open|awake)\b|\bRTH\b", re.I)

_HOURS_CLAIM = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*[- ]?hours?\s+(?:to|until|before|from)\s+(?:the\s+)?(?:next\s+)?"
    r"(?:genuine\s+)?price\s+discovery|"
    r"\basleep\s+for\s+~?\s*(\d+(?:\.\d+)?)\s*\+?\s*hours?", re.I)

_HEDGE_CLAIM = re.compile(
    r"\bno\s+hedge(?:\s+(?:menu|available|instruments?))?\b|\bhedge\s+menu\s+(?:is\s+)?empty\b|"
    r"\bempty\s+hedge\s+menu\b|\bnothing\s+(?:is\s+)?placeable\b", re.I)

_RISK_LAYER_CLAIM = re.compile(
    r"\b(?:the\s+|our\s+|desk'?s\s+)?(?:risk\s+layer|constitution|risk\s+kernel|risk\s+gate)\s+"
    r"(?:blocks?|blocked|forbids?|forbade|caps?|capped|prevents?|prevented|vetoe?s?|vetoed|"
    r"refuses?|refused|rejects?|rejected|narrows?|narrowed)\b|"
    r"\bblocked\s+by\s+(?:the\s+)?(?:risk\s+(?:layer|kernel|gate|limit)|constitution)\b|"
    r"\b(?:position|exposure|risk)\s+(?:limit|cap)\s+(?:is\s+|was\s+)?(?:reached|breached|hit|"
    r"binding|blocks?|prevents?)\b", re.I)
"""The desk's **own** risk layer named as the cause. News about a "kill switch proposal" (seq 262)
is not a claim about the Constitution and must not match."""

_MOVES_ENOUGH_CLAIM = re.compile(
    r"\bdirection,?\s+not\s+(?:size|magnitude)\b|\btape\s+moves\s+enough\b|"
    r"\bmoves?\s+(?:are\s+|is\s+)?(?:large|big)\s+enough\b", re.I)

_TOO_SMALL_CLAIM = re.compile(
    r"\bany\s+move\s+would\s+be\b[^.]{0,60}\bunlikely\s+to\s+clear\b|"
    r"\b(?:realistic|expected|likely|projected|remaining)\s+(?:upside|downside|move|range)\b"
    r"[^.]{0,60}\b(?:unlikely\s+to|will\s+not|won't|cannot)\s+(?:clear|cover|pay)\b|"
    r"\bnot\s+(?:large|big)\s+enough\s+to\s+(?:pay|clear|cover)\b|"
    r"\brange\s+(?:is\s+)?too\s+(?:narrow|small)\b", re.I)
"""A claim about the size of the coming move. "The remaining *edge* is unlikely to clear the
hurdle" does not match: an edge is a directional expectation, and grading it against an unsigned
move would test a claim the thesis did not make."""


@dataclass(frozen=True, slots=True)
class ReasonCheck:
    """One reason a refusal gave, set beside what the record says about it."""

    seq: int
    symbol: str
    claim: Claim
    stated: str
    """The words in the thesis that make the claim, verbatim."""

    recorded: str
    """What the record holds for the same fact, with the file it came from."""

    verdict: str
    """``consistent``, ``contradicted`` or ``ungradeable`` — three states, because a claim the
    record cannot reach is neither confirmed nor refuted and must not be counted as either."""

    def as_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "symbol": self.symbol, "claim": str(self.claim),
                "stated": self.stated, "recorded": self.recorded, "verdict": self.verdict}


_NEGATED = re.compile(r"\b(?:no|not|never|without|nor)\s+(?:\w+\s+)?$", re.I)


def _asserted(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """The first match that the thesis asserts rather than negates.

    Found on the first run (2026-09-25): "a weekend token with **no** anchor market open" (seq 139)
    and "the weekend session has **no** anchor market open" (seq 460) matched the open-market
    pattern and were graded as contradicting a weekend record — two false contradictions out of
    two. A reason grader that misreads a negation accuses the desk of the error it is looking for.
    """
    for found in pattern.finditer(text):
        if not _NEGATED.search(text[max(0, found.start() - 24):found.start()]):
            return found
    return None


def _excerpt(text: str, match: re.Match[str], *, pad: int = 60) -> str:
    start, end = max(0, match.start() - pad), min(len(text), match.end() + pad)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def check_reasons(
    entry: Mapping[str, Any], *, notes: Sequence[str] | None = None,
    risk: Mapping[str, Any] | None = None, mark: Mark | None = None,
) -> list[ReasonCheck]:
    """Every checkable reason one refusal's thesis gives, each graded against the record.

    ``notes`` are the desk's own notes for this decision (`data/desk_notes.jsonl`), ``risk`` its
    risk ruling (`data/risk_records.jsonl`), ``mark`` its later observation (`paper/marks.py`).
    Any of them may be absent; a claim that needs an absent record is ``ungradeable``, never
    assumed to hold.
    """
    thesis = str(entry.get("thesis") or "")
    seq, symbol = int(entry["seq"]), str(entry.get("symbol", ""))
    phase = str(entry.get("session_phase") or "").lower()
    out: list[ReasonCheck] = []

    def add(claim: Claim, match: re.Match[str], recorded: str, verdict: str) -> None:
        out.append(ReasonCheck(seq, symbol, claim, _excerpt(thesis, match), recorded, verdict))

    closed, opened = _asserted(_CLOSED_CLAIM, thesis), _asserted(_OPEN_CLAIM, thesis)
    recorded_phase = f"session_phase={phase or 'unrecorded'} (paper_ledger.jsonl)"
    if closed:
        add(Claim.SESSION_CLOSED, closed, recorded_phase,
            UNGRADEABLE if not phase else CONTRADICTED if phase == "rth" else CONSISTENT)
    if opened:
        add(Claim.SESSION_OPEN, opened, recorded_phase,
            UNGRADEABLE if not phase else CONSISTENT if phase == "rth" else CONTRADICTED)

    hours = _HOURS_CLAIM.search(thesis)
    if hours:
        quoted = float(hours.group(1) or hours.group(2))
        raw = entry.get("hours_to_discovery")
        if raw is None or raw == "":
            add(Claim.HOURS_TO_DISCOVERY, hours, "hours_to_discovery unrecorded", UNGRADEABLE)
        else:
            actual = float(raw)
            add(Claim.HOURS_TO_DISCOVERY, hours,
                f"hours_to_discovery={actual:g} (paper_ledger.jsonl); quoted {quoted:g}",
                CONSISTENT if abs(quoted - actual) <= HOURS_TOLERANCE else CONTRADICTED)

    hedge = _HEDGE_CLAIM.search(thesis)
    if hedge:
        if notes is None:
            add(Claim.NO_HEDGE, hedge, "no desk notes for this decision", UNGRADEABLE)
        elif any(n.lower().startswith("hedge menu empty") for n in notes):
            add(Claim.NO_HEDGE, hedge, "note 'hedge menu empty' (desk_notes.jsonl)", CONSISTENT)
        else:
            # `agents/desk.py:531` writes the line whenever the menu is empty, so notes that exist
            # without it are the record of a menu that had something on it.
            add(Claim.NO_HEDGE, hedge, "notes present, no 'hedge menu empty' line "
                "(desk_notes.jsonl)", CONTRADICTED)

    blamed = _RISK_LAYER_CLAIM.search(thesis)
    if blamed:
        if risk is None:
            add(Claim.RISK_LAYER, blamed, "no risk record for this decision", UNGRADEABLE)
        else:
            intervened = bool(risk.get("intervened"))
            add(Claim.RISK_LAYER, blamed,
                f"intervened={intervened}, binding_constraint="
                f"{risk.get('binding_constraint', '')} (risk_records.jsonl)",
                CONSISTENT if intervened else CONTRADICTED)

    for claim, pattern in ((Claim.MOVES_ENOUGH, _MOVES_ENOUGH_CLAIM),
                           (Claim.MOVE_TOO_SMALL, _TOO_SMALL_CLAIM)):
        found = pattern.search(thesis)
        if not found:
            continue
        if mark is None or _horizon_of(mark.horizon_hours) != HORIZONS[0][0]:
            add(claim, found, "no ~2h mark for this decision", UNGRADEABLE)
            continue
        cleared = abs(mark.move) > ROUND_TRIP_BPS
        borne_out = cleared if claim is Claim.MOVES_ENOUGH else not cleared
        add(claim, found,
            f"next move {mark.move:+.1f}bps over {mark.horizon_hours:.2f}h against a "
            f"{ROUND_TRIP_BPS:g}bps round trip (refusal_marks.jsonl)",
            CONSISTENT if borne_out else CONTRADICTED)
    return out


@dataclass(frozen=True, slots=True)
class ReasonReport:
    """Every graded reason across the desk's refusals, and what could not be graded."""

    checks: tuple[ReasonCheck, ...]
    refusals: int
    base_clearance: tuple[int, int]
    """``(cleared, marks)`` over every ~2h refusal mark: how often *any* passed-on move cleared
    the round trip. The yardstick a forward magnitude claim has to beat to mean anything."""

    as_of: datetime

    @property
    def refusals_with_a_claim(self) -> int:
        return len({c.seq for c in self.checks if c.verdict != UNGRADEABLE})

    def tally(self, claim: Claim) -> dict[str, int]:
        rows = [c for c in self.checks if c.claim is claim]
        return {v: sum(1 for c in rows if c.verdict == v)
                for v in (CONSISTENT, CONTRADICTED, UNGRADEABLE)}

    @property
    def contradicted(self) -> tuple[ReasonCheck, ...]:
        return tuple(c for c in self.checks if c.verdict == CONTRADICTED)

    def _share(self, claims: frozenset[Claim]) -> tuple[int, int]:
        graded = [c for c in self.checks if c.claim in claims and c.verdict != UNGRADEABLE]
        return sum(1 for c in graded if c.verdict == CONSISTENT), len(graded)

    @property
    def verdict(self) -> str:
        if self.refusals == 0:
            return "No refusal is on the record, so no stated reason can be graded — UNDEFINED."
        now_claims = frozenset(Claim) - FORWARD_CLAIMS
        held, graded = self._share(now_claims)
        text = (f"{self.refusals} refusal(s) on record; {self.refusals_with_a_claim} carried at "
                f"least one reason the record can check. Of {graded} decision-time reason(s) "
                f"graded, {held} agreed with the record and {graded - held} contradicted it.")
        if graded == 0:
            text = (f"{self.refusals} refusal(s) on record and none stated a reason the record "
                    f"can check; reason accuracy is UNDEFINED.")
        for claim in sorted(FORWARD_CLAIMS):
            test = self.against_base(claim)
            if test is None:
                continue
            held, n, base, p = test
            text += (f" `{claim}` claims were borne out by the next ~2h move {held} of {n} "
                     f"time(s) ({held / n:.0%}), against {base:.0%} for every ~2h refusal mark")
            text += (f" (one-sided exact binomial p = {p:.2f}); "
                     + ("the stated reason carries information beyond the base rate."
                        if p < 0.05 else
                        "no evidence the stated reason says more than the base rate does."))
        return text

    def against_base(self, claim: Claim) -> tuple[int, int, float, float] | None:
        """``(borne out, graded, base rate, p)`` for a forward claim, or ``None`` if none graded.

        The base rate is how often the claim would have been borne out had it been made on every
        ~2h refusal mark; ``p`` is the chance of doing at least this well at that rate. A claim
        that is right four times in five is uninformative when the tape does the same unasked."""
        tally = self.tally(claim)
        n = tally[CONSISTENT] + tally[CONTRADICTED]
        cleared, marks = self.base_clearance
        if n == 0 or marks == 0:
            return None
        base = cleared / marks if claim is Claim.MOVES_ENOUGH else (marks - cleared) / marks
        return tally[CONSISTENT], n, base, binomial_upper(tally[CONSISTENT], n, base)

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "refusals": self.refusals,
            "refusals_with_a_gradeable_claim": self.refusals_with_a_claim,
            "hours_tolerance": HOURS_TOLERANCE,
            "round_trip_bps": ROUND_TRIP_BPS,
            "base_clearance_about_2h": {"cleared": self.base_clearance[0],
                                        "marks": self.base_clearance[1]},
            "by_claim": {str(claim): self.tally(claim) for claim in Claim},
            "forward_against_base": {
                str(claim): (None if (t := self.against_base(claim)) is None else {
                    "borne_out": t[0], "graded": t[1], "base_rate": round(t[2], 4),
                    "p_one_sided": round(t[3], 4)})
                for claim in sorted(FORWARD_CLAIMS)},
            "forward_claims": sorted(str(c) for c in FORWARD_CLAIMS),
            "contradicted": [c.as_dict() for c in self.contradicted],
            "verdict": self.verdict,
            "method": (
                "Each checkable reason in a no_trade thesis is matched by a fixed pattern and set "
                "beside the record: session_phase and hours_to_discovery from the ledger, the "
                "desk's own hedge-menu note, the risk ruling, and the next ~2h mark for claims "
                "about the size of the coming move. Question and binary verdict from "
                "VisualWebArena's llm_ua_match (evaluation_harness/helper_functions.py:610-642, "
                "MIT); graded here by the record, not by a model."),
        }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def grade_reasons(
    *, ledger_path: Path = LEDGER_PATH, notes_path: Path = NOTES_PATH,
    risk_path: Path = RISK_PATH, marks: Sequence[Mark] | None = None,
    now: datetime | None = None,
) -> ReasonReport:
    """Grade the stated reason of every ``no_trade`` decision against the record."""
    loaded = list(read_marks() if marks is None else marks)
    by_seq = {m.seq: m for m in loaded}
    notes = {int(r["seq"]): [str(n) for n in r.get("notes") or []] for r in _jsonl(notes_path)}
    risk = {int(r["seq"]): r for r in _jsonl(risk_path)}
    refusals = [r for r in _jsonl(ledger_path)
                if r.get("kind", "decision") == "decision" and r.get("verdict") == "no_trade"]
    checks: list[ReasonCheck] = []
    for entry in refusals:
        seq = int(entry["seq"])
        checks.extend(check_reasons(entry, notes=notes.get(seq), risk=risk.get(seq),
                                    mark=by_seq.get(seq)))
    short = [m for m in loaded if _horizon_of(m.horizon_hours) == HORIZONS[0][0]]
    return ReasonReport(
        checks=tuple(checks), refusals=len(refusals),
        base_clearance=(sum(1 for m in short if abs(m.move) > ROUND_TRIP_BPS), len(short)),
        as_of=now or datetime.now(UTC),
    )


def reason_line(path: Path = REASONS_PATH) -> str | None:
    """The reason grading as one sentence for the console's record answers, or ``None`` when
    ``data/refusal_reasons.json`` is missing or graded nothing.

    Written for the place the console already quotes this module's lean grading
    (`lui/answer.py:_lean_grading`, inserted where a desk with no settled trade explains itself):
    the decision-time reasons checked against the record, the contradictions named by count, and
    the forward magnitude claim set beside its base rate — including when, as on every run so
    far, that comparison says the reason carries no information the tape did not.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        by_claim: Mapping[str, Mapping[str, int]] = report["by_claim"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    now = [c for c in Claim if c not in FORWARD_CLAIMS]
    held = sum(int(by_claim.get(str(c), {}).get(CONSISTENT, 0)) for c in now)
    wrong = sum(int(by_claim.get(str(c), {}).get(CONTRADICTED, 0)) for c in now)
    if held + wrong == 0:
        return None
    line = (f"The reasons the desk wrote for standing aside were checked against its own record: "
            f"of {held + wrong} that the record can settle (session, hours to price discovery, "
            f"hedge menu, risk layer), {held} agreed and {wrong} contradicted it")
    forward = (report.get("forward_against_base") or {}).get(str(Claim.MOVES_ENOUGH))
    if forward and forward.get("graded"):
        share = forward["borne_out"] / forward["graded"]
        line += (f"; \"direction, not size\" was borne out {forward['borne_out']} of "
                 f"{forward['graded']} times ({share:.0%}) against {forward['base_rate']:.0%} for "
                 f"every refusal (p = {forward['p_one_sided']:.2f}), "
                 + ("so it carries information beyond the tape"
                    if forward["p_one_sided"] < 0.05 else
                    "so it is true but says no more than the tape does"))
    return (line + f" — graded {str(report.get('as_of', ''))[:10]}, "
            f"`python -m argus.eval.refusal`.")


def run_reasons(*, out: Path = REASONS_PATH, now: datetime | None = None) -> dict[str, Any]:
    """Grade every refusal's stated reason and write ``data/refusal_reasons.json``."""
    report = grade_reasons(now=now).as_dict()
    artefact.write(out, report)
    return report


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
    reasons = run_reasons()
    print(f"\nREASON AGAINST RECORD — {reasons['verdict']}\nwritten to {REASONS_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
