"""Does the model *remember* the answer? Memorisation leakage, measured on our own facts.

Every evaluation in this project that asks a model about a past period has the same hole in it: if
the period predates the model's training cutoff, a good result may be **recall of what happened**
rather than judgement about what was going to. `eval/shadow.py` grades the desk's lean against the
move that followed; `research/*_study.py` sweeps historical windows; `eval/bakeoff.py` compares
strategies over history. None of them could say whether the model had simply read the answer.

**Read before written.** The reference is barj28's *A Validated Instrument for Memorization Leakage
in LLM Trading Evaluation* (MIT, DOI 10.5281/zenodo.20844335), cloned to
`research/repos-themed/barj28~llm-leakage-instrument`. Its black-box probe is
`src/vr_C_api.py:124-137`:

* distractors are **multiplicative** on the true value — wide ``{0.5, 0.7, 1.4, 2.0}`` separates
  recall from a random guess, tight ``{0.85, 0.93, 1.08, 1.18}`` separates *precise* recall from an
  order-of-magnitude estimate. Both sets are copied exactly.
* the option order is shuffled with a **deterministic per-fact seed** (`vr_C_api.py:130`), so the
  same fact always presents the same way and a rerun measures the model rather than the shuffle.
* the model answers with a single letter, and capacity is ``P(prefers the true value)`` against a
  chance level of ``1/len(options)``.

Their headline finding, which is the reason this module reports capacity and never stops there:
frontier models **recall financial fundamentals precisely and that recall does not become trading
skill**. Memorisation is therefore not automatically a disqualification — it is a fact about the
evaluation that has to be stated before any result from an overlapping window is believed.

**What is ours.** The facts. barj28 ships a revenue panel; we build probes from
`market/fundamentals.py`, which reads what companies actually filed with the SEC,
point-in-time and restatement-resolved. So the ground truth is verifiable from primary
sources we already hold, and the probe set can be regenerated rather than trusted as a
frozen CSV.

**The axis is the filing date, not the fiscal period.** A model can only have learned a number after
it was published, so a fact for Q2 2026 filed in August 2026 is post-cutoff for any model trained
before then, whatever its fiscal label says. Bucketing by period instead would smear the boundary
this instrument exists to locate.

**It refuses to conclude from a handful of probes.** Capacity is a proportion, and a proportion over
four observations has a confidence interval wider than the range it lives in. Buckets below
:data:`MIN_PROBES_PER_BUCKET` are reported with their counts and left out of the boundary
estimate.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

WIDE_MULTIPLIERS: tuple[float, ...] = (0.5, 0.7, 1.4, 2.0)
"""`vr_C_api.py` ``MULTS`` — recall against a random guess."""

TIGHT_MULTIPLIERS: tuple[float, ...] = (0.85, 0.93, 1.08, 1.18)
"""The log-matched set — *precise* recall against an order-of-magnitude estimate.

A model that knows NVIDIA's revenue is "tens of billions" answers the wide probe correctly without
having memorised anything. Only the tight probe separates knowing the number from knowing the scale.
"""

MIN_PROBES_PER_BUCKET = 8
"""Fewest probes before a bucket's capacity is used to locate the boundary."""

SIGNIFICANCE = 0.05
"""One-sided binomial threshold for "above chance". Tested with
`research/significance.py:binomial_p_value`, which this module imports rather than reimplements."""

LETTERS = "ABCDE"

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "leakage.json"
"""Where the measurement is stored, and where :func:`check_window` reads it from.

Module level rather than built inside the CLI: the gate has to find the same file the run
wrote, and a path constructed twice is a path that can differ.
"""


class LeakageError(RuntimeError):
    """Raised rather than reporting a capacity computed from a probe set that cannot support one."""


@dataclass(frozen=True, slots=True)
class Probe:
    """One multiple-choice question with a verifiable answer and the date it became public."""

    entity: str
    label: str
    published: date
    """When the fact was filed. The axis the boundary is measured on."""

    value: float
    unit: str
    options: tuple[float, ...]
    correct: str
    kind: str
    """``wide`` or ``tight``."""

    @property
    def chance(self) -> float:
        return 1.0 / len(self.options)

    def render(self) -> str:
        lines = "\n".join(
            f"{LETTERS[i]}) {_format(option)} {self.unit}" for i, option in enumerate(self.options)
        )
        return (
            f"Which of the following was {self.entity}'s {self.label}?\n{lines}\n"
            f"Reply with ONLY the single letter of the correct option."
        )

    def render_control(self) -> str:
        """The same question with the answer stated in the prompt. The sensitivity control.

        barj28's cascade insists a leakage detector is only credible if it is **specific** (silent
        where memorisation is impossible) *and* **sensitive** (it fires where recall is known to be
        present) — their positive control is a LoRA that injects the values (`armB_*`). We cannot
        fine-tune a hosted endpoint, so recall is made certain the only other way: the value is put
        in the prompt. A model that still cannot pick the right letter has not shown an absence of
        memorisation; it has shown that this instrument cannot measure it, and a null result from an
        instrument that never fires is not evidence of anything.
        """
        lines = "\n".join(
            f"{LETTERS[i]}) {_format(option)} {self.unit}" for i, option in enumerate(self.options)
        )
        return (
            f"{self.entity} reported {self.label} of {_format(self.value)} {self.unit}.\n"
            f"Which of the following matches that figure?\n{lines}\n"
            f"Reply with ONLY the single letter of the correct option."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity, "label": self.label, "published": self.published.isoformat(),
            "value": self.value, "unit": self.unit, "kind": self.kind,
            "options": list(self.options), "correct": self.correct,
        }


def _format(value: float) -> float | int:
    """`vr_C_api.py:120-121` — integers above 10, one decimal below."""
    return round(value) if value >= 10 else round(float(value), 1)


def build_probe(
    *, entity: str, label: str, published: date, value: float, unit: str = "USD billion",
    kind: str = "wide", seed: str | None = None,
) -> Probe:
    """One probe, built exactly as `vr_C_api.py:124-137` builds its multiple choice.

    The deterministic per-fact shuffle matters more than it looks: with a random one, a rerun that
    scores differently cannot be told apart from a model that answered differently, and the whole
    instrument becomes unreproducible.
    """
    if value <= 0:
        raise LeakageError("a probe needs a positive value")
    multipliers = TIGHT_MULTIPLIERS if kind == "tight" else WIDE_MULTIPLIERS
    true = _format(value)
    options = {true}
    for multiplier in multipliers:
        options.add(_format(value * multiplier))
    ordered = [option for option in options if option >= 0.1]
    if len(ordered) < 3:
        raise LeakageError(
            f"{entity} {label}: distractors collapsed onto the true value after rounding; this "
            f"fact cannot be probed at this precision"
        )
    rng = random.Random(seed or f"{entity}{label}{published.isoformat()}{kind}")
    rng.shuffle(ordered)
    return Probe(
        entity=entity, label=label, published=published, value=value, unit=unit,
        options=tuple(float(o) for o in ordered), correct=LETTERS[ordered.index(true)], kind=kind,
    )


def answer_text(reply: Any) -> str:
    """The model's answer, never its reasoning.

    **The bug the sensitivity control caught.** The first version passed `str(completion)` to
    :func:`parse_letter`, and `QwenClient.complete` returns a `Completion` whose repr carries both
    `content` and `reasoning`. So the regex scanned the chain of thought and returned whichever
    letter appeared there first, which is uncorrelated with the answer. The instrument scored 2/12
    on probes whose answer was printed in the prompt — and that impossible number is the only reason
    the defect was found rather than shipped as "no memorisation detected".
    """
    content = getattr(reply, "content", None)
    return str(content if content is not None else reply)


def parse_letter(text: str) -> str | None:
    """`vr_C_api.py:140-142`.

    ``None`` when the model did not answer with a letter at all, which is scored as wrong rather
    than dropped: a model that will not answer has not demonstrated recall.
    """
    # **A deliberate departure from the reference.** `vr_C_api.py:140-142` searches for `[A-E]`
    # anywhere in the reply, which matches the "C" inside "I cannot help with that" and scores a
    # refusal as a confident answer — inflating capacity with noise that looks like recall. We
    # require a standalone letter instead. Every genuine answer here is a bare letter, so the
    # stricter rule loses nothing and stops a refusal from being read as a guess.
    match = re.search(r"\b([A-E])\b", (text or "").upper())
    return match.group(1) if match else None


@dataclass(frozen=True, slots=True)
class Answer:
    probe: Probe
    replied: str | None
    correct: bool


@dataclass(frozen=True, slots=True)
class Bucket:
    """Capacity over one period of publication dates."""

    label: str
    probes: int
    hits: int
    chance: float

    @property
    def capacity(self) -> float:
        return self.hits / self.probes if self.probes else 0.0

    @property
    def p_value(self) -> float:
        """One-sided binomial probability of this many hits or more, under chance."""
        from argus.research.significance import binomial_p_value

        return binomial_p_value(self.hits, self.probes, p_null=self.chance)

    @property
    def above_chance(self) -> bool:
        return (
            self.probes >= MIN_PROBES_PER_BUCKET
            and self.capacity > self.chance
            and self.p_value <= SIGNIFICANCE
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "probes": self.probes, "hits": self.hits,
            "capacity": round(self.capacity, 4), "chance": round(self.chance, 4),
            "p_value": round(self.p_value, 6), "above_chance": self.above_chance,
            "counted": self.probes >= MIN_PROBES_PER_BUCKET,
        }


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Where the model's recall of published facts stops, and what that forbids."""

    model: str
    answers: tuple[Answer, ...]
    buckets: tuple[Bucket, ...]
    kind: str

    control_hits: int = 0
    control_probes: int = 0
    """The sensitivity control: the same probes with the answer stated in the prompt.

    Zero means it was not run, and :attr:`verdict` then refuses to interpret a null as an absence.
    """

    @property
    def control_rate(self) -> float | None:
        """Share of control probes answered correctly. ``None`` when the control was not run."""
        if self.control_probes <= 0:
            return None
        return self.control_hits / self.control_probes

    @property
    def instrument_is_sensitive(self) -> bool:
        """Did the instrument demonstrate it can fire at all?

        The threshold is deliberately blunt: with the answer in the prompt, anything short of a
        clear majority means the format, not the memory, is what the probes are measuring.
        """
        rate = self.control_rate
        return rate is not None and rate >= 0.8

    @property
    def boundary(self) -> str | None:
        """The latest bucket whose capacity is significantly above chance.

        The signature barj28 calls cutoff-boundedness: recall collapses to chance for facts
        published after training. ``None`` means no bucket cleared the bar, which is a *good* result
        for an evaluation and a weak one for an instrument — it can also mean the probes were too
        hard or too few, and the counts are reported so a reader can tell which.
        """
        clean = [b for b in self.buckets if b.above_chance]
        return clean[-1].label if clean else None

    @property
    def overall(self) -> Bucket:
        probes = len(self.answers)
        hits = sum(1 for a in self.answers if a.correct)
        chance = (
            sum(a.probe.chance for a in self.answers) / probes if probes else 0.2
        )
        return Bucket(label="all", probes=probes, hits=hits, chance=chance)

    def overlaps_recall(self, start: date, end: date) -> bool:
        """Does an evaluation window sit inside the region the model demonstrably remembers?

        The gate. A backtest or a shadow grading over a window the model can recall is not evidence
        of foresight, and this returns True so the caller has to say so.
        """
        for bucket in self.buckets:
            if not bucket.above_chance:
                continue
            first, last = _bucket_range(bucket.label)
            if first <= end and start <= last:
                return True
        return False

    @property
    def verdict(self) -> str:
        """Delegates to :func:`verdict_of` so the live path and :func:`replay` cannot diverge."""
        return verdict_of(
            model=self.model, kind=self.kind, overall=self.overall, buckets=self.buckets,
            boundary=self.boundary, control_rate=self.control_rate,
            instrument_is_sensitive=self.instrument_is_sensitive,
        )


    def render(self) -> str:
        lines = [
            f"MEMORISATION LEAKAGE — {self.model}, {self.kind} probes",
            "",
            f"  {'published':12} {'probes':>7} {'hits':>6} {'capacity':>9} {'chance':>7} {'p':>8}",
        ]
        for bucket in self.buckets:
            mark = " *" if bucket.above_chance else ("  " if bucket.probes >= MIN_PROBES_PER_BUCKET
                                                    else " ~")
            lines.append(
                f"  {bucket.label:12} {bucket.probes:7d} {bucket.hits:6d} "
                f"{bucket.capacity:8.0%} {bucket.chance:6.0%} {bucket.p_value:8.4f}{mark}"
            )
        lines += ["", "  * above chance   ~ too few probes to count"]
        if self.control_rate is not None:
            lines.append(
                f"  control: {self.control_hits}/{self.control_probes} correct with the answer in "
                f"the prompt ({self.control_rate:.0%}) — instrument "
                f"{'sensitive' if self.instrument_is_sensitive else 'NOT SENSITIVE'}"
            )
        lines += ["", f"  {self.verdict}"]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "model": self.model,
            "kind": self.kind,
            "probes": len(self.answers),
            "overall": self.overall.as_dict(),
            "boundary": self.boundary,
            "control_probes": self.control_probes,
            "control_hits": self.control_hits,
            "control_rate": None if self.control_rate is None else round(self.control_rate, 4),
            "instrument_is_sensitive": self.instrument_is_sensitive,
            "buckets": [b.as_dict() for b in self.buckets],
            "verdict": self.verdict,
        }


def verdict_of(
    *,
    model: str,
    kind: str,
    overall: Bucket,
    buckets: Sequence[Bucket],
    boundary: str | None,
    control_rate: float | None,
    instrument_is_sensitive: bool,
) -> str:
    """The verdict sentence, derived from measurements that have already been made.

    **A pure function of its arguments, and deliberately the only place this prose exists.** The
    stored artefact once carried *"Recall is above chance in every period probed"* while the
    `buckets` array printed beside it showed a counted bucket at ``above_chance: false``, p=0.056 —
    a sentence contradicted by data produced in the same run. Fixing the sentence in one place and
    leaving a second copy for replaying stored results would reintroduce exactly that class of
    divergence, so there is one implementation and both callers use it.
    """
    if overall.probes < MIN_PROBES_PER_BUCKET:
        return (
            f"{overall.probes} probe(s) is too few to measure anything; the instrument needs "
            f"at least {MIN_PROBES_PER_BUCKET}"
        )
    head = (
        f"{model} answered {overall.hits}/{overall.probes} {kind} probes correctly "
        f"({overall.capacity:.0%}) against a {overall.chance:.0%} chance level "
        f"(p={overall.p_value:.4f})."
    )
    if boundary is None and not instrument_is_sensitive:
        control = (
            "the control was not run" if control_rate is None
            else f"the control answered only {control_rate:.0%} correctly with the answer "
                 f"in the prompt"
        )
        return head + (
            f" No period is above chance — and {control}, so this instrument has not shown it "
            f"can detect recall at all. **The null is uninterpretable**: it is equally "
            f"consistent with no memorisation and with a probe format this model cannot follow"
        )
    if boundary is None:
        return head + (
            " No period's recall is significantly above chance, so nothing here shows "
            "the model has memorised these facts. That is a permissive result and not a "
            "proof of absence: with this many probes only large effects can be ruled out. "
            f"The instrument is sensitive — it answered {control_rate:.0%} of the control "
            f"probes correctly with the value in the prompt — so it would have fired"
        )
    counted = [b for b in buckets if b.probes >= MIN_PROBES_PER_BUCKET]
    newest = counted[-1].label if counted else boundary
    if boundary == newest:
        # No upper boundary was located because recall persists into the newest period probed.
        # Saying "and not after" would describe a collapse the data never showed — and saying
        # "in every period" would be equally wrong whenever a counted bucket sits below the
        # line, which is why the counts are stated rather than summarised. The first version of
        # this sentence claimed "every period probed" while 2026H1 was counted at p=0.056, and
        # the artefact it was printed beside disagreed with it.
        above = sum(1 for b in counted if b.above_chance)
        below = [b.label for b in counted if not b.above_chance]
        aside = (
            "" if not below
            else f" {', '.join(below)} sits below the line, so recall is uneven rather than "
                 f"universal —"
        )
        return head + (
            f" Recall is above chance in {above} of {len(counted)} periods with enough probes "
            f"to count, including the most recent ({newest}).{aside} this instrument located "
            f"**no upper boundary**, and the model's knowledge extends at least that far. Any "
            f"evaluation over a window in that range is contaminated: the model has "
            f"demonstrably read facts published inside it, and a good result may be recall "
            f"rather than judgement. What it does not show is that the recall becomes trading "
            f"skill — barj28's own finding is that the two come apart, and that question is "
            f"`eval/shadow.py`'s, not this module's"
        )
    return head + (
        f" Recall is above chance through {boundary} and collapses to chance after, which "
        f"is the design-based signature of memorisation. Any evaluation window ending on or "
        f"before {boundary} is contaminated: a good result there may be recall of the "
        f"outcome rather than judgement about it"
    )


def _bucket_label(when: date) -> str:
    return f"{when.year}H{1 if when.month <= 6 else 2}"


def _bucket_range(label: str) -> tuple[date, date]:
    year = int(label[:4])
    if label.endswith("H1"):
        return date(year, 1, 1), date(year, 6, 30)
    return date(year, 7, 1), date(year, 12, 31)


def score(
    answers: Sequence[Answer], *, model: str, kind: str,
    control_hits: int = 0, control_probes: int = 0,
) -> LeakageReport:
    """Bucket the answers by publication half-year and test each against chance."""
    if not answers:
        raise LeakageError("no answers to score")
    grouped: dict[str, list[Answer]] = {}
    for answer in answers:
        grouped.setdefault(_bucket_label(answer.probe.published), []).append(answer)
    buckets = [
        Bucket(
            label=label,
            probes=len(rows),
            hits=sum(1 for r in rows if r.correct),
            chance=sum(r.probe.chance for r in rows) / len(rows),
        )
        for label, rows in sorted(grouped.items())
    ]
    return LeakageReport(
        model=model, answers=tuple(answers), buckets=tuple(buckets), kind=kind,
        control_hits=control_hits, control_probes=control_probes,
    )


def probes_from_filings(
    tickers: Sequence[str], *, concept: str = "revenue", kind: str = "tight",
    as_of: datetime | None = None, per_ticker: int = 12,
) -> list[Probe]:  # pragma: no cover - network
    """Probes built from what companies actually filed, via `market/fundamentals.py`.

    Point-in-time and restatement-resolved, so the answer is checkable against the primary source
    rather than against a dataset somebody assembled. The filing date travels with each probe
    because that, not the fiscal period, is when the fact could first have been learned.
    """
    from argus.market.fundamentals import FundamentalsSource

    source = FundamentalsSource()
    when = as_of or datetime.now(UTC)
    out: list[Probe] = []
    for ticker in tickers:
        try:
            facts, _notes = source.facts(ticker, concept=concept, as_of=when)
        except Exception:
            continue
        for fact in facts[:per_ticker]:
            value = float(fact.value) / 1e9
            if value <= 0 or fact.filed is None:
                continue
            label = (
                f"{concept.replace('_', ' ')} for the quarter ending "
                f"{fact.end.isoformat()}"
            )
            try:
                out.append(build_probe(
                    entity=ticker, label=label, published=fact.filed, value=value,
                    unit="USD billion", kind=kind,
                ))
            except LeakageError:
                continue
    return out


# --- the gate ------------------------------------------------------------------------------------


class Contamination(StrEnum):
    """What a stored leakage measurement says about one evaluation window."""

    CONTAMINATED = "contaminated"
    CLEAN = "clean"
    UNMEASURED = "unmeasured"
    """No usable measurement — which is **not** the same as clean, and must never be reported as it.

    An unrun instrument, a stored report whose sensitivity control failed, and a genuinely
    uncontaminated window all produce "no evidence of leakage". Only the last of those is a finding.
    """


@dataclass(frozen=True, slots=True)
class WindowCheck:
    """Whether a model may simply have read the answers inside an evaluation window."""

    status: Contamination
    start: date
    end: date
    boundary: str | None
    note: str

    @property
    def usable(self) -> bool:
        """True only for a window shown to sit outside the model's demonstrated recall."""
        return self.status is Contamination.CLEAN

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "window": [self.start.isoformat(), self.end.isoformat()],
            "boundary": self.boundary,
            "note": self.note,
        }


def check_window(
    start: date, end: date, *, path: Any = None,
) -> WindowCheck:
    """Does this evaluation window sit inside the region the model demonstrably recalls?

    Reads the stored measurement rather than spending tokens: the instrument is a scheduled
    measurement, and a gate that re-ran it on every call would make every backtest cost money.

    Three refusals, all of which report UNMEASURED rather than CLEAN:

    * no stored report at all;
    * a stored report whose **sensitivity control failed** — an instrument that never fires cannot
      certify anything, and this is the exact case that nearly shipped as a clean bill of health;
    * a report with no bucket above chance *and* no passing control.
    """
    target = path or REPORT_PATH
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return WindowCheck(
            status=Contamination.UNMEASURED, start=start, end=end, boundary=None,
            note=(
                "no memorisation measurement on record, so this window is UNMEASURED, not clean. "
                "Run `python -m argus.eval.leakage` before treating a historical result as evidence"
            ),
        )
    if not blob.get("instrument_is_sensitive"):
        return WindowCheck(
            status=Contamination.UNMEASURED, start=start, end=end,
            boundary=blob.get("boundary"),
            note=(
                "the stored measurement's sensitivity control failed, so the instrument has not "
                "shown it can detect recall at all. This window is UNMEASURED, not clean"
            ),
        )
    overlapping = [
        bucket["label"] for bucket in blob.get("buckets", [])
        if bucket.get("above_chance")
        and _bucket_range(str(bucket["label"]))[0] <= end
        and start <= _bucket_range(str(bucket["label"]))[1]
    ]
    if overlapping:
        return WindowCheck(
            status=Contamination.CONTAMINATED, start=start, end=end,
            boundary=blob.get("boundary"),
            note=(
                f"the model recalls facts published in {', '.join(overlapping)}, which this window "
                f"({start.isoformat()} to {end.isoformat()}) overlaps. A good result here may be "
                f"recall of the outcome rather than judgement about it"
            ),
        )
    return WindowCheck(
        status=Contamination.CLEAN, start=start, end=end, boundary=blob.get("boundary"),
        note=(
            f"no period the model demonstrably recalls overlaps {start.isoformat()} to "
            f"{end.isoformat()}; the instrument is sensitive, so it would have said otherwise"
        ),
    )



def replay(path: Path) -> tuple[str, list[str]]:
    """Recompute the verdict from a stored artefact, without asking the model anything.

    **Why this exists.** The verdict is *derived* — every number it quotes is already in the file.
    Re-probing the model to refresh a sentence would cost tokens, and worse, would replace the
    evidence rather than re-reading it: the answers are the measurement, and a measurement is not
    something to take again because the prose around it was wrong.

    **It also audits the file.** ``capacity``, ``p_value``, ``above_chance`` and ``counted`` are all
    derived from ``probes``, ``hits`` and ``chance``, so they are recomputed here and checked
    against what was stored. A mismatch means the artefact was edited or written by a different
    version, and it raises rather than quietly refreshing prose over numbers it could not confirm.

    Returns the refreshed verdict and the list of discrepancies found (empty on a clean replay).
    """
    blob: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []

    buckets: list[Bucket] = []
    for raw in blob.get("buckets", ()):
        bucket = Bucket(
            label=str(raw["label"]), probes=int(raw["probes"]),
            hits=int(raw["hits"]), chance=float(raw["chance"]),
        )
        for field_name, recomputed in (
            ("p_value", round(bucket.p_value, 6)),
            ("above_chance", bucket.above_chance),
            ("counted", bucket.probes >= MIN_PROBES_PER_BUCKET),
        ):
            if field_name in raw and raw[field_name] != recomputed:
                problems.append(
                    f"{bucket.label}.{field_name}: stored {raw[field_name]!r}, "
                    f"recomputes to {recomputed!r}"
                )
        buckets.append(bucket)

    if problems:
        raise LeakageError(
            "the stored artefact disagrees with its own arithmetic, so its prose will not be "
            "refreshed over numbers that could not be confirmed: " + "; ".join(problems)
        )

    raw_overall = blob["overall"]
    overall = Bucket(
        label="all", probes=int(raw_overall["probes"]), hits=int(raw_overall["hits"]),
        chance=float(raw_overall["chance"]),
    )
    above = [b for b in buckets if b.above_chance]
    control_probes = int(blob.get("control_probes", 0))
    return verdict_of(
        model=str(blob.get("model", "qwen")),
        kind=str(blob.get("kind", "tight")),
        overall=overall,
        buckets=buckets,
        boundary=above[-1].label if above else None,
        control_rate=(
            int(blob.get("control_hits", 0)) / control_probes if control_probes > 0 else None
        ),
        instrument_is_sensitive=bool(blob.get("instrument_is_sensitive", False)),
    ), problems


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="has the model already read the answer?")
    parser.add_argument("--tickers", default="NVDA,AAPL,MSFT,META,GOOGL,AMZN,TSLA")
    parser.add_argument("--kind", default="tight", choices=["wide", "tight"])
    parser.add_argument("--per-ticker", type=int, default=10)
    parser.add_argument(
        "--control", type=int, default=10,
        help="sensitivity-control probes: the same questions with the answer in the prompt",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build and print the probe set without spending a single model token",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="recompute the verdict from the stored artefact and write it back, without asking "
             "the model anything. Audits the file's own arithmetic first and refuses on a mismatch",
    )
    args = parser.parse_args()

    if args.replay:
        path = REPORT_PATH
        refreshed, _ = replay(path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        was = str(blob.get("verdict", ""))
        blob["verdict"] = refreshed
        blob["verdict_replayed_at"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print("arithmetic re-derived from the stored counts: consistent")
        print(f"\nWAS: {was[:220]}")
        print(f"\nNOW: {refreshed[:220]}")
        print(
            f"\nwritten to {path}" if was != refreshed
            else "\nthe stored verdict already matched; nothing written"
        )
        return 0

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    probes = probes_from_filings(tickers, kind=args.kind, per_ticker=args.per_ticker)
    if not probes:
        print("no probes could be built from filings")
        return 1
    print(f"{len(probes)} {args.kind} probe(s) built from SEC filings")
    if args.dry_run:
        for probe in probes[:3]:
            print()
            print(probe.render())
            print(f"  (correct: {probe.correct}, published {probe.published})")
        print(f"\ndry run: {len(probes)} probes, 0 model calls")
        return 0

    from argus.llm.qwen import QwenClient

    client = QwenClient()
    answers: list[Answer] = []
    for index, probe in enumerate(probes, start=1):
        try:
            reply = client.complete(
                [{"role": "user", "content": probe.render()}], temperature=0.0, max_tokens=4,
            )
        except Exception as exc:
            print(f"  probe {index}: {type(exc).__name__}")
            answers.append(Answer(probe=probe, replied=None, correct=False))
            continue
        letter = parse_letter(answer_text(reply))
        answers.append(Answer(probe=probe, replied=letter, correct=letter == probe.correct))

    # The sensitivity control, on a sample of the same probes. Without it a null result cannot be
    # distinguished from an instrument that never fires, which is the difference between a finding
    # and a blank page.
    control_probes = probes[:: max(1, len(probes) // args.control)][: args.control]
    control_hits = 0
    for probe in control_probes:
        try:
            reply = client.complete(
                [{"role": "user", "content": probe.render_control()}],
                temperature=0.0, max_tokens=4,
            )
        except Exception:
            continue
        if parse_letter(answer_text(reply)) == probe.correct:
            control_hits += 1

    report = score(
        answers, model=getattr(client, "model", "qwen"), kind=args.kind,
        control_hits=control_hits, control_probes=len(control_probes),
    )
    print()
    print(report.render())
    out = REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "LETTERS",
    "MIN_PROBES_PER_BUCKET",
    "REPORT_PATH",
    "SIGNIFICANCE",
    "TIGHT_MULTIPLIERS",
    "WIDE_MULTIPLIERS",
    "Answer",
    "Bucket",
    "Contamination",
    "LeakageError",
    "LeakageReport",
    "Probe",
    "WindowCheck",
    "answer_text",
    "build_probe",
    "check_window",
    "parse_letter",
    "probes_from_filings",
    "score",
]
