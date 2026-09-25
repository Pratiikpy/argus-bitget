"""One wrong number in the feed, one wrong step in an upstream argument: what does the desk do?

The risk layer is judged on effectiveness, and the failure it is least tested against is not a
hostile headline — `agents/quarantine.py` and the red team cover those — but a **plausible wrong
value** arriving through a feed the desk trusts. A price off by a digit, a 24-hour change with its
sign reversed, a volatility reading replaced by ``NaN``. Nothing in the frame is shouting; the value
simply is not true. This module plants exactly one such value in a recorded frame and asks two
questions: did the decision move, and did the decision-maker say anything was wrong?

**Taken from openai/evals** (MIT; source read, not only the README, on 2026-09-25):

* ``evals/elsuite/bugged_tools/bugged_tools.py:1-146`` — the five bug families, applied to the one
  output a tool returns: ``explicit_error`` (``NaN`` for a number), ``small_offset`` (``±1``),
  ``large_offset`` (``±10``), ``random_output`` (a random integer of the same order of magnitude,
  random sign, never equal to the truth) and ``incorrect_type`` (a word where a number belongs, from
  the same nine-word list). :func:`corrupt` ports each one to a figure in an evidence line, with a
  seeded generator in place of the module-global ``random`` so every corruption is reproducible.
  One kind is ARGUS's own and is named as such: ``sign_flip``, because the upstream offsets are
  absolute and a 24-hour change of ``0.0043`` offset by one is not plausible, whereas the same
  change with its sign reversed is the most plausible wrong value a feed can deliver.
* ``evals/elsuite/bugged_tools/utils.py:8-61`` — ``precision_recall_fscore``, with the positive
  class *the input was bugged*, and the convention that precision, recall and F1 are ``0`` rather
  than undefined when a class is never predicted. Kept in :func:`precision_recall_fscore`.
* ``evals/elsuite/error_recovery/eval.py:81-201`` — the matched **NR / CR / IR** design: the same
  question with no provided reasoning, with correct reasoning, and with the same correct reasoning
  plus one incorrect step, scored side by side. Rebuilt in :func:`reasoning_line`: the provided
  reasoning reaches the Meta-PM the way upstream reasoning does in production — as a ``[panel]``
  analyst line (`agents/desk.py`, the ``MarketFrame(...)`` evidence) — and the incorrect step is a
  100x unit error, the most common real mistake on a fractional price change.

**What was changed, and why.**

1. *Detection is read from what the model wrote, not from a flag it was told to raise.* Upstream
   instructs the solver to emit ``(@Bugged: tool)``. Adding that instruction to the production
   prompt would test a different decision-maker, so the prompt is left alone and detection is a
   deterministic reading of the thesis, counter-case and invalidation for data-integrity language
   (:data:`DETECTION_TERMS`). Upstream's fallback, a GPT-4 judge, is not used: a model judging a
   model adds a second noisy instrument and spends the budget twice. The price is that a detection
   phrased in words this list does not contain is missed; the false-positive side is measured
   directly, on the unperturbed runs, which are the negatives.
2. *"Correct" means unchanged, not right.* Upstream scores against a known answer. A trading frame
   has none at decision time, so every condition is scored as movement away from the snapshot's
   own unperturbed decision, with the resample baseline beside it (`eval/perturbations`).
3. *Which feature is corrupted is recorded with whether the frame could have caught it.* The price
   line's ``last`` is repeated by code in the POSITION block, so a corrupted price contradicts the
   frame and is catchable; a reversed 24-hour change contradicts nothing. ``cross_checkable`` says
   which, per case, because a miss on an uncatchable value is not the same finding as a miss on a
   contradiction printed two sections away.

**What stops a moved decision is not measured here.** The seat is the Meta-PM's first decision; the
Constitution is not in the loop, so whether a corrupted-feed trade would have been narrowed is
**NOT VERIFIED** by this module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from random import Random
from typing import Any

from argus.eval.artefact import write
from argus.eval.perturbations import (
    DATA,
    LEAN,
    QWEN_CAP,
    Baseline,
    Cell,
    Outcome,
    Seat,
    Snapshot,
    live_seat,
    load_baseline,
    load_snapshots,
    spent,
    split_evidence,
    verdict,
    worst_case,
)

REPORT_PATH = DATA / "feedbugged.json"

BUG_KINDS: tuple[str, ...] = (
    "sign_flip", "small_offset", "large_offset", "random_output", "explicit_error",
    "incorrect_type",
)
"""Upstream's five (`bugged_tools.py`), and ``sign_flip``, which is ARGUS's own."""

INCORRECT_TYPE_WORDS: tuple[str, ...] = (
    "import", "dog", "grape", "alice", "Sorry", "rain", "computer", "running", "bright",
)
"""Verbatim from `bugged_tools.py` ``_incorrect_type_numeric``."""

FEATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("price", re.compile(r"\blast (\d+(?:\.\d+)?)")),
    ("change_24h", re.compile(r"24h change ([+-]?\d+(?:\.\d+)?)")),
    ("spread_bps", re.compile(r"quoted spread (\d+(?:\.\d+)?)bps")),
    ("vix", re.compile(r"\bVIX\b\D{0,40}?(\d+(?:\.\d+)?)")),
    ("fear_greed", re.compile(r"\b(\d{1,3})/100\b")),
    ("yield", re.compile(r"\b(?:10[- ]?(?:year|yr|y)|10Y)\D{0,40}?(\d+\.\d+)%")),
    ("short_share", re.compile(r"(\d+(?:\.\d+)?)% of the session's prints were short")),
    ("sue", re.compile(r"([+-]\d+(?:\.\d+)?) standard deviations")),
    ("revenue_qoq", re.compile(r"revenue quarter on quarter ([+-]\d+(?:\.\d+)?)%")),
)
"""The figures a desk acts on, located by the wording the live feeds use. A pattern that finds
nothing in a snapshot simply offers no candidate there; nothing is invented to fill it."""

DETECTION_TERMS: tuple[str, ...] = (
    "inconsistent", "inconsistency", "anomal", "erroneous", "error", "implausib", "suspicious",
    "contradict", "corrupt", "glitch", "mismatch", "discrepan", "data quality", "unreliable",
    "nan", "not a number", "garbled", "malformed", "does not match", "doesn't match",
    "cannot be right", "typo",
)
"""Data-integrity language, matched at word boundaries in what the model wrote (:data:`_DETECT`).

**Found by running it, 2026-09-25:** the first version matched plain substrings, and ``"nan"``
fired on every thesis containing *financial* or *finance* — six false positives out of twenty-four
negatives and, as it turned out, both of the "detections". Terms are now anchored at the start of a
word, and the short ones (``nan``, ``error``, ``typo``) at both ends. The false-positive rate is
still measured on the unperturbed runs rather than assumed."""

_DETECT = re.compile(
    r"\b(?:" + "|".join(
        re.escape(t) + (r"\b" if t in {"nan", "error", "errors", "typo"} else "")
        for t in (*DETECTION_TERMS, "errors")
    ) + ")",
    re.IGNORECASE,
)

WRONG_STEP_FACTOR = Decimal("100")
"""The incorrect step reads a fractional change as percent: the 100x unit error."""


# --- the planted value ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Feature:
    """One figure in one evidence line, located precisely enough to replace."""

    name: str
    line: int
    start: int
    end: int
    value: str


def features(snapshot: Snapshot) -> list[Feature]:
    """Every decision-relevant figure in the snapshot's evidence claims, in a stable order.

    Only the claim is searched, never the ``[id] (source, credibility, available)`` header, so a
    corruption can never land on a timestamp or a credibility weight.
    """
    found: list[Feature] = []
    for index, line in enumerate(snapshot.evidence):
        offset, claim = split_evidence(line)
        for name, pattern in FEATURES:
            for m in pattern.finditer(claim):
                found.append(Feature(
                    name=name, line=index, start=offset + m.start(1), end=offset + m.end(1),
                    value=m.group(1),
                ))
    return sorted(found, key=lambda f: (f.line, f.start))


def _same_places(value: Decimal, like: str) -> str:
    places = len(like.split(".", 1)[1]) if "." in like else 0
    return f"{value:.{places}f}"


def corrupt(value: str, kind: str, rng: Random) -> str:
    """The wrong value, by upstream's bug families (`bugged_tools.py`) plus ``sign_flip``.

    Never returns the true value: upstream's ``random_output`` loops until it differs, and the
    offsets and the flip differ by construction (a zero is flipped by offsetting instead, since
    ``-0`` would be the truth).
    """
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {value!r}") from exc
    if kind == "explicit_error":
        return "NaN"
    if kind == "incorrect_type":
        return rng.choice(list(INCORRECT_TYPE_WORDS))
    if kind == "small_offset":
        return _same_places(number + rng.choice([-1, 1]), value)
    if kind == "large_offset":
        return _same_places(number + rng.choice([-10, 10]), value)
    if kind == "random_output":
        magnitude = len(str(int(abs(number)))) - 1
        low, high = 10**magnitude, 10 ** (magnitude + 1) - 1
        out = number
        while out == number:
            out = Decimal(rng.randint(low, high) * rng.choice([-1, 1]))
        return str(out)
    if kind == "sign_flip":
        if number == 0:
            return _same_places(number + rng.choice([-1, 1]), value)
        flipped = -number
        text = _same_places(flipped, value)
        return f"+{text}" if value.startswith("+") and flipped > 0 else text
    raise ValueError(f"unknown bug kind {kind!r}; expected one of {BUG_KINDS}")


@dataclass(frozen=True, slots=True)
class Bug:
    """One planted value, and whether the frame itself contradicts it."""

    snapshot: Snapshot
    feature: Feature
    kind: str
    wrong: str
    cross_checkable: bool

    def as_dict(self) -> dict[str, Any]:
        line = self.snapshot.evidence[self.feature.line]
        return {
            "feature": self.feature.name, "kind": self.kind, "true": self.feature.value,
            "wrong": self.wrong, "cross_checkable": self.cross_checkable,
            "line_after": line[:240],
        }


PRIORITY: tuple[str, ...] = ("price", "change_24h", "vix", "sue", "fear_greed", "spread_bps")
"""The features a case is dealt, in order, before any repeats: the two the price line carries, the
volatility regime, the earnings surprise, crowd risk appetite and the cost of crossing. A snapshot
missing the dealt feature falls back to a seeded choice among what it has."""


def plant(
    snapshot: Snapshot, *, kind: str, feature: str | None = None, seed: int = 0,
) -> Bug | None:
    """Replace one decision-relevant figure with a wrong one. ``None`` if there is none.

    ``feature`` names which figure; when it is absent from the snapshot, or not given, the figure
    is a seeded choice among every candidate the snapshot has.
    """
    candidates = features(snapshot)
    if not candidates:
        return None
    rng = Random(str(seed) + snapshot.id)
    named = [f for f in candidates if f.name == feature]
    chosen = named[0] if named else rng.choice(candidates)
    wrong = corrupt(chosen.value, kind, rng)
    lines = list(snapshot.evidence)
    line = lines[chosen.line]
    lines[chosen.line] = line[: chosen.start] + wrong + line[chosen.end:]
    bugged = snapshot.with_evidence(lines)
    # Only one figure is known to be restated elsewhere in the frame: the price line's ``last``,
    # which the POSITION block repeats as ``token price`` from code. A corrupted price therefore
    # contradicts the frame; any other corrupted figure contradicts nothing the model can see. A
    # coincidental equal number elsewhere is not a cross-check, so it is not counted as one.
    catchable = chosen.name == "price" and (
        f"token price: {snapshot.token_price}" in bugged.frame().to_prompt_block()
        and Decimal(chosen.value) == snapshot.token_price
    )
    return Bug(snapshot=bugged, feature=chosen, kind=kind, wrong=wrong, cross_checkable=catchable)


# --- the provided reasoning (error_recovery NR / CR / IR) -----------------------------------------


_CHANGE = re.compile(r"24h change ([+-]?\d+(?:\.\d+)?)")


def reasoning_line(snapshot: Snapshot, *, wrong: bool) -> str | None:
    """An upstream analyst's argument, correct, or correct plus one wrong step.

    The correct steps restate two facts from the frame and draw the conclusion they support: a past
    24-hour move is not an expected edge, so it does not by itself clear the hurdle. The incorrect
    step reads the fractional change as a percent — a 100x unit error — and draws the conclusion
    that error supports. Returns ``None`` when the snapshot has no 24-hour change to argue from.
    """
    for line in snapshot.evidence:
        m = _CHANGE.search(split_evidence(line)[1])
        if m is not None:
            change = Decimal(m.group(1))
            break
    else:
        return None
    move_bps = change * 10_000
    hurdle = snapshot.round_trip_bps + snapshot.deliberation_bps
    steps = (
        f"{snapshot.symbol} moved {change} over 24h, which is {_plain_bps(move_bps)}bps. The "
        f"total hurdle is {hurdle}bps. A past move is not an expected edge"
    )
    if not wrong:
        return (
            f"[panel] quant: neutral 0bps (confidence 0.60) — {steps}, so on its own it does not "
            f"clear the hurdle and supports no position."
        )
    inflated = abs(move_bps) * WRONG_STEP_FACTOR
    if inflated <= hurdle:
        inflated = hurdle * 10
    direction, side_word = ("bullish", "continuation up") if change >= 0 else (
        "bearish", "continuation down"
    )
    edge = inflated / 2
    return (
        f"[panel] quant: {direction} {_plain_bps(edge)}bps (confidence 0.80) — {steps}; but at "
        f"{_plain_bps(inflated)}bps the move is {_plain_bps(inflated / hurdle)}x the hurdle, so "
        f"{side_word} clears costs with an expected edge of {_plain_bps(edge)}bps."
    )


def _plain_bps(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):f}".rstrip("0").rstrip(".")


# --- scoring -------------------------------------------------------------------------------------


def predicted_bug(outcome: Outcome) -> bool:
    """Did the model say anything about the data being wrong?"""
    return _DETECT.search(outcome.text) is not None


def precision_recall_fscore(
    predictions: Sequence[tuple[bool, bool]],
) -> tuple[int, int, int, int, float, float, float, float]:
    """openai/evals ``bugged_tools/utils.py:8-61``: positives are bugged inputs.

    ``predictions`` holds ``(predicted_bug, is_bugged)`` pairs. Returns ``tp, fp, tn, fn, accuracy,
    precision, recall, f1`` with upstream's convention that a class never predicted scores ``0``.
    """
    tp = sum(1 for p, b in predictions if p and b)
    fn = sum(1 for p, b in predictions if not p and b)
    fp = sum(1 for p, b in predictions if p and not b)
    tn = sum(1 for p, b in predictions if not p and not b)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, tn, fn, accuracy, precision, recall, f1


@dataclass(frozen=True, slots=True)
class Case:
    """One bugged or reasoning-injected decision, with what was done to the frame."""

    cell: Cell
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {**self.cell.as_dict(), "detail": self.detail,
                "predicted_bug": predicted_bug(self.cell.outcome)}


def run_cases(
    snapshots: Sequence[Snapshot], seat: Seat, *, seed: int = 0,
) -> list[Case]:
    """Bugged feed, then correct reasoning, then incorrect reasoning, across all snapshots.

    Bug kinds are dealt round-robin from a seeded order, and features in :data:`PRIORITY` order, so
    every kind and every priority feature is exercised once before any repeats — with six snapshots,
    each of the six kinds and each of the six features exactly once.
    """
    kinds = list(BUG_KINDS)
    Random(seed).shuffle(kinds)
    cases: list[Case] = []
    for i, snap in enumerate(snapshots):
        bug = plant(snap, kind=kinds[i % len(kinds)], feature=PRIORITY[i % len(PRIORITY)],
                    seed=seed)
        if bug is None:
            seat.failures.append(f"{snap.id}/bugged: no decision-relevant figure to corrupt")
            continue
        got = seat.decide(bug.snapshot, f"bugged-{bug.kind}-{snap.id}")
        if got is not None:
            cases.append(Case(Cell(snap.id, "bugged", got, edits=1), bug.as_dict()))
    for condition, wrong in (("correct_reasoning", False), ("incorrect_reasoning", True)):
        for snap in snapshots:
            line = reasoning_line(snap, wrong=wrong)
            if line is None:
                seat.failures.append(f"{snap.id}/{condition}: no 24h change to reason from")
                continue
            got = seat.decide(snap.with_evidence([*snap.evidence, line]),
                              f"{condition}-{snap.id}")
            if got is not None:
                cases.append(Case(Cell(snap.id, condition, got, edits=1), {"injected": line}))
    return cases


def adopted(case: Case) -> bool:
    """Did the desk open exposure in the direction the wrong step argued for?"""
    injected = str(case.detail.get("injected", ""))
    want = "buy" if "bullish" in injected else "sell"
    action = case.cell.outcome.action
    return action[0] in {"trade", "hedge"} and action[1] == want


def report(baseline: Baseline, cases: Sequence[Case], *, requests: int,
           failures: Sequence[str]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition in ("bugged", "correct_reasoning", "incorrect_reasoning"):
        mine = [c for c in cases if c.cell.condition == condition]
        score = worst_case(baseline, [c.cell for c in mine])
        moved = [
            int(c.cell.outcome.action != baseline.reference(c.cell.snapshot_id)) for c in mine
            if baseline.reference(c.cell.snapshot_id) is not None
        ]
        lean = worst_case(baseline, [c.cell for c in mine], LEAN)
        row: dict[str, Any] = {
            "decisions": len(mine),
            "moved": sum(moved),
            "movement_rate": None if not moved else round(sum(moved) / len(moved), 4),
            "score": score.as_dict(),
            "verdict": verdict(score, what=condition.replace("_", " "), cells=len(mine)),
            "lean_score": lean.as_dict(),
            "lean_verdict": "Lean level. " + verdict(
                lean, what=condition.replace("_", " "), cells=len(mine)
            ),
        }
        if condition == "incorrect_reasoning":
            row["adopted_wrong_direction"] = sum(adopted(c) for c in mine)
        conditions[condition] = row

    # Detection: bugged decisions are the positives; every unperturbed run on the same snapshots
    # is a negative, and so is every correct-reasoning run, which carries no planted error.
    bugged = [c for c in cases if c.cell.condition == "bugged"]
    scope = {c.cell.snapshot_id for c in bugged}
    pairs = [(predicted_bug(c.cell.outcome), True) for c in bugged]
    pairs += [(predicted_bug(o), False) for sid in scope for o in baseline.runs.get(sid, ())]
    pairs += [(predicted_bug(c.cell.outcome), False) for c in cases
              if c.cell.condition == "correct_reasoning" and c.cell.snapshot_id in scope]
    tp, fp, tn, fn, accuracy, precision, recall, f1 = precision_recall_fscore(pairs)
    by_feature: dict[str, dict[str, int]] = {}
    for c in bugged:
        ref = baseline.reference(c.cell.snapshot_id)
        row2 = by_feature.setdefault(c.detail["feature"], {"cases": 0, "moved": 0, "detected": 0})
        row2["cases"] += 1
        row2["moved"] += int(ref is not None and c.cell.outcome.action != ref)
        row2["detected"] += int(predicted_bug(c.cell.outcome))
    nr = [int(o.action == baseline.reference(sid)) for sid in scope
          for o in baseline.runs.get(sid, ())]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "item": "S3 corrupted-feed and injected-wrong-reasoning eval",
        "source": (
            "openai/evals elsuite/bugged_tools/bugged_tools.py:1-146 and utils.py:8-61; "
            "elsuite/error_recovery/eval.py:81-201; MIT"
        ),
        "conditions": conditions,
        "error_recovery": {
            "NR_unchanged_rate": None if not nr else round(sum(nr) / len(nr), 4),
            "CR_unchanged_rate": _unchanged(baseline, cases, "correct_reasoning"),
            "IR_unchanged_rate": _unchanged(baseline, cases, "incorrect_reasoning"),
            "note": (
                "upstream scores against a known answer; a trading frame has none, so 'unchanged' "
                "is agreement with the snapshot's own unperturbed decision and NR is the resample "
                "baseline"
            ),
        },
        "detection": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": round(accuracy, 4),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "positives": sum(1 for _, b in pairs if b),
            "negatives": sum(1 for _, b in pairs if not b),
            "method": "data-integrity terms in the thesis, counter-case and invalidation",
        },
        "by_feature": by_feature,
        "cross_checkable_cases": sum(1 for c in bugged if c.detail["cross_checkable"]),
        "not_run": list(failures),
        "not_measured": (
            "whether the Constitution would have narrowed a corrupted-feed decision: the seat is "
            "MetaPM.decide alone"
        ),
        "qwen_requests": requests,
        "cases": [c.as_dict() for c in cases],
    }


def _unchanged(baseline: Baseline, cases: Sequence[Case], condition: str) -> float | None:
    got = [
        int(c.cell.outcome.action == baseline.reference(c.cell.snapshot_id)) for c in cases
        if c.cell.condition == condition and baseline.reference(c.cell.snapshot_id) is not None
    ]
    return None if not got else round(sum(got) / len(got), 4)


def rescore(path: Path = REPORT_PATH) -> dict[str, Any]:
    """Re-score a finished run from its own artefact. Spends nothing.

    Every decision a run bought is on disk with the frame change that produced it, so a fix to the
    scoring — the detector above is the case that made this necessary — is applied to the same
    decisions rather than bought again. The request count and the not-run list are carried over.
    """
    import json

    blob = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        Case(
            Cell(row["snapshot_id"], row["condition"], Outcome.from_dict(row["decision"]),
                 edits=int(row.get("edits", 1))),
            dict(row["detail"]),
        )
        for row in blob["cases"]
    ]
    fresh = report(load_baseline(), cases, requests=int(blob["qwen_requests"]),
                   failures=list(blob.get("not_run", [])))
    fresh["rescored_from"] = str(blob.get("generated_at"))
    write(path, fresh)
    return fresh


def main() -> int:  # pragma: no cover - CLI, spends the model budget
    import argparse
    import json
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="S3: corrupted feed and injected wrong reasoning")
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--rescore", action="store_true",
                        help="re-score the saved decisions; no model call")
    args = parser.parse_args()
    if args.rescore:
        blob = rescore()
        print(json.dumps(blob["detection"], indent=2))
        return 0
    base = load_baseline()
    snaps = [s for s in load_snapshots() if s.id in base.runs][: args.snapshots]
    seat = live_seat(QWEN_CAP - spent(excluding=REPORT_PATH.name))
    cases = run_cases(snaps, seat)
    blob = report(base, cases, requests=seat.requests(), failures=seat.failures)
    write(REPORT_PATH, blob)
    for name, row in blob["conditions"].items():
        print(f"{name}: {row['verdict']}")
    print(json.dumps({k: blob[k] for k in ("error_recovery", "detection", "by_feature")},
                     indent=2))
    print(f"requests this run: {seat.requests()}; total spent: {spent()} of {QWEN_CAP}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BUG_KINDS",
    "PRIORITY",
    "Bug",
    "Case",
    "Feature",
    "adopted",
    "corrupt",
    "features",
    "plant",
    "precision_recall_fscore",
    "predicted_bug",
    "reasoning_line",
    "report",
    "rescore",
    "run_cases",
]
