"""Row 33 ("Market sentiment"): does the desk's sentiment read carry information about the next
move, measured against the general tools that lead the sub-theme — not asserted.

``Activity/28_CAPABILITY_CLOSE_PLAN_2.md`` section 33 names the row's blocker precisely: *"the
desk has opened no position from a sentiment read, and the rivals that lead the sub-theme have not
been run on the same input."* This module is the smallest complete measurement that answers it —
plan item (c)(1)-(3) — on real production data rather than designed cases: the 129 live paper
cycles in ``data/desk_notes.jsonl`` that recorded their evidence (2026-09-26/27), joined to the
realised move each decision's abstention settled against in ``data/paper_ledger.jsonl``.

**Four arms, identical point-in-time social evidence, one join.**

* **A — ARGUS's production ``SentimentAnalyst``** (``agents/analysts.py``), called through this
  exact class, unmodified, with production settings (``Thinking.LOW``, ``temperature=0``, the real
  ``role``/``COST_PREAMBLE``/``SCHEMA_NOTE``/``STANDING_INSTRUCTION`` — nothing here re-implements
  the prompt, so a future prompt edit cannot silently drift out of sync with what this module
  claims to test). Fed the **entire** ``source == "social"`` evidence slice exactly as
  ``agents/desk.py:484`` selects it for the live desk — every real tweet/Reddit post *and* the
  Bitget Skill/derivatives/technical-analysis items the live feed also tags ``"social"`` — because
  that is the actual production input, and a comparison that trimmed it would not be testing the
  thing that runs live.
* **B — VADER crowd tone** (``market/social_pulse.py``'s vendored, unmodified scorer and the
  production coordination-collapse weighting), scored only over the twitter/reddit **post text**
  subset of that same evidence — the ``~2,200`` real posts the task names, since a tone lexicon
  scoring a Skill's RSI readout is not a reproduction of anything social_pulse actually does
  (its own pipeline only ever sees ``Post`` objects built from ``TwitterSource``/``RedditSource``
  evidence: ``market/social_pulse.py:227-238``).
* **C — finBERT** (``ProsusAI/finbert``, the real published model, real local inference), mean tone
  over the same twitter/reddit posts, a lean by a threshold fixed here, before any scoring ran.
* **D — no-information baselines**: always flat, and a seeded coin flip.

**A real, verified data-fidelity gap, and the honest choice made about it.** Arm B's production
weighting collapses a *coordinated* story (three or more distinct **accounts** posting
near-duplicate text within two hours) to one vote, splitting the vote across its members
(``market/social_pulse.py:138-154``). That needs the poster's real identity, and it is not
recoverable here: ``agents/desk.py:765`` builds the frame the live desk actually reasons over with
``tuple(e.render() for e in evidence)``, and ``Evidence.render()`` (``truth/evidence.py:55-59``)
emits only ``id``, ``source``, ``credibility``, ``available_at`` and ``claim`` — never
``attributes["author"]``. ``data/desk_notes.jsonl`` is written from exactly that rendered tuple
(``paper/runner.py:1216,747`` -> ``run.evidence`` -> ``desk.py:765``), so real author identity was
already gone before this archive existed; grepping the id that names it
(``twitter-2103666575265989094``) across every file in ``data/`` finds it nowhere but here. This
module substitutes each post's own evidence id as its "voice" (:func:`build_posts`), which
reproduces the production **collapse rule** exactly on text alone — three-or-more near-duplicate
posts inside the coordination window still count once, whoever posted them — but cannot detect one
real account repeating itself, nor tell two real accounts apart from one. That half of "the
production voice weighting" is **NOT VERIFIED** for per-account fidelity; the near-duplicate
collapse mechanism itself, which is the part `truth/novelty.cluster` and `voice_weights` actually
compute, is reproduced unmodified.

**Budget.** The owner approved at most 140 real Qwen calls / ~250k tokens for this row. A pilot
measurement on three real cycles (drawn from the actual usable set, smallest/median/largest by
evidence size) found ~3,700-6,230 prompt tokens and ~1,070-1,500 completion tokens per call —
because the production prompt carries the desk's real evidence load, not a designed one-liner —
averaging **~6,560 tokens/call**, which makes the full 89-cycle join unaffordable (~584k tokens) and
is why :func:`select_sample` exists: a deterministic, evenly-spaced subsample sized to the budget,
never a cherry-picked one. ``main`` tracks real spend against ``--max-tokens``/``--max-calls`` and
stops before either is exceeded, saving whatever completed.

    python -m argus.eval.sentiment_signal_comparison --run --max-calls 130 --max-tokens 220000
    python -m argus.eval.sentiment_signal_comparison            # offline: re-scores a saved report
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from argus.agents.analysts import COST_PREAMBLE, SCHEMA_NOTE, SentimentAnalyst
from argus.agents.quarantine import STANDING_INSTRUCTION, render_for_prompt
from argus.eval.baselines.finbert_loader import FinbertClassifier
from argus.eval.compare import PairedBootstrap, paired_bootstrap, sign_test
from argus.llm.base import ChatModel
from argus.llm.qwen import Thinking
from argus.market import vader
from argus.market.social_pulse import Post, voice_weights
from argus.truth import artefact
from argus.truth.evidence import Evidence
from argus.truth.novelty import cluster
from argus.truth.paths import DATA_DIR

DESK_NOTES_PATH = DATA_DIR / "desk_notes.jsonl"
PAPER_LEDGER_PATH = DATA_DIR / "paper_ledger.jsonl"
REPORT_PATH = DATA_DIR / "sentiment_signal_comparison.json"

ROUND_TRIP_BPS = 12.0
"""The measured Bitget taker round trip. Same value and role as `agents/earnings.ROUND_TRIP_BPS`
and `eval/abstention_coverage.ROUND_TRIP_BPS` — not redefined differently here."""

FINBERT_LEAN_THRESHOLD = 0.05
"""Fixed **before** any finBERT scoring ran in this module (chosen when this file was written, not
tuned against a result). Mirrors VADER's own published boundary (`market/vader.POSITIVE_AT`) on the
same [-1, 1] signed scale, for the two text-tone arms to be read the same way."""

COIN_FLIP_SEED = 20260929
"""Today's date, the same seeding convention `eval/abstention_coverage.BOOTSTRAP_SEED` uses: fixed
so the "baseline" arm is reproducible and was not selected after seeing a result."""

BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260929

PILOT_TOKENS_ALREADY_SPENT = 19_673
"""Three real calibration calls (seq 760, 761, 775 — smallest/median/largest usable cycle by
social-evidence size) spent before this module existed, against the SAME approved 140-call/250k-
token ceiling for this row. Recorded here, not hidden, so a reader summing this module's own
`--max-tokens` argument against the 250k ceiling does not undercount the true total."""


class JoinError(RuntimeError):
    """The desk-notes/ledger join cannot proceed honestly."""


# =================================================================================================
# Evidence parsing — desk_notes.jsonl stores Evidence.render() strings, not structured records.
# =================================================================================================

_EVIDENCE_LINE = re.compile(
    r"^\[(?P<id>[^\]]+)\] \((?P<source>[a-z-]+), credibility (?P<cred>[0-9.]+), "
    r"available (?P<avail>[^)]+)\) (?P<claim>.*)$",
    re.DOTALL,
)
"""The exact inverse of `truth.evidence.Evidence.render()`: ``f"[{id}] ({source}, credibility "
"{cred:.2f}, available {available_at.isoformat()}) {claim}"``."""

SOCIAL_SOURCE = "social"
_SOCIAL_POST_PREFIXES = ("twitter-", "reddit-")
"""Real posts, as opposed to the Bitget Skill/derivatives/technical-analysis items the live feed
also tags ``source == "social"`` (ids like ``skill-technical_analysis-rsi-...`` and
``mirror-derivatives_sentiment-...``). Arm A reads all of `SOCIAL_SOURCE`; arms B and C read only
this subset — see the module docstring."""


def parse_evidence_line(line: str) -> Evidence:
    """One `Evidence.render()` line, parsed back into an `Evidence`. Raises on anything else."""
    m = _EVIDENCE_LINE.match(line)
    if not m:
        raise JoinError(f"not a rendered Evidence line: {line[:80]!r}")
    return Evidence(
        id=m.group("id"), claim=m.group("claim"), source=m.group("source"),
        available_at=datetime.fromisoformat(m.group("avail")), credibility=float(m.group("cred")),
    )


def is_social_post_id(evidence_id: str) -> bool:
    return evidence_id.startswith(_SOCIAL_POST_PREFIXES)


# =================================================================================================
# The join: desk_notes.jsonl evidence x paper_ledger.jsonl realised outcome.
# =================================================================================================


@dataclass(frozen=True, slots=True)
class Cycle:
    """One live paper cycle: the point-in-time social evidence it was given, and the realised move
    its (abstention) decision settled against."""

    seq: int
    symbol: str
    decided_at: str
    settled_at: str
    entry_price: float
    exit_price: float
    move_bps: float
    """`paper_ledger.Entry.counterfactual_move_bps`: ``(exit - entry) / entry * 10000``. Positive
    means the token rose over the horizon."""
    horizon_hours: float
    verdict: str
    social_evidence: tuple[Evidence, ...]
    """Every ``source == "social"`` item, exactly as `agents/desk.py:484` selects the real desk's
    sentiment-analyst input. Arm A's whole input."""
    posts: tuple[Evidence, ...]
    """The twitter/reddit subset of `social_evidence` — arms B and C's input."""

    @property
    def date(self) -> str:
        """The UTC calendar day, the cluster unit `eval.compare.paired_bootstrap` resamples —
        the same choice `eval.abstention_coverage.LeanCall.cluster` makes, for the same reason:
        cycles on one day share one market regime."""
        return self.decided_at[:10]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "symbol": self.symbol, "decided_at": self.decided_at,
            "settled_at": self.settled_at, "date": self.date,
            "entry_price": self.entry_price, "exit_price": self.exit_price,
            "move_bps": self.move_bps, "horizon_hours": round(self.horizon_hours, 3),
            "verdict": self.verdict, "n_social_evidence": len(self.social_evidence),
            "n_posts": len(self.posts),
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise JoinError(f"missing input: {path}")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cycles(
    desk_notes_path: Path = DESK_NOTES_PATH, ledger_path: Path = PAPER_LEDGER_PATH,
) -> list[Cycle]:
    """Every cycle that can honestly be scored: desk-notes evidence recorded, ledger settled with a
    realised `counterfactual_move_bps`, and at least one real social post among its evidence.

    Excludes ``"trade"`` rows (real positions settle through ``net_pnl``, not
    ``counterfactual_move_bps`` — see `paper/ledger.py:_settled_position` vs `_settled_abstention`)
    and any row still unsettled, matching what `eval.abstention_coverage.calls_from_entries` treats
    as ``unsettled``.
    """
    notes_by_seq = {row["seq"]: row for row in _read_jsonl(desk_notes_path) if "evidence" in row}
    ledger_by_seq = {
        row["seq"]: row for row in _read_jsonl(ledger_path)
        if row.get("kind", "decision") == "decision"
    }

    cycles: list[Cycle] = []
    for seq, ledger_row in sorted(ledger_by_seq.items()):
        if ledger_row.get("counterfactual_move_bps") is None:
            continue  # unsettled, or a real trade (settles through net_pnl instead)
        notes_row = notes_by_seq.get(seq)
        if notes_row is None:
            continue  # no recorded evidence for this cycle

        social = tuple(
            parse_evidence_line(line) for line in notes_row["evidence"]
            if (m := _EVIDENCE_LINE.match(line)) and m.group("source") == SOCIAL_SOURCE
        )
        posts = tuple(e for e in social if is_social_post_id(e.id))
        if not posts:
            continue  # nothing for arms B/C to score; not a comparable cycle

        entry_price = float(ledger_row["entry_price"])
        exit_price = float(ledger_row["exit_price"])
        decided_at = datetime.fromisoformat(ledger_row["decided_at"])
        settled_at = datetime.fromisoformat(ledger_row["settled_at"])
        cycles.append(Cycle(
            seq=seq, symbol=str(ledger_row["symbol"]), decided_at=ledger_row["decided_at"],
            settled_at=ledger_row["settled_at"], entry_price=entry_price, exit_price=exit_price,
            move_bps=float(ledger_row["counterfactual_move_bps"]),
            horizon_hours=(settled_at - decided_at).total_seconds() / 3600.0,
            verdict=str(ledger_row["verdict"]), social_evidence=social, posts=posts,
        ))
    return cycles


def select_sample(cycles: Sequence[Cycle], n: int) -> list[Cycle]:
    """An evenly-spaced, deterministic subsample of ``n`` cycles spanning the full range.

    Not a random draw and not "the first n": `cycles` is already sorted by `seq`, which is
    chronological, so the first n would hit only the earliest dates and under-represent later
    symbols in the rotation. Evenly-spaced indices spread across the whole join instead — the same
    kind of systematic, non-cherry-picked choice `eval.observatory`'s train/test split documents
    (`Activity/28_CAPABILITY_CLOSE_PLAN_2.md` explicitly asks that a selection never be tuned
    against a result). ``n >= len(cycles)`` returns everything.
    """
    total = len(cycles)
    if n >= total:
        return list(cycles)
    if n <= 1:
        return [cycles[0]] if cycles else []
    indices = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})
    return [cycles[i] for i in indices]


# =================================================================================================
# Arm result — one shape for all four arms.
# =================================================================================================


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm's read on one cycle: a lean in {-1, 0, +1} and a confidence in [0, 1]."""

    lean: int
    confidence: float
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lean not in (-1, 0, 1):
            raise ValueError(f"lean must be -1, 0 or 1, got {self.lean}")

    def as_dict(self) -> dict[str, Any]:
        return {"lean": self.lean, "confidence": round(self.confidence, 4), "raw": dict(self.raw)}


def net_bps_of(result: ArmResult, move_bps: float) -> float:
    """What trading this lean for the cycle's horizon would have netted, after the round trip.

    Zero for a flat lean — no trade, no cost, no P&L — matching
    `eval.abstention_coverage.LeanCall.net_bps`'s convention for a settled refusal.
    """
    if result.lean == 0:
        return 0.0
    return result.lean * move_bps - ROUND_TRIP_BPS


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


# =================================================================================================
# Arm A — ARGUS's production SentimentAnalyst, unmodified.
# =================================================================================================


def argus_prompt_hash(cycle: Cycle) -> str:
    """SHA-256 of the exact system+user text `SentimentAnalyst._ask` sends for this cycle.

    Built from the real, imported prompt pieces (`SentimentAnalyst.role`, `COST_PREAMBLE`,
    `SCHEMA_NOTE`, `STANDING_INSTRUCTION`, `render_for_prompt`) rather than a copied string, so a
    future edit to the production prompt changes this hash rather than silently going unnoticed —
    the opposite of `eval.sentiment_comparison._COORDINATION_INSTRUCTION`'s pinned literal, which
    needs manual re-pinning because that module deliberately ablates the real prompt. This one
    calls `SentimentAnalyst` unmodified, so there is nothing to keep in sync by hand.
    """
    system = (
        f"{SentimentAnalyst.role}\n\n{COST_PREAMBLE}\n\n{SCHEMA_NOTE}\n\n{STANDING_INSTRUCTION}"
    )
    body = (
        f"SYMBOL: {cycle.symbol}\nSOCIAL VOLUME (z-score vs 30d): +0.00\n\n"
        f"NARRATIVES:\n{render_for_prompt(list(cycle.social_evidence))}"
    )
    return hashlib.sha256(f"{system}\x1f{body}".encode()).hexdigest()[:16]


_SIGNAL_LEAN = {"bullish": 1, "bearish": -1, "neutral": 0, "insufficient_evidence": 0}


def run_argus_arm(client: ChatModel, cycle: Cycle) -> ArmResult:
    """One real call through the production `SentimentAnalyst`, on the production input.

    ``social_volume_z`` is left at its default (0.0): the live desk calls
    ``self.sentiment.analyse(symbol, social_evidence)`` with no z-score at this call site either
    (`agents/desk.py:517`), so leaving it out here matches production rather than adding an input
    the real decision path never had.
    """
    analyst = SentimentAnalyst(client, thinking=Thinking.LOW)
    view = analyst.analyse(cycle.symbol, list(cycle.social_evidence))
    return ArmResult(
        lean=_SIGNAL_LEAN.get(view.signal, 0), confidence=view.confidence,
        raw={"view": view.as_dict(), "prompt_hash": argus_prompt_hash(cycle)},
    )


# =================================================================================================
# Arm B — VADER crowd tone, production weighting (see the module docstring's fidelity note).
# =================================================================================================


def build_posts(cycle: Cycle) -> list[Post]:
    """The cycle's real posts as `social_pulse.Post` objects, ready for `truth.novelty.cluster`
    and `social_pulse.voice_weights`.

    ``source`` (the field those two functions read as the poster's identity) is set to the post's
    own evidence id rather than a real handle — real authorship is not retained in
    `data/desk_notes.jsonl`; see the module docstring's data-fidelity note. Every post is therefore
    its own "voice" unless near-duplicate text places it in a >=3-member cluster inside the 2h
    coordination window, in which case `voice_weights` still collapses it exactly as production
    does — the mechanism under test, reproduced on the one signal (text) this archive kept.
    """
    return [
        Post(id=e.id, claim=e.claim, source=e.id, available_at=e.available_at, channel="social")
        for e in cycle.posts
    ]


def run_vader_arm(cycle: Cycle, analyzer: vader.Analyzer) -> ArmResult:
    """The voice-weighted mean VADER compound over the cycle's posts, leaned by the production
    thresholds (`vader.label`). Reproduces `market.social_pulse._tone`'s aggregation
    (`market/social_pulse.py:168-206`) without importing that private function: the same
    `truth.novelty.cluster` -> `social_pulse.voice_weights` -> per-post weighted compound sum.
    """
    posts = build_posts(cycle)
    report = cluster(posts)
    weights = voice_weights(report)
    weighted_sum = 0.0
    voices = 0.0
    per_post = []
    for post in posts:
        compound = float(analyzer.polarity_scores(post.claim)["compound"])
        weight = weights.get(post.id, 1.0)
        weighted_sum += weight * compound
        voices += weight
        per_post.append({"id": post.id, "compound": round(compound, 4), "weight": round(weight, 4)})
    mean_compound = weighted_sum / voices if voices else 0.0
    label = vader.label(mean_compound)
    lean = {"positive": 1, "negative": -1, "neutral": 0}[label]
    confidence = min(1.0, abs(mean_compound))
    return ArmResult(
        lean=lean, confidence=confidence,
        raw={
            "mean_compound": round(mean_compound, 4), "label": label,
            "posts_scored": len(posts), "voices": round(voices, 4),
            "distinct_stories": report.distinct_stories,
            "coordinated_stories": sum(1 for c in report.clusters if c.coordinated),
            "per_post": per_post,
        },
    )


# =================================================================================================
# Arm C — finBERT, mean tone, a threshold fixed before scoring.
# =================================================================================================


def run_finbert_arm(
    classifier: FinbertClassifier, cycle: Cycle, *, threshold: float = FINBERT_LEAN_THRESHOLD,
) -> ArmResult:
    """finBERT's real per-post label+score, signed (positive: +score, negative: -score, neutral:
    0) and averaged over every post — not deduplicated, per the task's "mean tone over the same
    posts". `threshold` is decided once, in :data:`FINBERT_LEAN_THRESHOLD`, before any cycle here
    was scored."""
    texts = [post.claim for post in cycle.posts]
    results = classifier(texts)
    signed = [
        float(r["score"]) if r["label"] == "positive"
        else -float(r["score"]) if r["label"] == "negative" else 0.0
        for r in results
    ]
    mean_signed = statistics.mean(signed) if signed else 0.0
    lean = 1 if mean_signed >= threshold else (-1 if mean_signed <= -threshold else 0)
    confidence = min(1.0, abs(mean_signed))
    return ArmResult(
        lean=lean, confidence=confidence,
        raw={
            "mean_signed_score": round(mean_signed, 4), "threshold": threshold,
            "posts_scored": len(texts),
            "per_post": [
                {"label": r["label"], "score": round(float(r["score"]), 4)} for r in results
            ],
        },
    )


# =================================================================================================
# Arm D — no-information baselines.
# =================================================================================================


def run_flat_arm(_cycle: Cycle) -> ArmResult:
    return ArmResult(lean=0, confidence=0.0, raw={"policy": "always flat"})


def run_coin_flip_arms(
    cycles: Sequence[Cycle], *, seed: int = COIN_FLIP_SEED,
) -> dict[int, ArmResult]:
    """One seeded coin flip per cycle, in cycle (chronological) order — drawn once, not re-drawn,
    so the sequence is reproducible from `seed` alone."""
    rng = random.Random(seed)
    out = {}
    for cyc in cycles:
        lean = rng.choice((-1, 1))
        out[cyc.seq] = ArmResult(lean=lean, confidence=0.5, raw={"policy": "seeded coin flip",
                                                                  "seed": seed})
    return out


# =================================================================================================
# Scoring — per-arm metrics, honest about n.
# =================================================================================================


@dataclass(frozen=True, slots=True)
class ArmScore:
    arm: str
    n_cycles: int
    n_leaned: int
    hit_eligible: int
    """Leaned, and |move| exceeds the round trip — the only cycles a directional call could have
    profited on even with perfect timing."""
    hits: int
    hit_rate: float | None
    total_net_bps: float
    mean_net_bps_per_trade: float | None
    spearman_rho: float | None
    spearman_p: float | None
    spearman_n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "n_cycles": self.n_cycles, "n_leaned": self.n_leaned,
            "hit_eligible": self.hit_eligible, "hits": self.hits,
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 4),
            "total_net_bps": round(self.total_net_bps, 4),
            "mean_net_bps_per_trade": (
                None if self.mean_net_bps_per_trade is None
                else round(self.mean_net_bps_per_trade, 4)
            ),
            "spearman_rho": None if self.spearman_rho is None else round(self.spearman_rho, 4),
            "spearman_p": None if self.spearman_p is None else round(self.spearman_p, 6),
            "spearman_n": self.spearman_n,
        }


def score_arm(arm: str, cycles: Sequence[Cycle], results: Mapping[int, ArmResult]) -> ArmScore:
    """Directional hit rate (where the arm leans and the move clears the round trip), net bps of
    trading every lean, and Spearman rank correlation of lean x confidence against the move —
    exactly the three measures the row asks for, each with its own honest n."""
    rows = [(cyc, results[cyc.seq]) for cyc in cycles if cyc.seq in results]
    n_leaned = sum(1 for _, r in rows if r.lean != 0)
    eligible = [(cyc, r) for cyc, r in rows if r.lean != 0 and abs(cyc.move_bps) > ROUND_TRIP_BPS]
    hits = sum(1 for cyc, r in eligible if r.lean == _sign(cyc.move_bps))
    hit_rate = hits / len(eligible) if eligible else None

    net = [net_bps_of(r, cyc.move_bps) for cyc, r in rows if r.lean != 0]
    total_net = sum(net)
    mean_net = total_net / len(net) if net else None

    x = [r.lean * r.confidence for _cyc, r in rows]
    y = [cyc.move_bps for cyc, _r in rows]
    rho: float | None = None
    p_value: float | None = None
    if len(rows) >= 3 and len(set(x)) > 1 and len(set(y)) > 1:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        rho = float(result.statistic)
        p_value = float(result.pvalue)

    return ArmScore(
        arm=arm, n_cycles=len(rows), n_leaned=n_leaned, hit_eligible=len(eligible), hits=hits,
        hit_rate=hit_rate, total_net_bps=total_net, mean_net_bps_per_trade=mean_net,
        spearman_rho=rho, spearman_p=p_value, spearman_n=len(rows),
    )


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """A vs one rival: exact McNemar on directional correctness where both leaned, and a paired
    bootstrap on per-cycle net bps, clustered by date."""

    a_arm: str
    b_arm: str
    n_both_leaned: int
    a_correct_b_wrong: int
    b_correct_a_wrong: int
    both_correct: int
    both_wrong: int
    mcnemar_p: float | None
    bootstrap: PairedBootstrap | None
    n_dates: int
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "a_arm": self.a_arm, "b_arm": self.b_arm, "n_both_leaned": self.n_both_leaned,
            "a_correct_b_wrong": self.a_correct_b_wrong,
            "b_correct_a_wrong": self.b_correct_a_wrong,
            "both_correct": self.both_correct, "both_wrong": self.both_wrong,
            "mcnemar_p": None if self.mcnemar_p is None else round(self.mcnemar_p, 6),
            "bootstrap": None if self.bootstrap is None else {
                "mean_diff_net_bps": round(self.bootstrap.mean_diff, 4),
                "ci95": [round(self.bootstrap.low, 4), round(self.bootstrap.high, 4)],
                "units_dates": self.bootstrap.units, "draws": self.bootstrap.draws,
                "seed": self.bootstrap.seed,
            },
            "n_dates": self.n_dates, "note": self.note,
        }


def compare_arms(
    a_name: str, b_name: str, cycles: Sequence[Cycle],
    a_results: Mapping[int, ArmResult], b_results: Mapping[int, ArmResult],
) -> PairedComparison:
    """McNemar's exact test on discordant directional calls where both arms leaned
    (`eval.compare.sign_test` computes the identical two-sided exact binomial statistic
    `eval.sentiment_dedup_rival.exact_mcnemar` does — the same formula under a different name), and
    a bootstrap over per-cycle net bps clustered by UTC date (`eval.compare.paired_bootstrap`).
    """
    shared = [
        cyc for cyc in cycles
        if cyc.seq in a_results and cyc.seq in b_results
        and a_results[cyc.seq].lean != 0 and b_results[cyc.seq].lean != 0
    ]
    a_right_b_wrong = 0
    b_right_a_wrong = 0
    both_right = 0
    both_wrong = 0
    for cyc in shared:
        move_sign = _sign(cyc.move_bps)
        a_correct = a_results[cyc.seq].lean == move_sign
        b_correct = b_results[cyc.seq].lean == move_sign
        if a_correct and b_correct:
            both_right += 1
        elif a_correct and not b_correct:
            a_right_b_wrong += 1
        elif b_correct and not a_correct:
            b_right_a_wrong += 1
        else:
            both_wrong += 1
    mcnemar_p = (
        sign_test(a_right_b_wrong, b_right_a_wrong)
        if (a_right_b_wrong + b_right_a_wrong) > 0 else None
    )

    diffs_by_date: dict[str, list[float]] = {}
    for cyc in cycles:
        if cyc.seq not in a_results or cyc.seq not in b_results:
            continue
        diff = net_bps_of(a_results[cyc.seq], cyc.move_bps) - net_bps_of(
            b_results[cyc.seq], cyc.move_bps
        )
        diffs_by_date.setdefault(cyc.date, []).append(diff)
    n_dates = len(diffs_by_date)
    boot: PairedBootstrap | None = None
    if diffs_by_date and any(v for v in diffs_by_date.values()):
        boot = paired_bootstrap(diffs_by_date, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)

    note = f"{len(shared)} cycle(s) where both {a_name} and {b_name} leaned"
    if n_dates < 5:
        note += (
            f"; only {n_dates} distinct UTC date(s) in the sample, so the date-clustered bootstrap "
            f"interval is NOT a valid confidence interval here (resampling {n_dates} clusters "
            f"cannot approximate a sampling distribution) — read it as descriptive only"
        )
    if len(shared) < 30:
        note += f"; n={len(shared)} is small, so this comparison is reported as not separable below"

    return PairedComparison(
        a_arm=a_name, b_arm=b_name, n_both_leaned=len(shared),
        a_correct_b_wrong=a_right_b_wrong, b_correct_a_wrong=b_right_a_wrong,
        both_correct=both_right, both_wrong=both_wrong, mcnemar_p=mcnemar_p, bootstrap=boot,
        n_dates=n_dates, note=note,
    )


# =================================================================================================
# Report assembly and the CLI.
# =================================================================================================

MEASURED_TOKENS_PER_CALL = 6_558.0
"""Mean of the three real pilot calls (`PILOT_TOKENS_ALREADY_SPENT`): 4,777 / 7,275 / 7,621 total
tokens on the smallest/median/largest usable cycle by evidence size. Used only to *size* the
sample before spending; the real stopping decision during the run is the live `TokenBudget`, never
this estimate."""

ARMS: tuple[str, ...] = ("argus", "vader", "finbert", "flat", "coin_flip")


def _auto_sample_size(max_tokens: int, n_cycles: int) -> int:
    """How many cycles arm A can afford at the measured per-call rate, with a 5% margin, capped at
    the number of cycles actually available."""
    affordable = int(max_tokens / (MEASURED_TOKENS_PER_CALL * 1.05))
    return max(1, min(affordable, n_cycles))


def _load_saved_argus(path: Path) -> dict[int, ArmResult]:
    """Arm A responses a previous ``--run`` already saved to ``path``. Offline mode (the default)
    reuses these rather than re-spending; returns empty when no report exists yet."""
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, ArmResult] = {}
    for row in blob.get("cycles", []):
        a = (row.get("arms") or {}).get("argus")
        if a is None:
            continue
        out[int(row["seq"])] = ArmResult(
            lean=int(a["lean"]), confidence=float(a["confidence"]), raw=dict(a.get("raw") or {}),
        )
    return out


def _verdict_for(cmp: PairedComparison) -> str:
    """ARGUS ahead / tied / behind / not separable — read off the bootstrap interval, corroborated
    by McNemar, never off either alone. `PairedComparison.note` already states the caveats this
    verdict must be read alongside (small n, too few date clusters)."""
    if cmp.n_both_leaned < 10:
        return f"not separable: only {cmp.n_both_leaned} cycle(s) where both arms leaned"
    if cmp.bootstrap is None:
        return "not separable: no net-bps difference could be bootstrapped"
    lo, hi = cmp.bootstrap.low, cmp.bootstrap.high
    if lo > 0:
        return f"{cmp.a_arm} ahead: net-bps 95% interval [{lo:.2f}, {hi:.2f}] entirely above zero"
    if hi < 0:
        return f"{cmp.b_arm} ahead: net-bps 95% interval [{lo:.2f}, {hi:.2f}] entirely below zero"
    return f"not separable: net-bps 95% interval [{lo:.2f}, {hi:.2f}] spans zero"


def build_report(
    cycles: Sequence[Cycle], results_by_arm: Mapping[str, Mapping[int, ArmResult]], *,
    calls_made: int, tokens_spent: int, sample_seqs: Sequence[int],
) -> dict[str, Any]:
    """Everything the row's owner needs: the join, every arm's per-cycle lean and raw response,
    the three scores per arm, the two paired comparisons, and the honest verdict."""
    scores = {
        arm: score_arm(arm, cycles, res).as_dict() for arm, res in results_by_arm.items()
    }
    vs_vader = compare_arms(
        "argus", "vader", cycles, results_by_arm["argus"], results_by_arm["vader"],
    )
    vs_finbert = compare_arms(
        "argus", "finbert", cycles, results_by_arm["argus"], results_by_arm["finbert"],
    )
    rows = []
    for cyc in cycles:
        arms_here = {
            arm: res[cyc.seq].as_dict() for arm, res in results_by_arm.items() if cyc.seq in res
        }
        rows.append({**cyc.as_dict(), "arms": arms_here})

    n_argus = len(results_by_arm["argus"])
    dates = sorted({c.date for c in cycles})
    return {
        "capability_row": 33,
        "question": (
            "does the desk's sentiment read carry information about the next move, against the "
            "general tools that lead the sub-theme?"
        ),
        "join": {
            "usable_cycles": len(cycles),
            "argus_scored_cycles": n_argus,
            "argus_sample_seqs": list(sample_seqs),
            "dates_covered": dates,
            "symbols_covered": sorted({c.symbol for c in cycles}),
            "horizon_hours_mean": (
                round(statistics.mean(c.horizon_hours for c in cycles), 3) if cycles else None
            ),
            "horizon_note": (
                "production HOLD_HOURS=24 (paper/runner.py); actual settlement lag varies with "
                "the abstention-mark cron cadence, reported per cycle"
            ),
            "round_trip_bps": ROUND_TRIP_BPS,
        },
        "qwen_spend": {
            "calls_this_run": calls_made, "tokens_this_run": tokens_spent,
            "pilot_calls_already_spent": 3 if calls_made or tokens_spent else 0,
            "pilot_tokens_already_spent": PILOT_TOKENS_ALREADY_SPENT,
            "note": (
                "pilot figures are the three real calibration calls made before this module "
                "existed, against the same 140-call/250k-token ceiling for this row"
            ),
        },
        "cycles": rows,
        "scores": scores,
        "comparisons": {
            "argus_vs_vader": {**vs_vader.as_dict(), "verdict": _verdict_for(vs_vader)},
            "argus_vs_finbert": {**vs_finbert.as_dict(), "verdict": _verdict_for(vs_finbert)},
        },
        "scope_statement": (
            "Claimed: on real live paper-trading cycles (2026-09-26/27, N given in `join`), "
            "scored against the realised ~24h-horizon move net of the 12bps round trip, whichever "
            "of ARGUS's production sentiment analyst, VADER crowd tone and finBERT reads the "
            "sentiment channel better on the same point-in-time evidence. NOT claimed: a result "
            "for cycles this run's token budget did not reach (see `join.argus_scored_cycles` vs "
            "`join.usable_cycles`); anything about real-money P&L (paper prices only); exact "
            "per-account fidelity for arm B's coordination weighting (see the module docstring)."
        ),
    }


def _load_qwen_env(path: Path) -> None:
    """``KEY=VALUE`` lines into ``os.environ``, without ever printing or logging a value.

    Only used when the caller passes ``--qwen-env`` explicitly (see ``main``'s argparse help): no
    path to a secrets file is hard-coded in this committed module, matching the convention
    `eval/sentiment_comparison.py:main` already uses (it just checks the environment and tells the
    caller to load the file first). Never reads ``.secrets/bitget.env``.
    """
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Row 33: ARGUS sentiment analyst vs VADER/finBERT/no-info, on real paper cycles"
        ),
    )
    parser.add_argument("--run", action="store_true",
                        help="spend real Qwen calls for arm A; default reuses a saved report")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="cycles to run arm A on; 0 = auto-size from --max-tokens")
    parser.add_argument("--max-calls", type=int, default=140)
    parser.add_argument("--max-tokens", type=int, default=250_000)
    parser.add_argument(
        "--qwen-env", type=Path, default=None,
        help=(
            "optional path to a KEY=VALUE env file to load before --run (e.g. .secrets/qwen.env); "
            "default: BITGET_QWEN_API_KEY/BASE_URL must already be in the environment"
        ),
    )
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    import os

    if args.run:
        if args.qwen_env is not None:
            _load_qwen_env(args.qwen_env)
        if not os.environ.get("BITGET_QWEN_API_KEY"):
            print(
                "BITGET_QWEN_API_KEY not set; load .secrets/qwen.env into the environment first "
                "(or pass --qwen-env), matching eval/sentiment_comparison.py's own convention."
            )
            return 1

    cycles = load_cycles()
    if not cycles:
        print("no usable cycles joined (desk_notes.jsonl x paper_ledger.jsonl); nothing to score")
        return 1
    print(f"joined {len(cycles)} usable cycle(s)")

    analyzer = vader.load_analyzer()
    vader_results = {c.seq: run_vader_arm(c, analyzer) for c in cycles}

    from argus.eval.baselines.finbert_loader import load_finbert

    finbert_results = {c.seq: run_finbert_arm(load_finbert(), c) for c in cycles}
    flat_results = {c.seq: run_flat_arm(c) for c in cycles}
    coin_results = run_coin_flip_arms(cycles)

    calls_made = 0
    tokens_spent = 0
    sample_seqs: list[int] = []
    if args.run:
        from argus.llm.qwen import QwenClient, TokenBudget

        n = args.sample_size or _auto_sample_size(args.max_tokens, len(cycles))
        sample = select_sample(cycles, n)
        sample_seqs = [c.seq for c in sample]
        print(f"arm A sample: {len(sample)} of {len(cycles)} cycle(s), evenly spaced")

        from argus.llm.qwen import QwenError

        budget = TokenBudget(limit=args.max_tokens)
        client = QwenClient(budget=budget)
        argus_results: dict[int, ArmResult] = {}
        failures: list[dict[str, Any]] = []
        safety_margin = 9_000  # above the largest single-call spend seen in the pilot (7,621)
        results_by_arm_partial: dict[str, Mapping[int, ArmResult]] = {
            "argus": argus_results, "vader": vader_results, "finbert": finbert_results,
            "flat": flat_results, "coin_flip": coin_results,
        }
        for cyc in sample:
            if client.calls >= args.max_calls:
                print(f"stopping: reached --max-calls={args.max_calls}")
                break
            if budget.spent + safety_margin > args.max_tokens:
                print(
                    f"stopping to honour --max-tokens={args.max_tokens}: {budget.spent} spent, "
                    f"next call could push past it"
                )
                break
            # A single cycle's call failing (a schema the model could not satisfy in 3 attempts,
            # a transient transport error QwenClient's own retries did not clear) must not lose
            # every other cycle's already-spent budget and already-scored results — so it is
            # recorded and skipped, not raised.
            try:
                result = run_argus_arm(client, cyc)
            except QwenError as exc:
                failures.append({"seq": cyc.seq, "symbol": cyc.symbol, "error": str(exc)[:300]})
                print(f"  seq {cyc.seq} {cyc.symbol}: FAILED — {str(exc)[:150]}")
                continue
            argus_results[cyc.seq] = result
            print(
                f"  seq {cyc.seq} {cyc.symbol}: lean={result.lean:+d} "
                f"conf={result.confidence:.2f}  spent={budget.spent} calls={client.calls}"
            )
            # Saved after every real call, not only at the end: a killed process (the 10-minute
            # tool timeout, a network drop) must not lose spend that is already real and already
            # logged in `data/qwen_cost_ledger.jsonl`.
            interim = build_report(
                cycles, results_by_arm_partial, calls_made=client.calls, tokens_spent=budget.spent,
                sample_seqs=[c.seq for c in sample],
            )
            interim["argus_failures"] = failures
            artefact.write(args.out, interim)
        calls_made = client.calls
        tokens_spent = budget.spent
        print(f"arm A: {len(argus_results)} cycle(s) scored, {len(failures)} failed, "
              f"{calls_made} call(s), {tokens_spent} tokens this run")
    else:
        argus_results = _load_saved_argus(args.out)
        sample_seqs = sorted(argus_results)
        failures = []
        print(f"offline: reusing {len(argus_results)} previously saved arm-A response(s) "
              f"from {args.out}")

    results_by_arm: dict[str, Mapping[int, ArmResult]] = {
        "argus": argus_results, "vader": vader_results, "finbert": finbert_results,
        "flat": flat_results, "coin_flip": coin_results,
    }
    report = build_report(
        cycles, results_by_arm, calls_made=calls_made, tokens_spent=tokens_spent,
        sample_seqs=sample_seqs,
    )
    report["argus_failures"] = failures
    undefined = artefact.write(args.out, report)
    if undefined:
        print(f"note: {len(undefined)} non-finite field(s) written as null, e.g. {undefined[:5]}")
    print(json.dumps(report["comparisons"], indent=2, default=str))
    print(f"saved -> {args.out}")
    return 0


__all__ = [
    "ARMS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "COIN_FLIP_SEED",
    "DESK_NOTES_PATH",
    "FINBERT_LEAN_THRESHOLD",
    "PAPER_LEDGER_PATH",
    "PILOT_TOKENS_ALREADY_SPENT",
    "REPORT_PATH",
    "ROUND_TRIP_BPS",
    "ArmResult",
    "ArmScore",
    "Cycle",
    "JoinError",
    "PairedComparison",
    "argus_prompt_hash",
    "build_posts",
    "build_report",
    "compare_arms",
    "is_social_post_id",
    "load_cycles",
    "main",
    "net_bps_of",
    "parse_evidence_line",
    "run_argus_arm",
    "run_coin_flip_arms",
    "run_finbert_arm",
    "run_flat_arm",
    "run_vader_arm",
    "score_arm",
    "select_sample",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
