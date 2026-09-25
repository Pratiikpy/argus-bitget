"""S23 decision-quality primitives, each measured on the record rather than asserted.

Six primitives landed together on 2026-09-25 (``research/mypr-teardowns/_SYNTHESIS.md`` harvest,
item S23 of ``Activity/17_HARVEST_CHECKLIST.md``). None of them is allowed to claim an improvement
on the strength of its mechanism. This module measures each against the record that already exists
and writes ``data/decision_primitives.json`` — ties, losses and "cannot be measured on this record"
included, each with the reason.

* **Partially supported** (`agents/grounding.py`, `agents/claims.py`, after Self-RAG). Replays the
  grounding verdict of every decision in ``data/desk_notes.jsonl``: how many binary failures are
  partial at thesis level (some figures traced, some did not — exact, from the counts the desk
  recorded), how many unresolved figures are near misses of a value the desk demonstrably had (a
  *lower bound*: the desk never persisted its full set of known values, only a reconstructable
  subset), how often the near-miss band catches a figure *by chance* (each thesis paired with
  another decision's values), and whether partial theses differ in realised outcome from fully
  supported and unsupported ones. The recorded thesis-quality frames, where every known value *is*
  on disk, give a complete rerun on a small sample; and the support lines the live desk has printed
  since the state shipped are counted, which is the complete reading for those decisions, with a
  check that no failing decision since lacks one. For claims, the replay of the conviction rule
  that found six false contradictions, and a plain statement that partial claims cannot be
  recomputed because the evidence attributes were never persisted.
* **m-of-K agreement** (`decision/verdicts.py`, after GoEmotions). The analyst panels' individual
  signals were not persisted either; they are reconstructed from the conflict lines and the panel
  line by enumerating every assignment of signals consistent with them, and a panel is scored at a
  threshold only where every consistent assignment gives the same call. Decisiveness and hit rate
  against the settled move, at every threshold, beside the desk's own plurality and the base rate.
* **The rubric pair** (`agents/rubric.py`, after Self-Refine). No rubric has been asked for yet
  (flag off, no paid calls permitted for this item), so what *is* measured is the baseline it
  replaces: how much the single stated confidence says about each axis the desk measures on its own.
  The pair's own agreement is computed from the desk's ``[rubric]`` pair lines by the same function,
  and reads zero lines until the flag is switched on.
* **Prompt cache** (`llm/cache.py`, after Tree of Thoughts). The recorded thesis-quality run is
  replayed through the cache — every request in the order it was made — for the within-run hit
  rate, then again from the persisted cache for the cross-run rate; the live cost ledger gives the
  same figure on real desk traffic.
* **ID remap** (`llm/idmap.py`, after mem0). Exposure on the recorded frames, and the statement that
  the mangling rate itself is not on the record.
* **Forced reflection** (`agents/meta_pm.py`, after page-agent) needs model output to measure and
  none was permitted; it is reported as not measured, with the tests that cover its mechanics named.

**Qwen calls: none.** Every model answer used here is read from
``data/thesis_quality_recording.json`` through `eval/thesis_quality.py`'s offline client, which
raises rather than calls on a miss, and the replay's cost-ledger writes go to memory so the real
ledger is not polluted by replayed answers.

    python -m argus.eval.decision_primitives
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from argus.agents import claims as claims_mod
from argus.agents.grounding import (
    NEAR_MISS_UNITS,
    PARTIAL_TOLERANCE,
    TOLERANCE,
    Figure,
    NearMiss,
    Support,
    _candidates,
    _close,
    _distance,
    check,
    extract,
    nearest,
)
from argus.agents.meta_pm import SYSTEM_PROMPT, W_SUPPORT, MetaPM, given_values_with_units
from argus.agents.rubric import NAMES as RUBRIC_AXES
from argus.agents.rubric import agreement, meets_bar, parse_pair_note
from argus.decision.verdicts import DEFAULT_AGREEMENT, direction_of, m_of_k
from argus.desk.review import LOW_INDEPENDENCE, NARROW_SOURCES, panel_stats
from argus.eval.artefact import write as write_artefact
from argus.eval.shadow import DEAD_ZONE_BPS, move_of
from argus.llm.cache import PromptCache
from argus.llm.idmap import IdMap
from argus.llm.ledger import CostLedger
from argus.llm.qwen import Completion, Thinking

DATA = Path(__file__).resolve().parents[3] / "data"
NOTES_PATH = DATA / "desk_notes.jsonl"
LEDGER_PATH = DATA / "paper_ledger.jsonl"
CHAINS_PATH = DATA / "causal_chains.jsonl"
COST_LEDGER_PATH = DATA / "qwen_cost_ledger.jsonl"
REPORT_PATH = DATA / "decision_primitives.json"

NULL_SEEDS = 50
"""Shuffled pairings used for the near-miss chance rate."""

SIGNALS = ("bullish", "bearish", "neutral", "insufficient_evidence")
"""Every value `agents/analysts.py`'s ``_parse_view`` can give a view's signal."""

MAX_ENUMERATED_ANALYSTS = 6
"""Panels above this are skipped in the m-of-K reconstruction rather than enumerated (4^6 = 4,096
assignments is the most any recorded panel needs; the record's largest panel is 4)."""

OLD_CONVICTION = re.compile(
    r"(?:high[- ]conviction|conviction\s+(?:buy|purchase)|strong\s+insider\s+signal)", re.I
)
"""The conviction rule's pattern before 2026-09-25, kept here so the replay can compare the two."""


# --- reading the record ---------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            blob = json.loads(line)
        except ValueError:
            continue
        if isinstance(blob, dict):
            rows.append(blob)
    return rows


def decisions_by_seq(path: Path = LEDGER_PATH) -> dict[int, Any]:
    """Ledger decisions with settlement folded in, keyed by seq (the join `desk/review.py` uses)."""
    from argus.paper.ledger import PaperLedger

    out: dict[int, Any] = {}
    for entry in PaperLedger(path=path).entries:
        if str(getattr(entry, "kind", "decision")) == "decision":
            out[int(entry.seq)] = entry
    return out


def realised_direction(entry: Any) -> tuple[str | None, float | None]:
    """``("up" | "down" | None, move)`` from the settled move; ``None`` inside the dead zone or
    unsettled — the same grading `desk/review.py`'s ``defects_from_leans`` applies."""
    moved = move_of(entry) if entry is not None else None
    if moved is None:
        return None, None
    move = moved[0]
    if abs(move) <= DEAD_ZONE_BPS:
        return None, move
    return ("up" if move > 0 else "down"), move


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion; ``None`` with no trials."""
    if n == 0:
        return None
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the table [[a, b], [c, d]] (sum of tables no likelier)."""
    row1, col1, total = a + b, a + c, a + b + c + d

    def prob(x: int) -> float:
        return (math.comb(col1, x) * math.comb(total - col1, row1 - x)) / math.comb(total, row1)

    low, high = max(0, row1 - (total - col1)), min(row1, col1)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(low, high + 1) if prob(x) <= observed * 1.0000001))


# --- 1. partially supported: grounding on the live record -----------------------------------------

_GROUNDING_FAIL = re.compile(
    r"^\[grounding\]\s*(?P<bad>\d+) of (?P<total>\d+) figure\(s\) do not resolve[^:]*:"
    r"\s*(?P<names>.*)$"
)
_GROUNDING_PASS = re.compile(r"^\[grounding\]\s*all (?P<total>\d+) figure\(s\) resolve")


@dataclass(frozen=True)
class GroundingRow:
    """One recorded decision's grounding verdict, as the desk wrote it."""

    seq: int
    figures: int
    unresolved: int
    named: tuple[str, ...]
    """Unresolved figures the note names — sorted, de-duplicated, at most six (`render`'s cap)."""

    @property
    def level(self) -> Support:
        """Thesis-level support from the recorded counts alone. A failure with at least one figure
        that resolved is partial *exactly*; one where none resolved is none unless a near miss
        lifts it, which this reading cannot see."""
        if self.unresolved == 0:
            return Support.FULL
        return Support.PARTIAL if self.unresolved < self.figures else Support.NONE


def grounding_rows(notes: Sequence[dict[str, Any]]) -> list[GroundingRow]:
    out: list[GroundingRow] = []
    for row in notes:
        for note in row.get("notes", []) or []:
            text = str(note)
            fail = _GROUNDING_FAIL.match(text)
            if fail is not None:
                names = tuple(n.strip() for n in fail.group("names").split(",") if n.strip())
                out.append(GroundingRow(int(row.get("seq", 0)), int(fail.group("total")),
                                        int(fail.group("bad")), names))
                break
            ok = _GROUNDING_PASS.match(text)
            if ok is not None:
                out.append(GroundingRow(int(row.get("seq", 0)), int(ok.group("total")), 0, ()))
                break
    return out


def reconstructed_known(row: dict[str, Any], entry: Any,
                        chain: dict[str, Any] | None) -> list[tuple[str, float, str | None]]:
    """The part of the desk's known-value set that the record lets us rebuild *exactly*.

    Each value here was provably in the set `agents/desk.py` passed to the checker: the 12bps round
    trip; the token price and hours to discovery (both hashed into the decision); the analyst and
    distinct-source counts (printed on the panel line from the same objects); and the figures of the
    one evidence claim a causal chain recorded as its event. Everything else the desk knew — the
    hurdle, the per-analyst magnitudes, every other evidence item — was never persisted, so a near
    miss found against this subset is a lower bound on near misses, never an estimate of them.

    Each value carries the unit the desk declared for it — the fact's name suffix, or the unit the
    evidence claim was written in — because `agents/grounding.py` measures a near miss only between
    a figure and a value in the same unit. Unit-less facts carry ``None``.
    """
    known: list[tuple[str, float, str | None]] = [("round_trip_bps", 12.0, "bps")]
    if entry is not None:
        for name, attr in (("token_price", "entry_price"),
                           ("hours_to_discovery", "hours_to_discovery")):
            value = getattr(entry, attr, None)
            try:
                if value is not None:
                    known.append((name, float(value), None))
            except (TypeError, ValueError):
                continue
    stats = panel_stats(row)
    if stats:
        known.append(("analysts", stats["analysts"], None))
        known.append(("distinct_sources", stats["sources"], None))
    if chain is not None:
        known.extend(("chain_event", f.value, f.unit)
                     for f in extract(str(chain.get("event", ""))))
    return known


def _resolves(figure: Figure, known: Sequence[tuple[str, float, str | None]]) -> bool:
    written = _candidates(figure.value, figure.unit)
    return any(_close(c, v) for _, v, _unit in known for c in written)


def _named_figures(row: GroundingRow) -> list[Figure]:
    out: list[Figure] = []
    for name in row.named:
        found = extract(name)
        if found:
            out.append(found[0])
    return out


Matcher = Callable[[Figure, Sequence[tuple[str, float, str | None]]], NearMiss | None]


def quantity_blind_nearest(
    figure: Figure, known: Sequence[tuple[str, float, str | None]]
) -> NearMiss | None:
    """The first design of the near miss, kept only so its measured failure stays reproducible:
    every known value, through every unit conversion resolution uses, units ignored."""
    best: NearMiss | None = None
    for name, value, _unit in known:
        distance = min(_distance(c, value) for c in _candidates(figure.value, figure.unit))
        if distance <= PARTIAL_TOLERANCE and (best is None or distance < best.distance):
            best = NearMiss(source=name, known_value=value, distance=distance)
    return best


def _near_miss_run(
    pairs: Sequence[tuple[GroundingRow, Figure]],
    known_by_seq: dict[int, list[tuple[str, float, str | None]]],
    matcher: Matcher,
) -> dict[str, Any]:
    """Near misses under one matcher, and the same figures against a shuffled partner's values."""
    near = 0
    lifted: set[int] = set()
    examples: list[dict[str, Any]] = []
    for r, figure in pairs:
        hit = matcher(figure, known_by_seq[r.seq])
        if hit is None:
            continue
        near += 1
        if r.level is Support.NONE:
            lifted.add(r.seq)
        if len(examples) < 12:
            examples.append({"seq": r.seq, "figure": figure.raw, "near": hit.source,
                             "known_value": hit.known_value, "distance": round(hit.distance, 4)})
    seqs = sorted(known_by_seq)
    null_rates: list[float] = []
    for seed in range(NULL_SEEDS):
        rng = random.Random(20260925 + seed)
        shuffled = seqs[:]
        rng.shuffle(shuffled)
        partner = {s: shuffled[(i + 1) % len(shuffled)] for i, s in enumerate(seqs)}
        chance = sum(1 for r, fig in pairs
                     if not _resolves(fig, known_by_seq[partner[r.seq]])
                     and matcher(fig, known_by_seq[partner[r.seq]]) is not None)
        null_rates.append(chance / len(pairs) if pairs else 0.0)
    null_mean = sum(null_rates) / len(null_rates) if null_rates else None
    return {
        "near_misses_found": near,
        "near_miss_rate": round(near / len(pairs), 4) if pairs else None,
        "chance_rate_shuffled_pairing": {
            "seeds": NULL_SEEDS,
            "mean": None if null_mean is None else round(null_mean, 4),
            "max": round(max(null_rates), 4) if null_rates else None,
        },
        "no_support_theses_lifted_to_partial": len(lifted),
        "examples": examples,
    }


def near_miss_lower_bound(
    rows: Sequence[GroundingRow], notes_by_seq: dict[int, dict[str, Any]],
    ledger: dict[int, Any], chains: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Named unresolved figures that come within the band of a value the desk had, plus a null.

    Measured twice: under the shipped, unit-strict rule, and under the quantity-blind rule it
    replaced — the second is the measurement that rejected it, kept so it can be re-run.
    """
    failing = [r for r in rows if r.unresolved]
    known_by_seq = {r.seq: reconstructed_known(notes_by_seq.get(r.seq, {}), ledger.get(r.seq),
                                               chains.get(r.seq)) for r in failing}
    conflicts = 0
    pairs: list[tuple[GroundingRow, Figure]] = []
    for r in failing:
        for figure in _named_figures(r):
            if _resolves(figure, known_by_seq[r.seq]):
                # The original checker, holding a superset of these values, left this unresolved;
                # a subset cannot resolve it unless the rules moved since. Excluded, and counted.
                conflicts += 1
                continue
            pairs.append((r, figure))
    return {
        "band": PARTIAL_TOLERANCE,
        "resolution_tolerance": TOLERANCE,
        "named_unresolved_figures_examined": len(pairs),
        "written_in_a_near_miss_unit": sum(1 for _, f in pairs if f.unit in NEAR_MISS_UNITS),
        "excluded_as_reconstruction_conflicts": conflicts,
        "unit_strict": _near_miss_run(pairs, known_by_seq, nearest),
        "quantity_blind_rejected": _near_miss_run(pairs, known_by_seq, quantity_blind_nearest),
        "known_values_reconstructed_per_decision": {
            "min": min((len(v) for v in known_by_seq.values()), default=0),
            "max": max((len(v) for v in known_by_seq.values()), default=0),
            "declared_in_a_near_miss_unit": sum(
                1 for v in known_by_seq.values() for _, _, u in v if u in NEAR_MISS_UNITS
            ),
        },
        "reading": (
            "a lower bound: only values the record proves the desk held are used, and the "
            "hurdle — the figure theses misquote most — is not among them"
        ),
    }


def outcome_by_level(rows: Sequence[GroundingRow], ledger: dict[int, Any]) -> dict[str, Any]:
    """Lean hit rate and move size, by thesis-level support. Settled decisions only."""
    table: dict[str, dict[str, Any]] = {}
    tallies: dict[Support, tuple[int, int]] = {}
    for level in Support:
        members = [r for r in rows if r.level is level]
        graded = hits = settled = 0
        moves: list[float] = []
        verdicts: Counter[str] = Counter()
        for r in members:
            entry = ledger.get(r.seq)
            if entry is None:
                continue
            verdicts[str(getattr(entry, "verdict", ""))] += 1
            direction, move = realised_direction(entry)
            if move is None:
                continue
            settled += 1
            moves.append(abs(move))
            lean = str(getattr(entry, "lean", "none"))
            if direction is not None and lean in ("up", "down"):
                graded += 1
                hits += lean == direction
        tallies[level] = (hits, graded)
        table[str(level)] = {
            "decisions": len(members),
            "settled": settled,
            "lean_graded": graded,
            "lean_hits": hits,
            "lean_hit_rate": round(hits / graded, 4) if graded else None,
            "lean_hit_rate_95ci": wilson(hits, graded),
            "mean_abs_move_bps": round(sum(moves) / len(moves), 3) if moves else None,
            "verdicts": dict(verdicts),
        }

    def compare(a: Support, b: Support) -> dict[str, Any]:
        (ha, na), (hb, nb) = tallies[a], tallies[b]
        if not na or not nb:
            return {"p_value": None, "why": "one side has no graded lean"}
        return {"p_value": round(fisher_two_sided(ha, na - ha, hb, nb - hb), 4),
                "difference": round(ha / na - hb / nb, 4)}

    return {
        "by_level": table,
        "partial_vs_none": compare(Support.PARTIAL, Support.NONE),
        "partial_vs_full": compare(Support.PARTIAL, Support.FULL),
        "grading": (
            f"lean against the settled move (eval/shadow.move_of), moves within "
            f"{DEAD_ZONE_BPS}bps not graded; two-sided Fisher exact on hits"
        ),
    }


def grounding_on_record(notes: Sequence[dict[str, Any]], ledger: dict[int, Any],
                        chains: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows = grounding_rows(notes)
    by_seq = {int(r.get("seq", 0)): r for r in notes}
    fails = [r for r in rows if r.unresolved]
    partial = [r for r in fails if r.level is Support.PARTIAL]
    figures_total = sum(r.figures for r in rows)
    return {
        "decisions_with_a_grounding_note": len(rows),
        "binary_pass": sum(1 for r in rows if not r.unresolved),
        "binary_fail": len(fails),
        "binary_fail_that_is_partial_exactly": len(partial),
        "binary_fail_with_no_figure_traced": len(fails) - len(partial),
        "figures_checked": figures_total,
        "figures_unresolved": sum(r.unresolved for r in rows),
        "support_score_mean_lower_bound": (
            round(sum((r.figures - r.unresolved) for r in rows) / figures_total, 4)
            if figures_total else None
        ),
        "near_miss": near_miss_lower_bound(rows, by_seq, ledger, chains),
        "outcome": outcome_by_level(rows, ledger),
    }


# --- 1b. partially supported: complete rerun on the recorded frames -------------------------------


class _ReplayClient:
    """Builds the offline recording client, cost ledger in memory. Imported lazily: the thesis
    quality module is only needed for the replays."""

    @staticmethod
    def make(recording: dict[str, dict[str, Any]], cache: PromptCache | None = None) -> Any:
        from argus.eval import thesis_quality as tq

        class Replay(tq.CappedRecordingClient):
            def __init__(self) -> None:
                super().__init__(recording, live=False)
                self.ledger = CostLedger(None)
                if cache is not None:
                    self._cache = None  # measure the prompt cache alone, not the client's own
                self.prompt_cache = cache

            def complete(
                self, messages: list[dict[str, Any]], *, temperature: float = 0.0,
                max_tokens: int = 2048, json_mode: bool = False,
                tools: list[dict[str, Any]] | None = None, seed: int | None = None,
                thinking: Thinking = Thinking.LOW, stream: bool | None = None,
            ) -> Completion:
                def ask() -> Completion:
                    return tq.CappedRecordingClient.complete(
                        self, messages, temperature=temperature, max_tokens=max_tokens,
                        json_mode=json_mode, tools=tools, seed=seed, thinking=thinking,
                        stream=stream,
                    )

                if self.prompt_cache is None:
                    return ask()
                return self.prompt_cache.complete(
                    ask, messages, namespace="thesis-quality-recording",
                    temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                    tools=tools, seed=seed, thinking=thinking, stream=stream,
                )

        return Replay()


def recorded_frames_rerun() -> dict[str, Any]:
    """Every recorded Meta-PM answer re-checked with the complete set of values it was given."""
    from argus.eval import thesis_quality as tq

    frames = tq.load_frames()
    client = _ReplayClient.make(tq.load_recording())
    pm = MetaPM(client, max_tokens=tq.PM_MAX_TOKENS, thinking=tq.PM_THINKING, candidates=2)
    answers: list[dict[str, Any]] = []
    ranking_changes = 0
    ranked_pairs = 0
    stopped = ""
    for rf in frames:
        frame = rf.market_frame()
        try:
            deliberation = pm.deliberate(frame)
        except Exception as exc:  # the recording ends where the original run stopped
            stopped = f"{rf.symbol} {rf.at.isoformat()}: {str(exc)[:100]}"
            break
        checks = [(a.index, a.check) for a in deliberation.attempts if a.check is not None]
        if len(checks) == 2:
            ranked_pairs += 1
            by_coverage = min(checks, key=lambda ic: (len(ic[1].defects), -ic[1].score, ic[0]))
            # The same score with the support term read three-state instead of as coverage.
            by_support = min(checks, key=lambda ic: (
                len(ic[1].defects),
                -(ic[1].score + W_SUPPORT * (ic[1].grounding_support - ic[1].grounding_coverage)),
                ic[0]))
            ranking_changes += by_coverage[0] != by_support[0]
        for attempt in deliberation.attempts:
            if attempt.response is None or attempt.check is None:
                continue
            facts, values, units = given_values_with_units(frame, response=attempt.response)
            report = check(str(attempt.response.get("thesis", "")), facts=facts,
                           evidence_values=values, evidence_units=units)
            lean = str(attempt.response.get("lean", "none")).lower()
            realised = rf.realised_bps
            answers.append({
                "symbol": rf.symbol, "at": rf.at.isoformat(), "candidate": attempt.index,
                "support": str(report.support), "figures": len(report.resolutions),
                "unresolved": len(report.unresolved), "near_misses": [
                    {"figure": r.figure.raw, "near": r.near.source,
                     "known_value": r.near.known_value, "distance": round(r.near.distance, 4)}
                    for r in report.near_misses if r.near is not None
                ],
                "coverage": round(report.coverage, 4),
                "support_score": round(report.support_score, 4),
                "realised_bps": realised,
                "lean_hit": (None if lean not in ("up", "down") or abs(realised) <= DEAD_ZONE_BPS
                             else (lean == "up") == (realised > 0)),
            })
    levels = Counter(a["support"] for a in answers)
    return {
        "frames_replayed": len({(a["symbol"], a["at"]) for a in answers}),
        "answers_checked": len(answers),
        "by_level": dict(levels),
        "near_misses": sum(len(a["near_misses"]) for a in answers),
        "unresolved_figures": sum(a["unresolved"] for a in answers),
        "rank_pairs": ranked_pairs,
        "rank_changes_if_scored_by_support_instead_of_coverage": ranking_changes,
        "answers": answers,
        "stopped": stopped,
        "reading": (
            "complete known values, tiny sample: the recording holds the five frames the original "
            "thesis-quality run reached before its network failed"
        ),
    }


# --- 1c. claims -----------------------------------------------------------------------------------


def claims_on_record(notes: Sequence[dict[str, Any]], ledger: dict[int, Any]) -> dict[str, Any]:
    conviction = next(r for r in claims_mod.RULES if r.name == "conviction")
    examined = contradictions = 0
    by_rule: Counter[str] = Counter()
    replay: list[dict[str, Any]] = []
    for row in notes:
        for note in row.get("notes", []) or []:
            text = str(note)
            if not text.startswith("[claim]"):
                continue
            if "agree with" in text:
                examined += 1
            elif "is contradicted by" in text:
                contradictions += 1
                rule = text[len("[claim] "):].split(":", 1)[0]
                by_rule[rule] += 1
                snippet = text.split('"')[1] if text.count('"') >= 2 else ""
                replay.append({
                    "seq": int(row.get("seq", 0)), "rule": rule,
                    "old_conviction_pattern_fires": bool(OLD_CONVICTION.search(snippet)),
                    "new_conviction_pattern_fires": bool(conviction.regex.search(snippet)),
                })
    theses = [str(getattr(e, "thesis", "")) for e in ledger.values()]
    old_hits = sum(1 for t in theses if OLD_CONVICTION.search(t))
    new_hits = sum(1 for t in theses if conviction.regex.search(t))
    conviction_rows = [r for r in replay if r["rule"] == "conviction"]
    return {
        "sound_with_a_claim_examined": examined,
        "contradictions_recorded": contradictions,
        "contradictions_by_rule": dict(by_rule),
        "conviction_rule_replay": {
            "recorded_conviction_contradictions": len(conviction_rows),
            "still_fire_under_the_fixed_rule": sum(
                1 for r in conviction_rows if r["new_conviction_pattern_fires"]
            ),
            "ledger_theses_matching_old_pattern": old_hits,
            "ledger_theses_matching_new_pattern": new_hits,
            "ledger_theses": len(theses),
            "rows": replay,
        },
        "partial_claims_recomputable": False,
        "why_not": (
            "a claim is partial when some of its relevant records agree and others do not, and "
            "the desk never persisted the records' attributes — only the verdict line — so the "
            f"{examined} examined claims that passed cannot be re-split; the state applies to "
            "every decision from the one that first carries it"
        ),
    }


# --- 2. m-of-K on reconstructed panels ------------------------------------------------------------

_PANEL = re.compile(r"^panel:\s*(?P<n>\d+)\s*analysts?,.*?->\s*(?P<consensus>[a-z_]+)\s+at", re.I)
_CONFLICT = re.compile(
    r"^\[conflict:(?P<kind>direction|magnitude|conviction)\]\s*(?P<a>\S+) \((?P<sa>[a-z_]+)\) vs "
    r"(?P<b>\S+) \((?P<sb>[a-z_]+)\)", re.I
)


@dataclass(frozen=True)
class PanelReading:
    """What the notes say about one panel, and every signal assignment consistent with it."""

    seq: int
    analysts: int
    consensus: str
    assignments: tuple[tuple[str, ...], ...]
    skipped: str = ""

    def call_at(self, threshold: int) -> tuple[bool, str | None]:
        """``(determined, call)``: the m-of-K call if every consistent assignment agrees on it."""
        calls = {m_of_k(a, threshold).call for a in self.assignments}
        if len(calls) != 1:
            return False, None
        return True, next(iter(calls))


def read_panel(row: dict[str, Any]) -> PanelReading | None:
    """Reconstruct a panel's feasible signal assignments from its notes, or ``None`` without one.

    Constraints, each from `agents/conflict.py`'s own rules: every opposed pair of analysts produces
    a direction line naming both (``detect``, unconditionally), so a pair without one is not
    opposed; a line states each named analyst's signal outright; and the panel line's consensus is
    a most-common signal (``Panel.consensus``: ``max`` over the counts, the tie broken by an order
    the notes do not record, so any tied maximum is allowed). The absence of a magnitude or
    conviction line constrains nothing — those depend on sizes and confidences never persisted.
    """
    panel = None
    lines: list[re.Match[str]] = []
    conflict_record = False
    for note in row.get("notes", []) or []:
        text = str(note)
        match = _PANEL.match(text)
        if match is not None and panel is None:
            panel = match
        conflict_record = conflict_record or text.startswith("[conflict")
        conflict = _CONFLICT.match(text)
        if conflict is not None:
            lines.append(conflict)
    if panel is None:
        return None
    seq = int(row.get("seq", 0))
    n = int(panel.group("n"))
    consensus = panel.group("consensus").lower()
    if n == 0:
        return PanelReading(seq, 0, consensus, (), skipped="empty panel")
    if not conflict_record:
        # No conflict line at all means the conflict check did not run on this decision, so the
        # absence of a direction line proves nothing about who was opposed.
        return PanelReading(seq, n, consensus, (), skipped="no conflict record")
    fixed: dict[str, str] = {}
    opposed: set[frozenset[str]] = set()
    for line in lines:
        for name, signal in ((line.group("a"), line.group("sa")),
                             (line.group("b"), line.group("sb"))):
            signal = signal.lower()
            if fixed.setdefault(name, signal) != signal:
                return PanelReading(seq, n, consensus, (), skipped="one analyst, two signals")
        if line.group("kind").lower() == "direction":
            opposed.add(frozenset((line.group("a"), line.group("b"))))
    named = sorted(fixed)
    if len(named) > n:
        return PanelReading(seq, n, consensus, (), skipped="more analysts named than counted")
    if n > MAX_ENUMERATED_ANALYSTS:
        return PanelReading(seq, n, consensus, (), skipped="panel too large to enumerate")
    slots = [*named, *(f"_anon{i}" for i in range(n - len(named)))]
    options = [[fixed[s]] if s in fixed else list(SIGNALS) for s in slots]
    feasible: list[tuple[str, ...]] = []
    for combo in product(*options):
        ok = True
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                di, dj = direction_of(combo[i]), direction_of(combo[j])
                is_opposed = di is not None and dj is not None and di != dj
                if is_opposed != (frozenset((slots[i], slots[j])) in opposed):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        counts = Counter(combo)
        if counts.get(consensus, 0) != max(counts.values()):
            continue
        feasible.append(tuple(combo))
    return PanelReading(seq, n, consensus, tuple(feasible),
                        skipped="" if feasible else "no assignment fits the notes")


def m_of_k_on_record(notes: Sequence[dict[str, Any]], ledger: dict[int, Any]) -> dict[str, Any]:
    readings = [p for p in (read_panel(r) for r in notes) if p is not None]
    usable = [p for p in readings if p.assignments]
    skipped = Counter(p.skipped for p in readings if not p.assignments)
    thresholds: dict[str, Any] = {}
    max_k = max((p.analysts for p in usable), default=0)
    for m in range(1, max_k + 1):
        eligible = [p for p in usable if p.analysts >= m]
        determined = decisive = graded = hits = 0
        for p in eligible:
            known, call = p.call_at(m)
            if not known:
                continue
            determined += 1
            if call is None:
                continue
            decisive += 1
            direction, _move = realised_direction(ledger.get(p.seq))
            if direction is None:
                continue
            graded += 1
            hits += call == direction
        thresholds[f"{m}-of-K"] = {
            "panels_with_K_at_least_m": len(eligible),
            "determined": determined,
            "decisive": decisive,
            "decisiveness": round(decisive / determined, 4) if determined else None,
            "graded": graded,
            "hits": hits,
            "hit_rate": round(hits / graded, 4) if graded else None,
            "hit_rate_95ci": wilson(hits, graded),
            "stated_threshold": m == DEFAULT_AGREEMENT,
        }
    plurality_graded = plurality_hits = plurality_calls = 0
    up_graded = up_hits = 0
    for p in usable:
        direction, _move = realised_direction(ledger.get(p.seq))
        if direction is not None:
            up_graded += 1
            up_hits += direction == "up"
        call = direction_of(p.consensus)
        if call is None:
            continue
        plurality_calls += 1
        if direction is None:
            continue
        plurality_graded += 1
        plurality_hits += call == direction
    return {
        "panels_on_record": len(readings),
        "panels_reconstructed": len(usable),
        "panels_skipped": dict(skipped),
        "assignments_per_panel": {
            "min": min((len(p.assignments) for p in usable), default=0),
            "max": max((len(p.assignments) for p in usable), default=0),
            "exactly_one": sum(1 for p in usable if len(p.assignments) == 1),
        },
        "thresholds": thresholds,
        "desk_plurality": {
            "calls": plurality_calls, "graded": plurality_graded, "hits": plurality_hits,
            "hit_rate": round(plurality_hits / plurality_graded, 4) if plurality_graded else None,
            "hit_rate_95ci": wilson(plurality_hits, plurality_graded),
        },
        "base_rate_always_up": {
            "graded": up_graded, "hits": up_hits,
            "hit_rate": round(up_hits / up_graded, 4) if up_graded else None,
        },
        "grading": (
            f"call against the settled move's sign (eval/shadow.move_of), moves within "
            f"{DEAD_ZONE_BPS}bps not graded; a panel counts at a threshold only when every signal "
            f"assignment consistent with its notes gives the same call"
        ),
    }


# --- 3. the rubric's baseline ---------------------------------------------------------------------


def rubric_baseline(notes: Sequence[dict[str, Any]], ledger: dict[int, Any]) -> dict[str, Any]:
    """How much the one stated confidence says about each axis the desk measures independently."""
    rows = grounding_rows(notes)
    level_by_seq = {r.seq: r for r in rows}
    axes: dict[str, tuple[list[float], list[float], list[bool]]] = {
        "groundedness": ([], [], []),
        "evidence_breadth": ([], [], []),
        "independence": ([], [], []),
    }
    for row in notes:
        seq = int(row.get("seq", 0))
        entry = ledger.get(seq)
        if entry is None:
            continue
        raw_confidence = getattr(entry, "stated_confidence", None)
        if raw_confidence is None:
            continue
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        grounding = level_by_seq.get(seq)
        if grounding is not None and grounding.figures:
            coverage = 1 - grounding.unresolved / grounding.figures
            axes["groundedness"][0].append(confidence)
            axes["groundedness"][1].append(coverage)
            axes["groundedness"][2].append(grounding.unresolved == 0)
        stats = panel_stats(row)
        if stats:
            axes["evidence_breadth"][0].append(confidence)
            axes["evidence_breadth"][1].append(stats["sources"])
            axes["evidence_breadth"][2].append(stats["sources"] >= NARROW_SOURCES)
            axes["independence"][0].append(confidence)
            axes["independence"][1].append(stats["independence"])
            axes["independence"][2].append(stats["independence"] >= LOW_INDEPENDENCE)
    recorded = rubric_pairs_on_record(notes)
    return {
        "self_score_used": "the ledger's stated_confidence — the one number the rubric sits beside",
        "axes": {name: agreement(*cols) for name, cols in axes.items()},
        "edge_over_hurdle": (
            "not measured: the hurdle a decision faced is not persisted in the ledger or the notes"
        ),
        "rubric_pairs": recorded,
        "rubric_measured": bool(recorded["pair_notes"]),
        "why_not": (
            "" if recorded["pair_notes"] else
            "the rubric flag is off and this item was built with no paid model calls, so no "
            "decision has a self-rubric yet; the numbers above are the baseline it must beat, "
            "and rubric_pairs is computed from the desk's pair lines once the first is written"
        ),
    }


def rubric_pairs_on_record(notes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Each axis's self-score against the desk's measure, read from the pair lines the desk wrote.

    The same :func:`~argus.agents.rubric.agreement` as the baseline above, over the rubric instead
    of the single confidence, and the same bar (:func:`~argus.agents.rubric.meets_bar`) the pair was
    written with — so the two sections are directly comparable once pair lines exist.
    """
    columns: dict[str, tuple[list[float], list[float], list[bool]]] = {
        name: ([], [], []) for name in RUBRIC_AXES
    }
    lines = 0
    for row in notes:
        for note in row.get("notes", []) or []:
            parsed = parse_pair_note(str(note))
            if not parsed:
                continue
            lines += 1
            for axis, (own, measured) in parsed.items():
                bar = meets_bar(axis, measured) if axis in columns else None
                if measured is None or bar is None:
                    continue
                columns[axis][0].append(own)
                columns[axis][1].append(measured)
                columns[axis][2].append(bar)
    return {
        "pair_notes": lines,
        "axes": {name: agreement(*cols) for name, cols in columns.items() if cols[0]},
    }


# --- 1d. the three-state lines the live desk has printed ------------------------------------------

_SUPPORT_LINE = re.compile(r"^\[grounding\] support: (?P<level>partial|none)\b")
_NEAR_COUNT = re.compile(r"; (?P<n>\d+) more misquotes? one by under")
_PARTIAL_CLAIM = re.compile(r"^\[claim\] (?P<rule>[a-z_]+): partially supported")


def three_state_live(notes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What the shipped checkers have written into the live record since the three states landed.

    The rebuild in :func:`near_miss_lower_bound` is a lower bound because the record lacks most of
    what the desk knew; the desk itself holds every value, so the lines it prints are the complete
    reading, for the decisions made since. Also a completeness check on the integration: every
    failing grounding note written after the first support line must carry one.
    """
    first: int | None = None
    for row in notes:
        if any(_SUPPORT_LINE.match(str(n)) for n in row.get("notes", []) or []):
            seq = int(row.get("seq", 0))
            first = seq if first is None else min(first, seq)
    levels: Counter[str] = Counter()
    near = partial_claims = fails_since = fails_without_line = 0
    claim_rules: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in notes:
        seq = int(row.get("seq", 0))
        texts = [str(n) for n in row.get("notes", []) or []]
        for text in texts:
            claim = _PARTIAL_CLAIM.match(text)
            if claim is not None:
                partial_claims += 1
                claim_rules[claim.group("rule")] += 1
        if first is None or seq < first:
            continue
        failed = any(_GROUNDING_FAIL.match(t) for t in texts)
        support = next((m for t in texts if (m := _SUPPORT_LINE.match(t)) is not None), None)
        fails_since += failed
        if failed and support is None:
            fails_without_line += 1
        if support is None:
            continue
        levels[support.group("level")] += 1
        line = next(t for t in texts if _SUPPORT_LINE.match(t))
        count = _NEAR_COUNT.search(line)
        near += int(count.group("n")) if count is not None else 0
        if count is not None and len(examples) < 12:
            examples.append({"seq": seq, "line": line})
    return {
        "first_seq_with_a_support_line": first,
        "failing_decisions_since": fails_since,
        "failing_decisions_missing_the_support_line": fails_without_line,
        "support_lines_by_level": dict(levels),
        "near_miss_figures_printed": near,
        "near_miss_examples": examples,
        "partial_claims_printed": partial_claims,
        "partial_claims_by_rule": dict(claim_rules),
    }


# --- 4. the prompt cache on a recorded replay -----------------------------------------------------


def _replay_stream(cache: PromptCache) -> dict[str, Any]:
    """Every stage of the recorded thesis-quality run, offline, requests routed via ``cache``."""
    from argus.eval import thesis_quality as tq

    client = _ReplayClient.make(tq.load_recording(), cache)
    frames = tq.load_frames()
    s5, kept = tq.run_s5(frames, client)
    s6 = tq.run_s6(kept, client)
    s4 = tq.run_s4(kept, client)
    s7 = tq.run_s7(kept, client)
    return {
        "s5_rows": len(s5["rows"]), "s6_rows": len(s6["rows"]), "s4_rows": len(s4["rows"]),
        "s7_rows": len(s7["rows"]),
        "recording_lookups": int(client.replayed),
        "stopped": {k: v["stopped"] for k, v in (("s5", s5), ("s6", s6), ("s4", s4), ("s7", s7))
                    if v.get("stopped")},
    }


def prompt_cache_replay() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prompt_cache.jsonl"
        first_cache = PromptCache(path)
        first = _replay_stream(first_cache)
        second_cache = PromptCache(path)  # a new process's view: loaded from disk only
        second = _replay_stream(second_cache)
        stored = len(second_cache)
    rows = _jsonl(COST_LEDGER_PATH)
    outcomes = Counter(str(r.get("outcome", "")) for r in rows)
    requests = outcomes.get("ok", 0) + outcomes.get("cache_hit", 0) + outcomes.get("error", 0)
    return {
        "first_run_within_process": {**first, **first_cache.stats.as_dict()},
        "second_run_from_persisted_cache": {**second, **second_cache.stats.as_dict()},
        "answers_persisted": stored,
        "live_cost_ledger": {
            "requests": requests,
            "cache_hits": outcomes.get("cache_hit", 0),
            "hit_rate": round(outcomes.get("cache_hit", 0) / requests, 4) if requests else None,
            "first": min((str(r.get("ts", "")) for r in rows), default=""),
            "last": max((str(r.get("ts", "")) for r in rows), default=""),
        },
        "default_prompt_unchanged": MetaPM(_NoModel()).system_prompt() == SYSTEM_PROMPT,
    }


class _NoModel:
    """A model that must never be called — used only to read a MetaPM's prompt."""

    budget = None

    def complete(self, *_: Any, **__: Any) -> Completion:
        raise AssertionError("no model call is permitted in this evaluation")

    def complete_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("no model call is permitted in this evaluation")


# --- 5. id remap exposure -------------------------------------------------------------------------

_EVIDENCE_ID = re.compile(r"^\[(?P<id>[^\]]+)\]")
_LONG_DIGITS = re.compile(r"\d{10,}")


def id_exposure() -> dict[str, Any]:
    from argus.eval import thesis_quality as tq

    ids: list[str] = []
    for rf in tq.load_frames():
        for line in rf.evidence_lines:
            match = _EVIDENCE_ID.match(line)
            if match is not None:
                ids.append(match.group("id"))
    unique = list(dict.fromkeys(ids))
    remap = IdMap(unique)
    shown_chars = sum(len(i) for i in ids)
    handle_chars = sum(len(remap.handle(i)) for i in ids)
    return {
        "evidence_ids_shown_in_recorded_frames": len(ids),
        "distinct": len(unique),
        "mean_length_chars": round(shown_chars / len(ids), 2) if ids else None,
        "carrying_a_run_of_10_or_more_digits": sum(1 for i in unique if _LONG_DIGITS.search(i)),
        "characters_if_shown_as_handles": handle_chars,
        "characters_as_shown": shown_chars,
        "mangling_rate_on_record": None,
        "why_not": (
            "the analysts' raw source_ids are not persisted — only the panel's distinct-source "
            "count — so how often an echoed id was not one shown cannot be read back; the remap "
            "makes it impossible from the first decision that uses it"
        ),
    }


# --- the run --------------------------------------------------------------------------------------


def evaluate() -> dict[str, Any]:
    notes = _jsonl(NOTES_PATH)
    ledger = decisions_by_seq()
    chains = {int(c.get("seq", 0)): c for c in _jsonl(CHAINS_PATH)}
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "qwen_calls_made": 0,
        "inputs": {
            "desk_notes_rows": len(notes), "ledger_decisions": len(ledger),
            "causal_chains": len(chains),
        },
        "partially_supported": {
            "grounding_on_record": grounding_on_record(notes, ledger, chains),
            "grounding_recorded_frames": recorded_frames_rerun(),
            "claims_on_record": claims_on_record(notes, ledger),
            "live_three_state_lines": three_state_live(notes),
        },
        "m_of_k": m_of_k_on_record(notes, ledger),
        "rubric": rubric_baseline(notes, ledger),
        "prompt_cache": prompt_cache_replay(),
        "id_remap": id_exposure(),
        "forced_reflection": {
            "measured": False,
            "why_not": (
                "needs the model's answers under the reflection flag, and no paid call was "
                "permitted; its mechanics — required fields, the seq check against the memory "
                "block, the byte-identical default prompt — are covered by "
                "tests/test_decision_primitives.py"
            ),
        },
    }


def summary_lines(report: dict[str, Any]) -> Iterable[str]:
    g = report["partially_supported"]["grounding_on_record"]
    yield (f"grounding: {g['binary_fail']} binary fails; "
           f"{g['binary_fail_that_is_partial_exactly']} are partial at thesis level, "
           f"{g['binary_fail_with_no_figure_traced']} traced nothing")
    nm = g["near_miss"]
    for rule in ("unit_strict", "quantity_blind_rejected"):
        run = nm[rule]
        yield (f"near misses, {rule} (lower bound): {run['near_misses_found']} of "
               f"{nm['named_unresolved_figures_examined']} named figures; chance rate "
               f"{run['chance_rate_shuffled_pairing']['mean']}")
    oc = g["outcome"]
    for level, row in oc["by_level"].items():
        yield (f"  {level}: {row['decisions']} decisions, lean {row['lean_hits']}/"
               f"{row['lean_graded']} = {row['lean_hit_rate']}")
    yield f"  partial vs none: {oc['partial_vs_none']}; partial vs full: {oc['partial_vs_full']}"
    live = report["partially_supported"]["live_three_state_lines"]
    yield (f"live since seq {live['first_seq_with_a_support_line']}: "
           f"{live['support_lines_by_level']} support lines, "
           f"{live['failing_decisions_missing_the_support_line']} failing without one, "
           f"{live['near_miss_figures_printed']} near misses, "
           f"{live['partial_claims_printed']} partial claims")
    for name, row in report["m_of_k"]["thresholds"].items():
        yield (f"{name}: determined {row['determined']}, decisive {row['decisive']}, "
               f"hits {row['hits']}/{row['graded']} = {row['hit_rate']}")
    yield f"plurality: {report['m_of_k']['desk_plurality']}"
    for name, row in report["rubric"]["axes"].items():
        yield f"confidence vs {name}: {row}"
    pc = report["prompt_cache"]
    yield (f"prompt cache: first run {pc['first_run_within_process']['hits']}/"
           f"{pc['first_run_within_process']['lookups']}, second run "
           f"{pc['second_run_from_persisted_cache']['hits']}/"
           f"{pc['second_run_from_persisted_cache']['lookups']}, live {pc['live_cost_ledger']}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import contextlib

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    report = evaluate()
    write_artefact(REPORT_PATH, report)
    for line in summary_lines(report):
        print(line)
    print(f"written to {REPORT_PATH}")
    return 0


__all__ = [
    "REPORT_PATH",
    "GroundingRow",
    "PanelReading",
    "claims_on_record",
    "evaluate",
    "fisher_two_sided",
    "grounding_on_record",
    "grounding_rows",
    "id_exposure",
    "m_of_k_on_record",
    "outcome_by_level",
    "prompt_cache_replay",
    "quantity_blind_nearest",
    "read_panel",
    "reconstructed_known",
    "recorded_frames_rerun",
    "rubric_baseline",
    "rubric_pairs_on_record",
    "three_state_live",
    "wilson",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
