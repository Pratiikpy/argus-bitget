"""ARGUS's episodic ``recall`` vs. TradingAgents' real ``TradingMemoryLog`` — same fixtures, both.

``eval/standing.py``'s "Episodic memory across decisions" capability names its baseline precisely:
``TauricResearch/TradingAgents``'s reflection log, "which stores a model's prose lesson rather than
a graded outcome." This module runs the REAL vendored ``TradingMemoryLog`` (byte-verified in
``eval/baselines/tradingagents_memory.py``) against constructed fixtures, alongside ARGUS's real
``agents.recall.recall()``, rather than describing either from memory.

**What both get right, verified by running real code, not assumed.** Both systems implement a
point-in-time guard against future information leaking into a recall: TradingAgents'
``get_past_context(as_of=...)`` only returns entries whose ``resolved`` date is on or before
``as_of``; ARGUS's ``recall()`` only grades an episode whose ``settled_at`` is strictly before
``now``. Both correctly hide an unresolved outcome and correctly reveal it once resolved — this
module runs both real functions across the boundary and confirms it, rather than trusting either
docstring's own claim.

**Where they structurally diverge, precisely, not "ours is better" by assertion.**

1. **Granularity at the boundary.** TradingAgents compares DATE strings with ``<=`` (inclusive) —
   an entry resolved exactly on ``as_of`` is visible. ARGUS compares DATETIME instants with ``<``
   (strict) — an episode settled at exactly ``now`` is NOT yet graded. Different granularities for
   a reason: TradingAgents' resolution dates are day-level (a full day is either over or not);
   ARGUS's recall runs inside a live decision loop where ``now`` and an episode's own settlement
   could genuinely be the same instant, and treating "settled at this exact moment" as already-safe
   information is the more conservative (and, for an instant-granularity system, the more correct)
   choice. Verified by running both across the exact boundary, not derived on paper.
2. **Pending (ungraded) entries.** TradingAgents' ``get_past_context`` filters out every pending
   entry before doing anything else — an unresolved decision is invisible until it resolves. ARGUS's
   ``recall()`` deliberately includes ungraded episodes, marked as such (its own docstring: "knowing
   the desk has looked at this symbol nine times this week is itself worth knowing"). Neither is a
   bug; TradingAgents' own code makes the opposite design choice explicitly (`entries = [e for e in
   self.load_entries() if not e.get("pending")]`), confirmed by reading it, not assumed.
3. **What the write path costs.** TradingAgents' real reflection path
   (``tradingagents/graph/reflection.py``, not vendored — a thin LLM prompt-and-invoke wrapper with
   no logic of its own to compare) calls ``quick_thinking_llm.invoke(...)`` once per graded episode
   to produce the prose this module's own comparison then stores directly (no LLM call is made
   anywhere in this comparison — the prose is supplied as a fixed string, exercising the same
   storage/retrieval code either way). ARGUS's ``lessons()`` is pure arithmetic over graded ledger
   fields: zero tokens, zero model calls, on every single invocation.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.agents.recall import MIN_EPISODES_FOR_A_LESSON, recall
from argus.eval.baselines.tradingagents_loader import (
    TradingAgentsBaselineLoadError,
    load_trading_memory_log_class,
)


class RecallComparisonError(RuntimeError):
    """The comparison could not run — the baseline failed to load."""


@dataclass(frozen=True)
class LedgerEntry:
    """A minimal stand-in for ``argus.paper.ledger.Entry`` — ``recall()`` only reads these
    attributes (see its own docstring: "any sequence of Entry-shaped records"), so this avoids
    depending on the real ledger file for a comparison that only needs its shape."""

    seq: int
    symbol: str
    decided_at: str
    settled_at: str | None
    verdict: str
    session_phase: str
    stated_confidence: float
    thesis: str
    net_pnl: str | None = None
    direction_correct: bool | None = None
    counterfactual_move_bps: str | None = None


def run_argus_recall(entries: list[LedgerEntry], *, symbol: str, now: datetime) -> Any:
    """ARGUS's real ``recall()`` — no reimplementation."""
    return recall(entries, symbol=symbol, now=now)


def run_tradingagents_context(
    *,
    ticker: str,
    trade_date: str,
    decision_text: str,
    resolution_date: str | None,
    reflection_text: str = "noted",
    as_of: str | None,
    memory_log_class: type,
) -> str:
    """TradingAgents' real ``TradingMemoryLog`` — no paraphrase. Writes to a fresh temp file per
    call so each comparison scenario starts from a clean log, matching how a fresh ``entries``
    list is built per ARGUS scenario rather than sharing mutable state between cases."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "mem.md"
        log = memory_log_class({"memory_log_path": str(log_path)})
        log.store_decision(ticker, trade_date, decision_text)
        if resolution_date is not None:
            log.update_with_outcome(
                ticker, trade_date, raw_return=0.05, alpha_return=0.02, holding_days=1,
                reflection=reflection_text, resolution_date=resolution_date,
            )
        return str(log.get_past_context(ticker, as_of=as_of))


# =============================================================================================
# Designed cases — the point-in-time boundary, pending-inclusion, and the no-floor comparison.
# =============================================================================================


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    argus_visible: bool
    tradingagents_visible: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.name, "argus_visible": self.argus_visible,
            "tradingagents_visible": self.tradingagents_visible, "note": self.note,
        }


def run_boundary_cases(memory_log_class: type) -> tuple[BoundaryCase, ...]:
    """Both systems' point-in-time guard, run across the exact resolution boundary."""
    cases = []

    # ARGUS: settled strictly before `now` is graded.
    before = LedgerEntry(
        seq=1, symbol="NVDA", decided_at="2026-01-01T00:00:00+00:00",
        settled_at="2026-01-04T00:00:00+00:00", verdict="buy", session_phase="rth",
        stated_confidence=0.8, thesis="momentum", net_pnl="120", direction_correct=True,
        counterfactual_move_bps="50",
    )
    argus_after_settle = run_argus_recall(
        [before], symbol="NVDA", now=datetime(2026, 1, 5, tzinfo=UTC)
    )
    argus_before_settle = run_argus_recall(
        [before], symbol="NVDA", now=datetime(2026, 1, 3, tzinfo=UTC)
    )
    argus_exact_instant = run_argus_recall(
        [before], symbol="NVDA", now=datetime(2026, 1, 4, tzinfo=UTC)
    )

    ta_after = run_tradingagents_context(
        ticker="NVDA", trade_date="2026-01-01", decision_text="buy",
        resolution_date="2026-01-04", as_of="2026-01-05", memory_log_class=memory_log_class,
    )
    ta_before = run_tradingagents_context(
        ticker="NVDA", trade_date="2026-01-01", decision_text="buy",
        resolution_date="2026-01-04", as_of="2026-01-03", memory_log_class=memory_log_class,
    )
    ta_exact = run_tradingagents_context(
        ticker="NVDA", trade_date="2026-01-01", decision_text="buy",
        resolution_date="2026-01-04", as_of="2026-01-04", memory_log_class=memory_log_class,
    )

    cases.append(BoundaryCase(
        "well_after_resolution",
        argus_visible=bool(argus_after_settle.graded), tradingagents_visible=bool(ta_after),
        note="both should show the graded outcome",
    ))
    cases.append(BoundaryCase(
        "well_before_resolution",
        argus_visible=bool(argus_before_settle.graded), tradingagents_visible=bool(ta_before),
        note="both should hide the outcome",
    ))
    cases.append(BoundaryCase(
        "exact_resolution_instant",
        argus_visible=bool(argus_exact_instant.graded), tradingagents_visible=bool(ta_exact),
        note=(
            "granularity divergence: ARGUS's strict < excludes an episode settled at exactly "
            "`now`; TradingAgents' <= on date strings includes an entry resolved exactly on "
            "`as_of` — both real, both intentional, different time granularities"
        ),
    ))
    return tuple(cases)


@dataclass(frozen=True)
class PendingInclusionCase:
    argus_shows_pending: bool
    tradingagents_shows_pending: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "argus_shows_pending": self.argus_shows_pending,
            "tradingagents_shows_pending": self.tradingagents_shows_pending,
        }


def run_pending_inclusion_case(memory_log_class: type) -> PendingInclusionCase:
    """A decision with NO resolution yet — does each system's context include it at all?"""
    pending_entry = LedgerEntry(
        seq=2, symbol="TSLA", decided_at="2026-01-01T00:00:00+00:00", settled_at=None,
        verdict="buy", session_phase="rth", stated_confidence=0.6, thesis="pending thesis",
    )
    argus_result = run_argus_recall(
        [pending_entry], symbol="TSLA", now=datetime(2026, 1, 2, tzinfo=UTC)
    )
    ta_result = run_tradingagents_context(
        ticker="TSLA", trade_date="2026-01-01", decision_text="buy", resolution_date=None,
        as_of=None, memory_log_class=memory_log_class,
    )
    return PendingInclusionCase(
        argus_shows_pending=len(argus_result.episodes) > 0,
        tradingagents_shows_pending=bool(ta_result),
    )


@dataclass(frozen=True)
class NoFloorCase:
    argus_states_a_pattern_from_one: bool
    tradingagents_shows_content_from_one: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "argus_states_a_pattern_from_one": self.argus_states_a_pattern_from_one,
            "tradingagents_shows_content_from_one": self.tradingagents_shows_content_from_one,
        }


def run_no_floor_case(memory_log_class: type) -> NoFloorCase:
    """ARGUS refuses to state a pattern below MIN_EPISODES_FOR_A_LESSON; does TradingAgents have
    an equivalent floor, or does it inject whatever it has regardless of sample size?"""
    single_entry = LedgerEntry(
        seq=3, symbol="AAPL", decided_at="2026-01-01T00:00:00+00:00",
        settled_at="2026-01-02T00:00:00+00:00", verdict="no_trade", session_phase="rth",
        stated_confidence=0.5, thesis="single episode", counterfactual_move_bps="10",
    )
    argus_result = run_argus_recall(
        [single_entry], symbol="AAPL", now=datetime(2026, 1, 3, tzinfo=UTC)
    )
    ta_result = run_tradingagents_context(
        ticker="AAPL", trade_date="2026-01-01", decision_text="no_trade",
        resolution_date="2026-01-02", as_of="2026-01-03", memory_log_class=memory_log_class,
    )
    return NoFloorCase(
        argus_states_a_pattern_from_one=bool(
            argus_result.lessons(hurdle_bps=Decimal("5"))
        ),
        tradingagents_shows_content_from_one=bool(ta_result),
    )


# =============================================================================================
# Ablation — ARGUS's own MIN_EPISODES_FOR_A_LESSON floor, shown load-bearing.
# =============================================================================================


@dataclass(frozen=True)
class AblationResult:
    with_floor_states_pattern: bool
    without_floor_would_state_pattern: bool

    @property
    def floor_is_load_bearing(self) -> bool:
        return (not self.with_floor_states_pattern) and self.without_floor_would_state_pattern

    def as_dict(self) -> dict[str, Any]:
        return {
            "with_floor_states_pattern": self.with_floor_states_pattern,
            "without_floor_would_state_pattern": self.without_floor_would_state_pattern,
            "floor_is_load_bearing": self.floor_is_load_bearing,
        }


def run_ablation() -> AblationResult:
    """Four graded abstentions — one below MIN_EPISODES_FOR_A_LESSON (5) — real function, then
    the same data re-evaluated with the floor constant patched down to see the counterfactual."""
    entries = [
        LedgerEntry(
            seq=i, symbol="MSFT", decided_at=f"2026-01-0{i}T00:00:00+00:00",
            settled_at=f"2026-01-0{i + 1}T00:00:00+00:00", verdict="no_trade",
            session_phase="rth", stated_confidence=0.5, thesis="x",
            counterfactual_move_bps="15",
        )
        for i in range(1, 5)
    ]
    now = datetime(2026, 1, 10, tzinfo=UTC)
    with_floor = run_argus_recall(entries, symbol="MSFT", now=now)
    assert len(with_floor.graded) == 4 < MIN_EPISODES_FOR_A_LESSON

    import argus.agents.recall as recall_module

    # `Recall.lessons()` reads `MIN_EPISODES_FOR_A_LESSON` live from the module's own namespace
    # on every call (it is a plain module-level name, not baked into the dataclass at
    # construction) — so the patched-floor read must happen BEFORE `finally` restores it. A first
    # version of this function read `without_floor.lessons(...)` after the patch was already
    # undone and silently got the real, unablated result back (`floor_is_load_bearing` read as
    # False for the wrong reason) — caught by running this and checking the actual booleans, not
    # assumed correct on write.
    original_floor = recall_module.MIN_EPISODES_FOR_A_LESSON
    try:
        recall_module.MIN_EPISODES_FOR_A_LESSON = 4
        without_floor = recall(entries, symbol="MSFT", now=now)
        without_floor_states_pattern = bool(without_floor.lessons(hurdle_bps=Decimal("5")))
    finally:
        recall_module.MIN_EPISODES_FOR_A_LESSON = original_floor

    return AblationResult(
        with_floor_states_pattern=bool(with_floor.lessons(hurdle_bps=Decimal("5"))),
        without_floor_would_state_pattern=without_floor_states_pattern,
    )


# =============================================================================================
# Scope statement.
# =============================================================================================

SCOPE_STATEMENT = """\
Claimed: both systems' point-in-time guard against future information leaking into a recall was \
run, not assumed — both correctly hide an unresolved outcome and correctly reveal it once \
resolved, verified across the exact boundary on real code. ARGUS additionally shows ungraded \
episodes as context (TradingAgents' real code explicitly filters every pending entry out before \
building context — read directly, not inferred) and additionally refuses to state a pattern below \
a stated floor (MIN_EPISODES_FOR_A_LESSON=5, shown load-bearing by ablation) — TradingAgents' real \
get_past_context has no equivalent floor of any kind, verified by running it on a single episode \
and confirming it returns content. ARGUS's lessons() is pure arithmetic, zero tokens, zero model \
calls; TradingAgents' equivalent prose is produced by one LLM call per graded episode \
(reflection.py's quick_thinking_llm.invoke, read directly, not run — no LLM call was made \
anywhere in this comparison, matching this project's own hackathon-key budget discipline).

NOT claimed: that TradingAgents' design is a mistake. Excluding pending entries from context and \
having no minimum-sample floor are real, working design choices for a system whose memory is \
advisory prose re-read by a model that can itself discount thin evidence — a different premise \
from ARGUS's own, where the calibration scorecard (a separate ARGUS capability) already applies \
the same floor discipline and `lessons()` exists specifically so no model has to do that \
discounting itself. Also NOT claimed: that ARGUS's structured, falsifiable design produces BETTER \
trading decisions than prose reflection — that would need graded real-world outcomes from both \
systems on the same decisions, which this comparison does not have and does not claim to.
"""


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        RecallComparisonError: the vendored TradingAgents baseline failed to load.
    """
    try:
        memory_log_class = load_trading_memory_log_class()
    except TradingAgentsBaselineLoadError as exc:
        raise RecallComparisonError(
            f"could not load the vendored TradingAgents baseline: {exc}"
        ) from exc

    boundary = run_boundary_cases(memory_log_class)
    pending = run_pending_inclusion_case(memory_log_class)
    no_floor = run_no_floor_case(memory_log_class)
    ablation = run_ablation()

    return {
        "boundary_cases": [c.as_dict() for c in boundary],
        "boundary_agreement_on_clear_cases": all(
            c.argus_visible == c.tradingagents_visible
            for c in boundary if c.name != "exact_resolution_instant"
        ),
        "pending_inclusion": pending.as_dict(),
        "no_floor": no_floor.as_dict(),
        "ablation": ablation.as_dict(),
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = ["RECALL COMPARISON — ARGUS recall() vs. TradingAgents real TradingMemoryLog", ""]
    lines.append(
        f"boundary cases: {len(report['boundary_cases'])}, agree on the two clear-cut cases: "
        f"{report['boundary_agreement_on_clear_cases']}"
    )
    p = report["pending_inclusion"]
    lines.append(
        f"pending inclusion — ARGUS shows it: {p['argus_shows_pending']}, "
        f"TradingAgents shows it: {p['tradingagents_shows_pending']}"
    )
    nf = report["no_floor"]
    lines.append(
        f"no-floor — ARGUS states a pattern from 1 episode: "
        f"{nf['argus_states_a_pattern_from_one']}, TradingAgents shows content from 1: "
        f"{nf['tradingagents_shows_content_from_one']}"
    )
    a = report["ablation"]
    lines.append(
        f"ablation — MIN_EPISODES_FOR_A_LESSON floor load-bearing: {a['floor_is_load_bearing']}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from pathlib import Path as _Path

    result = main()
    print(render(result))
    out_path = _Path(__file__).resolve().parents[3] / "data" / "recall_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "AblationResult",
    "BoundaryCase",
    "LedgerEntry",
    "NoFloorCase",
    "PendingInclusionCase",
    "RecallComparisonError",
    "main",
    "render",
    "run_ablation",
    "run_argus_recall",
    "run_boundary_cases",
    "run_no_floor_case",
    "run_pending_inclusion_case",
    "run_tradingagents_context",
]
