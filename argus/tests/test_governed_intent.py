"""The ledger may only record what the Constitution allowed.

**This file exists because that property was violated on the live record and no test noticed.**

`paper/runner.py` selected the intent to record as
``proof.llm_revised_intent or proof.llm_original_intent``. The fallback bypasses the risk layer
entirely, and it is reached on exactly the decisions the desk chooses not to re-put to the model —
which is most of them, because it deliberately does not pay for a second opinion when nothing
bound. `agents/desk.py:657` computed the same value correctly from ``ruling.resulting_intent``.

The two disagreed on live seq 264 and 265. `agents/desk.py` emitted *"no order: final verdict
human_review with quantity 0"* and the risk record wrote ``quantity_after: 0,
binding_constraint: no_exposure``, while the ledger stored ``verdict: trade, quantity: 1`` and the
settlement pass booked **+9.6521** and **+5.3339** of P&L against positions the desk had refused
to take. Those two rows were the entire basis of the project's win-rate and net-P&L figures.

Every existing test fed the path an intent the Constitution had already allowed, so the divergent
branch was never exercised. These tests exercise it directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.agents.desk import DeskRun
from argus.decision.verdicts import (
    ConstitutionRuling,
    ConstitutionVerdict,
    Intent,
    Side,
    Verdict,
)
from argus.paper.runner import governed_intent

AT = datetime(2026, 9, 15, 19, 40, tzinfo=UTC)


def _intent(verdict: Verdict, quantity: str) -> Intent:
    return Intent(
        symbol="COINUSDT", side=Side.SELL, quantity=Decimal(quantity), verdict=verdict,
        stated_confidence=0.72, thesis="fixture", invalidation=("x",),
    )


class _Proof:
    """The two fields `governed_intent` reads, with nothing else in the way."""

    def __init__(self, original: Intent, revised: Intent | None) -> None:
        self.llm_original_intent = original
        self.llm_revised_intent = revised


def _run(original: Intent, revised: Intent | None, ruled: Intent | None) -> DeskRun:
    ruling = None
    if ruled is not None:
        ruling = ConstitutionRuling(
            verdict=ConstitutionVerdict.ALLOW, resulting_intent=ruled,
            binding_constraint="no_exposure", reason="no exposure proposed; nothing to narrow",
        )
    return DeskRun(
        symbol="COINUSDT", as_of=AT, panel=None,  # type: ignore[arg-type]
        proof=_Proof(original, revised),  # type: ignore[arg-type]
        ruling=ruling,
    )


class TestTheLedgerRecordsWhatTheRiskLayerDecided:
    def test_the_constitution_narrowing_is_used_when_the_model_was_not_re_asked(self) -> None:
        """**The exact live seq-264 shape.** The model proposed a real SELL; the Constitution
        refused it to quantity 0 and never re-put it to the model. The recorded intent must be
        the refusal, not the proposal."""
        proposed = _intent(Verdict.TRADE, "1")
        refused = _intent(Verdict.HUMAN_REVIEW, "0")
        got = governed_intent(_run(proposed, None, refused))
        assert got is refused
        assert got.quantity == Decimal("0"), "a refused position must not reach the ledger"

    def test_the_old_fallback_would_have_booked_the_refused_position(self) -> None:
        """Pins the defect itself, so a future edit cannot quietly restore it."""
        proposed = _intent(Verdict.TRADE, "1")
        refused = _intent(Verdict.HUMAN_REVIEW, "0")
        run = _run(proposed, None, refused)
        old = run.proof.llm_revised_intent or run.proof.llm_original_intent  # the 2026-09-20 bug
        assert old.quantity == Decimal("1"), "the old expression booked the unconstrained size"
        assert governed_intent(run).quantity == Decimal("0")

    def test_a_revision_still_wins_when_the_model_was_re_asked(self) -> None:
        """The revised intent remains authoritative — the fix narrows the fallback only."""
        revised = _intent(Verdict.TRADE, "3")
        got = governed_intent(_run(_intent(Verdict.TRADE, "9"), revised, _intent(Verdict.TRADE, "5")))
        assert got is revised

    def test_an_allowed_position_is_recorded_unchanged(self) -> None:
        """The fix must not suppress genuine trades — only unapproved ones."""
        allowed = _intent(Verdict.TRADE, "2")
        got = governed_intent(_run(_intent(Verdict.TRADE, "2"), None, allowed))
        assert got is allowed
        assert got.quantity == Decimal("2")

    def test_a_missing_ruling_refuses_rather_than_falling_back(self) -> None:
        """Fail-safe: no ruling means the Constitution never ruled, so nothing may be booked.
        Falling back to the model's intent here is precisely the original bug."""
        with pytest.raises(RuntimeError, match="no Constitution ruling"):
            governed_intent(_run(_intent(Verdict.TRADE, "1"), None, None), symbol="COINUSDT")
