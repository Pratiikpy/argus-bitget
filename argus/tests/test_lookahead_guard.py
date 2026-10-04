"""Build-list 2.2: no market state older than what the deciding model may have been trained on can
reach it, so a graded decision can never be recall of what followed (`agents/meta_pm.py`)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from argus.agents.meta_pm import (
    MODEL_KNOWLEDGE_BOUND,
    KnownPastError,
    MarketFrame,
    MetaPM,
    refuse_known_past,
)
from argus.truth.clocks import SessionPhase, SessionState


def _frame(as_of: datetime) -> MarketFrame:
    return MarketFrame(
        symbol="rNVDA", as_of=as_of,
        session=SessionState(as_of=as_of, phase=SessionPhase.WEEKEND, hours_to_next_discovery=30.0),
        token_price=Decimal("118.40"), position_quantity=Decimal("0"),
        round_trip_bps=Decimal("12"),
    )


class _Recording:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {"verdict": "NO_TRADE", "side": "BUY", "quantity": 0, "confidence": 0.5,
                "thesis": "no edge at 12bps"}


def test_an_old_state_never_reaches_the_model() -> None:
    client = _Recording()
    pm = MetaPM(client, candidates=1, repair_rounds=0)  # type: ignore[arg-type]
    old = _frame(datetime(2026, 3, 2, 21, 0, tzinfo=UTC))
    with pytest.raises(KnownPastError, match="predates 2026-09-01"):
        pm._ask([{"role": "user", "content": old.to_prompt_block()}], old)
    assert client.calls == 0


def test_a_scripted_answer_has_nothing_to_recall_and_is_exempt() -> None:
    client = _Recording()
    client.recalls_the_past = False  # type: ignore[attr-defined]
    pm = MetaPM(client, candidates=1, repair_rounds=0)  # type: ignore[arg-type]
    old = _frame(datetime(2026, 3, 2, 21, 0, tzinfo=UTC))
    pm._ask([{"role": "user", "content": old.to_prompt_block()}], old)
    assert client.calls == 1


def test_the_real_clients_do_not_claim_the_exemption() -> None:
    from argus.llm.cache import CachedModel
    from argus.llm.provider import FallbackClient
    from argus.llm.qwen import QwenClient

    for cls in (QwenClient, FallbackClient, CachedModel):
        assert getattr(cls, "recalls_the_past", True) is True


def test_a_state_after_the_bound_is_asked() -> None:
    client = _Recording()
    pm = MetaPM(client, candidates=1, repair_rounds=0)  # type: ignore[arg-type]
    fresh = _frame(datetime(2026, 9, 13, 3, 0, tzinfo=UTC))
    pm._ask([{"role": "user", "content": fresh.to_prompt_block()}], fresh)
    assert client.calls == 1


def test_the_bound_sits_before_every_decision_on_the_record() -> None:
    import json

    from argus.truth.paths import DATA_DIR

    first = min(json.loads(line)["decided_at"] for line in
                (DATA_DIR / "paper_ledger.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and '"decided_at"' in line)
    assert datetime.fromisoformat(first) >= MODEL_KNOWLEDGE_BOUND
    refuse_known_past(datetime.fromisoformat(first))  # the record's first decision is askable
