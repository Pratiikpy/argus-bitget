"""Population RCT's foundation: N gate-ablated `ConstitutionPolicy` variants ruled against the
SAME `attacked` intent one real cycle already produced, for the cost of N cheap deterministic
calls — never N more calls against a hackathon key with a stated limited balance.

Not the harness itself (that is a separate, not-yet-built module — see `Activity/PROGRESS.md`).
This pins the one piece `TradingDesk.run` now provides: `ablated_constitutions` in,
`DeskRun.ablated_rulings` out, with no extra LLM spend and no interference with the real ruling.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from argus.agents.desk import ConstitutionPolicy, TradingDesk
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionPhase, SessionState
from argus.truth.evidence import Evidence

AT = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


def _session() -> SessionState:
    return SessionState(
        phase=SessionPhase.RTH, as_of=AT, hours_to_next_discovery=0.0, nav_age_seconds=5.0
    )


def _evidence() -> list[Evidence]:
    return [
        Evidence(id="e1", claim="NVDA announced a new datacentre contract", source="news",
                 available_at=AT, credibility=0.9),
    ]


class _TradeProposingModel:
    """Every call returns a fixed, real TRADE — enough for the Constitution to have something to
    rule on, unlike the default NO_TRADE stub other test files use for panel/debate concerns."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        required = tuple(kwargs.get("required_keys", ()))
        if "verdict" in required:
            return {
                "verdict": "TRADE", "side": "buy", "quantity": "10000",
                "confidence": 0.9, "thesis": "guidance beat, not yet priced in",
                "invalidation": ["guidance reaffirmed lower"],
            }
        return {
            "signal": "bullish", "confidence": 0.8, "magnitude_bps": 40,
            "reasoning": "guidance beat", "chain": ["a", "b"],
        }

    def complete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused here
        raise NotImplementedError


def _run(model: _TradeProposingModel, **kw: Any) -> Any:
    desk = TradingDesk(model)
    return desk.run(
        symbol="NVDAUSDT", session=_session(), token_price=Decimal("200"),
        position=Decimal("0"), evidence=_evidence(),
        hedges=HedgeabilitySurface(candidates=()), decision_id="test-ablated",
        constitution=ConstitutionPolicy(), **kw,
    )


class TestAblatedRulings:
    def test_no_variants_means_an_empty_dict_not_an_error(self) -> None:
        run = _run(_TradeProposingModel())
        assert run.ablated_rulings == {}

    def test_a_tighter_variant_produces_a_different_ruling_than_the_real_one(self) -> None:
        model = _TradeProposingModel()
        run = _run(
            model,
            ablated_constitutions={
                "no_position_cap": ConstitutionPolicy(max_position_notional=Decimal("1")),
            },
        )
        assert run.ruling is not None
        assert run.ablated_rulings["no_position_cap"].binding_constraint == "max_position"
        assert run.ablated_rulings["no_position_cap"].resulting_intent.quantity == Decimal("1")
        # The real ruling is untouched by the variant existing at all.
        assert run.ruling.binding_constraint != "max_position"

    def test_multiple_variants_are_each_ruled_independently(self) -> None:
        run = _run(
            _TradeProposingModel(),
            ablated_constitutions={
                "tight": ConstitutionPolicy(max_position_notional=Decimal("1")),
                "loose": ConstitutionPolicy(max_position_notional=Decimal("999999")),
            },
        )
        assert run.ablated_rulings["tight"].resulting_intent.quantity == Decimal("1")
        assert run.ablated_rulings["loose"].binding_constraint != "max_position"

    def test_ablated_variants_cost_no_extra_llm_calls(self) -> None:
        """The whole point: N deterministic re-rulings, never N more model calls."""
        baseline = _TradeProposingModel()
        _run(baseline)
        baseline_calls = baseline.calls

        with_variants = _TradeProposingModel()
        _run(
            with_variants,
            ablated_constitutions={
                f"variant-{i}": ConstitutionPolicy(max_position_notional=Decimal(str(i)))
                for i in range(1, 6)
            },
        )
        assert with_variants.calls == baseline_calls

    def test_ablated_rulings_are_present_in_as_dict(self) -> None:
        run = _run(
            _TradeProposingModel(),
            ablated_constitutions={
                "tight": ConstitutionPolicy(max_position_notional=Decimal("1")),
            },
        )
        record = run.as_dict()
        assert record["ablated_rulings"]["tight"]["binding_constraint"] == "max_position"
        assert record["ablated_rulings"]["tight"]["resulting_quantity"] == "1"
