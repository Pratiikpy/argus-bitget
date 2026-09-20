"""An order the Constitution never approved must be **unconstructable**, not merely unchecked.

**This file exists because the may-only-reduce guarantee was true by habit rather than by
construction.** ARGUS enforces the asymmetry inside `decision/verdicts.apply_constraint` and proves
it across 2,177,280 swept states — and all of that binds only if a caller goes *through*
`apply_constraint`. Nothing forced them to. `OrderBook.submit` accepted a bare `Order`, and two
paths in this repository already built one and submitted it having never consulted the Constitution
at all. A new code path that forgot the check was a naked directional order, and the sweep would not
have caught it, because the sweep drives `ConstitutionPolicy` and not `OrderBook`.

The pattern was read from `Ritapossible/Ballast`'s `ballast/enforcer.py`, which had the property
first and whose own docstring names the same failure. What is added here is the **content binding**:
an authorisation is for a specific symbol, side and quantity, not merely proof that some ruling was
obtained. On 2026-09-20 this project found two live ledger rows recording `quantity: 1` against a
ruling of `quantity_after: 0` — a bare capability would have waved those through, because a ruling
*was* obtained. It simply was not the one the order carried.

A guard that cannot be shown to fail is not a guard, so most of what follows is forgery attempts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.decision.verdicts import (
    Authorised,
    ConstitutionRuling,
    ConstitutionVerdict,
    ConstitutionViolation,
    Intent,
    Side,
    Verdict,
    apply_constraint,
)
from argus.execution.orders import Order, OrderBook, OrderState

NOW = datetime(2026, 9, 20, 14, 30, tzinfo=UTC)


def _intent(*, symbol: str = "NVDAUSDT", side: Side = Side.SELL, quantity: str = "10") -> Intent:
    return Intent(
        symbol=symbol, side=side, quantity=Decimal(quantity), verdict=Verdict.TRADE,
        stated_confidence=0.8, thesis="fixture", invalidation=("x",),
    )


def _order(*, symbol: str = "NVDAUSDT", side: str = "SELL", quantity: str = "10") -> Order:
    return Order(
        client_order_id="c-1", symbol=symbol, side=side,
        quantity=Decimal(quantity), approved_intent_hash="hash-1",
    )


def _ruling(intent: Intent, verdict: ConstitutionVerdict = ConstitutionVerdict.ALLOW):
    return apply_constraint(
        intent, verdict=verdict, binding_constraint="none", reason="nothing bound",
    )


class TestTheCapabilityCannotBeForged:
    def test_constructing_an_authorised_by_hand_is_refused(self) -> None:
        """The whole property in one assertion."""
        with pytest.raises(ConstitutionViolation, match="cannot be constructed directly"):
            Authorised(_order(), _ruling(_intent()), _token=object())

    def test_a_guessed_token_does_not_work(self) -> None:
        """Identity, not equality — so a copied value cannot stand in for the real object."""
        for guess in (None, "authorised", True, 1, object(), ()):
            with pytest.raises(ConstitutionViolation):
                Authorised(_order(), _ruling(_intent()), _token=guess)

    def test_the_token_cannot_be_reached_from_a_ruling(self) -> None:
        """A caller holding a ruling must go through `authorise`; the sentinel is not exposed on
        the object, so it cannot be lifted off one and reused for a different order."""
        ruling = _ruling(_intent())
        assert not any("token" in name.lower() for name in dir(ruling))

    def test_the_order_book_refuses_a_bare_order(self) -> None:
        """The defect this replaced: `submit(order)` used to be a legal call."""
        with pytest.raises(AttributeError):
            OrderBook().submit(_order(), at=NOW)  # type: ignore[arg-type]


class TestTheAuthorisationIsBoundToTheOrder:
    """A ruling authorises *this* order, not any order."""

    def test_a_different_quantity_is_refused(self) -> None:
        """**The live seq-264 shape.** A ruling for 0, an order for 1."""
        refused = _intent(quantity="10")
        ruling = ConstitutionRuling(
            ConstitutionVerdict.RESIZE, "no_exposure", "narrowed to zero",
            Intent(symbol="NVDAUSDT", side=Side.SELL, quantity=Decimal("0"),
                   verdict=Verdict.NO_TRADE, stated_confidence=0.8, thesis="refused"),
        )
        assert refused.quantity != ruling.resulting_intent.quantity
        with pytest.raises(ConstitutionViolation, match="approved quantity"):
            ruling.authorise(_order(quantity="1"))

    def test_a_different_side_is_refused(self) -> None:
        with pytest.raises(ConstitutionViolation, match="approved"):
            _ruling(_intent(side=Side.SELL)).authorise(_order(side="BUY"))

    def test_a_different_symbol_is_refused(self) -> None:
        with pytest.raises(ConstitutionViolation, match="authorisation is for"):
            _ruling(_intent(symbol="NVDAUSDT")).authorise(_order(symbol="TSLAUSDT"))

    def test_a_rejected_ruling_cannot_authorise_anything(self) -> None:
        """REJECT means the order does not exist. It must not be able to mint a capability."""
        ruling = ConstitutionRuling(
            ConstitutionVerdict.REJECT, "gross_exposure", "over the cap", _intent(),
        )
        with pytest.raises(ConstitutionViolation, match="REJECT"):
            ruling.authorise(_order())

    def test_side_comparison_is_case_insensitive(self) -> None:
        """`Order.side` is an upper-case string and `Intent.side` is a lower-case enum. A guard
        that refused on that alone would be rejecting correct orders, which is how a safety check
        gets switched off."""
        assert _ruling(_intent(side=Side.SELL)).authorise(_order(side="SELL"))
        assert _ruling(_intent(side=Side.BUY)).authorise(_order(side="buy"))


class TestTheApprovedPathStillWorks:
    """A control is only useful if it lets the legitimate case through."""

    def test_an_approved_order_submits(self) -> None:
        book = OrderBook()
        got = book.submit(_ruling(_intent()).authorise(_order()), at=NOW)
        assert got.state is OrderState.SUBMITTED
        assert book.get("c-1").symbol == "NVDAUSDT"

    def test_a_resize_authorises_the_narrowed_quantity(self) -> None:
        """The common real case: the Constitution shrinks a position and the smaller order goes."""
        ruling = apply_constraint(
            _intent(quantity="10"), verdict=ConstitutionVerdict.RESIZE,
            binding_constraint="max_position", reason="over the per-symbol cap",
            resized_quantity=Decimal("4"),
        )
        assert ruling.resulting_intent.quantity == Decimal("4")
        assert ruling.authorise(_order(quantity="4"))
        with pytest.raises(ConstitutionViolation):
            ruling.authorise(_order(quantity="10"))  # the un-narrowed original

    def test_duplicate_detection_still_fires_through_the_capability(self) -> None:
        """The capability must not accidentally disable an existing control."""
        from argus.execution.orders import DuplicateOrder

        book = OrderBook()
        book.submit(_ruling(_intent()).authorise(_order()), at=NOW)
        with pytest.raises(DuplicateOrder):
            book.submit(_ruling(_intent()).authorise(_order()), at=NOW)


class TestNoBypassRemains:
    """Grep-level: a future edit must not reintroduce a path around the Constitution."""

    def test_no_module_submits_a_bare_order(self) -> None:
        """Scoped to the execution path deliberately, and the exclusion is narrow and named.

        `argus/sim/` is excluded because it is a different `Order` and a different `OrderBook`
        entirely — `argus.sim.book`, the agent-based market simulator, whose orders are synthetic
        participants' quotes inside a simulated venue. No real exchange is reachable from it and
        the Constitution has no jurisdiction over a simulated counterparty's limit order. This
        exclusion is written here rather than applied silently because a test that quietly skips
        the directory it was about to fail on is worse than no test.
        """
        import re
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "src" / "argus"
        offenders: list[str] = []
        for path in source.rglob("*.py"):
            if "sim" in path.relative_to(source).parts:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                # `.submit(Order(` or `.submit(order` with no authorisation in the expression.
                if re.search(r"\.submit\(\s*(Order\(|order\b)", line) and "authorise" not in line:
                    offenders.append(f"{path.name}:{number}: {line.strip()[:80]}")
        assert not offenders, "bare-order submits found:\n" + "\n".join(offenders)

    def test_the_exclusion_is_a_genuinely_different_type(self) -> None:
        """Pins the reason above. If `argus.sim.book.Order` ever becomes the execution `Order`,
        the exclusion stops being honest and this fails."""
        from argus.execution.orders import Order as ExecutionOrder
        from argus.sim.book import Order as SimulatedOrder

        assert SimulatedOrder is not ExecutionOrder

    def test_the_venue_client_requires_a_capability(self) -> None:
        """`place_order` is the last point before a real exchange."""
        import inspect

        from argus.execution.bitget_client import BitgetTradingClient

        first = list(inspect.signature(BitgetTradingClient.place_order).parameters)[1]
        assert first == "authorised", f"place_order takes {first!r}, not a capability"


class TestTheProverWouldCatchABrokenCapability:
    """A prover that cannot fail proves nothing — the discipline `eval/riskproof.py` already applies
    to its mutant policies, extended to the seventh invariant.

    The sweep asserts the capability property across all 2,177,280 states. That is only worth
    something if a broken capability would actually be reported, so here it is broken on purpose.
    """

    def test_a_ruling_that_authorises_anything_is_reported(self) -> None:
        from argus.eval.riskproof import State, _authorisation_violations
        from argus.truth.clocks import SessionPhase

        class PermissiveRuling(ConstitutionRuling):
            """The mutant: mints a capability for any order, ignoring what it approved."""

            def authorise(self, order: object) -> Authorised:  # type: ignore[override]
                return Authorised(order, self, _token=_sentinel())  # type: ignore[arg-type]

        broken = PermissiveRuling(
            ConstitutionVerdict.ALLOW, "none", "nothing bound", _intent(quantity="10"),
        )
        state = State(
            verdict=Verdict.TRADE, side=Side.SELL, quantity=Decimal("10"), confidence=0.8,
            hedges_empty=True, nav_stale=False, phase=SessionPhase.RTH,
        )
        found = _authorisation_violations(state, broken)
        assert found, "a ruling that authorises a tampered order must be reported"
        assert any("not bound to the size" in v.detail for v in found)

    def test_a_reject_that_mints_a_capability_is_reported(self) -> None:
        from argus.eval.riskproof import State, _authorisation_violations
        from argus.truth.clocks import SessionPhase

        class LeakyReject(ConstitutionRuling):
            def authorise(self, order: object) -> Authorised:  # type: ignore[override]
                return Authorised(order, self, _token=_sentinel())  # type: ignore[arg-type]

        broken = LeakyReject(
            ConstitutionVerdict.REJECT, "gross_exposure", "over the cap", _intent(),
        )
        state = State(
            verdict=Verdict.TRADE, side=Side.SELL, quantity=Decimal("10"), confidence=0.8,
            hedges_empty=True, nav_stale=False, phase=SessionPhase.RTH,
        )
        found = _authorisation_violations(state, broken)
        assert found and any("REJECT" in v.detail for v in found)

    def test_a_correct_ruling_produces_no_violation(self) -> None:
        """The control: the real implementation must be clean, or the two tests above prove
        nothing except that the checker fires on everything."""
        from argus.eval.riskproof import State, _authorisation_violations
        from argus.truth.clocks import SessionPhase

        state = State(
            verdict=Verdict.TRADE, side=Side.SELL, quantity=Decimal("10"), confidence=0.8,
            hedges_empty=True, nav_stale=False, phase=SessionPhase.RTH,
        )
        assert _authorisation_violations(state, _ruling(_intent())) == []


def _sentinel() -> object:
    """The private capability, reached the only way a test legitimately can.

    Importing the module-private name is deliberate and is confined to this file: the mutants above
    must be able to forge a capability in order to prove the prover notices. Production code cannot
    reach it, which is the property `TestTheCapabilityCannotBeForged` pins.
    """
    from argus.decision.verdicts import _AUTHORISATION

    return _AUTHORISATION
