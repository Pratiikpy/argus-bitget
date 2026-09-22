"""The risk budget gate: the circuit breaker and the sizing rule, wired so they bind.

Three modules were complete, tested, and connected to nothing. `risk/sizing.py`'s `size()` had **no
caller anywhere in `src/argus`**; `sizing.py:149` documented `risk_multiplier` as coming from
`circuit.risk_multiplier` and nothing implemented the link; `risk/circuit.py`'s own docstring set
the bar — *"the test is not 'does it run' but 'inject a 3-sigma adverse move mid-cycle and confirm
the desk produces a different answer, and says why'"* — and it could not be met.

So these tests are about a decision changing. The two properties that matter most are that an
**unproven** desk is sized down hard, and that proving calibration is the only thing that lifts it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from argus.agents.desk import ConstitutionPolicy
from argus.decision.verdicts import ConstitutionVerdict, Intent, Side, Verdict
from argus.eval.observatory import Prediction
from argus.risk.circuit import BookState
from argus.risk.hedgeability import HedgeabilitySurface
from argus.risk.sizing import FIXED_FRACTION, MAX_FRACTION, MIN_GRADED
from argus.truth.clocks import DualClock

NOW = datetime(2026, 9, 14, 18, tzinfo=UTC)
SESSION = DualClock().state(NOW, nav_age_seconds=60.0)
SURFACE = HedgeabilitySurface(candidates=(), session_note="test")
EQUITY = Decimal("100000")


def _intent(quantity: str = "19000", confidence: float = 0.80) -> Intent:
    return Intent(
        symbol="NVDAUSDT", side=Side.BUY, quantity=Decimal(quantity),
        verdict=Verdict.TRADE, stated_confidence=confidence, thesis="t", invalidation=("x",),
    )


def _book(*, equity: str = "100000", peak: str = "100000") -> BookState:
    return BookState(
        equity=Decimal(equity), peak_equity=Decimal(peak), session_open_equity=Decimal(peak),
    )


def _calibrated(n: int = 24, confidence: float = 0.80) -> list[Prediction]:
    """A record that earns the right to size on confidence: stated 0.80, right ~80% of the time."""
    hits = round(n * confidence)
    return [Prediction(confidence=confidence, correct=i < hits) for i in range(n)]


class TestAnUnprovenDeskIsSizedDown:
    def test_zero_graded_outcomes_caps_at_the_fixed_fraction(self) -> None:
        """The live state: 183 decisions, 69 settled, **none** of them gradeable. Before this gate
        the only limit was a flat 50,000 notional; now it is 5% of the book."""
        policy = ConstitutionPolicy(book_state=_book(), graded_predictions=[])
        ruling = policy.rule(_intent("19000"), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint == "risk_budget"
        assert ruling.resulting_intent.quantity == EQUITY * FIXED_FRACTION

    def test_the_reason_names_the_calibration_failure(self) -> None:
        """A cap with no stated cause is indistinguishable from an arbitrary one."""
        policy = ConstitutionPolicy(book_state=_book(), graded_predictions=[])
        ruling = policy.rule(_intent(), session=SESSION, hedges=SURFACE)
        assert "confidence not usable" in ruling.reason
        assert f"{MIN_GRADED} needed" in ruling.reason

    def test_a_request_inside_the_budget_is_untouched(self) -> None:
        """The gate is a cap, not a target. It must not inflate a modest request."""
        policy = ConstitutionPolicy(book_state=_book(), graded_predictions=[])
        ruling = policy.rule(_intent("1000"), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint != "risk_budget"
        assert ruling.resulting_intent.quantity == Decimal("1000")

    def test_too_few_graded_outcomes_still_fails(self) -> None:
        policy = ConstitutionPolicy(
            book_state=_book(), graded_predictions=_calibrated(MIN_GRADED - 1),
        )
        ruling = policy.rule(_intent("19000"), session=SESSION, hedges=SURFACE)
        assert ruling.resulting_intent.quantity == EQUITY * FIXED_FRACTION


class TestProvingCalibrationIsWhatLiftsTheCap:
    def test_a_calibrated_record_earns_more_size_than_an_unproven_one(self) -> None:
        """The gate must not be a constant. If twenty well-calibrated outcomes bought nothing, the
        calibration gate would be decoration and the fixed fraction would be the whole rule."""
        unproven = ConstitutionPolicy(book_state=_book(), graded_predictions=[])
        proven = ConstitutionPolicy(book_state=_book(), graded_predictions=_calibrated())

        small = unproven.rule(_intent("19000"), session=SESSION, hedges=SURFACE)
        large = proven.rule(_intent("19000"), session=SESSION, hedges=SURFACE)
        assert large.resulting_intent.quantity > small.resulting_intent.quantity

    def test_the_cap_never_exceeds_the_maximum_fraction(self) -> None:
        """Whatever the arithmetic says, no single position takes more than a quarter of the book.

        ``max_unhedged_notional`` is lifted here because gate 4 binds at 20,000 on an empty hedge
        surface — and 20,000 is *below* the quarter-book cap, so without lifting it this assertion
        would have passed on the hedge gate's number and proven nothing about the budget.
        """
        policy = ConstitutionPolicy(
            book_state=_book(), graded_predictions=_calibrated(),
            max_unhedged_notional=Decimal("10000000"),
        )
        ruling = policy.rule(_intent("999999"), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint == "risk_budget"
        assert ruling.resulting_intent.quantity <= EQUITY * MAX_FRACTION

    def test_lower_confidence_earns_less_size(self) -> None:
        """Once confidence is usable it must actually be used, or the gate would be sizing on
        nothing while claiming to size on calibration."""
        bold = ConstitutionPolicy(
            book_state=_book(), graded_predictions=_calibrated(24, 0.80),
            max_unhedged_notional=Decimal("10000000"),
        )
        timid = ConstitutionPolicy(
            book_state=_book(), graded_predictions=_calibrated(24, 0.60),
            max_unhedged_notional=Decimal("10000000"),
        )
        big = bold.rule(_intent("999999", confidence=0.80), session=SESSION, hedges=SURFACE)
        small = timid.rule(_intent("999999", confidence=0.60), session=SESSION, hedges=SURFACE)
        assert big.resulting_intent.quantity > small.resulting_intent.quantity


class TestTheBreakerChangesTheDecision:
    def test_a_drawdown_narrows_the_budget_and_says_why(self) -> None:
        """The same intent, the same market, a different book — and a different answer."""
        healthy = ConstitutionPolicy(book_state=_book(), graded_predictions=[])
        drawn = ConstitutionPolicy(book_state=_book(equity="93000"), graded_predictions=[])

        before = healthy.rule(_intent("19000"), session=SESSION, hedges=SURFACE)
        after = drawn.rule(_intent("19000"), session=SESSION, hedges=SURFACE)

        assert after.binding_constraint == "risk_budget"
        assert after.resulting_intent.quantity < before.resulting_intent.quantity
        assert "drawdown from peak" in after.reason

    def test_a_deep_drawdown_halts_rather_than_resizes(self) -> None:
        """At the halt threshold the multiplier is zero, and a budget of zero is a refusal — not a
        very small version of a trade nobody should be taking."""
        policy = ConstitutionPolicy(book_state=_book(equity="80000"), graded_predictions=[])
        ruling = policy.rule(_intent(), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint == "risk_budget"
        assert ruling.verdict is ConstitutionVerdict.REJECT

    def test_deeper_drawdowns_never_permit_more(self) -> None:
        sizes = []
        for equity in ("100000", "96000", "93000", "80000"):
            policy = ConstitutionPolicy(book_state=_book(equity=equity), graded_predictions=[])
            ruling = policy.rule(_intent("999999"), session=SESSION, hedges=SURFACE)
            sizes.append(ruling.resulting_intent.quantity)
        assert sizes == sorted(sizes, reverse=True)

    def test_the_session_throttle_is_not_double_counted_here(self) -> None:
        """`sizing.size` accepts a session multiplier and this gate deliberately passes 1, because
        the Constitution applies the throttle as its own later rule. Folding it in would charge the
        market's clock twice and make the binding constraint unattributable."""
        import inspect

        from argus.agents import desk

        source = inspect.getsource(desk.ConstitutionPolicy.rule)
        assert 'session_multiplier=Decimal("1")' in source


class TestAbsentStateIsNotZeroRisk:
    def test_no_book_state_means_the_gate_does_not_fire(self) -> None:
        policy = ConstitutionPolicy(book_state=None, graded_predictions=[])
        ruling = policy.rule(_intent(), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint != "risk_budget"

    def test_an_empty_ledger_is_a_book_at_full_equity_and_no_drawdown(self, tmp_path) -> None:
        """Corrected from an earlier revision that returned ``None`` here. A paper book that has
        closed nothing has realised equity of exactly the starting book and a drawdown of exactly
        zero — that is a measurement. Returning ``None`` disabled the position cap on precisely the
        book that has proven nothing, which is the opposite of the caution intended."""
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import STARTING_EQUITY, _book_state

        state = _book_state(PaperLedger(path=tmp_path / "l.jsonl"))
        assert state is not None
        assert state.equity == STARTING_EQUITY
        assert state.total_drawdown == Decimal("0")

    def test_an_unsettled_decision_does_not_move_equity(self, tmp_path) -> None:
        """Marking an open decision would let an unrealised swing move the breaker before the
        outcome that justifies it exists."""
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import STARTING_EQUITY, _book_state

        ledger = PaperLedger(path=tmp_path / "l.jsonl")
        ledger.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY", quantity=Decimal("1"),
            entry_price=Decimal("100"), stated_confidence=0.6, thesis="t",
            invalidation=("x",), market_state_hash="a" * 8, approved_intent_hash="b" * 8,
            session_phase="rth", hours_to_discovery=0.0, decided_at=NOW,
        )
        state = _book_state(ledger)
        assert state.equity == STARTING_EQUITY
        assert state.open_positions == 1

    def test_an_empty_ledger_grades_nothing(self, tmp_path) -> None:
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import _graded_predictions

        assert _graded_predictions(PaperLedger(path=tmp_path / "l.jsonl")) == []

    def test_the_live_record_states_a_calibration_verdict_with_its_number(self) -> None:
        """**This test has now been right twice and wrong twice, which is the point of rewriting
        it this way.** Its first version asserted the live record had too few graded outcomes to
        judge, and predicted its own obsolescence: *"when this starts failing, the record has
        begun to earn its confidence."* It did, on 2026-09-15. Its second version then asserted
        the opposite fixed fact — enough samples but ECE 0.195, above the 0.15 ceiling, so sizing
        still falls back. That broke too, on 2026-09-20: the live record reached **224 graded
        outcomes at ECE 0.071** and the gate now **passes**.

        Both versions pinned a transient fact about the world rather than a property of the code,
        so both were guaranteed to break as the desk accumulated history — which is the desk
        working, not a regression. What is actually invariant, and is what this now asserts: on a
        record that clears `MIN_GRADED`, the gate reaches a definite verdict, reports the ECE that
        produced it, and states its reason in the basis. Which way it lands is the live record's
        business, and it is printed so a reader can see today's answer without this file having to
        be edited again."""
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import LEDGER_PATH, _graded_predictions
        from argus.risk.sizing import MAX_ECE, size

        if not LEDGER_PATH.exists():
            return
        predictions = _graded_predictions(PaperLedger(path=LEDGER_PATH))
        assert len(predictions) >= MIN_GRADED
        result = size(
            win_probability=0.6, payoff=Decimal("1"), predictions=predictions,
            risk_multiplier=Decimal("1"), session_multiplier=Decimal("1"),
        )
        assert isinstance(result.gate.passed, bool)
        assert result.gate.ece is not None
        # The verdict and the threshold must agree with each other, whichever way it fell.
        assert result.gate.passed == (result.gate.ece <= MAX_ECE)
        assert "calibration error" in result.gate.reason


class TestTheGateOrder:
    """``max_unhedged_notional`` is raised out of the way where noted: with an empty hedge surface
    gate 4 binds on anything above 20,000 and would answer a question about gates 5 and 7."""

    def test_the_absolute_cap_still_binds_last(self) -> None:
        """`max_position` is a hard ceiling that must never be diluted by a multiplier, so it sits
        after the budget — and with no book state the budget does not fire at all.

        **`reference_price` fixed 2026-09-22.** Every notional-gated ceiling in `rule()` — this
        one included — requires `price is not None`, and `reference_price` defaults to `None`;
        without setting it this test's own `max_position` gate was silently skipped along with
        every other notional gate, and the chain fell through to the terminal `binding_constraint
        == "none"` — the exact "an absent cap that reads like a satisfied one" failure mode
        `rule()`'s own `unpriced` comment warns about, reached here by the test itself rather than
        by production code (which always back-fills `reference_price` before calling `rule()`).
        `price=1` makes notional equal quantity, the simplest value that still exercises the gate."""
        policy = ConstitutionPolicy(
            book_state=None, max_unhedged_notional=Decimal("10000000"),
            reference_price=Decimal("1"),
        )
        ruling = policy.rule(_intent("999999"), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint == "max_position"

    def test_the_budget_binds_before_the_absolute_cap(self) -> None:
        """An oversized request is reported as a budget refusal, because the budget is the reason
        the position is small — naming the 50,000 cap would hide the real constraint."""
        policy = ConstitutionPolicy(
            book_state=_book(), graded_predictions=[],
            max_unhedged_notional=Decimal("10000000"),
        )
        ruling = policy.rule(_intent("999999"), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint == "risk_budget"

    def test_the_tightest_ceiling_binds_and_the_others_are_still_named(self) -> None:
        """**This reverses a deliberate earlier call, and the old assertion was a 4x over-size.**

        It read `binding_constraint == "unhedgeable_gap"` and defended it as *"an unhedgeable
        position is refused on hedgeability, not sized on drawdown; budgeting a position nobody can
        cover would answer the wrong question."*

        The rhetoric is appealing and the arithmetic refutes it. On this exact state the ceilings
        are **unhedgeable_gap at 20,000** and **risk_budget at 5,000**. The old chain returned on
        the first to bind, so it approved **20,000 — four times what the risk budget allowed**.
        That is the bypass defect, sitting in a test that asserted it as correct.

        The position is not being *covered*, it is being *sized*, and 5,000 is the only size every
        constraint permits. The hedgeability fact is not lost: it is named in the reason on every
        ruling now, instead of only when it happened to be evaluated first.

        **`reference_price` fixed 2026-09-22.** This test's own docstring says the ceilings ARE
        unhedgeable_gap at 20,000 and risk_budget at 5,000 — but `unhedgeable_gap` is a notional
        gate, gated on `price is not None`, and this policy never set `reference_price` (default
        `None`), so `unhedgeable_gap` never actually entered `ceilings` and could not appear in
        the "also computed" trail the last assertion below checks for. The ruling was still
        correct (risk_budget doesn't need a price, and it was the only ceiling that fired either
        way) — only the reason string's completeness went untested. `price=1` makes notional equal
        quantity: `max_unhedged_notional / 1 == 20000`, matching the docstring's own numbers.
        """
        policy = ConstitutionPolicy(
            book_state=_book(), graded_predictions=[], reference_price=Decimal("1"),
        )
        ruling = policy.rule(_intent("999999"), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint == "risk_budget"
        assert ruling.resulting_intent.quantity == Decimal("5000.00")
        # The looser ceiling is reported rather than discarded.
        assert "unhedgeable_gap at 20000" in ruling.reason

    def test_confidence_still_binds_before_the_budget(self) -> None:
        """Order is load-bearing: a proposal the desk is not confident in is refused outright, never
        resized into a small version of a trade it did not believe in."""
        policy = ConstitutionPolicy(book_state=_book(), graded_predictions=[])
        ruling = policy.rule(_intent("19000", confidence=0.10), session=SESSION, hedges=SURFACE)
        assert ruling.binding_constraint == "min_confidence"


class TestTheWiring:
    def test_the_runner_supplies_the_book_and_the_record(self) -> None:
        """The defect this closes: all three modules were complete, tested, and called from
        nowhere."""
        import inspect

        from argus.paper import runner

        source = inspect.getsource(runner.run_once)
        assert "book_state=_book_state(ledger)" in source
        assert "graded_predictions=_graded_predictions(ledger)" in source

    def test_the_policy_calls_both_risk_modules(self) -> None:
        import inspect

        from argus.agents import desk

        source = inspect.getsource(desk.ConstitutionPolicy.rule)
        assert "from argus.risk.circuit import risk_multiplier" in source
        assert "from argus.risk.sizing import size" in source

    def test_the_gate_is_in_the_autopsy_chain(self) -> None:
        from argus.eval.autopsy import CHAIN

        names = [g.name for g in CHAIN]
        assert "risk_budget" in names
        assert names.index("risk_budget") < names.index("max_position")
        assert [g.order for g in CHAIN] == list(range(1, len(CHAIN) + 1))
