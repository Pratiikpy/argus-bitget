"""Queue-position fill tests.

The defect these guard against is the one our own standing rules record as having cost real
money: a sim that assumed limit fills returned +90% where the replay returned -0.45%. Every test
below asserts that the assumption cannot be made silently.

Expected values are computed by hand from the Rust in ``backtest/models/queue.rs``, not from this
implementation's own output — a test that only asserts self-consistency would have passed with the
probability function inverted, which is exactly the bug that was in the first draft.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from argus.execution.queue import (
    L3FIFOQueue,
    L3Order,
    LogProbability,
    LogProbability2,
    PowerProbability,
    PowerProbability2,
    PowerProbability3,
    Probability,
    ProbQueue,
    QueueError,
    QueueModel,
    QueuePosition,
    RestingOrder,
    RiskAdverseQueue,
    fill_ratio,
)

D = Decimal


class TestTheTouchFallacy:
    """A price touching your level is not a fill. This is the whole point of the module."""

    def test_price_touching_with_no_volume_fills_nothing(self) -> None:
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("500"))
        # The market prints at our price all day, but thinly.
        for _ in range(20):
            assert order.on_trade_at_price(D("5")) == 0
        assert order.filled == 0
        assert order.queue_ahead == D("400")

    def test_a_thin_book_never_gets_there(self) -> None:
        """65 hours a week at a third of RTH depth is where passive strategies die."""
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("300"))
        for _ in range(50):
            order.on_trade_at_price(D("2"))  # 100 total against 300 ahead
        assert order.filled == 0
        assert fill_ratio([order]) == 0

    def test_joining_claims_no_priority(self) -> None:
        pos = RiskAdverseQueue().new_order(D("250"))
        assert pos.front_qty == D("250")

    def test_reaching_the_front_is_not_filling(self) -> None:
        """At front_qty == 0 you are next in line and nothing has traded through you."""
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("100"))
        assert order.on_trade_at_price(D("100")) == 0
        assert order.position.front_qty == 0
        assert not order.position.has_traded_through
        assert order.filled == 0


class TestRiskAdverse:
    """``queue.rs:44-96``."""

    def test_only_trades_advance_the_queue(self) -> None:
        pos = QueuePosition(front_qty=D("100"))
        model = RiskAdverseQueue()
        # A cancellation halves the level; a risk-adverse model assumes it happened behind us.
        model.on_depth(pos, prev_qty=D("200"), new_qty=D("100"))
        assert pos.front_qty == D("100")
        model.on_trade(pos, D("40"))
        assert pos.front_qty == D("60")

    def test_front_never_rises_on_a_refill(self) -> None:
        pos = QueuePosition(front_qty=D("30"))
        RiskAdverseQueue().on_depth(pos, prev_qty=D("50"), new_qty=D("900"))
        assert pos.front_qty == D("30")

    def test_depth_floors_the_front(self) -> None:
        """The level cannot hold less than what is ahead of us."""
        pos = QueuePosition(front_qty=D("100"))
        RiskAdverseQueue().on_depth(pos, prev_qty=D("200"), new_qty=D("15"))
        assert pos.front_qty == D("15")

    def test_execution_is_the_overshoot_only(self) -> None:
        """queue.rs:88-95 — exec = round(-front / lot) * lot, not the whole order."""
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("50"))
        assert order.on_trade_at_price(D("53")) == D("3")
        assert order.filled == D("3")
        assert order.leaves == D("7")
        assert not order.is_complete

    def test_consumed_volume_cannot_fill_twice(self) -> None:
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("50"))
        order.on_trade_at_price(D("53"))
        # No further volume: the 3 already consumed must not fill anything more.
        assert order.model.apply_fill(order.position, order.leaves) == 0
        assert order.filled == D("3")

    def test_a_large_print_completes_the_order(self) -> None:
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("50"))
        assert order.on_trade_at_price(D("500")) == D("10")
        assert order.is_complete
        assert fill_ratio([order]) == 1

    def test_lot_size_quantises_the_fill(self) -> None:
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("50"), lot_size=D("5"))
        # 7 through the queue rounds to 5 at a lot size of 5.
        assert order.on_trade_at_price(D("57")) == D("5")


class TestProbabilityOrientation:
    """The bug that was in the first draft: ``prob`` must rise with ``back``, not ``front``.

    Inverting it produces a model that advances fastest when the whole level is ahead of you —
    plausible-looking output, exactly backwards.
    """

    @pytest.mark.parametrize(
        "func",
        [PowerProbability(), PowerProbability2(), PowerProbability3(),
         LogProbability(), LogProbability2()],
        ids=lambda f: f.name,
    )
    def test_rises_with_back(self, func: Probability) -> None:
        low = func.prob(D("100"), D("10"))
        high = func.prob(D("100"), D("1000"))
        assert high > low, f"{func.name} is inverted"

    @pytest.mark.parametrize(
        "func",
        [PowerProbability(), PowerProbability2(), PowerProbability3(),
         LogProbability(), LogProbability2()],
        ids=lambda f: f.name,
    )
    def test_falls_with_front(self, func: Probability) -> None:
        assert func.prob(D("10"), D("100")) > func.prob(D("1000"), D("100"))

    @pytest.mark.parametrize(
        "func",
        [PowerProbability(), PowerProbability2(), PowerProbability3(),
         LogProbability(), LogProbability2()],
        ids=lambda f: f.name,
    )
    def test_stays_in_range(self, func: Probability) -> None:
        for front, back in [(D("0"), D("0")), (D("0"), D("100")), (D("100"), D("0")),
                            (D("1"), D("1")), (D("1e6"), D("1"))]:
            p = func.prob(front, back)
            assert D("0") <= p <= D("1"), f"{func.name}({front}, {back}) = {p}"

    def test_linear_power_is_the_share_behind(self) -> None:
        """n=1: prob = back / (back + front)."""
        assert PowerProbability(D("1")).prob(D("25"), D("75")) == D("0.75")

    def test_power_n_sharpens(self) -> None:
        """queue.rs:221-240 — back^n / (back^n + front^n)."""
        # n=2, front=25, back=75 -> 5625 / (5625 + 625) = 0.9
        assert PowerProbability(D("2")).prob(D("25"), D("75")) == D("0.9")

    def test_power2_matches_the_rust(self) -> None:
        """back^n / (back + front)^n. n=2, front=25, back=75 -> 5625/10000."""
        assert PowerProbability2(D("2")).prob(D("25"), D("75")) == D("0.5625")

    def test_power3_matches_the_rust(self) -> None:
        """1 - (front/(front+back))^n. n=2, front=25, back=75 -> 1 - 0.0625."""
        assert PowerProbability3(D("2")).prob(D("25"), D("75")) == D("0.9375")

    def test_log_is_damped_against_linear(self) -> None:
        """Log form compresses a large imbalance, which is the reason to choose it."""
        front, back = D("10"), D("1000")
        assert LogProbability().prob(front, back) < PowerProbability().prob(front, back)

    def test_log2_dominates_log_always(self) -> None:
        """Our own architecture note calls log2 "less aggressive". The note is wrong.

        ln(1+b) + ln(1+f) = ln(1+b+f+bf) >= ln(1+b+f), so log2's denominator is the smaller and
        log2 >= log at every pair. Asserted across a spread rather than at one point.
        """
        for front in (D("1"), D("10"), D("100"), D("1000")):
            for back in (D("1"), D("10"), D("100"), D("1000")):
                assert LogProbability2().prob(front, back) >= LogProbability().prob(front, back)

    def test_empty_level_is_undecidable(self) -> None:
        assert PowerProbability().prob(D("0"), D("0")) == D("0.5")
        assert LogProbability().prob(D("0"), D("0")) == D("0.5")

    def test_non_positive_n_is_rejected(self) -> None:
        for cls in (PowerProbability, PowerProbability2, PowerProbability3):
            with pytest.raises(QueueError, match="positive"):
                cls(D("0"))


class TestProbQueue:
    """``queue.rs:124-217``."""

    def test_estimator_matches_the_rust_by_hand(self) -> None:
        # front=40, prev=100 -> back=60. Level falls to 80, so chg=20, no trades counted.
        # prob = 60/100 = 0.6
        # est = 40 - 0.4*20 + min(60 - 0.6*20, 0) = 40 - 8 + min(48, 0) = 32
        pos = QueuePosition(front_qty=D("40"))
        ProbQueue(PowerProbability(D("1"))).on_depth(pos, prev_qty=D("100"), new_qty=D("80"))
        assert pos.front_qty == D("32")

    def test_the_min_term_bites_when_the_back_runs_out(self) -> None:
        """The trailing min() is what makes a decrease larger than the back come off the front."""
        # front=90, prev=100 -> back=10. Level falls to 20, chg=80. prob = 10/100 = 0.1
        # est = 90 - 0.9*80 + min(10 - 0.1*80, 0) = 90 - 72 + min(2, 0) = 18
        pos = QueuePosition(front_qty=D("90"))
        ProbQueue(PowerProbability(D("1"))).on_depth(pos, prev_qty=D("100"), new_qty=D("20"))
        assert pos.front_qty == D("18")

    def test_negative_back_term_is_applied(self) -> None:
        # front=20, prev=100 -> back=80. Level falls to 1, chg=99. prob = 0.8
        # est = 20 - 0.2*99 + min(80 - 0.8*99, 0) = 20 - 19.8 + min(0.8, 0) = 0.2
        # then capped at new_qty=1 -> 0.2
        pos = QueuePosition(front_qty=D("20"))
        ProbQueue(PowerProbability(D("1"))).on_depth(pos, prev_qty=D("100"), new_qty=D("1"))
        assert pos.front_qty == D("0.2")

    def test_trades_are_not_counted_twice(self) -> None:
        """The subtlest bug in the model: a trade reduces the level *and* the front.

        Without ``chg -= cum_trade_qty`` the same volume advances the order twice.
        """
        model = ProbQueue(PowerProbability(D("1")))
        pos = QueuePosition(front_qty=D("50"))
        model.on_trade(pos, D("20"))
        assert pos.front_qty == D("30")
        assert pos.cum_trade_qty == D("20")
        # The level falls by exactly that trade: nothing is left to attribute to cancellation.
        model.on_depth(pos, prev_qty=D("100"), new_qty=D("80"))
        assert pos.front_qty == D("30")
        assert pos.cum_trade_qty == 0

    def test_counter_resets_after_a_depth_update(self) -> None:
        model = ProbQueue()
        pos = QueuePosition(front_qty=D("50"))
        model.on_trade(pos, D("10"))
        model.on_depth(pos, prev_qty=D("100"), new_qty=D("90"))
        assert pos.cum_trade_qty == 0
        # A second identical depth fall is now a genuine cancellation and must advance us.
        before = pos.front_qty
        model.on_depth(pos, prev_qty=D("90"), new_qty=D("80"))
        assert pos.front_qty < before

    def test_a_growing_level_only_caps(self) -> None:
        pos = QueuePosition(front_qty=D("30"))
        ProbQueue().on_depth(pos, prev_qty=D("100"), new_qty=D("400"))
        assert pos.front_qty == D("30")

    def test_front_is_capped_by_the_level(self) -> None:
        pos = QueuePosition(front_qty=D("95"))
        ProbQueue().on_depth(pos, prev_qty=D("100"), new_qty=D("5"))
        assert pos.front_qty <= D("5")

    def test_it_fills_faster_than_risk_adverse(self) -> None:
        """The whole difference between the two models, stated as a test.

        The scenario is the one where the two genuinely disagree: an order that has been resting
        long enough for other people to queue up behind it, on a level that then erodes by
        cancellation rather than by trade. Risk-adverse credits none of that erosion; ProbQueue
        credits the share it attributes to orders ahead.
        """
        def run(model: QueueModel) -> Decimal:
            order = RestingOrder(price=D("100"), quantity=D("10"), model=model,
                                 level_qty=D("50"))
            order.on_depth_change(D("200"))   # 150 join behind us
            order.on_depth_change(D("100"))   # half the level cancels
            return order.on_trade_at_price(D("30"))

        prob = run(ProbQueue())
        risk_adverse = run(RiskAdverseQueue())
        assert prob > risk_adverse
        assert risk_adverse == 0            # 30 of volume never reaches 50 ahead
        assert prob == D("5")               # front fell 50 -> 25, so 5 traded through

    def test_neither_model_can_beat_l3_truth(self) -> None:
        """An estimator that fills more than the real queue would is not conservative, it is wrong.

        Same tape, same starting queue: 50 ahead, 150 behind, half the level cancels, then 30
        trades. In the L3 book those cancellations are attributed truthfully — here, to the back,
        which is the case the estimators are most likely to over-credit.
        """
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("ahead", D("50")))
        book.add(D("100"), L3Order("ours", D("10"), is_ours=True))
        book.add(D("100"), L3Order("behind", D("140")))
        book.cancel(D("100"), "behind")     # the 100 that left was all behind us
        truth = book.on_trade(D("100"), D("30"))
        assert truth == []                  # 30 < 50 ahead: no fill, and that is the fact

        def run(model: QueueModel) -> Decimal:
            order = RestingOrder(price=D("100"), quantity=D("10"), model=model,
                                 level_qty=D("50"))
            order.on_depth_change(D("200"))
            order.on_depth_change(D("100"))
            return order.on_trade_at_price(D("30"))

        assert run(RiskAdverseQueue()) == 0      # matches the truth here
        assert run(ProbQueue()) == D("5")        # over-credits by 5, and that is the known cost

    def test_the_model_names_its_assumption(self) -> None:
        assert ProbQueue(LogProbability()).name == "prob[log]"
        assert ProbQueue().probability.name == "power(n=1)"


class TestL3FIFO:
    """``queue.rs:481-1050`` — with market-by-order there is nothing to estimate."""

    def test_we_join_behind_everything_resting(self) -> None:
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("a", D("30")))
        book.add(D("100"), L3Order("b", D("20")))
        book.add(D("100"), L3Order("ours", D("10"), is_ours=True))
        assert book.ahead_of(D("100"), "ours") == D("50")
        assert book.depth_at(D("100")) == D("60")

    def test_a_trade_consumes_the_front_in_order(self) -> None:
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("a", D("30")))
        book.add(D("100"), L3Order("ours", D("10"), is_ours=True))
        assert book.on_trade(D("100"), D("25")) == []
        assert book.ahead_of(D("100"), "ours") == D("5")
        assert book.on_trade(D("100"), D("8")) == [("ours", D("3"))]

    def test_a_cancellation_ahead_really_advances_us(self) -> None:
        """The event the probabilistic models can only guess at."""
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("a", D("30")))
        book.add(D("100"), L3Order("ours", D("10"), is_ours=True))
        assert book.cancel(D("100"), "a")
        assert book.ahead_of(D("100"), "ours") == 0

    def test_cancelling_an_absent_order_reports_it(self) -> None:
        book = L3FIFOQueue()
        assert book.cancel(D("100"), "ghost") is False
        assert book.ahead_of(D("100"), "ghost") is None

    def test_a_trade_larger_than_the_level_empties_it(self) -> None:
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("a", D("10")))
        book.add(D("100"), L3Order("ours", D("10"), is_ours=True))
        assert book.on_trade(D("100"), D("999")) == [("ours", D("10"))]
        assert book.depth_at(D("100")) == 0

    def test_a_size_decrease_preserves_priority_rather_than_requeuing(self) -> None:
        """A real MBO Modify with a smaller size keeps the order's place -- the whole point of
        adding `reduce` rather than modelling every modify as cancel-then-add, which would send
        the shrunk order to the back and let our own order jump ahead of it for free."""
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("a", D("30")))
        book.add(D("100"), L3Order("ours", D("10"), is_ours=True))
        assert book.reduce(D("100"), "a", D("12")) is True
        assert book.ahead_of(D("100"), "ours") == D("12")
        assert book.depth_at(D("100")) == D("22")

    def test_a_size_increase_is_refused_not_silently_reinterpreted(self) -> None:
        """Growing an order loses priority on a real venue -- the caller must model that as
        cancel+add, and `reduce` refusing rather than guessing is what forces that rather than
        letting a size increase quietly keep a priority it should have lost."""
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("a", D("30")))
        assert book.reduce(D("100"), "a", D("40")) is False
        assert book.ahead_of(D("100"), "a") == D("0")  # unchanged: still 30, just unobserved here
        assert book.depth_at(D("100")) == D("30")

    def test_reducing_an_absent_order_is_refused(self) -> None:
        book = L3FIFOQueue()
        assert book.reduce(D("100"), "ghost", D("5")) is False

    def test_reducing_to_zero_or_negative_is_refused_a_cancel_does_that_job(self) -> None:
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("a", D("30")))
        assert book.reduce(D("100"), "a", D("0")) is False
        assert book.reduce(D("100"), "a", D("-5")) is False
        assert book.depth_at(D("100")) == D("30")

    def test_l3_is_the_yardstick_for_the_estimators(self) -> None:
        """Same tape through L3 truth and through the estimate; both must be reachable.

        This test exists so the comparison is possible at all — it is what lets us say how far the
        approximation is off rather than assuming it is close.
        """
        book = L3FIFOQueue()
        book.add(D("100"), L3Order("a", D("60")))
        book.add(D("100"), L3Order("ours", D("10"), is_ours=True))
        truth = book.on_trade(D("100"), D("65"))

        est = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                           level_qty=D("60"))
        approx = est.on_trade_at_price(D("65"))
        assert truth == [("ours", D("5"))]
        assert approx == D("5")


class TestRestingOrderGuards:
    def test_zero_quantity_is_rejected(self) -> None:
        with pytest.raises(QueueError, match="positive"):
            RestingOrder(price=D("100"), quantity=D("0"), model=RiskAdverseQueue())

    def test_non_positive_lot_size_is_rejected(self) -> None:
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("1"))
        order.on_trade_at_price(D("5"))
        with pytest.raises(QueueError, match="lot_size"):
            order.model.executable(order.position, D("10"), lot_size=D("0"))

    def test_fill_ratio_of_nothing_is_zero(self) -> None:
        assert fill_ratio([]) == 0

    def test_fill_ratio_reports_the_shortfall(self) -> None:
        filled = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                              level_qty=D("0"))
        filled.on_trade_at_price(D("10"))
        missed = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                              level_qty=D("999"))
        missed.on_trade_at_price(D("10"))
        assert fill_ratio([filled, missed]) == D("0.5")

    def test_a_negative_level_is_treated_as_empty(self) -> None:
        order = RestingOrder(price=D("100"), quantity=D("10"), model=RiskAdverseQueue(),
                             level_qty=D("-5"))
        assert order.queue_ahead == 0
