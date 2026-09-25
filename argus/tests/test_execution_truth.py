"""Execution truth: fill confirmation, the consent gate, batch writes, shown ids, and the runner.

Every venue here is scripted and every clock simulated; nothing touches the network or Qwen. The
fault scenarios are the ones `eval/execution_truth.py` measures, pinned so a regression in any
mechanism fails a test rather than quietly lowering a number in an artefact.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.agents.desk import TradingDesk
from argus.decision.pause import AlreadyResolved, PauseStore
from argus.eval import execution_truth as et
from argus.eval.pausedrill import AT, ScriptedModel, answer_for, by_key, run_paused_half
from argus.execution import preflight
from argus.execution.batch import BatchResponse, ItemStatus, write_batch
from argus.execution.bitget_client import (
    DEMO_PRODUCT_TYPE,
    LIVE_PRODUCT_TYPE,
    BitgetOrderError,
    BitgetStatusSource,
    LiveOrderRefused,
)
from argus.execution.confirm import (
    ConfirmationError,
    FillVerdict,
    StatusReading,
    confirm_fill,
    confirm_in_book,
)
from argus.execution.consent import (
    CONSENT_ENV,
    ConsentRefused,
    LiveOrderConsent,
    consent_phrase,
    grant_live_consent,
    new_run_id,
    require_live_consent,
)
from argus.execution.grounded_ids import UnshownId, show, show_orders, show_positions
from argus.execution.orders import IllegalTransition, Order, OrderBook, OrderState
from argus.market.bitget import Ticker
from argus.paper import chains, runner
from argus.paper import ledger as ledger_module
from argus.paper.ledger import LedgerError, LedgerLockTimeout, PaperLedger, SettleRequest
from argus.paper.venue import (
    CEILING_FLAG,
    FILL_FLAG,
    LedgerVenue,
    confirm_recorded,
    paper_order_id,
    position_or_none,
)
from argus.truth.clocks import SessionPhase, SessionState

T0 = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
HALTED_APPROVE = by_key("underlying_halted/approve")


@pytest.fixture(autouse=True)
def _no_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BITGET_QWEN_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_depth_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_executable_spread_bps` reaches for the live book; every test here prices at the quote."""
    from argus.market import depth

    monkeypatch.setattr(depth, "measured_spread_bps", lambda *a, **k: None)


def _confirm(scenario: et.FillScenario, *, deadline_s: float = 30.0) -> Any:
    clock = et.SimClock()
    venue = et.ScriptedVenue(clock, scenario.status, scenario.position)
    got = confirm_fill(
        status=venue, position=venue, client_order_id="c", symbol="SIMUSDT", side="BUY",
        quantity=et.Q, position_before=Decimal("0"), deadline_s=deadline_s,
        clock=clock.now, sleep=clock.sleep,
    )
    return got, clock, venue


# --- fill confirmation ---------------------------------------------------------------------------


class TestFillConfirmation:
    @pytest.mark.parametrize("scenario", et.FILL_SCENARIOS, ids=lambda s: s.key)
    def test_every_injected_fault_lands_in_its_true_class(self, scenario: et.FillScenario) -> None:
        got, _, _ = _confirm(scenario)
        assert et._OURS[got.verdict] == scenario.truth, got.render()

    def test_a_clean_fill_is_believed_after_one_re_read_one_window_later(self) -> None:
        got, clock, venue = _confirm(et.FILL_SCENARIOS[0])
        assert got.verdict is FillVerdict.CONFIRMED
        assert (len(got.observations), clock.now(), venue.status_reads) == (2, 1.0, 2)

    def test_an_agreement_on_one_read_that_breaks_on_the_next_is_not_believed(self) -> None:
        """The case that made the stability re-read necessary: rejected at once, position booked
        half a second later. One read agrees on "nothing happened"; the re-read does not."""
        exposed = next(s for s in et.FILL_SCENARIOS if s.key == "rejected_but_exposed")
        got, _, _ = _confirm(exposed)
        assert got.verdict is FillVerdict.DISAGREED
        agreed_once = [o for o in got.observations if o.note.startswith("settled")]
        assert agreed_once, "the transient agreement is in the trace, not hidden"

    def test_with_no_stability_window_one_agreeing_read_settles(self) -> None:
        clock = et.SimClock()
        venue = et.ScriptedVenue(clock, ((0.0, et._st(OrderState.FILLED, "2")),),
                                 ((0.0, Decimal("2")),))
        got = confirm_fill(status=venue, position=venue, client_order_id="c", symbol="S",
                           side="BUY", quantity=et.Q, position_before=Decimal("0"),
                           stable_for_s=0.0, clock=clock.now, sleep=clock.sleep)
        assert (got.verdict, len(got.observations), clock.now()) == (FillVerdict.CONFIRMED, 1, 0.0)

    def test_an_agreement_seen_only_at_the_deadline_is_not_believed(self) -> None:
        clock = et.SimClock()
        venue = et.ScriptedVenue(clock, ((0.0, et._st(OrderState.FILLED, "2")),),
                                 ((0.0, Decimal("2")),))
        got = confirm_fill(status=venue, position=venue, client_order_id="c", symbol="S",
                           side="BUY", quantity=et.Q, position_before=Decimal("0"),
                           deadline_s=0.0, clock=clock.now, sleep=clock.sleep)
        assert got.verdict is FillVerdict.UNSETTLED

    def test_a_late_settle_is_polled_until_it_settles_not_until_the_deadline(self) -> None:
        late = next(s for s in et.FILL_SCENARIOS if s.key == "late_settle")
        got, clock, _ = _confirm(late)
        assert got.verdict is FillVerdict.CONFIRMED
        assert 3.0 <= clock.now() < 5.0, "settled when the position booked, not at 30s"

    def test_the_poll_never_sleeps_past_the_deadline(self) -> None:
        working = next(s for s in et.FILL_SCENARIOS if s.key == "still_working")
        got, clock, _ = _confirm(working, deadline_s=10.0)
        assert got.verdict is FillVerdict.UNSETTLED
        assert clock.now() == pytest.approx(10.0)
        assert got.observations[-1].elapsed_s == pytest.approx(10.0)

    def test_the_backoff_grows_and_is_capped(self) -> None:
        clock = et.SimClock()
        waits: list[float] = []

        def sleep(s: float) -> None:
            waits.append(s)
            clock.sleep(s)

        venue = et.ScriptedVenue(clock, ((0.0, et._st(OrderState.ACCEPTED)),),
                                 ((0.0, Decimal("0")),))
        confirm_fill(status=venue, position=venue, client_order_id="c", symbol="S", side="BUY",
                     quantity=et.Q, position_before=Decimal("0"), deadline_s=20.0,
                     clock=clock.now, sleep=sleep)
        assert waits[:5] == [0.25, 0.5, 1.0, 2.0, 4.0]
        assert max(waits) == 4.0

    def test_no_record_and_no_movement_is_never_a_confirmed_non_fill(self) -> None:
        """SWE-bench's rule: no parsed result and no sign the suite ran is invalid, not a pass."""
        empty = next(s for s in et.FILL_SCENARIOS if s.key == "empty_record")
        got, _, _ = _confirm(empty)
        assert got.verdict is FillVerdict.NO_EVIDENCE
        assert got.order_state is OrderState.UNKNOWN

    def test_rejected_while_the_position_moved_is_a_disagreement(self) -> None:
        exposed = next(s for s in et.FILL_SCENARIOS if s.key == "rejected_but_exposed")
        got, _, _ = _confirm(exposed)
        assert got.verdict is FillVerdict.DISAGREED
        assert got.order_state is OrderState.DISAGREED

    def test_without_a_position_source_a_final_status_stops_and_is_uncorroborated(self) -> None:
        clock = et.SimClock()
        venue = et.ScriptedVenue(clock, ((0.0, et._st(OrderState.FILLED, "2")),),
                                 ((0.0, Decimal("2")),))
        got = confirm_fill(status=venue, position=None, client_order_id="c", symbol="S",
                           side="BUY", quantity=et.Q, position_before=None, deadline_s=30.0,
                           clock=clock.now, sleep=clock.sleep)
        assert got.verdict is FillVerdict.UNCORROBORATED
        assert len(got.observations) == 1
        assert got.order_state is OrderState.UNKNOWN

    def test_a_zero_order_that_both_records_show_filled_is_an_overfill(self) -> None:
        clock = et.SimClock()
        venue = et.ScriptedVenue(clock, ((0.0, et._st(OrderState.FILLED, "1")),),
                                 ((0.0, Decimal("-1")),))
        got = confirm_fill(status=venue, position=venue, client_order_id="c", symbol="S",
                           side="SELL", quantity=Decimal("0"), position_before=Decimal("0"),
                           deadline_s=0.0, stable_for_s=0.0, clock=clock.now, sleep=clock.sleep)
        assert got.verdict is FillVerdict.OVERFILLED

    def test_an_unknown_side_is_refused_not_guessed(self) -> None:
        venue = et.ScriptedVenue(et.SimClock(), ((0.0, et._st(OrderState.FILLED, "2")),),
                                 ((0.0, Decimal("2")),))
        with pytest.raises(ConfirmationError, match="neither a buy nor a sell"):
            confirm_fill(status=venue, position=venue, client_order_id="c", symbol="S",
                         side="LONG_ISH", quantity=et.Q, position_before=Decimal("0"))


class TestTheBookRecordsTheVerdict:
    def _book(self) -> tuple[OrderBook, Order]:
        book = OrderBook()
        order = Order("c-1", "SIMUSDT", "BUY", Decimal("2"), "hash")
        book.submit(et._authorised(order), at=T0)
        return book, order

    def test_a_disagreement_is_its_own_live_state_and_adopts_neither_number(self) -> None:
        book, order = self._book()
        clock = et.SimClock()
        venue = et.ScriptedVenue(clock, ((0.0, et._st(OrderState.FILLED, "2")),),
                                 ((0.0, Decimal("1")),))
        got = confirm_in_book(book, client_order_id="c-1", status=venue, position=venue,
                              position_before=Decimal("0"), at=T0, deadline_s=5.0,
                              clock=clock.now, sleep=clock.sleep)
        assert got.verdict is FillVerdict.DISAGREED
        assert order.state is OrderState.DISAGREED
        assert order.filled_quantity == 0
        assert order in book.live_orders() and order in book.needs_reconciliation()

    def test_a_confirmed_fill_adopts_the_agreed_quantity(self) -> None:
        book, order = self._book()
        clock = et.SimClock()
        venue = et.ScriptedVenue(clock, ((0.0, et._st(OrderState.FILLED, "2")),),
                                 ((0.0, Decimal("2")),))
        confirm_in_book(book, client_order_id="c-1", status=venue, position=venue,
                        position_before=Decimal("0"), at=T0, clock=clock.now, sleep=clock.sleep)
        assert (order.state, order.filled_quantity) == (OrderState.FILLED, Decimal("2"))

    def test_an_expiry_seen_from_submitted_records_the_inferred_acceptance(self) -> None:
        book, order = self._book()
        book.apply_confirmation("c-1", target=OrderState.EXPIRED, filled=Decimal("0"), at=T0,
                                reason="venue expired it")
        assert [t.to for t in order.history][-2:] == [OrderState.ACCEPTED, OrderState.EXPIRED]
        assert order.history[-2].reason.startswith("inferred")

    def test_two_live_orders_on_one_symbol_cannot_be_attributed(self) -> None:
        book, _ = self._book()
        book.submit(et._authorised(Order("c-2", "SIMUSDT", "SELL", Decimal("1"), "h")), at=T0)
        venue = et.ScriptedVenue(et.SimClock(), ((0.0, et._st(OrderState.FILLED, "2")),),
                                 ((0.0, Decimal("1")),))
        with pytest.raises(ConfirmationError, match="cannot be attributed"):
            confirm_in_book(book, client_order_id="c-1", status=venue, position=venue,
                            position_before=Decimal("0"), at=T0)

    def test_disagreed_leaves_only_by_reconciliation(self) -> None:
        book, order = self._book()
        book.apply_confirmation("c-1", target=OrderState.DISAGREED, filled=None, at=T0,
                                reason="records clash")
        with pytest.raises(IllegalTransition):
            order.transition(OrderState.SUBMITTED, at=T0, reason="resend")
        book.reconcile("c-1", venue_state=OrderState.FILLED, at=T0, venue_filled=Decimal("2"))
        assert order.state is OrderState.FILLED

    def test_an_overfill_is_never_adopted(self) -> None:
        book, _ = self._book()
        with pytest.raises(ValueError, match="never adopted"):
            book.apply_confirmation("c-1", target=OrderState.FILLED, filled=Decimal("3"), at=T0,
                                    reason="x")


# --- the consent gate ----------------------------------------------------------------------------


@contextmanager
def _credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE"):
        monkeypatch.setenv(name, f"test-{name.lower()}")
    yield


def _order(oid: str = "argus-c-1") -> Order:
    return Order(oid, "SBTCSUSDT", "BUY", Decimal("0.01"), "fixture-hash")


class TestConsentGate:
    def test_the_default_is_refusal(self, tmp_path: Path) -> None:
        with pytest.raises(ConsentRefused, match="none was given"):
            grant_live_consent("run-1", env={}, spent_path=tmp_path / "s.jsonl")

    def test_the_exact_phrase_for_this_run_grants_once(self, tmp_path: Path) -> None:
        spent = tmp_path / "s.jsonl"
        consent = grant_live_consent(
            "run-1", env={CONSENT_ENV: consent_phrase("run-1")}, spent_path=spent,
        )
        assert consent.run_id == "run-1"
        with pytest.raises(ConsentRefused, match="already used"):
            grant_live_consent("run-1", token=consent_phrase("run-1"), spent_path=spent)

    def test_a_phrase_for_another_run_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConsentRefused, match="does not name run"):
            grant_live_consent("run-2", token=consent_phrase("run-1"),
                               spent_path=tmp_path / "s.jsonl")

    def test_true_is_not_consent(self, tmp_path: Path) -> None:
        """MLE-bench accepts `true`; a per-run gate must not."""
        with pytest.raises(ConsentRefused):
            grant_live_consent("run-3", token="true", spent_path=tmp_path / "s.jsonl")

    def test_a_consent_cannot_be_forged(self) -> None:
        with pytest.raises(ConsentRefused, match="cannot be constructed directly"):
            LiveOrderConsent("run-x", T0, T0 + timedelta(hours=1), _token=object())

    def test_an_expired_consent_is_refused(self, tmp_path: Path) -> None:
        consent = grant_live_consent("run-4", token=consent_phrase("run-4"), now=T0,
                                     spent_path=tmp_path / "s.jsonl")
        require_live_consent(consent, now=T0 + timedelta(minutes=5))
        with pytest.raises(ConsentRefused, match="expired"):
            require_live_consent(consent, now=T0 + timedelta(minutes=16))

    def test_a_run_id_with_whitespace_is_not_a_run_id(self) -> None:
        with pytest.raises(ConsentRefused):
            consent_phrase("run 5")

    def test_new_run_ids_differ(self) -> None:
        assert new_run_id(T0) != new_run_id(T0)

    def test_a_live_client_without_consent_sends_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _credentials(monkeypatch):
            live = et.RecordingClient(paper_trading=False)
            with pytest.raises(LiveOrderRefused) as caught:
                live.place_order(et._authorised(_order()))
        assert isinstance(caught.value, BitgetOrderError)
        assert live.sent == []

    def test_a_live_client_with_consent_sends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        consent = grant_live_consent("run-6", token=consent_phrase("run-6"),
                                     spent_path=tmp_path / "s.jsonl")
        with _credentials(monkeypatch):
            live = et.RecordingClient(paper_trading=False, live_consent=consent)
            live.place_order(et._authorised(_order()))
        assert any("place-order" in r for r in live.sent)

    def test_paper_needs_no_consent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with _credentials(monkeypatch):
            paper = et.RecordingClient(paper_trading=True)
            paper.place_order(et._authorised(_order()))
        assert paper.trades_real_money is False
        assert any("place-order" in r for r in paper.sent)

    def test_paper_mode_on_a_live_product_type_cannot_be_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The corner `trades_real_money` used to answer False for while sending real orders."""
        with _credentials(monkeypatch), pytest.raises(ValueError, match="real-money orders"):
            et.RecordingClient(paper_trading=True, product_type=LIVE_PRODUCT_TYPE)

    def test_the_predicate_follows_the_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with _credentials(monkeypatch):
            assert et.RecordingClient(
                paper_trading=False, product_type=DEMO_PRODUCT_TYPE,
            ).trades_real_money is False
            assert et.RecordingClient(
                paper_trading=True, product_type=LIVE_PRODUCT_TYPE, legacy_paptrading=True,
            ).trades_real_money is False
            assert et.RecordingClient(paper_trading=False).trades_real_money is True

    def test_the_measured_consent_scenarios_all_hold(self, tmp_path: Path) -> None:
        got = et.run_consent_scenarios(tmp_path / "s.jsonl")
        assert got["correct"] == got["scenarios"]
        assert got["real_money_orders_sent_without_valid_consent"] == 0


class TestBitgetStatusSource:
    def test_an_empty_detail_is_absence_not_a_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with _credentials(monkeypatch):
            client = et.RecordingClient(paper_trading=True)
        monkeypatch.setattr(client, "order_status", lambda **k: {})
        got = BitgetStatusSource(client).order_status("c", symbol="SBTCSUSDT")
        assert got.state is None

    def test_a_filled_detail_maps_with_its_volume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with _credentials(monkeypatch):
            client = et.RecordingClient(paper_trading=True)
        monkeypatch.setattr(client, "order_status",
                            lambda **k: {"state": "filled", "baseVolume": "0.01"})
        got = BitgetStatusSource(client).order_status("c", symbol="SBTCSUSDT")
        assert (got.state, got.filled) == (OrderState.FILLED, Decimal("0.01"))

    def test_the_preflight_probe_polls_past_an_unindexed_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single read after placement failed the probe on timing; the poll does not."""
        with _credentials(monkeypatch):
            client = et.RecordingClient(paper_trading=True)
        details = iter([{}, {}, {"state": "filled", "baseVolume": "0.01"}])
        monkeypatch.setattr(client, "order_status", lambda **k: next(details))
        monkeypatch.setattr(preflight, "ROUND_TRIP_DEADLINE_S", 5.0)
        step = preflight._order_round_trip(client, symbol="SBTCSUSDT", size=Decimal("0.01"))
        assert step.passed is True
        assert "uncorroborated" in step.detail


# --- batch writes --------------------------------------------------------------------------------


class TestWriteBatch:
    @pytest.mark.parametrize("scenario", et.BATCH_SCENARIOS, ids=lambda s: s.key)
    def test_no_duplicate_and_no_misreport_in_any_scenario(
        self, scenario: et.BatchScenario
    ) -> None:
        venue = scenario.venue()
        reported = et._ours_batch(scenario.ids, venue)
        assert all(n == 1 for n in venue.placements.values()), venue.placements
        assert et._batch_outcome_errors(reported, venue) == 0

    def test_a_partial_landing_is_probed_and_the_rest_sent_singly(self) -> None:
        venue = et.BatchVenue(raise_after=3, single_refuses={"o5": "insufficient margin"})
        report = write_batch(["o1", "o2", "o3", "o4", "o5", "o6"], key=str,
                             batch=venue.place_many, single=venue.place_one,
                             applied=venue.holds)
        statuses = {o.key: o.status for o in report.outcomes}
        assert statuses == {
            "o1": ItemStatus.ALREADY_APPLIED, "o2": ItemStatus.ALREADY_APPLIED,
            "o3": ItemStatus.ALREADY_APPLIED, "o4": ItemStatus.WRITTEN_SINGLY,
            "o5": ItemStatus.FAILED, "o6": ItemStatus.WRITTEN_SINGLY,
        }
        assert report.fell_back and report.failed_keys == ("o5",)

    def test_a_store_that_cannot_answer_the_probe_gets_nothing_resent(self) -> None:
        venue = et.BatchVenue(raise_after=2, probe_answers=False)
        report = write_batch(["o1", "o2", "o3"], key=str, batch=venue.place_many,
                             single=venue.place_one, applied=venue.holds)
        assert {o.status for o in report.outcomes} == {ItemStatus.UNKNOWN}
        assert venue.placements == {"o1": 1, "o2": 1}

    def test_the_response_s_failure_list_is_final_and_named(self) -> None:
        venue = et.BatchVenue(fail_in_response={"o2": "40762 insufficient margin"})
        report = write_batch(["o1", "o2"], key=str, batch=venue.place_many,
                             single=venue.place_one, applied=venue.holds)
        assert report.failed_keys == ("o2",)
        assert "40762" in report.failed[0].detail
        assert venue.placements == {"o1": 1}

    def test_validation_and_duplicates_are_refused_before_anything_is_sent(self) -> None:
        sent: list[list[str]] = []

        def batch(items: Any) -> BatchResponse:
            sent.append(list(items))
            return BatchResponse(succeeded=frozenset(items))

        report = write_batch(["a", "bad", "a"], key=str, batch=batch, single=lambda _: None,
                             validate=lambda k: "not allowed" if k == "bad" else None)
        assert sent == [["a"]]
        assert [o.status for o in report.outcomes] == [
            ItemStatus.WRITTEN_IN_BATCH, ItemStatus.REFUSED, ItemStatus.REFUSED,
        ]

    def test_keys_the_store_invented_are_recorded(self) -> None:
        report = write_batch(["a"], key=str, single=lambda _: None,
                             batch=lambda items: BatchResponse(succeeded=frozenset({"a", "z"})))
        assert report.unexpected == ("z",)


# --- shown ids -----------------------------------------------------------------------------------


class TestShownIds:
    def test_the_model_sees_short_ids_and_no_real_ones(self) -> None:
        book, shown = et.fixture_book(T0)
        text = shown.render()
        assert "[1]" in text and "[3]" in text
        assert not any(o.client_order_id in text for o in book.live_orders())

    def test_shown_ids_resolve_to_the_real_orders(self) -> None:
        _, shown = et.fixture_book(T0)
        assert shown.resolve("[2]") == shown.rows[1].target
        assert shown.resolve_all([1, " 3 "]) == (shown.rows[0].target, shown.rows[2].target)

    @pytest.mark.parametrize("token", ["7", "0", True, "1a", "", 1.5, "#2", None])
    def test_anything_not_shown_is_refused(self, token: object) -> None:
        _, shown = et.fixture_book(T0)
        with pytest.raises(UnshownId):
            shown.resolve(token)

    def test_a_real_id_even_of_a_shown_order_is_refused_and_says_why(self) -> None:
        _, shown = et.fixture_book(T0)
        with pytest.raises(UnshownId, match="real identifier"):
            shown.resolve(shown.rows[0].target)

    def test_an_order_that_exists_but_was_not_shown_is_refused(self) -> None:
        book, shown = et.fixture_book(T0)
        hidden = next(o for o in book.live_orders() if o.symbol == "COINUSDT")
        with pytest.raises(UnshownId):
            shown.resolve(hidden.client_order_id)

    def test_one_bad_token_refuses_the_whole_action_and_names_it(self) -> None:
        _, shown = et.fixture_book(T0)
        with pytest.raises(UnshownId) as caught:
            shown.resolve_all(["1", "9", "2", "2"])
        assert set(caught.value.bad) == {"9", "2"}

    def test_the_measured_scenarios_show_no_unsafe_action(self) -> None:
        got = et.run_id_scenarios()
        ours = got["by_method"]["argus_resolve_all"]
        assert (ours["correct"], ours["unsafe"]) == (got["scenarios"], 0)
        assert got["by_method"]["real_ids_any_in_book"]["unsafe"] > 0

    def test_positions_omit_flat_symbols_and_carry_direction(self) -> None:
        shown = show_positions({"NVDAUSDT": Decimal("2"), "TSLAUSDT": Decimal("0"),
                                "COINUSDT": Decimal("-1")})
        assert [r.line for r in shown.rows] == ["SHORT 1 COINUSDT", "LONG 2 NVDAUSDT"]
        assert shown.resolve("2") == "NVDAUSDT"

    def test_one_real_id_twice_in_a_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            show("x", ["a", "a"], target=str, describe=str)

    def test_the_digest_names_the_rendering(self) -> None:
        book, shown = et.fixture_book(T0)
        again = show_orders(o for o in book.live_orders() if o.symbol in ("NVDAUSDT", "TSLAUSDT"))
        other = show_orders(book.live_orders())
        assert shown.digest == again.digest != other.digest


# --- the paper venue, the batch settlement and the runner ----------------------------------------


def _ticker(symbol: str, last: str, at: datetime = T0) -> Ticker:
    price = Decimal(last)
    return Ticker(
        symbol=symbol, last=price, bid=price * Decimal("0.999"), ask=price * Decimal("1.001"),
        high_24h=price, low_24h=price, change_24h=Decimal("0"), base_volume=Decimal("1000"),
        funding_rate=Decimal("0"), fetched_at=at,
    )


def _record(led: PaperLedger, symbol: str, *, qty: str, side: str = "BUY", px: str = "100",
            at: datetime = T0) -> Any:
    return led.record(
        symbol=symbol, verdict="trade" if Decimal(qty) > 0 else "no_trade", side=side,
        quantity=Decimal(qty), entry_price=Decimal(px), stated_confidence=0.6, thesis="t",
        invalidation=("x",), market_state_hash="m" * 8, approved_intent_hash="a" * 8,
        session_phase="rth", hours_to_discovery=0.0, decided_at=at,
    )


class TestLedgerVenue:
    def test_a_recorded_position_is_confirmed_by_row_and_book(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "l.jsonl")
        venue = LedgerVenue(led.path)
        before = position_or_none(venue, "NVDAUSDT")
        entry = _record(led, "NVDAUSDT", qty="2", side="SELL")
        check = confirm_recorded(
            seq=entry.seq, symbol="NVDAUSDT", side="SELL", quantity=Decimal("2"), status=venue,
            position=venue, position_before=before, authorised_quantity=Decimal("2"),
            authorised_side="sell",
        )
        assert check.confirmation.verdict is FillVerdict.CONFIRMED
        assert check.confirmation.position_delta == Decimal("-2")
        assert not check.alarm

    def test_the_seq_264_shape_is_caught_against_its_ruling(self, tmp_path: Path) -> None:
        """Row and book agree on a position the Constitution had refused."""
        led = PaperLedger(path=tmp_path / "l.jsonl")
        venue = LedgerVenue(led.path)
        entry = _record(led, "COINUSDT", qty="1", side="SELL")
        check = confirm_recorded(
            seq=entry.seq, symbol="COINUSDT", side="SELL", quantity=Decimal("1"), status=venue,
            position=venue, position_before=Decimal("0"), authorised_quantity=Decimal("0"),
            authorised_side="sell",
        )
        assert check.alarm and check.within_authorisation is False
        assert CEILING_FLAG in check.render()
        assert runner._is_flag(check.render())

    def test_a_duplicated_sequence_number_is_not_confirmed(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "l.jsonl")
        entry = _record(led, "NVDAUSDT", qty="0")
        line = led.path.read_text(encoding="utf-8").splitlines()[0]
        with led.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        venue = LedgerVenue(led.path)
        check = confirm_recorded(
            seq=entry.seq, symbol="NVDAUSDT", side="BUY", quantity=Decimal("0"), status=venue,
            position=venue, position_before=Decimal("0"), authorised_quantity=None,
            authorised_side=None, deadline_s=0.0,
        )
        assert check.alarm
        assert FILL_FLAG in check.render() and runner._is_flag(check.render())

    def test_an_abstention_is_confirmed_and_not_flagged(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "l.jsonl")
        venue = LedgerVenue(led.path)
        entry = _record(led, "NVDAUSDT", qty="0")
        check = confirm_recorded(
            seq=entry.seq, symbol="NVDAUSDT", side="BUY", quantity=Decimal("0"), status=venue,
            position=venue, position_before=Decimal("0"), authorised_quantity=Decimal("0"),
            authorised_side="buy",
        )
        assert check.confirmation.verdict is FillVerdict.CONFIRMED_UNFILLED
        assert not check.alarm and not runner._is_flag(check.render())

    def test_a_torn_last_line_fails_the_read_rather_than_the_row(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "l.jsonl")
        _record(led, "NVDAUSDT", qty="0")
        with led.path.open("a", encoding="utf-8") as fh:
            fh.write('{"seq": 2, "symbol": "NVD')
        assert position_or_none(LedgerVenue(led.path), "NVDAUSDT") is None

    def test_a_status_for_another_symbol_is_absence(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "l.jsonl")
        entry = _record(led, "NVDAUSDT", qty="1")
        got = LedgerVenue(led.path).order_status(paper_order_id(entry.seq), symbol="TSLAUSDT")
        assert got.state is None


def _due_ledger(tmp_path: Path, name: str) -> PaperLedger:
    led = PaperLedger(path=tmp_path / name)
    _record(led, "NVDAUSDT", qty="0", px="100")
    _record(led, "TSLAUSDT", qty="2", side="SELL", px="50")
    _record(led, "AAPLUSDT", qty="0", px="200")
    _record(led, "MSFTUSDT", qty="1", side="BUY", px="400")
    return led


def _requests(at: datetime) -> list[SettleRequest]:
    return [
        SettleRequest(seq=1, price=Decimal("101"), settled_at=at),
        SettleRequest(seq=2, price=Decimal("49"), settled_at=at,
                      exit_spread_bps=Decimal("3.5")),
        SettleRequest(seq=3, price=Decimal("198"), settled_at=at),
        SettleRequest(seq=4, price=Decimal("404"), settled_at=at,
                      exit_spread_bps=Decimal("1.1")),
    ]


class TestSettleBatch:
    def test_a_batch_is_byte_identical_to_the_same_settlements_one_at_a_time(
        self, tmp_path: Path
    ) -> None:
        at = T0 + timedelta(hours=25)
        batched = _due_ledger(tmp_path, "batch.jsonl")
        singly = _due_ledger(tmp_path, "single.jsonl")
        result = batched.settle_batch(_requests(at))
        assert result.report.all_written and not result.report.fell_back
        for request in _requests(at):
            if singly.entries[request.seq - 1].is_abstention:
                singly.settle_abstention(request.seq, price_now=request.price, settled_at=at)
            else:
                singly.settle(request.seq, exit_price=request.price, settled_at=at,
                              exit_spread_bps=request.exit_spread_bps)
        assert batched.path.read_bytes() == singly.path.read_bytes()
        assert batched.verify()["chain_intact"] is True
        head = json.loads(batched.path.with_name("batch.jsonl.head").read_text(encoding="utf-8"))
        assert head["raw_entries"] == 8

    def test_a_poisoned_row_is_refused_and_named_and_the_rest_settle(
        self, tmp_path: Path
    ) -> None:
        led = _due_ledger(tmp_path, "l.jsonl")
        rows = led.path.read_text(encoding="utf-8").splitlines()
        poisoned = json.loads(rows[1])
        poisoned["side"] = "LONG"  # outside the hashed payload; a real upstream defect's shape
        rows[1] = json.dumps(poisoned)
        led.path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        led = PaperLedger(path=led.path)
        result = led.settle_batch(_requests(T0 + timedelta(hours=25)))
        assert result.report.failed_keys == ("2",)
        assert "expected BUY or SELL" in result.report.failed[0].detail
        assert [e.seq for e in result.settled] == [1, 3, 4]

    def test_a_row_already_settled_is_refused_not_settled_twice(self, tmp_path: Path) -> None:
        led = _due_ledger(tmp_path, "l.jsonl")
        at = T0 + timedelta(hours=25)
        led.settle_abstention(1, price_now=Decimal("101"), settled_at=at)
        result = led.settle_batch(_requests(at))
        assert result.report.failed_keys == ("1",)
        assert [o.status for o in result.report.failed] == [ItemStatus.REFUSED]

    def test_a_failed_batch_write_falls_back_to_one_at_a_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        led = _due_ledger(tmp_path, "l.jsonl")
        real = ledger_module._exclusive
        calls = {"n": 0}

        @contextmanager
        def flaky(path: Path) -> Iterator[None]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise LedgerLockTimeout("another writer held the lock")
            with real(path):
                yield

        monkeypatch.setattr(ledger_module, "_exclusive", flaky)
        result = led.settle_batch(_requests(T0 + timedelta(hours=25)))
        assert result.report.fell_back
        assert {o.status for o in result.report.outcomes} == {ItemStatus.WRITTEN_SINGLY}
        assert PaperLedger(path=led.path).verify()["chain_intact"] is True

    def test_an_unknown_seq_is_refused(self, tmp_path: Path) -> None:
        led = _due_ledger(tmp_path, "l.jsonl")
        result = led.settle_batch([SettleRequest(seq=99, price=Decimal("1"), settled_at=T0)])
        assert result.report.failed_keys == ("99",)
        with pytest.raises(LedgerError):
            led.settle(99, exit_price=Decimal("1"))


class TestRunnerSettlement:
    def test_a_poisoned_row_no_longer_stops_the_cycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(chains, "CHAINS_PATH", tmp_path / "chains.jsonl")
        led = _due_ledger(tmp_path, "l.jsonl")
        rows = led.path.read_text(encoding="utf-8").splitlines()
        poisoned = json.loads(rows[3])
        poisoned["side"] = "SIDEWAYS"
        rows[3] = json.dumps(poisoned)
        led.path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        led = PaperLedger(path=led.path)
        tickers = {s: _ticker(s, p, T0 + timedelta(hours=25)) for s, p in (
            ("NVDAUSDT", "101"), ("TSLAUSDT", "49"), ("AAPLUSDT", "198"), ("MSFTUSDT", "404"),
        )}
        out = runner._settle_due(led, tickers, T0 + timedelta(hours=25))
        failed = [r for r in out if "settlement_failed" in r]
        assert [r["seq"] for r in failed] == [4]
        assert sorted(int(str(r["seq"])) for r in out if "settlement_failed" not in r
                      and "seq" in r) == [1, 2, 3]


def _session(at: datetime) -> SessionState:
    return SessionState(phase=SessionPhase.RTH, as_of=at, hours_to_next_discovery=0.0,
                        nav_age_seconds=5.0)


@pytest.fixture
def _sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(runner, "NOTES_PATH", tmp_path / "desk_notes.jsonl")
    monkeypatch.setattr(runner, "RISK_PATH", tmp_path / "risk_records.jsonl")
    monkeypatch.setattr(chains, "CHAINS_PATH", tmp_path / "chains.jsonl")
    return tmp_path


def _held(tmp_path: Path) -> tuple[PauseStore, str]:
    store = PauseStore(tmp_path / "pauses")
    run, _ = run_paused_half(HALTED_APPROVE, store, decision_id="paper-NVDAUSDT-1")
    assert run.pause is not None
    return store, run.pause.request_id


def _desk() -> TradingDesk:
    return TradingDesk(ScriptedModel(quantity=HALTED_APPROVE.quantity,
                                     thesis=HALTED_APPROVE.thesis))


class TestTheHumanLoopInTheCycle:
    def test_an_answered_hold_is_resumed_and_booked_like_a_fresh_decision(
        self, _sidecars: Path
    ) -> None:
        store, request_id = _held(_sidecars)
        store.answer(answer_for(HALTED_APPROVE, request_id), now=AT + timedelta(minutes=4))
        now = AT + timedelta(minutes=5)
        runs, log = runner._resume_answered(_desk(), store, now=now, priced={"NVDAUSDT"})
        assert [(e["request_id"], e["outcome"]) for e in log] == [(request_id, "resumed")]
        led = PaperLedger(path=_sidecars / "ledger.jsonl")
        row = runner._record_run(
            runs[0], symbol="NVDAUSDT", ticker=_ticker("NVDAUSDT", "200", now),
            session=_session(now), ledger=led, commitments=(), guard=None, guard_note="",
            now=now,
        )
        assert row["pause"] == {"request_id": request_id, "state": "resumed",
                                "reviewer": "drill-reviewer", "action": "approve"}
        assert row["fill"] == "confirmed" and row["fill_alarm"] is False
        assert Decimal(led.entries[0].quantity) > 0
        notes = [json.loads(x) for x in (_sidecars / "desk_notes.jsonl").read_text(
            encoding="utf-8").splitlines()]
        assert any(n.startswith("[human]") for n in notes[0]["notes"])
        assert any(n.startswith("[fill]") for n in notes[0]["notes"])
        assert (_sidecars / "risk_records.jsonl").exists()
        # Acted on once: the next pass finds nothing answered.
        assert runner._resume_answered(_desk(), store, now=now, priced={"NVDAUSDT"}) == ([], [])

    def test_an_expired_answer_is_logged_not_raised(self, _sidecars: Path) -> None:
        store, request_id = _held(_sidecars)
        store.answer(answer_for(HALTED_APPROVE, request_id), now=AT + timedelta(minutes=4))
        runs, log = runner._resume_answered(_desk(), store, now=AT + timedelta(hours=2),
                                            priced={"NVDAUSDT"})
        assert runs == [] and log[0]["outcome"] == "expired"
        assert store.status(request_id) == "expired"

    def test_already_resolved_is_logged_not_raised(self, _sidecars: Path) -> None:
        store, request_id = _held(_sidecars)
        store.answer(answer_for(HALTED_APPROVE, request_id), now=AT + timedelta(minutes=4))

        class Racing:
            def resume(self, store: PauseStore, request_id: str, *, now: datetime) -> Any:
                raise AlreadyResolved(f"{request_id} is resumed; nothing to resume")

        runs, log = runner._resume_answered(Racing(), store,  # type: ignore[arg-type]
                                            now=AT + timedelta(minutes=5), priced={"NVDAUSDT"})
        assert runs == [] and log[0]["outcome"] == "already_resolved"

    def test_a_tampered_continuation_is_refused_without_stopping_the_cycle(
        self, _sidecars: Path
    ) -> None:
        store, request_id = _held(_sidecars)
        store.answer(answer_for(HALTED_APPROVE, request_id), now=AT + timedelta(minutes=4))
        path = store.root / "requests" / request_id / "continuation.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob["payload_sha256"] = "0" * 64
        path.write_text(json.dumps(blob), encoding="utf-8")
        runs, log = runner._resume_answered(_desk(), store, now=AT + timedelta(minutes=5),
                                            priced={"NVDAUSDT"})
        assert runs == [] and log[0]["outcome"] == "refused"

    def test_a_symbol_with_no_price_is_deferred_and_stays_answered(self, _sidecars: Path) -> None:
        store, request_id = _held(_sidecars)
        store.answer(answer_for(HALTED_APPROVE, request_id), now=AT + timedelta(minutes=4))
        runs, log = runner._resume_answered(_desk(), store, now=AT + timedelta(minutes=5),
                                            priced=set())
        assert runs == [] and log[0]["outcome"] == "deferred"
        assert store.status(request_id) == "answered"

    def test_the_cycle_runs_every_decision_with_the_store_and_resumes_first(self) -> None:
        import inspect

        source = inspect.getsource(runner.run_once)
        assert "pause_store=store" in source
        assert source.index("_resume_answered(") < source.index("desk.run(")

    def test_the_live_ledger_s_store_is_the_one_the_cli_answers_into(self) -> None:
        from argus.decision import pause

        assert runner.pause_root(runner.LEDGER_PATH).resolve() == pause.DEFAULT_ROOT.resolve()

    def test_a_cycle_on_another_ledger_never_touches_the_live_store(self, tmp_path: Path) -> None:
        """The 2026-09-26 finding: a replay on its own ledger wrote into ``data/pauses``."""
        import inspect

        assert runner.pause_root(tmp_path / "ledger.jsonl") == tmp_path / "pauses"
        assert "PauseStore(pause_root(ledger_path))" in inspect.getsource(runner.run_once)

    def test_the_fill_markers_are_flagged(self) -> None:
        assert runner._is_flag(f"[fill] seq 9: {FILL_FLAG} — x")
        assert runner._is_flag(f"[fill] seq 9: the recorded position {CEILING_FLAG} — x")


class TestTheReplayOfRecordedRuns:
    """The recorded history must pass the new checks, except where it is already known wrong."""

    def test_the_real_ledger_raises_no_false_alarm(self) -> None:
        raw = [json.loads(x) for x in et.LEDGER_PATH.read_text(encoding="utf-8").splitlines()
               if x.strip()]
        rows = [r for r in raw if r.get("kind", "decision") == "decision"]
        cycles, _ = et.parse_cycle_logs()
        got = et.replay_logged_decisions(cycles, rows, et._risk_ceilings())
        assert got["decisions_checked"] > 500
        assert set(got["alarm_seqs"]) <= {264, 265}, got["alarms"][:3]

    def test_the_incident_file_is_flagged_exactly_at_its_duplicates(self) -> None:
        got = et.positive_control_incident()
        if not got["available"]:
            pytest.skip("the 2026-09-12 incident file is not in this checkout")
        assert got["exactly_the_duplicates"] is True
        assert got["duplicated_seqs"] == [41, 47, 48, 49, 50, 51, 52]


def test_status_reading_holds_no_state_for_absence() -> None:
    assert StatusReading(state=None, filled=Decimal("0")).state is None
