"""Paper-trading ledger tests.

Fabricated or backfilled paper trading is an anti-pattern we named and banned. These tests assert
the structural defences actually hold, because "trust our log" is not a submission.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.paper.ledger import GENESIS, Entry, LedgerError, PaperLedger

T0 = datetime(2026, 3, 9, 14, 0, tzinfo=UTC)


def _ledger(tmp_path: Path) -> PaperLedger:
    return PaperLedger(path=tmp_path / "paper.jsonl")


def _record(led: PaperLedger, *, qty: str = "10", price: str = "220", n: int = 0) -> Entry:
    return led.record(
        symbol="NVDAUSDT", verdict="trade", side="BUY",
        quantity=Decimal(qty), entry_price=Decimal(price),
        stated_confidence=0.7, thesis="guidance cut not priced",
        invalidation=("guidance reaffirmed",),
        market_state_hash="abc123", approved_intent_hash="def456",
        session_phase="weekend", hours_to_discovery=30.5,
        decided_at=T0 + timedelta(hours=n),
    )


class TestHashChain:
    def test_first_entry_links_to_genesis(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        assert _record(led).prev_hash == GENESIS

    def test_each_entry_links_to_the_last(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        a = _record(led, n=0)
        b = _record(led, n=1)
        assert b.prev_hash == a.content_hash
        assert led.verify()["chain_intact"] is True

    def test_editing_history_breaks_the_chain_at_that_point(self, tmp_path: Path) -> None:
        """A doctored log must fail its own verification."""
        led = _ledger(tmp_path)
        for i in range(4):
            _record(led, n=i)
        assert led.verify()["chain_intact"] is True

        # Tamper: rewrite a past decision's thesis on disk.
        raw = led.path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in raw if x.strip()]
        rows[1]["thesis"] = "a story invented after the fact"
        led.path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        reloaded = PaperLedger(path=led.path)
        got = reloaded.verify()
        assert got["chain_intact"] is False
        assert got["first_break_at"] == 3  # the entry AFTER the edited one

    def test_a_reloaded_untampered_ledger_verifies(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        for i in range(3):
            _record(led, n=i)
        assert PaperLedger(path=led.path).verify()["chain_intact"] is True


class TestOutcomesAreWrittenOnce:
    def test_a_decision_is_recorded_without_an_outcome(self, tmp_path: Path) -> None:
        """There is no code path that writes a decision and its result together."""
        e = _record(_ledger(tmp_path))
        assert e.is_settled is False
        assert e.net_pnl is None
        assert e.direction_correct is None

    def test_settling_twice_is_refused(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _record(led)
        led.settle(1, exit_price=Decimal("230"), settled_at=T0 + timedelta(days=1))
        with pytest.raises(LedgerError, match="written once"):
            led.settle(1, exit_price=Decimal("300"))

    def test_settling_an_unknown_entry_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LedgerError, match="no entry"):
            _ledger(tmp_path).settle(99, exit_price=Decimal("1"))

    def test_settlement_does_not_change_the_chain(self, tmp_path: Path) -> None:
        """Outcomes sit outside the hash, so attaching one is distinguishable from tampering."""
        led = _ledger(tmp_path)
        _record(led, n=0)
        _record(led, n=1)
        before = led.verify()["head_hash"]
        led.settle(1, exit_price=Decimal("230"))
        assert led.verify()["head_hash"] == before
        assert led.verify()["chain_intact"] is True


class TestCosts:
    def test_entry_cost_is_charged_at_decision_time(self, tmp_path: Path) -> None:
        e = _record(_ledger(tmp_path))
        assert Decimal(e.entry_cost_bps) > 0

    def test_net_is_below_gross_by_the_round_trip(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _record(led, qty="10", price="220")
        got = led.settle(1, exit_price=Decimal("230"))
        assert Decimal(got.net_pnl or "0") < Decimal(got.gross_pnl or "0")

    def test_a_small_winner_can_be_a_net_loser(self, tmp_path: Path) -> None:
        """The measured reality on this venue: a 12bps round trip eats most small moves."""
        led = _ledger(tmp_path)
        _record(led, qty="10", price="220")
        got = led.settle(1, exit_price=Decimal("220.10"))   # ~4.5bps move
        assert Decimal(got.gross_pnl or "0") > 0
        assert Decimal(got.net_pnl or "0") < 0

    def test_performance_reports_when_cost_flipped_the_sign(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path)
        _record(led, qty="10", price="220")
        led.settle(1, exit_price=Decimal("220.10"))
        assert led.performance()["cost_flipped_the_sign"] is True


class TestAbstentions:
    def test_abstentions_are_counted_separately_from_trades(self, tmp_path: Path) -> None:
        """A NO_TRADE is a decision, not a missing row — the log must show the agent chose."""
        led = _ledger(tmp_path)
        led.record(
            symbol="NVDAUSDT", verdict="no_trade", side="BUY",
            quantity=Decimal("0"), entry_price=Decimal("220"),
            stated_confidence=0.9, thesis="edge does not clear 12bps",
            invalidation=(), market_state_hash="x", approved_intent_hash="y",
            session_phase="weekend", hours_to_discovery=30.5, decided_at=T0,
        )
        _record(led, n=1)
        perf = led.performance()
        assert perf["abstentions"] == 1
        assert perf["decisions"] == 2

    def test_an_empty_ledger_says_so_rather_than_reporting_zeroes(self, tmp_path: Path) -> None:
        assert "nothing to report" in _ledger(tmp_path).performance()["note"]
