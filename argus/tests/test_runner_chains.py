"""Tests for the settlement path and the causal chains it now grades.

`paper/chains.py` shipped complete and was never called. Its own docstring described the defect it
closes — a chain "constructed, counted, described as checkable, and dropped on the floor" — and
because nothing in `paper/runner.py` imported it, that sentence remained an accurate description of
the live path rather than of the past. These tests exist so the wiring cannot be removed silently:
an orphan module with a passing unit test looks exactly like an integrated one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.agents.causality import CausalChain, Link
from argus.market.bitget import Ticker
from argus.paper import chains
from argus.paper.ledger import PaperLedger
from argus.paper.runner import HOLD_HOURS, _grade_chains, _settle_due, _write_chain

DECIDED = datetime(2026, 9, 1, 14, tzinfo=UTC)
LATER = DECIDED + timedelta(hours=HOLD_HOURS + 1)


def _ticker(symbol: str, last: str) -> Ticker:
    price = Decimal(last)
    return Ticker(
        symbol=symbol, last=price, bid=price * Decimal("0.999"), ask=price * Decimal("1.001"),
        high_24h=price, low_24h=price, change_24h=Decimal("0"), base_volume=Decimal("1000"),
        funding_rate=Decimal("0"), fetched_at=LATER,
    )


def _chain(*, direction: str = "up", magnitude: int = 40) -> CausalChain:
    chain = CausalChain(
        event="8-K: guidance raised", as_of=DECIDED,
        predicted_direction=direction, predicted_magnitude_bps=magnitude,
    )
    chain.add(Link(
        step="guidance raised", claim="FY revenue guide lifted",
        falsifier="the filing does not change the guide",
    ))
    chain.add(Link(
        step="multiple expands", claim="a higher guide lifts the multiple",
        falsifier="the multiple does not move on the print",
    ))
    return chain


class _Run:
    """The only part of a DeskRun `_write_chain` touches."""

    def __init__(self, chain: CausalChain | None) -> None:
        self.causal_chain = chain


def _ledger(tmp_path: Path) -> PaperLedger:
    return PaperLedger(path=tmp_path / "ledger.jsonl")


def _record(ledger: PaperLedger, symbol: str, *, verdict: str, side: str, qty: str, px: str):
    return ledger.record(
        symbol=symbol, verdict=verdict, side=side, quantity=Decimal(qty),
        entry_price=Decimal(px), stated_confidence=0.6, thesis="t",
        invalidation=("x",), market_state_hash="a" * 8, approved_intent_hash="b" * 8,
        session_phase="rth", hours_to_discovery=0.0, decided_at=DECIDED,
    )


class TestTheChainIsPersisted:
    def test_a_chain_with_links_is_written(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "chains.jsonl"
        monkeypatch.setattr(chains, "CHAINS_PATH", path)
        _write_chain(7, "NVDAUSDT", _Run(_chain()))
        stored = chains.load(path)
        assert len(stored) == 1
        assert stored[0].seq == 7
        assert stored[0].symbol == "NVDAUSDT"
        assert len(stored[0].links) == 2

    def test_a_decision_with_no_chain_writes_nothing(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "chains.jsonl"
        monkeypatch.setattr(chains, "CHAINS_PATH", path)
        _write_chain(7, "NVDAUSDT", _Run(None))
        assert chains.load(path) == []

    def test_an_empty_chain_writes_nothing(self, tmp_path: Path, monkeypatch) -> None:
        """An empty row would inflate the count of claims this desk has made without adding one,
        and the count is the whole reason these are kept."""
        path = tmp_path / "chains.jsonl"
        monkeypatch.setattr(chains, "CHAINS_PATH", path)
        _write_chain(7, "NVDAUSDT", _Run(CausalChain(event="nothing", as_of=DECIDED)))
        assert chains.load(path) == []


class TestSettlementCollectsTheOutcome:
    def test_an_abstention_settles_and_yields_its_counterfactual(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _record(ledger, "NVDAUSDT", verdict="no_trade", side="BUY", qty="0", px="100")
        out = _settle_due(ledger, {"NVDAUSDT": _ticker("NVDAUSDT", "101")}, LATER)
        row = next(r for r in out if r.get("abstention"))
        assert float(str(row["counterfactual_move_bps"])) == pytest.approx(100.0, abs=0.1)

    def test_nothing_settles_before_the_hold_elapses(self, tmp_path: Path) -> None:
        """The property that keeps settlement from using information the decision already had."""
        ledger = _ledger(tmp_path)
        _record(ledger, "NVDAUSDT", verdict="no_trade", side="BUY", qty="0", px="100")
        early = DECIDED + timedelta(hours=HOLD_HOURS - 1)
        assert _settle_due(ledger, {"NVDAUSDT": _ticker("NVDAUSDT", "101")}, early) == []

    def test_a_symbol_with_no_ticker_is_left_open(self, tmp_path: Path) -> None:
        """Absence, never a substituted price. Settling at a guessed number would write an
        outcome that never happened into a record that cannot be revised."""
        ledger = _ledger(tmp_path)
        _record(ledger, "NVDAUSDT", verdict="no_trade", side="BUY", qty="0", px="100")
        assert _settle_due(ledger, {}, LATER) == []
        assert not ledger.entries[0].is_settled


class TestTheChainIsGraded:
    def test_settlement_grades_the_stored_chain(self, tmp_path: Path, monkeypatch) -> None:
        """The whole point of the wiring: a decision that settles takes its chain with it."""
        path = tmp_path / "chains.jsonl"
        monkeypatch.setattr(chains, "CHAINS_PATH", path)
        ledger = _ledger(tmp_path)
        entry = _record(ledger, "NVDAUSDT", verdict="no_trade", side="BUY", qty="0", px="100")
        chains.write(_chain(), seq=entry.seq, symbol="NVDAUSDT", path=path)
        assert not chains.load(path)[0].graded

        out = _settle_due(ledger, {"NVDAUSDT": _ticker("NVDAUSDT", "101")}, LATER)
        assert any("causal_chains_graded" in r for r in out)
        assert chains.load(path)[0].graded

    def test_a_chain_is_graded_against_the_move_not_the_pnl(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A chain predicts a price move. Grading it on net P&L would charge the forecast for the
        fee and mark a correct call wrong on a trade the cost model killed."""
        path = tmp_path / "chains.jsonl"
        monkeypatch.setattr(chains, "CHAINS_PATH", path)
        ledger = _ledger(tmp_path)
        entry = _record(ledger, "NVDAUSDT", verdict="trade", side="BUY", qty="1", px="100")
        chains.write(_chain(direction="up", magnitude=100), seq=entry.seq,
                     symbol="NVDAUSDT", path=path)

        # +100bps of price, which a 12bps round trip would leave net positive but smaller. The
        # chain must be graded on the 100, and its direction must come out correct.
        _settle_due(ledger, {"NVDAUSDT": _ticker("NVDAUSDT", "101")}, LATER)
        graded = chains.load(path)[0]
        assert graded.graded
        assert graded.realised_direction == "up"
        assert graded.realised_magnitude_bps == pytest.approx(100, abs=1)

    def test_grading_is_refused_when_the_outcome_does_not_postdate_the_decision(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The one mistake that would make every number in the file worthless."""
        path = tmp_path / "chains.jsonl"
        monkeypatch.setattr(chains, "CHAINS_PATH", path)
        chains.write(_chain(), seq=1, symbol="NVDAUSDT", path=path)
        graded = _grade_chains({1: (100.0, DECIDED - timedelta(hours=1))})
        assert graded == 0
        assert not chains.load(path)[0].graded

    def test_no_outcomes_grades_nothing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(chains, "CHAINS_PATH", tmp_path / "chains.jsonl")
        assert _grade_chains({}) == 0

    def test_a_chain_is_graded_once(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "chains.jsonl"
        monkeypatch.setattr(chains, "CHAINS_PATH", path)
        chains.write(_chain(), seq=1, symbol="NVDAUSDT", path=path)
        assert _grade_chains({1: (100.0, LATER)}) == 1
        assert _grade_chains({1: (100.0, LATER)}) == 0


class TestTheWiringItself:
    def test_the_runner_actually_imports_the_chains_module(self) -> None:
        """The test that would have caught the original defect. `paper/chains.py` was complete,
        tested in isolation, registered in `status.py` as importable — and called from nowhere."""
        import inspect

        from argus.paper import runner

        source = inspect.getsource(runner)
        assert "from argus.paper import chains" in source

    def test_settlement_calls_the_grader(self) -> None:
        import inspect

        from argus.paper import runner

        assert "_grade_chains(outcomes)" in inspect.getsource(runner._settle_due)

    def test_the_decision_path_calls_the_writer(self) -> None:
        """Every booked run goes through `_record_run` since 2026-09-25 — the fresh decision and
        the one a human resumed alike — so the writer is pinned there, and `run_once` is pinned to
        route both kinds of run through it."""
        import inspect

        from argus.paper import runner

        assert "_write_chain(" in inspect.getsource(runner._record_run)
        assert inspect.getsource(runner.run_once).count("_record_run(") == 2
