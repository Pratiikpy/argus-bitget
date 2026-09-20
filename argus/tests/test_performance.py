"""Paper-trading performance tests.

The property under test is the one that made the module necessary: the three numbers Track 2 is
scored on must come out of the *real* ledger, and must refuse to exist when the ledger cannot
support them. A printed ``0.0`` Sharpe on a log of pure abstentions is the failure mode — it reads
as a real measurement of a flat strategy rather than as the absence of a measurement.

Fixtures are written as JSONL and loaded back through ``PaperLedger(path=...)`` rather than poked
into the object with ``__new__``. That costs a temp file per test and buys the guarantee that these
tests break when :class:`Entry` changes shape, instead of passing against a structure the ledger
never actually writes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.backtest.metrics import MetricError
from argus.eval.performance import PERIODS_PER_YEAR, daily_series, evaluate_ledger
from argus.paper.ledger import Entry, PaperLedger

DAY0 = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
CAPITAL = Decimal("10000")


def _entry(
    seq: int,
    *,
    symbol: str = "NVDAUSDT",
    decided: datetime,
    settled: datetime | None = None,
    net_pnl: str | None = None,
    verdict: str = "open_long",
    quantity: str = "1",
) -> Entry:
    return Entry(
        seq=seq,
        decided_at=decided.isoformat(),
        symbol=symbol,
        verdict=verdict,
        side="long" if verdict == "open_long" else "flat",
        quantity=quantity,
        entry_price="100",
        stated_confidence=0.7,
        thesis="fixture",
        invalidation=("price below 95",),
        market_state_hash="m" * 16,
        approved_intent_hash="a" * 16,
        session_phase="rth",
        hours_to_discovery=0.0,
        entry_cost_bps="6",
        prev_hash="0" * 16,
        settled_at=settled.isoformat() if settled else None,
        exit_price="101" if settled else None,
        gross_pnl=net_pnl if settled else None,
        net_pnl=net_pnl if settled else None,
        direction_correct=(Decimal(net_pnl) > 0) if (settled and net_pnl) else None,
    )


def _abstention(seq: int, *, decided: datetime, settled: datetime | None = None) -> Entry:
    return _entry(
        seq, decided=decided, settled=settled, net_pnl="0", verdict="no_trade", quantity="0"
    )


def _write(entries: list[Entry], directory: Path) -> PaperLedger:
    """Persist fixtures as real JSONL and load them back through the real constructor."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "paper.jsonl"
    lines = [json.dumps(asdict(e), default=str) for e in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PaperLedger(path=path)


def _trades(directory: Path, pnls: list[str]) -> PaperLedger:
    """Settled trades on consecutive days, one per day, from the P&L sequence given."""
    entries = [
        _entry(
            i + 1,
            symbol="NVDAUSDT" if i % 2 == 0 else "TSLAUSDT",
            decided=DAY0 + timedelta(days=i),
            settled=DAY0 + timedelta(days=i + 1),
            net_pnl=p,
        )
        for i, p in enumerate(pnls)
    ]
    return _write(entries, directory)


def _mixed(directory: Path) -> PaperLedger:
    """Seven settled trades on consecutive days: four winners, three losers.

    Seven clears `MIN_TRADES_FOR_DRAWDOWN` (5) but deliberately NOT `MIN_DAYS_FOR_SHARPE` (30),
    so the exact drawdown and win-rate values the tests below assert stay exactly as chosen.
    Tests that need a Sharpe use :func:`_mixed_long` instead.
    """
    return _trades(directory, ["120", "-60", "80", "-30", "150", "-20", "90"])


def _mixed_long(directory: Path) -> PaperLedger:
    """Thirty-five settled trades — enough daily returns for a Sharpe to be defined.

    **Exists because a threshold that was too low shipped a real bug.** `MIN_DAYS_FOR_SHARPE` was
    2 — an arithmetic guard ("a standard deviation is not defined below this"), not a statistical
    one — so the seven-trade fixture above was enough to produce a Sharpe and the tests passed.
    When the live desk took its first two real trades on 2026-09-15, that same threshold printed
    **Sharpe 6.75** onto the public cockpit from two daily returns, beside prose still calling the
    figure undefined. The production fix raises the threshold to 30 (this codebase's own
    `backtest.dependence.MIN_OBSERVATIONS`); this fixture gives the Sharpe tests a sample that
    genuinely clears it, rather than weakening the threshold back to keep them green.

    The P&L pattern repeats the seven-value cycle above, so the winner/loser character is unchanged.
    """
    cycle = ["120", "-60", "80", "-30", "150", "-20", "90"]
    return _trades(directory, [cycle[i % len(cycle)] for i in range(35)])


class TestTheRefusalToInventNumbers:
    """This is the state the live ledger was actually in when this module was written."""

    def test_a_log_of_only_abstentions_has_no_sharpe_and_no_win_rate(self, tmp_path: Path) -> None:
        entries = [
            _abstention(i, decided=DAY0 + timedelta(hours=2 * i), settled=DAY0 + timedelta(days=2))
            for i in range(1, 28)
        ]
        perf = evaluate_ledger(_write(entries, tmp_path), capital=CAPITAL)

        assert perf.trades == 0
        assert perf.abstentions == 27
        assert perf.sharpe is None, "a zero-variance series must not report a Sharpe of 0.0"
        assert perf.win_rate is None, "a win rate over zero trades is undefined, not 0%"
        assert "sharpe" in perf.undefined and "win_rate" in perf.undefined
        # **This reverses an earlier deliberate call in this file**, which asserted
        # `max_drawdown == 0.0` and justified it as "that one is a real zero".
        #
        # The arithmetic behind that was right: the equity series never moves, so the realised
        # drawdown is genuinely 0.0 and nothing is being divided by zero. What it missed is that
        # **Bitget scores this exact number** — handbook line 254, "Paper trading Sharpe, max
        # drawdown, win rate" — and 0.0% is the single best-looking risk figure a desk can print.
        # A judge reading it sees excellent risk control; what produced it is an account that never
        # took a position.
        #
        # That is the absence-is-not-zero pattern precisely: a strategy that traded and never lost
        # and a strategy that never traded publish the identical number from entirely different
        # causes, and one of them is an achievement. `sharpe`, `sortino` and `win_rate` were
        # already withheld here for the same absence; drawdown was the one still producing a
        # flattering figure from it, and it was the most flattering of the four.
        #
        # One settled trade (MIN_TRADES_TO_REPORT = 1) makes it a real measurement again.
        assert perf.max_drawdown is None, (
            "a drawdown over an account that never took a position measures the absence of "
            "exposure, not risk control, and Track 2 scores this number"
        )
        assert "max_drawdown" in perf.undefined
        assert "never took a position" in perf.undefined["max_drawdown"]

    def test_the_report_carries_the_reason_not_just_a_null(self, tmp_path: Path) -> None:
        entries = [_abstention(1, decided=DAY0, settled=DAY0 + timedelta(days=1))]
        out = evaluate_ledger(_write(entries, tmp_path)).as_dict()
        assert out["sharpe"] is None
        assert "undefined" in out
        assert "win_rate" in out["undefined"]
        assert "zero" in out["undefined"]["win_rate"]

    def test_a_single_day_window_cannot_produce_a_standard_deviation(
        self, tmp_path: Path
    ) -> None:
        entries = [_entry(1, decided=DAY0, settled=DAY0 + timedelta(hours=3), net_pnl="50")]
        perf = evaluate_ledger(_write(entries, tmp_path))
        assert perf.sharpe is None
        assert "at least" in perf.undefined["sharpe"]

    def test_zero_capital_is_refused_rather_than_dividing(self, tmp_path: Path) -> None:
        ledger = _write([_entry(1, decided=DAY0)], tmp_path)
        with pytest.raises(MetricError, match="capital must be positive"):
            daily_series(ledger, capital=Decimal("0"))


class TestTheCalendarSeries:
    def test_days_with_no_settlement_appear_as_flat_days(self, tmp_path: Path) -> None:
        """A desk that abstains for a week has not earned a week's worth of Sharpe."""
        entries = [
            _entry(1, decided=DAY0, settled=DAY0 + timedelta(days=1), net_pnl="100"),
            _entry(2, decided=DAY0, settled=DAY0 + timedelta(days=6), net_pnl="100"),
        ]
        daily = daily_series(_write(entries, tmp_path), capital=CAPITAL)
        assert [d.day for d in daily] == [date(2026, 9, 1) + timedelta(days=i) for i in range(7)]
        assert sum(1 for d in daily if d.is_flat) == 5

    def test_a_trade_lands_on_the_day_it_settled_not_the_day_it_was_decided(
        self, tmp_path: Path
    ) -> None:
        entries = [_entry(1, decided=DAY0, settled=DAY0 + timedelta(days=3), net_pnl="100")]
        daily = daily_series(_write(entries, tmp_path), capital=CAPITAL)
        earning = [d for d in daily if d.net_pnl != 0]
        assert len(earning) == 1
        assert earning[0].day == date(2026, 9, 4)

    def test_several_settlements_on_one_day_are_summed(self, tmp_path: Path) -> None:
        settle = DAY0 + timedelta(days=1)
        entries = [
            _entry(1, decided=DAY0, settled=settle, net_pnl="100"),
            _entry(2, decided=DAY0, settled=settle, net_pnl="-40", symbol="TSLAUSDT"),
        ]
        daily = daily_series(_write(entries, tmp_path), capital=CAPITAL)
        day = next(d for d in daily if d.day == date(2026, 9, 2))
        assert day.net_pnl == Decimal("60")
        assert day.trades_settled == 2

    def test_an_open_position_contributes_nothing_until_it_settles(self, tmp_path: Path) -> None:
        entries = [
            _entry(1, decided=DAY0, settled=DAY0 + timedelta(days=1), net_pnl="100"),
            _entry(2, decided=DAY0 + timedelta(days=1)),  # still open
        ]
        perf = evaluate_ledger(_write(entries, tmp_path), capital=CAPITAL)
        assert perf.trades == 1
        assert perf.open_positions == 1
        assert perf.net_pnl == Decimal("100")


class TestTheThreeScoredNumbers:
    def test_all_three_are_produced_once_the_sample_supports_them(self, tmp_path: Path) -> None:
        """Note "supports them", not merely "trades exist" — the distinction is the 2026-09-20
        bug. A single settled trade used to satisfy every threshold here."""
        perf = evaluate_ledger(_mixed_long(tmp_path), capital=CAPITAL)
        assert perf.sharpe is not None
        assert perf.win_rate is not None
        assert perf.max_drawdown > 0
        assert perf.undefined == {}

    def test_a_thin_sample_yields_win_rate_but_refuses_sharpe(self, tmp_path: Path) -> None:
        """The real live shape on 2026-09-20: enough trades to count, not enough to infer.

        This is the regression guard for the shipped bug — a descriptive count is reported, the
        inferential statistics are refused, and the reason is stated rather than left blank.
        """
        perf = evaluate_ledger(_trades(tmp_path, ["120", "90"]), capital=CAPITAL)
        assert perf.win_rate == pytest.approx(1.0)
        assert perf.sharpe is None
        assert perf.max_drawdown is None
        assert "sharpe" in perf.undefined
        assert "max_drawdown" in perf.undefined

    def test_win_rate_counts_settled_trades_net_of_fees(self, tmp_path: Path) -> None:
        perf = evaluate_ledger(_mixed(tmp_path), capital=CAPITAL)
        assert perf.trades == 7
        assert perf.win_rate == pytest.approx(4 / 7)

    def test_drawdown_is_measured_from_the_running_peak(self, tmp_path: Path) -> None:
        """Up 120 then down 60 on 10k: the trough is 0.6% below the peak.

        The start-relative reading would divide the 60 by the original 10,000 and call it 0.593%,
        understating the trough. The peak-relative reading is the one that matches how a drawdown
        is quoted anywhere it matters.

        The three small gains after the trough exist only to clear `MIN_TRADES_FOR_DRAWDOWN` (5).
        They rise monotonically from the trough, so they cannot deepen it: the maximum drawdown
        over the series is still exactly the 0.6% this test was written to pin.
        """
        entries = [
            _entry(1, decided=DAY0, settled=DAY0 + timedelta(days=1), net_pnl="120"),
            _entry(2, decided=DAY0, settled=DAY0 + timedelta(days=2), net_pnl="-60"),
            _entry(3, decided=DAY0, settled=DAY0 + timedelta(days=3), net_pnl="10"),
            _entry(4, decided=DAY0, settled=DAY0 + timedelta(days=4), net_pnl="10"),
            _entry(5, decided=DAY0, settled=DAY0 + timedelta(days=5), net_pnl="10"),
        ]
        perf = evaluate_ledger(_write(entries, tmp_path), capital=CAPITAL)
        assert perf.max_drawdown == pytest.approx(0.006, rel=1e-9)
        assert perf.max_drawdown > 60 / 10_120, "start-relative would understate the trough"

    def test_capital_leaves_sharpe_alone_and_scales_return_about_tenfold(
        self, tmp_path: Path
    ) -> None:
        """Sharpe is scale-free exactly; compounded return is scale-free only to first order.

        Mean and standard deviation both scale with 1/capital, so their ratio is invariant to the
        last bit. Total return compounds, so a tenth of the capital earns slightly *more* than ten
        times the return — asserting exact linearity would be asserting that compounding does not
        happen.
        """
        small = evaluate_ledger(_mixed_long(tmp_path / "a"), capital=Decimal("10000"))
        large = evaluate_ledger(_mixed_long(tmp_path / "b"), capital=Decimal("100000"))
        assert small.sharpe is not None and large.sharpe is not None
        assert small.sharpe == pytest.approx(large.sharpe, rel=1e-9)
        # 8% rather than 1%: this fixture is now 35 days rather than 8, and the convexity the
        # docstring describes compounds over every one of them. Widened to match the longer
        # window, not to hide a discrepancy — the exact-invariance assertion above is unchanged
        # at 1e-9, and the direction of the gap is asserted below.
        assert small.total_return == pytest.approx(10 * large.total_return, rel=8e-2)
        assert small.total_return > 10 * large.total_return, "compounding is convex in size"


class TestTheSingleSymbolTrap:
    def test_one_instrument_carrying_the_result_is_visible_in_the_report(
        self, tmp_path: Path
    ) -> None:
        """A cross-sectional result that is really one lucky name has been produced here before."""
        entries = [
            _entry(1, symbol="MSTRUSDT", decided=DAY0, settled=DAY0 + timedelta(days=1),
                   net_pnl="500"),
            _entry(2, symbol="NVDAUSDT", decided=DAY0, settled=DAY0 + timedelta(days=2),
                   net_pnl="10"),
            _entry(3, symbol="AAPLUSDT", decided=DAY0, settled=DAY0 + timedelta(days=3),
                   net_pnl="-5"),
        ]
        perf = evaluate_ledger(_write(entries, tmp_path), capital=CAPITAL)
        assert perf.by_symbol[0].symbol == "MSTRUSDT"
        assert perf.largest_symbol_share > 0.95
        assert perf.as_dict()["largest_symbol_share_pct"] > 95


class TestAnnualisation:
    def test_the_venue_trades_every_day_of_the_year(self) -> None:
        """252 would understate a 7x24 perpetual series by about 20%."""
        assert PERIODS_PER_YEAR == 365

    def test_the_periods_per_year_is_stated_in_the_report(self, tmp_path: Path) -> None:
        entries = [_entry(1, decided=DAY0, settled=DAY0 + timedelta(days=1), net_pnl="10")]
        assert evaluate_ledger(_write(entries, tmp_path)).as_dict()["periods_per_year"] == 365


class TestTheLedgerRoundTrip:
    def test_fixtures_survive_the_real_loader(self, tmp_path: Path) -> None:
        """If :class:`Entry` gains or renames a field, these tests must fail rather than drift."""
        entries = [_entry(1, decided=DAY0, settled=DAY0 + timedelta(days=1), net_pnl="10")]
        ledger = _write(entries, tmp_path)
        assert len(ledger.entries) == 1
        assert ledger.entries[0].symbol == "NVDAUSDT"
        assert ledger.entries[0].is_settled
