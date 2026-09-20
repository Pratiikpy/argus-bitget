"""Baseline comparison tests.

A trading agent with no baseline has not been evaluated, it has been described. LLM-Trading-Lab ran
a real-money LLM portfolio for six months, computed an index comparison and then disclaimed it as
non-evidentiary; ai-hedge-fund has none at all. So the properties pinned here are the ones that keep
a baseline from flattering the desk:

* **no settled trades is undefined, not a tie** — a zero would read as "the desk matched the
  baseline", which is a performance claim made from an absence of performance;
* the baseline is **the desk's own picks, held** — not an index, because an index comparison
  measures which universe was hot;
* the baseline is **always long**, since holding has no short leg; scoring it with the desk's own
  direction would make it a copy of the desk rather than an alternative to it;
* a margin inside the noise band is a **tie**, not a victory.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from argus.eval.baseline import MATCH_TOLERANCE_BPS, Verdict, compare


@dataclass
class Row:
    """A settled ledger row, reduced to what the comparison reads."""

    symbol: str = "NVDAUSDT"
    side: str = "BUY"
    quantity: str = "10"
    entry_price: str = "100"
    exit_price: str | None = "110"
    net_pnl: str | None = None


class TestNoTradesIsUndefined:
    def test_an_empty_ledger_is_undefined(self) -> None:
        assert compare([]).verdict is Verdict.UNDEFINED

    def test_unsettled_trades_are_undefined(self) -> None:
        assert compare([Row(exit_price=None)]).verdict is Verdict.UNDEFINED

    def test_undefined_is_not_evidence_of_skill(self) -> None:
        assert not compare([]).verdict.is_evidence_of_skill

    def test_it_says_undefined_rather_than_tie(self) -> None:
        assert "undefined, not a tie" in " ".join(compare([]).render())

    def test_the_reason_counts_what_it_was_given(self) -> None:
        assert "2 trade(s) supplied" in compare([Row(exit_price=None)] * 2).reason

    def test_the_margin_is_none_rather_than_zero(self) -> None:
        assert compare([]).difference_bps is None

    def test_a_trade_with_no_usable_entry_is_not_scored(self) -> None:
        assert compare([Row(entry_price="0")]).verdict is Verdict.UNDEFINED


class TestTheBaselineIsTheDesksOwnPicksHeld:
    def test_a_long_that_captured_the_whole_move_matches_holding(self) -> None:
        """Bought at 100, out at 110, and the name closed at 110: holding did the same."""
        report = compare([Row()], marks={"NVDAUSDT": Decimal("110")})
        assert report.verdict is Verdict.MATCHED

    def test_selling_before_a_further_rally_loses_to_holding(self) -> None:
        report = compare([Row()], marks={"NVDAUSDT": Decimal("140")})
        assert report.verdict is Verdict.LOST

    def test_selling_before_a_collapse_beats_holding(self) -> None:
        report = compare([Row()], marks={"NVDAUSDT": Decimal("80")})
        assert report.verdict is Verdict.BEAT
        assert report.verdict.is_evidence_of_skill

    def test_the_baseline_is_long_even_when_the_desk_was_short(self) -> None:
        """Holding has no short leg; a desk that never bought the name had nothing to hold."""
        short = Row(side="SELL", entry_price="100", exit_price="90")
        report = compare([short], marks={"NVDAUSDT": Decimal("90")})
        # Holding from 100 to 90 loses 10 a share; the short made it.
        assert report.baseline_pnl < 0 < report.desk_pnl
        assert report.verdict is Verdict.BEAT

    def test_with_no_mark_the_desks_own_exit_is_used(self) -> None:
        """A real observed price, and the same one the desk got out at."""
        assert compare([Row()]).verdict is Verdict.MATCHED

    def test_the_margin_is_quoted_in_basis_points_of_deployed_notional(self) -> None:
        report = compare([Row()], marks={"NVDAUSDT": Decimal("80")})
        # Held: -20 a share on 10 shares = -200. Desk: +100. Difference 300 on 1000 notional.
        assert report.difference_bps == pytest.approx(Decimal("3000"), abs=1)


class TestASmallMarginIsATie:
    def test_inside_the_tolerance_is_matched(self) -> None:
        report = compare([Row(exit_price="100.02")], marks={"NVDAUSDT": Decimal("100")})
        assert abs(report.difference_bps or Decimal("0")) <= MATCH_TOLERANCE_BPS
        assert report.verdict is Verdict.MATCHED

    def test_a_tie_is_not_evidence_of_skill(self) -> None:
        assert not Verdict.MATCHED.is_evidence_of_skill

    def test_only_beating_counts_as_skill(self) -> None:
        for verdict in (Verdict.UNDEFINED, Verdict.MATCHED, Verdict.LOST):
            assert not verdict.is_evidence_of_skill
        assert Verdict.BEAT.is_evidence_of_skill


class TestManySymbolsAndManyTrades:
    def test_each_symbol_is_reported_separately(self) -> None:
        rows = [Row(symbol="NVDAUSDT"), Row(symbol="TSLAUSDT")]
        assert len(compare(rows).legs) == 2

    def test_the_symbols_beaten_count_is_visible(self) -> None:
        """One lucky name carrying the whole result must be legible, not hidden in the total."""
        rows = [Row(symbol="NVDAUSDT"), Row(symbol="TSLAUSDT")]
        report = compare(rows, marks={"NVDAUSDT": Decimal("80"), "TSLAUSDT": Decimal("140")})
        assert report.legs_beaten == 1

    def test_repeated_trades_in_one_name_all_count_toward_the_desk(self) -> None:
        rows = [Row(net_pnl="50"), Row(net_pnl="70")]
        report = compare(rows, marks={"NVDAUSDT": Decimal("110")})
        assert report.desk_pnl == Decimal("120")

    def test_the_baseline_buys_once_at_the_first_entry(self) -> None:
        """Turnover is the thing being tested, so the baseline may not trade."""
        rows = [Row(entry_price="100"), Row(entry_price="200")]
        report = compare(rows, marks={"NVDAUSDT": Decimal("110")})
        assert report.legs[0].baseline_pnl == Decimal("100")  # (110-100)*10, once

    def test_a_recorded_net_pnl_is_preferred_over_recomputing_it(self) -> None:
        """The ledger charged the fee; recomputing from prices would quietly drop it."""
        report = compare([Row(net_pnl="88")], marks={"NVDAUSDT": Decimal("110")})
        assert report.desk_pnl == Decimal("88")


class TestTheReportIsReadable:
    def test_a_loss_says_the_desk_should_have_gone_home(self) -> None:
        report = compare([Row()], marks={"NVDAUSDT": Decimal("140")})
        assert "choosing once and going home" in " ".join(report.render())

    def test_the_verdict_serialises_with_the_margin(self) -> None:
        payload = compare([Row()], marks={"NVDAUSDT": Decimal("80")}).as_dict()
        assert payload["verdict"] == "beat"
        assert payload["difference_bps"] is not None

    def test_every_leg_serialises(self) -> None:
        payload = compare([Row(symbol="A"), Row(symbol="B")]).as_dict()
        assert len(payload["legs"]) == 2

    def test_an_undefined_report_still_renders(self) -> None:
        assert compare([]).render()
