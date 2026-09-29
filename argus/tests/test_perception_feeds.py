"""The feeds added after the perception comparison counted them missing (2026-09-29): listed options
(market/options.py), dark-pool volume (market/darkpool.py), dividend history
(market/bitget_positioning.dividend_history), and the one-day date shift corrected in
market/bitget_mcp.py. All offline."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from argus.market import bitget_mcp, bitget_positioning, darkpool, options


def _row(sym: str, bid: float, ask: float, iv: float, delta: float, oi: float = 10.0,
         vol: float = 5.0) -> dict[str, Any]:
    return {"option": sym, "bid": bid, "ask": ask, "iv": iv, "delta": delta,
            "open_interest": oi, "volume": vol}


def _chain() -> dict[str, Any]:
    return {"timestamp": "2026-09-28 20:56:44", "data": {
        "symbol": "NVDA", "current_price": 228.6, "iv30": 31.3, "options": [
            # a same-week expiry, skipped for the implied move
            _row("NVDA260930C00230000", 1.0, 1.1, 0.40, 0.45),
            _row("NVDA260930P00230000", 2.0, 2.1, 0.40, -0.55),
            # the first expiry at least five days out
            _row("NVDA261005C00227500", 5.0, 5.2, 0.31, 0.52, oi=100, vol=40),
            _row("NVDA261005P00227500", 2.6, 2.8, 0.32, -0.48, oi=80, vol=20),
            _row("NVDA261005C00240000", 1.2, 1.3, 0.29, 0.24),
            _row("NVDA261005P00215000", 1.0, 1.1, 0.33, -0.26),
            # no two-sided quote: kept out of every figure but the ratios
            _row("NVDA261005C00300000", 0.0, 0.05, 0.0, 0.01),
            {"option": "not-an-occ-symbol"},
        ]}}


class TestOptions:
    def test_the_summary_reads_the_first_expiry_a_week_out(self) -> None:
        s = options.summarise(_chain(), today=date(2026, 9, 28))
        assert s.expiry == date(2026, 10, 5) and s.days_to_expiry == 7
        assert s.atm_strike == 227.5
        assert s.implied_move_pct == pytest.approx((5.1 + 2.7) / 228.6 * 100, abs=0.01)
        assert s.skew_25d == pytest.approx((0.33 - 0.29) * 100)
        assert s.contracts == 7 and s.quoted_contracts == 6

    def test_ratios_are_puts_over_calls(self) -> None:
        s = options.summarise(_chain(), today=date(2026, 9, 28))
        assert s.put_call_volume == pytest.approx(round((5 + 20 + 5) / (5 + 40 + 5 + 5), 3))

    def test_a_chain_without_a_price_refuses(self) -> None:
        with pytest.raises(options.OptionsError):
            options.summarise({"data": {"symbol": "NVDA", "options": []}}, today=date.today())

    def test_a_replayed_instant_gets_nothing_rather_than_todays_chain(self) -> None:
        got, status = options.options_evidence("NVDA", as_of=datetime(2026, 1, 5, tzinfo=UTC))
        assert got == [] and "not replayable" in status

    def test_a_flat_skew_is_called_flat(self) -> None:
        s = options.summarise(_chain(), today=date(2026, 9, 28))
        flat = options.OptionsSummary(**{**s.__dict__, "skew_25d": 0.0})
        assert "(flat)" in flat.claim()


class TestDarkPool:
    def _week(self, start: str, shares: int, published: str) -> darkpool.AtsWeek:
        return darkpool.AtsWeek(week_start=date.fromisoformat(start), shares=shares, trades=100,
                                notional=1e9, published=published)

    def test_a_holiday_week_is_compared_per_trading_day(self) -> None:
        labor_day = self._week("2026-09-07", 40_000_000, "2026-09-28")
        assert labor_day.trading_days == 4
        pool = darkpool.DarkPool("NVDA", "T1", (labor_day,
                                               self._week("2026-08-31", 50_000_000, "2026-09-21")))
        assert pool.vs_baseline == pytest.approx(10_000_000 / 10_000_000 - 1.0)

    def test_a_week_published_after_the_instant_is_not_shown(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        weeks = (self._week("2026-09-07", 1, "2026-09-28"),
                 self._week("2026-08-31", 1, "2026-09-21"))
        monkeypatch.setattr(darkpool, "dark_pool",
                            lambda symbol, timeout=0: darkpool.DarkPool(symbol, "T1", weeks))
        got, status = darkpool.dark_pool_evidence("NVDA", as_of=datetime(2026, 9, 25, tzinfo=UTC))
        assert status == "darkpool:NVDA: week of 2026-08-31"
        assert "2026-08-31" in got[0].claim

    def test_finras_empty_body_is_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(darkpool.http, "fetch", lambda *a, **k: b"")
        assert darkpool.week_for("SQQQ", "2026-09-07", "T1") is None


class TestTheOneDayShift:
    def test_calendar_and_dividend_dates_move_forward_a_day(self) -> None:
        rows = [{"ex_dividend_date": "2025-08-10", "payment_date": "2025-08-13",
                 "amount": 0.26, "split_valid_date": "2024-06-06"}]
        (fixed,) = bitget_mcp.correct_dates("equity_fundamental_dividends", rows)
        assert fixed["ex_dividend_date"] == "2025-08-11"   # Apple's Monday, not a Sunday
        assert fixed["payment_date"] == "2025-08-14"
        assert fixed["split_valid_date"] == "2024-06-06"   # checked, and not shifted

    def test_other_entries_are_left_as_served(self) -> None:
        rows = [{"filing_date": "2026-09-23"}]
        assert bitget_mcp.correct_dates("equity_ownership_insider_trading", rows) == rows

    def test_a_field_that_is_not_a_date_is_left_alone(self) -> None:
        (fixed,) = bitget_mcp.correct_dates("equity_calendar", [{"period_ending": None,
                                                                 "perf_brief_dsclsr_date": "tbd"}])
        assert fixed == {"period_ending": None, "perf_brief_dsclsr_date": "tbd"}


class TestDividendHistory:
    def test_last_dividend_twelve_months_and_last_split(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [{"ex_dividend_date": d, "amount": a, "payment_date": p} for d, a, p in (
            ("2025-09-11", 0.01, "2025-10-02"), ("2025-12-04", 0.01, ""),
            ("2026-03-11", 0.01, ""), ("2026-06-04", 0.25, ""),
            ("2026-09-10", 0.25, "2026-10-01"), ("2026-12-01", 0.30, ""))]
        rows.append({"ex_dividend_date": "2024-06-10", "amount": None, "split_numerator": "10",
                     "split_denominator": "1"})
        monkeypatch.setattr(bitget_positioning, "_safe", lambda entry, **kw: rows)
        line = bitget_positioning.dividend_history("NVDA", today=date(2026, 9, 29))
        assert line is not None
        assert "last cash dividend $0.25 a share, ex 10 Sep 2026, payable 2026-10-01" in line
        assert "$0.52 over the last 12 months in 4 payment(s)" in line   # the future row is not
        assert "last split 10-for-1 on 10 Jun 2024" in line

    def test_nothing_recorded_says_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bitget_positioning, "_safe", lambda entry, **kw: [])
        assert bitget_positioning.dividend_history("QQQ") is None


class TestTheAuditsFixes:
    """The fresh-eyes audit of 2026-09-29, one test per defect it verified."""

    def test_cboes_timestamp_is_shown_in_new_york_time(self) -> None:
        s = options.summarise(_chain(), today=date(2026, 9, 28))
        assert s.quoted_at == "2026-09-28 16:56"          # 20:56 UTC is 16:56 in New York
        assert "New York" in s.claim() and " ET)" not in s.claim()

    def test_a_skew_under_half_a_vol_point_is_flat(self) -> None:
        s = options.summarise(_chain(), today=date(2026, 9, 28))
        assert "(flat)" in options.OptionsSummary(**{**s.__dict__, "skew_25d": 0.3}).claim()
        assert "(puts bid)" in options.OptionsSummary(**{**s.__dict__, "skew_25d": 0.6}).claim()

    def test_the_confirmed_calendar_row_wins_over_a_forecast_placeholder(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [{"period_ending": "2026-07-30", "perf_briefing_fore_dsclsr_date": "2026-08-26",
                 "perf_brief_dsclsr_date": None},
                {"period_ending": "2026-07-26", "perf_briefing_fore_dsclsr_date": "2026-08-26",
                 "perf_brief_dsclsr_date": "2026-08-26"},
                {"period_ending": "2026-04-26", "perf_brief_dsclsr_date": "2026-05-20"}]
        service = bitget_mcp.BitgetDataService.__new__(bitget_mcp.BitgetDataService)
        monkeypatch.setattr(service, "results", lambda entry, **kw: rows, raising=False)
        got = service.next_earnings("NVDA")
        assert got["period_ending"] == "2026-07-26" and got["report_date"] == "2026-08-26"

    def test_the_served_timestamp_moves_with_the_dates(self) -> None:
        (fixed,) = bitget_mcp.correct_dates("equity_calendar", [
            {"period_ending": "2026-06-29", "time": 1782691200000}])
        assert fixed["time"] == 1782691200000 + 86_400_000

    def test_a_week_is_labelled_with_how_late_it_was_published(self) -> None:
        week = darkpool.AtsWeek(week_start=date(2026, 8, 24), shares=1, trades=1, notional=1.0,
                                published="2026-09-28")
        assert week.weeks_late == 4

    def test_tier_2_names_are_read_at_tier_2_first(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked: list[str] = []
        monkeypatch.setattr(darkpool, "latest_weeks",
                            lambda tier, timeout=0: asked.append(tier) or ["2026-08-24"])
        monkeypatch.setattr(darkpool, "week_for", lambda s, w, t, timeout=0: darkpool.AtsWeek(
            week_start=date.fromisoformat(w), shares=1, trades=1, notional=1.0,
            published="2026-09-28"))
        assert darkpool.dark_pool("TQQQ").tier == "T2" and asked == ["T2"]

    def test_a_read_with_a_missing_feed_is_cached_briefly(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import positioning

        calls: list[str] = []

        def read(ticker: str) -> Any:
            calls.append(ticker)
            return [], [], {"options": None, "dark_pool": None, "short_volume": None}

        monkeypatch.setattr(positioning, "_read", read)
        monkeypatch.setattr(positioning, "_CACHE", {})
        monkeypatch.setattr(positioning, "_PENDING", {})
        positioning.listed_positioning("ZZZZ")
        at, _ = positioning._CACHE["ZZZZ"]
        assert positioning._fresh("ZZZZ", at + positioning.MISS_TTL_S - 1) is not None
        assert positioning._fresh("ZZZZ", at + positioning.MISS_TTL_S + 1) is None

    def test_a_future_payment_is_payable_not_paid(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bitget_positioning, "_safe", lambda entry, **kw: [
            {"ex_dividend_date": "2026-09-25", "amount": 0.25, "payment_date": "2026-10-01"}])
        line = bitget_positioning.next_ex_dividend("NVDA", today=date(2026, 9, 28))
        assert line is not None and "payable 2026-10-01" in line

    def test_a_dividend_question_leads_with_the_dated_dividend_line(self) -> None:
        from argus.lui.research.fundamentals import _fundamentals_focus

        lines = ["Valuation on 2026-09-28: P/E (trailing 12m) 28.1.",
                 "NVDA corporate actions: last cash dividend $0.25 a share, ex 09 Sep 2026."]
        out = _fundamentals_focus(lines, "what is NVDA's dividend", "NVDA")
        assert out[0].startswith("Bottom line: NVDA corporate actions")
