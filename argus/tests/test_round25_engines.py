"""Round 25's new engines, run offline: stated periods, company names, earnings reactions and
macro-event moves. Each fixture is a small hand-made series whose answer can be read off it."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from argus.lui.research import earnings_moves, macro_moves, performance
from argus.market import company_names


class TestStatedPeriods:
    NOW = datetime(2026, 10, 3, 12, tzinfo=UTC)

    @pytest.mark.parametrize(("asked", "start", "end"), [
        ("between 1 July and 30 September", datetime(2026, 7, 1, tzinfo=UTC),
         datetime(2026, 10, 1, tzinfo=UTC)),
        ("from 15 August to 15 September", datetime(2026, 8, 15, tzinfo=UTC),
         datetime(2026, 9, 16, tzinfo=UTC)),
        # a calendar month ends when its last daily candle closes, 00:00 the day after
        ("in September", datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC)),
        ("last year", datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
    ])
    def test_a_stated_range_is_the_period_read(self, asked: str, start: datetime,
                                               end: datetime) -> None:
        period = performance.asked_period(asked, self.NOW)
        assert period is not None and (period.start, period.end) == (start, end)

    def test_since_and_spans(self) -> None:
        since = performance.asked_period("return on gold since 1 March", self.NOW)
        assert since is not None and since.start == datetime(2026, 3, 1, tzinfo=UTC)
        year = performance.asked_period("ETH one-year return", self.NOW)
        assert year is not None and (year.end - year.start).days == 365
        three = performance.asked_period("英伟达过去三个月涨了多少", self.NOW)
        assert three is not None and (three.end - three.start).days == 90


class TestCompanyNames:
    def test_names_resolve_and_words_do_not(self) -> None:
        assert company_names.us_ticker("When does Micron report next") == ("Micron", "MU")
        assert company_names.us_ticker("how did Ford do") == ("Ford", "F")
        assert company_names.us_ticker("what is the target price") is None
        assert company_names.us_ticker("Hello there") is None

    def test_build_maps_an_issuer_to_its_first_ticker(self) -> None:
        built = company_names.build({
            "0": {"cik_str": 1, "ticker": "BIGCO", "title": "Bigcorp Holdings Inc"},
            "1": {"cik_str": 1, "ticker": "BIGCO-PB", "title": "Bigcorp Holdings Inc"},
        })
        assert built["full"] == {"BIGCORP": "BIGCO"} or built["first"].get("BIGCORP") == "BIGCO"


class _Filing:
    def __init__(self, form: str, accepted: datetime, items: tuple[str, ...] = ()) -> None:
        self.form, self.accepted, self.items = form, accepted, items


class TestEarningsMoves:
    def test_an_after_close_release_moves_the_next_session(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import evidence

        # results after the close on Tue 21 Jul 2026 (20:05 UTC = 16:05 New York); 10-Q later
        released = datetime(2026, 7, 21, 20, 5, tzinfo=UTC)
        filings = [_Filing("8-K", released, ("2.02",)),
                   _Filing("10-Q", released + timedelta(days=5))]

        class _Edgar:
            def filings(self, *_a: object, **_k: object) -> list[_Filing]:
                return filings

        monkeypatch.setattr(evidence, "EdgarSource", _Edgar)
        closes = {date(2026, 7, 20): 100.0, date(2026, 7, 21): 100.0, date(2026, 7, 22): 90.0}
        market = {date(2026, 7, 21): 500.0, date(2026, 7, 22): 495.0}
        monkeypatch.setattr(earnings_moves, "_days",
                            lambda ticker: market if ticker == "SPY" else closes)
        found = earnings_moves.reactions("XYZ")
        assert len(found) == 1
        assert found[0].moved_day == date(2026, 7, 22) and found[0].move == pytest.approx(-0.10)
        assert found[0].market == pytest.approx(-0.01)
        lines = earnings_moves.reaction_lines(["XYZ"])
        assert lines is not None and lines[0].startswith("Bottom line: XYZ moved -10.0%")


class TestMacroMoves:
    def test_release_hour_moves_against_ordinary_hours(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.research import event_reactions

        base = datetime(2026, 3, 2, tzinfo=UTC)
        closes: dict[datetime, float] = {}
        price = 100.0
        for h in range(24 * 60):
            at = base + timedelta(hours=h)
            closes[at] = price
            price *= 1.001 if h % 2 else 0.999
        releases = [base + timedelta(days=d, hours=12, minutes=30) for d in (5, 15, 25, 35, 45)]
        for at in releases:
            after = at.replace(minute=0) + timedelta(hours=1)
            closes[after] = closes[after - timedelta(hours=1)] * 1.02
        monkeypatch.setattr(event_reactions, "cpi_releases", lambda: (releases, "test"))
        monkeypatch.setattr(macro_moves, "_closes", lambda symbol: closes)
        lines = macro_moves.event_move_lines("BTCUSDT", "CPI")
        assert lines is not None
        assert "on CPI releases BTC has moved 2.00% either way in the hour after" in lines[0]
        assert "over 5 releases" in lines[0]
