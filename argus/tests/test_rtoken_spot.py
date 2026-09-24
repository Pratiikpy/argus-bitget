"""Spot rTokens and the overnight hedge: the calendar, the night's return, the hedge, the console.

Everything here is offline. The live path (Bitget's spot and perpetual candles) is checked by
`python -m argus.eval.copilot_hedge` and by asking the console; its port was verified against
Ballast's own nights to 0.0 on 275 of 276 TSLA sessions (2026-09-24).
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pytest

from argus.desk.rtoken_hedge import HedgeUnavailable, overnight_hedge, render
from argus.lui.research import ResearchKind, _spot_rtoken, detect
from argus.market.rtoken_spot import (
    close_utc,
    early_closes,
    holidays,
    is_trading_day,
    next_session,
    open_utc,
    overnight_returns,
    overnight_window,
)


class TestTheCalendar:
    def test_good_friday_and_thanksgiving_2026_are_closed(self) -> None:
        assert date(2026, 4, 3) in holidays(2026)
        assert date(2026, 11, 26) in holidays(2026)
        assert not is_trading_day(date(2026, 4, 3))

    def test_a_saturday_new_year_is_not_observed(self) -> None:
        assert date(2021, 12, 31) not in holidays(2022)
        assert date(2022, 1, 1) not in holidays(2022)

    def test_the_day_after_thanksgiving_closes_at_one(self) -> None:
        assert date(2026, 11, 27) in early_closes(2026)
        assert close_utc(date(2026, 11, 27)) == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)

    def test_the_close_follows_daylight_saving(self) -> None:
        assert close_utc(date(2026, 7, 1)).hour == 20
        assert close_utc(date(2026, 1, 5)).hour == 21

    def test_friday_spans_the_weekend_and_a_holiday_is_skipped(self) -> None:
        start, end = overnight_window(date(2026, 4, 2))  # Thursday before Good Friday
        assert end == open_utc(date(2026, 4, 6))
        assert next_session(date(2026, 9, 25)) == date(2026, 9, 28)
        assert (end - start) > timedelta(hours=80)


def _bars(days: int, drift: float) -> dict[int, tuple[float, float]]:
    """Hourly bars whose price multiplies by ``exp(drift)`` every hour."""
    start = datetime(2026, 3, 2, tzinfo=UTC)
    out = {}
    price = 100.0
    for h in range(days * 24):
        ts = int((start + timedelta(hours=h)).timestamp() * 1000)
        nxt = price * math.exp(drift)
        out[ts] = (price, nxt)
        price = nxt
    return out


def test_the_night_runs_from_the_close_bar_to_the_open_bar() -> None:
    nights = overnight_returns(_bars(10, 0.001))
    monday = date(2026, 3, 2)
    assert monday in nights
    close_at, open_at = overnight_window(monday)
    # close at 21:00Z (EST) is the close of the 21:00 bar; the 14:30Z open is snapped to 15:00Z,
    # whose bar opens 18 hourly steps after the 21:00 bar closes
    hours = (open_at.replace(minute=0) + timedelta(hours=1) - close_at).total_seconds() / 3600
    assert nights[monday] == pytest.approx(0.001 * (hours - 1))
    assert overnight_returns({}) == {}


def _nights(n: int, slope: float, noise: float) -> tuple[dict[date, float], dict[date, float]]:
    first = date(2026, 1, 5)
    perp: dict[date, float] = {}
    spot: dict[date, float] = {}
    for i in range(n):
        d = first + timedelta(days=i)
        x = 0.01 * math.sin(i * 0.9)
        perp[d] = x
        spot[d] = slope * x + noise * math.cos(i * 2.1)
    return spot, perp


def test_the_same_name_hedge_is_sized_and_tested_out_of_sample() -> None:
    spot, perp = _nights(120, 1.0, 0.0002)
    h = overnight_hedge("RXUSDT", "XUSDT", spot, perp)
    assert h.ratio == pytest.approx(-1.0, abs=0.02)
    assert h.held_out_nights == 36
    assert h.held_out_variance_removed > 0.99
    assert h.cost_bp_per_night == pytest.approx(12.0, abs=0.3)
    lines = render(h, "X")
    assert lines[0].startswith("Actionable: to carry RXUSDT") and "short 1.00 XUSDT" in lines[0]
    assert any("Tested on nights it had not seen" in line for line in lines)


def test_too_few_nights_is_refused() -> None:
    spot, perp = _nights(20, 1.0, 0.0)
    with pytest.raises(HedgeUnavailable):
        overnight_hedge("RXUSDT", "XUSDT", spot, perp)


@pytest.mark.parametrize(("text", "ticker"), [
    ("hedge my rTSLA overnight", "TSLA"),
    ("I hold RNVDAUSDT, protect it over the weekend", "NVDA"),
    ("my AAPL rToken over the weekend", "AAPL"),
    ("how do I hedge tokenized MSFT", "MSFT"),
    ("rtoken of META overnight", "META"),
])
def test_a_spot_rtoken_is_read_as_a_holding(text: str, ticker: str) -> None:
    assert _spot_rtoken(text) == ticker
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.HEDGE
    assert request.spot == f"R{ticker}USDT" and request.symbols == (f"{ticker}USDT",)


def test_ordinary_words_are_not_rtokens() -> None:
    assert _spot_rtoken("what is the return on TSLA overnight") is None
    assert _spot_rtoken("reduce my risk") is None


def test_an_imperative_hedge_is_a_plan_request_not_an_order() -> None:
    request = detect("hedge my TSLA")
    assert request is not None and request.kind is ResearchKind.HEDGE
    assert request.spot is None


def test_the_spot_holding_survives_a_model_reading_of_the_same_kind(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """On the live console the model read "I hold RNVDAUSDT, protect it over the weekend" as a
    hedge with no holding and the answer was a refusal; the patterns' spot reading must stand."""
    from argus.lui import research, server

    class Hedger:
        def complete_json(self, messages: object, **kwargs: object) -> dict[str, object]:
            return {"kind": "hedge", "confidence": 0.95}

    captured: dict[str, object] = {}

    def fake_payload(text: str, prior: object, request: research.ResearchRequest, *args: object,
                     **kwargs: object) -> dict[str, object]:
        captured["request"] = request
        return {}

    monkeypatch.setattr(server, "_model_for", lambda visitor: Hedger())
    monkeypatch.setattr(server, "worth_asking_the_model", lambda text, now=None: True)
    monkeypatch.setattr(server, "_research_payload", fake_payload)
    server.handle_ask("I hold RNVDAUSDT, how do I protect it over the weekend?", [])
    request = captured["request"]
    assert isinstance(request, research.ResearchRequest)
    assert request.spot == "RNVDAUSDT"
