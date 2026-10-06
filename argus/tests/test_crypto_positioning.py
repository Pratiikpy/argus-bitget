"""Crypto positioning (`market/crypto_oi.py`, `lui/research/crypto_positioning.py`): perpetual open
interest across venues with one failing and named, the 7-day change over only the venues that have
history, Deribit put/call open-interest ratio and max pain on a book small enough to do by hand,
and None for questions that belong to other readers. Every fetch is replaced; nothing here
touches the network."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from argus.lui.research import crypto_positioning
from argus.market import crypto_oi, deribit
from argus.market.crypto_oi import Contract
from argus.truth import http
from argus.truth.failures import ErrorKind, RpcError

SRC = Path(__file__).resolve().parents[1] / "src" / "argus"
NOW = 1_791_219_300_000
TARGET = NOW - crypto_oi.WEEK_MS
TODAY = date(2026, 10, 5)

JUDGE_OI = ("What is the current Bitcoin perpetual open interest in USD across major exchanges "
            "and has it risen or fallen over the last 7 days?")
JUDGE_OPTIONS = ("What is the put/call open interest ratio for BTC options on Deribit right now "
                 "and where is max pain for the next monthly expiry?")


def _blocked(status: int = 451) -> RpcError:
    return RpcError(ErrorKind.DOMAIN, "Service unavailable from a restricted location",
                    http_status=status)


def _venues(*, bybit: Any = None, binance_history: bool = True, okx_rows: list[Any] | None = None,
            binance_down: bool = False) -> Callable[..., Any]:
    """A fake `http.fetch_json` serving every venue's documented response shape for BTC.

    Bitget 1,000 BTC at $80,000 (= $80m); Binance 2,000 BTC = $160m now and 1,900 BTC = $152m a
    week ago; OKX $80m; Hyperliquid 500 BTC at $80,000 = $40m; Deribit $40m. ``bybit`` None means
    Bybit refuses (HTTP 451), otherwise the dict of its responses.
    """

    def fake(url: str, *, params: dict[str, Any] | None = None, **_: Any) -> Any:
        p = params or {}
        if "/api/v3/market/tickers" in url:
            return {"code": "00000", "data": [{"symbol": "BTCUSDT", "openInterest": "1000",
                                               "markPrice": "80000", "ts": str(NOW)}]}
        if "openInterestHist" in url:
            if binance_down:
                raise _blocked()
            if p["period"] == "5m":
                return [{"sumOpenInterest": "2000", "sumOpenInterestValue": "160000000",
                         "timestamp": NOW}]
            return ([{"sumOpenInterest": "1900", "sumOpenInterestValue": "152000000",
                      "timestamp": TARGET}] if binance_history else [])
        if "bybit" in url:
            if bybit is None:
                raise _blocked()
            for key, body in bybit.items():
                if key in url:
                    return body
        if "/api/v5/public/open-interest" in url:
            return {"code": "0", "data": [{"oiUsd": "80000000", "oiCcy": "1000",
                                           "ts": str(NOW)}]}
        if "open-interest-history" in url:
            return {"code": "0", "data": okx_rows or []}
        if "hyperliquid" in url:
            return [{"universe": [{"name": "ETH"}, {"name": "BTC"}]},
                    [{"openInterest": "9", "markPx": "1"}, {"openInterest": "500",
                                                            "markPx": "80000"}]]
        if "get_book_summary_by_instrument" in url:
            return {"result": [{"open_interest": 40_000_000, "mark_price": 80000,
                                "creation_timestamp": NOW}]}
        raise AssertionError(f"unexpected url {url} {p}")

    return fake


BYBIT_OK = {
    "/v5/market/tickers": {"result": {"list": [{"openInterest": "2000", "singleOpenInterest":
                                                "1000", "markPrice": "80000"}]}, "time": NOW},
    "/v5/market/open-interest": {"result": {"list": [{"openInterest": "1800",
                                                      "singleOpenInterest": "900",
                                                      "timestamp": str(TARGET)}]}},
    "/v5/market/mark-price-kline": {"result": {"list": [[str(TARGET), "70000", "71000",
                                                         "69000", "70500"]]}},
}


# --------------------------------------------------------------------------------------------
# perpetual open interest


def test_the_sum_covers_the_venues_that_answered_and_names_the_one_that_did_not(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http, "fetch_json", _venues(bybit=None))
    reading = crypto_oi.read("BTC", now_ms=NOW)
    assert reading.total_usd == pytest.approx(160e6 + 80e6 + 80e6 + 40e6 + 40e6)
    assert set(reading.answered) == {"Binance", "OKX", "Bitget", "Hyperliquid", "Deribit"}
    assert [name for name, _ in reading.failed] == ["Bybit"]
    assert "451" in reading.failed[0][1]


def test_the_seven_day_change_uses_only_the_venues_with_both_readings(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http, "fetch_json", _venues(bybit=None))
    reading = crypto_oi.read("BTC", now_ms=NOW)
    # Only Binance has history; the other four venues must not move the figure.
    assert reading.change is not None
    assert reading.change.venues == ("Binance",)
    assert reading.change.now_usd == pytest.approx(160e6)
    assert reading.change.past_usd == pytest.approx(152e6)
    assert reading.change.pct == pytest.approx(160 / 152 - 1)
    assert reading.change.coin_pct == pytest.approx(2000 / 1900 - 1)


def test_bybit_counts_the_single_side_and_values_the_past_at_the_price_then(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http, "fetch_json", _venues(bybit=BYBIT_OK, binance_down=True))
    reading = crypto_oi.read("BTC", now_ms=NOW)
    bybit = next(v for v in reading.venues if v.venue == "Bybit")
    assert bybit.now is not None and bybit.past is not None
    assert bybit.now.coin == 1000 and bybit.now.usd == pytest.approx(80e6)  # not the doubled 2000
    assert bybit.past.usd == pytest.approx(900 * 70000)
    assert reading.change is not None and reading.change.venues == ("Bybit",)
    assert reading.change.pct == pytest.approx(80e6 / 63e6 - 1)
    assert [name for name, _ in reading.failed] == ["Binance"]


def test_a_venue_whose_history_is_empty_is_left_out_of_the_change_and_says_why(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http, "fetch_json", _venues(bybit=None, binance_history=False))
    reading = crypto_oi.read("BTC", now_ms=NOW)
    assert reading.change is None
    binance = next(v for v in reading.venues if v.venue == "Binance")
    assert binance.now is not None and "no reading" in binance.history


def test_okx_history_rows_are_used_when_the_endpoint_returns_them(
        monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [[str(TARGET), "7200000", "900", "63000000"]]
    monkeypatch.setattr(http, "fetch_json", _venues(bybit=None, okx_rows=rows,
                                                    binance_history=False))
    reading = crypto_oi.read("BTC", now_ms=NOW)
    assert reading.change is not None and reading.change.venues == ("OKX",)
    assert reading.change.pct == pytest.approx(80 / 63 - 1)


def test_a_reading_far_from_seven_days_ago_is_not_used() -> None:
    far = crypto_oi.Point(1.0, 1.0, TARGET - 20 * crypto_oi.HOUR_MS)
    assert crypto_oi._within(far, TARGET) is None


def test_every_venue_failing_gives_a_named_bottom_line_and_no_figure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(url: str, **_: Any) -> Any:
        raise _blocked(403)

    monkeypatch.setattr(http, "fetch_json", refuse)
    said = crypto_positioning.lines("BTC open interest across exchanges")
    assert said is not None and said[0].startswith("Bottom line: no exchange returned")
    assert "Binance (HTTP 403" in said[0] and said[-1] == "Not advice."


def test_lines_for_the_judge_open_interest_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http, "fetch_json", _venues(bybit=None))
    # the fixtures' histories are stamped around NOW: the clock is held there, or the 7-day
    # window walks off them a day after they were written
    monkeypatch.setattr(crypto_oi.time, "time", lambda: NOW / 1000)
    said = crypto_positioning.lines(JUDGE_OI)
    assert said is not None and said[-1] == "Not advice."
    assert said[0].startswith("Bottom line: BTC perpetual open interest is about $400.0m")
    assert "up 5.3%" in said[0] and "Binance" in said[0]
    text = "\n".join(said)
    assert "Did not answer: Bybit (HTTP 451" in text
    assert "/futures/data/openInterestHist" in text and "get_book_summary_by_instrument" in text
    assert "/api/v3/market/tickers" in text and "metaAndAssetCtxs" in text


def test_a_coin_without_a_deribit_perpetual_skips_deribit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(url: str, **kw: Any) -> Any:
        seen.append(url)
        raise _blocked()

    monkeypatch.setattr(http, "fetch_json", fake)
    reading = crypto_oi.read("SOL", now_ms=NOW)
    assert "Deribit" not in [v.venue for v in reading.venues]
    assert not any("deribit" in u for u in seen)


def test_an_unlisted_coin_is_refused() -> None:
    with pytest.raises(crypto_oi.OiError):
        crypto_oi.read("PEPE")


# --------------------------------------------------------------------------------------------
# Deribit options


def _book() -> list[Contract]:
    """A weekly with one call, and a monthly worked by hand:

    calls 90:3, 100:10, 110:5; puts 90:6, 100:8, 110:2. Payout if it settles at 90 / 100 / 110:
      90:  puts 8*10 + 2*20 = 120
      100: calls 3*10 = 30, puts 2*10 = 20 -> 50
      110: calls 3*20 + 10*10 = 160
    so max pain is 100 and holders collect 50.
    """
    weekly, monthly = date(2026, 10, 9), date(2026, 10, 30)
    return [Contract(weekly, 100.0, True, 10.0),
            Contract(monthly, 90.0, True, 3.0), Contract(monthly, 100.0, True, 10.0),
            Contract(monthly, 110.0, True, 5.0), Contract(monthly, 90.0, False, 6.0),
            Contract(monthly, 100.0, False, 8.0), Contract(monthly, 110.0, False, 2.0)]


def test_max_pain_on_a_book_done_by_hand() -> None:
    monthly = [c for c in _book() if c.expiry == date(2026, 10, 30)]
    assert crypto_oi.pain_at(monthly, 90) == 120
    assert crypto_oi.pain_at(monthly, 100) == 50
    assert crypto_oi.pain_at(monthly, 110) == 160
    assert crypto_oi.max_pain(monthly, spot=105.0) == (100.0, 50.0)


def test_max_pain_ties_go_to_the_strike_nearest_the_index() -> None:
    tied = [Contract(date(2026, 10, 30), 100.0, True, 1.0),
            Contract(date(2026, 10, 30), 110.0, False, 1.0)]
    # settling at 100 pays the put 10; at 110 pays the call 10: a tie either way
    assert crypto_oi.pain_at(tied, 100) == crypto_oi.pain_at(tied, 110) == 10
    assert crypto_oi.max_pain(tied, spot=109.0) == (110.0, 10.0)
    assert crypto_oi.max_pain(tied, spot=101.0) == (100.0, 10.0)
    assert crypto_oi.max_pain([], spot=1.0) is None


def test_the_monthly_is_the_last_friday_and_a_lapsed_expiry_is_never_offered() -> None:
    assert crypto_oi.is_monthly(date(2026, 10, 30)) and crypto_oi.is_monthly(date(2026, 11, 27))
    assert not crypto_oi.is_monthly(date(2026, 10, 23))  # a Friday, but not the last
    assert not crypto_oi.is_monthly(date(2026, 10, 29))  # the last Thursday
    expiries = [date(2026, 10, 2), date(2026, 10, 9), date(2026, 10, 30), date(2026, 11, 27)]
    assert crypto_oi.pick_expiry(expiries, TODAY) == date(2026, 10, 30)
    assert crypto_oi.pick_expiry(expiries, date(2026, 10, 31)) == date(2026, 11, 27)
    assert crypto_oi.pick_expiry(expiries, TODAY, "weekly") == date(2026, 10, 9)
    assert crypto_oi.pick_expiry(expiries, TODAY, "monthly", month=11) == date(2026, 11, 27)
    assert crypto_oi.pick_expiry(expiries, date(2026, 12, 1)) is None


def test_put_call_ratio_whole_book_and_monthly_expiry() -> None:
    book = crypto_oi.options_read("BTC", contracts=_book(), spot=104.0, today=TODAY)
    assert book.calls_oi == 28 and book.puts_oi == 16
    assert book.ratio == pytest.approx(16 / 28)
    assert book.expiry is not None and book.expiry.expiry == date(2026, 10, 30)
    assert book.expiry.calls_oi == 18 and book.expiry.puts_oi == 16
    assert book.expiry.ratio == pytest.approx(16 / 18)
    assert book.expiry.max_pain == 100.0 and book.expiry.pain_usd == 50.0
    assert book.expiry.pain_at_spot_usd == crypto_oi.pain_at(
        [c for c in _book() if c.expiry == date(2026, 10, 30)], 104.0)


def _deribit_rows() -> dict[str, Any]:
    rows = [{"instrument_name": f"BTC-30OCT26-{k}-{t}", "open_interest": oi}
            for k, t, oi in [(90, "C", 3), (100, "C", 10), (110, "C", 5), (90, "P", 6),
                             (100, "P", 8), (110, "P", 2)]]
    rows.append({"instrument_name": "BTC-9OCT26-100-C", "open_interest": 10})
    rows.append({"instrument_name": "garbage", "open_interest": 99})
    return {"result": rows}


def _deribit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str, **_: Any) -> Any:
        if "get_book_summary_by_currency" in url:
            return _deribit_rows()
        if "get_index_price" in url:
            return {"result": {"index_price": 104.0}}
        raise AssertionError(url)

    monkeypatch.setattr(http, "fetch_json", fake)
    monkeypatch.setattr(deribit, "today", lambda: TODAY)


def test_lines_for_the_judge_options_question(monkeypatch: pytest.MonkeyPatch) -> None:
    _deribit(monkeypatch)
    said = crypto_positioning.lines(JUDGE_OPTIONS)
    assert said is not None and said[-1] == "Not advice."
    first = said[0]
    assert first.startswith("Bottom line: Deribit's BTC put/call open-interest ratio is 0.57")
    assert "0.89 for the 30 Oct 2026 monthly expiry" in first and "max pain is $100" in first
    assert "3.8% below" in first
    text = "\n".join(said)
    assert "get_book_summary_by_currency?currency=BTC&kind=option" in text
    assert "(7 instruments" in text  # the unparseable row is not counted


def test_a_max_pain_question_leads_with_max_pain(monkeypatch: pytest.MonkeyPatch) -> None:
    _deribit(monkeypatch)
    said = crypto_positioning.lines("Where is BTC max pain on Deribit this month?")
    assert said is not None
    assert said[0].startswith("Bottom line: for Deribit's 30 Oct 2026 monthly BTC expiry max pain "
                              "is $100")


def test_an_option_book_that_will_not_load_is_said_not_raised(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(url: str, **_: Any) -> Any:
        raise _blocked(503)

    monkeypatch.setattr(http, "fetch_json", refuse)
    said = crypto_positioning.lines("ETH put/call ratio")
    assert said is not None and "did not answer" in said[0] and said[-1] == "Not advice."


def test_a_question_with_both_halves_answers_the_first_asked_first(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str, **kw: Any) -> Any:
        if "deribit" in url and ("by_currency" in url or "index_price" in url):
            return (_deribit_rows() if "by_currency" in url
                    else {"result": {"index_price": 104.0}})
        return _venues(bybit=None)(url, **kw)

    monkeypatch.setattr(http, "fetch_json", fake)
    monkeypatch.setattr(deribit, "today", lambda: TODAY)
    said = crypto_positioning.lines("BTC perpetual open interest across exchanges and the "
                                    "put/call ratio on Deribit options")
    assert said is not None and said[0].startswith("Bottom line: BTC perpetual open interest")
    assert any(line.startswith("Bottom line: Deribit's BTC put/call") for line in said)
    assert said.count("Not advice.") == 1 and said[-1] == "Not advice."


# --------------------------------------------------------------------------------------------
# what is not ours


@pytest.mark.parametrize("question", [
    "What is the BTC funding rate?",
    "What is the BTC price?",
    "What is the 25-delta skew on BTC options this month?",
    "Compare the VIX to BTC's at-the-money implied volatility",
    "What is the BTC put/call volume ratio?",
    "How much open interest is there?",
    "What is the open interest in AAPL options?",
    "hello",
    "",
])
def test_unrelated_questions_return_none(question: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def never(url: str, **_: Any) -> Any:
        raise AssertionError(f"a question that is not ours fetched {url}")

    monkeypatch.setattr(http, "fetch_json", never)
    assert crypto_positioning.lines(question) is None


# --------------------------------------------------------------------------------------------
# layering


def test_the_data_module_does_not_reach_up_and_hosts_come_from_endpoints() -> None:
    data = (SRC / "market" / "crypto_oi.py").read_text(encoding="utf-8")
    answer = (SRC / "lui" / "research" / "crypto_positioning.py").read_text(encoding="utf-8")
    assert "argus.lui" not in data
    assert "https://api.bitget.com" not in data + answer
    assert "from argus.truth.endpoints import BITGET_API" in data
