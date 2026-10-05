"""Market-breadth answers (`lui/research/market_breadth.py`): the six round-43 questions that were
answered from the wrong engine, read from synthetic daily series, offline."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pytest

from argus.lui.research import market_breadth as mb

TODAY = date(2026, 10, 5)

Q_SECTOR = ("Which S&P 500 sectors have led over the past month, and is money rotating from "
            "tech into defensives?")
Q_MAG7 = "How have the Magnificent Seven performed against equal-weight S&P this quarter?"
Q_REGIME = ("Is the S&P 500 in a bull or bear regime based on its 200 day moving average, and "
            "how far is it from it?")
Q_VOL = "Is volatility in a high or low regime right now for BTC and for the S&P 500?"
Q_VALUE = "Is the Nasdaq 100 more expensive than its 10 year average on forward earnings?"
Q_DRIVER = "Is gold or the dollar the stronger driver of silver this quarter?"


def _days(count: int, end: date = TODAY, weekends: bool = False) -> list[date]:
    out: list[date] = []
    d = end
    while len(out) < count:
        if weekends or d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _series(ticker: str, fn: Callable[[date, int], float], count: int = 400,
            weekends: bool = False) -> mb.Series:
    dates = _days(count, weekends=weekends)
    return mb.Series(ticker, dates, [fn(d, i) for i, d in enumerate(dates)])


def _install(monkeypatch: pytest.MonkeyPatch, table: dict[str, mb.Series]) -> None:
    def fake(ticker: str) -> mb.Series:
        if ticker not in table:
            raise mb.BreadthError(f"no fixture for {ticker}")
        return table[ticker]

    monkeypatch.setattr(mb, "_series", fake)
    monkeypatch.setattr(mb, "_today", lambda: TODAY)


def _step(base: float, when: date, after: float) -> Callable[[date, int], float]:
    """A price that is ``base`` through ``when`` and ``after`` once it has passed."""
    return lambda d, i: base if d <= when else after


# --------------------------------------------------------------------------- windows


def test_window_quarter_to_date_uses_the_prior_quarter_end() -> None:
    w = mb.parse_window("how did X do this quarter", TODAY)
    assert w.base == date(2026, 9, 30)
    assert "quarter to date" in w.label


def test_window_last_quarter_is_the_previous_full_quarter() -> None:
    w = mb.parse_window("last quarter", TODAY)
    assert (w.base, w.end) == (date(2026, 6, 30), date(2026, 9, 30))


def test_window_ytd_month_and_trailing_units() -> None:
    assert mb.parse_window("ytd", TODAY).base == date(2025, 12, 31)
    assert mb.parse_window("over the past 6 months", TODAY).base == date(2026, 4, 5)
    assert mb.parse_window("past year", TODAY).base == date(2025, 10, 5)
    assert mb.parse_window("last 2 weeks", TODAY).base == date(2026, 9, 21)
    assert mb.parse_window("this month", TODAY).base == date(2026, 9, 30)


def test_window_default_is_stated_not_silent() -> None:
    assert "no period was named" in mb.parse_window("hello", TODAY).label
    assert "year to date" in mb.parse_window("hello", TODAY, default="ytd").label
    assert "3 months" in mb.parse_window("hello", TODAY, default="3 months").label


def test_months_back_clamps_the_day() -> None:
    assert mb._months_back(date(2026, 3, 31), 1) == date(2026, 2, 28)
    assert mb._months_back(date(2026, 1, 15), 2) == date(2025, 11, 15)


# --------------------------------------------------------------------------- data layer


def test_parse_chart_uses_the_exchange_offset_and_drops_gaps() -> None:
    payload = {"chart": {"result": [{
        "meta": {"gmtoffset": -14400},
        "timestamp": [1_790_000_000, 1_790_086_400, 1_790_172_800],
        "indicators": {"quote": [{"close": [10.0, None, 12.0]}]}}]}}
    s = mb._parse_chart("X", payload)
    assert s.closes == [10.0, 12.0] and len(s.dates) == 2
    with pytest.raises(mb.BreadthError):
        mb._parse_chart("X", {"chart": {"result": None}})


def test_series_fetches_once_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    stamps = [1_700_000_000 + 86_400 * i for i in range(40)]
    payload = {"chart": {"result": [{"timestamp": stamps, "meta": {},
                                     "indicators": {"quote": [{"close": [5.0 + i for i in
                                                                         range(40)]}]}}]}}

    def fake(url: str, **kw: Any) -> dict[str, Any]:
        calls.append(url)
        return payload

    monkeypatch.setattr(mb.http, "fetch_json", fake)
    mb._cache.clear()
    first = mb._series("^GSPC")
    assert mb._series("^GSPC") is first and len(calls) == 1
    assert "%5EGSPC" in calls[0] and "period1=0" in calls[0]
    mb._cache.clear()


def test_series_too_short_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"chart": {"result": [{"timestamp": [1_700_000_000, 1_700_086_400], "meta": {},
                                     "indicators": {"quote": [{"close": [1.0, 2.0]}]}}]}}
    monkeypatch.setattr(mb.http, "fetch_json", lambda url, **kw: payload)
    mb._cache.clear()
    with pytest.raises(mb.BreadthError):
        mb._series("TINY")
    mb._cache.clear()


# --------------------------------------------------------------------------- 1. sector rotation


def _sector_table(tech_prior: float, tech_last: float) -> dict[str, mb.Series]:
    """Every fund drifts; XLK gets the given growth over the three months before the last month
    and over the last month; the defensives are flat then fall 3% in the last month."""
    month = mb._months_back(TODAY, 1)
    quarter = mb._months_back(TODAY, 4)
    table: dict[str, mb.Series] = {}
    for t in mb.SECTORS:
        def price(d: date, i: int, t: str = t) -> float:
            if t == "XLK":
                lead = tech_prior if d > quarter else 0.0
                lag = tech_last if d > month else 0.0
                return 100.0 * (1 + lead) * (1 + lag)
            if t in mb.DEFENSIVES:
                return 100.0 * (0.97 if d > month else 1.0)
            return 100.0
        table[t] = _series(t, price, count=300)
    return table


def test_sector_rotation_yes_when_tech_led_then_lagged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _sector_table(tech_prior=0.10, tech_last=-0.08))
    out = mb.lines(Q_SECTOR)
    assert out is not None
    assert out[0].startswith("Bottom line: ")
    assert "Yes, by price strength" in out[0]
    assert out[-1].startswith("Data: ") and out[-1].endswith("Not advice.")
    joined = "\n".join(out)
    # tech: -8% against defensives -3% over the last month, so lagging by 5.0 points
    assert "lagged them by 5.0 points" in joined
    assert "XLK -8.0%" in joined and "XLP -3.0%" in joined
    assert "not fund flows" in joined


def test_sector_rotation_no_when_tech_keeps_leading(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _sector_table(tech_prior=0.05, tech_last=0.04))
    out = mb.lines(Q_SECTOR)
    assert out is not None
    assert out[0].count("No:") == 1 and "Technology (XLK) +4.0%" in out[0]
    assert "7.0 points" in out[0]  # +4% against defensives at -3%


def test_sector_rotation_is_not_a_crypto_or_single_stock_question(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _sector_table(0.0, 0.0))
    assert mb.lines("Which crypto sectors are leading this month?") is None
    assert mb.lines("How does AAPL compare against its sector on valuation?") is None


def test_sector_failure_is_one_honest_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {})
    out = mb.lines(Q_SECTOR)
    assert out is not None and "could not be read just now" in out[0]


# --------------------------------------------------------------------------- 2. Magnificent Seven


def _mag_table(growth: dict[str, float], q3: dict[str, float]) -> dict[str, mb.Series]:
    """Flat through 30 Jun, ``q3`` growth by 30 Sep, then ``growth`` on top from 1 Oct."""
    table: dict[str, mb.Series] = {}
    for t in (*mb.MAG7, "RSP", "SPY"):
        def price(d: date, i: int, t: str = t) -> float:
            p = 100.0
            if d > date(2026, 6, 30):
                p *= 1 + q3[t]
            if d > date(2026, 9, 30):
                p *= 1 + growth[t]
            return p
        table[t] = _series(t, price, count=300)
    return table


def test_mag7_quarter_to_date_equal_weight_against_rsp(monkeypatch: pytest.MonkeyPatch) -> None:
    growth = dict.fromkeys(mb.MAG7, 0.0)
    growth.update(AAPL=0.07, NVDA=0.14, TSLA=-0.07, RSP=0.02, SPY=0.03)
    growth.update({"MSFT": 0.0, "GOOGL": 0.0, "AMZN": 0.0, "META": 0.0})
    q3 = dict.fromkeys((*mb.MAG7, "RSP", "SPY"), 0.0)
    q3.update(NVDA=0.30, RSP=-0.02, SPY=0.02)
    _install(monkeypatch, _mag_table(growth, q3))
    out = mb.lines(Q_MAG7)
    assert out is not None
    # (7 + 14 - 7 + 0*4) / 7 = 2.0% against RSP +2.0%: level, so it lagged by 0.0
    assert out[0].startswith("Bottom line: over the quarter to date the Magnificent Seven")
    assert "returned +2.0% against +2.0%" in out[0] and "+3.0% for the cap-weighted" in out[0]
    assert "lagged cap-weight by 1.0 points" in out[0]
    assert "NVDA +14.0%" in out[1] and "TSLA -7.0%" in out[1]
    assert "Measured from the close of 30 Sep 2026" in out[2]
    # two sessions into the quarter, so the full previous quarter is added
    prev = next(line for line in out if line.startswith("Only "))
    assert "Only 3 sessions" in prev and "last quarter (1 Jul to 30 Sep 2026)" in prev
    assert "returned +4.3%" in prev  # NVDA +30% / 7
    assert "-2.0% for the equal-weight S&P 500" in prev


def test_mag7_other_period_phrases_and_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    growth = dict.fromkeys((*mb.MAG7, "RSP", "SPY"), 0.0)
    q3 = dict.fromkeys((*mb.MAG7, "RSP", "SPY"), 0.01)
    _install(monkeypatch, _mag_table(growth, q3))
    for text in ("Did the Mag 7 beat RSP year to date?", "Mag7 vs equal weight over the past "
                 "6 months", "magnificent 7 ytd"):
        out = mb.lines(text)
        assert out is not None and out[0].startswith("Bottom line: over ")
    ytd = mb.lines("magnificent 7 ytd")
    assert ytd is not None and "the year to date" in ytd[0]
    assert not any(line.startswith("Only ") for line in ytd)


def test_mag7_missing_history_is_stated(monkeypatch: pytest.MonkeyPatch) -> None:
    table = _mag_table(dict.fromkeys((*mb.MAG7, "RSP", "SPY"), 0.0),
                       dict.fromkeys((*mb.MAG7, "RSP", "SPY"), 0.0))
    table["META"] = _series("META", lambda d, i: 100.0, count=20)
    _install(monkeypatch, table)
    out = mb.lines("magnificent seven year to date")
    assert out is not None and "META has no price history" in out[0]


def test_mag7_trigger_is_strict() -> None:
    assert mb.lines("How is NVDA doing this quarter?") is None


# --------------------------------------------------------------------------- 3. 200-day regime


def _wave(i: int) -> float:
    return 1000.0 * math.exp(0.0002 * i + 0.25 * math.sin(i / 90.0))


def test_regime_arithmetic_on_a_straight_line(monkeypatch: pytest.MonkeyPatch) -> None:
    # close = 1000 + i: the last 200 closes average (last - 99.5)
    s = _series("^GSPC", lambda d, i: 1000.0 + i, count=400)
    _install(monkeypatch, {"^GSPC": s})
    out = mb.lines(Q_REGIME)
    assert out is not None
    last = 1000.0 + 399
    avg = last - 99.5
    assert f"closed at {last:,.2f}" in out[0] and f"moving average of {avg:,.2f}" in out[0]
    assert f"{(last / avg - 1) * 100:.1f}% above" in out[0] and "a bull regime" in out[0]
    assert "201 sessions on that side" in out[0]  # 400 bars: 201 have an average
    assert "is rising" in out[1] and "at its highest close on record" in out[1]
    assert out[-1].startswith("Data: ") and out[-1].endswith("Not advice.")
    assert "No base rate is given" in "\n".join(out)  # 400 days is too short for one


def test_regime_below_with_base_rate_and_cross_date(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _series("^GSPC", lambda d, i: _wave(i), count=6500)
    _install(monkeypatch, {"^GSPC": s})
    out = mb.lines(Q_REGIME)
    assert out is not None
    text = "\n".join(out)
    closes = s.closes
    avg = sum(closes[-200:]) / 200
    assert f"moving average of {avg:,.2f}" in out[0]
    side = "above" if closes[-1] > avg else "below"
    assert f"% {side} its 200-day moving average" in out[0]
    assert "Base rate on the S&P 500's own history" in text
    assert "after a close above the 200-day average" in text and "after a close below" in text
    assert "cross was a" in text
    # the counts in the base-rate line add up to every session with a 63-day-forward window
    ups = [i for i in range(199, 6500 - 63)
           if closes[i] > sum(closes[i - 199:i + 1]) / 200]
    assert f"({len(ups):,} sessions)" in text


def test_regime_names_other_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {t: _series(t, lambda d, i: 100.0 + i % 7, count=300)
             for t in ("BTC-USD", "NVDA", "^NDX")}
    _install(monkeypatch, table)
    for text, ticker in (("Is bitcoin above its 200 day moving average?", "BTC-USD"),
                         ("Is NVDA in a bull regime on the 200-day MA?", "NVDA"),
                         ("How far is the Nasdaq 100 from its 200 day moving average?", "^NDX")):
        out = mb.lines(text)
        assert out is not None and ticker in out[-1]


def test_regime_too_little_history_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"BTC-USD": _series("BTC-USD", lambda d, i: 100.0 + i, count=120)})
    out = mb.lines("Is bitcoin above its 200 day moving average?")
    assert out is not None and "fewer than the 200 a 200-day average needs" in out[0]
    out = mb.lines("Is the S&P 500 below its 200 day moving average?")
    assert out is not None and "could not be read just now" in out[0]


def test_regime_trigger_needs_the_200_day_average_and_a_subject() -> None:
    assert mb.lines("Is the S&P 500 in a bull market?") is None
    assert mb.lines("what is the 200 day moving average?") is None  # no subject, no regime word


# --------------------------------------------------------------------------- 4. volatility regime


def _alternating(small: float, big: float, split: int) -> Callable[[date, int], float]:
    """Daily moves of +/-small up to bar ``split``, then +/-big."""
    state = {"p": 100.0}
    cache: dict[int, float] = {}

    def f(d: date, i: int) -> float:
        if i not in cache:
            move = small if i < split else big
            state["p"] *= (1 + move) if i % 2 else (1 - move)
            cache[i] = state["p"]
        return cache[i]
    return f


def test_vol_regime_labels_and_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    quiet = _series("^GSPC", _alternating(0.004, 0.004, 99999), count=1500)
    wild = _series("BTC-USD", _alternating(0.01, 0.04, 1500 - 40), count=2200, weekends=True)
    vix = _series("^VIX", lambda d, i: 10.0 + (i % 100) / 10.0, count=1500)
    _install(monkeypatch, {"^GSPC": quiet, "BTC-USD": wild, "^VIX": vix})
    out = mb.lines(Q_VOL)
    assert out is not None
    assert out[0].startswith("Bottom line: volatility right now: BTC is high")
    # alternating +/-4%: sample std of 30 log returns annualised by 365
    r_up, r_dn = math.log(1.04), math.log(0.96)
    mean = (r_up + r_dn) / 2
    std = math.sqrt(sum((r - mean) ** 2 for r in [r_up, r_dn] * 15) / 29)
    assert f"{std * math.sqrt(365):.1%}" in "\n".join(out)
    assert "the S&P 500 is normal" in out[0] or "the S&P 500 is low" in out[0] \
        or "the S&P 500 is high" in out[0]
    assert "the VIX (implied, S&P 500) is" in out[0] and "percentile of the last year" in out[0]
    assert "terciles of each asset's own five-year history" in "\n".join(out)
    assert out[-1].startswith("Data: ") and out[-1].endswith("Not advice.")


def test_vol_regime_low_when_calm_after_storm(monkeypatch: pytest.MonkeyPatch) -> None:
    calm = _series("ETH-USD", _alternating(0.05, 0.002, 1500), count=2000, weekends=True)
    _install(monkeypatch, {"ETH-USD": calm})
    out = mb.lines("Is ETH volatility in a low or high regime right now?")
    assert out is not None and "ETH is low" in out[0]


def test_vol_regime_vix_alone_reports_level_and_percentile(
        monkeypatch: pytest.MonkeyPatch) -> None:
    vix = _series("^VIX", lambda d, i: 10.0 + i * 0.02, count=1400)
    _install(monkeypatch, {"^VIX": vix})
    out = mb.lines("Is the VIX elevated right now?")
    assert out is not None and "the VIX is" in out[0] and "high" in out[0]


def test_vol_trigger_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {})
    assert mb.lines("Is volatility high?") is None  # no asset named
    assert mb.lines("What is BTC implied volatility on options, high or low?") is None
    assert mb.lines("What is the price of BTC?") is None


def test_vol_failure_is_stated(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {})
    out = mb.lines(Q_VOL)
    # nothing readable: every asset line says so rather than inventing a regime
    assert out is None or "not readable" in "\n".join(out)


def test_percentile_and_ordinal_helpers() -> None:
    values = sorted([1.0, 2.0, 3.0, 4.0])
    assert mb._percentile(values, 4.0) == pytest.approx(87.5)
    assert mb._percentile(values, 0.0) == 0.0
    assert mb._quantile(values, 0.5) == 2.5
    assert [mb._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 21, 112)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "21st", "112th"]
    assert mb._tercile(10) == "low" and mb._tercile(50) == "normal" and mb._tercile(90) == "high"


# --------------------------------------------------------------------------- 5. valuation


def _pe_rows() -> list[tuple[date, float]]:
    rows = []
    for m in range(0, 300):
        index = 2026 * 12 + 9 - m
        year, month = divmod(index, 12)
        rows.append((date(year, month + 1, 1), 20.0 if m < 120 else 10.0))
    rows[0] = (rows[0][0], 30.0)
    return rows


def test_valuation_nasdaq_cannot_be_answered_as_asked_but_gives_what_exists(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mb, "_sp_pe_history", _pe_rows)
    monkeypatch.setattr(mb, "_fund_multiple",
                        lambda t: (30.8, 29.2) if t == "QQQ" else (25.0, 24.8))
    out = mb.lines(Q_VALUE)
    assert out is not None
    assert out[0].startswith("Bottom line: that cannot be answered as asked")
    assert "QQQ, which tracks the Nasdaq 100, trades on 30.8x trailing earnings" in out[0]
    assert "29.2x on its holdings' earnings yield" in out[0]
    # latest 30.0 against ten years of 20.0 (and the latest): 120 rows incl. the 30.0 one
    avg = (30.0 + 119 * 20.0) / 120
    assert f"its 10-year average of {avg:.1f}x" in out[0]
    joined = "\n".join(out)
    assert "premium of 23% to the S&P 500" in joined  # 30.8 / 25.0 - 1
    assert "Yahoo returns no forward P/E" in joined
    assert "Nasdaq 100's own 10-year average is not available" in joined
    assert out[-1].startswith("Data: multpl.com") and out[-1].endswith("Not advice.")


def test_valuation_sp500_question_answers_from_the_history(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mb, "_sp_pe_history", _pe_rows)
    monkeypatch.setattr(mb, "_fund_multiple", lambda t: (None, None))
    out = mb.lines("Is the S&P 500 expensive compared with its 10 year average P/E?")
    assert out is not None
    assert out[0].startswith("Bottom line: The S&P 500's own multiple is 30.0x")
    assert "above its 10-year average" in out[0] and "forward one" not in out[0]


def test_valuation_history_failure_is_stated(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list[tuple[date, float]]:
        raise mb.BreadthError("multpl.com did not answer (TimeoutError)")

    monkeypatch.setattr(mb, "_sp_pe_history", boom)
    monkeypatch.setattr(mb, "_fund_multiple", lambda t: (30.8, None))
    out = mb.lines(Q_VALUE)
    assert out is not None
    joined = "\n".join(out)
    assert "P/E history could not be read just now" in joined
    assert "10-year average of" not in joined


def test_valuation_yahoo_failure_is_stated(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ticker: str) -> tuple[float | None, float | None]:
        raise RuntimeError("401")

    monkeypatch.setattr(mb, "_sp_pe_history", _pe_rows)
    monkeypatch.setattr(mb, "_fund_multiple", boom)
    out = mb.lines(Q_VALUE)
    assert out is not None
    assert "QQQ's current multiple could not be read" in out[0]
    assert any("Yahoo's fund multiples could not be read" in line for line in out)


def test_valuation_trigger_is_strict() -> None:
    assert mb.lines("Is AAPL expensive versus its 10 year average P/E?") is None
    assert mb.lines("What is the Nasdaq 100 at today?") is None
    assert mb.lines("What is the S&P 500 P/E?") is None  # no history asked: other readers


def test_parse_multpl_reads_dates_values_and_estimate_markers() -> None:
    html = ("<table><tr><th>Date</th><th>Value</th></tr>"
            '<tr class="odd"><td>Oct 2, 2026</td><td>\n<abbr title="Estimate">†</abbr>\n'
            "26.34\n</td></tr>"
            '<tr class="even"><td>Jun 1, 2026</td><td>\n&#x2002;\n25.22\n</td></tr></table>')
    assert mb.parse_multpl(html) == [(date(2026, 10, 2), 26.34), (date(2026, 6, 1), 25.22)]


def test_sp_pe_history_fetches_parses_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = "".join(f"<tr><td>Jan {1 + i % 28}, {2000 + i // 28}</td><td>{10 + i % 9}.5</td></tr>"
                   for i in range(300))
    calls: list[str] = []

    def fake(url: str, **kw: Any) -> str:
        calls.append(url)
        return f"<table>{rows}</table>"

    monkeypatch.setattr(mb.http, "fetch_text", fake)
    mb._pe_cache.clear()
    first = mb._sp_pe_history()
    assert len(first) == 300 and mb._sp_pe_history() is first and len(calls) == 1
    assert calls[0] == mb.MULTPL
    mb._pe_cache.clear()
    monkeypatch.setattr(mb.http, "fetch_text", lambda url, **kw: "<table></table>")
    with pytest.raises(mb.BreadthError):
        mb._sp_pe_history()
    mb._pe_cache.clear()


# --------------------------------------------------------------------------- 6. drivers


def _returns_table(n: int = 110) -> dict[str, mb.Series]:
    """Silver moves 1.5x gold plus a small independent wiggle; the dollar and yield are noise."""
    dates = _days(n)
    gold_r = [0.01 * math.sin(i * 1.7) + 0.004 * math.cos(i * 0.3) for i in range(n)]
    noise = [0.0015 * math.sin(i * 2.9 + 1.0) for i in range(n)]
    dollar_r = [0.004 * math.cos(i * 4.1) + 0.003 * math.sin(i * 0.7) for i in range(n)]

    def path(rets: list[float], start: float) -> list[float]:
        out = [start]
        for r in rets[1:]:
            out.append(out[-1] * (1 + r))
        return out

    silver_r = [1.5 * g + e for g, e in zip(gold_r, noise, strict=True)]
    yields = [4.0]
    for i in range(1, n):
        yields.append(yields[-1] + 0.02 * math.sin(i * 3.3))
    return {
        "SI=F": mb.Series("SI=F", dates, path(silver_r, 30.0)),
        "GC=F": mb.Series("GC=F", dates, path(gold_r, 2000.0)),
        "DX-Y.NYB": mb.Series("DX-Y.NYB", dates, path(dollar_r, 100.0)),
        "^TNX": mb.Series("^TNX", dates, yields),
        "^GSPC": mb.Series("^GSPC", dates, path(dollar_r[::-1], 5000.0)),
    }


def test_drivers_names_the_stronger_driver_by_r_squared(monkeypatch: pytest.MonkeyPatch) -> None:
    table = _returns_table()
    _install(monkeypatch, table)
    out = mb.lines("Is gold or the dollar the stronger driver of silver over the past 3 months?")
    assert out is not None
    assert out[0].startswith("Bottom line: over the past 3 months")
    assert "gold was the stronger driver of silver" in out[0]
    # independent recomputation of gold's R-squared and beta over the same shared days
    window = mb.parse_window("past 3 months", TODAY)
    s, g = table["SI=F"], table["GC=F"]
    idx = [i for i, d in enumerate(s.dates) if d >= window.base]
    ys = [100 * (s.closes[i] / s.closes[i - 1] - 1) for i in idx[1:]]
    xs = [100 * (g.closes[i] / g.closes[i - 1] - 1) for i in idx[1:]]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    beta = (sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
            / sum((a - mx) ** 2 for a in xs))
    corr = (sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
            / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)))
    assert f"beta {beta:+.2f}" in out[0] and f"correlation {corr:+.2f}" in out[0]
    assert f"R-squared {corr * corr:.2f}" in out[0]
    assert "Each alone: gold: R-squared" in out[1]
    assert "the US dollar index (DXY): R-squared" in out[1]
    assert "Together they explain" in out[2] and "unique contribution" in out[2]
    assert "cannot be told from noise" in out[3]
    assert out[-1].startswith("Data: ") and out[-1].endswith("Not advice.")
    assert "daily change in percentage points" not in out[-1]


def test_drivers_quarter_too_short_falls_back_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _returns_table())
    out = mb.lines(Q_DRIVER)
    assert out is not None
    assert "The quarter to date has only 3 daily returns" in out[0]
    assert "so the past 3 months are used instead" in out[0]
    assert "gold was the stronger driver of silver" in out[0]


def test_drivers_yield_enters_as_a_change(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _returns_table())
    out = mb.lines("Is gold or the 10-year yield the bigger driver of silver this year?")
    assert out is not None
    assert "the 10-year Treasury yield: R-squared" in out[1] and "per +10bp" in out[1]
    assert "daily change in percentage points" in out[-1]


def test_drivers_trigger_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _returns_table())
    assert mb.lines("What drives bitcoin?") is None
    assert mb.lines("Does the dollar move gold?") is None  # one candidate, no comparison
    assert mb.lines("Gold or the dollar for my portfolio?") is None  # no driver verb or target
    assert mb.lines("Is silver or gold the better buy?") is None


def test_drivers_failure_is_stated(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {})
    out = mb.lines(Q_DRIVER)
    assert out is not None and "could not be read just now" in out[0]


def test_ols_helpers() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.1, 3.9, 6.2, 7.8, 10.0]
    fit = mb.fit_one(y, x)
    assert fit is not None and fit.beta == pytest.approx(1.97)
    assert fit.r2 == pytest.approx(fit.corr**2)
    assert mb.fit_one([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert mb._r2(y, [x]) == pytest.approx(fit.r2)
    # a second, uninformative column cannot lower the fit
    both = mb._r2(y, [x, [0.3, -0.1, 0.2, -0.4, 0.1]])
    assert both is not None and both >= fit.r2 - 1e-9
    # identical columns are singular: no answer rather than a wrong one
    assert mb._r2(y, [x, x]) is None


# --------------------------------------------------------------------------- entry point


def test_lines_returns_none_for_other_questions() -> None:
    for text in ("What is the price of BTC?", "Should I buy NVDA?",
                 "What is the weather like?", "", "Explain the Fed dot plot"):
        assert mb.lines(text) is None


def test_lines_accepts_prior_and_every_answer_is_well_formed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _sector_table(0.1, -0.08))
    out = mb.lines(Q_SECTOR, ["earlier question"])
    assert out is not None
    assert out[0].startswith("Bottom line: ") and out[-1].startswith("Data: ")
    assert out[-1].endswith("Not advice.")
    assert all(isinstance(x, str) and x for x in out)
