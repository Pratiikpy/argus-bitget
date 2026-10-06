"""Portfolio research answers (`lui/research/portfolio_lab.py`): the eight round-45 questions that
were answered as something else, run on synthetic series with known structure, offline."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from argus.lui.research import portfolio_lab as pl
from argus.market import equity_history, history

TODAY = date(2026, 10, 5)  # a Monday


def _business_days(count: int, end: date = TODAY) -> list[date]:
    out: list[date] = []
    d = end
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _closes(returns: Iterable[float], start: float = 100.0) -> list[float]:
    out = [start]
    for r in returns:
        out.append(out[-1] * (1.0 + r))
    return out


def _normal(seed: int, n: int, vol: float) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, vol, n)


def _install(monkeypatch: pytest.MonkeyPatch,
             table: dict[str, tuple[list[date], list[float]]]) -> None:
    """Yahoo daily closes from ``table``; a ticker that is not in it fails to arrive."""

    def fake(ticker: str, **_: Any) -> list[equity_history.Day]:
        if ticker not in table:
            raise equity_history.HistoryError(f"no fixture for {ticker}")
        days, closes = table[ticker]
        return [equity_history.Day(d, c, c) for d, c in zip(days, closes, strict=True)]

    monkeypatch.setattr(equity_history, "daily", fake)
    monkeypatch.setattr(pl, "_today", lambda: TODAY)


def _text(out: list[str] | None) -> str:
    assert out is not None
    return "\n".join(out)


# --------------------------------------------------------------------------- 1. lead-lag

THESIS = "My thesis: semiconductor stocks lead Bitcoin by a few days. Test it against data."
LAG_Q = "What lag maximizes the correlation, and is it statistically significant?"


def _lead_tables(lead: int | None) -> dict[str, tuple[list[date], list[float]]]:
    days = _business_days(560)
    n = len(days) - 1
    x = _normal(1, n, 0.015)
    noise = _normal(2, n, 0.012)
    if lead is None:
        y = _normal(3, n, 0.02)
    else:
        y = noise.copy()
        y[lead:] += 0.8 * x[:n - lead]
    return {"SMH": (days, _closes(x)), "BTC-USD": (days, _closes(y))}


def test_planted_lead_is_found_and_significant(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _lead_tables(2))
    text = _text(pl.lines(THESIS))
    first = text.splitlines()[0]
    assert first.startswith("Bottom line: yes")
    assert "led Bitcoin by 2 trading days" in first
    assert "p<0.001" in first or "p=0.00" in first


def test_no_lead_says_no_with_the_chance_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _lead_tables(None))
    out = _text(pl.lines(THESIS))
    first = out.splitlines()[0]
    assert first.startswith("Bottom line: no")
    assert "does not show semiconductor stocks (SMH) leading Bitcoin" in first
    assert "five lags tried" in first
    assert "Data: Yahoo Finance daily closes (SMH and BTC-USD)" in out


def test_lag_table_matches_an_independent_correlation(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _lead_tables(3)
    _install(monkeypatch, tables)
    text = _text(pl.lines(THESIS))
    a = np.diff(tables["SMH"][1]) / np.array(tables["SMH"][1][:-1])
    b = np.diff(tables["BTC-USD"][1]) / np.array(tables["BTC-USD"][1][:-1])
    # the module keeps the last two years of the 560 days; recompute on that same tail
    days = tables["SMH"][0]
    keep = [i for i, d in enumerate(days) if d >= TODAY - timedelta(days=730)]
    a, b = a[keep[0]:], b[keep[0]:]
    expected = float(np.corrcoef(a[:len(a) - 3], b[3:])[0, 1])
    assert f"+3: {expected:+.2f}" in text


def test_follow_up_reads_the_thesis_from_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _lead_tables(2))
    out = pl.lines(LAG_Q, [THESIS])
    text = _text(out)
    assert out is not None
    assert text.startswith("Bottom line: the correlation is strongest at")
    assert "2 trading days with semiconductor stocks (SMH) first" in text.splitlines()[0]
    assert "significant" in text.splitlines()[0]
    assert "the best of lags 1 to 5 is 2 days" in text


def test_follow_up_with_the_same_day_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    days = _business_days(560)
    x = _normal(4, len(days) - 1, 0.015)
    y = 0.7 * x + _normal(5, len(days) - 1, 0.01)
    _install(monkeypatch, {"SMH": (days, _closes(x)), "BTC-USD": (days, _closes(y))})
    first = _text(pl.lines(LAG_Q, [THESIS])).splitlines()[0]
    assert "the same day (lag 0)" in first


def test_follow_up_without_a_thesis_is_not_claimed() -> None:
    assert pl.lines(LAG_Q, []) is None
    assert pl.lines(LAG_Q, ["What is Bitcoin trading at?", "How is NVDA doing?"]) is None


def test_thesis_directions_and_non_theses() -> None:
    th = pl._thesis("Does gold lead bitcoin?")
    assert th is not None and (th.leader, th.follower) == ("GC=F", "BTC-USD")
    th = pl._thesis("I think Bitcoin leads the Nasdaq")
    assert th is not None and (th.leader, th.follower) == ("BTC-USD", "^NDX")
    assert pl._thesis("What leads Bitcoin higher?") is None
    assert pl._thesis("Bitcoin leads Bitcoin") is None
    assert pl.lines("What does a leading indicator mean?") is None


def test_data_failure_is_one_line_not_an_invented_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"SMH": _lead_tables(2)["SMH"]})
    out = pl.lines(THESIS)
    assert out is not None and len(out) == 1
    assert out[0].startswith("Bottom line: not computed") and "BTC-USD" in out[0]


# --------------------------------------------------------------------------- 2. buy schedule

DCA_Q = ("What would have happened if I bought ETH every Monday for the last year with $200? "
         "Compare with buying it all at the start.")
SWAP_Q = "What if I did it on Bitcoin instead?"
DD_Q = "Which had the worse drawdown during the period?"


def _install_candles(monkeypatch: pytest.MonkeyPatch,
                     paths: dict[str, Callable[[int], float]], span: int = 420) -> None:
    def fake(symbol: str, *, days: int = 90, interval: str = "1H", **_: Any) -> list[Any]:
        assert interval == "1Dutc"
        if symbol not in paths:
            raise history.HistoryError("no candles")
        out = []
        for i in range(span):
            day = datetime(TODAY.year, TODAY.month, TODAY.day, tzinfo=UTC) - timedelta(
                days=span - 1 - i)
            if (TODAY - day.date()).days > days:
                continue
            o, c = paths[symbol](i), paths[symbol](i + 1)
            out.append(history.Candle(ts=day, open=Decimal(str(o)), high=Decimal(str(max(o, c))),
                                      low=Decimal(str(min(o, c))), close=Decimal(str(c)),
                                      volume=Decimal(1)))
        return out

    monkeypatch.setattr(history, "fetch_range", fake)
    monkeypatch.setattr("argus.lui.research.parse.last_price", lambda s: None)
    monkeypatch.setattr(pl, "_today", lambda: TODAY)


def _eth_up(i: int) -> float:
    return 100.0 + i


def _eth_crash(i: int) -> float:
    return 100.0 if i < 250 else 50.0 if i > 300 else 100.0 - (i - 250)  # halves, then flat


def _btc_dip(i: int) -> float:
    return 100.0 if i < 250 else 80.0 if i > 300 else 100.0 - 0.4 * (i - 250)


def test_schedule_numbers_are_the_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_up})
    out = _text(pl.lines(DCA_Q))
    mondays = [TODAY - timedelta(days=k) for k in range(365 + 1)
               if (TODAY - timedelta(days=k)).weekday() == 0]
    mondays = [m for m in mondays if m >= TODAY - timedelta(days=365)]
    assert len(mondays) == 53
    coins = sum(200 / _eth_up(419 - (TODAY - m).days) for m in mondays)
    invested = 200 * 53
    value = coins * _eth_up(420)
    first = out.splitlines()[0]
    assert "53 buys, $10,600 in" in first
    assert f"worth ${value:,.0f}" in first
    lump_coins = invested / _eth_up(419 - (TODAY - min(mondays)).days)
    assert f"worth ${lump_coins * _eth_up(420):,.0f}" in first
    assert "every Monday over the last year" in first
    assert "the schedule finishes" in first and "behind the lump sum" in first
    assert "Data: Bitget daily ETHUSDT" in out


def test_drawdown_is_flow_adjusted_and_nothing_in_scientific_notation(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_crash})
    out = _text(pl.lines(DCA_Q))
    assert re.search(r"lump sum -(?:4|5)\d\.\d%", out)
    assert not re.search(r"\d[eE][+-]\d", out)
    sched = float(re.search(r"schedule (-\d+\.\d)%", out).group(1))  # type: ignore[union-attr]
    lump = float(re.search(r"lump sum (-\d+\.\d)%", out).group(1))  # type: ignore[union-attr]
    assert lump < sched < 0   # money added late into the fall cannot hide it, but it dilutes it


def test_swap_follow_up_reuses_the_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_up, "BTCUSDT": _btc_dip})
    out = _text(pl.lines(SWAP_Q, [DCA_Q]))
    first = out.splitlines()[0]
    assert "$200 of BTC every Monday over the last year (53 buys, $10,600 in)" in first


def test_drawdown_follow_up_compares_the_two_coins(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_crash, "BTCUSDT": _btc_dip})
    out = _text(pl.lines(DD_Q, [DCA_Q, SWAP_Q]))
    first = out.splitlines()[0]
    assert first.startswith("Bottom line: ETH had the worse drawdown")
    assert "ETH -" in first and "BTC -" in first
    assert "lump sum" in first.lower() or "As a lump sum" in first


def test_drawdown_follow_up_with_one_coin_compares_to_the_lump_sum(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_crash})
    first = _text(pl.lines(DD_Q, [DCA_Q])).splitlines()[0]
    assert "ETH's schedule had a worst drawdown" in first and "for the lump sum" in first


def test_other_cadences_and_periods(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_up})
    daily = _text(pl.lines("What if I bought ETH every day for the last 3 months with $10?"))
    assert "every day over the last 3 months" in daily.splitlines()[0]
    assert re.search(r"\((9[0-2]) buys", daily.splitlines()[0])
    monthly = _text(pl.lines("If I had bought ETH monthly with $500 over the past 6 months, "
                             "what would it be worth?"))
    assert "on the first of every month over the last 6 months (6 buys" in monthly
    weekly = _text(pl.lines("What if I invested $50 in ETH weekly for the last year?"))
    assert "every week (Mondays)" in weekly.splitlines()[0]


def test_period_given_as_a_length_after_for_or_as_a_start_date(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_up})
    six = _text(pl.lines("What if I put $50 into ETH every week for 6 months?"))
    assert "over the last 6 months" in six.splitlines()[0]
    assert "Period not stated" not in six
    assert "$50 of ETH" in six.splitlines()[0]
    since = _text(pl.lines("If I had invested $100 in ETH on the 1st of every month since "
                           "February 2026, what would I have?"))
    assert "over the period since 1 Feb 2026 (9 buys, $900 in)" in since.splitlines()[0]
    assert "first purchase was Sunday 01 Feb 2026" in since


def test_schedule_drawdown_is_dated_by_its_low(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_crash})
    out = _text(pl.lines(DCA_Q))
    assert re.search(r"schedule -\d+\.\d% \(low on \d\d \w{3} 20\d\d\), lump sum -\d+\.\d% "
                     r"\(\d\d \w{3} 20\d\d peak to \d\d \w{3} 20\d\d\)", out)


def test_missing_amount_is_asked_for_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_up})
    out = pl.lines("What would have happened if I bought ETH every Monday for the last year?")
    assert out is not None and out[0].startswith("Bottom line: not computed")
    assert "say how much each purchase is" in out[0]


def test_dca_not_claimed_for_unrelated_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {"ETHUSDT": _eth_up})
    assert pl.lines("What if ETH drops 20% tomorrow?", [DCA_Q]) is None
    assert pl.lines("What is the price of ETH?", [DCA_Q]) is None
    assert pl.lines(SWAP_Q, []) is None
    assert pl.lines(DD_Q, []) is None


def test_candle_failure_is_a_plain_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candles(monkeypatch, {})
    out = pl.lines(DCA_Q)
    assert out is not None and out[0].startswith("Bottom line: not computed")


# --------------------------------------------------------------------------- 3. rebalance cap

BOOK = "I hold 40% NVDA, 30% AAPL, 30% BTC."
CAP_Q = "Suggest a rebalance that caps any single position at 25%."
CAP_DD_Q = "What would that rebalance have done to my worst drawdown?"


def _book_tables(nvda_crash: bool = True) -> dict[str, tuple[list[date], list[float]]]:
    days = _business_days(800)
    n = len(days) - 1
    nvda = _normal(11, n, 0.02)
    if nvda_crash:
        nvda[300:330] -= 0.02
    return {"NVDA": (days, _closes(nvda)), "AAPL": (days, _closes(_normal(12, n, 0.01))),
            "BTC-USD": (days, _closes(_normal(13, n, 0.015)))}


def test_cap_weights_unit_cases() -> None:
    w, cash = pl.cap_weights({"A": 0.4, "B": 0.3, "C": 0.3}, 0.25)
    assert w == pytest.approx({"A": 0.25, "B": 0.25, "C": 0.25}) and cash == pytest.approx(0.25)
    # 20 points come off A; B can take 10 (to the cap), C the other 10 less what B could not
    w, cash = pl.cap_weights({"A": 0.5, "B": 0.2, "C": 0.1}, 0.3)
    assert w == pytest.approx({"A": 0.3, "B": 0.3, "C": 0.2}) and cash == pytest.approx(0.2)
    # the excess is absorbed in proportion to weight, and what the others cannot take is cash
    w, cash = pl.cap_weights({"A": 0.6, "B": 0.3, "C": 0.1}, 0.35)
    assert w == pytest.approx({"A": 0.35, "B": 0.35, "C": 0.3}) and cash == pytest.approx(0.0)
    w, cash = pl.cap_weights({"A": 0.8, "B": 0.1, "C": 0.1}, 0.3)
    assert w == pytest.approx({"A": 0.3, "B": 0.3, "C": 0.3}) and cash == pytest.approx(0.1)
    # nothing above the cap: unchanged, and a cash balance the book already held is kept
    w, cash = pl.cap_weights({"A": 0.3, "B": 0.3}, 0.4)
    assert w == pytest.approx({"A": 0.3, "B": 0.3}) and cash == pytest.approx(0.4)
    for book in ({"A": 0.7, "B": 0.2, "C": 0.05}, {"A": 0.34, "B": 0.33, "C": 0.33}):
        w, cash = pl.cap_weights(book, 0.4)
        assert max(w.values()) <= 0.4 + 1e-9
        assert sum(w.values()) + cash == pytest.approx(1.0)


def test_proposal_says_where_the_excess_goes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _book_tables())
    out = pl.lines(CAP_Q, [BOOK])
    text = _text(out)
    assert out is not None
    assert out[0] == ("Bottom line: with no position above 25%, 40% NVDA, 30% AAPL, 30% BTC "
                      "becomes 25% NVDA, 25% AAPL, 25% BTC, 25% cash.")
    assert "NVDA 40% to 25%" in text and "all of it goes to cash" in text
    assert "no name is under the cap to take it" in text
    assert "Worst drawdown over" in text


def test_excess_goes_to_names_under_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _book_tables())
    out = _text(pl.lines("Cap any single position at 30%.", ["I hold 50% NVDA, 25% AAPL, "
                                                             "25% BTC"]))
    first = out.splitlines()[0]
    assert "50% NVDA, 25% AAPL, 25% BTC becomes 30% NVDA, 30% AAPL, 30% BTC, 10% cash" in first
    assert "goes to the names under the cap, up to the cap, and the other 10% to cash" in out


def test_drawdown_follow_up_compares_original_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _book_tables()
    _install(monkeypatch, tables)
    out = pl.lines(CAP_DD_Q, [BOOK, CAP_Q])
    text = _text(out)
    assert out is not None
    first = out[0]
    assert first.startswith("Bottom line: the 25% cap (25% NVDA, 25% AAPL, 25% BTC, 25% cash) "
                            "would have cut the worst drawdown from")
    # recompute both drawdowns independently: daily-rebalanced weights over the last three years
    days = tables["NVDA"][0]
    keep = [i for i, d in enumerate(days) if d >= TODAY - timedelta(days=3 * 365)]
    rets = {k: np.diff(v[1]) / np.array(v[1][:-1]) for k, v in tables.items()}
    for weights, label in (((0.4, 0.3, 0.3), "original"), ((0.25, 0.25, 0.25), "capped")):
        r = sum(w * rets[k][keep[0]:] for w, k in zip(weights, ("NVDA", "AAPL", "BTC-USD"),
                                                       strict=True))
        curve = np.concatenate([[1.0], np.cumprod(1 + r)])
        dd = float((curve / np.maximum.accumulate(curve) - 1).min())
        assert f"{label} {dd:.1%}" in text
    assert text.count("Worst drawdown over") == 1


def test_follow_up_text_is_not_the_previous_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _book_tables())
    first = pl.lines(CAP_Q, [BOOK])
    second = pl.lines(CAP_DD_Q, [BOOK, CAP_Q])
    assert first is not None and second is not None and first != second
    assert "worst drawdown from" in second[0]


def test_cap_without_a_book_or_a_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _book_tables())
    assert pl.lines(CAP_Q, []) is None
    out = pl.lines("Cap any single position at 50%.", [BOOK])
    assert out is not None and out[0].startswith("Bottom line: no change")
    assert pl.lines("Cap the leverage at 25x", [BOOK]) is None
    assert pl.lines("What is my worst drawdown?", [BOOK, CAP_Q]) is None


# --------------------------------------------------------------------------- 4. concentration

CONC_Q = (BOOK + " What is my biggest concentration risk and how correlated are these over the "
          "past year?")


def test_concentration_names_the_risk_and_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _book_tables(nvda_crash=False)
    _install(monkeypatch, tables)
    out = _text(pl.lines(CONC_Q))
    first = out.splitlines()[0]
    assert first.startswith("Bottom line: your biggest concentration risk is NVDA")
    assert "40% of the money" in first
    assert "High means 0.60 or more, moderate 0.30 to 0.60, low below 0.30." in out
    shares = [int(m) for m in re.findall(r"of the money, (\d+)% of the risk", out)]
    assert abs(sum(shares) - 100) <= 2
    assert shares[0] == max(shares)


def test_risk_shares_sum_to_one_and_follow_volatility() -> None:
    shares = pl.risk_shares({"A": 0.5, "B": 0.5}, [[0.04, 0.0], [0.0, 0.01]])
    assert shares == pytest.approx([0.8, 0.2])
    # fully correlated: each holding's share is weight x vol over the book's volatility
    cov = [[0.04, 0.02], [0.02, 0.01]]
    both = pl.risk_shares({"A": 0.5, "B": 0.5}, cov)
    vol = 0.5 * 0.2 + 0.5 * 0.1
    assert both == pytest.approx([0.5 * 0.2 / vol, 0.5 * 0.1 / vol])


def test_concentration_uses_the_thread_book(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _book_tables(nvda_crash=False))
    out = pl.lines("Where is most of my risk, and how concentrated is it?", [BOOK])
    assert out is not None and "your biggest concentration risk" in out[0]
    assert pl.lines("What is my biggest concentration risk?", []) is None
    assert pl.lines("Is Bitcoin a good diversifier?", [BOOK]) is None


def test_weights_over_one_hundred_percent_are_flagged_not_silently_rescaled(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _book_tables(nvda_crash=False))
    out = pl.lines("My book is 70% NVDA, 60% AAPL, 40% BTC. What is the concentration?")
    assert out is not None
    assert out[1] == ("Note: your weights add to 170%, which means leverage or an error; they are "
                      "shown as shares of that total, rescaled to 100%.")
    rebal = pl.lines("Cap any single position at 40%.", ["My book is 70% NVDA, 60% AAPL, 40% BTC"])
    assert rebal is not None and rebal[1].startswith("Note: your weights add to 170%")
    plain = pl.lines(CONC_Q)
    assert plain is not None and not any(line.startswith("Note:") for line in plain)


# --------------------------------------------------------------------------- 5. vol-adjusted

VOL_Q = "Did Bitcoin outperform gold over the past 12 months, adjusted for volatility?"


def _vol_tables(btc_drift: float) -> dict[str, tuple[list[date], list[float]]]:
    days = _business_days(500)
    n = len(days) - 1
    btc = _normal(21, n, 0.03) + btc_drift
    gold = _normal(22, n, 0.008) + 0.0004
    return {"BTC-USD": (days, _closes(btc)), "GC=F": (days, _closes(gold)),
            "^IRX": (days, [4.0] * len(days))}


def test_vol_adjusted_verdict_and_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _vol_tables(-0.004)
    _install(monkeypatch, tables)
    monkeypatch.setattr("argus.market.skill_mirror.fred_series",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    out = _text(pl.lines(VOL_Q))
    first = out.splitlines()[0]
    assert first.startswith("Bottom line: no") and "did not outperform gold" in first
    start = TODAY - timedelta(days=365)
    days, btc = tables["BTC-USD"]
    idx = [i for i, d in enumerate(days) if d >= start]
    series = np.array(btc)[idx[0]:]
    assert f"return {series[-1] / series[0] - 1:+.1%}" in out
    vol = float((np.diff(series) / series[:-1]).std(ddof=1)) * math.sqrt(252)
    assert f"volatility {vol:.1%} a year" in out
    assert "averaging 4.00%" in out
    assert "Treasury bill" in out


def test_vol_adjusted_can_flip_the_raw_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    days = _business_days(500)
    n = len(days) - 1
    btc = _normal(31, n, 0.04) + 0.0012      # higher return, far higher volatility
    gold = _normal(32, n, 0.004) + 0.0004
    _install(monkeypatch, {"BTC-USD": (days, _closes(btc)), "GC=F": (days, _closes(gold))})
    monkeypatch.setattr("argus.market.skill_mirror.fred_series", lambda *a, **k: [(TODAY, 0.0)])
    first = _text(pl.lines(VOL_Q)).splitlines()[0]
    assert first.startswith("Bottom line:")
    assert "once adjusted for volatility" in first


def test_index_and_etf_names_are_one_asset_not_two(monkeypatch: pytest.MonkeyPatch) -> None:
    days = _business_days(500)
    n = len(days) - 1
    _install(monkeypatch, {"BTC-USD": (days, _closes(_normal(51, n, 0.03))),
                           "^GSPC": (days, _closes(_normal(52, n, 0.01) + 0.0004)),
                           "^IRX": (days, [4.0] * len(days))})
    monkeypatch.setattr("argus.market.skill_mirror.fred_series",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    for question in ("Did Bitcoin beat the S&P 500 over the past 12 months on a risk-adjusted "
                     "basis?", "Did Bitcoin beat SPY over the past 12 months, adjusted for "
                     "volatility?"):
        out = pl.lines(question)
        assert out is not None and out[0].startswith("Bottom line:")
        assert "the S&P 500" in out[0]


def test_vol_adjusted_not_claimed_for_other_questions() -> None:
    assert pl.lines("Did Bitcoin outperform gold over the past 12 months?") is None
    assert pl.lines("What is the Sharpe of my book? 50% BTC, 50% gold, adjusted for risk") is None
    assert pl.lines("Is Bitcoin risk-adjusted better than anything?") is None


# --------------------------------------------------------------------------- 6. breadth

BREADTH_Q = "What percentage of S&P 500 stocks are above their 50-day average right now?"


@pytest.fixture(autouse=True)
def _fresh_breadth_cache() -> None:
    pl._breadth_cache.clear()


def _members(n: int = 500) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _install_breadth(monkeypatch: pytest.MonkeyPatch, members: list[str],
                     up: Callable[[int], bool], fail_list: bool = False, bars: int = 250) -> None:
    csv_body = "Symbol,Security\n" + "\n".join(f"{m},{m} Inc" for m in members)

    def fetch_text(url: str, **_: Any) -> str:
        if fail_list:
            raise OSError("down")
        return csv_body

    def fetch_json(url: str, **_: Any) -> Any:
        symbols = url.split("symbols=")[1].split("&")[0].split(",")
        assert len(symbols) <= 20
        out: dict[str, Any] = {}
        for s in symbols:
            i = int(s[1:])
            base = list(np.linspace(100, 150, bars)) if up(i) else list(np.linspace(150, 100, bars))
            stamps = [int(datetime(2026, 1, 1, tzinfo=UTC).timestamp()) + k * 86400
                      for k in range(bars + 1)]
            out[s] = {"timestamp": stamps, "close": [*base, None]}
        return out

    from argus.truth import http
    monkeypatch.setattr(http, "fetch_text", fetch_text)
    monkeypatch.setattr(http, "fetch_json", fetch_json)


def test_breadth_counts_members_above_their_average(monkeypatch: pytest.MonkeyPatch) -> None:
    members = _members(500)
    _install_breadth(monkeypatch, members, lambda i: i < 130)
    out = _text(pl.lines(BREADTH_Q))
    first = out.splitlines()[0]
    assert first.startswith("Bottom line: 26.0% of S&P 500 stocks (130 of 500) are above their "
                            "50-day average")
    assert "fewer than half" in first
    assert "26.0% (130 of 500) are above their 200-day average" in out
    assert "Today's members only" in out


def test_breadth_below_and_200_day_phrasing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_breadth(monkeypatch, _members(500), lambda i: i < 300)
    out = _text(pl.lines("What share of S&P 500 stocks are below their 200-day average?"))
    assert out.splitlines()[0].startswith("Bottom line: 40.0% of S&P 500 stocks (200 of 500) "
                                          "are below their 200-day average")


def test_breadth_with_short_history_skips_the_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_breadth(monkeypatch, _members(500), lambda i: i < 250, bars=120)
    out = _text(pl.lines(BREADTH_Q))
    assert out.splitlines()[0].startswith("Bottom line: 50.0% of S&P 500 stocks (250 of 500)")
    assert "200-day" not in out


def test_breadth_is_fetched_once_for_two_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_breadth(monkeypatch, _members(500), lambda i: i < 130)
    calls = {"n": 0}
    from argus.truth import http
    real = http.fetch_json

    def counting(url: str, **kw: Any) -> Any:
        calls["n"] += 1
        return real(url, **kw)

    monkeypatch.setattr(http, "fetch_json", counting)
    assert pl.lines(BREADTH_Q) is not None
    first = calls["n"]
    assert first == 25                       # 500 members, 20 a request
    assert pl.lines(BREADTH_Q.replace("50", "200")) is not None
    assert calls["n"] == first


def test_breadth_counter_unit() -> None:
    closes = {"A": [1.0] * 49 + [2.0], "B": [2.0] * 49 + [1.0], "C": [1.0, 2.0]}
    assert pl.breadth_counts(closes, 50) == (1, 2)


def test_breadth_falls_back_to_a_labelled_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_breadth(monkeypatch, _members(500), lambda i: True, fail_list=True)
    days = _business_days(300)
    _install(monkeypatch, {"SPY": (days, list(np.linspace(100, 130, 300))),
                           "RSP": (days, list(np.linspace(100, 90, 300)))})
    out = _text(pl.lines(BREADTH_Q))
    first = out.splitlines()[0]
    assert "could not be computed here" in first and "As a proxy only" in first
    assert "equal-weighted one (RSP) -" in first
    assert "It is not the percentage asked for" in out


def test_breadth_not_claimed_for_other_questions() -> None:
    assert pl.lines("Is the S&P 500 above its 50-day average?") is None
    assert pl.lines("What percentage of my book is NVDA?") is None


# --------------------------------------------------------------------------- 7. rolling correlation

ROLL_Q = "How does the DXY dollar index relate to BTC this year - rolling 60 day correlation?"


def _roll_tables() -> dict[str, tuple[list[date], list[float]]]:
    days = _business_days(400)
    n = len(days) - 1
    dxy = _normal(41, n, 0.004)
    btc = -2.0 * dxy + _normal(42, n, 0.01)
    return {"BTC-USD": (days, _closes(btc)), "DX-Y.NYB": (days, _closes(dxy, 100.0))}


def test_rolling_correlation_matches_an_independent_calculation(
        monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _roll_tables()
    _install(monkeypatch, tables)
    out = _text(pl.lines(ROLL_Q))
    days = tables["BTC-USD"][0]
    b = np.diff(tables["BTC-USD"][1]) / np.array(tables["BTC-USD"][1][:-1])
    d = np.diff(tables["DX-Y.NYB"][1]) / np.array(tables["DX-Y.NYB"][1][:-1])
    vals = {days[i]: float(np.corrcoef(b[i - 60:i], d[i - 60:i])[0, 1])
            for i in range(60, len(b) + 1)}
    ytd = {k: v for k, v in vals.items() if k >= date(TODAY.year, 1, 1)}
    latest = ytd[max(ytd)]
    lo, hi = min(ytd.values()), max(ytd.values())
    first = out.splitlines()[0]
    assert f"is {latest:+.2f} now" in first
    assert f"ranged from {lo:+.2f}" in first and f"to {hi:+.2f}" in first
    assert "opposite directions" in first
    assert "ICE US Dollar Index" in first and "so far this year" in first


def test_rolling_window_is_read_from_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _roll_tables())
    first = _text(pl.lines("What is the 30-day rolling correlation between Bitcoin and the "
                           "dollar over the past year?")).splitlines()[0]
    assert "rolling 30-day correlation" in first and "over the past 12 months" in first


def test_rolling_falls_back_to_fred_and_says_it_is_not_ice_dxy(
        monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _roll_tables()
    _install(monkeypatch, {"BTC-USD": tables["BTC-USD"]})
    days, closes = tables["DX-Y.NYB"]
    monkeypatch.setattr("argus.market.skill_mirror.fred_series",
                        lambda *a, **k: list(zip(days, closes, strict=True)))
    out = _text(pl.lines(ROLL_Q))
    assert "FRED DTWEXBGS, not ICE DXY" in out.splitlines()[0]
    assert "different basket" in out


def test_rolling_not_claimed_for_other_questions() -> None:
    assert pl.lines("What is the dollar index at?") is None
    assert pl.lines("How correlated is BTC with the Nasdaq?") is None
    assert pl.lines("What does dollar-cost averaging mean for BTC?") is None


# --------------------------------------------------------------------------- 8. funding

FUND_Q = ("Which Bitget perpetual has the most extreme funding right now, long or short side, "
          "and is it a squeeze setup?")


def _fund_feed(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, str]],
               meta: dict[str, dict[str, str]] | None = None) -> None:
    contracts = [{"symbol": r["symbol"], "fundInterval": (meta or {}).get(
        r["symbol"], {}).get("fundInterval", "8"), "symbolStatus": "normal"} for r in rows]

    def fake(path: str, params: dict[str, str] | None = None, **_: Any) -> Any:
        if path.endswith("/tickers"):
            return rows
        if path.endswith("/contracts"):
            return contracts
        if path.endswith("/current-fund-rate"):
            return [{"fundingRate": "-0.0075", "minFundingRate": "-0.02",
                     "maxFundingRate": "0.02"}]
        raise AssertionError(path)

    import argus.market.bitget as bitget
    monkeypatch.setattr(bitget, "public_get", fake)


def _row(symbol: str, funding: float, volume: float = 20e6, change: float = 0.01,
         holding: float = 1_000.0, last: float = 10.0) -> dict[str, str]:
    return {"symbol": symbol, "fundingRate": str(funding), "usdtVolume": str(volume),
            "lastPr": str(last), "change24h": str(change), "holdingAmount": str(holding)}


def _book_rows() -> list[dict[str, str]]:
    rows = [_row(f"X{i}USDT", 0.0001 * (i % 5)) for i in range(20)]
    rows += [_row("CARVUSDT", -0.00754, volume=9e6, change=0.06),
             _row("龙虾USDT", 0.00055),
             _row("THINUSDT", -0.5, volume=100.0),     # most negative of all, but illiquid
             _row("SANDUSDT", -0.00155)]
    return rows


def test_funding_names_both_sides_and_ranks_by_daily_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    _fund_feed(monkeypatch, _book_rows(), {"CARVUSDT": {"fundInterval": "4"},
                                           "龙虾USDT": {"fundInterval": "4"}})
    out = _text(pl.lines(FUND_Q))
    first = out.splitlines()[0]
    assert first.startswith("Bottom line: CARVUSDT has the most extreme funding")
    assert "-0.7540% every 4h (-4.52% a day)" in first and "short side" in first
    assert "On the other side the most extreme is 龙虾USDT" in first
    assert "THINUSDT" not in out
    assert "Squeeze reading: CARVUSDT is a crowded-short setup" in out
    assert "price has moved against them, +6.0% in 24 hours" in out
    assert "funding is at 38% of the contract's cap" in out
    assert "This is not a prediction." in out
    assert "at least $5,000,000 traded in 24 hours" in out


def test_funding_long_side_when_it_is_the_extreme(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(f"X{i}USDT", -0.0001) for i in range(15)] + [_row("HOTUSDT", 0.004,
                                                                   change=-0.03)]
    _fund_feed(monkeypatch, rows)
    out = _text(pl.lines("Which perp has the most extreme funding today, long or short side?"))
    assert out.splitlines()[0].startswith("Bottom line: HOTUSDT has the most extreme")
    assert "long side" in out.splitlines()[0]
    assert "crowded-long setup" in out and "price has moved against them, -3.0% in 24 hours" in out


def test_squeeze_follow_up_uses_the_same_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    _fund_feed(monkeypatch, _book_rows(), {"CARVUSDT": {"fundInterval": "4"}})
    out = pl.lines("Is that a short squeeze setup?", [FUND_Q])
    assert out is not None
    assert out[0].startswith("Bottom line: on the short side (negative funding, shorts pay), "
                             "CARVUSDT — yes, it reads as a crowded-short setup")
    assert pl.lines("Is that a short squeeze setup?", ["Hello"]) is None
    assert pl.lines("Is that a short squeeze setup?", []) is None


def test_plain_highest_funding_is_left_to_the_existing_answer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _fund_feed(monkeypatch, _book_rows())
    assert pl.lines("Which Bitget perp has the highest funding?") is None
    assert pl.lines("What is the lowest funding rate right now?") is None
    assert pl.lines("What is funding?") is None


def test_funding_feed_failure_is_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    import argus.market.bitget as bitget

    def boom(*a: Any, **k: Any) -> Any:
        raise OSError("down")

    monkeypatch.setattr(bitget, "public_get", boom)
    out = pl.lines(FUND_Q)
    assert out is not None and len(out) == 1 and out[0].startswith("Bottom line: not computed")


# --------------------------------------------------------------------------- general

@pytest.mark.parametrize("question", [
    "What is the price of Bitcoin?",
    "How is NVDA doing today?",
    "Explain what a perpetual future is",
    "Should I buy ETH?",
    "Compare Coinbase and Robinhood on revenue growth",
    "What is the max leverage on BTC?",
    "",
])
def test_unrelated_questions_return_none(question: str) -> None:
    assert pl.lines(question, []) is None
    assert pl.lines(question, [THESIS, DCA_Q, CAP_Q]) is None


def test_source_files_have_no_backspace_bytes_and_use_lf() -> None:
    for name in ("src/argus/lui/research/portfolio_lab.py", "tests/test_portfolio_lab.py"):
        raw = (Path(__file__).resolve().parents[1] / name).read_bytes()
        assert raw.count(bytes([8])) == 0
        assert b"\r" not in raw
