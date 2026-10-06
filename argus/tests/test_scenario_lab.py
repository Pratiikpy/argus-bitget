"""Round 44's judge findings answered by `lui/research/scenario_lab.py`, offline: every network
read is replaced by a small deterministic series or record, and each figure is recomputed in the
test by a different route than the reader's own."""

from __future__ import annotations

import math
import random
import statistics
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from argus.lui.research import scenario_lab as lab
from argus.lui.research import shareholder_metrics as sm
from argus.market.earnings_release import Release


def _bdays(start: date, count: int) -> list[date]:
    out: list[date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _joined(lines: list[str]) -> str:
    return "\n".join(lines)


# --------------------------------------------------------------------------- 4. Kelly payoff


@pytest.mark.parametrize("text, expected", [
    ("My edge: 55% win rate, average win 1.5R, average loss 1R.", 1.5),
    ("55% win rate, my average winner is 1.5 times my average loser", 1.5),
    ("average loss 1R and average win 2R", 2.0),
    ("avg win $150, avg loss $100", 1.5),
    ("payoff 1.5:1", 1.5),
    ("a 2:1 payoff", 2.0),
    ("win/loss ratio of 1.5", 1.5),
    ("payoff ratio 1.5", 1.5),
    ("1.5R payoff", 1.5),
    ("reward to risk ratio of 2", 2.0),
    ("55% win rate and a 1:1 payoff", 1.0),
    ("even money", 1.0),
    ("55% win rate", None),
    ("average win 1.5R, average loss 0", None),
])
def test_kelly_payoff_forms(text: str, expected: float | None) -> None:
    assert lab.kelly_payoff(text) == expected


def test_kelly_console_uses_the_stated_payoff() -> None:
    """The judge's input gave a 10.0% fraction and said no payoff was given; 55% at 1.5R is
    0.55 - 0.45 / 1.5 = 25.0% full, 12.5% half, 6.25% quarter."""
    from argus.lui import server

    said = server._kelly_lines(
        "My edge: 55% win rate, average win 1.5R, average loss 1R. What is the Kelly fraction "
        "and what should I actually bet?", [])
    assert said is not None
    text = _joined(said)
    assert "full Kelly is 25.0%" in text and "1.5:1 payoff" in text
    assert "no payoff was given" not in text
    assert "half Kelly 12.5%" in text and "quarter Kelly 6." in text
    assert "What to actually bet is half Kelly or less: 12.5% at half" in text
    again = server._kelly_lines(
        "55% win rate, my average winner is 1.5 times my average loser; Kelly?", [])
    assert again is not None and "full Kelly is 25.0%" in again[0]
    unstated = server._kelly_lines("Kelly for a 55% win rate", [])
    assert unstated is not None and "no payoff was given" in unstated[0]
    assert "full Kelly is 10.0%" in unstated[0]


def test_an_average_win_percentage_is_not_the_win_rate() -> None:
    from argus.lui import server

    said = server._kelly_lines(
        "Kelly: average win 15%, average loss 10%, and I win 55% of the time", [])
    assert said is not None and "55% win rate" in said[0] and "1.5:1" in said[0]


# --------------------------------------------------------------------------- 1. scenarios


def _points(base: date, marks: dict[date, float]) -> dict[date, float]:
    return {base: 1.0, **marks}


def _stub_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    long_ago = date(1999, 1, 4)
    series: dict[str, dict[date, float]] = {
        "SPY": _points(long_ago, {date(2022, 1, 3): 100.0, date(2022, 10, 12): 75.5,
                                  date(2020, 2, 19): 100.0, date(2020, 3, 23): 66.0,
                                  date(2008, 9, 12): 100.0, date(2009, 3, 9): 54.0}),
        "TLT": _points(long_ago, {date(2022, 1, 3): 100.0, date(2022, 10, 12): 70.7}),
        "GC=F": _points(long_ago, {date(2022, 1, 3): 1800.0, date(2022, 10, 12): 1677.5}),
        "QQQ": _points(long_ago, {date(2022, 1, 3): 100.0, date(2022, 10, 12): 65.0,
                                  date(2020, 2, 19): 100.0, date(2020, 3, 23): 72.0,
                                  date(2008, 9, 12): 100.0, date(2009, 3, 9): 60.0}),
        "BTC-USD": {date(2014, 9, 17): 1.0, date(2022, 1, 3): 100.0, date(2022, 10, 12): 40.0,
                    date(2020, 2, 19): 100.0, date(2020, 3, 23): 65.0},
        "SHV": _points(long_ago, {date(2022, 1, 3): 100.0, date(2022, 10, 12): 100.2,
                                  date(2020, 2, 19): 100.0, date(2020, 3, 23): 100.6,
                                  date(2008, 9, 12): 100.0, date(2009, 3, 9): 100.8}),
    }
    monkeypatch.setattr(lab, "_series", lambda ticker: series[ticker])


def test_stagflation_is_run_on_the_2022_window_and_weighted(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_prices(monkeypatch)
    said = lab.lines("How would a stagflation scenario hit a book of 50% SPY, 30% TLT, 20% gold?")
    assert said is not None
    assert said[0].startswith("Bottom line: ")
    # 0.5 x -24.5% + 0.3 x -29.3% + 0.2 x -6.8% (1677.5 / 1800 - 1 = -6.8%) = -22.4% (to 0.1)
    expected = 0.5 * (75.5 / 100 - 1) + 0.3 * (70.7 / 100 - 1) + 0.2 * (1677.5 / 1800 - 1)
    assert f"{expected:+.1%}" == "-22.4%"
    text = _joined(said)
    assert "-22.4% over the 2022 rate shock (03 Jan 2022 to 12 Oct 2022)" in said[0]
    assert "S&P 500 (SPY) -24.5% x 50% = -12.2%" in text
    assert "1973-74" in text and "not available" in text
    assert said[-1].startswith("Data: ") and said[-1].endswith("Not advice.")


def test_a_recession_names_both_windows_and_says_what_bitcoin_lacks(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_prices(monkeypatch)
    said = lab.lines("What would a recession do to a book of 40% QQQ, 40% BTC, 20% T-bills?")
    assert said is not None
    # 2020: 0.4 x -28% + 0.4 x -35% + 0.2 x +0.6% = -25.08%
    assert "-25.1% over the 2020 Covid crash (19 Feb 2020 to 23 Mar 2020)" in said[0]
    assert "cannot be computed for the 2008-09 financial crisis" in said[0]
    assert "Bitcoin (BTC-USD) had no price then" in said[0]
    # the 2022 window every asset has: 0.4 x -35% + 0.4 x -60% + 0.2 x +0.2% = -37.96%
    assert "-38.0% over the 2022 rate shock" in said[0]
    assert "not a recession" in _joined(said)
    # the rest of the 2008-09 window, on its own weights: 0.4 x -40% + 0.2 x +0.8% = -15.84%
    assert "contributed -15.8% of it" in _joined(said)


def test_a_scenario_needs_a_stated_book_and_weights_under_100(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_prices(monkeypatch)
    assert lab.lines("What is a recession?") is None
    assert lab.lines("How would a recession affect my portfolio?") is None
    over = lab.lines("A stagflation hitting 70% SPY and 60% TLT?")
    assert over is not None and "add up to 130%" in over[0]
    odd = lab.lines("A recession on 50% zzzzzz and 50% SPY?")
    assert odd is not None and "did not recognise" in odd[0] and "zzzzzz" in odd[0]
    part = lab.lines("A stagflation on 50% SPY and 30% TLT")
    assert part is not None and "the other 20% is treated as cash" in _joined(part)


def test_a_scenario_whose_history_fails_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ticker: str) -> dict[date, float]:
        raise OSError("down")

    monkeypatch.setattr(lab, "_series", boom)
    said = lab.lines("A recession on 50% SPY and 50% TLT")
    assert said is not None and "could not be read just now" in said[0]


# --------------------------------------------------------------------------- 2. risk parity


def _walk(seed: int, sigma: float, days: list[date], drift: float = 0.0) -> dict[date, float]:
    rng = random.Random(seed)
    level, out = 100.0, {}
    for day in days:
        level *= math.exp(rng.gauss(drift, sigma))
        out[day] = level
    return out


def test_risk_parity_weights_follow_the_realised_volatilities(
        monkeypatch: pytest.MonkeyPatch) -> None:
    days = _bdays(date(2025, 1, 1), 400)
    series = {"SPY": _walk(1, 0.010, days), "TLT": _walk(2, 0.006, days),
              "GLD": _walk(3, 0.015, days)}
    monkeypatch.setattr(lab, "_series", lambda ticker: series[ticker])
    said = lab.lines("Explain risk parity and build a risk parity weighting for stocks, bonds, "
                     "gold using realised vol")
    assert said is not None

    def vol(ticker: str, window: int) -> float:
        closes = [series[ticker][d] for d in days]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))][-window:]
        return statistics.stdev(rets) * math.sqrt(252)

    for window in (60, 252):
        inv = {t: 1 / vol(t, window) for t in series}
        total = sum(inv.values())
        for ticker, label in (("SPY", "stocks (SPY)"), ("TLT", "bonds (TLT)"),
                              ("GLD", "gold (GLD)")):
            line = next(x for x in said if x.startswith(f"{window}-day realised volatility"))
            assert f"{label} {vol(ticker, window):.1%} -> {inv[ticker] / total:.1%}" in line
    assert said[0].startswith("Bottom line: inverse-volatility (risk-parity) weights on 252-day")
    text = _joined(said)
    assert "Risk parity sizes each asset by how much risk it brings" in text
    assert "lever the bond leg" in text and "this does not" in text
    assert "Share of portfolio risk" in text
    assert "Weights that make those shares exactly equal" in text


def test_equal_risk_weights_equalise_the_shares() -> None:
    cov = [[0.04, 0.012, 0.006], [0.012, 0.0225, -0.003], [0.006, -0.003, 0.09]]
    weights = lab.equal_risk_weights(cov)
    assert weights is not None and abs(sum(weights) - 1) < 1e-9
    shares = lab._risk_shares(weights, cov)
    assert all(abs(s - 1 / 3) < 1e-4 for s in shares)
    # two assets: equal risk is inverse volatility whatever the correlation
    two = lab.equal_risk_weights([[0.04, 0.01], [0.01, 0.01]])
    assert two is not None and abs(two[0] - (1 / 0.2) / (1 / 0.2 + 1 / 0.1)) < 1e-3


def test_risk_parity_as_a_bare_concept_question_is_left_alone() -> None:
    assert lab.lines("What is risk parity?") is None


def test_risk_parity_with_a_missing_history_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ticker: str) -> dict[date, float]:
        raise OSError("down")

    monkeypatch.setattr(lab, "_series", boom)
    said = lab.lines("Build a risk parity weighting for SPY, TLT and GLD")
    assert said is not None and "could not be read just now" in said[0]


# --------------------------------------------------------------------------- 3. gold vs real yields


def _gold_world(monkeypatch: pytest.MonkeyPatch, *, last_residual: float,
                years: int = 14) -> tuple[list[date], list[float], list[float]]:
    days = _bdays(date(2012, 1, 2), 261 * years)
    yields = [1.0 + 1.5 * math.sin(i / 300) for i in range(len(days))]
    noise = [0.02 * math.sin(i * 0.7) for i in range(len(days))]
    noise[-1] = last_residual
    gold = [math.exp(7.5 - 0.3 * y + e) for y, e in zip(yields, noise, strict=True)]
    monkeypatch.setattr(lab, "_series", lambda ticker: dict(zip(days, gold, strict=True)))
    monkeypatch.setattr(lab, "_fred_series", lambda series, n: [
        (d.isoformat(), y) for d, y in zip(days, yields, strict=True)])
    return days, yields, gold


def test_gold_is_measured_against_the_real_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    _, yields, gold = _gold_world(monkeypatch, last_residual=0.15)
    said = lab.lines("Is gold cheap or expensive versus real yields right now?")
    assert said is not None
    first = said[0]
    assert first.startswith("Bottom line: gold at ")
    assert "looks expensive on both the full-span and the pre-2022 fit" in first
    assert f"the 10-year real yield ({yields[-1]:.2f}%, FRED DFII10" in first
    # the fit is ln(gold) on the yield; recompute the residual by the normal equations
    x = yields
    y = [math.log(g) for g in gold]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / sum(
        (a - mx) ** 2 for a in x)
    intercept = my - slope * mx
    fitted = math.exp(intercept + slope * x[-1])
    assert f"puts it at ${fitted:,.0f}" in first
    assert f"gold is ${gold[-1] - fitted:,.2f} above that" in first
    assert "the link has been steady" in first
    text = _joined(said)
    assert f"ln(gold) = {intercept:.3f} - {abs(slope):.3f} x real yield" in text
    assert "one more point of real yield goes with gold" in text and "lower" in text
    assert "Correlation of the levels" not in text  # reported in the headline, not twice
    assert "daily changes" in text
    assert said[-1].startswith("Data: ") and "DFII10" in said[-1]


def test_gold_below_its_fit_is_called_cheap(monkeypatch: pytest.MonkeyPatch) -> None:
    _gold_world(monkeypatch, last_residual=-0.15)
    said = lab.lines("Is gold cheap or expensive versus real yields right now?")
    assert said is not None and "looks cheap on both the full-span and the pre-2022 fit" in said[0]
    assert "below that" in said[0]


def test_gold_needs_ten_years_of_both_series(monkeypatch: pytest.MonkeyPatch) -> None:
    _gold_world(monkeypatch, last_residual=0.1, years=6)
    said = lab.lines("Is gold cheap or expensive versus real yields right now?")
    assert said is not None and "fewer than ten years" in said[0]


def test_gold_questions_about_other_things_are_left_alone() -> None:
    assert lab.lines("Is gold a good hedge against inflation?") is None
    assert lab.lines("What are real yields?") is None
    assert lab.lines("What is gold doing?") is None


# --------------------------------------------------------------------------- 5. earnings quality


def _company(**over: Any) -> sm.Company:
    base: dict[str, Any] = dict(
        ticker="MSFT", name="Microsoft", kind="us", end=date(2026, 6, 30),
        period="four quarters to 30 Jun 2026", filed="10-K filed 29 Jul 2026",
        rev=300e9, rev_prior=250e9, net=100e9, net_prior=80e9, cfo=150e9, cfo_prior=90e9,
        capex=60e9, buyback=20e9, mcap=2000e9, price=400.0, eps=10.0, eps_prior=8.0)
    base.update(over)
    return sm.Company(**base)


def test_earnings_quality_ratios_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["MSFT"])
    monkeypatch.setattr(lab, "_quality_inputs", lambda t: {
        "company": _company(), "assets": [900e9, 700e9, 600e9]})
    said = lab.lines("What is the quality of Microsoft's earnings: cash flow versus net income, "
                     "accruals?")
    assert said is not None
    # cash flow 150 / net income 100 = 1.50x; accruals (100 - 150) / ((900 + 700) / 2) = -6.25%
    assert said[0].startswith("Bottom line: Microsoft (MSFT), four quarters to 30 Jun 2026: "
                              "operating cash flow $150.00bn against net income $100.00bn, 1.50x")
    assert "accruals ratio -6.2% of average total assets" in said[0].replace("-6.3%", "-6.2%")
    text = _joined(said)
    assert "average total assets $800.00bn ($700.00bn a year earlier, $900.00bn now)" in text
    assert "free-cash-flow conversion" in text and "$90.00bn, 90% of net income" in text
    # a year earlier: 90 / 80 = 1.12x, accruals (80 - 90) / ((700 + 600) / 2) = -1.5%
    assert "cash flow 1.12x net income, accruals ratio -1.5%" in text
    assert "positive means profit has run ahead of cash" in text
    assert said[-1].startswith("Data: ") and "XBRL" in said[-1]


def test_earnings_quality_for_a_cash_poor_profit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["XYZ"])
    monkeypatch.setattr(lab, "_quality_inputs", lambda t: {
        "company": _company(ticker="XYZ", name="Xyz", net=100e9, cfo=40e9, capex=10e9,
                            net_prior=None, cfo_prior=None),
        "assets": [500e9, 500e9, None]})
    said = lab.lines("What is the earnings quality of XYZ, accruals?")
    assert said is not None
    assert "0.40x" in said[0] and "below reported profit" in said[0]
    assert "accruals ratio +12.0%" in said[0]


def test_earnings_quality_when_the_filings_do_not_answer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ticker: str) -> dict[str, Any]:
        raise OSError("edgar down")

    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["MSFT"])
    monkeypatch.setattr(lab, "_quality_inputs", boom)
    said = lab.lines("What is the quality of Microsoft's earnings, accruals?")
    assert said is not None and "could not be read from SEC EDGAR just now" in said[0]


def test_earnings_quality_needs_a_company_and_the_phrase() -> None:
    assert lab.lines("How do I improve my accruals accounting?") is None
    assert lab.lines("My earnings quality is poor") is None


# --------------------------------------------------------------------------- 6. insiders vs buyback


def test_insider_selling_is_set_against_the_buyback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["META"])
    monkeypatch.setattr(lab, "_insider_totals", lambda t, days=90: {
        "days": 90, "filings": 33, "sales_usd": 105e6, "sales_n": 21, "planned": 19,
        "buys_usd": 0.0, "buys_n": 0, "capped": False})
    monkeypatch.setattr(lab, "_buyback", lambda t: _company(
        ticker="META", name="Meta", buyback=3.33e9, period="four quarters to 30 Jun 2026"))
    said = lab.lines("Compare insider selling with buyback activity at Meta")
    assert said is not None
    assert said[0].startswith("Bottom line: META: insiders sold $105m on the open market and "
                              "bought nothing over the last 90 days (21 sale decisions, 19 under "
                              "10b5-1 plans); the company bought back $3.33bn in the four "
                              "quarters to 30 Jun 2026; net of the two, $3.23bn more stock was "
                              "bought than sold")
    # like for like: 3.33bn x 90 / 365 = $821m a quarter; insiders sold 105 / 821 = 12.8%
    text = _joined(said)
    assert "about $821m" in text and "12.8%" in text
    assert "the windows differ" in text


def test_insiders_who_sold_more_than_the_buyback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["AAPL"])
    monkeypatch.setattr(lab, "_insider_totals", lambda t, days=90: {
        "days": 90, "filings": 5, "sales_usd": 5e9, "sales_n": 3, "planned": 0,
        "buys_usd": 1e9, "buys_n": 2, "capped": True})
    monkeypatch.setattr(lab, "_buyback", lambda t: _company(buyback=1e9))
    said = lab.lines("Compare Apple's insider selling with its buyback: net signal?")
    assert said is not None
    assert "bought $1.00bn" in said[0] and "$3.00bn more stock was sold than bought" in said[0]
    assert "200-line cap" in _joined(said)


def test_insider_buyback_when_a_source_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise OSError("down")

    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["META"])
    monkeypatch.setattr(lab, "_insider_totals", boom)
    monkeypatch.setattr(lab, "_buyback", lambda t: _company(buyback=3e9))
    said = lab.lines("Compare insider selling with buyback activity at Meta")
    assert said is not None and "insider sales could not be read just now" in said[0]
    monkeypatch.setattr(lab, "_buyback", boom)
    both = lab.lines("Compare insider selling with buyback activity at Meta")
    assert both is not None and "neither its Form 4 filings" in both[0]


def test_insider_questions_without_a_buyback_stay_with_insider_flow() -> None:
    assert lab.lines("Tell me about insider buying at NVDA") is None


# --------------------------------------------------------------------------- 7. revisions


def _trend() -> dict[str, dict[str, Any]]:
    return {
        "0q": {"end": "2026-09-30", "now": 0.44, "d7": 0.44, "d30": 0.45, "d60": 0.45, "d90": 0.55,
               "up7": 1, "down7": 3, "up30": 1, "down30": 1, "analysts": 24},
        "0y": {"end": "2026-12-31", "now": 1.74, "d7": 1.76, "d30": 1.77, "d60": 1.78,
               "d90": 2.11, "up7": 1, "down7": 3, "up30": 1, "down30": 4, "analysts": 33},
        "+1y": {"end": "2027-12-31", "now": 2.14, "d7": 2.17, "d30": 2.16, "d60": 2.20,
                "d90": 2.56, "up7": 1, "down7": 3, "up30": 1, "down30": 4, "analysts": 31}}


def test_revisions_report_the_ninety_day_trend_and_the_counts(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["TSLA"])
    monkeypatch.setattr(lab, "_trend", lambda t: _trend())
    said = lab.lines("Have analysts been revising Tesla EPS estimates up or down over the last "
                     "3 months?")
    assert said is not None
    # 0.44 / 0.55 - 1 = -20.0%; 1.74 / 2.11 - 1 = -17.5%; 2.14 / 2.56 - 1 = -16.4%
    assert "analysts have cut Tesla's (TSLA) EPS estimates" in said[0]
    assert "the quarter to 30 Sep 2026 from $0.55 to $0.44 (-20.0%)" in said[0]
    assert "this year (to Dec 2026) from $2.11 to $1.74 (-17.5%)" in said[0]
    assert "next year (to Dec 2027) from $2.56 to $2.14 (-16.4%)" in said[0]
    text = _joined(said)
    assert "revisions in the last 30 days: 1 up, 4 down (1 up, 3 down in the last 7)" in text
    assert "not reported EPS" in text


def test_revisions_through_zero_are_not_given_a_percentage(
        monkeypatch: pytest.MonkeyPatch) -> None:
    trend = _trend()
    trend["0y"].update(now=-1.94, d90=0.62)
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["COIN"])
    monkeypatch.setattr(lab, "_trend", lambda t: trend)
    said = lab.lines("Are analysts revising COIN EPS estimates down?")
    assert said is not None
    assert "from $0.62 to -$1.94" in said[0] and "-411" not in said[0]


def test_revisions_never_present_reported_eps_as_revisions(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ticker: str) -> dict[str, dict[str, Any]]:
        raise OSError("down")

    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["TSLA"])
    monkeypatch.setattr(lab, "_trend", boom)
    said = lab.lines("Have analysts been revising Tesla EPS estimates down?")
    assert said is not None
    assert "are not read just now" in said[0] and "that is not a revision" in said[0]
    assert "$" not in said[0]


def test_revisions_ignore_other_analyst_questions() -> None:
    assert lab.lines("Analysts upgrade Tesla?") is None
    assert lab.lines("How accurate are analyst estimates?") is None


# --------------------------------------------------------------------------- 8. guidance


def _release(**over: Any) -> Release:
    base: dict[str, Any] = dict(
        ticker="NVDA", filed=date(2026, 8, 26), accession="0001-26-000073",
        url="https://www.sec.gov/Archives/edgar/data/1/q2.htm", headline="NVIDIA Announces",
        revenue=96.2e9, revenue_sentence="Revenue was $96.2 billion",
        outlook=("Revenue is expected to be $108.0 billion, plus or minus 2%.",
                 "GAAP gross margins are expected to be 74.0%, plus or minus 50 basis points."),
        guided_revenue=108e9, guided_band_pct=2.0)
    base.update(over)
    return Release(**base)


def _reaction(day: date, move: float, market: float | None = 0.0066) -> Any:
    return SimpleNamespace(released=datetime(day.year, day.month, day.day, 20, 21, tzinfo=UTC),
                           base_day=day, moved_day=day + timedelta(days=1), move=move,
                           market=market)


def test_nvidia_guidance_and_the_next_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["NVDA"])
    monkeypatch.setattr(lab, "_releases", lambda t: [
        _release(), _release(filed=date(2026, 5, 20), revenue=81.6e9, guided_revenue=91e9)])
    monkeypatch.setattr(lab, "_reactions", lambda t: [_reaction(date(2026, 8, 26), 0.0874)])
    said = lab.lines("What did Nvidia guide for next quarter and how did the stock react the "
                     "next day?")
    assert said is not None
    # 108.0 / 96.2 - 1 = +12.3%
    assert "NVDA guided next-quarter revenue to $108.00bn, plus or minus 2% (+12.3% on the " \
        "$96.20bn it just reported)" in said[0]
    assert "the stock moved +8.7% by the close of 27 Aug 2026" in said[0]
    assert "the S&P 500 +0.7% that day" in said[0]
    text = _joined(said)
    # reported 96.2 against the 91.0 guided a quarter earlier: +5.7%, above the +/-2% range
    assert "had guided $91.00bn plus or minus 2% and reported $96.20bn (+5.7%), above that range" \
        in text
    assert "Rest of the outlook" in text and "gross margins" in text
    assert "Source: https://www.sec.gov/Archives" in text


def test_apple_has_no_guidance_in_its_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["AAPL"])
    monkeypatch.setattr(lab, "_releases", lambda t: [_release(
        ticker="AAPL", filed=date(2026, 7, 30), revenue=109.4e9, outlook=(),
        guided_revenue=None, guided_band_pct=None)])
    monkeypatch.setattr(lab, "_reactions",
                        lambda t: [_reaction(date(2026, 7, 30), -0.0736, 0.0072)])
    monkeypatch.setattr(lab, "_last_surprise", lambda t: (date(2026, 6, 30), 2.02, 1.89))
    said = lab.lines("Did Apple beat or miss last quarter, and what did it guide for the next one?")
    assert said is not None
    # 2.02 / 1.89 - 1 = +6.9%
    assert said[0].startswith("Bottom line: Apple beat the consensus for the quarter to 30 Jun "
                              "2026: "
                              "EPS $2.02 against $1.89 expected, +7%")
    # the beat leads; the guidance is its own line, its name's capital kept
    assert said[1].startswith("Guidance: Apple's release")
    assert "contains no outlook or guidance section" in said[1]
    assert "-7.4% by the close of 31 Jul 2026" in said[1]


def test_guidance_reads_a_miss_and_survives_a_missing_surprise(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["NVDA"])
    monkeypatch.setattr(lab, "_releases", lambda t: [_release()])
    monkeypatch.setattr(lab, "_reactions", lambda t: [])
    monkeypatch.setattr(lab, "_last_surprise", lambda t: (date(2026, 6, 30), 1.0, 1.1))
    said = lab.lines("Did NVDA beat or miss, and what did it guide?")
    assert said is not None and "missed the consensus" in said[0]
    assert said[1].startswith("Guidance: ")
    assert "the price reaction could not be matched to that release" in said[1]

    def boom(t: str) -> Any:
        raise OSError("down")

    monkeypatch.setattr(lab, "_last_surprise", boom)
    unread = lab.lines("Did NVDA beat or miss, and what did it guide?")
    assert unread is not None
    assert "beat or miss against the consensus could not be read" in unread[0]


def test_guidance_when_the_release_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lab, "_tickers", lambda text, limit=3: ["NVDA"])
    monkeypatch.setattr(lab, "_releases", lambda t: [])
    said = lab.lines("What did NVDA guide for next quarter?")
    assert said is not None and "could not be read just now" in said[0]


def test_guidance_questions_about_the_fed_are_left_alone() -> None:
    assert lab.lines("What did the Fed guide?") is None
    assert lab.lines("What is the outlook for AAPL?") is None


# --------------------------------------------------------------------------- 9. research brief


def _facts(monkeypatch: pytest.MonkeyPatch, *, falling: bool = True) -> list[float]:
    days = _bdays(date(2025, 6, 2), 300)
    if falling:
        closes = [100.0] * 200 + [100.0 - 0.4 * i for i in range(1, 101)]
    else:
        closes = [100.0 + 0.4 * i for i in range(300)]
    facts = lab._Facts(
        symbol="COINUSDT", ticker="COIN", equity=True, days=days, closes=closes,
        where="the stock's own daily closes (Yahoo Finance, split-adjusted)",
        company=_company(ticker="COIN", name="Coinbase", rev=8e9, rev_prior=5e9, net=-1e9,
                         net_prior=2e9, cfo=1e9, cfo_prior=1e9, capex=0.2e9, buyback=2e9,
                         mcap=40e9, price=60.0, eps=-4.0),
        insider={"days": 90, "filings": 20, "sales_usd": 30e6, "sales_n": 5, "planned": 4,
                 "buys_usd": 0.0, "buys_n": 0, "capped": False},
        trend={"0y": {"end": "2026-12-31", "now": 1.0, "d90": 1.5, "up30": 1, "down30": 4},
               "0q": {"end": "2026-09-30", "now": 0.3, "d90": 0.5}},
        report=SimpleNamespace(day=date(2026, 10, 29), timing="after the close",
                               source="Yahoo Finance earnings calendar", estimated=True),
        ex_div=None, move=(0.09, 4), filings=[(date(2026, 10, 2), "5.02,9.01")], missing=[])
    monkeypatch.setattr(lab, "_gather", lambda symbol: facts)
    return closes


def test_the_brief_has_bull_bear_and_watch_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    closes = _facts(monkeypatch)
    said = lab.lines("Give me a research brief on COIN with bull/bear case and what to watch")
    assert said is not None
    assert said[0].startswith("Bottom line: COIN, on price history to ")
    assert "Bull case:" in said and "Bear case:" in said and "What to watch:" in said
    bull = said[said.index("Bull case:") + 1: said.index("Bear case:")]
    bear = said[said.index("Bear case:") + 1: said.index("What to watch:")]
    assert 1 <= len(bull) <= 4 and 2 <= len(bear) <= 4
    last, sma = closes[-1], sum(closes[-200:]) / 200
    assert f"Price ${last:,.2f} is {last / sma - 1:+.1%} against its 200-day average" in _joined(
        bear)
    assert "Revenue $8.00bn, +60.0% on the four quarters before" in _joined(bull)
    assert "bought back $2.00bn, 5.0% of its market cap" in _joined(bull)
    assert "Consensus EPS" in _joined(bear) and "from $1.50 to $1.00 (-33.3%)" in _joined(bear)
    watch = _joined(said[said.index("What to watch:"):])
    assert "29 Oct 2026 (after the close): next earnings report" in watch
    assert "date estimated" in watch and "moved 9.0% either way" in watch
    assert "item 5.02 (a change of officers or directors)" in watch
    assert said[-1].startswith("Data: ") and said[-1].endswith("Not advice.")


def test_the_follow_up_tests_the_bear_case_from_the_brief(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _facts(monkeypatch)
    brief = "Give me a research brief on COIN with bull and bear case"
    said = lab.lines("What would change your mind on the bear case?", [brief])
    assert said is not None
    assert said[0].startswith("Bottom line: what would change the bear case on COIN")
    body = _joined(said)
    assert "It would change if: a daily close above the 200-day average" in body
    assert "It would change if: net margin back above 40.0% over the next four quarters" in body
    assert "recomputed on today's data" in body
    assert "29 Oct 2026" in body
    bull = lab.lines("What would change your mind on the bull case?", [brief])
    assert bull is not None and "bull case on COIN" in bull[0]
    assert "revenue growth falling below zero in the next 10-Q" in _joined(bull)


def test_the_follow_up_without_a_brief_before_it_is_left_alone(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _facts(monkeypatch)
    assert lab.lines("What would change your mind on the bear case?", []) is None
    assert lab.lines("What would change your mind on the bear case?", ["hello"]) is None


def test_a_rising_stock_leads_with_the_bull_side(monkeypatch: pytest.MonkeyPatch) -> None:
    _facts(monkeypatch, falling=False)
    said = lab.lines("Give me a research brief on COIN with bull and bear case")
    assert said is not None
    assert "the strongest for is that" in said[0]
    bear = said[said.index("Bear case:") + 1: said.index("What to watch:")]
    assert not any(x.startswith("Price $") and "200-day" in x for x in bear)


def test_a_brief_needs_the_brief_words() -> None:
    assert lab.lines("What is the bull case for ETH?") is None
    assert lab.lines("What is the price of BTC?") is None


def test_a_brief_whose_prices_fail_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(symbol: str) -> Any:
        raise OSError("down")

    monkeypatch.setattr(lab, "_gather", boom)
    said = lab.lines("Give me a research brief on COIN with bull and bear case")
    assert said is not None and "could not be read just now" in said[0]


def test_a_coin_brief_names_its_unlock_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    days = _bdays(date(2025, 6, 2), 300)
    facts = lab._Facts(symbol="SOLUSDT", ticker="SOL", equity=False, days=days,
                       closes=[100.0 + 0.2 * i for i in range(300)], where="Bitget's daily closes",
                       unlock="Solana unlocks about 1,000 tokens in the next 90 days.", missing=[])
    monkeypatch.setattr(lab, "_gather", lambda symbol: facts)
    said = lab.lines("Give me a research brief on SOL with bull and bear case")
    assert said is not None
    assert "Solana unlocks about 1,000 tokens" in _joined(said)
    assert "A coin has no earnings, filings or insiders" in _joined(said)


# --------------------------------------------------------------------------- the entry point


@pytest.mark.parametrize("question", [
    "What is the price of BTC?", "What is a recession?", "Explain Kelly criterion",
    "How did NVDA do last quarter?", "Compare NVDA and AMD revenue growth",
    "Will Tesla beat earnings?", "How much is 20% of 50% SPY?",
    "I have a book of 60% SPY and 40% TLT; what is the drawdown?",
    "Is gold a good hedge against inflation?"])
def test_other_questions_are_left_to_the_other_readers(question: str) -> None:
    assert lab.lines(question) is None
    assert lab.lines(question, ["an earlier question"]) is None


def test_a_reader_that_raises_returns_an_honest_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(text: str, prior: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(lab, "_scenario", boom)
    said = lab.lines("A recession on 50% SPY and 50% TLT")
    assert said is not None and said[0].startswith("Bottom line: ")
    assert "could not be read just now" in said[0]
