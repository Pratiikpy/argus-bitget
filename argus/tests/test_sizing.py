"""Position size from a risk budget (a judge's audit, 2026-09-29: the question got the abstention
count), and the three other round-6 judge questions that the hosted routing still missed."""

from __future__ import annotations

import pytest

from argus.lui.research import detect, sizing
from argus.lui.research.fundamentals import pattern_reading_wins
from argus.lui.research.kinds import ResearchKind

JUDGE = "$50,000 account, risk at most 2% per trade with a 4% stop — what position size"


def test_the_judges_question_is_sized_net_of_the_round_trip() -> None:
    assert sizing.asks_for_size(JUDGE)
    lines, _, data = sizing.answer(JUDGE)
    sized = data["sizing"]
    assert sized["risk_usd"] == pytest.approx(1_000)
    assert sized["position_gross"] == pytest.approx(25_000)
    # the stop-out loss at the net size is exactly the budget, fees included
    assert sized["position_net"] * (0.04 + sizing.ROUND_TRIP) == pytest.approx(1_000)
    assert lines[0].startswith("Bottom line: to lose no more than $1,000 if the 4.00% stop")
    assert any("no leverage is needed" in line for line in lines)


def test_a_stop_price_and_an_entry_become_a_distance_and_units() -> None:
    q = "account of 20k, risking $200, entry 100, stop at 95 — how many shares should I buy"
    lines, _, data = sizing.answer(q)
    assert data["sizing"]["stop"] == pytest.approx(0.05)
    assert any(line.startswith("At 100.00 that is 39.06 units") for line in lines)


def test_a_named_name_sets_its_worst_day_beside_the_stop() -> None:
    q = "position size for NVDA with a 3% stop, risk 1% of a 50k account"
    lines, _, _ = sizing.answer(q, "NVDAUSDT", price=lambda s: 180.0, worst_day=lambda s: -0.05)
    assert any("worst observed 24 hours was -5.0%, wider than the 3.0% stop" in line
               for line in lines)
    assert any("of the $50,000 account" in line for line in lines)


def test_leverage_is_named_when_the_position_is_larger_than_the_account() -> None:
    lines, _, _ = sizing.answer("$10k account, risk 2%, 0.5% stop — what position size")
    assert any("x leverage" in line for line in lines)


@pytest.mark.parametrize("text", [
    "what position size is right for me",   # no budget, no stop
    "how big is the desk's position in NVDA",
    "what is a stop loss",
])
def test_a_question_without_a_budget_and_a_stop_is_not_sized(text: str) -> None:
    assert not sizing.asks_for_size(text)


@pytest.mark.parametrize(("text", "notional"), [
    ("what allocation split between BTC and ETH for $30,000", 30_000),
    ("how should I split $30k between BTC and ETH", 30_000),
])
def test_money_divided_between_names_is_a_construction_with_its_sum(text: str,
                                                                      notional: int) -> None:
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.CONSTRUCT
    assert request.symbols == ("BTCUSDT", "ETHUSDT")
    assert float(request.notional or 0) == notional
    assert pattern_reading_wins(request, text)


def test_splitting_an_order_is_still_an_execution_plan() -> None:
    request = detect("split a $50k order in NVDA")
    assert request is not None and request.kind is ResearchKind.EXECUTION


def test_an_outlook_on_a_listed_name_keeps_the_patterns_reading() -> None:
    request = detect("outlook on SOL")
    assert request is not None and request.symbols == ("SOLUSDT",)
    assert pattern_reading_wins(request, "outlook on SOL")


def test_a_brief_is_recognised_and_ordinary_questions_are_not() -> None:
    from argus.lui.server import _BRIEF_ASKED

    assert _BRIEF_ASKED.search("full research brief on COIN")
    assert _BRIEF_ASKED.search("give me a deep dive on NVDA")
    assert not _BRIEF_ASKED.search("what does adding COIN do to my risk")


def _synthetic(names: tuple[str, ...]) -> object:
    """Thirty days of hourly returns with a sharp fall at 03:00 UTC, outside US hours."""
    import math
    from datetime import UTC, datetime, timedelta

    from argus.lui.answer import Source
    from argus.lui.research.data import MarketData

    start = datetime(2026, 9, 1, tzinfo=UTC)
    raw: dict[str, dict[datetime, float]] = {}
    for k, name in enumerate((*names, "QQQUSDT")):
        series = {}
        for i in range(720):
            t = start + timedelta(hours=i)
            value = 0.002 * math.sin(i / (3 + k)) + 0.0005 * math.cos(i * (k + 1))
            if t.hour == 3 and t.day == 10:
                value = -0.05
            series[t] = value
        raw[name] = series
    return MarketData(raw=raw, provenance="synthetic", live=False,
                      source=Source(kind="computation", ref="test", detail="synthetic"))


def test_a_crypto_book_is_read_on_every_hour_and_a_stock_book_on_the_session() -> None:
    """A first-user audit, 2026-09-30: a BTC/ETH book was read on US open-session hours alone, so a
    fall at 03:00 UTC was invisible and its "worst 24 hours" spanned about 3.7 trading days."""
    from argus.lui.research.book import _book_report
    from argus.lui.research.kinds import ResearchRequest

    def us_open(t: object) -> bool:
        return 14 <= t.hour < 20  # type: ignore[attr-defined]

    crypto = ResearchRequest(kind=ResearchKind.BOOK, symbols=("BTCUSDT", "ETHUSDT"),
                             book={"BTCUSDT": 0.5, "ETHUSDT": 0.5})
    lines, _, payload = _book_report(crypto, _synthetic(("BTCUSDT", "ETHUSDT")), us_open)
    assert any("every hour, as crypto trades" in line for line in lines)
    assert payload["worst_window"]["move_pct"] < -4.0  # the 03:00 fall is inside the window

    stock = ResearchRequest(kind=ResearchKind.BOOK, symbols=("NVDAUSDT", "BTCUSDT"),
                            book={"NVDAUSDT": 0.5, "BTCUSDT": 0.5})
    lines, _, payload = _book_report(stock, _synthetic(("NVDAUSDT", "BTCUSDT")), us_open)
    assert any("open-session hours" in line for line in lines)
    assert payload["worst_window"]["move_pct"] > -4.0  # the session never saw it


@pytest.mark.parametrize(("text", "said"), [
    # a hostile review, 2026-09-30: each of these reached the stress or execution engine, and a $0
    # account was given a $250,000 order plan
    ("$0 account, risk 2%, stop 1%, how much BTC should I buy", "an account of $0"),
    ("risk 150% of my $20k account with a 5% stop, what size", "is not a risk budget"),
    ("$10k account, risk $200, entry 100, stop at 100", "a stop at the entry is no distance"),
    ("risk 2% of my 50k account, how big a position in NVDA", "no stop was given"),
    ("short TSLA, entry 350, stop 340, risk $200", "a short's stop sits above its entry"),
])
def test_an_input_no_honest_size_follows_from_is_refused_with_the_reason(text: str,
                                                                          said: str) -> None:
    assert sizing.asks_for_size(text)
    lines, _, data = sizing.answer(text)
    assert data["sizing"] is None
    assert said in lines[0]


def test_a_stop_above_the_entry_is_sized_as_a_short_and_says_so() -> None:
    lines, _, data = sizing.answer("$10k account, risk $100, entry 100, stop at 105")
    assert data["sizing"]["stop"] == pytest.approx(0.05)
    assert "(a short)" in lines[0]
    assert any("sized as a short" in line for line in lines)


def test_a_budget_and_a_stop_are_a_sizing_question_however_worded() -> None:
    assert sizing.asks_for_size("$10k account, risk $500, entry 100, stop 95")
    assert not sizing.asks_for_size("what is the risk of a 2% drop")
