"""Every number a question states reaches the request: the fixes `eval/figurecheck.py` drove."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from argus.eval.figurecheck import stated
from argus.lui import research
from argus.lui.kindmodel import LocalPlanner, kind_model
from argus.lui.research import ResearchKind, _pairs, detect
from argus.market import history, universe


@pytest.fixture(autouse=True)
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("live fetch disabled in tests")

    monkeypatch.setattr(research, "_fetch_live", _fail)
    monkeypatch.setattr(history, "fetch", _fail)
    monkeypatch.setattr(universe, "_fetch_live", _fail)
    monkeypatch.setattr(universe, "_CACHE", None)


def test_the_checker_reads_amounts_before_chinese_text() -> None:
    assert ("usd", 50_000.0) in stated("想加$50,000的AAPL,对整体有什么影响")
    assert ("usd", 5_000.0) in stated("want to add 5k of coin")
    assert ("pct", 30.0) in stated("holding spy etf 30% of book")


def test_a_word_between_a_ticker_and_its_weight_is_allowed() -> None:
    assert [(s, w) for _, s, w in _pairs("holding spy etf 30% of book, adding qqq")] == [
        ("SPYUSDT", 0.30)]
    assert [(s, w) for _, s, w in _pairs("NVDA stock 20%, BTC position 25%")] == [
        ("NVDAUSDT", 0.20), ("BTCUSDT", 0.25)]


def test_stated_cash_is_held_as_cash_not_scaled_away() -> None:
    request = detect("portfolio is nvda 40%, msft 20%, cash rest — should I add 10% coin?")
    assert request is not None and request.cash == pytest.approx(0.40)
    held = {s: w for s, w in request.book.items() if s != "COINUSDT"}
    assert held == pytest.approx({"NVDAUSDT": 0.40, "MSFTUSDT": 0.20})
    assert any("40%, read as cash" in n for n in request.notes)
    explicit = detect("I hold 50% BTC, 30% ETH and 20% in USDT, how risky is my book?")
    assert explicit is not None and explicit.cash == pytest.approx(0.20)


@pytest.mark.parametrize(("text", "shock"), [
    ("¿Qué pasaría con mi cuenta si el mercado de acciones cae un 20% como en 2008?", -20.0),
    ("Was passiert mit meinem Depot, wenn der Markt um 15% fällt?", -15.0),
    ("E se o mercado cair 12%, o que acontece com a minha carteira?", -12.0),
])
def test_a_market_drop_in_another_language_is_a_stress(text: str, shock: float) -> None:
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.STRESS
    assert request.shock_pct == pytest.approx(shock)


def test_a_german_cut_by_a_percentage_is_a_resize() -> None:
    request = detect("wie verändert sich mein Portfolio-Beta, wenn ich meine BTC-Perp-Position "
                     "um 40% reduziere?")
    assert request is not None and request.resize_by == pytest.approx(-0.40)


def test_an_analysis_request_that_starts_with_trim_is_not_an_order() -> None:
    request = detect("trim QQQ by 30% and show me the resulting net exposure")
    assert request is not None and request.resize_by == pytest.approx(-0.30)


@pytest.mark.parametrize(("text", "shock"), [
    ("run a -10% NVDA shock and show book impact", -10.0),
    ("I hold 50% BTC, 30% ETH, what if the market drops 8%?", -8.0),
])
def test_the_kind_models_shock_is_never_a_holding_weight(text: str, shock: float) -> None:
    model = kind_model()
    if model is None:
        pytest.skip("kind model export not present")
    plan = LocalPlanner(model).complete_json([{"role": "user", "content": text}])
    if plan.get("kind") == "stress":
        assert plan.get("shock_percent") == pytest.approx(shock)
    assert "shock_percent" not in plan or plan["shock_percent"] != 50.0


def test_value_at_risk_is_read_from_daily_history(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)

    class Bar:
        def __init__(self, i: int, close: float) -> None:
            self.ts, self.close = start + timedelta(days=i), Decimal(str(close))

    # 200 days: every 20th day falls 5%, the rest rise 0.1%
    closes, level = [], 100.0
    for i in range(200):
        level *= 0.95 if i % 20 == 19 else 1.001
        closes.append(Bar(i, level))
    monkeypatch.setattr(history, "fetch_window", lambda *a, **k: closes)
    lines, sources = research._var_lines({}, {"NVDAUSDT": 1.0}, "VaR at 95%")
    assert lines[0].startswith("Value at risk, 95%, one day: the book at these weights lost "
                               "more than 5.00% on 5% of its 199 past days")
    assert sources[0].ref == "argus.lui.research._var_lines"


@pytest.mark.parametrize(("text", "size", "stated_note"), [
    ("what does adding 10% ETH at 3x do to my risk? I hold 50% NVDA, 50% AAPL", 0.30,
     "the 10% stated is read as 30% of exposure"),
    ("wanna add 3x ETH perp longs, how bad does that wreck my portfolio beta? I hold 50% NVDA, "
     "50% AAPL", 0.60, "with no size given, 20% of margin is read as 60% of exposure"),
])
def test_an_add_at_a_multiple_is_sized_by_its_exposure(text: str, size: float,
                                                        stated_note: str) -> None:
    request = detect(text)
    assert request is not None and request.kind is ResearchKind.IMPACT
    assert request.size == pytest.approx(size) and request.leverage == 3.0
    assert any(stated_note in n for n in request.notes)


def test_the_multiple_is_applied_once_through_the_kind_model() -> None:
    model = kind_model()
    if model is None:
        pytest.skip("kind model export not present")
    text = "what does adding 10% ETH at 3x do to my risk?"
    planned, _ = research.plan_with_model(text, LocalPlanner(model))
    assert planned is not None and planned.size == pytest.approx(0.30)


def test_a_group_weight_is_split_over_its_named_or_theme_members() -> None:
    pairs = {s: w for _, s, w in _pairs("currently 60% crypto (btc+eth), 40% tech stocks")}
    assert pairs["BTCUSDT"] == pytest.approx(0.30) and pairs["ETHUSDT"] == pytest.approx(0.30)
    assert pairs["NVDAUSDT"] == pytest.approx(0.40 / 6)
    request = detect("currently 60% crypto (btc+eth), 40% tech stocks — adding coin on top, "
                     "too much?")
    assert request is not None and request.symbols[0] == "COINUSDT"
    assert any("60% crypto split equally across BTC, ETH" in n for n in request.notes)
    assert any("name the holdings to use your own" in n for n in request.notes)


def test_a_named_weight_is_not_displaced_by_a_group() -> None:
    pairs = {s: w for _, s, w in _pairs("I hold 40% NVDA, 60% crypto")}
    assert pairs["NVDAUSDT"] == pytest.approx(0.40)
    assert sum(pairs.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(("text", "multiple"), [
    ("run a stress test on my book assuming a 2x vol spike across all crypto perps", 2.0),
    ("what if volatility doubles on my book", 2.0),
    ("stress my book for a vol shock of 3x", 3.0),
    ("what if the nasdaq drops 10%", None),
])
def test_a_volatility_scenario_is_read(text: str, multiple: float | None) -> None:
    assert research._vol_multiple(text) == multiple


def test_a_volatility_scenario_scales_the_books_history(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)

    class Bar:
        def __init__(self, i: int, close: float) -> None:
            self.ts, self.close = start + timedelta(days=i), Decimal(str(close))

    closes, level = [], 100.0
    for i in range(200):
        level *= 0.95 if i % 20 == 19 else 1.001
        closes.append(Bar(i, level))
    monkeypatch.setattr(history, "fetch_window", lambda *a, **k: closes)
    lines, _ = research._var_lines({}, {"NVDAUSDT": 1.0}, "a 2x vol spike", scale=2.0)
    assert "value at risk goes from 5.00% to 10.00%" in lines[0]
    assert "worst day in 199 would have been -10.00% instead of -5.00%" in lines[0]
