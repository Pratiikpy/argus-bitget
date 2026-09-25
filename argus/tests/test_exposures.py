"""`argus.lui.exposures`: the sector map, the factor regression, the book arithmetic, the answer's
wording and labels, and which questions reach it. Offline: every series is synthetic or frozen."""

from __future__ import annotations

import glob
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from argus.lui import exposures as ex
from argus.lui.provenance import label

DATA = Path(__file__).resolve().parents[1] / "data"

POSITIVES = [
    "40% MSFT, 30% META, 30% GOOGL — what are my sector and factor exposures?",
    "what are my sector exposures?",
    "show me my factor exposures",
    "sector breakdown of my portfolio please",
    "how does adding 20% NVDA change my sector and factor exposures",
    "what's my factor tilt",
    "which sectors is my book in?",
    "what sectors does my portfolio hold",
    "am I overweight tech? give me the sector split",
    "do I have a momentum tilt in my book",
    "size factor exposure of my portfolio",
    "exposures by sector for 50% AAPL 50% XOM",
    "industry breakdown of my holdings",
    "what style exposures do I have with 60% QQQ and 40% IWM",
    "crypto beta of my book",
    "sector weights if I buy 10% JPM",
    "我的行业敞口是多少?",
    "组合的因子暴露怎么样",
    "40% 微软 30% 谷歌 30% 脸书 板块分布",
    "看看我的风格暴露",
    "加仓英伟达后行业占比会怎么变",
]

NEAR_MISSES = [
    "what's my exposure to NVDA?",
    "what is my crypto exposure",
    "what's the beta of my book to the S&P 500",
    "how risky is my portfolio",
    "how would adding TSLA change my portfolio risk",
    "is NVDA overbought",
    "what's the sector doing today",
    "best tech stocks to buy",
    "which sector will outperform next year",
    "explain the momentum factor",
    "what is a factor model",
    "if I add 500 shares of NVDA to the book what does that do to my beta and sector concentration",
    "在现有组合中加入200股TSLA，对组合的因子暴露（动量、波动率）有何影响？",  # noqa: RUF001
    "how concentrated is my book",
    "correlation between BTC and NVDA",
    "英伟达现在值得买吗?",
    "hedge my tech exposure",
    "my exposure is too high, what should I trim",
]


@pytest.mark.parametrize("text", POSITIVES)
def test_exposure_questions_reach_it(text: str) -> None:
    assert ex.asks_exposures(text), text


@pytest.mark.parametrize("text", NEAR_MISSES)
def test_near_misses_do_not(text: str) -> None:
    assert not ex.asks_exposures(text), text


def test_no_heldout_question_is_captured() -> None:
    files = glob.glob(str(DATA / "lui_*_2026-09-25.jsonl"))
    assert len(files) == 3
    texts = [json.loads(line)["text"] for f in files
             for line in Path(f).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(texts) == 680
    assert [t for t in texts if ex.asks_exposures(t)] == []


# --- the regression ---------------------------------------------------------------------------


def test_ols_recovers_known_coefficients() -> None:
    rng = random.Random(7)
    xs = [[rng.gauss(0, 0.01) for _ in range(400)] for _ in range(4)]
    truth = [0.0002, 1.3, -0.4, 0.25, 0.1]
    y = [truth[0] + sum(truth[i + 1] * xs[i][t] for i in range(4)) + rng.gauss(0, 0.002)
         for t in range(400)]
    fitted = ex.ols(y, xs)
    assert fitted is not None
    coef, ses, r2 = fitted
    for got, want, se in zip(coef[1:], truth[1:], ses[1:], strict=True):
        assert abs(got - want) < 4 * se
    assert r2 > 0.95


def test_ols_refuses_a_singular_design() -> None:
    x = [0.01 * i for i in range(100)]
    assert ex.ols([2 * v for v in x], [x, x]) is None


def _walk(returns: list[float], start: float = 100.0) -> list[float]:
    out = [start]
    for r in returns:
        out.append(out[-1] * (1.0 + r))
    return out


def _synthetic(n: int = 300, seed: int = 3) -> tuple[list[date], dict[str, list[float]]]:
    rng = random.Random(seed)
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(n + 1)]
    f = {k: [rng.gauss(0, 0.01) for _ in range(n)] for k in ("mkt", "smb", "mom", "btc")}
    return days, f


def _factor_closes(days: list[date], f: dict[str, list[float]]) -> dict[str, dict[date, float]]:
    spy = _walk(f["mkt"])
    iwm = _walk([m + s for m, s in zip(f["mkt"], f["smb"], strict=True)])
    mtum = _walk([m + s for m, s in zip(f["mkt"], f["mom"], strict=True)])
    btc = _walk(f["btc"], 80_000.0)
    return {k: dict(zip(days, v, strict=True))
            for k, v in (("SPY", spy), ("IWM", iwm), ("MTUM", mtum), ("BTC", btc))}


def _asset(days: list[date], f: dict[str, list[float]], b: tuple[float, float, float, float],
           noise: float = 0.001, seed: int = 11) -> dict[date, float]:
    rng = random.Random(seed)
    rets = [b[0] * f["mkt"][t] + b[1] * f["smb"][t] + b[2] * f["mom"][t] + b[3] * f["btc"][t]
            + rng.gauss(0, noise) for t in range(len(f["mkt"]))]
    return dict(zip(days, _walk(rets), strict=True))


def test_fit_holding_recovers_loadings_over_the_window() -> None:
    days, f = _synthetic()
    closes = _asset(days, f, (1.2, -0.5, 0.3, 0.05))
    fit = ex.fit_holding(closes, _factor_closes(days, f))
    assert fit is not None
    assert fit.n == ex.WINDOW_DAYS
    # simple returns of a compounded spread are not exactly the spread, so a small tolerance
    assert abs(fit.loadings["market"] - 1.2) < 0.03
    assert abs(fit.loadings["size"] + 0.5) < 0.03
    assert abs(fit.loadings["momentum"] - 0.3) < 0.03
    assert abs(fit.loadings["crypto"] - 0.05) < 0.02
    assert fit.t_stats["market"] > 20


def test_too_little_history_gives_no_loading() -> None:
    days, f = _synthetic(n=40)
    assert ex.fit_holding(_asset(days, f, (1, 0, 0, 0)), _factor_closes(days, f)) is None


# --- the book arithmetic ----------------------------------------------------------------------


def test_after_trade_adds_trims_and_exits() -> None:
    book = {"A": 0.4, "B": 0.3, "C": 0.3}
    added = ex.after_trade(book, {"D": 0.2})
    assert added == pytest.approx({"A": 0.32, "B": 0.24, "C": 0.24, "D": 0.2})
    trimmed = ex.after_trade(book, {"A": 0.1})
    assert trimmed == pytest.approx({"A": 0.1, "B": 0.45, "C": 0.45})
    assert sum(ex.after_trade(book, {"A": 0.0}).values()) == pytest.approx(1.0)
    assert "A" not in ex.after_trade(book, {"A": 0.0})


def test_effective_number() -> None:
    assert ex.effective_number({"A": 1.0}) == pytest.approx(1.0)
    assert ex.effective_number({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == pytest.approx(4.0)
    assert ex.effective_number({"A": 0.4, "B": 0.3, "C": 0.3}) == pytest.approx(1 / 0.34)


ROWS: dict[str, Any] = {
    "MSFTUSDT": {"kind": "stock", "ticker": "MSFT", "industry": "Software - Infrastructure",
                 "weights": {"Information Technology": 1.0}},
    "METAUSDT": {"kind": "stock", "ticker": "META", "industry": "Internet Content & Information",
                 "weights": {"Communication Services": 1.0}},
    "GOOGLUSDT": {"kind": "stock", "ticker": "GOOGL",
                  "industry": "Internet Content & Information",
                  "weights": {"Communication Services": 1.0}},
    "NVDAUSDT": {"kind": "stock", "ticker": "NVDA", "industry": "Semiconductors",
                 "weights": {"Information Technology": 1.0}},
    "QQQUSDT": {"kind": "etf", "ticker": "QQQ", "weights": {"Information Technology": 0.6,
                                                            "Communication Services": 0.4}},
}


def test_sector_weights_look_through_an_etf_and_name_gold_and_crypto(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "_is_rwa", lambda s: False)
    weights = ex.sector_weights({"QQQUSDT": 0.5, "XAUUSDT": 0.25, "BTCUSDT": 0.25}, ROWS)
    assert weights == pytest.approx({"Information Technology": 0.3, "Gold": 0.25,
                                     "Crypto": 0.25, "Communication Services": 0.2})


def test_index_products_are_looked_through_their_etf() -> None:
    info = ex.classify("NDX100USDT", ROWS)
    assert info.kind == "index" and info.weights == ROWS["QQQUSDT"]["weights"]


# --- rows from Yahoo --------------------------------------------------------------------------


def _price(last: float, currency: str = "USD", name: str = "X") -> dict[str, Any]:
    return {"regularMarketPrice": {"raw": last}, "currency": currency, "longName": name}


def test_a_stock_row_is_price_checked() -> None:
    summary = {"price": _price(516.2, name="Microsoft Corporation"),
               "quoteType": {"quoteType": "EQUITY"},
               "assetProfile": {"sector": "Technology", "industry": "Software - Infrastructure"}}
    row = ex.row_from_summary(summary, ticker="MSFT", bitget_last=515.0)
    assert row["check"] == "price" and row["weights"] == {"Information Technology": 1.0}
    collision = ex.row_from_summary(summary, ticker="MSFT", bitget_last=0.5)
    assert collision["check"] == "failed" and collision["kind"] == "unclassified"


def test_an_etf_row_is_looked_through_and_leverage_is_flagged() -> None:
    holdings = {"stockPosition": {"raw": 0.9}, "bondPosition": {"raw": 0.1},
                "sectorWeightings": [{"technology": {"raw": 0.6}},
                                     {"healthcare": {"raw": 0.4}}]}
    plain = ex.row_from_summary(
        {"price": _price(600.0, name="Invesco QQQ Trust"), "quoteType": {"quoteType": "ETF"},
         "fundProfile": {"categoryName": "Large Growth"}, "topHoldings": holdings},
        ticker="QQQ", bitget_last=601.0)
    assert plain["weights"] == pytest.approx({"Information Technology": 0.54,
                                              "Health Care": 0.36, "Bonds": 0.1})
    assert plain["leveraged"] is False
    tqqq = ex.row_from_summary(
        {"price": _price(90.0, name="ProShares UltraPro QQQ"), "quoteType": {"quoteType": "ETF"},
         "fundProfile": {"categoryName": "Trading--Leveraged Equity"}, "topHoldings": holdings},
        ticker="TQQQ", bitget_last=90.5)
    assert tqqq["leveraged"] is True
    sgov = ex.row_from_summary(
        {"price": _price(100.5, name="iShares 0-3 Month Treasury Bond ETF"),
         "quoteType": {"quoteType": "ETF"}, "fundProfile": {"categoryName": "Ultrashort Bond"},
         "topHoldings": {}}, ticker="SGOV", bitget_last=100.4)
    assert sgov["leveraged"] is False


# --- the frozen map ---------------------------------------------------------------------------


def test_the_frozen_map_covers_every_stock_and_etf_perpetual() -> None:
    from argus.market.universe import NOT_EQUITY

    frozen = json.loads((DATA / "sector_map.json").read_text(encoding="utf-8"))
    venue = json.loads((DATA / "venue_universe.json").read_text(encoding="utf-8"))
    equities = {s for s, v in venue["contracts"].items() if v["rwa"] and s not in NOT_EQUITY}
    rows = frozen["rows"]
    assert equities <= set(rows)
    assert rows["MSFTUSDT"]["weights"] == {"Information Technology": 1.0}
    assert rows["METAUSDT"]["weights"] == {"Communication Services": 1.0}
    assert rows["GOOGLUSDT"]["weights"] == {"Communication Services": 1.0}
    assert rows["MSFTUSDT"]["check"] == "price"
    for symbol, word in (("SAMSUNGUSDT", "Samsung Electronics"), ("SKHYNIXUSDT", "SK hynix"),
                         ("TENCENTUSDT", "Tencent"), ("SOFTBANKUSDT", "SoftBank"),
                         ("TOKYOELUSDT", "Tokyo Electron"), ("XIAOMIUSDT", "Xiaomi")):
        assert word.lower() in str(rows[symbol]["name"]).lower(), symbol
    classified = sum(1 for r in rows.values() if r["kind"] != "unclassified")
    assert classified / len(rows) > 0.9
    for row in rows.values():
        assert math.isclose(sum(row["weights"].values()), 1.0, abs_tol=1e-6)


# --- the answer -------------------------------------------------------------------------------


def _inputs() -> ex.Inputs:
    days, f = _synthetic(n=300, seed=5)
    factors = _factor_closes(days, f)
    closes = {
        "MSFTUSDT": _asset(days, f, (1.0, -0.4, 0.2, 0.02), seed=1),
        "METAUSDT": _asset(days, f, (1.3, -0.5, 0.4, 0.03), seed=2),
        "GOOGLUSDT": _asset(days, f, (1.1, -0.3, 0.1, 0.01), seed=3),
        "NVDAUSDT": _asset(days, f, (1.8, -0.2, 0.9, 0.08), seed=4),
    }
    return ex.Inputs(closes=closes, factors=factors,
                     origins={s: "synthetic closes" for s in closes})


def test_the_answer_before_and_after_a_trade() -> None:
    book = {"MSFTUSDT": 0.4, "METAUSDT": 0.3, "GOOGLUSDT": 0.3}
    lines, sources, data = ex.exposures_answer(book, {"NVDAUSDT": 0.2}, inputs=_inputs(),
                                               rows=ROWS)
    assert lines[0].startswith("Actionable: adding 20% NVDA moves ")
    assert data["sectors_before"] == pytest.approx({"Communication Services": 0.6,
                                                    "Information Technology": 0.4})
    assert data["sectors_after"] == pytest.approx({"Communication Services": 0.48,
                                                   "Information Technology": 0.52})
    want = 0.4 * 1.0 + 0.3 * 1.3 + 0.3 * 1.1
    assert data["loadings_before"]["market"] == pytest.approx(want, abs=0.05)
    assert data["loadings_after"]["market"] > data["loadings_before"]["market"]
    assert data["effective_positions"][1] > data["effective_positions"][0]
    after = next(line for line in lines if line.startswith("After the trade: "))
    assert "Information Technology 40% → 52%" in after
    labelled = {line.split(":")[0]: label(line) for line in lines}
    assert label(lines[0]) == "computed"
    assert labelled["After the trade"] == "computed"
    assert all(label(line) == "computed" for line in lines if " beta (" in line)
    assert all(label(line) == "assumed" for line in lines if line.startswith("Assumed:"))
    assert any("correlation" in line for line in lines)
    assert sources and all(s.ref for s in sources)


def test_the_answer_without_a_trade_names_the_largest_sector() -> None:
    book = {"MSFTUSDT": 0.4, "METAUSDT": 0.3, "GOOGLUSDT": 0.3}
    lines, _, data = ex.exposures_answer(book, inputs=_inputs(), rows=ROWS)
    assert lines[0].startswith("Actionable: your book is 60% Communication Services")
    assert data["sectors_after"] is None and data["loadings_after"] is None


def test_a_holding_without_history_is_said_to_be_missing() -> None:
    inputs = _inputs()
    del inputs.closes["GOOGLUSDT"]
    lines, _, data = ex.exposures_answer({"MSFTUSDT": 0.5, "GOOGLUSDT": 0.5}, inputs=inputs,
                                         rows=ROWS)
    gap = next(line for line in lines if line.startswith("Missing: GOOGL"))
    assert label(gap) == "missing"
    assert data["coverage_before"] == pytest.approx(0.5)


def test_parse_reads_the_book_and_the_trade() -> None:
    book, proposed, notes = ex.parse(
        "40% MSFT, 30% META, 30% GOOGL — what are my sector exposures if I add 20% NVDA?")
    assert book == pytest.approx({"MSFTUSDT": 0.4, "METAUSDT": 0.3, "GOOGLUSDT": 0.3})
    assert proposed == pytest.approx({"NVDAUSDT": 0.2})
    assert any("pro rata" in n for n in notes)


def test_parse_uses_the_saved_book_and_defaults_the_size() -> None:
    book, proposed, notes = ex.parse("how do my sector exposures change if I buy NVDA?",
                                     "50% AAPL, 50% XOM")
    assert book == pytest.approx({"AAPLUSDT": 0.5, "XOMUSDT": 0.5})
    assert proposed == {"NVDAUSDT": ex.DEFAULT_ADD}
    assert any("saved book" in n for n in notes) and any("no size" in n for n in notes)


def test_answer_asks_for_a_book_when_there_is_none() -> None:
    result = ex.answer("what are my sector exposures?")
    assert result is not None and result.lines[0].startswith("Actionable: tell me what you hold")
    assert ex.answer("is NVDA overbought?") is None
