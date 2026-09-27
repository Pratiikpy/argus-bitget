"""What the 2026-09-25 blind corpus exposed in the console's understanding, pinned so it stays
fixed.

240 questions written by an agent that never read this repository
(`data/lui_blind_corpus_2026-09-25.jsonl`) scored 66% on the patterns alone: nine SPY questions
refused, most Chinese phrasings refused, every stress on anything but the Nasdaq misread, and
chit-chat ("who won the lakers game") answered with a desk decision. Each class is tested here by
its mechanism, not by the corpus rows, so the corpus can still be used as a measurement.
"""
# ruff: noqa: RUF001 - real Chinese questions, full-width punctuation included

from __future__ import annotations

from typing import Any

import pytest

from argus.lui import research
from argus.lui.arbiter import in_domain
from argus.lui.question import Intent, classify
from argus.lui.research import ResearchKind
from argus.lui.research.parse import read_request
from argus.market import history, universe


@pytest.fixture(autouse=True)
def frozen_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("live fetch disabled in tests")

    monkeypatch.setattr(research.data, "_fetch_live", _fail)
    monkeypatch.setattr(history, "fetch", _fail)
    monkeypatch.setattr(universe, "_fetch_live", _fail)
    monkeypatch.setattr(universe, "_CACHE", None)
    # No live price either: an amount in shares or coins is left unpriced unless a test pins
    # prices itself (TestHoldingsStatedAsAmounts does).
    monkeypatch.setattr(research.parse, "_last_price", lambda symbol: None)


def _kind(text: str) -> str | None:
    request = read_request(text)
    return None if request is None else str(request.kind)


class TestTheTopicGate:
    @pytest.mark.parametrize("text", [
        "who won the lakers game last night",
        "help me with my resume, can you review it for a swe role",
        "explain quantum computing to me like i'm five",
    ])
    def test_chit_chat_is_off_topic(self, text: str) -> None:
        assert not in_domain(text)

    @pytest.mark.parametrize("text", [
        "what decisions has the desk made this week",
        "why did you do nothing all weekend",
        "is the log tamper-evident",
        "为什么没有交易",
        "what is NVDA doing",
    ])
    def test_desk_and_market_questions_are_on_topic(self, text: str) -> None:
        assert in_domain(text)


class TestLowercaseIndexProducts:
    def test_spy_beside_a_market_word_is_the_etf(self) -> None:
        assert _kind("spy live price") == "quote"
        assert _kind("what's support and resistance on spy") == "technicals"

    def test_spy_as_prose_is_not_a_ticker(self) -> None:
        assert research.research_symbols("i spy with my little eye")[0] == ()


class TestChinese:
    @pytest.mark.parametrize(("text", "kind"), [
        ("英伟达和特斯拉比,哪个波动更大", "compare"),
        ("比较一下亚马逊和谷歌,哪个风险更大", "compare"),
        ("我的仓位英伟达占比很高,应该怎么对冲", "hedge"),
        ("英伟达财报当天通常波动多大", "event"),
        ("英伟达今天为什么大跌", "news"),
        ("特斯拉的估值现在贵不贵", "fundamentals"),
        ("微软的机构持仓情况怎么样", "fundamentals"),
        ("特斯拉的热度是真的还是散户炒作", "sentiment"),
        ("美联储最近的政策方向是什么", "macro"),
        ("假设纳指暴跌10%,我的组合大概会跌多少", "stress"),
    ])
    def test_the_kind_is_read(self, text: str, kind: str) -> None:
        assert _kind(text) == kind

    def test_a_chinese_instruction_to_trade_is_an_order(self) -> None:
        from datetime import UTC, datetime

        text = "PLTR现在能买吗,帮我实盘下单买入"
        assert read_request(text) is None
        assert classify(text, now=datetime.now(UTC)).intent is Intent.ORDER


class TestAShockOnANamedInstrument:
    @pytest.mark.parametrize(("text", "subject", "shock"), [
        ("oil -20% shock, how does that hit my book", "CLUSDT", -20.0),
        ("if eth drops 25% what's my max drawdown look like", "ETHUSDT", -25.0),
        ("suppose mstr craters 30% in a day, how much of my book gets wiped", "MSTRUSDT", -30.0),
        ("gold spikes 10% (flight to safety) — what does that do to my equity longs", "XAUUSDT",
         10.0),
        ("如果标普明天跌5%,我的账户会亏多少", "SPYUSDT", -5.0),
        ("假设纳指暴跌10%,我的组合大概会跌多少", None, -10.0),
    ])
    def test_the_subject_and_size_are_read(self, text: str, subject: str | None,
                                           shock: float) -> None:
        request = read_request(text)
        assert request is not None and request.kind is ResearchKind.STRESS
        assert request.shock_on == subject
        assert request.shock_pct == shock

    def test_a_shock_percentage_is_not_a_holding_weight(self) -> None:
        request = read_request(
            "I hold 40% NVDA 60% MSFT; oil -20% shock, how does that hit my book")
        assert request is not None
        assert dict(request.book) == {"NVDAUSDT": 0.4, "MSFTUSDT": 0.6}
        assert request.shock_on == "CLUSDT"

    def test_the_answer_shocks_the_named_instrument_not_qqq(self) -> None:
        request = research.ResearchRequest(
            kind=ResearchKind.STRESS, symbols=("NVDAUSDT", "COINUSDT"),
            book={"NVDAUSDT": 0.5, "COINUSDT": 0.5}, shock_pct=-30.0, shock_on="MSTRUSDT")
        answer = research.run("if mstr craters 30% what happens to my book", request)
        assert not answer.refused
        moves = [line for line in answer.lines if line.startswith("If ")]
        assert moves and all(line.startswith("If MSTR moves") for line in moves)
        assert any(line.startswith("Hedge:") and "MSTR" in line for line in answer.lines)
        # The trim-or-hedge line is phrased against the shocked name, not QQQ.
        assert any(line.startswith("Bottom line:") and "QQQ" not in line for line in answer.lines)


class TestEnglishPhrasings:
    @pytest.mark.parametrize(("text", "kind"), [
        ("why did googl gap up", "news"),
        ("what happened to btc overnight", "news"),
        ("comparing tsla msft and nvda risk side by side", "compare"),
        ("how are rates looking right now", "macro"),
        ("ETH perp funding rate right now", "quote"),
        ("eth round trip cost if I buy and sell right now", "quote"),
        ("sell 500 shares mstr without tanking the price, how", "execution"),
        ("how do I scale into a 20k qqq position", "execution"),
        ("coin's typical earnings day move?", "event"),
        ("how does qqq typically behave the week of cpi", "event"),
    ])
    def test_the_kind_is_read(self, text: str, kind: str) -> None:
        assert _kind(text) == kind

    def test_an_order_on_an_unknown_ticker_is_stillnot_research(self) -> None:
        assert read_request("Execute a purchase of 5,000 shares of ZYXQ on my behalf at market "
                       "open.") is None


class TestChinesePercentHoldings:
    """Percentages beside Chinese names were not read at all until 2026-09-27: the question the
    console's own chip suggests in English, asked in Chinese, came back with no book."""

    @pytest.mark.parametrize("asked", [
        "我持有50%英伟达和50%苹果，加仓20%特斯拉会怎样影响风险",
        "我持有英伟达50%和苹果50%，加仓特斯拉20%风险会怎样",
    ])
    def test_the_book_and_the_add_are_read_in_either_order(self, asked: str) -> None:
        request = read_request(asked)
        assert request is not None and request.kind is ResearchKind.IMPACT
        assert request.symbols[0] == "TSLAUSDT"
        assert request.book == pytest.approx({"NVDAUSDT": 0.5, "AAPLUSDT": 0.5})
        assert request.size == pytest.approx(0.2)

    def test_a_three_name_book_separated_by_the_chinese_comma(self) -> None:
        request = read_request("我持有40%英伟达、30%微软、30%苹果，再买入15%特斯拉合适吗")
        assert request is not None and request.kind is ResearchKind.IMPACT
        assert request.book == pytest.approx(
            {"NVDAUSDT": 0.4, "MSFTUSDT": 0.3, "AAPLUSDT": 0.3})
        assert request.size == pytest.approx(0.15)

    def test_a_figure_belongs_to_the_name_it_touches(self) -> None:
        """"英伟达50%和苹果50%": the first 50% is NVIDIA's, never read as a weight on 苹果 across
        the 和 between them."""
        request = read_request("我持有英伟达50%和苹果50%，加仓特斯拉20%风险会怎样")
        assert request is not None and set(request.book) == {"NVDAUSDT", "AAPLUSDT"}


class TestHoldingsStatedAsAmounts:
    @pytest.fixture(autouse=True)
    def prices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = {"AAPLUSDT": 200.0, "TSLAUSDT": 400.0, "NVDAUSDT": 100.0, "BTCUSDT": 60000.0,
                 "ETHUSDT": 3000.0}
        monkeypatch.setattr(research.parse, "_last_price", table.get)

    def test_share_counts_in_chinese_become_weights_by_value(self) -> None:
        request = read_request("我持有200股苹果和50股特斯拉,现在加仓英伟达合适吗?")
        assert request is not None and request.kind is ResearchKind.IMPACT
        assert request.symbols[0] == "NVDAUSDT"
        # 200 x 200 = 40,000 of AAPL against 50 x 400 = 20,000 of TSLA.
        assert request.book == pytest.approx({"AAPLUSDT": 2 / 3, "TSLAUSDT": 1 / 3})

    def test_shares_and_dollars_mix(self) -> None:
        request = read_request("I hold 100 shares of AAPL and $20k in TSLA, should I add NVDA?")
        assert request is not None
        assert request.book == pytest.approx({"AAPLUSDT": 0.5, "TSLAUSDT": 0.5})

    def test_coin_amounts_feed_a_stress(self) -> None:
        request = read_request("I have 2 BTC and 20 ETH, what if bitcoin drops 20%, how bad is my "
                          "portfolio")
        assert request is not None and request.kind is ResearchKind.STRESS
        assert request.book == pytest.approx({"BTCUSDT": 2 / 3, "ETHUSDT": 1 / 3})
        assert request.shock_pct == -20.0

    def test_an_order_size_is_not_a_holding(self) -> None:
        request = read_request("sell 500 shares mstr without tanking the price, how")
        assert request is not None and request.kind is ResearchKind.EXECUTION
        assert not request.book


class TestEitherReaderCanSayNotResearch:
    def test_the_language_models_confident_none_is_binding(self) -> None:
        from argus.lui.arbiter import not_research

        audit = {"model": {"kind": "none", "confidence": 0.95, "why": "a price forecast"}}
        assert not_research(audit) == ("refuse", 0.95)

    def test_an_unsure_language_model_is_not(self) -> None:
        from argus.lui.arbiter import not_research

        assert not_research({"model": {"kind": "none", "confidence": 0.5}}) is None
        assert not_research({"model": {"kind": "quote", "confidence": 0.99}}) is None

    def test_the_kind_models_verdict_is_read_from_its_audit_line(self) -> None:
        from argus.lui.arbiter import not_research

        assert not_research({"model": {"why": "kind model: record at 0.41"}}) == ("record", 0.41)

    @pytest.mark.parametrize("text", [
        "what will gold price be exactly one year from now",
        "what will ETH be trading at in 3 months",
        "give me the exact price of BTC two months from now",
    ])
    def test_forecast_phrasings_are_forecasts(self, text: str) -> None:
        from argus.lui.research.parse import PRICE_FORECAST

        assert PRICE_FORECAST.search(text)


def test_the_status_page_reports_the_measured_understanding(tmp_path: Any) -> None:
    import json

    from argus.lui.status_page import sweep_lines

    (tmp_path / "lui_final_heldout_report.json").write_text(json.dumps({
        "measured_at": "2026-09-25", "written_by": "a blind writer",
        "console_before": {"no_book": "125/240 (52.1%)"},
        "console_after": {"no_book": "196/240 (81.7%)", "saved_book": "204/240 (85.0%)"},
        "kind_model_alone": {"correct": 219, "rows": 240}}), encoding="utf-8")
    line = dict(sweep_lines(tmp_path))["Understanding questions"]
    assert "with no language model" in line
    assert "125/240 (52.1%)" in line and "196/240 (81.7%)" in line and "219/240" in line


class TestTheJudgePass:
    """Found by reading twenty live answers as a judge would (2026-09-25)."""

    def test_a_stated_add_keeps_the_impact_reading(self) -> None:
        from argus.lui.research import detect, pattern_reading_wins

        text = "I'm a conservative investor, should I add 15% TSLA?"
        request = detect(text)
        assert request is not None and pattern_reading_wins(request, text)

    def test_a_dated_price_question_is_a_forecast(self) -> None:
        from argus.lui.research.parse import PRICE_FORECAST

        assert PRICE_FORECAST.search("what will BTC be next Friday")

    @pytest.mark.parametrize(("question", "rsi", "start"), [
        ("is TSLA overbought", "63.4", "Bottom line: No — TSLA is not overbought"),
        ("is TSLA overbought", "74.0", "Bottom line: Yes — TSLA is overbought"),
        ("英伟达超卖了吗", "25.0", "Bottom line: Yes — NVDA is oversold"),
    ])
    def test_the_overbought_question_gets_a_yes_or_no(self, question: str, rsi: str,
                                                      start: str) -> None:
        from argus.lui.research.technicals import _answer_the_state_asked

        symbol = "NVDAUSDT" if "英伟达" in question else "TSLAUSDT"
        lines = ["Bottom line: momentum turning down.", f"RSI(14, 4h) {rsi} — neutral."]
        out = _answer_the_state_asked(question, symbol, lines)
        assert out[0].startswith(start) and "momentum turning down" in out[0]
        assert len(out) == 2

    def test_a_semicolon_inside_brackets_is_not_a_sentence_end(self) -> None:
        from argus.lui.research.text import sentence_cut

        text = ("Stand aside: the social feed is chatter with nothing time-sensitive; the macro "
                "backdrop (VIX 14.81, normal regime; F&G 71; nothing scheduled) gives no edge "
                "either, and the anchor market is asleep for a long stretch of the weekend") * 2
        assert not sentence_cut(text, 200).endswith("F&G 71;")
