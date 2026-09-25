"""`lui/research.py` — the research questions a Track 3 judge types, answered by the desk's engines.

Deterministic throughout: parsing is pure, and every `run()` test answers from the frozen Bitget
fixture (`data/risk_layer_candles_fixture.json`, real history) with the live fetch forced to fail,
so no test depends on the venue being up. The live path is exercised by hand and reported in
`Activity/PROGRESS.md`; what is pinned here is the parsing, the arithmetic contract and the
refusals.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from argus.lui import research
from argus.lui.research import (
    DEFAULT_SIZE,
    RISK_BUDGET,
    ResearchKind,
    ResearchRequest,
    detect,
    plan_with_model,
    run,
)
from argus.market import bitget_mcp, history, skills, universe


@pytest.fixture(autouse=True)
def frozen_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the frozen fallback: every live fetch fails, so answers come from the fixture."""

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("live fetch disabled in tests")

    monkeypatch.setattr(research, "_fetch_live", _fail)
    monkeypatch.setattr(history, "fetch", _fail)
    # The contract registry answers from its frozen snapshot, so which names resolve cannot
    # change with the venue's listings between runs.
    monkeypatch.setattr(universe, "_fetch_live", _fail)
    monkeypatch.setattr(universe, "_CACHE", None)
    # SEC filings are a live source too; tests that need an earnings surprise supply one.
    monkeypatch.setattr(research, "_earnings_surprise", lambda ticker: None)


class TestDetectFindsResearchQuestions:
    def test_the_handbook_worked_example_is_a_trade_impact_question(self) -> None:
        """The Track 3 Open Theme's own example, which used to be answered as "Sharpe not
        available" because the ledger patterns read it as a performance question."""
        req = detect("how would adding TSLA change my portfolio risk")
        assert req is not None
        assert req.kind is ResearchKind.IMPACT
        assert req.symbols[0] == "TSLAUSDT"
        assert not req.size_stated and req.size == DEFAULT_SIZE

    def test_book_and_size_are_separated_by_the_verb(self) -> None:
        req = detect("I hold 50% NVDA and 50% AAPL. What happens if I add 20% TSLA?")
        assert req is not None and req.kind is ResearchKind.IMPACT
        assert req.symbols[0] == "TSLAUSDT"
        assert dict(req.book) == {"NVDAUSDT": 0.5, "AAPLUSDT": 0.5}
        assert req.size == pytest.approx(0.20) and req.size_stated

    def test_fraction_notation_is_read(self) -> None:
        req = detect("I hold NVDA=0.6 AAPL=0.4, should I add MSFT at 10%")
        assert req is not None and req.symbols[0] == "MSFTUSDT"
        assert dict(req.book) == {"NVDAUSDT": 0.6, "AAPLUSDT": 0.4}
        assert req.size == pytest.approx(0.10)

    def test_names_without_weights_are_an_equal_weight_book_and_it_is_said(self) -> None:
        req = detect("I've got a bunch of Nvidia and Apple, split evenly. Would throwing some "
                     "Tesla in there make things worse?")
        assert req is not None and req.kind is ResearchKind.IMPACT
        assert req.symbols[0] == "TSLAUSDT"
        assert dict(req.book) == {"NVDAUSDT": 0.5, "AAPLUSDT": 0.5}
        assert any("equal weight" in note for note in req.notes)

    def test_owning_is_holdings_not_a_trade(self) -> None:
        """Regression: "I own 40% MSFT" once made MSFT the trade and turned a stress question
        into an impact one."""
        req = detect("what if the market drops 10% — I own 40% MSFT, 30% META, 30% GOOGL")
        assert req is not None and req.kind is ResearchKind.STRESS
        assert req.shock_pct == pytest.approx(-10.0)
        assert dict(req.book) == pytest.approx(
            {"MSFTUSDT": 0.4, "METAUSDT": 0.3, "GOOGLUSDT": 0.3}
        )

    def test_colloquial_holdings_and_sell_off(self) -> None:
        req = detect("how exposed am I if tech sells off hard? I'm mostly in microsoft and meta")
        assert req is not None and req.kind is ResearchKind.STRESS
        assert set(req.book) == {"MSFTUSDT", "METAUSDT"}

    def test_a_held_name_is_never_the_new_trade(self) -> None:
        req = detect("thinking about getting into coinbase, is that dumb given I'm heavy on MSTR")
        assert req is not None and req.symbols[0] == "COINUSDT"
        assert dict(req.book) == {"MSTRUSDT": 1.0}

    def test_holdings_that_do_not_sum_to_one_are_scaled_and_the_scaling_is_stated(self) -> None:
        req = detect("I hold 30% NVDA and 30% AAPL, what would adding 10% TSLA do to my risk")
        assert req is not None
        assert sum(req.book.values()) == pytest.approx(1.0)
        assert any("scaled to 100%" in note for note in req.notes)

    def test_two_names_compared(self) -> None:
        req = detect("is TSLA riskier than NVDA")
        assert req is not None and req.kind is ResearchKind.COMPARE
        assert set(req.symbols) == {"TSLAUSDT", "NVDAUSDT"}

    def test_an_order_split_with_a_size(self) -> None:
        req = detect("how should I split a $50k order in NVDA")
        assert req is not None and req.kind is ResearchKind.EXECUTION
        assert req.notional is not None and int(req.notional) == 50_000


class TestDetectLeavesTheRecordAlone:
    """Questions about what the desk did belong to the ledger answerers, untouched."""

    @pytest.mark.parametrize("text", [
        "why did you pass on NVDA",
        "why did the desk skip TSLA yesterday",
        "how did we do this week",
        "buy NVDA",
        "sell half of that",
        "what is XYZQ beta",  # not listed on Bitget: nothing to analyse
        "",
    ])
    def test_not_research(self, text: str) -> None:
        assert detect(text) is None


class TestRunAnswersFromTheEngines:
    def test_impact_with_a_book_leads_with_an_actionable_sizing_line(self) -> None:
        req = detect("I hold 50% NVDA and 50% AAPL. What happens if I add 20% TSLA?")
        assert req is not None
        answer = run("q", req)
        assert not answer.refused and answer.is_grounded
        assert any(line.startswith("Actionable:") for line in answer.lines)
        assert any(line.startswith("Hedge:") for line in answer.lines)
        report = answer.data["report"]
        assert report["impact"]["risk_share_after"] is not None
        assert answer.data["live_data"] is False
        assert "frozen" in answer.lines[-1]

    def test_the_sizing_ceiling_actually_respects_the_budget(self) -> None:
        """The ceiling is the arithmetic, checked independently: at the stated ceiling the name's
        risk share is inside the budget, and one point above it is not."""
        req = detect("I hold 50% NVDA and 50% AAPL. What happens if I add 20% TSLA?")
        assert req is not None
        data = research.load(req.symbols)
        columns = research._open_columns(data.raw, research._is_open())
        ceiling = research.max_size_within_budget(add="TSLAUSDT", before=req.book,
                                                  columns=columns)
        assert ceiling is not None
        at = research.decompose(research.rebalance(req.book, "TSLAUSDT", ceiling), columns)
        assert at is not None and (at.share_of_risk("TSLAUSDT") or 0) <= RISK_BUDGET
        if ceiling < 1.0:
            above = research.decompose(
                research.rebalance(req.book, "TSLAUSDT", ceiling + 0.01), columns
            )
            assert above is not None and (above.share_of_risk("TSLAUSDT") or 0) > RISK_BUDGET

    def test_standalone_position_is_described_as_a_position(self) -> None:
        req = detect("what is the beta of MSTR")
        assert req is not None
        answer = run("q", req)
        assert not answer.refused
        assert any("this position" in line for line in answer.lines)
        assert not any("this book" in line for line in answer.lines)

    def test_stress_answers_the_shock_that_was_asked(self) -> None:
        req = detect("stress test my book if the market drops 7%: 60% QQQ 40% COIN")
        assert req is not None and req.kind is ResearchKind.STRESS
        answer = run("q", req)
        assert any("-7" in line for line in answer.lines)

    def test_display_uses_bare_tickers(self) -> None:
        req = detect("compare AAPL and MSFT")
        assert req is not None
        answer = run("q", req)
        assert not any("USDT" in line for line in answer.lines if not line.startswith("Data"))

    def test_every_answer_names_its_data_and_disclaims_advice(self) -> None:
        req = detect("should I buy NVDA")
        assert req is not None
        last = run("q", req).lines[-1]
        assert last.startswith("Data:") and "you make the call" in last


class _FakeModel:
    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply = reply

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class TestThePlannerCannotPutANumberInTheAnswer:
    def test_a_valid_plan_becomes_a_request(self) -> None:
        req, audit = plan_with_model("anything", _FakeModel({
            "kind": "impact", "candidate": "TSLA", "holdings": {"NVDA": 50, "AAPL": 50},
            "size_percent": 15, "confidence": 0.9,
        }))
        assert req is not None and audit["applied"]
        assert req.symbols[0] == "TSLAUSDT" and req.size == pytest.approx(0.15)
        assert req.parsed_by == "model"

    def test_names_bitget_does_not_list_are_dropped_not_coerced(self) -> None:
        req, _ = plan_with_model("anything", _FakeModel({
            "kind": "impact", "candidate": "XYZQ", "holdings": {"QWERTZ": 100},
            "confidence": 0.9,
        }))
        assert req is None

    def test_any_listed_contract_the_model_names_is_analysed(self) -> None:
        """AMD and PLTR are not among the desk's twelve, but Bitget lists both; a research
        console that refused them would be refusing a question it can answer."""
        req, _ = plan_with_model("anything", _FakeModel({
            "kind": "impact", "candidate": "AMD", "holdings": {"PLTR": 100}, "confidence": 0.9,
        }))
        assert req is not None
        assert req.symbols[0] == "AMDUSDT" and dict(req.book) == {"PLTRUSDT": 1.0}

    def test_low_confidence_keeps_the_refusal(self) -> None:
        req, audit = plan_with_model("anything about NVDA", _FakeModel({
            "kind": "impact", "candidate": "NVDA", "confidence": 0.3,
        }))
        assert req is None and not audit["applied"]

    def test_none_kind_and_transport_failure_both_decline(self) -> None:
        assert plan_with_model("x", _FakeModel({"kind": "none", "confidence": 0.99}))[0] is None
        assert plan_with_model("x", _FakeModel(TimeoutError()))[0] is None
        assert plan_with_model("x", None)[0] is None

    def test_an_absurd_size_is_discarded(self) -> None:
        req, _ = plan_with_model("add NVDA", _FakeModel({
            "kind": "impact", "candidate": "NVDA", "size_percent": 400, "confidence": 0.9,
        }))
        assert req is not None and req.size == DEFAULT_SIZE and not req.size_stated


def test_request_serialises() -> None:
    req = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("NVDAUSDT",))
    assert req.as_dict()["kind"] == "impact"


class TestQuotes:
    """A research workbench that refuses to quote a price is not one; the old console did."""

    @pytest.mark.parametrize("text", [
        "what is NVDA trading at right now", "whats the spread on TSLA", "price of AAPL and MSFT",
    ])
    def test_a_price_question_is_a_quote(self, text: str) -> None:
        req = detect(text)
        assert req is not None and req.kind is ResearchKind.QUOTE

    def test_a_quote_never_swallows_a_heavier_question(self) -> None:
        req = detect("I hold 50% NVDA, what does adding 20% TSLA at this price do to my risk")
        assert req is not None and req.kind is ResearchKind.IMPACT

    def test_the_model_can_ask_for_a_quote(self) -> None:
        req, _ = plan_with_model("nvda px?", _FakeModel({"kind": "quote", "candidate": "NVDA",
                                                         "confidence": 0.9}))
        assert req is not None and req.kind is ResearchKind.QUOTE

    def test_the_model_saying_record_is_not_research(self) -> None:
        req, audit = plan_with_model("why did the desk pass on NVDA", _FakeModel(
            {"kind": "record", "confidence": 0.95}))
        assert req is None and "record" in audit["detail"]


class TestTheTradersOwnBudget:
    """The sizing answer is written against the trader's own risk cap when they state one."""

    def test_a_stated_budget_is_read_and_never_mistaken_for_a_weight(self) -> None:
        req = detect("I hold 50% NVDA and 50% AAPL, keep any name under 15% of my risk. "
                     "What if I add 20% TSLA?")
        assert req is not None and req.budget == pytest.approx(0.15) and req.budget_stated
        assert dict(req.book) == {"NVDAUSDT": 0.5, "AAPLUSDT": 0.5}
        assert req.size == pytest.approx(0.20)

    def test_the_budget_changes_the_verdict(self) -> None:
        base = "I hold 50% NVDA and 50% AAPL. What happens if I add 20% TSLA?"
        tight = base + " My risk budget is 15%."
        default_req, tight_req = detect(base), detect(tight)
        assert default_req is not None and tight_req is not None
        default_first = run("q", default_req).lines[0]
        tight_first = run("q", tight_req).lines[0]
        assert "25%" in default_first and "15%" in tight_first and "your budget" in tight_first

    def test_a_saved_book_can_carry_the_budget(self) -> None:
        req = research.with_book(detect("should I add TSLA"), "50% NVDA, 50% AAPL, risk budget 15%")
        assert req is not None and req.budget == pytest.approx(0.15)
        assert dict(req.book) == {"NVDAUSDT": 0.5, "AAPLUSDT": 0.5}


class TestTheNewKindsAreDetected:
    @pytest.mark.parametrize(("text", "kind"), [
        ("what is the RSI on NVDA", ResearchKind.TECHNICALS),
        ("is TSLA overbought", ResearchKind.TECHNICALS),
        ("when does NVDA report earnings", ResearchKind.FUNDAMENTALS),
        ("who owns MSFT", ResearchKind.FUNDAMENTALS),
        ("has COIN been here before", ResearchKind.ANALOGUE),
        ("what happened last time NVDA was like this", ResearchKind.ANALOGUE),
    ])
    def test_kind(self, text: str, kind: ResearchKind) -> None:
        req = detect(text)
        assert req is not None and req.kind is kind


def _skills(rsi: float, hist: float) -> Any:
    def calls(calls: Any, timeout: int = 10) -> list[tuple[Any, str]]:
        payloads = {
            "rsi": {"rsi": rsi, "period": 14, "timeframe": "4h"},
            "macd": {"macd": 1.0, "signal": 1.0 - hist, "histogram": hist},
            "support_resistance": {"current_price": 100.0, "resistances": [100.5],
                                   "supports": [95.0]},
            "atr": {"atr": 2.5, "timeframe": "4h"},
        }
        return [(payloads[args["action"]], "ok") for _, args in calls]
    return calls


class TestTechnicals:
    def test_the_figures_are_the_skills_and_named_as_such(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(research, "_skill_calls", _skills(rsi=75.0, hist=0.5))
        answer = run("q", ResearchRequest(kind=ResearchKind.TECHNICALS, symbols=("NVDAUSDT",)))
        assert answer.lines[0].startswith("Actionable:") and "overbought" in answer.lines[0]
        assert "within 1% of resistance" in answer.lines[0]
        assert any(line.startswith("Caveat:") for line in answer.lines)
        assert "bitget-signal" in answer.lines[-1] and "24h ticker" not in answer.lines[-1]

    def test_a_neutral_tape_says_neutral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(research, "_skill_calls", _skills(rsi=50.0, hist=0.0))
        lines, _ = research._technicals("NVDAUSDT")
        assert "overbought" not in lines[0] and "oversold" not in lines[0]


class _FakeService:
    def __init__(self, *, report: str, scraped: str, holders: int) -> None:
        self.report, self.scraped, self.holders = report, scraped, holders

    def next_earnings(self, ticker: str) -> dict[str, Any]:
        return {"report_date": self.report, "period_ending": "2026-10-31"}

    def consensus(self, ticker: str) -> dict[str, Any]:
        return {"fore_mean": 4.2, "scraped_date": self.scraped, "fore_indicator_name": "EPS",
                "fore_org_num": 30, "min_fore_value": 3.9, "high_fore_value": 4.6}

    def institutional_holdings(self, ticker: str) -> list[dict[str, Any]]:
        return [{"org_name": f"Fund {i}", "principal_amount": 1000 * (i + 1),
                 "period_ending": "2026-06-30"} for i in range(self.holders)]

    def quote(self, ticker: str) -> dict[str, Any]:
        return {"total_market_cap": 3.5e12, "pb": 40.0}


def _service(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    monkeypatch.setattr(bitget_mcp, "BitgetDataService", lambda: _FakeService(**kwargs))


def _days_from_today(n: int) -> str:
    from datetime import UTC, datetime, timedelta
    return (datetime.now(UTC).date() + timedelta(days=n)).isoformat()


class TestFundamentals:
    def test_a_stale_consensus_is_withheld_not_quoted(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        _service(monkeypatch, report=_days_from_today(30), scraped="2018-10-25", holders=3)
        lines, _ = research._fundamentals("MSFTUSDT")
        text = " ".join(lines)
        assert "withheld" in text and "4.2" not in text

    def test_a_fresh_consensus_is_quoted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _service(monkeypatch, report=_days_from_today(30), scraped=_days_from_today(-10),
                 holders=3)
        lines, _ = research._fundamentals("MSFTUSDT")
        assert any("4.2" in line and "30 analysts" in line for line in lines)

    def test_a_near_report_leads_the_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _service(monkeypatch, report=_days_from_today(3), scraped="2018-10-25", holders=3)
        lines, _ = research._fundamentals("NVDAUSDT")
        assert lines[0].startswith("Actionable: NVDA reports in 3 day(s)")

    def test_a_past_report_gives_an_estimate_labelled_as_one(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        _service(monkeypatch, report=_days_from_today(-40), scraped="2018-10-25", holders=3)
        lines, _ = research._fundamentals("NVDAUSDT")
        assert lines[0].startswith("Actionable:") and "an estimate, not a date" in lines[0]

    def test_one_13f_record_is_not_dressed_up_as_a_table(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        _service(monkeypatch, report=_days_from_today(30), scraped="2018-10-25", holders=1)
        lines, _ = research._fundamentals("MSFTUSDT")
        line = next(x for x in lines if x.startswith("Institutional"))
        assert "one 13F record" in line and "not a full ownership table" in line

    def test_the_data_line_names_the_mcp_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _service(monkeypatch, report=_days_from_today(30), scraped="2018-10-25", holders=3)
        answer = run("q", ResearchRequest(kind=ResearchKind.FUNDAMENTALS, symbols=("MSFTUSDT",)))
        assert "bitget-mcp-server" in answer.lines[-1]


class TestAnalogues:
    def test_a_failed_live_fetch_is_answered_frozen_and_says_so(self) -> None:
        answer = run("q", ResearchRequest(kind=ResearchKind.ANALOGUE, symbols=("COINUSDT",)))
        assert not answer.refused
        assert answer.lines[0].startswith("Actionable:")
        assert "frozen" in answer.lines[-1]

    def test_a_thin_sample_is_called_thin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(research, "MIN_INDEPENDENT_EPISODES", 10**6)
        answer = run("q", ResearchRequest(kind=ResearchKind.ANALOGUE, symbols=("COINUSDT",)))
        assert answer.lines[0].startswith("Actionable: treat this as thin")
        assert not any(line.startswith("Spread of outcomes") for line in answer.lines)

    @staticmethod
    def _live(monkeypatch: pytest.MonkeyPatch, *, gap_hours: int) -> research.MarketData:
        """Live bars start where the frozen file's last 200 bars begin, plus ``gap_hours``."""
        import json
        from datetime import datetime, timedelta
        from decimal import Decimal

        fixture = json.loads(research.FIXTURE_PATH.read_text(encoding="utf-8"))
        rows = fixture["candles"]["COINUSDT"]
        start = datetime.fromisoformat(rows[-200][0]) + timedelta(hours=gap_hours)
        bars = [history.Candle(ts=start + timedelta(hours=i), open=Decimal(100),
                               high=Decimal(101), low=Decimal(99), close=Decimal(100 + i % 3),
                               volume=Decimal(1)) for i in range(300)]
        monkeypatch.setattr(history, "fetch", lambda *a, **k: bars)
        return research._analogue_data("COINUSDT")

    def test_live_bars_extend_backwards_with_contiguous_closed_bars(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = self._live(monkeypatch, gap_hours=0)
        assert data.live and "extended back" in data.provenance
        assert len(data.raw["COINUSDT"]) > 300

    def test_a_gap_drops_the_extension_rather_than_bridging_it(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = self._live(monkeypatch, gap_hours=500)
        assert data.live and "extended" not in data.provenance
        assert len(data.raw["COINUSDT"]) < 300


class TestEveryBitgetContract:
    """Research reaches every contract Bitget lists, not only the twelve the desk trades."""

    @pytest.mark.parametrize(("text", "kind", "symbol"), [
        ("where is gold trading", ResearchKind.QUOTE, "XAUUSDT"),
        ("is PLTR overbought", ResearchKind.TECHNICALS, "PLTRUSDT"),
        ("has oil been here before", ResearchKind.ANALOGUE, "CLUSDT"),
        ("when does NFLX report earnings", ResearchKind.FUNDAMENTALS, "NFLXUSDT"),
        ("is CVX overbought", ResearchKind.TECHNICALS, "CVXSTOCKUSDT"),
    ])
    def test_listed_names_resolve(self, text: str, kind: ResearchKind, symbol: str) -> None:
        req = detect(text)
        assert req is not None and req.kind is kind and req.symbols[0] == symbol

    def test_a_reading_that_is_not_literal_is_said_out_loud(self) -> None:
        req = detect("is CVX overbought")
        assert req is not None
        assert any("CVXUSDT is a different, crypto contract" in n for n in req.notes)

    @pytest.mark.parametrize("text", [
        "what should I buy now", "how is the US market", "tell me about ME",
    ])
    def test_prose_is_not_read_as_a_ticker(self, text: str) -> None:
        """Bitget lists tokens called NOW, US and ME; none of these is a question about them."""
        assert detect(text) is None

    def test_the_market_being_shocked_is_not_a_holding(self) -> None:
        req = detect("what happens to my portfolio if the nasdaq drops 10%")
        assert req is not None and req.kind is ResearchKind.STRESS and not req.book

    def test_a_weighted_index_product_is_a_holding(self) -> None:
        req = detect("I hold 30% SP500 and 70% NVDA, what if the market drops 10%")
        assert req is not None and dict(req.book)["SP500USDT"] == pytest.approx(0.3)

    def test_spx_is_the_index_not_the_memecoin(self) -> None:
        req = detect("where is the SPX trading")
        assert req is not None and req.symbols == ("SP500USDT",)
        assert any("memecoin" in n for n in req.notes)

    def test_a_commodity_has_no_earnings_and_says_so(self) -> None:
        lines, _ = research._fundamentals("CLUSDT")
        assert lines[0].startswith("Actionable: CL is a commodity")

    def test_crypto_has_no_filings_and_says_so(self) -> None:
        lines, _ = research._fundamentals("BTCUSDT")
        assert "crypto contract" in lines[0]

    def test_the_session_line_fits_what_the_contract_tracks(self) -> None:
        assert "US anchor" in research._session_line("NVDAUSDT", anchor_open=False,
                                                     us_listed=True)
        assert "around the clock" in research._session_line("BTCUSDT", anchor_open=False,
                                                            us_listed=False)
        gold = research._session_line("XAUUSDT", anchor_open=False, us_listed=True)
        assert "US anchor" not in gold and "own trading hours" in gold

    def test_a_ticker_collision_is_not_reported_as_a_premium(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CL is WTI crude on Bitget and Colgate-Palmolive on the NYSE."""
        class Colgate:
            def quote(self, ticker: str) -> dict[str, Any]:
                return {"last_price": 80.0}
        monkeypatch.setattr(bitget_mcp, "BitgetDataService", Colgate)
        from decimal import Decimal
        assert research._premium_line("CLUSDT", Decimal("91.08"), False) is None
        assert research._premium_line("PLTRUSDT", Decimal("186.9"), False) is None


class TestTheSkillsMacdIsCheckedNotTrusted:
    """bitget-signal returns MACD's signal line and histogram in each other's fields (measured on
    BTC, NVDA, TSLA and META, 2026-09-23). Arithmetic cannot see it — DIF minus either field is the
    other — so the reading is checked against a recomputation from Bitget's candles."""

    BTC: ClassVar[dict[str, Any]] = {"macd": 1568.0, "signal": -148.3, "histogram": 1716.3,
                                     "cross": "golden_cross"}

    def test_the_swap_is_seen_against_the_recomputed_signal_line(self) -> None:
        checked = skills.macd_fields(self.BTC, recomputed_signal=1715.0)
        assert checked is not None
        line, hist, swapped = checked
        assert swapped and line == pytest.approx(1716.3) and hist == pytest.approx(-148.3)

    def test_a_correct_reading_is_read_straight(self) -> None:
        fixed = {"macd": 1568.0, "signal": 1716.3, "histogram": -148.3}
        checked = skills.macd_fields(fixed, recomputed_signal=1715.0)
        assert checked is not None and not checked[2]

    def test_an_unchecked_signal_line_is_not_quoted(self) -> None:
        assert skills.macd_fields(self.BTC, recomputed_signal=None) is None

    def test_the_answer_drops_the_skills_cross_and_says_why(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def skill_calls(calls: Any, timeout: int = 10) -> list[tuple[Any, str]]:
            return [({"rsi": 65.0, "period": 14, "timeframe": "4h"} if a["action"] == "rsi"
                     else self.BTC if a["action"] == "macd" else {}, "ok") for _, a in calls]
        monkeypatch.setattr(research, "_skill_calls", skill_calls)
        monkeypatch.setattr(skills, "indicators", lambda s: {
            "dif": 1568.0, "dea": 1715.0, "histogram": -147.0, "cross": "", "atr": 900.0,
            "close": 85000.0, "bars": 300.0, "since": "2026-08-04"})
        lines, _ = research._technicals("BTCUSDT")
        assert "momentum turning down" in lines[0]
        assert not any("golden cross" in line for line in lines)
        assert any(line.startswith("Corrected:") for line in lines)


class TestComputedTechnicals:
    def _bars(self, n: int) -> list[Any]:
        from datetime import datetime, timedelta
        from decimal import Decimal
        start = datetime(2026, 8, 1)
        return [history.Candle(ts=start + timedelta(hours=4 * i), open=Decimal(100 + i),
                               high=Decimal(102 + i), low=Decimal(99 + i),
                               close=Decimal(101 + i), volume=Decimal(1)) for i in range(n)]

    def test_a_steady_uptrend_reads_overbought_and_rising(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(history, "fetch", lambda *a, **k: self._bars(200))
        lines, sources = research._technicals_computed("SP500USDT")
        assert lines[0].startswith("Actionable: RSI is overbought")
        assert sources[0].kind == "computation"
        assert not any(line.startswith("Short history") for line in lines)

    def test_a_young_listing_is_read_with_its_history_stated(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(history, "fetch", lambda *a, **k: self._bars(49))
        lines, _ = research._technicals_computed("CVXSTOCKUSDT")
        assert any(line.startswith("Short history: CVX has 49") for line in lines)

    def test_too_little_history_is_not_computed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(history, "fetch", lambda *a, **k: self._bars(20))
        assert research._technicals_computed("CVXSTOCKUSDT") == ([], [])


class TestTheDeskReadsACorrectedMacd:
    """The desk hands Skill payloads to the model as evidence; decision 586 cited the Skill's false
    "golden cross". The payload is corrected before it becomes evidence."""

    STATE: ClassVar[dict[str, Any]] = {"dif": 1568.0, "dea": 1715.0, "histogram": -147.0,
                                       "cross": "", "atr": 900.0, "close": 85000.0,
                                       "bars": 300.0, "since": "2026-08-04"}

    def test_a_swapped_payload_is_restored_and_its_cross_replaced(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(skills, "indicators", lambda s: self.STATE)
        got = skills.correct_macd({"symbol": "BTCUSDT", "macd": 1568.0, "signal": -148.3,
                                   "histogram": 1716.3, "cross": "golden_cross"})
        assert got["signal"] == pytest.approx(1716.3) and got["histogram"] == pytest.approx(-148.3)
        assert got["cross"] != "golden_cross" and "corrected" in got

    def test_an_unverifiable_signal_is_removed_not_passed_on(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(skills, "indicators", lambda s: None)
        got = skills.correct_macd({"symbol": "BTCUSDT", "macd": 1568.0, "signal": -148.3,
                                   "histogram": 1716.3, "cross": "golden_cross"})
        assert "cross" not in got and "signal" not in got and "unverified" in got


class TestTheEarningsSurpriseIsShownWhenFiled:
    def test_the_surprise_line_is_part_of_an_earnings_answer(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        _service(monkeypatch, report=_days_from_today(30), scraped="2018-10-25", holders=3)
        line = ("Earnings surprise: the quarter ending 2026-07-26 was a large beat — SUE +2.84")
        monkeypatch.setattr(research, "_earnings_surprise", lambda ticker: (
            line, research.Source(kind="computation", ref="argus.research.sue via SEC XBRL")))
        lines, sources = research._fundamentals("NVDAUSDT")
        assert line in lines
        assert any(s.ref == "argus.research.sue via SEC XBRL" for s in sources)

    def test_a_commodity_never_asks_the_sec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(ticker: str) -> None:
            raise AssertionError("SEC queried for a commodity")
        monkeypatch.setattr(research, "_earnings_surprise", boom)
        lines, _ = research._fundamentals("CLUSDT")
        assert "commodity" in lines[0]


class TestTheModelsStressBook:
    def test_names_without_weights_are_an_equal_weight_book(self) -> None:
        """Held-out B: "…propagate through my QQQ and META weights" lost its book when stress
        plans began keeping only weighted holdings."""
        req, _ = plan_with_model(
            "Apply a -400bps Nasdaq composite shock, propagate through my QQQ and META weights",
            _FakeModel({"kind": "stress", "holdings": {}, "shock_percent": -4,
                        "confidence": 0.9}))
        assert req is not None and dict(req.book) == {"QQQUSDT": 0.5, "METAUSDT": 0.5}

    def test_the_shocked_index_is_still_not_a_holding(self) -> None:
        req, _ = plan_with_model("what if the nasdaq drops 10%, I hold gold", _FakeModel(
            {"kind": "stress", "holdings": {}, "shock_percent": -10, "confidence": 0.9}))
        assert req is not None and "NDX100USDT" not in req.book and "XAUUSDT" in req.book


class TestASingleNamesProfile:
    def test_moments_match_the_adjusted_estimators(self) -> None:
        """Checked against scipy.stats.skew/kurtosis(bias=False) on live MSTR returns
        (2026-09-23): identical to 15 significant figures."""
        got = research._moments([0.0, 0.0, 0.0, 1.0, -1.0, 3.0])
        assert got is not None
        assert got[0] == pytest.approx(1.3745866, abs=1e-6)
        assert got[1] == pytest.approx(2.3545706, abs=1e-6)

    def test_a_standalone_question_leads_with_a_sizing_rule_and_shows_the_shape(self) -> None:
        answer = run("q", ResearchRequest(kind=ResearchKind.IMPACT, symbols=("MSTRUSDT",)))
        assert answer.lines[0].startswith("Actionable: size MSTR")
        assert any(line.startswith("Return shape") and "kurtosis" in line and "R²" in line
                   for line in answer.lines)


class TestTheFourthCorpusFixes:
    def test_a_book_of_only_the_name_asked_about_is_a_single_name_question(self) -> None:
        req, _ = plan_with_model("is intel a scary stock to hold", _FakeModel(
            {"kind": "impact", "candidate": "INTC", "holdings": {"INTC": 100},
             "confidence": 0.95}))
        assert req is not None and not req.book and req.symbols == ("INTCUSDT",)

    def test_a_crypto_price_target_is_refused_on_both_paths(self) -> None:
        text = "whats the price target for eth in 2027 gonna moon or what"
        req, audit = plan_with_model(text, _FakeModel(
            {"kind": "fundamentals", "candidate": "ETH", "confidence": 0.9}))
        assert req is None and "forecast" in audit["detail"]
        assert detect(text) is None

    def test_an_analyst_price_target_on_a_stock_is_still_fundamentals(self) -> None:
        req = detect("what is the analyst price target for NVDA")
        assert req is not None and req.kind is ResearchKind.FUNDAMENTALS

    def test_the_model_is_asked_about_names_the_registry_cannot_spell(self) -> None:
        assert research.worth_asking_the_model("is intel a scary stock to hold")
        assert research.worth_asking_the_model("aaple ka abhi price kya chal raha hai")
        assert not research.worth_asking_the_model("why did you pass on NVDA")


class TestTheModelsNamesList:
    def test_a_compare_reaches_names_only_the_model_can_spell(self) -> None:
        req, _ = plan_with_model("comparame oro y bitcoin", _FakeModel(
            {"kind": "compare", "candidate": "XAU", "names": ["XAU", "BTC"], "confidence": 0.9}))
        assert req is not None and req.kind is ResearchKind.COMPARE
        assert req.symbols[:2] == ("XAUUSDT", "BTCUSDT")

    def test_a_names_list_that_is_not_a_list_is_ignored(self) -> None:
        req, _ = plan_with_model("compare AMD and NVDA", _FakeModel(
            {"kind": "compare", "names": "AMD, NVDA", "confidence": 0.9}))
        assert req is not None and set(req.symbols) >= {"AMDUSDT", "NVDAUSDT"}


class TestAnalystPriceTargets:
    ROWS: ClassVar[list[dict[str, Any]]] = [
        {"published_date": "2026-09-09", "analyst_firm": "A", "price_target": 300.0,
         "price_target_previous": None, "latest_rating_cn": "增持"},
        {"published_date": "2026-09-03", "analyst_firm": "B", "price_target": 390.0,
         "price_target_previous": 350.0, "latest_rating_cn": "买入"},
        {"published_date": "2026-08-01", "analyst_firm": "B", "price_target": 350.0,
         "price_target_previous": 300.0, "latest_rating_cn": "买入"},
        {"published_date": "2026-08-20", "analyst_firm": "C", "price_target": 200.0,
         "price_target_previous": 250.0, "latest_rating_cn": "持有"},
        {"published_date": "2019-01-01", "analyst_firm": "D", "price_target": 40.0,
         "price_target_previous": None, "latest_rating_cn": "卖出"},
    ]

    def test_only_each_firms_latest_view_inside_the_window_counts(self) -> None:
        from datetime import date
        line = research._price_target_line("NVDA", self.ROWS, {"last_price": 250.0},
                                           date(2026, 9, 23))
        assert line is not None
        assert "3 firms" in line and "median 300" in line and "range 200 to 390" in line
        assert "+20% from the stock's 250" in line
        assert "2 buy, 1 hold, 0 sell" in line and "1 raised and 1 cut" in line

    def test_nothing_recent_is_nothing_said(self) -> None:
        from datetime import date
        assert research._price_target_line("NVDA", self.ROWS[-1:], {}, date(2026, 9, 23)) is None


class TestTheBookAsHeld:
    """"How risky is my portfolio" is about the book held, not a proposal to add to it."""

    @pytest.mark.parametrize("text", [
        "I hold 60% BTC, 30% ETH, 10% SOL. How risky is my portfolio?",
        "review my portfolio: 40% TSLA 30% COIN 30% MSTR",
        "I hold 50% NVDA, 30% MSFT, 20% AAPL - should I rebalance?",
    ])
    def test_a_book_question_is_a_book_request(self, text: str) -> None:
        req = detect(text)
        assert req is not None and req.kind is ResearchKind.BOOK and req.book

    def test_adding_a_name_is_still_impact(self) -> None:
        req = detect("I hold 50% NVDA, 50% AAPL - how risky is my portfolio if I add 20% TSLA")
        assert req is not None and req.kind is ResearchKind.IMPACT

    def test_a_small_book_gets_an_equal_risk_rebalance_not_a_cash_pile(self) -> None:
        answer = run("q", ResearchRequest(kind=ResearchKind.BOOK,
                                          symbols=("NVDAUSDT", "MSFTUSDT", "AAPLUSDT"),
                                          book={"NVDAUSDT": 0.5, "MSFTUSDT": 0.3,
                                                "AAPLUSDT": 0.2}))
        assert answer.lines[0].startswith("Actionable: the risk is concentrated in NVDA")
        assert "equal-risk rebalance" in answer.lines[0] and "fully invested" in answer.lines[0]
        assert any(line.startswith("Where the risk sits") for line in answer.lines)

    def test_a_stated_budget_is_enforced_by_trimming(self) -> None:
        answer = run("q", ResearchRequest(kind=ResearchKind.BOOK,
                                          symbols=("NVDAUSDT", "MSFTUSDT", "AAPLUSDT"),
                                          book={"NVDAUSDT": 0.5, "MSFTUSDT": 0.3,
                                                "AAPLUSDT": 0.2},
                                          budget=0.4, budget_stated=True))
        assert "over your 40% budget" in answer.lines[0]

    def test_equal_risk_weights_equalise_risk(self) -> None:
        data = research.load(("NVDAUSDT", "MSFTUSDT", "AAPLUSDT"))
        columns = research._open_columns(data.raw, research._is_open())
        weights = research._equal_risk_weights(("NVDAUSDT", "MSFTUSDT", "AAPLUSDT"), columns)
        assert weights is not None and sum(weights.values()) == pytest.approx(1.0)
        risk = research.decompose(weights, columns)
        assert risk is not None
        for c in risk.contributions:
            assert c.contribution / risk.volatility == pytest.approx(1 / 3, abs=0.01)

    def test_units_and_cash_are_valued_not_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(research, "_last_price", lambda s: 80_000.0 if s == "BTCUSDT"
                            else None)
        notes: list[str] = []
        book, cash = research._value_holdings(
            {"holdings_usd": {"TSLA": 5000}, "holdings_units": {"BTC": 2}, "cash_usd": 10000},
            notes)
        total = 5000 + 160_000 + 10_000
        assert book["BTCUSDT"] == pytest.approx(160_000 / total)
        assert book["TSLAUSDT"] == pytest.approx(5000 / total)
        assert cash == pytest.approx(10_000 / total)
        assert any("2 BTC valued at $160,000" in n for n in notes)

    def test_a_qqq_hedge_that_explains_little_says_so(self) -> None:
        line = research._hedge_line(0.95, 0.16)
        assert line is not None and "would remove little" in line
        assert "neutralises" in (research._hedge_line(1.1, 0.56) or "")


def _ticker(symbol: str, last: str = "100", change: str = "0.01", funding: str = "0.0001") -> Any:
    from datetime import UTC, datetime
    from decimal import Decimal

    from argus.market.bitget import Ticker
    return Ticker(symbol=symbol, last=Decimal(last), bid=Decimal(last), ask=Decimal(last),
                  high_24h=Decimal(last), low_24h=Decimal(last), change_24h=Decimal(change),
                  base_volume=Decimal("1000000"), funding_rate=Decimal(funding),
                  fetched_at=datetime.now(UTC))


class TestTheJudgesQuestionsRouteToTheRightEngine:
    """Every question here was answered wrongly or refused on the live console in a judge's probe
    on 2026-09-24."""

    @pytest.mark.parametrize(("text", "kind"), [
        ("Whats the news on NVDA today?", ResearchKind.NEWS),
        ("why is COIN down today", ResearchKind.NEWS),
        ("Why did the market drop today?", ResearchKind.NEWS),
        ("What is the macro backdrop - Fed, yields, dollar, and how does it affect tech stocks?",
         ResearchKind.MACRO),
        ("What will the S&P 500 do after the next FOMC meeting?", ResearchKind.MACRO),
        ("What is the fear and greed index right now?", ResearchKind.SENTIMENT),
        ("Explain the risk in shorting DOGE with 20x leverage", ResearchKind.LEVERAGE),
        ("Build me a portfolio of tech stocks", ResearchKind.CONSTRUCT),
        ("build a portfolio with NVDA, gold and BTC", ResearchKind.CONSTRUCT),
        ("I hold 60% BTC, 30% ETH, 10% SOL. How risky is my portfolio?", ResearchKind.BOOK),
        ("Compare AAPL and MSFT fundamentals", ResearchKind.FUNDAMENTALS),
    ])
    def test_route(self, text: str, kind: ResearchKind) -> None:
        req = detect(text)
        assert req is not None and req.kind is kind

    def test_why_did_a_price_move_is_news_but_why_did_you_is_the_record(self) -> None:
        assert not research.about_the_record("why did NVDA fall today")
        assert research.about_the_record("why did you pass on NVDA")
        assert research.about_the_record("why did the desk skip TSLA yesterday")

    def test_two_names_get_fundamentals_for_both(self) -> None:
        req = detect("Compare AAPL and MSFT fundamentals")
        assert req is not None and req.symbols == ("AAPLUSDT", "MSFTUSDT")

    def test_leverage_and_side_are_read(self) -> None:
        req = detect("Explain the risk in shorting DOGE with 20x leverage")
        assert req is not None and req.leverage == 20.0 and req.side == "short"

    def test_a_forecast_is_answered_as_a_backdrop_and_says_so(self) -> None:
        req = detect("What will the S&P 500 do after the next FOMC meeting?")
        assert req is not None and any("does not forecast" in n for n in req.notes)

    def test_the_market_being_shocked_by_a_named_theme_is_not_a_construct(self) -> None:
        req = detect("How should I split a $50k order in NVDA?")
        assert req is not None and req.kind is ResearchKind.EXECUTION


class TestLeverageArithmetic:
    def test_liquidation_distance_and_survivable_multiple(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import datetime, timedelta
        from decimal import Decimal
        start = datetime(2026, 9, 1)
        # A flat series with one 10% spike: a 20x short (5% distance) is liquidated in the windows
        # that contain the spike; the worst adverse 24h move is 10%, so 10x is the survivable cap.
        bars = []
        for i in range(200):
            high = Decimal("110") if i == 100 else Decimal("100")
            bars.append(history.Candle(ts=start + timedelta(hours=i), open=Decimal(100),
                                       high=high, low=Decimal(100), close=Decimal(100),
                                       volume=Decimal(1)))
        monkeypatch.setattr(history, "fetch", lambda *a, **k: bars)
        from argus.market import bitget
        monkeypatch.setattr(bitget, "fetch_tickers", lambda: {"DOGEUSDT": _ticker("DOGEUSDT")})
        monkeypatch.setattr(bitget, "maintenance_margin_rate", lambda symbol, notional: None)
        lines, _, payload = research._leverage("DOGEUSDT", 20.0, "short")
        assert payload["liquidation_distance"] == pytest.approx(0.05)
        assert payload["worst_adverse_24h"] == pytest.approx(0.10)
        assert payload["survivable_leverage"] == 10
        assert "only 10x or less would have survived" in lines[0]
        assert any("shorts" in line or "short receives" in line for line in lines)
        # With Bitget's maintenance margin the line moves closer and the survivor count falls:
        # 1/20 - 1% is 4%, and only 1/(10% + 1%) = 9x clears the 10% worst day.
        monkeypatch.setattr(bitget, "maintenance_margin_rate", lambda symbol, notional: 0.01)
        lines, _, payload = research._leverage("DOGEUSDT", 20.0, "short")
        assert payload["liquidation_distance"] == pytest.approx(0.04)
        assert payload["survivable_leverage"] == 9
        assert any("1.00% maintenance margin" in line for line in lines)


class TestFundingMeaning:
    def test_a_positive_rate_is_longs_paying_shorts_on_the_contracts_own_interval(self) -> None:
        line = research._funding_meaning("XAUUSDT", 0.01)
        assert line is not None and "longs pay shorts every 4h" in line
        assert "0.060% of the position a day" in line

    def test_a_flat_rate_is_flat(self) -> None:
        assert "flat" in (research._funding_meaning("BTCUSDT", 0.0) or "")


class TestExecutionReadsTheBook:
    def test_a_size_beyond_the_visible_book_is_a_lower_bound(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        from decimal import Decimal

        from argus.desk.workbench import plan_execution
        from argus.market import depth
        levels = tuple(depth.Level(Decimal(100.01 + i * 0.01), Decimal(10)) for i in range(50))
        bids = tuple(depth.Level(Decimal(100 - i * 0.01), Decimal(10)) for i in range(50))
        book = depth.OrderBook(symbol="PLTRUSDT", fetched_at=datetime.now(UTC), bids=bids,
                               asks=levels)
        monkeypatch.setattr(depth, "fetch_orderbook", lambda *a, **k: book)
        plan = plan_execution(symbol="PLTRUSDT", notional=Decimal("2000000"),
                              adv_notional=Decimal("6000000"))
        lines = research._depth_lines("PLTRUSDT", Decimal("2000000"), Decimal("6000000"), plan,
                                      "sell $2m of PLTR")
        assert lines[0].startswith("Actionable: in one market order this costs at least")
        assert any("sell side" in line for line in lines)
        assert any("larger than the visible book" in line for line in lines)
        assert any(line.startswith("Schedule:") and "days" in line for line in lines)


class TestMacroAndSentiment:
    def test_macro_reads_fred_and_states_the_curve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        values = {"DGS10": 4.9, "DGS2": 4.7, "DFF": 3.9, "T10YIE": 2.3, "DTWEXBGS": 119.0}
        monkeypatch.setattr(research, "_fred", lambda sid, days=45: [
            ("2026-08-10", values[sid] - 0.1), ("2026-09-22", values[sid])])
        monkeypatch.setattr(research, "_rate_sensitivity", lambda s: {
            "days": 60, "corr_10y": -0.4, "pct_per_10bp": -1.1, "corr_dollar": -0.3})
        from argus.market import evidence
        monkeypatch.setattr(evidence.RssSource, "headlines", lambda self, k, u: [])
        lines, _, readings = research._macro(None)
        joined = " ".join(lines)
        assert lines[0].startswith("Actionable: the 10-year is 4.90%")
        assert "rates have been driving it" in lines[0]
        assert "10-year minus 2-year is +20bp" in joined
        assert readings["DGS10"]["value"] == pytest.approx(4.9)

    def test_sentiment_reads_the_index_and_positioning(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io
        import json as _json
        import urllib.request
        body = _json.dumps({"data": [{"value": str(v), "value_classification": "Greed"}
                                     for v in (74, 70, 65, 60, 55, 52, 50, 48)]}).encode()
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
        from argus.market import bitget
        monkeypatch.setattr(bitget, "fetch_tickers", lambda: {
            "BTCUSDT": _ticker("BTCUSDT", funding="0.0005")})
        lines, _, payload = research._sentiment()
        assert payload["index"] == 74 and "greed" in lines[0]
        assert any("crowded long" in line for line in lines)


class TestTheRecordAnswersItsOwnQuestions:
    def test_a_track_record_question_is_answered_first(self) -> None:
        from datetime import UTC, datetime

        from argus.lui.question import Intent, classify
        assert classify("What is your track record?",
                        now=datetime.now(UTC)).intent is Intent.PERFORMANCE

    def test_what_we_got_wrong_reaches_the_review(self) -> None:
        from datetime import UTC, datetime

        from argus.lui.question import Intent, classify
        assert classify("What did you get wrong recently?",
                        now=datetime.now(UTC)).intent is Intent.REVIEW


class TestSecondRoundOfJudgeQuestions:
    @pytest.mark.parametrize(("text", "kind"), [
        ("What is Bitget's tokenized stock offering and should I use it instead of buying stocks "
         "directly?", ResearchKind.VENUE),
        ("I'm a conservative investor with $20k. Build me a portfolio", ResearchKind.CONSTRUCT),
        ("What is the outlook for gold over the next month?", ResearchKind.ANALOGUE),
        ("英伟达现在值得买吗?", ResearchKind.IMPACT),
        ("what is the analyst price target for NVDA", ResearchKind.FUNDAMENTALS),
    ])
    def test_route(self, text: str, kind: ResearchKind) -> None:
        req = detect(text)
        assert req is not None and req.kind is kind

    def test_a_price_forecast_is_not_a_research_request(self) -> None:
        assert detect("yo where will tsla be by friday closing lol give me a number") is None
        assert research._PRICE_FORECAST.search("Forecast BTC price for next week")

    def test_a_conservative_book_is_the_defensive_theme(self) -> None:
        req = detect("I'm a conservative investor with $20k. Build me a portfolio")
        assert req is not None and "SPYUSDT" in req.symbols and "XAUUSDT" in req.symbols

    def test_the_line_that_answers_the_question_leads(self) -> None:
        lines = ["Actionable: NVDA's next report date is not published yet.",
                 "Institutional holders: the source returned one 13F record.",
                 "Analyst price targets, last 90 days (21 firms): median 330."]
        led = research._lead_with_what_was_asked(lines, "what are analysts' price targets")
        assert led[0].startswith("Actionable: analyst price targets")
        led = research._lead_with_what_was_asked(lines, "which 13F funds own NVDA")
        assert led[0].startswith("Actionable: institutional holders")
        assert research._lead_with_what_was_asked(lines, "when does NVDA report") == lines

    def test_fred_falls_back_to_the_dated_snapshot(self, monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Any) -> None:
        import json as _json

        def unreachable(series: str, days: int) -> list[tuple[str, float]]:
            raise TimeoutError("FRED did not answer")
        snap = tmp_path / "macro_snapshot.json"
        snap.write_text(_json.dumps({"generated_at": "2026-09-24T00:00:00+00:00",
                                     "series": {"DGS10": [["2026-09-22", 4.96]]}}),
                        encoding="utf-8")
        monkeypatch.setattr(research, "_fred_live", unreachable)
        monkeypatch.setattr(research, "FRED_SNAPSHOT", snap)
        rows = research._fred("DGS10", days=10_000)
        assert rows == [("2026-09-22", 4.96)]
        assert research._FRED_USED_SNAPSHOT["DGS10"] == "2026-09-24"
        research._FRED_USED_SNAPSHOT.clear()

    def test_the_track_record_pattern_cannot_be_relabelled(self) -> None:
        from datetime import UTC, datetime

        from argus.lui.ngram import reclassify
        from argus.lui.question import Intent, classify
        q = classify("What is your track record? How many trades have you made?",
                     now=datetime.now(UTC))
        assert reclassify(q)[0].intent is Intent.PERFORMANCE


class TestOrdersStayRefusedWhateverTheyName:
    @pytest.mark.parametrize("text", ["Buy 0.5 BTC at market",
                                      "Place a limit order to sell 10 ETH at 5000"])
    def test_an_order_for_a_listed_name_is_refused(self, text: str) -> None:
        """A name-only fallback answered these as a risk profile on 2026-09-24; an order is
        refused as an order before any fallback."""
        from argus.lui import server
        payload = server.handle_ask(text, [])
        assert payload["refused"] and payload["intent"] == "order"


class TestFredBacksOff:
    def test_after_one_failure_the_snapshot_is_used_without_waiting(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        import json as _json
        calls: list[str] = []

        def slow_failure(series: str, days: int) -> list[tuple[str, float]]:
            calls.append(series)
            raise TimeoutError("FRED did not answer")
        snap = tmp_path / "macro_snapshot.json"
        snap.write_text(_json.dumps({"generated_at": "2026-09-24T00:00:00+00:00",
                                     "series": {"DGS10": [["2026-09-22", 4.96]],
                                                "DGS2": [["2026-09-22", 4.71]]}}),
                        encoding="utf-8")
        monkeypatch.setattr(research, "_fred_live", slow_failure)
        monkeypatch.setattr(research, "FRED_SNAPSHOT", snap)
        monkeypatch.setattr(research, "_FRED_DOWN_UNTIL", 0.0)
        research._fred("DGS10", days=10_000)
        research._fred("DGS2", days=10_000)
        assert calls == ["DGS10"]
        research._FRED_USED_SNAPSHOT.clear()
