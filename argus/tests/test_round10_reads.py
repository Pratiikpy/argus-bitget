"""Round 10 of the §27 audits (2026-09-30): a hostile review, a first-time user and a judge.

Each test pins one finding, named in the docstring of the code it covers. Everything here runs
offline: live sources are replaced by the rows they return.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pytest

# --- the product explains itself --------------------------------------------------------------


@pytest.mark.parametrize("asked", [
    "what is this site and who is it for", "what is ARGUS", "what can you do", "help",
    "who is this for?", "what is this and what can it do",
])
def test_a_stranger_asking_what_this_is_gets_the_intro(asked: str) -> None:
    from argus.lui.intro import INTRO_Q

    assert INTRO_Q.search(asked)


@pytest.mark.parametrize("asked", ["what is it", "what is NVDA", "what can you do about my TSLA"])
def test_the_intro_does_not_claim_other_questions(asked: str) -> None:
    from argus.lui.intro import INTRO_Q

    assert not INTRO_Q.search(asked)


def test_the_intro_counts_are_the_registers_own() -> None:
    from argus.eval.standing import REGISTER, State
    from argus.lui.intro import answer

    lines, _, data = answer()
    assert data["total"] == len(REGISTER)
    assert f"of {len(REGISTER)}, {sum(c.state is State.OWNED for c in REGISTER)} win" in lines[3]
    assert "/record" not in " ".join(lines)  # a route the console does not serve


def test_first_money_asked_as_help_getting_started_is_the_starter() -> None:
    from argus.lui.research.starter import amount_of

    assert amount_of("I have $800 saved, can you help me start investing") == 800.0
    assert amount_of("I have 3 kids, help me get started") is None


# --- routing ----------------------------------------------------------------------------------


def test_a_date_question_is_not_a_rival_comparison() -> None:
    from argus.lui.rivals import asks_about_a_rival, rival_tokens

    asked = "What was your position on 30/09/2026 versus September 30 2026?"
    assert not {"2026", "september", "30"} & set(rival_tokens(asked))
    assert not asks_about_a_rival(asked)
    assert asks_about_a_rival("how does ARGUS compare to Nautilus Trader")


def test_explicit_technical_analysis_wording_keeps_the_pattern_reading() -> None:
    from argus.lui.research.fundamentals import pattern_reading_wins
    from argus.lui.research.kinds import ResearchKind, ResearchRequest

    request = ResearchRequest(kind=ResearchKind.TECHNICALS, symbols=("SOLUSDT",))
    assert pattern_reading_wins(request, "show me the bitget-signal technical analysis for SOL")
    sentiment = ResearchRequest(kind=ResearchKind.SENTIMENT, symbols=("BTCUSDT",))
    assert pattern_reading_wins(sentiment, "what does the bitget-signal sentiment skill say of BTC")


def test_skill_health_questions_still_reach_the_matrix() -> None:
    from argus.lui.server import _SKILL_HEALTH

    assert _SKILL_HEALTH.search("are the bitget skills working")
    assert _SKILL_HEALTH.search("how well does bitget-signal work for BTC")
    assert not _SKILL_HEALTH.search("show me the bitget-signal technical analysis for SOL")


@pytest.mark.parametrize("asked", [
    "what does the Agent Hub dry-run for a $2k BTC buy look like?",
    "show me the bgc order dry run to buy $2k of NVDA",
])
def test_an_agent_hub_dry_run_question_is_an_execution_question(asked: str) -> None:
    from argus.lui.research.dispatch import _ASKS_FOR_DRY_RUN
    from argus.lui.research.parse import _EXECUTION

    assert _EXECUTION.search(asked) and _ASKS_FOR_DRY_RUN.search(asked)


def test_the_next_report_date_leads_when_asked_when() -> None:
    from argus.lui.research import dispatch
    from argus.lui.watchlist import Report

    found = ["Bottom line: expect a bigger move than usual — TSLA has moved 3.2x.", "detail"]
    sources: list[Any] = []
    report = Report("TSLA", date(2026, 10, 21), "after the close", "Bitget equity calendar")
    import argus.lui.watchlist as watchlist

    original = watchlist.earnings_date
    watchlist.earnings_date = lambda ticker, today: report  # type: ignore[assignment]
    try:
        out = dispatch._next_report(found, "TSLAUSDT",
                                    "when does TSLA report and how big is the move usually",
                                    sources)
        same = dispatch._next_report(found, "TSLAUSDT", "how big is TSLA's earnings move", [])
    finally:
        watchlist.earnings_date = original  # type: ignore[assignment]
    assert out[0].startswith("Bottom line: TSLA next reports on Wed 21 Oct 2026, after the close")
    assert "On the move: expect a bigger move" in out[0]
    assert same == found


# --- the desk's record ------------------------------------------------------------------------


@pytest.mark.parametrize("asked", ["how has the desk performed", "what is your performance",
                                   "how have you done"])
def test_the_desk_performance_asked_plainly_is_the_track_record(asked: str) -> None:
    from argus.lui.question import Intent, classify

    assert classify(asked, now=datetime(2026, 9, 30, tzinfo=UTC)).intent is Intent.PERFORMANCE


def test_money_on_the_record_reads_as_dollars() -> None:
    from argus.lui.answer import _dollars

    assert _dollars(-1.7781) == "-$1.78"
    assert _dollars(10000, signed=False) == "$10,000"
    assert _dollars(12.4) == "+$12.40"


def test_the_desk_record_points_to_the_separate_agent_record() -> None:
    from argus.lui.phrasebook import Language, t

    line = t("perf.agent_elsewhere", Language.EN)
    assert "Track 2 agent" in line and "/agent" in line
    assert "/agent" in t("perf.agent_elsewhere", Language.ZH)


# --- windows ----------------------------------------------------------------------------------

_WED = datetime(2026, 9, 30, 12, tzinfo=UTC)


@pytest.mark.parametrize(("asked", "day"), [
    ("what did the desk do last Friday", date(2026, 9, 25)),
    ("what happened on Monday", date(2026, 9, 28)),
    ("this past sunday", date(2026, 9, 27)),
    ("what did you do last Wednesday", date(2026, 9, 23)),
    ("上周五做了什么", date(2026, 9, 25)),
])
def test_a_named_weekday_is_that_day(asked: str, day: date) -> None:
    from argus.lui.question import resolve_window

    window = resolve_window(asked, now=_WED)
    assert window is not None and window.start.date() == day
    assert (window.end.date() - window.start.date()).days == 1


def test_a_weekday_inside_a_word_is_not_a_window() -> None:
    from argus.lui.question import resolve_window

    assert resolve_window("NVDA sat idle", now=_WED) is None


def test_a_list_over_a_named_day_is_not_relabelled_by_the_ngram_model() -> None:
    from argus.lui.ngram import reclassify
    from argus.lui.question import classify

    question = classify("what did the desk do last Friday", now=_WED)
    assert reclassify(question) == (question, "patterns")


def test_a_chinese_day_question_is_not_declined_as_off_topic() -> None:
    from argus.lui.arbiter import gate_ledger_reading
    from argus.lui.question import Intent, classify

    question = classify("上周五做了什么", now=_WED)
    kept, by = gate_ledger_reading(question, "patterns", "上周五做了什么", prior=[], audit={},
                                   desk_first=False)
    assert kept.intent is Intent.DECISION_LIST and by == "patterns"
    other = classify("what did the lakers do last friday", now=_WED)
    _, declined = gate_ledger_reading(other, "patterns", "what did the lakers do last friday",
                                      prior=[], audit={}, desk_first=False)
    assert declined == "declined-off-topic"


def test_why_not_is_a_follow_up() -> None:
    from argus.lui.server import _BARE_WHY, _WHY_THAT

    assert _WHY_THAT.match("Why not?") and _BARE_WHY.match("why not")


# --- the console page -------------------------------------------------------------------------


def test_the_pause_says_when_it_lifts() -> None:
    import time

    from argus.lui import server

    server._VISITS["198.51.100.7"] = [time.monotonic() - 3300.0]
    try:
        assert server.allowance_back_in("198.51.100.7") == 5
        assert "about 5 more minutes at most" in server.allowance_note("198.51.100.7")
    finally:
        del server._VISITS["198.51.100.7"]
    assert server.allowance_back_in("198.51.100.8") == 0


def test_a_typed_draft_survives_a_suggestion() -> None:
    from argus.lui.server import PAGE

    assert "if (draft) { qEl.value = draft; grow(); }" in PAGE
    assert "language model paused" in PAGE


def test_the_wrong_page_leads_with_the_plain_summary_and_its_kinds() -> None:
    from argus.lui.corrections_page import render

    @dataclass
    class C:
        kind: str
        headline: str = "h"
        detail: str = "d"
        artefact: str = ""

    page = render([C("bug"), C("loss"), C("loss")])  # type: ignore[list-item]
    assert page.index("class='plain'") < page.index("Every entry is read")
    assert "By kind: 1 we shipped it broken, 2 a baseline beat us." in page


def test_the_agent_page_says_a_few_trades_say_little() -> None:
    from argus.lui.agent_page import _metrics

    few = _metrics({"n_closed_trades": 3, "win_rate": 0.0, "sharpe_ann": 1.9}, {})
    assert "On 3 closed trades these figures say little yet" in few
    assert "say little" not in _metrics({"n_closed_trades": 40, "win_rate": 0.5}, {})


# --- the thesis tester ------------------------------------------------------------------------


def test_a_capex_thesis_is_one_demand_driver_reason_without_the_asking() -> None:
    from argus.lui.thesis import Kind, reasons

    found = reasons("I think NVDA runs on AI capex through 2027 - test my thesis")
    assert [(r.text, r.kind) for r in found] == [("NVDA runs on AI capex through 2027",
                                                  Kind.DRIVER)]
    two = reasons("my thesis is that TSLA is overvalued and demand is slowing, check it")
    assert [r.kind for r in two] == [Kind.VALUATION, Kind.DRIVER]
    # "check the" inside a reason is not the asking
    assert reasons("NVDA will check the 200-day and bounce")


@dataclass(frozen=True)
class _Row:
    start: date | None
    end: date
    value: float
    filed: date
    unit: str = "USD"


def test_year_to_date_cash_flow_is_split_into_quarters() -> None:
    from argus.lui.drivers import quarters

    rows = [_Row(date(2025, 1, 1), date(2025, 3, 31), 10.0, date(2025, 5, 1)),
            _Row(date(2025, 1, 1), date(2025, 6, 30), 25.0, date(2025, 8, 1)),
            _Row(date(2025, 1, 1), date(2025, 9, 30), 45.0, date(2025, 11, 1)),
            _Row(date(2025, 1, 1), date(2025, 12, 31), 70.0, date(2026, 2, 1))]
    assert [(q.end, q.value) for q in quarters(rows)] == [
        (date(2025, 3, 31), 10.0), (date(2025, 6, 30), 15.0), (date(2025, 9, 30), 20.0),
        (date(2025, 12, 31), 25.0)]


def _series(start_value: float, step: float, n: int = 12) -> list[Any]:
    from argus.lui.drivers import Quarter

    ends = [date(2023 + (3 * i + 2) // 12, (3 * i + 2) % 12 + 1, 28) for i in range(n)]
    return [Quarter(end, start_value * (1 + step) ** i) for i, end in enumerate(ends)]


def _facts(revenue_step: float, capex_step: float) -> dict[str, Any]:
    builders = {t: _series(10e9, capex_step) for t in ("MSFT", "GOOGL", "AMZN", "META")}
    return {"ticker": "NVDA", "cik": 1045810, "revenue": _series(20e9, revenue_step),
            "revenue_tag": "Revenues", "builders": builders,
            "builder_tags": dict.fromkeys(builders, "PaymentsToAcquirePropertyPlantAndEquipment"),
            "builder_ciks": dict.fromkeys(builders, 789019)}


def test_a_driver_that_still_grows_supports_the_thesis_and_names_what_is_untested() -> None:
    from argus.lui.drivers import test

    result, line, evidence = test("NVDA runs on AI capex through 2027", _facts(0.15, 0.10))
    assert result == "supported" and line.startswith("Today it holds")
    assert "cannot test is the part still ahead" in line
    assert any("Microsoft, Alphabet, Amazon and Meta together spent" in e[0] for e in evidence)
    assert any("not proof that one drives the other" in e[0] for e in evidence)


def test_a_falling_driver_contradicts_it_and_a_turn_claim_reads_the_trend() -> None:
    from argus.lui.drivers import test

    assert test("NVDA runs on AI capex", _facts(0.15, -0.05))[0] == "contradicted"
    assert test("AI capex is peaking", _facts(0.15, 0.10))[0] in ("contradicted",
                                                                   "not measurable")
    assert test("NVDA grows over the next year", _facts(0.10, 0.10))[0] == "supported"


def test_an_unreadable_filing_is_not_tested_rather_than_guessed() -> None:
    from argus.lui.drivers import test

    assert test("demand is growing", {"error": "the SEC's ticker map did not answer"})[0] \
        == "not tested"


def test_valuation_weighs_the_multiple_and_the_target_and_says_when_they_disagree() -> None:
    from argus.lui.thesis import Kind, Reason, Result, _valuation

    both = {"pe": 370.9, "sector_pe": 25.5, "pe_vs_sector": 14.6, "sector_fund": "XLY",
            "sector": "Consumer Cyclical", "target_mean": 393.38, "price": 353.54}
    split = _valuation(Reason("TSLA is overvalued", Kind.VALUATION), both, "TSLA")
    assert split.result is Result.NOT_MEASURABLE and "opposite ways" in split.line
    assert split.line.startswith("Trailing P/E 370.9 against Consumer Cyclical's 25.5 (XLY)")
    agree = dict(both, target_mean=300.0)
    assert _valuation(Reason("TSLA is overvalued", Kind.VALUATION), agree,
                      "TSLA").result is Result.SUPPORTED
    assert _valuation(Reason("TSLA is cheap", Kind.VALUATION), {},
                      "TSLA").result is Result.NOT_TESTED


def test_the_sector_read_hands_its_ratio_to_the_thesis() -> None:
    from argus.lui.research.sector import sector_valuation

    def fetch(ticker: str, modules: str) -> dict[str, Any]:
        if ticker == "XLK":
            return {"topHoldings": {"equityHoldings": {"priceToEarnings": {"raw": 0.03},
                                                       "priceToBook": {"raw": 0.09}}}}
        return {"assetProfile": {"sector": "Technology"},
                "summaryDetail": {"trailingPE": {"raw": 60.0}},
                "defaultKeyStatistics": {"priceToBook": {"raw": 20.0}}}

    found: dict[str, Any] = {}
    assert sector_valuation("NVDA", fetch=fetch, found=found) is not None
    assert found["pe_vs_sector"] == pytest.approx(1.8) and found["sector_fund"] == "XLK"


@pytest.mark.parametrize("asked", [
    "I think NVDA runs on AI capex through 2027 - test my thesis",
    "my thesis is that TSLA is overvalued", "poke holes in my idea: long SOL",
    "is my thesis right? ETH outperforms BTC",
])
def test_a_stated_thesis_reaches_the_tester(asked: str) -> None:
    from argus.lui.thesis_answer import asks

    assert asks(asked)


def test_a_thesis_with_no_name_asks_for_one_without_running_anything() -> None:
    from argus.lui.thesis_answer import answer

    lines, _, data = answer("test my thesis")
    assert lines[0].startswith("Bottom line: name the stock or coin") and data == {"thesis": None}


# --- the earnings event study counts results, not delivery reports ------------------------------


@dataclass(frozen=True)
class _Filing:
    form: str
    accepted: datetime
    items: tuple[str, ...] = ()


def _at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(hour=20, tzinfo=UTC)


def test_a_deliveries_release_filed_under_item_2_02_is_not_a_results_release() -> None:
    from argus.research.event_reactions import results_releases

    tesla = [_Filing("8-K", _at("2025-10-02"), ("2.02", "9.01")),
             _Filing("8-K", _at("2025-10-22"), ("2.02", "9.01")),
             _Filing("10-Q", _at("2025-10-23")),
             _Filing("8-K", _at("2026-01-02"), ("2.02", "9.01")),
             _Filing("8-K", _at("2026-01-28"), ("2.02", "9.01")),
             _Filing("10-K", _at("2026-01-29")),
             _Filing("8-K", _at("2026-04-02"), ("2.02",)),
             _Filing("8-K", _at("2026-04-22"), ("2.02",))]
    assert [d.date().isoformat() for d in results_releases(tesla)] == [
        "2025-10-22", "2026-01-28", "2026-04-22"]


def test_the_measured_move_is_asked_beside_the_date() -> None:
    from argus.lui.research.dispatch import _MOVE_ON_RESULTS

    assert _MOVE_ON_RESULTS.search("when does TSLA report and how big is the move usually")
    assert _MOVE_ON_RESULTS.search("how does NVDA usually react to earnings")
    assert not _MOVE_ON_RESULTS.search("when does TSLA report")
