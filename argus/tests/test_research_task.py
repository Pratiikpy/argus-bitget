"""The research task (`lui/task.py`) end to end, with its eight engines stubbed.

Track 3's required demo is one complete research task, "the full flow from question to actionable
insight". Until 2026-09-26 the flow started from three form fields, not a question, and nothing
tested what the task sends each engine; a judge audit found both. These tests drive the flow the
way a trader does, from their own words, and read what reached every engine, so they need no
network: each engine is replaced by a stub that records the request it was given.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from argus.lui import exposures
from argus.lui import task as task_mod
from argus.lui.research import ResearchKind, ResearchRequest
from argus.lui.server import Handler
from argus.lui.task import Reading, as_dict, read_question, research_task

LISTED = (
    "TSLAUSDT", "NVDAUSDT", "AAPLUSDT", "MSFTUSDT", "BTCUSDT", "XAUUSDT", "AMDUSDT", "MSTRUSDT",
)
ASKED = "I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?"


@dataclass
class _Answer:
    lines: list[str]
    refused: bool = False
    data: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> list[ResearchRequest]:
    """Every request the task sends, in the order the engines receive them."""
    seen: list[ResearchRequest] = []

    def run(question: str, request: ResearchRequest, ledger: Any = None) -> _Answer:
        seen.append(request)
        kind = request.kind.value
        return _Answer(lines=[f"{kind} reading", f"Actionable: act on the {kind} step"])

    monkeypatch.setattr(task_mod, "run", run)
    monkeypatch.setattr(task_mod, "_price_path", lambda symbol: [])
    monkeypatch.setattr(task_mod, "contracts", lambda: dict.fromkeys(LISTED))
    monkeypatch.setattr(exposures, "exposures_answer", _exposures([]))
    return seen


def _exposures(calls: list[Any]) -> Any:
    """A stub of the exposures engine that records the book and the add it was given."""

    def answer(book: Any, proposed: Any = None, **_: Any) -> tuple[list[str], list[Any], dict]:
        calls.append((dict(book), dict(proposed or {})))
        return (["Actionable: act on the exposure step", "Sector weights of the book now: x."],
                [], {"book": dict(book)})

    return answer


def _one(seen: list[ResearchRequest], kind: ResearchKind) -> ResearchRequest:
    (request,) = [r for r in seen if r.kind is kind]
    return request


class TestTheQuestionIsRead:
    def test_a_full_question_gives_the_name_the_size_and_the_book(self) -> None:
        reading = read_question(ASKED)
        assert isinstance(reading, Reading)
        assert reading.name == "TSLAUSDT"
        assert reading.size_pct == pytest.approx(15)
        assert reading.book == pytest.approx({"NVDAUSDT": 0.4, "MSFTUSDT": 0.3, "AAPLUSDT": 0.3})
        assert reading.summary == "add 15% TSLA to 40% NVDA / 30% MSFT / 30% AAPL"

    def test_the_saved_book_fills_in_when_the_question_names_none(self) -> None:
        reading = read_question("what does adding 20% TSLA do to my risk", "50% NVDA, 50% AAPL")
        assert isinstance(reading, Reading)
        assert reading.book == pytest.approx({"NVDAUSDT": 0.5, "AAPLUSDT": 0.5})
        assert any("saved book" in n for n in reading.notes)

    def test_a_question_that_only_names_a_contract_runs_on_it_and_says_what_it_assumed(
        self,
    ) -> None:
        for asked in ("do full research on TSLA for my book", "研究一下加仓TSLA"):
            reading = read_question(asked, "50% NVDA, 50% AAPL")
            assert isinstance(reading, Reading), asked
            assert reading.name == "TSLAUSDT"
            assert reading.size_pct == task_mod.UNSTATED_SIZE_PCT
            assert any("no size was given" in n for n in reading.notes)

    def test_a_held_name_stays_in_the_book(self) -> None:
        """"research NVDA" from a 40% NVDA holder used to run as a first NVDA position in a book
        with the NVDA taken out."""
        reading = read_question("research NVDA", "40% NVDA, 60% AAPL")
        assert isinstance(reading, Reading)
        assert reading.book == pytest.approx({"NVDAUSDT": 0.4, "AAPLUSDT": 0.6})
        assert reading.summary.startswith("add 20% more NVDA")
        assert any("already hold 40% NVDA" in n for n in reading.notes)

    def test_an_add_of_more_keeps_its_size(self) -> None:
        """"10% more NVDA" was read as no size at all and assessed at the 20% default."""
        reading = read_question("should I add 10% more NVDA?", "40% NVDA, 60% AAPL")
        assert isinstance(reading, Reading)
        assert reading.size_pct == pytest.approx(10)

    def test_a_dollar_size_sizes_the_execution(self) -> None:
        reading = read_question("how do I buy 50k of NVDA")
        assert isinstance(reading, Reading)
        assert reading.notional == Decimal(50000)
        assert "$50,000 to execute" in reading.summary

    @pytest.mark.parametrize(
        ("asked", "said"),
        [
            ("", "Type a question"),
            ("hello there", "could not find a name Bitget lists"),
            ("what if the Nasdaq drops 10%?", "console"),
        ],
    )
    def test_what_is_not_a_task_is_said_rather_than_guessed(self, asked: str, said: str) -> None:
        reading = read_question(asked, "50% NVDA, 50% AAPL")
        assert isinstance(reading, str)
        assert said in reading


class TestTheTaskRunsOnWhatWasRead:
    def test_every_engine_is_asked_about_the_name_and_the_book(
        self, engines: list[ResearchRequest]
    ) -> None:
        reading = read_question(ASKED)
        assert isinstance(reading, Reading)
        task = research_task(reading=reading, asked=ASKED)
        assert task.question == ASKED
        assert len(task.steps) == 8
        assert {r.kind for r in engines} == {
            ResearchKind.QUOTE,
            ResearchKind.TECHNICALS,
            ResearchKind.NEWS,
            ResearchKind.FUNDAMENTALS,
            ResearchKind.ANALOGUE,
            ResearchKind.IMPACT,
            ResearchKind.EXECUTION,
        }
        impact = _one(engines, ResearchKind.IMPACT)
        assert impact.symbols == ("TSLAUSDT", "NVDAUSDT", "MSFTUSDT", "AAPLUSDT")
        assert dict(impact.book) == pytest.approx(reading.book)
        assert impact.size == pytest.approx(0.15)
        assert impact.size_stated
        assert _one(engines, ResearchKind.EXECUTION).notional == Decimal(15000)
        assert all(r.symbols[0] == "TSLAUSDT" for r in engines)

    def test_the_conclusion_leads_with_the_book(self, engines: list[ResearchRequest]) -> None:
        reading = read_question(ASKED)
        assert isinstance(reading, Reading)
        task = research_task(reading=reading, asked=ASKED)
        assert task.conclusion[0] == ("What the trade does to your book", "act on the impact step")
        assert len(task.conclusion) == 8

    def test_a_held_candidate_is_sent_once_and_kept_in_the_book(
        self, engines: list[ResearchRequest]
    ) -> None:
        reading = read_question("research NVDA", "40% NVDA, 60% AAPL")
        assert isinstance(reading, Reading)
        research_task(reading=reading, asked="research NVDA")
        impact = _one(engines, ResearchKind.IMPACT)
        assert impact.symbols == ("NVDAUSDT", "AAPLUSDT")
        assert dict(impact.book) == pytest.approx({"NVDAUSDT": 0.4, "AAPLUSDT": 0.6})

    def test_the_exposure_step_moves_the_book_as_the_impact_engine_does(
        self, engines: list[ResearchRequest], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A held name ends at its scaled weight plus the add (`desk.portfolio.rebalance`)."""
        calls: list[Any] = []
        monkeypatch.setattr(exposures, "exposures_answer", _exposures(calls))
        reading = read_question("should I add 10% more NVDA?", "40% NVDA, 60% AAPL")
        assert isinstance(reading, Reading)
        task = research_task(reading=reading, asked="should I add 10% more NVDA?")
        ((book, proposed),) = calls
        assert book == pytest.approx({"NVDAUSDT": 0.4, "AAPLUSDT": 0.6})
        assert proposed == pytest.approx({"NVDAUSDT": 0.4 * 0.9 + 0.1})
        titles = [s.title for s in task.steps]
        assert titles.index(task_mod.EXPOSURE_TITLE) == titles.index(task_mod.IMPACT_TITLE) + 1

    def test_no_book_means_no_exposure_to_move(self, engines: list[ResearchRequest]) -> None:
        reading = read_question("should I add 10% NVDA?")
        assert isinstance(reading, Reading)
        task = research_task(reading=reading, asked="should I add 10% NVDA?")
        (step,) = [s for s in task.steps if s.title == task_mod.EXPOSURE_TITLE]
        assert not step.applicable

    def test_a_stated_dollar_size_reaches_execution(self, engines: list[ResearchRequest]) -> None:
        reading = read_question("how do I buy 50k of NVDA")
        assert isinstance(reading, Reading)
        research_task(reading=reading, asked="how do I buy 50k of NVDA")
        assert _one(engines, ResearchKind.EXECUTION).notional == Decimal(50000)

    def test_a_crypto_name_has_no_earnings_step(self, engines: list[ResearchRequest]) -> None:
        reading = read_question("should I add 10% BTC?", "50% NVDA, 50% AAPL")
        assert isinstance(reading, Reading)
        task = research_task(reading=reading, asked="should I add 10% BTC?")
        (fundamentals,) = [s for s in task.steps if s.title.startswith("Earnings")]
        assert not fundamentals.applicable
        assert ResearchKind.FUNDAMENTALS not in {r.kind for r in engines}

    def test_a_name_bitget_does_not_list_is_one_honest_line(
        self, engines: list[ResearchRequest]
    ) -> None:
        task = research_task("PLTR", 10, "50% NVDA, 50% AAPL")
        assert len(task.steps) == 1
        assert task.steps[0].refused
        assert "not listed on Bitget" in task.steps[0].lines[0]
        assert engines == []

    def test_the_json_says_what_was_read(self, engines: list[ResearchRequest]) -> None:
        reading = read_question(ASKED)
        assert isinstance(reading, Reading)
        payload = as_dict(research_task(reading=reading, asked=ASKED))
        assert payload["question"] == ASKED
        assert payload["read_as"]["summary"] == reading.summary
        assert as_dict(research_task())["read_as"] is None


@pytest.fixture
def base_url(engines: list[ResearchRequest]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _post(url: str, fields: dict[str, str]) -> tuple[int, str]:
    request = Request(url, data=urlencode(fields).encode(), method="POST")
    with urlopen(request) as response:
        return response.status, response.read().decode("utf-8")


class TestThePageTakesAQuestion:
    def test_a_posted_question_runs_the_task_and_shows_what_was_read(self, base_url: str) -> None:
        status, body = _post(f"{base_url}/research", {"q": ASKED, "book": ""})
        assert status == 200
        assert "Read as:</b> add 15% TSLA to 40% NVDA / 30% MSFT / 30% AAPL" in body
        assert body.count("<article") == 8
        assert '<form method="post" action="/research">' in body
        assert '<textarea name="q"' in body

    def test_the_saved_book_travels_with_the_question(self, base_url: str) -> None:
        status, body = _post(f"{base_url}/research", {"q": "research TSLA", "book": "50% SPY"})
        assert status == 200
        assert "add 20% TSLA to 100% SPY" in body

    def test_an_unreadable_question_is_answered_with_why(self, base_url: str) -> None:
        status, body = _post(f"{base_url}/research", {"q": "hello there"})
        assert status == 200
        assert "What I could not read" in body
        assert "could not find a name Bitget lists" in body

    def test_a_shared_link_with_a_question_works_too(self, base_url: str) -> None:
        query = urlencode({"q": ASKED, "format": "json"})
        with urlopen(f"{base_url}/research?{query}") as response:
            payload = json.loads(response.read())
        assert payload["read_as"]["summary"].startswith("add 15% TSLA")
        assert len(payload["steps"]) == 8

    def test_the_fields_still_run_without_a_question(self, base_url: str) -> None:
        status, body = _post(
            f"{base_url}/research?format=json",
            {"q": "", "name": "gold", "size": "10", "book": "50% SPY, 50% QQQ"},
        )
        assert status == 200
        payload = json.loads(body)
        assert payload["name"] == "XAU"
        assert payload["size_pct"] == 10


def _sizing(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "TSLAUSDT", "proposed": 0.15, "budget": 0.25, "budget_stated": False,
        "standalone": False, "alone": False, "held": 0.0, "final": 0.15,
        "share_after": 0.17, "share_before": None, "ceiling": 0.19, "trim_to": None,
        "crowded": {"symbol": "NVDAUSDT", "weight": 0.34, "share": 0.48, "trim_to": 0.2},
        "worst_24h_pct": -2.0,
    }
    return {**base, **over}


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The engines' data: change ``figures["sizing"]`` before running the task."""
    state: dict[str, Any] = {"sizing": _sizing()}

    def run(question: str, request: ResearchRequest, ledger: Any = None) -> _Answer:
        kind = request.kind.value
        data: dict[str, Any] = {}
        if request.kind is ResearchKind.IMPACT:
            data = {"sizing": state["sizing"]}
        elif request.kind is ResearchKind.EXECUTION:
            data = {"request": {"notional": str(request.notional)},
                    "execution": {"expected_cost_bps": "10.25", "slices": [
                        {"fraction": "0.6", "style": "near-touch limit", "cost_bps": "6"},
                        {"fraction": "0.4", "style": "market", "cost_bps": "6"}]}}
        return _Answer(lines=[f"Actionable: act on the {kind} step"], data=data)

    monkeypatch.setattr(task_mod, "run", run)
    monkeypatch.setattr(task_mod, "_price_path", lambda symbol: [])
    monkeypatch.setattr(task_mod, "contracts", lambda: dict.fromkeys(LISTED))
    monkeypatch.setattr(exposures, "exposures_answer", _exposures([]))
    return state


def _verdict(asked: str = ASKED) -> task_mod.Verdict | None:
    reading = read_question(asked)
    assert isinstance(reading, Reading)
    return research_task(reading=reading, asked=asked).verdict


class TestTheVerdictIsComposedFromTheFigures:
    def test_inside_the_budget_is_an_add_with_the_fill_plan(self, figures: dict[str, Any]) -> None:
        call = _verdict()
        assert call is not None
        assert call.call == "Add, at 15%"
        assert "17% of the book's risk, inside the 25% budget; 19% is the most" in call.lines[0]
        assert "trimming NVDA to 20% brings it inside" in call.lines[1]
        assert call.lines[2] == ("Fill it as 60% near-touch limit then 40% market, about 10.2 bps "
                                 "all in for $15,000.")

    def test_over_the_budget_is_an_add_smaller_with_the_ceiling(
        self, figures: dict[str, Any]
    ) -> None:
        figures["sizing"] = _sizing(proposed=0.4, share_after=0.45, ceiling=0.26, crowded=None)
        call = _verdict("I hold 80% NVDA, 20% AAPL — should I add 40% TSLA?")
        assert call is not None
        assert call.call == "Add smaller: at most 26%"
        assert "at the 26% ceiling that order is $26,000" in call.lines[-1]

    def test_a_held_name_over_budget_is_do_not_add_and_no_fill_plan(
        self, figures: dict[str, Any]
    ) -> None:
        figures["sizing"] = _sizing(symbol="MSTRUSDT", held=0.5, share_before=0.82, ceiling=None,
                                    trim_to=0.17, crowded=None)
        call = _verdict("I hold 50% MSTR, 50% AAPL — should I add 20% more MSTR?")
        assert call is not None
        assert call.call == "Do not add"
        assert call.lines == ("MSTR already carries 82% of the book's risk at 50%, over the 25% "
                              "budget, so any add takes it further over; trimming it to 17% "
                              "brings it inside.",)

    def test_no_book_sizes_by_the_worst_day(self, figures: dict[str, Any]) -> None:
        figures["sizing"] = _sizing(standalone=True, ceiling=None, crowded=None,
                                    worst_24h_pct=-4.4)
        call = _verdict("should I add 10% NVDA?")
        assert call is not None
        assert call.call == "Size it by its worst day"
        assert "-4.4% day" in call.lines[0]
        assert len(call.lines) == 1

    def test_no_sizing_no_verdict(self, engines: list[ResearchRequest]) -> None:
        assert _verdict() is None

    def test_the_page_and_the_json_lead_with_it(self, figures: dict[str, Any]) -> None:
        reading = read_question(ASKED)
        assert isinstance(reading, Reading)
        done = research_task(reading=reading, asked=ASKED)
        page = task_mod.render_task(done, "")
        assert page.index("<h2>Conclusion</h2>") < page.index("What to do, engine by engine")
        assert "Add, at 15%" in page
        assert as_dict(done)["verdict"]["call"] == "Add, at 15%"
