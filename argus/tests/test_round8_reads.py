"""How questions are read, as the round-8 role audits found them misread (2026-09-30)."""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.lui.research import detect
from argus.lui.research.kinds import ResearchKind

LEDGER = Path(__file__).resolve().parents[1] / "data" / "paper_ledger.jsonl"


def test_a_model_supplied_book_naming_the_shock_gives_way_to_the_stated_one() -> None:
    """A judge's audit: "stress test this portfolio for a 10% Nasdaq drop" with a stated book came
    back as 100% QQQ every time the hosted model planned it."""
    from argus.lui.research.parse import with_book

    planned = detect("what if QQQ drops 10%")
    assert planned is not None
    from dataclasses import replace

    modelled = replace(planned, book={"QQQUSDT": 1.0},
                       notes=("no weights were given for your holdings, so they were read as "
                              "equal weight",))
    kept = with_book(modelled, "40% NVDA, 30% TSLA, 30% COIN",
                     "Stress test this portfolio for a 10% Nasdaq drop tomorrow.")
    assert kept is not None
    assert kept.book == pytest.approx({"NVDAUSDT": 0.4, "TSLAUSDT": 0.3, "COINUSDT": 0.3})
    assert not any("equal weight" in note for note in kept.notes)


@pytest.mark.parametrize("text", ["What is your thesis on Bitcoin right now?",
                                  "whats the bull case for NVDA"])
def test_a_thesis_on_a_name_is_a_view_that_stands(text: str) -> None:
    from argus.lui.research.fundamentals import pattern_reading_wins

    request = detect(text)
    assert request is not None and request.kind is ResearchKind.IMPACT
    assert pattern_reading_wins(request, text)


@pytest.mark.parametrize(("text", "kind"), [
    ("how does buying actually work", "lines"),
    ("what could go wrong", "lines"),
    ("Guarantee me a 50% return on BTC", "lines"),
    ("i have $2000, how much should i put in", "reask"),
    ("what is your recommendation for a total beginner with $2000", "reask"),
    ("is now a good time to buy", "reask"),
])
def test_the_nameless_newcomer_questions_are_answered(text: str, kind: str) -> None:
    from argus.lui.newcomer import reply

    found = reply(text)
    assert found is not None and bool(found.lines) is (kind == "lines")


def test_a_named_question_is_left_to_its_engines() -> None:
    from argus.lui.newcomer import reply

    assert reply("what could go wrong with NVDA", named=True) is None
    assert reply("is now a good time to buy TSLA", named=True) is None


@pytest.mark.parametrize(("text", "guaranteed"), [
    ("Guarantee me a 50% return on BTC", True), ("can you guarantee profit", True),
    ("what is the guaranteed rate on USDC", False), ("what is the risk-free rate", False),
])
def test_only_a_request_for_a_certain_return_is_told_no(text: str, guaranteed: bool) -> None:
    from argus.lui.newcomer import _GUARANTEE

    assert bool(_GUARANTEE.search(text)) is guaranteed


@pytest.mark.parametrize("text", ["what could go wrong with NVDA",
                                  "what are the risks of holding TSLA"])
def test_the_risks_of_one_name_are_its_risk_profile(text: str) -> None:
    from argus.lui.research.fundamentals import pattern_reading_wins

    request = detect(text)
    assert request is not None and request.kind is ResearchKind.IMPACT
    assert pattern_reading_wins(request, text)


def test_the_allowance_question_is_answered_with_the_count() -> None:
    from argus.lui import server

    server._VISITS.pop("10.0.0.9", None)
    payload = server.handle_ask("how many questions do i have left", [], visitor="10.0.0.9")
    assert payload["classified_by"] == "allowance"
    assert payload["lines"][0].startswith(
        f"Bottom line: {server.MODEL_CALLS_PER_VISITOR_PER_HOUR} of "
        f"{server.MODEL_CALLS_PER_VISITOR_PER_HOUR} questions are left")


def test_the_architecture_answer_names_the_modules_and_the_reference() -> None:
    from argus.lui import architecture

    assert architecture.ARCHITECTURE_Q.search("Walk me through your architecture")
    assert architecture.ARCHITECTURE_Q.search("how is the desk built")
    assert not architecture.ARCHITECTURE_Q.search("how does NVDA work")
    lines, _, _ = architecture.answer()
    text = " ".join(lines)
    for module in ("paper/runner.py", "agents/desk.py", "paper/ledger.py", "lui/arbiter.py"):
        assert module in text
        assert (Path(__file__).resolve().parents[1] / "src" / "argus" / module).exists()


def test_the_record_on_one_name_counts_what_the_performance_counts() -> None:
    from argus.lui import edge
    from argus.paper.ledger import PaperLedger
    from argus.paper.performance import evaluate_ledger

    ledger = PaperLedger(path=LEDGER)
    if not ledger.entries:
        pytest.skip("no ledger in this checkout")
    symbol = ledger.entries[-1].symbol
    lines, _, data = edge.name_answer(ledger, symbol)
    mine = next((c for c in evaluate_ledger(ledger).by_symbol if c.symbol == symbol), None)
    assert data["trades"] == (mine.trades if mine else 0)
    assert data["decisions"] == sum(1 for e in ledger.entries if e.symbol == symbol)
    assert lines[0].startswith("Bottom line: the desk")
    assert edge.EDGE_ON_NAME_Q.search("What's your edge on NVDA specifically?")


def test_a_named_leverage_scenario_is_not_a_definition() -> None:
    """Live, 2026-09-30: "Explain the risk in shorting DOGE with 20x leverage" got the glossary
    entry for leverage; the scenario is the leverage engine's."""
    request = detect("Explain the risk in shorting DOGE with 20x leverage")
    assert request is not None and request.kind is ResearchKind.LEVERAGE
    source = (Path(__file__).resolve().parents[1] / "src" / "argus" / "lui" / "server.py"
              ).read_text(encoding="utf-8")
    assert "scenario.kind is ResearchKind.LEVERAGE" in source
