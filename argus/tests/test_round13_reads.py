"""Round 13 of the §27 audits (2026-09-30): a judge's pass on the product as a whole.

Each test pins one finding (tracker rows 552-559). Live sources are replaced by stubs, so nothing
here touches the network.
"""

from __future__ import annotations

from typing import Any

import pytest

_PERSONA = ("I'm a swing trader, my horizon is a few weeks, and I can't lose more than 10% of my "
            "book. I hold 40% NVDA, 30% MSFT, 30% AAPL.")
_THESIS = ("I think NVDA keeps running because AI capex is accelerating, gross margins are "
           "expanding, and the data center bookings backlog is huge. Should I add 15% TSLA?")
_FIRST_REVISION = ("Actually I was wrong about capex accelerating - it's slowing next quarter. "
                   "Does that change your view?")
_SECOND_REVISION = ("Forget capex and margins. My real reason now is just the bookings backlog. "
                    "Does that change your sizing?")


class TestTheFeatureListIsAskable:
    """Row 552: "What can you do? List your main capabilities and data sources." was answered
    with the desk's last decision."""

    @pytest.mark.parametrize("q", [
        "What can you do? List your main capabilities and data sources.",
        "List your main capabilities", "what are your features?", "list your data sources",
        "show me all your capabilities and data sources", "what is argus? list your features"])
    def test_the_question_reaches_the_feature_list(self, q: str) -> None:
        from argus.lui.intro import CAPABILITIES_Q

        assert CAPABILITIES_Q.search(q)

    @pytest.mark.parametrize("q", ["list NVDA holders", "what are the risks of NVDA",
                                   "list my holdings", "what are your sources"])
    def test_other_lists_are_not_taken(self, q: str) -> None:
        from argus.lui.intro import CAPABILITIES_Q

        assert not CAPABILITIES_Q.search(q)

    def test_sources_alone_go_to_the_sources_answer(self) -> None:
        from argus.lui.intro import SOURCES_ONLY_Q

        assert SOURCES_ONLY_Q.search("what data sources do you use")

    def test_the_list_names_each_capability_with_a_question_and_the_sources(self) -> None:
        from argus.lui.intro import capabilities_answer

        lines, _, data = capabilities_answer()
        text = " ".join(lines)
        for part in ("Thesis tester", "Risk and sizing", "Personal memory", "bitget-signal",
                     "SEC EDGAR", "FRED", "/proof"):
            assert part in text
        assert data["total"] == sum(data["standing"].values())


def test_a_loss_limit_is_not_a_request_for_a_sure_thing() -> None:
    """Row 553: "I can't lose more than 10% of my book" was told no return can be guaranteed."""
    from argus.lui.newcomer import _GUARANTEE

    assert not _GUARANTEE.search("I can't lose more than 10% of my book")
    assert not _GUARANTEE.search("I can't lose over $500")
    assert _GUARANTEE.search("give me a trade that can't lose")


class TestTheHorizonSaidInWords:
    """Row 554: "my horizon is a few weeks" was stored as the swing-trader default, one week."""

    @pytest.mark.parametrize(("text", "hours"), [
        (_PERSONA, "504"), ("my horizon is a few weeks. I am a swing trader", "504"),
        ("I hold for 3 months", "2160"), ("horizon 2 weeks", "336"),
        ("I usually hold for weeks", "168"), ("I am a day trader", "24"),
        ("my investment horizon is 5 years", "43800")])
    def test_the_count_and_the_unit_are_read(self, text: str, hours: str) -> None:
        from argus.lui.memory import extract

        assert [f.value for f in extract(text) if f.kind == "horizon"] == [hours]

    def test_the_remembered_horizon_reaches_the_mandate(self) -> None:
        from argus.lui import memory as mem
        from argus.lui.research import ResearchKind, ResearchRequest

        facts = mem.extract(_PERSONA)
        request = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("TSLAUSDT",),
                                  book={"NVDAUSDT": 1.0}, size=0.15, size_stated=True)
        applied, _ = mem.apply(request, facts, "should I add 15% TSLA?")
        assert "my horizon is 504 hours" in applied.mandate_text


class TestTheBookSaidInChat:
    """Row 555: the book stated in chat was forgotten by the next question."""

    def test_holdings_in_a_sentence_are_kept_as_the_book(self) -> None:
        from argus.lui.memory import extract

        kept = [f for f in extract(_PERSONA) if f.kind == "book"]
        assert [f.text for f in kept] == ["I hold 40% NVDA, 30% MSFT, 30% AAPL"]

    @pytest.mark.parametrize("text", ["Should I add 15% TSLA?", "I hold 70% NVDA, 50% TSLA",
                                      "I hold NVDA r-token overnight - how do I hedge it?"])
    def test_a_trade_or_an_impossible_book_is_not(self, text: str) -> None:
        from argus.lui.memory import extract

        assert not [f for f in extract(text) if f.kind == "book"]

    def test_the_remembered_book_is_used_and_said(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import server
        from argus.lui.memory import dumps, extract

        seen: list[str] = []

        def answer(text: str, prior: list[str], **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs.get("book", ""))
            return {"lines": ["Bottom line: x", "Assumed: used your saved book (40% NVDA)"],
                    "refused": False, "classified_by": "exposures", "sources": []}

        monkeypatch.setattr(server, "_answer", answer)
        monkeypatch.setattr(server, "_model_for", lambda *a, **k: None)
        payload = server.handle_ask("what are my exposures if I add 10% XOM?", [],
                                    memory=dumps(extract(_PERSONA)))
        assert seen == ["I hold 40% NVDA, 30% MSFT, 30% AAPL"]
        assert "used the book you gave on" in payload["lines"][1]
        server.handle_ask("what are my exposures?", [], book="50% BTC, 50% ETH",
                          memory=dumps(extract(_PERSONA)))
        assert seen[-1] == "50% BTC, 50% ETH"


class TestTheModelReaderKeepsOnlyWhatWasSaid:
    """Row 556: "40% NVDA" beside "I hold" was kept as a 40% risk budget, and "Long rNVDA" replaced
    the trader's NVDA thesis."""

    def test_a_holding_is_not_a_budget(self) -> None:
        from argus.lui.memory_model import _valid

        assert _valid("budget", "40", "", "40% NVDA", "I hold 40% NVDA, 30% MSFT.") is None
        assert _valid("budget", "12", "", "nothing over 12% of the book in one name") == (
            "0.12", "")

    def test_an_order_is_not_a_thesis(self) -> None:
        from argus.lui.memory_model import _valid

        assert _valid("thesis", "bull", "NVDA", "Long rNVDA over the weekend at 3x") is None
        assert _valid("thesis", "bull", "NVDA", "I think NVDA runs")[0] == "bull"  # type: ignore[index]


class TestAThesisChangedTwice:
    """Row 557: the second follow-up ("forget capex and margins, my real reason is just the
    backlog") was refused as an unlisted name; only the turn just before was read."""

    def test_the_claim_is_a_direction_not_a_loose_reason(self) -> None:
        from argus.lui.thesis import Kind, reasons

        assert reasons(_THESIS)[0].kind is Kind.DIRECTION

    def test_revisions_accumulate_from_the_thesis(self) -> None:
        from argus.lui import thesis
        from argus.lui.thesis_answer import _apply, _thesis_turn

        turns = [_PERSONA, _THESIS, _FIRST_REVISION]
        assert _thesis_turn(turns) == 1
        stated = list(thesis.reasons(_THESIS))
        kept, dropped = _apply(_FIRST_REVISION, stated)  # type: ignore[misc]
        assert [r.text for r in dropped] == ["AI capex is accelerating"]
        kept, dropped = _apply(_SECOND_REVISION, kept)  # type: ignore[misc]
        assert [r.text for r in kept] == ["NVDA keeps running",
                                          "the data center bookings backlog is huge"]
        assert [r.text for r in dropped] == ["gross margins are expanding"]

    def test_the_answer_names_what_was_set_aside_and_speaks_to_sizing(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import thesis_answer

        asked: list[str] = []

        def answer(text: str, **kwargs: Any) -> tuple[list[str], list[Any], dict[str, Any]]:
            asked.append(text)
            tested = [{"reason": r, "result": "supported", "implied": False}
                      for r in ("AI capex is accelerating",
                                "the data center bookings backlog is huge")]
            return ["Bottom line: tested", "Supported — backlog"], [], {
                "thesis": {"name": "NVDA", "tested": tested}}

        monkeypatch.setattr(thesis_answer, "answer", answer)
        out = thesis_answer.revise(_SECOND_REVISION, [_PERSONA, _THESIS, _FIRST_REVISION])
        assert out is not None
        lines, _, data = out
        assert "\"AI capex is accelerating\" and \"gross margins are expanding\" set aside" in \
            lines[0]
        assert any(line.startswith("On sizing:") for line in lines)
        rebuilt = next(a for a in asked if a.startswith("My bull case"))
        assert "gross margins" not in rebuilt and "backlog" in rebuilt
        assert all("TSLA" not in a for a in asked)
        assert data["dropped"] == ["AI capex is accelerating", "gross margins are expanding"]

    def test_a_question_after_the_thesis_is_split_off(self) -> None:
        from argus.lui.thesis_answer import split_other_question

        claim, also = split_other_question(_THESIS)
        assert also == "Should I add 15% TSLA?" and "TSLA" not in claim
        assert split_other_question("I think NVDA runs on AI capex. Test my thesis.")[1] == ""


def test_a_trim_of_a_name_not_held_says_so() -> None:
    """Row 558: "What if I trim TSLA to 10% of my book?" on a book without TSLA asked for the
    book it had."""
    from argus.lui.research import ResearchKind, ResearchRequest
    from argus.lui.research.riskmath import _resolve_resize

    request = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("TSLAUSDT",), target=0.10)
    said = _resolve_resize(request, {"NVDAUSDT": 0.4, "MSFTUSDT": 0.3, "AAPLUSDT": 0.3})
    assert isinstance(said, str) and said.startswith("TSLA is not in your book")
    assert "40% NVDA" in said and "adding 10% TSLA" in said
