"""Round 11 of the §27 audits (2026-09-30): a hostile review, a first-time user and a judge.

Each test pins one finding (tracker rows 502-529), named in the docstring of the code it covers.
Live sources are replaced by the rows they return, so nothing here touches the network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

_WED = datetime(2026, 9, 30, 12, tzinfo=UTC)


# --- sizing, newcomers, concepts --------------------------------------------------------------


def test_an_account_typed_without_a_dollar_sign_is_not_read_from_the_risk_percentage() -> None:
    from argus.lui.research.sizing import _ACCOUNT

    text = "size a trade: 10k account 1% risk stop 4% below entry"
    readings = [m.group("n") for pattern in _ACCOUNT for m in [pattern.search(text)] if m]
    assert "1" not in readings and "10" in readings


@pytest.mark.parametrize(("asked", "starts"), [
    ("is my money safe here", "Bottom line: this console never holds, moves or touches money"),
    ("are you financial advice", "Bottom line: no — this console never tells anyone"),
])
def test_newcomer_trust_questions_are_answered_plainly(asked: str, starts: str) -> None:
    from argus.lui.newcomer import reply

    answered = reply(asked)
    assert answered is not None and answered.lines[0].startswith(starts)


def test_should_i_sell_now_says_no_call_in_its_own_words() -> None:
    from argus.lui.newcomer import reply

    answered = reply("should i sell my crypto now")
    assert answered is not None and "whether to sell now" in answered.lead
    assert "a good time to sell" in str(answered.note)
    assert reply("whats the best coin to buy right now") is not None


@pytest.mark.parametrize(("asked", "name"), [
    ("wut is a stop loss", "stop loss"),
    ("how risky is leverage trading", "leverage"),
    ("is margin trading risky", "leverage"),
    ("bitcoin kya hai", "bitcoin"),
    ("stop loss kya hota hai", "stop loss"),
    ("explain apy vs apr", "APY and APR"),
    ("what does Agent Hub's dry run do", "Agent Hub dry run"),
])
def test_beginner_terms_as_typed_are_defined(asked: str, name: str) -> None:
    from argus.lui.concepts import concept_asked
    from argus.lui.research import research_symbols

    found = concept_asked(asked, research_symbols(asked)[0])
    assert found is not None and found.name == name


def test_asking_for_bitcoins_price_is_not_a_definition() -> None:
    from argus.lui.concepts import concept_asked
    from argus.lui.research import research_symbols

    asked = "what is the bitcoin price"
    assert concept_asked(asked, research_symbols(asked)[0]) is None


# --- the product answering for itself -----------------------------------------------------------


@pytest.mark.parametrize("asked", ["how do you know all this", "what model do you run on",
                                   "are you an AI?", "where does the data come from"])
def test_how_it_knows_and_which_model_are_answered(asked: str) -> None:
    from argus.lui.intro import HOW_Q, how_answer

    assert HOW_Q.search(asked)
    lines, _, _ = how_answer(asked)
    if "model" in asked:
        assert lines[0].startswith("Bottom line: Qwen 3.8 Max")


def test_the_how_answer_states_the_servers_own_allowance() -> None:
    from argus.lui.intro import how_answer
    from argus.lui.server import MODEL_CALLS_PER_VISITOR_PER_HOUR

    said = " ".join(how_answer()[0])
    assert f"up to {MODEL_CALLS_PER_VISITOR_PER_HOUR} questions an hour" in said


def test_the_scoreboard_asked_whole_is_the_registers_counts() -> None:
    from argus.eval.standing import REGISTER
    from argus.lui.intro import STANDING_Q, standing_answer
    from argus.lui.rivals import asks_about_a_rival

    asked = "how many capabilities have you tested against rivals"
    assert STANDING_Q.search(asked) and not asks_about_a_rival(asked)
    assert standing_answer()[0][0].startswith(f"Bottom line: {len(REGISTER)} capabilities")


def test_the_rivals_lead_reads_as_a_sentence() -> None:
    from argus.lui.rivals import answer

    assert answer("how does ARGUS compare to Nautilus Trader")[0][0].startswith(
        "Bottom line: in ARGUS's register, ")


@pytest.mark.parametrize(("asked", "one"), [
    ("what does the sentiment-analyst skill do", True),
    ("what are the bitget skills", False),
])
def test_the_skills_are_explained_from_bitgets_own_readme(asked: str, one: bool) -> None:
    from argus.lui import skills_explain

    assert skills_explain.asks(asked)
    lines, _, data = skills_explain.answer(asked)
    assert (data["skills"] == ["sentiment-analyst"]) is one
    assert "https://github.com/Bitget-AI/bitget-signal" in lines[-1]


def test_a_skill_named_beside_a_contract_is_that_contracts_reading_not_the_explainer() -> None:
    from argus.lui import skills_explain

    assert not skills_explain.asks("what does the bitget-signal sentiment skill say about BTC")
    assert not skills_explain.asks("are the bitget skills working")


# --- follow-ups and windows ----------------------------------------------------------------


def test_another_day_after_a_day_question_is_that_question_again() -> None:
    from argus.lui.server import _DAY_FOLLOW_UP, _IT_FOLLOW_UP

    found = _DAY_FOLLOW_UP.match("why not Thursday?")
    assert found is not None and found.group("day") == "Thursday"
    assert _DAY_FOLLOW_UP.match("and monday?")
    assert _IT_FOLLOW_UP.search("does it place a real order")


@pytest.mark.parametrize(("asked", "kept"), [
    ("what happened last Friday", True),
    ("what did the lakers do last friday", False),
])
def test_a_subject_less_day_question_is_the_records(asked: str, kept: bool) -> None:
    from argus.lui.arbiter import gate_ledger_reading
    from argus.lui.question import Intent, classify

    question = classify(asked, now=_WED)
    reading, by = gate_ledger_reading(question, "patterns", asked, prior=[], audit={},
                                      desk_first=False)
    assert (reading.intent is Intent.DECISION_LIST and by == "patterns") is kept


def test_a_bare_name_follow_up_is_not_told_it_asked_nothing() -> None:
    from argus.lui.arbiter import arbitrate

    reading = arbitrate("and ETH?", book="", model=None, instruction=False, desk_first=False,
                        audit={})
    notes = " ".join(reading.request.notes) if reading.request is not None else ""
    assert "no specific research question" not in notes


# --- the desk's record --------------------------------------------------------------------------


def test_the_desk_record_states_its_win_rate_without_unsaying_it() -> None:
    from argus.lui.phrasebook import Language, t

    line = t("perf.few_trades", Language.EN, trades=1, abstentions=878, days=18, net="-$1.78",
             win=t("perf.few_trades_win", Language.EN, pct=0.0, wins=0, trades=1))
    assert "Win rate 0% (0 of 1)." in line
    assert "too few trades to judge the desk by" in line
    assert "to say anything" not in line


# --- the thesis tester --------------------------------------------------------------------------


def test_a_closing_question_is_not_a_reason() -> None:
    from argus.lui.thesis import Kind, reasons

    found = reasons("I think NVDA is undervalued because its P/E is low relative to growth. "
                    "Is that thesis right?")
    assert [(r.text, r.kind) for r in found] == [("NVDA is undervalued", Kind.VALUATION),
                                                 ("its P/E is low relative to growth",
                                                  Kind.VALUATION)]
    assert [r.text for r in reasons("my thesis: inflation will crush COIN")] == [
        "inflation will crush COIN"]


def test_a_thesis_with_its_reason_reaches_the_tester_without_the_magic_words() -> None:
    from argus.lui.thesis_answer import asks

    assert asks("I think NVDA is undervalued because its P/E is low relative to growth. "
                "Is that thesis right?")
    assert asks("I think SOL will outrun ETH because activity is growing, am I wrong?")


@pytest.mark.parametrize(("text", "kind"), [
    ("SOL will outrun ETH", "one name against another"),
    ("MSTR goes higher", "a direction"),
    ("BTC will go up", "a direction"),
    ("Fed rate cuts will boost NVDA", "macro"),
    ("institutional adoption", "network activity"),
])
def test_each_new_reason_kind_is_read(text: str, kind: str) -> None:
    from argus.lui.thesis import reasons

    assert [r.kind.value for r in reasons(text)] == [kind]


def test_one_name_against_another_reads_the_record_so_far() -> None:
    from argus.lui.thesis import Kind, Reason, Result, _relative

    found = {"names": ["SOLUSDT", "ETHUSDT"], "SOLUSDT": {7: 0.049, 30: 0.161, 89: 0.471},
             "ETHUSDT": {7: 0.006, 30: 0.083, 89: 0.544}}
    split = _relative(Reason("SOL will outrun ETH", Kind.RELATIVE), found)
    assert split.result is Result.NOT_MEASURABLE and "no data tests that" in split.line
    ahead = dict(found, ETHUSDT={7: 0.0, 30: 0.0, 89: 0.1})
    assert _relative(Reason("SOL will outrun ETH", Kind.RELATIVE), ahead).result \
        is Result.SUPPORTED


def test_a_macro_premise_is_checked_against_the_fed_funds_rate() -> None:
    from argus.lui.macro_thesis import test

    found = {"fed_funds": {"now": 3.88, "then": 3.62, "since": "2026-06-02"},
             "rates": {"n": 60, "slope_per_10bp": -0.0025, "corr": -0.16, "t": -1.2,
                       "from": "2026-07-02", "to": "2026-09-28", "ten_year": 5.2}, "events": {}}
    result, line, _ = test("the Fed is cutting rates", found, "SP500")
    assert result == "contradicted" and "up, not down" in line
    result, line, _ = test("Fed rate cuts will boost NVDA", found, "NVDA")
    assert result == "not measurable" and "too weak to call" in line
    strong = dict(found, rates=dict(found["rates"], t=-3.1))
    assert test("Fed rate cuts will boost NVDA", strong, "NVDA")[0] == "supported"
    assert test("rate hikes will help NVDA", strong, "NVDA")[0] == "contradicted"


def test_institutional_adoption_reads_the_13f_ownership() -> None:
    from argus.lui.thesis import Kind, Reason, Result, _institutions

    found = {"holders": 1087, "share_pct": 64.07, "holding_vol": 233_861_671,
             "net_trade_vol": 12_456_708, "as_of": "2026-09-28"}
    tested = _institutions(Reason("institutional adoption", Kind.ACTIVITY), found, "MSTR")
    assert tested.result is Result.SUPPORTED and "+5.6%" in tested.line


def test_a_thesis_on_an_unlisted_company_says_so() -> None:
    from argus.lui.thesis_answer import answer

    lines, _, data = answer("I think Tencent will outperform because of gaming approvals, "
                            "test my thesis")
    assert lines[0].startswith("Bottom line: Tencent is not a contract Bitget lists")
    assert data["unlisted"] == "Tencent"


# --- sources that disagree, and the Skills --------------------------------------------------------


def test_two_treasury_feeds_that_disagree_are_both_said(monkeypatch: Any, tmp_path: Any) -> None:
    import json

    from argus.market import bitget_positioning

    (tmp_path / "etf_flows.json").write_text(json.dumps({"treasury": {
        "holding_btc": 847666, "recent": [{"date": "2026-09-28", "btc": 1665, "usd": 1.43e8}]}}),
        encoding="utf-8")
    monkeypatch.setattr("argus.truth.paths.DATA_DIR", tmp_path)
    said = bitget_positioning._other_treasury_record("MSTR", 846842.0)
    assert "847,666 BTC after the 2026-09-28 purchase" in said
    assert "SEC 8-K is the record" in said
    assert bitget_positioning._other_treasury_record("MSTR", 847666.0) == ""


def test_a_paused_model_says_so_in_the_questions_language() -> None:
    from argus.lui.translate import PAUSED, target_language

    assert target_language("Wie reagiert NVDA auf CPI-Daten?") == "de"
    assert target_language("Quelle est la thèse de valorisation pour NVDA et pourquoi?") == "fr"
    assert target_language("Is NVDA a buy on the dip for a long term hold?") is None
    for answered, declined in PAUSED.values():
        assert "{minutes}" in answered and "{minutes}" in declined


def test_the_deploy_bundle_loses_files_the_source_deleted(tmp_path: Any) -> None:
    from argus.demo.deploysync import _copy_tree

    source, target = tmp_path / "src", tmp_path / "dst"
    (source / "eval").mkdir(parents=True)
    (source / "eval" / "kept.toml").write_text("a", encoding="utf-8")
    (target / "eval").mkdir(parents=True)
    (target / "eval" / "kept.toml").write_text("a", encoding="utf-8")
    (target / "eval" / "gone.toml").write_text("b", encoding="utf-8")
    _copy_tree(source, target, dry_run=False)
    assert (target / "eval" / "kept.toml").exists()
    assert not (target / "eval" / "gone.toml").exists()


def test_the_saved_book_is_said_under_the_lead_not_at_the_foot() -> None:
    from argus.lui.server import saved_book_first

    lines = ["Bottom line: NVDA already carries 52% of this book's risk.", "a", "b",
             "Assumed: used your saved book (40% NVDA, 30% MSFT, 30% AAPL).", "Data: x."]
    moved = saved_book_first(lines)
    assert moved[1] == ("Assumed: read against your saved book in My book — 40% NVDA, 30% MSFT, "
                        "30% AAPL; clear it to ask without it.")
    assert len(moved) == len(lines) and moved[-1] == "Data: x."
    assert saved_book_first(["Bottom line: x.", "y"]) == ["Bottom line: x.", "y"]
