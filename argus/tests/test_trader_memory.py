"""The console's memory of what a trader told it (`lui/memory.py`), offline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from argus.lui import memory as mem
from argus.lui.provenance import label
from argus.lui.research import ResearchKind, ResearchRequest

NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)


@pytest.mark.parametrize(("text", "kind", "value"), [
    ("I can't lose more than 10% of my account", "max_loss", "0.1"),
    ("my risk budget is 20%", "budget", "0.2"),
    ("I'm a swing trader", "horizon", "168"),
    ("I usually hold for weeks", "horizon", "168"),
    ("im pretty aggressive", "style", "aggressive"),
    ("my account is 50k", "capital", "50000"),
    ("I think NVDA will outperform on AI capex", "thesis", "bull"),
    ("I expect TSLA to miss", "thesis", "bear"),
    ("I don't trade TSLA", "avoid", "avoid"),
])
def test_statements_become_facts(text: str, kind: str, value: str) -> None:
    facts = mem.extract(text, NOW)
    assert any(f.kind == kind and f.value == value for f in facts), facts
    assert all(f.at == "2026-09-25" for f in facts)


@pytest.mark.parametrize("text", ["what if I can't lose more than 10%?", "should I trade earnings",
                                  "what is my max drawdown", "is a 20% risk budget normal?"])
def test_questions_state_nothing(text: str) -> None:
    assert mem.extract(text, NOW) == []


def test_a_thesis_is_stamped_with_the_price_it_was_stated_at() -> None:
    facts = mem.extract("I think NVDA will outperform", NOW, price_of=lambda s: 200.0)
    assert facts[0].subject == "NVDAUSDT" and facts[0].price_at == 200.0
    line = mem.thesis_line(facts[0], "NVDA", 210.0)
    assert "+5.0%" in line and "with it" in line and label(line) == "memory"
    assert "flat so far" in mem.thesis_line(facts[0], "NVDA", 200.05)


def test_parse_drops_anything_malformed_and_caps_the_size() -> None:
    assert mem.parse("not json") == [] and mem.parse('{"kind": "budget"}') == []
    rows = [{"kind": "budget", "value": "0.2", "text": "x" * 999, "at": "2026-09-25"},
            {"kind": "evil", "value": "1"}, "junk",
            {"kind": "thesis", "subject": "NVDAUSDT", "value": "bull", "price_at": "NaN"}]
    facts = mem.parse(json.dumps(rows))
    assert [f.kind for f in facts] == ["budget", "thesis"]
    assert len(facts[0].text) == mem.MAX_TEXT and facts[1].price_at is None
    many = [{"kind": "style", "value": str(i), "text": "t", "at": "2026-09-25"} for i in range(99)]
    assert len(mem.parse(json.dumps(many))) == mem.MAX_FACTS


def test_a_newer_fact_replaces_the_older_one_of_its_kind() -> None:
    old = mem.extract("my risk budget is 20%", NOW) + mem.extract("I'm conservative", NOW)
    new = mem.extract("my risk budget is 30%", NOW)
    merged = mem.merge(old, new)
    assert [f.value for f in merged if f.kind == "budget"] == ["0.3"]
    assert any(f.kind == "style" for f in merged)
    assert mem.parse(mem.dumps(merged)) == merged


def test_apply_fills_what_the_question_left_out_and_says_so() -> None:
    facts = mem.extract("my risk budget is 20%", NOW) + mem.extract("I'm a swing trader", NOW)
    request = ResearchRequest(kind=ResearchKind.ANALOGUE, symbols=("BTCUSDT",), horizon_hours=24,
                              notes=("no horizon was stated, so the next 24 hours are read",))
    applied, used = mem.apply(request, facts, "will BTC go up?")
    assert applied.budget == 0.2 and applied.budget_stated
    assert applied.horizon_hours == 168 and not applied.notes
    assert len(used) == 2 and all(label(line) == "memory" for line in used)
    stated = ResearchRequest(kind=ResearchKind.ANALOGUE, symbols=("BTCUSDT",), horizon_hours=72)
    kept, _ = mem.apply(stated, facts, "will BTC go up over 3 days?")
    assert kept.horizon_hours == 72


def test_after_sets_a_loss_limit_against_the_loss_found() -> None:
    facts = mem.extract("I can't lose more than 10%", NOW)
    request = ResearchRequest(kind=ResearchKind.STRESS, symbols=("NVDAUSDT",))
    lines = ["Bottom line: If QQQ moves -10%: your book moves about -12.40%, hardest hit NVDA."]
    extra = mem.after(lines, request, facts)
    joined = " ".join(extra)
    assert extra and "past that limit" in joined and "-12.4%" in joined
    assert "uses 124% of your 10% limit" in joined
    inside = mem.after(["If QQQ moves -5%: your book moves about -6.00%"], request, facts)
    assert "stays inside it" in " ".join(inside)


def test_the_server_acknowledges_a_statement_and_returns_the_memory(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import server

    def declined(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"lines": ["I did not recognise that question"], "refused": True,
                "classified_by": "declined", "sources": []}

    monkeypatch.setattr(server, "_answer", declined)
    payload = server.handle_ask("I can't lose more than 10% of my account", [], memory="")
    assert payload["classified_by"] == "memory" and not payload["refused"]
    assert payload["lines"][0].startswith("Bottom line: noted")
    stored = mem.parse(payload["memory"])
    assert stored and stored[0].kind == "max_loss"
    again = server.handle_ask("my risk budget is 20%", [], memory=payload["memory"])
    assert {f.kind for f in mem.parse(again["memory"])} == {"max_loss", "budget"}


def test_memory_is_scoped_to_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import server

    seen: list[tuple[Any, ...]] = []

    def record(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(server._MEMORY.get())
        return {"lines": ["x"], "refused": False, "classified_by": "patterns", "sources": []}

    monkeypatch.setattr(server, "_answer", record)
    server.handle_ask("my risk budget is 20%", [], memory="")
    server.handle_ask("hello", [], memory="")
    assert seen[0] and seen[1] == ()
    assert server._MEMORY.get() == ()


class TestARememberedMandateReachesTheCheck:
    """Audit finding 55: "I'm conservative", said once, never reached the mandate check; the next
    "should I add 15% TSLA?" was answered as it was for anyone."""

    def _impact(self) -> ResearchRequest:
        return ResearchRequest(kind=ResearchKind.IMPACT, symbols=("TSLAUSDT",),
                               book={"NVDAUSDT": 1.0}, size=0.15, size_stated=True)

    def test_a_remembered_style_becomes_the_mandate_and_is_said(self) -> None:
        facts = mem.extract("I'm conservative", NOW) + mem.extract("my account is 50k", NOW)
        request, used = mem.apply(self._impact(), facts, "should I add 15% TSLA?")
        assert request.mandate_text == "I'm conservative"
        assert str(request.mandate_capital) == "50000"
        assert any("mandate in this answer is checked against it" in line for line in used)
        assert any("sizes on your $50,000" in line for line in used)

    def test_the_question_own_mandate_wins_and_memory_stays_out(self) -> None:
        facts = mem.extract("I'm conservative", NOW)
        request, used = mem.apply(self._impact(), facts,
                                  "I'm an aggressive trader, should I add 15% TSLA?")
        assert request.mandate_text == "" and not any("mandate" in line for line in used)

    def test_a_style_that_is_not_a_mandate_changes_nothing(self) -> None:
        facts = mem.extract("I mostly trade earnings", NOW)
        request, used = mem.apply(self._impact(), facts, "should I add 15% TSLA?")
        assert request.mandate_text == "" and used == []

    def test_the_mandate_check_reads_the_remembered_words(self) -> None:
        from datetime import timedelta

        from argus.lui.research.parse import _mandate_lines

        start = datetime(2026, 9, 1, tzinfo=UTC)
        series = {"TSLAUSDT": {start + timedelta(hours=i): (-0.004 if i % 2 else 0.003)
                               for i in range(200)}}
        assert _mandate_lines("TSLAUSDT", 0.15, series, "should I add 15% TSLA?") == []
        lines = _mandate_lines("TSLAUSDT", 0.15, series, "should I add 15% TSLA?",
                               "I'm conservative", None)
        assert lines and "conservative" in " ".join(lines)


# --- the checklist a trade review wrote, kept and run again (audit finding 56) ------------------

from datetime import date, timedelta  # noqa: E402

from argus.lui import journal  # noqa: E402

REVIEW = ("review my trades: bought NVDA at 188 on 2026-08-20, sold at 176 on 2026-08-27; "
          "bought NVDA at 180 on 2026-09-01, sold at 172 on 2026-09-08; "
          "bought NVDA at 175 on 2026-09-10, sold at 169 on 2026-09-15")


def test_a_reviews_checklist_becomes_check_facts_and_replaces_the_last_one() -> None:
    out = journal.review_trades(REVIEW, now=NOW, history=None, releases=None)
    assert out is not None
    checks = mem.checks_from(out[2], NOW)
    assert checks and all(f.kind == "check" and f.text for f in checks)
    assert {f.subject for f in checks} == {p["key"] for p in out[2]["journal"]["patterns"]}
    stale = mem.Fact(kind="check", subject="chasing", value="1", text="old check", at="2026-09-01")
    budget = mem.Fact(kind="budget", subject="", value="0.2", text="my risk budget is 20%",
                      at="2026-09-01")
    kept = mem.merge_checks([stale, budget], checks)
    assert budget in kept and stale not in kept
    assert mem.parse(mem.dumps(kept)) == kept


def _check(key: str, text: str) -> mem.Fact:
    return mem.Fact(kind="check", subject=key, value="2", text=text, at="2026-09-20")


def test_the_checklist_runs_on_an_entry_and_names_what_it_cannot_test() -> None:
    facts = [_check("chasing", "Wait a session after a big day."),
             _check("size_after_loss", "Keep the next trade at your usual size.")]
    request = ResearchRequest(kind=ResearchKind.IMPACT, symbols=("TSLAUSDT",))
    lines = mem.checklist_lines(request, facts, tester=lambda key: (
        "its last session was one of its largest up days" if key == "chasing" else None),
        today=date(2026, 9, 27))
    assert lines[0].startswith("Your checklist, from your trade review on 2026-09-20, run on this "
                               "TSLA buy over a default 5-day hold")
    assert lines[1] == ("Checklist: Wait a session after a big day. Now: its last session was "
                        "one of its largest up days.")
    assert lines[2] == ("Not testable from a question, kept as reminders: Keep the next trade at "
                        "your usual size.")


def test_a_check_that_fails_to_run_is_not_passed() -> None:
    def broken(key: str) -> str:
        raise TimeoutError

    request = ResearchRequest(kind=ResearchKind.EXECUTION, symbols=("NVDAUSDT",))
    lines = mem.checklist_lines(request, [_check("chasing", "Wait.")], tester=broken)
    assert lines[1] == "Checklist: Wait. Now: could not be run (TimeoutError), so it is not passed."


def test_the_checklist_is_not_run_where_no_entry_is_weighed() -> None:
    request = ResearchRequest(kind=ResearchKind.QUOTE, symbols=("NVDAUSDT",))
    assert mem.checklist_lines(request, [_check("chasing", "Wait.")], tester=lambda k: "x") == []


def _closes(last_move: float) -> journal.History:
    start = date(2025, 6, 1)
    closes, price = [], 100.0
    for i in range(400):
        price *= 1 + (0.004 if i % 2 else -0.0035)
        closes.append((start + timedelta(days=i), price))
    closes.append((start + timedelta(days=400), price * (1 + last_move)))
    return closes, "synthetic"


def test_chasing_is_measured_as_the_review_measured_it() -> None:
    today = date(2025, 6, 1) + timedelta(days=401)
    big = journal.retest("chasing", "BTCUSDT", "long", today=today, horizon_days=5,
                         history=lambda s: _closes(0.05))
    assert big is not None and "one of its largest up days" in big
    quiet = journal.retest("chasing", "BTCUSDT", "long", today=today, horizon_days=5,
                           history=lambda s: _closes(0.0))
    assert quiet is not None and quiet.endswith("this check passes")
    unread = journal.retest("chasing", "BTCUSDT", "long", today=today, horizon_days=5,
                            history=None)
    assert unread is not None and "not passed" in unread


def test_earnings_closures_and_repeats_are_tested_on_the_named_stock() -> None:
    today = date(2026, 9, 28)  # a Monday
    inside = journal.retest("earnings_losers", "NVDAUSDT", "long", today=today, horizon_days=5,
                            next_report=lambda s: date(2026, 10, 1))
    assert inside is not None and "reports on 2026-10-01, 3 day(s) away" in inside
    after = journal.retest("earnings_losers", "NVDAUSDT", "long", today=today, horizon_days=5,
                           next_report=lambda s: date(2026, 11, 19))
    assert after is not None and "after your 5-day hold" in after
    unread = journal.retest("earnings_losers", "NVDAUSDT", "long", today=today, horizon_days=5,
                            next_report=None)
    assert unread is not None and "not passed" in unread
    crossed = journal.retest("closure_losers", "NVDAUSDT", "long", today=today, horizon_days=7,
                             holidays=frozenset())
    assert crossed is not None and "crosses 2 closed US day(s), the first 2026-10-03" in crossed
    assert journal.retest("closure_losers", "BTCUSDT", "long", today=today, horizon_days=7,
                          holidays=frozenset()) is None
    assert "lose on repeatedly" in (journal.retest(
        "repeat_NVDAUSDT", "NVDAUSDT", "long", today=today, horizon_days=5) or "")
    assert journal.retest("size_after_loss", "NVDAUSDT", "long", today=today,
                          horizon_days=5) is None
