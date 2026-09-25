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
    lines = ["Actionable: If QQQ moves -10%: your book moves about -12.40%, hardest hit NVDA."]
    extra = mem.after(lines, request, facts)
    assert extra and "past that limit" in extra[0] and "-12.4%" in extra[0]
    inside = mem.after(["If QQQ moves -5%: your book moves about -6.00%"], request, facts)
    assert "stays inside it" in inside[0]


def test_the_server_acknowledges_a_statement_and_returns_the_memory(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import server

    def declined(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"lines": ["I did not recognise that question"], "refused": True,
                "classified_by": "declined", "sources": []}

    monkeypatch.setattr(server, "_answer", declined)
    payload = server.handle_ask("I can't lose more than 10% of my account", [], memory="")
    assert payload["classified_by"] == "memory" and not payload["refused"]
    assert payload["lines"][0].startswith("Actionable: noted")
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
