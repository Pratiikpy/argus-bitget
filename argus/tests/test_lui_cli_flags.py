"""The terminal console speaks to programs as well as people: --json, -q, and exit status
(research/harvest/51-cli-guidelines.md)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import pytest

from argus.lui import cli


@dataclass
class _Question:
    intent: str = "integrity"
    speed: Any = None


class _Answer:
    def __init__(self, *, refused: bool = False) -> None:
        from argus.lui.question import Speed

        self.question = _Question(speed=Speed.FAST)
        self.refused = refused
        self.reason = "the record cannot support it" if refused else ""
        self.lines = ["Hash chain intact.", "Each row carries the previous hash."]
        self.sources: list[Any] = []

    def as_dict(self) -> dict[str, Any]:
        return {"intent": "integrity", "refused": self.refused, "lines": self.lines}


@pytest.fixture
def console(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """`main` with the ledger, router and answerer replaced; records every question asked."""
    asked: list[str] = []

    def fake_ask(ledger: Any, text: str, **_: Any) -> tuple[Any, float]:
        asked.append(text)
        return _Answer(refused=text.startswith("refuse")), 12.0

    monkeypatch.setattr(cli, "ask", fake_ask)
    monkeypatch.setattr(cli, "PaperLedger", lambda **_: object())
    monkeypatch.setattr(cli, "build_router", lambda: None)
    return asked


def test_json_prints_one_parseable_object(console: list[str],
                                          capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--json", "is", "the", "log", "tamper-evident"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["intent"] == "integrity" and row["elapsed_ms"] == 12.0
    assert row["budget_ms"] == cli.BUDGET_MS[_Answer().question.speed]
    assert row["over_budget"] is False
    assert console == ["is the log tamper-evident"]


def test_quiet_prints_only_the_answer(console: list[str],
                                      capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["-q", "is the log tamper-evident"]) == 0
    assert capsys.readouterr().out == "Hash chain intact.\nEach row carries the previous hash.\n"


def test_a_refusal_exits_one_in_every_mode(console: list[str]) -> None:
    assert cli.main(["refuse this"]) == 1
    assert cli.main(["--json", "refuse this"]) == 1
    assert cli.main(["-q", "refuse this"]) == 1


def test_a_usage_error_exits_two(console: list[str]) -> None:
    with pytest.raises(SystemExit) as bad:
        cli.main(["--json", "-q", "x"])
    assert bad.value.code == 2
    with pytest.raises(SystemExit) as unknown:
        cli.main(["--no-such-flag"])
    assert unknown.value.code == 2


def test_piped_questions_answer_one_json_line_each(
        console: list[str], monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("first question\n\nsecond question\n"))
    assert cli.main(["--json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 2 and console == ["first question", "second question"]
