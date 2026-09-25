"""The per-step Qwen cost ledger: every call recorded, attributed, and summed exactly once."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from argus.llm import ledger
from argus.llm.ledger import CostLedger, Outcome, cost_step, read, role_of_the_llm, summarise
from argus.llm.qwen import (
    BudgetExhausted,
    Completion,
    QwenClient,
    QwenError,
    TokenBudget,
    Usage,
)

SECRET = "sk-never-in-the-ledger-0123456789"


def _client(monkeypatch: pytest.MonkeyPatch, path: Path | None, *,
            limit: int = 1_000_000, script: list[Any] | None = None) -> QwenClient:
    monkeypatch.setenv("BITGET_QWEN_API_KEY", SECRET)
    monkeypatch.setenv("BITGET_QWEN_BASE_URL", "https://hackathon.example.invalid/v1")
    client = QwenClient(budget=TokenBudget(limit=limit), ledger=CostLedger(path))
    queue = list(script or [])

    def fake_post(payload: dict[str, Any]) -> Completion:
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        assert isinstance(step, Completion)
        return step

    monkeypatch.setattr(client, "_post", fake_post)
    return client


def _answer(total: int, *, reasoning: int = 0, reported: bool = True) -> Completion:
    return Completion("ok", "", Usage(total // 2, total - total // 2, reasoning, total,
                                      reported=reported), "stop")


def _ask(client: QwenClient, text: str) -> None:
    client.complete([{"role": "user", "content": text}])


def test_the_ledger_sum_equals_the_budget_over_every_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    client = _client(monkeypatch, path, limit=10_000, script=[
        _answer(1200, reasoning=500), _answer(800, reported=False), QwenError("HTTP 504: gw"),
        _answer(3000, reasoning=2000),
    ])
    _ask(client, "a")
    _ask(client, "a")                      # cache hit, sends nothing
    _ask(client, "b")                      # answered with no usage block: unmeasured
    with pytest.raises(QwenError):
        _ask(client, "c")
    _ask(client, "d")
    assert client.budget is not None
    client.budget.spent = 9_000             # the next call would cross the cap
    with pytest.raises(BudgetExhausted):
        _ask(client, "e")

    outcomes = [e.outcome for e in client.ledger.entries]
    assert outcomes == [Outcome.OK, Outcome.CACHE_HIT, Outcome.OK, Outcome.ERROR, Outcome.OK,
                        Outcome.BUDGET_REFUSED]
    assert client.ledger.billed_tokens == 1200 + 3000
    entries, skipped = read(path)
    assert skipped == 0 and len(entries) == 6
    report = summarise(entries)
    assert report["billed_tokens"] == 4200
    assert report["calls_answered"] == 3 and report["unmeasured_calls"] == 1
    assert report["cache_hits"] == 1 and report["errors"] == 1 and report["budget_refusals"] == 1
    assert report["reasoning_tokens"] == 2500


def test_billed_tokens_equal_the_budgets_own_spend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    script: list[Any] = [_answer(n, reasoning=n // 3) for n in (17, 400, 2048, 9, 777)]
    client = _client(monkeypatch, tmp_path / "l.jsonl", script=script)
    for i in range(5):
        _ask(client, f"q{i}")
    entries, _ = read(tmp_path / "l.jsonl")
    assert client.budget is not None
    assert sum(e.billed_tokens for e in entries) == client.budget.spent == 3251


def test_each_call_is_tagged_with_the_argus_module_that_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(monkeypatch, None, script=[_answer(10), _answer(20)])
    caller: dict[str, Any] = {"__name__": "argus.desk.fake_caller", "client": client}
    exec("def go(text):\n    client.complete([{'role': 'user', 'content': text}])", caller)
    caller["go"]("one")
    with cost_step("thesis"):
        caller["go"]("two")
    first, second = client.ledger.entries
    assert first.module == second.module == "argus.desk.fake_caller"
    assert (first.step, second.step) == ("", "thesis")
    assert client.ledger.by_module() == {"argus.desk.fake_caller": 30}


def test_steps_do_not_leak_between_threads() -> None:
    seen: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        with cost_step(name):
            barrier.wait()
            seen[name] = ledger.current_step()

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("research", "decision")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"research": "research", "decision": "decision"}
    assert ledger.current_step() == ""


def test_nothing_the_ledger_writes_contains_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    client = _client(monkeypatch, path, script=[_answer(5), QwenError("HTTP 401: bad token")])
    _ask(client, "x")
    with pytest.raises(QwenError):
        _ask(client, "y")
    text = path.read_text(encoding="utf-8")
    assert SECRET not in text and "hackathon.example.invalid" in text


def test_a_torn_line_is_skipped_and_counted_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    book = CostLedger(path)
    shape = ledger.CallShape(model="qwen3.8-max", host="h", thinking="low", stream=False,
                             json_mode=True, tools=False)
    book.record(ledger.new_entry(outcome=Outcome.OK, shape=shape, total_tokens=100,
                                 prompt_tokens=60, completion_tokens=40, usage_reported=True,
                                 module="argus.agents.meta_pm"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"ts": "2026-09-25T00:00:00", "modu')
    entries, skipped = read(path)
    assert len(entries) == 1 and skipped == 1
    text = role_of_the_llm(path)
    assert "argus.agents.meta_pm: 1 calls, 100 tokens (100%)" in text
    assert "1 ledger line(s) were unreadable" in text


def test_a_test_run_never_writes_the_real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_QWEN_LEDGER", raising=False)
    assert CostLedger.default().path is None
    monkeypatch.setenv("ARGUS_QWEN_LEDGER", "off")
    assert CostLedger.default().path is None
    monkeypatch.setenv("ARGUS_QWEN_LEDGER", "/tmp/elsewhere.jsonl")
    assert CostLedger.default().path == Path("/tmp/elsewhere.jsonl")


def test_an_empty_ledger_says_so_rather_than_inventing_a_role(tmp_path: Path) -> None:
    assert "No Qwen call has been recorded" in role_of_the_llm(tmp_path / "none.jsonl")


def test_a_write_failure_never_reaches_the_model_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    blocked = tmp_path / "a_directory_not_a_file"
    blocked.mkdir()
    client = _client(monkeypatch, blocked, script=[_answer(5)])
    _ask(client, "x")
    assert client.ledger.write_failures == 1 and len(client.ledger.entries) == 1
    assert json.loads(json.dumps(summarise(client.ledger.entries)))["billed_tokens"] == 5


def test_a_module_run_with_dash_m_is_named_by_its_spec_not_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    client = _client(monkeypatch, None, script=[_answer(10)])
    spec = types.SimpleNamespace(name="argus.eval.some_study")
    caller: dict[str, Any] = {"__name__": "__main__", "__spec__": spec, "client": client}
    exec("def go():\n    client.complete([{'role': 'user', 'content': 'x'}])", caller)
    caller["go"]()
    assert client.ledger.entries[0].module == "argus.eval.some_study"
