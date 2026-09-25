"""The trace of record: each shown line's label comes from the step that made it, recorded while the
answer is built, and a line no step speaks for keeps its wording label, marked as such."""

from __future__ import annotations

import sys
import types
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pytest

from argus.lui.provenance import LABELS
from argus.truth import trace
from argus.truth.trace import (
    SourceRead,
    attach,
    carry,
    current,
    emit,
    every,
    instrumented,
    line_labels,
    recording,
    step,
    traced,
)


@dataclass
class Reply:
    """The two attributes the recorder reads off an answer: its lines, and whether it refused."""

    lines: list[str] = field(default_factory=list)
    refused: bool = False


@traced("computed")
def _beta_lines() -> list[str]:
    return ["If QQQ falls 10%, the book moves about -10.5% through beta alone."]


@traced("desk")
def _quote_desk(line: str) -> list[str]:
    return [line, "seq 664 · NVDAUSDT · no_trade · decided 2026-09-24T19:41:11+00:00"]


@traced()
def _undeclared() -> list[str]:
    return ["Assumed: no horizon was given, so a week is used.",
            "A sentence no wording rule knows about."]


@traced()
def _mixed() -> list[str]:
    emitted = "Tested on nights it had not seen: over the last 50 of 165 nights."
    emit([emitted], "record")
    return [emitted, "Another sentence no wording rule knows about."]


def _labels(recorded: trace.Trace, lines: list[str]) -> list[tuple[str | None, str]]:
    return [(x.label, x.origin) for x in line_labels(recorded, lines)]


def test_outside_a_recording_a_traced_engine_runs_unchanged() -> None:
    assert current() is None
    assert _beta_lines() == ["If QQQ falls 10%, the book moves about -10.5% through beta alone."]
    with step("a block") as nothing:
        assert nothing is None
    emit(["nothing records this"], "live")  # a no-op outside a recording


def test_a_declared_producer_labels_its_lines_from_the_trace() -> None:
    with recording("what does a 10% QQQ drop do to my book?") as recorded:
        lines = _beta_lines()
    labelled = line_labels(recorded, lines)
    assert [(x.label, x.origin) for x in labelled] == [("computed", "trace")]
    assert labelled[0].engine is not None and labelled[0].engine.endswith("_beta_lines")
    assert labelled[0].step == 1
    assert labelled[0].reason == "declared at the definition"


def test_passing_a_line_through_is_not_making_it() -> None:
    """The evals postprocessor rule: a step that returns a line it was handed did not produce it,
    so the label stays with the step that did."""
    with recording("q") as recorded:
        made = _beta_lines()
        shown = _quote_desk(made[0])
    assert _labels(recorded, shown) == [("computed", "trace"), ("desk", "trace")]
    passed = next(s for s in recorded.steps if s.engine.endswith("_quote_desk"))
    assert len(passed.output) == 2 and len(passed.produced) == 1


def test_a_line_passed_in_from_outside_any_step_keeps_its_wording_label() -> None:
    outside = "Assumed: no size was given, so NVDA is assessed at 20%."
    with recording("q") as recorded:
        shown = _quote_desk(outside)
    assert _labels(recorded, shown)[0] == ("assumed", "regex")


def test_an_undeclared_producer_falls_back_to_the_wording_rules_and_says_so() -> None:
    with recording("q") as recorded:
        lines = _undeclared()
    labelled = line_labels(recorded, lines)
    assert [(x.label, x.origin) for x in labelled] == [("assumed", "regex"), (None, "none")]
    assert all(x.engine is not None and x.engine.endswith("_undeclared") for x in labelled)
    assert "wording rules" in labelled[0].reason


def test_emit_labels_a_line_where_it_is_made() -> None:
    with recording("q") as recorded:
        lines = _mixed()
    labelled = line_labels(recorded, lines)
    assert (labelled[0].label, labelled[0].origin, labelled[0].engine) == ("record", "trace",
                                                                         "emit")
    assert labelled[0].reason == "emitted at the line site"
    assert (labelled[1].label, labelled[1].origin) == (None, "none")


def test_meta_lines_are_left_unlabelled_on_purpose() -> None:
    with recording("q") as recorded:
        pass
    assert _labels(recorded, ["Data: Bitget tickers, read 12:00 UTC", ""]) == [
        (None, "meta"), (None, "meta")]


def test_request_defaults_are_a_step_and_label_the_assumed_line() -> None:
    request = types.SimpleNamespace(kind="compare", symbols=("NVDAUSDT",),
                                    notes=("no size was given, so NVDA is assessed at 20%",))

    @traced()
    def engine(req: Any) -> list[str]:
        return ["Assumed: no size was given, so NVDA is assessed at 20%."]

    with recording("q") as recorded:
        lines = engine(request)
        engine(request)  # the same default seen twice is recorded once
    labelled = line_labels(recorded, lines)
    assert (labelled[0].label, labelled[0].origin, labelled[0].engine) == (
        "assumed", "trace", "request.defaults")
    assert sum(s.engine == "request.defaults" for s in recorded.steps) == 1
    called = next(s for s in recorded.steps if s.engine.endswith("engine"))
    assert called.inputs == {"req": "compare on NVDAUSDT"}
    assert called.facts["kind"] == "compare"


def test_a_refusal_after_a_failed_read_is_missing_and_names_the_source() -> None:
    from argus.eval.trace_audit import offline

    @traced("computed")  # a refused answer is never covered by its producer's declaration
    def quote(symbol: str) -> Reply:
        try:
            urllib.request.urlopen(
                f"https://api.bitget.com/api/v2/mix/market/ticker?symbol={symbol}", timeout=2)
        except (OSError, urllib.error.URLError):
            return Reply(["Bitget did not return a quote just now, and a stale price presented "
                          "as current is worse than none."], refused=True)
        return Reply(["unreachable in an offline test"])

    with offline(), recording("price of NVDA") as recorded:
        answer = quote("NVDAUSDT")
    labelled = line_labels(recorded, answer.lines)
    assert (labelled[0].label, labelled[0].origin) == ("missing", "trace")
    assert labelled[0].reason.startswith("a refusal after a failed read: ")
    reads = next(s for s in recorded.steps if s.engine.endswith("quote")).sources
    assert len(reads) == 1 and not reads[0].ok and reads[0].why


def test_a_refusal_with_no_failed_read_is_not_called_missing_by_the_trace() -> None:
    @traced("desk")
    def out_of_scope() -> Reply:
        return Reply(["That is an order; the console answers questions."], refused=True)

    with recording("buy 10 NVDA") as recorded:
        answer = out_of_scope()
    labelled = line_labels(recorded, answer.lines)
    assert labelled[0].origin in ("regex", "none")
    assert labelled[0].label != "desk"


def test_an_exception_is_recorded_on_its_step_and_still_raised() -> None:
    @traced()
    def broken() -> list[str]:
        raise ValueError("no candles")

    with recording("q") as recorded, pytest.raises(ValueError, match="no candles"):
        broken()
    failed = next(s for s in recorded.steps if s.engine.endswith("broken"))
    assert failed.error == "ValueError: no candles" and failed.finished_at


def test_carry_records_a_worker_thread_under_the_submitting_step() -> None:
    with (recording("q") as recorded, step("research.fanout") as parent,
          ThreadPoolExecutor(max_workers=2) as pool):
        lines = pool.submit(carry(_beta_lines)).result()
    worker = next(s for s in recorded.steps if s.engine.endswith("_beta_lines"))
    assert worker.parent is parent
    assert _labels(recorded, lines) == [("computed", "trace")]


def test_without_carry_a_plain_pool_loses_the_trace() -> None:
    """Why `carry` exists: a plain pool starts its tasks with an empty context."""
    with recording("q") as recorded, ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_beta_lines).result()
    assert not any(s.engine.endswith("_beta_lines") for s in recorded.steps)


def test_render_prints_thought_action_observation() -> None:
    with recording("q") as recorded, step("research.route", thought="read as kind 'compare'",
                                          inputs={"symbols": ["NVDAUSDT", "AMDUSDT"]}):
        _beta_lines()
    rendered = recorded.render()
    assert rendered[0] == "Thought 1: read as kind 'compare'"
    assert rendered[1] == "Action 1: research.route(symbols=[2 items])"
    assert rendered[2] == "Observation 1: 0 line(s) made"
    assert rendered[3].startswith("Action 2: ") and rendered[3].endswith("[inside step 1]")
    assert rendered[4] == "Observation 2: 1 line(s) made"


def test_bookkeeping_steps_are_dropped_from_the_record() -> None:
    @traced()
    def helper() -> int:
        return 3

    with recording("q") as recorded:
        helper()
        _beta_lines()
    engines = [row["action"]["engine"] for row in recorded.as_dict()["steps"]]
    assert len(engines) == 1 and engines[0].endswith("_beta_lines")


def test_attach_keeps_the_line_labels_shape_and_reconciles_the_transport_verdict() -> None:
    with recording("q") as recorded, step("mcp.read") as reading:
        assert reading is not None
        reading.sources.append(SourceRead("bitget-mcp-server us_fundamentals", "t", True))
        lines = _beta_lines()
    payload: dict[str, Any] = {
        "lines": ["Data: Bitget", *lines, "A sentence no wording rule knows about."],
        "data": {"coverage": {"did_not_answer": {"bitget-mcp-server us_fundamentals":
                                                     "isError in a 200"}}}}
    attach(payload, recorded)
    assert payload["line_labels"] == [None, "computed", None]
    assert payload["line_origins"] == ["meta", "trace", "none"]
    assert payload["line_steps"][1] is not None
    sources = payload["trace"]["steps"][0]["observation"]["sources"]
    assert sources == [{"source": "bitget-mcp-server us_fundamentals", "at": "t", "ok": False,
                        "why": "isError in a 200"}]


def test_instrument_wraps_a_modules_engines_and_the_answerer_table_then_restores_them() -> None:
    name = "argus_trace_test_console"
    module = types.ModuleType(name)
    # a synthetic console module, so instrumenting touches nothing real
    exec(
        "def engine():\n    return ['Assumed: a week is used.']\n"
        "def _t(x):\n    return x\n"
        "_ANSWERERS = {'k': engine}\n", module.__dict__)
    sys.modules[name] = module
    original = module.engine
    try:
        with instrumented((name,), server=False):
            assert module.engine is not original
            assert module._ANSWERERS["k"] is module.engine
            assert not getattr(module._t, "__argus_traced__", False)  # a display transform
            with recording("q") as recorded:
                lines = module._ANSWERERS["k"]()
            assert _labels(recorded, lines) == [("assumed", "regex")]
        assert module.engine is original and module._ANSWERERS["k"] is original
    finally:
        del sys.modules[name]


def test_the_real_console_answers_with_a_trace_offline() -> None:
    """End to end: the console itself, every engine wrapped by the monkeypatched wrapper the audit
    uses, every outbound connection refused."""
    from argus.eval.trace_audit import _console_wired, offline
    from argus.lui.server import handle_ask

    with offline(), _console_wired(), recording("why did the desk stand aside on NVDA?") as rec:
        payload = handle_ask("why did the desk stand aside on NVDA?", [])
    attach(payload, rec)
    lines = payload["lines"]
    assert lines and len(payload["line_labels"]) == len(lines) == len(payload["line_origins"])
    rows = payload["trace"]["steps"]
    assert any(r["action"]["engine"].startswith("argus.lui.") for r in rows)
    indices = {r["index"] for r in rows}
    decided = [i for i, o in enumerate(payload["line_origins"]) if o == "trace"]
    assert decided, "no line of this answer was labelled by the step that made it"
    for i in decided:
        assert payload["line_labels"][i] in LABELS
        assert payload["line_steps"][i] in indices
    assert rows[0]["started_at"] <= rows[-1]["finished_at"]
    assert all(t.startswith(("Thought ", "Action ", "Observation ")) for t in rec.render())


def test_every_declaration_names_an_engine_that_exists_and_what_was_read() -> None:
    import importlib

    for qualified, (declared, why) in trace.DECLARED.items():
        module, _, attr = qualified.rpartition(".")
        assert callable(getattr(importlib.import_module(module), attr)), qualified
        assert callable(declared) and why and ":" in why, qualified
    assert every("live")(trace.Step("e", None, "t"), 0) == "live"


def test_answered_is_the_one_call_a_surface_makes() -> None:
    seen: list[str] = []

    def ask() -> dict[str, Any]:
        return {"lines": _beta_lines(), "sources": []}

    def finish(payload: dict[str, Any]) -> None:
        seen.append("finished before the labels were read")
        payload["lines"].append("Data: Bitget candles")

    payload = trace.answered("what does a 10% QQQ drop do?", ask, finish)
    assert seen and payload["line_labels"] == ["computed", None]
    assert payload["line_origins"] == ["trace", "meta"]
    assert payload["trace"]["question"] == "what does a 10% QQQ drop do?"
    assert current() is None


def test_payload_labels_keeps_the_trace_labels_and_falls_back_to_wording() -> None:
    from argus.lui.provenance import payload_labels

    lines = ["A sentence no wording rule knows about.", "Assumed: a week is used."]
    assert payload_labels({"lines": lines, "line_labels": ["desk", None]}) == ["desk", None]
    # absent, the wrong length, or not a label: the wording rules speak instead
    assert payload_labels({"lines": lines}) == [None, "assumed"]
    assert payload_labels({"lines": lines, "line_labels": ["desk"]}) == [None, "assumed"]
    assert payload_labels({"lines": lines, "line_labels": ["bogus", None]}) == [None, "assumed"]
