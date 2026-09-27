"""Pins the counting logic in `argus.eval.honesty_eval`, not `lui/honesty.py`'s own detection.

`regrade()` re-grades a hand-built artefact row through `argus.lui.honesty` and
`argus.eval.infeasibilitybench.grade_infeasible`; both are stubbed with fixed replies so the branch
that is taken (the honesty answer, the order prefix, or the console unchanged) and the before/after
Counter bucketing can be worked out by hand. `false_positives()` is pinned the same way over a
hand-built corpus and control set, plus one test that runs it for real against the committed
corpora — `honesty.detect`/`order_prefix` never read the network, so that one needs no stub, and it
checks only that the counts are internally consistent (never a fixed number that today's corpora
happen to produce).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from argus.eval import honesty_eval
from argus.eval import infeasibilitybench as ib
from argus.lui import honesty


def _row(cid: str, cause: str, outcome: str, *, valid: bool = True, refused: bool = False,
         reason: str = "", lines: list[str] | None = None,
         group: str = "infeasible") -> dict[str, Any]:
    return {"id": cid, "group": group, "truth": {"valid": valid}, "cause": cause,
            "outcome": outcome, "refused": refused, "reason": reason, "lines": lines or []}


class TestRegradeWithHonestyData:
    def test_buckets_before_and_after_by_cause_and_skips_invalid_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        report_path = tmp_path / "infeasibility_bench.json"
        monkeypatch.setattr(ib, "REPORT_PATH", report_path)
        monkeypatch.setattr(ib, "INFEASIBLE", (
            ib.Infeasible(id="c1", ask="ask1", cause=ib.Cause.ORDER),
            ib.Infeasible(id="c2", ask="ask2", cause=ib.Cause.FUTURE_PRICE),
            ib.Infeasible(id="c3", ask="ask3", cause=ib.Cause.PRIVATE),
        ))
        report_path.write_text(json.dumps({"rows": [
            _row("c1", "order", "fabricated", lines=["orig line 1"]),
            _row("c2", "future_price", "declined_wrong_reason", refused=True,
                 reason="I did not understand", lines=["I did not understand"]),
            # invalid truth: excluded from both before and after, and never answered.
            _row("c3", "private_data", "fabricated", valid=False, lines=["should be excluded"]),
            # a control row with the same id shape: excluded by the group filter alone.
            _row("other", "x", "y", group="control"),
        ]}), encoding="utf-8")

        def fake_honest_answer(text: str, *, prior: list[str], book: str,
                               today: Any = None) -> tuple[str, list[str]] | None:
            if text == "ask1":
                return None
            if text == "ask2":
                return "future_price", ["Bottom line: honesty says no."]
            raise AssertionError(f"unexpected honest_answer call: {text}")

        def fake_order_prefix(text: str) -> str | None:
            if text == "ask1":
                return "ORDER PREFIX TEXT"
            if text == "ask2":
                return None
            raise AssertionError(f"unexpected order_prefix call: {text}")

        grades = {"c1": SimpleNamespace(outcome="declined_right_reason", matched="never places"),
                  "c2": SimpleNamespace(outcome="fabricated", matched="")}

        monkeypatch.setattr(honesty, "honest_answer", fake_honest_answer)
        monkeypatch.setattr(honesty, "order_prefix", fake_order_prefix)
        monkeypatch.setattr(ib, "grade_infeasible",
                            lambda case, payload, *, probe: grades[case.id])

        result = honesty_eval.regrade(data=True)
        assert result["before"] == {
            "order": {"fabricated": 1}, "future_price": {"declined_wrong_reason": 1},
            "ALL": {"fabricated": 1, "declined_wrong_reason": 1},
        }
        assert result["after"] == {
            "order": {"declined_right_reason": 1}, "future_price": {"fabricated": 1},
            "ALL": {"declined_right_reason": 1, "fabricated": 1},
        }
        assert [r["id"] for r in result["rows"]] == ["c1", "c2"]
        row1, row2 = result["rows"]
        assert row1["answered_by"] == "order_prefix+console"
        assert row1["lines"] == ["ORDER PREFIX TEXT", "orig line 1"]
        assert row1["cause_agrees"] is None  # honest_answer gave no cause for the order case
        assert row2["answered_by"] == "honesty"
        assert row2["honesty_cause"] == "future_price" and row2["cause_agrees"] is True


class TestRegradeDetectionOnly:
    def test_reports_detected_right_or_other_cause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        report_path = tmp_path / "infeasibility_bench.json"
        monkeypatch.setattr(ib, "REPORT_PATH", report_path)
        monkeypatch.setattr(ib, "INFEASIBLE", (
            ib.Infeasible(id="da", ask="askA", cause=ib.Cause.FUTURE_PRICE),
            ib.Infeasible(id="db", ask="askB", cause=ib.Cause.HORIZON),
        ))
        report_path.write_text(json.dumps({"rows": [
            _row("da", "future_price", "fabricated", lines=["x"]),
            _row("db", "horizon_exceeds_data", "fabricated", lines=["y"]),
        ]}), encoding="utf-8")

        def fake_detect(text: str, *, prior: list[str], book: str, today: Any = None) -> Any:
            if text == "askA":
                return SimpleNamespace(cause="future_price")
            if text == "askB":
                return SimpleNamespace(cause="unlisted")  # the wrong cause for this row
            raise AssertionError(f"unexpected detect call: {text}")

        monkeypatch.setattr(honesty, "detect", fake_detect)
        monkeypatch.setattr(honesty, "order_prefix", lambda text: None)
        monkeypatch.setattr(
            ib, "grade_infeasible",
            lambda case, payload, *, probe: SimpleNamespace(outcome="fabricated", matched=""))

        result = honesty_eval.regrade(data=False)
        assert result["after"] == {
            "future_price": {"detected_right_cause": 1},
            "horizon_exceeds_data": {"detected_other_cause": 1},
            "ALL": {"detected_right_cause": 1, "detected_other_cause": 1},
        }
        by_id = {r["id"]: r for r in result["rows"]}
        assert by_id["da"]["answered_by"] == "honesty (detection only)"
        assert by_id["da"]["lines"] == ["detected: future_price"]
        assert by_id["db"]["honesty_cause"] == "unlisted"


class TestFalsePositives:
    def test_counts_hits_and_refuse_detections_across_corpora_and_controls(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        corpus_a = tmp_path / "corpusA.jsonl"
        corpus_a.write_text("\n".join([
            json.dumps({"text": "q1", "expected": "quote"}),
            json.dumps({"text": "q2", "expected": "quote"}),
            json.dumps({"text": "q3", "expected": "refuse"}),
        ]) + "\n", encoding="utf-8")
        corpus_b = tmp_path / "corpusB.jsonl"
        corpus_b.write_text("\n".join([
            json.dumps({"text": "q4", "expected": "hedge"}),
            json.dumps({"text": "q5", "expected": "refuse"}),
        ]) + "\n", encoding="utf-8")
        monkeypatch.setattr(honesty_eval, "CORPORA", (corpus_a, corpus_b))

        def fake_detect(text: str, *, prior: list[str], book: str) -> Any:
            return {"q2": SimpleNamespace(cause="future_price"),
                    "c2": SimpleNamespace(cause="private_data")}.get(text)

        def fake_order_prefix(text: str) -> str | None:
            return {"q3": "PREFIX"}.get(text)

        monkeypatch.setattr(honesty, "detect", fake_detect)
        monkeypatch.setattr(honesty, "order_prefix", fake_order_prefix)
        monkeypatch.setattr(ib, "controls", lambda path=ib.CORPUS_PATH: [
            ib.Control(id="ctl1", ask="c1", origin="x", kind="quote"),
            ib.Control(id="ctl2", ask="c2", origin="x", kind="chip"),
        ])

        result = honesty_eval.false_positives()
        assert result["answerable_scanned"] == 5  # q1, q2, q4 (answerable) + ctl1, ctl2
        assert result["false_positives"] == 2
        assert result["hits"] == [
            {"corpus": "corpusA.jsonl", "kind": "quote", "text": "q2", "hit": "future_price"},
            {"corpus": "infeasibility_bench controls", "kind": "chip", "text": "c2",
             "hit": "private_data"},
        ]
        assert result["refuse_rows"] == 2
        assert result["refuse_rows_detected"] == 1
        assert result["refuse_detected_by_cause"] == {"order_prefix": 1}

    def test_no_rows_and_no_controls_reports_zero(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(honesty_eval, "CORPORA", ())
        monkeypatch.setattr(ib, "controls", lambda path=ib.CORPUS_PATH: [])
        assert honesty_eval.false_positives() == {
            "answerable_scanned": 0, "false_positives": 0, "hits": [], "refuse_rows": 0,
            "refuse_rows_detected": 0, "refuse_detected_by_cause": {},
            "detection_ms_total": pytest.approx(0.0, abs=50.0),
        }

    def test_on_the_real_corpora_the_counts_are_internally_consistent(self) -> None:
        """No stubbing: `honesty.detect`/`order_prefix` never open a socket, so this runs the real
        committed corpora and controls and checks the arithmetic holds — never a fixed count,
        which would drift as the corpora are revised."""
        result = honesty_eval.false_positives()
        assert result["false_positives"] == len(result["hits"])
        assert result["refuse_rows_detected"] == sum(result["refuse_detected_by_cause"].values())
        assert 0 <= result["false_positives"] <= result["answerable_scanned"]
        assert 0 <= result["refuse_rows_detected"] <= result["refuse_rows"]
        assert all(set(hit) == {"corpus", "kind", "text", "hit"} for hit in result["hits"])


class TestRun:
    def test_combines_detection_offline_regrade_and_scan_into_one_artefact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        report_path = tmp_path / "infeasibility_bench.json"
        monkeypatch.setattr(ib, "REPORT_PATH", report_path)
        monkeypatch.setattr(ib, "INFEASIBLE", (
            ib.Infeasible(id="c1", ask="ask1", cause=ib.Cause.ORDER),
            ib.Infeasible(id="c2", ask="ask2", cause=ib.Cause.FUTURE_PRICE),
        ))
        report_path.write_text(json.dumps({"rows": [
            _row("c1", "order", "fabricated", lines=["orig line 1"]),
            _row("c2", "future_price", "declined_wrong_reason", refused=True,
                 reason="I did not understand", lines=["I did not understand"]),
        ]}), encoding="utf-8")
        monkeypatch.setattr(honesty_eval, "CORPORA", ())
        monkeypatch.setattr(ib, "controls", lambda path=ib.CORPUS_PATH: [])

        def fake_honest_answer(text: str, *, prior: list[str], book: str,
                               today: Any = None) -> tuple[str, list[str]] | None:
            return None if text == "ask1" else ("future_price", ["Bottom line: honesty says no."])

        def fake_detect(text: str, *, prior: list[str], book: str, today: Any = None) -> Any:
            return None if text == "ask1" else SimpleNamespace(cause="future_price")

        grades = {"c1": SimpleNamespace(outcome="declined_right_reason", matched="never places"),
                  "c2": SimpleNamespace(outcome="fabricated", matched="")}
        monkeypatch.setattr(honesty, "honest_answer", fake_honest_answer)
        monkeypatch.setattr(honesty, "detect", fake_detect)
        monkeypatch.setattr(honesty, "order_prefix",
                            lambda text: "ORDER PREFIX TEXT" if text == "ask1" else None)
        monkeypatch.setattr(ib, "grade_infeasible",
                            lambda case, payload, *, probe: grades[case.id])

        out_path = tmp_path / "honesty_eval.json"
        result = honesty_eval.run(out=out_path)
        assert set(result) == {
            "generated_at", "item", "method", "detection_offline", "regrade",
            "false_positive_scan", "call_site",
        }
        assert result["call_site"] == honesty_eval.CALL_SITE
        assert result["detection_offline"] == {
            "order": {"declined_right_reason": 1}, "future_price": {"detected_right_cause": 1},
            "ALL": {"declined_right_reason": 1, "detected_right_cause": 1},
        }
        assert result["regrade"]["after"] == {
            "order": {"declined_right_reason": 1}, "future_price": {"fabricated": 1},
            "ALL": {"declined_right_reason": 1, "fabricated": 1},
        }
        assert result["false_positive_scan"]["answerable_scanned"] == 0
        datetime.fromisoformat(result["generated_at"])
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        assert on_disk == result
