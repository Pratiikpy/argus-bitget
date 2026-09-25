"""S1: seeded, meaning-preserving perturbations, scored on the worst case beside resampling.

No test here reaches the model. The Meta-PM is driven by a scripted seat whose policy is a function
of the prompt, so each test states exactly what a sensitive or an insensitive decision-maker does.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from argus.eval import perturbations as pt
from argus.eval.artefact import is_strict, write
from argus.llm.qwen import QwenClient

AS_OF = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)


def snapshot(sid: str = "NVDAUSDT-a", evidence: tuple[str, ...] | None = None) -> pt.Snapshot:
    return pt.Snapshot(
        id=sid, symbol="NVDAUSDT", as_of=AS_OF, phase="rth", hours_to_discovery=0.0,
        nav_age_seconds=4.0, token_price=Decimal("219.37"), position=Decimal("0"),
        round_trip_bps=Decimal("12"), deliberation_bps=Decimal("1.93"), hedge_menu=(),
        evidence=evidence or (
            "[mkt-NVDAUSDT] (news, credibility 1.00, available 2026-09-25T13:59:58+00:00) "
            "NVDAUSDT last 219.37, 24h change 0.0043, quoted spread 0.46bps, 24h base volume "
            "1234567",
            "[vix] (macro, credibility 0.90, available 2026-09-25T13:00:00+00:00) VIX 16.2, "
            "day change -0.4%, percentile 41",
            "[fng] (social, credibility 0.35, available 2026-09-25T00:00:00+00:00) crypto "
            "fear-greed 62/100 greed",
        ),
    )


def pm_answer(verdict: str = "no_trade", side: str = "BUY", quantity: int = 0) -> dict[str, Any]:
    return {
        "verdict": verdict, "side": side, "quantity": quantity, "confidence": 0.6,
        "thesis": "t", "invalidation": ["x"], "counter_case": "c", "lean": "UP",
        "lean_confidence": 0.5,
    }


class ScriptedPM:
    """A decision-maker whose answer is a pure function of the frame it is shown."""

    budget = None

    def __init__(self, policy: Callable[[str], dict[str, Any]]) -> None:
        self.policy = policy
        self.calls = 0

    def complete_json(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.calls += 1
        return self.policy(str(messages[-1]["content"]))

    def complete(self, *_: Any, **__: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("the decision seat uses complete_json only")


def seat(policy: Callable[[str], dict[str, Any]]) -> pt.Seat:
    model = ScriptedPM(policy)
    return pt.Seat(model=model, requests=lambda: model.calls)


def outcome(verdict: str = "no_trade", side: str = "buy", qty: str = "0") -> pt.Outcome:
    return pt.Outcome(verdict=verdict, side=side, quantity=qty, lean="up", confidence=0.5,
                      thesis="t", reasoning="r")


# --- the perturbations ---------------------------------------------------------------------------


def test_unit_restatement_is_exact_and_leaves_dates_alone() -> None:
    text, edits = pt.restate_units(
        "spread 12.4bps, change +0.43% and -0.046%, at 2026-09-25T06:37:00+00:00, vol 1234567"
    )
    assert text == (
        "spread 0.124%, change +43bps and -4.6bps, at 2026-09-25T06:37:00+00:00, vol 1,234,567"
    )
    assert edits == 4


def test_unit_restatement_is_reversible_to_the_same_numbers() -> None:
    once, _ = pt.restate_units("hurdle 18.80bps, move 0.5%")
    twice, _ = pt.restate_units(once)
    assert twice == "hurdle 18.8bps, move 0.5%"


def test_restyle_keeps_every_field() -> None:
    line = snapshot().evidence[1]
    out = pt.restyle(line)
    assert out.startswith("VIX 16.2, day change -0.4%, percentile 41 — source macro;")
    for field in ("credibility 0.90", "2026-09-25T13:00:00+00:00", "ref vix"):
        assert field in out
    assert pt.restyle("no header here") == "no header here"


def test_reorder_is_seeded_per_snapshot_like_helm() -> None:
    a = pt.Reorder().apply(snapshot("A"), seed=0)
    assert a.evidence == pt.Reorder().apply(snapshot("A"), seed=0).evidence
    assert a.evidence != snapshot("A").evidence
    assert sorted(a.evidence) == sorted(snapshot("A").evidence)
    # HELM seeds from str(seed) + id, so the same seed on another snapshot draws another order.
    draw = pt.Reorder().rng
    assert draw(snapshot("A"), 3).random() == draw(snapshot("A"), 3).random()
    assert draw(snapshot("A"), 3).random() != draw(snapshot("B"), 3).random()


def test_perturbations_never_mutate_the_original() -> None:
    snap = snapshot()
    for p in pt.PERTURBATIONS:
        p.apply(snap, seed=1)
    assert snap == snapshot()


def test_snapshot_round_trips_and_builds_the_production_frame() -> None:
    snap = snapshot()
    again = pt.Snapshot.from_dict(json.loads(json.dumps(snap.as_dict())))
    assert again == snap
    frame = again.frame()
    assert frame.evidence == snap.evidence
    assert "TOTAL HURDLE: 13.93 bps" in frame.to_prompt_block()


# --- the decision and its action -----------------------------------------------------------------


def test_an_abstention_has_no_side() -> None:
    assert outcome("no_trade", "buy").action == outcome("no_trade", "sell").action
    assert outcome("trade", "buy", "2").action != outcome("trade", "sell", "2").action


def test_the_seat_asks_the_production_meta_pm() -> None:
    prompts: list[str] = []

    def policy(prompt: str) -> dict[str, Any]:
        prompts.append(prompt)
        return pm_answer("trade", "SELL", 3)

    got = seat(policy).decide(snapshot(), "d1")
    assert got is not None and got.action == ("trade", "sell", "3")
    assert "EVIDENCE (each item was available at or before AS OF)" in prompts[0]


def test_the_request_ceiling_refuses_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITGET_QWEN_API_KEY", "test-only-not-a-key")
    monkeypatch.setenv("BITGET_QWEN_BASE_URL", "http://127.0.0.1:9")

    def never(*_: Any, **__: Any) -> Any:
        raise AssertionError("a request was sent past the ceiling")

    monkeypatch.setattr(QwenClient, "_post", never)
    client = pt.CappedQwen(cap=0)
    capped = pt.Seat(model=client, requests=lambda: client.requests)
    assert capped.decide(snapshot(), "d1") is None
    assert capped.failures == ["d1: not run, request ceiling reached"]
    assert client.requests == 0


def test_spent_sums_every_budget_artefact(tmp_path: Path) -> None:
    (tmp_path / "robustness_baseline.json").write_text(json.dumps({"qwen_requests": 18}))
    (tmp_path / "vocab_stress.json").write_text(json.dumps({"qwen_requests": 12}))
    (tmp_path / "unrelated.json").write_text(json.dumps({"qwen_requests": 999}))
    assert pt.spent(tmp_path) == 30
    assert pt.spent(tmp_path, excluding="vocab_stress.json") == 18


# --- scoring -------------------------------------------------------------------------------------


def test_worst_case_separates_perturbation_flips_from_noise() -> None:
    trade = outcome("trade", "buy", "2")
    hold = outcome()
    base = pt.Baseline(runs={
        "steady": (hold, hold, hold),        # unanimous: any flip here is the perturbation's
        "noisy": (hold, hold, trade),        # already split: a flip here is within noise
    })
    cells = [
        pt.Cell("steady", "reorder", trade, edits=1),
        pt.Cell("steady", "restyle", hold, edits=3),
        pt.Cell("noisy", "reorder", trade, edits=1),
    ]
    score = pt.worst_case(base, cells)
    assert score.snapshots == 2
    assert score.before == pytest.approx((1 + 2 / 3) / 2)
    assert score.resample_worst == 0.5          # the noisy snapshot fails on resampling alone
    assert score.worst == 0.0                   # and the perturbation breaks the steady one
    assert score.per_condition_worst == {"reorder": 0.0, "restyle": 1.0}
    assert score.clean_pair_disagreement == (2, 6)
    assert score.cross_pair_disagreement == (3 + 0 + 2, 9)
    assert score.flips_beyond_noise == ("steady/reorder: no_trade - 0 -> trade buy 2",)
    assert len(score.flips_within_noise) == 1


def test_a_snapshot_without_an_unperturbed_run_is_not_scored() -> None:
    score = pt.worst_case(pt.Baseline(runs={}), [pt.Cell("x", "reorder", outcome())])
    assert score.snapshots == 0 and score.before is None
    assert pt.verdict(score, what="perturbed", cells=1).startswith("UNDEFINED")


def test_an_order_sensitive_desk_is_caught_and_an_insensitive_one_is_not() -> None:
    snaps = [snapshot("A"), snapshot("B")]

    def first_line_decides(prompt: str) -> dict[str, Any]:
        # Trades whenever the price line is the first evidence item: order-dependent by design.
        block = prompt.split("EVIDENCE (each item was available at or before AS OF)\n", 1)[1]
        first = block.splitlines()[0]
        return pm_answer("trade", "BUY", 1) if "mkt-" in first else pm_answer()

    sensitive = seat(first_line_decides)
    base = pt.collect_baseline(snaps, sensitive, runs=3)
    cells = pt.perturbed_cells(snaps, sensitive)
    score = pt.worst_case(base, cells)
    assert score.resample_worst == 1.0
    assert score.worst == 0.0
    assert score.flips_beyond_noise

    steady = seat(lambda _: pm_answer())
    base = pt.collect_baseline(snaps, steady, runs=3)
    score = pt.worst_case(base, pt.perturbed_cells(snaps, steady))
    assert score.worst == 1.0 and not score.flips_beyond_noise
    assert "not a bound on the flip rate" in pt.verdict(score, what="perturbed", cells=8)


def test_the_baseline_round_robin_and_report_are_strict_json(tmp_path: Path) -> None:
    order: list[str] = []

    def policy(prompt: str) -> dict[str, Any]:
        order.append("A" if "A-marker" in prompt else "B")
        return pm_answer()

    snaps = [snapshot("A", ("A-marker",)), snapshot("B", ("B-marker",))]
    s = seat(policy)
    base = pt.collect_baseline(snaps, s, runs=3)
    assert order == ["A", "B", "A", "B", "A", "B"]
    assert pt.Baseline.from_dict(json.loads(json.dumps(base.as_dict()))) == base
    blob = pt.report(base, pt.perturbed_cells(snaps, s), requests=s.requests(), failures=[])
    path = tmp_path / "r.json"
    write(path, blob)
    assert is_strict(path)
    assert blob["qwen_requests"] == 14


def test_the_console_summary_reports_absence_as_absence(tmp_path: Path) -> None:
    got = pt.summary(tmp_path)
    assert got["lines"] == [
        "S1 perturbation: not run", "S2 renamed vocabulary: not run",
        "S3 corrupted feed: not run",
    ]
    assert got["qwen_requests"] == 0 and got["cap"] == pt.QWEN_CAP
    (tmp_path / "perturbation_robustness.json").write_text(
        json.dumps({"verdict": "3 snapshot(s)...", "qwen_requests": 7})
    )
    got = pt.summary(tmp_path)
    assert got["lines"][0] == "S1 perturbation: 3 snapshot(s)..." and got["qwen_requests"] == 7


def test_the_lean_is_scored_as_its_own_level() -> None:
    def lean(value: str) -> pt.Outcome:
        return pt.Outcome(verdict="no_trade", side="buy", quantity="0", lean=value,
                          confidence=0.5, thesis="t", reasoning="r")

    base = pt.Baseline(runs={"s": (lean("up"), lean("up"), lean("up"))})
    cells = [pt.Cell("s", "reorder", lean("down"), edits=1)]
    assert pt.worst_case(base, cells).worst == 1.0          # the action never moved
    view = pt.worst_case(base, cells, pt.LEAN)
    assert view.worst == 0.0 and view.flips_beyond_noise == ("s/reorder: up -> down",)
