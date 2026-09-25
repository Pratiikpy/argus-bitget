"""The eval spine: evaluators (S8), comparison reports (S8), the harness canary (S8), metaeval (S9).

The migration tests are the fix-to-pass / pass-to-pass check on eval code itself: the three
harnesses moved onto the spine (``claimcheck``, ``void``, ``overnight``) are re-scored from their
saved inputs and compared with the artefacts they wrote *before* the migration, frozen in
``tests/data/spine_legacy/``. Every field must match, and every verdict must be the one the old
artefact (and ``standing.py``'s prose) gave.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from argus.eval import compare, evaluators, harness_validity, metaeval
from argus.eval.compare import ComparisonReport, Outcome
from argus.eval.evaluators import Tier

LEGACY = Path(__file__).resolve().parent / "data" / "spine_legacy"


# --------------------------------------------------------------------------------------------
# evaluators.py


def test_evaluators_are_combined_with_and_by_multiplying_scores() -> None:
    grade = evaluators.combine((evaluators.rule("a", lambda _s: 1.0),
                                evaluators.statistic("b", lambda _s: 0.5)), None,
                               stop_at_first_failure=False)
    assert not grade.passed
    assert grade.score == 0.5
    assert grade.first_failure is not None and grade.first_failure.name == "b"


def test_a_model_judge_runs_last_and_never_after_a_cheap_rule_failed() -> None:
    calls: list[str] = []

    def judge(_s: object) -> bool:
        calls.append("judge")
        return True

    grade = evaluators.combine((evaluators.model_judge("judge", judge),
                                evaluators.rule("free", lambda _s: (False, "empty answer"))),
                               None, stop_at_first_failure=False)
    assert calls == []
    assert grade.model_calls == 0
    assert [c.name for c in grade.checks] == ["free", "judge"]
    assert grade.checks[1].skipped and grade.checks[1].tier is Tier.MODEL


def test_the_model_judge_is_paid_for_only_when_every_rule_passed() -> None:
    grade = evaluators.combine((evaluators.model_judge("judge", lambda _s: True),
                                evaluators.rule("free", lambda _s: True)), None)
    assert grade.passed and grade.model_calls == 1


def test_an_evaluator_that_raises_fails_the_and_with_its_location() -> None:
    def broken(_s: object) -> bool:
        raise KeyError("score")

    grade = evaluators.combine((evaluators.rule("broken", broken),), None)
    assert not grade.passed
    check = grade.checks[0]
    assert check.passed is None and "KeyError" in check.detail
    assert "test_eval_spine.py:" in check.detail


def test_all_or_nothing_publishes_no_aggregate_when_a_component_is_missing() -> None:
    whole = evaluators.all_or_nothing({"a": 1.0, "b": 0.0})
    assert whole.value == 0.5 and whole.complete
    part = evaluators.all_or_nothing({"a": 1.0, "b": None, "c": 0.0})
    assert part.value is None and part.missing == ("b",) and part.partial == 0.5
    assert evaluators.all_or_nothing({}).value is None


def test_a_grade_round_trips_through_json() -> None:
    grade = evaluators.combine((evaluators.rule("x", lambda _s: (True, "fine")),), None)
    again = evaluators.Grade.from_dict(json.loads(json.dumps(grade.as_dict())))
    assert again == grade


# --------------------------------------------------------------------------------------------
# compare.py


def _legacy_loop(by_unit: dict[str, list[float]], *, draws: int, seed: int) -> tuple[float, ...]:
    """The loop void/overnight carried before the migration, copied from their pre-spine source."""
    keys = sorted(by_unit)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [d for k in (rng.choice(keys) for _ in keys) for d in by_unit[k]]
        means.append(statistics.mean(sample))
    means.sort()
    point = statistics.mean(d for k in keys for d in by_unit[k])
    return point, means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def test_the_spine_bootstrap_is_the_legacy_loop_bit_for_bit() -> None:
    rng = random.Random(3)
    by_unit = {f"u{i}": [rng.gauss(0.2, 1) for _ in range(rng.randint(1, 5))] for i in range(40)}
    boot = compare.paired_bootstrap(by_unit, draws=2000, seed=7)
    assert (boot.mean_diff, boot.low, boot.high) == _legacy_loop(by_unit, draws=2000, seed=7)


def test_sign_test_is_exact() -> None:
    assert compare.sign_test(0, 0) == 1.0
    assert compare.sign_test(0, 6) == pytest.approx(2 / 64)
    assert compare.sign_test(3, 3) == 1.0
    assert compare.sign_test_outcome(0, 0) is Outcome.LEVEL
    assert compare.sign_test_outcome(6, 0) is Outcome.ARGUS_BETTER
    assert compare.sign_test_outcome(0, 6) is Outcome.RIVAL_BETTER
    assert compare.sign_test_outcome(4, 1) is Outcome.NOT_SEPARABLE


def _report(**over: Any) -> ComparisonReport:
    base: dict[str, Any] = dict(
        comparison="t", question="q", argus="a", rival="r", metric="mae", lower_is_better=True,
        argus_score=1.0, rival_score=2.0, n=10, unit="night", outcome=Outcome.ARGUS_BETTER,
        basis="b", ci95=(-2.0, -0.5), scored=10, total=12)
    base.update(over)
    return ComparisonReport(**base)


def test_a_verdict_that_contradicts_its_own_interval_is_demoted() -> None:
    fine = compare.finalise(_report())
    assert fine.valid and fine.outcome is Outcome.ARGUS_BETTER
    lying = compare.finalise(_report(ci95=(-1.0, 0.5)))
    assert not lying.valid and lying.outcome is Outcome.NO_RESULT
    assert lying.unvalidated_outcome is Outcome.ARGUS_BETTER


def test_a_missing_side_is_no_result_not_a_win() -> None:
    r = compare.finalise(_report(rival_score=None))
    assert r.outcome is Outcome.NO_RESULT
    assert r.validity is not None and r.validity.first_failure is not None
    assert r.validity.first_failure.name == "both_sides_scored"


def test_a_report_exists_even_when_scoring_raises() -> None:
    def build() -> ComparisonReport:
        raise FileNotFoundError("inputs.json")

    r = compare.emit(build, comparison="t", question="q", argus="a", rival="r", metric="m",
                     lower_is_better=True, unit="night")
    assert r.outcome is Outcome.NO_RESULT and not r.valid and not r.inputs_exist
    assert "FileNotFoundError" in r.errors[0] and "test_eval_spine.py:" in r.errors[0]


def test_every_group_is_all_or_nothing() -> None:
    won = compare.finalise(_report(groups={"A": "argus_better", "B": "argus_better"}))
    assert won.every_group == "argus_better_on_every_group"
    gap = compare.finalise(_report(groups={"A": "argus_better", "B": "no_result"}))
    assert gap.every_group is None
    mixed = compare.finalise(_report(groups={"A": "argus_better", "B": "level"}))
    assert mixed.every_group == "mixed"


def test_a_comparison_report_round_trips() -> None:
    r = compare.finalise(_report(groups={"A": "level"}))
    assert ComparisonReport.from_dict(json.loads(json.dumps(r.to_dict()))) == r


# --------------------------------------------------------------------------------------------
# Migration: old verdict == new verdict, on the saved artefacts.


def _legacy(name: str) -> dict[str, Any]:
    blob: dict[str, Any] = json.loads((LEGACY / f"{name}_comparison.json").read_text("utf-8"))
    return blob


def _rescored(module: types.ModuleType) -> dict[str, Any]:
    with mock.patch("argus.eval.artefact.write"):
        blob: dict[str, Any] = json.loads(json.dumps(module.score()))
    return blob


VOLATILE = {"generated"}
PROSE_CHANGED_BEFORE_MIGRATION = {"claimcheck": {"first_run_loss"}}
"""claimcheck's ``first_run_loss`` sentence was reworded in the source after its artefact was
written, before this migration; the re-scored text is the current source's, the figures agree."""

CONSOLE_CALLED = {
    "claimcheck": ({"replay"}, set()),
    "void": ({"console_replay"}, {"summary", "per_weekend"}),
    "overnight": ({"console_replay"}, {"summary", "paired", "per_stock", "ablation",
                                       "sunday_evening_vs_nocturne_claim"}),
}
"""(keys added, keys whose figures moved) when the three harnesses stopped scoring their own copy
of the console's arithmetic and started calling `lui/research` itself (2026-09-26). claimcheck's
replay of ``_claim_check`` gave the recorded verdict on all 38 claims, so nothing moved; void and
overnight read the console's implied price back from its sentence, printed to the cent, so their
ARGUS figures moved by that rounding (at most 0.5bps a row). The verdicts did not move, which the
test checks outright."""


@pytest.mark.parametrize(("name", "expected"), [
    ("claimcheck", [Outcome.LEVEL]),
    ("void", [Outcome.NOT_SEPARABLE, Outcome.NOT_SEPARABLE]),
    ("overnight", [Outcome.ARGUS_BETTER, Outcome.ARGUS_BETTER, Outcome.ARGUS_BETTER]),
])
def test_the_migrated_harness_reproduces_its_pre_migration_artefact_and_verdict(
        name: str, expected: list[Outcome]) -> None:
    module = __import__(f"argus.eval.{name}_comparison", fromlist=["score"])
    old, new = _legacy(name), _rescored(module)
    added, moved = CONSOLE_CALLED[name]
    skip = VOLATILE | PROSE_CHANGED_BEFORE_MIGRATION.get(name, set()) | moved
    assert {k: v for k, v in old.items() if k not in skip} == \
           {k: v for k, v in new.items() if k in old and k not in skip}
    assert set(new) - set(old) == {"comparison_reports"} | added
    from_old = [r.outcome for r in module.comparison_reports(old)]
    from_new = [ComparisonReport.from_dict(r).outcome for r in new["comparison_reports"]]
    assert from_old == from_new == expected
    assert all(r["valid"] for r in new["comparison_reports"])


def test_the_claimcheck_verdict_is_the_tie_standing_records() -> None:
    old = _legacy("claimcheck")
    assert old["argus_correct"] == old["mirrorline_correct"] == old["graded"] == 30
    (report,) = __import__("argus.eval.claimcheck_comparison",
                           fromlist=["x"]).comparison_reports(old)
    assert report.outcome is Outcome.LEVEL and report.p_value == 1.0
    assert report.every_group == "level_on_every_group"


def test_the_paired_verdict_strings_are_unchanged_by_the_migration() -> None:
    for name in ("void", "overnight"):
        module = __import__(f"argus.eval.{name}_comparison", fromlist=["score"])
        old, new = _legacy(name), _rescored(module)
        assert [p["verdict"] for p in old["paired"]] == [p["verdict"] for p in new["paired"]]


# --------------------------------------------------------------------------------------------
# harness_validity.py


def test_code_under_test_excludes_the_harness_and_rival_code() -> None:
    credited = ["argus/eval/overnight_comparison.py", "argus/eval/void_comparison.py",
                "argus/eval/baselines/gloaming_fairvalue.py", "argus/desk/odds.py"]
    assert harness_validity.code_under_test("void_comparison", credited) == ["argus/desk/odds.py"]


def test_standing_credits_are_read_from_the_register() -> None:
    claims = harness_validity.claimed_modules("stopquality_comparison")
    assert any("argus/desk/odds.py" in paths for paths in claims.values())


@pytest.mark.parametrize("oracle", sorted(harness_validity.ORACLES))
def test_each_migrated_scorer_gives_a_perfect_desk_full_marks(oracle: str) -> None:
    result = harness_validity.ORACLES[oracle]()
    assert result["oracle_perfect"] and result["anti_oracle_worse"], result


def test_sabotage_replaces_functions_and_methods_with_canaries() -> None:
    module = types.ModuleType("argus_fake_target")
    exec("def f():\n    return 1\nclass C:\n    def m(self):\n        return 2\n"
         "    def __init__(self):\n        pass\n", module.__dict__)
    guards = harness_validity._Guards()
    assert harness_validity._sabotage(module, guards) == 2
    with pytest.raises(harness_validity.CanaryError, match=r"---CANARY_argus_fake_target\.f---"):
        module.f()
    instance = module.C()  # dunder methods are left alone so objects still construct
    with pytest.raises(harness_validity.CanaryError):
        instance.m()
    assert guards.fired == ["---CANARY_argus_fake_target.f---",
                            "---CANARY_argus_fake_target.C.m---"]


TOKEN = "---CANARY_argus.desk.odds.stop_line---"


def _run(**canary: Any) -> harness_validity.HarnessRun:
    return harness_validity.HarnessRun(
        "x_comparison", {"cap": ["argus/desk/odds.py"]}, ["argus/desk/odds.py"], None,
        {"status": "ok", "network_attempts": 0}, canary)


def test_a_swallowed_canary_fails_and_a_surfaced_one_passes() -> None:
    swallowed = _run(status="ok", fired=[TOKEN], fired_in_entry=[TOKEN], returned='{"ok": 1}')
    assert swallowed.reached() == ["argus/desk/odds.py"]
    assert swallowed.surfaced() is False and not harness_validity.grade(swallowed).passed
    shown = _run(status="ok", fired=[TOKEN], fired_in_entry=[TOKEN], returned=f'{{"e": "{TOKEN}"}}')
    assert shown.surfaced() is True and harness_validity.grade(shown).passed
    raised = _run(status="raised during entry", fired=[TOKEN], fired_in_entry=[TOKEN])
    assert raised.surfaced() is True and harness_validity.grade(raised).passed
    at_import = _run(status="raised during import", fired=[TOKEN], fired_at_import=True)
    assert at_import.surfaced() is None and at_import.reached() == []
    assert not harness_validity.grade(at_import).passed
    never = _run(status="ok", fired=[], fired_in_entry=[], returned="{}")
    assert never.reached() == [] and never.surfaced() is None


def test_a_real_canary_run_is_offline_and_never_rewrites_a_published_artefact() -> None:
    published = harness_validity.DATA / "claimcheck_comparison.json"
    before = hashlib.sha256(published.read_bytes()).hexdigest()
    run = harness_validity.measure("claimcheck_comparison", timeout=300)
    assert hashlib.sha256(published.read_bytes()).hexdigest() == before
    assert run.baseline["status"] == "ok" and run.baseline["network_attempts"] == 0
    assert "argus/data/claimcheck_comparison.json" in run.baseline["writes_redirected"]


@pytest.mark.parametrize(("harness", "module", "function"), [
    ("claimcheck_comparison", "argus/lui/research.py", "argus.lui.research."),
    ("stopquality_comparison", "argus/desk/odds.py", "argus.desk.odds.directional_odds"),
    ("void_comparison", "argus/lui/research.py", "argus.lui.research._implied_open_line"),
])
def test_the_harness_runs_the_console_code_and_a_sabotaged_console_surfaces(
        harness: str, module: str, function: str) -> None:
    """On 2026-09-25 the canary found these harnesses never ran the console code standing credits
    them for. They now call it, so sabotaging that module must fire a canary inside the entry point
    and fail the run. Measured against the module directly, whatever `standing.py` credits today.
    (overnight is the same call as void through `overnight_comparison.console_implied_open`; it is
    left out only for its 30-second run.)"""
    canary = harness_validity._run_child(harness, "canary", [module], 300)
    assert canary["status"] == "raised during entry", canary.get("error")
    assert any(token.startswith(f"---CANARY_{function}") for token in canary["fired_in_entry"])
    assert canary["network_attempts"] == 0


# --------------------------------------------------------------------------------------------
# metaeval.py


def _row(i: int, got: str, want: str, kind: str = "human", judge: str = "j") -> dict[str, str]:
    return {"judge": judge, "item_id": str(i), "input": "q", "judge_label": got,
            "reference_label": want, "labeller": "L", "labeller_kind": kind,
            "labelled_at": "2026-09-25", "source": "s"}


def test_cohens_kappa_on_a_textbook_table() -> None:
    pairs = ([("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15)
    assert metaeval.cohens_kappa(pairs) == pytest.approx(0.4)
    assert metaeval.cohens_kappa([("a", "a")] * 5) is None
    assert metaeval.cohens_kappa([]) is None


def test_raw_agreement_flatters_a_judge_on_a_skewed_set_and_kappa_does_not() -> None:
    rows = [_row(i, "none", "none") for i in range(45)] + \
           [_row(45 + i, "none", "buy") for i in range(5)]
    result = metaeval.agreement("j", rows)
    assert result.metascore == 0.9 and result.kappa == 0.0


@pytest.mark.parametrize(("n", "kind", "status"), [
    (60, "human", metaeval.Status.MEASURED),
    (20, "human", metaeval.Status.PROVISIONAL),
    (60, "author", metaeval.Status.NOT_HUMAN),
    (0, "human", metaeval.Status.UNMEASURED),
])
def test_only_enough_human_labels_make_a_judge_measured(
        n: int, kind: str, status: metaeval.Status) -> None:
    rows = [_row(i, "a" if i % 3 else "b", "a" if i % 4 else "b", kind) for i in range(n)]
    assert metaeval.agreement("j", rows).status is status


def test_the_labelled_set_format_is_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="labeller_kind"):
        metaeval.validate(_row(1, "a", "a", kind="crowd"))
    with pytest.raises(ValueError, match="missing"):
        metaeval.validate({"judge": "j"})
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(json.dumps(_row(1, "a", "a")) for _ in range(2)), "utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        metaeval.load_labels(path)


def test_the_router_set_comes_from_recorded_routings_and_is_not_called_human() -> None:
    rows = metaeval.router_labels_from_bench()
    bench = json.loads((metaeval.DATA / "router_bench.json").read_text("utf-8"))
    reached = [c for c in bench["outcomes"] if c["reached_router"]]
    assert len(rows) == len(reached) == 24
    assert [r["judge_label"] for r in rows] == [c["routed_to"] or "none" for c in reached]
    assert {r["labeller_kind"] for r in rows} == {"author"}
    assert metaeval.agreement("lui_router", rows).status is metaeval.Status.NOT_HUMAN


def test_every_complete_json_call_site_is_classified() -> None:
    import re

    src = Path(metaeval.__file__).resolve().parents[1]
    sites = set()
    for path in src.rglob("*.py"):
        if "baselines" in path.parts or path.parts[-2] == "llm":
            continue
        for line in path.read_text("utf-8").splitlines():
            if re.search(r"\.complete_json\(|\.complete\(\s*$|client\.complete\(", line) \
                    and "def complete" not in line:
                sites.add(path.relative_to(src.parent).as_posix())
    covered = {j.module for j in metaeval.JUDGES} | {k.split(":")[0] for k in metaeval.NOT_JUDGES}
    assert sites <= covered, sorted(sites - covered)
