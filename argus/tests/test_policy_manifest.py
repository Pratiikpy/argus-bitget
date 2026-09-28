"""The risk policy as a file (research/harvest/26-opa-rule-gate.md §5 item 2): the committed file
loads and equals the code, an edit changes the policy only once sealed, a malformed file refuses,
the paper cycle reads it and decides nothing on a refusal, and the live record says which gates
could fire."""

from __future__ import annotations

import inspect
import json
from decimal import Decimal
from pathlib import Path

import pytest

from argus.paper import runner
from argus.risk import policy_manifest as pm
from argus.risk.constitution import ConstitutionPolicy

DATA = Path(__file__).resolve().parents[1] / "data"


def write(path: Path, blob: dict[str, object]) -> Path:
    path.write_text(json.dumps(blob), encoding="utf-8")
    return path


def test_the_committed_policy_is_the_code_policy() -> None:
    kwargs, version = pm.load(DATA / "risk_policy.json")
    assert pm.thresholds_of(ConstitutionPolicy(**kwargs)) == pm.thresholds_of(
        ConstitutionPolicy())
    assert version == pm.manifest()["digest"]


def test_every_threshold_is_listed_and_nothing_else() -> None:
    policy_fields = set(ConstitutionPolicy.__dataclass_fields__)
    assert set(pm.THRESHOLDS) <= policy_fields
    injected = {"reference_price", "book_state", "book", "factor_exposures", "stress_outcomes",
                "liquidation_cost_estimates", "graded_predictions", "session_risk", "baseline"}
    assert policy_fields - set(pm.THRESHOLDS) - injected == set()


def test_an_edit_counts_only_once_sealed(tmp_path: Path) -> None:
    blob = pm.manifest()
    blob["thresholds"]["max_position_notional"]["value"] = "25000"  # type: ignore[index]
    path = write(tmp_path / "risk_policy.json", blob)
    with pytest.raises(pm.PolicyError, match="seal"):
        pm.load(path)
    edited, _ = pm.load(path, sealed=False)
    write(path, pm.manifest(ConstitutionPolicy(**edited)))
    kwargs, _ = pm.load(path)
    assert kwargs["max_position_notional"] == Decimal("25000")
    policy, _ = pm.policy(path)
    assert policy.max_position_notional == Decimal("25000")


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda t: t.pop("max_unhedged_notional"), "missing"),
    (lambda t: t.update(max_leverage={"value": "3"}), "unknown"),
    (lambda t: t["session_horizon_bars"].update(value=24.5), "whole number"),
    (lambda t: t["max_position_notional"].update(value="lots"), "not a number"),
    (lambda t: t["max_position_notional"].update(value=None), "may not be null"),
])
def test_a_malformed_policy_refuses(tmp_path: Path, mutate: object, match: str) -> None:
    blob = pm.manifest()
    mutate(blob["thresholds"])  # type: ignore[operator]
    with pytest.raises(pm.PolicyError, match=match):
        pm.load(write(tmp_path / "risk_policy.json", blob), sealed=False)


def test_the_cycle_reads_the_file_and_refuses_on_a_bad_one(tmp_path: Path) -> None:
    assert runner.cycle_policy(tmp_path / "absent.json") == ({}, "code defaults", "")
    good = write(tmp_path / "risk_policy.json", pm.manifest())
    kwargs, version, refused = runner.cycle_policy(good)
    assert not refused and version == pm.manifest()["digest"] and kwargs
    bad = pm.manifest()
    bad["thresholds"]["max_position_notional"]["value"] = "1"  # type: ignore[index]
    _, _, refused = runner.cycle_policy(write(tmp_path / "risk_policy.json", bad))
    assert "seal" in refused
    source = inspect.getsource(runner.run_once)
    assert "cycle_policy(" in source and "**policy_kwargs" in source
    assert "killed or policy_refused" in source


def test_the_live_record_says_which_gates_can_fire() -> None:
    record = pm.live_record(ConstitutionPolicy(), "v")
    # With no book, factor exposures, stress or liquidation estimates supplied, those gates cannot
    # fire; the order-only caps can.
    assert "max_position_notional" in record["can_fire"]
    for gate in ("max_gross_exposure_notional", "factor_exposure_limits",
                 "max_scenario_loss_pct", "max_liquidation_cost_bps"):
        assert gate in record["cannot_fire"]
    assert record["gates"]["max_liquidation_cost_bps"]["set"] is False
