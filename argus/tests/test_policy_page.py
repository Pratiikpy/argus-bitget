"""The /policy page draws the risk policy file and the last cycle's record of which limits could
fire, and says so when either is missing (research/harvest/26-opa-rule-gate.md)."""

from __future__ import annotations

from pathlib import Path

from argus.lui.policy_page import load, render
from argus.risk import policy_manifest as pm

DATA = Path(__file__).resolve().parents[1] / "data"


def test_limits_parameters_and_their_state() -> None:
    policy, _ = load(DATA)
    assert policy is not None
    live = {"at": "2026-09-28T19:00:00+00:00",
            "per_symbol": {"NVDAUSDT": {"can_fire": ["max_position_notional"],
                                        "cannot_fire": ["max_scenario_loss_pct"]}}}
    page = render(policy, live)
    limits = len([n for n in pm.THRESHOLDS if n not in pm.PARAMETERS])
    assert f"1 of {limits} limits could fire" in page and "1 symbol)" in page
    assert page.count("can fire</span>") == 1
    assert page.count(">parameter</span>") == len(pm.PARAMETERS)
    assert "$50,000" in page and policy["digest"][:16] in page


def test_no_cycle_yet_and_no_file() -> None:
    policy, _ = load(DATA)
    assert "no cycle has recorded" in render(policy, None)
    assert "could not be read" in render(None, None)
