"""The /factors page draws data/factor_lab.json as it stands, failures included, and says so when
the record cannot be read (research/harvest/36-ai-scientist.md)."""

from __future__ import annotations

import json
from pathlib import Path

from argus.lui.factors_page import load, render

DATA = Path(__file__).resolve().parents[1] / "data"


def test_every_factor_and_its_reason_is_shown() -> None:
    blob = load(DATA)
    assert blob is not None
    page = render(blob)
    for factor in blob["factors"]:
        assert factor["name"] in page
        if factor.get("rejection_reason"):
            assert factor["rejection_reason"].split(" (")[0].replace("&", "&amp;") in page
    funnel = blob["funnel"]
    assert f"{funnel['proposed']} factors proposed, {funnel['certified']} certified" in page


def test_a_gate_never_reached_is_a_dash_not_a_zero(tmp_path: Path) -> None:
    blob = {"funnel": {"proposed": 1, "certified": 0, "rejected": 1}, "gate": {"trials": 1},
            "factors": [{"name": "x", "expression": "a<b", "state": "rejected",
                         "gross_sharpe": 0.5, "net_sharpe": None, "oos_sharpe": None,
                         "dsr": None, "rejection_reason": "died at costs"}]}
    (tmp_path / "factor_lab.json").write_text(json.dumps(blob), encoding="utf-8")
    page = render(load(tmp_path))
    assert "0.500" in page and page.count(">—</td>") == 3
    assert "a&lt;b" in page


def test_an_unreadable_record_says_so(tmp_path: Path) -> None:
    (tmp_path / "factor_lab.json").write_text("{not json", encoding="utf-8")
    assert load(tmp_path) is None
    assert "could not be read" in render(None)
