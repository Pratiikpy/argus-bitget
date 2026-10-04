"""Build-list 4.2: a figure in the desk's thesis that its grounding check could not trace is
marked where the thesis is shown; one that is arithmetic on two traced figures says so."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

answer = import_module("argus.lui.answer")  # the package re-exports a function of the same name

GROUNDING = ("[grounding] 2 of 7 figure(s) do not resolve to anything the desk was given: "
             "2.2%, 37bps")


@pytest.fixture
def notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "desk_notes.jsonl"
    path.write_text(json.dumps({"seq": 7, "notes": ["[panel] 3 of 3 analysts run", GROUNDING]})
                    + "\n" + json.dumps({"seq": 8, "notes": ["[grounding] all 3 figure(s) "
                                                             "resolve"]}) + "\n",
                    encoding="utf-8")
    monkeypatch.setattr(answer, "desk_notes_path", lambda: path)
    return path


def test_the_untraced_figures_are_read_from_the_note(notes: Path) -> None:
    assert answer.untraced_figures(7) == ("2.2%", "37bps")
    assert answer.untraced_figures(8) == ()
    assert answer.untraced_figures(99) == ()


def test_each_untraced_figure_is_marked_where_it_stands(notes: Path) -> None:
    thesis = ("The token trades at a ~2.2% premium ($163.45 vs $160.01); a 37bps edge is not "
              "enough. 37bps again.")
    said = answer.marked_thesis(7, thesis)
    # 2.2% is 163.45 / 160.01 - 1 = 2.15%: said to be arithmetic on two stated figures
    assert "~2.2% [not in the evidence; worked out as 163.45 / 160.01 - 1 = 2.15%] premium" in said
    # 37bps traces to nothing, every time it appears
    assert said.count("37bps [not in the evidence]") == 2
    # marking twice does not mark twice
    assert answer.marked_thesis(7, said) == said
    # a thesis whose figures all resolve is shown as logged
    assert answer.marked_thesis(8, "All 3 figures resolve.") == "All 3 figures resolve."


def test_a_figure_inside_a_longer_number_is_not_marked(notes: Path) -> None:
    assert answer.marked_thesis(7, "a 12.2% move and 137bps") == "a 12.2% move and 137bps"


def test_a_scaled_figure_is_marked_after_its_scale(notes: Path) -> None:
    path = notes
    path.write_text(json.dumps({"seq": 9, "notes": [
        "[grounding] 1 of 3 figure(s) do not resolve to anything the desk was given: 85"]}) + "\n",
        encoding="utf-8")
    assert answer.marked_thesis(9, "BTC above $85k on the proposal") == (
        "BTC above $85k [not in the evidence] on the proposal")


def test_evidence_behind_a_decision_is_the_desks_record() -> None:
    from argus.lui import server

    got = server.handle_ask("show me the evidence behind the latest MSTR decision", [])
    assert got["classified_by"] != "research-task"
    assert str(got["lines"][0]).startswith("Evidence behind seq")
