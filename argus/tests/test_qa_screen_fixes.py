"""Fixes from the QA screen pass (Activity/25_QA_SCREEN_RESULTS.md, 2026-09-28)."""

from __future__ import annotations

import pytest

from argus.lui import task


@pytest.mark.parametrize(("text", "name"), [
    ("should I buy Reliance", "Reliance"),
    ("should I buy Reliance Industries", "Reliance Industries"),
])
def test_an_unlisted_name_is_named_not_treated_as_gibberish(text: str, name: str) -> None:
    said = task.read_question(text, "")
    assert isinstance(said, str) and said.startswith(f"{name} is not a name Bitget lists")


@pytest.mark.parametrize("text", ["hello there", "what about risk"])
def test_no_name_still_gets_the_general_answer(text: str) -> None:
    said = task.read_question(text, "")
    assert isinstance(said, str) and said.startswith("I could not find a name Bitget lists")


def test_the_mobile_question_box_fits_its_two_line_example() -> None:
    from pathlib import Path

    from argus.lui import server

    source = Path(server.__file__).read_text(encoding="utf-8")
    assert "@media (max-width: 640px) { textarea#q { min-height:" in source
    assert "@media (max-width: 640px) {{ pre {{ white-space:pre-wrap" in source


def test_materials_counts_engines_and_tools_from_the_code() -> None:
    from argus.lui.materials_page import collect
    from argus.lui.task import STEPS
    from argus.truth.paths import DATA_DIR

    items = {i.label: i.shows for i in collect(DATA_DIR, tools=["argus_ask", "argus_quote"])}
    assert f"{len(STEPS)} engines" in items["Research task"] or "eight engines" in items[
        "Research task"]
    assert "two tools, from argus_ask to argus_quote." in items["MCP server"]
    assert "one tool" not in " ".join(items.values())
