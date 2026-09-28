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


def test_a_long_mcp_question_says_it_was_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import mcp_server, server

    seen: list[str] = []

    def handle_ask(text: str, prior: list[str], **_: object) -> dict[str, object]:
        seen.append(text)
        return {"lines": ["an answer"], "sources": [], "refused": False}

    monkeypatch.setattr(server, "handle_ask", handle_ask)
    text, _ = mcp_server.call_tool("argus_ask", {"question": "why?" * 200})
    assert len(seen[0]) == mcp_server.MAX_QUESTION
    assert text.startswith(f"Note: only the first {mcp_server.MAX_QUESTION} of the question's "
                           f"800 characters were read.")
    short, _ = mcp_server.call_tool("argus_ask", {"question": "why"})
    assert not short.startswith("Note:")


def test_an_unreadable_fill_table_says_it_needs_a_header() -> None:
    from argus.lui import mcp_server

    with pytest.raises(mcp_server.ToolError, match="needs a header row"):
        mcp_server.call_tool("argus_review_trades", {"fills": "1,2,3\n4,5,6"})
