"""A question about ARGUS against a named system is answered from the register (`lui/rivals.py`;
fresh-eyes audit of 2026-09-29: "compare to Nautilus Trader" got the risk layer's own counts)."""

from __future__ import annotations

import pytest

from argus.eval.standing import REGISTER
from argus.lui import rivals


@pytest.mark.parametrize("text", [
    "How does ARGUS's risk layer compare to Nautilus Trader?", "is ARGUS better than OpenBB",
    "how does your desk compare to TradingAgents", "how do you stack up against qlib",
])
def test_a_question_about_argus_against_a_named_rival_is_recognised(text: str) -> None:
    assert rivals.asks_about_a_rival(text)


@pytest.mark.parametrize("text", [
    "compare gold and bitcoin", "compare NVDA vs AMD", "is ARGUS better than bitcoin",
    "is TSLA riskier than NVDA", "how does argus compare to FooBarBaz",
])
def test_other_comparisons_are_not(text: str) -> None:
    assert not rivals.asks_about_a_rival(text)


def test_the_answer_is_every_register_row_naming_the_rival_with_its_state() -> None:
    lines, sources, data = rivals.answer("is ARGUS better than OpenBB")
    expected = {c.name: c.state.value for c in REGISTER if "openbb" in c.baseline.lower()}
    assert {r["name"]: r["state"] for r in data["rival_rows"]} == expected
    assert lines[0].startswith(f"Bottom line: in ARGUS's register, {len(expected)} capability row")
    assert "against OpenBB:" in lines[0]
    assert all(not line.split(" — ", 1)[-1].startswith(("RE-GRADED", "GRADED"))
               for line in lines[1:])
    assert "Activity/" not in " ".join(lines)
    assert sources[0].ref == "argus.eval.standing register"
