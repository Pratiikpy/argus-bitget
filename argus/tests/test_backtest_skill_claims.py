"""No document may credit bitget-signal's backtest tool while it cannot run (audit finding 114).

The tool parses ``strategy_config`` twice and refuses every shape
(`eval/skill_matrix.py` ``_BACKTEST_CONFIG``; data/skill_matrix.json). A sentence such as
"validated by Bitget's backtest Skill" would therefore describe a call that has never succeeded.
When the tool is fixed and a real call is recorded, this test is the one to change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ARGUS = Path(__file__).resolve().parents[1]
WORKSPACE = ARGUS.parent
DOCS = (ARGUS / "README.md", WORKSPACE / "README.md", WORKSPACE / "ARGUS-EXPLAINED.md",
        WORKSPACE / "ARGUS-ARCHITECTURE.md", WORKSPACE / "SUBMISSION-DRAFT.md")

CLAIM = re.compile(
    r"(?:validated|verified|confirmed|checked|backtested)\s+(?:\w+\s+){0,3}(?:by|with|on|through|"
    r"using)\s+(?:\w+'?s?\s+){0,3}(?:bitget[- ]signal'?s?\s+)?backtest(?:\s+(?:skill|tool))",
    re.I)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_document_credits_the_broken_backtest_tool(doc: Path) -> None:
    if not doc.exists():
        pytest.skip(f"{doc.name} is not in this checkout")
    found = [m.group(0) for m in CLAIM.finditer(doc.read_text(encoding="utf-8"))]
    assert not found, f"{doc.name} credits bitget-signal's backtest tool: {found}"


def test_the_pattern_catches_the_claim_it_exists_for() -> None:
    assert CLAIM.search("every rule was validated by Bitget's backtest Skill")
    assert CLAIM.search("backtested with the bitget-signal backtest tool")
    assert not CLAIM.search("the Nautilus backtest of the playbook")
