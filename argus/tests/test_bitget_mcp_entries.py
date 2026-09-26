"""`bitget_mcp.ANSWER_ENTRIES` is the list /status reports as "read by the console's answers";
it must equal the catalog entries the source actually names."""

from __future__ import annotations

import json
import re
from pathlib import Path

from argus.market.bitget_mcp import ANSWER_ENTRIES

PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src" / "argus"
# Where catalog names appear without being read by an answer: the constant itself (cut out below),
# the harnesses that sweep the whole catalog, and bitget-signal's Skill list (its tool names
# overlap the catalog).
EXCLUDED = {SRC / "market" / "skills.py"}
CONSTANT = re.compile(r"ANSWER_ENTRIES: tuple\[str, \.\.\.\] = \(.*?\n\)", re.S)


def _catalog() -> set[str]:
    blob = json.loads((PACKAGE / "data" / "data_coverage.json").read_text(encoding="utf-8"))
    return {str(p["entry_id"]) for p in blob["probes"]}


def _named_in_source(catalog: set[str]) -> set[str]:
    named: set[str] = set()
    for path in SRC.rglob("*.py"):
        if path in EXCLUDED or "eval" in path.relative_to(SRC).parts:
            continue
        text = CONSTANT.sub("", path.read_text(encoding="utf-8", errors="replace"))
        named |= {n for n in re.findall(r"[\"']([a-z0-9_]+)[\"']", text) if n in catalog}
    return named


def test_every_listed_entry_is_in_the_catalog() -> None:
    assert set(ANSWER_ENTRIES) <= _catalog()


def test_the_list_is_exactly_what_the_answers_name() -> None:
    catalog = _catalog()
    assert set(ANSWER_ENTRIES) == _named_in_source(catalog)
