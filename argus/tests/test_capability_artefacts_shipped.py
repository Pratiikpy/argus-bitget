"""Every artefact a capability row cites is in the repository.

A hostile review (2026-09-29) found three LOST and TIED rows quoting figures from
data/general_delib_comparison.json, data/session_arena.json and data/stopquality_prospective.json,
none of which had been published: the figures were real and could not be checked. This runs in
CI on a clean checkout, where a file that was never committed is a file that is not there.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITED = re.compile(r"\bdata/[A-Za-z0-9_./-]+\.(?:json|jsonl|csv|md)\b")


def test_every_artefact_a_capability_cites_is_shipped() -> None:
    cited: dict[str, set[str]] = {}
    for toml in sorted((ROOT / "src" / "argus" / "eval" / "capabilities").glob("*.toml")):
        for name in CITED.findall(toml.read_text(encoding="utf-8")):
            cited.setdefault(name, set()).add(toml.name)
    assert len(cited) > 50
    missing = {name: sorted(rows) for name, rows in cited.items() if not (ROOT / name).exists()}
    assert not missing, missing
