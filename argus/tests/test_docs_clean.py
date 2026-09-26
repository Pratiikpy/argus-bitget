"""The public documents carry no control characters.

A patch once wrote VERIFY.md's activation line as ``.venv\\Scripts<BEL>ctivate``: the ``\\a`` became
byte 0x07, and the one command a judge copies first no longer worked (2026-09-26)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
DOCS = sorted({*PACKAGE.glob("*.md"),
               *(PACKAGE.parent / name for name in ("README.md", "ARGUS-EXPLAINED.md",
                                                    "ARGUS-ARCHITECTURE.md")
                 if (PACKAGE.parent / name).exists())})
CONTROL = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_control_characters(doc: Path) -> None:
    found = CONTROL.search(doc.read_bytes())
    assert found is None, f"{doc.name}: control byte {found.group()!r} at offset {found.start()}"
