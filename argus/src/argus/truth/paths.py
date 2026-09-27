"""Where the package's data lives, named once.

Until 2026-09-27 some two hundred modules each wrote
``Path(__file__).resolve().parents[3] / "data"`` (audit finding 165): correct only while every
file sat at the same depth, and a module moved one level would have read or written somewhere else
without a word. Every module now takes the directory from here.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
"""The ``argus/`` directory that holds ``src/``, ``data/`` and ``tests/``."""

DATA_DIR = PACKAGE_ROOT / "data"

__all__ = ["DATA_DIR", "PACKAGE_ROOT"]
