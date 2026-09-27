"""Where the package's data lives, named once.

Until 2026-09-27 some two hundred modules each wrote
``Path(__file__).resolve().parents[3] / "data"`` (audit finding 165): correct only while every
file sat at the same depth, and a module moved one level would have read or written somewhere else
without a word. Every module now takes the directory from here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
"""The ``argus/`` directory that holds ``src/``, ``data/`` and ``tests/``."""

DATA_DIR = PACKAGE_ROOT / "data"

REPOSITORY_ROOT = PACKAGE_ROOT.parent
"""The directory holding ``argus/``: the private workspace, or the public repository's checkout."""


def _root_spellings(root: Path) -> tuple[str, ...]:
    """``root`` followed by a separator, in every spelling a Windows path takes in text and in
    JSON-escaped text — the same set `tools/publish.py` removes on the way out."""
    text = str(root)
    spellings = {text + "\\", text.replace("\\", "\\\\") + "\\\\", text.replace("\\", "/") + "/"}
    if len(text) > 3 and text[1:3] == ":\\":
        spellings.add(f"/{text[0].lower()}/" + text[3:].replace("\\", "/") + "/")
    return tuple(sorted(spellings, key=len, reverse=True))


def portable_bytes(data: bytes, root: Path = REPOSITORY_ROOT) -> bytes:
    """``data`` with this checkout's root removed wherever it is written into the text.

    Some artefacts record the absolute path they were written from. Publishing rewrites those to
    repository-relative paths, so the same artefact had one hash in the workspace and another in
    the public repository, and the register's groupwise check reached a different verdict in each
    (found 2026-09-28 by running the public CI's gates on a clean checkout). Hashing this form gives
    both copies one digest. Bytes that are not UTF-8 are returned unchanged."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    for prefix in _root_spellings(root):
        text = text.replace(prefix, "")
    return text.encode("utf-8")


def portable_digest(path: Path, root: Path = REPOSITORY_ROOT) -> str:
    """SHA-256 of :func:`portable_bytes` of the file at ``path``."""
    return hashlib.sha256(portable_bytes(path.read_bytes(), root)).hexdigest()


__all__ = ["DATA_DIR", "PACKAGE_ROOT", "REPOSITORY_ROOT", "portable_bytes", "portable_digest"]
