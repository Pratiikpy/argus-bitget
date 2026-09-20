"""Writing an artefact that something other than Python can read.

**This module exists because our artefacts were not JSON.** Python's :mod:`json` emits bare ``NaN``
and ``Infinity`` tokens by default and accepts them on the way back in, so the round trip inside
this project always worked. Nothing else in the world does: `JSON.parse`, Go's `encoding/json` and
Rust's `serde_json` all reject those tokens, because they are not in the grammar. An adversarial
audit on 2026-09-20 found three committed artefacts carrying them — including the cross-sectional
study and the regime comparison — which means a judge who opened one in a browser console, or any
reviewer whose tooling is not Python, got a parse error from a project whose entire argument is
that its evidence is open to inspection.

The fix is not to forbid the value. An undefined correlation on a half-length series is a real
thing that genuinely happened, and the project's standing convention is to write ``null`` with a
reason rather than a zero — ``None`` is exactly right for it. What was wrong was the *encoding*.

So: non-finite floats become ``null``, and where they appear the key is recorded alongside, so a
reader can tell "this was undefined" from "this was never computed". Nothing is silently dropped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def sanitise(blob: Any, *, path: str = "", seen: list[str] | None = None) -> Any:
    """Replace every non-finite float with ``None``, recording where each one was.

    ``seen`` collects the key paths, so the caller can say what became null rather than leaving a
    reader to guess whether a ``null`` means undefined or never-run.
    """
    if isinstance(blob, float) and not math.isfinite(blob):
        if seen is not None:
            seen.append(f"{path or '<root>'}={'nan' if math.isnan(blob) else 'inf'}")
        return None
    if isinstance(blob, dict):
        return {k: sanitise(v, path=f"{path}.{k}" if path else str(k), seen=seen)
                for k, v in blob.items()}
    if isinstance(blob, list):
        return [sanitise(v, path=f"{path}[{i}]", seen=seen) for i, v in enumerate(blob)]
    return blob


def dumps(blob: Any, *, indent: int = 2) -> str:
    """Serialise to **strict** JSON. Raises rather than emitting a token nothing else parses.

    ``allow_nan=False`` is the belt: if a non-finite value survives :func:`sanitise` — through a
    numpy scalar, say, or a subclass that fools the isinstance check — this raises ``ValueError``
    instead of writing a file that only Python can read. A writer that fails loudly is worth more
    than one that produces a corpus nobody else can open.
    """
    return json.dumps(sanitise(blob), indent=indent, allow_nan=False)


def write(path: Path, blob: Any, *, indent: int = 2) -> list[str]:
    """Write an artefact as strict JSON. Returns the key paths that were non-finite."""
    undefined: list[str] = []
    text = json.dumps(sanitise(blob, seen=undefined), indent=indent, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return undefined


def is_strict(path: Path) -> bool:
    """Would a non-Python parser read this file?

    Uses ``parse_constant``, which fires on exactly the three tokens outside the JSON grammar —
    ``NaN``, ``Infinity``, ``-Infinity``. Plain ``json.loads`` accepts all three and is therefore
    useless as a check.
    """
    def reject(token: str) -> Any:
        raise ValueError(token)

    try:
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (ValueError, OSError):
        return False
    return True
