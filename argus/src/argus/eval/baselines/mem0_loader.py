"""Run real, installed `mem0ai` in its own isolated venv, via a subprocess — never imported here.

Same discipline as `crypto_sor_loader.py` (a real subprocess against real vendored code, JSON in,
JSON out), for the reason `mem0_runner.py`'s own docstring states: mem0ai must never be installed
into the environment that has `argus` importable, so there is no `sys.modules` shim available —
the only way to run the real package is to hand a JSON request to the real interpreter that has it
installed and read back a JSON response.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_MEM0_PYTHON = Path.home() / ".venvs" / "mem0" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python")
"""Overridable via `ARGUS_MEM0_VENV_PYTHON` for a machine where the isolated venv lives elsewhere;
defaults to `~/.venvs/mem0`, where this comparison's setup created it."""

_RUNNER_SCRIPT = Path(__file__).resolve().parent / "mem0_runner.py"


class Mem0SubprocessError(RuntimeError):
    """The real mem0 subprocess failed to run or returned something that could not be parsed."""


class Mem0Unavailable(Mem0SubprocessError):
    """The isolated mem0 venv is not present here."""


def mem0_python() -> Path:
    override = os.environ.get("ARGUS_MEM0_VENV_PYTHON")
    return Path(override) if override else DEFAULT_MEM0_PYTHON


def mem0_available() -> bool:
    return mem0_python().is_file() and _RUNNER_SCRIPT.is_file()


def run_chains(
    chains: list[dict[str, Any]], *, max_workers: int = 6, timeout: float = 1800.0,
) -> dict[str, Any]:
    """Run every chain's ops, in order within a chain, against real mem0 — one subprocess call.

    ``chains``: ``[{"chain_id": str, "user_id": str, "ops": [{"op": "add"|"search"|"get_all",
    ...}]}]``. See `mem0_runner.py`'s docstring for the concurrency model (each worker owns one
    real `Memory` instance and one temp qdrant directory; chains are sharded across a thread pool).
    """
    if not mem0_available():
        raise Mem0Unavailable(
            f"isolated mem0 venv not found at {mem0_python()}. Create it and `pip install mem0ai "
            f"sentence-transformers fastembed spacy` inside it, per argus/CLAUDE.md's setup rule."
        )
    request = {"chains": chains, "max_workers": max_workers}
    try:
        result = subprocess.run(
            [str(mem0_python()), str(_RUNNER_SCRIPT)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Mem0SubprocessError(f"the real mem0 subprocess failed to run: {exc}") from exc
    if not result.stdout.strip():
        raise Mem0SubprocessError(
            f"the real mem0 subprocess produced no stdout (exit {result.returncode}); "
            f"stderr tail: {result.stderr[-4000:]}"
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise Mem0SubprocessError(
            f"the real mem0 subprocess's stdout was not valid JSON: {result.stdout[:1000]!r}; "
            f"stderr tail: {result.stderr[-2000:]}"
        ) from exc
    if "fatal_error" in response:
        raise Mem0SubprocessError(f"the real mem0 subprocess raised: {response['fatal_error']}")
    return dict(response)


def print_available() -> None:  # pragma: no cover - manual diagnostic
    print("mem0 python:", mem0_python(), "available:", mem0_available(), file=sys.stderr)


__all__ = [
    "DEFAULT_MEM0_PYTHON",
    "Mem0SubprocessError",
    "Mem0Unavailable",
    "mem0_available",
    "mem0_python",
    "run_chains",
]
