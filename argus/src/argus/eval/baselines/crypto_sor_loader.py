"""Run the real, vendored crypto_sor `CompositeOrderBook.newOrder()` via a real Node/ts-node
subprocess.

Unlike every other loader in this package, the real code here is TypeScript, not Python, so there
is no `sys.modules` shim to build — the whole point is to run the actual `node`/`ts-node`
interpreter against the actual vendored source (`crypto_sor_shim/src/lib/CompositeOrderBook.ts`),
the same discipline `eval/rdagent_comparison.py` already applies to RD-Agent's real subprocess
execution, just with a different real interpreter. `crypto_sor_shim/harness.ts` (not vendored —
this comparison's own thin JSON-in/JSON-out wrapper) is the entry point; its own header explains
exactly which real methods it calls.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_SHIM_DIR = Path(__file__).resolve().parent / "crypto_sor_shim"
_INSTALL_LOCK = threading.Lock()

# `npm`/`npx` are `.cmd` shims on Windows, and `subprocess.run` can only resolve those through a
# shell — hence `shell=True` below. On POSIX, `shell=True` with a *list* of args means something
# different: per the `subprocess` docs, only `args[0]` becomes the shell command string, and every
# other list item is passed as a positional parameter to the shell itself (`$1`, `$2`, ...), never
# appended to the command line. So on Linux CI this silently ran bare `npm` with no arguments —
# which exits 1 — instead of `npm ci --no-audit --no-fund --loglevel=error`, and `npx --no-install
# ts-node harness.ts` had the same problem. Gate `shell=True` to Windows, where the list form is
# converted to a full command string before being handed to the shell either way.
_USE_SHELL = sys.platform == "win32"


class CryptoSorSubprocessError(RuntimeError):
    """The real ts-node subprocess failed or returned something that could not be parsed."""


class CryptoSorUnavailable(CryptoSorSubprocessError):
    """Node.js is not installed here, so the real TypeScript cannot run at all."""


def node_available() -> bool:
    return shutil.which("npm") is not None and shutil.which("npx") is not None


def ensure_installed(timeout: float = 600.0) -> None:
    """Install the shim's pinned dependencies from its lockfile on first use.

    ``node_modules`` is not committed, and a fresh clone used to fail here: ``npx ts-node`` with no
    local install fetches whatever ts-node is current, without the pinned TypeScript it needs, and
    dies inside its own configuration loader. ``npm ci`` installs exactly the locked versions."""
    if (_SHIM_DIR / "node_modules" / "ts-node").is_dir():
        return
    if not node_available():
        raise CryptoSorUnavailable("Node.js (npm and npx) is required to run the real crypto_sor "
                                   "TypeScript; install Node 18 or later")
    with _INSTALL_LOCK:
        if (_SHIM_DIR / "node_modules" / "ts-node").is_dir():
            return
        try:
            result = subprocess.run(
                ["npm", "ci", "--no-audit", "--no-fund", "--loglevel=error"], cwd=_SHIM_DIR,
                capture_output=True, text=True, timeout=timeout, shell=_USE_SHELL, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CryptoSorSubprocessError(f"npm ci failed to run: {exc}") from exc
        if result.returncode != 0:
            raise CryptoSorSubprocessError(
                f"npm ci exited {result.returncode}: {result.stderr[-2000:]}")


def run_new_order(
    symbol: str,
    side: str,
    order_qty: float,
    levels: list[dict[str, Any]],
    *,
    exchanges: list[str] | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Feed the real, unmodified `CompositeOrderBook.newOrder()` a composite book and an order,
    and return the real `Execution[]` it produces.

    ``levels`` is a list of ``{"exchange": str, "side": "BUY"|"SELL", "price": float,
    "size": float}``. Each distinct ``exchange`` name is one competing quote source in the real
    class's own composite-book abstraction — this comparison feeds it real Bitget order-book
    sweeps for two different asset-class legs, one "exchange" per leg, exactly as the real class
    was designed to receive one "exchange" per real trading venue.
    """
    if not _SHIM_DIR.exists():
        raise CryptoSorSubprocessError(f"shim directory not found: {_SHIM_DIR}")
    ensure_installed()
    request = {
        "symbol": symbol, "side": side, "orderQty": order_qty,
        "exchanges": exchanges, "levels": levels,
    }
    try:
        result = subprocess.run(
            ["npx", "--no-install", "ts-node", "harness.ts"],
            cwd=_SHIM_DIR,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=_USE_SHELL,  # npx.cmd resolution on Windows needs the shell
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CryptoSorSubprocessError(f"the real ts-node subprocess failed to run: {exc}") from exc
    if result.returncode != 0:
        raise CryptoSorSubprocessError(
            f"the real ts-node subprocess exited {result.returncode}: {result.stderr[-2000:]}"
        )
    try:
        return list(json.loads(result.stdout))
    except json.JSONDecodeError as exc:
        raise CryptoSorSubprocessError(
            f"the real subprocess's stdout was not valid JSON: {result.stdout[:500]!r}"
        ) from exc


__all__ = ["CryptoSorSubprocessError", "CryptoSorUnavailable", "ensure_installed",
           "node_available", "run_new_order"]
