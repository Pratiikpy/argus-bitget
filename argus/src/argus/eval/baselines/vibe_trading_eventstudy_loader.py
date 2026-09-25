"""Load the two vendored event-significance baselines, re-checking their bytes on every load.

* ``vibe_trading_eventstudy.py`` — HKUDS/Vibe-Trading's ``agent/src/quantlib/eventstudy.py``
  (MIT), the whole file: ``event_study`` with t, Patell, BMP, Corrado and Cowan tests.
* ``ballast_stats.py`` — Ritapossible/Ballast's ``ballast/stats.py`` (MIT), whose ``t_stat`` is
  the pooled t behind Ballast's published Gate 1 / Gate 1b event verdicts.

Both are executed from the vendored file by path, the same way
``whale_signals_event_study_loader`` does it. Unlike that loader, this one refuses to execute a
body whose SHA256 does not match the pin: a comparison against a baseline someone edited is a
comparison against a paraphrase, which ``eval/standing.py``'s ``baseline_reproduced`` exists to
refuse. The pin is the SHA256 of the upstream blob at the cited commit, which was also checked
against ``git rev-parse HEAD:<path>`` in the clone when the file was vendored.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_event_significance_baseline_shim"
MARKER = b"# ============================== VENDORED FROM HERE ==============================\n"

VIBE_EVENTSTUDY_SHA256 = "9400e45bae876ca6f2e8099a57c3420d89b213dcb4f3d04aff8c3e9217712505"
VIBE_COMMIT = "a65427a3d587e43cd5c7541948b36524e8fd2f12"
BALLAST_STATS_SHA256 = "a731201da6aa63163569f39ae363993bf90a5fa26042aa19c93a599c18d9fa21"
BALLAST_COMMIT = "5cf675904bbb27ea338377b4cfe01a79d880086f"


class EventBaselineLoadError(RuntimeError):
    """A vendored event-significance baseline is missing, drifted, or could not be executed."""


def vendored_body_sha256(filename: str) -> str:
    """SHA256 of everything after the ``VENDORED FROM HERE`` marker in ``filename``."""
    raw = (_BASELINES_DIR / filename).read_bytes()
    if MARKER not in raw:
        raise EventBaselineLoadError(f"{filename} has no VENDORED FROM HERE marker")
    return hashlib.sha256(raw.split(MARKER, 1)[1]).hexdigest()


def _load(filename: str, dotted: str, pinned: str) -> ModuleType:
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise EventBaselineLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing
    got = vendored_body_sha256(filename)
    if got != pinned:
        raise EventBaselineLoadError(
            f"{filename} body hashes to {got}, not the pinned upstream {pinned}; the vendored "
            f"copy was edited and is no longer the baseline it claims to be"
        )
    path = _BASELINES_DIR / filename
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise EventBaselineLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    # Registered before execution: both files define frozen dataclasses, and `dataclasses`
    # resolves `sys.modules[cls.__module__]` while the class body is still executing.
    sys.modules[dotted] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[dotted]
        raise
    return module


def load_vibe_eventstudy() -> ModuleType:
    """Vibe-Trading's real ``quantlib.eventstudy``. Call ``.event_study(returns, market, events,
    event_window=..., estimation_window=..., estimation_gap=..., model="market")`` exactly as
    their own tests do."""
    return _load(
        "vibe_trading_eventstudy.py", "argus_eval_vibe_trading_eventstudy",
        VIBE_EVENTSTUDY_SHA256,
    )


def load_ballast_stats() -> ModuleType:
    """Ballast's real ``stats`` module. ``.t_stat(values)`` returns ``(mean, t)`` and refuses
    (NaN) below eight observations, as their file does."""
    return _load("ballast_stats.py", "argus_eval_ballast_stats", BALLAST_STATS_SHA256)


__all__ = [
    "BALLAST_COMMIT",
    "BALLAST_STATS_SHA256",
    "MARKER",
    "VIBE_COMMIT",
    "VIBE_EVENTSTUDY_SHA256",
    "EventBaselineLoadError",
    "load_ballast_stats",
    "load_vibe_eventstudy",
    "vendored_body_sha256",
]
