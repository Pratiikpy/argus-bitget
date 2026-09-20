"""Make the vendored serenity-guardrails journal importable without editing a line of it.

``serenity_journal.py`` contains, verbatim, ``from .guards import GuardError`` — correct inside
serenity-guardrails' own repo layout (``etoro_trading`` as a real package), meaningless here. This
module registers the sibling vendored ``serenity_guards.py`` in :data:`sys.modules` under the exact
dotted name the journal file expects, so its own import resolves without editing either vendored
file — the same shim pattern as every other baseline in this package.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_serenity_baseline_shim"


class SerenityBaselineLoadError(RuntimeError):
    """The vendored serenity-guardrails baseline could not be loaded, or a name it needs is
    already taken."""


def _load_module_from_file(dotted_name: str, path: Path, *, register: bool = False) -> ModuleType:
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    if spec is None or spec.loader is None:
        raise SerenityBaselineLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


def _shim_package(dotted_name: str) -> ModuleType:
    existing = sys.modules.get(dotted_name)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise SerenityBaselineLoadError(
                f"sys.modules[{dotted_name!r}] already holds a module this loader did not "
                f"create; refusing to overwrite it (fail-closed, matching the vendored code's "
                f"own posture)"
            )
        return existing
    module = types.ModuleType(dotted_name)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted_name] = module
    return module


def load_journal_module() -> ModuleType:
    """Load the vendored ``serenity_journal.py`` with its own unedited relative import satisfied.

    Returns:
        The executed module object — carries ``ChainedJournal``.
    """
    _shim_package("etoro_trading")

    guards_dotted = "etoro_trading.guards"
    if guards_dotted not in sys.modules or not getattr(
        sys.modules[guards_dotted], _SHIM_MARKER, False
    ):
        guards_module = _load_module_from_file(
            guards_dotted, _BASELINES_DIR / "serenity_guards.py", register=True
        )
        setattr(guards_module, _SHIM_MARKER, True)

    journal_dotted = "etoro_trading.journal"
    if journal_dotted in sys.modules and getattr(sys.modules[journal_dotted], _SHIM_MARKER, False):
        return sys.modules[journal_dotted]

    module = _load_module_from_file(
        journal_dotted, _BASELINES_DIR / "serenity_journal.py", register=True
    )
    setattr(module, _SHIM_MARKER, True)
    return module


def load_chained_journal_class() -> type:
    """Load the vendored file and return the ``ChainedJournal`` class.

    Raises:
        SerenityBaselineLoadError: A name this loader needs was already taken, or the vendored
            file could not be executed.
    """
    module = load_journal_module()
    result: type = module.ChainedJournal
    return result


__all__ = [
    "SerenityBaselineLoadError",
    "load_chained_journal_class",
    "load_journal_module",
]
