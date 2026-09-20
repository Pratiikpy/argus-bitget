"""Make the vendored TradingAgents memory log importable without editing a line of it.

``tradingagents_memory.py`` contains, verbatim, ``from tradingagents.agents.utils.rating import
parse_rating`` — correct inside TradingAgents' own repo layout, meaningless here. This module
registers the sibling vendored ``tradingagents_rating.py`` in :data:`sys.modules` under the exact
dotted name the memory file expects, so its own import resolves without editing either vendored
file — the same shim pattern as every other baseline in this package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_tradingagents_baseline_shim"

# The bare "tradingagents" top-level package is shimmed by more than one loader in this package
# (this one and `tradingagents_feedlist_loader.py`, which shims a different subpackage —
# `tradingagents.dataflows` — of the same real repo). A loader-private marker on that one shared
# name would make the second loader to run see "a module I did not create" and refuse it, even
# though both loaders' placeholder is equally empty and equally harmless. This marker is
# therefore shared by name (duplicated as a literal in each file, not imported, to keep each
# loader independently readable) — every OTHER shim this file creates keeps its own private
# marker above, since those dotted paths are not shared with any other loader.
_SHARED_TOP_LEVEL_MARKER = "_argus_tradingagents_toplevel_shim"


class TradingAgentsBaselineLoadError(RuntimeError):
    """The vendored TradingAgents baseline could not be loaded, or a name it needs is taken."""


def _load_module_from_file(dotted_name: str, path: Path, *, register: bool = False) -> ModuleType:
    """Execute ``path`` as a module named ``dotted_name``.

    ``register=True`` puts the module into :data:`sys.modules` before executing its body — needed
    here because ``TradingMemoryLog`` uses class-level compiled regex attributes, and (as with the
    other loaders in this package) registering first matches the standard import-machinery order
    rather than relying on it not mattering for this particular file.
    """
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    if spec is None or spec.loader is None:
        raise TradingAgentsBaselineLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


def _shim_package(dotted_name: str, *, marker: str = _SHIM_MARKER) -> ModuleType:
    existing = sys.modules.get(dotted_name)
    if existing is not None:
        if not getattr(existing, marker, False):
            raise TradingAgentsBaselineLoadError(
                f"sys.modules[{dotted_name!r}] already holds a module this loader did not "
                f"create; refusing to overwrite it (fail-closed, matching the vendored code's "
                f"own posture)"
            )
        return existing
    import types

    module = types.ModuleType(dotted_name)
    setattr(module, marker, True)
    sys.modules[dotted_name] = module
    return module


def load_memory_module() -> ModuleType:
    """Load the vendored ``tradingagents_memory.py`` with its own unedited import satisfied.

    Returns:
        The executed module object — carries ``TradingMemoryLog``.
    """
    _shim_package("tradingagents", marker=_SHARED_TOP_LEVEL_MARKER)
    for name in ("tradingagents.agents", "tradingagents.agents.utils"):
        _shim_package(name)

    rating_dotted = "tradingagents.agents.utils.rating"
    if rating_dotted not in sys.modules or not getattr(
        sys.modules[rating_dotted], _SHIM_MARKER, False
    ):
        rating_module = _load_module_from_file(
            rating_dotted, _BASELINES_DIR / "tradingagents_rating.py", register=True
        )
        setattr(rating_module, _SHIM_MARKER, True)

    memory_dotted = "argus_baseline_tradingagents_memory"
    if memory_dotted in sys.modules and getattr(sys.modules[memory_dotted], _SHIM_MARKER, False):
        return sys.modules[memory_dotted]

    module = _load_module_from_file(
        memory_dotted, _BASELINES_DIR / "tradingagents_memory.py", register=True
    )
    setattr(module, _SHIM_MARKER, True)
    return module


def load_trading_memory_log_class() -> type:
    """Load the vendored file and return the ``TradingMemoryLog`` class.

    Raises:
        TradingAgentsBaselineLoadError: A name this loader needs was already taken, or the
            vendored file could not be executed.
    """
    module = load_memory_module()
    result: type = module.TradingMemoryLog
    return result


__all__ = [
    "TradingAgentsBaselineLoadError",
    "load_memory_module",
    "load_trading_memory_log_class",
]
