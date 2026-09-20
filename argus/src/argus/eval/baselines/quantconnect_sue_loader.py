"""Make the vendored `quantconnect_sue.py` (QuantConnect's real SUE factor computation)
importable.

No shim needed: `SueSorter` references only real, ordinary `numpy` plus its own constructor-
supplied attributes, never an internal QuantConnect/Lean module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_quantconnect_sue_shim"


class QuantConnectSueLoadError(RuntimeError):
    """The vendored QuantConnect SUE excerpt could not be loaded."""


def load_sue_module() -> ModuleType:
    """Load `quantconnect_sue.py`. Construct `.SueSorter(eps_by_symbol, months_count,
    months_eps_change)` and call `.compute(stock, sue_by_symbol)` exactly as the real
    `FineSelectionAndSueSorting` does."""
    dotted = "argus_eval_quantconnect_sue"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise QuantConnectSueLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    path = _BASELINES_DIR / "quantconnect_sue.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise QuantConnectSueLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["QuantConnectSueLoadError", "load_sue_module"]
