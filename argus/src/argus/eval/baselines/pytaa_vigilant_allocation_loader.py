"""Make the vendored `pytaa_vigilant_allocation.py` (pytaa's real VAA breadth rule) importable.

Like `lean_pairs_ranking_loader.py`, this needs no shim: the vendored function's only imports are
real, ordinary `numpy`/`pandas`/`typing.List`, already installed, never an internal pytaa module.
`importlib.util.spec_from_file_location` loads it directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_pytaa_vigilant_allocation_shim"


class PytaaVigilantAllocationLoadError(RuntimeError):
    """The vendored pytaa VAA excerpt could not be loaded."""


def load_vigilant_allocation_module() -> ModuleType:
    """Load `pytaa_vigilant_allocation.py`. Call `.vigilant_allocation(data, risk_assets,
    safe_assets, top_k, step)` exactly as pytaa's real `strategies.py` does."""
    dotted = "argus_eval_pytaa_vigilant_allocation"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise PytaaVigilantAllocationLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    path = _BASELINES_DIR / "pytaa_vigilant_allocation.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise PytaaVigilantAllocationLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["PytaaVigilantAllocationLoadError", "load_vigilant_allocation_module"]
