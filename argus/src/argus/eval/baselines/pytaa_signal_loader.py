"""Make the vendored `pytaa_signal.py` (pytaa's real `Signal` momentum class) importable.

Like `pytaa_vigilant_allocation_loader.py`, no shim needed: `Signal` references only its own
`__init__`-set attributes and real, ordinary `numpy`/`pandas`, never an internal pytaa module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_pytaa_signal_shim"


class PytaaSignalLoadError(RuntimeError):
    """The vendored pytaa `Signal` excerpt could not be loaded."""


def load_signal_module() -> ModuleType:
    """Load `pytaa_signal.py`. Construct `.Signal(prices_dataframe)` and call
    `.momentum_score()` exactly as pytaa's real `strategy/signals.py` does."""
    dotted = "argus_eval_pytaa_signal"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise PytaaSignalLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    path = _BASELINES_DIR / "pytaa_signal.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise PytaaSignalLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["PytaaSignalLoadError", "load_signal_module"]
