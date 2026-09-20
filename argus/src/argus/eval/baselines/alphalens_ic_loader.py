"""Make the vendored `alphalens_ic.py` (Alphalens' real Information Coefficient computation)
importable and runnable.

**Shim needed, and why.** The real `factor_information_coefficient` calls
`utils.get_forward_returns_columns(...)` — in the original repo, `utils` is a sibling module
imported via `from . import utils`. This vendoring concatenates both real functions into ONE file
(the same technique already used for `qlib_eval_surface.py` and `rdagent_cache_utils.py`), so
`get_forward_returns_columns` lives in the same namespace `factor_information_coefficient` itself
is defined in, not in a separate `utils` module. Rather than editing the vendored body to call it
directly (which would mean the "verbatim" region no longer matches the real upstream source byte
for byte), this loader injects a self-referential name `utils` pointing back at the same module
before executing it — the same technique `qlib_loader.py` already uses in this directory for
`re`/`abc`, and `quantconnect_preholiday_loader.py` for `self`/`holidays`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_alphalens_ic_shim"


class AlphalensIcLoadError(RuntimeError):
    """The vendored Alphalens IC excerpt could not be loaded."""


def load_ic_module() -> ModuleType:
    """Load `alphalens_ic.py`. Call `.factor_information_coefficient(factor_data)` exactly as
    the real `alphalens.performance` module does — `factor_data` is a `pandas.DataFrame` with a
    `(date, asset)` `MultiIndex`, a `"factor"` column, and one or more forward-return columns
    named like `"1D"`/`"5D"` (matched by `get_forward_returns_columns`'s own real regex). Call
    `.plot_information_table(ic_data, return_df=True)` on THAT function's own real output — the
    real, default, uncorrected significance test the library ships (`scipy.stats.ttest_1samp`)."""
    dotted = "argus_eval_alphalens_ic"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise AlphalensIcLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    path = _BASELINES_DIR / "alphalens_ic.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise AlphalensIcLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    module.utils = module  # type: ignore[attr-defined]
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["AlphalensIcLoadError", "load_ic_module"]
