"""Make the vendored `whale_signals_event_study.py` (whale-signals' real hit-rate significance
test) importable.

No shim needed: `compute_hit_rates`/`compute_base_rate` reference only real, ordinary
`numpy`/`pandas`/`scipy.stats`, already installed, never an internal whale-signals module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_whale_signals_event_study_shim"


class WhaleSignalsEventStudyLoadError(RuntimeError):
    """The vendored whale-signals event-study excerpt could not be loaded."""


def load_event_study_module() -> ModuleType:
    """Load `whale_signals_event_study.py`. Call `.compute_hit_rates(events_df)` and
    `.compute_base_rate(price_df, direction, horizon, condition_mask=None)` exactly as
    whale-signals' real `src/analysis/event_study.py` does."""
    dotted = "argus_eval_whale_signals_event_study"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise WhaleSignalsEventStudyLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    path = _BASELINES_DIR / "whale_signals_event_study.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise WhaleSignalsEventStudyLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["WhaleSignalsEventStudyLoadError", "load_event_study_module"]
