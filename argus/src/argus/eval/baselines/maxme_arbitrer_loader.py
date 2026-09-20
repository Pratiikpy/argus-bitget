"""Make the vendored `maxme_arbitrer.py` (maxme/bitcoin-arbitrage's real profit-detection core)
importable and CALLABLE without editing it.

No shim needed at all: the excerpt's own wrapper class (`ArbitrerProfitDetector`, this vendoring's
own addition — see that file's header) takes `depths`/`max_tx_volume` as constructor arguments
instead of reading them off `self` the way the real `Arbitrer` class does, and every method body
below the marker is the real file's own, referencing only `self`, stdlib built-ins, and its own
arguments — no external import at all.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_maxme_arbitrer_shim"


class MaxmeArbitrerLoadError(RuntimeError):
    """The vendored maxme/bitcoin-arbitrage profit-detection excerpt could not be loaded."""


def load_arbitrer_module() -> ModuleType:
    """Load `maxme_arbitrer.py`. Construct `.ArbitrerProfitDetector(depths, max_tx_volume)` and
    call `.arbitrage_depth_opportunity(kask, kbid)` exactly as the real `Arbitrer` class does."""
    dotted = "argus_eval_maxme_arbitrer"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise MaxmeArbitrerLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    path = _BASELINES_DIR / "maxme_arbitrer.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise MaxmeArbitrerLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["MaxmeArbitrerLoadError", "load_arbitrer_module"]
