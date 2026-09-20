"""Make the vendored `lean_pairs_ranking.py` (Lean's real pearsonr-ranking excerpt) importable.

Unlike every other loader in this package, this excerpt needs no shim at all: it references only
`df`, `stop`, and `pearsonr` — its own real, unedited parameters — never a module-level import of
its own. `importlib.util.spec_from_file_location` loads it directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_lean_pairs_ranking_shim"


class LeanPairsRankingLoadError(RuntimeError):
    """The vendored Lean pairs-ranking excerpt could not be loaded."""


def load_pairs_ranking_module() -> ModuleType:
    """Load `lean_pairs_ranking.py`. Call `.rank_pairs_by_correlation(df, stop, pearsonr)`
    exactly as Lean's real `on_securities_changed` does."""
    dotted = "argus_eval_lean_pairs_ranking"
    existing = sys.modules.get(dotted)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise LeanPairsRankingLoadError(
                f"sys.modules[{dotted!r}] already holds a module this loader did not create"
            )
        return existing

    path = _BASELINES_DIR / "lean_pairs_ranking.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    if spec is None or spec.loader is None:
        raise LeanPairsRankingLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["LeanPairsRankingLoadError", "load_pairs_ranking_module"]
