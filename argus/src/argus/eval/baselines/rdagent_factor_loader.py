"""Make the vendored RD-Agent factor-execution path
(`rdagent_experiment.py`/`rdagent_factor.py`/`rdagent_costeer_task.py`/`rdagent_exception.py`/
`rdagent_cache_utils.py`) importable and CALLABLE without editing a line of any of them.

Unlike the qlib expression engine loaders in this package, every one of these four files (all but
`rdagent_cache_utils.py`) is vendored WHOLE and keeps its own original `from rdagent.X.Y import Z`
lines verbatim — so making them work is a `sys.modules` shim, not a name-prepopulation trick: each
vendored file is registered in `sys.modules` under its OWN REAL dotted path
(`rdagent.core.experiment`, `rdagent.components.coder.CoSTEER.task`, `rdagent.core.exception`,
`rdagent.components.coder.factor_coder.factor`), with every parent package
(`rdagent`, `rdagent.core`, `rdagent.components`, ...) shimmed as an empty namespace first, so a
real `from rdagent.core.experiment import Task` line inside another vendored file finds the REAL
vendored `Task` class when it runs — exactly the qlib loaders' own `_shim_package` technique,
applied to a deeper real package tree.

Real names satisfied, by category:

1. **Real, vendored, unmodified code** — `FBWorkspace` (`rdagent_experiment.py`), `CoSTEERTask`
   (`rdagent_costeer_task.py`), `CodeFormatError`/`CustomRuntimeError`/`NoOutputError`
   (`rdagent_exception.py`), `cache_with_pickle`/`md5_hash` (`rdagent_cache_utils.py`) — loaded
   from disk and never edited.
2. **Real third-party package** — `filelock.FileLock`, already installed (transitively via
   transformers/huggingface_hub, declared directly in `pyproject.toml`'s dev extras).
3. **Real default VALUES, not vendored logic** — `FACTOR_COSTEER_SETTINGS`'s four fields
   `execute()` actually reads (`python_bin`, `file_based_execution_timeout`, `data_folder`,
   `data_folder_debug`), copied from `rdagent/components/coder/factor_coder/config.py:13-29` at
   the vendored commit. The surrounding pydantic-settings/`CondaConf`/`LocalEnv`/`get_factor_env`
   machinery around those four lines is orthogonal to the execution-surface finding this
   comparison exists to demonstrate, so only the values are carried over, not vendored.
4. **A real, deliberately-set configuration value that makes real code take its own real
   fast-exit branch** — `RD_AGENT_SETTINGS.cache_with_pickle = False` (the real DEFAULT is `True`,
   `rdagent/core/conf.py:70`). The real, unmodified `cache_with_pickle` decorator's own first line
   is `if not RD_AGENT_SETTINGS.cache_with_pickle and not force: return func(*args, **kwargs)` —
   set this way, it genuinely returns immediately and never reaches `FileLock`/
   `UntrustedArtifactError`/`secure_pickle_dump`/`secure_pickle_load`, so those four names are
   shimmed as unreachable stubs (fail loud, not silent, if this analysis is ever wrong).
5. **Unreachable stand-ins for a code path never exercised** — `KAGGLE_IMPLEMENT_SETTING` (only
   read inside `self.target_task.version == 2`, never taken since every task here uses the real
   default `version=1`), `rdagent.core.serialization`'s three names (only reached past the
   `cache_with_pickle=False` fast exit), `rdagent.core.evaluation.Feedback` (only a `TypeVar`
   bound, never instantiated), `Experiment` (only assigned to two module-level aliases in
   `rdagent_factor.py`, `FactorExperiment`/`FeatureExperiment`, never constructed or subclassed
   here).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any

_BASELINES_DIR = Path(__file__).resolve().parent
_SHIM_MARKER = "_argus_rdagent_factor_shim"


class RdAgentFactorLoadError(RuntimeError):
    """The vendored RD-Agent factor-execution baseline could not be loaded."""


def _make_unreachable(name: str, reason: str) -> Any:
    """A stand-in for a CALLED name — raises `NotImplementedError` if invoked."""

    def _raise(*args: object, **kwargs: object) -> object:
        raise NotImplementedError(
            f"{name} is a stub — {reason} — reaching this means a test assumption is wrong"
        )

    _raise.__name__ = name
    return _raise


class _UnreachablePathLike:
    """A stand-in for an ATTRIBUTE used directly as a path-like value (e.g.
    `Path(KAGGLE_IMPLEMENT_SETTING.local_data_path)`) — a callable stub is the wrong shape here
    since the real code never CALLS it, only reads it and hands it to `Path()`/`os.fspath()`,
    which `_make_unreachable`'s function object cannot satisfy (found by a test that genuinely
    tried to reach this code path and got `TypeError: expected str, bytes or os.PathLike object,
    not function` instead of the intended fail-loud `NotImplementedError`)."""

    def __init__(self, name: str, reason: str) -> None:
        self._name = name
        self._reason = reason

    def __fspath__(self) -> str:
        raise NotImplementedError(
            f"{self._name} is a stub — {self._reason} — "
            f"reaching this means a test assumption is wrong"
        )

    def __str__(self) -> str:
        return self.__fspath__()


def _shim_package(dotted_name: str) -> ModuleType:
    """An empty namespace package, or the one this loader already created."""
    existing = sys.modules.get(dotted_name)
    if existing is not None:
        if not getattr(existing, _SHIM_MARKER, False):
            raise RdAgentFactorLoadError(
                f"sys.modules[{dotted_name!r}] already holds a module this loader did not create"
            )
        return existing
    module = types.ModuleType(dotted_name)
    setattr(module, _SHIM_MARKER, True)
    sys.modules[dotted_name] = module
    return module


def _shim_leaf(dotted_name: str, **attrs: Any) -> ModuleType:
    """A leaf module carrying real values/stand-ins as attributes — not loaded from a vendored
    file, because nothing here vendors it (it is genuinely external to the execution-surface
    finding, or is a real default VALUE rather than logic — see the module docstring)."""
    module = _shim_package(dotted_name)
    for k, v in attrs.items():
        setattr(module, k, v)
    return module


def _load_vendored(real_dotted_name: str, path: Path) -> ModuleType:
    """Load a whole vendored file and register it at its OWN real dotted path, so another
    vendored file's real `from <real_dotted_name> import X` line finds it."""
    existing = sys.modules.get(real_dotted_name)
    if existing is not None and getattr(existing, _SHIM_MARKER, False):
        return existing
    spec = importlib.util.spec_from_file_location(real_dotted_name, path)
    if spec is None or spec.loader is None:
        raise RdAgentFactorLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[real_dotted_name] = module
    spec.loader.exec_module(module)
    setattr(module, _SHIM_MARKER, True)
    return module


def load_factor_module() -> tuple[ModuleType, ModuleType]:
    """Load the vendored RD-Agent factor-execution chain with every real dependency satisfied.

    Returns:
        (experiment_module, factor_module) — `factor_module` carries the real, unmodified
        `FactorTask` and `FactorFBWorkspace` (whose `.execute()` is this comparison's subject).
    """
    for pkg in (
        "rdagent",
        "rdagent.core",
        "rdagent.components",
        "rdagent.components.coder",
        "rdagent.components.coder.CoSTEER",
        "rdagent.components.coder.factor_coder",
        "rdagent.app",
        "rdagent.app.kaggle",
        "rdagent.oai",
    ):
        _shim_package(pkg)

    # (4) RD_AGENT_SETTINGS: workspace_path is real (a temp dir under the scratch data folder);
    # cache_with_pickle=False deliberately takes the real decorator's own real fast-exit branch.
    settings_ns = types.SimpleNamespace(
        workspace_path=Path(__file__).resolve().parents[4] / "data" / "_rdagent_workspace_tmp",
        cache_with_pickle=False,
        use_file_lock=False,
        pickle_cache_folder_path_str=str(
            Path(__file__).resolve().parents[4] / "data" / "_rdagent_pickle_cache_tmp"
        ),
    )
    _shim_leaf("rdagent.core.conf", RD_AGENT_SETTINGS=settings_ns)

    class _FeedbackStub:
        """(5) Only used as `TypeVar("ASpecificFeedback", bound=Feedback)` — never instantiated."""

    _shim_leaf("rdagent.core.evaluation", Feedback=_FeedbackStub)

    _shim_leaf(
        "rdagent.app.kaggle.conf",
        KAGGLE_IMPLEMENT_SETTING=types.SimpleNamespace(
            local_data_path=_UnreachablePathLike(
                "KAGGLE_IMPLEMENT_SETTING.local_data_path",
                "only read for target_task.version == 2, never used here",
            ),
            competition=_UnreachablePathLike(
                "KAGGLE_IMPLEMENT_SETTING.competition",
                "only read for target_task.version == 2, never used here",
            ),
        ),
    )

    # (3) Real default VALUES from rdagent/components/coder/factor_coder/config.py:13-29 at the
    # vendored commit — copied, not computed; the pydantic-settings class itself is not vendored.
    _shim_leaf(
        "rdagent.components.coder.factor_coder.config",
        FACTOR_COSTEER_SETTINGS=types.SimpleNamespace(
            data_folder="git_ignore_folder/factor_implementation_source_data",
            data_folder_debug="git_ignore_folder/factor_implementation_source_data_debug",
            file_based_execution_timeout=3600,
            python_bin=sys.executable,
        ),
    )

    class UntrustedArtifactError(Exception):
        """(5) Only raised/caught past the cache_with_pickle=False fast exit, never reached."""

    _shim_leaf(
        "rdagent.core.serialization",
        UntrustedArtifactError=UntrustedArtifactError,
        dump=_make_unreachable(
            "serialization.dump", "only reached past the cache_with_pickle=False fast exit"
        ),
        load=_make_unreachable(
            "serialization.load", "only reached past the cache_with_pickle=False fast exit"
        ),
    )
    _shim_leaf("rdagent.oai.llm_conf", LLM_SETTINGS=types.SimpleNamespace())

    # rdagent_cache_utils.py's vendored `cache_with_pickle`/`md5_hash` bodies were extracted
    # WITHOUT their original files' own import lines (only the function bodies were vendored —
    # see that file's own header), so its free names ARE pre-populated (the one case where that
    # trick is correct: there is no `from X import Y` statement inside it to override them).
    import functools
    import hashlib

    from filelock import FileLock

    cache_spec = importlib.util.spec_from_file_location(
        "argus_eval_rdagent_cache_utils", _BASELINES_DIR / "rdagent_cache_utils.py"
    )
    if cache_spec is None or cache_spec.loader is None:
        raise RdAgentFactorLoadError("could not build an import spec for rdagent_cache_utils.py")
    cache_module = importlib.util.module_from_spec(cache_spec)
    cache_module.Any = Any  # type: ignore[attr-defined]
    cache_module.Callable = Any  # type: ignore[attr-defined]
    cache_module.functools = functools  # type: ignore[attr-defined]
    cache_module.Path = Path  # type: ignore[attr-defined]
    cache_module.FileLock = FileLock  # type: ignore[attr-defined]
    cache_module.RD_AGENT_SETTINGS = settings_ns  # type: ignore[attr-defined]
    cache_module.UntrustedArtifactError = UntrustedArtifactError  # type: ignore[attr-defined]
    cache_module.secure_pickle_dump = _make_unreachable(  # type: ignore[attr-defined]
        "secure_pickle_dump", "only reached past the cache_with_pickle=False fast exit"
    )
    cache_module.secure_pickle_load = _make_unreachable(  # type: ignore[attr-defined]
        "secure_pickle_load", "only reached past the cache_with_pickle=False fast exit"
    )
    cache_module.hashlib = hashlib  # type: ignore[attr-defined]
    sys.modules["argus_eval_rdagent_cache_utils"] = cache_module
    cache_spec.loader.exec_module(cache_module)
    setattr(cache_module, _SHIM_MARKER, True)

    # `rdagent_factor.py`'s real `from rdagent.core.utils import cache_with_pickle` and
    # `from rdagent.oai.llm_utils import md5_hash` lines need these two REAL dotted paths to
    # carry the real, vendored functions above.
    _shim_leaf("rdagent.core.utils", cache_with_pickle=cache_module.cache_with_pickle)
    _shim_leaf("rdagent.oai.llm_utils", md5_hash=cache_module.md5_hash)

    experiment_module = _load_vendored(
        "rdagent.core.experiment", _BASELINES_DIR / "rdagent_experiment.py"
    )
    _load_vendored(
        "rdagent.components.coder.CoSTEER.task", _BASELINES_DIR / "rdagent_costeer_task.py"
    )
    _load_vendored("rdagent.core.exception", _BASELINES_DIR / "rdagent_exception.py")

    # `rdagent_factor.py`'s own real `from rdagent.core.experiment import Experiment, FBWorkspace`
    # line needs `Experiment` on that real module too — the vendored excerpt stops at line 410,
    # right before the real `class Experiment(...)` — so it is added here as an unreachable stand-
    # in (see (5) above), on the SAME real module object `FBWorkspace` was just loaded onto.
    if not hasattr(experiment_module, "Experiment"):
        experiment_module.Experiment = _make_unreachable(  # type: ignore[attr-defined]
            "Experiment", "only assigned to FactorExperiment/FeatureExperiment aliases, "
            "never constructed or subclassed here"
        )

    factor_module = _load_vendored(
        "rdagent.components.coder.factor_coder.factor", _BASELINES_DIR / "rdagent_factor.py"
    )
    return experiment_module, factor_module


__all__ = ["RdAgentFactorLoadError", "load_factor_module"]
