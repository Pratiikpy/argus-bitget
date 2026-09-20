"""Import Blackout Desk's real `src/blackout/hedges.py` from a local clone, without vendoring it.

Every other baseline in this package is a verbatim copy sitting next to a provenance header,
because `baselines/__init__.py` permits vendoring only when "(1) its licence permits
redistribution". **Modemola/BITGET_HACK carries no licence at all** — no `LICENSE` file, and the
GitHub API reports ``license: null`` (checked with ``gh api repos/Modemola/BITGET_HACK --jq
.license`` while writing this). Under default copyright that means no redistribution right, so
copying `hedges.py` into this repository is not available to us however convenient it would be.
The route that remains is the one `eval/sentiment_comparison.py` already uses for finBERT: run the
real published artefact where it lives, and record exactly which bytes ran.

So this loader imports the real, unmodified file from a local clone and reports its resolved path
and git commit alongside every number derived from it. Two resolution routes, in order:

1. ``import blackout.hedges`` — succeeds when the clone has been ``pip install -e``'d, which is
   what the upstream repo's own ``CLAUDE.md`` tells a reader to do (``pip install -e ".[dev]"``).
2. Direct execution of ``<clone>/src/blackout/hedges.py`` by file path, with ``<clone>/src`` put on
   ``sys.path`` so the file's own verbatim ``from .clock import ClosureClock, Regime`` resolves.

Neither route edits a byte of their source, and route 2 is what makes the comparison runnable on a
machine where only the clone exists. If neither resolves, this raises and names the exact clone
command — deliberately, rather than degrading to a reimplementation of their method, because a
paraphrase of a baseline is the thing `eval/standing.py`'s ``baseline_reproduced`` condition exists
to refuse.

What is being loaded (read at ``src/blackout/hedges.py``, commit 0dfb298, before any of this was
written): ``quote_hedge`` at lines 96-125 measures a candidate hedge's correlation, OLS ratio,
``1 - residual_std/unhedged_std``, a rolling correlation and a sign-flip count; ``HedgeQuote``
at 44-71 carries ``is_stable`` = ``sign_flips == 0 and corr_min > 0.2`` and the verdict string;
``window_returns`` at 74-93 reduces an hourly frame to one return per closure window using their
own ``ClosureClock``; ``hedge_menu`` at 128-148 ranks candidates and attaches ``COSTS`` (34-38).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

# The clone this project keeps, relative to the argus package root's parent. Falls back to an
# importable installed package, so neither path is load-bearing on its own.
_DEFAULT_CLONE = (
    Path(__file__).resolve().parents[5] / "research" / "repos-rivals" / "modemola~BITGET_HACK"
)

CLONE_URL = "https://github.com/Modemola/BITGET_HACK.git"
READ_AT_COMMIT = "0dfb298a0759c1ce14f80125f7786c89a93adb4c"
"""The commit whose ``src/blackout/hedges.py`` was read line by line before this comparison was
designed. A later commit is not an error — :func:`provenance` reports whatever actually ran, and
the comparison publishes it — but a reader comparing our description of their mechanism against
their file should check out this one."""


class BlackoutLoadError(RuntimeError):
    """Blackout Desk's real `hedges.py` could not be located or executed."""


def _clone_candidates(clone: Path | None) -> list[Path]:
    return [clone] if clone is not None else [_DEFAULT_CLONE]


def load_hedges_module(*, clone: Path | None = None) -> ModuleType:
    """Return Blackout Desk's real, unmodified ``blackout.hedges`` module.

    Carries ``quote_hedge``, ``hedge_menu``, ``window_returns``, ``HedgeQuote``, ``COSTS`` and
    ``MIN_WINDOWS`` exactly as their file defines them.
    """
    try:
        import blackout.hedges as installed
    except ImportError:
        pass
    else:
        return cast(ModuleType, installed)

    for candidate in _clone_candidates(clone):
        src = candidate / "src"
        if not (src / "blackout" / "hedges.py").is_file():
            continue
        # Putting <clone>/src on sys.path rather than exec'ing the file under a synthetic package
        # name: `hedges.py` does `from .clock import ...` and `clock.py` in turn imports
        # exchange_calendars, so the cheapest way to satisfy both is to let Python's own import
        # machinery see the real package directory the real file expects to live in.
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            import blackout.hedges as from_clone
        except ImportError as exc:  # pragma: no cover - only on a broken clone
            raise BlackoutLoadError(
                f"found {src / 'blackout' / 'hedges.py'} but could not import it: {exc}. Their "
                f"`clock.py` needs `exchange_calendars`; `pip install exchange_calendars` fixes it."
            ) from exc
        return cast(ModuleType, from_clone)

    raise BlackoutLoadError(
        "Blackout Desk's real hedges.py is not importable and no clone was found at "
        f"{_DEFAULT_CLONE}. Get it with:  git clone {CLONE_URL} '{_DEFAULT_CLONE}'  then "
        "`pip install exchange_calendars`. This comparison will not substitute a reimplementation "
        "of their method for their own code."
    )


def load_clock_module(*, clone: Path | None = None) -> ModuleType:
    """Their real ``blackout.clock``, carrying ``ClosureClock`` and ``Regime``.

    Resolved only after :func:`load_hedges_module` has put the clone's ``src`` on ``sys.path``,
    because that is the step that makes the package importable at all.
    """
    load_hedges_module(clone=clone)
    import blackout.clock as clock_module

    return cast(ModuleType, clock_module)


def provenance(*, clone: Path | None = None) -> dict[str, str]:
    """Where the baseline that just ran actually came from, so a number can be traced to bytes.

    Reports the resolved file and the clone's current commit. A clone at a different commit from
    :data:`READ_AT_COMMIT` is reported rather than rejected — the honest record is what ran, not
    what we wish had run.
    """
    module = load_hedges_module(clone=clone)
    path = Path(str(module.__file__))
    repo = path.resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent
        commit = ""
    return {
        "file": str(path),
        "commit": commit or "unknown",
        "read_at_commit": READ_AT_COMMIT,
        "commit_matches_the_one_read": str(commit == READ_AT_COMMIT).lower(),
        "licence": "none published (GitHub API reports license: null; no LICENSE file in the "
                   "repository) — which is why this file is loaded from a clone and not vendored",
    }


__all__ = [
    "CLONE_URL",
    "READ_AT_COMMIT",
    "BlackoutLoadError",
    "load_clock_module",
    "load_hedges_module",
    "provenance",
]
