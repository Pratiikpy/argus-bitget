"""Import two unlicensed Season-2-era event-driven trading agents from local clones, unmodified.

Neither repository publishes a licence (``gh api repos/<owner>/<repo> --jq .license`` answered
``null`` for both on 2026-09-25, and neither has a LICENSE file), so under default copyright there
is no right to redistribute their code and none of it is vendored here. The route is the one
`blackout_hedges_loader.py` set: run the real files where they live, and record which bytes ran.

* **JohnboscoE/slimon** — an S2 Track 2 *Event-Driven Agent* entry trading Bitget US-stock
  perpetuals on the demo venue. Its perception layer (``slimon/perception.py``) turns closed
  5-minute candles into ``price_move`` / ``range_expansion`` / ``volume_spike`` events with its
  own thresholds from ``config/agent.toml``; its LLM (``qwen3.8-max``) then decides. The
  perception is deterministic and runs here unmodified; the LLM's real decisions are read from
  its own published, append-only decision log rather than re-bought.
* **Nicholas-03/trading-bot** — a news-trading bot (Alpaca news, ChatGPT decision, Tradier
  orders). Its deterministic half, ``news/filters.py``, is a hard-catalyst gate applied before the
  model is asked; it runs here unmodified on real headlines.

If a clone is missing this raises and names the clone command — it never substitutes a
reimplementation, because a paraphrase of a rival is what ``baseline_reproduced`` refuses.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

_WORKSPACE = Path(__file__).resolve().parents[5]
SLIMON_CLONE = _WORKSPACE / "research" / "repos-owned" / "slimon"
NICHOLAS_CLONE = _WORKSPACE / "research" / "repos-owned" / "Nicholas-03~trading-bot"
SLIMON_URL = "https://github.com/JohnboscoE/slimon.git"
NICHOLAS_URL = "https://github.com/Nicholas-03/trading-bot.git"
SLIMON_READ_AT = "7d0ae2dc22c197e3dc771ec8185afbb84c8e7d6f"
NICHOLAS_READ_AT = "7cf49b8d9822681ff5ca0a5d7dce783347e64c1e"
_NICHOLAS_DOTTED = "argus_eval_nicholas03_news_filters"


class AgentLoadError(RuntimeError):
    """A rival agent's real code could not be located or imported."""


def _require(path: Path, url: str, clone: Path) -> None:
    if not path.is_file():
        raise AgentLoadError(
            f"{path} not found. Get it with:  git clone {url} '{clone}'. This comparison will not "
            f"substitute a reimplementation of their code for their code."
        )


def load_slimon() -> dict[str, ModuleType]:
    """slimon's real ``perception``, ``market``, ``broker``, ``journal`` and ``report`` modules.

    ``SLIMON_STATE_DIR`` — their own documented override "so tests and dev runs never write into
    the published log" — is pointed at a throwaway directory before import, so nothing run here can
    touch the clone's committed state. ``SLIMON_LOG_DIR`` is pointed at the clone's own published
    logs, which their ``report.summary`` reads; nothing here writes a log (their ``Journal``, the
    only writer, is never constructed).
    """
    _require(SLIMON_CLONE / "slimon" / "perception.py", SLIMON_URL, SLIMON_CLONE)
    scratch = Path(tempfile.gettempdir()) / "argus_slimon_replay"
    os.environ.setdefault("SLIMON_STATE_DIR", str(scratch / "state"))
    os.environ.setdefault("SLIMON_LOG_DIR", str(SLIMON_CLONE / "logs"))
    root = str(SLIMON_CLONE)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import slimon.broker as broker  # type: ignore[import-not-found]
        import slimon.journal as journal  # type: ignore[import-not-found]
        import slimon.market as market  # type: ignore[import-not-found]
        import slimon.perception as perception  # type: ignore[import-not-found]
        import slimon.report as report  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only on a broken clone or missing deps
        raise AgentLoadError(f"slimon clone at {root} did not import: {exc}") from exc
    for module in (perception, market, broker, journal, report):
        if not str(module.__file__).startswith(root):
            raise AgentLoadError(f"{module.__name__} resolved outside the clone: {module.__file__}")
    return {"perception": perception, "market": market, "broker": broker, "journal": journal,
            "report": report}


def slimon_config() -> dict[str, Any]:
    """Their committed ``config/agent.toml``, parsed as-is."""
    path = SLIMON_CLONE / "config" / "agent.toml"
    _require(path, SLIMON_URL, SLIMON_CLONE)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_nicholas_filters() -> ModuleType:
    """Nicholas-03's real ``news/filters.py`` (stdlib only), under a private module name."""
    existing = sys.modules.get(_NICHOLAS_DOTTED)
    if existing is not None:
        return existing
    path = NICHOLAS_CLONE / "news" / "filters.py"
    _require(path, NICHOLAS_URL, NICHOLAS_CLONE)
    spec = importlib.util.spec_from_file_location(_NICHOLAS_DOTTED, path)
    if spec is None or spec.loader is None:
        raise AgentLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NICHOLAS_DOTTED] = module
    spec.loader.exec_module(module)
    return module


def _commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30, check=False).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent
        return "unknown"


def provenance() -> dict[str, dict[str, str]]:
    """Which commit of each clone actually ran, beside the commit that was read."""
    out = {}
    for name, repo, read_at in (("slimon", SLIMON_CLONE, SLIMON_READ_AT),
                                ("nicholas_03_trading_bot", NICHOLAS_CLONE, NICHOLAS_READ_AT)):
        commit = _commit(repo) if repo.exists() else "absent"
        # Relative to the workspace: an artefact must not carry the machine's own home path.
        where = repo.relative_to(_WORKSPACE).as_posix()
        out[name] = {"clone": where, "commit": commit, "read_at_commit": read_at,
                     "commit_matches_the_one_read": str(commit == read_at).lower(),
                     "licence": "none published (GitHub API license: null, no LICENSE file); "
                                "run from the clone, not vendored"}
    return out


__all__ = [
    "NICHOLAS_CLONE",
    "SLIMON_CLONE",
    "AgentLoadError",
    "load_nicholas_filters",
    "load_slimon",
    "provenance",
    "slimon_config",
]
