"""The five Cross-Asset Execution rivals, run from their own clones for `eval/xa_arena.py`.

Nothing here is vendored: none of Triad, Crossfire or VIGIL ships a licence file, and the
HedgeAgents replication's ``package.json`` says MIT with no LICENSE file either, so under this
project's licence rule they are *run*, never copied. (Omni is MIT and could be vendored; it is run
the same way so all five are treated alike.) Each runner in this directory is ARGUS's own thin
adapter: it streams the tape to the rival's own functions and maps the orders they emit onto the
arena book. Every rival source file a runner calls is pinned below by SHA-256 against the commit
named, and :func:`verify` refuses to run a clone whose bytes moved.

**Where the clones live.** By default under ``<workspace>/research/repos-rivals`` and
``<workspace>/research/repos-owned`` (``<workspace>`` is the directory that holds the ``argus``
project). Set ``ARGUS_RESEARCH_DIR`` to point anywhere else; the sub-directory names below are
the ones :func:`verify` expects, and its error message says what to clone where.

**What a pin hashes.** The committed bytes of each file, i.e. the working copy with CRLF line
endings folded to LF before hashing. A Windows checkout with ``core.autocrlf=true`` rewrites every
line ending, so hashing raw working-copy bytes would pin one machine's checkout rather than the
commit, and a clean Linux or macOS clone would be refused. Measured 2026-09-26: every pinned file
is LF in its commit (``git show HEAD:<path>``), and each hash below equals SHA-256 of that blob.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def research_dir() -> Path:
    """Root that holds ``repos-rivals/`` and ``repos-owned/`` (``ARGUS_RESEARCH_DIR`` wins)."""
    override = os.environ.get("ARGUS_RESEARCH_DIR")
    return Path(override) if override else Path(__file__).resolve().parents[6] / "research"


class RivalUnavailable(RuntimeError):
    """A clone is missing, drifted from its pinned bytes, or its runtime is not installed."""


PINS: dict[str, dict[str, object]] = {
    "triad": {
        "dir": ("repos-rivals", "danielamodu~Triad"),
        "url": "https://github.com/danielamodu/Triad", "commit": "d70c67b",
        "licence": "none (run only)",
        "files": {
            "config.py": "9740019e0ef80e8de1cd21db6c04adc3ef4b823cb7aebb8edf28c4b3c30af46e",
            "src/cli.py": "010d9830fd1782c618fc0fd8853d9387a114ca558088ef27667a20d56b121a53",
            "src/decision/engine.py":
                "6cb12d457de3fb5ab1ad524adfc10595285794b5d2bb53ec7a0922139f7fae50",
            "src/signals/price_divergence.py":
                "dc5091c85832d8ac9aaafff7f06ccc5de6662db1a846716c6253e366a2dd21c2",
            "src/signals/event_signal.py":
                "36559ed0581c79fd159473c17069aaeaa181ffb27205f28335152baf5b4c9c19",
            "src/signals/sentiment_signal.py":
                "ed08484aca68e3be5f83c58c146bd3da390346dc734dd97189308e8a28813f21",
            "src/risk/cage.py": "cbf04c119e3edaf98a391f0e1435583b8a41463c80cdf3032dee5191fceed118",
            "src/risk/state.py":
                "0d95f6b9a3c0fd612af96a3a12808f4451c97b2cc290f51d6b0b3e4f8d78f805",
            "src/execution/executor.py":
                "9e7704dac47a29d8475628bf5b22760e7a9a2935e3c1a4148d73b91bf99a7ac2",
            "backtest/harness.py":
                "59e011006705abb91ee3dbb62736c212c45dd31f97972266ce935576770fe9b1",
        },
    },
    "omni": {
        "dir": ("repos-rivals", "Jayanng~Omni"),
        "url": "https://github.com/Jayanng/Omni", "commit": "c50d566", "licence": "MIT",
        "files": {
            "omni/risk.py": "bced46ab78ba64209b3a4c6f9bf16e54be50fbcf0a9dd7b1dc845c48f97eaf23",
            "omni/policy.py": "15515e7c5a6d0f919ce8bfca956e30e01afd9a38e22fbb617b966974bb81f797",
            "omni/llm.py": "da844597729bd293561e3ba8718aa1d7eb92597e55f328599bd90ca3457cf030",
            "omni/scenario.py": "e8b73129d0e779fe58add62f5d52dd64865d5478b3d03ea1c9033dd98edf413d",
            "omni/session.py": "b12a7a832a789507ef0ecf36a4e9e64db4cf5e70f278e2f4c291beb0b6341e2b",
            "omni/collateral.py":
                "6e10492e89625d12572aaeb9733cca318631c34716c4ab1a0ee1b28fc90c7cb6",
            "omni/config.py": "2384bc7269b7d65296c6ec31fbfef0eda4fd62654caae2ff911060a718d371d6",
            "omni/venue_risk.py":
                "245c1d076e290cd89e68e5a2837e5ea6ebe2e1eeccce5783842de2166e2d7ce9",
            "omni/bitget_public.py":
                "96e0006ee6756666650319d4d3a91de76ec99e485fbb05a392442dee018f4461",
            "omni/executor.py":
                "24abd1db332a30a32768a6e2d4a08f1b0a3558b8968b00f47990df635118df1a",
        },
    },
    "crossfire": {
        "dir": ("repos-rivals", "CryptoCT01~Crossfire"),
        "url": "https://github.com/CryptoCT01/Crossfire", "commit": "7a3bdfa",
        "licence": "none (run only)",
        "files": {
            "crossfire/agent_engine.py":
                "3db3553bc102e24428402ada6d795126ee68893f544e1287e2df092d2fc8f5bc",
            "crossfire/config.py":
                "7cd089d036efba304eff771044ddd2eabd461c82f6579762c2daf3e7dec54f4c",
        },
    },
    "vigil": {
        "dir": ("repos-rivals", "norbert351~vigil"),
        "url": "https://github.com/norbert351/vigil", "commit": "3bfad2d",
        "licence": "package.json MIT, no LICENSE file (run only)",
        "files": {
            "src/risk.js": "4b3d6a0be7f2335b90fe008eed045b2359d71ee5ae217827aeaa0e437ff80d51",
            "src/regime.js": "6fb553465c437615f2ecf5dc7e31c0d52f8b69284624f7382d519492b1fc1973",
            "src/config.js": "3b28b709d8dd94f91fb2569aa8c20bf8d4113d9e3638b972e33d9848b4d3b054",
            "src/backtest.js": "569a72d3c9dffa13c69cc98cfea6fc7e86cffbb706aa81d6edc2340469236016",
        },
    },
    "hedgeagents": {
        "dir": ("repos-owned", "hedge-agents"),
        "url": "https://github.com/JansenAnalytics/hedge-agents", "commit": "8b87aaf",
        "licence": "package.json MIT, no LICENSE file (run only)",
        "files": {
            "src/utils/math.cjs":
                "a87f8bac51ae9885920cc4d47237440892143e6bbd7ad4666ee5e63d429ccbd7",
            "src/tools/domain.cjs":
                "92a70f3146e3ac885d84396497e6e7da46cd66f2044f7620a135cd91142f8ada",
            "src/tools/risk.cjs":
                "e080e90a062bcb73e5efdaf4d1bbbb146492978d79ff675f37b12fae4694d6f7",
            "src/conferences/emc.cjs":
                "a8d824146acff8b7a4adc22df9a710e6884d6a3c211c8e9ab94733d8cf2e9326",
            "src/conferences/bac.cjs":
                "afc3fab425a11a7e8f44ca12e7c0d6537af6504751539028058dbaa8a84800ac",
            "config/schedule.json":
                "4a4cfd34ef17a63671be31ff37135599b5da9fc7f9e84094fbba398a426c04cd",
            "config/portfolio.json":
                "57be44dd94d2aa752a7fdcf5c5f6f338a690af60eb361a6ee1a268a7aac4351c",
        },
    },
}


def committed_sha256(path: Path) -> str:
    """SHA-256 of a file's committed bytes: CRLF folded to LF (see the module docstring)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def clone_path(name: str) -> Path:
    where = PINS[name]["dir"]
    assert isinstance(where, tuple) and len(where) == 2
    group, folder = where
    return research_dir() / str(group) / str(folder)


def verify(name: str) -> Path:
    """The clone's path, after checking every pinned file's bytes. Raises on any drift."""
    pin = PINS[name]
    repo = clone_path(name)
    if not repo.is_dir():
        raise RivalUnavailable(f"{name}: clone missing at {repo}; clone {pin['url']} "
                               f"@{pin['commit']} there (or set ARGUS_RESEARCH_DIR)")
    files = pin["files"]
    assert isinstance(files, dict)
    for rel, want in files.items():
        path = repo / rel
        if not path.is_file():
            raise RivalUnavailable(f"{name}: {rel} missing")
        got = committed_sha256(path)
        if got != want:
            raise RivalUnavailable(f"{name}: {rel} changed since it was pinned "
                                   f"({got[:12]} != {want[:12]})")
    return repo


def node() -> str:
    exe = shutil.which("node")
    if exe is None:
        raise RivalUnavailable("Node.js is required to run VIGIL and HedgeAgents")
    return exe


def rival_python() -> str:
    """Interpreter for the Python rivals: ``ARGUS_RIVAL_PYTHON`` if set, else this one.

    The rivals import their own requirements, which ARGUS does not carry: Triad's ``config.py``
    imports ``python-dotenv`` (its ``requirements.txt``), which is absent from ARGUS's own
    environment (measured 2026-09-26: ``ModuleNotFoundError: No module named 'dotenv'``). Omni
    and Crossfire's keyless paths need only the standard library. Point this at an interpreter
    that has each rival's requirements installed; nothing is installed on its behalf.
    """
    return os.environ.get("ARGUS_RIVAL_PYTHON") or sys.executable


def commands() -> list[tuple[str, list[str], Path, dict[str, str]]]:
    """``(name, argv, cwd, extra_env)`` for every rival arm."""
    py = rival_python()
    return [
        ("triad", [py, str(HERE / "triad_runner.py"), "1"], verify("triad"), {}),
        ("triad-x10", [py, str(HERE / "triad_runner.py"), "10"], verify("triad"), {}),
        ("omni", [py, str(HERE / "omni_runner.py")], verify("omni"), {}),
        ("crossfire", [py, str(HERE / "crossfire_runner.py")], verify("crossfire"), {}),
        ("vigil", [node(), str(HERE / "vigil_runner.mjs")], verify("vigil"), {}),
        ("hedgeagents", [node(), str(HERE / "hedgeagents_runner.cjs"), "equal"],
         verify("hedgeagents"), {}),
        ("hedgeagents-optimizer", [node(), str(HERE / "hedgeagents_runner.cjs"), "optimizer"],
         verify("hedgeagents"), {}),
    ]


__all__ = [
    "PINS", "RivalUnavailable", "clone_path", "commands", "committed_sha256", "node",
    "research_dir", "rival_python", "verify",
]
