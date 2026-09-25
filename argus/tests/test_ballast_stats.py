"""The vendored Ballast ``stats.py`` (`argus.eval.baselines.ballast_stats`) is the upstream file.

The header names the upstream commit, the git blob id and a SHA-256 of the body below the
"VENDORED FROM HERE" marker. Both digests are recomputed here from the file itself, so an edit to
the vendored body fails the suite rather than silently changing the rival being measured.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "src" / "argus" / "eval" / "baselines" / "ballast_stats.py"
MARKER = b"# ============================== VENDORED FROM HERE ==============================\n"


def _body() -> bytes:
    raw = VENDORED.read_bytes().replace(b"\r\n", b"\n")
    at = raw.find(MARKER)
    assert at >= 0, "the vendoring marker is missing"
    return raw[at + len(MARKER):]


def _header(name: str) -> str:
    found = re.search(rf"^# {name}:\s+(\S+)", VENDORED.read_text(encoding="utf-8"), re.M)
    assert found is not None, f"header line {name!r} is missing"
    return found.group(1)


def _module() -> Any:
    return importlib.import_module("argus.eval.baselines.ballast_stats")


def test_the_body_matches_the_recorded_sha256_and_git_blob() -> None:
    body = _body()
    recorded = re.search(r"SHA256 of the vendored body: ([0-9a-f]{64})",
                         VENDORED.read_text(encoding="utf-8"))
    assert recorded is not None
    assert hashlib.sha256(body).hexdigest() == recorded.group(1)
    blob = hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()  # git's own object id
    assert blob == _header("Blob")


def test_t_stat_is_the_plain_pooled_one_sample_t() -> None:
    values = [0.01, -0.02, 0.03, 0.015, -0.005, 0.02, 0.01, 0.0, 0.025]
    mean, t = _module().t_stat(values)
    n = len(values)
    mu = sum(values) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))
    assert math.isclose(mean, mu, rel_tol=1e-12)  # statistics.mean sums exactly
    assert math.isclose(t, mu / (sd / math.sqrt(n)), rel_tol=1e-12)


def test_t_stat_declines_below_eight_observations() -> None:
    mean, t = _module().t_stat([0.1] * 7)
    assert math.isnan(mean) and math.isnan(t)
