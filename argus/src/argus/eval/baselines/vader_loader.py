"""Run VADER from the local clone of ``cjhutto/vaderSentiment``, unmodified, as a baseline.

VADER (Hutto & Gilbert, ICWSM 2014) is the lexicon-and-rules scorer written for social-media text,
and the one a reader who knows the field expects beside finBERT on a tweet benchmark: finBERT was
trained on financial news, VADER on the register tweets are written in. It is MIT-licensed
(``gh api repos/cjhutto/vaderSentiment --jq .license.spdx_id`` answers ``MIT`` on 2026-09-25), so it
could be vendored, but nothing is gained by a copy: the clone at
``research/gpt_suggestions/repos/cjhutto~vaderSentiment`` is imported by file path, exactly as
`blackout_hedges_loader.py` does for a baseline it may not copy, and :func:`provenance` records the
file and commit that ran.

What is used, read before this was written: ``SentimentIntensityAnalyzer`` at
``vaderSentiment/vaderSentiment.py:200-215`` (it loads ``vader_lexicon.txt`` and
``emoji_utf8_lexicon.txt`` from its own directory) and ``polarity_scores`` at ``:239``, whose
``compound`` score the project README maps to a class at ``README.rst:243-245`` — positive at
``>= 0.05``, negative at ``<= -0.05``, neutral between. :func:`label` applies exactly those
thresholds; they are the authors' published convention, not tuned here. The file's ``__main__``
block (``:521`` onward) imports ``requests`` for a translation demo (``:650``); loading the module
under another name does not run it, so no network is touched.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

_DEFAULT_CLONE = Path(__file__).resolve().parents[2] / "vendor" / "vader"
"""VADER vendored verbatim at ``READ_AT_COMMIT`` under its MIT licence
(``vendor/vader/LICENSE.txt``, 2026-09-26), because the crowd read now scores tone with it on every
sweep (`market/social_pulse.py`) and a public clone of ARGUS must run without the research
checkout. The clone layout
(``<root>/vaderSentiment/vaderSentiment.py``) is still accepted through ``clone=``."""
CLONE_URL = "https://github.com/cjhutto/vaderSentiment.git"
READ_AT_COMMIT = "44fc044cd877310ee8278a0eadf34bcd50d41d06"
POSITIVE_AT = 0.05
NEGATIVE_AT = -0.05
"""The README's thresholds on ``compound`` (``README.rst:243-245``)."""

_MODULE: ModuleType | None = None


class VaderLoadError(RuntimeError):
    """The VADER clone is missing or could not be imported."""


class Analyzer(Protocol):
    def polarity_scores(self, text: str) -> dict[str, float]: ...


def _module_path(clone: Path | None) -> Path:
    if clone is None:
        return _DEFAULT_CLONE / "vaderSentiment.py"
    return clone / "vaderSentiment" / "vaderSentiment.py"


def load_analyzer(*, clone: Path | None = None) -> Analyzer:
    """VADER's own ``SentimentIntensityAnalyzer``, from the clone, unmodified."""
    global _MODULE
    path = _module_path(clone)
    if _MODULE is None or Path(str(_MODULE.__file__)) != path:
        if not path.is_file():
            raise VaderLoadError(
                f"no VADER at {path}. The vendored copy ships in argus/vendor/vader; a clone "
                f"comes from  git clone {CLONE_URL}. Nothing substitutes a reimplementation."
            )
        spec = importlib.util.spec_from_file_location("argus_vader_baseline", path)
        if spec is None or spec.loader is None:  # pragma: no cover - importlib contract
            raise VaderLoadError(f"could not build an import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
    return cast(Analyzer, _MODULE.SentimentIntensityAnalyzer())


def label(compound: float) -> str:
    """``positive``, ``negative`` or ``neutral`` by the README's thresholds."""
    if compound >= POSITIVE_AT:
        return "positive"
    if compound <= NEGATIVE_AT:
        return "negative"
    return "neutral"


VENDORED_SHA256 = {
    "vaderSentiment.py": "4177fb710d6f0548c90e087be90b8611716e4606f48221fc22dbefbdec90bfa4",
    "vader_lexicon.txt": "19b824842e261209723ded0c241d5a1e7da43e605330603dcbda2e4af6e527d6",
}
"""The vendored files' hashes, taken from the clone at ``READ_AT_COMMIT`` when they were copied:
a vendored copy is only "unmodified" while these still match."""


def provenance(*, clone: Path | None = None) -> dict[str, str]:
    """The file and commit that ran, so every VADER number traces to bytes."""
    import hashlib

    path = _module_path(clone)
    if clone is None:
        digests = {name: hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
                   for name in VENDORED_SHA256}
        commit = READ_AT_COMMIT if digests == VENDORED_SHA256 else "modified"
        licence = "MIT (Copyright (c) 2016 C.J. Hutto); vendored verbatim, hashes checked"
    else:
        try:
            commit = subprocess.run(["git", "-C", str(path.parents[1]), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, timeout=30,
                                    check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent
            commit = ""
        licence = "MIT (Copyright (c) 2016 C.J. Hutto); run from a clone"
    return {
        "file": str(path),
        "commit": commit or "unknown",
        "read_at_commit": READ_AT_COMMIT,
        "commit_matches_the_one_read": str(commit == READ_AT_COMMIT).lower(),
        "licence": licence,
        "thresholds": f"compound >= {POSITIVE_AT} positive, <= {NEGATIVE_AT} negative "
                      "(README.rst:243-245)",
    }


__all__ = [
    "CLONE_URL",
    "NEGATIVE_AT",
    "POSITIVE_AT",
    "READ_AT_COMMIT",
    "Analyzer",
    "VaderLoadError",
    "label",
    "load_analyzer",
    "provenance",
]
