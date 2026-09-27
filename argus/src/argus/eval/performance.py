"""Re-export of :mod:`argus.paper.performance`, where the ledger's Sharpe, drawdown and win rate
are computed since 2026-09-27 (audit finding 163). Kept so the documents and artefacts that cite
``eval/performance.py`` still resolve."""

from __future__ import annotations

from argus.paper.performance import *  # noqa: F403
from argus.paper.performance import __all__ as __all__
