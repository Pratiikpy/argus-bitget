"""Importing hftbacktest from ``HFTBACKTEST_SITE`` without making it an ARGUS dependency."""

from __future__ import annotations

import os
import sys
from typing import Any


class RivalUnavailable(RuntimeError):
    """hftbacktest could not be imported; the rival run is reported as not run, never faked."""


def import_hftbacktest() -> Any:
    site = os.environ.get("HFTBACKTEST_SITE")
    if site and site not in sys.path:
        sys.path.insert(0, site)
    try:
        import hftbacktest  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        raise RivalUnavailable(
            "hftbacktest is not importable; set HFTBACKTEST_SITE to a directory holding "
            "`pip install --no-deps --target <dir> hftbacktest==2.4.4`"
        ) from exc
    return hftbacktest
