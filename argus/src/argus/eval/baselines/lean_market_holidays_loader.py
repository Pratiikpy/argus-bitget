"""Load the real, vendored `lean_market_holidays_usa.json` — the real US equity market holiday
dates from QuantConnect/Lean's own real market-hours database.

Source:  https://github.com/QuantConnect/Lean
Path:    Data/market-hours/market-hours-database.json, the "holidays" array of the real
         "Equity-usa-[*]" entry (293 real dates, 1998-2028) — the same real, authoritative
         calendar the real `TradingCalendar.GetDaysByType(TradingDayType.PublicHoliday, ...)`
         the QuantConnect/Tutorials "Pre-Holiday Effect" strategy calls is itself built from.
Commit:  23b735d99a357807dc0df9f4c51d30f05fe0d277 (2026-09-04) — the same commit already cited
         for `lean_pairs_ranking.py` in this directory (same clone, same checkout).
Licence: Apache License 2.0 (Copyright 2014 QuantConnect Corporation) — the repository's top-
         level LICENSE covers `Data/`; there is no separate licence file for that directory.

This is real, verbatim DATA, not code — every date string here is copied unchanged from the
real file, re-serialised as a clean, chronologically-sorted JSON array (the original entry mixes
insertion order across markets/decades). `argus.research.gap_study.study()` had never been given
a real holiday calendar at all: it calls `truth.clocks.DualClock()` with no `holidays` argument,
so `DualClock`'s own real, deliberate default (`frozenset()` — "a missing holiday file degrades
to 'we do not know', never to 'assume open'", per that module's own docstring) meant the
`SessionPhase.HOLIDAY` branch its own `_closed_sessions()` already classifies for has never once
fired in a real run. `argus.eval.afterhours_comparison` wires this real calendar in and reports
what changes.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

_BASELINES_DIR = Path(__file__).resolve().parent


class LeanMarketHolidaysLoadError(RuntimeError):
    """The vendored real holiday data could not be loaded."""


def load_usa_equity_holidays() -> frozenset[date]:
    """The real 293 US equity market holiday dates (1998-2028), as a `frozenset[date]` ready to
    pass straight to `truth.clocks.DualClock(holidays=...)`."""
    path = _BASELINES_DIR / "lean_market_holidays_usa.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeanMarketHolidaysLoadError(f"could not read {path}: {exc}") from exc
    return frozenset(datetime.strptime(s, "%m/%d/%Y").date() for s in raw)


__all__ = ["LeanMarketHolidaysLoadError", "load_usa_equity_holidays"]
