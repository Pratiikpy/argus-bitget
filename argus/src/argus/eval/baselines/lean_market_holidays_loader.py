"""Load the vendored US equity market holiday calendar (`truth/data/us_equity_holidays.json`,
until 2026-09-26 `eval/baselines/lean_market_holidays_usa.json`) — the holiday dates from
QuantConnect/Lean's own market-hours database.

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
insertion order across markets/decades). When this loader was written, `argus.research.gap_study
.study()` had never been given a real holiday calendar: it called `truth.clocks.DualClock()` with
no `holidays` argument, and `DualClock`'s default was then `frozenset()`, so the
`SessionPhase.HOLIDAY` branch its own `_closed_sessions()` classifies for had never fired in a
real run. `argus.eval.afterhours_comparison` wired this calendar in and reported what changed.
Since 2026-09-26 the same dates, byte for byte, live at `truth/data/us_equity_holidays.json` and
are every `DualClock`'s default; this loader reads them there.
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
    # The calendar now lives with the clock that uses it (truth/data), so truth does not
    # import from eval; this loader reads the same file.
    path = _BASELINES_DIR.parents[1] / "truth" / "data" / "us_equity_holidays.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeanMarketHolidaysLoadError(f"could not read {path}: {exc}") from exc
    return frozenset(datetime.strptime(s, "%m/%d/%Y").date() for s in raw)


__all__ = ["LeanMarketHolidaysLoadError", "load_usa_equity_holidays"]
