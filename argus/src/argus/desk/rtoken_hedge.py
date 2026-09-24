"""The overnight hedge for a spot rToken holder: short the same company's stock perpetual.

A holder of ``RTSLAUSDT`` carries the US market's closed hours — every night and, above all, the
weekend — in a token that keeps trading while its anchor does not. The instrument that tracks it
through those hours is not an index: it is ``TSLAUSDT``, the same company's perpetual, and on
Ballast's held-out nights a hedge sized by ordinary least squares removed 99.7% of the overnight
variance where ARGUS's index leg removed 10.1% (`eval/copilot_hedge.py`, the loss that produced this
module).

**What is computed, and why each piece is there.**

* The ratio is ARGUS's own :func:`~argus.desk.diversification.minimum_variance_hedge` on the
  nightly log returns (close to next open, `market.rtoken_spot.overnight_returns`), fitted on
  every night available — that is the size a holder would carry tonight.
* The evidence is fitted separately, on the first 70% of nights, and scored unrefitted on the last
  30% — Ballast's own protocol (``research/oos.py:63-87``). The number quoted to a holder is the
  held-out one, never the in-sample fit.
* Weekend nights (Friday close to Monday open) are reported on their own, because they are the
  longest window and the one an rToken holder most often asks about.
* The cost is Bitget's perpetual taker fee both ways, 0.06% a side (``ballast/costs.py``, and the
  rate ARGUS charges everywhere, `cost/model.py:188-197`), on the hedge's size: a night's
  insurance costs about 12 bp of the position before funding.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from argus.desk.diversification import DiversificationError, minimum_variance_hedge

FIT_SHARE = 0.7
PERP_TAKER_PER_SIDE_BP = 6.0
MIN_NIGHTS = 40


class HedgeUnavailable(ValueError):
    """Too few shared nights to size a hedge honestly."""


def _p95_abs(xs: Sequence[float]) -> float:
    ordered = sorted(abs(x) for x in xs)
    rank = 0.95 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


@dataclass(frozen=True, slots=True)
class OvernightHedge:
    spot: str
    perp: str
    ratio: float
    """Units of the perpetual per unit of the rToken; negative means short."""
    nights: int
    first: date
    last: date
    held_out_nights: int
    held_out_variance_removed: float
    held_out_tail_cut: float
    weekend_nights_held_out: int
    weekend_variance_removed: float | None
    typical_night_bp: float
    """The median absolute unhedged overnight move, in basis points."""
    typical_hedged_bp: float
    cost_bp_per_night: float

    def as_dict(self) -> dict[str, Any]:
        return {k: (v.isoformat() if isinstance(v, date) else
                    round(v, 5) if isinstance(v, float) else v)
                for k, v in ((f, getattr(self, f)) for f in self.__slots__)}


def overnight_hedge(spot: str, perp: str, spot_nights: Mapping[date, float],
                    perp_nights: Mapping[date, float]) -> OvernightHedge:
    days = sorted(set(spot_nights) & set(perp_nights))
    if len(days) < MIN_NIGHTS:
        raise HedgeUnavailable(f"{len(days)} shared nights; at least {MIN_NIGHTS} are needed")
    s = [spot_nights[d] for d in days]
    p = [perp_nights[d] for d in days]
    try:
        live = minimum_variance_hedge(s, p, instrument=perp)
        cut = int(len(days) * FIT_SHARE)
        fit = minimum_variance_hedge(s[:cut], p[:cut], instrument=perp)
    except DiversificationError as exc:
        raise HedgeUnavailable(str(exc)) from exc
    held_days, held_s, held_p = days[cut:], s[cut:], p[cut:]
    resid = [a + fit.ratio * b for a, b in zip(held_s, held_p, strict=True)]
    weekend = [i for i, d in enumerate(held_days) if d.weekday() == 4]
    weekend_removed = None
    if len(weekend) >= 5:
        ws = [held_s[i] for i in weekend]
        wr = [resid[i] for i in weekend]
        weekend_removed = 1 - statistics.variance(wr) / statistics.variance(ws)
    return OvernightHedge(
        spot=spot, perp=perp, ratio=live.ratio, nights=len(days), first=days[0], last=days[-1],
        held_out_nights=len(held_s),
        held_out_variance_removed=1 - statistics.variance(resid) / statistics.variance(held_s),
        held_out_tail_cut=1 - _p95_abs(resid) / _p95_abs(held_s),
        weekend_nights_held_out=len(weekend), weekend_variance_removed=weekend_removed,
        typical_night_bp=statistics.median(abs(x) for x in held_s) * 1e4,
        typical_hedged_bp=statistics.median(abs(x) for x in resid) * 1e4,
        cost_bp_per_night=abs(live.ratio) * 2 * PERP_TAKER_PER_SIDE_BP,
    )


def render(h: OvernightHedge, ticker: str) -> list[str]:
    side = "Short" if h.ratio < 0 else "Buy"
    lines = [
        f"Actionable: to carry {h.spot} through the hours the US market is shut, "
        f"{side.lower()} {abs(h.ratio):.2f} {h.perp} (Bitget's {ticker} stock perpetual) for "
        f"every 1 of {h.spot} you hold — the same company, so it moves with your token when "
        f"nothing else does.",
        f"Tested on nights it had not seen: over the last {h.held_out_nights} of {h.nights} "
        f"nights ({h.first:%d %b %Y} to {h.last:%d %b %Y}), that hedge removed "
        f"{h.held_out_variance_removed:.1%} of the overnight swing and cut the worst 1-in-20 "
        f"night by {h.held_out_tail_cut:.0%}. A typical night moved {h.typical_night_bp:.0f} bp "
        f"unhedged and {h.typical_hedged_bp:.0f} bp hedged.",
    ]
    if h.weekend_variance_removed is not None:
        lines.append(
            f"Weekends (Friday close to Monday open), the longest window: "
            f"{h.weekend_variance_removed:.1%} of the swing removed on "
            f"{h.weekend_nights_held_out} held-out weekends.")
    lines.append(
        f"Cost: about {h.cost_bp_per_night:.0f} bp of the position each night the hedge is put "
        f"on and taken off (Bitget's 0.06% perpetual taker fee each way), before funding. Held "
        f"across several nights, you pay the fee once and the funding rate instead.")
    return lines


__all__ = ["FIT_SHARE", "HedgeUnavailable", "OvernightHedge", "overnight_hedge", "render"]
