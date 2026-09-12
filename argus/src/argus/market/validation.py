"""Is this symbol actually a tokenized equity? Decided by behaviour, not by its ticker.

**This module exists because a hardcoded list was wrong and cost us a false result.** ``SPXUSDT``
was included in the rToken universe on the assumption that it tracked the S&P 500. It trades at
**$0.49**. It is SPX6900, a memecoin that happens to own the ticker, and a backtest variant scored
a 179% return and Sharpe 4.32 on it — pure memecoin drift presented as session alpha.

Bitget's own instrument metadata does not help: ``/api/v2/mix/market/contracts`` reports
``symbolType: perpetual`` for every one of them, tokenized equity and memecoin alike.

So the test is behavioural, and it falls straight out of our own measurement. A tokenized equity's
**index** series tracks an underlying that closes; a memecoin has no underlying and never closes.
Measure median absolute hourly index movement during RTH and during weekends, and take the ratio:

=============  ==========  ===========  =============
Symbol         RTH bps/h   wknd bps/h   attenuation
=============  ==========  ===========  =============
SQQQUSDT            46.59         5.31         8.78
TQQQUSDT            48.68         6.83         7.13
NVDAUSDT            31.71         4.52         7.01
METAUSDT            34.42         5.64         6.10
AMZNUSDT            24.01         3.97         6.05
TSLAUSDT            36.01         6.40         5.62
AAPLUSDT            24.92         4.52         5.51
MSFTUSDT            25.60         4.92         5.20
COINUSDT            63.83        12.76         5.00
GOOGLUSDT           24.67         5.42         4.56
QQQUSDT             15.65         3.58         4.37
MSTRUSDT            67.85        18.84         3.60
**SPXUSDT**         67.09        47.00     **1.43**   <- no anchor
=============  ==========  ===========  =============

Twelve genuine instruments cluster between 3.6x and 8.8x. The impostor sits at 1.43x. The gap is
wide enough that the threshold does not need tuning, which is the mark of a real signal rather than
a fitted one.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from argus.market.history import CandleType, fetch_range
from argus.truth.clocks import DualClock, SessionPhase

# Chosen from the observed gap: genuine instruments >= 3.60, impostor 1.43. Any threshold in
# (1.5, 3.5) separates them identically, so the exact value carries no fitted information.
ATTENUATION_THRESHOLD = 2.5

MIN_OBSERVATIONS = 200


@dataclass(frozen=True, slots=True)
class AnchorEvidence:
    """Whether a symbol behaves like something with a closing underlying."""

    symbol: str
    last_index: float
    rth_bps_per_hour: float
    weekend_bps_per_hour: float
    observations: int

    @property
    def attenuation(self) -> float:
        """How much quieter the index goes when the anchor market shuts.

        A genuine tokenized equity cannot have a lively index at 3am on a Sunday: there is no
        underlying trading to move it.
        """
        if self.weekend_bps_per_hour <= 0:
            return float("inf")
        return self.rth_bps_per_hour / self.weekend_bps_per_hour

    @property
    def has_anchor(self) -> bool:
        return self.attenuation >= ATTENUATION_THRESHOLD

    def as_dict(self) -> dict[str, float | str | bool | int]:
        return {
            "symbol": self.symbol,
            "last_index": round(self.last_index, 4),
            "rth_bps_per_hour": round(self.rth_bps_per_hour, 2),
            "weekend_bps_per_hour": round(self.weekend_bps_per_hour, 2),
            "attenuation": round(self.attenuation, 2),
            "has_anchor": self.has_anchor,
            "observations": self.observations,
        }


class NoAnchorError(RuntimeError):
    """A symbol was used as a tokenized equity but does not behave like one.

    Raised rather than warned. A study that silently skips an impostor and one that never noticed
    it produce the same output, and only one of them is honest.
    """


def measure_anchor(symbol: str, *, days: int = 90) -> AnchorEvidence:
    """Measure a symbol's session attenuation from its published index series."""
    clock = DualClock()
    index = fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.INDEX)
    if len(index) < MIN_OBSERVATIONS:
        raise NoAnchorError(
            f"{symbol}: only {len(index)} index observations, need {MIN_OBSERVATIONS} "
            f"to judge attenuation"
        )

    rth: list[float] = []
    weekend: list[float] = []
    for i in range(1, len(index)):
        prev, cur = float(index[i - 1].close), float(index[i].close)
        if prev <= 0:
            continue
        move = abs(cur - prev) / prev * 10_000
        phase = clock.phase(index[i].ts)
        if phase.has_price_discovery:
            rth.append(move)
        elif phase in (SessionPhase.WEEKEND, SessionPhase.HOLIDAY):
            weekend.append(move)

    if not rth or not weekend:
        raise NoAnchorError(f"{symbol}: sample lacks both RTH and weekend observations")

    return AnchorEvidence(
        symbol=symbol,
        last_index=float(index[-1].close),
        rth_bps_per_hour=median(rth),
        weekend_bps_per_hour=median(weekend),
        observations=len(index),
    )


def verify_universe(
    symbols: tuple[str, ...], *, days: int = 90
) -> tuple[tuple[str, ...], dict[str, dict[str, float | str | bool | int]]]:
    """Split a candidate universe into instruments with an anchor and those without.

    Returns the verified symbols and the full evidence for every candidate — including the
    rejections, because "which ones we excluded and why" is part of the result.
    """
    verified: list[str] = []
    evidence: dict[str, dict[str, float | str | bool | int]] = {}

    for symbol in symbols:
        try:
            got = measure_anchor(symbol, days=days)
        except Exception as exc:
            evidence[symbol] = {"symbol": symbol, "error": str(exc)[:120], "has_anchor": False}
            continue
        evidence[symbol] = got.as_dict()
        if got.has_anchor:
            verified.append(symbol)

    return tuple(verified), evidence
