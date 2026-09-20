"""Implied volatility — the regime these tokens actually trade in, from CBOE directly.

The data-source audit (`research/architecture/datasource-audit.md`) ranked a volatility gauge among
the highest-value additions available, and unlike the sentiment feeds it recommended, this one
answers: CBOE publishes both a delayed quote and the full daily history since 1990, keyless, and
both were probed live on 2026-09-13.

**Why this source is on-target and the crypto Fear & Greed index is not.** VIX is thirty-day
implied volatility on the S&P 500. The twelve instruments this desk trades are tokenised US
equities whose price is anchored to US equity prices, so S&P implied volatility is a statement
about *their* risk environment rather than about a related market. That is the difference between
evidence and borrowed relevance, and it is why this is carried at credibility 0.9 while
`argus.market.macro.fear_greed_evidence` is carried at 0.35.

**A level is not information; a percentile is.** "VIX is 15.8" means nothing without knowing that
15.8 sits in the calmest fifth of the last thirty-five years. So the module fetches the history
once, caches it, and reports where today sits in it. The regime bands are percentile-based rather
than absolute, because an absolute threshold written in 2026 would be wrong the moment the
volatility regime shifts — which is the thing it exists to detect.

**It is wired to a number the desk already uses.** :attr:`Reading.implied_daily_move_pct` converts
the index to the one-day move it implies, which is directly comparable with the desk's own
18.8bps weekend cost hurdle. A regime where the implied daily move is smaller than the round trip
is a regime in which almost nothing is worth trading, and the desk should be able to say so in
those terms rather than discovering it one abstention at a time.

Sources read for this: ``OpenBB-finance/OpenBB`` (``openbb_cboe`` provider) for the endpoint shapes,
and ``qlib``'s regime handling for the percentile-over-absolute choice.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from math import sqrt
from pathlib import Path
from typing import Any

from argus.truth.evidence import Evidence

QUOTE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX.json"
HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
USER_AGENT = "ARGUS research desk (contact@argus.invalid)"

CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "vix_history.csv"
CACHE_MAX_AGE_DAYS = 7
"""How stale the cached history may be before it is refetched.

A week, not a day. The history is 35 years of daily closes and one week's absence moves a
percentile by nothing measurable, while refetching a megabyte every cycle would be a self-inflicted
rate limit. The live level always comes from the quote endpoint, so freshness where it matters is
never cached.
"""

TRADING_DAYS = 252
"""Used only to convert an annualised index into a one-day implied move."""

MIN_HISTORY = 250
"""Fewest daily closes before a percentile is reported at all.

A percentile over a hundred points is a statement about a hundred points. Below this the module
reports the level and refuses the regime rather than producing a number that reads like context.
"""


class VolatilityError(RuntimeError):
    """The volatility surface could not be read. Never a silent neutral 20."""


class Regime(StrEnum):
    """Where today sits in its own history. Bands are percentiles, never absolute levels."""

    CALM = "calm"
    """Bottom quintile. Implied moves are small; the cost hurdle bites hardest here."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    STRESSED = "stressed"
    """Top 5%. Historically the regime in which correlations converge and hedges stop hedging."""

    UNKNOWN = "unknown"
    """Not enough history to place today. Reported as absence, not as NORMAL."""


def classify(percentile: float | None) -> Regime:
    """Percentile in [0, 1] to a regime. ``None`` means unknown, and stays unknown."""
    if percentile is None:
        return Regime.UNKNOWN
    if percentile >= 0.95:
        return Regime.STRESSED
    if percentile >= 0.80:
        return Regime.ELEVATED
    if percentile <= 0.20:
        return Regime.CALM
    return Regime.NORMAL


@dataclass(frozen=True, slots=True)
class Reading:
    """One observation of the volatility surface, placed in its own history."""

    as_of: datetime
    level: float
    change: float | None
    """Move since the prior close, or ``None`` when the quote did not carry one.

    **``None``, not ``0.0``.** ``price_change`` was defaulted to zero, and zero is the value that
    means *"the index closed unchanged"* — a real and reportable event. A feed that omitted the
    field and a genuinely flat day rendered identically as ``(+0.00 on the day)``.
    """

    day_high: float | None
    day_low: float | None
    """The session's extremes, or ``None`` when the quote did not carry them.

    **``None``, not the current level.** Both defaulted to ``level``, which does not read as
    missing data — it reads as *a session that has not moved at all*, a zero-width range, the
    rarest and most remarkable VIX day there is. The most extreme possible reading was what this
    module published whenever the feed was quiet.
    """
    percentile: float | None
    observations: int

    @property
    def regime(self) -> Regime:
        return classify(self.percentile)

    @property
    def implied_daily_move_pct(self) -> float:
        """The one-day move this index implies, in percent.

        VIX is quoted as an annualised percentage, so the daily figure is the level divided by the
        root of the trading-day count. This is the number that can be compared with a cost hurdle.
        """
        return self.level / sqrt(TRADING_DAYS)

    @property
    def implied_daily_move_bps(self) -> float:
        return self.implied_daily_move_pct * 100.0

    def clears_hurdle(self, hurdle_bps: float) -> bool:
        """Does a typical day even move far enough to pay a round trip?

        The question the desk's abstention record keeps answering the long way. A one-sigma day is
        not a guaranteed move, so this is a statement about the regime rather than about any trade
        — but a regime whose implied daily move is below the round-trip cost is one where a desk
        should expect to stand aside, and being able to say that in advance is different from
        discovering it afterwards.
        """
        return self.implied_daily_move_bps > hurdle_bps

    def render(self) -> str:
        where = (
            f"the {self.percentile:.0%} percentile of {self.observations:,} daily closes"
            if self.percentile is not None
            else f"an unplaced level ({self.observations:,} closes is too few to rank it)"
        )
        move = (
            "day change not reported" if self.change is None
            else f"{self.change:+.2f} on the day"
        )
        return (
            f"VIX {self.level:.2f} ({move}), {where} — "
            f"{self.regime.value} regime. Implies a one-day move of about "
            f"{self.implied_daily_move_bps:.0f}bps on US equities, which is what these tokens "
            f"track."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "level": round(self.level, 2),
            "change": None if self.change is None else round(self.change, 2),
            "day_high": None if self.day_high is None else round(self.day_high, 2),
            "day_low": None if self.day_low is None else round(self.day_low, 2),
            "percentile": None if self.percentile is None else round(self.percentile, 4),
            "observations": self.observations,
            "regime": self.regime.value,
            "implied_daily_move_bps": round(self.implied_daily_move_bps, 1),
        }


def _get(url: str, *, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise VolatilityError(f"could not read {url}: {exc}") from exc


def parse_history(text: str) -> list[tuple[date, float]]:
    """CBOE's daily CSV to dated closes. Rows that do not parse are skipped, not guessed.

    The file spans 1990 to today and its early rows carry the same value in every column, which is
    genuine — the index was reconstructed back to 1990 from a different methodology — so those rows
    are kept rather than filtered as suspicious.
    """
    out: list[tuple[date, float]] = []
    for row in csv.DictReader(io.StringIO(text)):
        raw_date = (row.get("DATE") or "").strip()
        raw_close = (row.get("CLOSE") or "").strip()
        if not raw_date or not raw_close:
            continue
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw_date, fmt).date()
                break
            except ValueError:
                continue
        else:
            continue
        try:
            out.append((parsed, float(raw_close)))
        except ValueError:
            continue
    out.sort()
    return out


def history(*, path: Path = CACHE_PATH, timeout: int = 40, refresh: bool = False) -> list[
    tuple[date, float]
]:
    """Daily VIX closes, cached on disk.

    A cache miss or a stale cache refetches; a fetch failure with a usable cache **uses the cache
    and does not raise**. That asymmetry is deliberate: the history is nearly static, so an old
    copy is almost as good as a new one, while an exception here would take down a decision cycle
    over a CDN hiccup. A fetch failure with no cache at all does raise, because then there is
    genuinely nothing to report.
    """
    fresh_enough = False
    if path.exists() and not refresh:
        age = datetime.now(UTC) - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        fresh_enough = age < timedelta(days=CACHE_MAX_AGE_DAYS)
    if fresh_enough:
        return parse_history(path.read_text(encoding="utf-8"))
    try:
        text = _get(HISTORY_URL, timeout=timeout).decode("utf-8", errors="replace")
    except VolatilityError:
        if path.exists():
            return parse_history(path.read_text(encoding="utf-8"))
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return parse_history(text)


def percentile_of(level: float, closes: list[float]) -> float | None:
    """Share of history at or below ``level``. ``None`` when there is not enough history."""
    if len(closes) < MIN_HISTORY:
        return None
    below = sum(1 for c in closes if c <= level)
    return below / len(closes)


def _optional_float(raw: object) -> float | None:
    """A float, or ``None`` when the field was absent or unparseable — never a stand-in number."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fetch(*, timeout: int = 25, path: Path = CACHE_PATH) -> Reading:
    """The live level, placed in its own history."""
    payload = json.loads(_get(QUOTE_URL, timeout=timeout).decode("utf-8", errors="replace"))
    data = payload.get("data") or {}
    try:
        level = float(data["current_price"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VolatilityError(f"the VIX quote had no usable level: {exc}") from exc
    stamp = str(payload.get("timestamp") or "").strip()
    try:
        as_of = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        # The quote is delayed but dated by CBOE; if the stamp is unreadable, saying "now" would
        # claim a freshness the payload did not provide, so the failure is named instead.
        raise VolatilityError(f"the VIX quote carried an unreadable timestamp: {stamp!r}") from None

    try:
        closes = [c for _, c in history(path=path, timeout=timeout)]
    except VolatilityError:
        closes = []
    return Reading(
        as_of=as_of,
        level=level,
        # Absent fields stay absent. A defaulted change reads as an unchanged index and a
        # defaulted high/low reads as a zero-width session; neither is what a quiet feed means.
        change=_optional_float(data.get("price_change")),
        day_high=_optional_float(data.get("high")),
        day_low=_optional_float(data.get("low")),
        percentile=percentile_of(level, closes),
        observations=len(closes),
    )


def evidence(reading: Reading, *, as_of: datetime) -> list[Evidence]:
    """One dated item for the panel, at the credibility an official index earns."""
    return [
        Evidence(
            id=f"vix-{reading.as_of.date().isoformat()}",
            claim=reading.render(),
            source="macro",
            available_at=reading.as_of,
            # 0.9, not 1.0: CBOE is authoritative about the index, and the index is a market's
            # opinion about the future rather than a fact about it.
            credibility=0.9,
            attributes={
                "regime": reading.regime.value,
                "implied_daily_move_bps": f"{reading.implied_daily_move_bps:.0f}",
            },
        )
    ]


def status(reading: Reading | None, error: str = "") -> str:
    """One line for the cycle's feed-health record."""
    if reading is None:
        return f"vix: unavailable ({error or 'not fetched'})"
    return (
        f"vix: {reading.level:.2f} {reading.regime.value} "
        f"({reading.observations:,} closes of history)"
    )


__all__ = [
    "CACHE_MAX_AGE_DAYS",
    "CACHE_PATH",
    "HISTORY_URL",
    "MIN_HISTORY",
    "QUOTE_URL",
    "TRADING_DAYS",
    "Reading",
    "Regime",
    "VolatilityError",
    "classify",
    "evidence",
    "fetch",
    "history",
    "parse_history",
    "percentile_of",
    "status",
]
