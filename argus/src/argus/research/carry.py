"""Funding carry — the one subject where ARGUS is allowed to hold a position.

**Why this subject and not another.** Every directional study in this repository reached the same
answer. `desk/analogue.py`, `desk/shapematch.py` and `research/cointegration.py` each measured no
directional edge on the rTokens, the register committed both direction claims at exactly 0.50 to say
so in public, and `agents/desk.py:651` consequently returns on ``quantity <= 0`` — which is why
`eval/autopsy.py` reports five of the six constitution gates as UNREACHED. The desk has never
refused a trade; it has never been offered one.

Carry is not a direction. A funding payment is collected for *holding* a side of a perpetual, not
for being right about where it goes, so the 0.50 finding does not apply to it and a non-zero
quantity can be proposed without claiming a forecast we do not have. That is the whole reason this
module exists: it is the honest route to a decision that reaches gate two.

**What the venue's own record says** — 270 settlements per instrument over 90 days, pulled from
`/api/v3/market/history-fund-rate` (public; `agent-sdk/src/generated/catalog.ts:115` documents
``limit`` to a maximum of 200 and the live endpoint rejects anything above 100 with
``40020 Parameter limit error``, so the catalogue is wrong and the API is what this follows):

* **The median settlement pays nothing.** On all twelve instruments the median funding rate is
  exactly zero; between 67% and 95% of all settlements are zero.
* **Funding is one-sided.** Negative settlements are close to non-existent — on NVDAUSDT, 85.9% are
  zero and 13.7% positive, leaving 0.4% negative. Longs pay shorts, or nobody pays anybody.
* So the mean is a **tail statistic**, not a typical outcome, and quoting an annualised mean without
  an interval around it would be the single most misleading number available here. Every mean below
  carries a bootstrap interval, and the ones whose interval contains zero are reported as such.

**The structure that makes it harvestable.** Collecting funding means being short, and a naked short
perp is a directional position wearing a carry costume — over any horizon long enough for the carry
to matter, equity beta dwarfs it. What makes this tradeable at all is that three of the twelve
instruments are *mechanically* related: QQQ, TQQQ and SQQQ track the same index at nominal +1x, +3x
and -3x. A basket weighted by the measured relationship is delta-neutral by construction rather than
by a statistical coincidence that can decay.

**Measured, not assumed, and the difference is material.** The nominal multipliers are +/-3. Over
2,158 matched hourly bars the realised betas are **+2.859** and **-2.866**, with R2 of 0.941 and
0.917 —
so roughly 6% and 8% of the variance does not hedge. Sizing to ±3 would leave a systematic residual
delta, and treating the residual as zero would turn an unhedged directional bet into a reported
carry. The hedge ratio here is the minimum-variance ratio from `risk/effectiveness.py`, which is
Ederington (1979), and the leftover variance is charged against the carry rather than ignored.

**The cost that decides it.** Every leg is round-tripped at the taker fee — `desk/allocation.py:60`
puts taker at 6bps — so entry plus exit costs **12bps of gross notional**, which is 16bps to 24bps
of the first leg's notional depending on how large the hedge leg is. A carry of a few percent a year
is a fraction of a basis point per hour, so the break-even holding period runs to weeks, and that,
not the headline annualised number, is what decides whether the trade exists at all.

**And the headline number is not what decides it either.** Funding is only one leg of the P&L. The
basket has to be *held*, and holding it means wearing the price path of both instruments for the
whole period — which for a leveraged token is not the index's path scaled by three. So every basket
here is replayed through the real series over every window in the record, and the replay outranks
the funding table wherever they disagree. On the first run they disagreed: the basket paying the
second-highest funding lost money in three windows out of four, and the basket that paid best was
earning four fifths of its return from volatility decay rather than from funding at all. Neither
fact is visible in a funding rate.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "carry_study.json"

ENDPOINT = "https://api.bitget.com/api/v3/market/history-fund-rate"
MAX_LIMIT = 100
"""The live maximum. The generated catalogue says 200 and the venue answers ``40020`` to it."""

SETTLEMENTS_PER_DAY = 3
"""An eight-hour funding interval. `cost/model.py:60` reads ``fundingRateInterval: "8"`` from the
venue for every rToken rather than assuming the common default."""

MIN_SETTLEMENTS = 90
"""Thirty days. Below this the zero-inflated mean is not worth an interval."""

BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260914

STRUCTURAL_BASKETS: tuple[tuple[str, str, str], ...] = (
    ("QQQUSDT", "TQQQUSDT", "same index at nominal 1x and 3x"),
    ("QQQUSDT", "SQQQUSDT", "same index at nominal 1x and -3x"),
    ("TQQQUSDT", "SQQQUSDT", "same index at nominal 3x and -3x"),
)
"""Pairs whose relationship is mechanical rather than statistical.

Every other pair in the universe is two different companies, and a hedge between them rests on a
correlation that can and does break. These three track one index by construction, which is the only
reason a delta-neutral weight is defensible over a holding period measured in weeks.
"""


class CarryUnavailable(RuntimeError):
    """The venue did not answer with funding history. Absence, never a substituted number."""


@dataclass(frozen=True, slots=True)
class Settlement:
    at: datetime
    rate_bps: float


@dataclass(frozen=True, slots=True)
class FundingProfile:
    """What one instrument's funding has actually done, with the shape of it stated."""

    symbol: str
    settlements: int
    span_days: float
    zero_share: float
    positive_share: float
    negative_share: float
    mean_bps: float
    median_bps: float
    mean_low_bps: float
    mean_high_bps: float

    @property
    def annualised_pct(self) -> float:
        return self.mean_bps * SETTLEMENTS_PER_DAY * 365 / 100

    @property
    def interval_excludes_zero(self) -> bool:
        """Whether the mean is distinguishable from zero at all. With 67-95% of settlements exactly
        zero, this is the question, and an annualised figure without it is decoration."""
        return self.mean_low_bps > 0 or self.mean_high_bps < 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "settlements": self.settlements,
            "span_days": round(self.span_days, 1),
            "zero_share": round(self.zero_share, 4),
            "positive_share": round(self.positive_share, 4),
            "negative_share": round(self.negative_share, 4),
            "mean_bps": round(self.mean_bps, 4),
            "median_bps": round(self.median_bps, 4),
            "mean_ci95_bps": [round(self.mean_low_bps, 4), round(self.mean_high_bps, 4)],
            "annualised_pct": round(self.annualised_pct, 3),
            "interval_excludes_zero": self.interval_excludes_zero,
        }


def fetch_funding(symbol: str, *, pages: int = 3, timeout: float = 25.0) -> list[Settlement]:
    """Every settlement the venue will serve, oldest last. Raises rather than returning a guess."""
    out: list[Settlement] = []
    for cursor in range(1, pages + 1):
        url = (
            f"{ENDPOINT}?category=USDT-FUTURES&symbol={symbol}"
            f"&limit={MAX_LIMIT}&cursor={cursor}"
        )
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise CarryUnavailable(f"{symbol}: {exc}") from exc
        if payload.get("code") != "00000":
            raise CarryUnavailable(f"{symbol}: {payload.get('code')} {payload.get('msg')}")
        rows = (payload.get("data") or {}).get("resultList") or []
        if not rows:
            break
        out.extend(
            Settlement(
                at=datetime.fromtimestamp(int(row["fundingRateTimestamp"]) / 1000, tz=UTC),
                rate_bps=float(row["fundingRate"]) * 10_000,
            )
            for row in rows
        )
        if len(rows) < MAX_LIMIT:
            break
    if not out:
        raise CarryUnavailable(f"{symbol}: the venue returned no funding history")
    return sorted(out, key=lambda s: s.at)


def _bootstrap_mean(values: list[float], *, seed: int) -> tuple[float, float]:
    """A percentile bootstrap interval on the mean.

    A t-interval assumes something about the shape, and this distribution is 85% a point mass at
    zero with a thin positive tail — the assumption is exactly wrong here. Resampling makes no
    shape assumption, which is the only reason the interval means anything on this data.
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def profile(symbol: str, settlements: list[Settlement] | None = None) -> FundingProfile:
    """One instrument's funding profile, interval included."""
    rows = settlements if settlements is not None else fetch_funding(symbol)
    rates = [s.rate_bps for s in rows]
    n = len(rates)
    span = (rows[-1].at - rows[0].at).total_seconds() / 86_400 if n > 1 else 0.0
    low, high = _bootstrap_mean(rates, seed=BOOTSTRAP_SEED)
    return FundingProfile(
        symbol=symbol, settlements=n, span_days=span,
        zero_share=sum(1 for r in rates if r == 0) / n,
        positive_share=sum(1 for r in rates if r > 0) / n,
        negative_share=sum(1 for r in rates if r < 0) / n,
        mean_bps=sum(rates) / n, median_bps=median(rates),
        mean_low_bps=low, mean_high_bps=high,
    )


# --- the delta-neutral basket ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BasketReplay:
    """What holding the basket actually returned, over every window in the record.

    **This exists because the funding number on its own is a pitch, not a result.** Two of the three
    baskets here are same-side — short the index and short the inverse leveraged token — and a
    position like that carries the leveraged token's volatility decay alongside its funding. Decay
    does not appear anywhere in a funding rate, and over a holding period long enough for a few
    basis points a settlement to matter it can be larger than the entire carry. So the basket is
    held through the real price series and the total return is reported next to the funding, and
    where the two disagree the replay is the one that counts. Our own standing rules were written
    after a local sim showed +90% and the replay showed -0.45%.
    """

    holding_days: int
    windows: int
    mean_total_pct: float
    median_total_pct: float
    mean_funding_pct: float
    mean_price_pct: float
    share_positive: float
    worst_pct: float
    best_pct: float

    @property
    def funding_survives_the_price(self) -> bool:
        """Whether holding the basket actually paid, not merely whether the funding did."""
        return self.mean_total_pct > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "holding_days": self.holding_days, "windows": self.windows,
            "mean_total_pct": round(self.mean_total_pct, 4),
            "median_total_pct": round(self.median_total_pct, 4),
            "mean_funding_pct": round(self.mean_funding_pct, 4),
            "mean_price_pct": round(self.mean_price_pct, 4),
            "share_positive": round(self.share_positive, 4),
            "worst_pct": round(self.worst_pct, 4),
            "best_pct": round(self.best_pct, 4),
            "funding_survives_the_price": self.funding_survives_the_price,
        }


@dataclass(frozen=True, slots=True)
class CarryPair:
    """A structurally hedged pair, its measured carry, and what it costs to hold.

    ``legs`` carries **signed notional weights** — positive is long, negative is short — rather than
    a `long_leg`/`short_leg` pair. The first version used the latter and emitted an instruction
    nobody could execute: when the minimum-variance ratio is negative the two legs sit on the *same*
    side, and labelling one of them "long" inverted a leg of the basket while the arithmetic behind
    it was correct. A shape that cannot express the position is a shape that will misreport it.
    """

    legs: tuple[tuple[str, float], ...]
    structure: str
    hedge_ratio: float
    r_squared: float
    correlation_low: float
    observations: int
    gross_notional: float
    net_bps_per_settlement: float
    net_low_bps: float
    net_high_bps: float
    round_trip_bps: float
    replay: BasketReplay | None = None

    @property
    def description(self) -> str:
        """The position as an instruction, with no side left to infer."""
        return " + ".join(
            f"{'long' if weight > 0 else 'short'} {abs(weight):.3f} {symbol}"
            for symbol, weight in self.legs
        )

    @property
    def annualised_pct(self) -> float:
        return self.net_bps_per_settlement * SETTLEMENTS_PER_DAY * 365 / (self.gross_notional * 100)

    @property
    def breakeven_days(self) -> float:
        """Days of carry needed to repay entry and exit. ``inf`` when the carry is negative."""
        per_day = self.net_bps_per_settlement * SETTLEMENTS_PER_DAY
        return float("inf") if per_day <= 0 else self.round_trip_bps / per_day

    @property
    def unhedged_variance(self) -> float:
        """What the hedge does not remove — charged against the carry rather than ignored."""
        return 1.0 - self.r_squared

    @property
    def carry_survives_its_interval(self) -> bool:
        return self.net_low_bps > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.description,
            "legs": [{"symbol": s, "weight": round(w, 4)} for s, w in self.legs],
            "structure": self.structure,
            "replay": self.replay.as_dict() if self.replay else None,
            "hedge_ratio": round(self.hedge_ratio, 4),
            "r_squared": round(self.r_squared, 4),
            "correlation_low": round(self.correlation_low, 4),
            "observations": self.observations,
            "gross_notional": round(self.gross_notional, 4),
            "net_bps_per_settlement": round(self.net_bps_per_settlement, 4),
            "net_ci95_bps": [round(self.net_low_bps, 4), round(self.net_high_bps, 4)],
            "annualised_pct": round(self.annualised_pct, 3),
            "round_trip_bps": round(self.round_trip_bps, 2),
            "breakeven_days": round(self.breakeven_days, 1)
            if self.breakeven_days != float("inf") else None,
            "unhedged_variance": round(self.unhedged_variance, 4),
            "carry_survives_its_interval": self.carry_survives_its_interval,
        }


_SERIES_CACHE: dict[tuple[str, int], list[tuple[datetime, float]]] = {}


def _series(symbol: str, *, days: int) -> list[tuple[datetime, float]]:
    """One instrument's hourly closes, fetched once per process.

    The study reads each of three instruments twice per basket across three baskets; without this
    the same ninety days of bars are pulled twelve times and the run is bounded by the venue rather
    than by the work. Cached per (symbol, days) because a different window is a different series,
    not a slice of this one — `market/history.py` decides the range and it is not ours to truncate.
    """
    key = (symbol, days)
    if key not in _SERIES_CACHE:
        from argus.market.history import CandleType, fetch_range

        bars = fetch_range(symbol, days=days, interval="1H", candle_type=CandleType.MARKET)
        _SERIES_CACHE[key] = [(bar.ts, float(bar.close)) for bar in bars]
    return _SERIES_CACHE[key]


def _aligned_closes(a: str, b: str, *, days: int) -> tuple[list[float], list[float]]:
    """Two price series on matched timestamps. An inner join, never a forward fill."""
    left = {ts: px for ts, px in _series(a, days=days)}
    right = {ts: px for ts, px in _series(b, days=days)}
    keys = sorted(set(left) & set(right))
    return [left[k] for k in keys], [right[k] for k in keys]


def replay_basket(
    legs: tuple[tuple[str, float], ...], settlements: dict[str, list[Settlement]],
    *, days: int = 90, holding_days: int = 14, round_trip_bps: float = 0.0,
) -> BasketReplay | None:
    """Hold the basket through every window in the real series and report what it returned.

    Overlapping windows, deliberately. Independent windows would leave four or five observations in
    ninety days, which supports nothing; overlapping ones are correlated and that is stated rather
    than hidden, because the purpose here is the *shape* of the outcome — how often it pays, how bad
    the worst one is — and not a significance test the sample cannot carry.

    Funding accrues to a signed notional as ``-w x rate``: a long pays what a short collects, which
    falls out of the sign rather than needing a special case for each side.
    """
    series = {symbol: _series(symbol, days=days) for symbol, _weight in legs}
    stamps = sorted(
        set.intersection(*({t for t, _ in rows} for rows in series.values()))
    )
    prices = {sym: {t: p for t, p in rows} for sym, rows in series.items()}
    horizon = holding_days * 24
    if len(stamps) <= horizon + 1:
        return None

    gross = sum(abs(w) for _s, w in legs)
    totals: list[float] = []
    fundings: list[float] = []
    price_moves: list[float] = []
    for start in range(0, len(stamps) - horizon):
        opened, closed = stamps[start], stamps[start + horizon]
        price_pnl = sum(
            weight * (prices[symbol][closed] / prices[symbol][opened] - 1.0)
            for symbol, weight in legs
        )
        funding_pnl = sum(
            -weight * sum(
                s.rate_bps / 10_000 for s in settlements[symbol] if opened < s.at <= closed
            )
            for symbol, weight in legs
        )
        totals.append((price_pnl + funding_pnl - round_trip_bps / 10_000) / gross * 100)
        fundings.append(funding_pnl / gross * 100)
        price_moves.append(price_pnl / gross * 100)

    ordered = sorted(totals)
    return BasketReplay(
        holding_days=holding_days, windows=len(totals),
        mean_total_pct=sum(totals) / len(totals),
        median_total_pct=ordered[len(ordered) // 2],
        mean_funding_pct=sum(fundings) / len(fundings),
        mean_price_pct=sum(price_moves) / len(price_moves),
        share_positive=sum(1 for t in totals if t > 0) / len(totals),
        worst_pct=ordered[0], best_pct=ordered[-1],
    )


def build_pair(
    first: str, second: str, structure: str, *, profiles: dict[str, FundingProfile],
    settlements: dict[str, list[Settlement]], days: int = 90, holding_days: int = 14,
    taker_bps: float | None = None,
) -> CarryPair | None:
    """One structural pair, sized to the minimum-variance ratio, costed, and then actually held.

    Direction is **derived, not chosen**: funding is collected by the short and paid by the long, so
    whichever configuration nets positive is the one reported. Picking the side first and then
    finding a number to support it is how a carry study becomes a carry pitch.
    """
    from argus.desk.allocation import TAKER_BPS
    from argus.risk.effectiveness import EffectivenessError, measure

    taker = TAKER_BPS if taker_bps is None else taker_bps
    left, right = _aligned_closes(first, second, days=days)
    try:
        hedge = measure(left, right, spot=first, hedge=second, window_days=days)
    except EffectivenessError:
        return None

    ratio = hedge.hedge_ratio
    if ratio == 0:
        return None

    # The ratio is measured on log changes, so it is already a *notional* weight rather than a unit
    # count — which matters because funding is charged on notional and these three instruments quote
    # at 705, 68 and 41. A negative ratio means the two move oppositely, so a delta-flat basket puts
    # both legs on the same side; the sign carries that, and nothing has to assume they are opposed.
    weight_second = abs(ratio)
    gross = 1.0 + weight_second
    same_side = ratio < 0

    rng = random.Random(BOOTSTRAP_SEED)
    first_rates = [s.rate_bps for s in settlements[first]]
    second_rates = [s.rate_bps for s in settlements[second]]

    best: tuple[float, tuple[tuple[str, float], ...], float, float] | None = None
    for first_short in (True, False):
        first_weight = -1.0 if first_short else 1.0
        second_weight = (first_weight if same_side else -first_weight) * weight_second
        # Funding accrues as -w x rate: a short (w < 0) collects a positive rate.
        net = -(first_weight * profiles[first].mean_bps
                + second_weight * profiles[second].mean_bps)
        if best is None or net > best[0]:
            best = (net, ((first, first_weight), (second, second_weight)),
                    first_weight, second_weight)
    assert best is not None
    net, legs, first_weight, second_weight = best

    draws = sorted(
        -(first_weight * (sum(rng.choices(first_rates, k=len(first_rates))) / len(first_rates))
          + second_weight
          * (sum(rng.choices(second_rates, k=len(second_rates))) / len(second_rates)))
        for _ in range(BOOTSTRAP)
    )
    round_trip = 2 * taker * gross

    return CarryPair(
        legs=legs, structure=structure,
        hedge_ratio=ratio, r_squared=hedge.r_squared, correlation_low=hedge.correlation_low,
        observations=hedge.observations, gross_notional=gross,
        net_bps_per_settlement=net,
        net_low_bps=draws[int(0.025 * BOOTSTRAP)],
        net_high_bps=draws[int(0.975 * BOOTSTRAP) - 1],
        round_trip_bps=round_trip,
        replay=replay_basket(
            legs, settlements, days=days, holding_days=holding_days, round_trip_bps=round_trip,
        ),
    )


def run(*, days: int = 90, holding_days: int = 14) -> dict[str, Any]:
    """The whole study: every instrument's funding profile, then the structural baskets."""
    from argus.market.bitget import RTOKEN_SYMBOLS

    profiles: dict[str, FundingProfile] = {}
    settlements: dict[str, list[Settlement]] = {}
    unavailable: list[str] = []
    for symbol in RTOKEN_SYMBOLS:
        try:
            rows = fetch_funding(symbol)
        except CarryUnavailable as exc:
            unavailable.append(str(exc))
            continue
        settlements[symbol] = rows
        profiles[symbol] = profile(symbol, rows)

    pairs: list[CarryPair] = []
    for first, second, structure in STRUCTURAL_BASKETS:
        if first in profiles and second in profiles:
            built = build_pair(
                first, second, structure, profiles=profiles, settlements=settlements,
                days=days, holding_days=holding_days,
            )
            if built is not None:
                pairs.append(built)

    # Tradeable means the *held* basket paid, not merely that the funding line did. A basket whose
    # funding interval clears zero and whose replay does not is a basket that loses money.
    tradeable = [
        p for p in pairs
        if p.carry_survives_its_interval and p.replay and p.replay.funding_survives_the_price
    ]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": ENDPOINT,
        "settlements_per_day": SETTLEMENTS_PER_DAY,
        "holding_days": holding_days,
        "profiles": [profiles[s].as_dict() for s in sorted(profiles)],
        "unavailable": unavailable,
        "pairs": [p.as_dict() for p in pairs],
        "tradeable": [p.description for p in tradeable],
        "verdict": _verdict(list(profiles.values()), pairs),
    }


def effective_windows(windows: int, holding_days: int, span_days: float) -> float:
    """How many *independent* holds the overlapping windows actually represent.

    1,823 overlapping fourteen-day windows drawn from ninety days of history are not 1,823
    observations; they are roughly six, reused. A win rate quoted off the raw count reads as three
    hundred times more certain than the sample can support, which is exactly the mistake that makes
    an overlapping-window backtest look conclusive. So the number is computed and printed next to
    the win rate rather than left for a reader to work out.
    """
    if holding_days <= 0 or span_days <= 0:
        return 0.0
    return max(1.0, span_days / holding_days)


def _verdict(profiles: list[FundingProfile], pairs: list[CarryPair]) -> str:
    if not profiles:
        return "no funding history reachable — nothing is claimed"
    zero = sum(p.zero_share for p in profiles) / len(profiles)
    one_sided = sum(1 for p in profiles if p.negative_share < 0.05)
    head = (
        f"Across {len(profiles)} instruments the median funding settlement is exactly zero on "
        f"{sum(1 for p in profiles if p.median_bps == 0)} of them, {zero:.0%} of all settlements "
        f"pay nothing, and {one_sided} settle negative less than 5% of the time — funding here is "
        f"one-sided and sparse, so a mean is a tail statistic and carries a bootstrap interval "
        f"rather than being annualised on its own."
    )
    priced = [p for p in pairs if p.replay is not None]
    if not priced:
        return head + " No structural basket could be replayed, so no carry is claimed."

    held = [p for p in priced if p.replay and p.replay.funding_survives_the_price]
    # A basket the funding line recommends and the replay refuses. Selected on that contradiction
    # rather than on rank: the richest carry is not necessarily the one that fails, and looking only
    # at the top of the table would have missed this entirely on the first run.
    promised = [
        p for p in priced
        if p.carry_survives_its_interval and p.replay and not p.replay.funding_survives_the_price
    ]

    body = (
        f" Of {len(pairs)} delta-neutral baskets, "
        f"{sum(1 for p in pairs if p.carry_survives_its_interval)} have a funding rate whose 95% "
        f"interval excludes zero — and that is the number this study exists to distrust."
    )

    for lost in promised:
        assert lost.replay is not None
        body += (
            f" **{lost.description} is a carry that loses money.** It pays "
            f"{lost.annualised_pct:+.2f}% a year in funding with an interval clear of zero, and "
            f"held through the real series over {lost.replay.holding_days}-day windows it returns "
            f"{lost.replay.mean_total_pct:+.3f}%, positive in only "
            f"{lost.replay.share_positive:.0%} of them: the funding arrives "
            f"({lost.replay.mean_funding_pct:+.3f}%) and the price leg gives back more "
            f"({lost.replay.mean_price_pct:+.3f}%). A funding table alone would have recommended "
            f"precisely this trade, which is why there is a replay."
        )

    if not held:
        return head + body + (
            " No basket survives being held, so none is proposed. The funding is real and the "
            "trade is not, and that distinction is the entire finding."
        )

    best = max(held, key=lambda p: p.replay.mean_total_pct if p.replay else 0.0)
    assert best.replay is not None
    carry_share = (
        best.replay.mean_funding_pct / best.replay.mean_total_pct
        if best.replay.mean_total_pct else 0.0
    )
    independent = effective_windows(
        best.replay.windows, best.replay.holding_days,
        max((p.span_days for p in profiles), default=0.0),
    )
    body += (
        f" {len(held)} of {len(priced)} pay when actually held. The best is {best.description} at "
        f"{best.replay.mean_total_pct:+.3f}% per {best.replay.holding_days}-day hold net of its "
        f"{best.round_trip_bps:.0f}bps round trip, positive in {best.replay.share_positive:.0%} of "
        f"{best.replay.windows} overlapping windows — which is about {independent:.0f} independent "
        f"holds, not {best.replay.windows}, and the win rate should be read against that."
    )
    if carry_share < 0.5:
        body += (
            f" **It should not be called a carry.** Only {carry_share:.0%} of that return is "
            f"funding ({best.replay.mean_funding_pct:+.3f}%); the rest is the price leg "
            f"({best.replay.mean_price_pct:+.3f}%), which for a basket short both a leveraged "
            f"token and its inverse is volatility decay. That is a short-gamma position — a high "
            f"win "
            f"rates with a {best.replay.worst_pct:+.2f}% worst window are the shape of one — and "
            f"selling it as funding carry would misdescribe both the source of the return and the "
            f"risk being run."
        )
    body += (
        f" {best.unhedged_variance:.1%} of the basket's variance is unhedged, and that residual is "
        f"what the position is actually paid for bearing."
    )
    return head + body



def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = run()
    print("FUNDING CARRY — the venue's own settlement record\n")
    print(
        f"  {'symbol':11} {'n':>4} {'zero':>6} {'pos':>6} {'neg':>6} {'mean':>7} "
        f"{'95% interval':>18} {'ann%':>7}"
    )
    for row in report["profiles"]:
        lo, hi = row["mean_ci95_bps"]
        print(
            f"  {row['symbol']:11} {row['settlements']:4d} {row['zero_share']:6.1%} "
            f"{row['positive_share']:6.1%} {row['negative_share']:6.1%} {row['mean_bps']:7.3f} "
            f"[{lo:+7.3f},{hi:+7.3f}] {row['annualised_pct']:7.2f}"
        )

    print(
        f"\n  {'position':44} {'R2':>6} {'fund bps':>9} {'ann%':>7} {'b/e':>6}"
        f" | {'held':>7} {'of which fund':>14} {'px':>7} {'win%':>6} {'worst':>7}"
    )
    for row in report["pairs"]:
        be = row["breakeven_days"]
        rep = row["replay"]
        line = (
            f"  {row['position']:44} {row['r_squared']:6.3f} "
            f"{row['net_bps_per_settlement']:+9.4f} {row['annualised_pct']:+7.2f} "
            f"{(f'{be:.0f}d' if be else 'never'):>6}"
        )
        if rep:
            line += (
                f" | {rep['mean_total_pct']:+7.3f} {rep['mean_funding_pct']:+14.3f} "
                f"{rep['mean_price_pct']:+7.3f} {rep['share_positive']:6.0%} "
                f"{rep['worst_pct']:+7.2f}"
            )
        print(line)
    print(
        f"\n  held / fund / px / win% / worst are {report['holding_days']}-day holds over every"
        f" overlapping window, net of the round trip."
    )

    print(f"\n  {report['verdict']}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP",
    "ENDPOINT",
    "MAX_LIMIT",
    "MIN_SETTLEMENTS",
    "REPORT_PATH",
    "SETTLEMENTS_PER_DAY",
    "STRUCTURAL_BASKETS",
    "CarryPair",
    "CarryUnavailable",
    "FundingProfile",
    "Settlement",
    "build_pair",
    "fetch_funding",
    "profile",
    "run",
]
