"""Portfolio risk for a book of tokenized equities — and why one beta is the wrong number.

Track 3's Open Theme names a portfolio-aware AI PM that evaluates how a proposed trade changes
"beta, sector and factor exposures, correlation, and concentration, with stress tests or hedge
suggestions". `argus.desk.workbench.assess_trade` covered two of those — notional concentration and
sector share — and its own docstring claimed beta without computing one. This module is the rest.

**The finding this module is built around, measured before it was written.** An rToken trades 7x24
while its anchor equity does not, so roughly 82% of hourly bars fall while the US market is shut.
Beta estimated over all bars is therefore an average dominated by the session with the least price
discovery. Measured on 30 days of hourly Bitget candles against QQQUSDT
(`data/session_beta.json`, reproducible):

===========  =========  ======  ======
symbol       blended    open    shut
===========  =========  ======  ======
AAPLUSDT        0.254   0.668   0.090
TSLAUSDT        1.069   1.594   0.861
METAUSDT        0.682   1.290   0.441
MSFTUSDT        0.340   0.577   0.245
NVDAUSDT        1.412   1.709   1.296
===========  =========  ======  ======

**Open-session beta exceeds shut-session beta on nine of eleven rTokens.** AAPL's blended 0.254
would tell a trader the name is nearly market-neutral; during the session that actually prices it,
the figure is 0.668. A copilot reporting one beta is reporting the wrong one.

The two exceptions are the check that the effect is real rather than an artefact: TQQQUSDT (2.905
open, 2.908 shut) and SQQQUSDT (-2.910, -2.885) are leveraged ETFs on the benchmark itself, so their
beta is mechanically fixed by construction and *should* not vary by session. Exactly the two
instruments that are structurally pinned show no session effect, and all nine single names do.

**That "leveraged ETF" claim used to rest on this docstring alone.** It is now a declared fact,
sourced and dated, in `market.instruments.REGISTRY` — checked against the issuer's own fund page,
not recalled from training data — and :func:`leverage_consistency` below checks the measured figure
against it on every call rather than asserting the two agree.

So every exposure here is reported **per session**, and the blended figure is carried alongside with
a warning rather than dropped — a reader who expects one number should see it, and see why it
misleads.

**Risk contribution, not notional share.** Concentration measured in notional says a position is 10%
of the book. What a trader needs is that it is 40% of the *risk*. The decomposition used is the
standard one, and it is checked rather than asserted: marginal contribution to risk
``MCR_i = (Sigma w)_i / sigma_p``, risk contribution ``RC_i = w_i * MCR_i``, and the contributions
sum exactly to ``sigma_p`` — :func:`decompose` verifies that identity on every call and raises if it
fails, because a decomposition that does not add up is arithmetic, not risk.

**Where this sits against the references, which were read first.** The estimators match
PyPortfolioOpt's beta (``pypfopt/expected_returns.py:304-305``, ``Cov(i,mkt)/Var(mkt)``) and
riskparity.py's Euler decomposition (``src/riskparityportfolio/rpp.py:96-98``,
``RC_i = w_i (Sigma w)_i``); the volatility-units form here sums to ``sigma_p`` rather than to one,
which is the same decomposition read in the unit a trader thinks in. Two guards are ours because
the references do not have them: ``cov_to_corr`` at ``risk_models.py:366-386`` divides by
``sqrt(diag)`` with no zero-variance check, so a constant price yields NaN there and ``None`` here;
and PyPortfolioOpt's beta has no minimum-observation guard at all. The effective number of bets
appears in none of the four repos read — it is standard textbook, built here on top of the
contributions. Neither does any of them implement scenario stress, which is noted rather than
claimed.

**No positive-semidefinite repair, deliberately.** PyPortfolioOpt eigen-clips
(``risk_models.py:57-113``) because it accepts ragged, pairwise-complete covariance, which can be
indefinite. :func:`align` intersects timestamps instead, so every covariance here comes from one
complete observation matrix with far more rows than columns — 719 aligned hourly bars against at
most twelve rTokens — and is positive semi-definite by construction. Adding a repair step would
hide a misalignment bug rather than fix one.

Pure Python throughout, no numpy: this package is stdlib-only and the matrices are at most twelve
by twelve.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from argus.market.instruments import InstrumentMasterError, LeverageCheck, leverage_check

MIN_OBSERVATIONS = 20
"""Fewest aligned returns before a beta or correlation is reported at all.

Below this the estimate is noise wearing a decimal point. Returning ``None`` and saying why is the
whole difference between a copilot and a number generator."""

IDENTITY_TOLERANCE = 1e-9
"""How far the sum of risk contributions may drift from portfolio volatility before it is an error.

Floating point only; anything larger is a bug in the decomposition, not rounding."""

SAME_TRADE_RHO = 0.70
"""|rho| at or above which a candidate is substantially the position already held.

Above this the second name adds notional without adding a bet, which is the case the diversification
warning exists for. These twelve instruments routinely correlate above 0.9, so the bar is set where
the relationship stops being incidental rather than at a textbook 0.5."""

DISTINCT_TRADE_RHO = 0.30
"""|rho| at or below which the candidate is a materially different bet.

Between the two constants the honest answer is "partly" — and saying so beats picking a side."""


class Session(StrEnum):
    """Which bars an estimate was computed over."""

    OPEN = "open"
    """The anchor equity market is open and genuinely pricing the underlying."""

    SHUT = "shut"
    """The token trades, the anchor does not. ~82% of hourly bars."""

    BLENDED = "blended"
    """All bars. Reported because readers expect it, and flagged because it misleads."""


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def variance(xs: Sequence[float]) -> float | None:
    """Sample variance, Bessel-corrected. ``None`` below :data:`MIN_OBSERVATIONS`."""
    if len(xs) < MIN_OBSERVATIONS:
        return None
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


def covariance(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Sample covariance over paired observations. ``None`` when too short or unpaired."""
    if len(xs) != len(ys) or len(xs) < MIN_OBSERVATIONS:
        return None
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (len(xs) - 1)


def beta(asset: Sequence[float], benchmark: Sequence[float]) -> float | None:
    """OLS slope of asset returns on benchmark returns.

    ``None`` when the benchmark has no variance: an unmoving benchmark cannot explain anything, and
    dividing by it would produce an enormous number that reads as an enormous exposure.
    """
    cov = covariance(benchmark, asset)
    var = variance(benchmark)
    if cov is None or var is None or var <= 0:
        return None
    return cov / var


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation. ``None`` when either series is flat."""
    cov = covariance(xs, ys)
    vx, vy = variance(xs), variance(ys)
    if cov is None or vx is None or vy is None or vx <= 0 or vy <= 0:
        return None
    return max(-1.0, min(1.0, cov / math.sqrt(vx * vy)))


def returns(prices: Sequence[tuple[datetime, float]]) -> dict[datetime, float]:
    """Simple returns keyed by the timestamp of the *later* bar.

    Keyed by the closing instant rather than the opening one so that a return is stamped with the
    moment it became knowable. Stamping it at the start of the bar would date a fact to before it
    existed, which is the look-ahead this project refuses everywhere else.
    """
    out: dict[datetime, float] = {}
    for i in range(1, len(prices)):
        (_, before), (stamp, after) = prices[i - 1], prices[i]
        if before > 0:
            out[stamp] = (after - before) / before
    return out


def align(
    series: Mapping[str, Mapping[datetime, float]],
) -> tuple[list[datetime], dict[str, list[float]]]:
    """Restrict every series to the timestamps they all share.

    Aligning on the intersection rather than forward-filling is deliberate. A filled bar is an
    invented observation, and inventing observations to make a covariance matrix bigger is how a
    correlation estimate becomes confident about a relationship nobody measured.
    """
    if not series:
        return [], {}
    common = set.intersection(*(set(v) for v in series.values()))
    stamps = sorted(common)
    return stamps, {name: [values[t] for t in stamps] for name, values in series.items()}


@dataclass(frozen=True, slots=True)
class Exposure:
    """One asset's exposure to the benchmark, per session."""

    symbol: str
    session: Session
    beta: float | None
    observations: int
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "session": str(self.session),
            "beta": None if self.beta is None else round(self.beta, 4),
            "observations": self.observations, "reason": self.reason,
        }


def session_betas(
    asset: Mapping[datetime, float],
    benchmark: Mapping[datetime, float],
    *,
    symbol: str,
    is_open: Any,
) -> dict[Session, Exposure]:
    """Beta in the open session, the shut session, and blended across both.

    ``is_open`` is a predicate on a timestamp — in production
    ``lambda t: DualClock().phase(t).has_price_discovery``. Injected rather than imported so the
    session definition stays the clock's business and this module stays arithmetic.
    """
    stamps = sorted(set(asset) & set(benchmark))
    buckets: dict[Session, list[tuple[float, float]]] = {
        Session.OPEN: [], Session.SHUT: [], Session.BLENDED: [],
    }
    for stamp in stamps:
        pair = (benchmark[stamp], asset[stamp])
        buckets[Session.BLENDED].append(pair)
        buckets[Session.OPEN if is_open(stamp) else Session.SHUT].append(pair)

    out: dict[Session, Exposure] = {}
    for session, pairs in buckets.items():
        value = beta([a for _, a in pairs], [b for b, _ in pairs]) if pairs else None
        reason = ""
        if value is None:
            reason = (
                f"{len(pairs)} paired observation(s); fewer than {MIN_OBSERVATIONS} is noise"
                if len(pairs) < MIN_OBSERVATIONS
                else "the benchmark did not move over these bars, so nothing can be attributed"
            )
        out[session] = Exposure(
            symbol=symbol, session=session, beta=value, observations=len(pairs), reason=reason,
        )
    return out


def leverage_consistency(exposures: Mapping[Session, Exposure]) -> LeverageCheck | None:
    """Check a leveraged instrument's measured beta against its declared target.

    **The check this module's own docstring used to only assert.** TQQQUSDT and SQQQUSDT were
    described in prose as "leveraged ETFs on the benchmark itself" with no declared source behind
    the number — the docstring and the measurement were the same claim, restated, not two
    independent facts agreeing. `market.instruments.leverage_check` is the second, independent
    fact: a target read from the issuer's own fund page.

    Uses the blended session, because a fund's daily-reset target is a whole-day fact and does not
    have a session-specific version to check against. Returns ``None`` for an instrument with no
    declared leverage, no blended reading, or **no entry in the Instrument Master at all** — this
    function is an optional enrichment on top of generic session-beta arithmetic, which
    legitimately runs on symbols the registry has never heard of (a test fixture, a future
    non-rToken instrument), and for those "nothing to check" is the honest answer, not an error.
    `market.instruments.identity_of` still raises for a caller that specifically needs an
    instrument's identity and would be wrong to proceed without one — this is not that caller.
    """
    blended = exposures.get(Session.BLENDED)
    if blended is None or blended.beta is None:
        return None
    try:
        return leverage_check(blended.symbol, blended.beta)
    except InstrumentMasterError:
        return None


@dataclass(frozen=True, slots=True)
class Contribution:
    """One position's share of the portfolio's risk, beside its share of the money."""

    symbol: str
    weight: float
    marginal: float
    """Marginal contribution to risk: how portfolio volatility moves per unit of this weight."""

    contribution: float
    """This position's share of portfolio volatility, in volatility units."""

    @property
    def share(self) -> float:
        """Fraction of total portfolio risk. The number notional concentration hides."""
        return self.contribution

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "weight": round(self.weight, 6),
            "marginal": round(self.marginal, 8), "contribution": round(self.contribution, 8),
        }


@dataclass(frozen=True, slots=True)
class RiskDecomposition:
    """Where the portfolio's risk actually sits."""

    volatility: float
    contributions: tuple[Contribution, ...]

    def share_of_risk(self, symbol: str) -> float | None:
        """Fraction of portfolio risk carried by one name, or ``None`` if it is not held."""
        if self.volatility <= 0:
            return None
        for item in self.contributions:
            if item.symbol == symbol:
                return item.contribution / self.volatility
        return None

    @property
    def effective_positions(self) -> float | None:
        """Inverse Herfindahl of risk shares: across how many positions the risk is spread.

        **This is not the "effective number of bets", and the distinction was found by testing.**
        Three perfectly correlated positions score 2.71 here, not 1, because the inverse Herfindahl
        measures how *evenly risk is spread across positions* and says nothing about whether those
        positions are the same trade wearing three names. An earlier docstring claimed "a book of
        ten names that all move together is one bet", which this arithmetic does not deliver.

        Correlation is reported separately, by :attr:`TradeImpact.max_correlation`. A genuine
        correlation-aware count — Meucci's effective number of bets — needs a principal-component
        decomposition of the covariance matrix, which is **NOT BUILT**; claiming it from this number
        would be the overclaim this project exists to avoid.
        """
        if self.volatility <= 0:
            return None
        shares = [c.contribution / self.volatility for c in self.contributions]
        total = sum(s * s for s in shares)
        return 1.0 / total if total > 0 else None

    @property
    def most_concentrated(self) -> Contribution | None:
        return max(self.contributions, key=lambda c: c.contribution, default=None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "volatility": round(self.volatility, 8),
            "effective_positions": (
                None if self.effective_positions is None
                else round(self.effective_positions, 3)
            ),
            "contributions": [c.as_dict() for c in self.contributions],
        }


class PortfolioError(RuntimeError):
    """The arithmetic did not hold. Never swallowed: a decomposition that does not add up is not
    a smaller risk number, it is a wrong one."""


def covariance_matrix(
    columns: Mapping[str, Sequence[float]],
) -> tuple[list[str], list[list[float]]] | None:
    """Sample covariance over aligned columns, or ``None`` if any pair is too short."""
    names = sorted(columns)
    if not names:
        return None
    size = len(columns[names[0]])
    if size < MIN_OBSERVATIONS:
        return None
    matrix: list[list[float]] = []
    for row in names:
        line: list[float] = []
        for col in names:
            value = covariance(columns[row], columns[col])
            if value is None:
                return None
            line.append(value)
        matrix.append(line)
    return names, matrix


def decompose(
    weights: Mapping[str, float], columns: Mapping[str, Sequence[float]]
) -> RiskDecomposition | None:
    """Split portfolio volatility into per-position contributions that sum back to it.

    ``MCR_i = (Sigma w)_i / sigma_p`` and ``RC_i = w_i * MCR_i``, so ``sum(RC) == sigma_p`` exactly.
    The identity is **verified on every call** rather than trusted: it is the one check that catches
    a transposed matrix, a misaligned weight vector or a dropped name, and all three produce
    plausible-looking numbers.
    """
    # Held names only. Including a zero-weight column put an unheld symbol into the result with a
    # contribution of -0.0, so "not in the book" and "contributes no risk" became the same answer.
    # They are different facts and a copilot must not conflate them.
    held = {n: c for n, c in columns.items() if float(weights.get(n, 0.0)) != 0.0}
    built = covariance_matrix(held)
    if built is None:
        return None
    names, sigma = built
    weight_vector = [float(weights[name]) for name in names]

    # Sigma @ w
    product = [
        sum(sigma[i][j] * weight_vector[j] for j in range(len(names)))
        for i in range(len(names))
    ]
    var = sum(weight_vector[i] * product[i] for i in range(len(names)))
    if var <= 0:
        return None
    vol = math.sqrt(var)

    contributions = tuple(
        Contribution(
            symbol=name,
            weight=weight_vector[i],
            marginal=product[i] / vol,
            contribution=weight_vector[i] * product[i] / vol,
        )
        for i, name in enumerate(names)
    )

    total = sum(c.contribution for c in contributions)
    if abs(total - vol) > max(IDENTITY_TOLERANCE, abs(vol) * 1e-9):
        raise PortfolioError(
            f"risk contributions sum to {total!r} but portfolio volatility is {vol!r}; "
            f"the decomposition does not add up and must not be reported"
        )
    return RiskDecomposition(volatility=vol, contributions=contributions)


@dataclass(frozen=True, slots=True)
class TradeImpact:
    """What a proposed trade does to the shape of the risk, not only the size of the book."""

    symbol: str
    beta_before: float | None
    beta_after: float | None
    risk_share_before: float | None
    risk_share_after: float | None
    effective_positions_before: float | None
    effective_positions_after: float | None
    session: Session
    max_correlation: tuple[str, float] | None
    """The existing holding this trade is most correlated with, and by how much."""

    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)
        return {
            "symbol": self.symbol, "session": str(self.session),
            "beta_before": r(self.beta_before), "beta_after": r(self.beta_after),
            "risk_share_before": r(self.risk_share_before),
            "risk_share_after": r(self.risk_share_after),
            "effective_positions_before": r(self.effective_positions_before),
            "effective_positions_after": r(self.effective_positions_after),
            "max_correlation": (
                None if self.max_correlation is None
                else {"symbol": self.max_correlation[0], "rho": round(self.max_correlation[1], 4)}
            ),
            "notes": list(self.notes),
        }

    def render(self) -> list[str]:
        lines: list[str] = []
        if self.beta_before is not None and self.beta_after is not None:
            lines.append(
                f"[portfolio] {self.session}-session beta {self.beta_before:.2f} -> "
                f"{self.beta_after:.2f}"
            )
        if self.risk_share_after is not None:
            share = f"{self.risk_share_after:.0%}"
            before = (
                "new position" if self.risk_share_before is None
                else f"{self.risk_share_before:.0%}"
            )
            lines.append(
                f"[portfolio] {self.symbol} would carry {share} of total portfolio risk "
                f"(was {before})"
            )
        if (
            self.effective_positions_before is not None
            and self.effective_positions_after is not None
        ):
            lines.append(
                f"[portfolio] risk spread across {self.effective_positions_before:.1f} -> "
                f"{self.effective_positions_after:.1f} effective position(s)"
            )
        if self.max_correlation is not None:
            other, rho = self.max_correlation
            # The verdict must follow the number it quotes. This clause used to be appended
            # unconditionally, and `max_correlation` is non-None for any non-empty prior book, so
            # it fired on every run regardless: a four-name book whose effective positions ROSE
            # 2.4 -> 3.2 at a max |rho| of 0.26 was told, two lines below that, that the trade
            # "buys less diversification than the position count suggests". The two sentences
            # contradicted each other and the correlation one was wrong.
            if abs(rho) >= SAME_TRADE_RHO:
                verdict = "adding it buys less diversification than the position count suggests"
            elif abs(rho) <= DISTINCT_TRADE_RHO:
                verdict = "largely a different bet from what you already own"
            else:
                verdict = (
                    "partly the same trade; the diversification is real but smaller than the "
                    "position count suggests"
                )
            lines.append(f"[portfolio] most correlated with {other} at {rho:+.2f}; {verdict}")
        lines.extend(f"[portfolio] {note}" for note in self.notes)
        return lines


def assess(
    *,
    symbol: str,
    weights_before: Mapping[str, float],
    weights_after: Mapping[str, float],
    columns: Mapping[str, Sequence[float]],
    benchmark: Sequence[float],
    session: Session = Session.OPEN,
) -> TradeImpact:
    """How a proposed trade changes beta, risk share, diversification and correlation.

    ``session`` defaults to :attr:`Session.OPEN` because that is the honest default on this venue:
    the blended figure is dominated by shut bars and understates open-session exposure on nine of
    eleven rTokens. A caller who wants the blended number must ask for it.
    """
    notes: list[str] = []

    def portfolio_beta(weights: Mapping[str, float]) -> float | None:
        total = 0.0
        seen = False
        for name, weight in weights.items():
            series = columns.get(name)
            if series is None:
                continue
            value = beta(series, benchmark)
            if value is None:
                continue
            total += weight * value
            seen = True
        return total if seen else None

    before = decompose(weights_before, columns) if weights_before else None
    after = decompose(weights_after, columns)

    if after is None:
        notes.append(
            "risk could not be decomposed: too few aligned observations, or the book has no "
            "variance. Reported as unknown rather than as zero risk"
        )

    rho: tuple[str, float] | None = None
    mine = columns.get(symbol)
    if mine is not None:
        best: tuple[str, float] | None = None
        for other, series in columns.items():
            if other == symbol or weights_before.get(other, 0.0) == 0:
                continue
            value = correlation(mine, series)
            if value is None:
                continue
            if best is None or abs(value) > abs(best[1]):
                best = (other, value)
        rho = best

    if session is Session.BLENDED:
        notes.append(
            "this is the blended figure across open and shut bars; roughly 82% of bars fall while "
            "the anchor market is shut, so it understates open-session exposure"
        )

    return TradeImpact(
        symbol=symbol,
        beta_before=portfolio_beta(weights_before),
        beta_after=portfolio_beta(weights_after),
        risk_share_before=None if before is None else before.share_of_risk(symbol),
        risk_share_after=None if after is None else after.share_of_risk(symbol),
        effective_positions_before=None if before is None else before.effective_positions,
        effective_positions_after=None if after is None else after.effective_positions,
        session=session,
        max_correlation=rho,
        notes=tuple(notes),
    )




# =============================================================================================
# Stress — the part no reference implements
# =============================================================================================

WORST_WINDOW_BARS = 24
"""Length of the realised worst-case window, in bars. One day of hourly data."""


@dataclass(frozen=True, slots=True)
class Shock:
    """A named move in the benchmark, propagated to the book through each position's beta."""

    name: str
    benchmark_move_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "benchmark_move_pct": self.benchmark_move_pct}


STANDARD_SHOCKS: tuple[Shock, ...] = (
    Shock("benchmark -2%", -2.0),
    Shock("benchmark -5%", -5.0),
    Shock("benchmark -10%", -10.0),
    Shock("benchmark +5%", 5.0),
)
"""Sized to the instrument, not to folklore. A 10% move in QQQ is a severe but real day; going to
-30% would produce a number nobody acts on and would say more about the author than the book."""


@dataclass(frozen=True, slots=True)
class StressOutcome:
    """What one shock does to the book."""

    shock: str
    portfolio_move_pct: float | None
    worst_position: tuple[str, float] | None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "shock": self.shock,
            "portfolio_move_pct": (
                None if self.portfolio_move_pct is None else round(self.portfolio_move_pct, 3)
            ),
            "worst_position": (
                None if self.worst_position is None
                else {
                    "symbol": self.worst_position[0],
                    "move_pct": round(self.worst_position[1], 3),
                }
            ),
            "reason": self.reason,
        }


def stress_by_beta(
    *,
    weights: Mapping[str, float],
    columns: Mapping[str, Sequence[float]],
    benchmark: Sequence[float],
    shocks: Sequence[Shock] = STANDARD_SHOCKS,
) -> list[StressOutcome]:
    """Propagate a benchmark shock through each position's beta.

    **Not each position shocked independently**, which is the usual mistake: positions in one book
    move together, and a table that shocks each one on its own reports a diversification that does
    not exist. Here one move in the benchmark reaches every position through its own beta, so the
    correlation is carried by the betas rather than assumed away.

    What this deliberately does *not* claim is the idiosyncratic part. Beta explains the share of a
    move the benchmark drives; the residual is real and is not modelled here. So these figures are
    the **market-driven** component of a shock, and :func:`worst_window` is the counterweight that
    uses realised moves instead.
    """
    out: list[StressOutcome] = []
    for shock in shocks:
        total = 0.0
        worst: tuple[str, float] | None = None
        seen = False
        for symbol, weight in weights.items():
            series = columns.get(symbol)
            if series is None:
                continue
            b = beta(series, benchmark)
            if b is None:
                continue
            move = b * shock.benchmark_move_pct
            total += weight * move
            seen = True
            if worst is None or move < worst[1]:
                worst = (symbol, move)
        out.append(
            StressOutcome(shock=shock.name, portfolio_move_pct=total if seen else None,
                          worst_position=worst,
                          reason="" if seen else "no position had an estimable beta")
        )
    return out


@dataclass(frozen=True, slots=True)
class WorstWindow:
    """The worst run this exact book would actually have had, from realised returns."""

    bars: int
    move_pct: float | None
    start_index: int | None
    contributors: tuple[tuple[str, float], ...]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "bars": self.bars,
            "move_pct": None if self.move_pct is None else round(self.move_pct, 3),
            "start_index": self.start_index,
            "contributors": [
                {"symbol": s, "move_pct": round(v, 3)} for s, v in self.contributors
            ],
            "reason": self.reason,
        }

    def render(self) -> str:
        if self.move_pct is None:
            return f"[stress] no {self.bars}-bar window could be evaluated: {self.reason}"
        worst = ", ".join(f"{s} {v:+.2f}%" for s, v in self.contributors[:3])
        return (
            f"[stress] the worst {self.bars}-bar window in the observed history would have moved "
            f"this book {self.move_pct:+.2f}% — driven by {worst}"
        )


def worst_window(
    *,
    weights: Mapping[str, float],
    columns: Mapping[str, Sequence[float]],
    bars: int = WORST_WINDOW_BARS,
) -> WorstWindow:
    """The worst realised run for this book, found by sliding over the actual history.

    Non-parametric and correlation-exact. It asks nothing of a covariance matrix and assumes no
    distribution: it replays what the market actually did to *these weights*, so the co-movement in
    the answer is the co-movement that happened rather than the co-movement a model believes in.

    This is the part none of the four reference libraries implement. PyPortfolioOpt, cvxportfolio,
    qlib and riskparity.py were read for it and the search returned nothing — cvxportfolio replays
    history, but as ordinary backtesting rather than as a stress question, and PyPortfolioOpt's CVaR
    takes a tail quantile of unordered returns, which discards the ordering that makes a drawdown a
    drawdown.
    """
    held = {s: list(c) for s, c in columns.items() if float(weights.get(s, 0.0)) != 0.0}
    if not held:
        return WorstWindow(bars=bars, move_pct=None, start_index=None, contributors=(),
                           reason="no position carries a weight")
    length = min(len(c) for c in held.values())
    if length < bars:
        return WorstWindow(
            bars=bars, move_pct=None, start_index=None, contributors=(),
            reason=f"{length} aligned bar(s) is fewer than the {bars}-bar window",
        )

    best_start, best_move = None, None
    for start in range(length - bars + 1):
        total = 0.0
        for symbol, series in held.items():
            # Compounded, not summed: a run of returns multiplies, and summing them overstates a
            # fall. Over 24 hourly bars the difference is small and it is still the wrong sum.
            growth = 1.0
            for value in series[start: start + bars]:
                growth *= 1.0 + value
            total += float(weights[symbol]) * (growth - 1.0)
        if best_move is None or total < best_move:
            best_start, best_move = start, total

    contributors: list[tuple[str, float]] = []
    if best_start is not None:
        for symbol, series in held.items():
            growth = 1.0
            for value in series[best_start: best_start + bars]:
                growth *= 1.0 + value
            contributors.append((symbol, (growth - 1.0) * 100.0))
        contributors.sort(key=lambda pair: pair[1])

    return WorstWindow(
        bars=bars,
        move_pct=None if best_move is None else best_move * 100.0,
        start_index=best_start,
        contributors=tuple(contributors),
    )





# =============================================================================================
# Tail risk — who carries the loss in the worst hours, not only the variance
# =============================================================================================

TAIL_CONFIDENCE = 0.95
"""The CVaR level: the average of the worst 5% of bars."""


@dataclass(frozen=True, slots=True)
class TailRisk:
    """The book's historical CVaR and each position's share of it."""

    confidence: float
    cvar: float
    """Average loss over the worst ``1 - confidence`` of bars, as a positive fraction."""

    bars: int
    contributions: tuple[tuple[str, float], ...]
    """``(symbol, contribution)``; contributions sum to :attr:`cvar`."""

    def share(self, symbol: str) -> float | None:
        for name, value in self.contributions:
            if name == symbol:
                return value / self.cvar if self.cvar > 0 else None
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"confidence": self.confidence, "cvar": round(self.cvar, 8), "bars": self.bars,
                "contributions": {k: round(v, 8) for k, v in self.contributions}}


def tail_contributions(
    weights: Mapping[str, float], columns: Mapping[str, Sequence[float]],
    confidence: float = TAIL_CONFIDENCE,
) -> TailRisk | None:
    """Historical CVaR of the book and its exact Euler split across positions.

    **The definition is skfolio's** (``measures/_measures.py:602-655``, BSD-3): with ``k = (1 -
    beta) * n`` and ``ik = max(0, ceil(k) - 1)``, CVaR is minus the sum of the ``ik`` worst returns
    over ``k``, plus the next-worst return weighted by ``(ik / k - 1)`` — the fractional bar a
    finite sample needs. CVaR is positively homogeneous in the weights, so its gradient times the
    weights sums to it exactly (Euler); the gradient is each name's return over those same bars in
    the same proportions. skfolio's ``Portfolio.contribution(CVaR)`` estimates the same quantity by
    central finite differences (``portfolio/_portfolio.py:1032-1082, 1575-1603``); the two agree
    wherever the ordering of the worst bars does not change inside its step, and the agreement is
    checked against skfolio itself (`data/tail_contribution_oracle.json`).

    **Why beside the variance split.** Variance treats a name that jumps up like one that crashes.
    A position can carry 19% of a book's variance and far more of its worst hours — the
    concentration a trader actually feels. Computed over every aligned bar, open and shut, because
    the losses that matter to a stock-perpetual holder are often the ones that land while the US
    market is closed.
    """
    held = {s: list(c) for s, c in columns.items() if float(weights.get(s, 0.0)) != 0.0}
    if not held:
        return None
    n = min(len(c) for c in held.values())
    if n < MIN_OBSERVATIONS:
        return None
    book = [sum(float(weights[s]) * held[s][t] for s in held) for t in range(n)]
    k = (1.0 - confidence) * n
    ik = max(0, math.ceil(k) - 1)
    order = sorted(range(n), key=lambda t: book[t])
    worst, edge = order[:ik], order[ik]
    tail_weight = ik / k - 1.0

    def loss(series: Sequence[float]) -> float:
        return -sum(series[t] for t in worst) / k + series[edge] * tail_weight

    cvar = loss(book)
    contributions = tuple((s, float(weights[s]) * loss(c)) for s, c in sorted(held.items()))
    total = sum(v for _, v in contributions)
    if abs(total - cvar) > max(IDENTITY_TOLERANCE, abs(cvar) * 1e-9):
        raise PortfolioError(f"tail contributions sum to {total!r}, CVaR is {cvar!r}")
    return TailRisk(confidence=confidence, cvar=cvar, bars=n, contributions=contributions)


# =============================================================================================
# Factor exposure — the last item on the Open Theme's list
# =============================================================================================


@dataclass(frozen=True, slots=True)
class FactorExposure:
    """How much of the book's movement one factor explains, and in which direction."""

    factor: str
    exposure: float | None
    """Regression slope of portfolio returns on the factor's returns."""

    observations: int
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "exposure": None if self.exposure is None else round(self.exposure, 4),
            "observations": self.observations,
            "reason": self.reason,
        }


def portfolio_returns(
    weights: Mapping[str, float], columns: Mapping[str, Sequence[float]]
) -> list[float]:
    """The book's own return series, bar by bar.

    Built from held names only and from aligned columns, so bar *i* of the result is the book's
    return over the same instant for every position. A portfolio series stitched from differently
    aligned inputs would produce a factor exposure to nothing in particular.
    """
    held = {s: c for s, c in columns.items() if float(weights.get(s, 0.0)) != 0.0}
    if not held:
        return []
    length = min(len(c) for c in held.values())
    return [
        sum(float(weights[s]) * held[s][i] for s in held)
        for i in range(length)
    ]


def factor_exposures(
    *,
    weights: Mapping[str, float],
    columns: Mapping[str, Sequence[float]],
    factors: Mapping[str, Sequence[float]],
) -> list[FactorExposure]:
    """Regress the book's returns on each factor's returns, one factor at a time.

    **Univariate on purpose.** A multivariate regression would report exposures net of the other
    factors, which is the right answer for a risk model and the wrong one for a copilot: a trader
    asking "how exposed am I to momentum" wants the total, not the part left over once four
    correlated factors have argued about it. Our factors are session-structural and overlap heavily,
    so the net figures would be unstable and would read as precision.

    The factor series are the ones `argus.research.factor_lab` evaluates, which means this is the
    one place the factor laboratory and the portfolio copilot meet: a factor that survived the
    anti-overfit gates can be asked about the book directly.
    """
    book = portfolio_returns(weights, columns)
    out: list[FactorExposure] = []
    for name, series in sorted(factors.items()):
        length = min(len(book), len(series))
        if length < MIN_OBSERVATIONS:
            out.append(FactorExposure(
                factor=name, exposure=None, observations=length,
                reason=f"{length} aligned bar(s); fewer than {MIN_OBSERVATIONS} is noise",
            ))
            continue
        value = beta(book[:length], list(series[:length]))
        out.append(FactorExposure(
            factor=name, exposure=value, observations=length,
            reason="" if value is not None else "the factor did not vary over these bars",
        ))
    return out


def parse_book(pairs: Sequence[str]) -> dict[str, float]:
    """``NVDAUSDT=0.2`` pairs into weights. Refuses a book that does not sum to one."""
    book: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected SYMBOL=weight, got {pair!r}")
        symbol, _, raw = pair.partition("=")
        try:
            book[symbol.strip().upper()] = float(raw)
        except ValueError:
            raise SystemExit(f"{raw!r} is not a weight") from None
    total = sum(book.values())
    if book and abs(total - 1.0) > 0.01:
        raise SystemExit(
            f"the weights sum to {total:.3f}, not 1.0 — a book that does not add up would make "
            f"every number below wrong in a way that is hard to see"
        )
    return book


@dataclass(frozen=True, slots=True)
class CopilotReport:
    """One complete answer to *what does this trade do to this book* — the research task the
    Track 3 Open Theme names, as data rather than as printed text.

    Built by :func:`copilot`, which is pure: it takes aligned returns and never fetches. The CLI
    (:func:`main`) and the natural-language console (`argus.lui.research`) both call it, so a judge
    typing a question and a reader running the command get the same numbers from the same code.
    An empty ``weights_before`` is legal and means "no book given": the report is then a standalone
    risk profile of the one symbol, which is the honest answer when nobody has said what they own.
    """

    symbol: str
    benchmark: str
    weights_before: Mapping[str, float]
    weights_after: Mapping[str, float]
    bars_aligned: int
    bars_open_session: int
    impact: TradeImpact
    risk_after: RiskDecomposition | None
    stress: tuple[StressOutcome, ...]
    worst: WorstWindow
    exposures: tuple[FactorExposure, ...]
    session_beta: Mapping[Session, Exposure]
    tail: TailRisk | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "benchmark": self.benchmark,
            "weights_before": dict(self.weights_before),
            "weights_after": {k: round(v, 4) for k, v in self.weights_after.items()},
            "bars_aligned": self.bars_aligned,
            "bars_open_session": self.bars_open_session,
            "impact": self.impact.as_dict(),
            "risk_after": None if self.risk_after is None else self.risk_after.as_dict(),
            "stress_by_beta": [s.as_dict() for s in self.stress],
            "worst_window": self.worst.as_dict(),
            "factor_exposures": [e.as_dict() for e in self.exposures],
            "session_beta": {
                str(k): (None if v.beta is None else round(v.beta, 4))
                for k, v in self.session_beta.items()
            },
            "tail": None if self.tail is None else self.tail.as_dict(),
        }


def rebalance(before: Mapping[str, float], add: str, size: float) -> dict[str, float]:
    """The book after buying ``add`` to a target weight of ``size``, the rest scaled pro rata.

    An empty book becomes ``{add: 1.0}`` — a standalone position — rather than ``{add: size}``,
    because a book that sums to ``size`` is not a book, and every risk share computed from it would
    be read as a fraction of a whole that does not exist.
    """
    if not 0.0 < size <= 1.0:
        raise PortfolioError(f"target weight must be in (0, 1], got {size}")
    if not before:
        return {add: 1.0}
    after = {s: w * (1.0 - size) for s, w in before.items()}
    after[add] = after.get(add, 0.0) + size
    return after


def copilot(
    *,
    add: str,
    before: Mapping[str, float],
    size: float,
    raw: Mapping[str, Mapping[datetime, float]],
    benchmark: str,
    is_open: Any,
) -> CopilotReport:
    """The whole portfolio-copilot research task over already-fetched returns.

    ``raw`` maps each symbol (the book, the candidate and the benchmark) to its per-bar returns.
    ``is_open`` is the session predicate — injected, as in :func:`session_betas`, so the clock
    stays the clock's business. Impact and beta-propagated stress are read over **open-session**
    bars only (see this module's docstring for why the blended figure misleads); the realised
    worst window and factor exposures use every aligned bar, because the book is exposed to the
    shut session too and a stress test that skipped 82% of the hours would understate it.
    """
    after = rebalance(before, add, size)
    stamps, columns = align(raw)
    if benchmark not in columns:
        raise PortfolioError(f"no aligned returns for the benchmark {benchmark}")
    open_rows = [i for i, t in enumerate(stamps) if is_open(t)]
    open_columns = {k: [v[i] for i in open_rows] for k, v in columns.items()}

    impact = assess(
        symbol=add, weights_before=before, weights_after=after,
        columns=open_columns, benchmark=open_columns[benchmark], session=Session.OPEN,
    )
    # Exposure to our own session-structural factors. These are slices of the same market series,
    # so they overlap heavily and each figure is a TOTAL exposure, not a residual — see
    # `factor_exposures` for why that is the right answer for a copilot. Built over `columns`, the
    # SAME aligned bars the book series uses: an earlier version built it from the open-session
    # subset and regressed it against the full book, pairing two different periods and reporting
    # exposures of about zero. Truncating two series to a common length is not aligning them.
    market = portfolio_returns({s: 1.0 / len(columns) for s in columns}, columns)
    exposures = factor_exposures(
        weights=after, columns=columns,
        factors={
            "equal-weight market": market,
            "open session only": [
                m if is_open(stamps[i]) else 0.0 for i, m in enumerate(market[: len(stamps)])
            ],
            "shut session only": [
                0.0 if is_open(stamps[i]) else m for i, m in enumerate(market[: len(stamps)])
            ],
        },
    )
    return CopilotReport(
        symbol=add, benchmark=benchmark,
        weights_before=dict(before), weights_after=after,
        bars_aligned=len(stamps), bars_open_session=len(open_rows),
        impact=impact,
        risk_after=decompose(after, open_columns),
        stress=tuple(stress_by_beta(
            weights=after, columns=open_columns, benchmark=open_columns[benchmark],
        )),
        worst=worst_window(weights=after, columns=columns),
        exposures=tuple(exposures),
        session_beta=session_betas(raw[add], raw[benchmark], symbol=add, is_open=is_open),
        tail=tail_contributions(after, columns),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """One complete research task: what does this trade do to this book?

    Track 3 asks for a full flow from question to actionable insight. The question here is the one
    a portfolio-aware copilot exists to answer — *should I add this, given what I already own* —
    and every number printed is computed from live Bitget candles rather than asserted. The
    arithmetic is :func:`copilot`; this function only fetches and prints.
    """
    import argparse

    from argus.market.history import CandleType, fetch_range
    from argus.truth.clocks import DualClock

    parser = argparse.ArgumentParser(
        description="Portfolio copilot: what a proposed trade does to your book."
    )
    parser.add_argument(
        "--book", nargs="+", required=True, metavar="SYMBOL=WEIGHT",
        help="current holdings, e.g. TSLAUSDT=0.4 MSFTUSDT=0.4 METAUSDT=0.2",
    )
    parser.add_argument("--add", required=True, help="the symbol being considered")
    parser.add_argument(
        "--size", type=float, default=0.2,
        help="target weight for the new position; existing holdings are scaled down pro rata",
    )
    parser.add_argument("--benchmark", default="QQQUSDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    before = parse_book(args.book)
    add = args.add.upper()
    raw: dict[str, dict[datetime, float]] = {}
    for symbol in sorted({*before, add, args.benchmark}):
        candles = fetch_range(
            symbol, days=args.days, interval="1H", candle_type=CandleType.MARKET
        )
        raw[symbol] = returns([(c.ts, float(c.close)) for c in candles])

    clock = DualClock()
    report = copilot(
        add=add, before=before, size=args.size, raw=raw, benchmark=args.benchmark,
        is_open=lambda t: clock.phase(t).has_price_discovery,
    )

    if args.json:
        payload = {"question": f"should I add {add} to this book?", **report.as_dict()}
        print(json.dumps(payload, indent=2))
        return 0

    print(f"question: should I add {add} to this book?")
    print(
        f"{report.bars_aligned} aligned hourly bars, {report.bars_open_session} of them while the "
        f"anchor market was open; benchmark {args.benchmark}"
    )
    print()
    for line in report.impact.render():
        print(" ", line)
    if report.risk_after is not None:
        print("\n  weight against risk, after the trade:")
        for item in sorted(report.risk_after.contributions, key=lambda c: -c.contribution):
            share = item.contribution / report.risk_after.volatility
            print(f"    {item.symbol:10} weight {item.weight:6.1%}   risk {share:6.1%}")
    print("\n  if the benchmark moves (through each position's open-session beta):")
    for outcome in report.stress:
        if outcome.portfolio_move_pct is None:
            print(f"    {outcome.shock:16} unavailable — {outcome.reason}")
            continue
        worst = outcome.worst_position
        tail = f"   worst {worst[0]} {worst[1]:+.2f}%" if worst else ""
        print(f"    {outcome.shock:16} book {outcome.portfolio_move_pct:+7.2f}%{tail}")
    print()
    print(" ", report.worst.render())
    print()
    print("  factor exposure (univariate, so each is a total rather than a residual):")
    for exposure in report.exposures:
        if exposure.exposure is None:
            print(f"    {exposure.factor:22} unavailable — {exposure.reason}")
        else:
            print(f"    {exposure.factor:22} {exposure.exposure:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IDENTITY_TOLERANCE",
    "MIN_OBSERVATIONS",
    "STANDARD_SHOCKS",
    "WORST_WINDOW_BARS",
    "Contribution",
    "CopilotReport",
    "Exposure",
    "FactorExposure",
    "PortfolioError",
    "RiskDecomposition",
    "Session",
    "Shock",
    "StressOutcome",
    "TailRisk",
    "TradeImpact",
    "WorstWindow",
    "align",
    "assess",
    "beta",
    "copilot",
    "correlation",
    "covariance",
    "covariance_matrix",
    "decompose",
    "factor_exposures",
    "main",
    "parse_book",
    "portfolio_returns",
    "rebalance",
    "returns",
    "session_betas",
    "stress_by_beta",
    "tail_contributions",
    "variance",
    "worst_window",
]
