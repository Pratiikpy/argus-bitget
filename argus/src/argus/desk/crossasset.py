"""Cross-asset hedge routing: hedge only the risk and carry the allocation did not sign up for.

The handbook's Cross-Asset Execution sub-theme asks how an agent manages rToken and crypto positions
at once — "hedge crypto when rToken anomalies; dynamic cross-market allocation after macro shocks".
`desk/execution.py` answered a narrower question (which of two legs is cheaper to hold, entry plus
funding) and never asked whether the leg hedged anything. This module answers the whole question
for a book that holds rToken spot and crypto perpetuals: **when to hedge, through which perpetual,
how much, and when not to.**

**The objective, stated once.** A holder's allocation already says how much risk they accept in
normal conditions. Reverse optimisation (the Black-Litterman equilibrium step) turns that allocation
into the expected returns that make it optimal under the normal covariance,
``mu = gamma * S_normal * w``, plus the normal funding carry for perpetuals. The router then
maximises the certainty equivalent over the horizon to the next session boundary,

    CE(x) = mu' v - (gamma / 2 NAV) v' S_H v - trading cost(x - x0) - exit cost - funding(x),

over the perpetual notionals ``x`` (spot holdings are never traded), where ``S_H`` is the
covariance *forecast for this horizon*: conditioned on the session phase (US cash market open or
shut) and scaled by each instrument's current volatility regime. Under normal risk and normal carry
the optimum is the holder's own book — no trade. It hedges exactly when the forecast risk, or the
funding, departs from normal by more than the fees, the walked slippage and the funding of the
hedge.

That single rule covers both handbook examples without ad-hoc triggers: an rToken anomaly or a
macro shock raises the forecast variance (and, through the phase-conditioned correlations, the
cross-asset covariance), which raises the value of hedging; a funding spike on the crypto leg raises
its carry above normal, which makes reducing it worth the round trip.

**What was read before this was written, and what was taken.**

* Reverse optimisation / implied returns: He & Litterman (1999), the equilibrium step
  ``Pi = delta * Sigma * w_mkt``. Taken as the anchor that keeps a hedge overlay from drifting the
  holder's allocation toward global minimum variance.
* Minimum-variance hedge ratio and its variance reduction ``r^2``: ARGUS's own
  `desk/diversification.py:420-453`; the covariance route here generalises it to several
  instruments at once.
* Sign-stable proxies only: Modemola/BITGET_HACK `src/blackout/hedges.py:58-60` (``is_stable``,
  read in `eval/hedge_tradeable.py`), rebuilt **sign-symmetric** because theirs rejects every
  inverse hedge (`eval/hedge_tradeable.py` finding 1). A cross-asset correlation is used only when
  it has the same sign in both halves of the lookback and ``|r| >= 0.2`` in each.
* Session phases: the NYSE calendar of `market/rtoken_spot.py:110-160` (Ballast's rules). The bar
  containing the cash open is counted as shut, because the gap is realised in it.
* Costs: perpetual taker fee and slippage walked on a recorded book (`market/depth.py`), funding
  from the venue's settlements (`research/carry.py`), as in `cost/model.py`.
* HedgeAgents' Extreme Market Conference (arXiv 2502.13165 §3.4.3) fires on fixed 5% / 10%
  amplitude thresholds. Rejected as a trigger: a fixed amplitude is blind to which session the move
  happened in and to what the hedge costs. Its idea — respond to regime shocks — is kept, as the
  volatility-regime multiplier in ``S_H``.
* Rival behaviour this is compared against, run unmodified in `eval/xa_arena.py`: Triad (sell BTC
  on a RAAPL-vs-BTC 24 h gap), Omni (short the mapped stock perpetual when a margin model breaches a
  budget), Crossfire (policy HOLD without its LLM), VIGIL (Fear & Greed rotation), HedgeAgents
  (periodic budget allocation).

**What it deliberately does not do.** It never trades the spot rTokens (10 bps taker plus a thin
book; the holder chose them), takes no directional view (no expected return beyond the implied
one), and refuses to act — keeps the current overlay — when there is too little history in the
phase to estimate a covariance (``min_blocks``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from argus.desk.diversification import DiversificationError, jacobi_eigen
from argus.market.rtoken_spot import close_utc, is_trading_day, open_utc

HOUR_MS = 3_600_000

PERP_OF_SPOT: dict[str, str] = {
    "RNVDAUSDT": "NVDAUSDT", "RAAPLUSDT": "AAPLUSDT", "RTSLAUSDT": "TSLAUSDT",
    "RQQQUSDT": "QQQUSDT", "RSPYUSDT": "SPYUSDT", "BTCUSDT": "BTCUSDT", "ETHUSDT": "ETHUSDT",
}
"""A spot instrument and the perpetual on the same underlying."""

OPEN, SHUT = "open", "shut"


def _underlying(key: str) -> str:
    kind, sym = key.split(":", 1)
    return PERP_OF_SPOT.get(sym, sym) if kind == "spot" else sym


def same_underlying(a: str, b: str) -> bool:
    return _underlying(a) == _underlying(b)


# --- session phases ---------------------------------------------------------------------------


def _ceil_hour(when: datetime) -> datetime:
    floored = when.replace(minute=0, second=0, microsecond=0)
    return floored if floored == when else floored + timedelta(hours=1)


def phase_of_bar(open_ms: int) -> str:
    """``open`` when the hour bar starting at ``open_ms`` lies wholly inside a US cash session
    (after the bar containing the open), else ``shut``."""
    t = datetime.fromtimestamp(open_ms / 1000, UTC)
    # A US session's UTC span lies within one UTC calendar day (13:30-21:00 at the latest).
    d = t.date()
    if not is_trading_day(d):
        return SHUT
    start = _ceil_hour(open_utc(d))
    end = close_utc(d).replace(minute=0, second=0, microsecond=0)
    return OPEN if start <= t < end else SHUT


def phase_calendar(hours: Sequence[int]) -> list[str]:
    cache: dict[date, tuple[datetime, datetime] | None] = {}
    out: list[str] = []
    for ms in hours:
        t = datetime.fromtimestamp(ms / 1000, UTC)
        d = t.date()
        if d not in cache:
            cache[d] = ((_ceil_hour(open_utc(d)),
                         close_utc(d).replace(minute=0, second=0, microsecond=0))
                        if is_trading_day(d) else None)
        span = cache[d]
        out.append(OPEN if span is not None and span[0] <= t < span[1] else SHUT)
    return out


def hours_to_boundary(phases: Sequence[str], i: int, *, cap: int) -> int:
    """How many bars from ``i`` (inclusive) share bar ``i``'s phase, capped. The calendar is a
    published schedule, so reading ahead in it is not look-ahead."""
    p = phases[i]
    n = 0
    while i + n < len(phases) and phases[i + n] == p and n < cap:
        n += 1
    return max(1, n)


# --- configuration ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterConfig:
    """Pre-registered constants. Chosen before the scored period, never fitted on it."""

    gamma: float = 5.0
    """Risk aversion of the certainty equivalent. The primary evaluation uses the same number."""

    lookback_hours: int = 60 * 24
    block_hours: Mapping[str, int] = field(default_factory=lambda: {OPEN: 3, SHUT: 6})
    """Block length per phase for covariance estimation: long enough to wash out the asynchronous
    trading of a thin rToken book, short enough to leave a usable number of blocks."""

    normal_block_hours: int = 24
    ewma_lambda: float = 0.97
    """Hourly decay of the volatility-regime estimate (half-life about 23 hours)."""

    regime_clip: tuple[float, float] = (0.5, 6.0)
    shrink: float = 0.2
    """Shrinkage of every correlation matrix toward the identity."""

    min_blocks: int = 20
    stability_min_abs_corr: float = 0.2
    horizon_cap_hours: int = 72
    min_trade_usd: float = 50.0
    max_abs_notional_nav: float = 2.0
    use_funding: bool = True
    use_costs: bool = True
    use_stability_gate: bool = True
    use_regime: bool = True
    use_phase: bool = True
    cross_asset: bool = True
    """False restricts every exposure to the perpetual on its own underlying (ablation)."""


@dataclass(frozen=True)
class RiskModel:
    """One hour's covariance forecast, shared by every router variant that decides at that hour."""

    keys: tuple[str, ...]
    horizon_hours: int
    phase: str
    cov_h: list[list[float]]
    """Forecast covariance of returns over the horizon (phase-conditioned, regime-scaled, gated)."""

    cov_normal_h: list[list[float]]
    """Unconditional covariance over the same horizon: what the allocation was chosen under."""

    regime: dict[str, float]
    gated_pairs: tuple[tuple[str, str], ...]
    blocks: int
    ok: bool
    reason: str = ""


# --- estimation -------------------------------------------------------------------------------


def _log_ret(a: float, b: float) -> float:
    return math.log(b / a) if a > 0 and b > 0 else 0.0


def _cov(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(rows)
    k = len(rows[0])
    means = [sum(r[j] for r in rows) / n for j in range(k)]
    out = [[0.0] * k for _ in range(k)]
    for a in range(k):
        for b in range(a, k):
            s = sum((r[a] - means[a]) * (r[b] - means[b]) for r in rows) / (n - 1)
            out[a][b] = out[b][a] = s
    return out


def _corr_of(cov: Sequence[Sequence[float]]) -> tuple[list[float], list[list[float]]]:
    sd = [math.sqrt(max(cov[i][i], 0.0)) for i in range(len(cov))]
    corr = [[(cov[i][j] / (sd[i] * sd[j]) if sd[i] > 0 and sd[j] > 0 else (1.0 if i == j else 0.0))
             for j in range(len(cov))] for i in range(len(cov))]
    return sd, corr


def _psd_corr(corr: list[list[float]]) -> list[list[float]]:
    """Nearest correlation-like matrix by eigenvalue clipping, used after gating zeroes entries."""
    try:
        values, vectors = jacobi_eigen(corr)
    except DiversificationError:
        return corr
    if min(values) > 1e-8:
        return corr
    n = len(corr)
    clipped = [max(v, 1e-6) for v in values]
    rebuilt = [[sum(clipped[k] * vectors[k][i] * vectors[k][j] for k in range(n))
                for j in range(n)] for i in range(n)]
    d = [math.sqrt(rebuilt[i][i]) for i in range(n)]
    return [[rebuilt[i][j] / (d[i] * d[j]) for j in range(n)] for i in range(n)]


def _blocks(closes: Mapping[str, Sequence[float]], keys: Sequence[str], phases: Sequence[str],
            lo: int, hi: int, *, phase: str | None, length: int) -> list[tuple[int, list[float]]]:
    """Returns of consecutive ``length``-bar blocks in ``[lo, hi]`` whose bars all share
    ``phase`` (or any phase when None), cut from the start of each phase segment."""
    out: list[tuple[int, list[float]]] = []
    i = lo
    while i + length - 1 <= hi:
        if phase is not None and not all(phases[j] == phase for j in range(i, i + length)):
            i += 1
            continue
        start, end = i - 1, i + length - 1
        if start < 0:
            i += 1
            continue
        out.append((end, [_log_ret(closes[k][start], closes[k][end]) for k in keys]))
        i += length
    return out


def estimate(closes: Mapping[str, Sequence[float]], keys: Sequence[str], phases: Sequence[str],
             i: int, cfg: RouterConfig, *, real: Mapping[str, Sequence[bool]] | None = None,
             ) -> RiskModel:
    """The covariance forecast at the close of bar ``i`` for the horizon to the next phase
    boundary. Uses bars ``<= i`` only (``phases`` may extend past ``i``: it is a calendar)."""
    lo = max(1, i - cfg.lookback_hours + 1)
    upcoming = phases[i + 1] if i + 1 < len(phases) else phases[i]
    horizon = hours_to_boundary(phases, min(i + 1, len(phases) - 1), cap=cfg.horizon_cap_hours)
    phase = upcoming if cfg.use_phase else None
    length = cfg.block_hours[upcoming] if cfg.use_phase else cfg.normal_block_hours
    cond = _blocks(closes, keys, phases, lo, i, phase=phase, length=length)
    normal = _blocks(closes, keys, phases, lo, i, phase=None, length=cfg.normal_block_hours)
    if len(cond) < cfg.min_blocks or len(normal) < max(10, len(keys) + 2):
        return RiskModel(tuple(keys), horizon, upcoming, [], [], {}, (), len(cond), False,
                         f"{len(cond)} {upcoming}-phase blocks in the lookback, "
                         f"{cfg.min_blocks} required")

    def shrunk(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[list[float]]]:
        sd, corr = _corr_of(_cov(rows))
        d = cfg.shrink
        corr = [[(1 - d) * corr[a][b] + (d if a == b else 0.0) for b in range(len(keys))]
                for a in range(len(keys))]
        return sd, corr

    rows_c = [r for _, r in cond]
    sd_c, corr_c = shrunk(rows_c)
    sd_n, corr_n = shrunk([r for _, r in normal])

    gated: list[tuple[str, str]] = []
    if cfg.use_stability_gate:
        half = len(rows_c) // 2
        _, c1 = _corr_of(_cov(rows_c[:half]))
        _, c2 = _corr_of(_cov(rows_c[half:]))
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if same_underlying(keys[a], keys[b]):
                    continue
                r1, r2 = c1[a][b], c2[a][b]
                if (r1 > 0) != (r2 > 0) or min(abs(r1), abs(r2)) < cfg.stability_min_abs_corr:
                    gated.append((keys[a], keys[b]))
                    corr_c[a][b] = corr_c[b][a] = 0.0
                    corr_n[a][b] = corr_n[b][a] = 0.0
        if gated:
            corr_c = _psd_corr(corr_c)
            corr_n = _psd_corr(corr_n)

    regime = {k: 1.0 for k in keys}
    if cfg.use_regime:
        regime = _regime(closes, keys, phases, lo, i, cfg, real=real)

    scale_c = horizon / length
    scale_n = horizon / cfg.normal_block_hours
    n = len(keys)
    m = [math.sqrt(regime[k]) for k in keys]
    cov_h = [[corr_c[a][b] * sd_c[a] * sd_c[b] * m[a] * m[b] * scale_c for b in range(n)]
             for a in range(n)]
    cov_nh = [[corr_n[a][b] * sd_n[a] * sd_n[b] * scale_n for b in range(n)] for a in range(n)]
    return RiskModel(tuple(keys), horizon, upcoming, cov_h, cov_nh, regime, tuple(gated),
                     len(cond), True)


def _regime(closes: Mapping[str, Sequence[float]], keys: Sequence[str], phases: Sequence[str],
            lo: int, i: int, cfg: RouterConfig, *,
            real: Mapping[str, Sequence[bool]] | None) -> dict[str, float]:
    """EWMA of squared hourly returns, each standardised by its own phase's lookback RMS, so the
    ratio reads "how agitated is this instrument relative to what is normal for this phase".
    A spot rToken borrows the multiplier of the perpetual on its underlying: its own hourly
    returns are zero on every untraded hour, which would read as calm."""
    out: dict[str, float] = {}
    by_underlying: dict[str, float] = {}
    lo_c, hi_c = cfg.regime_clip
    for k in keys:
        if k.startswith("spot:") and f"perp:{_underlying(k)}" in closes:
            continue
        c = closes[k]
        rets: dict[str, list[float]] = {OPEN: [], SHUT: []}
        seq: list[tuple[str, float]] = []
        for j in range(lo, i + 1):
            if real is not None and not real[k][j]:
                continue
            r = _log_ret(c[j - 1], c[j])
            rets[phases[j]].append(r * r)
            seq.append((phases[j], r * r))
        norm = {p: (sum(v) / len(v) if v else 0.0) for p, v in rets.items()}
        ew = 1.0
        for p, sq in seq:
            base = norm[p]
            z2 = sq / base if base > 0 else 1.0
            ew = cfg.ewma_lambda * ew + (1 - cfg.ewma_lambda) * z2
        out[k] = min(max(ew, lo_c), hi_c)
        by_underlying[_underlying(k)] = out[k]
    for k in keys:
        if k not in out:
            out[k] = by_underlying.get(_underlying(k), 1.0)
    return out


# --- the decision -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CostCurve:
    fee_bps: float
    slippage: tuple[tuple[float, float], ...]
    """``(notional, slippage_bps)`` points, ascending; linear between, flat beyond the ends."""

    def bps(self, notional: float) -> float:
        n = abs(notional)
        pts = self.slippage
        if not pts:
            return self.fee_bps
        if n <= pts[0][0]:
            return self.fee_bps + pts[0][1]
        for (x0, y0), (x1, y1) in itertools.pairwise(pts):
            if n <= x1:
                return self.fee_bps + y0 + (y1 - y0) * (n - x0) / (x1 - x0)
        # Beyond the deepest measured size, slippage grows in proportion (a conservative extension
        # rather than a flat one, which would make an enormous order look cheap).
        x_last, y_last = pts[-1]
        return self.fee_bps + y_last * n / x_last


@dataclass(frozen=True)
class RouterInput:
    nav: float
    spot_usd: Mapping[str, float]
    """Current USD value of each spot holding, keyed ``spot:SYMBOL``."""

    perp_usd: Mapping[str, float]
    """Current signed USD notional of each perpetual, keyed ``perp:SYMBOL`` (all candidates)."""

    base_perp_usd: Mapping[str, float]
    """The holder's own (un-hedged) perpetual notional at current prices."""

    funding_now: Mapping[str, float]
    """Last settled funding rate per perpetual, as a fraction per settlement (0 if unobserved)."""

    funding_normal: Mapping[str, float]
    """Mean settled rate over the lookback (0 if unobserved)."""

    settlements_ahead: int
    costs: Mapping[str, CostCurve]


@dataclass(frozen=True)
class RouterDecision:
    target_usd: dict[str, float]
    orders_usd: dict[str, float]
    ce_gain_usd: float
    """Modelled certainty-equivalent improvement of the target over holding, net of all costs."""

    risk_before_usd: float
    risk_after_usd: float
    trade_cost_usd: float
    funding_usd: float
    horizon_hours: int
    phase: str
    refused: bool
    reason: str
    gated_pairs: tuple[tuple[str, str], ...]
    regime: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_usd": {k: round(v, 2) for k, v in self.target_usd.items()},
            "orders_usd": {k: round(v, 2) for k, v in self.orders_usd.items()},
            "ce_gain_usd": round(self.ce_gain_usd, 4),
            "risk_before_usd": round(self.risk_before_usd, 4),
            "risk_after_usd": round(self.risk_after_usd, 4),
            "trade_cost_usd": round(self.trade_cost_usd, 4),
            "funding_usd": round(self.funding_usd, 4),
            "horizon_hours": self.horizon_hours, "phase": self.phase,
            "refused": self.refused, "reason": self.reason,
            "gated_pairs": [list(p) for p in self.gated_pairs],
            "regime": {k: round(v, 3) for k, v in self.regime.items()},
        }

    def explain(self) -> str:
        """One plain sentence a holder can read."""
        if self.refused:
            return f"No hedge change: {self.reason}."
        if not self.orders_usd:
            return (f"Holding: over the next {self.horizon_hours}h ({self.phase} session) no "
                    f"hedge removes more risk than it costs in fees, slippage and funding.")
        legs = ", ".join(f"{'buy' if v > 0 else 'sell'} ${abs(v):,.0f} {k.split(':', 1)[1]}"
                         for k, v in self.orders_usd.items())
        return (f"{legs}: over the next {self.horizon_hours}h ({self.phase} session) this removes "
                f"${self.risk_before_usd - self.risk_after_usd:,.2f} of risk cost for "
                f"${self.trade_cost_usd:,.2f} of trading cost and ${self.funding_usd:,.2f} funding "
                f"(net certainty-equivalent gain ${self.ce_gain_usd:,.2f}).")


def _objective_parts(model: RiskModel, cfg: RouterConfig, inp: RouterInput,
                     x: Mapping[str, float]) -> tuple[float, float, float, float]:
    """(implied-return term, risk term, trading cost, funding) in USD for perpetual notionals x."""
    keys = model.keys
    v = [inp.spot_usd.get(k, 0.0) if k.startswith("spot:") else x.get(k, 0.0) for k in keys]
    w = [inp.spot_usd.get(k, 0.0) if k.startswith("spot:") else inp.base_perp_usd.get(k, 0.0)
         for k in keys]
    n = len(keys)
    nav = inp.nav
    sw = [sum(model.cov_normal_h[a][b] * w[b] for b in range(n)) / nav for a in range(n)]
    mu = [cfg.gamma * sw[a] for a in range(n)]
    implied = sum(mu[a] * v[a] for a in range(n))
    if cfg.use_funding:
        implied += sum(inp.funding_normal.get(k, 0.0) * inp.settlements_ahead * v[a]
                       for a, k in enumerate(keys) if k.startswith("perp:"))
    risk = cfg.gamma / (2 * nav) * sum(v[a] * model.cov_h[a][b] * v[b]
                                       for a in range(n) for b in range(n))
    trade = 0.0
    if cfg.use_costs:
        for k in keys:
            if not k.startswith("perp:"):
                continue
            x0 = inp.perp_usd.get(k, 0.0)
            d = x.get(k, 0.0) - x0
            if d:
                trade += abs(d) * inp.costs[k].bps(d) / 1e4
            b = inp.base_perp_usd.get(k, 0.0)
            extra = abs(x.get(k, 0.0) - b) - abs(x0 - b)
            if extra > 0:
                trade += extra * inp.costs[k].bps(extra) / 1e4
    fund = 0.0
    if cfg.use_funding:
        fund = sum(inp.funding_now.get(k, 0.0) * inp.settlements_ahead * x.get(k, 0.0)
                   for k in keys if k.startswith("perp:"))
    return implied, risk, trade, fund


def _ce(model: RiskModel, cfg: RouterConfig, inp: RouterInput, x: Mapping[str, float]) -> float:
    implied, risk, trade, fund = _objective_parts(model, cfg, inp, x)
    return implied - risk - trade - fund


def route(model: RiskModel, inp: RouterInput, cfg: RouterConfig,
          *, tradeable: Sequence[str]) -> RouterDecision:
    """Choose perpetual notionals that maximise the certainty equivalent (see module docstring).

    Coordinate descent on a convex objective: each coordinate is a one-dimensional convex
    piecewise quadratic (quadratic risk, absolute-value trading and exit costs), minimised exactly
    by checking its kinks and the stationary point of each piece.
    """
    current = {k: inp.perp_usd.get(k, 0.0) for k in model.keys if k.startswith("perp:")}
    if not model.ok:
        return RouterDecision(dict(current), {}, 0.0, 0.0, 0.0, 0.0, 0.0, model.horizon_hours,
                              model.phase, True, model.reason, (), {})
    keys = model.keys
    idx = {k: a for a, k in enumerate(keys)}
    allowed = [k for k in tradeable if k in idx]
    if not cfg.cross_asset:
        held = {_underlying(k) for k, v in inp.spot_usd.items() if v} | {
            _underlying(k) for k, v in inp.base_perp_usd.items() if v}
        allowed = [k for k in allowed if _underlying(k) in held]
    nav = inp.nav
    g = cfg.gamma / (2 * nav)
    x = dict(current)
    spot = {k: inp.spot_usd.get(k, 0.0) for k in keys if k.startswith("spot:")}
    w = {k: (inp.spot_usd.get(k, 0.0) if k.startswith("spot:") else inp.base_perp_usd.get(k, 0.0))
         for k in keys}
    n = len(keys)
    mu = {k: cfg.gamma * sum(model.cov_normal_h[idx[k]][b] * w[keys[b]] for b in range(n)) / nav
          for k in keys}
    bound = cfg.max_abs_notional_nav * nav

    def vec(k: str) -> float:
        return spot[k] if k in spot else x.get(k, 0.0)

    # Hedge-only: the overlay on each perpetual may only move against the holder's book. The sign
    # of an instrument's covariance with the book (under this horizon's forecast) says which side
    # hedges; a gated or zero covariance allows no overlay at all. Without this bound the implied
    # returns make a calm forecast an invitation to add exposure — measured on the first run
    # (period B, 2026-09-25): an unbounded router levered the crypto sleeve up in quiet shut
    # sessions and raised the book's volatility from 23% to 29%, which is timing, not hedging.
    v_base = [w[k] for k in keys]
    grad = {k: sum(model.cov_h[idx[k]][b] * v_base[b] for b in range(n)) for k in allowed}
    limits: dict[str, tuple[float, float]] = {}
    for k in allowed:
        base = inp.base_perp_usd.get(k, 0.0)
        if grad[k] > 0:
            limits[k] = (-bound, base)
        elif grad[k] < 0:
            limits[k] = (base, bound)
        else:
            limits[k] = (base, base)
        lo_k, hi_k = limits[k]
        x[k] = min(max(x[k], lo_k), hi_k) if lo_k <= current[k] <= hi_k else current[k]

    fee_cache: dict[str, float] = {k: (inp.costs[k].bps(0.0) / 1e4 if cfg.use_costs else 0.0)
                                   for k in allowed}
    for _outer in range(3):
        for _sweep in range(60):
            moved = 0.0
            for k in allowed:
                a_i = idx[k]
                a = g * model.cov_h[a_i][a_i]
                if a <= 0:
                    continue
                cross = sum(model.cov_h[a_i][b] * vec(keys[b]) for b in range(n) if b != a_i)
                lin = -mu[k] + 2 * g * cross
                if cfg.use_funding:
                    lin += (inp.funding_now.get(k, 0.0)
                            - inp.funding_normal.get(k, 0.0)) * inp.settlements_ahead
                x0 = current[k]
                base = inp.base_perp_usd.get(k, 0.0)
                c = fee_cache[k]
                lo_k, hi_k = limits[k]
                lo_k, hi_k = min(lo_k, x0), max(hi_k, x0)
                new = _argmin_1d(a, lin, x0, base, c, c, bound, lo=lo_k, hi=hi_k)
                moved = max(moved, abs(new - x[k]))
                x[k] = new
            if moved < 1.0:
                break
        if cfg.use_costs:
            fee_cache = {k: inp.costs[k].bps(max(abs(x[k] - current[k]), 1.0)) / 1e4
                         for k in allowed}

    # The sweeps price trading at a constant marginal rate per coordinate, but the walked slippage
    # grows with size, so they can overshoot to a point whose true certainty equivalent is worse
    # than holding even though a smaller move in the same direction is better (found by the
    # constructed funding-spike case in `eval/xa_arena.py:adversarial_cases`, 2026-09-26: +$334 at
    # a -$75,000 move, yet the unsearched router returned no order). The true objective is concave
    # along the segment from the current book to the sweep's point (risk and walked cost are both
    # convex), so a golden-section search on that segment finds the best step exactly.
    x = _best_on_segment(model, cfg, inp, current, x)
    for k in allowed:
        if abs(x[k] - current[k]) < cfg.min_trade_usd:
            x[k] = current[k]

    implied0, risk0, _, fund0 = _objective_parts(model, cfg, inp, current)
    implied1, risk1, trade1, fund1 = _objective_parts(model, cfg, inp, x)
    gain = (implied1 - risk1 - trade1 - fund1) - (implied0 - risk0 - fund0)
    if gain <= 0:
        x = dict(current)
        implied1, risk1, trade1, fund1 = implied0, risk0, 0.0, fund0
        gain = 0.0
    orders = {k: x[k] - current[k] for k in x if abs(x[k] - current[k]) >= cfg.min_trade_usd}
    return RouterDecision(
        target_usd=x, orders_usd=orders, ce_gain_usd=gain, risk_before_usd=risk0,
        risk_after_usd=risk1, trade_cost_usd=trade1, funding_usd=fund1 - fund0,
        horizon_hours=model.horizon_hours, phase=model.phase, refused=False,
        reason="", gated_pairs=model.gated_pairs, regime=model.regime,
    )


def _best_on_segment(model: RiskModel, cfg: RouterConfig, inp: RouterInput,
                     start: Mapping[str, float], end: Mapping[str, float],
                     *, iterations: int = 48) -> dict[str, float]:
    """The point ``start + t (end - start)``, ``t`` in [0, 1], with the highest certainty
    equivalent under the full objective (golden-section search on a concave function)."""

    def at(t: float) -> dict[str, float]:
        return {k: start[k] + t * (end.get(k, start[k]) - start[k]) for k in start}

    def value(t: float) -> float:
        return _ce(model, cfg, inp, at(t))

    lo, hi = 0.0, 1.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = value(a), value(b)
    for _ in range(iterations):
        if fa >= fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = value(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = value(b)
    candidates = {0.0: value(0.0), 1.0: value(1.0), 0.5 * (lo + hi): value(0.5 * (lo + hi))}
    best = max(candidates, key=lambda t: candidates[t])
    return at(best)


def _argmin_1d(a: float, b: float, x0: float, base: float, c1: float, c2: float,
               bound: float, *, lo: float | None = None, hi: float | None = None) -> float:
    """argmin of ``a x^2 + b x + c1 |x - x0| + c2 max(0, |x - base| - |x0 - base|)``, a > 0,
    over ``[lo, hi]`` (default ``[-bound, bound]``). Convex, so the minimum is at a kink or at a
    piece's stationary point."""
    lo = -bound if lo is None else max(lo, -bound)
    hi = bound if hi is None else min(hi, bound)
    k = abs(x0 - base)
    kinks = sorted({x0, base - k, base + k, lo, hi})
    kinks = [p for p in kinks if lo <= p <= hi]

    def f(x: float) -> float:
        return a * x * x + b * x + c1 * abs(x - x0) + c2 * max(0.0, abs(x - base) - k)

    candidates = list(kinks)
    edges = [lo, *kinks, hi]
    for lo, hi in itertools.pairwise(edges):
        if hi <= lo:
            continue
        mid = 0.5 * (lo + hi)
        s1 = 1.0 if mid > x0 else -1.0
        s2 = 0.0
        if abs(mid - base) > k:
            s2 = 1.0 if mid > base else -1.0
        xs = -(b + c1 * s1 + c2 * s2) / (2 * a)
        if lo <= xs <= hi:
            candidates.append(xs)
    return min(candidates, key=f)


def settlements_between(start_ms: int, hours: int, *, every_hours: int = 8) -> int:
    """Funding settlements at 00:00/08:00/16:00 UTC falling in ``(start_ms, start_ms + hours]``."""
    first = (start_ms // (every_hours * HOUR_MS) + 1) * every_hours * HOUR_MS
    end = start_ms + hours * HOUR_MS
    return 0 if first > end else (end - first) // (every_hours * HOUR_MS) + 1


__all__ = [
    "OPEN", "PERP_OF_SPOT", "SHUT", "CostCurve", "RiskModel", "RouterConfig", "RouterDecision",
    "RouterInput", "estimate", "hours_to_boundary", "phase_calendar", "phase_of_bar", "route",
    "same_underlying", "settlements_between",
]
