"""A pair trade on two named instruments, tested the way a quant desk tests one: is the spread
mean-reverting, what is the hedge ratio, did the relation hold on data the fit never saw, and what
would prove the trade wrong.

Round 43's brief: "I want to run a pair trade long NVDA short AMD. Is the spread mean-reverting,
and what is the hedge ratio?", then "Given that, how would I size it on a 100k account?", then
"What would prove this pair trade wrong?". No reader here took two instruments and a spread, so the
questions went to single-name readers; a pair is a different object (a regression, a stationarity
test on its residual, two legs to size). The follow-ups also say "it" and "this pair trade", so the
pair has to come from an earlier turn (``prior``).

**What was read before writing it** (this repository's own copy of each is cited, not memory):

- ``argus/research/cointegration.py`` is ARGUS's own pure-Python Engle-Granger, ADF, MacKinnon
  (2010) critical values and p-values and Ornstein-Uhlenbeck half-life, transcribed from
  statsmodels' ``tsa/stattools.py`` (``coint`` at 1702, ``adfuller`` at 289-357) and
  ``tsa/adfvalues.py`` and checked against statsmodels 0.14.6 to 1e-9 in ``tests/
  test_cointegration.py``. This module calls it rather than carrying a second copy, so the two
  can never disagree about what "cointegrated at 5%" means. statsmodels is installed here (0.15.0)
  but is a dev-only dependency (``pyproject.toml``: "never imported by argus's own src/ code"), so
  ``argus.research.cointegration.engle_granger`` is the production path and statsmodels is its
  oracle.
- ``research/repos-themed/tradermonty~claude-trading-skills/skills/pair-trade-screener/scripts/
  analyze_spread.py``: the hedge ratio is an OLS slope (``calculate_hedge_ratio``, line 125), the
  half-life comes from an AR(1) on the spread (``calculate_half_life``, line 171; it returns
  nothing when the coefficient is outside (0, 1)), and the z-score is a rolling window
  (``calculate_zscore_series``, line 188). Taken: all three ideas. Changed: this module regresses
  **log** prices, so the slope is an elasticity and the dollar hedge ratio is the slope itself,
  and it regresses on a training slice only (below).
- ``research/repos-themed/gbeced~basana/samples/strategies/pairs_trading.py:28`` and
  ``research/repos-t2/hkuds~vibe-trading/agent/src/quantlib/timeseries.py:181`` both reduce the
  test to ``statsmodels.tsa.stattools.coint(...)[1]`` on the whole sample, with the hedge ratio
  fitted on the same data the test then judges. Rejected: that is in-sample only, so it cannot say
  whether the relation survived.

**Method.** Daily closes through ``rule_test.daily_closes`` (Bitget's for a coin, Yahoo's
split-adjusted for a stock), on the days both have a close, the last two years of them. Then:

1. The first 70% of the days fit ``log Y = a + b log X`` (the long leg is ``Y``). ``b`` is the
   hedge ratio, with its OLS standard error, and Engle-Granger tests the residual (the N=2
   MacKinnon table, because the residual is fitted).
2. The last 30% are tested with that ``b`` and ``a`` **frozen**: an ADF on the held-out spread. A
   spread that is stationary only because it was fitted to be, fails here.
3. The half-life is ``ln 2 / lambda`` from the AR(1) regression of spread changes on the lagged
   spread, in each segment; no half-life is reported when the slope is not negative.
4. Hedge-ratio stability: ``b`` fitted on each half of the window; "stable" means the two are
   within 25% of each other (a stated rule of thumb, not a test).
5. The z-score is the current spread against the whole window's mean and standard deviation and,
   separately, against the last 60 days.
6. Return correlation is the daily log-return correlation, over the window and the last 60 days.

The OLS standard error assumes stationary errors; for two trending price series it is optimistic,
and the answer says so. The stability line is the honest uncertainty on the ratio.

**Sizing.** The long leg is ``L`` and the short leg ``b * L`` (a log-log slope is a dollar
elasticity), so the hedge-ratio-neutral version holds ``A / (1 + b)`` long and ``b`` times that
short on gross exposure ``A``; the dollar-neutral version holds ``A / 2`` each and carries a net
exposure of ``(b - 1) * A / 2`` to the common driver, which the answer states. One unit of spread
standard deviation costs ``L * sd`` dollars on the hedge-ratio version. The stop is a z-score level
one standard deviation beyond the entry (never closer than 3), with the dollar loss at that level.
Costs: four legs at Bitget's 6 bp taker fee (``argus.research.cointegration.TAKER_BPS_PER_LEG``).

**Checked live on 2026-10-06** (``rule_test.daily_closes``: Yahoo for NVDA and AMD, Bitget for ETH
and SOL). NVDA on AMD, 500 shared days (7 Oct 2024 to 5 Oct 2026): first-70% hedge ratio 0.57,
Engle-Granger -4.09 against -3.35 (p = 0.005), then -2.22 against -2.88 out of sample (p = 0.199),
so "cointegrated in sample, broke out of sample"; the ratio by half was 0.74 then 0.18, the
whole-window ratio 0.35 and z-score -0.3. ETH on SOL, 731 days: Engle-Granger -1.31 (p = 0.83),
"not mean-reverting", ratio 0.61, half-life 101 days. The first live read of a pair takes ~20 s
(Bitget's daily history is paged); the statistics themselves take ~0.1 s.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from typing import Final

from argus.research.cointegration import (
    LEGS_PER_ROUND_TRIP,
    MIN_OBSERVATIONS,
    TAKER_BPS_PER_LEG,
    ADFResult,
    CointegrationError,
    CointegrationResult,
    adf,
    engle_granger,
    half_life,
    ols,
    zscores,
)

PAIR: Final = re.compile(
    r"\bpairs?[\s-]+trad\w*|\bpairs?[\s-]+(?:strategy|position|spread|setup|trade)\b|"
    r"\bcointegrat\w*|\bstat(?:istical)?[\s-]+arb\w*|"
    r"\b(?:long|short)\s+[A-Za-z$][\w.]{0,11}\s*(?:,|\band\b|/)?\s*(?:short|long)\s+[A-Za-z$]"
    r"[\w.]{0,11}[^.?]{0,80}\b(?:spread|hedge\s+ratio|mean[\s-]*revert\w*|cointegrat\w*)", re.I)
"""A question that names a pair trade, or a long/short of two names with a spread cue."""
SIZE: Final = re.compile(
    r"\b(?:siz(?:e|ed|ing)|how\s+(?:much|many)|legs?|position\s+size|notional|dollar[\s-]neutral|"
    r"allocate|put\s+on|account)\b", re.I)
"""A request for leg sizes."""
WRONG: Final = re.compile(
    r"\b(?:prove[sd]?|disprove[sd]?|invalidat\w*|falsif\w*|wrong|break(?:s|ing)?\s+down|"
    r"stop[\s-]*(?:loss|out|level)?|get\s+out|bail|exit\s+(?:rule|level|if)|what\s+(?:would|could)"
    r"\s+(?:kill|end)|when\s+(?:do|would|should)\s+i\s+(?:exit|quit|close))\b", re.I)
"""A request for what would show the trade wrong."""
_LONG: Final = re.compile(r"\blong\s+(?:the\s+)?\$?(?P<t>[A-Za-z][\w.]{0,11})", re.I)
_SHORT: Final = re.compile(r"\bshort\s+(?:the\s+)?\$?(?P<t>[A-Za-z][\w.]{0,11})", re.I)

DAYS: Final = 730
"""The window of shared days kept: the last two years."""
MIN_DAYS: Final = 150
"""Fewer shared days than this and a 70/30 split leaves too little to test."""
TRAIN: Final = 0.7
RECENT: Final = 60
STABLE_WITHIN: Final = 0.25
ENTRY_Z: Final = 2.0
STOP_FLOOR_Z: Final = 3.0
ROLL_DAYS: Final = 252
BREAK_P: Final = 0.10
SLOW_HALF_LIFE: Final = 60.0
DEFAULT_ACCOUNT: Final = 10_000.0


class PairError(Exception):
    """The pair cannot be studied from the data on hand; the message is the reason, in words."""


@dataclass(frozen=True)
class Pair:
    long: str
    short: str
    stated: bool
    """True when the question said which leg is long ("long NVDA short AMD")."""


@dataclass(frozen=True)
class Study:
    pair: Pair
    days: list[date]
    ly: list[float]
    lx: list[float]
    price_y: float
    price_x: float
    said_y: str
    said_x: str
    split: int
    train: CointegrationResult
    train_se: float
    held_out: ADFResult | None
    life_train: float | None
    life_held: float | None
    half_a: tuple[float, float]
    half_b: tuple[float, float]
    beta: float
    beta_se: float
    spread: list[float]
    spread_sd: float
    z_whole: float
    z_recent: float | None
    corr_all: float
    corr_recent: float | None
    rolling_p: float | None
    rolling_beta: float | None
    sd_recent: float | None


def _symbols(text: str) -> list[str]:
    from argus.lui.research.parse import research_symbols

    found = list(research_symbols(text)[0])
    return list(dict.fromkeys(found))


def _resolve(token: str) -> str | None:
    from argus.lui.research.parse import resolve_name

    hit = resolve_name(token)
    return hit[0] if hit else None


def read_pair(text: str) -> Pair | None:
    """The two instruments a question names, the long leg first, or None."""
    symbols = _symbols(text)
    if len(symbols) != 2:
        return None
    long_leg = next((s for m in _LONG.finditer(text) if (s := _resolve(m.group("t"))) in symbols),
                    None)
    short_leg = next((s for m in _SHORT.finditer(text)
                      if (s := _resolve(m.group("t"))) in symbols), None)
    if long_leg and short_leg and long_leg != short_leg:
        return Pair(long_leg, short_leg, True)
    if long_leg:
        return Pair(long_leg, next(s for s in symbols if s != long_leg), True)
    if short_leg:
        return Pair(next(s for s in symbols if s != short_leg), short_leg, True)
    return Pair(symbols[0], symbols[1], False)


def _pair_from(text: str, prior: Sequence[str]) -> Pair | None:
    own = read_pair(text)
    if own is not None:
        return own
    for turn in reversed(list(prior)[-8:]):
        if PAIR.search(turn):
            found = read_pair(turn)
            if found is not None:
                return found
    return None


def asked(text: str, prior: Sequence[str] = ()) -> bool:
    """Whether :func:`lines` answers this question: a pair trade named in it, or a sizing or
    "what would prove it wrong" follow-up to a pair trade in the last turns that names no other
    instrument."""
    if PAIR.search(text):
        return _pair_from(text, prior) is not None
    if not (SIZE.search(text) or WRONG.search(text)):
        return False
    if _symbols(text):
        return False
    return any(PAIR.search(t) and read_pair(t) is not None for t in list(prior)[-8:])


def account_in(text: str, prior: Sequence[str]) -> float | None:
    """The account size the question or an earlier turn states, in dollars."""
    from argus.lui.research.sizing import account_of, stated_capital

    for turn in (text, *reversed(list(prior)[-8:])):
        found = account_of(turn) or stated_capital(turn)
        if found:
            return found
    return None


def _day(moment: datetime) -> date:
    return moment.date()


def _aligned(first: str, second: str) -> tuple[list[date], list[float], list[float], str, str]:
    from argus.lui.research import rule_test

    t1, c1, said1 = rule_test.daily_closes(first)
    t2, c2, said2 = rule_test.daily_closes(second)
    left = {_day(t): c for t, c in zip(t1, c1, strict=True) if c > 0}
    right = {_day(t): c for t, c in zip(t2, c2, strict=True) if c > 0}
    shared = sorted(left.keys() & right.keys())
    if shared:
        shared = [d for d in shared if (shared[-1] - d).days <= DAYS]
    return shared, [left[d] for d in shared], [right[d] for d in shared], said1, said2


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0


def _returns(logs: Sequence[float]) -> list[float]:
    return [b - a for a, b in pairwise(logs)]


def _fit(ly: Sequence[float], lx: Sequence[float]) -> tuple[float, float, float]:
    fit = ols(ly, [lx, [1.0] * len(lx)])
    return fit.params[0], fit.stderr[0], fit.params[1]


def study(pair: Pair) -> Study:
    """Every figure the answers use, or :class:`PairError` saying why there are none."""
    days, cy, cx, said_y, said_x = _aligned(pair.long, pair.short)
    if len(days) < MIN_DAYS:
        raise PairError(f"{_name(pair.long)} and {_name(pair.short)} share only {len(days)} daily "
                        f"closes, under the {MIN_DAYS} needed to fit and test a pair")
    ly = [math.log(v) for v in cy]
    lx = [math.log(v) for v in cx]
    n = len(days)
    split = int(n * TRAIN)
    try:
        train = engle_granger(ly[:split], lx[:split])
    except CointegrationError as exc:
        raise PairError(f"the pair cannot be tested ({exc})") from exc
    _, train_se, _ = _fit(ly[:split], lx[:split])
    frozen = [y - train.hedge_ratio * x - train.intercept for y, x in zip(ly, lx, strict=True)]
    held = frozen[split:]
    held_out = adf(held, regression="c") if len(held) >= MIN_OBSERVATIONS else None
    life_train = half_life(frozen[:split])
    life_held = half_life(held) if len(held) >= MIN_OBSERVATIONS else None
    mid = n // 2
    ba, sa, _ = _fit(ly[:mid], lx[:mid])
    bb, sb, _ = _fit(ly[mid:], lx[mid:])
    beta, beta_se, alpha = _fit(ly, lx)
    spread = [y - beta * x - alpha for y, x in zip(ly, lx, strict=True)]
    mean = sum(spread) / n
    sd = math.sqrt(sum((s - mean) ** 2 for s in spread) / n)
    z_whole = (spread[-1] - mean) / sd if sd > 0 else 0.0
    z_recent = zscores(spread, lookback=RECENT)[-1]
    ry, rx = _returns(ly), _returns(lx)
    corr_recent = _pearson(ry[-RECENT:], rx[-RECENT:]) if len(ry) > RECENT else None
    rolling_p: float | None = None
    rolling_beta: float | None = None
    if n >= ROLL_DAYS:
        try:
            rolling_p = engle_granger(ly[-ROLL_DAYS:], lx[-ROLL_DAYS:]).pvalue
            rolling_beta = _fit(ly[-ROLL_DAYS:], lx[-ROLL_DAYS:])[0]
        except CointegrationError:
            rolling_p = None
    recent = spread[-RECENT:]
    sd_recent: float | None = None
    if len(recent) == RECENT:
        mr = sum(recent) / RECENT
        sd_recent = math.sqrt(sum((s - mr) ** 2 for s in recent) / RECENT)
    return Study(
        pair=pair, days=days, ly=ly, lx=lx, price_y=cy[-1], price_x=cx[-1], said_y=said_y,
        said_x=said_x, split=split, train=train, train_se=train_se, held_out=held_out,
        life_train=life_train, life_held=life_held, half_a=(ba, sa), half_b=(bb, sb), beta=beta,
        beta_se=beta_se, spread=spread, spread_sd=sd, z_whole=z_whole, z_recent=z_recent,
        corr_all=_pearson(ry, rx), corr_recent=corr_recent, rolling_p=rolling_p,
        rolling_beta=rolling_beta, sd_recent=sd_recent)


def _name(symbol: str) -> str:
    return symbol.removesuffix("USDT").removesuffix("STOCK")


def _usd(value: float) -> str:
    return f"${value:,.0f}"


def _span(a: date, b: date) -> str:
    return f"{a:%d %b %Y} to {b:%d %b %Y}"


def stable(s: Study) -> tuple[bool, float]:
    """Whether the two halves' hedge ratios agree within :data:`STABLE_WITHIN`, and the gap as a
    share of their mean."""
    low, high = s.half_a[0], s.half_b[0]
    mean = (abs(low) + abs(high)) / 2
    gap = abs(low - high) / mean if mean > 0 else math.inf
    return gap <= STABLE_WITHIN, gap


def verdict(s: Study) -> str:
    """A plain verdict on the pair."""
    ys, xs = _name(s.pair.long), _name(s.pair.short)
    in_sample = s.train.cointegrated_at_5pct
    held = s.held_out is not None and s.held_out.stationary_at_5pct
    if in_sample and held:
        return (f"cointegrated at 5% in the first {TRAIN:.0%} of the window and the relation held "
                f"out of sample, so the {ys}/{xs} spread has been mean-reverting")
    if in_sample:
        return (f"cointegrated at 5% in the first {TRAIN:.0%} of the window but the relation "
                f"broke out of sample, so the spread cannot be relied on to revert")
    return (f"not mean-reverting: the {ys}/{xs} spread is not cointegrated at 5% in the first "
            f"{TRAIN:.0%} of the window, so a pair trade here is two directional bets")


def _life(value: float | None) -> str:
    return "none (the spread does not revert)" if value is None else f"{value:.0f} days"


def _side(s: Study) -> tuple[str, str, bool]:
    """(long leg, short leg, whether the spread, long Y short X, is the side held)."""
    if s.pair.stated or s.z_whole <= 0:
        return s.pair.long, s.pair.short, True
    return s.pair.short, s.pair.long, False


def _legs(s: Study, account: float) -> tuple[float, float, float, float]:
    """(hedge-ratio long dollars, its short dollars, dollar-neutral dollars each, the net factor
    exposure of the dollar-neutral version)."""
    beta = max(s.beta, 0.0)
    first = account / (1 + beta)
    return first, beta * first, account / 2, (beta - 1) * account / 2


def sizing_lines(s: Study, account: float, assumed: bool) -> list[str]:
    ys, xs = _name(s.pair.long), _name(s.pair.short)
    out: list[str] = []
    if s.beta <= 0.05:
        return [f"The hedge ratio is {s.beta:.2f}, so {xs} does not hedge {ys}: there is no "
                f"hedge-ratio-neutral size, and holding both is two bets, not a pair."]
    first, second, each, net = _legs(s, account)
    held_long, held_short, as_fitted = _side(s)
    flip = "" if as_fitted else (f" Today's z-score is {s.z_whole:+.1f} (the spread is rich), so "
                                 f"the side that pays on reversion is long {_name(held_long)}, "
                                 f"short {_name(held_short)}; legs below are swapped.")
    la, sa_ = (ys, xs) if as_fitted else (xs, ys)
    long_dollars, short_dollars = (first, second) if as_fitted else (second, first)
    out.append(
        f"Hedge-ratio neutral on {_usd(account)} gross: long {_usd(first if as_fitted else second)}"
        f" {la} and short {_usd(second if as_fitted else first)} {sa_}"
        f" (about {_shares(s, la, long_dollars)} and {_shares(s, sa_, short_dollars)}), since "
        f"each $1 of {ys} is hedged by ${s.beta:.2f} of {xs}.{flip}")
    direction = "long" if net > 0 else "short"
    out.append(f"Dollar neutral: {_usd(each)} each side. That leaves a net {direction} tilt of "
               f"about {_usd(abs(net))} to whatever moves both names, because the hedge ratio is "
               f"{s.beta:.2f}, not 1.")
    out.extend(_stop_lines(s, first, account))
    cost = 2 * TAKER_BPS_PER_LEG / 10_000 * account
    out.append(f"Costs: {LEGS_PER_ROUND_TRIP} legs (open and close both) at a "
               f"{TAKER_BPS_PER_LEG:.0f} bp taker fee come to about {_usd(cost)} for the round "
               f"trip on {_usd(account)} gross, before funding.")
    if assumed:
        out.append(f"No account size was stated, so the legs are shown on {_usd(account)} gross; "
                   "they scale in proportion.")
    return out


def _shares(s: Study, name: str, dollars: float) -> str:
    price = s.price_y if name == _name(s.pair.long) else s.price_x
    return f"{dollars / price:,.2f} units at {price:,.2f}"


def _levels(s: Study) -> tuple[int, float, float]:
    """(the sign of z at which the held side is entered, the entry z-score, the stop z-score).

    The entry is today's z-score when it already favours the held side by at least one standard
    deviation, else the usual two; the stop is one standard deviation beyond it, never nearer
    than :data:`STOP_FLOOR_Z`."""
    sign = -1 if (s.pair.stated or s.z_whole <= 0) else 1
    on_side = (s.z_whole < 0) == (sign < 0)
    entry = s.z_whole if on_side and abs(s.z_whole) >= 1.0 else sign * ENTRY_Z
    return sign, entry, sign * max(STOP_FLOOR_Z, abs(entry) + 1.0)


def _stop_lines(s: Study, long_dollars: float, account: float) -> list[str]:
    _, entry, stop = _levels(s)
    loss = long_dollars * s.spread_sd * (abs(stop) - abs(entry))
    reached = "now" if entry == s.z_whole else "when it is reached"
    return [f"Stop: exit if the spread z-score reaches {stop:+.1f} (entry {entry:+.1f}, "
            f"{reached}); one spread standard deviation is {s.spread_sd:.1%} of the long leg, so "
            f"that stop costs about {_usd(loss)}, or {loss / account:.1%} of {_usd(account)}."]


def falsifiers(s: Study) -> list[str]:
    """What would prove the trade wrong, each with today's reading."""
    ys, xs = _name(s.pair.long), _name(s.pair.short)
    out: list[str] = []
    _, _, stop = _levels(s)
    out.append(f"The spread z-score goes beyond {stop:+.1f} against the position (today "
               f"{s.z_whole:+.1f}): a spread that keeps widening is not reverting.")
    low, high = s.beta * (1 - STABLE_WITHIN), s.beta * (1 + STABLE_WITHIN)
    now = "no rolling fit" if s.rolling_beta is None else f"{s.rolling_beta:.2f} over the last year"
    out.append(f"The hedge ratio drifts outside {low:.2f} to {high:.2f} (fitted {s.beta:.2f}; "
               f"now {now}): the legs no longer offset and the pair becomes a directional bet.")
    p_now = "not computable" if s.rolling_p is None else f"{s.rolling_p:.3f}"
    out.append(f"The Engle-Granger p-value on a rolling {ROLL_DAYS}-day window rises above "
               f"{BREAK_P:.2f} (now {p_now}): the evidence of cointegration has gone.")
    sd_now = (f"{s.sd_recent / s.spread_sd:.1f}x" if s.sd_recent is not None and s.spread_sd > 0
              else "not computable")
    corr = "not computable" if s.corr_recent is None else f"{s.corr_recent:.2f}"
    out.append(f"A regime break: the spread's 60-day standard deviation passes twice its "
               f"window level (now {sd_now}) or the 60-day return correlation falls below half of "
               f"its {s.corr_all:.2f} window level (now {corr}); news that changes one of "
               f"{ys} or {xs} alone (earnings, a ban, a listing) is such a break.")
    return out


def _tripped(s: Study) -> list[str]:
    hits: list[str] = []
    if s.rolling_p is not None and s.rolling_p > BREAK_P:
        hits.append(f"the rolling {ROLL_DAYS}-day Engle-Granger p-value is already "
                    f"{s.rolling_p:.2f}")
    if s.rolling_beta is not None and abs(s.rolling_beta - s.beta) > STABLE_WITHIN * abs(s.beta):
        hits.append(f"the last year's hedge ratio ({s.rolling_beta:.2f}) is already outside "
                    f"{STABLE_WITHIN:.0%} of the fitted {s.beta:.2f}")
    if s.sd_recent is not None and s.spread_sd > 0 and s.sd_recent > 2 * s.spread_sd:
        hits.append("the spread's recent volatility is already over twice its window level")
    if not (s.train.cointegrated_at_5pct and s.held_out is not None
            and s.held_out.stationary_at_5pct):
        hits.append("the study itself does not show a relation that held out of sample")
    if abs(s.z_whole) >= STOP_FLOOR_Z:
        hits.append(f"the z-score is already {s.z_whole:+.1f}")
    return hits


def _data_line(s: Study) -> str:
    return (f"Data: daily closes, {s.pair.long} = {s.said_y}; {s.pair.short} = {s.said_x}; "
            f"{len(s.days)} shared days, {_span(s.days[0], s.days[-1])}; Engle-Granger and ADF "
            f"with MacKinnon (2010) critical values (argus.research.cointegration, checked "
            f"against statsmodels); fees at Bitget's taker rate. Not advice.")


def _body(s: Study) -> list[str]:
    ys, xs = _name(s.pair.long), _name(s.pair.short)
    t = s.train
    train_span = _span(s.days[0], s.days[s.split - 1])
    out = [
        f"In sample ({train_span}, {s.split} days): hedge ratio {s.train.hedge_ratio:.2f} "
        f"(+/- {s.train_se:.2f} OLS standard error) from log {ys} on log {xs}; "
        f"Engle-Granger statistic {t.statistic:.2f} against a 5% critical value "
        f"{t.critical_values['5%']:.2f} (p = {t.pvalue:.3f}); half-life {_life(s.life_train)}."]
    if s.held_out is None:
        out.append("Out of sample: too few days left after the split to test the frozen ratio.")
    else:
        o = s.held_out
        out.append(
            f"Out of sample ({_span(s.days[s.split], s.days[-1])}, {len(s.days) - s.split} days, "
            f"hedge ratio frozen): ADF statistic {o.statistic:.2f} against a 5% critical value "
            f"{o.critical_values['5%']:.2f} (p = {o.pvalue:.3f}); half-life "
            f"{_life(s.life_held)}.")
    ok, gap = stable(s)
    out.append(
        f"Hedge ratio by half: {s.half_a[0]:.2f} (+/- {s.half_a[1]:.2f}) then "
        f"{s.half_b[0]:.2f} (+/- {s.half_b[1]:.2f}), a {gap:.0%} gap: "
        f"{'stable' if ok else 'not stable'} (stable means within {STABLE_WITHIN:.0%}). The "
        f"standard errors assume stationary errors and are optimistic for trending prices; the gap "
        f"is the honest uncertainty.")
    out.append(
        f"Current spread (whole-window ratio {s.beta:.2f}): z-score {s.z_whole:+.1f} against the "
        f"whole window"
        + ("" if s.z_recent is None else f" and {s.z_recent:+.1f} against the last {RECENT} days")
        + f"; return correlation {s.corr_all:.2f} over the window"
        + ("" if s.corr_recent is None else f", {s.corr_recent:.2f} over the last {RECENT} days")
        + ".")
    if s.life_train is not None and s.life_train > SLOW_HALF_LIFE:
        out.append(f"A half-life of {s.life_train:.0f} days is slow: a holding period that long "
                   f"carries funding and event risk the spread may never repay.")
    if not t.cointegrated_at_5pct:
        out.append("The test has little power on two years of daily data, so a miss does not prove "
                   "there is no relation; it means this window does not show one.")
    return out


STRONG_WRONG: Final = re.compile(r"\b(?:prove[sd]?|disprove[sd]?|invalidat\w*|falsif\w*|wrong|"
                                 r"break(?:s|ing)?\s+down|kill|end)\b", re.I)


def _mode(text: str) -> str:
    """"wrong" for what would prove it wrong, "size" for leg sizes, else "study"."""
    if STRONG_WRONG.search(text) and WRONG.search(text):
        return "wrong"
    if SIZE.search(text):
        return "size"
    return "wrong" if WRONG.search(text) else "study"


def _answer(s: Study, text: str, prior: Sequence[str]) -> list[str]:
    ys, xs = _name(s.pair.long), _name(s.pair.short)
    account = account_in(text, prior)
    mode = _mode(text)
    if mode == "wrong":
        tripped = _tripped(s)
        lead = (f"Bottom line: the {ys}/{xs} pair trade is wrong if the z-score reaches "
                f"{_levels(s)[2]:+.1f}, the hedge ratio leaves {s.beta * 0.75:.2f} to "
                f"{s.beta * 1.25:.2f}, the rolling Engle-Granger p-value passes {BREAK_P:.2f}, or "
                f"the spread breaks regime.")
        out = [lead, *falsifiers(s)]
        out.append("Already tripped today: " + "; ".join(tripped) + "." if tripped
                   else "None of these is tripped today.")
        out.append(f"The study behind it: {verdict(s)}.")
        out.append(_data_line(s))
        return out
    if mode == "size":
        base = account if account is not None else DEFAULT_ACCOUNT
        first, second, each, _ = _legs(s, base)
        lead = (f"Bottom line: on {_usd(base)} gross, hedge-ratio neutral is {_usd(first)} "
                f"{ys} against {_usd(second)} {xs} (hedge ratio {s.beta:.2f}); dollar neutral is "
                f"{_usd(each)} each side"
                + ("" if s.beta > 0.05 else ", but the hedge ratio is too low to hedge at all")
                + ".")
        out = [lead, *sizing_lines(s, base, account is None)]
        out.append(f"Whether it is worth putting on: {verdict(s)}.")
        out.append(_data_line(s))
        return out
    hl = _life(s.life_train)
    lead = (f"Bottom line: {verdict(s)}; hedge ratio {s.beta:.2f} (+/- {s.beta_se:.2f}) of log "
            f"{ys} on log {xs} over the whole window, z-score now {s.z_whole:+.1f}, "
            f"half-life {hl} in the fitting period.")
    out = [lead, *_body(s)]
    if account is not None:
        out.extend(sizing_lines(s, account, False))
    out.append(_data_line(s))
    return out


def lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """The pair-trade answer, or None when the question is not about a pair trade this reader
    can place (two named instruments here or in an earlier turn)."""
    if not asked(text, prior):
        return None
    pair = _pair_from(text, prior)
    if pair is None:
        return None
    try:
        s = study(pair)
    except PairError as exc:
        return [f"Bottom line: the {_name(pair.long)}/{_name(pair.short)} pair could not be "
                f"studied just now: {exc}.", "Data: daily closes via Bitget and Yahoo Finance. "
                "Not advice."]
    except Exception:
        return [f"Bottom line: the daily history for {_name(pair.long)} and "
                f"{_name(pair.short)} could not be read just now, so the pair cannot be tested; "
                f"ask again in a minute.", "Data: daily closes via Bitget and Yahoo Finance. "
                "Not advice."]
    return _answer(s, text, prior)
