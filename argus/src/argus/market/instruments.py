"""The Instrument Master — declared identity, not inferred behaviour.

Foundation 1 of the product architecture (`Activity/08_TRADING_OS_PLAN.md` §3). An external review
of this project named the exact failure this module closes:

    "Behavioral 'session attenuation' is not proof of legal instrument identity... a beta estimated
    in one regime is not a contract specification."

Before this module, the only thing anywhere in this codebase that knew TQQQUSDT tracks a 3x
leveraged fund was `desk/portfolio.py`'s own docstring, asserting it from a *measured* session beta
(2.905 open, 2.908 shut) — and `market/bitget.underlying_ticker` is pure string manipulation, `s[:
-len("USDT")]`, that would map TQQQUSDT to "TQQQ" with no idea what TQQQ *is*. A measured beta is
behaviour; it can drift, be estimated on a short window, or be wrong. A declared identity is a fact
about the instrument, sourced and dated, that behaviour is then checked *against*.

**Every fact below was verified against the issuer's own published fund page on 2026-09-15**, not
recalled from training data — see `source` and `source_fetched_at` on each entry. Where the
declared figure and this project's independently measured figure disagree, :func:`leverage_check`
reports the disagreement explicitly rather than silently preferring one number.

**What this module deliberately does not claim.** Bitget's rToken is a tracker on top of the
underlying (`market/bitget.py:100`, "the underlying equity this token tracks") — this module
declares the underlying's own structure (is TQQQ 3x the Nasdaq-100? is COIN a plain equity?), not
Bitget's own tracking mechanism, which is a separate and already-documented fact
(`execution/guard.py`'s `is_rwa` flag).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any


class InstrumentKind(StrEnum):
    """What kind of thing the underlying actually is. Not interchangeable, and conflating them
    is exactly the failure this module exists to prevent — an index ETF and a 3x leveraged ETF on
    the same index are structurally different instruments, not two tickers on one theme."""

    EQUITY = "equity"
    """A single company's common stock."""

    INDEX_ETF = "index_etf"
    """An unleveraged fund tracking a published index at ~1x."""

    LEVERAGED_ETF = "leveraged_etf"
    """A fund seeking a stated multiple (positive or negative) of an index's *daily* return."""


class InstrumentMasterError(ValueError):
    """Raised rather than guessing at an instrument's identity. See STANDING RULE #2."""


@dataclass(frozen=True, slots=True)
class InstrumentIdentity:
    """Declared facts about one underlying, sourced and dated — never inferred from price data."""

    symbol: str
    """The Bitget rToken symbol, e.g. ``TQQQUSDT``."""

    underlying: str
    """The ticker this rToken tracks, e.g. ``TQQQ``."""

    underlying_name: str
    issuer: str
    kind: InstrumentKind

    benchmark: str | None
    """The index this underlying is measured against. ``None`` for a single-company equity, which
    has no benchmark it is *defined* relative to — it has a beta to one, which is a different and
    measured (not declared) fact."""

    target_multiple: Decimal | None
    """The issuer's *stated* daily target, signed — ``Decimal("3")`` for TQQQ, ``Decimal("-3")``
    for SQQQ, ``None`` for anything not built to track a multiple of something else.

    This is a **daily** target. Every leveraged fund in this registry resets daily (`source` below
    quotes the issuer on this), which is the specific mechanical reason a multi-day *measured* beta
    is expected to differ from the *declared* multiple — path-dependent compounding, not error.
    """

    daily_reset: bool
    """Does the fund rebalance to its target multiple every trading day? Named explicitly because
    it is *why* declared and measured can honestly disagree over more than one day — a fact a
    reader needs before drawing a conclusion from any gap :func:`leverage_check` reports."""

    source: str
    source_fetched_at: date

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "underlying_name": self.underlying_name,
            "issuer": self.issuer,
            "kind": str(self.kind),
            "benchmark": self.benchmark,
            "target_multiple": None if self.target_multiple is None else str(self.target_multiple),
            "daily_reset": self.daily_reset,
            "source": self.source,
            "source_fetched_at": self.source_fetched_at.isoformat(),
        }


# --- the registry ---------------------------------------------------------------------------------
#
# One entry per symbol in `market.bitget.RTOKEN_SYMBOLS`. Every non-leveraged, single-company
# equity is declared with `benchmark=None, target_multiple=None` explicitly — omitting them would
# make "not leveraged" indistinguishable from "not yet looked up", and the whole point of a
# declared registry is that those two states must never be confused.

_TODAY = date(2026, 9, 15)

_EQUITIES: tuple[tuple[str, str, str], ...] = (
    # (rToken symbol, underlying ticker, company name)
    ("NVDAUSDT", "NVDA", "NVIDIA Corporation"),
    ("TSLAUSDT", "TSLA", "Tesla, Inc."),
    ("AAPLUSDT", "AAPL", "Apple Inc."),
    ("MSFTUSDT", "MSFT", "Microsoft Corporation"),
    ("METAUSDT", "META", "Meta Platforms, Inc."),
    ("GOOGLUSDT", "GOOGL", "Alphabet Inc."),
    ("AMZNUSDT", "AMZN", "Amazon.com, Inc."),
    ("COINUSDT", "COIN", "Coinbase Global, Inc."),
    ("MSTRUSDT", "MSTR", "Strategy Inc. (formerly MicroStrategy)"),
)

REGISTRY: dict[str, InstrumentIdentity] = {
    symbol: InstrumentIdentity(
        symbol=symbol,
        underlying=ticker,
        underlying_name=name,
        issuer=name,
        kind=InstrumentKind.EQUITY,
        benchmark=None,
        target_multiple=None,
        daily_reset=False,
        source="single-company common stock; no fund prospectus applies",
        source_fetched_at=_TODAY,
    )
    for symbol, ticker, name in _EQUITIES
}

REGISTRY["QQQUSDT"] = InstrumentIdentity(
    symbol="QQQUSDT",
    underlying="QQQ",
    underlying_name="Invesco QQQ Trust",
    issuer="Invesco Capital Management LLC",
    kind=InstrumentKind.INDEX_ETF,
    benchmark="Nasdaq-100 Index",
    target_multiple=Decimal("1"),
    daily_reset=False,
    source="https://www.invesco.com/qqq-etf/en/about.html — fetched and quoted verbatim: "
    '"tracks the Nasdaq-100 Index"; unleveraged, no daily-reset mechanism described',
    source_fetched_at=_TODAY,
)

REGISTRY["TQQQUSDT"] = InstrumentIdentity(
    symbol="TQQQUSDT",
    underlying="TQQQ",
    underlying_name="ProShares UltraPro QQQ",
    issuer="ProShares",
    kind=InstrumentKind.LEVERAGED_ETF,
    benchmark="Nasdaq-100 Index",
    target_multiple=Decimal("3"),
    daily_reset=True,
    source="https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq — fetched and quoted "
    'verbatim: "seeks daily investment results, before fees and expenses, that correspond to '
    'three times (3x) the daily performance of the Nasdaq-100 Index"',
    source_fetched_at=_TODAY,
)

REGISTRY["SQQQUSDT"] = InstrumentIdentity(
    symbol="SQQQUSDT",
    underlying="SQQQ",
    underlying_name="ProShares UltraPro Short QQQ",
    issuer="ProShares",
    kind=InstrumentKind.LEVERAGED_ETF,
    benchmark="Nasdaq-100 Index",
    target_multiple=Decimal("-3"),
    daily_reset=True,
    source="https://www.proshares.com/our-etfs/leveraged-and-inverse/sqqq — fetched and quoted "
    'verbatim: "seek daily investment results... that correspond to three times the inverse '
    '(-3x) of the daily performance" of the Nasdaq-100 Index; explicitly "a daily investment '
    'objective"',
    source_fetched_at=_TODAY,
)


def identity_of(symbol: str) -> InstrumentIdentity:
    """The declared identity for a symbol. Raises rather than guessing at an unregistered one."""
    try:
        return REGISTRY[symbol]
    except KeyError:
        raise InstrumentMasterError(
            f"{symbol!r} is not in the Instrument Master. A capability that needs an instrument's "
            f"identity and finds none here must say so, not assume it behaves like whatever it is "
            f"closest to."
        ) from None


@dataclass(frozen=True, slots=True)
class LeverageCheck:
    """A declared target compared against a measured figure for the same instrument."""

    symbol: str
    declared: Decimal
    measured: float
    difference: float
    daily_reset: bool

    @property
    def within_tolerance(self) -> bool:
        return self.difference <= LEVERAGE_TOLERANCE

    def render(self) -> str:
        note = (
            " (the fund resets daily, so multi-day compounding is the expected, honest cause "
            "of any gap — not evidence the target is wrong)"
            if self.daily_reset else ""
        )
        verdict = "within tolerance" if self.within_tolerance else "DISAGREES beyond tolerance"
        return (
            f"{self.symbol}: declared {self.declared:+} vs measured {self.measured:+.3f} "
            f"(|Δ|={self.difference:.3f}) — {verdict}{note}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "declared": str(self.declared),
            "measured": round(self.measured, 4),
            "difference": round(self.difference, 4),
            "within_tolerance": self.within_tolerance,
            "daily_reset": self.daily_reset,
        }


LEVERAGE_TOLERANCE = Decimal("0.15")
"""How far a measured beta may sit from the declared target before it is reported as a
disagreement rather than expected daily-reset drift.

Not tuned against a held-out set — stated as what it is. `desk/portfolio.py`'s own measurement
(2.905 to 2.908 against a declared 3, 2.885 to 2.910 against a declared -3) sits 0.09 to 0.115 from
target, so 0.15 accommodates the observed drift without being wide enough to accept a materially
wrong declaration silently.
"""


def leverage_check(symbol: str, measured_beta: float) -> LeverageCheck | None:
    """Compare a measured beta against the declared target multiple for a leveraged instrument.

    Returns ``None`` for an instrument with no declared leverage to check against — never a
    fabricated check against nothing. Raises via :func:`identity_of` for an unregistered symbol
    rather than silently skipping it.
    """
    identity = identity_of(symbol)
    if identity.target_multiple is None:
        return None
    declared = identity.target_multiple
    diff = abs(float(declared) - measured_beta)
    return LeverageCheck(
        symbol=symbol, declared=declared, measured=measured_beta, difference=diff,
        daily_reset=identity.daily_reset,
    )


def registry_as_dict() -> dict[str, dict[str, Any]]:
    """The whole registry, serialisable — for an artefact, not only for import-time use."""
    return {symbol: identity.as_dict() for symbol, identity in sorted(REGISTRY.items())}
