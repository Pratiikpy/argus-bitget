"""Position size from a risk budget — "a $50,000 account, risk at most 2% with a 4% stop".

A judge asked exactly that and was told how many times the desk had abstained (a judge's audit,
2026-09-29): the ledger classifier read "risk" and "per trade" as a question about the record. It is
the most common sizing question a trader asks and it has one honest answer, the fixed-fractional
rule every trading text states (van Tharp's "percent risk" model; Elder's 2% rule):

    money at risk   = account x risk fraction
    position value  = money at risk / stop distance (as a fraction of the entry)

Two things the textbook formula leaves out, and this answer does not:

* **Costs.** A stopped-out trade pays the round trip as well as the stop distance. At Bitget's
  measured 0.06% taker per side (`cost/model.py`, ``CostModel.bitget_perp``) the loss on a stop is
  ``position x (stop + 0.12%)``, so the size that keeps the loss inside the budget is
  ``risk / (stop + round trip)``. The answer gives both sizes and says which one holds the cap.
* **Gaps.** A stop is an order, not a guarantee: a stock perpetual that gaps through it over a
  weekend or an earnings print fills worse. When a name is given, its worst observed 24 hours is
  set beside the stop so the reader sees whether the stop has ever been jumped.

Leverage is stated when the position is larger than the account; the answer never recommends any.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from argus.cost.model import CostModel
from argus.lui.answer import Source
from argus.lui.trace import trace_module

SIZING_Q = re.compile(
    r"\bposition\s+siz\w*|\bhow\s+(?:big|large|much|many\s+(?:shares|units|coins|contracts))\b"
    r"[^?]{0,40}?\b(?:position|trade|buy|size|should\s+i\s+(?:buy|hold|trade))|\bsize\s+(?:my|the|a|"
    r"this)\s+(?:position|trade)|\bhow\s+much\s+(?:should|can)\s+i\s+(?:buy|put\s+on|trade)", re.I)
"""A question asking for a size."""

_ACCOUNT = (
    re.compile(r"\$\s*(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k|m|thousand|million)?\s*(?:dollars?\s+|usd\s+)?"
               r"(?:account|capital|portfolio|balance|bankroll)", re.I),
    re.compile(r"\b(?:account|capital|portfolio|balance|bankroll)\s+(?:of|is|=|:|size\s+(?:of|is))?\s*"
               r"\$?\s*(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k|m|thousand|million)?", re.I),
    re.compile(r"\b(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k|m|thousand|million)?\s+(?:dollars?\s+|usd\s+|"
               r"usdt\s+)?(?:account|capital|portfolio|balance|bankroll)", re.I),
)
"""The account size, written "$50,000 account", "account of 50k" or "50k account"."""
_RISK_PCT = re.compile(
    r"\brisk(?:ing)?\s+(?:at\s+most\s+|max(?:imum)?\s+|no\s+more\s+than\s+|up\s+to\s+|only\s+)?"
    r"(?P<p>\d+(?:\.\d+)?)\s*%|(?P<p2>\d+(?:\.\d+)?)\s*%\s+(?:risk|of\s+(?:my\s+)?(?:account|"
    r"capital)\s+(?:per|a|each)\s+trade)", re.I)
_RISK_USD = re.compile(
    r"\brisk(?:ing)?\s+(?:at\s+most\s+|max(?:imum)?\s+|no\s+more\s+than\s+|up\s+to\s+|only\s+)?"
    r"\$\s*(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k)?", re.I)
_STOP_PCT = re.compile(
    r"(?P<p>\d+(?:\.\d+)?)\s*%\s*(?:stop|stop[-\s]loss|sl)\b|\b(?:stop|stop[-\s]loss|sl)\s+(?:of|at|"
    r"is|=)?\s*(?P<p2>\d+(?:\.\d+)?)\s*%", re.I)
_STOP_PRICE = re.compile(
    r"\b(?:stop|stop[-\s]loss|sl)\s+(?:at|of|is|=)?\s*\$?\s*"
    r"(?P<v>\d[\d,]*(?:\.\d+)?)(?!\s*%)", re.I)
_ENTRY_PRICE = re.compile(
    r"\b(?:entry|enter|entering|buy(?:ing)?|in)\s+(?:at|@|is|=)?\s*\$?\s*(?P<v>\d[\d,]*(?:\.\d+)?)"
    r"(?!\s*%)|@\s*\$?\s*(?P<v2>\d[\d,]*(?:\.\d+)?)", re.I)

ROUND_TRIP = float(CostModel.bitget_perp().taker_bps * Decimal(2)) / 10_000
"""The taker round trip a stopped trade pays: 0.06% a side on Bitget's perpetuals."""


def _value(n: str, k: str | None) -> float:
    base = float(n.replace(",", ""))
    scale = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get((k or "").lower(), 1.0)
    return base * scale


def _group(m: re.Match[str], *names: str) -> str | None:
    for name in names:
        value = m.groupdict().get(name)
        if value:
            return value
    return None


def account_of(text: str) -> float | None:
    """The account size the question states, in dollars, or None."""
    for pattern in _ACCOUNT:
        m = pattern.search(text)
        if m is not None:
            return _value(m.group("n"), m.group("k"))
    return None


def asks_for_size(text: str) -> bool:
    """A size is asked for and there is a risk budget and a stop to size it from."""
    return bool(SIZING_Q.search(text) and (_RISK_PCT.search(text) or _RISK_USD.search(text))
                and (_STOP_PCT.search(text) or _STOP_PRICE.search(text)))


def answer(text: str, symbol: str | None = None, *,
           price: Callable[[str], float] | None = None,
           worst_day: Callable[[str], float | None] | None = None,
           ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The fixed-fractional size, net of costs, in dollars and (for a named name) in units."""
    notes: list[str] = []
    account = account_of(text)
    risk_usd: float | None = None
    if (m := _RISK_USD.search(text)) is not None:
        risk_usd = _value(m.group("n"), m.group("k"))
    elif (m := _RISK_PCT.search(text)) is not None:
        fraction = float(_group(m, "p", "p2") or 0) / 100
        if account is None:
            account = 10_000.0
            notes.append("no account size was given, so $10,000 is worked — the sizes scale "
                         "with it")
        risk_usd = account * fraction
    assert risk_usd is not None  # asks_for_size() guaranteed a risk budget

    entry: float | None = None
    if (m := _ENTRY_PRICE.search(text)) is not None:
        entry = float((_group(m, "v", "v2") or "0").replace(",", ""))
    if entry is None and symbol is not None and price is not None:
        try:
            entry = price(symbol)
            notes.append(f"entry read as {symbol.removesuffix('USDT')}'s last Bitget price, "
                         f"{entry:,.2f}")
        except Exception:  # a missing quote only drops the unit count
            entry = None
    stop_frac: float | None = None
    if (m := _STOP_PCT.search(text)) is not None:
        stop_frac = float(_group(m, "p", "p2") or 0) / 100
    elif (m := _STOP_PRICE.search(text)) is not None and entry:
        stop_frac = abs(entry - float(m.group("v").replace(",", ""))) / entry
    if not stop_frac or stop_frac <= 0 or stop_frac >= 1:
        lines = ["Bottom line: a stop price needs an entry price to become a distance — say "
                 "\"entry 150, stop 144\", or give the stop as a percentage (\"a 4% stop\")."]
        return lines, [], {"sizing": None}

    gross = risk_usd / stop_frac
    net = risk_usd / (stop_frac + ROUND_TRIP)
    who = f" of {symbol.removesuffix('USDT')}" if symbol else ""
    lines = [
        f"Bottom line: to lose no more than ${risk_usd:,.0f} if the {stop_frac:.2%} stop is hit, "
        f"the position{who} is ${net:,.0f} — ${risk_usd:,.0f} divided by the stop distance plus "
        f"the {ROUND_TRIP:.2%} taker round trip a stopped trade also pays.",
        f"The textbook figure without costs is ${gross:,.0f}; at that size a stop-out loses "
        f"${gross * (stop_frac + ROUND_TRIP):,.0f}, {ROUND_TRIP / stop_frac:.1%} over the budget.",
    ]
    if entry:
        lines.append(f"At {entry:,.2f} that is {net / entry:,.4g} units ({gross / entry:,.4g} "
                     f"without costs).")
    if account:
        ratio = net / account
        lines.append(
            f"That is {ratio:.0%} of the ${account:,.0f} account"
            + (f" — it needs {ratio:.2f}x leverage, and leverage is what turns a gap through the "
               f"stop into more than the budget." if ratio > 1 else
               ", so no leverage is needed."))
    if symbol and worst_day is not None:
        worst = worst_day(symbol)
        if worst is not None:
            jumped = abs(worst) > stop_frac
            lines.append(
                f"A stop is an order, not a guarantee: {symbol.removesuffix('USDT')}'s worst "
                f"observed 24 hours was {worst:+.1%}"
                + (f", wider than the {stop_frac:.1%} stop — a gap through it fills worse and "
                   f"loses more than ${risk_usd:,.0f}." if jumped else
                   f", inside the {stop_frac:.1%} stop."))
    elif not symbol:
        lines.append("A stop is an order, not a guarantee: a price that gaps through it (a "
                     "weekend, an earnings print) fills worse. Name the instrument to see its "
                     "worst observed 24 hours beside the stop.")
    lines.extend(f"Assumed: {note}." for note in notes)
    lines.append("This is arithmetic on your own rule, not advice on whether to take the trade.")
    sources = [Source(kind="computation", ref="argus.lui.research.sizing",
                      detail="fixed-fractional size: risk / (stop + taker round trip)"),
               Source(kind="computation", ref="argus.cost.model.CostModel.bitget_perp",
                      detail="0.06% taker a side")]
    data = {"risk_usd": risk_usd, "stop": stop_frac, "round_trip": ROUND_TRIP,
            "position_net": net, "position_gross": gross, "entry": entry, "account": account}
    return lines, sources, {"sizing": data}


trace_module(globals())
