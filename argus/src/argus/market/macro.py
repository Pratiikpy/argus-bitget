"""The Treasury yield curve — a working macro source, replacing one that does not work.

Bitget's official `macro-analyst` Skill exposes a `rates_yields` tool, and the handbook names it as
part of the perception layer for this track. Probed on 2026-09-13 it answers, and what it answers
with is every tenor as ``{"error": ""}`` followed by ``spread_10y2y: 0.0, inverted: false`` — a
yield curve of zeros derived from legs that never arrived. `market/skills.py:hollow()` now keeps
that out of the evidence, which is correct and leaves the desk with no rates data at all.

This module supplies it from the issuer. The US Treasury publishes the daily par yield curve as
XML, keyless, with a date on every observation — ``home.treasury.gov/resource-center/...``,
verified live: 175 daily entries for 2026, the most recent carrying 1-month through 30-year.
Nothing here is scraped from a page or inferred; the field names below (``BC_10YEAR``,
``NEW_DATE``) are the issuer's own.

**Why this source rather than FRED.** FRED covers far more series, and its CSV endpoint timed out
from this network when probed, so it is not claimed as a source. Treasury answers, and for the one
question a trading desk asks of rates — what is the curve today, and is it inverted — Treasury is
the primary publisher rather than a redistributor. A source that redistributes is a second place
for a number to go stale.

**The curve is dated, and the date is used.** Every :class:`Curve` carries the observation date the
issuer stamped, not the moment it was fetched. The curve published on a Friday is still Friday's
curve when read on Sunday, and reporting it as Sunday's would be the same point-in-time error the
rest of this system exists to avoid. :meth:`Curve.age_days` says how stale it is, and
:func:`evidence` refuses to emit anything older than :data:`MAX_AGE_DAYS`.

    python -m argus.market.macro
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from argus.truth.evidence import Evidence

CURVE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    "?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)

USER_AGENT = "ARGUS research"
"""The same User-Agent policy the SEC reader learned the hard way: a bare library default is
rejected by several public issuers, and a contactable string is the documented courtesy."""

MAX_AGE_DAYS = 7
"""Oldest curve that may still become evidence.

A week covers a long weekend plus a holiday. Beyond it the curve is reported as stale rather than
quietly presented as current — an old rate looks exactly like a new one."""

_TENORS: tuple[tuple[str, str], ...] = (
    ("BC_1MONTH", "1m"), ("BC_2MONTH", "2m"), ("BC_3MONTH", "3m"), ("BC_6MONTH", "6m"),
    ("BC_1YEAR", "1y"), ("BC_2YEAR", "2y"), ("BC_3YEAR", "3y"), ("BC_5YEAR", "5y"),
    ("BC_7YEAR", "7y"), ("BC_10YEAR", "10y"), ("BC_20YEAR", "20y"), ("BC_30YEAR", "30y"),
)
"""Issuer field name to the short label a person reads. Taken from a live response, not guessed."""


class MacroError(RuntimeError):
    """The curve could not be read. Raised rather than returning an empty curve of zeros —
    which is precisely the failure mode this module was written to replace."""


@dataclass(frozen=True, slots=True)
class Curve:
    """One day's par yield curve, as the Treasury published it."""

    as_of: date
    yields: dict[str, Decimal]

    def get(self, tenor: str) -> Decimal | None:
        return self.yields.get(tenor)

    @property
    def spread_10y2y_bps(self) -> Decimal | None:
        """The 10y-2y spread in basis points.

        The issuer publishes yields in **per cent** (4.96 means 4.96%), so the difference of two
        tenors is in percentage points and a basis-point figure is a hundred times larger. The
        first version of this module printed "+0.33bps" for a 33bp spread — wrong by two orders of
        magnitude, in the direction that makes an inversion look like a rounding error. Both units
        are now named in the property, so a caller cannot pick the wrong one silently.
        """
        spread = self.spread_10y2y
        return None if spread is None else spread * 100

    @property
    def spread_10y2y(self) -> Decimal | None:
        """The recession signal. ``None`` when either leg is missing — never zero.

        A spread of zero is a real and meaningful state (a flat curve). Returning zero for
        "we do not have the data" makes the two indistinguishable, which is the exact defect
        found in the Skill this module replaces.
        """
        ten, two = self.get("10y"), self.get("2y")
        return None if ten is None or two is None else ten - two

    @property
    def spread_10y3m(self) -> Decimal | None:
        """The other standard inversion measure, preferred by much of the literature."""
        ten, three = self.get("10y"), self.get("3m")
        return None if ten is None or three is None else ten - three

    @property
    def is_inverted(self) -> bool | None:
        spread = self.spread_10y2y
        return None if spread is None else spread < 0

    def age_days(self, *, now: date | None = None) -> int:
        return ((now or datetime.now(UTC).date()) - self.as_of).days

    def is_stale(self, *, now: date | None = None, max_age_days: int = MAX_AGE_DAYS) -> bool:
        return self.age_days(now=now) > max_age_days

    def render(self) -> str:
        parts = ", ".join(f"{label} {self.yields[label]}%" for _, label in _TENORS
                          if label in self.yields)
        bps = self.spread_10y2y_bps
        tail = (
            "10y-2y unavailable" if bps is None
            else f"10y-2y {bps:+}bps{' — INVERTED' if bps < 0 else ''}"
        )
        return f"US Treasury par curve {self.as_of.isoformat()}: {parts}; {tail}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "yields": {k: str(v) for k, v in self.yields.items()},
            "spread_10y2y_pct_points": (
                None if self.spread_10y2y is None else str(self.spread_10y2y)
            ),
            "spread_10y2y_bps": (
                None if self.spread_10y2y_bps is None else str(self.spread_10y2y_bps)
            ),
            "spread_10y3m": None if self.spread_10y3m is None else str(self.spread_10y3m),
            "inverted": self.is_inverted,
            "age_days": self.age_days(),
        }


_ENTRY = re.compile(r"<m:properties>(.*?)</m:properties>", re.S)
_DATE = re.compile(r"<d:NEW_DATE[^>]*>([^<]+)</d:NEW_DATE>")


def _field(block: str, name: str) -> Decimal | None:
    match = re.search(rf"<d:{name}[^>]*>([^<]+)</d:{name}>", block)
    if match is None:
        return None
    try:
        return Decimal(match.group(1).strip())
    except ArithmeticError:
        return None


def parse(xml: str) -> list[Curve]:
    """Every dated curve in one year's feed, oldest first.

    Parsed with regular expressions rather than an XML library on purpose: the payload is a fixed,
    flat Atom feed from one issuer, the field names are stable, and pulling in a parser to walk
    twelve numeric elements would add a dependency for no additional correctness. A malformed
    entry is skipped rather than aborting the year — one bad day must not cost the other 174.
    """
    out: list[Curve] = []
    for block in _ENTRY.findall(xml):
        stamp = _DATE.search(block)
        if stamp is None:
            continue
        try:
            observed = datetime.fromisoformat(stamp.group(1).strip()).date()
        except ValueError:
            continue
        yields: dict[str, Decimal] = {}
        for field_name, label in _TENORS:
            value = _field(block, field_name)
            if value is not None:
                yields[label] = value
        if yields:
            out.append(Curve(as_of=observed, yields=yields))
    return sorted(out, key=lambda c: c.as_of)


def fetch(*, year: int | None = None, timeout: int = 30) -> list[Curve]:
    """Read one calendar year of daily curves from the Treasury."""
    target = year or datetime.now(UTC).year
    request = urllib.request.Request(
        CURVE_URL.format(year=target), headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MacroError(f"could not read the Treasury yield curve: {exc}") from exc
    curves = parse(body)
    if not curves:
        raise MacroError(
            f"the Treasury feed for {target} carried no readable curve; refusing to report an "
            f"empty curve as a flat one"
        )
    return curves


def latest(*, year: int | None = None, timeout: int = 30) -> Curve:
    """The most recent published curve."""
    return fetch(year=year, timeout=timeout)[-1]


def evidence(curve: Curve, *, as_of: datetime) -> list[Evidence]:
    """Turn a curve into dated evidence the desk can read.

    ``available_at`` is the **issuer's** observation date, not the fetch time. A Friday curve read
    on a Sunday is Friday's information, and dating it Sunday would grant it a freshness the
    issuer never claimed.

    A stale curve produces nothing. The desk is told about the absence through the feed-status
    line, in the same way every other dark source is, rather than being handed an old number that
    reads exactly like a current one.
    """
    if curve.is_stale(now=as_of.date()):
        return []
    stamp = datetime.combine(curve.as_of, datetime.min.time(), tzinfo=UTC)
    out = [
        Evidence(
            id=f"ust-curve-{curve.as_of.isoformat()}",
            claim=curve.render(),
            source="macro",
            available_at=stamp,
            # The issuer of the instrument, published on a fixed schedule. Nothing in the evidence
            # set is more authoritative about rates than the Treasury is.
            credibility=1.0,
            attributes={k: str(v) for k, v in curve.yields.items()},
        )
    ]
    spread = curve.spread_10y2y_bps
    if spread is not None:
        out.append(Evidence(
            id=f"ust-spread-{curve.as_of.isoformat()}",
            claim=(
                f"US 10y-2y spread {spread:+}bps on {curve.as_of.isoformat()}"
                + (
                    " — the curve is inverted, which has preceded every US recession since 1955 "
                    "and is a statement about the rate path, not a trading signal"
                    if spread < 0 else ""
                )
            ),
            source="macro",
            available_at=stamp,
            credibility=1.0,
            attributes={"spread_10y2y_bps": str(spread), "inverted": str(spread < 0)},
        ))
    return out


FNG_URL = "https://api.alternative.me/fng/?limit=1"
"""Crypto Fear & Greed, keyless. The only free sentiment feed that answers from this network.

Three were recommended by the sentiment audit and probed on 2026-09-13: StockTwits returns 403 to
four different User-Agents, CNN's index returns 418, and this one returns 200. That is the whole
free-sentiment landscape as it actually is, rather than as a repository README describes it.

**It is labelled for what it measures.** This is a crypto-wide risk appetite index, not equity
sentiment and not per-symbol. It is genuine information about the risk environment the rTokens
trade in, and it is *not* a view on NVDA. The evidence it produces says so, because a
crypto-sentiment number presented as a stock signal is the kind of borrowed relevance that makes an
evidence count look larger than it is."""


@dataclass(frozen=True, slots=True)
class FearGreed:
    """One dated reading of the crypto Fear & Greed index."""

    as_of: datetime
    value: int
    classification: str

    def render(self) -> str:
        return (
            f"Crypto Fear & Greed {self.value}/100 ({self.classification}) at "
            f"{self.as_of.date().isoformat()} — a market-wide risk-appetite reading for the venue "
            f"these tokens trade on, not a view on any single equity"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(), "value": self.value,
            "classification": self.classification,
        }


def fetch_fear_greed(*, timeout: int = 20) -> FearGreed:
    """Read the index. Raises rather than returning a neutral 50 on failure."""
    request = urllib.request.Request(FNG_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise MacroError(f"could not read the Fear & Greed index: {exc}") from exc
    rows = payload.get("data") or []
    if not rows:
        raise MacroError("the Fear & Greed feed returned no reading")
    row = rows[0]
    try:
        return FearGreed(
            as_of=datetime.fromtimestamp(int(row["timestamp"]), tz=UTC),
            value=int(row["value"]),
            classification=str(row.get("value_classification", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MacroError(f"the Fear & Greed reading could not be parsed: {exc}") from exc


def fear_greed_evidence(reading: FearGreed, *, as_of: datetime) -> list[Evidence]:
    """One dated, honestly-scoped piece of evidence."""
    return [
        Evidence(
            id=f"fng-{reading.as_of.date().isoformat()}",
            claim=reading.render(),
            # `macro`, not `social`. Routing it to `social` was wrong and had a measurable cost:
            # `agents/selection.py:72` sends anything on the social channel to the sentiment
            # analyst, so one market-wide index made a DEMOTED analyst run on all eleven symbols
            # and charged deliberation for it. Caught by ablating the selection layer
            # (`argus.eval.ablations`) rather than by reading the code.
            #
            # The classification is also simply more accurate: this is a venue-level risk-appetite
            # reading, the same kind of object as the Treasury curve and the volatility index, and
            # not a narrative about any company.
            source="macro",
            available_at=reading.as_of,
            # Deliberately low. It is a real published number about a related market, which is a
            # weaker thing than a filing about this company, and the credibility says so.
            credibility=0.35,
            attributes={"value": str(reading.value), "scope": "crypto-wide"},
        )
    ]


def status(curve: Curve | None, error: str = "", *, now: date | None = None) -> str:
    """One line for the feed-health list, so a dark curve is visible rather than absent.

    ``now`` exists for the same reason :meth:`Curve.is_stale` and :meth:`Curve.age_days` already
    take it, and its absence here was a real defect: this function read the wall clock directly,
    so **any test of it was a time bomb**. `tests/test_macro.py` pinned a fixed 2026-09-11 curve
    and asserted the line said ``ok`` — true when written, and it began failing four days later
    when real time moved the fixture past `MAX_AGE_DAYS`. A test that passes only near the date it
    was written is measuring the calendar, not the code. Defaults to today, so no caller changes.
    """
    if curve is None:
        return f"treasury-curve: unavailable ({error or 'no reason given'})"
    if curve.is_stale(now=now):
        return (
            f"treasury-curve: last published {curve.as_of.isoformat()}, "
            f"{curve.age_days(now=now)} days old — too stale to use"
        )
    return f"treasury-curve: {curve.as_of.isoformat()} ({curve.age_days(now=now)}d old), ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="US Treasury par yield curve")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--history", type=int, default=0, help="print the last N curves")
    args = parser.parse_args(argv)

    try:
        curves = fetch(year=args.year)
    except MacroError as exc:
        print(status(None, str(exc)))
        return 1

    curve = curves[-1]
    print(status(curve))
    print(curve.render())
    for item in evidence(curve, as_of=datetime.now(UTC)):
        print(f"  evidence: {item.claim[:150]}")
    if args.history:
        for older in curves[-args.history:]:
            print(json.dumps(older.as_dict(), default=str))
    return 0


__all__ = [
    "CURVE_URL",
    "FNG_URL",
    "MAX_AGE_DAYS",
    "Curve",
    "FearGreed",
    "MacroError",
    "evidence",
    "fear_greed_evidence",
    "fetch",
    "fetch_fear_greed",
    "latest",
    "main",
    "parse",
    "status",
]


if __name__ == "__main__":
    raise SystemExit(main())
