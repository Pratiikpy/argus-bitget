"""Consensus estimates and revision momentum — the expectation a reported number is judged against.

**This module exists because the desk asked for it.** Ledger seq 41, writing about NVDA, ended:
*"fundamentals cannot be assessed against expectations without consensus estimates."* It was right.
`argus.market.fundamentals` reads what a company **reported**; nothing told the desk what the market
had **expected**, and a reported number without an expectation is not a surprise, it is a fact with
no direction.

**What it adds beyond the level.** Consensus is a point estimate; the more informative object is the
*revision trail* — how many analysts moved their number up or down in the last 7, 30, 60 and 90
days. A beat against a number that forty analysts have been raising for a month is a different
event from a beat against a stale one.

**Provenance, and why it is not Seeking Alpha.** OpenBB reaches forward estimates through Seeking
Alpha's keyless `api/v3/symbol_data/estimates`. Probed from this machine on 2026-09-13, both that
endpoint and its ticker-id search returned **HTTP 403** under the full browser header set, so that
route is closed to us and saying otherwise would be wishful. Yahoo's `quoteSummary` carries the same
class of data and answers, but only after a handshake: an unauthenticated call now returns
**HTTP 401**, and a cookie must be collected before a crumb can be fetched and passed as a query
parameter. That handshake is implemented here and re-run on expiry.

**The point-in-time position, stated rather than implied.** Yahoo gives no date on which a consensus
was formed, so every figure here is dated at **fetch time** and never earlier. A backtest may
therefore not use this module for a past decision — there is no way to know what the consensus was
then, and back-dating it would manufacture exactly the look-ahead `argus.truth` exists to prevent.
The revision counts are the exception worth noting: they are explicitly trailing windows, which is a
genuine point-in-time property, and they are carried as such.
"""

from __future__ import annotations

import http.cookiejar
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from argus.truth.evidence import Evidence

TIMEOUT = 20
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

COOKIE_URL = "https://fc.yahoo.com"
"""Visited only to be handed a cookie. It answers 404 and that is fine — the cookie is the point,
and treating the status code as failure would abort a handshake that succeeded."""

CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"

PERIOD_LABELS = {
    "0q": "the current quarter",
    "+1q": "the next quarter",
    "0y": "the current fiscal year",
    "+1y": "the next fiscal year",
}
"""Yahoo's relative period codes. Translated because `0q` in a decision record means nothing to a
reader six months later."""

STRONG_REVISION_RATIO = 3.0
"""Up-revisions per down-revision before the trail is called one-sided.

A ratio rather than a count: forty analysts raising and thirty cutting is disagreement, while four
raising and none cutting is a direction. Chosen, not measured — and labelled as a description of the
trail, never as a prediction."""


class EstimatesError(RuntimeError):
    """The source could not be read honestly."""


def _fresh_opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [
        ("User-Agent", _UA),
        ("Accept", "*/*"),
        ("Accept-Language", "en-US,en;q=0.9"),
    ]
    return opener


@dataclass(frozen=True, slots=True)
class Revisions:
    """How the estimate has been moving. Trailing windows, so genuinely point-in-time."""

    up_7d: int
    down_7d: int
    up_30d: int
    down_30d: int

    @property
    def net_30d(self) -> int:
        return self.up_30d - self.down_30d

    @property
    def direction(self) -> str:
        """``up``, ``down`` or ``mixed`` — a description of the trail, never a forecast."""
        if self.down_30d == 0 and self.up_30d == 0:
            return "mixed"
        if self.down_30d == 0:
            return "up" if self.up_30d else "mixed"
        ratio = self.up_30d / self.down_30d
        if ratio >= STRONG_REVISION_RATIO:
            return "up"
        if ratio <= 1 / STRONG_REVISION_RATIO:
            return "down"
        return "mixed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "up_7d": self.up_7d, "down_7d": self.down_7d,
            "up_30d": self.up_30d, "down_30d": self.down_30d,
            "net_30d": self.net_30d, "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class Consensus:
    """What the market expects for one period, and how firmly."""

    ticker: str
    period: str
    period_label: str
    end_date: str
    eps_avg: float | None
    eps_low: float | None
    eps_high: float | None
    analysts: int | None
    revenue_avg: float | None
    revisions: Revisions
    fetched_at: str
    """When we asked. Deliberately the only date on this record: Yahoo does not say when the
    consensus was formed, so this is the earliest instant we may honestly claim to have known it."""

    @property
    def dispersion(self) -> float | None:
        """High minus low, in EPS. Wide dispersion means the consensus is an average of guesses."""
        if self.eps_low is None or self.eps_high is None:
            return None
        return round(self.eps_high - self.eps_low, 4)

    def surprise_vs(self, reported_eps: float) -> float | None:
        """Reported against expected, as a fraction of the expectation.

        ``None`` when there is no expectation, rather than zero: "in line" and "nobody had a view"
        are different states, and returning 0.0 for the second would invent a consensus.
        """
        if self.eps_avg in (None, 0) or self.eps_avg is None:
            return None
        return round((reported_eps - self.eps_avg) / abs(self.eps_avg), 4)

    def render(self) -> str:
        eps = "unknown" if self.eps_avg is None else f"{self.eps_avg:.4f}"
        who = "an unknown number of analysts" if not self.analysts else f"{self.analysts} analysts"
        trail = (
            f"revisions {self.revisions.direction}: "
            f"{self.revisions.up_30d} up / {self.revisions.down_30d} down in 30d"
        )
        return (
            f"{self.ticker} consensus EPS {eps} for {self.period_label} "
            f"(ending {self.end_date}), from {who}; {trail}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "period": self.period,
            "period_label": self.period_label,
            "end_date": self.end_date,
            "eps_avg": self.eps_avg,
            "eps_low": self.eps_low,
            "eps_high": self.eps_high,
            "dispersion": self.dispersion,
            "analysts": self.analysts,
            "revenue_avg": self.revenue_avg,
            "revision_direction": self.revisions.direction,
            **self.revisions.as_dict(),
            "fetched_at": self.fetched_at,
        }


def _raw(node: Any) -> Any:
    """Yahoo wraps every number as ``{"raw": x, "fmt": "..."}``; absent fields arrive as ``{}``."""
    if isinstance(node, dict):
        return node.get("raw")
    return node


def _number(node: Any) -> float | None:
    value = _raw(node)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _count(node: Any) -> int:
    value = _raw(node)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def parse(payload: dict[str, Any], *, ticker: str, fetched_at: datetime) -> list[Consensus]:
    """Turn one ``quoteSummary`` response into consensus records.

    A period with no EPS estimate is kept rather than dropped: "forty analysts cover the next
    quarter and none has published a number for the one after" is information about coverage, and
    silently omitting the row would hide it.
    """
    result = (payload.get("quoteSummary") or {}).get("result") or []
    if not result:
        raise EstimatesError(f"no quoteSummary result for {ticker}")
    trend = (result[0].get("earningsTrend") or {}).get("trend") or []
    stamp = fetched_at.isoformat()

    out: list[Consensus] = []
    for row in trend:
        period = str(row.get("period", ""))
        earnings = row.get("earningsEstimate") or {}
        revenue = row.get("revenueEstimate") or {}
        revisions = row.get("epsRevisions") or {}
        out.append(Consensus(
            ticker=ticker,
            period=period,
            period_label=PERIOD_LABELS.get(period, period or "an unnamed period"),
            end_date=str(row.get("endDate") or ""),
            eps_avg=_number(earnings.get("avg")),
            eps_low=_number(earnings.get("low")),
            eps_high=_number(earnings.get("high")),
            analysts=_count(earnings.get("numberOfAnalysts")) or None,
            revenue_avg=_number(revenue.get("avg")),
            revisions=Revisions(
                up_7d=_count(revisions.get("upLast7days")),
                down_7d=_count(revisions.get("downLast7days")),
                up_30d=_count(revisions.get("upLast30days")),
                down_30d=_count(revisions.get("downLast30days")),
            ),
            fetched_at=stamp,
        ))
    return out


class EstimatesSource:
    """Yahoo consensus estimates, keyless, behind a cookie-and-crumb handshake."""

    def __init__(self) -> None:
        self._opener: urllib.request.OpenerDirector | None = None
        self._crumb: str | None = None

    def _handshake(self) -> None:
        """Collect a cookie, then a crumb. Both are required; neither needs an account."""
        opener = _fresh_opener()
        try:
            # Answers 404 and sets the cookie anyway. The status is not the point and treating it
            # as failure would abort a handshake that has already done its job.
            opener.open(COOKIE_URL, timeout=TIMEOUT).close()
        except urllib.error.HTTPError:
            pass
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EstimatesError(f"could not reach the cookie endpoint: {exc}") from exc

        try:
            with opener.open(CRUMB_URL, timeout=TIMEOUT) as response:
                crumb = response.read().decode(errors="replace").strip()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EstimatesError(f"could not fetch a crumb: {exc}") from exc

        # A crumb is a short opaque token. An HTML body here means we were served a consent or
        # block page, and using it as a crumb would produce a confusing 401 later instead of a
        # clear failure now.
        if not crumb or "<" in crumb or len(crumb) > 64:
            raise EstimatesError(
                f"the crumb endpoint returned something that is not a crumb ({crumb[:40]!r}); "
                f"this is usually a consent or block page"
            )
        self._opener, self._crumb = opener, crumb

    def fetch(self, ticker: str, *, as_of: datetime | None = None) -> list[Consensus]:
        """Consensus records for one ticker, newest coverage first.

        Retries the handshake **once** on a 401: crumbs expire, and a single refresh is the
        difference between a source that works for an hour and one that works.
        """
        stamp = as_of or datetime.now(UTC)
        for attempt in (1, 2):
            if self._opener is None or self._crumb is None:
                self._handshake()
            assert self._opener is not None and self._crumb is not None
            query = urllib.parse.urlencode(
                {"modules": "earningsTrend", "crumb": self._crumb}
            )
            url = f"{SUMMARY_URL.format(ticker=urllib.parse.quote(ticker))}?{query}"
            try:
                with self._opener.open(url, timeout=TIMEOUT) as response:
                    payload = json.loads(response.read().decode(errors="replace"))
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403) and attempt == 1:
                    self._opener = self._crumb = None  # crumb expired; redo the handshake once
                    continue
                raise EstimatesError(f"quoteSummary for {ticker}: HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                raise EstimatesError(f"quoteSummary for {ticker}: {exc}") from exc
            return parse(payload, ticker=ticker, fetched_at=stamp)
        raise EstimatesError(f"quoteSummary for {ticker}: authentication failed twice")

    def evidence(
        self, ticker: str, *, as_of: datetime
    ) -> tuple[list[Evidence], list[str]]:
        """Consensus as evidence the earnings analyst can act on.

        Only the two nearest periods. The 2028 fiscal year is a real number and not a fact about
        this week's decision, and an evidence list padded with it buries what matters.
        """
        try:
            records = self.fetch(ticker, as_of=as_of)
        except EstimatesError as exc:
            return [], [f"estimates:{ticker}: unavailable ({exc})"]

        near = [r for r in records if r.period in ("0q", "+1q")] or records[:2]
        out = [
            Evidence(
                id=f"consensus-{ticker}-{r.period}-{int(as_of.timestamp())}",
                claim=r.render(),
                source="filing",
                # Dated at the fetch, never earlier: Yahoo does not say when the consensus formed.
                available_at=as_of,
                # A survey of analyst opinion, not a filed figure. Below a 10-Q, above a headline.
                credibility=0.8,
                attributes=r.as_dict(),
            )
            for r in near
        ]
        status = [f"estimates:{ticker}: {len(records)} period(s), {len(out)} forwarded"]
        return out, status


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch consensus estimates for a ticker.")
    parser.add_argument("ticker", nargs="?", default="NVDA")
    args = parser.parse_args()

    for record in EstimatesSource().fetch(args.ticker):
        print(record.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PERIOD_LABELS",
    "STRONG_REVISION_RATIO",
    "Consensus",
    "EstimatesError",
    "EstimatesSource",
    "Revisions",
    "main",
    "parse",
]
