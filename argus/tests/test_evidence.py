"""Evidence-feed tests. No network: every source is driven by fixtures.

The property under test is the one that made the module necessary: the desk must receive real
catalysts, and it must never receive a fact stamped after the decision instant. The second half is
where a fast feed becomes a leak, and it is asserted at the gate, not trusted at the source.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from argus.market.evidence import (
    RSS_FEEDS,
    EvidenceError,
    Filing,
    Gathered,
    RssSource,
    _is_empty,
    gather,
    underlying_ticker,
)
from argus.truth.evidence import Evidence

NOW = datetime(2026, 9, 12, 18, 0, tzinfo=UTC)

# The three routing sets the desk actually uses (desk.py). Emitting any other string starves an
# analyst silently, which is the bug this module exists to fix.
EVENT_SOURCES = {"sec-edgar", "news", "macro"}
SENTIMENT_SOURCES = {"social"}
EARNINGS_SOURCES = {"filing", "transcript"}
ALL_ROUTED = EVENT_SOURCES | SENTIMENT_SOURCES | EARNINGS_SOURCES


class _Edgar:
    """Fixture source: returns whatever filings it is built with."""

    def __init__(self, filings: list[Filing]) -> None:
        self._f = filings

    def evidence(self, symbol: str, *, as_of: datetime, lookback: timedelta) -> list[Evidence]:
        out = []
        for f in self._f:
            if f.accepted > as_of or f.accepted < as_of - lookback:
                continue
            src = "sec-edgar" if f.is_event else "filing"
            out.append(Evidence(id=f"edgar-{f.accession}", claim=f"{f.form} {f.item_summary}",
                                source=src, available_at=f.accepted))
        return out


class _Rss:
    def __init__(self, items: list[Evidence], status: list[str]) -> None:
        self._i, self._s = items, status

    def evidence(self, symbol: str, *, as_of: datetime, lookback: timedelta, keywords=()):  # type: ignore[no-untyped-def]
        return [e for e in self._i if e.available_at <= as_of], list(self._s)


class _Bitget:
    def __init__(self, items: list[Evidence], status: str) -> None:
        self._i, self._s = items, status

    def evidence(self, symbol: str, *, as_of: datetime):  # type: ignore[no-untyped-def]
        return list(self._i), [self._s]


class _Twitter:
    """Fixture source: same shape as the real `TwitterSource`, no subprocess."""

    def __init__(self, items: list[Evidence], status: str) -> None:
        self._i, self._s = items, status

    def evidence(self, symbol: str, *, as_of: datetime):  # type: ignore[no-untyped-def]
        return [e for e in self._i if e.available_at <= as_of], [self._s]


class _Reddit:
    """Fixture source: same shape as the real `RedditSource`, no subprocess."""

    def __init__(self, items: list[Evidence], status: str) -> None:
        self._i, self._s = items, status

    def evidence(self, symbol: str, *, as_of: datetime):  # type: ignore[no-untyped-def]
        return [e for e in self._i if e.available_at <= as_of], [self._s]


def _filing(form: str, hours_ago: float, items: tuple[str, ...] = ()) -> Filing:
    t = NOW - timedelta(hours=hours_ago)
    return Filing(form=form, filed=t, accepted=t, accession=f"acc-{form}-{hours_ago}",
                  items=items, description="")


class TestTickerMapping:
    def test_rtoken_symbols_map_to_the_anchor(self) -> None:
        assert underlying_ticker("NVDAUSDT") == "NVDA"
        assert underlying_ticker("mstrusdt") == "MSTR"
        assert underlying_ticker("COINUSDC") == "COIN"

    def test_a_bare_ticker_passes_through(self) -> None:
        assert underlying_ticker("AAPL") == "AAPL"

    def test_a_suffix_alone_is_not_a_ticker(self) -> None:
        assert underlying_ticker("USDT") == "USDT"


class TestFilingSemantics:
    def test_eight_k_is_an_event_and_routes_to_the_event_analyst(self) -> None:
        f = _filing("8-K", 3, ("2.02", "9.01"))
        assert f.is_event and not f.is_periodic
        assert "2.02 results of operations" in f.item_summary

    def test_ten_q_is_periodic_and_routes_to_earnings(self) -> None:
        f = _filing("10-Q", 3)
        assert f.is_periodic and not f.is_event
        assert f.item_summary == "no items listed"

    def test_unknown_item_codes_are_kept_not_dropped(self) -> None:
        assert "6.05" in _filing("8-K", 1, ("6.05",)).item_summary


class TestTheAsOfGate:
    """A feed cannot leak the future into a decision by being fast."""

    def test_future_filings_never_reach_the_desk(self) -> None:
        edgar = _Edgar([_filing("8-K", -2, ("8.01",)), _filing("8-K", 5, ("8.01",))])
        g = gather("NVDAUSDT", as_of=NOW, edgar=edgar, rss=_Rss([], []), bitget=_Bitget([], "ok"))
        assert len(g.evidence) == 1
        assert all(e.available_at <= NOW for e in g.evidence)

    def test_the_gate_raises_if_a_source_misbehaves(self) -> None:
        """Sources apply the rule too; the gate is the line a test can point at."""
        leak = Evidence(id="x", claim="from the future", source="news",
                        available_at=NOW + timedelta(minutes=1))

        class _Bad(_Rss):
            def evidence(self, symbol, *, as_of, lookback, keywords=()):  # type: ignore[no-untyped-def,override]
                return [leak], ["bad: leaked"]

        with pytest.raises(EvidenceError, match="after as_of"):
            gather("NVDAUSDT", as_of=NOW, edgar=_Edgar([]), rss=_Bad([], []),
                   bitget=_Bitget([], "ok"))

    def test_naive_as_of_is_refused(self) -> None:
        with pytest.raises(EvidenceError, match="timezone-aware"):
            gather("NVDAUSDT", as_of=datetime(2026, 9, 12, 18, 0), edgar=_Edgar([]),
                   rss=_Rss([], []), bitget=_Bitget([], "ok"))


class TestLookbackWindows:
    def test_filings_use_the_longer_window(self) -> None:
        """A weekend cycle with a 72h filing window sees nothing and concludes there was never a
        catalyst. Filings are sparse; the window is a week."""
        edgar = _Edgar([_filing("8-K", 5 * 24, ("5.02",))])
        g = gather("AMZNUSDT", as_of=NOW, edgar=edgar, rss=_Rss([], []), bitget=_Bitget([], "ok"))
        assert g.catalysts == 1

    def test_filings_older_than_the_window_are_excluded(self) -> None:
        edgar = _Edgar([_filing("8-K", 9 * 24, ("5.02",))])
        g = gather("AMZNUSDT", as_of=NOW, edgar=edgar, rss=_Rss([], []), bitget=_Bitget([], "ok"))
        assert g.catalysts == 0


class TestRoutingAndStatus:
    def test_every_emitted_source_is_one_the_desk_routes(self) -> None:
        edgar = _Edgar([_filing("8-K", 1, ("8.01",)), _filing("10-Q", 2)])
        rss = _Rss([Evidence(id="r", claim="h", source="news", available_at=NOW),
                    Evidence(id="m", claim="fed", source="macro", available_at=NOW)], ["feed: ok"])
        bit = _Bitget([Evidence(id="s", claim="fng", source="social", available_at=NOW)],
                      "bitget: ok")
        g = gather("NVDAUSDT", as_of=NOW, edgar=edgar, rss=rss, bitget=bit)
        assert {e.source for e in g.evidence} <= ALL_ROUTED
        assert {e.source for e in g.evidence} & EVENT_SOURCES
        assert {e.source for e in g.evidence} & EARNINGS_SOURCES
        assert {e.source for e in g.evidence} & SENTIMENT_SOURCES

    def test_catalysts_count_events_not_price(self) -> None:
        g = Gathered(evidence=[
            Evidence(id="a", claim="8-K", source="sec-edgar", available_at=NOW),
            Evidence(id="b", claim="10-Q", source="filing", available_at=NOW),
            Evidence(id="c", claim="fed", source="macro", available_at=NOW),
            Evidence(id="d", claim="headline", source="news", available_at=NOW),
            Evidence(id="e", claim="fng", source="social", available_at=NOW),
        ])
        assert g.catalysts == 3

    def test_a_dark_feed_is_reported_not_hidden(self) -> None:
        """Visible error recovery: the status travels with the evidence."""
        g = gather("NVDAUSDT", as_of=NOW, edgar=_Edgar([]),
                   rss=_Rss([], ["coindesk: unavailable (URLError)"]),
                   bitget=_Bitget([], "bitget:sentiment_index: reachable, returned no data"))
        joined = " ".join(g.status)
        assert "unavailable" in joined and "returned no data" in joined

    def test_evidence_is_newest_first(self) -> None:
        edgar = _Edgar([_filing("8-K", 30, ("8.01",)), _filing("8-K", 2, ("8.01",))])
        g = gather("NVDAUSDT", as_of=NOW, edgar=edgar, rss=_Rss([], []), bitget=_Bitget([], "ok"))
        assert g.evidence[0].available_at > g.evidence[1].available_at


class TestGatherWithTwitter:
    """`twitter` is opt-in, same shape as `insider`/`fundamentals` — omitted entirely, `gather`
    never touches it (no real subprocess call happens unless a caller explicitly asks)."""

    def test_omitted_by_default_no_social_from_twitter(self) -> None:
        g = gather(
            "NVDAUSDT", as_of=NOW, edgar=_Edgar([]), rss=_Rss([], []), bitget=_Bitget([], "ok")
        )
        assert not any(e.id.startswith("twitter-") for e in g.evidence)

    def test_supplied_source_contributes_evidence_and_status(self) -> None:
        tw = _Twitter(
            [Evidence(id="twitter-1", claim="bullish thread", source="social", available_at=NOW,
                      credibility=0.4)],
            "twitter: 1 tweets",
        )
        g = gather("NVDAUSDT", as_of=NOW, edgar=_Edgar([]), rss=_Rss([], []),
                   bitget=_Bitget([], "ok"), twitter=tw)
        assert any(e.id == "twitter-1" for e in g.evidence)
        assert "twitter: 1 tweets" in g.status

    def test_the_as_of_gate_still_applies_to_twitter_evidence(self) -> None:
        future = Evidence(id="twitter-future", claim="leaked", source="social",
                           available_at=NOW + timedelta(hours=1), credibility=0.4)
        tw = _Twitter([future], "twitter: 1 tweets")
        g = gather("NVDAUSDT", as_of=NOW, edgar=_Edgar([]), rss=_Rss([], []),
                   bitget=_Bitget([], "ok"), twitter=tw)
        assert not any(e.id == "twitter-future" for e in g.evidence)


class TestGatherWithReddit:
    """Same shape as `TestGatherWithTwitter` — `reddit` is opt-in the same way."""

    def test_omitted_by_default_no_social_from_reddit(self) -> None:
        g = gather(
            "NVDAUSDT", as_of=NOW, edgar=_Edgar([]), rss=_Rss([], []), bitget=_Bitget([], "ok")
        )
        assert not any(e.id.startswith("reddit-") for e in g.evidence)

    def test_supplied_source_contributes_evidence_and_status(self) -> None:
        rd = _Reddit(
            [Evidence(id="reddit-1", claim="DD thread", source="social", available_at=NOW,
                      credibility=0.35)],
            "reddit: 1 posts",
        )
        g = gather("NVDAUSDT", as_of=NOW, edgar=_Edgar([]), rss=_Rss([], []),
                   bitget=_Bitget([], "ok"), reddit=rd)
        assert any(e.id == "reddit-1" for e in g.evidence)
        assert "reddit: 1 posts" in g.status

    def test_the_as_of_gate_still_applies_to_reddit_evidence(self) -> None:
        future = Evidence(id="reddit-future", claim="leaked", source="social",
                           available_at=NOW + timedelta(hours=1), credibility=0.35)
        rd = _Reddit([future], "reddit: 1 posts")
        g = gather("NVDAUSDT", as_of=NOW, edgar=_Edgar([]), rss=_Rss([], []),
                   bitget=_Bitget([], "ok"), reddit=rd)
        assert not any(e.id == "reddit-future" for e in g.evidence)


class TestEmptyPayloadDetection:
    """The exact shapes Bitget's server returned on probe. An empty answer must read as empty."""

    def test_rss_shape_with_no_items(self) -> None:
        assert _is_empty([{"feed": "coindesk", "error": "", "items": []}])

    def test_bare_error_object(self) -> None:
        assert _is_empty({"error": ""})
        assert _is_empty({"alt_me_error": ""})

    def test_a_real_error_is_also_empty_of_data(self) -> None:
        assert _is_empty({"error": "Unknown action: latest"})

    def test_real_data_is_not_empty(self) -> None:
        assert not _is_empty({"value": 42, "value_classification": "Fear"})
        assert not _is_empty([{"feed": "coindesk", "error": "", "items": [{"title": "x"}]}])


class TestRssParsing:
    def test_items_without_a_publish_time_are_dropped(self) -> None:
        """An item that cannot be placed on the clock does not enter the record."""
        xml = b"""<rss><channel>
          <item><title>dated NVDA</title><link>a</link>
            <pubDate>Sat, 12 Sep 2026 14:00:00 +0000</pubDate></item>
          <item><title>undated NVDA</title><link>b</link></item>
        </channel></rss>"""

        class _Src(RssSource):
            def _fetch(self, url: str) -> bytes:
                return xml

        hs = _Src().headlines("test", "http://x")
        assert [h.title for h in hs] == ["dated NVDA"]
        assert hs[0].published.tzinfo is not None

    def test_the_verified_feed_list_excludes_the_stale_ones(self) -> None:
        """WSJ Markets and Blockworks returned months-old newest items on 2026-09-12."""
        assert "wsj-markets" not in RSS_FEEDS and "blockworks" not in RSS_FEEDS
        assert {src for _, src in RSS_FEEDS.values()} <= EVENT_SOURCES


class TestTheSecUserAgent:
    """SEC's filter rejects a User-Agent containing a bare domain, and that was a live outage.

    Probed on 2026-09-13: ``ARGUS research (github.com/Pratiikpy/argus-bitget)`` returned **403**,
    and so did the same string without the parentheses, while ``ARGUS research``,
    ``ARGUS-research/0.1`` and ``ARGUS research contact@argus.invalid`` all returned **200**. It is
    the domain that is rejected, not a missing email.

    Every EDGAR call was failing because of it — 8-K and 10-Q filings, Form 4 insider transactions
    and XBRL fundamentals — while the cycle faithfully logged ``edgar: unavailable (HTTPError)``
    that nobody read.
    """

    def test_the_default_agent_carries_no_bare_domain(self) -> None:
        from argus.market import evidence

        assert "github.com" not in evidence._UA
        assert "://" not in evidence._UA

    def test_it_still_identifies_the_requester(self) -> None:
        """SEC rejects anonymous clients; an empty agent would 403 for a different reason."""
        from argus.market import evidence

        assert evidence._UA.strip()
        assert "argus" in evidence._UA.lower()

    def test_a_contact_from_the_environment_is_used_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deployment path SEC's fair-access policy actually asks for."""
        monkeypatch.setenv("ARGUS_CONTACT_EMAIL", "someone@example.com")
        import importlib

        from argus.market import evidence

        reloaded = importlib.reload(evidence)
        try:
            assert "someone@example.com" in reloaded._UA
            assert "://" not in reloaded._UA
        finally:
            monkeypatch.delenv("ARGUS_CONTACT_EMAIL", raising=False)
            importlib.reload(evidence)


class TestTheUserAgentSecActuallyAccepts:
    """This has broken twice. Both times every EDGAR path went dark and the cause was the
    fallback agent, so the shape is pinned rather than trusted.

    Measured against both SEC hosts on 2026-09-13: `data.sec.gov` accepts a bare "ARGUS research",
    `www.sec.gov` returns 403 for it, and `www.sec.gov` is where company_tickers.json lives. A
    partial outage is harder to notice than a total one, which is why the first fix looked done.
    """

    def test_the_fallback_carries_a_contact_token(self) -> None:
        import argus.market.evidence as ev

        assert "@" in ev._UA, ev._UA

    def test_the_fallback_carries_no_domain_url(self) -> None:
        """SEC's filter rejects an agent containing a bare domain, which is how it broke the
        first time."""
        import argus.market.evidence as ev

        assert "://" not in ev._UA
        assert "github.com" not in ev._UA

    def test_the_fallback_address_is_a_reserved_domain(self) -> None:
        """RFC 2606 reserves .invalid, so the fallback names nobody and can receive no mail."""
        import argus.market.evidence as ev

        assert ev._FALLBACK_CONTACT.endswith(".invalid")

    def test_no_personal_address_is_compiled_into_the_source(self) -> None:
        from pathlib import Path

        import argus.market.evidence as ev

        text = Path(ev.__file__).read_text(encoding="utf-8")
        assert "gmail.com" not in text and "prate" not in text.lower()


class TestEightKMaterialityIsAProperty:
    """A 9.01 exhibit list and a 4.02 non-reliance arrived as the same kind of object until
    2026-09-13. The second is a restatement."""

    def test_the_taxonomy_matches_the_secs_own_form(self) -> None:
        """Thirty-three items, transcribed from https://www.sec.gov/files/form8-k.pdf."""
        from argus.market.evidence import ITEM_LABELS

        assert len(ITEM_LABELS) == 33

    def test_every_labelled_item_has_a_materiality(self) -> None:
        from argus.market.evidence import ITEM_LABELS, ITEM_MATERIALITY

        assert set(ITEM_LABELS) == set(ITEM_MATERIALITY)

    def test_a_restatement_is_severe(self) -> None:
        from argus.market.evidence import Materiality, materiality_of

        assert materiality_of(["4.02"]) is Materiality.SEVERE

    def test_a_cybersecurity_incident_is_severe(self) -> None:
        from argus.market.evidence import Materiality, materiality_of

        assert materiality_of(["1.05"]) is Materiality.SEVERE

    def test_an_exhibit_list_is_routine(self) -> None:
        from argus.market.evidence import Materiality, materiality_of

        assert materiality_of(["9.01"]) is Materiality.ROUTINE

    def test_the_strongest_item_wins_rather_than_the_average(self) -> None:
        """Otherwise a company could dilute a restatement by attaching routine items, which is a
        thing that actually happens."""
        from argus.market.evidence import Materiality, materiality_of

        assert materiality_of(["4.02", "9.01", "5.07"]) is Materiality.SEVERE

    def test_a_catch_all_item_is_opaque_not_routine(self) -> None:
        """7.01 and 8.01 carry anything from a conference schedule to a resignation."""
        from argus.market.evidence import Materiality, materiality_of

        assert materiality_of(["8.01"]) is Materiality.OPAQUE
        assert materiality_of(["7.01"]) is Materiality.OPAQUE

    def test_an_unknown_code_is_opaque_not_routine(self) -> None:
        """The SEC adds items — 1.05 arrived in 2023 — so an unknown code must mean 'go and read
        it', never 'nothing to see'."""
        from argus.market.evidence import Materiality, materiality_of

        assert materiality_of(["9.99"]) is Materiality.OPAQUE

    def test_no_items_at_all_is_routine(self) -> None:
        from argus.market.evidence import Materiality, materiality_of

        assert materiality_of([]) is Materiality.ROUTINE

    def test_a_periodic_report_is_material_despite_carrying_no_items(self) -> None:
        """Calling a 10-K routine because it has no item codes would be exactly backwards."""
        from argus.market.evidence import Filing, Materiality

        at = datetime(2026, 9, 1, tzinfo=UTC)
        tenq = Filing(form="10-Q", filed=at, accepted=at, accession="a", items=(),
                      description="10-Q")
        assert tenq.materiality is Materiality.MATERIAL

    def test_severe_and_opaque_both_demand_a_read(self) -> None:
        from argus.market.evidence import Filing

        at = datetime(2026, 9, 1, tzinfo=UTC)
        for items in (("4.02",), ("8.01",)):
            assert Filing(form="8-K", filed=at, accepted=at, accession="a", items=items,
                          description="").demands_a_read
        assert not Filing(form="8-K", filed=at, accepted=at, accession="a", items=("9.01",),
                          description="").demands_a_read
