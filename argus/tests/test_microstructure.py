"""Short volume and halts: real column names, published reason codes, no silent zeros."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime

import pytest

from argus.market.microstructure import (
    HALT_REASONS,
    MAX_LOOKBACK_DAYS,
    Halt,
    HaltKind,
    MicrostructureError,
    ShortVolume,
    evidence,
    fetch_halts,
    fetch_short_volume,
    kind_of,
    parse_halts,
    parse_short_volume,
    status,
)

LIVE = os.environ.get("ARGUS_LIVE_VENUE") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set ARGUS_LIVE_VENUE=1 to hit FINRA and Nasdaq")

AT = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)

# The real file's header and three real rows, copied from the live fetch on 2026-09-13.
FINRA = """Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
20260911|NVDA|13390506.971560|99044|35300143.518841|B,Q,N
20260911|TSLA|8613653.678808|25549.500000|14382570.698167|B,Q,N
20260911|QQQ|5930446.472981|17002|8781631.443767|B,Q,N
"""

HALTS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:ndaq="http://www.nasdaqtrader.com/"><channel>
<item>
  <title>NVDA</title>
  <ndaq:HaltDate>09/11/2026</ndaq:HaltDate>
  <ndaq:HaltTime>19:50:00.000</ndaq:HaltTime>
  <ndaq:IssueSymbol>NVDA</ndaq:IssueSymbol>
  <ndaq:IssueName>NVIDIA Corporation</ndaq:IssueName>
  <ndaq:ReasonCode>T1</ndaq:ReasonCode>
  <ndaq:ResumptionQuoteTime />
  <ndaq:ResumptionTradeTime />
</item>
<item>
  <title>HUBC</title>
  <ndaq:HaltDate>09/10/2026</ndaq:HaltDate>
  <ndaq:HaltTime>14:05:00.000</ndaq:HaltTime>
  <ndaq:IssueSymbol>HUBC</ndaq:IssueSymbol>
  <ndaq:IssueName>Hub Cyber Security Ltd. Ord</ndaq:IssueName>
  <ndaq:ReasonCode>LUDP</ndaq:ReasonCode>
  <ndaq:ResumptionQuoteTime>14:10:00</ndaq:ResumptionQuoteTime>
  <ndaq:ResumptionTradeTime>14:15:00</ndaq:ResumptionTradeTime>
</item>
</channel></rss>"""


class TestTheFinraFileIsParsedNotGuessed:
    def test_it_reads_the_real_pipe_delimited_columns(self) -> None:
        got = parse_short_volume(FINRA)
        assert set(got) == {"NVDA", "TSLA", "QQQ"}
        assert got["NVDA"].total == pytest.approx(35300143.518841)

    def test_the_short_share_matches_the_published_numbers(self) -> None:
        assert parse_short_volume(FINRA)["NVDA"].short_share == pytest.approx(0.3793, abs=1e-3)

    def test_the_header_row_is_not_a_ticker(self) -> None:
        assert "Symbol" not in parse_short_volume(FINRA)

    def test_a_filter_keeps_only_what_was_asked_for(self) -> None:
        """The real file has over 12,000 rows; keeping them all to read twelve is waste."""
        got = parse_short_volume(FINRA, wanted=frozenset({"NVDA"}))
        assert set(got) == {"NVDA"}

    def test_an_unparseable_row_is_skipped_not_zeroed(self) -> None:
        """A zero short share reads as 'nobody is short', which is a claim, not an absence."""
        broken = FINRA + "20260911|BAD|not-a-number|0|123|Q\n"
        assert "BAD" not in parse_short_volume(broken)

    def test_a_zero_total_does_not_divide_by_zero(self) -> None:
        got = ShortVolume(as_of=date(2026, 9, 11), ticker="X", short=0.0, exempt=0.0, total=0.0)
        assert got.short_share == 0.0

    def test_the_claim_says_it_is_about_the_underlying_not_the_token(self) -> None:
        text = parse_short_volume(FINRA)["NVDA"].render()
        assert "underlying equity rather than the token" in text

    def test_it_serialises(self) -> None:
        got = parse_short_volume(FINRA)["TSLA"].as_dict()
        assert got["ticker"] == "TSLA" and 0 < got["short_share"] < 1


class TestHaltsCarryTheirPublishedMeaning:
    def test_every_code_in_the_table_is_nasdaqs_own(self) -> None:
        """Transcribed from nasdaqtrader.com/trader.aspx?id=TradeHaltCodes, fetched 2026-09-13."""
        assert HALT_REASONS["T1"] == "news pending"
        assert HALT_REASONS["H10"] == "SEC trading suspension"
        assert len(HALT_REASONS) == 10

    def test_news_pending_and_a_volatility_pause_are_different_kinds(self) -> None:
        """Both are 'halted'. One says information is coming, the other that the tape moved."""
        assert kind_of("T1") is HaltKind.INFORMATION
        assert kind_of("LUDP") is HaltKind.VOLATILITY

    def test_a_regulatory_halt_is_its_own_kind(self) -> None:
        for code in ("H4", "H10", "H11", "D"):
            assert kind_of(code) is HaltKind.REGULATORY

    def test_an_unpublished_code_is_unknown_not_bucketed(self) -> None:
        """Nasdaq adds codes. A guess here would be a confident misreading."""
        assert kind_of("ZZ9") is HaltKind.UNKNOWN
        halt = Halt(ticker="X", name="X", halted_at=AT, reason_code="ZZ9",
                    resumption_quote="", resumption_trade="")
        assert "unpublished code" in halt.reason

    def test_the_feed_is_parsed_newest_first(self) -> None:
        got = parse_halts(HALTS)
        assert [h.ticker for h in got] == ["NVDA", "HUBC"]

    def test_a_resumed_halt_is_marked_resumed(self) -> None:
        got = {h.ticker: h for h in parse_halts(HALTS)}
        assert got["HUBC"].resumed and not got["NVDA"].resumed

    def test_an_unresolved_information_halt_explains_what_it_means_here(self) -> None:
        """The point of the whole module: the underlying stopped and the token did not."""
        text = {h.ticker: h for h in parse_halts(HALTS)}["NVDA"].render()
        assert "price discovery is stopped while the token keeps trading" in text

    def test_a_resumed_volatility_pause_does_not_get_that_warning(self) -> None:
        text = {h.ticker: h for h in parse_halts(HALTS)}["HUBC"].render()
        assert "price discovery is stopped" not in text

    def test_a_row_without_a_symbol_is_skipped(self) -> None:
        broken = HALTS.replace("<ndaq:IssueSymbol>NVDA</ndaq:IssueSymbol>", "")
        assert len(parse_halts(broken)) == 1

    def test_unparseable_xml_raises_rather_than_returning_nothing(self) -> None:
        """An empty tuple would read as 'no halts', which is the opposite of 'I could not look'."""
        with pytest.raises(MicrostructureError, match="not parseable"):
            parse_halts("<rss>")


class TestEvidenceIsScopedToTheSymbol:
    def _short(self) -> ShortVolume:
        return parse_short_volume(FINRA)["NVDA"]

    def test_a_halt_on_another_ticker_is_not_evidence_about_this_one(self) -> None:
        got = evidence(ticker="NVDA", as_of=AT, halts=parse_halts(HALTS))
        assert all("HUBC" not in e.claim for e in got)
        assert len(got) == 1

    def test_short_volume_is_dated_by_the_session_not_by_now(self) -> None:
        got = evidence(ticker="NVDA", as_of=AT, short=self._short())[0]
        assert got.available_at.date() == date(2026, 9, 11)

    def test_short_volume_credibility_is_below_one_for_a_stated_reason(self) -> None:
        """Short volume counts prints, not positions: a market maker's hedge prints short too."""
        got = evidence(ticker="NVDA", as_of=AT, short=self._short())[0]
        assert 0.9 < got.credibility < 1.0

    def test_a_halt_is_the_venue_stating_its_own_state_so_it_is_certain(self) -> None:
        got = evidence(ticker="NVDA", as_of=AT, halts=parse_halts(HALTS))[0]
        assert got.credibility == 1.0

    def test_the_reason_code_travels_as_an_attribute(self) -> None:
        got = evidence(ticker="NVDA", as_of=AT, halts=parse_halts(HALTS))[0]
        assert got.attributes["reason_code"] == "T1"
        assert got.attributes["kind"] == "information"

    def test_nothing_available_produces_no_evidence_rather_than_an_empty_claim(self) -> None:
        assert evidence(ticker="NVDA", as_of=AT) == []

    def test_ids_are_stable(self) -> None:
        a = evidence(ticker="NVDA", as_of=AT, short=self._short())[0]
        b = evidence(ticker="NVDA", as_of=AT, short=self._short())[0]
        assert a.id == b.id


class TestFailureIsNamed:
    def test_a_lookback_that_finds_nothing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty mapping and 'every symbol had zero short volume' are indistinguishable."""
        import argus.market.microstructure as ms

        def _dead(url: str, *, timeout: int) -> bytes:
            raise MicrostructureError("404")

        monkeypatch.setattr(ms, "_get", _dead)
        with pytest.raises(MicrostructureError, match="no FINRA short-volume file"):
            fetch_short_volume(lookback=2)

    def test_it_walks_back_over_non_trading_days(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Sunday request has to reach Friday. Six days covers a holiday weekend."""
        import argus.market.microstructure as ms

        seen: list[str] = []

        def _partial(url: str, *, timeout: int) -> bytes:
            seen.append(url)
            if len(seen) < 3:
                raise MicrostructureError("404")
            return FINRA.encode()

        monkeypatch.setattr(ms, "_get", _partial)
        got = fetch_short_volume(on=date(2026, 9, 13))
        assert len(seen) == 3 and "NVDA" in got

    def test_the_lookback_covers_a_holiday_weekend(self) -> None:
        assert MAX_LOOKBACK_DAYS >= 5

    def test_the_status_line_names_the_absence(self) -> None:
        assert "unavailable" in status(None, None, "connection refused")

    def test_the_status_line_reports_what_was_found(self) -> None:
        text = status(parse_short_volume(FINRA), parse_halts(HALTS))
        assert "3 ticker(s)" in text and "halt(s)" in text


@live_only
class TestAgainstTheRegulators:
    RTOKENS = frozenset({"NVDA", "TSLA", "AAPL", "MSFT", "META", "GOOGL",
                         "AMZN", "COIN", "MSTR", "QQQ", "TQQQ", "SQQQ"})

    def test_finra_carries_every_one_of_our_underlyings(self) -> None:
        got = fetch_short_volume(wanted=self.RTOKENS)
        assert set(got) == self.RTOKENS
        for reading in got.values():
            assert 0.0 < reading.short_share < 1.0

    def test_the_nasdaq_halt_feed_answers(self) -> None:
        got = fetch_halts()
        assert isinstance(got, tuple)
        for halt in got:
            assert halt.ticker and halt.reason
