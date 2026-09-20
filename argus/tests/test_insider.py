"""Form 4 tests.

Two properties, both learned from real filings rather than imagined.

**The code decides, not the direction.** The first live Form 4 this module parsed was 172,507 NVDA
shares "acquired" at a price of zero — an RSU vest, transaction code ``A``. Counted as a purchase it
is the largest insider buy of the quarter and it means nothing at all.

**One decision is one event.** The second live filing reported a director's single disposal across
seven price tiers. Emitted per line it becomes seven pieces of evidence about one choice.

Fixtures are trimmed from genuine SEC documents so the XML shape under test is the shape that
actually arrives.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.agents.claims import check as check_claims
from argus.market.insider import (
    CONVICTION_CODES,
    INFORMATIVE_CODES,
    InsiderSource,
    InsiderTrade,
    aggregate,
    parse_form4,
)
from argus.truth.evidence import Evidence

ACCEPTED = datetime(2026, 9, 11, 21, 4, 47, tzinfo=UTC)


def _form4(
    *,
    code: str = "P",
    shares: str = "1000",
    price: str = "150.25",
    acquired: str = "A",
    pre_arranged: str = "0",
    officer: str = "1",
    director: str = "0",
    title: str = "EVP, Worldwide Field Ops",
    extra_lines: str = "",
) -> str:
    """The real ownershipDocument shape, trimmed to what the parser reads."""
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0609</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2026-09-09</periodOfReport>
  <issuer>
    <issuerCik>0001045810</issuerCik>
    <issuerName>NVIDIA CORP</issuerName>
    <issuerTradingSymbol>NVDA</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0002152188</rptOwnerCik>
      <rptOwnerName>Parker Nicholas P.</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>{director}</isDirector>
      <isOfficer>{officer}</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>{title}</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <aff10b5One>{pre_arranged}</aff10b5One>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-09-09</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>{code}</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{acquired}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>{shares}</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>{extra_lines}
  </nonDerivativeTable>
  <derivativeTable></derivativeTable>
</ownershipDocument>"""


def _line(shares: str, price: str, code: str = "S") -> str:
    return f"""
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-09-09</value></transactionDate>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>"""


def _parse(xml: str):  # type: ignore[no-untyped-def]
    return parse_form4(xml, accepted_at=ACCEPTED, accession="0002152188-26-000005")


class TestTheCodeDecidesNotTheDirection:
    """The failure that motivates the whole module."""

    def test_an_rsu_vest_is_not_a_purchase(self) -> None:
        """172,507 shares "acquired" at zero. Code A. The executive expressed no view."""
        trade = _parse(_form4(code="A", shares="172507", price="0"))[0]
        assert trade.acquired is True
        assert trade.is_informative is False
        assert trade.meaning == "grant or award"

    def test_tax_withholding_on_a_vest_is_not_a_sale(self) -> None:
        trade = _parse(_form4(code="F", acquired="D"))[0]
        assert trade.is_informative is False

    def test_an_option_exercise_is_mechanical(self) -> None:
        assert _parse(_form4(code="M"))[0].is_informative is False

    def test_an_open_market_purchase_is_informative_and_carries_conviction(self) -> None:
        trade = _parse(_form4(code="P"))[0]
        assert trade.is_informative is True
        assert trade.is_conviction is True

    def test_an_open_market_sale_is_informative_but_not_conviction(self) -> None:
        """Insiders sell for houses and tax bills; that is not a view on the business."""
        trade = _parse(_form4(code="S", acquired="D"))[0]
        assert trade.is_informative is True
        assert trade.is_conviction is False

    def test_the_two_code_sets_are_stated_not_inlined(self) -> None:
        assert CONVICTION_CODES < INFORMATIVE_CODES

    def test_a_zero_price_can_never_be_informative(self) -> None:
        """Whatever the code says, nobody expressed a view at a price of zero."""
        assert _parse(_form4(code="P", price="0"))[0].is_informative is False


class TestThePreArrangedFlag:
    def test_a_10b5_1_purchase_is_not_informative(self) -> None:
        """The decision was taken months ago under different information, and the filing says so."""
        trade = _parse(_form4(code="P", pre_arranged="1"))[0]
        assert trade.is_informative is False
        assert "pre-arranged" in trade.render()

    def test_the_same_trade_without_the_flag_is_informative(self) -> None:
        assert _parse(_form4(code="P", pre_arranged="0"))[0].is_informative is True


class TestOneDecisionIsOneEvent:
    """A market order large enough to walk the book is reported across price tiers."""

    def test_tiers_in_one_filing_collapse_to_a_single_decision(self) -> None:
        xml = _form4(
            code="S", shares="100000", price="230.00", acquired="D",
            extra_lines=_line("50000", "231.00") + _line("50000", "232.00"),
        )
        decisions = aggregate(_parse(xml))
        assert len(decisions) == 1
        assert decisions[0].tiers == 3
        assert decisions[0].shares == Decimal("200000")

    def test_the_average_price_is_volume_weighted(self) -> None:
        """A plain mean across tiers would misprice an uneven fill."""
        xml = _form4(
            code="S", shares="90000", price="100.00", acquired="D",
            extra_lines=_line("10000", "200.00"),
        )
        decision = aggregate(_parse(xml))[0]
        assert decision.average_price == Decimal("110")  # not 150

    def test_the_tier_count_is_reported_so_the_reader_can_see_it(self) -> None:
        xml = _form4(code="S", acquired="D", extra_lines=_line("500", "231.00"))
        assert "across 2 price tiers" in aggregate(_parse(xml))[0].render()

    def test_a_single_line_decision_does_not_mention_tiers(self) -> None:
        assert "price tiers" not in aggregate(_parse(_form4(code="P")))[0].render()

    def test_a_purchase_and_a_sale_in_one_filing_stay_separate(self) -> None:
        """Opposite directions are two decisions however they were filed."""
        xml = _form4(code="P", shares="1000", price="150.00",
                     extra_lines=_line("2000", "151.00", code="S"))
        assert len(aggregate(_parse(xml))) == 2

    def test_aggregating_nothing_returns_nothing(self) -> None:
        assert aggregate([]) == []


class TestTheParserIsRobust:
    def test_malformed_xml_yields_nothing_rather_than_raising(self) -> None:
        """One unreadable filing must not take the evidence feed down."""
        assert parse_form4("<not-xml", accepted_at=ACCEPTED, accession="x") == []

    def test_a_missing_transaction_code_is_skipped(self) -> None:
        xml = _form4().replace("<transactionCode>P</transactionCode>", "")
        assert _parse(xml) == []

    def test_a_malformed_number_becomes_zero_rather_than_raising(self) -> None:
        xml = _form4(shares="not-a-number")
        assert _parse(xml)[0].shares == Decimal("0")

    def test_the_role_falls_back_through_the_relationship_flags(self) -> None:
        director = _parse(_form4(officer="0", director="1", title=""))[0]
        assert director.role == "director"

    def test_an_officer_title_is_preferred_over_the_generic_role(self) -> None:
        assert _parse(_form4())[0].role == "EVP, Worldwide Field Ops"

    def test_the_issuer_ticker_is_read_from_the_filing(self) -> None:
        assert _parse(_form4())[0].ticker == "NVDA"

    @pytest.mark.parametrize("code", ["P", "S", "A", "F", "M"])
    def test_every_common_code_has_a_stated_meaning(self, code: str) -> None:
        trade = _parse(_form4(code=code))[0]
        assert not trade.meaning.startswith("code "), f"{code} has no plain-language meaning"

    def test_an_unknown_code_still_renders_rather_than_failing(self) -> None:
        assert _parse(_form4(code="Z"))[0].meaning == "code Z"


class TestTheRecordSerialises:
    def test_a_trade_carries_everything_needed_to_check_it(self) -> None:
        payload = _parse(_form4(code="P"))[0].as_dict()
        for key in ("ticker", "insider", "role", "code", "meaning", "shares", "price",
                    "notional", "pre_arranged", "informative", "conviction", "accession"):
            assert key in payload

    def test_a_decision_carries_its_tier_count_and_average(self) -> None:
        payload = aggregate(_parse(_form4(code="P")))[0].as_dict()
        assert payload["tiers"] == 1
        assert payload["average_price"] == "150.25"


class TestTheEvidenceCarriesItsRecord:
    """A claim about a filing can only be checked against the filing's own fields.

    `argus.agents.claims` settles "these sales are pre-arranged" against ``pre_arranged``. If the
    evidence reaching the desk were prose alone, that check would have nothing to read and the
    seq-41 hallucination would still be invisible.
    """

    class _Stub(InsiderSource):
        def __init__(self, trades: list[InsiderTrade]) -> None:  # no network
            self._trades = trades

        def trades(  # type: ignore[override]
            self, ticker: str, *, since: datetime
        ) -> tuple[list[InsiderTrade], list[str]]:
            return self._trades, []

    def _evidence(self, **kw: str) -> list[Evidence]:
        source = self._Stub(_parse(_form4(**kw)))
        got, _ = source.evidence("NVDA", as_of=ACCEPTED)
        return got

    def test_the_structured_fields_reach_the_desk(self) -> None:
        attributes = self._evidence(code="S")[0].attributes
        for key in ("pre_arranged", "acquired", "conviction", "code", "accession"):
            assert key in attributes

    def test_the_field_that_caught_seq_41_says_what_the_filing_says(self) -> None:
        assert self._evidence(code="S", pre_arranged="0")[0].attributes["pre_arranged"] is False

    def test_the_checker_fires_on_evidence_built_the_real_way(self) -> None:
        """End to end: parse a filing, build evidence, refute the thesis that named it."""
        evidence = self._evidence(code="S", pre_arranged="0")
        report = check_claims(
            "These insider sales are pre-arranged 10b5-1 plans.",
            records=[(e.id, e.attributes) for e in evidence],
        )
        assert not report.sound
        assert report.contradictions[0].rule == "pre_arranged"
