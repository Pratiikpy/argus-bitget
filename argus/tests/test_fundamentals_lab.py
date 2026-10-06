"""Offline tests for `argus.lui.research.fundamentals_lab` (round 45 judge, M6-M10).

Every network read in the module is one small function; each test replaces them with fixtures, so
nothing here touches SEC EDGAR, Yahoo or Bitget. The figures are built so the expected answers can
be worked out by hand in the test.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pytest

from argus.lui.research import fundamentals_lab as lab

LAST = date(2026, 6, 30)


def ends(count: int, last: date = LAST) -> list[date]:
    """Quarter ends 91 days apart, oldest first (so a year is exactly four of them)."""
    return [last - timedelta(days=91 * k) for k in reversed(range(count))]


def rows(values: dict[date, float], span: int = 91) -> list[dict[str, Any]]:
    return [{"start": (d - timedelta(days=span - 1)).isoformat(), "end": d.isoformat(), "val": v,
             "form": "10-Q", "filed": (d + timedelta(days=30)).isoformat()}
            for d, v in values.items()]


def facts_of(**series: dict[date, float]) -> dict[str, Any]:
    """An EDGAR us-gaap fact block with one USD tag per keyword (tag name -> quarterly values)."""
    return {tag: {"units": {"USD": rows(values)}} for tag, values in series.items()}


def install(monkeypatch: pytest.MonkeyPatch, facts: dict[str, dict[str, Any]] | None = None,
            **readers: Callable[..., Any]) -> None:
    """Route the module's data readers to fixtures: ``facts`` maps ticker -> company facts."""
    table = facts or {}

    def read_facts(ticker: str) -> dict[str, Any] | None:
        return table.get(ticker)

    monkeypatch.setattr(lab, "_facts", read_facts)
    for name, fn in readers.items():
        monkeypatch.setattr(lab, "_" + name, fn)


def growing(start: float, step: float, count: int = 9) -> dict[date, float]:
    return {d: start + step * i for i, d in enumerate(ends(count))}


# ----------------------------------------------------------------------------- M6


def _context(ident: str, start: str, end: str, axis: str, member: str) -> str:
    return (f'<xbrli:context id="{ident}"><xbrli:entity/><xbrli:period>'
            f"<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>"
            f'</xbrli:period><xbrli:segment><xbrldi:explicitMember dimension="{axis}">{member}'
            "</xbrldi:explicitMember></xbrli:segment></xbrli:context>")


def _fact(context: str, millions: str) -> str:
    return (f'<ix:nonFraction name="us-gaap:Revenues" contextRef="{context}" scale="6">'
            f"{millions}</ix:nonFraction>")


PRODUCT = "srt:ProductOrServiceAxis"
SEGMENT_DOC = "".join([
    _context("now_dc", "2026-04-01", "2026-06-30", PRODUCT, "nvda:DataCenterMember"),
    _context("then_dc", "2025-04-01", "2025-06-30", PRODUCT, "nvda:DataCenterMember"),
    _context("now_gm", "2026-04-01", "2026-06-30", PRODUCT, "nvda:GamingMember"),
    _context("now_os", "2026-04-01", "2026-06-30", "srt:ConsolidationItemsAxis",
             "us-gaap:OperatingSegmentsMember"),
    _fact("now_dc", "89,023"), _fact("then_dc", "41,096"), _fact("now_gm", "4,000"),
    _fact("now_os", "96,221")])
NAMES = {"nvda:DataCenterMember": "Data Center", "nvda:GamingMember": "Gaming"}
NVDA_REVENUE = {d: v for d, v in zip(ends(9), [20, 25, 30, 35, 46.74, 57, 68, 81, 96.221],
                                      strict=True)}
NVDA_REVENUE = {d: v * 1e9 for d, v in NVDA_REVENUE.items()}
NVDA_Q = ("What did Nvidia report for revenue and data center growth last quarter, and how did "
          "the stock react?")


def closes() -> list[tuple[date, float]]:
    days = [date(2026, 8, 24) + timedelta(days=k) for k in range(8)]
    days = [d for d in days if d.weekday() < 5]
    prices = [200.0, 205.0, 209.66, 227.98, 230.0, 231.0]
    return list(zip(days, prices, strict=False))


def nvda(monkeypatch: pytest.MonkeyPatch, *, document: Any = (SEGMENT_DOC, "10-Q", "2026-06-30",
                                                                NAMES),
         reports: list[tuple[date, bool]] | None = None) -> None:
    install(monkeypatch, {"NVDA": facts_of(Revenues=NVDA_REVENUE)},
            document=lambda t: document, reports=lambda t: (
                reports if reports is not None else [(date(2026, 8, 26), True),
                                                     (date(2026, 5, 27), True)]),
            closes=lambda t: closes())


def test_results_segment_and_reaction_are_all_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    nvda(monkeypatch)
    out = lab.lines(NVDA_Q)
    assert out is not None
    head = out[0]
    assert head.startswith("Bottom line: NVDA reported, for the quarter to 30 Jun 2026")
    assert "revenue of $96.22bn (+105.9% on a year earlier)" in head
    assert "Data Center revenue of $89.02bn (+116.6% on a year earlier)" in head
    assert "the stock moved +8.7% on 27 Aug 2026" in head and "after the close" in head
    assert "Gaming" not in head and out[-1].startswith("Data:")
    assert not any("e+" in line.lower() and "e+0" in line for line in out)


def test_reaction_before_the_open_uses_the_same_day(monkeypatch: pytest.MonkeyPatch) -> None:
    nvda(monkeypatch, reports=[(date(2026, 8, 26), False)])
    out = lab.lines(NVDA_Q)
    assert out is not None
    # 26 Aug close 209.66 against 25 Aug close 205.00
    assert "moved +2.3% on 26 Aug 2026" in out[0] and "before the open" in out[0]


def test_a_missing_reaction_says_so_and_still_gives_the_rest(monkeypatch: pytest.MonkeyPatch
                                                             ) -> None:
    nvda(monkeypatch, reports=[(date(2026, 3, 1), True)])
    out = lab.lines(NVDA_Q)
    assert out is not None
    assert "revenue of $96.22bn" in out[0] and "Data Center revenue of $89.02bn" in out[0]
    assert "the stock's reaction could not be read" in out[0]


def test_results_and_a_segment_without_a_reaction_cue_still_answer_both_parts(
        monkeypatch: pytest.MonkeyPatch) -> None:
    nvda(monkeypatch)
    out = lab.lines("What did Nvidia report for revenue and data center growth last quarter?")
    assert out is not None
    assert "Data Center revenue of $89.02bn (+116.6% on a year earlier)" in out[0]
    assert "stock" not in out[0] and "8-K" not in out[-1]


def test_a_segment_label_that_already_says_revenue_is_not_doubled(
        monkeypatch: pytest.MonkeyPatch) -> None:
    doc = SEGMENT_DOC.replace("nvda:DataCenterMember", "coin:TransactionRevenueMember")
    nvda(monkeypatch, document=(doc, "10-Q", "2026-06-30",
                                {"coin:TransactionRevenueMember": "Transaction revenue"}))
    out = lab.lines("What did Nvidia report for revenue and transaction revenue growth, and how "
                    "did shares react?")
    assert out is not None
    assert "Transaction revenue of $89.02bn" in out[0] and "revenue revenue" not in out[0]


def test_a_segment_the_filing_does_not_tag_is_named_not_dropped(monkeypatch: pytest.MonkeyPatch
                                                                 ) -> None:
    nvda(monkeypatch)
    out = lab.lines("What did Nvidia report for revenue and automotive growth last quarter, and "
                    "how did the stock react?")
    assert out is not None
    assert "no 'automotive' revenue line is tagged in its latest 10-Q" in out[0]
    assert "Data Center $89.02bn" in out[0]  # what it does tag
    assert "the stock moved +8.7%" in out[0]


def test_an_unreadable_10q_says_the_segment_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    nvda(monkeypatch, document=None)
    out = lab.lines(NVDA_Q)
    assert out is not None
    assert "its latest 10-Q could not be read, so no segment revenue is given" in out[0]
    assert "revenue of $96.22bn" in out[0] and "the stock moved +8.7%" in out[0]


def test_an_unreadable_filing_set_is_one_honest_line(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {}, document=lambda t: None, reports=lambda t: [], closes=lambda t: [])
    out = lab.lines(NVDA_Q)
    assert out is not None and out[0].startswith("Bottom line: NVDA: its filings could not be read")


def test_results_without_a_reaction_cue_are_not_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    nvda(monkeypatch)
    assert lab.lines("What did Nvidia report last quarter?") is None
    assert lab.lines("What was Nvidia's revenue last quarter?") is None
    assert lab.lines("How did Nvidia stock move the day after its last earnings report?") is None
    assert lab.lines("What move are options pricing for Nvidia's earnings report and how will the "
                     "stock react to revenue?") is None


def test_results_for_two_companies_are_not_claimed(monkeypatch: pytest.MonkeyPatch) -> None:
    nvda(monkeypatch)
    assert lab.lines("What did Nvidia and AMD report for revenue last quarter and how did the "
                     "stocks react?") is None


def test_segment_reader_gives_each_member_with_its_year_ago_figure() -> None:
    found = {s.label: s for s in lab.segments(SEGMENT_DOC, NAMES)}
    assert found["Data Center"].now == 89_023e6 and found["Data Center"].then == 41_096e6
    assert found["Gaming"].then is None
    assert "Operating Segments" not in found  # the total, not a segment


# ----------------------------------------------------------------------------- M7


def fcf_facts(revenue: float, cfo: float, capex: float, *, capex_tag: bool = True
              ) -> dict[str, Any]:
    quarters = ends(8)
    parts: dict[str, dict[date, float]] = {
        "Revenues": {d: revenue / 4 for d in quarters},
        "NetCashProvidedByUsedInOperatingActivities": {d: cfo / 4 for d in quarters}}
    if capex_tag:
        parts["PaymentsToAcquirePropertyPlantAndEquipment"] = {d: capex / 4 for d in quarters}
    return facts_of(**parts)


def test_fcf_margin_after_a_pair_names_both_and_gives_the_verdict(monkeypatch: pytest.MonkeyPatch
                                                                  ) -> None:
    install(monkeypatch, {"NVDA": fcf_facts(300e9, 134e9, 7e9), "AMD": fcf_facts(40e9, 10e9, 2e9)})
    out = lab.lines("Which of the two has the better free cash flow margin?", ["NVDA vs AMD"])
    assert out is not None
    # NVDA (134-7)/300 = 42.3%; AMD (10-2)/40 = 20.0%
    assert out[0].startswith("Bottom line: NVDA has the better free cash flow margin: 42.3% "
                             "against AMD's 20.0% (22.3 points higher)")
    assert "NVDA: free cash flow $127.00bn on revenue $300.00bn = 42.3%" in out[1]
    assert "operating cash flow $134.00bn less capital spending $7.00bn" in out[1]
    assert out[2].startswith("AMD: free cash flow $8.00bn on revenue $40.00bn = 20.0%")


def test_fcf_margin_uses_the_latest_pair_when_there_are_several(monkeypatch: pytest.MonkeyPatch
                                                                ) -> None:
    install(monkeypatch, {"NVDA": fcf_facts(300e9, 134e9, 7e9), "AMD": fcf_facts(40e9, 10e9, 2e9),
                          "INTC": fcf_facts(50e9, 5e9, 20e9)})
    out = lab.lines("Which has the better FCF margin?", ["NVDA vs AMD", "how is the weather",
                                                           "what about Intel"])
    assert out is not None
    # the newest earlier question naming two is NVDA vs AMD; a one-name question is a fallback
    assert "NVDA" in out[0] and "AMD" in out[0] and "INTC" not in out[0]


def test_names_in_the_question_override_the_earlier_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"NVDA": fcf_facts(300e9, 134e9, 7e9), "AMD": fcf_facts(40e9, 10e9, 2e9),
                          "INTC": fcf_facts(50e9, 5e9, 20e9)})
    out = lab.lines("Is Intel's free cash flow margin better than AMD's?", ["NVDA vs AMD"])
    assert out is not None and "INTC" in out[0] and "NVDA" not in out[0]
    assert "AMD has the better free cash flow margin: 20.0% against INTC's -30.0%" in out[0]


def test_fcf_margin_with_no_name_anywhere_asks_which_companies() -> None:
    out = lab.lines("Which of the two has the better free cash flow margin?")
    assert out is not None and out[0].startswith("Bottom line: no company is named")


def test_fcf_margin_with_no_capex_tag_is_not_computed(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"NVDA": fcf_facts(300e9, 134e9, 7e9),
                          "AMD": fcf_facts(40e9, 10e9, 2e9, capex_tag=False)})
    out = lab.lines("Which has the better free cash flow margin, NVDA or AMD?")
    assert out is not None
    assert out[0].startswith("Bottom line: NVDA's free cash flow margin over the four quarters")
    assert any(line.startswith("AMD: capital spending is not tagged") for line in out)


def test_fcf_margin_single_company_and_unreadable_company(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"NVDA": fcf_facts(300e9, 134e9, 7e9)})
    out = lab.lines("What is Nvidia's FCF margin?")
    assert out is not None and "42.3%" in out[0]
    out = lab.lines("What is AMD's FCF margin?")
    assert out is not None and "AMD: its filings could not be read" in out[0]


def test_fcf_margin_needs_four_consecutive_quarters(monkeypatch: pytest.MonkeyPatch) -> None:
    short = fcf_facts(300e9, 134e9, 7e9)
    gap = {d for d in ends(8)[:5]}  # drop the quarters so only three consecutive remain at the end
    for tag in short:
        short[tag]["units"]["USD"] = [r for r in short[tag]["units"]["USD"]
                                      if date.fromisoformat(r["end"]) not in gap
                                      or date.fromisoformat(r["end"]) == ends(8)[3]]
    install(monkeypatch, {"NVDA": short})
    out = lab.lines("What is NVDA's free cash flow margin?")
    assert out is not None and "fewer than four consecutive quarters" in out[0]


# ----------------------------------------------------------------------------- M8


def risk_page(headings: list[str], *, extra_toc: bool = True) -> str:
    bold = 'style="font-weight:bold;font-size:10pt;"'
    body = "".join(f"<p><span {bold}>{h}</span><span> Detail text follows here.</span></p>"
                   for h in headings)
    toc = "<table><tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr></table>" if extra_toc \
        else ""
    return (f"<html><body>{toc}<p>Item 1.</p><p>Business text.</p>"
            f"<p><span {bold}>ITEM 1A. RISK FACTORS</span></p>"
            f"<p><span {bold}>STRATEGIC RISKS</span></p>{body}<p>PART I</p><p>Item 1A</p>"
            "<p><span " + bold + ">Item 1B. Unresolved Staff Comments</span></p><p>None.</p>"
            "<p>Item 2. Properties</p></body></html>")


THIS_YEAR = [
    "We face intense competition across all markets for our products and services.",
    "Our cloud and AI strategy requires heavy investment and depends on customer demand.",
    "Cyberattacks and security vulnerabilities could lead to reduced revenue or liability.",
    "We may be unable to develop and expand adequate infrastructure for our datacenters.",
    "Adverse economic or market conditions could harm our business and results.",
    "Our business depends on our ability to attract and retain talented employees.",
]
LAST_YEAR = [
    "We face intense competition across all markets for our products and services.",
    "Cyberattacks and security vulnerabilities could lead to reduced revenue or liability.",
    "Adverse economic or market conditions could harm our business and results.",
    "Our business depends on our ability to attract and retain talented employees.",
    "We may have excessive outages and data losses if we fail to maintain an operations "
    "infrastructure for online services.",
    "Our focus on mobile devices presents execution risks.",
]
MSFT_FILINGS = [(date(2026, 7, 29), date(2026, 6, 30), "https://sec.test/msft-2026.htm"),
                (date(2025, 7, 30), date(2025, 6, 30), "https://sec.test/msft-2025.htm")]


def msft(monkeypatch: pytest.MonkeyPatch, pages: dict[str, str],
         filings: list[tuple[date, date, str]] | None = None) -> None:
    def fetch(url: str) -> str:
        if url not in pages:
            raise OSError("not served")
        return pages[url]

    install(monkeypatch, {}, tenk_filings=lambda t: MSFT_FILINGS if filings is None else filings,
            fetch=fetch)


RISK_Q = ("Summarize the main risk factors in Microsoft's latest 10-K, and flag anything new "
          "versus the prior year.")


def test_risk_factors_flag_new_and_removed_headings_with_citations(monkeypatch: pytest.MonkeyPatch
                                                                   ) -> None:
    msft(monkeypatch, {"https://sec.test/msft-2026.htm": risk_page(THIS_YEAR),
                       "https://sec.test/msft-2025.htm": risk_page(LAST_YEAR)})
    out = lab.lines(RISK_Q)
    assert out is not None
    assert out[0].startswith("Bottom line: MSFT's latest 10-K (fiscal year to 30 Jun 2026, filed "
                             "29 Jul 2026) lists 6 risk-factor headings")
    assert ("against the prior year's 10-K (filed 30 Jul 2025, 6 headings), 2 new and 2 of its "
            "headings no longer there") in out[0]
    new = next(line for line in out if line.startswith("New in this year's 10-K"))
    assert "Our cloud and AI strategy requires heavy investment" in new
    assert "We may be unable to develop and expand adequate infrastructure" in new
    assert "competition" not in new
    gone = next(line for line in out if line.startswith("In the prior year's 10-K"))
    assert "Our focus on mobile devices presents execution risks" in gone
    assert any("https://sec.test/msft-2026.htm" in line and "https://sec.test/msft-2025.htm" in line
               for line in out if line.startswith("Data:"))
    assert out[1].startswith("Main risks, in the order the 10-K gives them: We face intense")


def test_risk_factors_unchanged_year_reports_none(monkeypatch: pytest.MonkeyPatch) -> None:
    page = risk_page(THIS_YEAR)
    msft(monkeypatch, {"https://sec.test/msft-2026.htm": page,
                       "https://sec.test/msft-2025.htm": page})
    out = lab.lines(RISK_Q)
    assert out is not None
    assert "0 new and 0 of its headings no longer there" in out[0]
    assert "New in this year's 10-K (Item 1A): none." in out


def test_risk_factors_say_so_when_the_prior_10k_cannot_be_read(monkeypatch: pytest.MonkeyPatch
                                                               ) -> None:
    msft(monkeypatch, {"https://sec.test/msft-2026.htm": risk_page(THIS_YEAR)})
    out = lab.lines(RISK_Q)
    assert out is not None
    assert ("no comparison with the prior year is made, because the prior year's 10-K "
            "(filed 30 Jul 2025) could not be read for its Item 1A just now") in out[0]
    assert not any(line.startswith("New in this year's") for line in out)


def test_risk_factors_with_only_one_10k_on_file(monkeypatch: pytest.MonkeyPatch) -> None:
    msft(monkeypatch, {"https://sec.test/msft-2026.htm": risk_page(THIS_YEAR)},
         filings=MSFT_FILINGS[:1])
    out = lab.lines(RISK_Q)
    assert out is not None
    assert "the prior year's 10-K is not among the company's recent EDGAR filings" in out[0]


def test_risk_factors_when_the_latest_10k_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    msft(monkeypatch, {})
    out = lab.lines(RISK_Q)
    assert out is not None and "could not be read for its Item 1A risk factors" in out[0]
    msft(monkeypatch, {}, filings=[])
    out = lab.lines(RISK_Q)
    assert out is not None and "10-K filings could not be read from SEC EDGAR" in out[0]


def test_risk_factors_not_asked_against_a_prior_year_are_not_owned(monkeypatch: pytest.MonkeyPatch
                                                                   ) -> None:
    msft(monkeypatch, {})
    assert lab.lines("Summarize the main risk factors in Microsoft's latest 10-K.") is None
    assert lab.lines("What is new in the risk factors of Microsoft and Apple?") is None


def test_heading_reader_skips_page_headers_and_the_contents_entry() -> None:
    found = lab.risk_headings(risk_page(THIS_YEAR))
    assert found == THIS_YEAR  # no "STRATEGIC RISKS" (under four words), no "Item 1A", no TOC


def test_heading_reader_takes_the_bold_lead_of_a_run_in_heading() -> None:
    page = ("<p><b>Item 1A. Risk Factors</b></p><p><b>We may experience supply problems.</b> "
            "There are limited suppliers for critical parts.</p><p>Plain text only here today.</p>"
            "<p><b>Item 1B. Unresolved</b></p>")
    assert lab.risk_headings(page) == ["We may experience supply problems."]


def test_heading_reader_returns_nothing_without_an_item_1a() -> None:
    assert lab.risk_headings("<p>No such section here at all.</p>") == []


def test_reworded_headings_are_matched_but_new_ones_are_not() -> None:
    now = ["Investment in new business strategies, commercial relationships and acquisitions "
           "could disrupt the Company's ongoing business.",
           "Brand new quantum computing exposure arises."]
    before = ["Investment in new business strategies and acquisitions could disrupt the Company's "
              "ongoing business.", "The Company's retail stores face risks."]
    new, gone = lab.risk_changes(now, before)
    assert new == ["Brand new quantum computing exposure arises."]
    assert gone == ["The Company's retail stores face risks."]


# ----------------------------------------------------------------------------- M9


def apple_facts(gross: list[float], operating: list[float], revenue: float = 100.0
                ) -> dict[str, Any]:
    quarters = ends(len(gross))
    return facts_of(
        RevenueFromContractWithCustomerExcludingAssessedTax={d: revenue for d in quarters},
        GrossProfit={d: g for d, g in zip(quarters, gross, strict=True)},
        OperatingIncomeLoss={d: o for d, o in zip(quarters, operating, strict=True)})


APPLE_Q = "What are Apple's gross margin and operating margin trends over the last 8 quarters?"


def test_margin_trend_gives_eight_points_and_the_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    gross = [46.2, 46.9, 47.1, 46.5, 47.2, 48.2, 49.3, 50.1, 50.5]  # nine quarters on file
    operating = [31.2, 34.5, 31.0, 30.0, 31.6, 35.4, 32.3, 32.6, 33.0]
    install(monkeypatch, {"AAPL": apple_facts(gross, operating)})
    out = lab.lines(APPLE_Q)
    assert out is not None
    assert "AAPL's gross margin is widening, 50.5% in the quarter to Jun 2026 against 46.9% in " \
           in out[0]
    gline = next(line for line in out if line.startswith("AAPL gross margin, 8 quarters:"))
    assert gline.count("%") >= 8 + 2 and "46.2%" not in gline.split("Trend")[0]  # 8, not 9
    assert "Trend +" in gline and "high 50.5%, low 46.5%" in gline
    oline = next(line for line in out if line.startswith("AAPL operating margin, 8 quarters:"))
    third = ends(9)[2]  # the third quarter on file is the second shown: 31.0% of 100 revenue
    assert f"{third:%b %Y} 31.0%" in oline


def test_margin_trend_flat_and_narrowing(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"AAPL": apple_facts([40, 40.3, 39.8, 40.1, 39.9, 40.2, 40.0, 40.1],
                                              [30, 29, 28, 27, 26, 25, 24, 23])})
    out = lab.lines(APPLE_Q)
    assert out is not None
    assert "gross margin is broadly flat" in out[0] and "operating margin is narrowing" in out[0]
    assert "(-7.0 points)" in out[0]


def test_margin_trend_says_when_fewer_quarters_are_tagged(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"AAPL": apple_facts([40, 41, 42, 43, 44], [30, 31, 32, 33, 34])})
    out = lab.lines(APPLE_Q)
    assert out is not None
    assert any("5 quarters (5 of the 8 quarters asked for are tagged)" in line for line in out)


def test_margin_trend_follows_the_quarter_count_and_the_margin_named(monkeypatch: pytest.MonkeyPatch
                                                                     ) -> None:
    install(monkeypatch, {"AAPL": apple_facts([40, 41, 42, 43, 44, 45], [30, 31, 32, 33, 34, 35])})
    out = lab.lines("How has Apple's gross margin trended over the last four quarters?")
    assert out is not None
    assert any(line.startswith("AAPL gross margin, 4 quarters:") for line in out)
    assert not any("operating margin" in line for line in out)


def test_margin_trend_rebuilds_a_fiscal_fourth_quarter_from_year_to_date() -> None:
    """A 10-K gives the year; the fourth quarter is the year less the nine months."""
    from argus.market.hyperscaler_capex import quarters

    spans = [("2025-09-28", "2025-12-27", 40.0), ("2025-09-28", "2026-03-28", 70.0),
             ("2025-09-28", "2026-06-27", 90.0), ("2025-09-28", "2026-09-26", 130.0)]
    got = quarters([{"start": s, "end": e, "val": v, "form": "10-K"} for s, e, v in spans])
    assert got[date(2026, 9, 26)] == pytest.approx(40.0)  # 130 - 90
    assert got[date(2026, 3, 28)] == pytest.approx(30.0)  # 70 - 40


def test_margin_trend_for_a_company_with_no_filings(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {})
    out = lab.lines(APPLE_Q)
    assert out is not None and out[0].startswith("Bottom line: AAPL: its filings could not be read")


def test_a_plain_margin_question_is_not_a_trend_question() -> None:
    assert lab.lines("What is Apple's gross margin?") is None
    assert lab.lines("What was Apple's operating margin in its latest 10-Q?") is None


def test_slope_and_runs_are_exact() -> None:
    assert lab._slope([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert lab._slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)
    quarters = ends(6)
    assert lab._recent_run(quarters) == quarters
    # a missing quarter (a 182-day gap) ends the run: only the quarters after it are used
    assert lab._recent_run([*quarters[:2], *quarters[3:]]) == quarters[3:]


# ----------------------------------------------------------------------------- M10


def summary(cap: float | None, trailing: float | None, forward: float | None, eps: float | None,
            ev: float | None) -> dict[str, Any]:
    def node(x: float | None) -> dict[str, float] | None:
        return None if x is None else {"raw": x}

    return {"summaryDetail": {"marketCap": node(cap), "trailingPE": node(trailing),
                              "forwardPE": node(forward)},
            "defaultKeyStatistics": {"enterpriseValue": node(ev), "trailingEps": node(eps),
                                     "forwardPE": node(forward)}}


def revenue_only(values: list[float]) -> dict[str, Any]:
    return facts_of(Revenues={d: v * 1e9 for d, v in zip(ends(len(values)), values, strict=True)})


def test_valuation_comparison_gives_every_measure_and_a_cheaper_side(monkeypatch: pytest.MonkeyPatch
                                                                     ) -> None:
    coin = revenue_only([1.5, 1.2, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2])    # last 4 = 5.4, prior 6.4
    hood = revenue_only([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])    # last 4 = 5.4? see below
    yahoo = {"COIN": summary(54e9, None, 66.0, -3.84, 50e9),
             "HOOD": summary(108e9, 50.0, 33.0, 2.27, 100e9)}
    install(monkeypatch, {"COIN": coin, "HOOD": hood}, yahoo=lambda t: yahoo[t])
    out = lab.lines("Compare Coinbase and Robinhood on revenue growth and valuation.")
    assert out is not None
    # COIN: last four 1.6+1.5+1.4+1.3? the four newest are 1.5,1.4,1.3,1.2? worked in assertions:
    assert "HOOD is growing faster" in out[0] and "trailing four quarters" in out[0]
    assert "HOOD is the only one with a meaningful trailing P/E (50.0x; COIN has none)" in out[0]
    assert "HOOD is cheaper on forward P/E (33.0x against 66.0x for COIN)" in out[0]
    assert "COIN is cheaper on price/sales" in out[0] and "COIN is cheaper on EV/sales" in out[0]
    coin_line = next(line for line in out if line.startswith("COIN:"))
    assert "market cap $54.00bn" in coin_line
    assert "trailing P/E none (loss-making, trailing EPS -3.84)" in coin_line
    assert "forward P/E 66.0x" in coin_line
    # price/sales = market cap over the last four quarters' revenue: 54 / (1.5+1.4+1.3+1.2 = 5.4)
    assert "price/sales 10.0x" in coin_line and "EV/sales 9.3x" in coin_line
    hood_line = next(line for line in out if line.startswith("HOOD:"))
    # 108 / (1.2+1.3+1.4+1.5 = 5.4) = 20.0
    assert "price/sales 20.0x" in hood_line
    assert "revenue growth +" in hood_line
    assert any("not the same as better value" in line for line in out)


def test_valuation_comparison_computes_trailing_growth_against_the_four_a_year_before(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # nine quarters 1,1,1,1,2,2,2,2,2: the newest four sum to 8; each lies four quarters (364
    # days) after one of [1, 1, 1, 2] (indices 1-4), which sum to 5: growth 8/5 - 1 = +60%
    series = revenue_only([1.0] * 4 + [2.0] * 5)
    install(monkeypatch, {"COIN": series, "HOOD": series},
            yahoo=lambda t: summary(10e9, 20.0, 15.0, 1.0, 9e9))
    out = lab.lines("Compare Coinbase and Robinhood on revenue growth and valuation.")
    assert out is not None
    assert "revenue growth +60.0% (four quarters to 30 Jun 2026, $8.00bn)" in next(
        line for line in out if line.startswith("COIN:"))


def test_valuation_comparison_names_what_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(t: str) -> dict[str, Any]:
        raise RuntimeError("yahoo down")

    install(monkeypatch, {"COIN": revenue_only([1.0] * 9), "HOOD": revenue_only([1.0] * 9)},
            yahoo=broken)
    out = lab.lines("Compare Coinbase and Robinhood on revenue growth and valuation.")
    assert out is not None
    assert any("Yahoo Finance's valuation fields could not be read" in line for line in out)
    assert out[0].startswith("Bottom line: ")


def test_valuation_comparison_with_one_company_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"COIN": revenue_only([1.0] * 9)}, yahoo=lambda t: summary(1e9, 5.0, 4.0,
                                                                                   1.0, 1e9))
    out = lab.lines("Compare Coinbase and Robinhood on revenue growth and valuation.")
    assert out is not None and "fewer than two of the companies could be read" in out[0]


def tsla(monkeypatch: pytest.MonkeyPatch, *, net: list[float], trailing: float | None,
         forward: float | None, revenue: list[float] | None = None) -> None:
    rev = revenue or [22.0, 24.0, 22.5, 25.0, 28.0, 25.0, 22.0, 24.0, 28.2]
    facts = facts_of(
        RevenueFromContractWithCustomerExcludingAssessedTax={
            d: v * 1e9 for d, v in zip(ends(9), rev, strict=True)},
        NetIncomeLoss={d: v * 1e9 for d, v in zip(ends(9), net, strict=True)})
    install(monkeypatch, {"TSLA": facts},
            yahoo=lambda t: summary(1500e9, trailing, forward, 1.08, 1470e9))


TSLA_Q = "Is Tesla's P/E justified given its latest quarterly net income and revenue growth?"


def test_justified_sets_the_implied_growth_against_the_delivered_growth(
        monkeypatch: pytest.MonkeyPatch) -> None:
    tsla(monkeypatch, net=[1.0] * 4 + [1.17, 1.37, 0.84, 0.48, 1.11], trailing=350.0, forward=175.0,
         revenue=[22.0, 24.0, 22.5, 25.0, 22.5, 28.1, 24.9, 22.4, 28.236])
    out = lab.lines(TSLA_Q)
    assert out is not None
    head = out[0]
    # implied = 350/175 - 1 = +100%; delivered quarter growth = 28.236/22.5 - 1 = 25.5%
    assert "priced for more growth than the latest quarter delivered" in head
    assert "350.0x trailing and 175.0x forward assume +100% earnings growth" in head
    assert "about 3.9 times the 25.5% revenue growth delivered" in head
    assert "net income fell" in head or "net income rose" in head
    delivered = next(line for line in out if line.startswith("Delivered, latest quarter"))
    assert "net income $1.11bn" in delivered
    assert "revenue $28.24bn, +25.5% on a year earlier" in delivered
    assert any(line.startswith("Arithmetic: at 25.5% a year") and "3.1 years" in line
               for line in out)
    assert any("no buy or sell view is given" in line for line in out)
    assert not any(w in " ".join(out).lower() for w in ("you should buy", "you should sell"))


def test_justified_when_the_growth_delivered_covers_the_multiple(monkeypatch: pytest.MonkeyPatch
                                                                 ) -> None:
    tsla(monkeypatch, net=[1.0] * 9, trailing=30.0, forward=28.0,
         revenue=[10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 12.0, 13.2])
    out = lab.lines(TSLA_Q)
    assert out is not None
    # implied growth 30/28 - 1 = +7%, against 13.2/12 - 1 = +10.0% delivered
    assert "assumes +7% earnings growth next year, within 1.5 times the 10.0% revenue" in out[0]
    assert "priced for more growth" not in out[0]


def test_justified_with_a_loss_has_no_trailing_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    tsla(monkeypatch, net=[-1.0] * 9, trailing=None, forward=80.0)
    out = lab.lines(TSLA_Q)
    assert out is not None
    assert "needs both a trailing and a forward P/E" in out[0]
    assert "negative" in out[0]


def test_justified_when_forward_is_not_below_trailing(monkeypatch: pytest.MonkeyPatch) -> None:
    tsla(monkeypatch, net=[1.0] * 9, trailing=20.0, forward=22.0)
    out = lab.lines(TSLA_Q)
    assert out is not None and "analysts expect no earnings growth" in out[0]


def test_justified_when_revenue_did_not_grow(monkeypatch: pytest.MonkeyPatch) -> None:
    tsla(monkeypatch, net=[1.0] * 9, trailing=60.0, forward=30.0,
         revenue=[10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 12.0, 11.0])
    out = lab.lines(TSLA_Q)
    assert out is not None and "revenue did not grow in the latest quarter" in out[0]


def test_a_single_stock_valuation_without_a_justified_cue_is_not_owned() -> None:
    assert lab.lines("What is Tesla's P/E?") is None
    assert lab.lines("What is Tesla's market cap?") is None


# ----------------------------------------------------------------------------- not ours


@pytest.mark.parametrize("text", [
    "", "   ", "hello", "What is the price of Bitcoin?",
    "Compare Bitcoin and Ethereum on valuation.",
    "How much would I lose if Nvidia fell 10%?", "Compare Nvidia and AMD.",
    "What did the Fed say?",
])
def test_questions_it_does_not_own_return_none(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {})
    assert lab.lines(text) is None
    assert lab.lines(text, ["NVDA vs AMD"]) is None


def test_no_number_is_printed_in_scientific_notation(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"NVDA": fcf_facts(3e11, 1.34e11, 7e9), "AMD": fcf_facts(4e10, 1e10, 2e9)})
    out = lab.lines("Which has the better free cash flow margin, NVDA or AMD?")
    assert out is not None
    assert not any(re.search(r"\d[eE][+-]?\d", line) for line in out)


def test_every_answer_leads_with_the_bottom_line_and_ends_with_its_source(
        monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {"NVDA": fcf_facts(300e9, 134e9, 7e9), "AMD": fcf_facts(40e9, 10e9, 2e9),
                          "AAPL": apple_facts([40.0, 41.0, 42.0], [30.0, 31.0, 32.0])},
            document=lambda t: (SEGMENT_DOC, "10-Q", "2026-06-30", NAMES),
            reports=lambda t: [(date(2026, 8, 26), True)], closes=lambda t: closes())
    asked = [("Which has the better free cash flow margin, NVDA or AMD?", ()),
             ("Which of the two has the better free cash flow margin?", ("NVDA vs AMD",)),
             (APPLE_Q, ())]
    for text, prior in asked:
        out = lab.lines(text, prior)
        assert out is not None
        assert out[0].startswith("Bottom line: ") and out[-1].startswith("Data: ")
        assert all(line.strip() for line in out)
