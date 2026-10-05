"""Shareholder-return and valuation comparisons, from SEC XBRL facts and Yahoo (all offline)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from argus.lui.research import filing_figures
from argus.lui.research import shareholder_metrics as sm

QUARTERS = [("2024-07-01", "2024-09-30"), ("2024-10-01", "2024-12-31"),
            ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
            ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
            ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30")]


def _row(start: str | None, end: str, val: float, form: str = "10-Q") -> dict[str, Any]:
    row: dict[str, Any] = {"end": end, "val": val, "form": form, "filed": "2026-07-30"}
    if start:
        row["start"] = start
    return row


def _quarterly(values: list[float]) -> list[dict[str, Any]]:
    return [_row(s, e, v) for (s, e), v in zip(QUARTERS, values, strict=True)]


def _tag(rows: list[dict[str, Any]], unit: str = "USD") -> dict[str, Any]:
    """Dollar rows are written in millions above and served in dollars, as SEC serves them."""
    if unit == "USD":
        rows = [{**r, "val": r["val"] * 1_000_000} for r in rows]
    return {"units": {unit: rows}}


def _us_doc() -> dict[str, Any]:
    """Eight quarters to 30 Jun 2026 for a made-up company, in the shapes SEC serves them: some
    tags as three-month rows, others as year-to-date rows a quarter must be taken out of."""
    gaap = {
        "Revenues": _tag(_quarterly([100, 110, 120, 125, 130, 140, 150, 160])),
        "OperatingIncomeLoss": _tag(_quarterly([18, 19, 20, 22, 26, 28, 30, 32])),
        "NetIncomeLoss": _tag(_quarterly([9, 10, 12, 13, 14, 15, 16, 17])),
        "PaymentsOfDividendsCommonStock": _tag(_quarterly([3, 3, 4, 4, 4, 4, 4, 4])),
        # only the year and the half are tagged: TTM = FY2025 + H1 2026 - H1 2025 = 40 + 25 - 18
        "PaymentsForRepurchaseOfCommonStock": _tag([
            _row("2025-01-01", "2025-12-31", 40, "10-K"), _row("2026-01-01", "2026-06-30", 25),
            _row("2025-01-01", "2025-06-30", 18)]),
        # cash flow as year-to-date figures; a quarter is the difference between two of them
        "NetCashProvidedByUsedInOperatingActivities": _tag([
            _row("2025-01-01", "2025-03-31", 20), _row("2025-01-01", "2025-06-30", 45),
            _row("2025-01-01", "2025-09-30", 70), _row("2025-01-01", "2025-12-31", 100, "10-K"),
            _row("2026-01-01", "2026-03-31", 25), _row("2026-01-01", "2026-06-30", 55)]),
        "PaymentsToAcquirePropertyPlantAndEquipment": _tag([
            _row("2025-01-01", "2025-03-31", 8), _row("2025-01-01", "2025-06-30", 18),
            _row("2025-01-01", "2025-09-30", 28), _row("2025-01-01", "2025-12-31", 40, "10-K"),
            _row("2026-01-01", "2026-03-31", 10), _row("2026-01-01", "2026-06-30", 22)]),
        "EarningsPerShareDiluted": _tag([
            _row("2024-07-01", "2024-09-30", 0.9), _row("2024-01-01", "2024-09-30", 2.7),
            _row("2024-01-01", "2024-12-31", 3.6, "10-K"),
            _row("2025-01-01", "2025-03-31", 1.0), _row("2025-04-01", "2025-06-30", 1.0),
            _row("2025-07-01", "2025-09-30", 1.0), _row("2025-01-01", "2025-09-30", 3.0),
            _row("2025-01-01", "2025-12-31", 4.0, "10-K"),
            _row("2026-01-01", "2026-03-31", 1.0), _row("2026-04-01", "2026-06-30", 3.5)],
            "USD/shares"),
        "CommonStockDividendsPerShareDeclared": _tag([
            _row("2025-01-01", "2025-03-31", 0.5), _row("2025-04-01", "2025-06-30", 0.5),
            _row("2025-07-01", "2025-09-30", 0.5), _row("2025-01-01", "2025-09-30", 1.5),
            _row("2025-01-01", "2025-12-31", 2.0, "10-K"),
            _row("2026-01-01", "2026-03-31", 0.55), _row("2026-04-01", "2026-06-30", 0.55)],
            "USD/shares"),
        "CommonStockSharesOutstanding": _tag([
            _row(None, "2025-06-30", 1_000_000_000), _row(None, "2026-06-30", 990_000_000)],
            "shares"),
        "StockRepurchasedDuringPeriodShares": _tag([
            _row("2025-01-01", "2025-12-31", 8_000_000, "10-K"),
            _row("2026-01-01", "2026-06-30", 4_000_000),
            _row("2025-01-01", "2025-06-30", 3_000_000)],
            "shares"),
    }
    return {"us-gaap": gaap}


def _ifrs_doc() -> dict[str, Any]:
    def year(tag_val: tuple[float, float], unit: str = "USD") -> dict[str, Any]:
        return _tag([_row("2024-01-01", "2024-12-31", tag_val[1], "20-F"),
                     _row("2025-01-01", "2025-12-31", tag_val[0], "20-F")], unit)

    return {"ifrs-full": {
        "Revenue": year((1000, 900)),
        "ProfitLossAttributableToOwnersOfParent": year((100, 90)),
        "CashFlowsFromUsedInOperatingActivities": year((200, 180)),
        "DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities": year((50, 45)),
        "PaymentsToAcquireOrRedeemEntitysShares": year((30, 20)),
        "WeightedAverageShares": year((500e6, 520e6), "shares"),
    }}


def _yahoo(price: float = 100.0, rate: float | None = None, forward: float | None = 12.0,
           cap: float = 50e9) -> dict[str, Any]:
    return {"summaryDetail": {"dividendRate": {"raw": rate} if rate else {},
                              "forwardPE": {"raw": forward} if forward else {},
                              "marketCap": {"raw": cap}},
            "financialData": {"currentPrice": {"raw": price}},
            "defaultKeyStatistics": {"enterpriseValue": {"raw": 60_000.0},
                                     "trailingEps": {"raw": 6.5}}}


def _no_report(ticker: str) -> tuple[str, date, date] | None:
    return None


def _facts(**docs: dict[str, Any] | None) -> Any:
    def get(ticker: str) -> dict[str, Any] | None:
        if ticker not in docs:
            raise RuntimeError("EDGAR unreachable")
        return docs[ticker]

    return get


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_listed", lambda text: [])
    monkeypatch.setattr(sm, "_named", lambda text: [])
    sm._FACTS_CACHE.clear()


def _co(doc: dict[str, Any], ticker: str = "AAA", **yahoo: Any) -> sm.Company:
    return sm.read_company(ticker, facts_of=lambda t: doc, summary_of=lambda t: _yahoo(**yahoo),
                           report_of=_no_report)


# --- reading a US filer -------------------------------------------------------------------------


def test_trailing_four_quarters_and_the_four_before_are_summed() -> None:
    c = _co(_us_doc())
    assert c.error is None and c.end == date(2026, 6, 30)
    assert c.rev == 580e6 and c.rev_prior == 455e6
    assert c.growth == pytest.approx(580 / 455 - 1)
    assert c.margin("operating") == pytest.approx(116 / 580)
    assert c.margin("operating", prior=True) == pytest.approx(79 / 455)
    assert c.net == 62e6


def test_a_year_is_assembled_from_the_fiscal_year_and_two_year_to_date_figures() -> None:
    c = _co(_us_doc())
    assert c.buyback == 47e6
    assert c.buyback_share_of_cap == pytest.approx(47e6 / 50e9)
    assert c.buyback_shares == 8_000_000 + 4_000_000 - 3_000_000


def test_cash_flow_quarters_are_the_difference_of_year_to_date_figures() -> None:
    c = _co(_us_doc())
    assert c.cfo == 110e6 and c.capex == 44e6 and c.fcf == 66e6
    assert c.fcf_margin == pytest.approx(66 / 580)


def test_dividend_is_the_latest_declared_times_four_and_the_yield_uses_the_price() -> None:
    c = _co(_us_doc(), price=110.0)
    assert c.dps == 0.55 and c.dps_kind == "declared" and c.dps_annual == pytest.approx(2.2)
    assert c.yield_ == pytest.approx(2.2 / 110.0)
    assert c.div_paid == 16e6 and c.payout == pytest.approx(16 / 62)


def test_diluted_eps_gives_the_trailing_pe_and_a_one_off_quarter_is_flagged() -> None:
    c = _co(_us_doc())
    assert c.eps == pytest.approx(6.5) and c.eps_prior == pytest.approx(3.8)
    assert c.trailing_pe == pytest.approx(100 / 6.5)
    flag = sm._one_off(c)
    assert flag is not None and "30 Jun 2026" in flag and "twice" in flag
    assert sm._one_off(_co(_ordinary_eps_doc())) is None


def _ordinary_eps_doc() -> dict[str, Any]:
    doc = _us_doc()
    doc["us-gaap"]["EarningsPerShareDiluted"] = _tag([
        _row("2025-07-01", "2025-09-30", 1.0), _row("2025-10-01", "2025-12-31", 1.1),
        _row("2026-01-01", "2026-03-31", 1.0), _row("2026-04-01", "2026-06-30", 1.2)],
        "USD/shares")
    return doc


def test_share_count_change_uses_the_balance_sheet_at_the_quarter_ends() -> None:
    c = _co(_us_doc())
    assert (c.shares_now, c.shares_then) == (990_000_000, 1_000_000_000)
    assert c.share_change == pytest.approx(-0.01)
    said = sm._p_shares(c)
    assert "fell by 10m" in said and "-1.00%" in said and "8m" not in said


def test_a_company_with_no_dividend_tag_and_no_yahoo_rate_pays_none() -> None:
    doc = _us_doc()
    for tag in ("PaymentsOfDividendsCommonStock", "CommonStockDividendsPerShareDeclared"):
        del doc["us-gaap"][tag]
    c = _co(doc)
    assert not c.pays and c.yield_ is None and c.payout is None
    assert "pays no dividend" in sm._p_div(c)


def test_a_loss_gives_no_pe_and_no_percentage_across_it() -> None:
    doc = _us_doc()
    doc["us-gaap"]["EarningsPerShareDiluted"] = _tag([
        _row("2025-07-01", "2025-09-30", -1.0), _row("2025-10-01", "2025-12-31", -1.0),
        _row("2026-01-01", "2026-03-31", -1.0), _row("2026-04-01", "2026-06-30", -1.0)],
        "USD/shares")
    c = _co(doc)
    assert c.trailing_pe is None and "a loss" in sm._p_pe(c) and "-$4.00" in sm._p_pe(c)
    assert c.eps_growth is None


def test_zero_repurchases_tagged_for_the_last_year_are_read_as_no_buyback() -> None:
    doc = _us_doc()
    doc["us-gaap"]["PaymentsForRepurchaseOfCommonStock"] = _tag([
        _row("2025-01-01", "2025-12-31", 0, "10-K")])
    c = _co(doc)
    assert c.buyback == 0 and any("treated as no buyback" in n for n in c.notes)
    assert "no buyback" in sm._p_buyback(c)


def test_a_filing_newer_than_the_facts_feed_is_disclosed() -> None:
    c = sm.read_company("AAA", facts_of=lambda t: _us_doc(), summary_of=lambda t: _yahoo(),
                        report_of=lambda t: ("10-Q", date(2026, 9, 30), date(2026, 10, 29)))
    assert any("not yet in SEC's XBRL feed" in n and "30 Jun 2026" in n for n in c.notes)


def test_yahoo_failing_still_answers_from_the_filings() -> None:
    def broken(ticker: str) -> dict[str, Any]:
        raise RuntimeError("401")

    out = sm.lines("Does Apple have a buyback?", facts_of=_facts(AAPL=_us_doc()),
                   summary_of=broken, report_of=_no_report)
    assert out is not None
    text = "\n".join(out)
    assert "bought back $47m" in text and "Yahoo Finance did not answer" in text
    assert out[0].startswith("Bottom line: ") and out[-1].endswith("Not advice.")


# --- a 20-F filer ------------------------------------------------------------------------------


def test_an_ifrs_filer_is_read_for_its_fiscal_year_and_the_one_before() -> None:
    c = _co(_ifrs_doc(), ticker="SHEL", price=10.0, rate=0.8, cap=5_000.0)
    assert c.kind == "ifrs" and c.end == date(2025, 12, 31)
    assert c.growth == pytest.approx(1000 / 900 - 1)
    assert c.payout == pytest.approx(0.5) and c.buyback == 30e6
    assert c.yield_ == pytest.approx(0.08)
    assert c.share_change == pytest.approx(500 / 520 - 1)
    assert c.margin("net") == pytest.approx(0.1)
    assert c.gross is None and c.capex is None and c.fcf is None
    assert "not computed" in sm._p_fcf(c)
    assert sm._ifrs_gaps(c) == ["gross profit", "operating profit", "capital spending",
                                "diluted EPS"]


# --- the whole answer ---------------------------------------------------------------------------


def _answer(question: str, **docs: dict[str, Any] | None) -> list[str]:
    out = sm.lines(question, facts_of=_facts(**docs), summary_of=lambda t: _yahoo(),
                   report_of=_no_report)
    assert out is not None
    return out


def test_the_answer_has_a_bottom_line_the_support_and_a_data_line() -> None:
    out = _answer("Does Apple pay a dividend and how large is its buyback relative to market "
                  "cap?", AAPL=_us_doc())
    assert out[0].startswith("Bottom line: Apple (AAPL), four quarters to 30 Jun 2026")
    assert "$2.20 a year" in out[0] and "yield 2.20% at $100.00" in out[0]
    assert "bought back $47m" in out[0] and "0.09% of its" in out[0]
    assert out[-1].startswith("Data: ") and out[-1].endswith("Not advice.")
    assert "AAPL 10-Q filed 30 Jul 2026" in out[-1]


def test_two_companies_are_ranked_on_each_metric_asked() -> None:
    cheap = _us_doc()
    cheap["us-gaap"]["CommonStockDividendsPerShareDeclared"] = _tag([
        _row("2026-01-01", "2026-03-31", 1.0), _row("2026-04-01", "2026-06-30", 1.0)],
        "USD/shares")
    out = _answer("Rank Exxon and Chevron by dividend yield and payout ratio.",
                  XOM=_us_doc(), CVX=cheap)
    first = out[0]
    assert first.startswith("Bottom line: dividend yield (highest first): CVX 4.00%, XOM 2.20%")
    assert "payout ratio (highest first): XOM 26%, CVX 26%" in first


def test_margin_trend_names_the_company_that_improved_more_and_says_it_is_no_forecast() -> None:
    flat = _us_doc()
    flat["us-gaap"]["OperatingIncomeLoss"] = _tag(_quarterly([20, 20, 22, 24, 26, 28, 30, 32]))
    out = _answer("Which has the better outlook for margins, Microsoft or Alphabet?",
                  MSFT=_us_doc(), GOOGL=flat)
    first = out[0]
    assert ("operating margin trend: MSFT 20.0% from 17.4% (+2.6 pts); "
            "GOOGL 20.0% from 18.9% (+1.1 pts)") in first
    assert "MSFT's improved the most" in first and "not a forecast" in first


def test_forward_pe_alone_does_not_add_the_trailing_pe() -> None:
    out = _answer("Compare Microsoft and Alphabet: operating margin, revenue growth and "
                  "forward P/E.", MSFT=_us_doc(), GOOGL=_us_doc())
    assert "forward P/E (lowest first)" in out[0] and "trailing P/E" not in "\n".join(out)
    assert "revenue growth (highest first): MSFT +27.5%" in out[0]


def test_dividends_or_buybacks_are_totalled_for_each_company() -> None:
    quiet = _us_doc()
    for tag in ("PaymentsOfDividendsCommonStock", "CommonStockDividendsPerShareDeclared",
                "PaymentsForRepurchaseOfCommonStock"):
        del quiet["us-gaap"][tag]
    out = sm.lines("How much has Circle paid out versus Coinbase in dividends or buybacks?",
                   facts_of=_facts(CRCL=quiet, COIN=_us_doc()), summary_of=lambda t: _yahoo(),
                   report_of=_no_report)
    assert out is not None
    assert out[0].startswith("Bottom line: dividends plus buybacks over the period (highest "
                             "first): COIN $63m, CRCL $0")


def test_an_unreadable_company_is_named_and_the_rest_still_answer() -> None:
    out = _answer("Compare Apple and Tesla dividend yield", AAPL=_us_doc(), TSLA=None)
    text = "\n".join(out)
    assert "Tesla (TSLA): its filings could not be read from SEC EDGAR just now." in text
    assert out[0].startswith("Bottom line: Apple (AAPL), four quarters to 30 Jun 2026")


def test_every_source_failing_gives_one_honest_line() -> None:
    out = sm.lines("What is the dividend yield of Apple?", facts_of=_facts(),
                   summary_of=lambda t: _yahoo(), report_of=_no_report)
    assert out is not None and out[0].startswith("Bottom line: Apple (AAPL)")
    assert "could not be read just now" in out[0] and out[-1].endswith("Not advice.")


def test_the_ifrs_answer_says_it_is_annual_and_what_is_not_tagged() -> None:
    out = sm.lines("Rank Shell and BP by dividend yield and payout ratio.",
                   facts_of=_facts(SHEL=_ifrs_doc(), BP=_ifrs_doc()),
                   summary_of=lambda t: _yahoo(price=10.0, rate=0.8 if t == "SHEL" else 0.4),
                   report_of=_no_report)
    assert out is not None
    text = "\n".join(out)
    assert "dividend yield (highest first): SHEL 8.00%, BP 4.00%" in out[0]
    assert "file a 20-F under IFRS" in text and "annual figures only" in text
    assert "does not tag gross profit" in text


def test_a_payout_over_three_times_profit_is_marked_not_useful() -> None:
    tiny = _ifrs_doc()
    tiny["ifrs-full"]["ProfitLossAttributableToOwnersOfParent"] = _tag([
        _row("2024-01-01", "2024-12-31", 90, "20-F"), _row("2025-01-01", "2025-12-31", 5, "20-F")])
    out = sm.lines("What is the payout ratio of BP?", facts_of=_facts(BP=tiny),
                   summary_of=lambda t: _yahoo(rate=0.4), report_of=_no_report)
    assert out is not None and "1,000%" in out[0] and "not a useful measure" in out[0]


# --- non-company assets -------------------------------------------------------------------------


def test_bitcoin_has_no_pe_and_no_dividend_and_that_is_the_answer() -> None:
    out = sm.lines("What is the P/E ratio and dividend yield of Bitcoin?")
    assert out is not None
    assert out[0] == ("Bottom line: Bitcoin has no earnings, so it has no P/E ratio and no EPS to "
                      "grow. Bitcoin pays no dividend, so its dividend yield is nil: nothing is "
                      "distributed to holders.")
    assert out[-1].startswith("Data: ") and out[-1].endswith("Not advice.")


def test_eps_growth_of_btc_and_pe_of_eth_get_the_same_stated_answer() -> None:
    btc = sm.lines("EPS growth of BTC")
    eth = sm.lines("What is the P/E of ETH?")
    assert btc is not None and "no earnings" in btc[0]
    assert eth is not None and "no earnings" in eth[0] and "Staking rewards" in eth[0]


def test_gold_pays_no_dividend() -> None:
    out = sm.lines("Does gold pay a dividend?")
    assert out is not None and "Gold pays no dividend" in out[0]


def test_a_crypto_asset_beside_a_company_is_answered_alongside_it() -> None:
    out = _answer("Compare Bitcoin and Apple P/E", AAPL=_us_doc())
    text = "\n".join(out)
    assert out[0].startswith("Bottom line: ") and "Apple (AAPL)" in text
    assert "Bitcoin has no earnings" in text


# --- what the reader takes and leaves ----------------------------------------------------------


@pytest.mark.parametrize("question", [
    "When does Apple report earnings?", "BTC price?", "P/E ratio of the S&P 500",
    "Why did Exxon cut its dividend?", "how many bp is 0.25%", "Shell company tax",
    "What was Apple's 10-Q gross margin and free cash flow",
    "What is the staking yield on ETH?", "Should I buy Tesla?"])
def test_questions_that_belong_to_other_readers_are_left_alone(question: str) -> None:
    assert sm.lines(question) is None and not sm.asks(question)


@pytest.mark.parametrize("question", [
    "Which has the better outlook for margins, Exxon or Chevron, and how do their dividend "
    "yields compare?",
    "Rank Exxon, Shell and BP by dividend yield and payout ratio.",
    "Does Microsoft have a buyback and how many shares has it retired over the last year?",
    "Compare Tesla and Ford on gross margin and free cash flow.",
    "Does Apple pay a dividend and how large is its buyback relative to market cap?",
    "What is the P/E ratio and dividend yield of Bitcoin?"])
def test_the_questions_this_reader_exists_for_are_taken(question: str) -> None:
    assert sm.asks(question)


def test_metrics_asked_are_only_those_named() -> None:
    assert sm.asked_metrics("dividend yield and payout ratio") == ["div", "payout"]
    assert sm.asked_metrics("how many shares has it retired") == ["buyback", "shares"]
    assert sm.asked_metrics("gross margin and free cash flow") == ["gross", "fcf"]
    assert sm.asked_metrics("operating margin, revenue growth and forward P/E") == [
        "operating", "growth", "fpe"]
    assert sm.asked_metrics("P/E") == ["pe", "fpe"]
    assert sm.asked_metrics("margins") == ["gross", "operating", "net"]
    assert sm.asked_metrics("compare fundamentals") == [
        "gross", "operating", "net", "growth", "pe", "fpe"]
    assert sm.asked_metrics("dividends or buybacks") == ["div", "buyback", "returned"]


def test_names_resolve_in_the_order_they_are_written() -> None:
    assert sm.resolve("Rank Exxon, Shell and BP by yield") == ["XOM", "SHEL", "BP"]
    assert sm.resolve("Microsoft vs Google, then Apple") == ["MSFT", "GOOGL", "AAPL"]
    assert sm.resolve("a 5 bp move in the shell game, circle back") == []
    assert sm.resolve("Compare Circle and Coinbase, and Ford") == ["CRCL", "COIN", "F"]


def test_more_than_four_companies_are_cut_and_said() -> None:
    names = "Apple, Microsoft, Tesla, Ford and Exxon"
    docs = {t: _us_doc() for t in ("AAPL", "MSFT", "TSLA", "F", "XOM")}
    out = _answer(f"Compare dividend yield of {names}", **docs)
    assert "Only the first 4 companies named are compared." in out


# --- the predecessor registrant -----------------------------------------------------------------


def test_company_document_merges_a_predecessor_registrants_facts(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def facts(**tags: float) -> dict[str, Any]:
        return {"facts": {"us-gaap": {tag: {"units": {"USD": [_row("2025-01-01", "2025-12-31", v)]}}
                                      for tag, v in tags.items()}}}

    served = {2115436: facts(Revenues=3), 34088: facts(Revenues=2, NetIncomeLoss=1)}

    class Edgar:
        def cik_for(self, ticker: str) -> int | None:
            return 2115436 if ticker == "XOM" else None

        def _get(self, url: str) -> Any:
            return served[int(url.rsplit("CIK", 1)[1].split(".")[0])]

    monkeypatch.setattr("argus.market.evidence.EdgarSource", Edgar)
    doc = filing_figures.company_document("XOM", (34088,))
    assert doc is not None
    assert [r["val"] for r in doc["us-gaap"]["Revenues"]["units"]["USD"]] == [3, 2]
    assert "NetIncomeLoss" in doc["us-gaap"]
    assert filing_figures.company_document("NOPE") is None
    assert sm.PREDECESSORS["XOM"] == (34088,)
