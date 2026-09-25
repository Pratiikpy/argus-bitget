"""Answer a numeric question about a 10-K from SEC's own XBRL — or say exactly why not.

**What this is for.** The T3 Information Extraction sub-theme asks how a system pulls a specific,
checkable number out of a filing. The approach every LLM benchmark measures is *read the prose*:
retrieve pages of the PDF, hand them to a model, parse its sentence. FinanceBench (Patronus AI,
arXiv 2311.11944) measured what that produces on exactly this question class, and its own
published transcripts show the failure that matters: a fluent, confident, wrong number.

This module takes the other route. It never reads prose. It resolves the question to

    (company → CIK, fiscal year(s) → one anchor 10-K, metric → a formula over statement lines)

and evaluates the formula on the numbers the company itself filed in XBRL
(:mod:`argus.market.statement_facts`). Every step can fail, and every failure is an **abstention
with a named reason** — an unknown company, a year with no 10-K, a metric that is not in the
library, a line the filing does not tag. There is no path from "could not resolve" to a number.

**The metric library is standard definitions, not a benchmark's answers.** Each derived metric is
the textbook formula (the operating cash flow ratio is cash from operations over current
liabilities; DPO is 365 times accounts payable over cost of goods sold; a margin is a line over
revenue), written once here with the variants that analysts genuinely disagree on made explicit
and chosen by what the question says:

* **Average or point-in-time denominators.** "Return on assets" is net income over total assets —
  or over the *average* of this year's and last year's. The question decides: if it says
  "average", both years are read from the anchor 10-K's own comparative column; if it does not,
  the year-end figure is used.
* **DPO on purchases.** When a question defines DPO over "COGS + change in inventory" (purchases),
  that is what is computed; otherwise plain COGS.

The ratio names and formulas were cross-checked against ``JerBouma/FinanceToolkit`` (MIT,
``research/repos-themed/JerBouma~FinanceToolkit/financetoolkit/ratios/``) — see
:data:`FORMULA_SOURCES` — and where the toolkit and a question's own stated definition differ, the
question's definition wins because it is the thing being asked.

**Scope, stated so nobody has to infer it.** Single-company, single-metric numeric questions whose
inputs are face-statement lines of a 10-K. Not segment data, not footnote tables, not MD&A prose,
not non-GAAP figures a company defines for itself, not comparisons across companies. Those are
abstentions, by design, and the evaluation counts them as abstentions rather than hiding them.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from argus.market.evidence import _UA
from argus.market.statement_facts import (
    COMPONENTS,
    Anchor,
    CompanyFacts,
    CompanyFactsSource,
    ConceptSource,
    FaceStatementSource,
    Line,
    Unresolved,
)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ENTITY_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?keysTyped={q}"
TIMEOUT = 30

FORMULA_SOURCES: dict[str, str] = {
    "current_ratio": "financetoolkit/ratios/liquidity_model.py get_current_ratio",
    "quick_ratio": "financetoolkit/ratios/liquidity_model.py get_quick_ratio",
    "cash_ratio": "financetoolkit/ratios/liquidity_model.py get_cash_ratio",
    "operating_cash_flow_ratio": "financetoolkit/ratios/liquidity_model.py "
                                 "get_operating_cash_flow_ratio",
    "working_capital": "financetoolkit/ratios/liquidity_model.py get_working_capital",
    "return_on_assets": "financetoolkit/ratios/profitability_model.py get_return_on_assets",
    "return_on_equity": "financetoolkit/ratios/profitability_model.py get_return_on_equity",
    "gross_margin": "financetoolkit/ratios/profitability_model.py get_gross_margin",
    "operating_margin": "financetoolkit/ratios/profitability_model.py get_operating_margin",
    "net_margin": "financetoolkit/ratios/profitability_model.py get_net_profit_margin",
    "effective_tax_rate": "financetoolkit/ratios/profitability_model.py get_effective_tax_rate",
    "inventory_turnover": "financetoolkit/ratios/efficiency_model.py get_inventory_turnover_ratio",
    "days_inventory": "financetoolkit/ratios/efficiency_model.py get_days_of_inventory_outstanding",
    "dso": "financetoolkit/ratios/efficiency_model.py get_days_of_sales_outstanding",
    "dpo": "financetoolkit/ratios/efficiency_model.py get_days_of_accounts_payable_outstanding",
    "cash_conversion_cycle": "financetoolkit/ratios/efficiency_model.py get_cash_conversion_cycle",
    "asset_turnover": "financetoolkit/ratios/efficiency_model.py get_asset_turnover_ratio",
    "fixed_asset_turnover": "financetoolkit/ratios/efficiency_model.py "
                            "get_fixed_asset_turnover",
    "interest_coverage": "financetoolkit/ratios/solvency_model.py get_interest_coverage_ratio",
    "dividend_payout_ratio": "financetoolkit/ratios/valuation_model.py get_payout_ratio",
    "retention_ratio": "1 - payout ratio (definition used by the question when it states one)",
}


# --- expressions ------------------------------------------------------------------------------


Getter = Callable[[str, int], Line]


@dataclass(frozen=True)
class Expr:
    """A formula over statement lines, evaluated for a base fiscal year.

    ``op`` is one of ``line``, ``const``, ``add``, ``sub``, ``mul``, ``div``, ``avg``, ``pow``,
    ``growth``. A ``line`` names a component and a year offset from the base year (0 = the year
    asked about, -1 = the year before).
    """

    op: str
    args: tuple[Expr, ...] = ()
    component: str = ""
    offset: int = 0
    const: float = 0.0

    def evaluate(self, get: Getter, year: int, used: list[Line]) -> float:
        if self.op == "line":
            line = get(self.component, year + self.offset)
            used.append(line)
            return line.value
        if self.op == "opt_line":
            # A line a formula may legitimately lack (marketable securities in a quick ratio):
            # a company whose balance sheet has no such line holds none of it. Recorded as a
            # zero-valued step so the answer shows the assumption rather than hiding it.
            try:
                line = get(self.component, year + self.offset)
            except Unresolved:
                return 0.0
            used.append(line)
            return line.value
        if self.op == "const":
            return self.const
        vals = [a.evaluate(get, year, used) for a in self.args]
        if self.op == "add":
            return sum(vals)
        if self.op == "sub":
            return vals[0] - vals[1]
        if self.op == "mul":
            out = 1.0
            for v in vals:
                out *= v
            return out
        if self.op == "div":
            if vals[1] == 0:
                raise Unresolved("division by a reported zero")
            return vals[0] / vals[1]
        if self.op == "avg":
            return sum(vals) / len(vals)
        if self.op == "pow":
            if vals[0] < 0:
                raise Unresolved("a growth base is negative; a compound rate is undefined")
            return float(vals[0] ** vals[1])
        raise ValueError(f"unknown op {self.op}")

    def render(self) -> str:
        if self.op in ("line", "opt_line"):
            mark = "?" if self.op == "opt_line" else ""
            return self.component + mark + ("" if self.offset == 0 else f"[t{self.offset:+d}]")
        if self.op == "const":
            return f"{self.const:g}"
        sym = {"add": " + ", "sub": " - ", "mul": " * ", "div": " / ", "pow": " ^ "}
        if self.op == "avg":
            return "avg(" + ", ".join(a.render() for a in self.args) + ")"
        return "(" + sym[self.op].join(a.render() for a in self.args) + ")"


def L(component: str, offset: int = 0) -> Expr:
    return Expr("line", component=component, offset=offset)


def O(component: str, offset: int = 0) -> Expr:  # noqa: E743
    """An optional line: zero when the filing has no such line (see ``opt_line``)."""
    return Expr("opt_line", component=component, offset=offset)


def K(value: float) -> Expr:
    return Expr("const", const=value)


def add(*a: Expr) -> Expr:
    return Expr("add", a)


def sub(a: Expr, b: Expr) -> Expr:
    return Expr("sub", (a, b))


def mul(*a: Expr) -> Expr:
    return Expr("mul", a)


def div(a: Expr, b: Expr) -> Expr:
    return Expr("div", (a, b))


def avg(*a: Expr) -> Expr:
    return Expr("avg", a)


def avg2(component: str) -> Expr:
    """This year's and last year's year-end balance, averaged."""
    return avg(L(component, 0), L(component, -1))


# --- metric library ---------------------------------------------------------------------------

Output = Literal["money", "ratio", "percent", "days", "per_share"]


@dataclass(frozen=True)
class Metric:
    """A named metric: the phrases that name it and the formula that computes it.

    ``formula`` takes the parsed question (so a stated definition can choose a variant) and
    returns an expression. ``output`` is the natural unit when the question does not ask for one.
    """

    name: str
    patterns: tuple[str, ...]
    formula: Callable[[ParsedQuestion], Expr]
    output: Output
    note: str = ""


def _averaged(q: ParsedQuestion, component: str) -> Expr:
    """The balance a flow-over-balance ratio divides by: averaged unless the question says not.

    FinanceToolkit's default for every such ratio is the average of the opening and closing
    balance (``efficiency_model.py:8-145``, ``profitability_model.py:179-240``), so that is the
    default here. The one exception is a question that states its own definition and does not
    say "average" — then the year-end balance it names is what it asked for.
    """
    if q.mentions_average:
        return avg2(component)
    if q.defines:
        return L(component)
    return avg2(component)


def _dpo(q: ParsedQuestion) -> Expr:
    purchases = (
        add(L("cogs"), sub(L("inventory"), L("inventory", -1)))
        if re.search(r"change in inventor", q.text) else L("cogs")
    )
    return div(mul(K(365), _averaged(q, "accounts_payable")), purchases)


def _dso(q: ParsedQuestion) -> Expr:
    return div(mul(K(365), _averaged(q, "accounts_receivable")), L("revenue"))


def _dio(q: ParsedQuestion) -> Expr:
    return div(mul(K(365), _averaged(q, "inventory")), L("cogs"))


DERIVED: tuple[Metric, ...] = (
    Metric("operating_cash_flow_ratio", (r"operating cash flow ratio",),
           lambda q: div(L("cfo"), L("current_liabilities")), "ratio"),
    Metric("cash_conversion_cycle", (r"cash conversion cycle", r"\bccc\b"),
           lambda q: sub(add(_dso(q), _dio(q)), _dpo(q)), "days"),
    Metric("dpo", (r"days payables? outstanding", r"\bdpo\b"), _dpo, "days"),
    Metric("dso", (r"days sales outstanding", r"\bdso\b"), _dso, "days"),
    Metric("days_inventory", (r"days inventory outstanding", r"\bdio\b",
                              r"days (of|in) inventory"), _dio, "days"),
    Metric("inventory_turnover", (r"inventory turnover",),
           lambda q: div(L("cogs"), _averaged(q, "inventory")), "ratio"),
    Metric("fixed_asset_turnover", (r"fixed asset turnover",),
           lambda q: div(L("revenue"), _averaged(q, "ppe_net")), "ratio"),
    Metric("asset_turnover", (r"(total )?asset turnover",),
           lambda q: div(L("revenue"), _averaged(q, "total_assets")), "ratio"),
    Metric("return_on_assets", (r"return on (total )?assets", r"\broa\b"),
           lambda q: div(L("net_income"), _averaged(q, "total_assets")), "ratio"),
    Metric("return_on_equity", (r"return on (shareholders'? |stockholders'? )?equity",
                                r"\broe\b"),
           lambda q: div(L("net_income"), _averaged(q, "total_equity")), "ratio"),
    Metric("quick_ratio", (r"quick ratio", r"acid[- ]test"),
           lambda q: div(add(L("cash"), O("short_term_investments"),
                             L("accounts_receivable")), L("current_liabilities")), "ratio"),
    Metric("cash_ratio", (r"\bcash ratio",),
           lambda q: div(add(L("cash"), O("short_term_investments")),
                         L("current_liabilities")), "ratio"),
    Metric("current_ratio", (r"current ratio", r"working capital ratio"),
           lambda q: div(L("current_assets"), L("current_liabilities")), "ratio"),
    Metric("working_capital", (r"(net )?working capital",),
           lambda q: sub(L("current_assets"), L("current_liabilities")), "money"),
    Metric("retention_ratio", (r"retention ratio", r"plowback ratio"),
           lambda q: sub(K(1), div(L("dividends_paid"), L("net_income"))), "ratio"),
    Metric("dividend_payout_ratio", (r"(dividend )?payout ratio",),
           lambda q: div(L("dividends_paid"), L("net_income")), "ratio"),
    Metric("interest_coverage", (r"interest coverage",),
           lambda q: div(L("operating_income"), L("interest_expense")), "ratio"),
    Metric("effective_tax_rate", (r"effective tax rate",),
           lambda q: div(L("income_tax"), L("pretax_income")), "percent"),
    Metric("free_cash_flow", (r"free cash flow", r"\bfcf\b"),
           lambda q: sub(L("cfo"), L("capex")), "money"),
    Metric("ebitda_margin", (r"ebitda (% )?margin", r"ebitda margin"),
           lambda q: div(add(L("operating_income"), L("dna")), L("revenue")), "percent"),
    Metric("ebitda", (r"\bebitda\b",),
           lambda q: add(L("operating_income"), L("dna")), "money"),
)
"""Checked in this order; the first whose phrase appears wins. Longer, more specific names come
before the shorter names they contain ("dividend payout ratio" before "dividends")."""

LINE_ITEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("capex", (r"capital expenditures?", r"\bcapex\b", r"purchases? of property")),
    ("ppe_net", (r"(net )?(ppne|pp&e|ppe)\b", r"(net )?property,? plant,? (and|&) equipment")),
    ("accounts_receivable", (r"accounts receivables?", r"trade receivables?", r"\bnet ar\b",
                             r"\bar\b")),
    ("accounts_payable", (r"accounts payables?", r"\bnet ap\b")),
    ("inventory", (r"inventor(y|ies)",)),
    ("cogs", (r"\bcogs\b", r"cost of goods sold", r"cost of sales", r"cost of revenues?")),
    ("gross_profit", (r"gross profit",)),
    ("operating_income", (r"operating (income|profit)",)),
    ("net_income", (r"net (income|earnings|profit)",)),
    ("pretax_income", (r"(pre-?tax income|income before (income )?taxes)",)),
    ("income_tax", (r"income tax expense", r"provision for income taxes")),
    ("interest_expense", (r"interest expense",)),
    ("sga", (r"sg&a", r"selling,? general,? and administrative")),
    ("rnd", (r"r&d", r"research and development")),
    ("dividends_paid", (r"dividends",)),
    ("buybacks", (r"(share|stock) repurchases?", r"buybacks?",
                  r"repurchases? of (common )?stock")),
    ("cfo", (r"cash (flows? )?from operations", r"cash flows? from operating activities",
             r"cash from operating activities", r"operating cash flows?",
             r"net cash (provided by|from|generated by) operating( activities)?",
             r"cash (generated|provided) (from|by) operati(ons|ng activities)")),
    ("dna", (r"depreciation (and|&) amortization", r"\bd&a\b")),
    ("total_assets", (r"total assets",)),
    ("total_liabilities", (r"total liabilities",)),
    ("current_assets", (r"current assets",)),
    ("current_liabilities", (r"current liabilities",)),
    ("cash", (r"cash and cash equivalents",)),
    ("total_equity", (r"(shareholders|stockholders)'? equity",)),
    ("long_term_debt", (r"long[- ]term debt",)),
    ("eps_diluted", (r"diluted (eps|earnings per share)",)),
    ("eps_basic", (r"basic (eps|earnings per share)",)),
    ("revenue", (r"revenues?", r"net sales", r"total sales")),
)
"""Single statement lines. Order matters only among names that contain one another."""

LINE_OUTPUT: dict[str, Output] = {"eps_diluted": "per_share", "eps_basic": "per_share"}

UNSUPPORTED_CUES: tuple[tuple[str, str, str], ...] = (
    # (pattern, why it is out of scope, pattern that lifts the objection)
    (r"\bgross (ppne|pp&e|property)", "gross PP&E is not a face-statement net line", ""),
    (r"\bsegments?\b", "segment data lives in dimensional XBRL this module does not read", ""),
    (r"\badjusted\b", "an adjusted (non-GAAP) figure is the company's own definition, not a "
                     "filed line", ""),
    (r"\bguidance\b", "guidance is prose, not a filed line item", ""),
    (r"\bper share\b", "a per-share figure other than EPS", r"\beps\b|earnings per share"),
    (r"\b(employees|headcount|stores|square feet|customers|subscribers)\b",
     "an operating statistic, not a financial statement line", ""),
    # FinanceBench's open questions, 2026-09-26: "Has CVS Health paid dividends to common
    # shareholders in Q2 of FY2022?" was answered with the full year's $2,907M, and a Q2-on-Q2
    # margin question with an annual figure. A quarter is read from a 10-Q, not this 10-K.
    (r"\bq[1-4]\b|\bquarters?\b|\bquarterly\b|\b(first|second) half\b|\b10-q\b",
     "a quarterly figure is filed in a 10-Q; this engine reads the annual 10-K", ""),
    # "What drove the increase in Ulta Beauty's merchandise inventories?" was answered with the
    # inventory balance, and "were there any events ... that increased net income" with net income.
    (r"\bwhat (drove|caused|contributed|explains?)\b|\bwhy (did|was|were|is|has|have)\b|"
     r"\bwere there any\b|\bany (potential )?events?\b|\breasons? (for|behind)\b|"
     r"\bdrivers? (of|behind)\b",
     "the question asks for a cause or an event, which a filing states in prose, not as a "
     "number", ""),
)


# --- question parsing -------------------------------------------------------------------------


@dataclass
class ParsedQuestion:
    text: str                    # normalised, lowercase
    raw: str
    years: tuple[int, ...]
    mentions_average: bool
    scale: float | None          # 1e6 for "USD millions", ...
    scale_label: str
    wants_percent: bool
    decimals: int | None
    span_average: int | None     # "3 year average" -> 3
    growth: str | None           # "growth" | "cagr" | None
    defines: bool = False        # the question states its own formula

    def average_denominator(self) -> bool:
        """"average total assets between FY2021 and FY2022" averages a balance, not a metric."""
        return bool(re.search(
            r"average (total |net )?(assets|inventor|accounts|equity|ppne|pp&e|property|fixed|"
            r"receivable|payable|balance)", self.text))


_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "zero": 0}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'").replace("\u2013", "-")
    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_question(raw: str) -> ParsedQuestion:
    text = _norm(raw)
    years = sorted({int(y if len(y) == 4 else "20" + y)
                    for y in re.findall(r"\bfy\s?'?(\d{4}|\d{2})\b", text)})
    if not years:
        years = sorted({int(y) for y in re.findall(r"\bfiscal (?:year )?(\d{4})\b", text)})
    if not years:
        years = sorted({int(y) for y in re.findall(r"\b(199\d|20[0-3]\d)\b", text)})
    scale, label = None, ""
    for pattern, s, name in (
        (r"\b(usd|\$) ?thousands?\b|\bin thousands\b", 1e3, "USD thousands"),
        (r"\b(usd|\$) ?millions?\b|\bin millions\b|\busd ?mm\b", 1e6, "USD millions"),
        (r"\b(usd|\$) ?billions?\b|\bin billions\b|\busd ?bn\b", 1e9, "USD billions"),
    ):
        if re.search(pattern, text):
            scale, label = s, name
            break
    wants_percent = bool(re.search(r"%|percent|\bin units of percents?\b", text))
    decimals = None
    m = re.search(r"round(?:ed)?(?: your answer| answer| the answer)? to (\w+) decimal", text)
    if m:
        word = m.group(1)
        decimals = int(word) if word.isdigit() else _WORDS.get(word)
    span = None
    m = re.search(r"\b(\d+|one|two|three|four|five)[- ]year average\b", text)
    if m:
        word = m.group(1)
        span = int(word) if word.isdigit() else _WORDS.get(word)
    growth = None
    if re.search(r"\bcagr\b|compound(ed)? annual growth", text):
        growth = "cagr"
    elif re.search(r"growth rate|\bgrowth\b|% change|percent(age)? change|\byoy\b"
                   r"|year[- ]over[- ]year", text):
        growth = "growth"
    return ParsedQuestion(
        text=text, raw=raw, years=tuple(years),
        mentions_average=bool(re.search(r"\baverage\b", text)) and span is None,
        scale=scale, scale_label=label, wants_percent=wants_percent, decimals=decimals,
        span_average=span, growth=growth,
        defines=bool(re.search(r"defined as|\bdefine\b|calculated as|using the formula|"
                               r"\bis computed as\b", text)),
    )


# --- company resolution -----------------------------------------------------------------------

_SUFFIXES = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "plc", "ltd", "limited", "llc",
    "lp", "holdings", "holding", "group", "the", "sa", "nv", "ag", "se", "de", "cv", "sab", "ny",
    "new", "com", "del", "md", "nj", "pa", "tx", "va",
})
_QUESTION_WORDS = frozenset({
    "what", "how", "when", "where", "which", "who", "assume", "answer", "according", "based",
    "using", "we", "you", "please", "give", "compute", "respond", "provide", "address",
    "approach", "calculate", "base", "here", "is", "does", "did", "has", "was", "were", "a", "an",
    "in", "on", "for", "by", "if", "as", "of", "and", "or", "to", "from", "with", "that", "this",
    "total", "net", "year", "fiscal", "statement", "income", "cash", "balance", "sheet", "flow",
    "financial", "investment", "equity", "public", "analyst", "banker", "question", "only",
})
_ACRONYMS = frozenset({
    "FY", "USD", "COGS", "EBITDA", "EBIT", "DPO", "DSO", "DIO", "ROA", "ROE", "ROIC", "AR", "AP",
    "PPE", "PPNE", "EPS", "CAGR", "GAAP", "SEC", "US", "USA", "CEO", "CFO", "IT", "PNL", "TTM",
    "LTM", "NWC", "FCF", "OCF", "AI", "ESG", "MD", "NA", "UK", "EU", "IPO", "EV", "YOY", "QOQ",
    "BPS", "CCC", "SGA", "RND", "DNA", "OK", "LLC", "INC", "CO", "PLC", "CORP", "Q", "K", "II",
    "III", "IV", "MM", "BN", "NYSE", "NASDAQ", "ETF", "REIT", "ADR", "GDP", "CPI", "FOMC",
})


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("&", " and ")
    return re.findall(r"[a-z0-9]+", text)


def _name_tokens(title: str) -> tuple[str, ...]:
    toks = _tokens(title)
    while toks and toks[-1] in _SUFFIXES:
        toks.pop()
    while toks and toks[0] == "the":
        toks.pop(0)
    return tuple(toks)


@dataclass(frozen=True)
class Company:
    cik: int
    name: str
    ticker: str
    matched: str
    how: str


class CompanyResolver:
    """Question text → one SEC registrant, or an abstention naming the ambiguity.

    Two sources, both SEC's own and keyless: ``company_tickers.json`` (current registrants, with
    titles) and EDGAR's entity-name index for registrants that have since been acquired or
    delisted (Activision Blizzard, Foot Locker), whose filings are still on EDGAR.
    """

    def __init__(self, *, user_agent: str = _UA, snapshot: Path | None = None,
                 offline: bool = False) -> None:
        self._ua = user_agent
        self._snapshot = snapshot
        self._offline = offline
        self._titles: list[tuple[int, str, str, tuple[str, ...]]] | None = None
        self._prefix_counts: dict[tuple[str, ...], set[int]] = {}
        self._tickers: dict[str, tuple[int, str]] = {}
        self._search_cache: dict[str, list[tuple[int, str]]] = {}
        self.requests = 0

    def _get(self, url: str) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": self._ua,
                                                   "Accept": "application/json"})
        self.requests += 1
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())

    def _load(self) -> None:
        if self._titles is not None:
            return
        data: dict[str, Any] | None = None
        if self._snapshot is not None and self._snapshot.exists():
            snap = json.loads(self._snapshot.read_text(encoding="utf-8"))
            data = snap.get("tickers")
            self._search_cache = {k: [(int(c), n) for c, n in v]
                                  for k, v in (snap.get("search") or {}).items()}
        if data is None:
            if self._offline:
                raise Unresolved("offline and no company snapshot")
            data = self._get(TICKERS_URL)
        titles: list[tuple[int, str, str, tuple[str, ...]]] = []
        seen: set[int] = set()
        for row in data.values():
            cik, ticker, title = int(row["cik_str"]), str(row["ticker"]), str(row["title"])
            self._tickers.setdefault(ticker.upper(), (cik, title))
            if cik in seen:
                continue
            seen.add(cik)
            toks = _name_tokens(title)
            if toks:
                titles.append((cik, ticker, title, toks))
        self._titles = titles
        for cik, _t, _n, toks in titles:
            for k in range(1, len(toks) + 1):
                self._prefix_counts.setdefault(toks[:k], set()).add(cik)
        self._raw_tickers = data

    def snapshot(self) -> dict[str, Any]:
        """What this resolver read, so an evaluation can replay it offline."""
        self._load()
        return {"tickers": self._raw_tickers,
                "search": {k: [[c, n] for c, n in v] for k, v in self._search_cache.items()}}

    def _search(self, span: str) -> list[tuple[int, str]]:
        key = span.lower()
        if key in self._search_cache:
            return self._search_cache[key]
        if self._offline:
            return []
        try:
            data = self._get(ENTITY_SEARCH_URL.format(q=urllib.parse.quote(key)))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return []
        hits = [(int(h["_id"]), str(h["_source"].get("entity", "")))
                for h in (data.get("hits") or {}).get("hits", [])[:5]]
        self._search_cache[key] = hits
        return hits

    def resolve(self, raw: str) -> Company:
        self._load()
        assert self._titles is not None
        q_tokens = _tokens(raw)
        q_text = " " + " ".join(q_tokens) + " "
        # Capitalisation of each token as written, so "Block" the company is not "block" the word.
        written = re.findall(r"[A-Za-z0-9&]+", unicodedata.normalize("NFKC", raw))
        capitalised = {w.lower() for w in written if w[:1].isupper() or w[:1].isdigit()}

        best: list[tuple[int, int, int, Company]] = []   # (score, full, -order, company)
        for order, (cik, ticker, title, toks) in enumerate(self._titles):
            for k in range(len(toks), 0, -1):
                span = toks[:k]
                if f" {' '.join(span)} " not in q_text:
                    continue
                full = k == len(toks)
                unique = len(self._prefix_counts.get(span, set())) == 1
                if not (full or unique):
                    break
                if not all(t in capitalised or t.isdigit() or any(c.isdigit() for c in t)
                           for t in span):
                    break
                if k == 1 and span[0] in _QUESTION_WORDS:
                    break
                # A number is not a name. "3 year average" once resolved to a registrant titled
                # "3 E Network Technology Group" through the unique prefix "3". A match must carry
                # a letter, and a prefix (not the whole title) must carry a real word.
                if not any(any(c.isalpha() for c in t) for t in span):
                    break
                if not full and not any(sum(c.isalpha() for c in t) >= 3 for t in span):
                    break
                best.append((k, int(full), -order,
                             Company(cik, title, ticker, " ".join(span),
                                     "name" if full else "unique name prefix")))
                break
        for word in written:
            if word.isupper() and 2 <= len(word) <= 5 and word not in _ACRONYMS \
                    and word in self._tickers:
                cik, title = self._tickers[word]
                best.append((1, 1, 0, Company(cik, title, word, word, "ticker")))
        if best:
            best.sort(key=lambda b: (b[0], b[1], b[2]), reverse=True)
            top = best[0]
            rivals = {b[3].cik for b in best if (b[0], b[1]) == (top[0], top[1])}
            if len(rivals) > 1:
                names = sorted({b[3].name for b in best if b[3].cik in rivals})
                raise Unresolved(f"the question names more than one registrant: {names}")
            return top[3]
        # Not a current registrant: EDGAR's entity index still has acquired/delisted filers.
        spans = re.findall(r"\b([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*)", raw)
        for span in sorted(spans, key=len, reverse=True):
            s_toks = _name_tokens(span)
            s_toks = tuple(t for t in s_toks if t not in _QUESTION_WORDS) if len(s_toks) > 1 \
                else s_toks
            if not s_toks or (len(s_toks) == 1 and s_toks[0] in _QUESTION_WORDS):
                continue
            if all(t.upper() in _ACRONYMS or re.fullmatch(r"fy\d+|\d+", t) for t in s_toks):
                continue
            for cik, entity in self._search(" ".join(s_toks)):
                if _name_tokens(entity)[: len(s_toks)] == s_toks:
                    return Company(cik, entity, "", " ".join(s_toks), "EDGAR entity index")
        raise Unresolved("no SEC registrant named in the question")


# --- answering --------------------------------------------------------------------------------


@dataclass
class Answer:
    question: str
    status: Literal["answered", "abstained"]
    value: float | None = None
    unit: str = ""
    text: str = ""
    reason: str = ""
    company: str = ""
    cik: int | None = None
    metric: str = ""
    formula: str = ""
    fiscal_years: tuple[int, ...] = ()
    anchor_accn: str = ""
    lines: list[Line] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "text": self.text,
            "reason": self.reason,
            "company": self.company,
            "cik": self.cik,
            "metric": self.metric,
            "formula": self.formula,
            "fiscal_years": list(self.fiscal_years),
            "anchor_accn": self.anchor_accn,
            "lines": [ln.as_dict() for ln in self.lines],
            "steps": self.steps,
        }


# --- understanding a question -----------------------------------------------------------------

Mode = Literal["point", "growth", "cagr", "span_mean", "change"]


@dataclass
class Understanding:
    """What a question asks for, and which characters of it that account covers.

    ``consumed`` is the heart of the safety argument. A parse that matches *part* of a question —
    "capex" inside "capex as a % of revenue", "working capital" inside "working capital ratio",
    "EBITDA" inside "EBITDA less capex" — returns a real, correctly-extracted, wrong answer. The
    frozen v1 of this module did exactly that on six held-out FinanceBench questions. So every
    phrase that changes the arithmetic (see :data:`OPERATOR_CUES`) must fall inside a span some rule
    actually used, or the question is abstained on and the unused phrase is named.
    """

    name: str
    expr: Expr
    output: Output
    mode: Mode = "point"
    source: str = "library"
    consumed: list[tuple[int, int]] = field(default_factory=list)
    definitions: dict[str, str] = field(default_factory=dict)


OPERATOR_CUES: tuple[str, ...] = (
    r"\bas an? (%|percent(age)?|share|proportion|fraction) of\b",
    r"%\s*of\b",
    r"\bpercent(age)? of\b",
    r"\bless\b",
    r"\bminus\b",
    r"\bplus\b",
    r"\bdivided by\b",
    r"\bratio\b",
    # "changed", "compare", "grow": the comparison words a question uses without the word
    # "change" itself. Unconsumed, each one meant a single year answered for a two-year question.
    r"\bchang(?:e|ed|es|ing)\b",
    r"\bcompar(?:e|ed|es|ing|ison)\b",
    r"\bincreas(?:e|ed|es|ing)\b",
    r"\bdecreas(?:e|ed|es|ing)\b",
    r"\b(?:grow|grew|grown)\b",
    r"\bdeclin(?:e|ed|es|ing)\b",
    r"\bdifference\b",
    r"\bmargin\b",
    r"\bturnover\b",
    r"\bcagr\b",
    r"\bgrowth\b",
    r"\baverage\b",
    r"\bper\b",
    r"\bexcluding\b",
    r"\bnet of\b",
    r"\bmultiple\b",
    r"\byield\b",
    r"\bcoverage\b",
    r"\breturn on\b",
    r"\bdays\b",
    r"\bcombined\b",
    r"\btotal of\b",
)
"""Phrases that change what is computed. An unconsumed one is an abstention, never a guess."""

_FILLER = frozenset({
    "total", "net", "the", "a", "an", "of", "from", "in", "on", "reported", "unadjusted", "gaap",
    "company", "company's", "companys", "annual", "year", "yearly", "fiscal", "end", "year-end",
    "consolidated", "statement", "statements", "cash", "flow", "flows", "income", "balance",
    "sheet", "p&l", "amount", "figure", "value", "line", "item", "as", "shown", "its", "their",
    "for", "at", "and", "amounts", "balances", "position", "financial", "operations",
})

_DEF_PATTERNS: tuple[str, ...] = (
    r"\b(?:we |please )?define (?P<name>[a-z0-9&%()'/ -]+?) as:? (?P<body>.+?)(?=\.(?:\s|$)|\?|$)",
    r"(?P<name>[a-z0-9&%()'/ -]+?) (?:is|are) (?:defined|calculated|computed) as:? "
    r"(?P<body>.+?)(?=\.(?:\s|$)|\?|$)",
)

_TOKEN_SPLIT = re.compile(
    r"(\s/\s|/|\bdivided by\b|\s\*\s|\*|\btimes\b|\bmultiplied by\b|\s\+\s|\+|\bplus\b"
    r"|\bminus\b|\bless\b|\s-\s|\(|\))"
)


def _shift(expr: Expr, k: int) -> Expr:
    """The same formula ``k`` fiscal years later (negative: earlier)."""
    if expr.op in ("line", "opt_line"):
        return Expr(expr.op, component=expr.component, offset=expr.offset + k)
    if expr.op == "const":
        return expr
    return Expr(expr.op, tuple(_shift(a, k) for a in expr.args))


def _norm_name(text: str) -> str:
    words = re.findall(r"[a-z0-9&%]+", text.lower())
    return " ".join(w for w in words if w not in {"the", "a", "an"})


def _match_line(phrase: str) -> tuple[str, Expr, Output] | None:
    """A phrase that *is* one statement line or one library metric, with nothing left over.

    "Nothing left over" means every word outside the matched name is filler ("total", "the",
    "reported", "from the cash flow statement"). "Capex as a % of revenue" is not the line
    "capex": "% of revenue" is not filler, so this returns None and the caller must account for it.
    """
    text = phrase.strip(" .,:;").lower()
    best: tuple[int, str, Expr, Output] | None = None
    candidates: list[tuple[str, tuple[str, ...], Expr, Output]] = []
    for metric in DERIVED:
        candidates.append((metric.name, metric.patterns, metric.formula(_DEFAULT_Q),
                           metric.output))
    for name, patterns in LINE_ITEMS:
        candidates.append((name, patterns, L(name), LINE_OUTPUT.get(name, "money")))
    for name, patterns, expr, output in candidates:
        for p in patterns:
            found = re.search(p, text)
            if found is None:
                continue
            rest = (text[:found.start()] + " " + text[found.end():]).replace("-", " ")
            leftover = [w for w in re.findall(r"[a-z0-9&'%]+", rest) if w not in _FILLER]
            if leftover:
                continue
            length = found.end() - found.start()
            if best is None or length > best[0]:
                best = (length, name, expr, output)
    if best is None:
        return None
    return best[1], best[2], best[3]


_TERM_NOISE = r"\b(unadjusted|reported|total|the)\b"


class Terms:
    """The question's own definitions, resolved in any order.

    FinanceBench's cash-conversion-cycle question defines CCC as "DIO + DSO - DPO" *before* it
    defines DIO, DSO and DPO. Parsing definitions in reading order resolved those three names
    against the library's defaults — a DPO without the purchases adjustment the question asked
    for — and returned a real, wrong number. Definitions are therefore collected first and
    resolved on demand, with a cycle check.
    """

    def __init__(self, raw: dict[str, str], target: int) -> None:
        self.raw = raw
        self.target = target
        self.parsed: dict[str, Expr] = {}
        self._resolving: set[str] = set()

    @staticmethod
    def key(text: str) -> str:
        return _norm_name(re.sub(_TERM_NOISE, " ", text))

    def names_for(self, name: str) -> set[str]:
        """The keys a defined name answers to: itself, and a parenthesised abbreviation."""
        keys = {self.key(re.sub(r"\([^)]*\)", " ", name)), self.key(name)}
        short = re.search(r"\(([a-z&]{2,6})\)", name)
        if short:
            keys.add(short.group(1))
        return {k for k in keys if k}

    def lookup(self, text: str) -> Expr | None:
        key = self.key(text)
        for name in self.raw:
            if key in self.names_for(name):
                return self.resolve(name)
        return None

    def resolve(self, name: str) -> Expr:
        if name in self.parsed:
            return self.parsed[name]
        if name in self._resolving:
            raise Unresolved(f"the question's definition of '{name}' refers to itself")
        self._resolving.add(name)
        try:
            self.parsed[name] = parse_definition(self.raw[name], self.target, self)
        finally:
            self._resolving.discard(name)
        return self.parsed[name]


def _phrase(tok: str, target: int, terms: Terms | None) -> Expr:
    """One operand of a stated definition: a line, a defined term, an average or a change."""
    p = re.sub(r"\s+", " ", tok.strip(" .,:;")).lower()
    m = re.fullmatch(r"(?:the )?average (?:of )?(?P<x>.+?) (?:between|from) fy\s?(?P<a>\d{4}) "
                     r"(?:and|to|-) fy\s?(?P<b>\d{4})", p)
    if m:
        return avg(_operand(m.group("x"), int(m.group("a")), target, terms),
                   _operand(m.group("x"), int(m.group("b")), target, terms))
    m = re.fullmatch(r"(?:the )?change in (?P<x>.+?) (?:between|from) fy\s?(?P<a>\d{4}) "
                     r"(?:and|to|-) fy\s?(?P<b>\d{4})", p)
    if m:
        return sub(_operand(m.group("x"), int(m.group("b")), target, terms),
                   _operand(m.group("x"), int(m.group("a")), target, terms))
    m = re.fullmatch(r"(?:the )?average (?:of )?(?P<x>.+)", p)
    if m:
        inner = _operand(m.group("x"), target, target, terms)
        return avg(inner, _shift(inner, -1))
    m = re.fullmatch(r"(?:the )?fy\s?(?P<y>\d{4}) (?P<x>.+)", p)
    if m:
        return _operand(m.group("x"), int(m.group("y")), target, terms)
    m = re.fullmatch(r"(?P<x>.+?) (?:in|for|of) fy\s?(?P<y>\d{4})", p)
    if m:
        return _operand(m.group("x"), int(m.group("y")), target, terms)
    return _operand(p, target, target, terms)


def _operand(text: str, year: int, target: int, terms: Terms | None) -> Expr:
    if terms is not None:
        defined = terms.lookup(text)
        if defined is not None:
            return _shift(defined, year - target)
    found = _match_line(text)
    if found is None:
        raise Unresolved(f"'{text.strip()}' is not a statement line this module reads")
    return _shift(found[1], year - target)


def parse_definition(body: str, target: int, terms: Terms | None = None) -> Expr:
    """A question's own stated formula, as an expression. Raises :class:`Unresolved` if any part
    of it is not understood — a half-read definition is never evaluated."""
    body = re.sub(r"\[[^\]]*\]", " ", body)
    body = re.sub(r"\((?:from|per|in|on) (?:the )?[a-z &]*statements?\)", " ", body)
    body = re.sub(r"\b(from|in|on) the (cash flow|income|balance sheet|p&l)[a-z ]*statement\b",
                  " ", body)
    tokens = [t.strip() for t in _TOKEN_SPLIT.split(body) if t and t.strip()]
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def take() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise Unresolved(f"the stated definition '{body.strip()}' ends early")
        tok = tokens[pos]
        pos += 1
        return tok

    def expression() -> Expr:
        left = term()
        while peek() in ("+", "plus", "-", "minus", "less"):
            op = take()
            right = term()
            left = add(left, right) if op in ("+", "plus") else sub(left, right)
        return left

    def term() -> Expr:
        left = factor()
        while peek() in ("*", "times", "multiplied by", "/", "divided by"):
            op = take()
            right = factor()
            left = mul(left, right) if op in ("*", "times", "multiplied by") else div(left, right)
        return left

    def factor() -> Expr:
        tok = take()
        if tok == "(":
            inner = expression()
            if take() != ")":
                raise Unresolved(f"unbalanced parentheses in '{body.strip()}'")
            return inner
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            return K(float(tok))
        if tok in ("+", "plus", "-", "minus", "less", "*", "times", "/", "divided by", ")"):
            raise Unresolved(f"the stated definition '{body.strip()}' is not a formula")
        return _phrase(tok, target, terms)

    expr = expression()
    if pos != len(tokens):
        raise Unresolved(f"could not read all of the stated definition '{body.strip()}'")
    return expr


def _infer_output(name: str, expr: Expr) -> Output:
    """The unit a defined term is in: the library metric it names, else what its name says."""
    alias = _match_line(re.sub(r"\([^)]*\)", " ", name)) or _match_line(name)
    if alias is not None:
        return alias[2]
    if re.search(r"margin|%|percent|rate\b|yield", name):
        return "percent"
    if re.search(r"\bdays?\b|\bdpo\b|\bdso\b|\bdio\b|outstanding|cycle", name):
        return "days"
    if expr.op in ("line", "add", "sub", "opt_line"):
        return "money"
    return "ratio"


def _outside(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return not any(a <= span[0] and span[1] <= b for a, b in spans)


_GROWTH_RE = (r"(?:year[- ]over[- ]year|\byoy\b|y/y|annual|annualized)\s+(?:% |percent(?:age)? )?"
              r"(?:growth(?: rate)?|change|increase)|growth rate|\bgrowth\b|% change"
              r"|percent(?:age)? change")
_MARGIN_AFTER = r"\s*(?:\([^)]*\)\s*)?(?:%\s*)?margin\b"


def understand(q: ParsedQuestion) -> Understanding:
    """Question → formula, with every arithmetic phrase accounted for. Raises Unresolved."""
    text = q.text
    target = max(q.years) if q.years else 0
    consumed: list[tuple[int, int]] = []
    raw: dict[str, str] = {}

    # 1. The question's own definitions win over any library formula. Collected first, resolved
    #    on demand (see Terms), so a definition may use a name defined after it.
    for pattern in _DEF_PATTERNS:
        for m in re.finditer(pattern, text):
            if not _outside(m.span(), consumed):
                continue
            name = m.group("name").strip()
            name = re.sub(r"^.*\b(?:define|and|where|with|using)\s+", "", name).strip(" :,")
            name = re.sub(r"^(?:the|an?|we|please)\s+", "", name).strip()
            raw.setdefault(name, m.group("body").strip())
            consumed.append(m.span())
    terms = Terms(raw, target)
    for name in raw:
        terms.resolve(name)

    und: Understanding | None = None
    # 2. Base metric: a defined term used elsewhere in the question, else the library.
    for name in sorted(raw, key=len, reverse=True):
        probes = sorted({name, *terms.names_for(name)}, key=len, reverse=True)
        for probe in probes:
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(probe)}(?![a-z0-9])", text):
                if _outside(m.span(), consumed):
                    und = Understanding(name, terms.parsed[name],
                                        _infer_output(name, terms.parsed[name]),
                                        source="definition", consumed=[m.span()])
                    after = re.match(_MARGIN_AFTER, text[m.end():])
                    if after is not None and und.output == "money":
                        und.expr = div(und.expr, L("revenue"))
                        und.name, und.output = f"{name}_margin", "percent"
                        und.consumed.append((m.end(), m.end() + after.end()))
                    break
            if und is not None:
                break
        if und is not None:
            break
    if und is None and len(raw) == 1:
        name = next(iter(raw))
        und = Understanding(name, terms.parsed[name], _infer_output(name, terms.parsed[name]),
                            source="definition", consumed=[])
    if und is None:
        und = _library(q, consumed)
    und.definitions = dict(raw)
    und.consumed.extend(consumed)
    if und.source == "definition":
        # The library's own name for the same defined metric ("days payable outstanding" beside
        # "DPO") is the same metric mentioned twice, not a second one.
        alias = _match_line(re.sub(r"\([^)]*\)", " ", und.name)) or _match_line(und.name)
        if alias is not None:
            patterns = next((mt.patterns for mt in DERIVED if mt.name == alias[0]),
                            dict(LINE_ITEMS).get(alias[0], ()))
            for p in patterns:
                for m in re.finditer(p, text):
                    if _outside(m.span(), und.consumed):
                        und.consumed.append(m.span())

    # 3. Modifiers that compose the base with more statement lines.
    pct = re.search(r"(?:as an? )?(?:%|percent(?:age)?|share|proportion) of (?:total |net )?"
                    r"(?:revenues?|sales|net sales)\b", text)
    if pct and _outside(pct.span(), und.consumed):
        if und.output in ("ratio", "percent", "days"):
            raise Unresolved("a ratio expressed as a % of revenue is not a defined quantity")
        if und.consumed and all(pct.start() <= a and b <= pct.end() for a, b in und.consumed):
            # "wages expense as a percent of net sales": no library line matched the numerator,
            # so the only match was "net sales" inside the denominator, and the answer was
            # revenue / revenue = 100%. The numerator has to be something the filing tags.
            before = text[max(0, pct.start() - 40): pct.start()].strip()
            raise Unresolved(f"the quantity taken as a share of revenue (...{before}) is not a "
                             f"line this engine reads")
        und.expr = div(und.expr, L("revenue"))
        und.name = f"{und.name}_to_revenue"
        und.output = "percent"
        und.consumed.append(pct.span())
    for m in re.finditer(r"\b(less|minus|plus)\s+(?P<y>[a-z0-9&'() -]+?)(?=\s+(?:for|of|in|at|"
                         r"during|between|from|by|according)\b|[?,.;]|$)", text):
        if not _outside(m.span(), und.consumed):
            continue
        rhs = _operand(m.group("y"), target, target, terms)
        und.expr = add(und.expr, rhs) if m.group(1) == "plus" else sub(und.expr, rhs)
        und.name = f"{und.name}_{m.group(1)}_{_norm_name(m.group('y')).replace(' ', '_')}"
        und.consumed.append(m.span())

    # 4. How the years combine.
    is_ratio = und.output in ("ratio", "percent", "days")
    if q.growth == "cagr":
        gm = re.search(r"\bcagr\b|compound(?:ed)? annual growth(?: rate)?", text)
        und.mode = "cagr"
        if gm:
            und.consumed.append(gm.span())
    else:
        gm = re.search(_GROWTH_RE, text)
        if gm and _outside(gm.span(), und.consumed):
            und.mode = "growth"
            und.consumed.append(gm.span())
    # A comparison asked in other words: "how much has the effective tax rate changed between
    # FY2021 and FY2022", "how does it compare to FY2021", "did Pfizer grow its PP&E between FY20
    # and FY21", "increase or decrease in FY2023". Answered as the change, with both endpoints.
    comparison = (r"\bchang(?:ed|es)\b|\bchange\b(?!\s+in\b)|\bcompar(?:e|ed|es|ing|ison)\b|"
                  r"\bincrease or decrease\b|"
                  r"\b(?:grow|grew|grown)\b|\bincreas(?:e|ed)\b|\bdecreas(?:e|ed)\b|"
                  r"\bdeclin(?:e|ed)\b")
    cues = [m for m in re.finditer(comparison, text) if _outside(m.span(), und.consumed)]
    if und.mode == "point" and cues and not re.search(r"\bchange in\b", text):
        und.mode = "change"
        und.consumed.extend(m.span() for m in cues)
    cm = re.search(r"\bchange in\b", text)
    if und.mode == "point" and cm and _outside(cm.span(), und.consumed):
        if len(q.years) < 2:
            raise Unresolved("a change needs two fiscal years")
        if is_ratio:
            und.mode = "change"          # a change in a ratio is a difference (points)
        elif q.wants_percent:
            und.mode = "growth"          # "change in revenue, in percent" is relative growth
        elif q.scale is not None:
            und.mode = "change"          # "change in capex, in USD millions" is a difference
        else:
            raise Unresolved("'change in' a money amount with no unit asked is ambiguous: a "
                             "difference and a percentage change are both called change")
        und.consumed.append(cm.span())
    sm = re.search(r"\b(\d+|one|two|three|four|five)[- ]year average\b", text)
    if sm and _outside(sm.span(), und.consumed):
        und.mode = "span_mean"
        und.consumed.append(sm.span())
    elif und.mode == "point" and len(q.years) >= 2:
        am = re.search(r"\baverage\b", text)
        if am and _outside(am.span(), und.consumed) and not q.average_denominator():
            und.mode = "span_mean"
            und.consumed.append(am.span())
    if und.mode in ("growth", "cagr") and is_ratio:
        raise Unresolved(
            f"growth of a ratio ({und.name}) is ambiguous: a percentage-point change and a "
            f"relative change are both called growth"
        )

    # 5. Every arithmetic phrase must have been used by something above.
    for cue in OPERATOR_CUES:
        for m in re.finditer(cue, text):
            if _outside(m.span(), und.consumed):
                window = text[max(0, m.start() - 30): m.end() + 30]
                raise Unresolved(
                    f"the question says '{m.group(0)}' (...{window}...), which the parsed "
                    f"formula {und.expr.render()} does not use"
                )
    return und


_AVERAGING = frozenset({
    "dpo", "dso", "days_inventory", "cash_conversion_cycle", "inventory_turnover",
    "fixed_asset_turnover", "asset_turnover", "return_on_assets", "return_on_equity",
})


def _library(q: ParsedQuestion, consumed: list[tuple[int, int]]) -> Understanding:
    text = q.text
    for metric in DERIVED:
        for p in metric.patterns:
            m = re.search(p, text)
            if m and _outside(m.span(), consumed):
                spans = [m.span()]
                if metric.name in _AVERAGING and q.mentions_average:
                    spans.extend(a.span() for a in re.finditer(r"\baverage\b", text))
                return Understanding(metric.name, metric.formula(q), metric.output,
                                     consumed=spans)
    for name, patterns in LINE_ITEMS:
        if name == "revenue":
            continue
        for p in patterns:
            m = re.search(p + r"\s*(\([^)]*\)\s*)?(%\s*)?margin", text)
            if m and _outside(m.span(), consumed):
                return Understanding(f"{name}_margin", div(L(name), L("revenue")), "percent",
                                     consumed=[m.span()])
    m = re.search(r"\bgross (profit )?margin", text)
    if m:
        return Understanding("gross_margin", div(sub(L("revenue"), L("cogs")), L("revenue")),
                             "percent", consumed=[m.span()])
    for name, patterns in LINE_ITEMS:
        for p in patterns:
            m = re.search(p, text)
            if m and _outside(m.span(), consumed):
                return Understanding(name, L(name), LINE_OUTPUT.get(name, "money"),
                                     source="line", consumed=[m.span()])
    raise Unresolved("no metric in the library matches the question")


def _format(value: float, output: Output, q: ParsedQuestion) -> tuple[float, str, str]:
    """Scale to the unit the question asked for, round as asked, and render."""
    if output == "money":
        scale = q.scale or 1e6
        label = q.scale_label or "USD millions"
        v = value / scale
        d = q.decimals if q.decimals is not None else 2
        return round(v, d), label, f"${round(v, d):,.{d}f} ({label})"
    if output == "per_share":
        d = q.decimals if q.decimals is not None else 2
        return round(value, d), "USD per share", f"${round(value, d):.{d}f} per share"
    if output == "days":
        d = q.decimals if q.decimals is not None else 2
        return round(value, d), "days", f"{round(value, d):.{d}f} days"
    percent = output == "percent" or q.wants_percent
    if percent:
        v = value * 100
        d = q.decimals if q.decimals is not None else 1
        return round(v, d), "percent", f"{round(v, d):.{d}f}%"
    d = q.decimals if q.decimals is not None else 2
    return round(value, d), "ratio", f"{round(value, d):.{d}f}"


def _format_change(facts: CompanyFacts, und: Understanding, q: ParsedQuestion, value: float,
                   years: tuple[int, ...], anchor: Anchor, qa: FilingQA,
                   as_of: date | None) -> tuple[float, str, str]:
    """A change rendered with the two values it is the difference of, and a change in a rate in
    percentage points: "-3.0%" for a tax rate that went from 24.6% to 21.6% reads as a relative
    change, which it is not."""
    prior, target = years

    def get(component: str, year: int) -> Line:
        return facts.value(component, year, anchor=anchor, as_of=as_of, faces=qa.faces,
                           concepts=qa.concepts)

    end_v = und.expr.evaluate(get, target, [])
    start_v = und.expr.evaluate(get, prior, [])
    if und.output in ("percent", "ratio") and (und.output == "percent" or q.wants_percent):
        d = q.decimals if q.decimals is not None else 1
        points = round(value * 100, d)
        return (points, "percentage points",
                f"{points:+.{d}f} percentage points: {start_v * 100:.{d}f}% in FY{prior}, "
                f"{end_v * 100:.{d}f}% in FY{target}")
    _, unit, _ = _format(value, und.output, q)
    change, _, change_text = _format(value, und.output, q)
    _, _, start_text = _format(start_v, und.output, q)
    _, _, end_text = _format(end_v, und.output, q)
    sign = "+" if value >= 0 else ""
    # "how much did revenue grow" is asked in dollars and in percent alike; the answer gives both
    relative = f" ({value / start_v:+.1%})" if start_v > 0 else ""
    return (change, unit, f"{sign}{change_text}{relative}: {start_text} in FY{prior}, "
                          f"{end_text} in FY{target}")


class FilingQA:
    """The public entry point: ``FilingQA().answer("What is 3M's FY2018 capex in USD millions?")``.

    Every network read is SEC's own and keyless. ``as_of`` makes the answer point-in-time: a 10-K
    accepted after it is treated as not yet filed.

    ``coverage_guard`` (default on) is the rule that every arithmetic phrase in the question must
    be used by the parsed formula; switching it off exists only so the evaluation can measure what
    it prevents.
    """

    def __init__(self, *, facts: CompanyFactsSource | None = None,
                 companies: CompanyResolver | None = None,
                 faces: FaceStatementSource | None = None,
                 concepts: ConceptSource | None = None,
                 use_faces: bool = True, coverage_guard: bool = True) -> None:
        self.facts = facts or CompanyFactsSource()
        self.companies = companies or CompanyResolver()
        self.faces = (faces or FaceStatementSource()) if use_faces else None
        self.concepts = (concepts or ConceptSource()) if use_faces else None
        self.coverage_guard = coverage_guard

    def answer(self, question: str, *, as_of: date | None = None) -> Answer:
        q = parse_question(question)
        out = Answer(question=question, status="abstained")
        for pattern, why, unless in UNSUPPORTED_CUES:
            if re.search(pattern, q.text) and not (unless and re.search(unless, q.text)):
                out.reason = f"out of scope: {why}"
                return out
        if not q.years:
            out.reason = "the question names no fiscal year"
            return out
        try:
            und = understand(q) if self.coverage_guard else _understand_unguarded(q)
        except Unresolved as exc:
            out.reason = str(exc)
            return out
        out.metric = und.name
        try:
            company = self.companies.resolve(question)
        except Unresolved as exc:
            out.reason = str(exc)
            return out
        out.company, out.cik = company.name, company.cik
        out.steps.append(f"company: {company.name} (CIK {company.cik}) via {company.how} "
                         f"'{company.matched}'")
        for name, body in und.definitions.items():
            out.steps.append(f"the question defines {name} as: {body}")
        try:
            facts = self.facts.get(company.cik)
        except Unresolved as exc:
            out.reason = str(exc)
            return out
        try:
            value, used, formula, years, anchor = self._evaluate(facts, und, q, as_of=as_of)
        except Unresolved as exc:
            out.reason = str(exc)
            return out
        out.formula = formula
        out.fiscal_years = years
        out.anchor_accn = anchor.accn
        out.lines = used
        output = und.output
        if und.mode in ("growth", "cagr"):
            output = "percent"
        out.value, out.unit, out.text = _format(value, output, q)
        if und.mode == "change":
            out.value, out.unit, out.text = _format_change(facts, und, q, value, years, anchor,
                                                           self, as_of)
        out.status = "answered"
        out.steps.append(f"anchor: FY{anchor.fiscal_year} {anchor.form} {anchor.accn} filed "
                         f"{anchor.filed.isoformat()}")
        out.steps.append(f"formula: {formula}")
        out.steps.extend(ln.cite() for ln in used)
        return out

    def _evaluate(self, facts: CompanyFacts, und: Understanding, q: ParsedQuestion, *,
                  as_of: date | None
                  ) -> tuple[float, list[Line], str, tuple[int, ...], Anchor]:
        target = max(q.years)
        anchor = facts.anchor(target, as_of=as_of)

        def get(component: str, year: int) -> Line:
            if component not in COMPONENTS:
                raise Unresolved(f"unknown line {component}")
            return facts.value(component, year, anchor=anchor, as_of=as_of,
                               faces=self.faces, concepts=self.concepts)

        used: list[Line] = []
        years = q.years
        expr = und.expr
        if und.mode == "cagr":
            if len(years) < 2:
                raise Unresolved("a compound growth rate needs a start and an end fiscal year")
            first = min(years)
            n = target - first
            end_v = expr.evaluate(get, target, used)
            start_v = expr.evaluate(get, first, used)
            if start_v <= 0 or end_v <= 0:
                raise Unresolved("a compound growth rate over a non-positive value is undefined")
            value = (end_v / start_v) ** (1.0 / n) - 1.0
            rendered = f"({expr.render()}[FY{target}] / {expr.render()}[FY{first}])^(1/{n}) - 1"
            return value, used, rendered, (first, target), anchor
        if und.mode == "growth":
            prior = min(years) if len(years) >= 2 else target - 1
            end_v = expr.evaluate(get, target, used)
            start_v = expr.evaluate(get, prior, used)
            if start_v == 0:
                raise Unresolved("growth from a reported zero is undefined")
            value = end_v / start_v - 1.0
            rendered = f"{expr.render()}[FY{target}] / {expr.render()}[FY{prior}] - 1"
            return value, used, rendered, (prior, target), anchor
        if und.mode == "change":
            prior = min(years) if len(years) >= 2 else target - 1
            end_v = expr.evaluate(get, target, used)
            start_v = expr.evaluate(get, prior, used)
            rendered = f"{expr.render()}[FY{target}] - {expr.render()}[FY{prior}]"
            return end_v - start_v, used, rendered, (prior, target), anchor
        if und.mode == "span_mean":
            first = min(years) if len(years) >= 2 else target - (q.span_average or 1) + 1
            span = list(range(first, target + 1))
            if q.span_average is not None and len(span) != q.span_average:
                raise Unresolved(
                    f"the question asks for a {q.span_average}-year average over "
                    f"FY{first}-FY{target}, which is {len(span)} years"
                )
            vals = [expr.evaluate(get, y, used) for y in span]
            value = sum(vals) / len(vals)
            return value, used, f"mean over FY{first}..FY{target} of {expr.render()}", \
                tuple(span), anchor
        value = expr.evaluate(get, target, used)
        return value, used, expr.render(), (target,), anchor


def _understand_unguarded(q: ParsedQuestion) -> Understanding:
    """``understand`` with the coverage rule off — for the ablation only. It keeps whatever the
    first rule matched and ignores every phrase nothing used."""
    try:
        return understand(q)
    except Unresolved as exc:
        if "does not use" not in str(exc):
            raise
    und = _library(q, [])
    if q.growth == "cagr" and und.output == "money":
        und.mode = "cagr"
    elif q.growth == "growth" and und.output == "money":
        und.mode = "growth"
    elif re.search(r"\b(\d+|one|two|three|four|five)[- ]year average\b", q.text):
        und.mode = "span_mean"
    return und


_DEFAULT_Q = ParsedQuestion(text="", raw="", years=(), mentions_average=False, scale=None,
                            scale_label="", wants_percent=False, decimals=None,
                            span_average=None, growth=None)


__all__ = [
    "DERIVED",
    "FORMULA_SOURCES",
    "LINE_ITEMS",
    "OPERATOR_CUES",
    "Answer",
    "Company",
    "CompanyResolver",
    "Expr",
    "FilingQA",
    "Metric",
    "ParsedQuestion",
    "Terms",
    "Understanding",
    "parse_definition",
    "parse_question",
    "understand",
]
