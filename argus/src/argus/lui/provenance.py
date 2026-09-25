"""Where each line of an answer comes from, said on the line.

Three Season 2 desks converge on this: MirrorLine files every claim under observed facts, derived
inferences, assumptions or unknowns (`lib/brief/types.ts:64-81`); baserate tags every number
observed, computed, estimated, target or replay (`web/src/domain/types.ts:35`); optic marks each
row computed or not. ARGUS showed its sources per answer, not per line, so a reader could not tell
a price read a second ago from a figure computed on it or from a record measured weeks ago.

**The labels, one per line:**

* ``live`` — read from a source just now and stated as read: a quote, a funding rate, open
  interest, a headline, a macro series value, a prediction-market price.
* ``computed`` — arithmetic on live data done for this answer: costs, beta, stress, VaR, the
  implied open, a hedge ratio, a premise verdict.
* ``record`` — a measured track record: a backtest, a held-out test, a head-to-head against a named
  rival. The figure is from a stated past run, not from now.
* ``desk`` — the desk's own logged decision and reasoning, quoted from the hash-chained ledger.
  The analyst text in it was written by a model at the time; the console quotes it, never edits it.
* ``assumed`` — a default the answer applied because the question did not say.
* ``missing`` — something that could not be read or checked, said as such.

**How a line gets its label.** The rivals attach a label where the number is made; ARGUS's lines
are made in several hundred places, so the label is read from the wording those places share —
every assumption begins "Assumed:", every record names its sample ("on 1,800 test books", "over
the last 50 of 165 nights"), every live quote names its venue. A line matching no rule gets **no
label** rather than a guess, and the share left unlabelled is measured and published
(`eval/provenance_audit.py`) rather than assumed small.

Meta lines — "Data:", "Sources reached", "Quoted", "Caveat:" — describe provenance themselves and
are left unlabelled on purpose.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

LABELS = ("live", "computed", "record", "desk", "assumed", "missing")

_META = re.compile(r"^(?:Data:|Sources reached|Sources:|Quoted \d|Caveat:|Method:|Corrected:|"
                   r"以下|\(\d+ decisions? matched|Answerable today\b)", re.I)

# Order matters: the first rule that matches decides. Missing and assumed come first because a
# line that says it could not read something must never be labelled as if it had.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("missing", re.compile(
        r"\bnot\s+checkable\b|\bdid\s+not\s+(?:answer|arrive|respond)\b|\bnot\s+known\s+yet\b|"
        r"\bunavailable\b|\bcould\s+not\s+(?:load|read|reach|fetch)\b|\btoo\s+few\b|"
        r"\bno\s+(?:data|reading|quote)\b|\bnothing\s+in\s+\d+h\b|\bso\s+I\s+need\s+what\s+you\s+"
        r"hold\b|\bnot\s+available\s+—|\b(?<!none )withheld\b|\bnot\s+(?:yet\s+)?published\b|"
        r"^I did not recognise\b", re.I)),
    ("assumed", re.compile(r"^Assumed:|\bno\s+\w+\s+was\s+(?:stated|given)\b|"
                           r"\bis\s+assessed\s+at\b|\bwere\s+scaled\s+to\s+100%|"
                           r"^Sized on a \$[\d,]+ book\b", re.I)),
    ("live", re.compile(r"^Crypto fear & greed:", re.I)),
    # A conclusion or a derived figure that quotes its inputs or its own record is still derived:
    # "Implied open: ... missed by 30bps" is the figure first and its record second.
    ("computed", re.compile(
        # A lead that is a live reading promoted to the top stays live.
        r"^Actionable:(?!\s+(?:Open interest:|Crypto fear))|^Your premise\b|^Implied open:|"
        r"^Versus the stock:|\bexplains [+-]\d|"
        r"^A \$[\d,]+ order is\b|^Order book \(|^Earnings surprise:", re.I)),
    ("desk", re.compile(
        r"^seq\s+\d+|^Stated confidence|^Thesis:|^Invalidation:|^Not yet settled|^Entry hash|"
        r"^How the desk reached|^Evidence screen:|^Panel:|^Agreement,|^Memory:|^Debate:|"
        r"^Grounding check:|^Entity check:|^Adversary:|^Constitution:|^Protocol:|"
        r"\bthe desk's own (?:last )?call\b|\bdecision \d+\b|^No open positions\b|"
        r"^The (?:record|log) holds\b|^\d+ abstention|^An abstention here\b|"
        r"\bhave a counterfactual recorded\b|^Track record: \d|^Net PnL\b|\bon the ledger\b",
        re.I)),
    ("record", re.compile(
        r"\bTested(?:,| on| not)|\bheld[\s-]out\b|\bout of sample\b|\bhow far to trust\b|"
        r"\bon [\d,]+ (?:test |past |held-out )?(?:books|nights|weekends|days|windows|stocks?)\b|"
        r"\bover (?:the last )?[\d,]+ (?:of [\d,]+ )?(?:past )?(?:nights|weekends|"
        r"\d-day windows|windows|sessions)\b|\bover [\d,]+ past (?:[\d-]+ )?(?:days|windows)\b|"
        r"\bof its last [\d,]+ (?:weekends|nights|days)\b|\bmissed (?:the|by)\b|"
        r"\b(?:S2|Season 2) (?:desk|entry)\b|"
        r"\bsince (?:19|20)\d\d\b|\bin (?:the )?[\d,]+ (?:comparable|past) (?:past )?"
        r"(?:states|weekends)\b|"
        r"\b[\d,]+ (?:past|of) (?:its |the )?(?:last )?[\d,]+? ?(?:weekends|nights)\b|"
        r"\bcleared that in \d+% of past\b|\bits own record\b|\bwhat actually happened\b|"
        r"\brealised worst case\b|\bworst 24-bar window\b|"
        r"\bthe systematic signals this desk tested\b|\bleans went the right way\b|"
        r"^Where a stop sits in the noise\b|\bfinished (?:higher|lower) \d+% of [\d,]+ windows\b|"
        r"^Spread of outcomes\b",
        re.I)),
    ("live", re.compile(
        r"\blast [\d,.]+ USDT on Bitget\b|^(?:Actionable: )?Open interest:|\bheld open\b|"
        r"^Funding(?: is|:)|"
        r"\bfunding [+-]?\d|^\d+h ago —|^Prediction market\b|\bfear & greed\b|"
        r"\b(?:Treasury|fed funds|breakeven|dollar index)\b.*\bon \d{4}-\d\d-\d\d\b|"
        r"^Scheduled:|\bspot BTC ETFs\b|\bheadlines? name\b|\b8-K\b|"
        r"\bis [+-]\d+\.\d+% over 24 hours\b|"
        r"^RSI\(|^MACD\b|^ATR\b|^Price [\d,.]+: nearest support\b|^News:|"
        r"\bthe US (?:stock market|anchor market)\b.*\b(?:open|closed|shut)\b|"
        r"^Crypto fear|^Most recent report on file\b|^Institutional holders\b|\bmarket cap \$|"
        r"^Valuation on \d{4}|^Institutions: [\d,]+ holders\b|^Insider filings\b|"
        r"^Analyst price targets\b|^Now: \w+ is [+-]\d|^Federal Reserve, \d+ \w+:|"
        r"^Bitget's own US-stock brief\b|^Crowd on X and Reddit\b|\bpositioning: \d+% of\b|"
        r"\bwallets hold \$|\bliquidations on \d{4}-\d\d-\d\d\b", re.I)),
    ("computed", re.compile(
        r"^Actionable:|^Your premise\b|^Implied open:|^Versus the stock:|\bbeta\b|^Hedge:|"
        r"^If \w+ (?:falls|moves|rises)\b|\bvalue at risk\b|\bexpected shortfall\b|"
        r"\bof (?:your )?(?:total |book )?risk\b|^After the trade:|^Risk spread\b|"
        r"\bcorrelated\b|\bmove together\b|^Resizing\b|^Adding\b|^Plan:|^Slice \d|^Schedule\b|"
        r"^In the book's worst \d+% of hours\b|^Cadence:|^Waiting costs\b|^Session:|"
        r"\bround trip\b|^Typical \d|^Clearing the cost\b|"
        r"^Curve:|^Real 10-year\b|\bcorrelation [+-]?\d|^Return shape\b|^Next \d+ sessions\b|"
        r"^Path match\b|\bliquidation\b|\bmoves? (?:\d\.\d+x|about [+-])|"
        r"\brealised vol\b|^Cost:|^Why not\b|^Weekends \(|^Going into\b|^The liquidation\b|"
        r"\bthe worst point was within\b|^Neither is the share\b|^Bitget's stock perpetuals\b|"
        r"^The desk itself trades\b|^A crypto contract\b|^Funding means\b|\bfunding across\b|"
        r"\bexpect it around\b|\blooks coordinated\b|^No story is carried\b",
        re.I)),
)


def label(line: str) -> str | None:
    """The line's provenance, or None for a meta line or one no rule recognises."""
    text = line.strip()
    if not text or _META.search(text):
        return None
    for name, pattern in _RULES:
        if pattern.search(text):
            return name
    return None


def labels(lines: Sequence[str]) -> list[str | None]:
    return [label(str(line)) for line in lines]


__all__ = ["LABELS", "label", "labels"]
