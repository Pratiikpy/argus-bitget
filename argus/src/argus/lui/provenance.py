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
* ``memory`` — something the trader told the console earlier (`lui/memory.py`), shown where it
  shaped the answer.

**How a line gets its label.** The rivals attach a label where the number is made; ARGUS's lines
are made in several hundred places, so the label is read from the wording those places share —
every assumption begins "Assumed:", every record names its sample ("on 1,800 test books", "over
the last 50 of 165 nights"), every live quote names its venue. A line matching no rule gets **no
label** rather than a guess, and the share left unlabelled is measured and published
(`eval/provenance_audit.py`) rather than assumed small.

Meta lines — "Data:", "Sources reached", "Quoted", "Caveat:" — describe provenance themselves and
are left unlabelled on purpose.

**These rules are the fallback, not the record** (2026-09-26). `truth/trace.py` records, while an
answer is built, which step made each line, what it read and when; where that step declares what
its lines are, or a refusal followed a failed read, the label comes from the step
(``origin: "trace"``) and these rules are not consulted. A line no step speaks for keeps the label
these rules give it, marked ``origin: "regex"``. `eval/trace_audit.py` publishes how many lines
each decides on the held-out corpora and every line on which the two disagree.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

LABELS = ("live", "computed", "record", "desk", "assumed", "missing", "memory")

_META = re.compile(r"^(?:Data:|Sources reached|Sources:|Quoted \d|Caveat:|Method:|Corrected:|"
                   r"Computed by ARGUS\b|Read as filed:|"
                   r"以下|\(\d+ decisions? matched|Answerable today\b)", re.I)

_LIVE_LEAD = re.compile(r"\S+ last [\d,.]+ USDT on Bitget\b|Open interest:|Crypto fear|"
                        r"Long/short on Bitget\b|\S+ funding is [+-]?\d|RSI\(|"
                        r"US spot (?:BTC|ETH) ETFs\b", re.I)
"""A live reading moved to the lead unchanged — the price, open interest, the long/short split,
the funding rate asked for — keeps its live label."""

# Order matters: the first rule that matches decides. Missing and assumed come first because a
# line that says it could not read something must never be labelled as if it had.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # A sentence from a filing answer (`research/document_qa.py`) ends with the number of the
    # passage it cites, and a sentence without one never reaches the page: read from EDGAR now.
    ("live", re.compile(r"\.\s*(?:\[\d{1,2}\])+\s*$")),
    # the past states an analogue answer was matched to: worked out from the candles just read
    ("computed", re.compile(r"^Closest past states: ")),
    # a saved book written as amounts, priced at Bitget's last price for this answer
    ("live", re.compile(r"^Priced at Bitget's last price: ")),
    # the stress answer's scenario tree (`desk/stress_tree.py`) and its loss-share line: every
    # figure worked out for this answer from the candles it just read
    ("computed", re.compile(r"^Scenario tree: \d+ scenarios|^\d+(?:\.\d+){0,3}\. (?:Stated shock|"
                            r"With the beta hedge|In the worst|\w+ on its own|Worst realised|"
                            r"Halving|History check|At the most extreme|Tail:|Trimming|Session:|"
                            r"If the )|\bis \d+% of the book but \d+% of its loss\b")),
    # A measured record over past sweeps or past releases (`eval/skill_matrix.py`, the "into
    # earnings" history): counted from what already happened, not read now.
    ("record", re.compile(r"^(?:Actionable: )?across \d+ sweep|^bitget-mcp-server: every call|"
                          r"^bitget-signal's news|^Answering with data in the latest sweep|"
                          r"^Why the rest did not, by kind|^Into earnings: |"
                          r"^The most recent one, ", re.I)),
    # what an as-of answer left out, and why: the answer describing itself
    ("assumed", re.compile(r"^Point in time: \d+ line\(s\) about today")),
    # A FRED series served from the snapshot shipped with the console is a past reading. Checked
    # before "missing": its line also says FRED did not answer, which is why it is a record and not
    # a gap (the CPI lead was tagged computed on the live page, 2026-09-25).
    ("record", re.compile(r"\(the shipped reading\)|last reading shipped with the console", re.I)),
    ("missing", re.compile(
        r"\bnot\s+checkable\b|\bdid\s+not\s+(?:answer|arrive|respond)\b|\bnot\s+known\s+yet\b|"
        r"\bunavailable\b|\bcould\s+not\s+(?:load|read|reach|fetch)\b|\btoo\s+few\b|"
        r"\bno\s+(?:data|reading|quote)\b|\bnothing\s+in\s+\d+h\b|\bso\s+I\s+need\s+what\s+you\s+"
        r"hold\b|\bnot\s+available\s+—|\b(?<!none )withheld\b|\bnot\s+(?:yet\s+)?published\b|"
        r"^I did not recognise\b|^Missing:|\bholds no\b[^.;]{0,40}\bfigure\b|"
        r"\bhas not verified\b|\bcould not be measured\b|\bwas not (?:read|measured|fetched)\b",
        re.I)),
    ("assumed", re.compile(r"^Assumed:|\bno\s+\w+\s+was\s+(?:stated|given)\b|"
                           r"\bis\s+assessed\s+at\b|\bwere\s+scaled\s+to\s+100%|"
                           r"^Sized on a \$[\d,]+ book\b", re.I)),
    # the trader's own words, kept by `lui/memory.py` and shown where they shaped the answer
    ("memory", re.compile(r"^Remembered:|^Your thesis on\b|^(?:Actionable: )?noted —", re.I)),
    ("live", re.compile(r"^Crypto fear & greed:", re.I)),
    # a FRED series served from the snapshot shipped with the console is a past reading
    ("record", re.compile(r"\(the shipped reading\)|last reading shipped with the console", re.I)),
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
        r"^Spread of outcomes\b|\boverlapping windows of its own history\b|"
        r"\bdeepest fall from a high\b",
        re.I)),
    ("live", re.compile(
        r"\blast [\d,.]+ USDT on Bitget\b|^(?:Actionable: )?Open interest:|\bheld open\b|"
        r"^Funding(?: is|:)|"
        r"\bfunding (?:is )?[+-]?\d|^\d+h ago —|^Prediction market\b|\bfear & greed\b|"
        r"\b(?:Treasury|fed funds|breakeven|dollar index)\b.*\bon \d{4}-\d\d-\d\d\b|"
        r"^Scheduled:|\bspot BTC ETFs\b|\bheadlines? name\b|\b8-K\b|"
        r"\bis [+-]\d+\.\d+% over 24 hours\b|"
        r"^RSI\(|^MACD\b|^ATR\b|^Price [\d,.]+: nearest support\b|^News:|"
        r"\bthe US (?:stock market|anchor market)\b.*\b(?:open|closed|shut)\b|"
        r"^Crypto fear|^Most recent report on file\b|^Next report:|^Analyst consensus\b|"
        r"^Institutional holders\b|\bmarket cap \$|"
        r"^Valuation on \d{4}|^Institutions: [\d,]+ holders\b|^Insider filings\b|"
        r"^Analyst price targets\b|^Now: \w+ is [+-]\d|^Federal Reserve, \d+ \w+:|"
        r"^Bitget's own US-stock brief\b|^Crowd on X and Reddit\b|\bpositioning: \d+% of\b|"
        r"\bwallets hold \$|\bliquidations on \d{4}-\d\d-\d\d\b|^Long/short on Bitget\b|"
        r"^Listed now:|\bright now\b[^.]{0,60}\ba year\b|^Quarters before it:|"
        r"\bspot (?:BTC|ETH) ETFs?\b|^RSI\(14, 1D\)", re.I)),
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
        r"\bexpect it around\b|\blooks coordinated\b|^No story is carried\b|"
        r"^Shorter term \(4h\)|^At \S+ last \d+ days of volatility\b|^At \d+(?:\.\d+)?x\b|"
        r"^Size \S+ so that\b|^(?:No|A) position signal\b|\bown positioning is\b|"
        r"^Use the perpetual\b|^The formula uses\b|^Liquidation price:|\bmoving average is\b|"
        r"^\S+'s 50-day average is\b|^Of the hedges you named\b|^Better than\b",
        re.I)),
)


def label(line: str) -> str | None:
    """The line's provenance, or None for a meta line or one no rule recognises."""
    text = line.strip()
    if not text or _META.search(text):
        return None
    lead = re.match(r"^Actionable(?: \(\w+\))?:\s*(.+)$", text)
    if lead is not None:
        # A lead is labelled by what it is, not by being first: a live reading promoted to the
        # top ("Actionable: NVDA last 226.3 USDT on Bitget") stays live, a record stays a record,
        # and a lead that says a figure is missing is missing. Anything else a lead says is the
        # answer's conclusion, computed for it.
        rest = lead.group(1)[0].upper() + lead.group(1)[1:]
        inner = label(rest)
        if inner == "live":
            # only a reading promoted as it was read; a conclusion drawn on live counts ("2
            # headlines name NVDA ... so the move is sentiment") is computed, as reviewed
            return "live" if _LIVE_LEAD.match(rest) else "computed"
        if inner == "missing":
            # missing only when the lead's own first clause is the gap; "XAU removes 11%; TLT
            # could not be measured ..." answers with a figure and notes one gap (seen live)
            first = re.split(r";\s|\.\s", rest, maxsplit=1)[0]
            return "missing" if label(first) == "missing" else "computed"
        return inner if inner in ("record", "desk", "memory") else "computed"
    for name, pattern in _RULES:
        if pattern.search(text):
            return name
    return None


def labels(lines: Sequence[str]) -> list[str | None]:
    return [label(str(line)) for line in lines]


def payload_labels(payload: Mapping[str, Any]) -> list[str | None]:
    """The labels an answer carries, one per line: the ones attached with its trace
    (`truth/trace.attach`) when there is one per line, else these wording rules.

    The one call for a surface that shows an answer (the MCP server, the Telegram bot), so a line
    labelled by the step that made it is not relabelled from its wording on the way out."""
    lines = [str(line) for line in payload.get("lines") or []]
    carried = payload.get("line_labels")
    if (isinstance(carried, list) and len(carried) == len(lines)
            and all(x is None or x in LABELS for x in carried)):
        return list(carried)
    return labels(lines)


__all__ = ["LABELS", "label", "labels", "payload_labels"]
