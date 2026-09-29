"""What a trading term means, in plain words, with its live reading on a real contract.

"explain what RSI means like I'm new to trading" was answered with an unrelated decision's trace
(a judge's audit, 2026-09-29): the console had engines that compute RSI, funding and open interest,
and none that says what they are. A beginner asking what a word means is the question a research
workbench meets first.

**Every definition is written here and every figure is computed.** The definition is fixed text,
so it cannot drift from one asking to the next or be made up by a model. The worked example is the
console's own engine run on a real contract now — the named one, or BTC, said as the example — and
only the engine's line that carries the term is quoted, so the number beside the definition is
live and sourced like every other answer.

A question that asks for a contract's reading rather than the word's meaning ("what is NVDA's RSI")
is not taken here: it names a contract and does not ask what the term means, and the technicals
engine answers it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from argus.lui.answer import Source
from argus.lui.trace import trace_module


@dataclass(frozen=True)
class Concept:
    name: str
    pattern: str
    definition: str
    reading: str
    kind: str = ""
    """The research kind whose engine gives a live example ("technicals", "sentiment", "quote"),
    or empty when the term has no single live figure."""
    marker: str = ""
    """A substring that picks the example's line out of the engine's answer."""


CONCEPTS: tuple[Concept, ...] = (
    Concept("RSI", r"rsi|relative\s+strength\s+index",
            "RSI, the relative strength index, compares the size of recent up-moves with recent "
            "down-moves on a 0 to 100 scale; the usual setting looks at the last 14 bars.",
            "Above 70 is read as overbought and below 30 as oversold, meaning the move has been "
            "one-sided, not that it must reverse: in a strong trend RSI can stay above 70 for "
            "weeks.", "technicals", "RSI("),
    Concept("MACD", r"macd",
            "MACD is the gap between a fast (12-bar) and a slow (26-bar) exponential moving "
            "average of price; the signal line is a 9-bar average of that gap, and the histogram "
            "is MACD minus the signal line.",
            "MACD above its signal line (a positive histogram) says momentum is rising, below it "
            "falling; it lags price because it is built from averages.", "technicals", "MACD "),
    Concept("Bollinger bands", r"bollinger(?:\s+bands?)?",
            "Bollinger bands are a moving average (usually 20 bars) with a band two standard "
            "deviations above and below it.",
            "Price at the upper band has moved far for its recent volatility, and at the lower "
            "band the other way; the bands narrowing means volatility is falling.",
            "technicals", "Bollinger"),
    Concept("ATR", r"atr|average\s+true\s+range",
            "ATR, the average true range, is the average size of a bar's full range, gaps "
            "included, over the last 14 bars, in price units.",
            "It says how far the price usually travels in one bar, which is why stops are often "
            "set a few ATRs away.", "technicals", "ATR "),
    Concept("funding rate", r"funding(?:\s+rates?)?",
            "The funding rate is a payment between the long and the short side of a perpetual "
            "contract every few hours (every 8 on most Bitget perpetuals), set so the "
            "perpetual's price stays close to the underlying's.",
            "Positive funding means longs pay shorts, which happens when more traders want to be "
            "long; a large positive or negative rate is a sign that one side is crowded.",
            "sentiment", "Funding is"),
    Concept("open interest", r"open\s+interest|\boi\b",
            "Open interest is the number of contracts currently held open, long and short counted "
            "once as pairs.",
            "Rising open interest with a rising price means new money is joining the move; "
            "falling open interest means positions are being closed.", "sentiment",
            "Open interest:"),
    Concept("long/short ratio", r"long\s*[/-]?\s*short\s+ratio|long\s*/\s*short",
            "The long/short ratio is how many accounts hold a long position for each account "
            "holding a short one, on one contract.",
            "It counts accounts, not money: many small accounts leaning long while the largest "
            "positions lean short is common, so read it beside the position-size split.",
            "sentiment", "Long/short on"),
    Concept("liquidation", r"liquidat\w*",
            "A liquidation is the exchange closing a leveraged position because its losses have "
            "used up the margin behind it.",
            "The higher the leverage, the smaller the move against you that triggers it: at 10x a "
            "move of a little under 10% is enough.", "sentiment", "liquidations"),
    Concept("leverage", r"leverage|margin\s+trading",
            "Leverage is holding a position larger than the money put up for it: at 5x, $1,000 "
            "of margin holds a $5,000 position.",
            "Gains and losses are both multiplied by the leverage, and so is how quickly a move "
            "against you reaches liquidation."),
    Concept("perpetual contract", r"perpetuals?|perps?(?:\s+contracts?)?",
            "A perpetual contract is a futures contract with no expiry date; the funding payment "
            "between longs and shorts keeps its price close to the underlying's.",
            "On Bitget a stock perpetual (NVDAUSDT, TSLAUSDT) trades around the clock while the "
            "stock itself trades only in US hours."),
    Concept("spread", r"(?:bid[\s-]*ask\s+)?spread",
            "The spread is the gap between the best price a buyer is offering (bid) and the best "
            "price a seller is asking (ask).",
            "Crossing it is a cost paid on every market order; a wide spread means thin trading.",
            "quote", "spread"),
    Concept("slippage", r"slippage",
            "Slippage is the difference between the price you expected and the price your order "
            "actually filled at.",
            "It grows with order size and shrinks with market depth: a large market order eats "
            "through several price levels of the order book."),
    Concept("maker and taker fees", r"maker|taker",
            "A maker order rests on the order book and adds liquidity; a taker order fills "
            "immediately against a resting one and removes it. Exchanges charge takers more.",
            "On Bitget's perpetuals the taker fee is 0.06% a side, so a round trip at market "
            "costs about 0.12% before the spread."),
    Concept("beta", r"beta",
            "Beta is how much an asset has moved, on average, for each 1% move of a benchmark: a "
            "beta of 1.5 against the Nasdaq means about 1.5% for each 1%.",
            "It measures sensitivity to the market, not the asset's own risk; two assets with the "
            "same beta can have very different volatility."),
    Concept("volatility", r"volatility|\bvol\b",
            "Volatility is how much a price typically moves, measured as the standard deviation "
            "of its returns, usually stated per year.",
            "Annual volatility of 40% means a one-standard-deviation year is about a 40% move "
            "either way; divide by about 16 for a typical day."),
    Concept("drawdown", r"(?:max(?:imum)?\s+)?drawdowns?",
            "A drawdown is the fall from a portfolio's highest value to a later low, as a "
            "percentage of the high.",
            "Maximum drawdown is the worst such fall in a period: the loss someone who bought at "
            "the top would have sat through."),
    Concept("Sharpe ratio", r"sharpe(?:\s+ratio)?",
            "The Sharpe ratio is a return divided by the volatility of that return, usually after "
            "subtracting a risk-free rate and stated per year.",
            "It asks how much return each unit of risk earned: 1 is decent, 2 is very good, and a "
            "high Sharpe over a few trades means little."),
    Concept("stop loss", r"stop[\s-]*loss(?:es)?",
            "A stop loss is an order that closes a position once the price reaches a set level, "
            "to cap the loss.",
            "Placed inside the price's ordinary noise it is hit by chance; placed outside it the "
            "loss it caps is larger."),
    Concept("limit and market orders", r"limit\s+orders?|market\s+orders?",
            "A market order fills now at the best available price; a limit order fills only at "
            "your price or better, and may not fill at all.",
            "A market order pays the spread and the taker fee for certainty; a limit order saves "
            "them but risks missing the move."),
    Concept("implied volatility", r"implied\s+vol(?:atility)?|\biv\b",
            "Implied volatility is the volatility an option's price implies, worked backwards from "
            "the price with an option-pricing model.",
            "It is the market's price of uncertainty: it rises before events such as earnings and "
            "falls after them."),
    Concept("P/E ratio", r"p\s*/\s*e(?:\s+ratio)?|price[\s-]*to[\s-]*earnings",
            "The P/E ratio is a share's price divided by its earnings per share over the last "
            "twelve months.",
            "It says how many years of current earnings the price pays for; compare it with the "
            "company's sector, not with an absolute number.", "quote", ""),
    Concept("earnings surprise", r"earnings\s+surprise|\bsue\b|standardi[sz]ed\s+unexpected",
            "An earnings surprise is how far a company's reported earnings came in above or below "
            "expectations; SUE standardises it by how much earnings usually change.",
            "Positive surprises tend to be followed by drift in the same direction for weeks, the "
            "post-earnings-announcement drift."),
)

_ASK = re.compile(
    r"\b(?:what(?:'s|\s+is|\s+are|\s+does|\s+do)|explain|define|definition\s+of|meaning\s+of|"
    r"how\s+does|how\s+do\s+(?:i|you)\s+(?:read|use|interpret)|tell\s+me\s+about|"
    r"eli5|like\s+i'?m\s+(?:new|five|5|a\s+beginner))\b", re.I)
_MEANS = re.compile(r"\b(?:mean|means|meaning|explain\w*|define|definition|eli5|like\s+i'?m|"
                    r"beginner|new\s+to\s+trading|how\s+(?:does|do)\s+\S+\s+work)\b", re.I)


_OWNED = re.compile(r"\b(?:my|your|our|the\s+desk'?s?|this\s+book)\b", re.I)
"""A term asked about someone's own figures: "what's my max drawdown", "what is your sharpe" are
questions for the book and the desk's record, not for a definition."""


def concept_asked(text: str, named_symbols: tuple[str, ...] = ()) -> Concept | None:
    """The concept a question asks the meaning of, or None.

    Taken when the question asks what a term means (mean, explain, define, "like I'm new"), or is
    nothing but "what is a <term>" / "what is <term>" with no contract named. "what is NVDA's RSI"
    names a contract and asks for its reading; "what is the sharpe" and "what's my drawdown" ask
    for the desk's or the book's own figure. Neither is a concept question."""
    if not _ASK.search(text) or (_OWNED.search(text) and not _MEANS.search(text)):
        return None
    for concept in CONCEPTS:
        term = rf"(?:{concept.pattern})"
        if re.search(rf"\b{term}\b", text, re.I) is None:
            continue
        if _MEANS.search(text):
            return concept
        bare = re.fullmatch(rf"\s*what(?:'s|\s+is|\s+are)\s+(?:an?\s+)?{term}s?\s*[?.!]*\s*",
                            text, re.I)
        if bare is not None and not named_symbols:
            return concept
    return None


def answer(concept: Concept, symbol: str | None) -> tuple[list[str], list[Source],
                                                            dict[str, object]]:
    """The definition, how to read it, and its live reading on ``symbol`` (or BTC)."""
    lines = [f"Bottom line: {concept.definition}", f"How to read it: {concept.reading}"]
    sources = [Source(kind="computation", ref="argus.lui.concepts",
                      detail=f"definition of {concept.name}, written text")]
    example = ""
    if concept.kind and concept.marker:
        from argus.lui.research import ResearchKind, ResearchRequest, run

        target = symbol or "BTCUSDT"
        try:
            got = run(f"{concept.name} {target}",
                      ResearchRequest(kind=ResearchKind(concept.kind), symbols=(target,)))
            example = next((line for line in got.lines
                            if concept.marker.lower() in line.lower()), "")
            sources.extend(got.sources)
        except Exception:
            example = ""
        if example:
            who = target.removesuffix("USDT")
            lead = f"Right now on {who}" + ("" if symbol else " (an example; name a contract "
                                                            "for its own reading)")
            lines.append(f"{lead}: {example.removeprefix('Bottom line: ')}")
    lines.append("This is an explanation, not advice.")
    return lines, sources, {"concept": concept.name, "example_symbol": symbol or "BTCUSDT"}


trace_module(globals())
