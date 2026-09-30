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
from argus.lui.provenance import explains
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
    example: str = "BTCUSDT"
    """The contract the live example runs on when the question names none: a company figure (a P/E,
    an earnings surprise) needs a stock, not BTC."""
    cited: str = ""
    """Where a fact stated in the text comes from, when the text states one (a fee, an interval)."""
    asset: bool = False
    """The term is itself a contract ("what is bitcoin"): asked bare, it is a definition even
    though it names the contract."""


CONCEPTS: tuple[Concept, ...] = (
    Concept("basis point", r"bps|basis\s+points?|\bbp\b",
            "A basis point (bp) is a hundredth of a percent: 100bps is 1%, and 12bps is 0.12%.",
            "Costs, funding and small moves are quoted in them because the numbers are small: a "
            "12bps round trip means paying 0.12% of the position to get in and out."),
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
            "sentiment", "Funding is",
            cited="Bitget's contract specifications: funding settles every 8 hours on most USDT-M "
                  "perpetuals, and some settle every 4 or every 1"),
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
            "costs about 0.12% before the spread.",
            cited="Bitget's USDT-M perpetual fee schedule (0.02% maker, 0.06% taker at the base "
                  "tier), the rate cost.CostModel.bitget_perp charges on every cost figure here"),
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
            "company's sector, not with an absolute number.", "fundamentals", "P/E (trailing",
            example="NVDAUSDT"),
    Concept("earnings surprise", r"earnings\s+surprise|\bsue\b|standardi[sz]ed\s+unexpected",
            "An earnings surprise is how far a company's reported earnings came in above or below "
            "expectations; SUE standardises it by how much earnings usually change.",
            "Positive surprises tend to be followed by drift in the same direction for weeks, the "
            "post-earnings-announcement drift."),
    # Options
    Concept("call option", r"call\s+options?|calls?\s+vs\.?\s+puts?|\bcalls?\b(?=.*\bputs?\b)",
            "A call option is the right, not the obligation, to buy a stock at a set price (the "
            "strike) until a set date (the expiry); the buyer pays a premium for it.",
            "A call gains when the stock rises above the strike by more than the premium paid; if "
            "it does not, the most the buyer loses is the premium."),
    Concept("put option", r"put\s+options?|\bputs?\b(?=\s+(?:option|contract))|\bputs\b",
            "A put option is the right, not the obligation, to sell a stock at a set price (the "
            "strike) until a set date (the expiry); the buyer pays a premium for it.",
            "A put gains when the stock falls below the strike by more than the premium; traders "
            "buy puts to bet on a fall or to insure shares they hold."),
    Concept("strike price", r"strike(?:\s+price)?",
            "The strike price is the price at which an option lets its holder buy (a call) or sell "
            "(a put) the underlying.",
            "An option is in the money when exercising it now would pay (a call's strike below the "
            "price, a put's above) and out of the money when it would not."),
    Concept("option expiry", r"expir(?:y|ation)(?:\s+date)?|expiry\s+date",
            "An option's expiry is the last day it can be used; after it, an option that is out of "
            "the money is worth nothing.",
            "The closer the expiry, the faster an option's time value decays, which works against "
            "the buyer and for the seller."),
    Concept("option premium", r"(?:option\s+)?premium",
            "An option's premium is the price paid for it: intrinsic value (what exercising now "
            "would pay) plus time value (what the chance of more is worth).",
            "The premium rises with the underlying's implied volatility and with the time left to "
            "expiry."),
    Concept("in the money", r"in[\s-]the[\s-]money|out[\s-]of[\s-]the[\s-]money|\bitm\b|\botm\b",
            "An option is in the money (ITM) when exercising it now would pay: a call with the "
            "price above its strike, a put with the price below. Out of the money (OTM) is the "
            "opposite.",
            "Out-of-the-money options are cheaper and need a bigger move to pay off."),
    Concept("straddle", r"straddles?",
            "A straddle is a call and a put on the same stock with the same strike and expiry, "
            "bought together.",
            "It pays if the stock moves a lot either way; its price is the market's estimate of "
            "how big the move will be, which is why it is read before earnings.",
            "sentiment", "straddle", example="NVDAUSDT"),
    # Charts and trend
    Concept("overbought and oversold", r"overbought|oversold",
            "Overbought and oversold describe a price that has moved one way unusually far and "
            "fast, usually read from RSI above 70 or below 30.",
            "They say the move has been one-sided, not that it must reverse: a strong trend can "
            "stay overbought for weeks.", "technicals", "RSI("),
    Concept("moving average crossover",
            r"moving[\s-]averages?\s+cross(?:over|es|ing)?|golden\s+cross|death\s+cross|"
            r"\bma\s+cross(?:over)?",
            "A moving average crossover is a shorter moving average crossing a longer one: above "
            "it (a golden cross, for the 50-day and 200-day) or below it (a death cross).",
            "It marks a change in trend after the fact; because both averages lag, the signal "
            "arrives late and gives many false starts in a sideways market."),
    Concept("moving average", r"moving\s+averages?|\bsma\b|\bema\b",
            "A moving average is the average price over the last N bars, recalculated each bar; "
            "an exponential one (EMA) weights recent bars more.",
            "Price above a rising average is read as an uptrend; the longer the average, the "
            "slower it turns."),
    Concept("support and resistance", r"support(?:\s+(?:and|&)\s+resistance)?|resistance",
            "Support is a price level where buying has repeatedly stopped a fall; resistance is "
            "one where selling has repeatedly stopped a rise.",
            "A level broken with conviction often swaps roles: old resistance becomes support.",
            "technicals", "support"),
    Concept("momentum", r"momentum",
            "Momentum is the tendency of a price that has been rising (or falling) to keep going "
            "for a while; indicators such as RSI and MACD measure it.",
            "It is strongest over weeks to months and reverses at extremes; it describes the move, "
            "not its cause.", "technicals", "MACD "),
    Concept("VWAP and TWAP", r"\bvwap\b|\btwap\b",
            "VWAP is the average price weighted by volume over a period; TWAP is the plain average "
            "over time. Both are benchmarks, and names for orders split to match them.",
            "An execution beats VWAP when it bought below it; a large order is split over time to "
            "avoid moving the price."),
    # Shorts
    Concept("short interest", r"short\s+interest",
            "Short interest is the number of a company's shares sold short and not yet bought "
            "back, usually shown as a share of the shares available to trade.",
            "High short interest means many traders bet on a fall; it can also fuel a sharp rise "
            "if they rush to buy back (a short squeeze)."),
    Concept("short squeeze", r"short\s+squeez\w*",
            "A short squeeze is a fast rise driven by short sellers buying back to cap their "
            "losses, which pushes the price up further.",
            "It needs heavy short positioning and a trigger; the rise tends to fade once the "
            "buying back is done."),
    Concept("short selling", r"short\s+sell\w*|going\s+short|to\s+short\b",
            "Short selling is selling something you do not own (borrowed, or through a perpetual "
            "or future) to buy it back later, cheaper.",
            "The gain is capped at the fall to zero and the loss is not capped, which is why "
            "shorts are sized smaller than longs."),
    # Accounting and tax
    Concept("wash sale", r"wash\s+sales?",
            "A wash sale, under US tax rules, is selling a security at a loss and buying the same "
            "or a substantially identical one within 30 days before or after; the loss cannot be "
            "deducted then.",
            "The disallowed loss is added to the cost basis of the new shares. It is a tax rule "
            "for US taxpayers, not a trading rule; ask a tax adviser for your case."),
    Concept("cost basis", r"cost\s+basis",
            "Cost basis is what you paid for a holding, fees included, used to work out the gain "
            "or loss when you sell.",
            "With several buys it can be averaged or tracked lot by lot (first in, first out), and "
            "the two give different realised profits on the same sale."),
    Concept("realised and unrealised profit",
            r"(?:un)?reali[sz]ed\s+(?:p\s*&\s*l|pnl|profits?|gains?|loss(?:es)?)",
            "Realised profit or loss is locked in by selling; unrealised is the gain or loss on "
            "what you still hold at today's price.",
            "Unrealised profit can vanish before you sell; only the realised part is money you "
            "have."),
    Concept("dividend", r"dividends?|ex[\s-]dividend",
            "A dividend is cash a company pays its shareholders out of profits; the ex-dividend "
            "date is the first day a buyer no longer gets the next payment.",
            "On the ex-dividend date the share price usually drops by about the dividend, so "
            "buying just before it is not free money."),
    Concept("market cap", r"market\s+cap(?:itali[sz]ation)?",
            "Market capitalisation is a company's share price times the number of its shares: "
            "what the market values the whole company at.",
            "It sorts companies by size (large, mid, small cap); it is not what the company would "
            "sell for, and it moves with every trade."),
    Concept("EPS", r"\beps\b|earnings\s+per\s+share",
            "EPS, earnings per share, is a company's profit over a period divided by its number of "
            "shares.",
            "It is the E in P/E, and the figure an earnings report is judged on against analysts' "
            "estimates."),
    # Portfolio
    Concept("dollar-cost averaging", r"dollar[\s-]cost\s+averag\w*|\bdca\b",
            "Dollar-cost averaging (DCA) is buying a fixed amount at regular intervals instead of "
            "all at once.",
            "It spreads the entry over time, so no single price decides the result; when the "
            "market rises steadily, buying all at once has usually done better, and DCA's value "
            "is the smaller regret when it falls."),
    Concept("diversification", r"diversif\w*",
            "Diversification is holding assets that do not all move together, so one loss is a "
            "smaller part of the whole.",
            "It helps only as far as the holdings are uncorrelated: five tech stocks are closer to "
            "one position than five.", "", ""),
    Concept("correlation", r"correlat\w*",
            "Correlation measures how closely two prices move together, from +1 (in step) through "
            "0 (unrelated) to -1 (opposite).",
            "It changes over time and tends to rise in a crash, when diversification is most "
            "needed."),
    Concept("hedging", r"hedg\w*",
            "Hedging is taking a second position that gains when the first one loses, to cut the "
            "risk you did not want to carry.",
            "A hedge costs something (fees, funding, the upside it gives away); a good one removes "
            "the risk you meant to remove and little else."),
    Concept("position sizing", r"position\s+siz\w*",
            "Position sizing is deciding how much to put into one trade, usually from how much you "
            "can accept losing if it goes wrong.",
            "Sizing from the worst ordinary move (and a stop outside the noise) keeps one bad "
            "trade from sinking the account."),
    Concept("risk/reward", r"risk[\s/-]*(?:to[\s-])?reward",
            "Risk/reward compares what a trade stands to lose (to its stop) with what it stands "
            "to gain (to its target).",
            "A 1:3 ratio needs to be right only one time in four to break even before costs; the "
            "ratio means nothing if the target is rarely reached."),
    # Derivatives mechanics
    Concept("mark price", r"mark\s+price",
            "The mark price is the fair price an exchange uses to value open positions and trigger "
            "liquidations, built from the index price rather than the last trade.",
            "It keeps one thin trade from liquidating everyone; the last traded price can differ "
            "from it for a moment."),
    Concept("basis", r"\bbasis\b(?!\s+points?)|contango|backwardation",
            "Basis is the gap between a futures or perpetual price and the spot price; contango is "
            "futures above spot, backwardation below.",
            "A wide positive basis means traders pay up to be long, which funding then charges "
            "them for on a perpetual."),
    Concept("rToken", r"\brtokens?\b|tokeni[sz]ed\s+stocks?",
            "An rToken is Bitget's tokenised stock: a token tracking a US share (rNVDA for "
            "NVIDIA), traded around the clock on Bitget.",
            "Outside US market hours it trades while the share it tracks does not, so its price "
            "can drift from the stock's last close and gap back at the open."),
    Concept("stablecoin", r"stablecoins?",
            "A stablecoin is a crypto token designed to hold a fixed value, usually one US dollar "
            "(USDT, USDC).",
            "Bitget's USDT-M contracts are priced and settled in USDT, so their P&L is in dollars "
            "as long as USDT holds its peg."),
    Concept("leveraged ETF decay", r"(?:volatility\s+)?decay|leveraged\s+etfs?",
            "A leveraged ETF (TQQQ is 3x the Nasdaq-100) resets its leverage every day, so over "
            "time it does not return 3x the index.",
            "In a choppy market that goes nowhere it loses value (volatility decay); in a steady "
            "trend it can return more than 3x."),
    Concept("order book", r"order\s+book|market\s+depth|\bdepth\b",
            "The order book is the list of resting buy (bid) and sell (ask) orders at each price; "
            "depth is how much is resting near the current price.",
            "A deep book absorbs a large order with little slippage; a thin one moves on it.",
            "quote", "spread"),
    Concept("liquidity", r"liquidity|illiquid",
            "Liquidity is how easily something can be bought or sold without moving its price: a "
            "tight spread and a deep order book.",
            "Illiquid markets cost more to trade and can gap when news arrives."),
    # Macro
    Concept("CPI", r"\bcpi\b|consumer\s+price\s+index",
            "CPI, the consumer price index, measures the change in prices US consumers pay for a "
            "basket of goods and services; its yearly change is the headline inflation rate.",
            "A higher-than-expected CPI makes rate cuts less likely, which usually weighs on "
            "stocks and on crypto."),
    Concept("FOMC", r"\bfomc\b|federal\s+open\s+market",
            "The FOMC is the Federal Reserve committee that sets US interest rates, at eight "
            "scheduled meetings a year.",
            "Markets move on the decision and on what the statement and the chair's press "
            "conference say about the next ones."),
    Concept("yield curve", r"yield\s+curve|inverted\s+curve",
            "The yield curve plots government bond yields by maturity; normally longer bonds yield "
            "more.",
            "An inverted curve (short yields above long) has preceded most US recessions, with a "
            "long and variable lag."),
    Concept("bull and bear market", r"bull\s+market|bear\s+market",
            "A bull market is a sustained rise; a bear market is a fall of 20% or more from a "
            "recent high.",
            "The labels describe what has happened, not what will; rallies inside bear markets are "
            "common and sharp."),
    Concept("Agent Hub dry run", r"dry[\s-]*runs?|agent\s+hub",
            "Agent Hub is Bitget's official toolkit for AI agents (its bgc command line, MCP "
            "server and SDK); a dry run (--dry-run) builds the exact order request and shows "
            "it without sending it, so no order is placed.",
            "This console only ever shows dry-run commands: ask \"how should I split a $2k "
            "order in BTC\" and the answer ends with the bgc command that previews it."),
    Concept("APY and APR", r"\bapy\b|\bapr\b|annual\s+percentage\s+(?:yield|rate)",
            "APR is the yearly interest rate without compounding; APY is the same rate with "
            "compounding counted, so APY is always at least APR.",
            "A 10% APR paid daily and reinvested is about 10.5% APY; a quoted APY on a crypto "
            "product says nothing about the risk of the thing paying it."),
    Concept("bitcoin", r"bitcoin|btc",
            "Bitcoin is a digital currency that runs on a public network no company or government "
            "controls; its supply is capped at 21 million coins and it trades around the clock.",
            "Its price is set only by what buyers pay: it has no earnings or dividends, and it "
            "has fallen more than 70% from a peak more than once.", "quote", " last ", "BTCUSDT",
            asset=True),
    Concept("ethereum", r"ethereum|\beth\b|ether\b",
            "Ethereum is a public blockchain that runs programs (smart contracts); ether (ETH) is "
            "the coin used to pay for them and trades around the clock.",
            "Much of crypto's lending, trading and stablecoin activity runs on it, so its price "
            "moves with that activity as well as with crypto overall.", "quote", " last ",
            "ETHUSDT",
            asset=True),
    Concept("fear and greed index", r"fear\s*(?:&|and)\s*greed",
            "The fear and greed index is a 0 to 100 score of market mood built from price, "
            "volatility and positioning measures.",
            "Extreme greed means the crowd is stretched long; like RSI, it describes a mood, not a "
            "turning point.", "sentiment", "fear & greed"),
)

explains(*(c.definition for c in CONCEPTS), *(c.reading for c in CONCEPTS))

_ASK = re.compile(
    r"\b(?:what(?:'s|\s+is|\s+are|\s+does|\s+do)|explain|define|definition\s+of|meaning\s+of|"
    r"how\s+does|how\s+do\s+(?:i|you)\s+(?:read|use|interpret)|tell\s+me\s+about|"
    r"eli5|like\s+i'?m\s+(?:new|five|5|a\s+beginner))\b", re.I)
_MEANS = re.compile(r"\bdo\s*[?.!]*\s*$|"
                    r"\b(?:mean|means|meaning|explain\w*|define|definition|eli5|like\s+i'?m|"
                    r"beginner|new\s+to\s+trading|how\s+(?:does|do)\s+\S+\s+work|"
                    r"difference\s+between|in\s+plain\s+(?:terms|english|words))\b", re.I)


_OWNED = re.compile(r"\b(?:my|your|our|the\s+desk'?s?|this\s+book)\b", re.I)
"""A term asked about someone's own figures: "what's my max drawdown", "what is your sharpe" are
questions for the book and the desk's record, not for a definition."""


def concept_asked(text: str, named_symbols: tuple[str, ...] = ()) -> Concept | None:
    """The concept a question asks the meaning of, or None.

    Taken when the question asks what a term means (mean, explain, define, "like I'm new"), or is
    nothing but "what is a <term>" / "what is <term>" with no contract named. "what is NVDA's RSI"
    names a contract and asks for its reading; "what is the sharpe" and "what's my drawdown" ask
    for the desk's or the book's own figure. Neither is a concept question."""
    # Typed as a newcomer types: "whats a perp", "waht is dca", "wats sharpe ratio" were declined
    # or answered with the desk's own Sharpe (a first-time-user audit, 2026-09-30).
    text = re.sub(r"\b(?:what|wat|waht|whta|wht)'?s(?=\s)", "what is",
                  text, flags=re.I)
    text = re.sub(r"\b(?:waht|whta|wht|wat|wut|whaat)\b", "what", text, flags=re.I)
    # Hinglish, as typed: "bitcoin kya hai" / "stop loss kya hota hai" is "what is ..." (a
    # first-time user, round 11, got BTC's trading cost).
    hinglish = re.fullmatch(r"\s*(.+?)\s+kya\s+(?:hai|hota\s+hai|hoti\s+hai|h)\s*[?.!]*\s*", text,
                            re.I)
    if hinglish is not None:
        text = f"what is {hinglish.group(1)}"
    # "how risky is leverage trading" asks what the term carries (round 11).
    risky = (re.match(r"^\s*how\s+risky\s+(?:is|are)\s+(.+?)[?.!\s]*$", text, re.I)
             or re.match(r"^\s*(?:is|are)\s+(.+?)\s+(?:risky|dangerous)[?.!\s]*$", text, re.I))
    if risky is not None:
        text = f"explain {risky.group(1)}"
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
        if bare is not None and (not named_symbols or concept.asset):
            return concept
    return None


def second_concept(text: str, first: Concept) -> Concept | None:
    """The other term of "what is the difference between calls and puts", when the question
    compares two."""
    if not re.search(r"\b(?:difference|differ|vs\.?|versus|compared?\s+(?:to|with))\b", text,
                     re.I):
        return None
    for concept in CONCEPTS:
        if concept is not first and concept.definition != first.definition and re.search(
                rf"\b(?:{concept.pattern})\b", text, re.I):
            return concept
    return None


def answer(concept: Concept, symbol: str | None, also: Concept | None = None
           ) -> tuple[list[str], list[Source], dict[str, object]]:
    """The definition, how to read it, and its live reading on ``symbol`` (or the concept's own
    example contract); with ``also``, the second term of a comparison beside it."""
    lines = [f"Bottom line: {concept.definition}", f"How to read it: {concept.reading}"]
    if also is not None:
        lines += [f"{also.name[:1].upper()}{also.name[1:]}: {also.definition}",
                  f"How to read it: {also.reading}"]
    sources = [Source(kind="computation", ref="argus.lui.concepts",
                      detail=f"definition of {concept.name}, written text")]
    if concept.cited:
        sources.append(Source(kind="venue", ref="Bitget", detail=concept.cited))
    example = ""
    if concept.kind and concept.marker:
        from argus.lui.research import ResearchKind, ResearchRequest, run

        target = symbol or concept.example
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
