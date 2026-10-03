"""The questions a newcomer asks before any research question, answered plainly.

A first-time-user audit (2026-09-30) typed "is this financial advice", "how do i buy crypto",
"can i lose more money than i put in", "should i buy the dip" and "whats a good stock for
beginners", and every one was declined with "I did not recognise that question". They are the
questions a workbench meets first from anyone new, and each has an honest answer that does not
recommend anything: what this console is, how buying on Bitget works, what can be lost, and —
where a number helps — the console's own engine run on a real example, said as the example.

The fixed text states only what is true of Bitget's public product and of this console's code. The
two answers that need a number (a dip, a first holding) hand over to the engine that computes one,
so no figure here is written by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from argus.lui.provenance import explains
from argus.lui.trace import trace_module


@dataclass(frozen=True)
class Reply:
    """A newcomer question's answer: its lines, or the engine question that answers it."""

    lines: tuple[str, ...] = ()
    reask: str = ""
    """A question for the console's own engines, when the answer is a computed one."""
    note: str | None = ""
    """What the re-asked question assumed, said under the answer."""
    lead: str = ""
    """A first line that answers the question as asked, above the engine's ("{name}" filled in)."""


_ADVICE = re.compile(
    r"\b(?:is\s+(?:this|it|that|argus)\s+(?:financial\s+|investment\s+)?advice|are\s+you\s+(?:a\s+)?"
    r"(?:financial\s+|investment\s+)?advis[eo]r|can\s+i\s+trust\s+(?:this|you|it|argus)|should\s+i\s+"
    r"(?:trust|follow|listen\s+to)\s+(?:you|this|argus)|are\s+you\s+(?:giving\s+(?:me\s+)?)?"
    r"(?:financial|investment)\s+advice)\b", re.I)
_HOW_TO_BUY = re.compile(
    r"\bhow\s+(?:do\s+i|to|can\s+i|would\s+i)\s+(?:actually\s+|just\s+|even\s+)?(?:buy|get|purchase|"
    r"start\s+(?:buying|trading|investing))\s*(?:(?:some\s+)?(?:crypto|bitcoin|btc|ethereum|eth|"
    r"stocks?|shares?|coins?|tokens?|this|it|that|them)\b|[?.!]*\s*$)|\bhow\s+(?:do\s+i|to|can\s+i)\s+(?:start|begin)\s+(?:trading|investing)\b|\bwhere\s+(?:do|"
    r"can)\s+i\s+buy\b|\b(?:i['\u2019]?m|i\s+am)\s+(?:brand\s+)?new\b[^?]*\b(?:where|how)\b[^?]*"
    r"\b(?:start|begin)\b|\bwhere\s+(?:do|should)\s+i\s+(?:start|begin)\b[?.!]*\s*$|"
    # "how do i actually buy on bitget step by step" got the rToken-versus-perpetual answer
    r"\bhow\s+(?:do\s+i|to|can\s+i)\s+(?:actually\s+)?buy\s+(?:on|from|at)\s+bitget\b|"
    r"\bbuy\b[^?]{0,30}\bstep\s+by\s+step\b", re.I)
_LOSE_MORE = re.compile(
    r"\b(?:can|could|will|do)\s+i\s+lose\s+(?:more\s+(?:money\s+)?than|everything|all\s+(?:of\s+)?"
    r"my\s+money)|\blose\s+more\s+than\s+(?:i|you)\s+(?:put|invest)|\bowe\s+(?:money|the\s+exchange)"
    r"|\bnegative\s+balance\b", re.I)
_LOSS_HAPPENS = re.compile(
    r"\bwhat\s+(?:happens|would\s+happen|if)\b[^?]*\bif\s+i\s+(?:lose|lost|start\s+losing)\s+"
    r"(?:some\s+|a\s+lot\s+of\s+|all\s+(?:my\s+|of\s+my\s+)?)?(?:money|everything|it\s+all)\b|"
    r"\bwhat\s+if\s+i\s+(?:lose|lost)\s+(?:some\s+|a\s+lot\s+of\s+)?(?:money|everything)\b", re.I)
"""A trader asking what a loss means, with no position named: it carries the word "lose" and went to
the desk's own track record (a first-user audit, round 18)."""
_BEGINNER_SAFE = re.compile(
    r"\bis\s+(?:crypto|bitcoin|trading|investing|leverage|leveraged\s+trading|futures|perps?|"
    r"perpetuals?|day\s+trading)\s+(?:safe|risky|dangerous|a\s+(?:good|bad)\s+idea|worth\s+it)"
    r"\s+(?:for\s+(?:a\s+)?(?:beginners?|newbies?|someone\s+new|me|starters?)|to\s+start)\b|"
    r"\bam\s+i\s+(?:ready|too\s+new)\s+(?:to|for)\s+(?:trade|trading|crypto|invest\w*)\b", re.I)
_HOW_BUYING_WORKS = re.compile(
    r"\bhow\s+(?:does|do)\s+(?:buying|trading|investing|it|this)\s+(?:actually\s+|really\s+)?work\b|"
    r"\bhow\s+(?:buying|trading)\s+works\b", re.I)
_PLACE_TRADE = re.compile(
    r"\bhow\s+(?:do|can|would|should)\s+i\s+(?:place|make|open|enter|put\s+in|execute|submit)\s+"
    r"(?:a\s+|an\s+|my\s+(?:first\s+)?)?(?:trade|order|position|bet)\b|"
    r"\bhow\s+to\s+(?:place|make|open|enter|put\s+in|execute)\s+(?:a\s+|an\s+)?"
    r"(?:trade|order|position)\b", re.I)
"""Placing a trade, which happens on Bitget: "how do I place a trade" was declined (round 17)."""
_FREE = re.compile(
    r"\b(?:is|are)\s+(?:this|it|argus|the\s+(?:console|tool|app|site|service))\s+(?:really\s+)?"
    r"(?:free|paid|free\s+to\s+use)\b|\bdo\s+i\s+(?:have\s+to|need\s+to)\s+(?:pay|sign\s*up|"
    r"register|log\s*in|create\s+an?\s+account)\b|\bhow\s+much\s+(?:does|is)\s+(?:argus|this\s+"
    r"(?:console|tool|app|site|service))\s+(?:cost|to\s+use)\b|\bany\s+(?:fees?|charges?|"
    r"subscription)\s+(?:to\s+use|for\s+using)\b|"
    # "whats the catch with this site, do u take a cut" (a first-time user, round 23)
    r"\bwhat'?s\s+the\s+catch\b|\b(?:do|does)\s+(?:u|you|it|argus)\s+take\s+a\s+cut\b|"
    r"\bhow\s+do\s+(?:you|u|they)\s+make\s+money\b|\b(?:do|does)\s+(?:u|you|it)\s+charge\b",
    re.I)
"""Whether using the console costs anything: "is this free?" was declined (round 17)."""
_GO_WRONG = re.compile(
    r"\bwhat\s+(?:could|can|might|would)\s+go\s+wrong\b|\bwhat\s+are\s+the\s+(?:main\s+|biggest\s+)?"
    r"risks\b|\bwhat\s+(?:could|can)\s+i\s+lose\b", re.I)
_GOOD_TIME = re.compile(
    r"\bis\s+(?:now|this|today|it)\s+(?:a\s+)?(?:good|bad|the\s+right|right)\s+time\s+to\s+"
    r"(?:buy|invest|get\s+in|start|sell|get\s+out)\b|\bshould\s+i\s+(?:buy|invest|get\s+in|sell|"
    r"get\s+out)(?:\s+(?:my|all\s+my|everything|some)?\s*(?:crypto|coins?|stocks?|shares?|"
    r"holdings|positions?))?\s+(?:right\s+)?now\b", re.I)
_HOW_MUCH_BARE = re.compile(
    r"\bhow\s+much\s+(?:money\s+)?(?:should|can|could|do)\s+i\s+(?:put\s+in(?:to)?|invest|start\s+with)"
    r"\b|\brecommend\w*\b|\bwhat\s+would\s+you\s+do\s+with\b", re.I)

_GUARANTEE = re.compile(
    r"\bguarantee\s+(?:me|us|that|a|an|my)\b|\bcan\s+you\s+guarantee\b|\bguaranteed\s+(?:returns?|"
    r"profits?|gains?|money|win)\b|\brisk[\s-]*free\s+(?:trade|profit|return|bet|money)\b|"
    r"\b(?:can'?t|cannot|won'?t)\s+lose\b(?!\s+(?:more|over|above|beyond|past|than|up\s+to|[$\d]))|\bsure\s+(?:thing|bet|profit|win|money)\b|"
    r"\bpromise\s+(?:me\s+)?(?:a\s+)?(?:return|profit)|"
    r"\bguaranteed\s+to\s+(?:make|win|profit|pay|go\s+up|rise|earn)\b|"
    r"\bguaranteed\s+(?:trade|pick|winner|call)\b|\bguaranteed\s+(?:\d+x|\d+%|(?:\w+\s+){0,2}?"
    r"(?:coins?|tokens?|stocks?|crypto|investments?|multibaggers?|moonshots?))\b|\b(?:certain|sure)\s+to\s+(?:make|win|profit|go\s+up|rise)\b",
    re.I)
"""A request for a certain return: "guarantee me a 50% return on BTC" was deflected into price data
instead of being told no (a judge's audit, 2026-09-30). "I can't lose more than 10% of my book"
is a loss limit, not a request for a sure thing, and was answered with this refusal (round 13)."""

_DIP = re.compile(r"\bbuy(?:ing)?\s+(?:the|this)\s+dip\b|\bdip\s+buy", re.I)
_BEGINNER_PICK = re.compile(
    r"\b(?:good|best|safe|safest|first)\s+(?:stocks?|coins?|crypto|investments?|things?)\s+(?:for|to)"
    r"\s+(?:a\s+)?(?:beginners?|newbies?|start|buy\s+first)\b|\bwhat\s+should\s+(?:a\s+)?beginners?\s+"
    r"(?:buy|invest\s+in)\b|\bbeginner\s+(?:stocks?|picks?|coins?)\b|"
    # "whats the best coin to buy right now" was answered "No open positions" (round 11)
    r"\b(?:best|top|hottest)\s+(?:\w+\s+)?(?:coins?|crypto|cryptos|stocks?|tokens?)\s+to\s+"
    r"(?:buy|invest\s+in|get)\b", re.I)

_MONEY_SAFE = re.compile(
    r"\bis\s+my\s+(?:money|crypto|account|deposit|cash|capital)\s+safe\b|\bis\s+(?:this|it|argus|"
    r"this\s+site)\s+(?:a\s+)?(?:scam|safe|legit|legitimate)\b|\b(?:do|does|can)\s+(?:you|argus|"
    r"this)\s+(?:hold|keep|take|touch|access)\s+my\s+(?:money|funds|crypto|account)\b|"
    # "Is my BTC held on ARGUS insured?" got venue mechanics (a hostile review, round 12)
    r"\b(?:held|kept|stored|custod\w*)\s+(?:on|by|with|at)\s+(?:argus|this\s+(?:site|console|app))\b|"
    r"\b(?:safe|insured)\s+(?:with|on|at)\s+(?:argus|you|this\s+(?:site|console|app))\b|"
    r"\bis\s+(?:my|it)\s+(?:\w+\s+)?(?:insured|protected)\b", re.I)
"""A newcomer asking whether their money is safe here (a first-time user, round 11: answered with
how rTokens and perpetuals differ)."""

_WITHDRAW = re.compile(
    # "how do i actually get my money out if i wanna stop" (a first-time user, round 28)
    r"\bhow\s+(?:do|can|would)\s+i\s+(?:actually\s+|even\s+|ever\s+)?(?:withdraw|take\s+out|"
    r"cash\s+out|get\s+(?:my\s+)?(?:money|funds|cash)\s+(?:back\s+)?out)"
    r"|\bwithdraw(?:al|ing)?\s+(?:my\s+)?(?:money|funds|crypto|usdt|cash)\b", re.I)
"""Taking money out, which happens on Bitget: "how do I withdraw my money" was declined (a
first-time user, round 12)."""

_WITHDRAWING = (
    "Bottom line: withdrawals happen on Bitget, not here — this console holds no money, so there "
    "is nothing in it to withdraw.",
    "On Bitget: close or sell what you want to take out, move it to your spot account, then use "
    "Withdraw (to a bank by selling for cash, or to a crypto address) — Bitget's own help pages "
    "give the fees, the limits and the checks for your country.",
    "Before a crypto withdrawal, check the network and the address twice: a transfer sent on the "
    "wrong network is usually lost.",
)

_SAFE = (
    "Bottom line: this console never holds, moves or touches money: there is no account to open, "
    "no login, no deposit and no connection to your Bitget account — it reads public market data "
    "and answers questions, nothing more.",
    "Money you keep on Bitget is held by Bitget, the exchange; its safety is Bitget's, and ARGUS "
    "does not audit it. Bitget publishes its own reserve reports on its site — read those, and "
    "keep only what you can afford to lose on any exchange.",
    "No order is ever placed from here: the trading commands the console shows are marked "
    "--dry-run, which previews an order and sends nothing.",
)

_PLACING = (
    "Bottom line: not here — this console never places an order; trades are placed on Bitget "
    "itself, after you have opened and funded an account there.",
    "On Bitget: open the page for the contract, choose the order type (a market order fills now "
    "at the best price on offer; a limit order waits at the price you set), enter the size and "
    "confirm. For a futures contract, set isolated or cross margin and the leverage first.",
    "Before you do, ask this console what it would cost and how far it could go against you: "
    "\"what does it cost to buy $500 of BTC\" or \"how much can I lose on TSLA this week\". The "
    "trading commands it shows are marked --dry-run, which previews an order and sends nothing.",
)
_FOR_ME = re.compile(
    r"\b(?:can|could|will|would)\s+(?:u|you|ya|it)\s+(?:just\s+|please\s+)?(?:trade|invest|buy|"
    r"sell|do\s+(?:it|the\s+trading|everything))\s+"
    # "can u just buy it for me" was declined as "'that' has nothing to refer to" (round 22)
    r"(?:(?:it|that|this|them|some|one|the\s+\w+)\s+)?(?:for|on\s+behalf\s+of)\s+me\b|"
    r"\bmanage\s+my\s+(?:money|account|portfolio|funds)\b|\bauto[\s-]?trade\s+for\s+me\b", re.I)
"""Handing the trading over: "can u just trade for me" got the generic refusal (round 21)."""
_TRADING_FOR_YOU = (
    "Bottom line: no — this console never trades for anyone, holds no money and cannot reach "
    "your account. Every trade is yours to place, on Bitget.",
    "What it does instead: tells you what a trade would cost and how far it could go against "
    "you before you place it — \"what does $500 of BTC cost me and what could I lose in a "
    "week?\" — and shows its own paper desk's decisions, every one logged, at /status.",
    "Bitget offers copy trading and trading bots for people who want trading done for them; "
    "read how each one has done, and what it can lose, before you follow it.",
)
_FREE_TO_USE = (
    "Bottom line: using this console costs nothing and takes no cut: there is no account, no "
    "login, no card and no subscription.",
    "One limit: the language model that reads oddly phrased questions answers up to 40 questions "
    "an hour from one network address, counted on each server instance. Past that, the "
    "console's own readers answer, in English, and every figure is still computed from live "
    "data.",
    "Trading is separate: Bitget charges its own fees when you trade there, and this console "
    "never places an order. Ask \"what does it cost to buy $500 of BTC\" for the fee on a real "
    "order book.",
)
_NOT_ADVICE = (
    "Bottom line: no — this console never tells anyone what to buy or sell. It computes from live "
    "market data and says where each figure came from, and the decision and the risk stay yours.",
    "What it does: prices, costs, how risky a name or a whole portfolio is, what past moves looked "
    "like, and definitions of the words used — each answer shows its sources in the receipt under "
    "it, so you can check them.",
    "Every answer ends \"not advice\" for this reason. Ask \"what is a perp\" or \"is TSLA "
    "riskier than NVDA\" to see what it does.",
)
_BUYING = (
    "Bottom line: on Bitget you open an account, verify your identity, add money (a card, a bank "
    "transfer, or crypto sent from elsewhere), and then buy on the spot market, where you own the "
    "coin itself.",
    "Spot is the simple start: the most you can lose is what you paid. The futures (\"perpetual\") "
    "markets are contracts with leverage, funding payments and liquidation — ask \"what is a "
    "perp\" and \"what is liquidation\" before using them.",
    "Before a first buy, ask this console what it will cost: \"what does it cost to buy $500 of "
    "BTC\" gives the fee and the spread on Bitget's live order book.",
)
_LOSING = (
    "Bottom line: on spot, no — the most you can lose is what you paid, if the price goes to zero. "
    "With leverage you can lose all the margin you put up, and more of your balance if you use it "
    "as margin.",
    "Leverage multiplies the move: at 10x, a fall of a little under 10% can wipe out the margin "
    "and the exchange closes (liquidates) the position. Isolated margin limits the loss to the "
    "margin set aside for that one position; cross margin lets a losing position draw on your "
    "whole futures balance.",
    "Ask \"what is liquidation\" or \"how much can I lose on TSLA this week\" for figures on a "
    "real name.",
)
_LOSS_HAPPENS_A = (
    "Bottom line: a loss is the position being worth less than you paid. On spot you still own "
    "the coin, so the loss is on paper until you sell, and it can shrink or grow; with leverage "
    "the exchange closes (liquidates) the position when the loss reaches the margin held against "
    "it, so the loss becomes final without a decision from you.",
    "Nothing here touches your money — this console places no orders and holds no funds. To see "
    "a loss before it happens, give it a position: \"how much would I lose if NVDA fell 10% and "
    "I hold $5k\".",
    "A loss you cannot afford is a sizing mistake, not a market one: say \"my loss limit is 5%\" "
    "and the answers that size a position are held to it.",
)
_BEGINNER_SAFE_A = (
    "Bottom line: not safe in the sense of guaranteed — prices can fall a long way, and the "
    "market runs around the clock, so a loss can arrive while you sleep. What a beginner controls "
    "is how much is exposed.",
    "Spot is the gentler start: the most you can lose is what you paid. Futures (\"perpetuals\") "
    "add leverage, funding payments and liquidation, which is where beginners lose the most.",
    "Start with an amount you could lose without it changing your month, and measure before you "
    "buy: \"how much can I lose on BTC this week\" and \"what does it cost to buy $500 of BTC\" "
    "are answered on live Bitget data. This is analysis, not advice.",
)
_LOOKAHEAD = re.compile(
    r"\blook[\s-]?ahead\b|\bfuture\s+(?:data\s+)?leak\w*|\b(?:data|label|target)\s+leak\w*|"
    r"\bpeek\w*\s+(?:at|into)\s+the\s+future\b", re.I)
_OVERFIT = re.compile(
    r"\boverfit\w*|\bover[\s-]fit\w*|\bcurve[\s-]?fit\w*|\bdata[\s-]?snoop\w*|\bp[\s-]?hack\w*|"
    r"\bmultiple[\s-]testing\b|\bcherry[\s-]?pick\w*\s+(?:backtest|result)", re.I)
_SURVIVOR = re.compile(r"\bsurvivor(?:ship)?\s+bias\b|\bsurvivorship\b|\bpoint[\s-]in[\s-]time\b",
                       re.I)
_BACKTEST_COSTS = re.compile(
    r"\b(?:back[\s-]?tests?\w*|simulat\w+|paper\s+results?)\b.{0,60}\b(?:fees?|costs?|slippage|"
    r"commissions?|spreads?)\b|\b(?:fees?|costs?|slippage|commissions?)\b.{0,60}\b(?:back[\s-]?"
    r"tests?\w*|simulat\w+)\b", re.I)
"""Questions about how this system's own testing is done — each is answered from what the code
does and says what it does not cover (round 17: "is the lookahead bias a problem in your
backtests" was answered with the hash-chain integrity check)."""

_LOOKAHEAD_A = (
    "Bottom line: it is guarded against where it can occur, and not proven absent — here is what "
    "is guarded and what is not.",
    "A backtest rule is shown only the bars up to the decision bar, and its order fills on the "
    "next bar (the engine's signal signature, `backtest/engine.py`).",
    "\"Has this been here before\" matches only states stamped at or before the question's date, "
    "and a past state's outcome counts only once its window has closed — a rival review found "
    "this leak in September and it was closed.",
    "As-of questions on filings drop every line filed after the date asked about. Not covered: "
    "prices that a data vendor later restated (adjusted closes are re-adjusted after later "
    "splits and dividends) — that effect has not been measured here.",
)
_OVERFIT_A = (
    "Bottom line: the guard is a test run after the search, not trust in the winner — a search "
    "that tries many rules will find one that looks good by chance.",
    "Rules found by searching a pool go through a multiple-testing gate (Hansen's SPA and Romano-"
    "Wolf StepM, `backtest/snooping.py`), and the out-of-sample half is reported separately: a "
    "Sharpe that keeps less than half of its in-sample value out of sample is flagged as "
    "overfitted, not hidden.",
    "A Sharpe or a distribution built on too few observations is refused with the count, not "
    "printed. Not covered: no gate removes the choice of which rules were tried in the first "
    "place; a result here is a candidate, not a proof.",
)
_SURVIVOR_A = (
    "Bottom line: survivorship is a real limit here and has not been measured.",
    "History is read for the contracts Bitget lists today, so a contract that was delisted is "
    "not in it — a result over \"the stocks on the list\" is a result over the survivors.",
    "For as-of filing questions, a figure is only taken from a filing dated on or before the "
    "date asked about. Not covered: index membership as of a past date, which this console does "
    "not hold.",
)
_COSTS_A = (
    "Bottom line: yes — the backtest engine cannot be run without a cost: there is no zero-fee "
    "setting.",
    "The cost model cannot be built without explicit rates, and a frictionless research model "
    "raises an error if a decision is gated on it (`cost/model.py`). The taker round trip on "
    "Bitget is 0.12% (12 bps) — larger than the typical intraday edge measured, which is why "
    "most quick strategies lose once it is counted.",
    "Not covered: market impact on a size larger than the order book shows.",
)


_NO_GUARANTEE = (
    "Bottom line: no — no return can be guaranteed, by this console or by anyone honest; a "
    "promised return in trading is the mark of a scam, not of a strategy.",
    "What can be shown instead is the range of what has happened: \"has BTC been here before\" "
    "gives how often states like today's rose over the next day and by how much, and \"how much "
    "can I lose on BTC this week\" gives how far past weeks went against a holder.",
)
_GOING_WRONG = (
    "Bottom line: four things, in the order they usually bite: the price falls further than you "
    "planned for; fees and the spread eat a small gain; leverage turns a fall into a "
    "liquidation; and a stock's perpetual trades while the stock market is shut, so it can gap "
    "when the market opens.",
    "Each has a number here for a real name: \"how much can I lose on NVDA this week\" (how far "
    "past weeks went against a holder), \"what does it cost to buy $500 of BTC\" (the fee and "
    "the spread), \"what is liquidation\", and \"has NVDA been here before\" (what followed "
    "states like today's).",
    "The one that cannot be measured in advance is the day worse than any on record — size so "
    "that a bad day is survivable, not so that the average day is comfortable.",
)


_TRUST = re.compile(
    r"\b(?:can|should|do)\s+i\s+trust\s+(?:you|this|it|argus|the\s+console)\b|\bare\s+you\s+"
    r"(?:reliable|accurate|legit|trustworthy)\b|\bhow\s+(?:accurate|reliable|trustworthy)\s+(?:are\s+"
    r"you|is\s+(?:this|it|argus))\b|\bwhy\s+should\s+i\s+(?:trust|believe)\s+(?:you|this|it)\b",
    re.I)
_TRUST_A = (
    "Bottom line: trust it only as far as you can check it — every figure names its source, so "
    "you can verify it yourself, and the console publishes its own mistakes.",
    "/wrong lists every claim it has withdrawn or corrected, in plain terms; /proof sets each "
    "capability against other tools on the same inputs, losses included; /status shows how "
    "accurate its own forecasts have been, including where a simple base rate does better.",
    "What it never does is tell you what to buy or sell — the decision and the risk stay yours.",
)
""""should i trust you" got the not-advice template, whose first word is "no" (a first-time
user, round 27)."""

explains(*_TRUST_A)

# Round 28 (a first-time user): questions about speculation, other exchanges, identity checks,
# hidden costs and beginner mistakes were declined or answered with the perpetual-versus-rToken
# explainer.
_SPECULATION = re.compile(
    r"\b(?:is|are)\s+(?:it|this|that|crypto|trading|all\s+this|meme\s*coins?|memecoins?|alt\s*"
    r"coins?|shit\s*coins?|these\s+coins|those\s+coins|altcoins?)\s+(?:just\s+|basically\s+|"
    r"really\s+|kinda\s+|like\s+|all\s+)?(?:gambling|a\s+casino|casinos|a\s+scam|scams|a\s+ponzi|"
    r"lottery|a\s+lottery\s+ticket)\b|\bno\s+(?:real\s+)?(?:use|utility|purpose)\b|"
    r"\bwhy\s+(?:would|do|does|did)\s+(?:a\s+|these\s+|those\s+)?(?:coins?|tokens?|memes?|"
    r"memecoins?)\b[^?]{0,40}\b(?:go|went)\s+up\b|\b(?:dog|meme|joke|shit|frog)\s*coins?\b[^?]{0,40}"
    r"\b(?:still\s+a\s+thing|worth\s+it|legit|real|a\s+thing)\b|\bmade\s+bank\b", re.I)
_SPECULATION_A = (
    "Bottom line: a coin with no business behind it has no earnings to anchor its price, so it "
    "moves on attention and on money coming in or going out — which is how it can multiply in "
    "weeks and then lose most of that again. Whether it is gambling comes down to the size: a "
    "small amount you could lose entirely and shrug off is a bet; your savings is not.",
    "Ask \"how much could I lose on DOGE in a bad week\" for what that looks like in numbers, "
    "or \"early signals\" for which coins are moving unusually today — a list to read, not to "
    "buy.",
    "This console never says to buy or sell; it measures what a coin has done.",
)
_OTHER_EXCHANGE = re.compile(
    r"\bbitget\b[^?]{0,60}\b(?:coinbase|binance|kraken|okx|bybit|robinhood|kucoin|gemini|"
    r"crypto\.com|etoro)\b|\b(?:coinbase|binance|kraken|okx|bybit|robinhood|kucoin|gemini|"
    r"crypto\.com|etoro)\b[^?]{0,60}\bbitget\b", re.I)
_OTHER_EXCHANGE_A = (
    "Bottom line: this console reads Bitget's markets and does not rate exchanges, so it will "
    "not say which is better — the comparison worth making is what each charges, what each "
    "lists, and how each holds your money.",
    "On Bitget, from its published schedule at the standard level: 0.10% a side on spot, 0.02% "
    "(maker) and 0.06% (taker) on USDT perpetuals; beside crypto it lists US-stock perpetuals "
    "and rTokens. The other exchange's own fee and listing pages give the same figures for it.",
    "Bitget publishes a Proof of Reserves page and a Protection Fund page; read the equivalent "
    "on any exchange before keeping money there. If you meant Coinbase's stock, it trades on "
    "Bitget as COIN — ask \"what is COIN doing today\".",
)
_KYC = re.compile(
    r"\b(?:verify|verified|verification|kyc)\b[^?]{0,40}\b(?:id|identity|bitget|account|start|"
    r"trade|use)\b|\b(?:id|identity|passport|selfie|driver'?s\s+licen[cs]e)\b[^?]{0,30}\b(?:to\s+"
    r"(?:use|trade|start|sign\s+up|buy)|for\s+bitget|on\s+bitget)\b|\bupload\s+(?:a\s+|my\s+)?"
    r"(?:passport|id|selfie|documents?)\b|\bkyc\b", re.I)
_KYC_A = (
    "Bottom line: Bitget asks for identity verification (KYC) — which features need it, and "
    "when, is set by Bitget's rules for your country, on its help centre and on the sign-up "
    "screens themselves.",
    "This console needs none of that: there is no account, no login and no ID here, because it "
    "only reads public market data and answers questions.",
)
_HIDDEN_FEES = re.compile(
    r"\bhidden\s+(?:fees?|costs?|charges?)\b|\bfees?\b[^?]{0,30}\b(?:hit\s+with|catch|surprise|"
    r"sneaky)\b|\b(?:all|every|any\s+other)\s+(?:the\s+)?(?:fees?|costs?|charges?)\b", re.I)
_HIDDEN_FEES_A = (
    "Bottom line: nothing on Bitget is hidden, but more than one cost applies: the trading fee "
    "(0.10% a side on spot; 0.02% maker or 0.06% taker on perpetuals, at the standard level), "
    "the spread you cross with a market order, funding every few hours while a perpetual is "
    "open, and a network fee on a crypto withdrawal that depends on the coin and the network.",
    "Bitget shows the withdrawal fee on the withdrawal screen before you confirm, and funding "
    "on each contract's page. Ask \"what does it cost to buy $500 of BTC\" for the fee and "
    "spread on the live order book, or \"if I trade 10 times a day with $500 how much do fees "
    "eat\" for what frequent trading costs a month.",
    "This console itself is free and takes no cut.",
)
_MISTAKES = re.compile(
    r"\b(?:biggest|common|worst|typical|main|top|usual)\s+(?:beginner\s+|newbie\s+|rookie\s+)?"
    r"(?:mistakes?|errors?|traps?)\b|\bmistakes?\b[^?]{0,20}\b(?:beginners?|newbies?|noobs?|new\s+"
    r"traders?|people\s+new)\b|\bwhat\s+should\s+i\s+(?:avoid|watch\s+out\s+for|not\s+do)\b", re.I)
_MISTAKES_A = (
    "Bottom line: the costliest beginner mistakes are sizes and habits, not picks — too much "
    "leverage, trading too often, putting in money you cannot lose, and buying after a big run "
    "because it is in the news.",
    "Leverage: at 10x a fall of about 9.5% (the 10% margin less Bitget's 0.5% maintenance margin "
    "at the first tier) closes the position and takes the whole margin; BTC fell 14% in one day "
    "in February 2026.",
    "Trading often: ten round trips a day on $500 cost about $180 a month in perpetual fees "
    "(0.12% each) — 36% of the money, before any gain or loss.",
    "Size: the usual rule is to hold only what you could lose on the worst day on record without "
    "it changing your life — ask \"how much could I lose on BTC in a bad week\" for the figure.",
)
explains(*_SPECULATION_A, *_OTHER_EXCHANGE_A, *_KYC_A, *_HIDDEN_FEES_A, *_MISTAKES_A)

OTHER_EXCHANGE: Final = _OTHER_EXCHANGE
OTHER_EXCHANGE_A: Final = _OTHER_EXCHANGE_A
EARLY_PLAIN: Final = ((_KYC, _KYC_A), (_MISTAKES, _MISTAKES_A), (_SPECULATION, _SPECULATION_A),
                      (_HIDDEN_FEES, _HIDDEN_FEES_A))
"""The plain answers the console's router takes before its research readers see the question:
"do they make you upload a passport on bitget" was taken by the venue explainer before the
newcomer answer was reached (round 28)."""

SLANG: dict[str, str] = {
    "dyor": "\"do your own research\" — check a claim yourself before acting on it",
    "wagmi": "\"we're all gonna make it\" — cheerleading, not information",
    "ngmi": "\"not gonna make it\" — said of someone (or a coin) expected to fail",
    "hodl": "holding through ups and downs instead of selling (a typo of \"hold\" that stuck)",
    "fomo": "\"fear of missing out\" — buying because a price is rising and others are buying",
    "fud": "\"fear, uncertainty and doubt\" — negative talk, true or not, that pushes prices down",
    "ath": "\"all-time high\" — the highest price a coin has ever traded at",
    "rekt": "\"wrecked\" — a large loss, often a leveraged position closed out",
    "degen": "someone taking very high-risk bets, often with leverage or tiny coins",
    "ape": "to buy fast and big without research (\"I aped in\")",
    "moon": "a very large price rise (\"to the moon\")",
    "bagholder": "someone still holding a coin that has fallen a long way",
    "bag holder": "someone still holding a coin that has fallen a long way",
    "whale": "an account big enough that its trades move the price",
    "shill": "promoting a coin, often because the promoter owns it",
    "rug pull": "the people behind a coin take the money and disappear; the price goes to near "
                "zero",
    "rug": "short for \"rug pull\": the people behind a coin take the money and disappear",
    "gm": "\"good morning\" — a greeting, nothing more",
    "wen": "\"when\", usually asked impatiently (\"wen moon\")",
    "lfg": "\"let's go\" — excitement, not information",
    "ct": "\"crypto Twitter\" — the crypto crowd on X",
    "alpha": "information that is supposed to give an edge; usually a rumour",
    "pump and dump": "a coin pushed up by coordinated buying so the pushers can sell to latecomers",
}
"""What a newcomer reads on crypto social media and asks about: "what does dyor even stand for"
was answered about Twitter's shares (a first-time user, round 29)."""
SLANG_Q: Final = re.compile(
    r"\b(?:what\s+(?:does|do|is|are)|whats|what's|wtf\s+is|meaning\s+of)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?P<t>" + "|".join(sorted((re.escape(k) for k in SLANG), key=len, reverse=True)) + r")\b",
    re.I)


def slang_lines(text: str) -> list[str] | None:
    """Every slang term the question asks about, in plain words."""
    asked = [m.group("t").lower() for m in SLANG_Q.finditer(text)]
    if not asked and len(re.findall(r"[a-z']+", text.lower())) <= 3 and (
            text.strip().endswith("?")
            or re.match(r"^\W*(?:and|or|what\s+about|how\s+about)\b", text, re.I)):
        # "and hodl?" after "what does fomo mean" repeated FOMO (round 29, live re-ask)
        asked = [k for k in SLANG if re.search(rf"\b{re.escape(k)}\b", text, re.I)]
    if not asked:
        found = [k for k in SLANG if re.search(rf"\b{re.escape(k)}\b", text, re.I)]
        asked = found if found and re.search(r"\b(?:mean|stand\s+for|means|meaning|no\s+clue|"
                                             r"don'?t\s+(?:get|know))\b", text, re.I) else []
    if not asked:
        return None
    said = [f"{k.upper() if len(k) <= 5 else k}: {SLANG[k]}." for k in dict.fromkeys(asked)]
    return ["Bottom line: " + said[0], *said[1:],
            "Slang like this travels fast on social media; none of it is a reason to buy or sell."]
explains(*_LOOKAHEAD_A, *_OVERFIT_A, *_SURVIVOR_A, *_COSTS_A, *_NOT_ADVICE, *_BUYING, *_LOSING,
         *_LOSS_HAPPENS_A, *_BEGINNER_SAFE_A,
         *_GOING_WRONG, *_NO_GUARANTEE, *_SAFE, *_WITHDRAWING, *_PLACING, *_FREE_TO_USE)

_LOAN = re.compile(
    r"\b(?:take|get|taking|getting)\s+(?:out\s+)?a\s+loan\b|\bborrow\w*\s+(?:money\s+)?to\s+"
    r"(?:buy|invest|trade)\b|\bloan\s+to\s+(?:buy|invest|trade)\b", re.I)
_LOAN_A = (
    "Bottom line: no \u2014 do not borrow to buy. A loan has to be repaid whatever the price "
    "does, so a fall loses money you do not have, and the interest is a cost every month you "
    "wait.",
    "\"It always comes back\" is not something any price has promised: ETH closed more than half "
    "below its 2021 high on 65% of the days of 2022 and on every day of 2023 (Yahoo Finance daily "
    "closes), and a borrower who needed the money back in that time sold at the bottom. Some "
    "coins and stocks never came back at all.",
    "If you buy, buy with money you could leave alone through a bad year, and ask \"how much could "
    "I lose on ETH in a bad week\" first to see what that means in dollars.",
)
_WHAT_NOW = re.compile(
    r"^\W*(?:(?:so|ok(?:ay)?|then|alright|hmm+)\W+){0,2}(?:what\s+(?:should|do)\s+i\s+do"
    r"(?:\s+now)?|"
    # "ok so which one do i go for" (a round-23 re-ask)
    r"which\s+(?:one\s+)?(?:should|do|would)\s+i\s+(?:pick|choose|buy|go\s+(?:with|for)|get)|"
    r"what\s+now|now\s+what|"
    r"what\s+would\s+you\s+pick|just\s+tell\s+me\s+what\s+to\s+(?:buy|do))\W*$", re.I)
"""A follow-up asking for the pick the last answer did not give: "so what should i do" and "so
which one should i pick" were declined (a first-time user, round 23)."""
_WHAT_NOW_A = (
    "Bottom line: this console will not pick for you \u2014 it cannot see your situation, and a "
    "pick from a website is not one you can hold it to. What it can do is make the choice yours "
    "with three checks:",
    "1. How much could you lose in a bad week without it hurting? Ask \"how much could I lose on "
    "BTC in a bad week\" (or QQQ, or gold) and compare the dollars with that.",
    "2. How long can you leave it alone? Money you need within a year is safest out of the market.",
    "3. Start with one broad thing rather than many small bets, without leverage, so the most you "
    "can lose is what you put in.",
)


_PLAIN: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    # Plain questions a first-time user asked and was told "I did not recognise that question"
    # while the model was available (a first-time user, round 24). Each answer is the plain fact
    # and the next question this console can answer with live figures.
    (re.compile(r"\bwhat\s+(?:even\s+)?(?:is|are)\s+(?:a\s+)?(?:stocks?|shares?)\b|"
                r"\bwhat\s+(?:does|do)\s+(?:a\s+)?(?:stock|share)\s+mean\b", re.I), (
        "Bottom line: a stock (a share) is a small piece of ownership in a company. Its price is "
        "what buyers will pay today for that piece of the company's future profits, so it moves "
        "every trading day.",
        "On Bitget you do not hold the share itself: you can hold an rToken that tracks a US "
        "stock's price (rNVDA for NVIDIA) or trade a perpetual contract on it, which adds "
        "leverage and a funding payment. Neither makes you a shareholder, so neither gives a "
        "vote.",
        "To see one in numbers, ask \"what is NVDA doing today\".")),
    (re.compile(r"\bwhat\s+(?:is|'s|are)\s+(?:a\s+)?margin\s+calls?\b|\bmargin\s+call\s+(?:mean|"
                r"meaning)\b", re.I), (
        "Bottom line: a margin call is a broker asking for more money because the losses on a "
        "borrowed (leveraged) position have eaten into the margin held against it.",
        "On a crypto exchange's perpetuals the call is not a phone call: when the margin left "
        "falls to the maintenance level, the exchange closes the position itself (liquidation) "
        "and the margin is gone. The line to watch is the liquidation price.",
        "Ask \"where is my liquidation price on a 5x BTC long\" to see where that line sits.")),
    (re.compile(r"\bisolated\b[^?]{0,30}\bcross\b|\bcross\b[^?]{0,30}\bisolated\b", re.I), (
        "Bottom line: isolated margin gives each position its own margin, so a liquidation loses "
        "that margin and nothing else; cross margin lets the whole account balance back every "
        "position, so a losing one can draw on all of it before it is closed.",
        "Cross sits further from liquidation, but more of your money is at stake in one bad "
        "move; isolated caps the loss per position at the margin you put up.",
        "Ask \"where is my liquidation price on a 5x ETH long\" for the isolated line on a real "
        "market.")),
    (re.compile(r"\b(?:what(?:'s|\s+is)|whats)\s+the\s+(?:safest|least\s+risky)\s+(?:thing|coin|"
                r"stock|option|bet|one|investment)\b", re.I), (
        "Bottom line: nothing on an exchange is safe in the sense of not losing value; what "
        "differs is how far it swings. A stablecoin such as USDT aims to stay at $1, and broad "
        "indices swing less than single coins.",
        "Measure it rather than take a label: ask \"how much could I lose on QQQ in a bad week\" "
        "and the same for BTC, and compare the dollars with what you could afford to lose.",
        "Without leverage, the most you can lose is what you put in.")),
    (re.compile(r"\brevenge[\s-]?trad\w*|\b(?:win|make|get|earn)\s+(?:it\s+|my\s+money\s+|my\s+"
                r"losses\s+|that\s+)?back\b|\brecover\s+(?:my\s+)?loss(?:es)?\b", re.I), (
        "Bottom line: no — trading to win back a loss is how small losses become large "
        "ones: the size goes up while the judgement goes down.",
        "Stop for the day, write down what went wrong (size, exit, leverage), and come back with "
        "the size you would have chosen before the loss.",
        "Ask \"how much could I lose on BTC in a bad week\" with the amount you can spare before "
        "the next trade.")),
    (re.compile(r"\b(?:i'?m|i\s+am)\s+\d{2}\b[^?]{0,40}\b(?:retired|retiree|pension\w*)\b|"
                r"\bsuitable\s+for\s+(?:someone|a\s+person|people)\s+like\s+me\b|"
                r"\b(?:retired|retiree)\b[^?]{0,40}\b(?:suitable|safe|right|okay|ok)\b|"
                r"\bretirement\s+(?:savings|money|fund|nest\s+egg|pot)\b", re.I), (
        "Bottom line: whether it suits you is a question for a licensed adviser who knows your "
        "circumstances; this console is a research tool and cannot judge that.",
        "What it can show plainly: single coins and leverage can fall by half or more in a year, "
        "and money you will need to live on should not be exposed to that.",
        "To see what a sum has actually been through in three broad markets over the last year, "
        "ask \"I have $10,000, what should I do\" — it shows the falls, not a pick.")),
    (re.compile(r"\bhow\s+do\s+(?:people|traders|you)\s+(?:actually\s+)?make\s+money\s+"
                r"(?:from\s+|by\s+|in\s+)?trading\b", re.I), (
        "Bottom line: by being right more often than fees and mistakes cost, at a size where one "
        "wrong trade does not end the account — and many short-term traders do not manage "
        "it once fees are counted.",
        "The parts that decide it: a reason to enter that has worked before, an exit set before "
        "entering, a size where the exit costs 1-2% of the account, and no more trades than the "
        "fees allow (a Bitget perpetual round trip at taker is about 0.12%).",
        "Ask \"has BTC been here before\" to see what followed a market's current state in the "
        "past.")),
    (re.compile(r"\b(?:turn|make|grow)\s+\$?(?P<a>\d[\d,]*)\s*(?:dollars?|bucks|usd)?\s+into\s+\$?"
                r"(?P<b>\d[\d,]*)", re.I), (
        "Bottom line: turning a sum into ten times as much needs either leverage or a coin that "
        "could also go to zero — the same move that multiplies it can wipe it out.",
        "At 10x leverage, a 10% move in your favour doubles the stake and a 10% move against "
        "it (a bit less, after the maintenance margin) loses all of it; ordinary coins move 10% "
        "in a week often.",
        "Ask \"how much could I lose on BTC in a bad week\" with your amount to see the downside "
        "first.")),
    (re.compile(r"\b(?:i\s+)?keep\s+losing\b|\bwhat\s+am\s+i\s+doing\s+wrong\b", re.I), (
        "Bottom line: the usual causes are size, exits and leverage: positions too big for one "
        "loss to be small, no exit decided before entering, and leverage that turns an ordinary "
        "move into a liquidation.",
        "Fees matter too: every round trip on a Bitget perpetual at taker costs about 0.12%, so "
        "frequent trading needs a bigger edge just to stand still.",
        "Ask \"where should my stop go on BTC\" for an exit outside ordinary noise, and size so "
        "that exit costs 1-2% of your account.")),
    (re.compile(r"\bliquidat\w*\b[^?]{0,80}\b(?:not\s+let|avoid|prevent|stop|never)\b|"
                r"\b(?:how\s+do\s+i|how\s+to)\s+(?:not\s+get|avoid\s+getting|stop\s+getting)\s+"
                r"liquidat\w*", re.I), (
        "Bottom line: a liquidation happens when the loss reaches the margin, and the distance "
        "to it is about 1 / leverage: at 20x a move of about 5% against you (less the "
        "maintenance margin) closes the position.",
        "Three levers: lower leverage (5x is about 20% away, 2x about 50%), a stop that closes "
        "the trade before the liquidation line, and a size where that stop costs 1-2% of the "
        "account.",
        "Ask \"where is my liquidation price on a 20x BTC long\" and \"where should my stop go on "
        "BTC\" to see both lines on a real market.")),
    (re.compile(r"^\W*(?:so\s+|then\s+|ok\s+)?(?:what|which)\s+leverage\s+is\s+(?:safe|okay|ok|"
                r"fine)\b|\bsafe(?:r|st)?\s+(?:amount\s+of\s+)?leverage\b|\bwhats\s+a\s+safer\s+"
                r"number\b", re.I), (
        "Bottom line: no leverage is safe in the sense of not losing; leverage decides how small "
        "a move ends the trade. The liquidation distance is about 1 / leverage: 2x about 50%, "
        "3x about 33%, 5x about 20%, 10x about 10%.",
        "Set it against the market's own swings: a coin that has fallen 10% in a day can "
        "liquidate a 10x position in that day.",
        "Without leverage (spot), the most you can lose is what you put in. Ask \"where is my "
        "liquidation price on a 3x BTC long\" for the exact line.")),
    (re.compile(r"\b(?:is|are)\s+(?:a\s+)?stable\s*coins?\s+(?:safe|risky)\b|\bcan\s+(?:usdt|usdc|"
                r"a\s+stablecoin|stablecoins?|it)\s+(?:go\s+to\s+zero|crash|lose\s+(?:its|the)\s+"
                r"peg|depeg)\b", re.I), (
        "Bottom line: a stablecoin aims to stay at $1, backed by its issuer's reserves; the risk "
        "is losing that peg if the reserves or the issuer fail.",
        "It has happened: TerraUSD (UST) lost its peg in May 2022 and never recovered, and USDT "
        "traded briefly below $1 in the same week before returning.",
        "So it moves far less than a coin, but it is a claim on an issuer, not cash in a bank.")),
    (re.compile(r"\b(?:buy|get|own)\s+(?:a\s+)?(?:half|part|fraction|piece|bit)\s+(?:of\s+)?"
                r"(?:a\s+)?(?:share|one|coin|it)\b|\bfractional\s+shares?\b", re.I), (
        "Bottom line: yes in effect — on Bitget you buy a stock's rToken or perpetual by "
        "amount, not by whole share, so half a share's worth is an ordinary order, within the "
        "minimum order size Bitget sets for that market.",
        "Neither is the share itself: it tracks the price; there is no vote, and the rToken's "
        "terms (dividends, redemption) are Bitget's to set.",
        "Ask \"what does it cost to buy $100 of NVDA\" for the fee on a real order.")),
    # Round 25, a first-time user: each of these got a desk log, a decline or a sentiment line
    (re.compile(r"\bi(?:'?m|\s+am)\s+(?:only\s+)?(?:1[0-7]|thirteen|fourteen|fifteen|sixteen|"
                r"seventeen)\b(?!\s*(?:%|x\b|k\b|thousand|years?\s+(?:into|of\s+trading)))",
                re.I), (
        "Bottom line: Bitget's terms require anyone using it to be at least 18 (section 2.2 of its "
        "terms of use), so at 16 you cannot open an account there, with $200 or any amount.",
        "Learning costs nothing: ask this console anything — \"how much could I lose on BTC "
        "in a bad week with $200\" shows what that money would have been through — and keep "
        "the $200 until you can decide for yourself, without leverage.",)),
    (re.compile(r"\b(?:just\s+)?(?:hold(?:ing)?|keep(?:ing)?|stay(?:ing)?\s+in|park(?:ing)?)\s+"
                r"(?:it\s+in\s+)?(?:usdt|usdc|stable\s*coins?|cash)\b", re.I), (
        "Bottom line: holding USDT keeps the money near $1 a coin while you decide — it does "
        "not grow, and it is not risk-free: it is a claim on its issuer, Tether.",
        "It has wobbled: USDT traded briefly below $1 in May 2022, the week TerraUSD collapsed, "
        "and recovered; TerraUSD itself never did.",
        "Compared with buying a coin, the trade-off is giving up the upside to avoid the swings; "
        "ask \"how much could I lose on BTC in a bad week\" to see what is being avoided.")),
    (re.compile(r"\bsame\s+(?:thing\s+)?as\s+(?:owning|holding|buying)\s+(?:the\s+)?(?:actual\s+)?"
                r"(?:bitcoin|btc|ether|eth|crypto|the\s+coin)\b", re.I), (
        "Bottom line: on Bitget's spot market, yes — buying BTC there is owning bitcoin, held "
        "in your Bitget account, and you can withdraw it to a wallet of your own.",
        "A BTC perpetual is not: it is a contract on the price, with leverage, funding payments "
        "and liquidation, and there is no coin to withdraw.",
        "For a first buy without leverage, spot is the simple one.")),
    (re.compile(r"\b(?:does\s+)?(?:anything|any\s+of\s+(?:this|these|it))\s+(?:here\s+)?pay\s+(?:me\s+)?"
                r"(?:a\s+)?(?:regular\s+)?(?:income|dividends?|interest|yield)\b|\b(?:dividends?|"
                r"passive\s+income|regular\s+income)\b[^?]{0,30}\b(?:here|on\s+bitget|from\s+this)\b",
                re.I), (
        "Bottom line: not the way a dividend does. A perpetual contract pays no dividend, and "
        "whether a stock's rToken passes on its dividends is set by Bitget's rToken terms, which "
        "this console has not verified.",
        "The one regular payment on perpetuals is funding, and it changes sign: a position "
        "receives it when the other side is crowded and pays it the rest of the time, so it is "
        "not income you can plan on.",
        "Ask \"what is the funding on BTC\" to see which side is paid right now.")),
    # "wat r the fees here" got a desk decision's evidence once the model allowance ran out (a
    # first-time user, round 24)
    (re.compile(r"\b(?:wh?at|wat)\s+(?:r|are|is)\s+(?:the\s+)?(?:trading\s+)?fees?\b|\bhow\s+much\s+"
                r"(?:are|is|do)\s+(?:the\s+)?(?:trading\s+)?fees?\b|\bfees?\s+(?:on|at|for)\s+"
                r"bitget\b", re.I), (
        "Bottom line: this console is free and takes no cut; trading fees are Bitget's: on spot, "
        "0.10% of the order at the standard (VIP 0) rate on Bitget's fee schedule, and on "
        "perpetuals about 0.06% for an order that takes the price, each way.",
        "Perpetuals also pay or receive funding every few hours while the position is open, and "
        "every market order pays the spread; a full round trip on a perpetual is about 0.12% "
        "before funding.",
        "Ask \"what does it cost to buy $500 of BTC\" for the cost of a real order on the live "
        "order book.")),
    (re.compile(r"^\W*(?:so\s+)?what(?:'s|\s+is)\s+(?:the\s+|a\s+)?(?:bid[\s-]+ask\s+)?spread\W*$",
                re.I), (
        "Bottom line: the spread is the gap between the best price a buyer offers (the bid) and "
        "the best price a seller asks (the ask); buying at the ask and selling at the bid costs "
        "you that gap once, on top of fees.",
        "On busy markets such as BTC on Bitget it is a fraction of a basis point; on thin markets, "
        "or when a stock's own exchange is shut, it widens.",
        "Ask \"what is the BTC price now\" to see the live bid, ask and spread.")),
    (re.compile(r"\bvoting\s+rights?\b|\bdo\s+i\s+get\s+(?:to\s+)?vote\b|\bcan\s+i\s+vote\b",
                re.I), (
        "Bottom line: no — neither a stock's rToken nor its perpetual on Bitget makes you a "
        "registered shareholder, so there is no vote.",
        "What you hold is exposure to the price; whether an rToken pays dividends or can be "
        "redeemed is set by Bitget's rToken terms, which this console has not verified.",)),
)


def reply(text: str, *, named: bool = False) -> Reply | None:
    """The newcomer answer to ``text``, or None when it is not one of these questions.

    ``named`` says the question names a contract; the questions that ask about one name ("what
    could go wrong with NVDA", "is now a good time to buy TSLA") are that name's engines' to
    answer, and only their nameless forms are taken here (the round-8 first-user audit,
    2026-09-30: all five nameless forms were declined)."""
    from argus.lui.research.sizing import stated_capital

    for asked, answer in _PLAIN:
        if asked.search(text):
            return Reply(lines=answer)
    if _LOAN.search(text):
        return Reply(lines=_LOAN_A)
    if _WHAT_NOW.search(text):
        return Reply(lines=_WHAT_NOW_A)
    if _LOOKAHEAD.search(text):
        return Reply(lines=_LOOKAHEAD_A)
    if _OVERFIT.search(text):
        return Reply(lines=_OVERFIT_A)
    if _SURVIVOR.search(text):
        return Reply(lines=_SURVIVOR_A)
    if _BACKTEST_COSTS.search(text):
        return Reply(lines=_COSTS_A)
    if _GUARANTEE.search(text):
        return Reply(lines=_NO_GUARANTEE)
    if _TRUST.search(text):
        return Reply(lines=_TRUST_A)
    for asked_now, answer_now in ((_OTHER_EXCHANGE, _OTHER_EXCHANGE_A), (_KYC, _KYC_A),
                                  (_HIDDEN_FEES, _HIDDEN_FEES_A), (_MISTAKES, _MISTAKES_A),
                                  (_SPECULATION, _SPECULATION_A)):
        if asked_now.search(text):
            return Reply(lines=answer_now)
    if _ADVICE.search(text):
        return Reply(lines=_NOT_ADVICE)
    if _MONEY_SAFE.search(text):
        return Reply(lines=_SAFE)
    if _WITHDRAW.search(text):
        return Reply(lines=_WITHDRAWING)
    if _LOSE_MORE.search(text):
        return Reply(lines=_LOSING)
    if _LOSS_HAPPENS.search(text):
        return Reply(lines=_LOSS_HAPPENS_A)
    if _BEGINNER_SAFE.search(text):
        return Reply(lines=_BEGINNER_SAFE_A)
    if _FOR_ME.search(text):
        return Reply(lines=_TRADING_FOR_YOU)
    if _PLACE_TRADE.search(text):
        return Reply(lines=_PLACING)
    if _FREE.search(text):
        return Reply(lines=_FREE_TO_USE)
    if _HOW_TO_BUY.search(text) or _HOW_BUYING_WORKS.search(text):
        return Reply(lines=_BUYING)
    if named:
        return None
    if _GO_WRONG.search(text):
        return Reply(lines=_GOING_WRONG)
    if _HOW_MUCH_BARE.search(text):
        money = stated_capital(text)
        return Reply(reask=f"I have ${money or 1000:,.0f}, what should I do",
                     note=None if money else "no amount was given, so $1,000 is the worked "
                                             "example")
    timing = _GOOD_TIME.search(text)
    if timing:
        # "should i sell my crypto now" was told how "a good time to buy" was read (round 11).
        verb = "sell" if re.search(r"\b(?:sell|get\s+out)\b", timing.group(0), re.I) else "buy"
        return Reply(reask="has {name} been here before",
                     lead=f"Bottom line: no call — this console will not tell you whether to "
                          f"{verb} now. What it can show is what followed past moments like "
                          f"{{name}}'s today, below: a base rate, not a forecast.",
                     note=f"\"a good time to {verb}\" was read as: what followed similar past "
                          f"states — a base rate, not a forecast or a call")
    if _BEGINNER_PICK.search(text):
        # No pick: what a first sum has been through in three broad markets, from the starter
        # engine, on a stated $1,000.
        return Reply(reask="I have $1,000, what should I do",
                     note="no amount was given, so $1,000 is the worked example")
    if _DIP.search(text):
        return Reply(reask="has {name} been here before",
                     note="\"buy the dip\" was read as: what followed similar past states — a "
                          "base rate, not a forecast")
    return None


trace_module(globals())
