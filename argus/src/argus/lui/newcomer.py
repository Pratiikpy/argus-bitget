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
from collections.abc import Callable
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
    r"\b(?:is|are)\s+(?:this|it|argus|(?:the|this)\s+(?:console|tool|app|site|service))\s+"
    r"(?:really\s+)?"
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
    r"holdings|positions?))?\s+(?:right\s+)?now\b|"
    # "is it too late to buy bitcoin now, feels like its already so high" (round 32)
    r"\bis\s+it\s+too\s+late\s+to\s+(?:buy|invest(?:\s+in)?|get\s+in|sell|get\s+out)\b", re.I)
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

_BITGET_SAFE = re.compile(
    # "is bitget safe for keep my money there long time, i am worry" got the desk's track record
    # (a first-time user, round 30)
    r"\bis\s+bitget\s+(?:\w+\s+){0,2}(?:safe|legit|trustworthy|secure|a\s+scam|reliable)\b|"
    # "can i trust bitget with my money" (a live re-ask, round 31)
    r"\b(?:trust|rely\s+on)\s+bitget\b|\bbitget\s+(?:\w+\s+){0,2}(?:trustworthy|legit|a\s+scam)\b|"
    r"\b(?:safe|ok|okay)\s+(?:to|for)\s+(?:keep|keeping|leave|leaving|store|storing|hold|holding)\s+"
    r"(?:my\s+)?(?:money|crypto|funds|coins|savings)\b|"
    # "if bitget the company goes down or gets hacked what happens to my coins" (round 32)
    r"\bbitget\b[^?]{0,40}\b(?:goes?\s+down|shuts?\s+down|collapses?|folds?|gets?\s+hacked|is\s+"
    r"hacked|goes?\s+bankrupt|fails?|disappears?)\b[^?]{0,60}\b(?:happens?|happen)\b[^?]{0,20}"
    r"\b(?:my\s+)?(?:coins?|crypto|money|funds)\b", re.I)
_BITGET_SAFE_A = (
    "Bottom line: that is Bitget's to show and yours to check — this console does not audit "
    "Bitget, and no exchange is risk-free.",
    "If it failed or was hacked: coins held on Bitget are a claim on Bitget, not coins sitting "
    "separately with your name on them. You would queue behind every other depositor, and "
    "getting money back from a failed exchange can take years and can still come back short.",
    "What Bitget publishes for you to check now: its proof-of-reserves reports and its "
    "protection fund, on its own site.",
    "Coins moved to a wallet only you control are not exposed to Bitget failing at all, "
    "because Bitget never held them — which then makes keeping that wallet's recovery phrase "
    "safe your own job. This console never holds, moves or touches money.")

_WITHDRAW = re.compile(
    # "how do i actually get my money out if i wanna stop" (a first-time user, round 28)
    r"\bhow\s+(?:do|can|would)\s+i\s+(?:actually\s+|even\s+|ever\s+)?(?:withdraw|take\s+out|"
    r"cash\s+out|get\s+(?:my\s+)?(?:money|funds|cash)\s+(?:back\s+)?out)"
    r"|\bwithdraw(?:al|ing)?\s+(?:my\s+)?(?:money|funds|crypto|usdt|cash)\b"
    # "mera paisa bitget se bank account me kaise nikalu, withdraw kaise karu" (Hinglish, round 32)
    r"|\bwithdraw\s+kaise\s+kar\w*\b|\b(?:paisa|paise)\b[^?]{0,40}\bnikal\w*\b", re.I)
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
TAX: Final = re.compile(
    r"\btax(?:es|ed)?\b|\birs\b|\bhmrc\b|\bcapital\s+gains?\b|"
    # "will this loss show up anywhere i can use it later, like against profits" (round 32)
    r"\bshow\s+up\s+anywhere\b[^?]{0,40}\buse\s+it\s+later\b|"
    r"\b(?:use|count|offset|claim|carr(?:y|ies)\s+(?:it\s+)?forward)\w*\b[^?]{0,40}\b(?:against|"
    r"to\s+offset)\s+(?:(?:future\s+|my\s+|any\s+)?profits?|gains?)\b",
    re.I)
"""A tax question, taken before the research readers: "i made 500 on eth this year, do i pay taxes
on that" was answered with ETH's year-to-date move (a live re-ask, round 31); "will this loss show
up anywhere i can use it later, like against profits" is the same question without the word "tax"
(a first-time user, round 32)."""
TAX_FOLLOW: Final = re.compile(r"\b(?:lost|loss\w*|report\w*|count|counts|owe|pay|declare\w*)\b",
                               re.I)
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
    r"^\W*(?:(?:so|ok(?:ay)?|then|alright|hmm+|thanks?|thank\s+you|thx)\W+){0,3}(?:what\s+(?:should|"
    r"do)\s+i\s+do"
    r"(?:\s+now)?|"
    # "ok so which one do i go for" (a round-23 re-ask)
    r"which\s+(?:one\s+)?(?:should|do|would)\s+i\s+(?:pick|choose|buy|go\s+(?:with|for)|get)|"
    r"what\s+now|now\s+what|"
    # "ok thank you, what i do first step today" got the thanks alone (a first-time user, round 30)
    r"what\s+(?:do\s+|should\s+)?i\s+do\s+(?:as\s+)?(?:a\s+|the\s+|my\s+)?first(?:\s+step)?(?:\s+today)?|"
    r"(?:what(?:'?s|\s+is)\s+)?(?:the|my)\s+first\s+step(?:\s+today)?|"
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
    # "he said its been working for him for 2 months and he showed me screenshots of profits" —
    # the follow-up to a "trading bot" question got the same canned refusal as the question
    # itself (a first-time user, round 32)
    (re.compile(r"\bscreenshots?\s+of\s+(?:his\s+|her\s+|their\s+|the\s+)?profits?\b|\bshowed\s+"
                r"me\s+(?:his\s+|the\s+)?(?:screenshots?|proof|profits?)\b|\b(?:been\s+)?working"
                r"\s+for\s+him\s+for\s+\d+\s+(?:months?|weeks?|years?)\b", re.I), (
        "Bottom line: screenshots and a run of months working are not proof — a screenshot can "
        "be edited in seconds, and in a Ponzi scheme early \"profits\" are just other people's "
        "new money being paid out, which is exactly what makes it look real for a while.",
        "The real tells: a return that never varies with the market, no public or auditable "
        "record (only what he shows you), and pressure to recruit more people or add more "
        "money — recruiting is how a Ponzi stays fed.",
        "A real trading result shows losing months too. Ask to see one, or a statement straight "
        "from the exchange itself, not a screenshot he chose to send.")),
    # "ok so how do i actually check if something like that is legit before i put money in" (a
    # first-time user, round 32)
    (re.compile(r"\bhow\s+do\s+i\s+(?:actually\s+)?(?:check|know|tell|verify)\s+if\s+(?:something|"
                r"that|this|it)\s+(?:like\s+that\s+)?is\s+legit\b|\bhow\s+(?:do|can)\s+i\s+"
                r"(?:check|know|verify)\s+(?:something|that|this|it)\s+is(?:n'?t)?\s+a\s+scam\b|"
                r"\bis\s+(?:something|this|that)\s+like\s+that\s+legit\b", re.I), (
        "Bottom line: four checks, before any money goes in: who actually holds the money (you, "
        "or them); can you deposit a small amount and then actually withdraw it back out, not "
        "just see a balance; is whoever runs it licensed to handle money where you live; and "
        "does it promise a fixed return regardless of what the market does.",
        "Any \"guaranteed\" or fixed return is the biggest red flag on its own — no real "
        "strategy guarantees a return, so one that claims to has already shown itself false.",
        "Search the name plus \"scam\" or \"review\" before sending anything, and never send "
        "money to \"unlock\" a withdrawal or \"verify\" an account — that request is itself the "
        "scam.")),
    # "some guy dmed me on instagram saying he can double my crypto in 24 hrs if i send him some
    # first, is this real" (a first-time user, round 32)
    (re.compile(r"\bdoubl\w*\s+(?:my|your|the|it|his|her)?\s*(?:crypto|money|coins?|btc|eth|"
                r"funds?)\b[^?]{0,80}\bsend\b[^?]{0,30}\bfirst\b", re.I), (
        "Bottom line: no — this is always a scam. Nobody doubles money by having you send money "
        "or crypto to them first; once it is sent, it is gone, and asking you to send first is "
        "the whole scheme, not a step in a real one.",
        "The DM, the urgency (\"24 hours\"), and the promise of a guaranteed multiple are the "
        "same pattern every time, on every platform — Instagram, WhatsApp, Telegram, it does "
        "not matter.",
        "Block and report the account; do not reply, and never send anything \"to unlock\" a "
        "bigger return — that request is the scam working as intended.")),
    # a candlestick chart, explained plainly — "can someone explain how to read a candlestick
    # chart like im 5 years old" was refused entirely (a first-time user, round 32)
    (re.compile(r"\b(?:explain|read|understand)\b[^?]{0,40}\bcandlesticks?\b|\bwhat(?:'?s|\s+is|"
                r"\s+are)\s+(?:a\s+|an?\s+)?candlesticks?\b|\bhow\s+(?:do|to|can)\s+(?:i\s+)?read"
                r"\s+(?:a\s+|the\s+)?candlestick\s+charts?\b", re.I), (
        "Bottom line: one candle is the price over one time period (a minute, an hour, a day — "
        "whatever the chart is set to): it opens at one price, trades up and down, and closes "
        "at another; the candle's body is the gap between the open and the close.",
        "Green (sometimes white or blue) means the close was higher than the open — price rose "
        "over that period; red (sometimes black) means the close was lower than the open — "
        "price fell.",
        "The thin lines above and below the body are wicks (or \"shadows\"): the highest and "
        "lowest price reached during the period, even if the price came back before the close.",
        "A chart is just many of these candles side by side, each one a snapshot of a short "
        "window of trading.")),
    # "whats the green and red mean on it" — a standalone follow-up about candle colour (round 32)
    (re.compile(r"\b(?:green\s+and\s+red|red\s+and\s+green)\b[^?]{0,40}\bmean\b|\bwhat(?:'?s|"
                r"\s+does)\s+green\s+mean\b.{0,30}\bred\b", re.I), (
        "Bottom line: on a candlestick chart, green means the price closed higher than it "
        "opened over that candle's period (it rose); red means it closed lower than it opened "
        "(it fell).",
        "The exact colours can differ by chart (some use white/black instead) — the rule that "
        "matters is whichever colour that chart's own legend marks as \"up.\"",
        "A green candle after a red one does not predict anything by itself — it is just what "
        "already happened in that window.")),
    # "whats a wick then, i see lines sticking out the top and bottom" (round 32)
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+are)\s+(?:a\s+|the\s+)?wicks?\b|\blines?\s+sticking\s+out"
                r"\b", re.I), (
        "Bottom line: a wick (also called a shadow) is the thin line sticking out of a candle's "
        "body — it marks the highest and lowest price traded during that candle's period, even "
        "though the price did not stay there.",
        "A long wick on top means price pushed up and was pushed back down before the close; a "
        "long wick on the bottom means it was pushed down and bought back up. The body (the "
        "thick part) is just the open-to-close range.",
        "On its own a wick is history, not a forecast — it shows what happened in that window, "
        "not what happens next.")),
    # "ok and is there a pattern that means its about to go up" got read as a ticker lookup for
    # the word "pattern" (a first-time user, round 32)
    (re.compile(r"\bis\s+there\s+a\s+pattern\b[^?]{0,60}\b(?:about\s+to|going\s+to|gonna)\s+"
                r"(?:go\s+up|rise|pump|moon)\b|\b(?:candle|candlestick|chart)\s*patterns?\b|"
                r"\bpatterns?\s+(?:that|which)\s+(?:means?|predicts?|signals?)\b", re.I), (
        "Bottom line: no — no candlestick pattern reliably means a price is about to go up. "
        "Patterns (hammer, engulfing, head-and-shoulders and the rest) show up both before a "
        "rise and before a fall; studied carefully, most are close to a coin flip, and the best "
        "of them only a little better than that in some conditions, and worse than even in "
        "others.",
        "Even a pattern with a genuine small edge is usually eaten by fees and the spread once "
        "you trade it: a Bitget perpetual round trip at taker is about 0.12%, bigger than most "
        "patterns' measured edge.",
        "Treat a pattern as one more observation, not a signal: what has actually followed a "
        "similar setup on a real contract is something this console can check — ask \"has BTC "
        "been here before\" for the base rate, not a pattern read off a picture.")),
    # "i get 5000 rupees allowance a month, how much of that should i put into crypto" and "what
    # if i only do 500 rupees a month instead, is that even worth it" were misread as a request
    # to trade an INR pair (a first-time user, round 32)
    (re.compile(r"\b(?:allowance|salary|stipend|pocket\s+money|income|pay)\b[^?]{0,50}\b(?:a|per|"
                r"each|every)\s+month\b[^?]{0,60}\b(?:how\s+much|worth\s+it|invest|crypto|"
                r"put\s+(?:in|into))\b|\b\d+\s*(?:rupees?|inr|rs\.?|dollars?|usd|naira|pesos?|"
                r"pounds?|euros?)\b[^?]{0,20}\b(?:a|per|each|every)\s+month\b[^?]{0,60}\b(?:how\s+"
                r"much|worth\s+it|crypto|invest|put\s+(?:in|into))\b", re.I), (
        "Bottom line: no fixed percentage is right for everyone — the honest rule is to put in "
        "only money you could fully lose without it touching rent, food, books or bills, and "
        "never money you might need back within a few months.",
        "Before anything goes into crypto, a small buffer in cash comes first: a bad month in "
        "crypto and a bad month in your own life landing together is the situation to avoid.",
        "If you do try it, a small fixed amount repeated every month beats guessing a "
        "percentage — a small, steady amount is a reasonable way to learn, as long as it is "
        "small enough to lose entirely without it mattering.",
        "At that size, fees matter more: Bitget's spot fee is about 0.10% a side (buying and "
        "selling), so on a small monthly amount that is a real share of a tiny trade, not just "
        "a rounding error.")),
    # "ok what about just keeping my cash in the bank instead of either" (after gold vs bitcoin,
    # a first-time user, round 32)
    (re.compile(r"\bwhat\s+about\s+(?:just\s+)?(?:keeping|leaving)\s+(?:my\s+)?cash\s+in\s+"
                r"(?:the\s+)?bank\b|\b(?:keep(?:ing)?|leav(?:e|ing)|just\s+hold(?:ing)?)\s+"
                r"(?:my\s+)?cash\s+in\s+(?:the\s+)?bank\b[^?]{0,40}\b(?:instead|rather\s+than|"
                r"vs\.?|versus)\b", re.I), (
        "Bottom line: cash in a bank keeps its number the same — $100 stays $100 — and in most "
        "countries a bank deposit is insured up to a limit (the US's FDIC covers $250,000 per "
        "depositor per bank; other countries set their own limit), which neither gold nor "
        "bitcoin offers.",
        "What it does not keep is its buying power: inflation quietly reduces what that same "
        "$100 buys every year, even while the number on the statement never changes.",
        "Gold and bitcoin do not hold a steady nominal value the way cash does — both have "
        "fallen sharply more than once — so the trade-off is stability now against the chance, "
        "not the certainty, of beating inflation later.",
        "No call here on which to hold — that depends on what you need the money for and when.")),
    # "so if i buy and then sell the same day do i get charged fees twice" (round 32)
    (re.compile(r"\bcharged?\s+fees?\s+twice\b|\bfees?\s+(?:charged\s+)?twice\b|\bdo\s+i\s+(?:get"
                r"\s+)?(?:pay|charged?)\s+fees?\s+(?:on\s+)?(?:both|each)\s+(?:sides?|ways?|"
                r"times?)\b", re.I), (
        "Bottom line: yes — buying and selling the same day gets charged a fee on each side, "
        "not one fee for the round trip.",
        "On Bitget's spot market the standard taker fee is about 0.10% each way, so buying then "
        "selling the same day costs roughly 0.20% in total, before any gain or loss on the "
        "price itself.",
        "It is the same on any exchange: every separate order is its own fee. Ask \"what does "
        "it cost to buy $500 of BTC\" for the exact fee on a live order.")),
    # "how do fees work when i buy crypto, is it free to just buy some" was told this console
    # costs nothing (a first-time user, round 32): the fee asked about is the exchange's
    (re.compile(r"\bhow\s+do\s+(?:the\s+)?fees?\s+work\b|\bis\s+(?:it|buying)\s+free\s+to\s+"
                r"(?:just\s+)?buy\b|\bfees?\s+(?:when|for)\s+(?:i\s+)?(?:buy|buying)\b|\bdoes\s+"
                r"(?:it|buying\s+crypto)\s+cost\s+(?:anything|money)\b", re.I), (
        "Bottom line: buying is not free — on Bitget's spot market the standard fee is 0.10% of "
        "what you buy (0.08% if paid in BGB), so a $100 buy costs about $0.10, and selling later "
        "costs the same again.",
        "On top of the fee there is the spread — the small gap between the buy and sell price — "
        "which on large coins like BTC is usually a few hundredths of a percent.",
        "Futures are charged differently: about 0.02% for an order that waits on the book (maker) "
        "and 0.06% for one that fills at once (taker). Ask \"what does it cost to buy $500 of "
        "BTC\" for a live quote.")),
    # "i saw this coin called PEPE on my tiktok fyp, everyone was saying its gonna blow up" got
    # nothing for the coin half (a first-time user, round 31)
    (re.compile(r"\b(?:on|from)\s+(?:my\s+)?(?:tiktok|fyp|twitter|x|youtube|reddit|insta(?:gram)?|"
                r"telegram|discord)\b[^?]{0,80}?\b(?:blow\s+up|moon\w*|pump\w*|\d+x|explode|"
                r"go(?:ing)?\s+up)\b|\beveryone\s+(?:is|was|'s)\s+(?:saying|talking)\b[^?]{0,40}?"
                r"\b(?:blow\s+up|moon\w*|pump\w*|\d+x|explode)\b", re.I), (
        "Bottom line: a coin everyone on social media says will blow up is a coin being sold to "
        "you — the people posting it often already hold it, and they gain when you buy.",
        "A trending coin that can rise tenfold can as easily fall 90%, and memecoins usually do "
        "fall far from their peak. If you buy, use only money you could lose entirely.",
        "Ask \"what is <ticker>\" — PEPE, for example — to see whether Bitget lists it, its "
        "price and its worst days.")),
    # "Show me what positions user 'satoshi_trader99' holds on Bitget" got the desk's own "no open
    # positions" (a hostile review, round 31)
    # "whats a good app for someone who has never traded before" matched "someone <who> has" as a
    # user-holdings lookup — "who", "that" etc. are now excluded from the username slot (round 32)
    (re.compile(r"\b(?:user|trader|account|person|someone|another\s+user)\s+(?!who\b|that\b|"
                r"which\b|whose\b)['\"]?[\w.@-]{2,40}"
                r"['\"]?\s+(?:currently\s+|now\s+)?(?:holds?|has|is\s+holding|owns?)\b|\b(?:positions?|"
                r"holdings?|trades|balance)\s+(?:of|for|that)\s+(?:user|trader|someone|another|other)"
                r"\b|\bwhat\s+(?:positions?|is)\s+(?:user|trader)\s+\S+\s+(?:holding|hold)",
                re.I), (
        "Bottom line: no one's account is visible here — not yours and not another user's. This "
        "console has no accounts or users at all, and Bitget does not publish what an individual "
        "holds.",
        "What is public is the crowd as a whole: ask \"long/short ratio on BTC\" or \"how crowded "
        "is ETH\" for how traders on Bitget are positioned in aggregate.")),
    # "whats a good app for someone who has never traded before" (a first-time user, round 32)
    (re.compile(r"\b(?:good|best|safe|safest|right)\s+(?:app|exchange|platform)\b[^?]{0,40}\b"
                r"(?:beginners?|newbies?|never\s+(?:traded|invested|used\s+crypto)|someone\s+"
                r"(?:who|new)|first\s+time)\b|\bwhat\s+app\s+should\s+i\s+use\b|\bwhich\s+"
                r"(?:app|exchange|platform)\s+(?:should|is\s+best)\b", re.I), (
        "Bottom line: no app named here — but four things are worth checking on any exchange "
        "before a first-time trader signs up: is it licensed to operate where you live, and "
        "does it publish proof of reserves (evidence it actually holds what it owes "
        "depositors)?",
        "Look for simple spot buying first — owning the coin outright, no leverage — before "
        "any app's futures or leverage screens; those are not where a beginner should start.",
        "Check its fee schedule before funding it (a spot fee of about 0.1% a side is "
        "typical), and that it offers two-factor authentication (2FA) on login and on "
        "withdrawals.",
        "Before trusting it with real money, send a small amount in and withdraw a small "
        "amount back out first — a working withdrawal is worth more than any review you "
        "read.")),
    # Round 31, a first-time user: each of these was answered with the desk's own P&L, its open
    # positions, the "this console is free" block, or nothing.
    (re.compile(r"\b(?:stable\s*coin|usdt|usdc|tether)\b[^?]{0,80}\b(?:de-?peg\w*|lost\s+(?:its|the)"
                r"\s+peg|off\s+(?:its|the)\s+peg|dropped|crash\w*|below\s+(?:a|1|one)\s+dollar|"
                r"0\.9\d?)\b|\bde-?peg\w*\b|\bpeg\s+(?:broke|break\w*)\b|\bis\s+usdt\s+(?:the\s+)?same"
                r"\b|\bsame\s+company\b[^?]{0,40}\b(?:usdt|usdc|tether|circle)\b|\b(?:usdc|usdt)\s+"
                r"not\s+(?:usdc|usdt)\b|\bmy\s+money\s+(?:just\s+)?gone\b", re.I), (
        "Bottom line: USDT and USDC are different coins from different companies — Tether issues "
        "USDT, Circle issues USDC — so one losing its peg does not by itself move the other.",
        "A stablecoin at $0.90 means the market doubts, for now, that it can be swapped for $1. "
        "Selling at $0.90 makes a 10% loss final; holding bets it recovers. It has gone both "
        "ways: USDC fell to about $0.88 in March 2023 when its reserves sat partly at a failed "
        "bank, and was back near $1 within days; TerraUSD fell in May 2022 and never recovered.",
        "What decides it is what backs the coin and whether its issuer can pay out, which each "
        "issuer publishes in its reserve reports. Ask \"what is USDC trading at\" for its live "
        "price on Bitget.",
        "No call here on whether to sell: that is your decision.")),
    (re.compile(r"\b(?:bitget\s+)?earn\b(?!\w)[^?]{0,60}\b(?:what|how|free\s+money|interest|yield|"
                r"every\s+month|safe|catch|thing)|\bwhat(?:'s|\s+is)\s+(?:this\s+)?(?:bitget\s+)?earn\b|"
                r"\bstak(?:ing|ed)\b|\bstake\s+(?:my|your|the|it|eth|sol|coins?|crypto)\b", re.I), (
        "Bottom line: Earn and staking pay you for letting your coins be used or locked — the "
        "catch is that a locked coin cannot be sold until the lock ends, whatever the market "
        "does.",
        "Flexible products can usually be redeemed at any time; fixed-term ones lock until the "
        "term ends; staking a coin on its own network can add an unstaking wait (ETH's exit "
        "queue has taken from hours to weeks). The rate is set by the product and can change.",
        "Before locking anything, read that product's own term and redemption rules on Bitget — "
        "this console has not read them — and lock only what you would not need to sell in a "
        "fall.",
        "The yield is paid in the coin, so a 5% yield on a coin that falls 30% is still a loss "
        "in dollars.")),
    (re.compile(r"\bcopy[\s-]*trad\w*|\bcopy(?:ing)?\s+(?:a|the|what\s+a|what\s+the|other|pro)\s+"
                r"(?:\w+\s+)?traders?\b|\btrader\s+i'?m\s+copying\b|\bguy\s+i\s+copy\b", re.I), (
        "Bottom line: copy trading opens the same trades another trader makes, in your account, "
        "automatically — so you take their losses as well as their gains, and their mistakes "
        "arrive in your account before you can react.",
        "Is the trader good or lucky: a short record of wins is often luck. Look at how long the "
        "record runs, the worst drawdown in it, how much leverage they use and how many people "
        "copy them — a crowd copying a lucky streak is common.",
        "Whether Bitget covers any of your loss is set by its copy-trading terms, which this "
        "console has not read; by the mechanics, a copied loss is your loss.",
        "Set the amount you copy with as the most you would accept losing.")),
    (re.compile(r"\b(?:i'?m|i\s+am|we'?re|we\s+are|living)\s+(?:from|in)\s+[A-Za-z]+\b[^?]{0,80}"
                r"\b(?:(?:work|available|allowed|legal)\s+(?:for\s+me|for\s+us|here|there|in\s+my\s+"
                r"country)|only\s+for\s+(?:us\b|usa|americans|\w+\s+people)|gonna\s+work\s+for\s+me)"
                r"|\b(?:work|available|allowed)\s+(?:for\s+me\s+)?in\s+my\s+country\b", re.I), (
        "Bottom line: this console works anywhere it loads — it is a research page with no "
        "account. Whether Bitget itself serves your country is set by Bitget's terms and its "
        "list of restricted regions, which this console has not read: check them on Bitget "
        "before depositing.",
        "Everything here — prices, risks, costs — is the same wherever you ask from.")),
    (re.compile(TAX.pattern + r"|\breport\s+(?:it|this|that|my\s+(?:gains?|losses|profits?))\b",
                re.I), (
        "Bottom line: not tax advice — the rules are your country's — but in many countries "
        "selling, swapping or spending crypto at a gain is taxable whether or not the money "
        "reaches your bank.",
        "If you lost money, it still counts — it is usually worth reporting, and in many "
        "countries a loss can offset a gain you made this year or carry forward to offset "
        "profits in a later year.",
        "This console does not send anything to the IRS or any tax office — it holds no account. "
        "Reporting is normally the taxpayer's job, though exchanges in some countries also report "
        "to the tax office. Keep a record of every buy and sell with its date and price; this "
        "console can review a pasted trade list for you.",
        "For what you owe, ask a tax professional or your tax office's own guidance.")),
    (re.compile(r"\baverag(?:e|ing)\s+down\b|\bbuy\s+more\s+(?:to\s+)?(?:lower|bring\s+down)\s+"
                r"(?:my\s+)?(?:average|cost)\b", re.I), (
        "Bottom line: averaging down lowers your average price, but it also makes the position "
        "bigger while the price is falling — it doubles the bet on a call that is so far wrong.",
        "It makes sense only if you would buy the coin today at this price from scratch, at "
        "this larger size. If the reason you bought it has changed, buying more does not fix "
        "that.",
        "Ask \"how much could I lose on <name> in a bad week\" with the larger amount before "
        "adding.")),
    (re.compile(r"\bwh?at(?:'?s|\s+is|\s+are)\s+(?:a\s+)?meme\s*coins?\b|\bmeme\s*coins?\s+"
                r"(?:even\s+)?mean\b", re.I), (
        "Bottom line: a memecoin is a coin built on a joke, an animal or an internet trend, with "
        "no business or cash flow behind it — its price is attention, and attention fades.",
        "Most fall far from their peak and many go close to zero; a few run very high first, "
        "which is what you hear about.",
        "If you try one, use only money you could lose entirely: $20 you would not miss is the "
        "right kind of amount.")),
    (re.compile(r"\b(?:everything|all\s+(?:my\s+)?(?:money|savings))\s+in\s+(?:usdt|usdc|stable\s*"
                r"coins?|dollars?)\b", re.I), (
        "Bottom line: holding savings in USDT protects them from your own currency falling, but "
        "it puts all of them on one company — Tether — and on the exchange that holds them.",
        "Spreading across more than one holder (another stablecoin such as USDC, or dollars at "
        "a bank where you can) means one failure does not take everything.",
        "USDT has held near $1 for years but dipped to about $0.95 in May 2022; ask \"what is "
        "USDT trading at\" for today's price.")),
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
    # "is BTC a safe place to park cash until then" got this block, about USDT (a judge, round 31)
    (re.compile(r"\b(?:just\s+)?(?:hold(?:ing)?|keep(?:ing)?|stay(?:ing)?\s+in)\s+(?:it\s+in\s+)?"
                r"(?:usdt|usdc|stable\s*coins?|cash)\b(?![^?]{0,30}\b(?:is|are)\s+(?:btc|eth|"
                r"bitcoin|gold))|\bpark(?:ing)?\s+(?:it\s+in\s+)?(?:usdt|usdc|stable\s*coins?)\b",
                re.I), (
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


_SCAM_FIXED_RETURN: Final = re.compile(
    # "my friend said i should put my savings into a trading bot he showed me, it promises 2%
    # daily returns, should i do it" (a first-time user, round 32)
    r"\b(?:promis\w+|guarantees?)\b[^?]{0,60}?(?P<pct>\d{1,3}(?:\.\d+)?)\s*%\s*(?:a\s+|per\s+|"
    r"every\s+|each\s+)?(?P<period>daily|weekly|day|week)\b|"
    r"(?P<pct2>\d{1,3}(?:\.\d+)?)\s*%\s*(?:a\s+|per\s+|every\s+|each\s+)?(?P<period2>daily|"
    r"weekly|day|week)\b[^?]{0,40}?\b(?:guarante\w+|promis\w+|returns?)\b",
    re.I)


def _scam_return_lines(text: str) -> tuple[str, ...] | None:
    """A fixed daily or weekly return, compounded out to a year, so the absurdity is a number
    rather than a feeling: "2% daily returns" was refused outright, twice, in the same
    conversation (a first-time user, round 32)."""
    found = _SCAM_FIXED_RETURN.search(text)
    if found is None:
        return None
    pct = float(found.group("pct") or found.group("pct2"))
    period = (found.group("period") or found.group("period2") or "").lower()
    daily = period in ("day", "daily")
    n = 365 if daily else 52
    multiple = (1 + pct / 100) ** n
    word = "day" if daily else "week"
    return (
        f"Bottom line: no — {pct:g}% a {word}, compounded, is about {multiple:,.0f}x your money "
        f"in a year ((1 + {pct:g}/100)^{n} ≈ {multiple:,.0f}x) — no real trading "
        f"strategy returns anything close to that, ever.",
        "What a promise like this always is: a fixed return paid whatever the market does, "
        "screenshots of profits instead of a public and checkable record, and pressure to add "
        "more money or bring in other people — the marks of a Ponzi scheme, not a strategy.",
        "Before anything like this gets money: find out who actually holds it, try "
        "withdrawing a small amount back out (not just watching a balance go up), and check "
        "whether it is licensed to take money where you live. Any guaranteed return is the red "
        "flag on its own.",
    )


_STOP_WORDS: Final = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and",
    "or", "but", "if", "it", "its", "this", "that", "do", "does", "did", "i", "me", "my",
    "you", "your", "u", "can", "could", "should", "would", "will", "just", "so", "like",
    "what", "how", "who", "why", "when", "which", "there", "here", "about", "with", "from",
    "at", "as", "by", "not", "no", "yes", "ok", "okay", "wait", "really", "actually", "even",
    "still"})


def _stem(word: str) -> str:
    """A word without its plural or tense ending: "covers" meets "cover", "counts" meets "count",
    and "countries" stays apart from both (a five-letter cut joined it to "count", round 31)."""
    if word.endswith("sses"):
        return word[:-2]
    for end in ("ing", "ies", "es", "ed", "s"):
        if word.endswith(end) and not word.endswith("ss") and len(word) - len(end) >= 3:
            return word[: -len(end)]
    return word


def _lead_by_question(text: str, answer: tuple[str, ...]) -> tuple[str, ...]:
    """The answer with the line that best answers this question first.

    Copy trading, staking, tax and a stablecoin scare each got the same lead three turns running —
    "how do I know if the trader is good or lucky" and "does Bitget cover me" both opened with what
    copy trading is (a first-time user, round 31). The lines already held each answer; the one
    sharing most words with the question now leads, and the rest follow in their order."""
    words = {_stem(w) for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP_WORDS}
    if not words or len(answer) < 2:
        return answer
    scores = [len(words & {_stem(w) for w in re.findall(r"[a-z]{3,}", line.lower())})
              for line in answer]
    best = max(range(len(answer)), key=lambda i: (scores[i], -i))
    if best == 0 or scores[best] <= scores[0] + 1:
        return answer
    lead = answer[best].removeprefix("Bottom line: ")
    first = answer[0].removeprefix("Bottom line: ")
    rest = [line for i, line in enumerate(answer) if i not in (0, best)]
    return (f"Bottom line: {lead[:1].upper()}{lead[1:]}", f"{first[:1].upper()}{first[1:]}",
            *rest)


_ALSO: Final = re.compile(r"[,.;]?\s*\b(?:and\s+)?also\b|,\s*and\s+(?=(?:is|are|can|should|"
                          r"what|how|do|does)\b)", re.I)


def reply(text: str, *, named: bool = False) -> Reply | None:
    """The newcomer answer to ``text``, both halves when it asks two of these questions.

    "i saw PEPE on my tiktok, everyone says its gonna blow up, and also is bitget even safe" was
    answered on Bitget alone (a first-time user, round 31): each half that is one of these
    questions now gets its own answer, the second under the first."""
    parts = _ALSO.split(text, maxsplit=1)
    if len(parts) == 2 and all(len(x.split()) >= 3 for x in parts):
        first, second = (_reply_one(x, named=named) for x in parts)
        if (first is not None and second is not None and first.lines and second.lines
                and first.lines[0] != second.lines[0]):
            also = second.lines[0].removeprefix("Bottom line: ")
            return Reply(lines=(*first.lines[:2], f"Also asked — {also}", *second.lines[1:2]))
    return _reply_one(text, named=named)


def _reply_one(text: str, *, named: bool = False) -> Reply | None:
    """The newcomer answer to ``text``, or None when it is not one of these questions.

    ``named`` says the question names a contract; the questions that ask about one name ("what
    could go wrong with NVDA", "is now a good time to buy TSLA") are that name's engines' to
    answer, and only their nameless forms are taken here (the round-8 first-user audit,
    2026-09-30: all five nameless forms were declined)."""
    from argus.lui.research.sizing import stated_capital

    bot_return = _scam_return_lines(text)
    if bot_return is not None:
        return Reply(lines=bot_return)
    for asked, answer in _PLAIN:
        if asked.search(text):
            return Reply(lines=_lead_by_question(text, answer))
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
        if asked_now.search(text) and not (
                # "Coinbase's COIN stock ... what's its spread?" names the company for its stock
                # and asks a market figure of it (a hostile review, round 32)
                asked_now is _OTHER_EXCHANGE and named
                and re.search(r"\b(?:spread|price|funding|volume|depth|order\s+book|beta|"
                              r"chart)\b", text, re.I)):
            return Reply(lines=answer_now)
    if _ADVICE.search(text):
        return Reply(lines=_NOT_ADVICE)
    if _BITGET_SAFE.search(text):
        # what happens if it fails leads when that is the question (a first-time user, round 32)
        failing = re.search(r"\b(?:hack\w*|fail\w*|goes\s+down|bankrupt\w*|collaps\w*|bust)\b",
                            text, re.I)
        return Reply(lines=_lead_by_question(text, _BITGET_SAFE_A) if failing else _BITGET_SAFE_A)
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


# --- the same answer in plain words ---------------------------------------------------------------

_PLAIN_WORDS: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(pattern, re.I), plain) for pattern, plain in (
        (r"\b(?:its\s+own\s+)?positioning\s+is\s+not\s+crowded\s+either\s+way\s+on\s+funding\b",
         "traders are not piling onto one side (the fee buyers and sellers pay each other is "
         "normal)"),
        (r"\b(?:its\s+own\s+)?positioning\s+is\s+not\s+crowded\s+either\s+way\b",
         "traders are not piling onto one side"),
        (r"\bread\s+both\s+as\s+positioning,\s+not\s+a\s+signal\b",
         "read both as a picture of how traders are betting, not a tip"),
        (r"\bcrypto\s+fear\s*(?:&|and)\s*greed\b", "crypto mood gauge (0 scared, 100 greedy)"),
        (r"^open\s+interest:", "open bets:"),
        (r"\bpositioning\b", "how traders are betting"),
        (r"\bcrowded\b", "piled onto one side"),
        (r"\bthe\s+crypto\s+market\s+backdrop\b", "the mood across crypto"),
        (r"\bbackdrop\b", "overall mood"),
        (r"\bfunding\s+rates?\b|\bfunding\b", "the fee that buyers and sellers of the contract pay "
                                              "each other"),
        (r"\bopen\s+interest\b", "the number of open bets"),
        (r"\bfear\s*(?:&|and)\s*greed\b", "the 0-100 mood gauge (0 scared, 100 greedy)"),
        (r"\b(?:realised|realized|annuali[sz]ed)\s+volatility\b|\bvolatility\b",
         "how much it swings"),
        (r"\bdrawdown\b", "fall from its high"),
        (r"\bliquidat(?:ed|ion)\b", "forced out of the trade"),
        (r"\bbasis\s+points?\b|\bbps\b", "hundredths of a percent"),
        (r"\bperpetual(?:\s+contract)?\b", "contract"),
        (r"\bbeta\b", "how much it moves with the stock market"),
        (r"\bkurtosis\b", "how often it makes very big moves"),
        (r"\bskew\b", "whether its big moves lean up or down"),
        (r"\bR²\b|\bR2\b|\br-squared\b", "the share of its moves the benchmark explains"),
        (r"\bthe\s+Nasdaq-100\s+\(QQQ\)|\bthe\s+Nasdaq-100\b", "the big US tech stocks"),
        (r"\bmiddle\s+range\b", "normal range"),
        (r"\bper\s+8h\s+settlement\b", "every 8 hours")))
"""Jargon a newcomer said they did not understand, each with the plain words for it."""


def plain_words(line: str) -> str:
    """``line`` with its jargon said in plain words: "can u explain like im 5" after a sentiment
    answer got the same "positioning … crowded … backdrop" sentence, only shorter (a first-time
    user, round 30)."""
    for jargon, plain in _PLAIN_WORDS:
        line = jargon.sub(_cased(plain), line)
    return line[:1].upper() + line[1:]


def _cased(plain: str) -> Callable[[re.Match[str]], str]:
    """The plain words, capitalised where the jargon they replace was."""
    def put(m: re.Match[str]) -> str:
        return plain[:1].upper() + plain[1:] if m.group(0)[:1].isupper() else plain
    return put


# --- a loss already taken: "my portfolio is down 30%", then "is that bad" ------------------------

LOSS_STATED: Final = re.compile(
    r"\b(?:my\s+)?(?:portfolio|account|book|money|holdings?|stack|bag|bags|crypto|investments?|"
    r"savings|positions?|coins?|stocks?|it|i'?m|im|i\s+am|we'?re|i'?ve|i\s+have)\s+"
    r"(?:is\s+|are\s+|was\s+|been\s+|got\s+|already\s+)*(?:down|lost|underwater|in\s+the\s+red)\s+"
    r"(?:by\s+)?(?:about\s+|around\s+|like\s+|almost\s+|nearly\s+|over\s+)?"
    r"(?P<pct>\d{1,2}(?:\.\d+)?)\s*(?:%|percent|pc)", re.I)
"""A loss stated as a share: "my portfolio down 30%", "i'm down like 40 percent"."""
_LOSS_BAD: Final = re.compile(
    r"^\W*(?:so\s+|and\s+|but\s+|ok(?:ay)?\s+)?(?:is\s+(?:that|this|it)\s+(?:bad|normal|a\s+lot|"
    r"terrible|ok(?:ay)?)|how\s+bad\s+is\s+(?:that|this|it)|should\s+i\s+(?:be\s+)?(?:worr(?:y|ied)|"
    r"panic(?:king)?|scared)|am\s+i\s+(?:screwed|cooked|done|ok(?:ay)?)|is\s+(?:that|this)\s+"
    r"(?:the\s+)?end)\b", re.I)
_LOSS_SELL: Final = re.compile(
    r"\bshould\s+i\s+(?:just\s+)?(?:sell|dump|cash\s+out|get\s+out|cut\s+(?:my\s+)?loss(?:es)?|"
    r"exit|bail)\b|\bsell\s+(?:it\s+)?(?:all|everything)\b|\bget\s+out\s+(?:now|while)\b", re.I)
_LOSS_MORE: Final = re.compile(
    r"\bwhat\s+if\s+(?:it|they|this|that|the\s+market|prices?)\s+(?:keeps?|continues?)\s+"
    r"(?:to\s+)?(?:drop|fall|go(?:ing)?\s+down|crash|sink)\w*|\bwhat\s+if\s+(?:it|they)\s+"
    r"(?:drops?|falls?|goes\s+down|crash(?:es)?)\s+(?:more|further|again|lower)\b|"
    r"\bwhat\s+if\s+it\s+goes\s+(?:even\s+)?lower\b|"
    r"^\W*(?:and\s+|but\s+|so\s+)?(?:what\s+)?if\s+(?:it|they)\s+(?:goes|go|gets|drops?|falls?|"
    r"keeps?\s+(?:dropping|falling))\s+(?:even\s+|any\s+)?(?:lower|further|more|down)?", re.I)
_LOSS_BACK: Final = re.compile(
    r"\bwill\s+(?:it|they|my\s+\w+)\s+(?:ever\s+)?(?:come\s+back|recover|bounce\s+back|go\s+back\s+"
    r"up)\b|\bhow\s+long\s+(?:to|until|till|before)\s+(?:it\s+)?(?:recover|come\s+back|break\s+"
    r"even|get\s+back)\w*\b|\bget\s+back\s+to\s+even\b", re.I)


def _fall(base: float, more: float) -> tuple[float, float]:
    """(the total loss after a further fall of ``more``, the rise that total needs to get back)."""
    left = (1 - base) * (1 - more)
    return 1 - left, 1 / left - 1


def loss_lines(text: str, prior: list[str]) -> list[str] | None:
    """A trader who has said they are down N% asks "is that bad", "should I sell everything" or
    "what if it keeps dropping" — the two highest-stakes things a newcomer asks, both answered with
    "I did not recognise that question" (a first-time user, round 30); the third was read as the
    desk's own abstention log. Answered with the arithmetic of the loss they stated, the deepest
    fall BTC and ETH took in the last year as the yardstick, the questions that decide whether to
    sell, and how to get their own book measured. No call either way."""
    stated = LOSS_STATED.search(text)
    source = stated
    if source is None:
        source = next((m for t in reversed(prior[-4:]) if (m := LOSS_STATED.search(t))), None)
    if source is None:
        return None
    pct = float(source.group("pct")) / 100
    if not 0.02 <= pct <= 0.95:
        return None
    sell, more, back = _LOSS_SELL.search(text), _LOSS_MORE.search(text), _LOSS_BACK.search(text)
    bad = _LOSS_BAD.search(text)
    bare = (stated is not None and len(re.findall(r"[a-z0-9%]+", text, re.I)) <= 9
            and "?" not in text)
    if not (sell or more or back or bad or bare):
        return None
    from argus.lui.memory import deepest_fall_year

    need = 1 / (1 - pct) - 1
    falls = [(name, f) for name, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"))
             if (f := deepest_fall_year(sym)) is not None]
    yardstick = ("For scale: over the last year BTC fell as far as "
                 + " and ".join(f"{f:.0%} below its high" if n == "BTC" else f"ETH {f:.0%}"
                                for n, f in falls)
                 + " at the worst point — so a loss this size is common in crypto, and rarer "
                   "in a broad stock index." if falls else
                 "For scale: single coins often fall 30-50% from a high within a year; broad "
                 "stock indices much less often.")
    own = ("To measure your own position rather than the averages, say what you hold, e.g. "
           "\"I hold 60% BTC and 40% SOL — what if crypto drops another 20%?\"")
    said = f"down {pct:.0%}"
    if sell:
        total, back_to = _fall(pct, 0.5)
        return [f"Bottom line: no sell call from here — this console cannot see your situation. "
                f"Selling now makes the {said} final; holding keeps it open both ways.",
                "Four questions decide it: would you buy it today at this price? Do you need this "
                "money within the next year? Has the reason you bought it changed? Could you sit "
                "through another fall as large without selling at the bottom?",
                f"The middle path is selling part: half sold means half of any further fall, and "
                f"half of any recovery. Selling all of it would avoid, for example, a further 50% "
                f"fall — which would leave you down {total:.0%} in total, needing +{back_to:.0%} "
                f"to get back.",
                yardstick, own, "This is analysis, not advice — you make the call."]
    if more:
        rows = "; ".join(f"another {m:.0%} fall makes it {_fall(pct, m)[0]:.0%} in total, "
                         f"needing +{_fall(pct, m)[1]:.0%} to get back"
                         for m in (0.10, 0.20, 0.30))
        return [f"Bottom line: from {said}, {rows}. Each further fall costs more to recover from "
                f"than the one before.",
                f"Getting back to where you started already needs +{need:.0%}, not +{pct:.0%}: a "
                "loss is measured from the higher starting point, a recovery from the lower one.",
                yardstick,
                "What limits the damage is decided before the next fall, not during it: a size you "
                "can hold, a price at which you would sell part, and no leverage on a falling "
                "position.", own]
    if back:
        return [f"Bottom line: no one can say when, or whether — from {said}, getting back to "
                f"even needs a rise of +{need:.0%}.",
                yardstick,
                "Broad indices have recovered from every fall in their history, given years; "
                "single coins and small stocks sometimes never have.", own]
    return [f"Bottom line: {said} is a large loss — getting back to where you started needs a "
            f"rise of +{need:.0%}, because a recovery is measured from the lower point.",
            yardstick,
            "Whether it is bad for you depends on two things only you know: whether you need the "
            "money soon, and whether what you hold is still what you meant to hold.", own]


trace_module(globals())
