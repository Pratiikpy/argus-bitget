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
    r"\bwhat'?s\s+the\s+catch\b(?![^?]{0,20}\bindia\b)|\b(?:do|does)\s+(?:u|you|it|argus)\s+"
    r"take\s+a\s+cut\b|"
    r"\bhow\s+do\s+(?:you|u|they)\s+make\s+money\b|\b(?:do|does)\s+(?:u|you|it)\s+charge\b",
    re.I)
"""Whether using the console costs anything: "is this free?" was declined (round 17)."""
_GO_WRONG = re.compile(
    r"\bwhat\s+(?:could|can|might|would)\s+go\s+wrong\b|\bwhat\s+are\s+the\s+(?:main\s+|biggest\s+)?"
    r"risks\b|\bwhat\s+(?:could|can)\s+i\s+lose\b", re.I)
_GOOD_TIME = re.compile(
    # "wassup is now good time buy" dropped both the article and the infinitive, and the strict
    # match missed it even with the model available (round 36)
    r"\bis\s+(?:now|this|today|it)\s+(?:a\s+)?(?:good|bad|the\s+right|right)\s+time\s+(?:to\s+)?"
    r"(?:buy|invest|get\s+in|start|sell|get\s+out)\b|\bshould\s+i\s+(?:buy|invest|get\s+in|sell|"
    r"get\s+out)(?:\s+(?:my|all\s+my|everything|some)?\s*(?:crypto|coins?|stocks?|shares?|"
    # "ok but fr should i buy rn" — "rn" is texting shorthand for "right now" (round 33)
    r"holdings|positions?))?\s+(?:right\s+now|now|today|rn)\b|"
    # "is it too late to buy bitcoin now, feels like its already so high" (round 32)
    r"\bis\s+it\s+too\s+late\s+to\s+(?:buy|invest(?:\s+in)?|get\s+in|sell|get\s+out)\b", re.I)
_HOW_MUCH_BARE = re.compile(
    # "how much money should i actually start with" put a filler word between "i" and the verb,
    # which the gap-free match missed outright, and fell through to a ticker misread (round 36)
    r"\bhow\s+much\s+(?:money\s+)?(?:should|can|could|do)\s+i\s+(?:actually\s+|really\s+|even\s+|"
    r"just\s+)?(?:put\s+in(?:to)?|invest|start\s+with)"
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
    # "if bitget the company goes down or gets hacked what happens to my coins" (round 32); "what
    # happens to my money if bitget the app goes down or crashes" put the effect-question first
    # and named a verb ("crashes") the list did not have, and both were refused (round 36)
    r"\bbitget\b[^?]{0,40}\b(?:goes?\s+down|shuts?\s+down|collapses?|folds?|gets?\s+hacked|is\s+"
    r"hacked|goes?\s+bankrupt|fails?|disappears?|crash(?:es|ed)?)\b[^?]{0,80}\b(?:happens?|"
    r"happen)\b[^?]{0,20}\b(?:my\s+)?(?:coins?|crypto|money|funds)\b|"
    r"\bwhat\s+happens\s+to\s+(?:my\s+)?(?:coins?|crypto|money|funds)\b[^?]{0,100}\bbitget\b"
    r"[^?]{0,40}\b(?:goes?\s+down|shuts?\s+down|collapses?|folds?|gets?\s+hacked|is\s+hacked|"
    r"goes?\s+bankrupt|fails?|disappears?|crash(?:es|ed)?)\b|"
    # "if bitget shuts down one day where does my money actually go" names the exchange and the
    # failure first, then asks where the money ends up — not "what happens" (round 36)
    r"\bbitget\b[^?]{0,40}\b(?:goes?\s+down|shuts?\s+down|collapses?|folds?|gets?\s+hacked|is\s+"
    r"hacked|goes?\s+bankrupt|fails?|disappears?|crash(?:es|ed)?)\b[^?]{0,80}\bwhere\s+(?:does|"
    r"is|do)\s+(?:my\s+)?(?:coins?|crypto|money|funds)\b",
    re.I)
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
    r"\bmanage\s+my\s+(?:money|account|portfolio|funds)\b|\bauto[\s-]?trade\s+for\s+me\b|"
    # "ok just buy me 100 dollars of bitcoin then" is a blunt imperative with no modal verb at
    # all, and it was silently filed as a remembered position size instead of refused — the
    # single most serious finding of round 36
    r"\b(?:ok(?:ay)?\s+)?(?:just\s+|please\s+|go\s+ahead\s+and\s+)*(?:buy|sell|get|short|long)"
    r"\s+me\s+(?:\$\d|an?\s+\d|\d|some\b|a\s+(?:bit|little)\s+of\b)|"
    r"\b(?:just\s+|please\s+|go\s+ahead\s+and\s+)*(?:buy|sell|trade|get|short|long)\s+(?:it\b|"
    r"that\b|this\b|them\b|some\b|my\s+\w+\b|\d[\d,.]*\b)[^?]{0,30}\bfor\s+me\b", re.I)
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
    r"coins?|shit\s*coins?|these\s+coins|those\s+coins|altcoins?)\s+"
    # "is crypto basically just gambling then" chained two qualifiers where only one was
    # allowed, so the whole match failed and the prior turn got repeated instead (a first-time
    # user, round 34)
    r"(?:(?:just|basically|really|kinda|like|all)\s+){0,3}(?:gambling|a\s+casino|casinos|a\s+"
    r"scam|scams|a\s+ponzi|lottery|a\s+lottery\s+ticket)\b|\bno\s+(?:real\s+)?(?:use|utility|"
    r"purpose)\b|"
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
    # "my frend says diamond hands lol wdym by that" went undefined (a first-time user, round 34)
    "diamond hands": "holding through a big drop without selling it — a compliment when the "
                      "coin recovers, a joke when it doesn't",
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
    if asked:
        # "what does rekt mean and also fomo" matched only REKT via the "what does" trigger;
        # a second term named after "and" or "and also" without repeating that trigger was
        # still being asked about, and went undefined (a first-time user, round 34)
        tail = re.split(r"\band\s+(?:also\s+)?", text, maxsplit=1)[-1]
        asked += [k for k in SLANG if k not in asked
                  and re.search(rf"\b{re.escape(k)}\b", tail, re.I)]
    if not asked and len(re.findall(r"[a-z']+", text.lower())) <= 3 and (
            text.strip().endswith("?")
            or re.match(r"^\W*(?:and|or|what\s+about|how\s+about)\b", text, re.I)):
        # "and hodl?" after "what does fomo mean" repeated FOMO (round 29, live re-ask)
        asked = [k for k in SLANG if re.search(rf"\b{re.escape(k)}\b", text, re.I)]
    if not asked:
        found = [k for k in SLANG if re.search(rf"\b{re.escape(k)}\b", text, re.I)]
        # "wdym by that" (what do you mean) named "diamond hands" with no "what does" prefix
        # at all (a first-time user, round 34)
        asked = found if found and re.search(r"\b(?:mean|stand\s+for|means|meaning|no\s+clue|"
                                             r"wdym|don'?t\s+(?:get|know))\b", text, re.I) else []
    if not asked:
        # "wen moon" has no "?" and no "mean" keyword, but both words are slang on their own
        # (a first-time user, round 33)
        bare_words = re.findall(r"[a-z']+", text.lower())
        if 2 <= len(bare_words) <= 3 and all(w in SLANG for w in bare_words):
            asked = list(dict.fromkeys(bare_words))
    if not asked:
        return None
    said = [f"{k.upper() if len(k) <= 5 else k}: {SLANG[k]}." for k in dict.fromkeys(asked)]
    return ["Bottom line: " + said[0], *said[1:],
            "Slang like this travels fast on social media; none of it is a reason to buy or sell."]
explains(*_LOOKAHEAD_A, *_OVERFIT_A, *_SURVIVOR_A, *_COSTS_A, *_NOT_ADVICE, *_BUYING, *_LOSING,
         *_LOSS_HAPPENS_A, *_BEGINNER_SAFE_A,
         *_GOING_WRONG, *_NO_GUARANTEE, *_SAFE, *_WITHDRAWING, *_PLACING, *_FREE_TO_USE)

_DATED_MONEY: Final = re.compile(r"\bdown\s*payment\b|\bdeposit\b|\bhouse\b|\bhome\b|\bin\s+"
                                 r"(?:like\s+)?\d+\s+years?\b|\bdate\s+on\s+it\b|\bsafer\b", re.I)
"""The turn before a "what's safer" or "keep it in a bank" follow-up: money with a date on it."""
_HOUSE_CASH: Final = (
    "Savings accounts, fixed-term deposits and short-term government bills all pay a little "
    "interest; one that matures before the purchase date cannot be caught by a bad month.",
    "In the US, bank deposits are insured by the FDIC up to $250,000 per depositor, per bank; "
    "other countries have their own schemes and limits. Inflation still nibbles at cash, but over "
    "two years it is a far smaller risk than a market fall.")

_FOLLOW_UPS: Final[tuple[tuple[re.Pattern[str], re.Pattern[str], tuple[str, ...]], ...]] = (
    # "ngmi fr fr is this normal", after "WHY IS EVERYTHING RED TODAY" (round 40 newcomer, #22)
    (re.compile(r"\b(?:is\s+(?:this|that|it)\s+normal|normal\s+(?:for\s+crypto|right)|happens?\s+"
                r"a\s+lot)\b", re.I),
     re.compile(r"\bred\b|\bdown\b|\bcrash\w*|\bdump\w*|\bfall\w*|\bdropp?\w*|"
                r"\bbleed\w*|📉", re.I), (
        "Bottom line: yes — red days are normal in crypto: Bitcoin falls 3% or more in a day many "
        "times in a typical year, and smaller coins swing more than it does.",
        "A red day says little about the next one; what decides how it feels is how much you "
        "hold. If one bad day hurts, the position is bigger than it should be.",
        "Ask \"how much could I lose on BTC in a bad week with $X\" with your own amount to see "
        "what a normal bad stretch means in dollars.")),
    # Round 39 newcomer: "so do i need one to use this site" after "whats a wallet" was declined,
    # though "whats a wallet and do i need one to use this site" in one sentence was answered
    (re.compile(r"\b(?:do|would|will)\s+i\s+(?:even\s+|really\s+|actually\s+)?need\s+(?:one|it|"
                r"a\s+wallet)\b", re.I),
     re.compile(r"\bwallet\b", re.I), (
        "Bottom line: no — you do not need a wallet to use Bitget or this console. A Bitget "
        "account holds your coins for you, the way a bank holds cash.",
        "A wallet of your own becomes useful later, for coins you plan to keep for a long time: "
        "coins there do not depend on any exchange staying healthy, and the recovery phrase "
        "becomes your job to keep safe.")),
    # "whats safer than crypto for this", "if i just keep it in a bank is that better", after a
    # house-deposit question (round 39 newcomer, X2 and X3)
    (re.compile(r"\bsafer\s+than\s+(?:crypto|that|this|bitcoin)\b|\bwhat(?:'?s|\s+is)\s+safer\b",
                re.I),
     _DATED_MONEY, (
        "Bottom line: for money you need on a set date, the usual places are an insured bank "
        "savings account, a fixed-term deposit (a CD in the US) that matures before you buy, or "
        "short-term government bills — none of them can fall the way a coin can.",
        *_HOUSE_CASH[1:])),
    (re.compile(r"\b(?:keep|put|leave)\s+it\s+in\s+(?:a|the)\s+(?:bank|savings)\b|\bbank\b[^?]{0,30}"
                r"\b(?:better|safer|instead)\b", re.I),
     _DATED_MONEY, (
        "Bottom line: yes — for money you need on a date a couple of years away, an insured bank "
        "savings account is the better fit: it cannot fall, and crypto can drop a third or more "
        "in a few weeks.",
        *_HOUSE_CASH)),
    # "wait so is that only in futures", after "can i lose more money than i put in" (round 37
    # newcomer, L2)
    (re.compile(r"\b(?:only|just)\s+(?:in|on|with|for)\s+(?:futures|perps?|perpetuals?|leverage|"
                r"margin)\b|\b(?:futures|perps?|perpetuals?)\s+only\b|\bwhat\s+about\s+spot\b|"
                r"\b(?:does|is)\s+(?:that|it|this)\s+(?:also\s+)?(?:apply|happen|true)\s+(?:to|on|in|"
                r"for)\s+(?:spot|futures)\b", re.I),
     re.compile(r"\blose\s+more\b|\bliquidat\w*|\bleverage\b|\bnegative\s+balance\b|"
                r"\bmargin\b", re.I), (
        "Bottom line: yes, in practice — losing more than you put in can only happen where you "
        "borrow: futures (Bitget's perpetuals) and margin trading. Buying on spot with your own "
        "money, the most you can lose is what you paid.",
        "On futures, isolated margin caps the loss at the margin behind that one position; cross "
        "margin lets it draw on your whole futures balance before it is liquidated.",
        "Ask \"spot vs futures\" for the difference in full, or \"where is my liquidation price "
        "on a 3x BTC long\" for the exact line.")),
    # "does that apply here on bitget", after "whats impermanent loss" (round 37 newcomer, K2)
    (re.compile(r"\b(?:does|do|would|can|could)\s+(?:that|it|this)\s+(?:also\s+)?(?:apply|happen|"
                r"matter|affect\s+me)\b|\b(?:is|does)\s+(?:that|it|this)\s+(?:a\s+thing|exist)\b",
                re.I),
     re.compile(r"\bimpermanent\s+loss\b", re.I), (
        "Bottom line: no — impermanent loss only happens when you deposit two tokens into a "
        "liquidity pool. Buying spot or trading perpetuals on Bitget has none.",
        "What applies on Bitget instead: on spot, the price itself and the fee (0.10% a side); on "
        "perpetuals, also funding every few hours and, with leverage, liquidation.",
        "Ask \"what does $500 of BTC cost me\" for the fee on a real order.")),
    # "is that different from a stock token on here", after "whats an etf" (round 38 newcomer)
    (re.compile(r"\b(?:different|difference|same)\b[^?]{0,40}\b(?:stock\s+token|rtoken|token)\b",
                re.I),
     re.compile(r"\betf\b|\bexchange[\s-]traded\s+fund\b", re.I), (
        "Bottom line: yes — an ETF is a fund you own shares of, run by a fund manager under "
        "securities law; a stock token on Bitget (an rToken) is a token that tracks one share's "
        "price, issued under Bitget's terms.",
        "An ETF holds many companies (or bitcoin itself) and trades only in stock-market hours; "
        "an rToken tracks one company and trades around the clock on Bitget.",
        "Neither makes you a shareholder with a vote; the ETF's protections are the fund's and "
        "the regulator's, the rToken's are Bitget's.")),
    # "how is that different from just buying right now", after "whats a limit order"
    (re.compile(r"\b(?:different|difference|better|worse)\b[^?]{0,40}\b(?:buying|selling|"
                r"buy|sell)\b[^?]{0,20}\b(?:right\s+now|now|straight\s+away|immediately)\b|"
                r"\bwhy\s+not\s+just\s+buy\b", re.I),
     re.compile(r"\blimit\s+orders?\b", re.I), (
        "Bottom line: buying right now is a market order — it fills at once at the best price on "
        "offer and pays the taker fee; a limit order waits for your price, may never fill, and "
        "pays the lower maker fee if it does.",
        "For a small order in a liquid coin the gap is tiny; it matters when the price is moving "
        "fast or the order is large, where a market order can fill worse than the screen showed.",
        "Ask \"what does it cost to buy $500 of BTC right now\" for the market order's real "
        "cost.")),
    # "is that safe to do", after "whats staking"
    (re.compile(r"\b(?:is|are)\s+(?:that|it|this|staking)\s+(?:safe|risky|dangerous|worth\s+it)\b",
                re.I),
     re.compile(r"\bstak\w*|\bearn\b|\byield\b", re.I), (
        "Bottom line: not risk-free — the coin can fall while it is locked (a 5% yield on a coin "
        "that falls 30% is still a loss), you cannot sell during the lock, and the coins sit "
        "with the exchange or protocol doing the staking.",
        "Safer in practice: a flexible product you can leave at any time, on a coin you would "
        "hold anyway, with an amount you could lose.",
        "Ask \"how much could I lose on ETH in a bad month\" to see the price risk the yield sits "
        "on.")),
    # "ok then why does the news keep saying its banned there", after India's legality
    (re.compile(r"\b(?:banned|ban|illegal)\b", re.I),
     re.compile(r"\bindia\b", re.I), (
        "Bottom line: because it was close to banned, twice, and headlines lag — in 2018 the RBI "
        "barred banks from dealing with crypto firms (struck down by the Supreme Court in March "
        "2020), and in 2021 a bill to ban private cryptocurrencies was listed but never passed.",
        "Since then the government taxes it (30% on gains, 1% TDS) rather than banning it, and "
        "regulates exchanges through FIU-IND registration.",
        "Not legal advice — check the current position before you put money in.")),
    # "so can bitget actually operate in india or not"
    (re.compile(r"\b(?:bitget|exchange)\b[^?]{0,40}\b(?:operate|work|available|allowed|legal)\b",
                re.I),
     re.compile(r"\bindia\b", re.I), (
        "Bottom line: that depends on its registration with India's Financial Intelligence Unit "
        "(FIU-IND), which offshore exchanges serving Indian users must hold — this console has "
        "not verified Bitget's current status.",
        "FIU-IND publishes the list of registered exchanges; check it, and Bitget's own help "
        "pages for India, before depositing.",
        "Not legal advice.")),
    # "whats the catch then if its legal"
    (re.compile(r"\bcatch\b", re.I),
     re.compile(r"\bindia\b", re.I), (
        "Bottom line: the tax — gains are taxed at a flat 30% with no deduction but the cost of "
        "buying, a loss on one coin cannot be set against a gain on another, and 1% TDS is "
        "deducted on each transfer.",
        "So a trade has to clear the fees and the tax to leave you anything; frequent trading is "
        "hit hardest.",
        "Not tax advice — check the current rules or a tax adviser.")),
    # "Then why should I listen to it at all?", after the desk's lean was shown to have no skill,
    # was refused (round 37 judge, M-8)
    (re.compile(r"\bwhy\s+(?:should|would|do)\s+(?:i|anyone|we)\s+(?:even\s+)?(?:listen\s+to|trust|"
                r"use|believe|care\s+about|bother\s+with)\b|\bwhat(?:'?s|\s+is)\s+(?:the\s+)?point\b",
                re.I),
     re.compile(r"\blean\b|\bpredictive\b|\bskill\b|\bcalibrat\w*|\bconfidence\b", re.I), (
        "Bottom line: not for the lean — it is published as a test it has not passed, and the "
        "console says so rather than dressing it up. Its own calls are not the product.",
        "What is: the research engines. Every figure is computed when you ask, from Bitget's live "
        "data, SEC filings or FRED, and names its source — a price and what a trade costs, how "
        "far a position can fall, a rule backtested after fees with a permutation test, a book "
        "under a shock, an earnings date and its usual move.",
        "Those you can check yourself, line by line; the lean you can watch fail or improve on "
        "/status, graded the same way every day.")),
)
"""A follow-up that only makes sense after the question before it: the first pattern is the
follow-up, the second must be in the previous turns, and the lines are the answer."""


def followup_lines(text: str, prior: list[str]) -> list[str] | None:
    """The answer to a short follow-up of a newcomer answer, read with the turns before it."""
    if len(text.split()) > 14:
        return None
    before = " ".join(prior[-2:])
    for asked, earlier, answer in _FOLLOW_UPS:
        if asked.search(text) and earlier.search(before):
            return list(answer)
    return None


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
    r"(?:what(?:'?s|\s+is)\s+)?(?:the|my|a\s+good)\s+first\s+step(?:\s+today)?"
    # "ok so whats a good first step for someone like me" (round 38 newcomer)
    r"(?:\s+for\s+(?:someone|people|a\s+beginner|beginners|me)(?:\s+like\s+me)?)?|"
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


_FIRST: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    # "Is now a 100% safe time to go all-in on NVDA call options with my entire savings?" was
    # answered as a plain NVDA holding (round 40 hostile, N2): an option can expire at zero
    (re.compile(r"\ball[\s-]?in\b[^?]{0,60}\b(?:call|put)\s+options?\b|\b(?:call|put)\s+options?\b"
                r"[^?]{0,60}\b(?:all\s+my|entire|whole)\s+(?:savings|money|account|portfolio)\b",
                re.I), (
        "Bottom line: no — nothing is 100% safe, and options are the riskiest way to do this: a "
        "call bought with your entire savings goes to zero if the stock is below the strike at "
        "expiry, even when the stock itself only dipped a little.",
        "Options also lose value every day just from time passing (time decay), and their price "
        "jumps with implied volatility, so you can be right about the direction and still lose "
        "if the move is too small or too late.",
        "Savings are money you need back; if you want exposure, a small amount in the shares or "
        "the perpetual without leverage caps the loss at what you put in. Ask \"how much could I "
        "lose on NVDA in a bad week with $X\" with an amount you could lose in full.")),
    # Round 40, a first-time user, second half (round40_newcomer.md #9, #24, #31-#36, #39)
    (re.compile(r"\b(?:got|been|was|just\s+got)\s+liquidated\b|\bliquidated\b[^?]{0,80}\b(?:lost|"
                r"what\s+(?:even\s+)?happened|everything)\b", re.I), (
        "Bottom line: liquidation is the exchange closing a leveraged position for you once the "
        "loss has eaten the margin behind it — at 10x a move of about 10% against you is "
        "enough, at 20x about 5%, and the margin is gone with it.",
        "It happens fast because leverage multiplies the move: borrowed money lets a small "
        "deposit hold a big position, and the exchange closes it before the loss can exceed what "
        "you put in. On isolated margin that position's margin is lost; on cross margin the "
        "whole futures balance can be drawn on first.",
        "Stop for today. Before any next trade: lower leverage (or none), a smaller size, and a "
        "stop you set in advance. Ask \"where is my liquidation price on a 3x BTC long\" to see "
        "how far the line sits.")),
    (re.compile(r"\b(?:avoid|prevent|stop)\s+(?:that|this|it|getting\s+liquidated|liquidation)\s+"
                r"(?:from\s+happening\s+)?(?:again|next\s+time)\b|\bnot\s+get\s+liquidated\b",
                re.I), (
        "Bottom line: three levers, in order — use less leverage (at 2x a move of about 50% is "
        "needed to liquidate you, at 20x about 5%), trade a smaller size, and set your exit "
        "before you enter.",
        "Isolated margin caps what one position can take to its own margin; a stop-loss closes "
        "it at a level you chose instead of the exchange's liquidation line.",
        "Ask \"where is my liquidation price on a 3x BTC long\" or \"what leverage would have "
        "survived BTC's worst day\" — both are worked from Bitget's own prices.")),
    (re.compile(r"\b(?:is\s+it\s+)?normal\s+to\s+feel\b|\bfeel\w*\s+(?:so\s+)?(?:messed\s+up|"
                r"sick|terrible|awful|devastated|depressed)\b[^?]{0,60}\b(?:los\w+|money)\b",
                re.I), (
        "Bottom line: yes — feeling awful after losing money is normal, and it is a sign to "
        "step away from trading for now, not a sign that you are bad at this.",
        "The feeling pushes people to win it back fast, with more size or leverage; that is the "
        "move that turns a bad loss into a worse one. No trades today.",
        "Talk to someone you trust. If it is affecting your sleep or work, or you cannot stop "
        "thinking about trading, free confidential help exists — in the US, the National Problem "
        "Gambling Helpline at 1-800-GAMBLER.")),
    (re.compile(r"\b(?:quit|leave)\s+my\s+(?:job|work)\b[^?]{0,60}\b(?:trad\w+|day[\s-]?trad\w*)\b|"
                r"\bday[\s-]?trad\w*\s+(?:full[\s-]?time|for\s+a\s+living)\b", re.I), (
        "Bottom line: for almost everyone, no — the evidence on people who day trade for a "
        "living is that nearly all lose money after costs, so a job is the safer income while "
        "you test whether you have an edge.",
        "A study of Brazil's index-futures day traders (Chague, De-Losso and Giovannetti, 2019) "
        "found 97% of those who kept going for 300 days or more lost money, and only about 1% "
        "earned more than Brazil's minimum wage from it.",
        "If you want to try, do it part-time with a fixed amount you can lose, keep a record of "
        "every trade with its fees, and judge it after a hundred trades, not a lucky week.")),
    (re.compile(r"\b(?:do|does)\s+most\s+(?:day\s+)?traders?\s+(?:even\s+)?(?:come\s+out\s+ahead|"
                r"make\s+money|profit|win|lose)\b|\bhow\s+many\s+(?:day\s+)?traders?\s+(?:actually\s+)?"
                r"(?:make\s+money|profit|lose)\b", re.I), (
        "Bottom line: no — in the large studies of real trading accounts, most day traders lose "
        "money once fees are counted.",
        "Brazil's index-futures day traders: 97% of those who kept at it for 300 days or more "
        "lost money (Chague, De-Losso and Giovannetti, 2019). The pattern is the same across "
        "markets: costs are certain, an edge is rare.",
        "This console holds no dataset of other people's accounts; its own measured fact is the "
        "cost side — ask \"what does a round trip in BTC cost\" for the fee every trade pays.")),
    (re.compile(r"(?:\$\s?(?:10|20|25|30|50)\b|\b(?:10|20|25|30|50)\s*(?:bucks|dollars|usd)\b)"
                r"[^?]{0,60}\b(?:point|worth\s+it|enough|bother)\b", re.I), (
        "Bottom line: yes, as a way to learn — with $50 the lessons are real and the most you "
        "can lose is $50, as long as it stays on spot without leverage.",
        "The cost to watch is fees: about 0.1% a side on Bitget spot, so a $50 buy and sale "
        "costs about $0.10 — small, but trading often eats into a small balance quickly.",
        "Spot is the safer place for a small balance: you own the coin and cannot lose more "
        "than you paid; futures with leverage can wipe out the whole $50 on one move.")),
    (re.compile(r"\bis\s+spot\s+(?:trading\s+)?safer\b|\bspot\b[^?]{0,40}\bsafer\b", re.I), (
        "Bottom line: yes — on spot you buy the coin outright, so the most you can lose is what "
        "you paid; there is no leverage and nothing to liquidate.",
        "That does not make the price safe: a coin can still fall 50% or more, so the size is "
        "what protects you.",
        "Futures (perpetuals) add leverage, funding payments and liquidation; leave them until "
        "you have traded spot for a while.")),
    (re.compile(r"\bstop[\s-]?loss\b[^?]{0,80}\b(?:not|doesn'?t|didn'?t|fail\w*)\s+(?:work|fill|"
                r"trigger|protect)\w*\b", re.I), (
        "Bottom line: a stop-loss is an order that sells once the price falls to a level you "
        "set — and in a fast drop it can fill well below that level, because it becomes an "
        "order to sell at the next price available.",
        "That gap between your stop and the fill is slippage; a stop-limit order avoids selling "
        "below your floor, but in a crash it may not fill at all.",
        "So a stop limits a loss rather than guaranteeing a price: size the position so that "
        "even a fill past the stop is a loss you can take.")),
    (re.compile(r"\btax\w*\b[^?]{0,60}\b(?:us|usa|united\s+states|america|irs)\b[^?]{0,60}"
                r"\bstocks?\b|\b(?:us|usa|irs)\b[^?]{0,40}\btax\w*\b[^?]{0,60}\bdifferent\b",
                re.I), (
        "Bottom line: in the US, crypto is taxed much like stocks — gains on what you held over "
        "a year get the lower long-term rates (0%, 15% or 20%), and a year or less is taxed as "
        "ordinary income — but the IRS treats it as property, not a security.",
        "The difference that catches people: spending crypto or swapping one coin for another "
        "is a taxable sale too, not only selling for dollars; each one needs its own gain or "
        "loss worked out.",
        "Losses offset gains, as with stocks. Not tax advice — check the IRS guidance or a tax "
        "professional for your own case.")),
    # Round 40, a first-time user: each was declined, or answered with a desk log, a ticker
    # refusal, a coin definition or the wrong scam.
    (re.compile(r"\bexplain\s+(?:the\s+)?fees\b|\bfees?\b[^?]{0,40}\b(?:simple|plain|actually\s+"
                r"pay\w*|what\s+am\s+i\s+paying)\b|\bwhat\s+(?:fees|am\s+i\s+(?:actually\s+)?"
                r"paying)\b", re.I), (
        "Bottom line: on Bitget you pay a small cut of each trade — about 0.1% of the amount on "
        "spot, so $1.00 on a $1,000 buy and $1.00 again on the sale.",
        "On futures (perpetuals) it is 0.02% when your order waits in the book and 0.06% when it "
        "takes a price straight away, plus funding: a small payment between longs and shorts "
        "every 8 hours, which you pay or receive while the position is open.",
        "Two costs are not on the fee table: the spread (the gap between the buy and sell price, "
        "tiny on BTC, wider on small coins) and the network fee when you withdraw coins. Ask "
        "\"what does it cost to buy $100 of BTC\" for the exact figure on a real order.")),
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+does)\s+(?:a\s+|the\s+)?spot(?:\s+(?:trading|market|"
                r"buying))?(?:\s+mean)?\b|\bspot\s+(?:trading|market)\s+(?:mean|means)\b", re.I), (
        "Bottom line: spot means buying the coin itself, paid in full, right now — you own it, "
        "and the most you can lose is what you paid.",
        "It is called spot because the trade settles \"on the spot\": the coin lands in your "
        "account straight away.",
        "The other kind is futures (perpetuals): a contract on the price, often with borrowed "
        "money (leverage), where you never hold the coin and a loss can wipe out what you put "
        "in. Beginners usually start on spot.")),
    # an overnight gap keeps its own answer below (round 35); this is a fast move through a stop
    (re.compile(r"^(?!.*\b(?:overnight|asleep|weekend)\b).*?(?:\bgaps?\b[^?]{0,60}\bstop[\s-]?loss\b|"
                r"\bstop[\s-]?loss\b[^?]{0,60}\b(?:gap\w*|skip\w*|jump\w*\s+past|slip\w*)\b)",
                re.I), (
        "Bottom line: not always at your price — it still triggers, but a stop-loss becomes an "
        "order to sell at the next price available, so after a fast gap it fills lower than the "
        "stop (that difference is called slippage).",
        "A stop-limit order avoids selling too low by setting a floor, but in a fast fall it may "
        "not fill at all, leaving you holding the position.",
        "Crypto trades around the clock, so gaps are rarer than in stocks overnight, but a crash "
        "can still jump past levels in seconds; size the position so the gap you fear is a loss "
        "you can take.")),
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+are)\s+(?:an?\s+)?airdrops?\b|\bairdrops?\b[^?]{0,40}"
                r"\bfree\s+(?:money|coins?|crypto)\b|"
                # "Airdrop là gì" (Vietnamese) while the model that translates was paused
                # (round 40 newcomer, #26)
                r"\bairdrops?\s+(?:là\s+gì|qué\s+es|c'est\s+quoi|was\s+ist)\b", re.I), (
        "Bottom line: an airdrop is a project handing out its new token for free, usually to "
        "people who used its app early — a marketing move to spread ownership, not free money "
        "on demand.",
        "The real ones are announced on the project's own site and verified accounts, and "
        "often reward past use, so you cannot simply sign up for them afterwards; many tokens "
        "fall sharply once people sell what they were given.",
        "The fake ones come to you: a DM or a link asking you to connect a wallet or pay a fee "
        "to claim. Never connect your main wallet to a claim link — that is how wallets are "
        "drained.")),
    # "what is an nft" keeps its own definition below (round 34); this is whether they are over
    (re.compile(r"\bnfts?\b[^?]{0,40}\b(?:still\s+a\s+thing|dead|die\w*|over|done)\b", re.I), (
        "Bottom line: NFTs still exist but the boom is over — trading fell steeply from its "
        "2021-22 peak, and many collections bought then now sell for a fraction of what was "
        "paid, if they sell at all.",
        "An NFT is a unique token on a blockchain that records ownership of something, usually "
        "a digital image; owning one does not mean owning the copyright to the picture.",
        "They are not something Bitget's markets or this console price, and they are hard to "
        "sell quickly, so treat any money put in one as money you may not get back.")),
    (re.compile(r"\b(?:recover|get\s+back|retrieve)\b[^?]{0,40}\b(?:losses|lost\s+(?:money|"
                r"crypto|funds)|my\s+(?:money|crypto|funds))\b[^?]{0,80}\b(?:send|pay|fee|"
                r"deposit)\b|\b(?:telegram|discord|whatsapp|instagram|dm\w*)\b[^?]{0,60}"
                r"\brecover\w*\b", re.I), (
        "Bottom line: that is a scam — a \"recovery\" scam, aimed at people who have already "
        "lost money. Nobody can reverse a crypto transfer for a fee, and paying them only adds "
        "to the loss.",
        "Regulators warn about exactly this, including the US Federal Trade Commission: real "
        "help never asks for money up front, and never contacts you first.",
        "Do not send anything or share your wallet or login details. Block and report the "
        "account, and if you were defrauded, report it to the police and your country's fraud "
        "reporting service.")),
    (re.compile(r"\b(?:guaranteed|sure[\s-]?fire|100%)\b[^?]{0,40}\bsignals?\b|\bsignals?\s+"
                r"(?:group|channel|service)\b[^?]{0,60}\b(?:fee|pay\w*|subscri\w*|monthly|"
                r"legit|worth|scam)\b", re.I), (
        "Bottom line: a \"guaranteed\" signal group is a red flag on its own — no one can "
        "guarantee trading results, and a seller who could would trade the signals instead of "
        "selling them.",
        "Common patterns: screenshots of wins with the losses left out, groups that pump a "
        "small coin onto their own subscribers, and an upsell to a \"VIP\" tier or a managed "
        "account.",
        "If you want to judge any strategy, ask for its full record with every trade and the "
        "fees, or test the rule yourself here: \"backtest buying BTC when RSI drops below 30\".")),
    (re.compile(r"\bhow\s+long\s+(?:should|do|would|can)\s+(?:i|you|people)\s+(?:actually\s+|"
                r"usually\s+|normally\s+)?hold\b|\bhow\s+long\s+(?:is\s+)?(?:normal|usual)\s+to\s+"
                r"hold\b", re.I), (
        "Bottom line: there is no normal — it depends on why you bought. A trade on a short-term "
        "move can last hours; a bet that a coin grows over years is held through falls of 50% "
        "or more.",
        "Decide the reason and the exit before you buy: what would make you sell (a price, a "
        "date, or the reason no longer being true), so the market does not decide it for you.",
        "Holding longer usually costs less in fees, and in some countries (the US, for one) "
        "gains on what you held over a year are taxed at a lower rate.")),
    (re.compile(r"\bhow\s+(?:do|can|would)\s+(?:you|i|people)\s+(?:actually\s+)?(?:know|tell|"
                r"judge|check)\s+(?:if|whether)\s+(?:a\s+)?(?:trading\s+)?strateg\w+\s+(?:is\s+)?"
                r"(?:good|works?|any\s+good)\b", re.I), (
        "Bottom line: test it on data it was not built on, after fees, over enough trades — a "
        "strategy that only looks good on the period it was designed on usually fails next.",
        "The checks that matter: dozens of trades, not a handful; results after fees and "
        "slippage; the worst losing stretch (drawdown), not only the return; and a comparison "
        "with simply buying and holding.",
        "This console runs those checks on a rule in plain words: ask \"backtest buying BTC when "
        "RSI drops below 30\" and it reports the out-of-sample result, the fees and the worst "
        "stretch.")),
    (re.compile(r"\bindia\w*\b.{0,60}\btax\w*\b|\btax\w*\b.{0,60}\bindia\w*\b", re.I),
     (
        "Bottom line: India taxes crypto gains at a flat 30% (plus cess), with no deduction "
        "other than what you paid for the coin, and 1% TDS is withheld on each sale above the "
        "threshold.",
        "A loss on one coin cannot be set against a gain on another, or carried forward, which "
        "is why it is called harsh: frequent trading has to clear fees and tax before anything "
        "is left.",
        "Not tax advice — rules change each budget, so check the current Income Tax Department "
        "guidance or a chartered accountant.")),
    # Round 39, a first-time user: each of these was declined, or answered with a definition, a
    # rates briefing or the desk's own record instead of what was asked.
    (re.compile(r"\b(?:lost|down)\s+(?:like\s+|about\s+|almost\s+|over\s+)?(?:half|most|a\s+lot|"
                r"\d+\s*%)\b[^?]{0,80}\b(?:freak\w*|panick?\w*|scared|terrified|don'?t\s+know\s+"
                r"what\s+to\s+do|dont\s+know\s+what\s+to\s+do|can'?t\s+(?:sleep|breathe|think))\b|"
                r"\b(?:freak\w*\s+out|panick?ing)\b[^?]{0,60}\b(?:lost|savings|money)\b", re.I), (
        "Bottom line: stop for today — make no trade while you feel like this, because the move "
        "that turns a big loss into a bigger one is trading fast to win it back.",
        "What you lost is real, but what is left is still yours: if any of it is in futures or "
        "leverage, that part can still be liquidated, so check that first; coins bought on spot "
        "cannot fall below zero.",
        "Tomorrow, with a clear head, ask \"how much could I lose on BTC in a bad week with $X\" "
        "using what you hold now — that number is the one to decide on, not the loss.",
        "If the worry is taking over, tell someone you trust today; if the urge to trade it "
        "back will not stop, the US National Problem Gambling Helpline (1-800-GAMBLER) is free, "
        "confidential and open around the clock.")),
    (re.compile(r"\b(?:can'?t|cant|cannot|couldn'?t)\s+sleep\b|\bsell\s+(?:it\s+)?(?:all|"
                r"everything)\s+(?:right\s+)?now\b", re.I), (
        "Bottom line: no call from here on selling — but losing sleep over a position is the "
        "clearest sign it is bigger than you can carry, whatever the price does next.",
        "Selling all of it makes the loss final; holding all of it keeps the worry. Many traders "
        "cut to a size they can sleep with — selling part — so the decision is not all or "
        "nothing.",
        "Before deciding, ask \"how much could I lose on BTC in a bad week with $X\" with what "
        "you hold: if that number would hurt, the position is too large for you.")),
    (re.compile(r"\b(?:wife|husband|partner|girlfriend|boyfriend|parents?|family)\s+(?:doesn'?t|"
                r"doesnt|don'?t|dont|does\s+not|do\s+not)\s+know\b", re.I), (
        "Bottom line: this console cannot judge your relationship, but people who have been "
        "through it say the same thing: hiding a money loss tends to grow — both the secret and "
        "the trading done to fix it before anyone finds out.",
        "Telling them with the numbers in hand — what went in, what is left, and that you have "
        "stopped adding — is usually easier than the conversation after a bigger loss.",
        "If you feel pulled to keep trading to undo it, the National Problem Gambling Helpline "
        "(1-800-GAMBLER in the US) is free and confidential, and it helps partners too.")),
    # "ill feel so dumb if it keeps going up after i sell" is a worry about selling, not this
    (re.compile(r"(?<!\bill\s)(?<!i'll\s)(?<!will\s)(?<!would\s)(?<!gonna\s)"
                r"\b(?:feel|felt|am|i'?m|im)\s+(?:so\s+|really\s+|very\s+|such\s+an?\s+)?"
                r"(?:stupid|dumb|idiot|an\s+idiot|ashamed|embarrassed)\b", re.I), (
        "Bottom line: losing money on a first trade is common, not stupid — most people who "
        "start trading lose early, and the ones who keep going learn the most from exactly this.",
        "What turns it into a lesson: write down what you bought, how much, why, and what you "
        "would do differently (smaller size, no leverage, a planned exit).",
        "Then ask \"how much could I lose on BTC in a bad week with $X\" before the next trade, "
        "so the size is decided in advance, not in the moment.")),
    (re.compile(r"\b(?:i\s+)?(?:bought|buy)\s+(?:some\s+)?\$?[\w$]{2,10}\s+(?:and|but|then)\s+"
                r"(?:it|the\s+price)\s+(?:went|go(?:es)?|dropped|fell|crashed|tanked)\b[^?]{0,40}"
                r"\b(?:what\s+(?:do|should|now)|now\s+what|help)\b|\b(?:bought|buy)\b[^?]{0,30}"
                r"\b(?:went|goes|go)\s+down\s+(?:right\s+)?(?:after|straight\s+away|"
                r"immediately)\b", re.I), (
        "Bottom line: a fall right after buying happens to almost everyone — prices move both "
        "ways all day, so the first hours after a buy say little about the trade.",
        "The question that matters is whether your reason for buying is still true. If it is, a "
        "dip is noise; if you bought because it was rising, that reason has already gone.",
        "Two rules that save beginners the most: do not add more or use leverage to \"fix\" it, "
        "and know in advance how far you would let it fall — ask \"how much could I lose on BTC "
        "in a bad week with $X\" with what you hold.")),
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+even\s+is)\s+(?:a\s+)?usdt\b|\busdt\b[^?]{0,40}\b(?:mean|"
                r"stand\s+for)\b|\bwhy\s+(?:does\s+)?(?:everything|every\s*thing|all\s+prices?|it)"
                r"\s+(?:cost|priced?|in|say|is\s+in)\s+usdt\b", re.I), (
        "Bottom line: USDT (Tether) is a stablecoin — a token built to stay worth one US dollar, "
        "so a price of \"85,000 USDT\" on Bitget means about $85,000.",
        "Exchanges price coins in USDT because it moves on the same networks as the coins "
        "themselves: you can swap between BTC and a dollar stand-in instantly, without a bank.",
        "It is not a bank dollar: it is Tether's promise, backed by reserves Tether says it holds "
        "and reports on quarterly. It has stayed close to $1, but ask \"is USDT safe to hold\" "
        "for what happened when other stablecoins lost the peg.")),
    (re.compile(r"\bhow\s+do\s+(?:people|traders|they|you|u)\s+(?:actually\s+|really\s+)?"
                r"(?:make|earn)\s+money\b|\b(?:can|does)\s+(?:this|trading|crypto|it)\s+"
                r"(?:actually\s+|really\s+)?(?:make|earn)\s+(?:real\s+)?money\b[^?]{0,40}\b"
                r"(?:luck|lucky|gambl\w*)\b|\b(?:real\s+)?money\s+or\s+(?:is\s+it\s+)?(?:all\s+|"
                r"just\s+)?luck\b", re.I), (
        "Bottom line: three ways, and only one is reliable: holding something that grows in value "
        "over years, earning a fee or yield for providing a service (exchanges and market makers "
        "earn on every trade), and short-term trading — which most people lose at.",
        "That last part is measured, not opinion: a study of Brazil's index-futures day traders "
        "(Chague, De-Losso and Giovannetti, 2019) found 97% of those who kept at it for 300 days "
        "or more lost money.",
        "Short-term results are mostly luck until there are many trades; ask \"what is this "
        "desk's track record\" to see how ARGUS's own paper desk is scored, and why a handful of "
        "trades proves nothing either way.")),
    (re.compile(r"\bbuy\s+(?:it\s+)?(?:all\s+)?(?:at\s+)?once\b[^?]{0,40}\b(?:every|each|monthly|"
                r"weekly|over\s+time|bit\s+by\s+bit|little)\b|\b(?:all\s+at\s+once|lump\s+sum)\s+"
                r"or\s+(?:a\s+little|monthly|bit\s+by\s+bit|every)\b", re.I), (
        "Bottom line: neither is right for everyone — buying all at once has usually ended ahead "
        "slightly more often in markets that mostly rise, and buying a little every month means "
        "no single day's price decides the whole amount.",
        "Monthly buying (dollar-cost averaging) suits money that arrives monthly anyway, and "
        "anyone who would regret putting everything in on the worst possible day.",
        "Either way, the amount matters more than the timing: ask \"how much could I lose on BTC "
        "in a bad week with $X\" for the total you would end up holding.")),
    (re.compile(r"\b(?:bitget|the\s+exchange|an\s+exchange|it)\s+(?:goes|go|went|going)\s+"
                r"(?:bust|bankrupt|under|broke)\b|\bif\s+bitget\s+(?:fails|collapses|shuts\s+down|"
                r"closes|disappears)\b|\bbitget\s+(?:goes|went)\s+(?:down|offline)\s+forever\b",
                re.I), (
        "Bottom line: your coins on Bitget are a claim on Bitget, not coins kept apart in your "
        "name — its own terms say user assets are not segregated on-chain and it is not a "
        "trustee of them (Terms of Use, 19.12). If it failed, you would be a creditor in line.",
        "That has happened before: FTX customers waited about two years and were repaid at their "
        "balances' dollar value on the day it collapsed, missing the rally since.",
        "What to check and do: Bitget publishes proof-of-reserves reports and a protection fund "
        "on its own site; keep there only what you are trading, and move what you hold long term "
        "to a wallet only you control.")),
    (re.compile(r"\b(?:use|download|get)\s+the\s+app\s+or\s+(?:the\s+)?(?:website|site|web|"
                r"browser)\b|\b(?:app|website)\s+(?:or|vs\.?|versus)\s+(?:the\s+)?(?:app|website|"
                r"site|web)\b", re.I), (
        "Bottom line: either — the app and the website are the same Bitget account with the "
        "same markets, balances and fees, so it is a question of where you like to read charts.",
        "The app adds price alerts and quick logins; the website gives bigger charts.",
        "The one rule that matters more: install the app only from the official App Store or "
        "Google Play listing, and reach the website by typing the address yourself — fake "
        "Bitget apps and look-alike sites exist to steal logins.")),
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+are)\s+(?:a\s+|the\s+)?candles?\b|\bcandles?\s+on\s+(?:the|a)"
                r"\s+chart\b", re.I), (
        "Bottom line: a candle is the price over one stretch of time — a minute, an hour, a "
        "day: its body runs from where the price opened to where it closed, green if it closed "
        "higher, red if lower.",
        "The thin lines above and below are wicks: the highest and lowest prices reached in that "
        "stretch, even if the price came back before it closed.",
        "When this console says \"Bitget hourly candles\", it means one of these per hour.")),
    (re.compile(r"\b(?:dad|mom|mum|father|mother|parents?|friends?|wife|husband|people)\s+"
                r"(?:says?|said|thinks?|thought|tells?\s+me)\s+(?:that\s+)?(?:crypto|bitcoin|"
                r"trading|this|it)\s+is\s+(?:a\s+|just\s+a\s+|all\s+a\s+)?(?:scam|fraud|ponzi|"
                r"gambling|fake)\b", re.I), (
        "Bottom line: partly right — Bitcoin itself is a real network that has run since 2009 "
        "and trades in the open on regulated markets, but crypto is also full of scams, and "
        "most of the money people lose to \"crypto\" goes to those.",
        "What makes something a scam is the offer, not the coin: a guaranteed return, pressure "
        "to buy now or recruit friends, or someone else holding your money for you.",
        "And even the real coins are very risky: Bitcoin fell about 77% from its November 2021 "
        "high to its November 2022 low. A fair answer to give back: \"it's not all a scam, but "
        "it can lose most of its value\".")),
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+does)\s+(?:a\s+|an?\s+)?(?:short|shorting|short\s+"
                r"(?:sell(?:ing)?|position))(?:\s+mean)?\s*[?.!]*\s*$|\bwhat\s+does\s+(?:it\s+mean\s+"
                r"to\s+)?(?:go(?:ing)?\s+)?short\b|\bhow\s+(?:does|do)\s+(?:shorting|a\s+short)\s+"
                r"work\b|\bexplain\s+(?:shorting|short\s+selling|going\s+short)\b", re.I), (
        "Bottom line: a short is a bet that the price falls — you sell first and buy back later; "
        "if the price has dropped, you keep the difference.",
        "On Bitget it is done with perpetual futures: open a short, and it gains as the price "
        "falls and loses as it rises.",
        "The risk is the reverse of buying: a price can rise without limit, so a short with "
        "leverage can be liquidated by a sharp jump. Ask \"what is liquidation\" next.")),
    (re.compile(r"\b(?:explain|describe)\s+(?:leverage|margin|liquidation)\b[^?]{0,40}\b(?:pizza|"
                r"food|cake|burgers?|like\s+i'?m\s+(?:5|five)|simple\s+analogy)\b", re.I), (
        "Bottom line: leverage with pizza — you have $10, borrow $40, and buy $50 of pizza to "
        "resell. That is 5x: you control five times the money you put in.",
        "If pizza prices rise 10%, you sell for $55, repay the $40 and keep $15 — a 50% gain on "
        "your $10. If they fall 20% to $40, repaying the $40 leaves you nothing: your $10 is "
        "gone.",
        "On an exchange that last part is liquidation — it closes the position for you before "
        "the loss can eat into the borrowed part. Leverage multiplies wins and losses alike.")),
    (re.compile(r"\bnever\s+(?:traded|invested|bought)\b[^?]{0,50}\b(?:where|how)\s+(?:do\s+i|to|"
                r"should\s+i)\s+(?:even\s+)?(?:begin|start)\b|\b(?:where|how)\s+do\s+i\s+even\s+"
                r"(?:begin|start)\b", re.I), (
        "Bottom line: begin by learning with no money at risk — ask this console what a coin did "
        "in its worst week, what a perpetual is, what liquidation is — then, if you go ahead, "
        "start small on the spot market, where the most you can lose is what you paid.",
        "The steps on Bitget: open an account, verify your identity (you must be 18 or older), "
        "add a small amount, and buy on spot — no futures, no leverage.",
        "Decide the size before you buy: ask \"how much could I lose on BTC in a bad week with "
        "$100\" with the amount you have in mind.")),
    (re.compile(r"\b(?:which|what)\s+coins?\s+(?:should|do|would|can)\s+(?:a\s+)?(?:total\s+|"
                r"complete\s+)?(?:beginner|newbie|noob|new\s+person|i)\s+(?:even\s+)?(?:look\s+at|"
                r"start\s+with|buy|pick|get)\b|\bbest\s+coins?\s+for\s+(?:a\s+)?(?:beginners?|"
                r"newbies?)\b", re.I), (
        "Bottom line: no pick from here — but most beginners who start look at the two largest, "
        "Bitcoin and Ether, because they have the deepest markets and the longest histories.",
        "Largest does not mean safe: Bitcoin fell about 77% from its November 2021 high to its "
        "November 2022 low, and Ether fell further. Small coins have fallen 90% or more and many "
        "never came back.",
        "Ask \"how much could I lose on BTC in a bad week with $100\" for any name you are "
        "considering, before you buy it.")),
    (re.compile(r"\bhow\s+do\s+i\s+(?:actually\s+|even\s+|really\s+)?know\s+when\s+to\s+"
                r"(?:buy|sell|get\s+in)\b|\bwhen\s+(?:is|'?s)\s+the\s+(?:right|best)\s+time\s+to\s+"
                r"buy\b", re.I), (
        "Bottom line: nobody knows the right moment in advance, and anyone who says they do is "
        "guessing or selling something — so the useful question is how you will buy, not when.",
        "Two plain approaches: buy a fixed amount on a schedule (every week or month), or write "
        "down a rule before you buy (\"after a 10% fall\") and follow it, instead of deciding "
        "on the day.",
        "To see what a rule has done before, ask \"what happened after BTC fell 10% in a "
        "week\" — that is past behaviour, not a promise.")),
    (re.compile(r"\b(?:i\s+think\s+)?(?:i'?m|im|i\s+am)\s+ready\b[^?]{0,30}\b(?:first\s+step|"
                r"what\s+(?:now|next|do\s+i\s+do))\b", re.I), (
        "Bottom line: the first step is the account: open one on Bitget, verify your identity "
        "(18 or older), and switch on two-factor login before you add money.",
        "Then add only an amount you could lose in full, buy on spot (not futures), and write "
        "down why you bought and the price at which you would sell.",
        "Ask \"how much could I lose on BTC in a bad week with $X\" with your amount first.")),
    (re.compile(r"\b(?:use|borrow|on|through|with)\s+(?:my\s+)?(?:mom'?s|mum'?s|dad'?s|"
                r"parents?'?|mother'?s|father'?s|brother'?s|sister'?s|friend'?s|someone\s+"
                r"else'?s)\s+(?:bitget\s+|exchange\s+)?account\b", re.I), (
        "Bottom line: no — Bitget's terms say an account is used only by the person it belongs "
        "to, not on behalf of anyone else (Terms of Use, 5.2), and that it is for adults 18 or "
        "older (2.2).",
        "Breaking that can get the account frozen, money included, and legally every coin in "
        "it is your parent's, not yours.",
        "What you can do now, free: ask this console how any coin has behaved — its worst week, "
        "what leverage does — and keep your own money until you can open your own account.")),
    # a bare "fomo" is left to the FOMO-and-rekt answers already here (round 34)
    (re.compile(r"\b(?:scared|afraid|worried)\s+of\s+missing\s+out\b|\beveryone\s+"
                r"(?:at\s+school\s+)?(?:is\s+)?(?:talks?|talking)\s+about\s+(?:crypto|bitcoin)\b|"
                r"\bfriends?\b[^?]{0,40}\btalk\w*\s+about\s+(?:crypto|bitcoin)\b", re.I), (
        "Bottom line: the fear of missing out is the feeling markets use against beginners — "
        "by the time everyone around you is talking about a coin, it has usually already risen.",
        "There is no last train: Bitcoin has been around since 2009 and will still be there "
        "next year; what you can actually miss by waiting is a few weeks of gains, and what you "
        "can avoid is buying at a top.",
        "Learning costs nothing: ask what a coin did in its worst week, or what leverage does, "
        "and decide later with your own money and your own reasons.")),
    (re.compile(r"\b(?:youngest|minimum|min)\s+age\b[^?]{0,40}\b(?:trad\w*|invest\w*|crypto|"
                r"bitget|start)\b|\bhow\s+old\s+(?:do\s+)?(?:you|i)\s+(?:have\s+to|need\s+to|must)"
                r"\s+be\b", re.I), (
        "Bottom line: on Bitget the youngest age to start trading is 18 — its terms require every "
        "individual user to be at least 18 (Terms of Use, 2.2), and most exchanges and brokers "
        "set the same age.",
        "Before 18, investing is possible in other ways: in many countries a parent can open a "
        "custodial account for a child at a regular broker, which the parent controls until the "
        "child comes of age.",
        "Learning needs no account: ask this console what a coin did in its worst week, or "
        "what a perpetual is.")),
    (re.compile(r"\bhow\s+much\s+(?:would|do|should)\s+i\s+(?:actually\s+)?(?:need\s+to\s+)?save"
                r"\b[^?]{0,40}\b(?:month|week)\b[^?]{0,60}\bdown\s*payment\b", re.I), (
        "Bottom line: divide the deposit by the months you have: 20% of a $300,000 home is "
        "$60,000, and over 24 months that is $2,500 a month; for a $400,000 home it is $80,000, "
        "about $3,333 a month.",
        "Money saved in an insured savings account or short-term government bills earns a "
        "little on top, which lowers that monthly figure slightly; money put in crypto could "
        "fall a third or more in the meantime.",
        "Say the price of the home and the months you have, and the same sum gives your own "
        "monthly figure. Closing costs come on top of the deposit.")),
    (re.compile(r"\b(?:is\s+it\s+)?safe\s+to\s+(?:put|keep|store|leave)\s+(?:my\s+)?(?:savings|"
                r"life\s+savings|money|ipon)\s+(?:here|in\s+(?:here|this|crypto|bitget)|on\s+"
                r"(?:here|bitget))\b", re.I), (
        "Bottom line: savings — money you would need back — should not be put into crypto "
        "trading. Prices here can fall a third in a week, and Bitget is an exchange, not an "
        "insured bank.",
        "If you do put money in, use only what you could lose in full, keep it on spot without "
        "leverage, and keep the savings themselves in an insured bank account.",
        "Ask \"how much could I lose on BTC in a bad week with $X\" with the amount you are "
        "thinking of, to see it in your own money.")),
    (re.compile(r"\b(?:is\s+it\s+)?safe\s+to\s+(?:use|start\s+with|learn\s+with|try\s+with)\s+"
                r"(?:a\s+)?(?:small|little|tiny)\s+(?:amount|money|sum)\b|\bsmall\s+(?:amount\s+of\s+)?"
                r"money\s+to\s+learn\b", re.I), (
        "Bottom line: yes, a small amount you could lose in full is the sensible way to learn — "
        "the lessons are real and the cost is capped.",
        "Keep it on the spot market, where the most you can lose is what you put in, and avoid "
        "futures and leverage until you know exactly how liquidation works.",
        "Fees matter more on small sums: on Bitget spot a trade costs about 0.1% a side, so "
        "buying and selling often eats into a small balance fastest.")),
    (re.compile(r"\b(?:get|put|move|send|transfer|deposit|add|load)\s+(?:my\s+)?(?:money|cash|"
                r"funds|dollars|rupees|euros)\s+(?:on(?:to)?|in(?:to)?|to)\s+(?:bitget|the\s+"
                r"exchange|(?:my\s+)?(?:bitget\s+)?account)\b|\bhow\s+(?:do\s+i|to|can\s+i)\s+"
                r"(?:deposit|fund\s+my|"
                r"top\s+up)\b", re.I), (
        "Bottom line: on Bitget itself, not here — this console never holds or moves money. "
        "In Bitget's app or site you either deposit crypto you already hold from another wallet, "
        "or buy crypto with local money through its buy-crypto options (card, bank transfer or "
        "peer-to-peer, depending on your country), after identity verification.",
        "Start small: a first deposit you could lose in full while you learn how fees and swings "
        "feel. Ask \"what does it cost to buy $100 of BTC\" for the fee on a real order.",
        "Which payment methods your country gets is Bitget's to say; this console has not "
        "checked them for you.")),
    (re.compile(r"\b(?:lose|losing)\s+(?:my\s+)?(?:house|home|car|savings|everything\s+i\s+own)\b|"
                r"\bgo\s+(?:into|in)\s+debt\b|\bend\s+up\s+owing\b", re.I), (
        "Bottom line: not from trading on Bitget itself — on spot the most you can lose is what "
        "you put in, and with leverage the exchange closes the position before your account "
        "goes below zero. What costs people their house is borrowing to trade, or putting in "
        "money they need.",
        "So the rules that keep a house safe: never borrow to trade, never use leverage you do "
        "not understand, and only put in what you could lose in full.",
        "Ask \"can I lose more than I put in\" for how leverage and liquidation work.")),
    (re.compile(r"\b(?:good|best|safe|safest|recommended)\s+(?:app|exchange|platform|site|wallet)"
                r"\s+(?:for|to)\s+(?:crypto|bitcoin|trading|buy\w*|start\w*)\b|\bwhich\s+"
                r"(?:app|exchange|platform)\s+should\s+i\s+use\b", re.I), (
        "Bottom line: this console will not rank apps — it is built on Bitget's data, so it is "
        "not a neutral judge of exchanges.",
        "What to check in any of them: that it is allowed to serve your country, that it "
        "publishes proof of reserves, what it charges per trade and per withdrawal, and that you "
        "can withdraw to your own wallet.",
        "Ask \"what does it cost to buy $500 of BTC on Bitget\" to see Bitget's side of that.")),
    (re.compile(r"\b(?:sell|selling|sold|share|sharing)\s+(?:my|our|your\s+users'?)\s+"
                r"(?:data|info\w*|questions)\b|\bwhat\s+do\s+you\s+do\s+with\s+my\s+"
                r"(?:data|questions)\b|\b(?:is|are)\s+(?:my\s+)?(?:data|questions)\s+"
                r"(?:stored|saved|kept|private)\b", re.I), (
        "Bottom line: no — nothing you type is sold or kept: the console keeps one anonymous line "
        "per answered question (which engine answered, how long it took, never the words), "
        "counted by a daily hash that cannot be traced back to you.",
        "Your book and what you tell it stay in your own browser. One thing leaves the page: an "
        "oddly phrased question is read by a language model (Qwen, through Bitget's hackathon "
        "service) to pick the engine that answers it.",
        "It makes no money from you either — no fees, no ads, no account.")),
    (re.compile(r"\b(?:make|making|earn)\s+money\s+(?:off|from|out\s+of)\s+(?:me|us|users|"
                r"people)\b|\bbusiness\s+model\b", re.I), (
        "Bottom line: it makes no money from you — no fees, no ads, no account, and nothing to "
        "sell: it is a hackathon project, free to use.",
        "What it keeps of your data: one anonymous line per answered question (which engine "
        "answered, how long it took — never the words you typed), counted by a daily hash that "
        "cannot be traced back to you. Your book and what you tell it stay in your own browser.",
        "One thing leaves the page: an oddly phrased question is read by a language model "
        "(Qwen, through Bitget's hackathon service) to pick the engine that answers it.")),
    (re.compile(r"\b(?:i\s+)?(?:don'?t|dont|do\s+not|never)\s+(?:really\s+)?trust\s+(?:ai|bots?|"
                r"this|you|chatbots?|robots?)\b|\bwhy\s+should\s+i\s+trust\s+(?:your|the|these|"
                r"those)\s+(?:numbers|figures|answers|data)\b", re.I), _TRUST_A),
    (re.compile(r"\bhalal\b|\bharam\b|\bshariah?\b|\bislamic(?:ally)?\b|\b(?:allowed|permitted|"
                r"permissible|ok(?:ay)?|forbidden)\s+in\s+islam\b|\bislam\s+(?:allow|permit)s?\b",
                re.I), (
        "Bottom line: that is a religious ruling, and this console cannot make it — scholars "
        "disagree, so ask a scholar you trust.",
        "The facts they usually weigh, as they apply on Bitget: buying a coin or an rToken on "
        "spot is owning it, with no interest; perpetual futures charge or pay a funding fee "
        "every few hours and are often used with borrowed leverage; and many coins have no "
        "business behind them, which some treat as excessive uncertainty (gharar).",
        "Bitget's own terms say whether it offers an Islamic or swap-free account; this console "
        "has not checked.")),
    (re.compile(r"\bis\s+(?:crypto|bitcoin|trading\s+crypto|bitget)\s+(?:even\s+|actually\s+)?"
                r"(?:legal|allowed|banned|illegal)\s+in\s+india\b|\bcrypto\s+(?:ban|legal\w*)\s+"
                r"(?:in\s+)?india\b", re.I), (
        "Bottom line: yes — owning and trading crypto is legal in India, and taxed: the Supreme "
        "Court struck down the RBI's 2018 banking ban in March 2020, and since 2022 gains are "
        "taxed at 30% with 1% TDS on each transfer.",
        "There is still no law that regulates crypto as such, and exchanges serving Indian users "
        "have to register with India's Financial Intelligence Unit (FIU-IND). Whether a given "
        "exchange is registered is on FIU-IND's list; this console has not checked Bitget's "
        "status.",
        "Not legal or tax advice — the rules change; check the current position before you "
        "put money in.")),
    (re.compile(r"\bwhat\s+time\s+does\s+(?:the\s+)?(?:us\s+|american\s+)?stock\s+market\s+"
                r"(?:open|close)\b|\bwhen\s+does\s+(?:the\s+)?(?:us\s+)?stock\s+market\s+"
                r"(?:open|close)\b|\bstock\s+market\s+hours\b|"
                # "what hours is the NYSE open" read NYSE as a ticker (a round-38 re-ask)
                r"\bwhat\s+(?:hours|time)\s+(?:is|does|are)\s+(?:the\s+)?(?:nyse|nasdaq|us\s+"
                r"market|stock\s+market|wall\s+street)\b|\b(?:nyse|nasdaq)\s+(?:trading\s+)?"
                r"hours\b|\bwhen\s+(?:is|does)\s+(?:the\s+)?(?:nyse|nasdaq)\s+(?:open|close)\b",
                re.I), (
        "Bottom line: the US stock market (NYSE and Nasdaq) opens at 9:30 and closes at 16:00 New "
        "York time, Monday to Friday, except market holidays — 13:30 to 20:00 UTC in summer, "
        "14:30 to 21:00 UTC in winter.",
        "On Bitget the stock perpetuals and rTokens trade around the clock, so their price moves "
        "while the stock market is shut and jumps toward the stock at the open.",
        "Ask \"is the US market open right now\" for today's session and any holiday.")),
    (re.compile(r"^\W*(?:hi\W+|hey\W+)?(?:i'?m|im|i\s+am)\s+(?:completely\s+|totally\s+|really\s+|"
                r"very\s+|super\s+)?new\s+(?:to\s+(?:all\s+)?(?:this|trading|crypto|investing)|"
                r"here)\W*$", re.I), (
        "Bottom line: welcome — this console answers questions about Bitget's markets in plain "
        "words, shows where every number comes from, and never trades or touches money.",
        "Good first questions: \"I have $100, where should I start?\", \"how much could I lose "
        "on BTC in a bad week\", \"what is a stop loss and do I need one\" — or click one of the "
        "\"New to trading\" suggestions under the box.",
        "Nothing you ask here can cost you anything, so there is no wrong question.")),
    (re.compile(r"\bwhat\s+should\s+i\s+(?:actually\s+)?(?:click|press|do)\s+(?:on\s+)?(?:on\s+)?"
                r"(?:this|the)\s+(?:site|page|website|console)\b|\bhow\s+do\s+i\s+use\s+this\s+"
                r"(?:site|page|website)\b", re.I), (
        "Bottom line: type a question in the box at the top and press Ask — in your own words — "
        "or click one of the suggestions under \"New to trading\".",
        "\"Guided research\" walks through five questions about a book you hold; the menu at the "
        "top has the pages that show the console's record (Status) and what it got wrong.",
        "Nothing on this site places an order or touches money.")),
    (re.compile(r"\b(?:mess|screw)\s+(?:something|it|anything)\s+up\b|\bcan\s+i\s+undo\b|"
                r"\bbreak\s+something\b", re.I), (
        "Bottom line: nothing here can be messed up — the console places no orders and holds no "
        "money, so every question is safe to ask and can be asked again.",
        "What you tell it (your book, your limits) stays in your own browser, and you can forget "
        "any of it from the list under My book; pinned questions can be unpinned.",
        "On Bitget itself it is different: a filled order cannot be undone, only closed with "
        "another order, at the price then.")),
    # a presale has its own answer (round 34), so it is left to it
    (re.compile(r"^(?!.*\bpre[\s-]?sales?\b).*?"
                r"\b(?:friend|cousin|brother|sister|mate|colleague|buddy)\b[^?.]{0,40}\b(?:made|"
                r"got|turned)\b[^?.]{0,30}\b(?:\d+\s*x|\d{2,}\s*%|rich|a\s+lot|so\s+much|bank)\b"
                r"[^?]{0,60}\b(?:should|can|do)\s+i\b", re.I), (
        "Bottom line: their gain says nothing about yours — a coin that made 5x has already "
        "moved, and you would be buying after the move, at the higher price.",
        "For every friend who made 5x on a meme coin there are many who bought late and lost "
        "most of it; you hear from the winners. Meme coins have no business behind them, so "
        "their price runs on attention, which can leave as fast as it came.",
        "If you still want some: only an amount you could lose in full, no leverage — and ask "
        "\"how much could I lose on DOGE in a bad week\" first to see what that means.")),
    (re.compile(r"\bis\s+\$?\s?(?P<amt>\d[\d,]*)\s*(?:dollars?|bucks|usd|\$)?\s+enough\s+to\s+"
                r"(?:learn|start|begin|trade|invest|try)\b", re.I), (
        "Bottom line: yes, to learn — Bitget's smallest spot order is about $1 for bitcoin, so "
        "$100 can buy real coins, and what you learn is how fees, swings and your own nerves "
        "feel with real money.",
        "Keep it spot, without leverage, so the most you can lose is the $100; a 0.1% fee on each "
        "side means each round trip costs about 20 cents on $100.",
        "Ask \"I have $100, where should I start?\" for what that sum has been through in three "
        "broad markets.")),
)
"""Questions no market engine should take, asked before any of them reads the question: a
deposit question went to the account-access refusal and "is crypto allowed in islam" to the
sentiment reader (round 38 re-asks)."""


_PLAIN: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    *_FIRST,
    # Round 37, a first-time user: each of these was declined as unrecognised.
    # "so you dont actually buy anything for me right" (A2) is the trust question asked as a check
    (re.compile(r"\b(?:you|u|it|this|argus)\s+(?:do\s*n'?o?t|dont|don't|never|can'?t|cannot|"
                r"won'?t)\s+(?:actually\s+|really\s+|ever\s+)?(?:buy|sell|trade|place|touch|move|"
                r"invest)\b[^?]{0,40}\b(?:right|correct|yeah|yes|do\s+you|for\s+me|my\s+money|"
                r"orders?)\b|\b(?:do|will|can|could)\s+(?:you|u|it|this|argus)\s+(?:ever\s+|"
                r"actually\s+)?(?:place|make|execute)\s+(?:any\s+|an?\s+)?(?:orders?|trades?)\b",
                re.I), _TRADING_FOR_YOU),
    # "are you the same thing as the bitget app itself" (A3) was told "the same" referred to nothing
    (re.compile(r"\b(?:are|is)\s+(?:you|u|this|argus|it)\s+(?:the\s+same\s+(?:thing\s+)?as|part\s+of|"
                r"made\s+by|run\s+by|owned\s+by|from|affiliated\s+with|(?:the\s+|an?\s+)?"
                r"official\w*)\b[^?]{0,30}"
                r"\bbitget\b|\b(?:are|is)\s+(?:you|u|this|argus|it)\s+(?:the\s+)?bitget(?:'?s)?\s+"
                r"(?:app|own|official|exchange|website|site)\b", re.I), (
        "Bottom line: no — ARGUS is a separate research console, a Bitget AI hackathon entry, not "
        "the Bitget app and not run by Bitget. It reads Bitget's public market data and Bitget's "
        "own AI tools; it cannot see or touch your Bitget account.",
        "The Bitget app is where you hold money and place trades. This console only answers "
        "questions — what a market is doing, what a trade would cost, what it could lose — and "
        "every figure names its source.",
        "So trades, deposits and withdrawals happen in the Bitget app, never here.")),
    # "can you watch nvda for me and tell me when it dips" (E)
    (re.compile(r"\b(?:can|could|will|would)\s+(?:you|u|it|this)\s+(?:please\s+)?(?:watch|monitor|"
                r"track|keep\s+an\s+eye\s+on|alert\s+me|notify\s+me|ping\s+me|tell\s+me\s+when)\b|"
                r"\b(?:set|create|make)\s+(?:me\s+)?(?:an?\s+)?(?:price\s+)?alerts?\b|\bprice\s+alerts?\b",
                re.I), (
        "Bottom line: this web page cannot watch a price and tell you when it dips or jumps — it "
        "answers when you ask and keeps nothing running between visits.",
        "Alerts exist in ARGUS's Telegram bot when it runs on an always-on machine: \"/watch NVDA "
        "below 170\" messages you once the price trades there (pip install the project, then "
        "\"argus-bot setup\" with your own BotFather token). The hosted bot, @argusbitgetbot, "
        "answers questions but runs as short calls, so it cannot keep a watch.",
        "To check yourself here: ask \"where is NVDA trading\" any time, or pin that answer (the "
        "pin button under it) and it is asked again, live, every time you open this page.")),
    # "ok so how do i check it myself then", after the alert answer (E, turn 2)
    (re.compile(r"\bhow\s+(?:do|can|would)\s+i\s+(?:check|see|watch|track|follow|look\s+up)\s+"
                r"(?:it|that|this|the\s+price|prices?|them)\s+(?:my\s*self|on\s+my\s+own|myself)\b",
                re.I), (
        "Bottom line: ask \"where is NVDA trading\" (or any name) here — the price is read live "
        "from Bitget every time you ask.",
        "Pin that answer with the pin button under it and it is asked again, live, every time you "
        "open this page, so the check is one visit.",
        "For a message when a level trades, run ARGUS's Telegram bot on your own machine and use "
        "\"/watch NVDA below 170\".")),
    # "i lost 30% of my money this week what now" (D1) — a loss stated, and nothing asked but help
    (re.compile(r"(?:\b(?:i(?:'?ve)?|just)\s+(?:just\s+)?lost|\b(?:i'?m|i\s+am|im|i'?ve\s+been|"
                r"been)?\s*down)\s+(?:like\s+|about\s+|almost\s+|over\s+)?"
                r"(?:\$?\d[\d,.]*\s*(?:%|k\b|dollars?|bucks|usd\w*)?|half|most|a\s+lot|everything|"
                r"all)[^?]{0,60}\b(?:what\s+(?:now|do\s+i\s+do|should\s+i\s+do|next)|help|now\s+what|"
                r"what\s+(?:can|should)\s+i)\b", re.I), (
        "Bottom line: first, stop adding money or leverage while it still hurts — the most common "
        "way a big loss becomes a far bigger one is trading bigger to win it back quickly.",
        "Then look at what is left with the numbers, not the feeling: what you hold now, whether "
        "any of it is leveraged (a leveraged position can still be liquidated), and how much more "
        "it could fall in a bad week — ask \"how much could I lose on BTC in a bad week with "
        "$700\" with your own name and sum.",
        "Whether to hold or sell is your call, and a loss already taken is not a reason either "
        "way; the question is whether you would buy what you hold today, at today's price, with "
        "money you could afford to lose.")),
    # "should i put more money in to average it out" (D2): averaging down
    (re.compile(r"\baverag\w*\s+(?:it\s+|my\s+\w+\s+|the\s+\w+\s+)?(?:out|down)\b|\bput\s+(?:more|"
                r"extra)\s+(?:money\s+)?in\b[^?]{0,40}\b(?:averag\w*|lower|bring\s+down|recover|"
                r"make\s+(?:it\s+)?back)\b|\bbuy\s+more\s+to\s+(?:lower|average|bring\s+down)\b",
                re.I), (
        "Bottom line: averaging down lowers your average price but raises how much you have on the "
        "same bet — if it keeps falling, the loss grows faster, not slower.",
        "It helps only if the reason you bought is still true and the extra money is money you "
        "could lose in full; it never makes the earlier loss smaller, it just needs a smaller "
        "rebound to break even on a bigger stake.",
        "Never add with leverage or with money you need: ask \"how much could I lose on BTC in a "
        "bad week with $1,000\" with the total you would have in, before adding.")),
    # "ok so how much should i actually risk then" (G2)
    (re.compile(r"^\W*(?:(?:ok(?:ay)?|so|then|alright|but)\W+){0,3}how\s+much\s+(?:money\s+)?"
                # "how much should i invest" keeps its own answer, the starting-sum one
                r"(?:should|can|do)\s+i\s+(?:actually\s+|really\s+)?(?:risk|bet)\b"
                r"(?:\s+(?:then|tho|though|at\s+most|max|per\s+trade))?\W*$", re.I), (
        "Bottom line: only money you could lose in full without it changing your life — never "
        "rent, bills or an emergency fund — and on any one trade, a common rule is to lose no "
        "more than 1 to 2% of your trading money if it goes wrong.",
        "That 1 to 2% is the loss, not the position: with $1,000 and a stop 5% below your entry, a "
        "1% risk ($10) means a $200 position.",
        "Ask \"size a trade: $1,000 account, 1% risk, stop 5% below entry\" with your own numbers "
        "and the console works it out, fees included.")),
    # "omg i think i lost access to my 2fa app how do i get back into my bitget account" was
    # answered with revenge-trading advice, because "get back into" matched the only answer
    # that mentions getting money "back" (a first-time user, round 33)
    (re.compile(r"\blost\s+(?:access\s+to\s+)?(?:my\s+)?2fa\b|\b2fa\b[^?]{0,40}\b(?:lost|lose|"
                r"gone|broken|deleted|new\s+phone)\b|\b(?:get\s+back\s+into|regain\s+access\s+"
                r"to|locked\s+out\s+of)\s+(?:my\s+)?(?:bitget\s+)?account\b", re.I), (
        "Bottom line: recover it through Bitget's own account-recovery process (it will ask "
        "you to verify your identity) — this console has no access to any Bitget account and "
        "cannot reset your 2FA or log you in.",
        "On Bitget's own login page, look for its account-recovery or \"reset 2FA\" option; "
        "check Bitget's help centre for the exact steps, since they depend on what "
        "verification you already have on file and can change.",
        "Never give a 2FA code, a password or your seed phrase to anyone who contacts you "
        "offering to \"help\" recover your account — that is always a scam. Bitget support "
        "will never ask you for these.")),
    # "if i forgot my seed phrase too is my money just gone forever" was answered about a
    # stablecoin losing its peg, because "my money just gone" is also that pattern's text (a
    # first-time user, round 33)
    (re.compile(r"\b(?:forgot|forgotten|lost)\s+(?:my\s+)?seed\s*phrase\b|\bseed\s*phrase\b"
                r"[^?]{0,40}\b(?:gone\s+forever|lost\s+forever|can'?t\s+(?:get|access)|"
                r"forever)\b", re.I), (
        "Bottom line: if your coins are sitting on Bitget and were never moved to your own "
        "wallet, they are not tied to a seed phrase at all — what matters there is logging "
        "back into your account, which is Bitget's own account-recovery process.",
        "A seed phrase only matters for a self-custody wallet. If you moved coins into one of "
        "those and have lost both the phrase and the only device that wallet lives on, with "
        "no backup anywhere else, that access is gone for good — nobody, not even the "
        "wallet's maker, can restore it.",
        "If you still have the device, or a backup of the phrase written down elsewhere, you "
        "are not locked out — the risk is only losing the phrase and the device together.")),
    # "ok and whats a seed phrase, why is everyone so scared about it" (a first-time user,
    # round 33)
    (re.compile(r"\bwhat(?:'?s|\s+is)\s+(?:a\s+)?seed\s*phrase\b|\bseed\s*phrase\b[^?]{0,40}\b"
                r"(?:scared|afraid|worried|important|matter)\b|\bwhy\s+(?:is\s+)?(?:everyone"
                r"\s+)?(?:so\s+)?scared\s+(?:of|about)\s+(?:a\s+|the\s+|their\s+)?seed\s*"
                r"phrase\b", re.I), (
        "Bottom line: a seed phrase (also called a recovery phrase) is a list of usually 12 "
        "or 24 words that is the master key to a self-custody wallet — anyone who has those "
        "words can move every coin in that wallet, no password needed.",
        "That is why people are scared of it: unlike a forgotten password, there is no reset. "
        "Lose the words with no backup and the coins are gone for good; let someone else see "
        "them and they can take everything, instantly, with no way to undo it.",
        "It only applies to a wallet you hold yourself — money kept in a Bitget account does "
        "not use a seed phrase at all.")),
    # "what even is a crypto wallet lol do i need one to use bitget" was refused outright (a
    # first-time user, round 33)
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+even\s+is|\s+are)\s+(?:a\s+|an\s+)?(?:crypto\s+|"
                r"hardware\s+|software\s+)?wallets?\b|\bdo\s+i\s+(?:even\s+|really\s+)?need\s+"
                r"a\s+(?:crypto\s+)?wallet\b", re.I), (
        "Bottom line: no — a crypto wallet is not required to use Bitget. An exchange account "
        "like Bitget's holds coins for you, the way a bank holds cash, so you log in and "
        "trade without ever touching a wallet.",
        "A crypto wallet is software (or a physical device) that holds the keys controlling "
        "coins directly on the blockchain itself — this is called self-custody, and it's "
        "optional, not a requirement to buy, sell or hold on Bitget.",
        "Some people later move coins off an exchange into their own wallet for more control. "
        "That is a separate, extra step, not something you need on day one.")),
    # "im withdrawing usdt from bitget, which network should i pick erc20 or trc20 idk the
    # diff" got generic withdrawal boilerplate, never the actual difference (a first-time
    # user, round 33)
    (re.compile(r"(?=.*\berc[\s-]?20\b)(?=.*\btrc[\s-]?20\b)|\b(?:erc|trc)[\s-]?20\b[^?]{0,60}"
                r"\b(?:network|which|pick|diff\w*|idk)\b|\b(?:network|which|pick|diff\w*)\b"
                r"[^?]{0,60}\b(?:erc|trc)[\s-]?20\b", re.I), (
        "Bottom line: ERC20 and TRC20 are two different blockchain networks that USDT can "
        "move on — Ethereum (ERC20) and TRON (TRC20) — and the receiving address must be on "
        "the same network you send to, or the transfer can be lost.",
        "They are not interchangeable: an Ethereum address accepts ERC20, a TRON address "
        "accepts TRC20, and sending ERC20 USDT to a TRC20-only address (or the reverse) is "
        "the classic way to lose a transfer.",
        "Fees differ too, and change over time — Bitget shows the exact network fee for each "
        "option right on the withdrawal screen before you confirm, so check there rather than "
        "guessing.",
        "If you are not sure which network the receiving wallet or exchange actually "
        "supports, ask it directly before you pick one here — matching networks is the one "
        "thing that matters most.")),
    # "oh no i think i sent my usdt using the wrong network to my other wallet, is it gone"
    # was refused outright (a first-time user, round 33)
    (re.compile(r"\bsent\s+(?:my\s+)?(?:usdt|usdc|crypto|coins?|btc|eth)\s+(?:using\s+|on\s+)?"
                r"(?:the\s+)?wrong\s+network\b|\bwrong\s+network\b[^?]{0,40}\b(?:sent|send|"
                r"gone|lost)\b", re.I), (
        "Bottom line: not always gone — it depends on whether the address you sent to also "
        "exists and is usable on the network you actually sent through. Sometimes it can be "
        "recovered; sometimes it genuinely cannot.",
        "Find the transaction hash (Bitget's withdrawal history has it) and contact the "
        "support team of the platform or wallet that received it — not Bitget's, since the "
        "funds already left Bitget — and give them the hash and both networks involved.",
        "Being honest about the odds: recovery is not guaranteed, it can take time, and the "
        "receiving platform may charge a fee or simply not be able to help, depending on how "
        "its wallet is built. Check Bitget's help centre for its own guidance on this, since "
        "the exact process can differ by coin and network.")),
    # "whats an etf, is bitcoin an etf" got a 20-line market data dump instead of a definition
    # (a first-time user, round 33)
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+are)\s+(?:an?\s+)?etfs?\b|\bis\s+bitcoin\s+an?\s+etf\b"
                r"|\bis\s+btc\s+an?\s+etf\b|"
                # "someone told me to just buy an etf instead of coins, what even is that"
                # named the term earlier in the sentence and asked about it referentially, not
                # right next to "what's" (round 36)
                r"\betfs?\b[^?]{0,60}\bwhat(?:'?s|\s+is|\s+even\s+is)\s+that\b", re.I), (
        "Bottom line: an ETF (exchange-traded fund) is a fund that holds a basket of assets "
        "(stocks, bonds, gold, or something else) and itself trades on a stock exchange like "
        "a share, so you buy one thing and get exposure to everything inside it.",
        "Bitcoin itself is not an ETF — it is a coin, traded directly on exchanges like "
        "Bitget. What does exist are spot bitcoin ETFs: funds that hold actual bitcoin and "
        "trade on traditional stock exchanges, a separate product from holding bitcoin "
        "directly.",
        "Buying bitcoin on Bitget and buying a spot bitcoin ETF through a stockbroker both "
        "end up tracking bitcoin's price, but through very different accounts, fees and "
        "ownership — the ETF holder owns fund shares, not the coin.")),
    # "whats a perpetual, ppl keep saying perp this perp that" has no bare "define perpetual"
    # pattern anywhere in this file — only "what is a perp" (the console's own suggested
    # phrasing, used elsewhere in this file) worked; a first-time user's own wording of the
    # identical question was refused (round 36)
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+are|\s+even\s+is)\s+(?:a\s+|an?\s+)?perpetuals?\b|"
                r"\bperps?\s+this\s+perps?\s+that\b|\b(?:everyone|people|ppl)\s+(?:keep\s+)?"
                r"(?:saying|says|say|talking\s+about)\s+perps?\b", re.I), (
        "Bottom line: a perpetual (\"perp\") is a futures contract with no expiry date — unlike "
        "an ordinary futures contract, which settles on a fixed date, a perpetual just keeps "
        "trading, so it needs its own way to track the real price.",
        "That mechanism is funding: a small payment every few hours between everyone long and "
        "everyone short the contract, which pulls the perpetual's price back toward the spot "
        "price instead of letting it drift away.",
        "It also adds leverage, which spot buying does not: you can trade a larger position "
        "than the cash you put up, and a large enough move against you closes (liquidates) "
        "it. Ask \"what is liquidation\" next.")),
    # "theres this coin at $0.0001 thats so cheap right i can buy millions of them" was read
    # as the stock ticker COIN (Coinbase), even when named=True (a first-time user, round 33)
    (re.compile(r"\$?0\.0+\d+\b[^?]{0,60}\bcheap\b|\bcheap\b[^?]{0,60}\$?0\.0+\d+\b|\$0\.0+\d+"
                r"\b[^?]{0,60}\bbuy\s+(?:millions?|thousands?|lots?|a\s+ton)\b", re.I), (
        "Bottom line: a low unit price does not mean a coin is cheap — what matters is market "
        "cap (price times how many coins exist) and total supply, not the price of one coin.",
        "A coin at $0.0001 with trillions of coins in existence can be worth more overall than "
        "a coin at $50,000 with 19 million coins — BTC looks \"expensive\" per coin only "
        "because so few of them exist, not because it is a better deal.",
        "Buying \"millions\" of a $0.0001 coin costs the same $100 as buying a tiny fraction "
        "of an expensive one — neither is cheaper in what you actually get for your money.",
        "Ask for a coin's market cap by name, not its price alone, to see what it is really "
        "worth.")),
    # "ok so whats actually safer than crypto for retirement money then" repeated Q13's
    # answer almost word for word and never named an alternative (a first-time user, round 33)
    (re.compile(r"\bsafer\s+than\s+crypto\s+for\s+retirement\b|\bwhat(?:'?s|\s+is)\s+"
                r"(?:actually\s+)?safer\s+(?:than\s+crypto\s+)?for\s+retirement\b|\bsafest\s+"
                r"(?:place|option|thing)\s+for\s+retirement\s+(?:money|savings|funds)\b",
                re.I), (
        "Bottom line: no pick here, but the things usually called \"safer\" for money someone "
        "will need to live on are: government bonds or T-bills (a government's own IOU, among "
        "the lowest-risk assets that exist), broad stock-market index funds (hundreds of "
        "companies at once, so no single company's failure wipes it out), and savings "
        "accounts protected by a government deposit-insurance scheme up to a set limit.",
        "None of those is risk-free either — bonds lose value if sold before maturity when "
        "rates rise, and index funds can still fall 30-50% in a bad year — they are just far "
        "less volatile than a single coin or a leveraged crypto position.",
        "Which mix actually suits your parents is a question for a licensed financial adviser "
        "who knows their full situation and how soon they need the money — not something "
        "this console can judge.")),
    # "how is that different than just buying once with all my money at the start" repeated
    # the DCA definition it had just given instead of answering the follow-up (a first-time
    # user, round 33)
    (re.compile(r"\bhow\s+is\s+that\s+different\s+(?:than|from)\s+(?:just\s+)?buying\s+once\s+"
                r"(?:with\s+)?(?:all\s+)?(?:my\s+)?money\s+at\s+the\s+start\b|\bdca\b[^?]{0,40}"
                r"\blump\s+sum\b|\blump\s+sum\b[^?]{0,40}\bdca\b", re.I), (
        "Bottom line: buying it all at once (a lump sum) puts the whole amount to work from "
        "day one; dollar-cost averaging (DCA) spreads that same money across several "
        "purchases over time instead of committing it all on one day's price.",
        "The trade-off: if the price rises afterward, the lump sum does better because all "
        "the money was already in; if it falls first, DCA does better because later "
        "purchases buy in cheaper. In markets that trend upward most of the time, the lump "
        "sum has usually won slightly more often, on average.",
        "What DCA actually buys you is not a better average price — it is less regret: you "
        "are never the person who put everything in on the single worst possible day, "
        "because no one day held all of it.")),
    # "whats DCA i keep seeing it everywhere" has no pattern at all for a bare definition — only
    # the lump-sum-comparison form above is covered, and the bare term reads as an unknown
    # ticker instead (round 36; reproduced identically in Hinglish: "...usko kya bolte hain, DCA
    # wagera?")
    (re.compile(r"\bwhat(?:'?s|\s+is|\s+does)\s+dca\b(?![^?]{0,40}\blump\s+sum\b)|\bdca\b[^?]"
                r"{0,20}\b(?:wagera|waghera|mean|means|stand\s+for|or\s+something|is\s+that)\b|"
                r"\bi\s+keep\s+seeing\s+dca\b", re.I), (
        "Bottom line: DCA stands for dollar-cost averaging — putting in a fixed amount at "
        "regular intervals (say, $50 every week) instead of all at once.",
        "It does not beat a lump sum on average in a market that mostly trends upward, but it "
        "means no single day's price decides the whole amount, which is why people nervous "
        "about timing use it.",
        "It is not a Bitget product or a coin — it is a buying habit, done by placing the "
        "same order on a schedule.")),
    # "idk man just tell me yes or no" pushed for a binary call after a timing question got the
    # generic refusal, and got nothing at all — not even a restatement of why (round 36)
    (re.compile(r"\b(?:just\s+)?(?:tell\s+me|give\s+me|say)\s+(?:a\s+)?yes\s+or\s+no\b|\byes\s+"
                r"or\s+no\s*[?!.]*\s*$", re.I), (
        "Bottom line: no yes or no here — this console will not give a flat call on whether to "
        "buy, sell or trade; no honest source can, and one that claims to is guessing or "
        "selling something.",
        "What it gives instead: what has actually happened in similar past moments for a real "
        "name — ask \"has BTC been here before\" for how often a setup like today's rose or "
        "fell afterward, and by how much. That is a base rate, not a prediction.",
        "The decision stays yours either way; this just gives you something real to decide "
        "with.")),
    # (voice run-on) "...or should i split it into different coins to be safe" was ignored —
    # only the lump-sum-risk half of the question got answered (a first-time user, round 33)
    (re.compile(r"\bsplit\w*\s+(?:it\s+|the\s+money\s+|my\s+money\s+)?(?:up\s+)?(?:into|"
                r"between|across|among)\s+(?:\d+\s+)?(?:different\s+)?coins?\b|\bspread\w*\s+"
                r"(?:it\s+)?(?:across|between|among)\s+(?:different\s+)?coins?\b|"
                r"\bdiversify\w*\s+(?:across|between|among|into)\s+(?:different\s+)?coins?\b",
                re.I), (
        "Bottom line: splitting money across several coins helps less than it sounds like it "
        "should — most crypto coins rise and fall together, heavily tied to what bitcoin "
        "does, so a bad day for bitcoin is usually a bad day for the rest of the portfolio "
        "too, just by different amounts.",
        "That is different from spreading money across unrelated things (stocks and bonds, "
        "say, which often move in opposite directions) — inside crypto alone, the coins "
        "mostly move the same way at once, so splitting mainly changes how much you lose or "
        "gain, not whether you do.",
        "It still removes the risk of one project failing outright — a hack, a scam, a coin "
        "going to zero specifically — that part is real. It just does not protect against "
        "crypto itself having a bad week, the way holding something outside crypto entirely "
        "would.")),
    # "is it better to keep my coins on bitget or move them somewhere else" matched nothing —
    # the same custody answer "is it safe to keep crypto on bitget" already gets, needed
    # without relying on the language model (a first-time user, round 33)
    (re.compile(r"\bkeep\s+(?:my\s+)?(?:coins?|crypto|money|funds?)\s+on\s+bitget\b[^?]{0,40}"
                r"\bmove\s+(?:them|it)\s+(?:somewhere\s+else|elsewhere|off(?:\s+(?:of\s+)?"
                r"(?:it|bitget))?|to\s+(?:a\s+|my\s+own\s+)?wallet)\b", re.I), _BITGET_SAFE_A),
    # "so should i move it to a wallet instead then" was a bare follow-up with no "keep on
    # bitget" of its own in the same message, and read as an unknown ticker lookup for the word
    # "wallet" instead (round 36)
    (re.compile(r"\bshould\s+i\s+move\s+(?:it|them|everything|my\s+(?:coins?|crypto|money|"
                r"funds?))\s+(?:to\s+(?:a\s+|my\s+own\s+)?wallet\b|somewhere\s+else\b|"
                r"elsewhere\b|off\s+(?:of\s+)?(?:it|bitget)\b)", re.I), _BITGET_SAFE_A),
    # "what time does the crypto market close like what time should i trade" was answered
    # around the desk's own session state, never plainly "crypto never closes" (a first-time
    # user, round 33)
    (re.compile(r"\bwhat\s+time\s+does\s+(?:the\s+)?crypto\s+market\s+close\b|\bwhen\s+does\s+"
                r"(?:the\s+)?crypto\s+market\s+close\b|\bdoes\s+(?:the\s+)?crypto\s+market\s+"
                r"(?:ever\s+)?close\b|\bwhat\s+time\s+(?:should|do)\s+i\s+trade\s+crypto\b|"
                r"\bcrypto\s+market\s+(?:hours|close\s+time|closing\s+time)\b", re.I), (
        "Bottom line: crypto never closes — it trades 24 hours a day, 7 days a week, "
        "including weekends and holidays, because it is not tied to any country's stock "
        "exchange hours.",
        "There is no \"best time\" set by the market being open or shut, the way there is "
        "for stocks. Volume (how much is trading) rises and falls through the day and the "
        "week, which can affect the spread, but the market itself is always open.",
        "Bitget's US-stock perpetual contracts also trade around the clock, 24/7 — unlike "
        "the actual stock (which only trades during its own exchange's hours), the "
        "perpetual on it keeps trading even while that stock market is shut, and can gap "
        "when the real market reopens.")),
    # "whats the difference between a coin and a token" was refused outright (a first-time
    # user, round 33)
    (re.compile(r"\b(?:difference|diff)\s+between\s+(?:a\s+)?coins?\s+and\s+(?:a\s+)?tokens?\b"
                r"|\bcoins?\s+(?:vs\.?|versus)\s+tokens?\b", re.I), (
        "Bottom line: a coin runs on its own blockchain (Bitcoin, Ether, Solana) and that "
        "blockchain's own network validates and processes it; a token is built on top of "
        "someone else's blockchain (most USDT and thousands of others run on Ethereum or "
        "similar chains) using that chain's own rules.",
        "A coin pays its network's own transaction (gas) fees; a token's fees are paid in "
        "the coin of whichever blockchain it is built on, not in the token itself — sending "
        "USDT on Ethereum, for example, costs ETH in gas, not USDT.",
        "Day to day the difference rarely matters for buying and holding on Bitget — both "
        "are just assets you can trade — but it decides which network a withdrawal uses, "
        "which is why Bitget asks you to pick one (see \"erc20 vs trc20\", for example).")),
    # "if i buy bitcoin on bitget do i actually own it or does bitget just owe me" got the
    # desk's volatility stats and never touched ownership (a first-time user, round 33)
    (re.compile(r"\bdo\s+i\s+(?:actually\s+)?own\s+it\b[^?]{0,40}\b(?:bitget\s+)?owe\s+me\b|"
                r"\bown\s+(?:it|the\s+coin|the\s+crypto|bitcoin)\s+or\s+(?:does\s+)?bitget\s+"
                r"(?:just\s+)?owe\s+me\b|\bdoes\s+bitget\s+(?:just\s+)?owe\s+me\b", re.I), (
        "Bottom line: on Bitget, you hold a claim on Bitget for that coin, not the coin "
        "itself sitting separately with your name on it — Bitget owes it to you and is "
        "expected to deliver it on request, the way a bank owes you the cash in your "
        "account.",
        "What backs that claim: Bitget publishes proof-of-reserves reports on its own site "
        "showing it holds enough to cover what it owes depositors — read those for yourself "
        "rather than taking it on faith.",
        "The only way to hold the coin itself, with nobody owing you anything, is to "
        "withdraw it to a self-custody wallet that only you control — at the cost of then "
        "being responsible for its own seed phrase and security.")),
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
                # "explain candle wicks to me like im new" was declined (a live re-ask, round 32)
                r"\b|\b(?:candle(?:stick)?\s+)?wicks?\b[^?]{0,40}\b(?:mean|like\s+i'?m|explain\w*|"
                r"new)\b|\b(?:explain|what\s+do)\b[^?]{0,20}\b(?:candle(?:stick)?\s+)?wicks?\b",
                re.I), (
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
    # "explain RSI like im 12 years old" is the explain-it-simply idiom, not an age: it got this
    # refusal, which also said "16" whatever age was typed (round 37 newcomer, C1)
    (re.compile(r"(?<!like\s)(?<!if\s)(?<!pretend\s)(?<!imagine\s)\bi(?:'?m|\s+am)\s+(?:only\s+)?"
                r"(?:1[0-7]|thirteen|fourteen|fifteen|sixteen|seventeen)\b(?!\s*(?:%|x\b|k\b|"
                r"thousand|years?\s+(?:into|of\s+trading)))", re.I), (
        "Bottom line: Bitget's terms require anyone using it to be at least 18 (section 2.2 of its "
        "terms of use), so under 18 you cannot open an account there, with any amount.",
        "Learning costs nothing: ask this console anything — \"how much could I lose on BTC "
        "in a bad week with $200\" shows what a sum would have been through — and keep the "
        "money until you can decide for yourself, without leverage.",)),
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
    # Round 34, a first-time user: a DM airdrop-claim link, "is airdrops real", finding legit
    # ones, a 20% stablecoin APY, Bitget convert vs trade, checking a coin before buying,
    # presale risk, slang follow-ups, a conceptual gambling question, a third-person age
    # question, NFTs and a recession — each refused or answered a different question entirely.
    # "someone DMed me a link to claim an airdrop should i connect my wallet to it" was refused
    # outright — the single most dangerous miss in the run (a first-time user, round 34)
    (re.compile(r"\b(?:dm'?d|dmed|dm\w*)\b[^?]{0,40}\blink\b[^?]{0,40}\bairdrop\b|\bairdrop\b"
                r"[^?]{0,40}\blink\b[^?]{0,40}\b(?:dm'?d|dmed|dm\w*)\b|\bconnect\s+(?:my\s+)?"
                r"wallet\b[^?]{0,60}\bairdrop\b|\bairdrop\b[^?]{0,60}\bconnect\s+(?:my\s+)?"
                r"wallet\b", re.I), (
        "Bottom line: no — never connect your wallet or sign anything from a link in a DM, "
        "even one about a real-sounding airdrop. That is how a \"wallet drainer\" works: the "
        "site has you connect, then sign something that looks routine, and that signature is "
        "what hands it permission to move everything in the wallet — no seed phrase needed.",
        "A claim link sent to you in a DM is close to always this scam, on every platform — "
        "Telegram, Discord, X, Instagram, it makes no difference — because a real airdrop "
        "never has to find you first. It is announced on the project's own site and its own "
        "verified account, which you can go check yourself, without the link.",
        "To check a real one: go to the project's own official channel directly, not through "
        "the message, and claim with a wallet that holds little or nothing — never your main "
        "one.")),
    # "yo is airdrops real?... sounds like a scam ngl" was refused (a first-time user, round 34)
    (re.compile(r"\bis\s+(?:an?\s+)?airdrops?\s+(?:even\s+|actually\s+)?real\b|\bare\s+"
                r"airdrops?\s+(?:even\s+|actually\s+)?real\b|\bairdrops?\b[^?]{0,30}\b(?:sounds?"
                r"\s+like\s+a\s+scam|a\s+scam)\b", re.I), (
        "Bottom line: yes, real airdrops exist — a project sends free tokens to wallets that "
        "meet some rule (held a coin, used an app, were on a list) to spread ownership and get "
        "attention. \"Is this real\" and \"is this specific one real\" are different "
        "questions, though: the space is also full of fake ones.",
        "The tell for a fake one: it asks you to connect a wallet to an unknown site, sign "
        "something, or pay a \"gas fee\" upfront to claim — a real airdrop never needs money "
        "or a wallet signature from you to hand over free tokens.",
        "This console cannot tell you whether one specific airdrop is real; it has no list of "
        "them. What it can say is which checks catch the fake ones.")),
    # "ok so if its real how do i find legit ones without getting scammed" was refused (a
    # first-time user, round 34)
    (re.compile(r"\bairdrops?\b[^?]{0,80}\b(?:find|spot|tell)\s+legit\s+ones\b[^?]{0,30}"
                r"\bscamm?ed\b|\b(?:find|spot|tell)\s+legit\s+airdrops?\b[^?]{0,30}"
                r"\bscamm?ed\b", re.I), (
        "Bottom line: go straight to the project's own official site and its own verified "
        "account for the announcement — never through a link someone sent you — and treat "
        "anything that disagrees with that as fake.",
        "A legit airdrop never asks for a fee, for you to send crypto first, or to connect a "
        "wallet holding real money just to \"qualify\" — those three asks are the scam, every "
        "time, and no real one ever needs your seed phrase or private key.",
        "Claim with a wallet that holds little or nothing, and treat a coin trending only on "
        "social media, with nothing on a major exchange, as unverified.")),
    # "whats a stablecoin yield and why do some apps offer 20% apy on usdt is that legit" and
    # "how do i tell if an apy is too good to be true" were both refused (a first-time user,
    # round 34)
    (re.compile(r"\bstable\s*coin\s+yield\b|\bapy\s+(?:too\s+good\s+to\s+be\s+true|legit)\b|"
                r"\bhow\s+do\s+i\s+tell\s+if\s+(?:an?\s+)?apy\s+is\s+too\s+good\s+to\s+be\s+"
                r"true\b|\b\d{1,3}\s*%\s*apy\b[^?]{0,30}\b(?:legit|real|safe|catch)\b", re.I), (
        "Bottom line: a stablecoin yield pays you for lending your USDT or USDC to borrowers, "
        "or for supplying it to a trading pool — real yield comes from real borrowing demand "
        "or trading fees, usually low single digits up to around 8-10% on reputable platforms. "
        "A steady 20% APY, every month, whatever the market does, is not normal.",
        "Where a number that high usually comes from: a separate token the platform prints to "
        "attract deposits (which that token's own price then has to support), or the platform "
        "simply not having the money to pay withdrawals later — both can look real for months "
        "before they stop.",
        "Before trusting any yield: can you withdraw a small amount back out right now, not "
        "just watch a balance grow; are the platform's reserves public; would the yield "
        "survive no new deposits tomorrow. A guaranteed, fixed return is the red flag on its "
        "own.")),
    # "whats the difference between convert and trade on the bitget app im confused" was
    # refused (a first-time user, round 34)
    (re.compile(r"\bdifference\s+between\s+convert\s+and\s+trade\b|\bconvert\s+(?:vs\.?|"
                r"versus|or)\s+trade\b|\btrade\s+(?:vs\.?|versus|or)\s+convert\b", re.I), (
        "Bottom line: on Bitget, Convert is a quick swap — you're shown a price for swapping "
        "one coin for another right now and you take it or leave it, with no order book and "
        "no choice of order type; Trade is the full market, with an order book and your "
        "choice of a market or limit order.",
        "Convert suits a small, one-off swap because nothing needs setting up first, but its "
        "quoted rate can sit a little worse than what the order book actually shows, since "
        "that convenience is priced in.",
        "A limit order (set your own price; it may not fill) versus a market order (fills now "
        "at the best price on offer) is a Trade-only choice, and it shows the live spread and "
        "depth that Convert hides. For the exact fee gap, check Bitget's own help pages.")),
    # "how do i even know if a coin is legit before buying it" was refused — distinct from
    # checking a service or a promised return (a first-time user, round 34)
    (re.compile(r"\bhow\s+do\s+i\s+(?:even\s+|actually\s+)?know\s+if\s+(?:a\s+|the\s+)?coin\s+"
                r"is\s+legit\b|\bis\s+(?:this|that)\s+coin\s+legit\b|\bhow\s+(?:can|do)\s+i\s+"
                r"(?:know|tell)\s+(?:if\s+)?a\s+coin\s+is(?:n'?t)?\s+a\s+scam\b", re.I), (
        "Bottom line: no single check proves a coin is legit, but these narrow it down fast: "
        "a real, named team with a track record (not anonymous); an independent smart-contract "
        "audit from a known firm; real trading volume (not a thin order book one trade can "
        "move); and whether a handful of wallets hold most of the supply, which a blockchain "
        "explorer shows for free.",
        "Also check: is it actually listed on a major exchange such as Bitget, Binance or "
        "Coinbase (each sets its own listing bar — not a guarantee, but a filter), and is a "
        "large token unlock for the team or early investors due soon, since that can put a "
        "lot of new supply on the market at once.",
        "Red flags on their own: a promised or guaranteed return, pressure to buy now or "
        "recruit others, and a contract whose owner can change fees or block selling (a "
        "honeypot). None of this is a buy signal either way — it only rules out the obvious "
        "scams.")),
    # "my friend made bank on a new coin presale launch should i join presales too" got the
    # generic shitcoin answer, missing what makes a presale distinct (a first-time user, round
    # 34)
    (re.compile(r"\b(?:new\s+)?(?:coin\s+)?presales?\b[^?]{0,40}\b(?:join|do|get\s+in|buy\s+"
                r"(?:into|in))\b|\bshould\s+i\s+(?:join|do|get\s+into)\s+presales?\b", re.I), (
        "Bottom line: a presale is riskier than buying an already-listed coin — there is no "
        "token trading yet, no price chart, and no way to sell if you change your mind. You "
        "hand over money on a promise, before there is anything to check.",
        "What makes it different from an ordinary risky coin: the team can vanish with the "
        "money before any token or liquidity exists at all (an \"exit scam\"), and even a real "
        "team usually vests (locks) its own and early buyers' tokens, so buyers at listing can "
        "be selling into that unlock for months.",
        "A friend making money on one presale is one data point, not a pattern — most "
        "presales list far below the presale price, or never list at all. If you try one "
        "anyway, use only money you could lose entirely.")),
    # "wait so if i fomo into a coin and it dumps is that the same as getting rekt or
    # different" was refused despite REKT having just been defined in-thread (a first-time
    # user, round 34)
    (re.compile(r"(?=.*\bfomo\w*\b)(?=.*\brekt\b)(?=.*\b(?:same\s+as|or\s+different|different"
                r"\s+(?:from|than))\b)", re.I), (
        "Bottom line: related, not identical. FOMO is the decision — buying because a price is "
        "rising and you don't want to miss it; getting rekt is the outcome — a large loss. "
        "FOMOing into a coin that then dumps is one of the most common ways to get rekt, but "
        "not the only one.",
        "You can also get rekt with no FOMO at all (a leveraged position liquidated by an "
        "ordinary move), and you can FOMO in and get lucky. Think of FOMO as a cause and rekt "
        "as one possible effect, not two words for the same thing.")),
    # "ok so if i get rekt from fomoing into a coin can i like get my money bak somehow or its
    # gone forever lol" was refused, same boilerplate (a first-time user, round 34)
    (re.compile(r"(?=.*\brekt\b)(?=.*\b(?:get\s+(?:my\s+)?money\s+(?:bak|back)|money\s+back|"
                r"gone\s+forever)\b)", re.I), (
        "Bottom line: gone — once a coin is sold at a loss, or a leveraged position is "
        "liquidated, there is no chargeback, no customer-service reversal, and no \"undo\" "
        "the way a bank transfer sometimes has. Crypto transactions do not get reversed.",
        "The one exception: if you are still holding — not sold, not liquidated — and the "
        "coin recovers, the loss was only on paper and can shrink or disappear. Once it is "
        "sold, or you are liquidated, that outcome is locked in.",
        "There is no legitimate way to \"recover\" crypto lost this way — anyone who offers to "
        "get it back for a fee is a second scam stacked on the first.")),
    # "is there a way to trade without it basically being gambling" got an unsolicited $100k
    # hedge pitch the user never asked for (a first-time user, round 34)
    (re.compile(r"\bway\s+to\s+trade\s+without\s+it\s+(?:basically\s+)?being\s+gambling\b|"
                r"\btrade\s+without\s+(?:it\s+(?:basically\s+)?being\s+|being\s+)?gambling\b|"
                r"\b(?:not\s+(?:make|turn)|avoid\s+(?:making|turning))\s+(?:it|trading)\s+"
                r"(?:into\s+)?(?:basically\s+)?gambling\b", re.I), (
        "Bottom line: yes — what separates trading from gambling is a reason to enter that has "
        "worked before (not a hunch), a plan for being wrong decided before you enter, and a "
        "size small enough that one bad trade doesn't end the account. Gambling is a bet on "
        "pure chance with no edge and no plan.",
        "A coin flip and buying because a chart looks exciting are both gambling either way. "
        "Checking what actually followed similar past moments (ask \"has BTC been here "
        "before\" for a real base rate), knowing the cost (fees eat any edge — a Bitget "
        "perpetual round trip is about 0.12%), and sizing for survival is what keeps it "
        "analysis rather than a bet.",
        "This console will not size a position or pitch a hedge unless you give it a real "
        "position to size — a plain question like this gets a plain answer, not a trade.")),
    # "should a 16 year old be trading this stuff" repeated the whale definition verbatim
    # instead (a first-time user, round 34)
    (re.compile(r"\b(?:should|can|could)\s+a\s+1[0-7][\s-]?year[\s-]?old\s+(?:be\s+)?trad(?:e|"
                r"ing)\b|\bis\s+(?:it\s+)?(?:ok|okay|fine|legal)\s+for\s+a\s+1[0-7][\s-]?year"
                r"[\s-]?old\s+to\s+trade\b", re.I), (
        "Bottom line: not with an account of their own — Bitget's terms require anyone using "
        "it to be at least 18 (section 2.2), so a 16-year-old cannot open a Bitget account, "
        "with any amount.",
        "Whether Bitget offers any custodial or parental option for a minor is worth checking "
        "on its own help centre; this console has not verified that either way.",
        "Learning costs nothing at any age: ask this console anything, e.g. \"how much could "
        "I lose on BTC in a bad week\" — the money can wait until they can decide for "
        "themselves, without leverage.")),
    # "what is an nft and should i buy one" was hit by the hourly quota wall with zero plain-
    # English fallback (a first-time user, round 34)
    (re.compile(r"\bwhat\s+(?:is|are)\s+(?:an?\s+)?nfts?\b|\bwhats?\s+(?:is|are)?\s*(?:an?\s+)?"
                r"nfts?\b", re.I), (
        "Bottom line: an NFT (non-fungible token) is a token on a blockchain that stands for "
        "ownership of one specific, unique thing — art, a collectible, an in-game item. Unlike "
        "a coin, no two NFTs are interchangeable; each one is its own record.",
        "Whether to buy one isn't this console's call, but the facts: most NFTs are worth far "
        "less than at their 2021 peak, many trade with little or no active market at all (hard "
        "to sell quickly even if you want to), and the price is driven by hype and scarcity, "
        "not any earnings behind it.",
        "If you buy one, treat it like a collectible you would be fine owning even if its "
        "price never recovers — not something you are counting on to sell higher later.")),
    # "if nfts crash too then whats even the point of buying one" was hit by the same quota
    # wall (a first-time user, round 34)
    (re.compile(r"\bnfts?\b[^?]{0,60}\bpoint\s+(?:of|in)\s+buying\s+(?:an?\s+nfts?\b|one\b)|"
                r"\bwhats?\s+(?:even\s+)?the\s+point\s+(?:of|in)\s+buying\s+(?:an?\s+)?nfts?\b|"
                r"\bpoint\s+(?:of|in)\s+buying\s+(?:an?\s+)?nfts?\b", re.I), (
        "Bottom line: for most buyers the honest point is not profit — it's owning something "
        "you actually like, a community or access that comes with holding it, or a piece of "
        "digital art — the same reason someone buys a print that might never be worth more "
        "than they paid.",
        "As an investment specifically it's a hard case: on top of the price risk, many NFTs "
        "have little to no active market, so wanting to sell doesn't mean you can, at any "
        "price.",
        "Treat \"it might go up\" as a bonus, not the reason — if resale hope is the only "
        "reason to buy, that is the same gamble as any other speculative asset, with worse "
        "liquidity.")),
    # "what happens to crypto if theres a recession does it crash too" was hit by the same
    # quota wall (a first-time user, round 34)
    (re.compile(r"\bcrypto\b[^?]{0,40}\brecession\b|\brecession\b[^?]{0,40}\bcrypto\b", re.I), (
        "Bottom line: crypto has mostly moved with the stock market, not against it — in the "
        "2022 downturn (rising interest rates, recession fears) BTC fell about 65% from its "
        "high, roughly in step with high-growth tech stocks, not acting as a safe haven.",
        "The \"uncorrelated\" or \"digital gold\" story gets tested in exactly these moments, "
        "and it has not held up consistently so far: when investors get cautious and sell risk "
        "assets broadly, crypto has usually been sold alongside them, often more sharply "
        "because of leverage in the system.",
        "That is a pattern from the recessions crypto has actually lived through, not a law — "
        "there is no guarantee the next one plays out the same way.")),
    # Round 35, a first-time user: an emotional loss message, a bear market, stop-loss and
    # take-profit together, a Bitget-vs-Google price mismatch and its follow-up, reading the 24h
    # change, a bare "fees?", a Hinglish leverage question, a friend's all-in tip, Earn vs
    # Futures, explaining bitcoin plainly, funding rate "for me", taking profit on a 50% pump,
    # sell regret, and Bitget vs a wallet — each refused or answered a different question.
    # "i lost half my savings on a trade last week i feel sick rn ngl" was refused outright, the
    # most serious miss in that run (a first-time user, round 35)
    (re.compile(r"\b(?:lost|lose|down)\b[^?]{0,60}\b(?:savings|money|my\s+(?:whole|entire)?\s*"
                r"(?:savings|money|account|portfolio))\b[^?]{0,80}\b(?:feel\s+sick|feel\s+"
                r"(?:terrible|awful|horrible)|panick\w*|can'?t\s+sleep|freaking\s+out|feel\s+"
                r"like\s+(?:crying|dying)|feel\s+so\s+(?:stupid|dumb))\b", re.I), (
        "Bottom line: that is a real loss, and feeling sick about it is a normal reaction, not "
        "a weakness — the most useful thing right now is to stop trading, not to fix it "
        "tonight.",
        "Don't try to win it back straight away: trading bigger, right after a big loss, is how "
        "a bad week turns into a bad year.",
        "Step away from the screen and talk to someone you trust about it. Only look again at "
        "what happened — the size, the leverage, what the plan was — once you're calm, not "
        "while you still feel like this.")),
    # "wats a bear market and how long do they usually last" was refused — a textbook glossary
    # question (a first-time user, round 35)
    # only the definition: "how did BTC do in the 2022 bear market?" and "is NVDA in a bear
    # market right now?" are measured questions, and got this definition (round 35 re-check)
    (re.compile(r"\b(?:what(?:'?s|\s+is|s)?|wat(?:'?s|s)?|wats|define|meaning\s+of|explain)\s+"
                r"(?:a\s+|an\s+|the\s+)?bear\s+markets?\b|\bbear\s+markets?\b[^?]{0,30}\bhow\s+"
                r"long\b|\bhow\s+long\s+(?:do|does|will|did)\s+(?:a\s+)?bear\s+markets?\b", re.I), (
        "Bottom line: a bear market is a sustained fall in prices, commonly defined as about a "
        "20% drop from a recent high. How long they usually last varies, but crypto bear "
        "markets have historically run roughly a year or more before recovering, sometimes "
        "longer.",
        "The 20% figure is a common convention, not an official rule. Its opposite, a bull "
        "market, is a sustained rise; no length is guaranteed, and not everything that falls "
        "comes all the way back.")),
    # "wat is a stop loss and take profit in simple words" was refused — the single most basic
    # order-type question a newcomer can ask (a first-time user, round 35)
    (re.compile(r"(?=.*\bstop[\s-]*loss(?:es)?\b)(?=.*\btake[\s-]*profits?\b)", re.I), (
        "Bottom line: a stop-loss automatically closes your position if the price falls to a "
        "level you choose, to cap a loss; a take-profit automatically closes it if the price "
        "rises to a level you choose, to lock in a gain. Both just sit there until the price "
        "reaches them.",
        "In general terms: you set the stop-loss below what you paid, by however much you are "
        "willing to lose, and the take-profit above it, by however much gain you would be "
        "happy to walk away with.",
        "The exact screen and labels for setting one can change — check Bitget's help centre, "
        "or the order form itself, for the current steps.")),
    # "what if price gaps past my stop loss overnight while im asleep does it still protect me"
    # was answered with an unrelated hedge-sizing pitch (a first-time user, round 35)
    (re.compile(r"\b(?:gaps?|jumps?)\s+past\s+(?:my\s+)?stop[\s-]*loss\b|\bstop[\s-]*loss\b"
                r"[^?]{0,40}\b(?:gap(?:s|ped)?|overnight|while\s+(?:i'?m\s+)?asleep)\b|\bdoes\s+"
                r"(?:a\s+|my\s+)?stop[\s-]*loss\s+(?:still\s+)?(?:protect|guarantee)\b", re.I), (
        "Bottom line: not always — a stop-loss triggers an order once the price reaches your "
        "level, but it does not guarantee the fill happens at exactly that price. If the price "
        "gaps (jumps) past it with no trading in between, such as overnight, the order can fill "
        "lower than you set it.",
        "Crypto trades 24/7 so there is no close like a stock market has, but big news can "
        "still move the price fast enough to jump past a level while you are asleep.",
        "A stop-loss limits the damage; it does not eliminate it. Check Bitget's help centre "
        "for whether a guaranteed-stop option exists on the contract you are using.")),
    # "why is the btc price on bitget diff from wat google shows me rn" read "google" as the
    # stock ticker GOOGL and quoted Alphabet's share price next to BTC's (a first-time user,
    # round 35) — answered here before any ticker lookup runs, so it holds even when a ticker is
    # also detected in the text (named=True)
    (re.compile(r"\b(?:price|btc|bitcoin|eth|ethereum)\b[^?]{0,60}\bgoogle\b(?!\s+(?:stock|"
                r"shares?|inc\b))[^?]{0,40}\b(?:shows?|says?)\b|\bgoogle\b(?!\s+(?:stock|"
                r"shares?|inc\b))[^?]{0,40}\b(?:shows?|says?)\b[^?]{0,40}\b(?:price|btc|"
                r"bitcoin|eth|ethereum)\b|\bwhy\s+is\s+(?:the\s+)?(?:btc|bitcoin|eth|ethereum)"
                r"\s+price\b[^?]{0,60}\bgoogle\b(?!\s+(?:stock|shares?|inc\b))", re.I), (
        "Bottom line: Bitget and Google are not quoting the same thing. Google shows a price "
        "from its own data partner, on its own small delay; Bitget shows its own live order "
        "book. Different source, different instant in time — a small gap is normal, not a bug.",
        "Two more differences to watch for: USD vs USDT (Bitget's crypto pairs usually quote in "
        "USDT, a dollar-pegged stablecoin that tracks $1 but is not literally a dollar) and "
        "spot vs perpetual (a perpetual futures contract can trade a little above or below the "
        "spot price, especially when one side is crowded).",
        "None of this makes one price \"wrong\" — each is correct for what it is quoting.")),
    # "ok so which one is the real price then" was read as an unknown ticker symbol, dropping a
    # natural follow-up to a price-mismatch question (a first-time user, round 35) — answered
    # here before any ticker lookup runs, so it holds even when named=True
    (re.compile(r"\bwhich\s+(?:one\s+)?(?:is|'?s)\s+(?:the\s+)?real\s+price\b|\bwhats?\s+the\s+"
                r"(?:actual|real)\s+price\s+then\b|\bso\s+which\s+(?:price|one)\s+is\s+right"
                r"\b", re.I), (
        "Bottom line: there is no single \"real\" price — each venue's own price is real for "
        "trading on that venue. If you are buying or selling on Bitget, the price that matters "
        "is Bitget's own live order book, because that is what you would actually pay or "
        "receive there.",
        "A number shown elsewhere, such as on a search engine, is a useful rough check, not an "
        "authority — it is sourced and delayed differently.",
        "Ask \"what is the BTC price now\" here for Bitget's own live bid, ask and spread.")),
    # "how do i even read the 24h change number on the app" was read as an unknown ticker symbol
    # — a basic UI-literacy question, never answered (a first-time user, round 35)
    # only how to read it: "SOL, XRP and DOGE: last price and 24h change, as a table" asks for
    # the figures and got this explainer (a live re-ask, round 35)
    (re.compile(r"\b(?:read|mean|means|meaning|understand|what\s+(?:is|does|'s)\s+(?:the|that|"
                r"this)?)\b[^?]{0,40}\b24\s*[-\s]?h(?:our)?\s+change\b|\b24\s*[-\s]?h(?:our)?\s+"
                r"change\b[^?]{0,30}\b(?:mean|means|meaning|tell\s+me)\b", re.I), (
        "Bottom line: the 24h change is how much the price has moved over the last 24 hours, "
        "shown as a percentage. A plus number (often shown in green) means it is higher now "
        "than 24 hours ago; a minus number (often red) means it is lower.",
        "It is a rolling window, not a fixed calendar day — it always compares right now to "
        "exactly 24 hours before, so it updates continuously rather than resetting at midnight.",
        "It says nothing about what happens next: a coin up a lot in the last 24h can keep "
        "rising or reverse. It measures the past, not a signal for the future.")),
    # "fees?" — the same one-word format as "ath?" one turn later in a different conversation,
    # refused (a first-time user, round 35)
    (re.compile(r"^\W*fees?\W*$", re.I), (
        "Bottom line: Bitget's standard spot trading fee is 0.10% a side (about 0.20% to buy "
        "and later sell); perpetual futures are about 0.02% for an order that waits on the "
        "book (maker) and 0.06% for one that fills at once (taker), plus funding every few "
        "hours while the position stays open.",
        "On top of that: the spread (the small gap between the buy and sell price) on every "
        "market order, and a withdrawal fee that depends on the coin and network.",
        "This console itself is free. Ask \"what does it cost to buy $500 of BTC\" for the "
        "exact fee on a live order.")),
    # "bhai ye leverage wala cheez kitna risky hai seriously batao" got no "Read as:" line at "
    # all and fell through to the generic refusal, while the identical question in plain English
    # was answered two conversations earlier (a first-time user, round 35) — matched directly on
    # the Hinglish, with no translation needed
    (re.compile(r"\bleverage\b[^?]{0,60}\bkitna\s+(?:risky|risk)\b|\bkitna\s+(?:risky|risk)\b"
                r"[^?]{0,60}\bleverage\b", re.I), (
        "Bottom line: leverage is risky — it multiplies both gains and losses, so the higher "
        "the number, the smaller the price move needed to wipe out your margin.",
        "Roughly, the move to liquidation is about 1 divided by the leverage: 2x can take "
        "about a 50% move against you, 5x about 20%, 10x about 10%, 20x about 5%. A beginner "
        "is usually safer on low leverage (2x-3x) or none at all (spot), where the most you "
        "can lose is what you put in.",
        "Start small, and only use money you could fully lose without it hurting.")),
    # the run-on "...should i just trust him and go all in" latched onto only the slang word
    # "moon" and dropped the far more important ask: all his savings, one coin, a friend's tip,
    # leverage on top (a first-time user, round 35)
    (re.compile(r"\b(?:friend|bro|mate|cousin|brother|sister)\b[^?]{0,120}\b(?:all\s+(?:my\s+)?"
                r"(?:savings|money)|go\s+all\s*-?\s*in|all\s+in\b)|\btrust\s+(?:him|her|them)"
                r"\b[^?]{0,40}\b(?:go\s+all\s*-?\s*in|all\s+in)\b", re.I), (
        "Bottom line: no — do not put all your savings into one coin on a friend's confidence, "
        "and do not add leverage on top of that. \"He seems confident\" is not evidence a coin "
        "will rise, and leverage only makes a wrong call more expensive, faster.",
        "Putting everything into one thing means one bad week can take all of it — most "
        "promised big moves do not happen, and coins chasing \"moon\" talk usually fall hard "
        "afterward too.",
        "If you try it at all, use a small amount you could lose completely without it "
        "mattering, with no leverage, and pick the size yourself rather than matching his "
        "confidence.")),
    # "whats the diff between the Earn tab and the Futures tab on bitget" was refused — a basic
    # product-navigation question (a first-time user, round 35)
    (re.compile(r"\bearn\s+tab\b[^?]{0,40}\bfutures\s+tab\b|\bfutures\s+tab\b[^?]{0,40}\bearn"
                r"\s+tab\b|\bdiff\w*\b[^?]{0,20}\bearn\b[^?]{0,20}\bfutures\b", re.I), (
        "Bottom line: Earn is for putting coins you already hold to work for a yield (lending "
        "them out or staking), with little price risk beyond the coin's own; Futures is for "
        "trading a contract on a coin's price, with optional leverage, where you can lose more "
        "than you put in if you use it.",
        "Earn pays you over time while the coin mostly just sits there — locked for a fixed "
        "term, or free to redeem, depending on the product. Futures changes value with every "
        "price move and can be closed (liquidated) by the exchange if a leveraged loss gets "
        "too large.",
        "A beginner typically starts with neither, and buys spot first — owning the coin "
        "outright. Check Bitget's help centre for what each product on Earn currently offers.")),
    # "can u explain bitcoin to me like im explaining to my grandma" got a good plain definition
    # with live bid/ask spread and funding-rate figures tacked on — exactly the jargon a
    # grandma-level explanation should omit (a first-time user, round 35)
    (re.compile(r"\bexplain\s+bitcoin\b[^?]{0,40}\bgrandma\b|\bgrandma\b[^?]{0,40}\bbitcoin\b|"
                r"\bbitcoin\b[^?]{0,40}\blike\s+(?:i'?m|i\s+am)\s+(?:explaining\s+to\s+)?"
                r"(?:my\s+)?grandma\b", re.I), (
        "Bottom line: Bitcoin is digital money that is not controlled by any bank, company or "
        "government — it runs on a shared public record (the blockchain) that thousands of "
        "computers around the world keep copies of and agree on together, instead of one "
        "company's database.",
        "Only 21 million bitcoins will ever exist, fixed by the software and unchangeable — "
        "unlike ordinary money, where a central bank can print more.",
        "You can send it to anyone in the world without a bank in the middle, but its price "
        "moves a lot — it has risen or fallen by half or more within a single year more than "
        "once, so it is not something to treat like a savings account.")),
    # "wats funding rate mean like if im holding a long position" was answered well, but the
    # plainer "what does funding rate mean for me" form is asked just as often (a first-time
    # user, round 35)
    (re.compile(r"\bfunding\s+rate\b[^?]{0,40}\bmeans?\s+for\s+me\b|\bwhat(?:'?s|\s+is)\s+"
                r"(?:the\s+)?funding\s+rate\b[^?]{0,40}\bmean\b", re.I), (
        "Bottom line: the funding rate is a small payment made every few hours (every 8 hours "
        "on most Bitget perpetuals) between everyone long and everyone short a perpetual "
        "contract — it keeps the contract's price close to the real spot price; Bitget does "
        "not keep it.",
        "What it means for you: if the rate is positive, longs pay shorts, so holding a long "
        "costs you a little each interval; if it is negative, shorts pay longs. It only "
        "applies while you hold an open perpetual — spot holdings never pay or receive it.",
        "Ask \"what is the funding on BTC\" here to see which side is paying right now on a "
        "real contract.")),
    # "my coin pumped 50% today should i take profit now" silently substituted BTC for the
    # user's own coin and never disclosed it (a first-time user, round 35)
    (re.compile(r"\b(?:pumped|is\s+up|went\s+up|rose|gained|up)\s+\d{1,4}\s*%[^?]{0,80}\btake"
                r"\s+profits?\b|\btake\s+profits?\b[^?]{0,80}\b(?:pumped|up|rose|gained)\s+"
                r"\d{1,4}\s*%", re.I), (
        "Bottom line: no call here on whether to sell — but this is how people usually decide: "
        "before a big move happens, set a plan (sell a third at +50%, another third at +100%, "
        "keep the rest, say), so the decision is not made in the heat of the moment.",
        "A common middle path is taking some profit, not all: selling part locks in a real "
        "gain while leaving the rest open if it keeps rising, so you are not fully out and not "
        "fully exposed either.",
        "Another approach is a trailing stop — a stop-loss that moves up as the price rises, "
        "so a pullback locks in most of the gain automatically instead of you watching and "
        "deciding in real time.",
        "What it should not be based on: hope that it keeps going, or regret about selling "
        "\"too early\" — those are feelings, not a plan.")),
    # "but what if it keeps going up after i sell ill feel so dumb" was read as an unknown
    # ticker symbol, dropping the emotional profit-taking framing entirely (a first-time user,
    # round 35)
    (re.compile(r"\bwhat\s+if\s+it\s+keeps?\s+(?:going\s+up|rising|climbing)\s+after\s+i\s+"
                r"sell\b|\bi'?ll\s+feel\s+(?:so\s+)?(?:dumb|stupid)\s+(?:if|after)\b.{0,40}"
                r"\bsell\b|\bafraid\s+i'?ll\s+regret\s+selling\b", re.I), (
        "Bottom line: it might keep going up after you sell, and that is fine — the point of "
        "taking profit is locking in a real gain, not catching the exact top, which nobody "
        "does reliably.",
        "Selling part instead of all of it is the usual fix for this exact feeling: you still "
        "benefit if it keeps rising, and you have already banked something if it does not.",
        "Regret either way is normal — selling early feels dumb if it keeps rising, holding "
        "too long feels dumb if it falls. Judge the decision by the plan you had, not by what "
        "the price does afterward.")),
    # "Was ist der Unterschied zwischen Bitget und einer Wallet?" translated correctly, but the
    # underlying concept question still got the generic refusal (a first-time user, round 35)
    (re.compile(r"\bdifference\s+between\s+bitget\s+and\s+(?:a\s+)?wallet\b|\bbitget\s+(?:vs\.?"
                r"|versus)\s+(?:a\s+)?wallet\b|\bwallet\s+(?:vs\.?|versus)\s+bitget\b", re.I), (
        "Bottom line: Bitget is an exchange — it holds coins for you, the way a bank holds "
        "cash, and you log in with a password (plus 2FA) to buy, sell and trade. A wallet is "
        "software or a physical device that holds the keys to coins directly on the "
        "blockchain, under your own control, with no company in between.",
        "On Bitget, Bitget can help recover your account if you lose access, through its own "
        "verification process; in a self-custody wallet nobody can — lose the seed phrase and "
        "the coins are gone for good.",
        "Most beginners start on an exchange like Bitget because it is simpler, and move some "
        "coins to their own wallet later, once they want more control.")),
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


def first_reply(text: str) -> list[str] | None:
    """The answer to one of :data:`_FIRST`, or None."""
    for asked, answer in _FIRST:
        if asked.search(text):
            return list(_lead_by_question(text, answer))
    return None


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
