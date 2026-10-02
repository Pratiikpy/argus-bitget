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
    r"\b(?:start|begin)\b|\bwhere\s+(?:do|should)\s+i\s+(?:start|begin)\b[?.!]*\s*$", re.I)
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
    r"subscription)\s+(?:to\s+use|for\s+using)\b", re.I)
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
    r"\bhow\s+(?:do|can|would)\s+i\s+(?:withdraw|take\s+out|cash\s+out|get\s+(?:my\s+)?money\s+out)"
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
    "Bottom line: yes — using this console costs nothing: there is no account, no login, no card "
    "and no subscription.",
    "One limit: the language model that reads oddly phrased questions answers up to 40 questions "
    "an hour from one network address. Past that, the console's own readers answer, in English, "
    "and every figure is still computed from live data.",
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


explains(*_LOOKAHEAD_A, *_OVERFIT_A, *_SURVIVOR_A, *_COSTS_A, *_NOT_ADVICE, *_BUYING, *_LOSING,
         *_LOSS_HAPPENS_A, *_BEGINNER_SAFE_A,
         *_GOING_WRONG, *_NO_GUARANTEE, *_SAFE, *_WITHDRAWING, *_PLACING, *_FREE_TO_USE)

def reply(text: str, *, named: bool = False) -> Reply | None:
    """The newcomer answer to ``text``, or None when it is not one of these questions.

    ``named`` says the question names a contract; the questions that ask about one name ("what
    could go wrong with NVDA", "is now a good time to buy TSLA") are that name's engines' to
    answer, and only their nameless forms are taken here (the round-8 first-user audit,
    2026-09-30: all five nameless forms were declined)."""
    from argus.lui.research.sizing import stated_capital

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
