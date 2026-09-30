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
    r"\bhow\s+(?:do\s+i|to|can\s+i|would\s+i)\s+(?:buy|get|purchase|start\s+(?:buying|trading|"
    r"investing))\s*(?:(?:some\s+)?(?:crypto|bitcoin|btc|ethereum|eth|stocks?|shares?|coins?|"
    r"tokens?)\b|[?.!]*\s*$)|\bhow\s+(?:do\s+i|to|can\s+i)\s+(?:start|begin)\s+(?:trading|investing)\b|\bwhere\s+(?:do|"
    r"can)\s+i\s+buy\b", re.I)
_LOSE_MORE = re.compile(
    r"\b(?:can|could|will|do)\s+i\s+lose\s+(?:more\s+(?:money\s+)?than|everything|all\s+(?:of\s+)?"
    r"my\s+money)|\blose\s+more\s+than\s+(?:i|you)\s+(?:put|invest)|\bowe\s+(?:money|the\s+exchange)"
    r"|\bnegative\s+balance\b", re.I)
_HOW_BUYING_WORKS = re.compile(
    r"\bhow\s+(?:does|do)\s+(?:buying|trading|investing|it|this)\s+(?:actually\s+|really\s+)?work\b|"
    r"\bhow\s+(?:buying|trading)\s+works\b", re.I)
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
    r"\bpromise\s+(?:me\s+)?(?:a\s+)?(?:return|profit)", re.I)
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


explains(*_NOT_ADVICE, *_BUYING, *_LOSING, *_GOING_WRONG, *_NO_GUARANTEE, *_SAFE,
         *_WITHDRAWING)

def reply(text: str, *, named: bool = False) -> Reply | None:
    """The newcomer answer to ``text``, or None when it is not one of these questions.

    ``named`` says the question names a contract; the questions that ask about one name ("what
    could go wrong with NVDA", "is now a good time to buy TSLA") are that name's engines' to
    answer, and only their nameless forms are taken here (the round-8 first-user audit,
    2026-09-30: all five nameless forms were declined)."""
    from argus.lui.research.sizing import stated_capital

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
