"""A beginner's first question — "I have 5,000 dollars, what should I do" — answered without advice.

It was refused as unrecognised (a first-time-user audit, 2026-09-29), the one question a newcomer
is most likely to type first. This console does not tell anyone what to buy, and says so; what it
can do honestly is show what that exact sum has actually been through over the last year in the
three broad things Bitget lists that a beginner reads about — the Nasdaq-100, bitcoin and gold —
end to end and at its worst point, and name the three questions that decide the choice more than
any market does.

Every figure is from real daily closes (Yahoo Finance's adjusted chart, `market/equity_history`):
QQQ for the Nasdaq-100, BTC-USD, and GLD for gold. It is what happened over one year, not a
forecast, and the answer says that too.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import trace_module

STARTER_Q = re.compile(
    # "i got like 700 bucks lying around, wat shud i buy" and "I'm new and have $2,000 - what's
    # a sensible way to begin?" were filed as notes and never answered (round 23)
    # "ok so i put in like 300 bucks total wut do i do with it" was kept as a note and its
    # question dropped (a first-time user, round 28)
    # "i saved up 150 bucks from my part time job, what should i actually do with it" was
    # declined as off-topic (a first-time user, round 29)
    r"\b(?:i|and)\s+(?:only\s+|just\s+|already\s+)?(?:have|got|own|saved(?:\s+up)?|put\s+in|"
    r"deposited|can\s+(?:put|invest|spare))\s+"
    r"(?:about\s+|around\s+|"
    r"roughly\s+|only\s+|like\s+|maybe\s+|just\s+)?\$?\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>k|thousand|grand|dollars?|usd|usdt|bucks)?\b.{0,300}?(?:wh?at\s+(?:should|shud|"
    r"shld|do|can|would)\s+i\s+(?:even\s+|actually\s+)?(?:do|buy|get)|w(?:u|ha|a|ah)t\s+(?:do|"
    r"should|shud|can)\s+i\s+(?:even\s+|actually\s+)?do\s+with\s+it|now\s+what|what\s+now|"
    r"(?:sensible|good|best|smart|right)\s+(?:way|place)\s+"
    r"to\s+(?:begin|start)|(?:way|how)\s+to\s+(?:begin|start)|where\s+(?:should|do|can)\s+"
    r"i\s+(?:put|invest|start)|how\s+(?:should|do|can)\s+i\s+(?:invest|start|begin)|what\s+(?:should|"
    r"do)\s+i\s+(?:buy|invest\s+in)|to\s+invest|invest\s+it|where\s+to\s+start|what\s+to\s+buy|"
    r"any\s+(?:advice|ideas|suggestions)|(?:don'?t|do\s+not)\s+know\s+where\s+to\s+start|"
    r"crypto\s+or\s+stocks|stocks\s+or\s+crypto|"
    # "I have $800 saved, can you help me start investing" was declined (a first-time user,
    # 2026-09-30).
    r"help\s+me\s+(?:start|begin|get\s+started)|start(?:ing)?\s+(?:to\s+)?invest\w*|"
    r"get(?:ting)?\s+started|"
    # "i only got like 200 bucks where do i even start" and "...make me a plan" were filed as
    # notes and never answered (a first-time user, round 27)
    r"where\s+(?:do|should|can)\s+i\s+even\s+(?:start|begin)|(?:make|give|build)\s+me\s+a\s+"
    r"plan|plan\s+for\s+me)", re.I | re.S)
STARTER_LOSS_Q = re.compile(
    # "im 22, student, can lose maybe $300 total, want to grow it slowly. make me a plan": the sum
    # a newcomer can afford to lose is the first sum (a first-time user, round 27)
    r"\bcan\s+(?:afford\s+to\s+)?(?:lose|risk)\s+(?:about\s+|around\s+|maybe\s+|up\s+to\s+|"
    r"like\s+)?\$\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>k|thousand|grand)?\b.{0,200}?"
    r"(?:(?:make|give|build)\s+me\s+a\s+plan|plan\s+for\s+me|where\s+(?:do|should|can)\s+i\s+"
    r"(?:even\s+)?(?:start|begin)|what\s+should\s+i\s+(?:do|buy)|grow\s+it)", re.I | re.S)
"""A first-money question: an amount, and a request for what to do with it."""

_MARKETS: tuple[tuple[str, str], ...] = (("QQQ", "the Nasdaq-100 (QQQ)"), ("BTC-USD", "bitcoin"),
                                         ("GLD", "gold (GLD)"))


STARTER_GROW_Q = re.compile(
    # "i want to grow my $300 slowly, i'm 22" was declined (a first-time user, round 27)
    r"\b(?:want|wanna|trying|looking|hope|hoping|need|plan(?:ning)?)\s+to\s+(?:grow|build(?:\s+up)?|"
    r"invest|start\s+with|turn)\s+(?:my\s+|this\s+|the\s+|about\s+|like\s+)?\$\s*"
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>k|thousand|grand)?\b", re.I)
"""A first sum and what the newcomer wants it to do."""
STARTER_IF_ME_Q = re.compile(
    # "if you were me with like 500 bucks what would you actually do with it" was told "that" has
    # nothing to refer to (a first-time user, round 30)
    r"\b(?:if\s+you\s+were\s+me|if\s+i\s+were\s+you|in\s+(?:my|your)\s+(?:shoes|position|place)|"
    r"what\s+would\s+you\s+(?:actually\s+|personally\s+)?do)\b.{0,60}?\$?\s*(?P<amount>\d[\d,]*"
    r"(?:\.\d+)?)\s*(?P<unit>k|thousand|grand|dollars?|usd|usdt|bucks)?\b", re.I | re.S)
"""A sum, and what the console would do with it in the asker's place."""


def amount_of(text: str) -> float | None:
    """The sum a first-money question names, in dollars."""
    m = (STARTER_Q.search(text) or STARTER_LOSS_Q.search(text) or STARTER_GROW_Q.search(text)
         or STARTER_IF_ME_Q.search(text))
    if m is None:
        return None
    value = float(m.group("amount").replace(",", ""))
    unit = (m.group("unit") or "").lower()
    dollars = "$" in text[max(0, m.start("amount") - 2):m.start("amount")] or bool(unit)
    if not dollars and value < 100:
        return None  # "I have 3 kids, what should I do" names no money
    return value * 1000 if unit in ("k", "thousand", "grand") else value


def answer(text: str, *, today: date, daily: Callable[[str], Any] | None = None
           ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """What the named sum went through over the last year in three broad markets."""
    stake = amount_of(text) or 5000.0
    if daily is None:
        from argus.market.equity_history import daily as daily_closes

        daily = daily_closes
    since = date(today.year - 1, today.month, min(today.day, 28))
    rows: list[str] = []
    data: dict[str, Any] = {}
    for ticker, label in _MARKETS:
        try:
            days = [d for d in daily(ticker) if d.day >= since]
        except Exception:
            continue
        if len(days) < 20:
            continue
        change = days[-1].close / days[0].close - 1.0
        lowest = min(d.close for d in days) / days[0].close
        peak, worst = days[0].close, 0.0
        for d in days:
            peak = max(peak, d.close)
            worst = min(worst, d.close / peak - 1.0)
        # "at its lowest it was worth $500" when it never went below the start read as odd (a
        # first-time user, round 20, row 691)
        floor = (f"it never fell below the ${stake:,.0f} it started at" if lowest >= 0.9995 else
                 f"at its lowest it was worth ${stake * lowest:,.0f}")
        rows.append(f"In {label}: ${stake:,.0f} would be ${stake * (1 + change):,.0f} a year later "
                    f"({change:+.0%}); {floor}, and its worst fall from a high on the way was "
                    f"{abs(worst):.0%}.")
        data[ticker] = {"change": change, "worst": worst, "from": str(days[0].day),
                        "to": str(days[-1].day)}
    lines = [f"Bottom line: this console will not tell you what to buy — it does not know your "
             f"situation — but it can show what ${stake:,.0f} has actually been through over the "
             f"last year in three broad markets Bitget lists."]
    lines += rows or ["The price histories did not arrive just now, so the year could not be "
                      "shown; ask again in a moment."]
    # A newcomer's paragraph often names a term too ("how do I protect it with a stop loss"): its
    # plain definition is added rather than the question dropped without a word (first-user
    # audit, 2026-09-29).
    import re as _re

    from argus.lui.concepts import CONCEPTS

    also = [c for c in CONCEPTS if _re.search(rf"\b(?:{c.pattern})\b", text, _re.I)][:2]
    lines += [f"You also asked about {c.name}: {c.definition} {c.reading}" for c in also]
    lines += ["Three questions decide it more than any market: how long before you need the "
              "money; how much of it you could lose without it changing your life; and whether "
              "you would sell in a fall like the ones above.",
              "Ask next: \"how much could I lose on QQQ this week\", \"what is dollar-cost "
              "averaging\", or \"should I DCA into BTC\".",
              "That is one year of real closes, not a forecast, and not advice."]
    sources = [Source(kind="venue", ref="https://query1.finance.yahoo.com/v8/finance/chart",
                      detail="QQQ, BTC-USD and GLD daily closes over the last year")]
    return lines, sources, {"starter": data, "stake": stake}


trace_module(globals())
