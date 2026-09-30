"""What each of Bitget's five research Skills does, and where this console calls it.

"What does the sentiment-analyst skill do" was declined (a hostile review, 2026-09-30), although
Track 3 scores "Skill integration count and effectiveness" by name and the console calls all
five Skills in its engines. What each Skill reads is quoted from bitget-signal's own README
(the Capabilities table, `bitget-signal/README.md`); where ARGUS calls it is the module and the
tool names found in this repository. How often each answers is the Skill-health sweep's to say, and
the answer points there rather than restating a figure that moves.
"""

from __future__ import annotations

import re
from typing import Any

from argus.lui.provenance import explains
from argus.lui.trace import trace_module
from argus.truth.source import Source

SKILLS: tuple[tuple[str, str, str], ...] = (
    ("macro-analyst",
     "Fed policy, FOMC news, the yield curve and cross-asset correlation (BTC against gold, the "
     "dollar index, the Nasdaq, the S&P 500 and the 10-year)",
     "the macro engine (lui/research/macro.py) calls its macro_indicators and rates_yields tools"),
    ("market-intel",
     "on-chain capital flows, ETF data, DeFi TVL and cycle indicators",
     "the on-chain answers (lui/onchain.py) call defi_analytics, and the trending answer "
     "(lui/trending.py) calls crypto_market"),
    ("sentiment-analyst",
     "the Fear & Greed index, long/short ratios, open interest, funding rates, the taker ratio "
     "and Reddit signals",
     "the sentiment engine (lui/research/sentiment.py) calls sentiment_index and "
     "crypto_derivatives, and the trending answer calls social_trending"),
    ("technical-analysis",
     "23 indicators in six groups: trend, volatility, oscillators, volume, momentum, and "
     "support and resistance",
     "the technicals engine (lui/research/technicals.py) calls technical_analysis for RSI, MACD, "
     "support and resistance and ATR, and recomputes MACD from Bitget's own candles because the "
     "Skill returns the signal line and the histogram in each other's fields"),
    ("news-briefing",
     "44 RSS and Atom feeds, social trending boards and a narrative summary",
     "the news engine (lui/research/news.py) calls news_feed"),
)
"""Name, what it reads (bitget-signal README, Capabilities), and where ARGUS calls it."""

README = "https://github.com/Bitget-AI/bitget-signal"

SKILLS_Q = re.compile(
    r"\b(?:macro-?analyst|market-?intel|sentiment-?analyst|technical-?analysis|news-?briefing)\s+"
    r"skill\b|\bwhat\s+(?:does|do|is|are)\s+(?:the\s+)?(?:bitget(?:[\s-]+signal)?\s+)?skills?\b"
    r"(?!\s+(?:say|think|read|show)\b)|\bwhich\s+(?:bitget\s+)?skills\s+(?:do\s+you|does\s+"
    r"(?:argus|it))\s+use\b|\bwhat\s+is\s+bitget[\s-]+signal\b|"
    # "How many Skills do you have?" after the Skills answer was refused (round 12)
    r"\bhow\s+many\s+(?:bitget\s+)?skills\b", re.I)
"""A question about what the Skills are, not a request for one Skill's reading of a name."""


def asks(text: str) -> bool:
    """A question about the Skills themselves: not their health ("are they working"), and not one
    Skill's reading of a named contract ("what does the sentiment-analyst skill say about BTC"),
    which the engine that calls it answers."""
    from argus.lui.research import research_symbols

    return bool(SKILLS_Q.search(text)) and not research_symbols(text)[0] and not re.search(
        r"\b(?:working|work|reliab\w*|answer\w*|health|up|down|effective\w*)\b", text, re.I)


def answer(text: str) -> tuple[list[str], list[Source], dict[str, Any]]:
    named = [s for s in SKILLS if re.search(s[0].replace("-", "-?"), text, re.I)]
    chosen = named or list(SKILLS)
    lines = [("Bottom line: " + (
        f"{chosen[0][0]} is one of the five research Skills in Bitget's bitget-signal package: it "
        f"reads {chosen[0][1]}." if named and len(named) == 1 else
        "bitget-signal is Bitget's package of five research Skills, keyless, served by Bitget's "
        "public data service; this console calls all five inside its engines."))]
    for name, reads, used in chosen:
        lines.append(f"{name}: reads {reads}. Here, {used}.")
    lines.append(f"Their own description: {README}. How often each answers right now — they "
                 f"time out often — ask \"are the bitget skills working\".")
    explains(*lines)
    return lines, [Source("computation", "bitget-signal README (Capabilities)", README)], {
        "skills": [s[0] for s in chosen]}


trace_module(globals())
