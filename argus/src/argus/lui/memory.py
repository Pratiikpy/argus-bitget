"""What a trader has told the console, kept across sessions and used in every later answer.

Track 3 judges a *personalised thesis*. Until this module, the only thing the console remembered was
the book a trader typed into one field; "I can't lose more than 10%", "I hold for weeks", "I think
NVDA runs on AI capex" were read once and forgotten, so the next answer was the same for everyone.

**What was taken from mem0 (Apache-2.0; `mem0/memory/main.py:881-1210`, `:143-172`), and what was
not.** Taken: extraction on every turn, facts deduplicated so the same statement is stored once, a
newer fact of the same kind replacing the older one, the date each fact was stated anchoring it
(`prompts.py:524-540`), and identity scoping — a fact can only belong to the trader whose browser
holds it. Not taken: the LLM extraction call (mem0 spends one per turn; the Qwen key here is a
budget, and the facts a trading answer needs have fixed shapes a pattern reads exactly), the vector
store (a trader's memory is tens of facts, not millions), and the server-side store: the memory
lives in the trader's own browser and is sent with each question, so the hosted console keeps
nothing about anyone and a second visitor can never read the first one's facts — mem0's three
cross-tenant issues (#4490, #6277, #6655) cannot happen when there is no shared store.

**What memory changes.** A remembered risk budget, holding period, loss limit, style or capital is
applied where the question did not state its own — and every use is said on a ``Remembered:`` line,
so a trader sees which of their words shaped the answer. A remembered thesis on a name is shown
beside every later answer about that name with the move since it was stated: the expectation gap,
measured, not narrated.

Everything read from the client is untrusted: parsed, bounded, and never executed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from argus.lui.research.kinds import bare_symbol

MAX_FACTS = 40
MAX_TEXT = 200

KINDS = ("budget", "max_loss", "loss_usd", "horizon", "style", "capital", "thesis", "avoid",
         "check", "goal", "book", "trade_risk", "cap")
"""``trade_risk`` is the share of the account the trader risks on one trade ("I won't risk more
than 2% per trade"), and ``cap`` the largest weight one position may have ("no single position
above 30% of the book"). Both were lost or misread as a risk budget (a judge, round 21)."""
"""``check`` is one line of the checklist a review of the trader's own trades wrote
(`lui/journal.py`): its subject is the habit's key, its text the check. It is kept so the check is
run again on a later entry (:func:`after`), which the review promised and nothing did until
2026-09-27 (audit finding 56)."""

DEFAULT_CHECK_DAYS = 5
"""The hold a checklist is tested over when the trader has stated none: one trading week."""


@dataclass(frozen=True, slots=True)
class Fact:
    kind: str
    subject: str
    """A symbol for a thesis or an avoid; empty for a fact about the trader."""
    value: str
    text: str
    """The trader's own words, as stated."""
    at: str
    """The date it was stated, ISO."""
    price_at: float | None = None
    """For a thesis: the instrument's price when it was stated."""
    replaces: str = ""
    """What this fact replaced, as "“the earlier words” (its date)", or empty. One step of
    history, kept with the fact: mem0 keeps every version (``history()``, Apache-2.0); a trader's
    browser-held memory keeps the step that matters, the one being overridden, so a changed mind
    is shown rather than silently applied (research/harvest/04-mem0.md, decision 1)."""

    def key(self) -> tuple[str, str]:
        return self.kind, self.subject


_BUDGET = re.compile(r"\b(?:my\s+)?risk\s+budget\s+(?:is\s+|of\s+)?(\d{1,2}(?:\.\d+)?)\s*%|"
                     r"\bno\s+(?:single\s+)?(?:name|position|holding)\s+(?:above|over|more\s+than)\s+"
                     r"(\d{1,2}(?:\.\d+)?)\s*%\s+of\s+(?:my\s+)?risk", re.I)
_TRADE_RISK = re.compile(
    r"\b(?:i\s+)?(?:won'?t|will\s+not|never|don'?t|do\s+not)\s+risk\s+(?:more\s+than\s+)?"
    r"(?P<a>\d{1,2}(?:\.\d+)?)\s*%(?:\s+of\s+(?:my\s+)?(?:account|capital|money|portfolio|book))?"
    r"\s+(?:per|a|each|on\s+(?:any|a|one))\s+(?:single\s+)?(?:trade|position|bet)\b|"
    r"\b(?:i\s+)?(?:only\s+)?risk\s+(?P<b>\d{1,2}(?:\.\d+)?)\s*%\s+(?:of\s+(?:my\s+)?(?:account|"
    r"capital)\s+)?(?:per|a|each)\s+trade\b|\brisk\s+per\s+trade\s+(?:is\s+|of\s+)?"
    r"(?P<c>\d{1,2}(?:\.\d+)?)\s*%", re.I)
_CAP = re.compile(
    r"\bno\s+(?:single\s+)?(?:position|name|holding|stock|coin|asset)\s+(?:may\s+|can\s+|should\s+|"
    r"will\s+|is\s+to\s+)?(?:exceed|be\s+(?:above|over|more\s+than|bigger\s+than)|above|over|"
    r"bigger\s+than|larger\s+than|more\s+than)\s+(?P<a>\d{1,2}(?:\.\d+)?)\s*%(?!\s+of\s+(?:my\s+)?"
    r"(?:the\s+)?risk)|\bmax(?:imum)?\s+(?P<b>\d{1,2}(?:\.\d+)?)\s*%\s+(?:per|in\s+(?:any\s+)?one|"
    r"in\s+a\s+single)\s+(?:position|name|holding|stock)\b|\b(?:position|single[\s-]name)\s+cap\s+"
    r"(?:is\s+|of\s+)?(?P<c>\d{1,2}(?:\.\d+)?)\s*%", re.I)
_MAX_LOSS = re.compile(
    r"\b(?:i\s+)?(?:can'?t|cannot|can\s+not|don'?t\s+want\s+to|won'?t|never)\s+(?:afford\s+to\s+)?"
    r"lose\s+"
    r"(?:more\s+than\s+)?(\d{1,2}(?:\.\d+)?)\s*%|\bmax(?:imum)?\s+(?:loss|drawdown)\s+(?:limit\s+|tolerance\s+)?(?:is\s+|of\s+)?"
    r"(\d{1,2}(?:\.\d+)?)\s*%|\b(?:my\s+)?(?:loss|drawdown)\s+limit\s+(?:is\s+|of\s+)?"
    r"(\d{1,2}(?:\.\d+)?)\s*%|"
    # "my max loss should probably be around 10%" (the mem0 comparison, round 12)
    r"\bmax(?:imum)?\s+loss\s+(?:should\s+(?:probably\s+)?be|would\s+be|of)\s+(?:around\s+|about\s+|"
    r"roughly\s+|~)?(\d{1,2}(?:\.\d+)?)\s*%", re.I)
_HORIZON = re.compile(
    r"\bi(?:'?m|\s+am)\s+an?\s+(?:[\w-]+\s+){0,3}?(day|swing|position|long[\s-]term)\s+"
    r"(?:trader|investor)|"
    # "I'm a swing trader and hold positions about 3 months" kept the style's 7 days (a judge,
    # round 22): the hold said after "and" is the horizon
    r"\b(?:i(?:'?ll|\s+will|'?d)?|and)\s+(?:usually\s+|normally\s+|typically\s+|only\s+|"
    r"plan\s+to\s+)?hold\s+(?:it\s+|this\s+(?:one\s+)?|positions?\s+|trades?\s+)?"
    r"(?:for\s+)?(?:about\s+|around\s+|roughly\s+|up\s+to\s+)?"
    r"(?P<n1>a\s+few|several|a\s+couple(?:\s+of)?|\d{1,3}|one|two|three|four|six|an?)?\s*"
    r"(?P<u1>hours?|days?|weeks?|months?|years?)\b|"
    r"\b(?P<explicit>(?:my\s+)?(?:time\s+|trading\s+|holding\s+|investment\s+)?horizon\s+"
    r"(?:is\s+|of\s+|=\s*|:\s*)?(?:about\s+|around\s+|roughly\s+)?)"
    r"(?P<n2>a\s+few|several|a\s+couple(?:\s+of)?|\d{1,3}|one|two|three|four|six|an?)?\s*"
    r"(?P<unit>hours?|days?|weeks?|months?|years?)\b", re.I)
_HOLDS = re.compile(
    r"\b(?:i\s+(?:currently\s+|now\s+)?(?:hold|own|have)|i(?:'?m|\s+am)\s+(?:holding|long|in)|"
    r"my\s+(?:current\s+)?(?:book|portfolio|holdings?|allocation)\s+(?:is|are|reads|looks\s+like|:)"
    r")\b[^.?!;\n]*", re.I)
"""Holdings said in a sentence: "I hold 40% NVDA, 30% MSFT, 30% AAPL". Kept as the book, so a later
"what are my exposures if I add 10% XOM" is asked of it (a judge, round 13, 2026-09-30: the book
was said once in chat, and the next answer read "your book is 100% Energy")."""

_HOW_MANY = {"a few": 3, "several": 4, "a couple": 2, "a couple of": 2, "one": 1, "a": 1, "an": 1,
             "two": 2, "three": 3, "four": 4, "six": 6}
""""A few weeks" is three weeks, not one: "my horizon is a few weeks" was stored as the swing-trader
default of one week (a judge, round 13, 2026-09-30)."""
_STYLE = re.compile(
    r"\bi(?:'?m|\s+am)\s+(?:a\s+|an\s+|pretty\s+|quite\s+|very\s+)?(conservative|aggressive|"
    r"risk[\s-]averse|cautious)\b|\bi\s+(?:mostly\s+|only\s+|usually\s+)?trade\s+(earnings|momentum|"
    r"mean[\s-]reversion|breakouts?|news|macro|crypto|stocks)\b|"
    r"\b(event[\s-]driven|earnings[\s-]driven|momentum|news[\s-]driven|macro)\s+(?:swing\s+|day\s+|"
    r"position\s+)?(?:trader|investor|trading)\b", re.I)
"""Style, including the one said in passing: "I am a tech-stock event-driven swing trader" was not
remembered at all on the hosted console (readiness audit, finding 40)."""
_CAPITAL = re.compile(
    r"\bmy\s+(?:account|book|portfolio|capital)\s+is\s+(?:actually\s+|now\s+|really\s+)?"
    r"(?:about\s+|around\s+)?[$€£]?\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\b|\bi\s+have\s+(?:about\s+|around\s+)?\$\s*(\d[\d,]*(?:\.\d+)?)"
    r"\s*(k|m)?\s+(?:to\s+(?:trade|invest)|in\s+my\s+account)|"
    r"\b(?:an?|my)\s+\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s+(?:account|book|portfolio)\b|"
    r"\b(?:trading|investing)\s+(?:with\s+)?\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\b|"
    # "I only have about $1,000 total" was not kept (a first-time user, round 19, row 635)
    r"\bi\s+(?:only\s+)?have\s+(?:about\s+|around\s+|roughly\s+|only\s+|just\s+)?\$\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s+(?:total|in\s+total|altogether|overall|saved|to\s+my\s+name)\b|"
    # "I have $5,000 of margin in my account" was kept by the model as a bare "$5,000" and read
    # as a position (a judge, round 20, row 713); the words are kept with it
    r"\bi\s+have\s+(?:about\s+|around\s+)?\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s+(?:of\s+)?"
    r"(?:margin|collateral|buying\s+power)\b",
    re.I)
_THESIS = re.compile(
    r"\bi\s+(?:think|believe|expect|reckon|bet)\s+(?:that\s+)?(.{2,40}?)\s+(?:will|is\s+going\s+to|"
    r"gonna|should|can|to)\s+(.{3,120}?)(?=\s*(?:[,.;!?]|\b(?:and|but|while)\s+i\s|$))", re.I)
"""One view per match, ending at the clause's end, so "I think NVDA will rise and I think BTC will
crash" is two theses. Until 2026-09-28 the claim ran to the end of the message and only the first
view was kept (research/harvest/04-mem0.md, the multi-topic case mem0 is built for)."""
_IDEA = re.compile(
    r"\bi\s+(?:want|plan|intend|would\s+like)\s+to\s+(short|long|buy|sell)\s+(?:some\s+)?"
    r"(.{2,30}?)\s+(?:because|since|as)\s+(.{3,120}?)(?=\s*(?:[,.;!?]|$))", re.I)
"""A trade idea said with its reason: "I want to short TSLA because its deliveries missed" is a
thesis to keep, though it carries no "I think ... will". A judge asked for it back as "my TSLA idea"
and it was never stored (round 15, 2026-10-01)."""
_GOAL = re.compile(
    r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+(?:saving|trying\s+to\s+save|putting\s+money\s+aside)\s+"
    r"(?:up\s+)?(?:for\s+)?(.{3,80}?)(?=\s*(?:[,.;!?]|\bso\b|\band\b|$))|"
    r"\b(?:i'?m|i\s+am)\s+(?:trying|planning|hoping|aiming)\s+to\s+(retire(?:\s+early)?|buy\s+a\s+"
    r"(?:house|home|car|flat)|pay\s+(?:off|for)\s+.{3,40}?)(?=\s*(?:[,.;!?]|\bso\b|\band\b|$))|"
    r"\bmy\s+goal\s+is\s+(?:to\s+)?(.{3,80}?)(?=\s*(?:[,.;!?]|$))|"
    r"\b(?:i|we)\s+need\s+(?:this|the)\s+money\s+(.{3,60}?)(?=\s*(?:[,.;!?]|$))", re.I)
"""What the money is for, said in passing: "I am saving up for a house down payment next year",
"I'm trying to retire early". No kind read these, and mem0 kept both (the mem0 comparison, round
12, `eval/memory_comparison.py`). Kept as the trader's words and shown beside any answer about
risk, never turned into a number the console did not hear."""

_SENTENCE = re.compile(r"(?<=[.?!])\s+|\s+(?:--|—)\s+")
"""Sentences and dashed asides: a message that opens with a question can still state a fact in its
next sentence ("What is my risk tolerance? I think ... my max loss should be around 10%")."""

_ASKING = re.compile(r"^\s*(?:what|how|should|is|are|does|do|can|could|why|when|which|will|would|"
                     r"where|who)\b|\?\s*$", re.I)
"""A question states nothing about the trader: "what if I can't lose more than 10%?" and "should
I trade earnings" wrote false memories on the first extraction run (`eval/memory_eval.py`)."""
_AVOID = re.compile(r"\bi\s+(?:don'?t|never|won'?t)\s+(?:trade|touch|buy|hold|want)\s+(?:any\s+)?"
                    r"(.{2,30}?)(?:[,.;!?]|$)", re.I)
_STANCE = re.compile(
    r"\bi(?:'?m|\u2019m|\s+am)\s+(?:now\s+|still\s+|really\s+|very\s+|quite\s+|pretty\s+|"
    r"turning\s+|getting\s+)?(?P<side>bullish|bearish)\s+(?:on\s+|about\s+)?"
    r"(?P<name>[A-Za-z][A-Za-z.]{1,11})\b", re.I)
"""A view said as a stance: "I'm bearish on TSLA", "Actually I'm now bullish on TSLA". Neither was
kept, and "What's my view on TSLA?" was told nothing was remembered (a hostile review, round 19,
row 649)."""
_UNTIL_REPORT = re.compile(
    r"\b(?:hold|keep|stay\s+in|ride)\w*\s+(?:it\s+|them\s+|this\s+|the\s+position\s+)?"
    r"(?:until|till|through|into|past|over)\s+(?:its\s+|their\s+|the\s+)?(?:next\s+)?"
    r"(?:earnings|results|report|print)\b", re.I)
"""A holding period tied to the name's next report."""
_AVOID_CLASS = re.compile(
    r"\bi\s+(?:really\s+)?(?:hate|avoid|don'?t\s+(?:like|trust|do|want|use)|do\s+not\s+want|"
    r"can'?t\s+stand|never\s+(?:trade|touch|buy|use))\s+(?:to\s+(?:use|trade|touch|buy)\s+)?"
    r"(?:anything\s+(?:with|in|like)\s+|any\s+|to\s+use\s+)?(?P<what>[^.;!?]{2,60})", re.I)
_AVOID_WORDS = re.compile(
    r"\b(?P<w>meme\s*coins?|memecoins?|altcoins?|crypto(?:currenc(?:y|ies))?|coins|stocks|equities|"
    r"leverage|margin|options|futures|perps?|perpetuals|shorting|shorts)\b", re.I)
_AVOID_SAME = {"memecoin": "meme coins", "memecoins": "meme coins", "meme coin": "meme coins",
               "altcoin": "altcoins", "cryptocurrency": "crypto", "cryptocurrencies": "crypto",
               "coins": "crypto", "equities": "stocks", "perp": "perpetuals", "perps": "perpetuals",
               "margin": "leverage", "shorts": "shorting"}
"""The classes a trader stays away from, each kept as said: "I avoid meme coins" was kept as all of
crypto, and "anything with leverage or meme coins" kept nothing (a first-time user, round 20,
rows 680-681)."""
"""A whole class of instrument the trader stays away from: "I hate crypto and never trade it"."""
_BULL = re.compile(r"\b(?:up|rise|rally|outperform|beat|higher|moon|rip|double|recover|bounce)\w*",
                   re.I)
_BEAR = re.compile(r"\b(?:down|fall|drop|crash|underperform|miss|lower|dump|tank|decline|lose)\w*",
                   re.I)


def extract(question: str, now: datetime | None = None,
            price_of: Any = None) -> list[Fact]:
    """The facts a question states about the trader. ``price_of(symbol)`` stamps a thesis with
    the price at the time it was stated; it may be None (no price, no gap measured later)."""
    from argus.lui.research import research_symbols
    from argus.lui.research.sizing import spelled_out

    day = (now or datetime.now(UTC)).date().isoformat()
    # European figures and spelled percentages first: "My account is 25.000 USD" was not kept
    # (a hostile review, round 20, row 696)
    text = spelled_out(question.strip()[:600])
    facts: list[Fact] = []
    # Questions state nothing about the trader, but the sentence after one can: "What is my risk
    # tolerance? I think I can handle it but my max loss should probably be around 10%" kept
    # nothing, because the whole message was dropped as a question (the mem0 comparison, round 12).
    sentences = [part for part in _SENTENCE.split(text) if part.strip()]
    kept = [part for part in sentences
            if not (_ASKING.search(part) and not re.match(r"\s*i\b", part, re.I))]
    if not kept:
        return facts
    text = " ".join(kept)

    def add(kind: str, subject: str, value: str, words: str,
            price: float | None = None) -> None:
        facts.append(Fact(kind=kind, subject=subject, value=value, text=words.strip()[:MAX_TEXT],
                          at=day, price_at=price))

    def last(pattern: re.Pattern[str]) -> re.Match[str] | None:
        """The last statement of a kind in the message, the earlier one noted as replaced: "My
        risk budget is 20% -- actually, no single name above 30%" corrects itself, and "I hold for
        weeks normally -- well, this one for months" too (the mem0 comparison, round 12)."""
        found = list(pattern.finditer(text))
        return found[-1] if found else None

    def earlier(pattern: re.Pattern[str]) -> str:
        found = list(pattern.finditer(text))
        return f"“{found[0].group(0).strip()}” (earlier in the same message)" if len(found) > 1 \
            else ""

    if (m := last(_BUDGET)) is not None:
        add("budget", "", str(float(m.group(1) or m.group(2)) / 100), m.group(0))
        facts[-1] = replace(facts[-1], replaces=earlier(_BUDGET))
    if (m := last(_TRADE_RISK)) is not None:
        add("trade_risk", "", str(float(m.group("a") or m.group("b") or m.group("c")) / 100),
            m.group(0))
    if (m := last(_CAP)) is not None:
        add("cap", "", str(float(m.group("a") or m.group("b") or m.group("c")) / 100), m.group(0))
    if (m := last(_MAX_LOSS)) is not None:
        add("max_loss", "", str(float(m.group(1) or m.group(2) or m.group(3) or m.group(4))
                                / 100), m.group(0))
        facts[-1] = replace(facts[-1], replaces=earlier(_MAX_LOSS))
    from argus.lui.research.sizing import LOSS_USD

    dollars = list(LOSS_USD.finditer(text))
    if dollars:
        m = dollars[-1]
        amount = float((m.group("n") or m.group("n2")).replace(",", "")) * (
            1000 if (m.group("k") or m.group("k2")) else 1)
        add("loss_usd", "", f"{amount:.0f}", m.group(0))
        if len(dollars) > 1:
            facts[-1] = replace(facts[-1], replaces=f"“{dollars[0].group(0).strip()}” (earlier in "
                                                    f"the same message)")
    stated = [h for h in _HORIZON.finditer(text) if h.group("explicit") is not None]
    if (m := stated[-1] if stated else last(_HORIZON)) is not None:
        # A horizon said in words beats the one a trading style implies: "I'm a swing trader, my
        # horizon is a few weeks" is three weeks, whichever order the two come in.
        word = (m.group(1) or m.group("u1") or m.group("unit") or "").lower()
        count = (m.group("n1") or m.group("n2") or "").lower()
        count = re.sub(r"\s+", " ", count)
        times = int(count) if count.isdigit() else _HOW_MANY.get(count, 1)
        hours = {"day": 24, "swing": 24 * 7, "position": 24 * 30}.get(word) or (
            24 * 90 if word.startswith("long") else
            1 if word.startswith("hour") else 24 if word.startswith("day") else
            168 if word.startswith("week") else 720 if word.startswith("month") else 24 * 365)
        if word not in ("day", "swing", "position") and not word.startswith("long"):
            hours *= max(1, times)
        add("horizon", "", str(hours), re.sub(r"^and\s+", "I ", m.group(0)))
        facts[-1] = replace(facts[-1], replaces=earlier(_HORIZON))
    if (m := _STYLE.search(text)) is not None:
        add("style", "", re.sub(r"[\s-]+", "-", (m.group(1) or m.group(2) or m.group(3)
                                                  or "").lower()), m.group(0))
    if (m := _CAPITAL.search(text)) is not None:
        amount = float((m.group(1) or m.group(3) or m.group(5) or m.group(7) or m.group(9)
                        or m.group(11) or "0").replace(",", ""))
        unit = (m.group(2) or m.group(4) or m.group(6) or m.group(8) or m.group(10)
                or m.group(12) or "").lower()
        amount *= 1_000 if unit == "k" else 1_000_000 if unit == "m" else 1
        if amount >= 100:
            add("capital", "", f"{amount:.0f}", m.group(0))
    for m in _THESIS.finditer(text):
        named = research_symbols(m.group(1))[0]
        if named:
            claim = m.group(2)
            lean = ("bull" if _BULL.search(claim) and not _BEAR.search(claim) else
                    "bear" if _BEAR.search(claim) and not _BULL.search(claim) else "view")
            price = None
            if price_of is not None:
                try:
                    price = float(price_of(named[0]))
                except Exception:
                    price = None
            add("thesis", named[0], lean, m.group(0), price)
    for m in _IDEA.finditer(text):
        named = research_symbols(m.group(2))[0]
        if named:
            price = None
            if price_of is not None:
                try:
                    price = float(price_of(named[0]))
                except Exception:
                    price = None
            add("thesis", named[0], "bear" if m.group(1).lower() in ("short", "sell") else "bull",
                m.group(0), price)
    for m in _STANCE.finditer(text):
        named = research_symbols(m.group("name"))[0]
        if named:
            price = None
            if price_of is not None:
                try:
                    price = float(price_of(named[0]))
                except Exception:
                    price = None
            # The reason is part of the thesis: "I'm bullish on NVDA because hyperscaler capex
            # keeps growing" was kept as "I'm bullish on NVDA" (a judge, round 20, row 713)
            said = re.split(r"(?<=[.;!?])\s", text[m.start():], maxsplit=1)[0][:220]
            add("thesis", named[0], "bull" if m.group("side").lower() == "bullish" else "bear",
                said.rstrip(" .;!?") or m.group(0), price)
            if (until := _UNTIL_REPORT.search(text)) is not None and price_of is not None:
                # "I plan to hold until earnings" is a horizon, and it was not kept (row 713)
                try:
                    from argus.lui.journal import next_report_date

                    report = next_report_date(named[0])
                except Exception:
                    report = None
                if report is not None:
                    days = (report - (now or datetime.now(UTC)).date()).days + 1
                    if days > 0:
                        add("horizon", "", str(days * 24),
                            f"{until.group(0)} ({report:%d %b %Y})")
    for m in _AVOID_CLASS.finditer(text):
        for w in _AVOID_WORDS.finditer(m.group("what")):
            word = re.sub(r"\s+", " ", w.group("w").lower())
            what = _AVOID_SAME.get(word, word)
            if not any(f.kind == "avoid" and f.value == what for f in facts):
                add("avoid", what, what, m.group(0))
    for m in _HOLDS.finditer(text):
        from argus.lui.research.parse import holding_pairs

        pairs = holding_pairs(m.group(0))
        if pairs and sum(w for _, _, w in pairs) <= 1.0001:
            add("book", "", str(len(pairs)), m.group(0))
    if (m := _GOAL.search(text)) is not None:
        goal = next(g for g in m.groups() if g)
        add("goal", "", goal.strip(" ."), m.group(0))
    for m in _AVOID.finditer(text):
        named = research_symbols(m.group(1))[0]
        if named:
            add("avoid", named[0], "avoid", m.group(0))
    return facts


def checks_from(data: Any, now: datetime | None = None) -> list[Fact]:
    """The checklist of a trade review's ``data`` (`journal.review_trades`) as facts to keep; a
    new review replaces the old checklist line by line, and a habit it no longer finds is dropped
    by :func:`merge_checks`."""
    journal = (data or {}).get("journal") if isinstance(data, dict) else None
    patterns = (journal or {}).get("patterns") or []
    day = (now or datetime.now(UTC)).date().isoformat()
    return [Fact(kind="check", subject=str(p["key"])[:24], value=str(p.get("count", "")),
                 text=str(p["check"])[:MAX_TEXT], at=day)
            for p in patterns if isinstance(p, dict) and p.get("key") and p.get("check")]


def merge_checks(old: list[Fact], checks: list[Fact]) -> list[Fact]:
    """A new review's checklist in place of the last one's, the other facts kept."""
    return merge([f for f in old if f.kind != "check"], checks)


def parse(raw: str | None) -> list[Fact]:
    """The client's stored memory, validated. Anything malformed is dropped, never trusted."""
    if not raw:
        return []
    try:
        rows = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(rows, list):
        return []
    out: list[Fact] = []
    for row in rows[:MAX_FACTS]:
        if not isinstance(row, dict) or row.get("kind") not in KINDS:
            continue
        try:
            price = row.get("price_at")
            out.append(Fact(
                kind=str(row["kind"]), subject=str(row.get("subject", ""))[:24],
                value=str(row.get("value", ""))[:40], text=str(row.get("text", ""))[:MAX_TEXT],
                at=str(row.get("at", ""))[:10],
                price_at=float(price) if isinstance(price, (int, float)) else None,
                replaces=str(row.get("replaces", ""))[:MAX_TEXT + 20]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def merge(old: list[Fact], new: list[Fact]) -> list[Fact]:
    """A newer fact of the same kind and subject replaces the older, and records what it replaced
    (a repeat of the same statement keeps the record it already had); the rest are kept, newest
    first, up to :data:`MAX_FACTS`."""
    prior = {f.key(): f for f in old}
    stamped: list[Fact] = []
    for fact in new:
        was = prior.get(fact.key())
        if was is not None and (was.value, was.text) != (fact.value, fact.text):
            fact = replace(fact, replaces=f"“{was.text[:120]}” ({was.at})")
        elif was is not None and was.replaces and not fact.replaces:
            fact = replace(fact, replaces=was.replaces)
        stamped.append(fact)
    replaced = {f.key() for f in stamped}
    kept = [*stamped, *(f for f in old if f.key() not in replaced)]
    return kept[:MAX_FACTS]


_SOLD = re.compile(
    r"\bi\s+(?:just\s+|already\s+|have\s+|'ve\s+)?(?:sold|closed|exited|dumped|got\s+out\s+of)\s+"
    r"(?:all\s+(?:of\s+)?)?(?:my\s+)?(?P<name>[A-Za-z][A-Za-z.]{1,11})\b", re.I)
"""A holding the trader says is gone: "I sold all my AAPL" left the remembered book at 50% AAPL
(a hostile review, round 19, row 648)."""


def apply_sales(facts: list[Fact], text: str, now: datetime | None = None) -> list[Fact]:
    """The remembered book with each name the message says was sold taken out, the rest scaled to
    the whole; the old book kept as what was replaced."""
    from argus.lui.research import research_symbols
    from argus.lui.research.parse import holding_pairs

    book = get(facts, "book")
    if book is None:
        return facts
    sold = {s for m in _SOLD.finditer(text) for s in research_symbols(m.group("name"))[0]}
    if not sold:
        return facts
    held: dict[str, float] = {}
    for _, symbol, weight in holding_pairs(book.text):
        held[symbol] = held.get(symbol, 0.0) + weight
    if not sold & set(held):
        return facts
    left = {s: w for s, w in held.items() if s not in sold}
    total = sum(left.values())
    day = (now or datetime.now(UTC)).date().isoformat()
    if not left or total <= 0:
        rest = [f for f in facts if f is not book]
        return rest
    # what was sold becomes cash; the rest keep their weights: selling the AAPL of 40% NVDA, 60%
    # AAPL leaves 40% NVDA and 60% cash, not 100% NVDA (a hostile review, round 20, row 696)
    cash = max(0.0, 1.0 - total)
    words = "I hold " + ", ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in left.items()) \
        + (f", {cash:.0%} cash" if cash >= 0.005 else "")
    updated = Fact(kind="book", subject="", value=str(len(left)), text=words, at=day,
                   replaces=f"“{book.text[:120]}” ({book.at})")
    return [updated, *(f for f in facts if f is not book)]


_CORRECTED = re.compile(
    r"\b(?:actually|correction|sorry|i\s+meant|make\s+that|rather)\b[^.?!]{0,40}?"
    r"\b(?:it'?s|it\s+is|i\s+hold|i\s+have|make\s+it|that'?s)\s+(?P<pct>\d+(?:\.\d+)?)\s*%\s+"
    r"(?P<name>[A-Za-z][A-Za-z.]{1,11})\b", re.I)
"""A holding's weight corrected: "Actually it's 30% NVDA, not 40%." was answered with the desk's
record and the book kept 40% (a hostile review, round 20, row 696)."""


def apply_corrections(facts: list[Fact], text: str, now: datetime | None = None) -> list[Fact]:
    """The remembered book with a corrected weight set; the rest kept, any gap held as cash."""
    from argus.lui.research import research_symbols
    from argus.lui.research.parse import holding_pairs

    book = get(facts, "book")
    m = _CORRECTED.search(text)
    if book is None or m is None:
        return facts
    named = research_symbols(m.group("name"))[0]
    if not named:
        return facts
    held: dict[str, float] = {}
    for _, symbol, weight in holding_pairs(book.text):
        held[symbol] = held.get(symbol, 0.0) + weight
    if named[0] not in held:
        return facts
    held[named[0]] = float(m.group("pct")) / 100
    total = sum(held.values())
    if total > 1.0001:
        return facts
    cash = max(0.0, 1.0 - total)
    words = "I hold " + ", ".join(f"{w:.0%} {s.removesuffix('USDT')}" for s, w in held.items()) \
        + (f", {cash:.0%} cash" if cash >= 0.005 else "")
    day = (now or datetime.now(UTC)).date().isoformat()
    updated = Fact(kind="book", subject="", value=str(len(held)), text=words, at=day,
                   replaces=f"“{book.text[:120]}” ({book.at})")
    return [updated, *(f for f in facts if f is not book)]


def dumps(facts: list[Fact]) -> str:
    return json.dumps([asdict(f) for f in facts], separators=(",", ":"))


def get(facts: list[Fact], kind: str, subject: str = "") -> Fact | None:
    return next((f for f in facts if f.kind == kind and f.subject == subject), None)


_FX = (("€", re.compile(r"€|\beur(?:os?)?\b", re.I), "EURUSDUSDT"),
       ("£", re.compile(r"£|\bgbp\b|\bpounds?\b", re.I), "GBPUSDUSDT"))


def in_dollars(capital: Fact) -> tuple[float, str]:
    """A remembered account in US dollars, with how it was said: "€1,234,568" was priced as
    $1,234,568 (a hostile review, round 21). Converted at Bitget's own EURUSD or GBPUSD perpetual,
    and said so; dollars, or a rate that does not answer, are taken as stated."""
    amount = float(capital.value)
    for sign, said, pair in _FX:
        if said.search(capital.text):
            try:
                from argus.lui.research.parse import last_price

                rate = last_price(pair)
            except Exception:
                rate = None
            if rate:
                return amount * rate, (f"{sign}{amount:,.0f} (about ${amount * rate:,.0f} at "
                                       f"Bitget's {pair.removesuffix('USDT')} {rate:.4f})")
            return amount, f"{sign}{amount:,.0f} (no exchange rate answered, so taken as dollars)"
    return amount, f"${amount:,.0f}"


_SHARE_OF_CAPITAL = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?(?:it|that|this|my\s+(?:account|capital|money|"
    r"portfolio|savings))\b", re.I)
""""How much is 5% of that in NVDA?": a share of the remembered account, not all of it."""


def risk_budget_usd(facts: list[Fact]) -> tuple[float, str, Fact] | None:
    """The dollars the trader said they can lose on a trade: a limit stated in dollars, else a
    percentage limit on the account size they gave (or the book they described). None when
    neither is held, so an answer asks instead of assuming one."""
    dollars = get(facts, "loss_usd")
    if dollars is not None:
        return float(dollars.value), f"loss limit of ${float(dollars.value):,.0f}", dollars
    per_trade, account = get(facts, "trade_risk"), get(facts, "capital")
    if per_trade is not None and account is not None:
        # "I won't risk more than 2% per trade" and "$50,000" were both kept, and sizing still
        # said no loss limit was given (a judge, round 21)
        usd = float(per_trade.value) * float(account.value)
        return usd, (f"{float(per_trade.value):.0%} per trade on your "
                     f"${float(account.value):,.0f}"), per_trade
    limit = get(facts, "max_loss")
    capital = get(facts, "capital")
    if limit is not None and capital is None:
        held = get(facts, "book")
        if held is not None:
            from argus.lui.research.sizing import stated_capital

            said = stated_capital(held.text)
            if said:
                capital = Fact(kind="capital", subject="", value=str(said), text=held.text,
                               at=held.at)
    if limit is not None and capital is not None:
        usd = float(limit.value) * float(capital.value)
        return usd, (f"{float(limit.value):.0%} loss limit on your "
                     f"${float(capital.value):,.0f}"), limit
    return None


def remembered_book(facts: list[Fact]) -> Fact | None:
    """The holdings the trader last said in chat, if any: the book an answer uses when none is
    saved in My book."""
    return get(facts, "book")


def remembered_line(fact: Fact, use: str) -> str:
    """How a remembered fact is shown when it shapes an answer."""
    earlier = f" (replacing {fact.replaces})" if fact.replaces else ""
    return f"Remembered: you said “{fact.text}” on {fact.at}{earlier} — {use}."


def thesis_line(fact: Fact, name: str, price_now: float | None) -> str:
    """A stated thesis beside today's price: the gap between what was expected and what happened
    so far, measured from the price when it was said."""
    base = f"Your thesis on {name} ({fact.at}): “{fact.text}”"
    if fact.price_at and price_now:
        move = price_now / fact.price_at - 1
        agrees = ((fact.value == "bull" and move > 0) or (fact.value == "bear" and move < 0))
        direction = ("flat so far" if abs(move) < 0.001 else
                     ("with it" if agrees else "against it") if fact.value in ("bull", "bear")
                     else "since")
        return (f"{base}. Since then {name} has moved {move:+.1%} ({fact.price_at:,.6g} to "
                f"{price_now:,.6g}) — {direction}"
                + ("." if fact.value not in ("bull", "bear") else
                   "; a thesis is judged at its horizon, not on the first move."))
    return base + "."


def apply(request: Any, facts: list[Fact], question: str) -> tuple[Any, list[str]]:
    """The request with what the trader told the console filled in where the question itself
    was silent, and one ``Remembered:`` line for each fact that shaped it."""
    from dataclasses import replace

    from argus.lui.research import ResearchKind

    used: list[str] = []
    budget = get(facts, "budget")
    if (budget is not None and not request.budget_stated
            # "your 2% risk budget is applied" was printed on a return comparison and a rates
            # answer that use no budget (a judge, round 21)
            and request.kind in (ResearchKind.IMPACT, ResearchKind.BOOK, ResearchKind.CONSTRUCT)):
        request = replace(request, budget=float(budget.value), budget_stated=True)
        used.append(remembered_line(budget, f"your {float(budget.value):.0%} risk budget is "
                                            f"applied"))
    position_cap = get(facts, "cap")
    if position_cap is not None and request.kind in (ResearchKind.BOOK, ResearchKind.IMPACT,
                                                     ResearchKind.CONSTRUCT):
        request = replace(request, position_cap=float(position_cap.value))
    horizon = get(facts, "horizon")
    if (horizon is not None and horizon.value.isdigit() and int(horizon.value) >= 336
            and request.kind is ResearchKind.TECHNICALS and request.horizon_hours is None):
        # "I hold positions about 3 months. Is NVDA overbought?" was read on the 4-hour RSI
        # alone (a judge, round 22): a holder for weeks or months is read on the daily chart
        request = replace(request, horizon_hours=int(horizon.value))
        used.append(remembered_line(horizon, "read on the daily chart, which fits that holding "
                                             "period, with the 4-hour reading after it"))
    defaulted = any("no horizon was stated" in note for note in request.notes)
    if (horizon is not None and request.kind is ResearchKind.ANALOGUE and defaulted
            and request.horizon_hours is not None):
        hours = int(horizon.value)
        request = replace(request, horizon_hours=hours, notes=tuple(
            n for n in request.notes if "no horizon was stated" not in n))
        used.append(remembered_line(horizon, f"read over your {_span(hours)} holding period"))
    capital = get(facts, "capital")
    if (capital is not None and request.notional is None
            and request.kind in (ResearchKind.BOOK, ResearchKind.HEDGE)
            and not re.search(r"\$|\d+\s*k\b", question, re.I)):
        from decimal import Decimal

        request = replace(request, notional=Decimal(capital.value))
        used.append(remembered_line(capital, f"sized on your ${float(capital.value):,.0f}"))
    if (capital is not None and request.notional is None and request.kind is ResearchKind.IMPACT
            and not request.book and not re.search(r"\$|\d+\s*k\b", question, re.I)):
        # "is bitcoin a good investment" after "I only have about $1,000" priced "a $10,000
        # position" (a first-time user, round 19, row 635): the money the trader said they have
        # is the position a one-name answer is priced on
        from decimal import Decimal

        dollars, said = in_dollars(capital)
        share = _SHARE_OF_CAPITAL.search(question)
        part = float(share.group("pct")) / 100 if share else 1.0
        request = replace(request, notional=Decimal(str(round(dollars * part, 2))))
        used.append(remembered_line(capital, (
            f"priced on {part:.0%} of your {said}, ${dollars * part:,.0f}" if share else
            f"priced on your {said}") + (
            " as an unlevered position — margin lets you hold more than that, and lose more"
            if re.search(r"\bmargin\b", capital.text, re.I) else "")))
    if request.kind is ResearchKind.IMPACT:
        request, mandate_used = _apply_mandate(request, facts, question)
        used.extend(mandate_used)
    goal = get(facts, "goal")
    if goal is not None and request.kind in (ResearchKind.IMPACT, ResearchKind.BOOK,
                                             ResearchKind.STRESS, ResearchKind.LEVERAGE,
                                             ResearchKind.CONSTRUCT, ResearchKind.HEDGE):
        used.append(remembered_line(goal, "weigh this answer against what the money is for"))
    return request, used


def _apply_mandate(request: Any, facts: list[Fact], question: str) -> tuple[Any, list[str]]:
    """A remembered style or loss limit, handed to the mandate check when this question states no
    mandate of its own; a remembered account size, sized against. Until 2026-09-27 a trader who
    had said "I'm conservative" got the same answer to "should I add 15% TSLA?" as anyone else
    (audit finding 55): the mandate check read the current question only."""
    from dataclasses import replace
    from decimal import Decimal

    from argus.lui.research import stated_profile
    from argus.lui.research.parse import TRIM_TO_BUDGET

    limit, horizon_fact = get(facts, "max_loss"), get(facts, "horizon")
    if request.target == TRIM_TO_BUDGET and limit is not None:
        # "size the trim so I stay inside my drawdown limit" names the limit without its number;
        # the remembered one sizes it (a judge, round 21)
        words = limit.text + (f". my horizon is {int(horizon_fact.value)} hours"
                              if horizon_fact is not None and horizon_fact.value.isdigit() else "")
        return replace(request, mandate_text=words), [
            remembered_line(limit, "the trim is sized inside it")]
    if stated_profile(question) is not None:
        return request, []
    said = [f for f in (get(facts, "style"), get(facts, "max_loss")) if f is not None]
    words = ". ".join(f.text for f in said)
    horizon = get(facts, "horizon")
    if said and horizon is not None and horizon.value.isdigit():
        # A remembered horizon reaches the mandate in the words its reader takes: the mandate
        # printed "a 1-week horizon (a default — say yours)" after "my horizon is a few weeks"
        # (a judge, round 13, 2026-09-30).
        said.append(horizon)
        words += f". my horizon is {int(horizon.value)} hours"
    if not said or stated_profile(words) is None:
        return request, []
    used = [remembered_line(f, "the mandate in this answer is checked against it") for f in said]
    capital = get(facts, "capital")
    amount = None
    if capital is not None:
        amount = Decimal(capital.value)
        used.append(remembered_line(capital, f"the mandate sizes on your "
                                              f"${float(capital.value):,.0f}"))
    return replace(request, mandate_text=words, mandate_capital=amount), used


def after(lines: list[str], request: Any, facts: list[Fact],
          price_now: Any = None, *, tester: Any = None) -> list[str]:
    """Lines an answer gains from memory once it is computed: a stated loss limit set against the
    loss the answer found, a thesis beside the name it is about, and a name the trader said they
    avoid, and the kept checklist run on the entry being weighed."""
    from argus.lui.research.kinds import bare_symbol

    extra: list[str] = []
    limit = get(facts, "max_loss")
    capital = get(facts, "capital")
    if limit is not None and capital is None:
        held = get(facts, "book")
        if held is not None:
            from argus.lui.research.sizing import stated_capital

            said = stated_capital(held.text)
            if said:
                capital = Fact(kind="capital", subject="", value=str(said), text=held.text,
                               at=held.at)
    sized = next((m for line in lines
                  for m in [re.match(r"Bottom line: (?:size \S+ so that its|\S+'s) worst observed "
                                     r"24 hours \((-\d+(?:\.\d+)?)%\)", line)] if m), None)
    if limit is not None and capital is not None and sized is not None:
        # "How big should the SOL short leg be given my 4% drawdown limit?" was answered on a
        # $10,000 default that ignored both (a judge, round 14): the stated limit and book, sized.
        drop = abs(float(sized.group(1))) / 100
        money = float(capital.value)
        allowed = money * float(limit.value)
        if drop > 0:
            if allowed / drop > money:
                # the limit never binds: even the whole account in this leg loses less than it
                # allows, so "costs exactly that" would be false (round 18, 2026-10-01)
                extra.append(remembered_line(limit, (
                    f"{float(limit.value):.0%} of your ${money:,.0f} is ${allowed:,.0f}; a repeat "
                    f"of the worst 24 hours above ({-drop:.1%}) would cost ${money * drop:,.0f} "
                    f"even if this leg were the whole ${money:,.0f} — inside your limit at any "
                    f"size up to the account, before what your other holdings do beside it")))
            else:
                leg = allowed / drop
                extra.append(remembered_line(limit, (
                    f"{float(limit.value):.0%} of your ${money:,.0f} is ${allowed:,.0f}; a repeat "
                    f"of the worst 24 hours above ({-drop:.1%}) on a ${leg:,.0f} leg "
                    f"({leg / money:.0%} of the book) costs exactly that, and any larger leg "
                    f"costs more — the ceiling for this leg alone, before what your other "
                    f"holdings do beside it")))
    shock = next((m for line in lines
                  for m in [re.match(r"Bottom line: If (\S+) moves (-?\d+(?:\.\d+)?)%: your "
                                     r"book moves about (-\d+(?:\.\d+)?)%", line)] if m), None)
    if limit is not None and shock is not None:
        # "How much of my drawdown budget would a 10% NVDA drop use?" got the book's move and no
        # share of the stated limit (a judge, round 15, 2026-10-01): the stated shock, as a share
        # of the limit, before the worst of the scenarios below it.
        cap = float(limit.value) * 100
        move = abs(float(shock.group(3)))
        dollars = (f", ${move / 100 * float(capital.value):,.0f} of your "
                   f"${float(capital.value):,.0f}" if capital is not None else "")
        extra.append(remembered_line(limit, (
            f"the {shock.group(1)} {float(shock.group(2)):+g}% shock moves the book -{move:.2f}%"
            f"{dollars}, which uses {move / cap:.0%} of your {cap:g}% limit"
            + (f" — {move - cap:.2f} points past it" if move > cap else " — inside it"))))
    if limit is not None and not (capital is not None and sized is not None):
        worst = [abs(float(m.group(1))) for line in lines
                 for m in re.finditer(r"(?:book|position)\s+moves\s+(?:about\s+)?(-\d+(?:\.\d+)?)%",
                                      line)]
        if worst:
            deepest = max(worst)
            cap = float(limit.value) * 100
            # "the worst move above, -14.6%" named an assumed QQQ -10% scenario as the worst
            # move, and told the trader how to size (a first-time user, round 20, row 682)
            extra.append(remembered_line(limit, (
                f"the largest scenario move above, -{deepest:.1f}%, is an assumed shock, not a "
                f"day that happened, and it is past your {cap:.0f}% limit" if deepest > cap else
                f"the largest scenario move above, -{deepest:.1f}%, stays inside it")))
    dollars_limit = get(facts, "loss_usd")
    if dollars_limit is not None:
        # "I can only afford to lose $300" was kept and never set beside a -22.5% stress on a
        # $3,000 book, about $675 (a first-time user, round 20, row 680)
        try:
            cap_usd = float(dollars_limit.value)
        except ValueError:
            cap_usd = 0.0
        account = None
        if capital is not None:
            try:
                account = float(capital.value)
            except ValueError:
                account = None
        moves = [abs(float(m.group(1))) for line in lines
                 for m in re.finditer(r"(?:book|position)\s+(?:moves|falls)\s+(?:about\s+)?"
                                      r"-?(\d+(?:\.\d+)?)%", line)]
        priced = next((float(m.group(1).replace(",", "")) for line in lines
                       for m in [re.search(r"\b(?:of|on) \$(\d[\d,]*)\)", line)] if m), None)
        base = priced or account
        if cap_usd > 0 and moves and base:
            deepest = max(moves)
            cost = deepest / 100 * base
            extra.append(remembered_line(dollars_limit, (
                f"the largest move above, -{deepest:.1f}%, is about ${cost:,.0f} of your "
                f"${base:,.0f} — " + (f"${cost - cap_usd:,.0f} past your ${cap_usd:,.0f} limit"
                                      if cost > cap_usd else
                                      f"inside your ${cap_usd:,.0f} limit"))))
    cap_fact = get(facts, "cap")
    book = dict(getattr(request, "book", {}) or {})
    if cap_fact is not None and book:
        # "No single position may exceed 30%" was never held against the book (a judge, round 21)
        cap_limit = float(cap_fact.value)
        weights = {s: w for s, w in book.items()}
        moved = next((m for line in lines for m in [re.search(
            r"(?:takes it from|from) \d+(?:\.\d+)?% to (\d+(?:\.\d+)?)%|"
            r"^At (\d+(?:\.\d+)?)% of the book", line)] if m), None)
        named = (getattr(request, "symbols", ()) or (None,))[0]
        if moved is not None and named:
            after_weight = float(moved.group(1) or moved.group(2)) / 100
            before_weight = weights.get(named, 0.0)
            rest = 1.0 - before_weight
            scale = (1.0 - after_weight) / rest if rest > 0 else 0.0
            weights = {s: (after_weight if s == named else w * scale) for s, w in weights.items()}
            weights.setdefault(named, after_weight)
        over = sorted(((s, w) for s, w in weights.items() if w > cap_limit + 1e-9),
                      key=lambda kv: -kv[1])
        where = "after this trade" if moved is not None else "as the book stands"
        extra.append(remembered_line(cap_fact, (
            f"{', '.join(f'{bare_symbol(s)} at {w:.0%}' for s, w in over)} "
            f"{'is' if len(over) == 1 else 'are'} over your {cap_limit:.0%} cap {where}"
            if over else f"every position is inside your {cap_limit:.0%} cap {where}")))
    for avoided in (f for f in facts if f.kind == "avoid" and not f.subject.endswith("USDT")):
        kind_of = avoided.value
        if kind_of == "leverage" and any(re.search(r"\bHedge: (?:short|long)\b|\b\d+x\b", line)
                                      for line in lines):
            extra.append(remembered_line(avoided, (
                "you said you avoid leverage: a hedge or a position like the one above is held "
                "on a perpetual with margin, so it is shown as information, not as a fit for your "
                "rule")))
        if kind_of in ("meme coins", "crypto", "altcoins"):
            hit = [s for s in getattr(request, "symbols", ())
                   if (kind_of == "meme coins" and s in MEME_COINS)
                   or (kind_of in ("crypto", "altcoins") and _is_crypto(s)
                       and not (kind_of == "altcoins" and s in ("BTCUSDT", "ETHUSDT")))]
            if hit:
                extra.append(remembered_line(avoided, (
                    f"{', '.join(bare_symbol(s) for s in hit)} "
                    f"{'is' if len(hit) == 1 else 'are'} among the {kind_of} you said you avoid")))
    for symbol in getattr(request, "symbols", ())[:3]:
        thesis = get(facts, "thesis", symbol)
        if thesis is not None:
            now = None
            if price_now is not None:
                try:
                    now = float(price_now(symbol))
                except Exception:
                    now = None
            extra.append(thesis_line(thesis, bare_symbol(symbol), now))
        avoid = get(facts, "avoid", symbol)
        if avoid is not None:
            extra.append(remembered_line(
                avoid, f"{bare_symbol(symbol)} is a name you said you stay out of"))
    extra.extend(checklist_lines(request, facts, tester=tester))
    return extra


def against_view(request: Any, facts: list[Fact]) -> str | None:
    """A trade weighed against the trader's own remembered view of the name: after "I'm now
    bearish on NVDA", "should I buy NVDA?" was answered as a plain buy, with the bearish thesis
    eleven lines below (a judge, round 20, row 713). Said under the lead, and not as advice."""
    from argus.lui.research.kinds import ResearchKind, bare_symbol

    if getattr(request, "kind", None) is not ResearchKind.IMPACT or not request.symbols:
        return None
    symbol = request.symbols[0]
    buying = (getattr(request, "side", None) != "short"
              and getattr(request, "target", None) != 0.0)
    for avoided in (f for f in facts if f.kind == "avoid"):
        # "I avoid meme coins. Should I buy DOGE?" said so eight lines down (round 20 re-ask)
        hit = (avoided.subject == symbol
               or (avoided.value == "meme coins" and symbol in MEME_COINS)
               or (avoided.value in ("crypto", "altcoins") and _is_crypto(symbol)
                   and not (avoided.value == "altcoins" and symbol in ("BTCUSDT", "ETHUSDT"))))
        if hit and buying:
            what = (f"among the {avoided.value} you said you avoid"
                    if avoided.subject != symbol else "a name you said you stay out of")
            return (f"Against your own rule: you said “{avoided.text}” on {avoided.at}, and "
                    f"{bare_symbol(symbol)} is {what}. What follows prices the trade; it does "
                    f"not back it.")
    thesis = get(facts, "thesis", symbol)
    if thesis is None or thesis.value not in ("bull", "bear"):
        return None
    held = (getattr(request, "book", {}) or {}).get(symbol, 0.0)
    target = getattr(request, "target", None)
    short = getattr(request, "side", None) == "short"
    adds = (not short and (target is None or target > held) and target != 0.0)
    cuts = short or (target is not None and target < held)
    name = bare_symbol(symbol)
    if thesis.value == "bear" and adds:
        return (f"Against your own view: you said “{thesis.text}” on {thesis.at} — buying {name} "
                f"runs the other way. What follows prices the trade; it does not back it.")
    if thesis.value == "bull" and cuts:
        return (f"Against your own view: you said “{thesis.text}” on {thesis.at} — cutting or "
                f"shorting {name} runs the other way. What follows prices the trade; it does not "
                f"back it.")
    return None


MEME_COINS = frozenset({"DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT",
                        "1000PEPEUSDT", "1000BONKUSDT", "1000SHIBUSDT", "TRUMPUSDT", "POPCATUSDT",
                        "BRETTUSDT", "MOGUSDT", "NEIROUSDT", "PNUTUSDT"})
"""Coins a trader means by "meme coins", for checking a remembered dislike against an answer."""


def _is_crypto(symbol: str) -> bool:
    from argus.lui.research.parse import is_us_equity
    from argus.market import universe

    return not is_us_equity(symbol) and universe.NOT_EQUITY.get(symbol, "crypto") == "crypto"


def checklist_lines(request: Any, facts: list[Fact], *, tester: Any = None,
                    today: Any = None) -> list[str]:
    """The kept checklist, run on the entry this question asks about: each check that a question
    can test, with what it finds now; the rest listed as reminders, said to be untested.

    Run only where an entry is being weighed (adding a name to a book, executing an order), on the
    first name the question names, over the trader's remembered holding period or, failing one,
    :data:`DEFAULT_CHECK_DAYS`. ``tester`` is ``journal.retest``'s signature, injected in tests."""
    from argus.lui.research import ResearchKind

    checks = [f for f in facts if f.kind == "check"]
    symbols = getattr(request, "symbols", ())
    if not checks or not symbols or getattr(request, "kind", None) not in (
            ResearchKind.IMPACT, ResearchKind.EXECUTION):
        return []
    from datetime import date as date_type

    from argus.lui import journal

    symbol = symbols[0]
    side = str(getattr(request, "side", "long") or "long")
    horizon = get(facts, "horizon")
    days = max(1, int(horizon.value) // 24) if horizon is not None else DEFAULT_CHECK_DAYS
    day = today if isinstance(today, date_type) else datetime.now(UTC).date()
    run = tester or (lambda key: journal.retest(
        key, symbol, side, today=day, horizon_days=days,
        next_report=journal.next_report_date, holidays=journal.holidays()))
    held = ("your remembered " if horizon is not None else "a default ") + f"{days}-day hold"
    out = [f"Your checklist, from your trade review on {checks[0].at}, run on this "
           f"{bare_symbol(symbol)} {'buy' if side == 'long' else 'sell'} over {held}:"]
    untested: list[str] = []
    for fact in checks:
        try:
            found = run(fact.subject)
        except Exception as exc:  # a check that cannot run is said so, never passed
            found = f"could not be run ({type(exc).__name__}), so it is not passed"
        if found is None:
            untested.append(fact.text)
        else:
            out.append(f"Checklist: {fact.text} Now: {found}.")
    if untested:
        out.append("Not testable from a question, kept as reminders: " + " ".join(untested))
    return out


def _span(hours: int) -> str:
    for size, unit in ((24 * 30, "month"), (24 * 7, "week"), (24, "day"), (1, "hour")):
        if hours >= size and hours % size == 0:
            count = hours // size
            return f"{count}-{unit}"
    return f"{hours}-hour"


_RECALL = re.compile(
    r"\bwhat\s+(?:do|did|have)\s+(?:you|u)\s+(?:still\s+)?(?:remember|know|recall|noted?|stored?|"
    r"kept?|got|have)\b.*\b(?:about\s+me|me\b|my\s+\w+)|"
    r"\bwhat\s+(?:have|did)\s+i\s+(?:told|tell|said|say)\s+you\b|"
    r"\b(?:remind|tell|show)\s+me\s+(?:again\s+)?what\s+(?:i|we)\s+(?:told|said|gave)\s+you\b|"
    r"\bwhat\s+(?:was|were)\s+(?:it|that|those|the\s+\w+)\s+(?:i|we)\s+(?:told|said|gave)\s+you\b|"
    r"\b(?:list|show|tell)\s+(?:me\s+)?(?:everything|all)\s+(?:you\s+)?(?:remember|know|noted)\b|"
    r"\bwhat\s+(?:is|are)\s+(?:in\s+)?(?:your|the)\s+memory\b", re.I)
"""A question about what the console has kept of the trader (round 14: "What do you remember about
me, my book, my horizon and my limits?" was declined while memory held seven facts)."""

_RECALL_ORDER = ("book", "capital", "budget", "cap", "trade_risk", "max_loss", "loss_usd",
                 "horizon", "style", "goal", "thesis", "avoid", "check")
_RECALL_LABEL = {
    "book": "Your book", "capital": "Your account size", "budget": "Your risk budget",
    "cap": "Your position cap", "trade_risk": "Your risk per trade",
    "max_loss": "Your loss limit", "loss_usd": "Your loss limit", "horizon": "Your horizon",
    "style": "Your style",
    "goal": "Your goal", "thesis": "Your thesis", "avoid": "What you avoid",
    "check": "A check kept from a review",
}


_EARLIER = re.compile(
    r"\bwhat\s+(?:did|have)\s+i\s+(?:say|said|tell|told|write|wrote|type|typed|ask|asked)\b"
    r"(?:\s+(?:you|u))?\b.*\b(?:first|original(?:ly)?|earliest|at\s+the\s+(?:start|beginning)|"
    r"(?:previous|last)\s+(?:message|question|one))\b|"
    r"\bmy\s+(?:very\s+)?(?:first|earliest|original|previous|last)\s+(?:message|question|"
    r"statement|sentence)\b|"
    r"\b(?:very\s+)?first\s+(?:thing|words?)\s+i\s+(?:said|told|asked|wrote|typed)\b", re.I)
"""A question about the trader's own earlier messages, not about what is remembered now: "What did
I say I hold in my very first message?" after the book was changed was answered with the desk's own
open position (a hostile review, 2026-10-01)."""


def earlier_asked(question: str) -> bool:
    return bool(_EARLIER.search(question))


def earlier_lines(question: str, prior: list[str], facts: list[Fact]) -> list[str]:
    """The trader's own earlier message, quoted, and what is held now beside it. The messages come
    from the turn history the page keeps (the last twelve), so a longer conversation is said to be
    cut rather than passed off as complete."""
    if not prior:
        return ["Bottom line: this is your first message in this conversation, so there is nothing "
                "earlier to repeat."]
    wants_last = re.search(r"\b(?:previous|last)\b", question, re.I) is not None
    quoted = prior[-1] if wants_last else prior[0]
    cut = not wants_last and len(prior) >= 12
    which = "previous" if wants_last else ("earliest kept" if cut else "first")
    lines = [f"Bottom line: your {which} message was “{quoted.strip()}”."]
    if cut:
        lines.append("Only the last twelve messages are kept, so an earlier one may have been "
                     "dropped.")
    changed = [f for f in facts if f.replaces]
    for fact in changed:
        lines.append(f"{_RECALL_LABEL.get(fact.kind, fact.kind.capitalize())} now: “{fact.text}” "
                     f"— said {fact.at}, replacing {fact.replaces}.")
    return lines


_WHAT_KEPT = (r"(?P<what>(?:max(?:imum)?\s+)?loss(?:\s+limit)?|drawdown(?:\s+limit)?|"
              r"risk\s+budget|horizon|(?:trading\s+)?style|account(?:\s+size)?|capital|goal)")
_RECALL_ONE_FORMS = tuple(re.compile(p, re.I) for p in (
    r"\bwhat(?:\s+(?:was|is|were|are)|'?s)\s+my\s+" + _WHAT_KEPT
    + r"\b(?:\s+(?:i|we)\s+(?:told|gave|said|set|mentioned)\b[^?]*)?\s*\??\s*$",
    # "the" only with "I told you": "stress test my book ..., what's the drawdown" asks for a
    # computed figure, not a remembered one.
    r"\bwhat(?:\s+(?:was|is|were|are)|'?s)\s+the\s+" + _WHAT_KEPT
    + r"\s+(?:i|we)\s+(?:told|gave|said|set|mentioned)\b[^?]*\??\s*$",
    r"\b(?:remind|tell)\s+me\s+(?:again\s+)?what\s+(?:my\s+)?" + _WHAT_KEPT
    + r"\b(?:\s+(?:was|is))?(?:\s+(?:i|we)\s+(?:told|gave|said|set|mentioned)\b[^?]*)?\s*\??\s*$",
    r"\bwhat\s+" + _WHAT_KEPT + r"\s+did\s+(?:i|we)\s+(?:tell|give|say|set|mention)\b[^?]*\??\s*$",
))
"""One remembered fact asked for by name: "what was my loss limit" went to a desk-decision reader
and returned a stale decision (a first-user audit, round 17, 2026-10-01)."""
_RECALL_KINDS = {"loss": ("loss_usd", "max_loss"), "drawdown": ("max_loss", "loss_usd"),
                 "risk": ("budget",), "horizon": ("horizon",), "style": ("style",),
                 "trading": ("style",), "account": ("capital", "book"), "capital": ("capital",),
                 "goal": ("goal",), "max": ("loss_usd", "max_loss"), "maximum": ("loss_usd",
                                                                                   "max_loss")}


_SCENARIO_STATED = re.compile(r"\b(?:if|when|should)\b[^?]*?\d+(?:\.\d+)?\s*(?:%|percent\b|"
                              r"bps\b|basis\s+points?\b)", re.I)
"""A move stated in the question: "if SPY falls 4%..., what's my loss?" asks for a computed loss,
not the remembered loss limit (a hostile review, round 22)."""


def _recall_match(question: str) -> re.Match[str] | None:
    if _SCENARIO_STATED.search(question):
        return None
    return next((m for p in _RECALL_ONE_FORMS if (m := p.search(question))), None)


_RECALL_MANY = re.compile(
    r"\bwhat\s+do\s+i\s+(?:hold|own)\b|\bwhat(?:'s|\s+is|\s+are)\s+my\s+(?:account(?:\s+size)?|"
    r"capital|holdings|book|positions?|max(?:imum)?\s+loss(?:\s+per\s+trade)?|loss\s+limit|"
    r"risk\s+budget|horizon|style|goal)\b|\bhow\s+(?:big|large|much)\s+is\s+my\s+(?:account|"
    r"book|portfolio|capital)\b", re.I)
"""A question that asks back for the trader's own facts. "What is my account size, what do I
hold, and what is my max loss per trade?" went to the desk's track record and "What do I hold?"
to the desk's open positions (a hostile review, round 19, row 648)."""
_VIEW_ASKED = re.compile(
    r"\bwhat(?:'s|\s+is|\s+was)\s+my\s+(?:view|thesis|take|stance|position|call)\s+on\s+"
    r"(?P<name>[A-Za-z][A-Za-z.]{1,11})\b|\bam\s+i\s+(?:bullish|bearish)\s+on\s+"
    r"(?P<name2>[A-Za-z][A-Za-z.]{1,11})\b", re.I)


def recall_asked(question: str) -> bool:
    return (bool(_RECALL.search(question)) or _recall_match(question) is not None
            or len(_RECALL_MANY.findall(question)) >= 2
            or bool(re.match(r"^\W*(?:what\s+do\s+i\s+(?:hold|own)|what(?:'s|\s+is|\s+are)\s+my\s+"
                             r"(?:positions?|holdings|book))\W*$", question, re.I))
            or bool(_VIEW_ASKED.search(question)))


_ASKED_KIND = (
    (re.compile(r"\baccount(?:\s+size)?\b|\bcapital\b", re.I), ("capital",), "account size"),
    (re.compile(r"\bwhat\s+do\s+i\s+(?:hold|own)\b|\bholdings\b|\bmy\s+(?:book|positions)\b", re.I),
     ("book",), "holdings"),
    (re.compile(r"\bmax(?:imum)?\s+loss|\bloss\s+limit", re.I), ("max_loss", "loss_usd"),
     "loss limit"),
    (re.compile(r"\brisk\s+budget", re.I), ("budget",), "risk budget"),
    (re.compile(r"\bhorizon\b", re.I), ("horizon",), "horizon"),
    (re.compile(r"\bstyle\b", re.I), ("style",), "style"),
)


def recall_missing(question: str, facts: list[Fact]) -> list[str]:
    """What the question asked back that was never said: a list of facts that silently skipped
    the loss limit read as if it had answered it (round 19, row 648)."""
    have = {f.kind for f in facts}
    missing = [label for pattern, kinds, label in _ASKED_KIND
               if pattern.search(question) and not have & set(kinds)]
    if not missing:
        return []
    return [f"Not remembered: your {' or '.join(missing)} — you have not said "
            f"{'it' if len(missing) == 1 else 'them'} here yet; say it in a sentence "
            f"(\"my loss limit is $500 a trade\") and it is kept."]


def recall_view(question: str, facts: list[Fact]) -> list[str] | None:
    """The trader's own remembered view on the name asked about, or None if not asked."""
    from argus.lui.research import research_symbols

    m = _VIEW_ASKED.search(question)
    if m is None:
        return None
    named = research_symbols(m.group("name") or m.group("name2") or "")[0]
    if not named:
        return None
    views = [f for f in facts if f.kind == "thesis" and f.subject == named[0]]
    name = named[0].removesuffix("USDT")
    if not views:
        return [f"Bottom line: no view on {name} is remembered for you in this browser — say it "
                f"(\"I'm bearish on {name} because …\") and it is kept."]
    view = views[-1]
    side = {"bull": "bullish", "bear": "bearish"}.get(view.value, "a view")
    earlier = f", replacing {view.replaces}" if view.replaces else ""
    return [f"Bottom line: you are {side} on {name} — you said “{view.text}” on {view.at}"
            f"{earlier}."]


def recall_one(question: str, facts: list[Fact]) -> list[str] | None:
    """The one fact asked for, in the trader's own words; None when the question is not that."""
    m = _recall_match(question)
    if m is None:
        return None
    word = m.group("what").split()[0].lower()
    kinds = _RECALL_KINDS.get(word, ())
    held = [f for k in kinds for f in facts if f.kind == k and not f.subject]
    label = m.group("what").strip().lower()
    if not held:
        return [f"Bottom line: no {label} is remembered for you in this browser — say it in a "
                f"sentence (\"my loss limit is $300\") and it is kept."]
    fact = held[0]
    earlier = f", replacing {fact.replaces}" if fact.replaces else ""
    return [f"Bottom line: you said “{fact.text}” on {fact.at}{earlier}."]


def recall_lines(facts: list[Fact]) -> list[str]:
    """Every remembered fact, in the trader's own words with the date each was said."""
    if not facts:
        return ["Bottom line: nothing is remembered about you yet. Tell it in a sentence — "
                "\"I hold 50% ETH and 50% SOL\", \"my loss limit is 5%\", \"I trade over two "
                "weeks\" — and it is kept in this browser only and shapes the answers it applies "
                "to."]
    order = {kind: i for i, kind in enumerate(_RECALL_ORDER)}
    ranked = sorted(facts, key=lambda f: (order.get(f.kind, 99), f.subject))
    lines = [f"Bottom line: {len(facts)} thing{'s' if len(facts) != 1 else ''} remembered about "
             f"you, all in your own words and kept in this browser only."]
    for fact in ranked:
        label = _RECALL_LABEL.get(fact.kind, fact.kind.capitalize())
        earlier = f" (replacing {fact.replaces})" if fact.replaces else ""
        lines.append(f"{label}: “{fact.text}” — said {fact.at}{earlier}.")
    lines.append("Forget any of them from the list under My book.")
    return lines


def acknowledgement(new: list[Fact]) -> list[str]:
    """The reply to a message that only tells the console something about the trader."""
    # One sentence can carry the same words as two facts (a style and a horizon): said once.
    unique = list({(f.text, f.replaces): f for f in new}.values())
    said = "; ".join(f"“{f.text}”" + (f", replacing {f.replaces}" if f.replaces else "")
                     for f in unique)
    return [f"Bottom line: noted — {said}. It is kept in this browser only and shapes every later "
            f"answer it applies to, each time with a Remembered: line saying so; forget it from "
            f"the list under your book."]


__all__ = [
    "DEFAULT_CHECK_DAYS",
    "KINDS",
    "MAX_FACTS",
    "Fact",
    "acknowledgement",
    "after",
    "apply",
    "checklist_lines",
    "checks_from",
    "dumps",
    "earlier_asked",
    "earlier_lines",
    "extract",
    "get",
    "merge",
    "merge_checks",
    "parse",
    "recall_asked",
    "recall_lines",
    "remembered_book",
    "remembered_line",
    "thesis_line",
]
